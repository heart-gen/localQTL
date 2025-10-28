import torch
import pandas as pd
from typing import Optional, Tuple, List

from .haplotypeio import InputGeneratorCis, InputGeneratorCisWithHaps
from .regression_kernels import (
    Residualizer,
    run_batch_regression,
    run_batch_regression_with_permutations
)
from .stats import beta_approx_pval
from .preproc import impute_mean_and_filter, allele_stats, filter_by_maf

def _residualize_matrix_with_covariates(
        Y: torch.Tensor, C: Optional[pd.DataFrame], device: str
) -> Tuple[torch.Tensor, Optional[Residualizer]]:
    """
    Residualize (features x samples) matrix Y against covariates C across samples.
    Returns residualized Y and a Residualizer (or None if no covariates).
    """
    if C is None:
        return Y, None
    C_t = torch.tensor(C.values, dtype=torch.float32, device=device)
    rez = Residualizer(C_t)
    (Y_resid,) = rez.transform(Y, center=True)
    return Y_resid, rez


def _residualize_batch(
        y: torch.Tensor, G: torch.Tensor, H: Optional[torch.Tensor],
        rez: Optional[Residualizer], center: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """
    Residualize y (1D), G (2D), and optional H (3D) with the same Residualizer.
    """
    if rez is None:
        return y, G, H

    # Prepare matrices as (features x samples)
    mats: List[torch.Tensor] = [G]  # (m x n)
    H_shape = None
    if H is not None:
        m, n, pH = H.shape
        H_shape = (m, n, pH)
        H_flat = H.reshape(m * pH, n)  # (m*pH x n)
        mats.append(H_flat)

    # Apply once
    mats_resid = rez.transform(*mats, center=center)

    G_resid = mats_resid[0]
    H_resid = None
    if H is not None:
        H_resid = mats_resid[1].reshape(H_shape)

    # y is (n,), make it (1 x n) for transform
    (y_resid_mat,) = rez.transform(y.unsqueeze(0), center=center)
    y_resid = y_resid_mat.squeeze(0)

    return y_resid, G_resid, H_resid


def _run_nominal_core(ig, variant_df, rez, nperm, device, maf_threshold: float = 0.0):
    """
    Shared inner loop for nominal (and optional permutation) mapping.
    Handles both InputGeneratorCis (no haps) and InputGeneratorCisWithHaps (haps).
    """
    out_rows = []
    # Iterate phenotypes / groups
    for batch in ig.generate_data():
        if len(batch) == 5 and not isinstance(batch[3], (list, tuple)):
            p, G_block, v_idx, H_block, pid = batch
        elif len(batch) == 4:
            p, G_block, v_idx, pid = batch
            H_block = None
        else:
            raise ValueError(f"Unexpected batch shape from generator: len={len(batch)}")

        # Tensors
        y_t = torch.tensor(p, dtype=torch.float32, device=device)
        G_t = torch.tensor(G_block, dtype=torch.float32, device=device)
        H_t = torch.tensor(H_block, dtype=torch.float32, device=device) if H_block is not None else None

        if H_t is not None:
            H_t = H_t[:, :, :-1]  # drop last ancestry to avoid collinearity

        # Impute and drop monomorphic
        G_t, keep_mask, _ = impute_mean_and_filter(G_t)
        if keep_mask.numel() == 0 or keep_mask.sum().item() == 0:
            continue

        # Optional MAF filter
        if maf_threshold and maf_threshold > 0:
            keep_maf, _ = filter_by_maf(G_t, maf_threshold, ploidy=2)
            keep_mask = keep_mask & keep_maf
            if keep_mask.sum().item() == 0:
                continue

        # Mask G, H, and indices consistently
        G_t = G_t[keep_mask]
        v_idx = v_idx[keep_mask.detach().cpu().numpy()]
        if H_t is not None:
            H_t = H_t[keep_mask]

        # Allele statistics on imputed genotypes
        af_t, ma_samples_t, ma_count_t = allele_stats(G_t, ploidy=2)
 
        # Residualize this batch (makes y/G/H orthogonal to covariates)
        y_t, G_t, H_t = _residualize_batch(y_t, G_t, H_t, rez, center=True)

        k_eff = rez.Q_t.shape[1] if rez is not None else 0        
        # Permuted phenotypes (if requested)
        if nperm is not None:
            perms = [torch.randperm(y_t.shape[0], device=device) for _ in range(nperm)]
            y_perm = torch.stack([y_t[idx] for idx in perms], dim=1)  # (n x nperm)
            betas, ses, tstats, r2_perm = run_batch_regression_with_permutations(
                y=y_t, G=G_t, H=H_t, y_perm=y_perm, k_eff=k_eff, device=device
            )
            perm_max_r2 = r2_perm.max().item()
        else:
            betas, ses, tstats = run_batch_regression(
                y=y_t, G=G_t, H=H_t, k_eff=k_eff, device=device
            )
            r2_perm = perm_max_r2 = None

        # Variant metadata for this window
        var_ids = variant_df.index.values[v_idx]
        var_pos = variant_df.iloc[v_idx]["pos"].values

        # Distances to phenotype start/end
        start_pos = ig.phenotype_start[pid]
        end_pos = ig.phenotype_end[pid]
        start_distance = var_pos - start_pos
        end_distance = var_pos - end_pos

        # Assemble result rows
        out = {
            "phenotype_id": pid,
            "variant_id": var_ids,
            "pos": var_pos,
            "start_distance": start_distance,
            "end_distance": end_distance,
            "beta": betas[:, 0].detach().cpu().numpy(),
            "se": ses[:, 0].detach().cpu().numpy(),
            "tstat": tstats[:, 0].detach().cpu().numpy(),
            "af": af_t.detach().cpu().numpy(),
            "ma_samples": ma_samples_t.detach().cpu().numpy(),
            "ma_count": ma_count_t.detach().cpu().numpy(),
        }
        df = pd.DataFrame(out)

        if perm_max_r2 is not None:
            df["perm_max_r2"] = perm_max_r2 
        out_rows.append(df)

    return pd.concat(out_rows, axis=0).reset_index(drop=True)


def _run_permutation_core(ig, variant_df, rez, nperm: int, device: str,
                          beta_approx: bool = True, maf_threshold: float = 0.0) -> pd.DataFrame:
    """
    One top association per phenotype with empirical permutation p-value.
    Compatible with InputGeneratorCis and InputGeneratorCisWithHaps.
    (Group mode is not handled here.)
    """
    out_rows = []
    if nperm is None or nperm <= 0:
        raise ValueError("nperm must be a positive integer for map_permutations.")

    for batch in ig.generate_data():
        # Accept shapes: (p, G, v_idx, pid) or (p, G, v_idx, H, pid)
        if len(batch) == 4:
            p, G_block, v_idx, pid = batch
            H_block = None
        elif len(batch) == 5 and not isinstance(batch[3], (list, tuple)):
            p, G_block, v_idx, H_block, pid = batch
        else:
            # Skip groups in this core (keep parity with tensorQTL's per-phenotype map_cis)
            raise ValueError("Group mode not supported in _run_permutation_core.")

        # Tensors
        y_t = torch.tensor(p, dtype=torch.float32, device=device)
        G_t = torch.tensor(G_block, dtype=torch.float32, device=device)
        H_t = torch.tensor(H_block, dtype=torch.float32, device=device) if H_block is not None else None

        # Impute and drop monomorphic
        G_t, keep_mask, _ = impute_mean_and_filter(G_t)
        if keep_mask.numel() == 0 or keep_mask.sum().item() == 0:
            continue

        # Optional MAF filter
        if maf_threshold and maf_threshold > 0:
            keep_maf, _ = filter_by_maf(G_t, maf_threshold, ploidy=2)
            keep_mask = keep_mask & keep_maf
            if keep_mask.sum().item() == 0:
                continue

        # Mask G, H, and indices consistently
        G_t = G_t[keep_mask]
        v_idx = v_idx[keep_mask.detach().cpu().numpy()]
        if H_t is not None:
            H_t = H_t[keep_mask]

        # Minor-allele stats before residualization
        af_t, ma_samples_t, ma_count_t = allele_stats(G_t, ploidy=2)

        # Residualize y/G/H against covariates
        y_t, G_t, H_t = _residualize_batch(y_t, G_t, H_t, rez, center=True)

        # Build permuted phenotypes (n x nperm)
        perms = [torch.randperm(y_t.shape[0], device=device) for _ in range(nperm)]
        y_perm = torch.stack([y_t[idx] for idx in perms], dim=1)

        # Run batched regression with permutations
        betas, ses, tstats, r2_perm = run_batch_regression_with_permutations(
            y=y_t, G=G_t, H=H_t, y_perm=y_perm, device=device
        )

        # Choose top variant by partial R^2 for genotype predictor
        # For multiple regression, partial R^2 for a single predictor:
        # R^2_partial = t^2 / (t^2 + dof), dof = n - p
        n = y_t.shape[0]
        p_pred = 1 + (H_t.shape[2] if H_t is not None else 0)   # genotype + (k-1) ancestry columns
        dof = max(n - p_pred, 1)
        t_g = tstats[:, 0]                                      # (m,)
        r2_nominal_vec = (t_g.double().pow(2) / (t_g.double().pow(2) + dof)).cpu().numpy()
        ix = int(np.nanargmax(r2_nominal_vec))

        # Extract top variant stats
        beta = float(betas[ix, 0].detach().cpu().numpy())
        se = float(ses[ix, 0].detach().cpu().numpy())
        tval = float(tstats[ix, 0].detach().cpu().numpy())
        r2_nominal = float(r2_nominal_vec[ix])

        # Nominal p (two-sided t)
        pval_nominal = float(2.0 * stats.t.sf(np.abs(tval), df=dof))

        # Empirical permutation p (max across variants each perm)
        r2_perm_np = r2_perm.detach().cpu().numpy()  # (nperm,)
        pval_perm = float((np.sum(r2_perm_np >= r2_nominal) + 1) / (r2_perm_np.size + 1))

        # Optional Beta approximation
        if beta_approx:
            pval_beta, a_hat, b_hat = _beta_approx_pval(r2_perm_np, r2_nominal)
        else:
            pval_beta, a_hat, b_hat = np.nan, np.nan, np.nan

        # Metadata
        var_id = variant_df.index.values[v_idx[ix]]
        var_pos = int(variant_df.iloc[v_idx[ix]]["pos"])
        start_pos = ig.phenotype_start[pid]
        end_pos = ig.phenotype_end[pid]
        start_distance = int(var_pos - start_pos)
        end_distance = int(var_pos - end_pos)
        num_var = int(G_t.shape[0])

        out_rows.append(pd.Series({
            "phenotype_id": pid,
            "variant_id": var_id,
            "pos": var_pos,
            "start_distance": start_distance,
            "end_distance": end_distance,
            "num_var": num_var,
            "beta": beta,
            "se": se,
            "tstat": tval,
            "r2_nominal": r2_nominal,
            "pval_nominal": pval_nominal,
            "pval_perm": pval_perm,
            "pval_beta": pval_beta,
            "beta_shape1": a_hat,
            "beta_shape2": b_hat,
            "af": float(af_t[ix].detach().cpu().numpy()),
            "ma_samples": int(ma_samples_t[ix].detach().cpu().numpy()),
            "ma_count": float(ma_count_t[ix].detach().cpu().numpy()),
            "dof": dof,
        }))

    return pd.DataFrame(out_rows).reset_index(drop=True)


def map_nominal(
        genotype_df: pd.DataFrame, variant_df: pd.DataFrame, phenotype_df: pd.DataFrame,
        phenotype_pos_df: pd.DataFrame, covariates_df: Optional[pd.DataFrame] = None,
        haplotypes: Optional[object] = None, loci_df: Optional[pd.DataFrame] = None,
        maf_threshold: float = 0.0, window: int = 1_000_000,
        nperm: Optional[int] = None, device: str = "cuda",
) -> pd.DataFrame:
    """
    Nominal cis-QTL scan with optional permutations and local ancestry.

    Adjusts for covariates by residualizing y, G, and H across samples using the
    same Residualizer (projection onto the orthogonal complement of C).
    """
    device = device if device in ("cuda", "cpu") else ("cuda" if torch.cuda.is_available() else "cpu")

    # Build the appropriate input generator
    if haplotypes is not None:
        ig = InputGeneratorCisWithHaps(
            genotype_df=genotype_df, variant_df=variant_df, phenotype_df=phenotype_df,
            phenotype_pos_df=phenotype_pos_df, window=window, haplotypes=haplotypes,
            loci_df=loci_df,
        )
    else:
        ig = InputGeneratorCis(
            genotype_df=genotype_df, variant_df=variant_df, phenotype_df=phenotype_df,
            phenotype_pos_df=phenotype_pos_df, window=window,
        )

    # Residualize all phenotypes once (features x samples)
    Y = torch.tensor(ig.phenotype_df.values, dtype=torch.float32, device=device)
    Y_resid, rez = _residualize_matrix_with_covariates(Y, covariates_df, device)

    # Put residuals back so the generator yields the same ordering/IDs
    phenotype_df_resid = pd.DataFrame(
        Y_resid.cpu().numpy(),
        index=ig.phenotype_df.index,
        columns=ig.phenotype_df.columns,
    )
    ig.phenotype_df = phenotype_df_resid
    
    return _run_nominal_core(ig, variant_df, rez, nperm, device,
                             maf_threshold=maf_threshold)


def map_permutations(
        genotype_df: pd.DataFrame, variant_df: pd.DataFrame, phenotype_df: pd.DataFrame,
        phenotype_pos_df: pd.DataFrame, covariates_df: Optional[pd.DataFrame] = None,
        haplotypes: Optional[object] = None, loci_df: Optional[pd.DataFrame] = None,
        maf_threshold: float = 0.0, window: int = 1_000_000, nperm: int = 10_000,
        device: str = "cuda", beta_approx: bool = True
) -> pd.DataFrame:
    """
    Empirical cis-QTL mapping (one top variant per phenotype) with permutations.
    Returns a DataFrame with empirical p-values (and optional Beta approximation).
    """
    device = device if device in ("cuda", "cpu") else ("cuda" if torch.cuda.is_available() else "cpu")

    # Build the appropriate input generator
    if haplotypes is not None:
        ig = InputGeneratorCisWithHaps(
            genotype_df=genotype_df, variant_df=variant_df, phenotype_df=phenotype_df,
            phenotype_pos_df=phenotype_pos_df, window=window, haplotypes=haplotypes, loci_df=loci_df,
        )
    else:
        ig = InputGeneratorCis(
            genotype_df=genotype_df, variant_df=variant_df, phenotype_df=phenotype_df,
            phenotype_pos_df=phenotype_pos_df, window=window,
        )

    # Residualize phenotypes once
    Y = torch.tensor(ig.phenotype_df.values, dtype=torch.float32, device=device)
    Y_resid, rez = _residualize_matrix_with_covariates(Y, covariates_df, device)
    ig.phenotype_df = pd.DataFrame(Y_resid.cpu().numpy(), index=ig.phenotype_df.index,
                                   columns=ig.phenotype_df.columns)

    return _run_permutation_core(ig, variant_df, rez, nperm=nperm,
                                 device=device, beta_approx=beta_approx,
                                 maf_threshold=maf_threshold)


class CisMapper:
    """
    Convenience wrapper: build an InputGenerator and run nominal scans.
    """
    def __init__(
            self, genotype_df: pd.DataFrame, variant_df: pd.DataFrame,
            phenotype_df: pd.DataFrame, phenotype_pos_df: pd.DataFrame,
            covariates_df: Optional[pd.DataFrame] = None,
            haplotypes: Optional[object] = None,
            loci_df: Optional[pd.DataFrame] = None,
            device: str = "auto", window: int = 1_000_000,
            maf_threshold: float = 0.0,
    ):
        self.device = ("cuda" if (device == "auto" and torch.cuda.is_available()) else
                       device if device in ("cuda", "cpu") else "cpu")
        self.variant_df = variant_df
        self.window = window
        self.maf_threshold = maf_threshold

        if haplotypes is not None:
            self.ig = InputGeneratorCisWithHaps(
                genotype_df=genotype_df,
                variant_df=variant_df,
                phenotype_df=phenotype_df,
                phenotype_pos_df=phenotype_pos_df,
                window=window,
                haplotypes=haplotypes,
                loci_df=loci_df,
            )
            self.with_haps = True
        else:
            self.ig = InputGeneratorCis(
                genotype_df=genotype_df,
                variant_df=variant_df,
                phenotype_df=phenotype_df,
                phenotype_pos_df=phenotype_pos_df,
                window=window,
            )
            self.with_haps = False

        # Residualize all phenotypes once and store
        Y = torch.tensor(self.ig.phenotype_df.values, dtype=torch.float32, device=self.device)
        Y_resid, self.rez = _residualize_matrix_with_covariates(Y, covariates_df, self.device)
        self.ig.phenotype_df = pd.DataFrame(
            Y_resid.cpu().numpy(),
            index=self.ig.phenotype_df.index,
            columns=self.ig.phenotype_df.columns,
        )

    def map_nominal(self, nperm: int | None = None, maf_threshold: float | None = None) -> pd.DataFrame:
        mt = self.maf_threshold if maf_threshold is None else maf_threshold
        return _run_nominal_core(self.ig, self.variant_df, self.rez, nperm,
                                 self.device, maf_threshold=mt)

    def map_permutations(self, nperm: int=10_000, beta_approx: bool=True,
                         maf_threshold: float | None = None) -> pd.DataFrame:
        """Empirical cis-QTLs (top per phenotype) with permutation p-values."""
        mt = self.maf_threshold if maf_threshold is None else maf_threshold
        return _run_permutation_core(
            self.ig, self.variant_df, self.rez, nperm=nperm, device=self.device,
            beta_approx=beta_approx, maf_threshold=mt
        )

    def map_independent(self, cis_df, fdr=0.05):
        """Forward–backward conditional mapping"""
        pass

    # Helper functions
    def _residualize(self, Y, C):
        C_t = torch.tensor(C, dtype=torch.float32, device=self.device)
        Y_t = torch.tensor(Y, dtype=torch.float32, device=self.device)
        CtC_inv = torch.linalg.inv(C_t.T @ C_t)
        proj = C_t @ (CtC_inv @ (C_t.T @ Y_t))
        return (Y_t - proj).cpu().numpy()

    def _regress(self, X, y):
        # closed-form OLS
        XtX = X.T @ X
        XtX_inv = torch.linalg.inv(XtX)
        betas = XtX_inv @ (X.T @ y)
        y_hat = X @ betas
        resid = y - y_hat
        k_eff = self.rez.Q_t.shape[1] if self.rez is not None else 0
        p = X.shape[-1]
        dof = X.shape[0] - k_eff - p
        sigma2 = (resid.transpose(1,2) @ resid).squeeze() / dof
        se = torch.sqrt(torch.diag(XtX_inv) * sigma2)
        tstats = betas / se
        return tstats, betas, se
