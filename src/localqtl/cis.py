import torch
import pandas as pd
from typing import Optional, Tuple, List

from .haplotypeio import InputGeneratorCis, InputGeneratorCisWithHaps
from .regression_kernels import (
    Residualizer,
    run_batch_regression,
    run_batch_regression_with_permutations
)

def _impute_and_filter(G_t: torch.Tensor):
    """Mean-impute per variant (treat -9 or non-finite as missing) and drop monomorphic variants."""
    miss = (~torch.isfinite(G_t)) | (G_t == -9)
    if miss.any():
        num = torch.where(miss, torch.zeros_like(G_t), G_t).sum(dim=1, keepdim=True)
        den = (~miss).sum(dim=1, keepdim=True).clamp_min(1)
        row_mean = num / den
        G_t = torch.where(miss, row_mean, G_t)
    keep = G_t.var(dim=1, unbiased=False) > 0
    return G_t, keep


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


def _run_nominal_core(ig, variant_df, rez, nperm, device):
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

        # Mean-impute and drop monomorphic BEFORE residualization; update indices/H accordingly
        G_t, keep_mask = _impute_and_filter(G_t)
        if keep_mask.numel() == 0 or keep_mask.sum().item() == 0:
            continue  # no valid variants
        if keep_mask.shape[0] != G_t.shape[0]:  # safety
            keep_mask = torch.ones(G_t.shape[0], dtype=torch.bool, device=device)
        keep_np = keep_mask.detach().cpu().numpy()
        v_idx = v_idx[keep_np]
        if H_t is not None:
            H_t = H_t[keep_mask]

        # Allele frequency & minor-allele stats from RAW (imputed) G
        n_samp = y_t.shape[0]
        sum_g_over_05 = torch.where(G_t > 0.5, G_t, torch.zeros_like(G_t)).sum(dim=1)
        af_t = (G_t.sum(dim=1) / (2.0 * n_samp))
        ma_samples_t = torch.where(af_t <= 0.5, (G_t > 0.5).sum(dim=1), (G_t < 1.5).sum(dim=1)).to(torch.int32)
        ma_count_t = torch.where(af_t <= 0.5, sum_g_over_05, 2 * n_samp - sum_g_over_05)
 
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


def map_cis_nominal(
        genotype_df: pd.DataFrame, variant_df: pd.DataFrame, phenotype_df: pd.DataFrame,
        phenotype_pos_df: pd.DataFrame, covariates_df: Optional[pd.DataFrame] = None,
        haplotype_reader: Optional[object] = None, window: int = 1_000_000,
        nperm: Optional[int] = None, device: str = "cuda",
) -> pd.DataFrame:
    """
    Nominal cis-QTL scan with optional permutations and local ancestry.

    Adjusts for covariates by residualizing y, G, and H across samples using the
    same Residualizer (projection onto the orthogonal complement of C).
    """
    device = device if device in ("cuda", "cpu") else ("cuda" if torch.cuda.is_available() else "cpu")

    # Build the appropriate input generator
    if haplotype_reader is not None and getattr(haplotype_reader, "haplotypes", None) is not None:
        ig = InputGeneratorCisWithHaps(
            genotype_df=genotype_df,
            variant_df=variant_df,
            phenotype_df=phenotype_df,
            phenotype_pos_df=phenotype_pos_df,
            window=window,
            haplotypes=haplotype_reader.haplotypes,  # (variants x samples x ancestries)
            loci_df=getattr(haplotype_reader, "loci_df", None),
        )
        with_haps = True
    else:
        ig = InputGeneratorCis(
            genotype_df=genotype_df,
            variant_df=variant_df,
            phenotype_df=phenotype_df,
            phenotype_pos_df=phenotype_pos_df,
            window=window,
        )
        with_haps = False

    # Residualize all phenotypes once (features x samples)
    Y = torch.tensor(ig.phenotype_df.values, dtype=torch.float32, device=device)  # (n_pheno x n_samples)
    Y_resid, rez = _residualize_matrix_with_covariates(Y, covariates_df, device)
    # Put residuals back so the generator yields the same ordering/IDs
    phenotype_df_resid = pd.DataFrame(
        Y_resid.cpu().numpy(),
        index=ig.phenotype_df.index,
        columns=ig.phenotype_df.columns,
    )
    ig.phenotype_df = phenotype_df_resid
    
    return _run_nominal_core(ig, variant_df, rez, nperm, device)


class SimpleCisMapper:
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
            rez: Optional[Residualizer] = None,
    ):
        self.device = ("cuda" if (device == "auto" and torch.cuda.is_available()) else
                       device if device in ("cuda", "cpu") else "cpu")
        self.variant_df = variant_df
        self.window = window

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

    def map_nominal(self, nperm: Optional[int] = None) -> pd.DataFrame:
        return _run_nominal_core(self.ig, self.variant_df, self.rez, nperm, self.device)

    def map_permutations(self, nperm=1000, window=1_000_000):
        """Permutation-based empirical cis-QTLs"""
        # same loop, but shuffle phenotype each time and record max r2
        pass

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
        k_eff = rez.Q_t.shape[1] if rez is not None else 0
        p = X.shape[-1]
        dof = X.shape[0] - k_eff - p
        sigma2 = (resid.transpose(1,2) @ resid).squeeze() / dof
        se = torch.sqrt(torch.diag(XtX_inv) * sigma2)
        tstats = betas / se
        return tstats, betas, se
