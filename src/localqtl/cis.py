import torch
import pandas as pd
from typing import Optional, Tuple, List

from .haplotypeio import InputGeneratorCis, InputGeneratorCisWithHaps
from .regression_kernels import (
    Residualizer,
    run_batch_regression,
    run_batch_regression_with_permutations
)

def _residualize_matrix_with_covariates(
    Y: torch.Tensor,           # (features x samples)
    C: Optional[pd.DataFrame], # covariates_df or None
    device: str
) -> Tuple[torch.Tensor, Optional[Residualizer]]:
    """
    Residualize (features x samples) matrix Y against covariates C across samples.
    Returns residualized Y and a Residualizer (or None if no covariates).
    """
    if C is None:
        return Y, None
    C_t = torch.tensor(C.values, dtype=torch.float32, device=device)  # (samples x k)
    rez = Residualizer(C_t)
    (Y_resid,) = rez.transform(Y, center=True)
    return Y_resid, rez


def _residualize_batch(
    y: torch.Tensor,  # (samples,)
    G: torch.Tensor,  # (m x samples)
    H: Optional[torch.Tensor],  # (m x samples x pH) or None
    rez: Optional[Residualizer],
    center: bool = True,
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


def map_cis_nominal(
    genotype_df: pd.DataFrame,      # variants x samples
    variant_df: pd.DataFrame,       # index=variant_id, columns ['chrom','pos']
    phenotype_df: pd.DataFrame,     # phenotypes x samples
    phenotype_pos_df: pd.DataFrame, # index=phenotype_id, columns ['chr','pos'] or ['chr','start','end']
    covariates_df: Optional[pd.DataFrame] = None,
    haplotype_reader: Optional[object] = None,  # expects attributes haplotypes, loci_df (RFMixReader)
    window: int = 1_000_000,
    nperm: Optional[int] = None,
    device: str = "cuda",
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
    
    results = []
    # Iterate phenotypes / groups
    for batch in ig.generate_data():
        # Unpack
        if with_haps:
            p, G_block, v_idx, H_block, pid = batch  # order as defined in InputGeneratorCisWithHaps._postprocess_batch
        else:
            p, G_block, v_idx, pid = batch
            H_block = None

        # Tensors (samples dimension is the last axis)
        y_t = torch.tensor(p, dtype=torch.float32, device=device)
        G_t = torch.tensor(G_block, dtype=torch.float32, device=device)
        H_t = torch.tensor(H_block, dtype=torch.float32, device=device) if H_block is not None else None

        # Residualize this batch (makes y/G/H orthogonal to covariates)
        y_t, G_t, H_t = _residualize_batch(y_t, G_t, H_t, rez, center=True)
        
        # Permuted phenotypes (if requested)
        if nperm is not None:
            perms = [torch.randperm(y_t.shape[0], device=device) for _ in range(nperm)]
            y_perm = torch.stack([y_t[idx] for idx in perms], dim=1)  # (n x nperm)
            betas, ses, tstats, r2_perm = run_batch_regression_with_permutations(
                y=y_t, G=G_t, H=H_t, y_perm=y_perm, device=device
            )
            perm_max_r2 = r2_perm.max().item()
        else:
            betas, ses, tstats = run_batch_regression(y=y_t, G=G_t, H=H_t, device=device)
            perm_max_r2 = None

        # Variant metadata for this window
        var_ids = variant_df.index.values[v_idx]
        var_pos = variant_df.iloc[v_idx]["pos"].values

        # Assemble result rows (genotype effect == column 0)
        out = {
            "phenotype_id": pid,
            "variant_id": var_ids,
            "pos": var_pos,
            "beta": betas[:, 0].detach().cpu().numpy(),
            "se": ses[:, 0].detach().cpu().numpy(),
            "tstat": tstats[:, 0].detach().cpu().numpy(),
        }
        df = pd.DataFrame(out)

        if perm_max_r2 is not None:
            df["perm_max_r2"] = perm_max_r2 

        results.append(df)

    return pd.concat(results, axis=0).reset_index(drop=True)


class SimpleCisMapper:
    """
    Convenience wrapper: build an InputGenerator and run nominal scans.
    """
    def __init__(
        self,
        genotype_df: pd.DataFrame,
        variant_df: pd.DataFrame,
        phenotype_df: pd.DataFrame,
        phenotype_pos_df: pd.DataFrame,
        covariates_df: Optional[pd.DataFrame] = None,
        haplotypes: Optional[object] = None,  # array-like (variants x samples x ancestries) or None
        loci_df: Optional[pd.DataFrame] = None,
        device: str = "auto",
        window: int = 1_000_000,
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

    def map_nominal(self, window=1_000_000):
        """Nominal cis-QTL scan"""
        results = []
        for pheno_id, pheno_vec, geno_block, var_df in self.geno_reader.generate_data(window=window):
            pheno_t = torch.tensor(pheno_vec, dtype=torch.float32, device=self.device)
            geno_t = torch.tensor(geno_block, dtype=torch.float32, device=self.device)
            impute_mean(geno_t)

            if self.H is not None:
                hap_t = torch.tensor(self.H[var_df.index.values], device=self.device)
                # concatenate genotype+haplotype as predictors
                X = torch.cat([geno_t.unsqueeze(1), hap_t], dim=1)
            else:
                X = geno_t.unsqueeze(1)

            # fast correlation method
            tstat, slope, se = self._regress(X, pheno_t)
            df = pd.DataFrame({
                "variant_id": var_df.index,
                "phenotype_id": pheno_id,
                "tstat": tstat.cpu().numpy(),
                "slope": slope.cpu().numpy(),
                "slope_se": se.cpu().numpy(),
            })
            results.append(df)
        return pd.concat(results)

    def map_permutations(self, nperm=1000, window=1_000_000):
        """Permutation-based empirical cis-QTLs"""
        # same loop, but shuffle phenotype each time and record max r2
        pass

    def map_independent(self, cis_df, fdr=0.05):
        """Forward–backward conditional mapping"""
        pass

    # --- helper functions ---
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
        dof = X.shape[0] - X.shape[1]
        sigma2 = (resid @ resid) / dof
        se = torch.sqrt(torch.diag(XtX_inv) * sigma2)
        tstats = betas / se
        return tstats, betas, se
