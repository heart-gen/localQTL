import torch
import pandas as pd

from .haplotypeio import InputGeneratorCis

def map_cis_nominal(genotype_df, variant_df, phenotype_df, phenotype_pos_df,
                    covariates_df=None, haplotype_reader=None, window=1_000_000,
                    nperm=None, device="cuda"):
    """
    Run nominal (and optionally permutation) cis-QTL mapping.
    """
    # Residualize phenotypes once
    if covariates_df is not None:
        residualizer = Residualizer(torch.tensor(covariates_df.values, dtype=torch.float32).to(device))
        Y_resid = residualize_phenotypes(phenotype_df.values, covariates_df.values, device=device)
        phenotype_df_resid = pd.DataFrame(Y_resid, index=phenotype_df.index, columns=phenotype_df.columns)
    else:
        phenotype_df_resid = phenotype_df

    # Create cis-window generator
    igc = InputGeneratorCis(genotype_df, variant_df,
                            phenotype_df_resid, phenotype_pos_df,
                            window=window, haplotype_reader=haplotype_reader)

    results = []
    for pheno_id, y, G_block, var_df, H_block in igc.generate_data():
        y_t = torch.tensor(y, dtype=torch.float32, device=device)
        G_t = torch.tensor(G_block, dtype=torch.float32, device=device)
        H_t = torch.tensor(H_block, dtype=torch.float32, device=device) if H_block is not None else None

        # Permuted phenotypes (if requested)
        if nperm is not None:
            y_perm = torch.stack([y_t[torch.randperm(len(y_t))] for _ in range(nperm)], dim=1)
        else:
            y_perm = None

        # Run batched regression
        betas, ses, tstats, r2_perm = run_batch_regression_with_permutations(
            y_t, G_t, H=H_t, y_perm=y_perm, device=device
        )

        df = pd.DataFrame({
            "phenotype_id": pheno_id,
            "variant_id": var_df.index.values,
            "tstat": tstats[:,0].cpu().numpy(),
            "beta": betas[:,0].cpu().numpy(),
            "se": ses[:,0].cpu().numpy(),
            "pos": var_df["pos"].values,
        })

        if nperm is not None:
            df["max_r2_perm"] = r2_perm.cpu().numpy().max()  # store empirical null

        results.append(df)

    return pd.concat(results, axis=0).reset_index(drop=True)


class SimpleCisMapper:
    def __init__(self, genotype_reader, phenotype_df, phenotype_pos_df,
                 covariates_df=None, haplotypes=None, device="auto"):
        # genotype_reader: like tensorQTL genotypeio.InputGeneratorCis
        self.geno_reader = genotype_reader
        self.phenotype_df = phenotype_df
        self.phenotype_pos_df = phenotype_pos_df
        self.device = "cuda" if device=="auto" and torch.cuda.is_available() else "cpu"

        # residualize phenotypes
        if covariates_df is not None:
            self.residualizer = Residualizer(torch.tensor(covariates_df.values, dtype=torch.float32).to(self.device))
            self.Y = self._residualize(phenotype_df.values, covariates_df.values)
        else:
            self.residualizer = None
            self.Y = phenotype_df.values

        # handle haplotypes (drop one ancestry col)
        if haplotypes is not None:
            self.H = haplotypes[:, :, :-1]  # (variants × samples × n_pops-1)
        else:
            self.H = None

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


import torch

def run_batch_regression_with_permutations(y, G, H=None, y_perm=None, device="cuda"):
    """
    Batched OLS regression for one phenotype across all variants in a cis-window,
    with optional haplotype ancestry and phenotype permutations.

    Parameters
    ----------
    y : torch.Tensor
        (n,) phenotype vector (samples)
    G : torch.Tensor
        (m × n) genotype matrix (variants × samples)
    H : torch.Tensor, optional
        (m × n × (k-1)) haplotype ancestry matrix (variants × samples × ancestries-1)
    y_perm : torch.Tensor, optional
        (n × nperm) permuted phenotype matrix
    device : str
        "cuda" or "cpu"

    Returns
    -------
    betas : torch.Tensor
        (m × p) regression coefficients (for true phenotype)
    ses : torch.Tensor
        (m × p) standard errors (for true phenotype)
    tstats : torch.Tensor
        (m × p) t-statistics (for true phenotype)
    r2_perm : torch.Tensor, optional
        (nperm,) max R² across variants for each permutation
    """
    y = y.to(device)
    G = G.to(device)
    if y_perm is not None:
        y_perm = y_perm.to(device)

    n = y.shape[0]

    # Expand y across variants: (m × n × 1)
    y_exp = y.unsqueeze(0).expand(G.shape[0], -1).unsqueeze(-1)

    # Build design matrix
    G_exp = G.unsqueeze(-1)  # (m × n × 1)
    if H is not None:
        H = H.to(device)  # (m × n × (k-1))
        X = torch.cat([G_exp, H], dim=2)  # (m × n × p)
    else:
        X = G_exp
    m, n, p = X.shape

    # Compute XtX, Xty (true phenotype)
    XtX = torch.matmul(X.transpose(1, 2), X)      # (m × p × p)
    Xty = torch.matmul(X.transpose(1, 2), y_exp)  # (m × p × 1)

    # Solve for betas
    betas = torch.linalg.solve(XtX, Xty).squeeze(-1)  # (m × p)

    # Residuals and variance
    y_hat = torch.matmul(X, betas.unsqueeze(-1))      # (m × n × 1)
    resid = y_exp - y_hat
    dof = n - p
    sigma2 = (resid.transpose(1,2) @ resid).squeeze() / dof  # (m,)

    # SEs and t-stats
    XtX_inv = torch.linalg.inv(XtX)                       # (m × p × p)
    var_betas = XtX_inv * sigma2.view(-1,1,1)             # (m × p × p)
    ses = torch.sqrt(torch.diagonal(var_betas, dim1=1, dim2=2))  # (m × p)
    tstats = betas / ses

    # --- Permutations ---
    r2_perm = None
    if y_perm is not None:
        # y_perm: (n × nperm) -> (1 × n × nperm), then expand across variants
        y_perm_exp = y_perm.unsqueeze(0).expand(m, -1, -1)  # (m × n × nperm)

        # Project genotypes onto y_perm
        # Compute betas for each permutation
        Xty_perm = torch.matmul(X.transpose(1,2), y_perm_exp)   # (m × p × nperm)
        betas_perm = torch.linalg.solve(XtX, Xty_perm)          # (m × p × nperm)

        # Predicted values
        y_hat_perm = torch.matmul(X, betas_perm)                # (m × n × nperm)
        resid_perm = y_perm_exp - y_hat_perm

        sigma2_perm = (resid_perm.transpose(1,2) @ resid_perm)  # (m × nperm × nperm) too big!
        # Instead: use correlation-based shortcut: R² = var(Xb) / var(y)
        # Here we compute R² for genotype effect only (first column of betas_perm)
        b_g = betas_perm[:,0,:]             # (m × nperm)
        # variance explained by genotype predictor
        y_g_hat = G.unsqueeze(-1) * b_g.unsqueeze(1)  # (m × n × nperm)
        r2_perm = (y_g_hat.var(dim=1) / y_perm.var(dim=0)).max(dim=0).values  # (nperm,)

    return betas, ses, tstats, r2_perm

