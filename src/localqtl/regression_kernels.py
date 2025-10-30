import torch

__all__ = [
    "Residualizer",
    "run_batch_regression",
    "run_batch_regression_with_permutations",
]

class Residualizer(object):
    """
    Residualizer for regressing out covariates from genotype/phenotype matrices.
    """
    def __init__(self, C_t: torch.Tensor):
        # Center and orthongonalize covariates
        C_t = C_t - C_t.mean(0)
        self.Q_t, _ = torch.linalg.qr(C_t, mode='reduced')
        self.P = self.Q_t @ self.Q_t.T
        self.dof = C_t.shape[0] - 2 - C_t.shape[1]

    def transform(self, *matrices: torch.Tensor, center: bool=True
                  ) -> tuple[torch.Tensor]:
        """
        Residualize one or more matrices in a single GPU pass.
        """
        dev = self.Q_t.device
        matrices = tuple(M.to(dev) for M in matrices)
        if len(matrices) == 1:
            M_t = matrices[0]
            if center:
                M_t = M_t - M_t.mean(1, keepdim=True)
            return (M_t - M_t @ self.P,)

        # Concatenate features along rows
        M_cat = torch.cat(matrices, dim=0)
        if center:
            M_cat = M_cat - M_cat.mean(1, keepdim=True)

        # Project once with cached P
        M_cat_resid = M_cat - M_cat @ self.P

        # Split back into original blocks
        out = []
        start = 0
        for M in matrices:
            end = start + M.shape[0]
            out.append(M_cat_resid[start:end])
            start = end
        return tuple(out)

    def check_orthogonality(self, M_t: torch.Tensor, atol: float = 1e-6) -> float:
        """
        Check maximum absolute correlation between residualized matrix and covariates.
        """
        # Residualize
        M_resid = self.transform(M_t, center=True)[0]

        # Project residuals onto Q
        proj = M_resid @ self.Q_t
        max_corr = proj.abs().max().item()

        if max_corr > atol:
            print(f"Warning: residuals not fully orthogonal (max={max_corr:.2e})")
        return max_corr


def run_batch_regression(y, G, H=None, k_eff: int = 0, device="cuda"):
    """
    Batched OLS regression for one phenotype and all variants in a cis-window.

    Parameters
    ----------
    y : torch.Tensor
        (n,) phenotype vector (samples)
    G : torch.Tensor
        (m × n) genotype matrix (variants × samples)
    H : torch.Tensor, optional
        (m × n × (k-1)) haplotype ancestry matrix (variants × samples × ancestries-1)
    k_eff : int, optional
        Number of covariate columns projected out beforehand (effective dof reduction).
    device : str
        "cuda" or "cpu"

    Returns
    -------
    betas : torch.Tensor
        (m × p) regression coefficients (per variant, per predictor)
    ses : torch.Tensor
        (m × p) standard errors
    tstats : torch.Tensor
        (m × p) t-statistics
    """
    y = y.to(device)
    G = G.to(device)

    n = y.shape[0]

    # Expand y across variants for batching: (m × n × 1)
    y_exp = y.unsqueeze(0).expand(G.shape[0], -1).unsqueeze(-1)

    # Build design matrix X for each variant
    # G -> (m × n × 1)
    G_exp = G.unsqueeze(-1)

    if H is not None:
        H = H.to(device)  # (m × n × (k-1))
        X = torch.cat([G_exp, H], dim=2)  # (m × n × p)
    else:
        X = G_exp  # (m × n × 1)

    m, n, p = X.shape

    # Compute XtX and Xty in batch
    XtX = torch.matmul(X.transpose(1, 2), X)      # (m × p × p)
    Xty = torch.matmul(X.transpose(1, 2), y_exp)  # (m × p × 1)

    # Solve for betas
    betas = torch.linalg.solve(XtX, Xty).squeeze(-1)  # (m × p)

    # Residuals and variance estimate
    y_hat = torch.matmul(X, betas.unsqueeze(-1))      # (m × n × 1)
    resid = y_exp - y_hat                             # (m × n × 1)
    dof = n - int(k_eff) - p
    sigma2 = (resid.transpose(1,2) @ resid).squeeze() / dof  # (m,)

    # Standard errors: sqrt(diag(XtX^-1) * sigma2)
    XtX_inv = torch.linalg.inv(XtX)                   # (m × p × p)
    var_betas = XtX_inv * sigma2.view(-1,1,1)         # broadcast sigma2
    ses = torch.sqrt(torch.diagonal(var_betas, dim1=1, dim2=2))  # (m × p)

    # T-statistics
    tstats = betas / ses

    return betas, ses, tstats


def run_batch_regression_with_permutations(y, G, H=None, y_perm=None, k_eff: int = 0, device="cuda"):
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
    k_eff : int, optional
        Number of covariate columns projected out beforehand (effective dof reduction).
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
    dof = n - int(k_eff) - p
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
