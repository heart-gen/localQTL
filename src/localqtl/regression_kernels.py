import torch

def run_batch_regression(y, G, H=None, device="cuda"):
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
    dof = n - p
    sigma2 = (resid.transpose(1,2) @ resid).squeeze() / dof  # (m,)

    # Standard errors: sqrt(diag(XtX^-1) * sigma2)
    XtX_inv = torch.linalg.inv(XtX)                   # (m × p × p)
    var_betas = XtX_inv * sigma2.view(-1,1,1)         # broadcast sigma2
    ses = torch.sqrt(torch.diagonal(var_betas, dim1=1, dim2=2))  # (m × p)

    # T-statistics
    tstats = betas / ses

    return betas, ses, tstats


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
