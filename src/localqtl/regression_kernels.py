import torch, warnings
from ._torch_utils import move_to_device, resolve_device

try:
    torch.backends.cuda.matmul.fp32_precision = 'tf32'
    torch.backends.cudnn.conv.fp32_precision = 'tf32'
except Exception:
    pass

__all__ = [
    "Residualizer",
    "run_batch_regression",
    "run_batch_regression_with_permutations",
]

class Residualizer(object):
    """
    Residualizer for regressing out covariates from genotype/phenotype matrices.
    """
    def __init__(self, C_t: torch.Tensor, tensorqtl_flavor: bool = False):
        self.tensorqtl_flavor = tensorqtl_flavor
        # Center covariates and drop columns with zero variance (e.g., intercept-only)
        n_samples = C_t.shape[0]
        C_centered = C_t - C_t.mean(0)
        norms = torch.linalg.norm(C_centered, dim=0)
        keep = norms > 0

        if keep.any():
            C_use = C_centered[:, keep]
            self.Q_t, _ = torch.linalg.qr(C_use, mode='reduced')
            self.P = self.Q_t @ self.Q_t.T
            self.rank = int(self.Q_t.shape[1])
        else:
            # No informative covariates remain; act as identity residualizer.
            self.Q_t = C_t.new_zeros((n_samples, 0))
            self.P = None
            self.rank = 0

        if tensorqtl_flavor:
            self.dof = n_samples - 2 - C_t.shape[1]
            self.k_eff = self.rank
        else:
            self.dof = n_samples - 2 - self.rank
            self.k_eff = C_t.shape[1]

        if tensorqtl_flavor and self.rank < C_t.shape[1]:
            print(f"[warning] Covariate matrix has {C_t.shape[1] - self.rank} "
                  "constant or colinear columns. "
                  "tensorQTL flavor uses full column count.")

    def transform(self, *matrices: torch.Tensor, center: bool=True
                  ) -> tuple[torch.Tensor]:
        """
        Residualize one or more matrices in a single GPU pass.
        """
        dev = resolve_device(self.Q_t.device if self.Q_t.numel() else matrices[0].device)
        matrices = tuple(move_to_device(M, dev) for M in matrices)
        if len(matrices) == 1:
            M_t = matrices[0]
            if center:
                M_t = M_t - M_t.mean(1, keepdim=True)
            if self.rank == 0:
                return (M_t,)
            return (M_t - M_t @ self.P,)

        # Concatenate features along rows
        M_cat = torch.cat(matrices, dim=0)
        if center:
            M_cat = M_cat - M_cat.mean(1, keepdim=True)

        if self.rank == 0:
            out = []
            start = 0
            for M in matrices:
                end = start + M.shape[0]
                out.append(M_cat[start:end])
                start = end
            return tuple(out)

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


def _max_ignore_nan(x: torch.Tensor, dim: int):
    """
    Torch <=1.12 doesn't have torch.nanmax.
    Replace NaNs with -inf then take max; return (values, indices).
    Columns that were all-NaN become -inf; we map those to 0.0.
    """
    x2 = torch.nan_to_num(x, nan=float("-inf"), posinf=float("inf"), neginf=float("-inf"))
    vals, idx = torch.max(x2, dim=dim)
    vals = torch.where(torch.isfinite(vals), vals, torch.zeros_like(vals))
    return vals, idx


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
    device = resolve_device(device)

    y = move_to_device(y, device)
    G = move_to_device(G, device)

    n = y.shape[0]

    # Expand y across variants for batching: (m × n × 1)
    y_exp = y.unsqueeze(0).expand(G.shape[0], -1).unsqueeze(-1)

    # Build design matrix X for each variant
    # G -> (m × n × 1)
    G_exp = G.unsqueeze(-1)

    if H is not None:
        H = move_to_device(H, device)     # (m × n × (k-1))
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


@torch.no_grad()
def run_batch_regression_with_permutations(
        y: torch.Tensor, G: torch.Tensor, H: torch.Tensor | None = None,
        y_perm: torch.Tensor | None = None, k_eff: int = 0, device: str = "cuda",
):
    """
    Efficient regression with optional covariates and permutation testing
    using Gauss-Markov projection trick.
    """
    device = resolve_device(device)
    y = move_to_device(y, device)
    G = move_to_device(G, device)
    if y_perm is not None:
        y_perm = y_perm.to(device)

    EPS = 1e-8
    n = y.shape[0]
    m = G.shape[0]

    if H is None:
        # Fast vectorized implementation (no H covariates)
        dof = max(n - 1 - int(k_eff), 1)
        Gnorm2 = (G * G).sum(dim=1) + EPS          # (m,)
        ynorm2 = (y * y).sum() + EPS               # scalar
        Gy     = G @ y                             # (m,)

        r      = Gy / torch.sqrt(Gnorm2 * ynorm2)  # (m,)
        one_minus_r2 = 1.0 - r * r
        t      = r * torch.sqrt(dof / torch.clamp(one_minus_r2, min=EPS))
        beta   = Gy / Gnorm2                       # (m,)
        se     = torch.abs(beta) / torch.clamp(t.float(), min=1e-8)

        betas  = beta.unsqueeze(1)                 # (m,1)
        ses    = se.unsqueeze(1)                   # (m,1)
        tstats = t.unsqueeze(1)                    # (m,1)

        # Permutations
        r2_perm = None
        if y_perm is not None:
            Ypnorm2 = (y_perm * y_perm).sum(dim=0) + EPS
            GYp   = G @ y_perm
            denom = torch.sqrt(Gnorm2).unsqueeze(1) * torch.sqrt(Ypnorm2).unsqueeze(0)
            R2    = (GYp / torch.clamp(denom, min=EPS)) ** 2
            r2_vals, _ = _max_ignore_nan(R2, dim=0)
            r2_perm = r2_vals.float()

        return betas, ses, tstats, r2_perm
    else:
        # Gauss-Markov Projection (with H covariates)
        H = move_to_device(H, device)

        betas = torch.empty((m, 1), device=device)
        ses = torch.empty((m, 1), device=device)
        tstats = torch.empty((m, 1), device=device)
        r2_perm = torch.zeros((y_perm.shape[1],), device=device) if y_perm is not None else None
        def project_out(v, Q):
            return v - Q @ (Q.T @ v)

        for i in range(m):
            Hi = H[i]
            Gi = G[i]
            Qi, _ = torch.linalg.qr(Hi)

            # Residualize y and G[i]
            y_resid  = project_out(y, Qi)
            Gi_resid = project_out(Gi, Qi)

            Gnorm2 = (Gi_resid ** 2).sum() + EPS
            ynorm2 = (y_resid ** 2).sum() + EPS
            Gy     = Gi_resid @ y_resid

            r    = Gy / torch.sqrt(Gnorm2 * ynorm2)
            one_minus_r2 = 1.0 - r * r            
            dof  = max(n - Qi.shape[1] - 1 - int(k_eff), 1)
            t    = r * torch.sqrt(dof / torch.clamp(one_minus_r2, min=EPS))
            beta = Gy / Gnorm2
            se   = torch.abs(beta) / torch.clamp(t.float(), min=1e-8)

            betas[i, 0]  = beta
            ses[i, 0]    = se
            tstats[i, 0] = t

            # Permutations
            if y_perm is not None:
                Yp_resid = project_out(y_perm, Qi)
                Ypnorm2  = (Yp_resid ** 2).sum(dim=0) + EPS
                GiYp     = Gi_resid @ Yp_resid
                denom    = torch.sqrt(Gnorm2 * Ypnorm2)
                R2 = (GiYp / torch.clamp(denom, min=EPS)) ** 2
                r2_vals, _ = _max_ignore_nan(R2.unsqueeze(0), dim=0)
                r2_perm  = torch.maximum(r2_perm, r2_vals.float())

        return betas, ses, tstats, r2_perm
