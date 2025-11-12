import torch
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
    "perm_chunk_r2", "prep_ctx_for_perm",
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
            self.k_eff = C_t.shape[1]
        else:
            self.dof = n_samples - 2 - self.rank
            self.k_eff = self.rank

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
        H = move_to_device(H, device)  # (m × n × (k-1))
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


def _is_cuda(*tensors):
    return all(getattr(t, "is_cuda", False) for t in tensors if t is not None)


@torch.no_grad()
def run_batch_regression_with_permutations(
        y: torch.Tensor, G: torch.Tensor, H: torch.Tensor | None = None,
        y_perm: torch.Tensor | None = None, k_eff: int = 0, device: str = "cuda",
):
    """
    Return nominal betas/ses/tstats for G~y, and per-permutation max r^2 across variants.
    No (chunk x chunk) allocations; memory scales with (m*n + (m+n)*chunk).
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
        beta   = Gy / Gnorm2                         # (m,)
        se     = torch.abs(beta) / torch.clamp(t.float(), min=1e-8)

        betas  = beta.unsqueeze(1)                   # (m,1)
        ses    = se.unsqueeze(1)                     # (m,1)
        tstats = t.unsqueeze(1)                      # (m,1)

        # Permutations (chunked)
        r2_perm = None
        if y_perm is not None:
            Ypnorm2 = (y_perm * y_perm).sum(dim=0) + EPS
            if _is_cuda(G, y_perm):
                try:
                    with torch.amp.autocast("cuda", dtype=torch.float16):
                        GYp = G @ y_perm             # (m, chunk)
                except Exception:
                    # older Torch
                    with torch.cuda.amp.autocast(dtype=torch.float16):
                        GYp = G @ y_perm
            else:
                GYp = G @ y_perm

            GYp = GYp.float()
            denom = torch.sqrt(Gnorm2).unsqueeze(1) * torch.sqrt(Ypnorm2).unsqueeze(0)
            R2 = (GYp / torch.clamp(denom, min=EPS)) ** 2
            r2_vals, _ = _max_ignore_nan(R2, dim=0)
            r2_perm = r2_vals.float()

        return betas, ses, tstats, r2_perm
    else:
        # Gauss-Markov Projection (with H covariates)
        H = move_to_device(H, device)
        assert H.dim() == 3, "H must be (m, n, pH)"
        m, n, pH = H.shape

        dof = max(n - (1 + pH) - int(k_eff), 1)

        Gy = (G @ y)                             # (m,)
        G2 = (G * G).sum(dim=1) + EPS            # (m,)
        y2 = (y * y).sum() + EPS                 # scalar
        Hy = torch.einsum("mnp,n->mp", H, y)     # (m, pH)
        Hg = torch.einsum("mnp,mn->mp", H, G)    # (m, pH)

        # Precompute common variables
        with (torch.amp.autocast("cuda", dtype=torch.float16) if _is_cuda(G, H) else torch.no_grad()):
            # S = H^T H  (m, pH, pH)
            S = torch.einsum("mnp,mnq->mpq", H, H).to(torch.float32)

        # Cholesky with gentle regularization
        I = torch.eye(pH, device=device, dtype=torch.float32).expand_as(S)
        L, info = torch.linalg.cholesky_ex(S)
        if (info != 0).any():
            S = S + (1e-6 * I)
            L = torch.linalg.cholesky(S)

        # Solve S * alpha = RHS in batch
        alpha_y = torch.cholesky_solve(Hy.unsqueeze(-1), L).squeeze(-1)
        alpha_g = torch.cholesky_solve(Hg.unsqueeze(-1), L).squeeze(-1)

        # Norms after projecting out H (variant-specific!)
        y_resid2 = (y2 - (Hy * alpha_y).sum(dim=1)) + EPS        # (m,)
        G_resid2 = (G2 - (Hg * alpha_g).sum(dim=1)) + EPS        # (m,)

        # Cross term after projection: <G_resid, y_resid> = Gy - Hg^T alpha_y
        Gy_resid = (Gy - (Hg * alpha_y).sum(dim=1))              # (m,)

        # Nominal partial correlation -> t, beta, se
        r = Gy_resid / torch.sqrt(G_resid2 * y_resid2)
        r = torch.clamp(r, min=-1 + 1e-7, max=1 - 1e-7)
        t = r * torch.sqrt(torch.tensor(dof, device=device, dtype=torch.float32) /
                           torch.clamp(1.0 - r * r, min=EPS))
        beta = Gy_resid / G_resid2
        se   = torch.abs(beta) / torch.clamp(t, min=1e-8)

        betas  = beta.view(m, 1)
        ses    = se.view(m, 1)
        tstats = t.view(m, 1)

        # Permutations
        r2_perm = None
        if y_perm is not None:
            y_perm = y_perm.to(device, non_blocking=True)        # (n, chunk)
            Ypnorm2 = (y_perm * y_perm).sum(dim=0) + EPS         # (chunk,)
            U = torch.einsum("mnp,nc->mpc", H, y_perm).to(torch.float32)
            alpha_perm = torch.cholesky_solve(U, L)
            T0 = (G @ y_perm).to(torch.float32)
            T1 = torch.einsum("mp,mpc->mc", Hg, alpha_perm)
            # Numerator dot after projection
            GiYp_proj = T0 - T1                                  # (m, chunk)
            Yp_resid2 = (Ypnorm2.unsqueeze(0)
                         - (U * alpha_perm).sum(dim=1)) + EPS    # (m, chunk)
            denom = torch.sqrt(G_resid2).unsqueeze(1) * torch.sqrt(Yp_resid2)
            R2 = (GiYp_proj / torch.clamp(denom, min=EPS)) ** 2  # (m, chunk)
            # max across variants for each permutation
            r2_perm = torch.nan_to_num(R2, nan=0.0).amax(dim=0).to(torch.float32)

        return betas, ses, tstats, r2_perm


@torch.no_grad()
def prep_ctx_for_perm(y: torch.Tensor, G: torch.Tensor, H: torch.Tensor, k_eff: int):
    """
    Precompute all variant-wise quantities for permutations when H is present.
    Returns a ctx dict with Cholesky factor and cached pieces, plus nominal stats.
    Shapes: y(n,), G(m,n), H(m,n,pH)
    """
    device = y.device
    m, n, pH = H.shape
    EPS = 1e-8
    dof = max(n - (1 + pH) - int(k_eff), 1)

    S = torch.einsum("mnp,mnq->mpq", H, H).to(torch.float32)
    I = torch.eye(pH, device=device, dtype=torch.float32).expand_as(S)
    L, info = torch.linalg.cholesky_ex(S)
    if (info != 0).any():
        S = S + (1e-6 * I)
        L = torch.linalg.cholesky(S)

    Gy = (G @ y)
    G2 = (G * G).sum(dim=1) + EPS
    y2 = (y * y).sum() + EPS
    Hy = torch.einsum("mnp,n->mp", H, y)
    Hg = torch.einsum("mnp,mn->mp", H, G)

    alpha_y = torch.cholesky_solve(Hy.unsqueeze(-1), L).squeeze(-1)
    alpha_g = torch.cholesky_solve(Hg.unsqueeze(-1), L).squeeze(-1)

    y_resid2 = (y2 - (Hy * alpha_y).sum(dim=1)) + EPS
    G_resid2 = (G2 - (Hg * alpha_g).sum(dim=1)) + EPS
    Gy_resid = (Gy - (Hg * alpha_y).sum(dim=1))

    # nominal stats
    r = Gy_resid / torch.sqrt(G_resid2 * y_resid2)
    r = torch.clamp(r, min=-1 + 1e-7, max=1 - 1e-7)
    t = r * torch.sqrt(torch.tensor(dof, device=device, dtype=torch.float32) /
                       torch.clamp(1.0 - r * r, min=EPS))
    beta = Gy_resid / G_resid2
    se = torch.abs(beta) / torch.clamp(t, min=1e-8)

    ctx = {
        "L": L, "Hg": Hg, "G_resid2": G_resid2, "dof": dof,
    }
    betas  = beta.view(m, 1)
    ses    = se.view(m, 1)
    tstats = t.view(m, 1)
    return ctx, betas, ses, tstats


@torch.no_grad()
def perm_chunk_r2(ctx: dict, H: torch.Tensor, G: torch.Tensor, y_perm: torch.Tensor) -> torch.Tensor:
    """
    Compute max R^2 across variants for a permutation chunk (n x chunk),
    using precomputed ctx from prep_ctx_for_perm.
    Returns (chunk,) on the same device.
    """
    EPS = 1e-8
    L, Hg, G_resid2 = ctx["L"], ctx["Hg"], ctx["G_resid2"]

    # U = H^T y_perm: (m,pH,chunk); Alpha solves S*Alpha = U via Cholesky
    U = torch.einsum("mnp,nc->mpc", H, y_perm).to(torch.float32)
    _alpha = torch.cholesky_solve(U, L)

    T0 = (G @ y_perm).to(torch.float32)
    T1 = torch.einsum("mp,mpc->mc", Hg, _alpha)
    GiYp_proj = T0 - T1

    Ypnorm2 = (y_perm * y_perm).sum(dim=0) + EPS
    Yp_resid2 = (Ypnorm2.unsqueeze(0) - (U * _alpha).sum(dim=1)) + EPS

    R2 = (GiYp_proj ** 2) / (G_resid2.unsqueeze(1) * Yp_resid2)
    return torch.nan_to_num(R2, nan=0.0).amax(dim=0).to(torch.float32)
