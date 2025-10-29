"""
Statistical helpers (kept separate to avoid pulling SciPy into low-level kernels).
"""
from __future__ import annotations
import torch
import numpy as np
from scipy import stats

def beta_approx_pval(r2_perm: np.ndarray, r2_true: float) -> tuple[float, float, float]:
    """
    Fit Beta(a,b) to permutation R^2 by method of moments and return:
        (pval_beta, a_hat, b_hat).
    Falls back to empirical tail if variance is ~0 or invalid.
    """
    r2_perm = np.asarray(r2_perm, dtype=np.float64)
    if r2_perm.size == 0 or not np.isfinite(r2_perm).all():
        # Degenerate input: return NA-like outputs
        return float("nan"), float("nan"), float("nan")

    mu = r2_perm.mean()
    var = r2_perm.var(ddof=1)
    if not np.isfinite(mu) or not np.isfinite(var) or var <= 1e-12:
        # Use empirical tail as a safe fallback
        p_emp = (np.sum(r2_perm >= r2_true) + 1) / (r2_perm.size + 1)
        return float(p_emp), float("nan"), float("nan")

    # Method-of-moments for Beta
    k = mu * (1.0 - mu) / var - 1.0
    a = max(mu * k, 1e-6)
    b = max((1.0 - mu) * k, 1e-6)

    p_beta = 1.0 - stats.beta.cdf(r2_true, a, b)
    return float(p_beta), float(a), float(b)


def get_t_pval(t, df, two_tailed: bool = True, log10: bool = False):
    """
    Numerically stable p-values for Student's t.
    t, df can be scalars or arrays (NumPy or torch). If log10=True, return -log10(p).
    """
    try: # Accept torch
        if isinstance(t, torch.Tensor):
            t = t.detach().cpu().numpy()
        if isinstance(df, torch.Tensor):
            df = df.detach().cpu().numpy()
    except Exception:
        pass

    t = np.asarray(t, dtype=np.float64)
    df = np.asarray(df, dtype=np.float64)

    # Compute one- or two-tailed p-value
    if log10:
        logp = stats.t.logsf(np.abs(t), df)
        if two_tailed:
            logp = np.log(2.0) + logp
        return -(logp / np.log(10.0))
    else:
        p = stats.t.sf(np.abs(t), df)
        p = 2.0 * p if two_tailed else p
        return p


def t_two_sided_pval_torch(t_abs: torch.Tensor, dof: int | torch.Tensor) -> torch.Tensor:
    """
    Two-sided p-value for t-statistics using regularized incomplete beta:
        p = I_{ν/(ν + t^2)}(ν/2, 1/2)
    Works fully on GPU. Falls back to CPU/NumPy if torch.special.betainc is unavailable.
    """
    # Ensure tensors with good precision on the same device as t_abs
    dev = t_abs.device
    dtype_work = torch.float64
    t2  = t_abs.to(dtype_work)**2
    nu  = (torch.as_tensor(dof, device=dev, dtype=dtype_work)
           .expand_as(t_abs.to(dtype_work)))
    x   = (nu / (nu + t2)).clamp(0.0, 1.0)            # ν/(ν+t^2)
    a   = 0.5 * nu                                    # ν/2
    b   = torch.full_like(a, 0.5)                     # 1/2

    try:
        p = torch.special.betainc(a, b, x)            # regularized I_x(a,b)
        return p.to(t_abs.dtype)
    except Exception: # Fallback (CPU)
        p_np = get_t_pval(t_abs.detach().cpu().numpy(), dof=int(nu.flatten()[0].item()), log=False)
        return torch.as_tensor(p_np, device=dev, dtype=t_abs.dtype)


def nominal_pvals_tensorqtl(
        y_t: torch.Tensor, G_resid: torch.Tensor, H_resid: torch.Tensor | None,
        k_eff: int, tstats: torch.Tensor,
):
    """
    TensorQTL-style nominal p-values.
    - If H is None: use correlation route (dof = n - k_eff - 2) to match tensorQTL.
    - Else: use the genotype-column t-stat (column 0) with dof = n - k_eff - p.
    Returns (pvals, dof_used)
    """
    n = y_t.shape[0]
    dev = y_t.device
    dt  = y_t.dtype

    if H_resid is None:
        # correlation-based t -> p (tensorQTL uses n - k_eff - 2)
        dof_used = max(n - int(k_eff) - 2, 1)
        y_c = y_t - y_t.mean()
        G_c = G_resid - G_resid.mean(dim=1, keepdim=True)
        num = torch.mv(G_c, y_c)                                   # (m,)
        den = (G_c.norm(dim=1) * y_c.norm()).clamp_min(1e-30)
        r   = (num / den).clamp(-1.0 + 1e-12, 1.0 - 1e-12)
        r2  = r * r
        t_abs = (r.abs() * torch.sqrt(torch.tensor(dof_used, device=dev, dtype=dt)
                                      / (1.0 - r2))).to(dt)
        pvals = t_two_sided_pval_torch(t_abs.abs(), dof_used)
        return pvals, dof_used
    else:
        # multi-predictor: use genotype t-stat (column 0)
        p_pred = 1 + H_resid.shape[2]
        dof_used = max(n - int(k_eff) - int(p_pred), 1)
        t_abs = tstats[:, 0].abs()
        pvals = t_two_sided_pval_torch(t_abs, dof_used)
        return pvals, dof_used
