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


def nominal_pvals_tensorqtl(
        y_t: torch.Tensor, G_resid: torch.Tensor, H_resid: torch.Tensor | None,
        k_eff: int, tstats: torch.Tensor | None = None, use_torch_cdf: bool = True,
        return_t: bool = False,
):
    """
    Compute tensorQTL-style nominal p-values per variant for a single phenotype window.

    - If H_resid is None: use correlation-based t (t = r * sqrt(dof / (1 - r^2)),
      dof = n - k_eff - 2).
    - Else: use OLS t-stat for genotype coefficient (col 0) with
      dof = n - k_eff - p, where p = 1 + (K-1).

    Returns:
      pvals_torch (m,), optionally t_used (m,) and dof (int).
    """
    assert y_t.ndim == 1, "y_t must be (n,)"
    assert G_resid.ndim == 2, "G_resid must be (m, n)"

    n = y_t.shape[0]
    device = y_t.device

    if H_resid is None:
        # correlation-based
        # center (rez.transform(center=True) should already do it, but be defensive)
        y_c = y_t - y_t.mean()
        G_c = G_resid - G_resid.mean(dim=1, keepdim=True)

        # r per variant
        num = torch.mv(G_c, y_c)
        den = (torch.linalg.norm(G_c, dim=1) * torch.linalg.norm(y_c)).clamp_min(1e-12)
        r = num / den
        r2 = (r * r).clamp(max=1 - 1e-12)

        dof = max(int(n) - int(k_eff) - 2, 1)
        t_used = r * torch.sqrt(torch.tensor(dof, dtype=r.dtype, device=device) / (1.0 - r2))
    else:
        # multi-predictor: use OLS t for genotype coeff (col 0)
        assert tstats is not None and tstats.ndim == 2 and tstats.shape[1] >= 1, \
            "tstats (m, p) with genotype in col 0 is required when H_resid is not None"
        p_pred = 1 + H_resid.shape[2]  # genotype + (K-1)
        dof = max(int(n) - int(k_eff) - int(p_pred), 1)
        t_used = tstats[:, 0]

    if use_torch_cdf:
        # two-sided p = 2 * CDF_t(-|t|)
        dist = torch.distributions.StudentT(df=float(dof))
        pvals = 2.0 * dist.cdf(-t_used.abs())
    else:
        pvals_np = get_t_pval(t_used.detach().cpu().numpy(), dof, log=False)
        pvals = torch.from_numpy(np.asarray(pvals_np)).to(device=device, dtype=t_used.dtype)

    return (pvals, t_used, dof) if return_t else (pvals, dof)
