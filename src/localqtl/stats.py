"""
Statistical helpers (kept separate to avoid pulling SciPy into low-level kernels).
"""
from __future__ import annotations
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


def get_t_pval(t, df, log=False):
    """
    Get p-value corresponding to t statistic and degrees of freedom (df). t and/or df can be arrays.
    If log=True, returns -log10(P).
    """
    # Compute the two-tailed p-value
    p = 2 * stats.t.cdf(-abs(t), df)
    
    if log:
        p = np.maximum(p, np.finfo(float).tiny)
        return -np.log10(p)
    else:
        return p
