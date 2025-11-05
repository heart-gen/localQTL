"""
Statistical helpers (kept separate to avoid pulling SciPy into low-level kernels).
Adapted from tensorqtl:
  - https://github.com/broadinstitute/tensorqtl/blob/master/tensorqtl/core.py
  - https://github.com/broadinstitute/tensorqtl/blob/master/tensorqtl/post.py
"""
from __future__ import annotations
import torch
import scipy
import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import betaln
from py_qvalue import qvalue, pi0est
from typing import Optional, Sequence, Union

__all__ = [
    "beta_approx_pval",
    "get_t_pval",
    "t_two_sided_pval_torch",
    "nominal_pvals_tensorqtl",
    "calculate_qvalues",
    "pval_from_corr_r2"
]

def get_t_pval(t, df, two_tailed: bool = True, log10: bool = False):
    """
    Numerically stable p-values for Student's t.
    Accepts scalars/arrays and torch tensors.
    If log10=True, returns -log10(p) (still via log-survival path).
    """
    t = np.asarray(t.detach().cpu() if isinstance(t, torch.Tensor) else t, dtype=np.float64)
    df = np.asarray(df.detach().cpu() if isinstance(df, torch.Tensor) else df, dtype=np.float64)

    # Use log-survival to avoid underflow at large |t|
    log_sf = stats.t.logsf(np.abs(t), df)
    log_p  = log_sf + (0.0 if not two_tailed else np.log(2.0))
    if log10:
        return -log_p / np.log(10.0)
    return np.minimum(np.exp(log_p), 1.0)


def pval_from_corr_r2(r2, dof, two_tailed: bool = True, log10: bool = False):
    """
    Convert R^2 to p-value using Student's t-distribution.
    """
    r2 = np.clip(np.asarray(r2, dtype=np.float64), 0.0, 1.0 - 1e-15)
    dof = np.asarray(dof, dtype=np.float64)
    t = np.sqrt(dof * r2 / np.maximum(1.0 - r2, 1e-15))
    return get_t_pval(t, dof, two_tailed=two_tailed, log10=log10)


def _approx_beta_alpha(r2_perm, dof):
    """Estimate shape1 (alpha) for Beta using p-value distribution."""
    p = pval_from_corr_r2(r2_perm, dof)
    m, v = p.mean(), p.var()
    if not np.isfinite(v) or v <= 1e-12:
        return 1.0 
    return max(m * (m * (1.0 - m) / v - 1.0), 1e-6)


def _solve_true_dof(r2_perm, dof_init, tol=1e-4):
    """Estimate best-fitting DoF for mapping R^2 -> p-values -> Beta dist."""
    r2_perm = np.asarray(r2_perm, dtype=np.float64)
    if r2_perm.size == 0 or not np.all(np.isfinite(r2_perm)):
        return float(dof_init)

    try:
        f = lambda log_dof: np.log(_approx_beta_alpha(r2_perm, np.exp(log_dof)))
        log_dof = scipy.optimize.newton(f, np.log(dof_init), tol=tol, maxiter=50)
        dof_est = float(np.exp(log_dof))
        return dof_est if np.isfinite(dof_est) else float(dof_init)
    except Exception:
        res = scipy.optimize.minimize(
            lambda x: abs(_approx_beta_alpha(r2_perm, x[0]) - 1.0),
            x0=[dof_init], method="Nelder-Mead", tol=tol,
        )
        dof_est = float(res.x[0])
        return dof_est if res.success and np.isfinite(dof_est) else float(dof_init)


def _beta_mle_on_p(p):
    """Fit Beta(a, b) by maximum likelihood on [0, 1] p-values."""
    p = np.clip(p, 1e-15, 1 - 1e-15)
    m, v = p.mean(), p.var()
    a0 = max(m * (m * (1.0 - m) / v - 1.0), 1e-6)
    b0 = max(a0 * (1.0 / m - 1.0), 1e-6)

    def nll(ab):
        a, b = ab
        if a <= 0 or b <= 0: return np.inf
        return -((a-1)*np.log(p).sum() + (b-1)*np.log(1.0-p).sum() - len(p)*betaln(a,b))
    result = scipy.optimize.minimize(nll, x0=[a0, b0], method="Nelder-Mead")
    a, b = result.x
    return float(a), float(b)


def beta_approx_pval(r2_perm, r2_true, dof_init, log10=False):
    """
    Compute beta-approximate p-value from R^2 using tensorQTL-like method.
    """
    r2_perm = np.asarray(r2_perm, dtype=np.float64)
    r2_true = float(r2_true)
    dof_init = float(dof_init)

    if r2_perm.size == 0 or not np.all(np.isfinite(r2_perm)):
        p_true = pval_from_corr_r2([r2_true], dof_init, log10=log10)[0]
        return float("nan"), float("nan"), float("nan"), dof_init, p_true

    true_dof = _solve_true_dof(r2_perm, dof_init)
    p_perm = pval_from_corr_r2(r2_perm, true_dof)
    a, b = _beta_mle_on_p(p_perm)
    p_true = pval_from_corr_r2([r2_true], true_dof, log10=False)[0]
    p_beta = stats.beta.cdf(p_true, a, b)
    return (
        -np.log10(p_beta) if log10 else p_beta,
        a, b, true_dof,
        -np.log10(p_true) if log10 else p_true
    )


def beta_approx_pval_old(r2_perm, r2_true, dof_init):
    """
    Fit Beta(a,b) to permutation R^2 by method of moments and return:
        (pval_beta, a_hat, b_hat).
    Falls back to empirical tail if variance is ~0 or invalid.
    """
    r2_perm = np.asarray(r2_perm, dtype=np.float64)
    if r2_perm.size == 0 or not np.isfinite(r2_perm).all():
        # Degenerate input: return NA-like outputs but keep DoF/p_true consistent
        p_true = pval_from_corr_r2(np.array([r2_true]), dof_init)[0]
        return (
            float("nan"),
            float("nan"),
            float("nan"),
            float(dof_init),
            float(p_true),
        )

    mu = r2_perm.mean()
    var = r2_perm.var(ddof=1)
    if not np.isfinite(mu) or not np.isfinite(var) or var <= 1e-12:
        # Use empirical tail as a safe fallback
        p_emp = (np.sum(r2_perm >= r2_true) + 1) / (r2_perm.size + 1)
        p_true = pval_from_corr_r2(np.array([r2_true]), dof_init)[0]
        return (
            float(p_emp),
            float("nan"),
            float("nan"),
            float(dof_init),
            float(p_true),
        )

    # Method-of-moments for Beta
    k = mu * (1.0 - mu) / var - 1.0
    a = max(mu * k, 1e-6)
    b = max((1.0 - mu) * k, 1e-6)

    p_beta = 1.0 - stats.beta.cdf(r2_true, a, b)
    p_true = pval_from_corr_r2(np.array([r2_true]), dof_init)[0]
    return float(p_beta), float(a), float(b), float(dof_init), float(p_true)

    
def t_two_sided_pval_torch(t_abs: torch.Tensor, dof: int | torch.Tensor) -> torch.Tensor:
    """
    Two-sided p-value for |t| with df degrees of freedom, returned on the same
    device/dtype as t_abs.

    Strategy:
      - For large dof (>120): Normal approx on GPU: p = 2 * Φ(-|t|)
      - Otherwise: CPU fallback via SciPy get_t_pval, then move back to device.
    """
    if not torch.is_tensor(t_abs):
        t_abs = torch.as_tensor(t_abs)
    assert torch.all(t_abs >= 0), "t_two_sided_pval_torch expects nonnegative |t|"

    dev, dt = t_abs.device, t_abs.dtype

    # Broadcast df to t_abs shape, on-device
    if torch.is_tensor(dof):
        nu = dof.to(device=dev, dtype=dt)
    else:
        nu = torch.as_tensor(dof, device=dev, dtype=dt)
    if nu.shape != t_abs.shape:
        nu = nu.expand_as(t_abs)

    p_out = torch.empty_like(t_abs)

    # Large-df Normal approximation on GPU
    # Φ(-x) = 0.5 * erfc(x / sqrt(2))
    large = nu > 120
    if large.any():
        z = t_abs[large] / torch.sqrt(torch.tensor(2.0, device=dev, dtype=dt))
        p_norm = torch.erfc(z)  # == 2 * Φ(-|t|) because erfc = 2*(1-Φ)
        p_out[large] = p_norm.clamp_min(torch.finfo(dt).tiny)

    # CPU fallback (SciPy) for the rest
    rest = ~large
    if rest.any():
        t_cpu   = t_abs[rest].detach().cpu().numpy()
        dof_cpu = nu[rest].detach().cpu().numpy()
        p_cpu   = get_t_pval(t_cpu, dof_cpu, log10=False)  # vectorized
        p_cpu   = np.maximum(p_cpu, np.finfo(float).tiny)
        p_out[rest] = torch.as_tensor(p_cpu, device=dev, dtype=dt)

    return p_out


def nominal_pvals_tensorqtl(
        y_t: torch.Tensor, G_resid: torch.Tensor, H_resid: torch.Tensor | None,
        k_eff: int, tstats: torch.Tensor,
):
    """
    TensorQTL-style nominal p-values.
    - If H is None: use correlation route (dof = n - k_eff - 2) to match tensorQTL.
    - Else: use the genotype-column t-stat (column 0) with dof = n - k_eff - p.
    Returns (pvals, dof)
    """
    n = int(y_t.shape[0])
    if H_resid is None:
        # correlation-based t -> p (tensorQTL uses n - k_eff - 2)
        y_c = y_t - y_t.mean()
        G_c = G_resid - G_resid.mean(dim=1, keepdim=True)
        num = torch.mv(G_c, y_c)
        den = (torch.linalg.norm(G_c, dim=1) * torch.linalg.norm(y_c)).clamp_min(1e-12)
        r   = num / den
        r2  = (r * r).clamp(max=1 - 1e-12)

        dof = max(n - int(k_eff) - 2, 1)
        # t = r * sqrt(dof / (1 - r^2))
        t_from_r = r * torch.sqrt(torch.tensor(dof, dtype=r.dtype, device=r.device) / (1.0 - r2))
        pvals = t_two_sided_pval_torch(t_from_r.abs(), dof)
        return pvals, dof
    else:
        # multi-predictor: use genotype t-stat (column 0)
        p_pred = 1 + H_resid.shape[2]
        dof = max(n - int(k_eff) - p_pred, 1)
        t_g = tstats[:, 0].abs()
        pvals = t_two_sided_pval_torch(t_g, dof)
        return pvals, dof


def calculate_qvalues(res_df: pd.DataFrame, fdr: float = 0.05,
                      qvalue_lambda: Optional[Union[float, Sequence[float]]] = None,
                      logger: Optional[SimpleLogger] = None) -> pd.DataFrame:
    """Annotate permutation results with q-values, p-value threshold"""
    logger = logger or SimpleLogger()

    if res_df.empty:
        logger.write("Computing q-values: Input is empty, returning unchanged.")
        return res_df.copy()

    # Choose p-value column
    use_beta = ("pval_beta" in res_df.columns) and (~res_df["pval_beta"].isna()).any()
    pcol = "pval_beta" if use_beta else "pval_perm"
    logger.write(f"Computing q-values on '{pcol}' ({res_df.shape[0]} phenotypes).")

    # Optional correlation report (only if we have >2 finite pairs)
    if "pval_perm" in res_df.columns and "pval_beta" in res_df.columns:
        mask = res_df["pval_perm"].notna() & res_df["pval_beta"].notna()
        if mask.sum() >= 3:
            r = stats.pearsonr(res_df.loc[mask, "pval_perm"], res_df.loc[mask, "pval_beta"])[0]
            logger.write(f"  * Corr(p_perm, p_beta) = {r:.4f}")
        else:
            logger.write("  * Skipping corr(p_perm, p_beta): insufficient non-NaN values.")

    # Compute q-values
    p = res_df[pcol].to_numpy(dtype=float)
    if not np.all(np.isfinite(p)):
        logger.write("  * WARNING: non-finite p-values found; replacing with 1.0 for q-value calc.")
        p = np.where(np.isfinite(p), p, 1.0)

    if qvalue_lambda is not None:
        logger.write(f'  * Calculating q-values with lambda = {qvalue_lambda:.3f}')

    qvals_res = qvalue(p, lambda_=qvalue_lambda)
    qvals, pi0 = qvals_res["qvalues"], qvals_res["pi0"]

    res_df['qval'] = qvals
    logger.write(f'  * Proportion of significant phenotypes (1-pi0): {1-pi0:.2f}')
    logger.write(f"  * QTL phenotypes @ FDR {fdr:.2f}: {(res_df['qval'] <= fdr).sum()}")

    # Nominal threshold per phenotype via Beta params
    res_df["pval_nominal_threshold"] = np.nan
    if use_beta:
        have_params = res_df["beta_shape1"].notna() & res_df["beta_shape2"].notna()
        if have_params.any():
            lb = res_df.loc[(res_df["qval"] <= fdr) & have_params, "pval_beta"].sort_values()
            ub = res_df.loc[(res_df["qval"] >  fdr) & have_params, "pval_beta"].sort_values()
            if not lb.empty:
                # global threshold p* between last sig and first non-sig (tensorQTL convention)
                p_star = lb.iloc[-1] if ub.empty else 0.5 * (lb.iloc[-1] + ub.iloc[0])
                # phenotype-wise nominal cutoff from Beta quantile
                from scipy.stats import beta as beta_dist
                idx = have_params[have_params].index
                res_df.loc[idx, "pval_nominal_threshold"] = beta_dist.ppf(
                    p_star, a=res_df.loc[idx, "beta_shape1"], b=res_df.loc[idx, "beta_shape2"]
                )
                logger.write(f"  * min p-value threshold @ FDR {fdr:.2f}: {p_star:.6g} (beta-quantile per phenotype)")
        else:
            logger.write("  * No beta parameters present; skipping nominal threshold computation.")

    return res_df
