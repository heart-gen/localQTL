"""
Statistical helpers (kept separate to avoid pulling SciPy into low-level kernels).
Adapted from tensorqtl:
  - https://github.com/broadinstitute/tensorqtl/blob/master/tensorqtl/core.py
  - https://github.com/broadinstitute/tensorqtl/blob/master/tensorqtl/post.py
"""
from __future__ import annotations
import torch
import numpy as np
import pandas as pd
from scipy import stats
from py_qvalue import qvalue, pi0est
from typing import Optional, Sequence, Union

__all__ = [
    "beta_approx_pval",
    "get_t_pval",
    "t_two_sided_pval_torch",
    "nominal_pvals_tensorqtl",
    "calculate_qvalues",
]

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

    qvals_res = qvalue(p, lambda_qvalue=qvalue_lambda)
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
