"""
Statistical helpers, kept separate so that SciPy is not pulled into the
low-level kernels.

Adapted from tensorQTL
(`core.py <https://github.com/broadinstitute/tensorqtl/blob/master/tensorqtl/core.py>`_,
`post.py <https://github.com/broadinstitute/tensorqtl/blob/master/tensorqtl/post.py>`_).
"""
from __future__ import annotations
import torch
import scipy
import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import betaln
from py_qvalue import qvalue
from typing import Optional, Sequence, Union

from .utils import SimpleLogger
__all__ = [
    "beta_approx_pval",
    "fit_beta_parameters",
    "get_t_pval",
    "get_f_pval",
    "test_beta_difference",
    "t_two_sided_pval_torch",
    "nominal_pvals_tensorqtl",
    "calculate_qvalues",
    "pval_from_corr_r2",
    "NULL_SUMMARY_LEVELS",
    "NULL_SUMMARY_TOP",
    "summarise_null_p",
    "empirical_nominal_threshold",
    "NullSummary",
    "null_summary_of",
]

# --------------------------------------------------------------------------- empirical null summaries (0.4.5)
# The permutation null of a phenotype is kept as a compact summary so that the nominal
# threshold of the pair extraction can be read from the null itself instead of from Beta
# parameters: the ``NULL_SUMMARY_TOP`` smallest null p-values exactly (the extreme tail,
# where any later extrapolation would be fitted) and the quantiles at ``NULL_SUMMARY_LEVELS``
# (a geometric grid of cumulative probabilities from 1e-4 to 1, ratio 10^(1/20)) for the body.
NULL_SUMMARY_LEVELS: np.ndarray = np.unique(np.round(np.logspace(-4.0, 0.0, 81), 10))
NULL_SUMMARY_TOP: int = 250


class NullSummary:
    """
    The null-summary table of a permutation scan, as it travels in
    ``DataFrame.attrs["null_summary"]``.  pandas compares ``attrs`` with ``==`` when it
    concatenates frames and deep-copies them on ``copy()``; a bare DataFrame there raises
    on the comparison and is copied wholesale, so the table is wrapped: equal by identity
    of the table, shared on copy (the summary is read-only).  ``.df`` is the table, one
    row per phenotype (``phenotype_id``, ``group_id``, ``nperm``, ``K``, ``dof``,
    ``levels``, ``null_lead_p_top``, ``null_lead_p_levels``, ``null_max_p_top``,
    ``null_max_p_levels``).  :func:`null_summary_of` reads it from a result frame.
    """
    __slots__ = ("df",)

    def __init__(self, df: pd.DataFrame):
        self.df = df

    def __eq__(self, other):
        return isinstance(other, NullSummary) and other.df is self.df

    def __hash__(self):
        return id(self.df)

    def __deepcopy__(self, memo):
        return NullSummary(self.df)

    def __len__(self):
        return len(self.df)

    def __repr__(self):
        return f"NullSummary({len(self.df)} rows)"


def null_summary_of(res_df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """The null-summary table carried by a permutation result (``attrs["null_summary"]``), or ``None``."""
    v = getattr(res_df, "attrs", {}).get("null_summary")
    if v is None:
        return None
    return v.df if isinstance(v, NullSummary) else v


def summarise_null_p(p_null: np.ndarray, top: int = NULL_SUMMARY_TOP,
                     levels: np.ndarray = NULL_SUMMARY_LEVELS) -> tuple[np.ndarray, np.ndarray]:
    """
    ``(p_top, p_levels)`` of one phenotype's permutation null ``p_null`` (``nperm``
    p-values): the ``top`` smallest values in increasing order, and for every
    cumulative probability ``l`` in ``levels`` the ``max(1, ceil(l * nperm))``-th
    smallest value (the empirical quantile).  Non-finite values count as 1 (a
    permutation whose statistic was unusable is the least extreme draw).
    """
    p = np.asarray(p_null, dtype=np.float64).reshape(-1)
    p = np.where(np.isfinite(p), p, 1.0)
    p.sort()
    n = p.size
    if n == 0:
        return np.empty(0, dtype=np.float32), np.full(len(levels), np.nan, dtype=np.float32)
    k = np.clip(np.ceil(np.asarray(levels, dtype=np.float64) * n).astype(int), 1, n) - 1
    return p[:min(int(top), n)].astype(np.float32), p[k].astype(np.float32)


def empirical_nominal_threshold(p_top: np.ndarray, p_levels: np.ndarray, nperm: int, p_star: float,
                                levels: np.ndarray = NULL_SUMMARY_LEVELS) -> float:
    """
    The nominal p-value threshold ``t`` of one phenotype at the gene-level cutoff
    ``p_star``, read from the empirical null of the window-minimum p-value: ``t`` is
    the ``k``-th smallest null value with ``k = floor(p_star * (nperm + 1))``, so a
    variant whose nominal p-value is at most ``t`` has an empirical p-value against
    that null of at most ``(k + 1) / (nperm + 1)``, i.e. ``p_star`` to the resolution
    of the permutations (the analogue of ``beta.ppf(p_star, a, b)``).  ``k`` within
    the stored top values is exact; above it ``t`` is interpolated between the
    stored quantiles, linearly in ``log t`` against ``log`` cumulative probability.
    ``p_star`` below ``1 / (nperm + 1)`` has no empirical resolution: NaN.
    """
    n = int(nperm)
    if not (np.isfinite(p_star) and n > 0):
        return float("nan")
    k = int(np.floor(float(p_star) * (n + 1) + 1e-6))     # p* is read from float32 p-values: 1 / (n + 1) is k = 1
    if k < 1:
        return float("nan")
    k = min(k, n)
    top = np.asarray(p_top, dtype=np.float64).reshape(-1)
    if k <= top.size:
        return float(top[k - 1])
    lv = np.asarray(levels, dtype=np.float64)
    q = np.asarray(p_levels, dtype=np.float64).reshape(-1)
    ok = np.isfinite(q) & (q > 0)
    if ok.sum() < 2:
        return float("nan")
    lv, q = lv[ok], q[ok]
    level = k / n                                       # the cumulative probability of the k-th smallest
    if level <= lv[0]:
        return float(q[0])
    if level >= lv[-1]:
        return float(q[-1])
    return float(np.exp(np.interp(np.log(level), np.log(lv), np.log(q))))

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


def get_f_pval(f, dfn, dfd, log10: bool = False):
    """Upper-tail p-value of an F statistic with ``(dfn, dfd)`` degrees of freedom."""
    f = np.asarray(f, dtype=np.float64)
    log_p = stats.f.logsf(f, dfn, dfd)
    if log10:
        return -log_p / np.log(10.0)
    return np.minimum(np.exp(log_p), 1.0)


def test_beta_difference(
    beta1: np.ndarray,
    se1: np.ndarray,
    beta2: np.ndarray,
    se2: np.ndarray,
    correlation: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Wald test for H0: beta1 = beta2.

    Parameters
    ----------
    beta1, beta2 : array-like
        Effect size estimates from two conditions (e.g. ancestry-specific betas).
    se1, se2 : array-like
        Standard errors of the effect size estimates.
    correlation : float
        Estimated correlation between the two beta estimates (default 0, which
        is conservative when the estimates come from non-overlapping samples or
        orthogonal ancestry channels).

    Returns
    -------
    z : np.ndarray
        Z-statistics for the difference.
    pval : np.ndarray
        Two-sided p-values.
    """
    beta1 = np.asarray(beta1, dtype=np.float64)
    beta2 = np.asarray(beta2, dtype=np.float64)
    se1 = np.asarray(se1, dtype=np.float64)
    se2 = np.asarray(se2, dtype=np.float64)

    var_diff = se1 ** 2 + se2 ** 2 - 2 * correlation * se1 * se2
    var_diff = np.maximum(var_diff, 1e-30)
    z = (beta1 - beta2) / np.sqrt(var_diff)
    pval = 2.0 * stats.norm.sf(np.abs(z))
    return z, pval


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


TRUE_DOF_MAX_RATIO = 10.0


def _accept_true_dof(dof_est, dof_init) -> float:
    """
    The effective degrees of freedom are a modest correction of the nominal ones (tensorQTL's
    ``true_df`` sits within a factor of about two of ``dof``).  A root more than
    ``TRUE_DOF_MAX_RATIO`` away on either side is the optimiser running off (observed: 1e14 on an
    interaction scan, which overflowed the int32 result column in <= 0.4.1); the nominal
    ``dof_init`` is used instead, as on optimiser failure.
    """
    dof_init = float(dof_init)
    if not np.isfinite(dof_est):
        return dof_init
    if dof_est < dof_init / TRUE_DOF_MAX_RATIO or dof_est > dof_init * TRUE_DOF_MAX_RATIO:
        return dof_init
    return float(dof_est)


def _solve_true_dof(r2_perm, dof_init, tol=1e-4):
    """Estimate best-fitting DoF for mapping R^2 -> p-values -> Beta dist."""
    r2_perm = np.asarray(r2_perm, dtype=np.float64)
    if r2_perm.size == 0 or not np.all(np.isfinite(r2_perm)):
        return float(dof_init)

    try:
        f = lambda log_dof: np.log(_approx_beta_alpha(r2_perm, np.exp(log_dof)))
        log_dof = scipy.optimize.newton(f, np.log(dof_init), tol=tol, maxiter=50)
        return _accept_true_dof(float(np.exp(log_dof)), dof_init)
    except Exception:
        res = scipy.optimize.minimize(
            lambda x: abs(_approx_beta_alpha(r2_perm, x[0]) - 1.0),
            x0=[dof_init], method="Nelder-Mead", tol=tol,
        )
        dof_est = float(res.x[0])
        return _accept_true_dof(dof_est, dof_init) if res.success else float(dof_init)


def _beta_mle_on_p(p):
    """
    Fit Beta(a, b) by maximum likelihood on [0, 1] p-values.

    Degenerate inputs (a single permutation, or zero variance) cannot be
    fitted; ``(nan, nan)`` is returned so that ``pval_beta`` is NaN and callers
    fall back to the empirical ``pval_perm``, instead of running an optimiser
    that never converges.  The optimiser is iteration-capped.
    """
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-15, 1 - 1e-15)
    m, v = float(p.mean()), float(p.var())
    if p.size < 2 or not np.isfinite(v) or v <= 1e-12:
        return float("nan"), float("nan")
    a0 = max(m * (m * (1.0 - m) / v - 1.0), 1e-6)
    b0 = max(a0 * (1.0 / m - 1.0), 1e-6)
    if not (np.isfinite(a0) and np.isfinite(b0)):
        return float("nan"), float("nan")

    def nll(ab):
        a, b = ab
        if a <= 0 or b <= 0: return np.inf
        return -((a-1)*np.log(p).sum() + (b-1)*np.log(1.0-p).sum() - len(p)*betaln(a,b))
    result = scipy.optimize.minimize(nll, x0=[a0, b0], method="Nelder-Mead",
                                     options={"maxiter": 2000, "maxfev": 4000})
    a, b = result.x
    if not (np.isfinite(a) and np.isfinite(b) and a > 0 and b > 0):
        return float(a0), float(b0)
    return float(a), float(b)


def fit_beta_parameters(r2_perm, dof_init, tol: float = 1e-4, return_minp: bool = False):
    """
    Fit the Beta approximation to a vector of permutation maximum ``r2`` values
    (tensorQTL ``fit_beta_parameters``): estimate the effective degrees of
    freedom ``true_dof`` such that the moment-matched first Beta shape is 1,
    then fit ``Beta(a, b)`` by maximum likelihood to the p-values obtained
    with ``true_dof``.

    Returns ``(beta_shape1, beta_shape2, true_dof)`` or, with
    ``return_minp``, ``(beta_shape1, beta_shape2, true_dof, pval_true_dof)``
    where the last element is the per-permutation minimum p-value vector.
    """
    r2_perm = np.asarray(r2_perm, dtype=np.float64)
    true_dof = _solve_true_dof(r2_perm, float(dof_init), tol=tol)
    p_perm = pval_from_corr_r2(r2_perm, true_dof)
    a, b = _beta_mle_on_p(p_perm)
    if return_minp:
        return float(a), float(b), float(true_dof), p_perm
    return float(a), float(b), float(true_dof)


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
    Two-sided p-value for ``|t|`` with ``df`` degrees of freedom, on the same
    device and dtype as ``t_abs``.

    For large ``dof`` (> 120) the normal approximation ``p = 2 * Phi(-|t|)`` is
    computed on the GPU; otherwise the values fall back to SciPy
    (:func:`get_t_pval`) on the CPU and are moved back to the device.
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

    ``k_eff`` counts the degrees of freedom already consumed by centring (the
    intercept) and the projected-out covariates (``Residualizer.k_eff``).

    Without ``H`` the correlation route is used, with ``dof = n - k_eff - 1``
    (tensorQTL's ``n - 2 - k`` for ``k`` covariates); otherwise the
    genotype-column t-statistic (column 0) with ``dof = n - k_eff - p``.

    Returns ``(pvals, dof)``.
    """
    n = int(y_t.shape[0])
    if H_resid is None:
        # correlation-based t -> p
        y_c = y_t - y_t.mean()
        G_c = G_resid - G_resid.mean(dim=1, keepdim=True)
        num = torch.mv(G_c, y_c)
        den = (torch.linalg.norm(G_c, dim=1) * torch.linalg.norm(y_c)).clamp_min(1e-12)
        r   = num / den
        r2  = (r * r).clamp(max=1 - 1e-12)

        dof = max(n - int(k_eff) - 1, 1)
        # t = r * sqrt(dof / (1 - r^2))
        t_from_r = r * torch.sqrt(torch.tensor(dof, dtype=r.dtype, device=r.device) / (1.0 - r2))
        t_cpu = t_from_r.abs().detach().cpu().numpy()
        p_cpu = get_t_pval(t_cpu, dof, log10=False)
        pvals = torch.as_tensor(p_cpu, device=t_from_r.device, dtype=torch.float32)
        return pvals, dof
    else:
        # multi-predictor: use genotype t-stat (column 0)
        p_pred = 1 + H_resid.shape[2]
        dof = max(n - int(k_eff) - p_pred, 1)
        t_g = tstats[:, 0].abs()
        t_cpu = t_g.detach().cpu().numpy()
        p_cpu = get_t_pval(t_cpu, dof, log10=False)
        pvals = torch.as_tensor(p_cpu, device=t_g.device, dtype=torch.float32)
        return pvals, dof


def _fit_method_of(res_df: pd.DataFrame) -> np.ndarray:
    """``pval_fit_method`` per row: the column when present, else ``"beta"`` where a Beta
    p-value exists and ``"empirical"`` elsewhere (tables written before 0.4.5)."""
    if "pval_fit_method" in res_df.columns:
        m = res_df["pval_fit_method"].astype(object).where(res_df["pval_fit_method"].notna(), None).to_numpy()
        has_beta = res_df["pval_beta"].notna().to_numpy() if "pval_beta" in res_df.columns else np.zeros(len(res_df), bool)
        return np.array([str(v) if v is not None else ("beta" if b else "empirical") for v, b in zip(m, has_beta)], dtype=object)
    if "pval_beta" in res_df.columns:
        return np.where(res_df["pval_beta"].notna().to_numpy(), "beta", "empirical").astype(object)
    return np.full(len(res_df), "empirical", dtype=object)


def _null_summary_lookup(res_df: pd.DataFrame, null_summary: pd.DataFrame | None) -> tuple[pd.DataFrame | None, str | None]:
    """The null summary indexed by the key that matches ``res_df`` (``group_id`` of a grouped
    scan, else ``phenotype_id``), or ``(None, None)``."""
    if null_summary is None:
        null_summary = null_summary_of(res_df)
    elif isinstance(null_summary, NullSummary):
        null_summary = null_summary.df
    if null_summary is None or len(null_summary) == 0:
        return None, None
    if "group_id" in res_df.columns and "group_id" in null_summary.columns and null_summary["group_id"].notna().any():
        key = "group_id"
    else:
        key = "phenotype_id"
    return null_summary.drop_duplicates(key).set_index(key), key


def calculate_qvalues(res_df: pd.DataFrame, fdr: float = 0.05,
                      qvalue_lambda: Optional[Union[float, Sequence[float]]] = None,
                      logger: Optional[SimpleLogger] = None,
                      null_summary: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Annotate permutation results with q-values and the nominal p-value threshold.

    Two inference products, read per row from ``pval_fit_method``:

    * ``"beta"`` (the correlation scans, tensorQTL's construction): the phenotype-level
      p-value is ``pval_beta`` and the nominal threshold of the pair extraction is the
      Beta quantile ``beta.ppf(p*, shape1, shape2)``.
    * ``"empirical"`` (the ancestry-specific dosage scan since 0.4.5, and every scan run
      with ``beta_approx=False``): the phenotype-level p-value is the calibrated
      ``pval_perm`` of the selected-lead null, and the nominal threshold is the
      ``p*``-quantile of the phenotype's own window-minimum null
      (:func:`empirical_nominal_threshold` on the ``null_summary`` the scan stored:
      ``res_df.attrs["null_summary"]`` or the ``null_summary`` argument, the table the
      CLI writes as ``<prefix>.cis_qtl.null_summary.parquet``).  Without a summary the
      threshold is NaN and the pair extraction skips the phenotype.

    Tables written before 0.4.5 carry no ``pval_fit_method``: a finite ``pval_beta`` is
    read as ``"beta"``.  A phenotype whose beta approximation could not be fitted
    (``pval_beta`` NaN) is tested with its ``pval_perm``.  Q-values are Storey's
    (``py_qvalue``) over all rows; ``p*`` is the gene-level cutoff between the largest
    significant and the smallest non-significant phenotype-level p-value (tensorQTL's
    convention) and is shared by both products.
    """
    logger = logger or SimpleLogger()

    if res_df.empty:
        logger.write("Computing q-values: Input is empty, returning unchanged.")
        return res_df.copy()

    method = _fit_method_of(res_df)
    is_beta = method == "beta"
    n_beta, n_emp = int(is_beta.sum()), int((~is_beta).sum())
    logger.write(f"Computing q-values ({res_df.shape[0]} phenotypes: {n_beta} on pval_beta, {n_emp} on pval_perm).")

    # Optional correlation report (only if we have >2 finite pairs)
    if "pval_perm" in res_df.columns and "pval_beta" in res_df.columns:
        mask = res_df["pval_perm"].notna() & res_df["pval_beta"].notna()
        if mask.sum() >= 3:
            r = stats.pearsonr(res_df.loc[mask, "pval_perm"], res_df.loc[mask, "pval_beta"])[0]
            logger.write(f"  * Corr(p_perm, p_beta) = {r:.4f}")

    # The phenotype-level p-value of each row
    p_perm = res_df["pval_perm"].to_numpy(dtype=float) if "pval_perm" in res_df.columns else np.full(len(res_df), np.nan)
    p_beta = res_df["pval_beta"].to_numpy(dtype=float) if "pval_beta" in res_df.columns else np.full(len(res_df), np.nan)
    p = np.where(is_beta, p_beta, p_perm)
    fill = ~np.isfinite(p) & np.isfinite(p_perm)
    if fill.any():
        # A beta fit that could not be made (degenerate null, see ``_beta_mle_on_p``) leaves
        # ``pval_beta`` NaN: that phenotype is tested with its empirical permutation p-value
        # rather than set to p = 1 and dropped from the discoveries.  The stored columns are
        # not modified, and the phenotype keeps a NaN ``pval_nominal_threshold``.
        logger.write(f"  * {int(fill.sum())} phenotypes without a beta approximation: using pval_perm for their q-values.")
        p = np.where(fill, p_perm, p)
    if not np.all(np.isfinite(p)):
        logger.write("  * WARNING: non-finite p-values found; replacing with 1.0 for q-value calc.")
        p = np.where(np.isfinite(p), p, 1.0)

    # Only the q-values and pi0 are used: the local FDR (a kernel density of the p-values) is not requested, as it
    # fails on tables with tied or very few p-values, which the empirical p of a small scan can produce.
    if qvalue_lambda is None:
        qvals_res = qvalue(p, lfdr_out=False)
    elif isinstance(qvalue_lambda, (list, tuple, np.ndarray)):
        logger.write(f'  * Calculating q-values with lambda = {qvalue_lambda}')
        qvals_res = qvalue(p, lambda_=np.asarray(qvalue_lambda, dtype=float), lfdr_out=False)
    else:
        # A single lambda (the CLI's --qvalue_lambda, R's qvalue(lambda=x)): pi0 is the plain Storey estimate
        # at that lambda; py_qvalue's pi0est needs a grid for its smoother, so pi0 is passed to it directly.
        lam = float(qvalue_lambda)
        if not 0.0 <= lam < 1.0:
            raise ValueError(f"qvalue_lambda must be in [0, 1), got {qvalue_lambda!r}")
        pi0_fixed = float(min(1.0, np.mean(p >= lam) / (1.0 - lam)))
        if pi0_fixed <= 0.0:
            raise ValueError(f"qvalue_lambda={lam:g}: the estimated pi0 is 0 (no p-value at or above lambda); "
                             "use a smaller lambda or the default grid")
        logger.write(f'  * Calculating q-values with lambda = {lam:.3f} (pi0 = {pi0_fixed:.3f})')
        qvals_res = qvalue(p, pi0=pi0_fixed, lfdr_out=False)
    qvals, pi0 = qvals_res["qvalues"], qvals_res["pi0"]

    res_df['qval'] = qvals
    logger.write(f'  * Proportion of significant phenotypes (1-pi0): {1-pi0:.2f}')
    logger.write(f"  * QTL phenotypes @ FDR {fdr:.2f}: {(res_df['qval'] <= fdr).sum()}")

    # The gene-level cutoff p*: between the last significant and the first non-significant p
    res_df["pval_nominal_threshold"] = np.nan
    sig = res_df["qval"].to_numpy(dtype=float) <= fdr
    if not sig.any():
        logger.write("  * No phenotype passes the FDR: no nominal thresholds.")
        return res_df
    lb, ub = p[sig].max(), (p[~sig].min() if (~sig).any() else np.nan)
    p_star = float(lb if not np.isfinite(ub) else 0.5 * (lb + ub))
    p_star = float(np.clip(p_star, 1e-16, 1 - 1e-16))  # Avoid edge cases

    # Product 2a: the Beta quantile where a Beta was fitted
    if "beta_shape1" in res_df.columns and "beta_shape2" in res_df.columns:
        have_params = is_beta & res_df["beta_shape1"].notna().to_numpy() & res_df["beta_shape2"].notna().to_numpy()
        if have_params.any():
            from scipy.stats import beta as beta_dist
            idx = res_df.index[have_params]
            a = res_df.loc[idx, "beta_shape1"].clip(lower=1e-6, upper=1e6)
            b = res_df.loc[idx, "beta_shape2"].clip(lower=1e-6, upper=1e6)
            res_df.loc[idx, "pval_nominal_threshold"] = beta_dist.ppf(p_star, a=a, b=b)
            logger.write(f"  * min p-value threshold @ FDR {fdr:.2f}: {p_star:.6g} (beta quantile, {int(have_params.sum())} phenotypes)")

    # Product 2b: the empirical quantile of the window-minimum null where the scan stored it
    if n_emp:
        summary, key = _null_summary_lookup(res_df, null_summary)
        if summary is None or key not in res_df.columns:
            logger.write(f"  * {n_emp} phenotypes on pval_perm without a null summary: their nominal threshold is NaN.")
        else:
            keys = res_df[key].to_numpy()
            thr = np.full(len(res_df), np.nan)
            missing = 0
            for i in np.flatnonzero(~is_beta):
                k = keys[i]
                if k not in summary.index:
                    missing += 1
                    continue
                row = summary.loc[k]
                levels = np.asarray(row["levels"]) if "levels" in summary.columns else NULL_SUMMARY_LEVELS
                thr[i] = empirical_nominal_threshold(np.asarray(row["null_max_p_top"]), np.asarray(row["null_max_p_levels"]),
                                                     int(row["nperm"]), p_star, levels=levels)
            res_df.loc[res_df.index[~is_beta], "pval_nominal_threshold"] = thr[~is_beta]
            got = int(np.isfinite(thr[~is_beta]).sum())
            logger.write(f"  * min p-value threshold @ FDR {fdr:.2f}: {p_star:.6g} (empirical quantile of the window-minimum "
                         f"null, {got} of {n_emp} phenotypes" + (f", {missing} without a stored null" if missing else "") + ")")
            if got < n_emp - missing:
                logger.write(f"  * {n_emp - missing - got} phenotypes: p* below the resolution of their permutations, threshold NaN.")

    return res_df
