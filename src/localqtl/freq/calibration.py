"""
Frequency- and ancestry-stratified calibration summaries for association
p-values (data only; no plotting dependency).
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats as sps

__all__ = [
    "DEFAULT_AF_BINS",
    "genomic_control_lambda",
    "qq_table",
    "calibration_by_frequency",
    "calibration_by_frequency_dosage",
]

DEFAULT_AF_BINS = (0.0, 0.005, 0.01, 0.05, 0.10, 0.25, 0.5)
_CHI2_MEDIAN = sps.chi2.ppf(0.5, 1)  # 0.4549...


def genomic_control_lambda(pvals) -> float:
    """Median-based genomic-control lambda (chi2 with 1 df)."""
    p = np.asarray(pvals, dtype=float)
    p = p[np.isfinite(p) & (p > 0) & (p <= 1)]
    if p.size == 0:
        return float("nan")
    return float(sps.chi2.ppf(1 - np.median(p), 1) / _CHI2_MEDIAN)


def qq_table(pvals, n_points: int = 200) -> pd.DataFrame:
    """Expected vs observed ``-log10(p)`` at ``n_points`` quantiles."""
    p = np.sort(np.asarray(pvals, dtype=float))
    p = p[np.isfinite(p) & (p > 0) & (p <= 1)]
    n = p.size
    if n == 0:
        return pd.DataFrame(columns=["quantile", "expected_neglog10", "observed_neglog10"])
    q = np.unique(np.clip(np.round(np.linspace(0, n - 1, min(n_points, n))).astype(int), 0, n - 1))
    expected = (q + 0.5) / n
    return pd.DataFrame({"quantile": q / n, "expected_neglog10": -np.log10(expected),
                         "observed_neglog10": -np.log10(p[q])})


def _bin_series(values: np.ndarray, bins: Sequence[float], fold: bool) -> pd.Categorical:
    v = np.asarray(values, dtype=float)
    if fold:
        v = np.minimum(v, 1 - v)
    edges = list(bins)
    labels = [f"({edges[i]:g},{edges[i+1]:g}]" for i in range(len(edges) - 1)]
    return pd.cut(v, edges, labels=labels, include_lowest=True)


def calibration_by_frequency(
        results_df: pd.DataFrame,
        *,
        pval_col: str = "pval_nominal",
        by: str = "af",
        bins: Sequence[float] = DEFAULT_AF_BINS,
        fold: bool = True,
        alpha: Sequence[float] = (0.05, 0.01, 1e-3),
        min_n: int = 20,
        extra_group_cols: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """
    Per-frequency-bin calibration of ``pval_col``.

    ``by`` names the column to bin on (``af``, ``af_anc{k}``, ``af_ref_<g>``,
    ``af_ref_anc{k}`` after :func:`~localqtl.freq.qc.annotate_with_frequency`,
    or an allele-count column with ``fold=False`` and integer ``bins``).

    Returns a tidy DataFrame with one row per bin (plus ``"all"``):
    ``bin, n, lambda_gc, frac_p_lt_<alpha>..., median_p, kolmogorov_d,
    kolmogorov_p``.  ``kolmogorov_*`` test uniformity of the p-values; under
    the null every bin should give ``lambda_gc ~ 1`` and
    ``frac_p_lt_a ~ a``.  Bins with fewer than ``min_n`` p-values report NaN
    statistics.
    """
    if pval_col not in results_df.columns:
        raise ValueError(f"{pval_col!r} not in results_df")
    if by not in results_df.columns:
        raise ValueError(f"{by!r} not in results_df")
    p = results_df[pval_col].to_numpy(dtype=float)
    cat = _bin_series(results_df[by].to_numpy(), bins, fold)
    frame = pd.DataFrame({"p": p, "bin": cat.astype(object)})
    if extra_group_cols:
        for c in extra_group_cols:
            frame[c] = results_df[c].to_numpy()
    frame_all = frame.copy()
    frame_all["bin"] = "all"
    frame = pd.concat([frame_all, frame], ignore_index=True)
    frame = frame.dropna(subset=["bin"])

    group_cols = ["bin"] + list(extra_group_cols or [])
    rows = []
    for key, sub in frame.groupby(group_cols, sort=False, observed=True):
        key = key if isinstance(key, tuple) else (key,)
        pv = sub["p"].to_numpy(dtype=float)
        pv = pv[np.isfinite(pv)]
        n = int(pv.size)
        row = dict(zip(group_cols, key))
        row["n"] = n
        if n >= min_n:
            row["lambda_gc"] = genomic_control_lambda(pv)
            for a in alpha:
                row[f"frac_p_lt_{a:g}"] = float(np.mean(pv < a))
            row["median_p"] = float(np.median(pv))
            ks = sps.kstest(pv, "uniform")
            row["kolmogorov_d"] = float(ks.statistic)
            row["kolmogorov_p"] = float(ks.pvalue)
        else:
            row["lambda_gc"] = np.nan
            for a in alpha:
                row[f"frac_p_lt_{a:g}"] = np.nan
            row["median_p"] = row["kolmogorov_d"] = row["kolmogorov_p"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def calibration_by_frequency_dosage(
        results_df: pd.DataFrame,
        *,
        ancestry_labels: Optional[Sequence[str]] = None,
        by: str = "af_anc",
        bins: Sequence[float] = DEFAULT_AF_BINS,
        fold: bool = True,
        include_joint: bool = True,
        **kwargs,
) -> pd.DataFrame:
    """
    :func:`calibration_by_frequency` for every ancestry of a dosage scan.

    For each ``k`` with ``pval_anc{k}`` present, ``pval_anc{k}`` is binned by
    ``f"{by}{k}"`` (``af_anc{k}`` in-sample by default; use
    ``by="af_ref_anc"`` after :func:`~localqtl.freq.qc.annotate_with_frequency`
    or ``by="ma_count_anc"`` with integer bins and ``fold=False``).  With
    ``include_joint`` the joint test ``pval_joint`` is binned by pooled ``af``
    and labelled ``"joint"``.  The result has an ``ancestry`` column (label
    from ``ancestry_labels`` or ``anc{k}``).
    """
    ks = sorted(int(c[len("pval_anc"):]) for c in results_df.columns
                if c.startswith("pval_anc") and c[len("pval_anc"):].isdigit())
    if not ks:
        raise ValueError("results_df has no pval_anc{k} columns (run ancestry_model='dosage').")
    parts = []
    for k in ks:
        col = f"{by}{k}"
        if col not in results_df.columns:
            raise ValueError(f"{col!r} not in results_df")
        t = calibration_by_frequency(results_df, pval_col=f"pval_anc{k}", by=col, bins=bins,
                                     fold=fold, **kwargs)
        t.insert(0, "ancestry", str(ancestry_labels[k]) if ancestry_labels is not None else f"anc{k}")
        parts.append(t)
    if include_joint and "pval_joint" in results_df.columns and "af" in results_df.columns:
        t = calibration_by_frequency(results_df, pval_col="pval_joint", by="af", bins=bins,
                                     fold=True, **kwargs)
        t.insert(0, "ancestry", "joint")
        parts.append(t)
    return pd.concat(parts, ignore_index=True)
