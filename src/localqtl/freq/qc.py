"""
Ancestry-aware allele-frequency QC: compare in-sample (per-ancestry) allele
frequencies with an external :class:`~localqtl.freq.table.FrequencyTable` and
annotate mapping results with reference frequencies.
"""
from __future__ import annotations

from typing import Mapping, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats as sps

from .table import FrequencyTable, resolve_ancestry_map

__all__ = [
    "DEFAULT_MAF_BINS",
    "compare_allele_frequencies",
    "summarize_af_concordance",
    "annotate_with_frequency",
]

DEFAULT_MAF_BINS = (0.0, 0.005, 0.01, 0.05, 0.10, 0.25, 0.5)


def _labels_from_columns(df: pd.DataFrame) -> list[str]:
    """Ancestry labels from ``af_anc_<label>`` columns (insample layout)."""
    return [c[len("af_anc_"):] for c in df.columns if c.startswith("af_anc_")]


def _two_proportion_z(p1, n1, p2, n2):
    """z for H0: p1 == p2 given allele numbers n1, n2 (pooled variance)."""
    with np.errstate(invalid="ignore", divide="ignore"):
        pool = (p1 * n1 + p2 * n2) / (n1 + n2)
        var = pool * (1 - pool) * (1 / n1 + 1 / n2)
        return (p1 - p2) / np.sqrt(var)


def compare_allele_frequencies(
        in_sample_df: pd.DataFrame,
        freq_table: FrequencyTable,
        ancestry_map: Mapping[str, str],
        *,
        variant_df: Optional[pd.DataFrame] = None,
        min_an: float = 20.0,
        abs_diff_threshold: float = 0.10,
        z_threshold: float = 4.0,
        position_only: bool = False,
) -> pd.DataFrame:
    """
    Per-variant comparison of in-sample and reference allele frequencies.

    Parameters
    ----------
    in_sample_df : DataFrame
        Output of :func:`localqtl.freq.ancestry_allele_frequencies` (columns
        ``af``, ``af_anc_<label>``, ``an_anc_<label>``; index = variant IDs;
        ``chrom``/``pos``/``ref``/``alt`` carried along), or a dosage-scan
        result collapsed per variant after
        :func:`localqtl.freq.anc_columns_to_labels`.
    freq_table : FrequencyTable
    ancestry_map : dict
        ``{panel_label: table_group}``; required, no defaults.
    variant_df : DataFrame, optional
        Supplies ``chrom/pos/ref/alt`` when ``in_sample_df`` lacks them.
    min_an : float
        Comparisons with fewer alleles than this on either side are NaN.
    abs_diff_threshold, z_threshold : float
        A per-ancestry comparison is ``discordant`` when ``|af_in - af_ref| >
        abs_diff_threshold`` **or** ``|z| > z_threshold``.

    Returns
    -------
    pandas.DataFrame
        Indexed like ``in_sample_df``, with per label ``a`` mapped to group
        ``g``: ``af_in_<a>``, ``an_in_<a>``, ``af_ref_<g>``, ``an_ref_<g>``,
        ``af_diff_<a>``, ``logodds_diff_<a>``, ``z_<a>`` and
        ``discordant_<a>``; the pooled ``af``, ``af_ref_expected``
        (admixture-weighted reference) and ``af_diff_pooled``; the lookup
        flags ``match_type``, ``allele_flipped``, ``strand_ambiguous`` and
        ``missing_in_reference``; and ``discordant`` (any ancestry).
    """
    labels = _labels_from_columns(in_sample_df)
    if not labels:
        raise ValueError("in_sample_df has no 'af_anc_<label>' columns; run ancestry_allele_frequencies "
                         "or anc_columns_to_labels first.")
    amap = resolve_ancestry_map(ancestry_map, labels, freq_table)

    keys_src = in_sample_df if {"chrom", "pos"}.issubset(in_sample_df.columns) else variant_df
    if keys_src is None:
        raise ValueError("Provide variant_df: in_sample_df lacks chrom/pos columns.")
    keys = keys_src.loc[in_sample_df.index] if keys_src is not in_sample_df else keys_src
    ref = freq_table.lookup(keys, groups=list(dict.fromkeys(amap.values())), position_only=position_only)

    out = pd.DataFrame(index=in_sample_df.index)
    for c in ("chrom", "pos", "ref", "alt", "rsid"):
        if c in keys.columns:
            out[c] = keys[c].to_numpy()
    out["af"] = in_sample_df["af"].to_numpy(dtype=float)
    weights = []
    expected = np.zeros(len(out))
    any_disc = np.zeros(len(out), dtype=bool)
    for a, g in amap.items():
        af_in = in_sample_df[f"af_anc_{a}"].to_numpy(dtype=float)
        an_in = in_sample_df[f"an_anc_{a}"].to_numpy(dtype=float)
        af_ref = ref[f"af_{g}"].to_numpy(dtype=float)
        an_ref = ref[f"an_{g}"].to_numpy(dtype=float) if f"an_{g}" in ref.columns else np.full(len(out), np.inf)
        ok = (an_in >= min_an) & (an_ref >= min_an) & np.isfinite(af_in) & np.isfinite(af_ref)
        diff = np.where(ok, af_in - af_ref, np.nan)
        with np.errstate(divide="ignore", invalid="ignore"):
            eps = 1e-6
            lo = np.log((af_in + eps) / (1 - af_in + eps)) - np.log((af_ref + eps) / (1 - af_ref + eps))
        z = _two_proportion_z(af_in, an_in, af_ref, np.where(np.isfinite(an_ref), an_ref, 1e9))
        z = np.where(ok, z, np.nan)
        disc = ok & ((np.abs(diff) > abs_diff_threshold) | (np.abs(z) > z_threshold))
        out[f"af_in_{a}"] = af_in
        out[f"an_in_{a}"] = an_in
        out[f"af_ref_{g}"] = af_ref
        out[f"an_ref_{g}"] = an_ref
        out[f"af_diff_{a}"] = diff
        out[f"logodds_diff_{a}"] = np.where(ok, lo, np.nan)
        out[f"z_{a}"] = z
        out[f"discordant_{a}"] = disc
        any_disc |= disc
        w = np.where(np.isfinite(an_in), an_in, 0.0)
        weights.append(w)
        expected = expected + w * np.where(np.isfinite(af_ref), af_ref, 0.0)
    wsum = np.sum(weights, axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        out["af_ref_expected"] = np.where(wsum > 0, expected / np.where(wsum > 0, wsum, 1), np.nan)
    out["af_diff_pooled"] = out["af"] - out["af_ref_expected"]
    for c in ("match_type", "allele_flipped", "strand_ambiguous", "missing_in_reference"):
        out[c] = ref[c].to_numpy()
    out["discordant"] = any_disc
    return out


def summarize_af_concordance(cmp_df: pd.DataFrame, *,
                             maf_bins: Sequence[float] = DEFAULT_MAF_BINS) -> pd.DataFrame:
    """
    Tidy per-ancestry summary of :func:`compare_allele_frequencies` output:
    ``ancestry, maf_bin, n, pearson_r, spearman_r, mean_abs_diff,
    frac_discordant, frac_flipped, frac_missing``; bins use the folded
    reference MAF, plus an ``"all"`` row per ancestry.
    """
    labels = [c[len("af_in_"):] for c in cmp_df.columns if c.startswith("af_in_")]
    rows = []
    edges = list(maf_bins)
    bin_labels = [f"({edges[i]:g},{edges[i+1]:g}]" for i in range(len(edges) - 1)]
    for a in labels:
        ref_col = [c for c in cmp_df.columns if c.startswith("af_ref_") and c != "af_ref_expected"]
        # the group paired with this label is the af_ref_ column appearing alongside its z_ column
        g_col = _ref_col_for_label(cmp_df, a, ref_col)
        af_in = cmp_df[f"af_in_{a}"].to_numpy(dtype=float)
        af_ref = cmp_df[g_col].to_numpy(dtype=float)
        ok = np.isfinite(af_in) & np.isfinite(af_ref)
        maf_ref = np.minimum(af_ref, 1 - af_ref)
        bins = pd.cut(maf_ref, edges, labels=bin_labels, include_lowest=True)
        disc = cmp_df[f"discordant_{a}"].to_numpy(dtype=bool)
        flipped = cmp_df["allele_flipped"].to_numpy(dtype=bool)
        missing = cmp_df["missing_in_reference"].to_numpy(dtype=bool)
        groups = [("all", np.ones(len(cmp_df), dtype=bool))] + [
            (lab, np.asarray(bins == lab)) for lab in bin_labels]
        for lab, m in groups:
            sel = m & ok
            n = int(sel.sum())
            pr = sr = np.nan
            if n >= 3:
                pr = float(np.corrcoef(af_in[sel], af_ref[sel])[0, 1])
                sr = float(sps.spearmanr(af_in[sel], af_ref[sel]).statistic)
            rows.append({
                "ancestry": a, "maf_bin": lab, "n": n,
                "pearson_r": pr, "spearman_r": sr,
                "mean_abs_diff": float(np.mean(np.abs(af_in[sel] - af_ref[sel]))) if n else np.nan,
                "frac_discordant": float(disc[sel].mean()) if n else np.nan,
                "frac_flipped": float(flipped[m].mean()) if m.any() else np.nan,
                "frac_missing": float(missing[m].mean()) if m.any() else np.nan,
            })
    return pd.DataFrame(rows)


def _ref_col_for_label(cmp_df: pd.DataFrame, label: str, ref_cols: list[str]) -> str:
    """Find the ``af_ref_<group>`` column written right after ``an_in_<label>``."""
    cols = list(cmp_df.columns)
    i = cols.index(f"an_in_{label}")
    for c in cols[i + 1:]:
        if c in ref_cols:
            return c
    raise ValueError(f"No reference column found for ancestry {label!r}")


def annotate_with_frequency(
        results_df: pd.DataFrame,
        variant_df: pd.DataFrame,
        freq_table: FrequencyTable,
        *,
        ancestry_map: Optional[Mapping[str, str]] = None,
        ancestry_labels: Optional[Sequence[str]] = None,
        groups: Optional[Sequence[str]] = None,
        position_only: bool = False,
) -> pd.DataFrame:
    """
    Add reference allele frequencies to a nominal / permutation / independent
    result table (one vectorised merge on ``variant_id``).

    Adds ``af_ref_<g>`` and ``an_ref_<g>`` for ``groups`` (default: the values
    of ``ancestry_map``, else every group) plus the lookup flags.  When
    ``ancestry_map`` and ``ancestry_labels`` (panel labels in channel order)
    are given, aliases ``af_ref_anc{k}`` are added so dosage columns
    (``pval_anc{k}``) can be binned by reference frequency.
    """
    if "variant_id" not in results_df.columns:
        raise ValueError("results_df must have a 'variant_id' column.")
    if groups is None:
        groups = list(dict.fromkeys(ancestry_map.values())) if ancestry_map else freq_table.groups
    groups = freq_table.validate_groups(groups)
    vids = pd.Index(results_df["variant_id"].unique())
    missing = vids.difference(variant_df.index)
    if len(missing):
        raise ValueError(f"{len(missing)} variant_ids in results_df are not in variant_df")
    ref = freq_table.lookup(variant_df.loc[vids], groups=groups, position_only=position_only)
    keep = [c for c in ref.columns if c.startswith(("af_", "an_"))] + \
           ["match_type", "allele_flipped", "strand_ambiguous", "missing_in_reference"]
    ref = ref[keep].rename(columns={**{f"af_{g}": f"af_ref_{g}" for g in groups},
                                    **{f"an_{g}": f"an_ref_{g}" for g in groups}})
    if ancestry_map and ancestry_labels is not None:
        for k, lab in enumerate(ancestry_labels):
            g = ancestry_map.get(str(lab))
            if g is not None and f"af_ref_{g}" in ref.columns:
                ref[f"af_ref_anc{k}"] = ref[f"af_ref_{g}"]
    out = results_df.merge(ref, left_on="variant_id", right_index=True, how="left")
    out.index = results_df.index
    return out
