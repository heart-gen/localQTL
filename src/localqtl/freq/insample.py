"""
In-sample per-ancestry allele frequencies from genotypes and a local-ancestry panel.

:func:`ancestry_allele_frequencies` reproduces the ``an_anc{k}`` / ``ac_anc{k}``
/ ``af_anc{k}`` statistics of the dosage model without running a QTL scan, with
one difference: missing genotypes and no-call ancestry are *excluded* from the
counts here, whereas the mapping path mean-imputes genotypes first.
"""
from __future__ import annotations

import re
from typing import Optional, Sequence

import numpy as np
import pandas as pd

__all__ = ["ancestry_labels", "anc_columns_to_labels", "ancestry_allele_frequencies"]


def ancestry_labels(panel) -> list[str]:
    """Ancestry labels in channel order (all ``n_pops`` labels, even when collapsed)."""
    return [str(a) for a in panel.ancestries]


def anc_columns_to_labels(df: pd.DataFrame, labels: Sequence[str]) -> pd.DataFrame:
    """Rename every ``_anc{k}`` token (k in channel order) to ``_anc_<label>``, incl. pairwise columns."""
    pat = re.compile(r"_anc(\d+)(?=_|$)")

    def _ren(col: str) -> str:
        return pat.sub(lambda m: f"_anc_{labels[int(m.group(1))]}" if int(m.group(1)) < len(labels) else m.group(0), col)

    return df.rename(columns={c: _ren(c) for c in df.columns if pat.search(c)})


def _expand_channels(L: np.ndarray, n_pops: int) -> np.ndarray:
    """``(v, s, 1)`` collapsed two-ancestry counts -> ``(v, s, 2)``; else unchanged."""
    if L.shape[2] == 1 and n_pops == 2:
        return np.concatenate([L, 2.0 - L], axis=2)
    return L


def ancestry_allele_frequencies(
        genotype_df: pd.DataFrame,
        variant_df: pd.DataFrame,
        panel,
        *,
        chroms: Optional[Sequence[str]] = None,
        missing: float = -9.0,
        chunk_size: int = 50_000,
        locus_method: str = "stepwise",
        nearest_tolerance: Optional[int] = None,
) -> pd.DataFrame:
    """
    Per-variant pooled and per-ancestry in-sample allele statistics.

    Parameters
    ----------
    genotype_df : DataFrame
        Variants x samples dosages of the counted (alt) allele; ``missing`` or
        NaN marks missing calls.
    variant_df : DataFrame
        Indexed like ``genotype_df`` with ``chrom``, ``pos`` (and optionally
        ``ref``, ``alt``, ``rsid``, which are carried through).
    panel : LocalAncestryPanel
        Samples must match ``genotype_df.columns`` as a set; columns are
        reordered to the panel order.
    chroms : sequence of str, optional
        Restrict to these chromosomes (labels as in ``variant_df['chrom']``).

    Returns
    -------
    DataFrame indexed by variant ID with ``chrom, pos, [ref, alt, rsid],
    n_called, af`` and, for each panel ancestry label ``L``,
    ``an_anc_L, ac_anc_L, af_anc_L, ma_count_anc_L``.  Variants with no
    covering local-ancestry locus have NaN per-ancestry columns.
    """
    if not variant_df.index.equals(genotype_df.index):
        raise ValueError("variant_df and genotype_df must share the same index (variant IDs).")
    panel_samples = [str(s) for s in panel.sample_ids]
    geno_samples = [str(s) for s in genotype_df.columns]
    if set(panel_samples) != set(geno_samples):
        raise ValueError("Panel samples and genotype_df columns differ; build the panel with "
                         "select_samples=list(genotype_df.columns).")
    if panel_samples != geno_samples:
        genotype_df = genotype_df[panel_samples]

    labels = ancestry_labels(panel)
    K = len(labels)
    vchrom = variant_df["chrom"].astype(str).to_numpy()
    vpos = pd.to_numeric(variant_df["pos"]).to_numpy(dtype=np.int64)
    if chroms is not None:
        keep = np.isin(vchrom, [str(c) for c in chroms])
    else:
        keep = np.ones(len(vchrom), dtype=bool)

    n_var = len(variant_df)
    an = np.full((n_var, K), np.nan, dtype=np.float64)
    ac = np.full((n_var, K), np.nan, dtype=np.float64)
    af_pool = np.full(n_var, np.nan, dtype=np.float64)
    n_called = np.zeros(n_var, dtype=np.int64)

    G_all = genotype_df.to_numpy(dtype=np.float32)
    for chrom in pd.unique(vchrom[keep]):
        rows = np.flatnonzero((vchrom == chrom) & keep)
        loc = panel.locus_index(chrom, vpos[rows], method=locus_method, tolerance=nearest_tolerance)
        counts = None
        for start in range(0, len(rows), int(chunk_size)):
            r = rows[start:start + int(chunk_size)]
            l = loc[start:start + int(chunk_size)]
            G = G_all[r].astype(np.float64)
            miss = ~np.isfinite(G) | (G == missing)
            Gm = np.where(miss, 0.0, G)
            valid_g = (~miss).astype(np.float64)
            n_called[r] = valid_g.sum(axis=1).astype(np.int64)
            denom = 2.0 * valid_g.sum(axis=1)
            af_pool[r] = np.where(denom > 0, (Gm * valid_g).sum(axis=1) / np.where(denom > 0, denom, 1), np.nan)

            mapped = l >= 0
            if not mapped.any():
                continue
            if counts is None:
                counts = panel.chrom_counts(chrom)
            L = counts[l[mapped]].astype(np.float64)                    # (v, s, channels)
            L = np.where(L < 0, np.nan, L)                                # int8 -1 = no call
            L = _expand_channels(L, K)
            valid = valid_g[mapped][:, :, None] * np.isfinite(L).all(axis=2, keepdims=True)
            Lz = np.where(np.isfinite(L), L, 0.0) * valid
            an[r[mapped]] = Lz.sum(axis=1)
            ac[r[mapped]] = (Gm[mapped][:, :, None] * Lz * 0.5).sum(axis=1)

    with np.errstate(invalid="ignore", divide="ignore"):
        af = np.where(an > 0, ac / np.where(an > 0, an, 1.0), np.nan)
    ma_count = np.minimum(ac, an - ac)

    out = pd.DataFrame(index=variant_df.index)
    out["chrom"] = vchrom
    out["pos"] = vpos
    for col in ("ref", "alt", "rsid"):
        if col in variant_df.columns:
            out[col] = variant_df[col].to_numpy()
    out["n_called"] = n_called
    out["af"] = af_pool.astype(np.float32)
    for k, lab in enumerate(labels):
        out[f"an_anc_{lab}"] = an[:, k].astype(np.float32)
        out[f"ac_anc_{lab}"] = ac[:, k].astype(np.float32)
        out[f"af_anc_{lab}"] = af[:, k].astype(np.float32)
        out[f"ma_count_anc_{lab}"] = ma_count[:, k].astype(np.float32)
    if chroms is not None:
        out = out.loc[keep]
    return out
