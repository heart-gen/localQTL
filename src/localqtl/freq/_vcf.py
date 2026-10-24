"""
Optional pysam/tabix helpers for the frequency importers.

``pysam`` is an optional dependency (``pip install "localqtl[annotate]"``);
importing :mod:`localqtl.freq` never imports it.  Region queries are built
from a ``variant_df`` (or phenotype positions) so that only the relevant
slices of the very large gnomAD / ALFA VCFs are read.
"""
from __future__ import annotations

import logging
from typing import Iterator, Optional

import numpy as np
import pandas as pd

from .table import CHROM_TO_GRCH38_REFSEQ, normalize_chrom

__all__ = [
    "pysam_available",
    "merge_intervals",
    "regions_from_variant_df",
    "regions_from_phenotype_pos",
    "resolve_contig_name",
    "iter_regions",
]

_PYSAM_AVAILABLE: Optional[bool] = None
_INSTALL_HINT = 'pysam is required for VCF/tabix access: pip install "localqtl[annotate]"'


def _try_get_pysam(force: bool = True):
    """Lazy pysam import mirroring :func:`localqtl.utils._try_get_cupy`."""
    global _PYSAM_AVAILABLE
    if _PYSAM_AVAILABLE is False and not force:
        return None
    try:
        import pysam  # noqa: F401
    except Exception as e:  # pragma: no cover - depends on environment
        _PYSAM_AVAILABLE = False
        if force:
            raise ImportError(_INSTALL_HINT) from e
        logging.warning(_INSTALL_HINT)
        return None
    _PYSAM_AVAILABLE = True
    return pysam


def pysam_available() -> bool:
    return _try_get_pysam(force=False) is not None


def merge_intervals(regions: pd.DataFrame, gap: int = 0) -> pd.DataFrame:
    """Merge ``chrom,start,end`` rows (1-based inclusive) closer than ``gap``."""
    out = []
    df = regions.sort_values(["chrom", "start"])
    for chrom, sub in df.groupby("chrom", sort=False):
        cur_s = cur_e = None
        for s, e in zip(sub["start"].to_numpy(), sub["end"].to_numpy()):
            if cur_s is None:
                cur_s, cur_e = int(s), int(e)
            elif int(s) <= cur_e + gap:
                cur_e = max(cur_e, int(e))
            else:
                out.append((chrom, cur_s, cur_e))
                cur_s, cur_e = int(s), int(e)
        if cur_s is not None:
            out.append((chrom, cur_s, cur_e))
    return pd.DataFrame(out, columns=["chrom", "start", "end"])


def regions_from_variant_df(variant_df: pd.DataFrame, gap: int = 10_000) -> pd.DataFrame:
    """One merged region per cluster of variant positions (canonical ``chr`` labels)."""
    df = pd.DataFrame({
        "chrom": [normalize_chrom(c) for c in variant_df["chrom"].to_numpy()],
        "start": pd.to_numeric(variant_df["pos"]).astype("int64"),
    })
    df = df.dropna(subset=["chrom"])
    df["end"] = df["start"]
    return merge_intervals(df, gap=gap)


def regions_from_phenotype_pos(phenotype_pos_df: pd.DataFrame, window: int) -> pd.DataFrame:
    """cis windows ``[start-window, end+window]`` per phenotype, merged."""
    if "pos" in phenotype_pos_df.columns:
        start = end = pd.to_numeric(phenotype_pos_df["pos"]).astype("int64")
    else:
        start = pd.to_numeric(phenotype_pos_df["start"]).astype("int64")
        end = pd.to_numeric(phenotype_pos_df["end"]).astype("int64")
    df = pd.DataFrame({
        "chrom": [normalize_chrom(c) for c in phenotype_pos_df["chr"].to_numpy()],
        "start": np.maximum(start.to_numpy() - int(window), 1),
        "end": end.to_numpy() + int(window),
    }).dropna(subset=["chrom"])
    return merge_intervals(df, gap=0)


def resolve_contig_name(vf, chrom: str) -> Optional[str]:
    """Return the contig name used in ``vf`` for canonical ``chrom`` (``chr7``)."""
    contigs = set(vf.header.contigs) if vf.header.contigs else set()
    if not contigs and hasattr(vf, "contigs"):
        contigs = set(vf.contigs)
    core = chrom[3:] if chrom.startswith("chr") else chrom
    candidates = [chrom, core, f"chr{core}", "MT" if core == "M" else None,
                  "chrMT" if core == "M" else None, CHROM_TO_GRCH38_REFSEQ.get(core)]
    for c in candidates:
        if c and c in contigs:
            return c
    return None


def iter_regions(vf, regions: pd.DataFrame, *, logger=None) -> Iterator[tuple[str, object]]:
    """
    Yield ``(canonical_chrom, record)`` for every record in ``regions``
    (``chrom,start,end`` 1-based inclusive).  Regions on contigs absent from the
    file are skipped with a log line.
    """
    for chrom, start, end in regions[["chrom", "start", "end"]].itertuples(index=False):
        contig = resolve_contig_name(vf, str(chrom))
        if contig is None:
            if logger is not None:
                logger.write(f"    ** contig {chrom} not in VCF; skipping {int(end - start + 1):,} bp")
            continue
        for rec in vf.fetch(contig, max(int(start) - 1, 0), int(end)):
            yield str(chrom), rec
