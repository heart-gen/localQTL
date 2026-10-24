"""
Dependency-free reader for ``bcftools query`` output, plus region-file helpers.

Recipes (run once per source; ``-R`` restricts to your variant regions):

gnomAD v4.1 genomes, AFR + NFE groups::

    bcftools query -H -R regions.bed -i 'FILTER="PASS"' \\
      -f '%CHROM\\t%POS\\t%ID\\t%REF\\t%ALT\\t%INFO/AF_afr\\t%INFO/AC_afr\\t%INFO/AN_afr\\t%INFO/AF_nfe\\t%INFO/AC_nfe\\t%INFO/AN_nfe\\n' \\
      gnomad.genomes.v4.1.sites.chr1.vcf.bgz > gnomad.chr1.freq.tsv

ALFA (sample columns are populations; regions BED must use RefSeq names, see
:func:`regions_to_refseq_bed`)::

    bcftools query -H -R regions.refseq.bed -s SAMN10492703,SAMN10492695 \\
      -f '%CHROM\\t%POS\\t%ID\\t%REF\\t%ALT[\\t%AN\\t%AC]\\n' freq.vcf.gz > alfa.freq.tsv

Then ``from_bcftools_query_tsv("gnomad.chr1.freq.tsv")``.
"""
from __future__ import annotations

import re
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from .alfa import ALFA_POPULATIONS
from .table import CHROM_TO_GRCH38_REFSEQ, FrequencyTable, normalize_chrom

__all__ = ["from_bcftools_query_tsv", "regions_to_bed", "regions_to_refseq_bed"]

_HEADER_TOKEN = re.compile(r"^\[\d+\](.+)$")


def _standard_name(token: str) -> str:
    """Map a ``bcftools query -H`` column token to the table column name."""
    t = _HEADER_TOKEN.sub(r"\1", token.strip())
    if t.startswith("INFO/"):
        t = t[5:]
    upper = t.upper()
    if upper in ("CHROM", "#CHROM"):
        return "chrom"
    if upper == "POS":
        return "pos"
    if upper == "REF":
        return "ref"
    if upper == "ALT":
        return "alt"
    if upper == "ID":
        return "rsid"
    m = re.match(r"^(SAMN\d+):(AN|AC)$", t)
    if m:
        code = ALFA_POPULATIONS.get(m.group(1), m.group(1)).lower()
        return f"{m.group(2).lower()}_{code}"
    m = re.match(r"^(AF|AC|AN)_(.+)$", t)
    if m:
        return f"{m.group(1).lower()}_{m.group(2).lower()}"
    if upper in ("AF", "AC", "AN"):
        return f"{upper.lower()}_all"
    return t.lower()


def from_bcftools_query_tsv(path: str, *, columns: Optional[Sequence[str]] = None,
                            source: str = "bcftools", sep: str = "\t") -> FrequencyTable:
    """
    Read a ``bcftools query`` TSV into a :class:`FrequencyTable`.

    With ``-H`` the header line names the columns; otherwise pass ``columns``
    using the standard names (``chrom, pos, ref, alt, [rsid], af_<g>, ac_<g>,
    an_<g>``).  Multi-allelic rows (``ALT`` with commas) are expanded one row
    per ALT, splitting comma-separated ``af_``/``ac_`` values alongside; ``.``
    is read as missing; ``af_<g>`` is computed from ``ac``/``an`` when absent.
    """
    with open(path) as fh:
        first = fh.readline()
    if first.startswith("#"):
        names = [_standard_name(tok) for tok in first.lstrip("#").strip().split(sep)]
        df = pd.read_csv(path, sep=sep, comment=None, skiprows=1, header=None, names=names,
                         dtype=str, na_values=["."], keep_default_na=False)
    else:
        if columns is None:
            raise ValueError("No '#' header line; pass columns=[...] with standard names.")
        df = pd.read_csv(path, sep=sep, header=None, names=list(columns), dtype=str,
                         na_values=["."], keep_default_na=False)
    missing = [c for c in ("chrom", "pos", "ref", "alt") if c not in df.columns]
    if missing:
        raise ValueError(f"TSV lacks required columns {missing}; got {list(df.columns)}")

    multi = df["alt"].astype(str).str.contains(",")
    per_alt_cols = [c for c in df.columns if c.startswith(("af_", "ac_"))]
    if multi.any():
        expanded = []
        for _, r in df.loc[multi].iterrows():
            alts = str(r["alt"]).split(",")
            for i, a in enumerate(alts):
                row = r.copy()
                row["alt"] = a
                for c in per_alt_cols:
                    v = str(r[c]).split(",") if pd.notna(r[c]) else []
                    row[c] = v[i] if i < len(v) else (v[-1] if v else np.nan)
                expanded.append(row)
        df = pd.concat([df.loc[~multi], pd.DataFrame(expanded)], ignore_index=True)

    for c in df.columns:
        if c.startswith(("af_", "ac_", "an_")):
            df[c] = pd.to_numeric(df[c], errors="coerce")
    groups = {c[3:] for c in df.columns if c.startswith(("ac_", "an_"))}
    for g in groups:
        if f"af_{g}" not in df.columns and {f"ac_{g}", f"an_{g}"} <= set(df.columns):
            an = df[f"an_{g}"]
            df[f"af_{g}"] = np.where(an > 0, df[f"ac_{g}"] / an.where(an > 0, 1), np.nan)
    if "rsid" in df.columns:
        df["rsid"] = df["rsid"].where(df["rsid"].astype(str).str.startswith("rs"), None)
    return FrequencyTable(df, source=source, metadata={"file": str(path)})


def regions_to_bed(regions: pd.DataFrame, path: str, *, style: str = "chr") -> str:
    """Write ``chrom,start,end`` (1-based inclusive) as a 0-based BED for ``bcftools -R``."""
    out = pd.DataFrame({
        "chrom": [normalize_chrom(c, style=style) for c in regions["chrom"]],
        "start": regions["start"].astype(int) - 1,
        "end": regions["end"].astype(int),
    })
    out.to_csv(path, sep="\t", header=False, index=False)
    return path


def regions_to_refseq_bed(regions: pd.DataFrame, path: str) -> str:
    """Like :func:`regions_to_bed` but with GRCh38 RefSeq accessions (for ALFA)."""
    out = pd.DataFrame({
        "chrom": [CHROM_TO_GRCH38_REFSEQ[normalize_chrom(c, style="plain")] for c in regions["chrom"]],
        "start": regions["start"].astype(int) - 1,
        "end": regions["end"].astype(int),
    })
    out.to_csv(path, sep="\t", header=False, index=False)
    return path
