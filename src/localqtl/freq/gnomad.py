"""
gnomAD sites-VCF importer (v4.x genetic-ancestry groups and the local-ancestry
tract releases).

Only the regions covering your variants are read (tabix), so the per-chromosome
``*.sites.chr*.vcf.bgz`` files can stay where they are.  INFO fields are read
as ``AF_<grp>`` / ``AC_<grp>`` / ``AN_<grp>``; the exact tags of the
local-ancestry releases differ per file, so they are detected from the header
(``tract_fields="auto"``) rather than hardcoded, and recorded in
``FrequencyTable.metadata["tract_fields"]``.

Requires ``pysam`` (``pip install "localqtl[annotate]"``).  Without pysam, use
the ``bcftools query`` recipe in :mod:`localqtl.freq.bcftools`.
"""
from __future__ import annotations

import glob
import os
import re
from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd

from ..utils import SimpleLogger
from ._vcf import _try_get_pysam, iter_regions, regions_from_variant_df
from .table import FrequencyTable, normalize_chrom

__all__ = [
    "GNOMAD_V4_GROUPS",
    "GNOMAD_V4_EXTRA_FIELDS",
    "GNOMAD_INFO_PATTERN",
    "inspect_gnomad_header",
    "from_gnomad_vcf",
]

#: gnomAD v4 genetic-ancestry group keys (INFO suffixes of ``AF_<grp>``).
GNOMAD_V4_GROUPS = ("afr", "ami", "amr", "asj", "eas", "fin", "mid", "nfe", "remaining", "sas")
#: Global fields worth keeping alongside the groups.
GNOMAD_V4_EXTRA_FIELDS = ("AF_grpmax", "fafmax_faf95_max")
GNOMAD_INFO_PATTERN = re.compile(r"^(AF|AC|AN)_(.+)$")
_AUTO_EXCLUDE = ("xx", "xy", "raw", "grpmax", "joint", "faf", "non_ukb", "female", "male")


def _expand_paths(paths: Union[str, Sequence[str]]) -> list[str]:
    if isinstance(paths, str):
        paths = [paths]
    out: list[str] = []
    for p in paths:
        hits = sorted(glob.glob(p)) if any(ch in p for ch in "*?[") else [p]
        if not hits:
            raise FileNotFoundError(f"No VCF matches {p}")
        out.extend(hits)
    return out


def inspect_gnomad_header(path: str) -> pd.DataFrame:
    """INFO fields of a gnomAD VCF: ``field, number, type, description``."""
    pysam = _try_get_pysam()
    with pysam.VariantFile(path) as vf:
        rows = [(k, v.number, v.type, v.description) for k, v in vf.header.info.items()]
    return pd.DataFrame(rows, columns=["field", "number", "type", "description"])


def _triplets(info_keys: set[str]) -> dict[str, tuple[str, str, str]]:
    """suffix -> (AF, AC, AN) field names for every complete triplet in the header."""
    found: dict[str, dict[str, str]] = {}
    for k in info_keys:
        m = GNOMAD_INFO_PATTERN.match(k)
        if m:
            found.setdefault(m.group(2), {})[m.group(1)] = k
    return {s: (d["AF"], d["AC"], d["AN"]) for s, d in found.items() if {"AF", "AC", "AN"} <= set(d)}


def _value(rec, field: str, alt_ix: int):
    v = rec.info.get(field)
    if v is None:
        return np.nan
    if isinstance(v, (tuple, list)):
        if len(v) == 0:
            return np.nan
        v = v[alt_ix] if alt_ix < len(v) else v[-1]
    return np.nan if v is None else float(v)


def from_gnomad_vcf(
        paths: Union[str, Sequence[str]],
        regions: Optional[Union[pd.DataFrame, str]] = None,
        *,
        variant_df: Optional[pd.DataFrame] = None,
        groups: Sequence[str] = GNOMAD_V4_GROUPS,
        include_global: bool = True,
        extra_fields: Sequence[str] = GNOMAD_V4_EXTRA_FIELDS,
        tract_fields: Union[str, Sequence[str], None] = "auto",
        biallelic_only: bool = True,
        pass_only: bool = True,
        region_gap: int = 10_000,
        logger: Optional[SimpleLogger] = None,
) -> FrequencyTable:
    """
    Build a :class:`FrequencyTable` from gnomAD sites VCF(s).

    Parameters
    ----------
    paths : str or sequence
        VCF path(s) or glob(s), e.g. ``"gnomad.genomes.v4.1.sites.chr*.vcf.bgz"``.
    regions : DataFrame or "all", optional
        ``chrom,start,end`` (1-based inclusive) regions to read.  Defaults to
        clusters around ``variant_df`` positions (``region_gap``).  ``"all"``
        streams whole files (slow).
    variant_df : DataFrame, optional
        localQTL variant table (``chrom``, ``pos``); used to derive ``regions``
        and to keep only rows at those positions.
    groups : sequence of str
        gnomAD group suffixes to read (``AF_<grp>`` etc.); missing ones are
        skipped with a log line.
    include_global : bool
        Also read the unsuffixed ``AF``/``AC``/``AN`` as group ``"all"``.
    extra_fields : sequence of str
        Additional INFO fields copied as ``info_<field>`` columns (e.g.
        ``info_af_grpmax``); they are not treated as groups.
    tract_fields : "auto", sequence or None
        Local-ancestry tract triplets to keep as groups (``"auto"``: every
        ``AF_/AC_/AN_`` triplet in the header whose suffix is not in ``groups``
        and not a sex / raw / grpmax / joint stratification).
    biallelic_only : bool
        Skip multi-allelic records (otherwise one row per ALT).
    pass_only : bool
        Keep only ``FILTER == PASS``.
    """
    pysam = _try_get_pysam()
    logger = logger or SimpleLogger(verbose=False)
    files = _expand_paths(paths)
    if regions is None:
        if variant_df is None:
            raise ValueError("Pass regions=<DataFrame>, regions='all', or variant_df to derive regions.")
        regions = regions_from_variant_df(variant_df, gap=region_gap)
    groups = [g.lower() for g in groups]
    wanted_pos: Optional[set] = None
    if variant_df is not None:
        wanted_pos = set(zip((normalize_chrom(c) for c in variant_df["chrom"]),
                             pd.to_numeric(variant_df["pos"]).astype("int64")))

    rows: list[dict] = []
    tract_used: dict[str, tuple[str, str, str]] = {}
    for path in files:
        with pysam.VariantFile(path) as vf:
            info_keys = set(vf.header.info.keys())
            trip = _triplets(info_keys)
            cols: dict[str, tuple[str, str, str]] = {}
            if include_global and {"AF", "AC", "AN"} <= info_keys:
                cols["all"] = ("AF", "AC", "AN")
            for g in groups:
                if g in trip:
                    cols[g] = trip[g]
                else:
                    logger.write(f"    ** {os.path.basename(path)}: no AF_/AC_/AN_{g} fields; skipped")
            if tract_fields == "auto":
                for s, t in trip.items():
                    sl = s.lower()
                    if sl in groups or any(tok in sl for tok in _AUTO_EXCLUDE):
                        continue
                    cols[f"lai_{sl}"] = t
                    tract_used[f"lai_{sl}"] = t
            elif tract_fields:
                for s in tract_fields:
                    if s in trip:
                        cols[f"lai_{s.lower()}"] = trip[s]
                        tract_used[f"lai_{s.lower()}"] = trip[s]
                    else:
                        raise ValueError(f"tract field triplet for {s!r} not found in {path}")
            extras = [f for f in extra_fields if f in info_keys]

            if isinstance(regions, str) and regions == "all":
                it = ((normalize_chrom(rec.chrom), rec) for rec in vf.fetch())
            else:
                it = iter_regions(vf, regions, logger=logger)
            for chrom, rec in it:
                if chrom is None:
                    continue
                if pass_only and rec.filter.keys() and "PASS" not in rec.filter.keys():
                    continue
                alts = rec.alts or ()
                if not alts or (biallelic_only and len(alts) > 1):
                    continue
                if wanted_pos is not None and (chrom, int(rec.pos)) not in wanted_pos:
                    continue
                for i, alt in enumerate(alts):
                    row = {"chrom": chrom, "pos": int(rec.pos), "ref": rec.ref, "alt": alt,
                           "rsid": rec.id if rec.id and str(rec.id).startswith("rs") else None}
                    for g, (af_f, ac_f, an_f) in cols.items():
                        row[f"af_{g}"] = _value(rec, af_f, i)
                        row[f"ac_{g}"] = _value(rec, ac_f, i)
                        row[f"an_{g}"] = _value(rec, an_f, i)
                    for f in extras:
                        row[f"info_{f.lower()}"] = _value(rec, f, i)
                    rows.append(row)
    if not rows:
        raise ValueError("No gnomAD records matched the requested regions/filters.")
    df = pd.DataFrame(rows)
    meta = {"files": [os.path.basename(f) for f in files], "groups": groups,
            "tract_fields": {k: list(v) for k, v in tract_used.items()},
            "pass_only": pass_only, "biallelic_only": biallelic_only}
    return FrequencyTable(df, source=f"gnomad:{os.path.basename(files[0])}", metadata=meta)
