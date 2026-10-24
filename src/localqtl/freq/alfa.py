"""
NCBI ALFA (Allele Frequency Aggregator) importer.

ALFA ships one GRCh38 VCF (``freq.vcf.gz`` + ``.tbi``, ~18 GB) whose *sample*
columns are BioSample population accessions (``SAMN...``) with ``FORMAT
AN:AC:...`` per population and ``AC`` per ALT allele.  Chromosomes are RefSeq
accessions (``NC_000007.14`` = chr7); a few rows carry non-GRCh38 accessions
and are dropped.  Only the regions covering your variants are read (tabix).

Requires ``pysam`` (``pip install "localqtl[annotate]"``).  Without pysam, use
the ``bcftools query`` recipe in :mod:`localqtl.freq.bcftools`.
"""
from __future__ import annotations

import os
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from ..utils import SimpleLogger
from ._vcf import _try_get_pysam, iter_regions, regions_from_variant_df
from .table import GRCH38_REFSEQ_TO_CHROM, FrequencyTable, normalize_chrom

__all__ = ["ALFA_URL", "ALFA_POPULATIONS", "ALFA_POPULATION_NAMES", "from_alfa_vcf"]

ALFA_URL = "https://ftp.ncbi.nih.gov/snp/population_frequency/latest_release/freq.vcf.gz"

#: BioSample accession -> population code (ALFA release 5, build 20260205170148).
ALFA_POPULATIONS: dict[str, str] = {
    "SAMN10492695": "EUR", "SAMN10492696": "AFO", "SAMN10492697": "EAS",
    "SAMN10492698": "AFA", "SAMN10492699": "LAC", "SAMN10492700": "LEN",
    "SAMN10492701": "OAS", "SAMN10492702": "SAS", "SAMN11605645": "OTR",
    "SAMN10492703": "AFR", "SAMN10492704": "ASN", "SAMN10492705": "TOT",
}
ALFA_POPULATION_NAMES: dict[str, str] = {
    "EUR": "European", "AFO": "African Others", "EAS": "East Asian",
    "AFA": "African American", "LAC": "Latin American 1 (Afro-Caribbean)",
    "LEN": "Latin American 2 (European/Native American)", "OAS": "Other Asian",
    "SAS": "South Asian", "OTR": "Other", "AFR": "African (AFO + AFA)",
    "ASN": "Asian (excl. South Asian)", "TOT": "Total",
}
_CODE_TO_SAMN = {v: k for k, v in ALFA_POPULATIONS.items()}


def from_alfa_vcf(
        path: str,
        regions: Optional[pd.DataFrame] = None,
        *,
        variant_df: Optional[pd.DataFrame] = None,
        populations: Sequence[str] = tuple(ALFA_POPULATIONS.values()),
        region_gap: int = 10_000,
        logger: Optional[SimpleLogger] = None,
) -> FrequencyTable:
    """
    Build a :class:`FrequencyTable` from the ALFA VCF.

    ``populations`` are ALFA codes (``"EUR"``, ``"AFA"``, ...; see
    :data:`ALFA_POPULATION_NAMES`); group keys in the table are the
    lower-cased codes (``af_eur``, ``af_afa``, ...).  Regions default to
    clusters around ``variant_df`` positions; when ``variant_df`` is given the
    output is restricted to its positions.
    """
    pysam = _try_get_pysam()
    logger = logger or SimpleLogger(verbose=False)
    if regions is None:
        if variant_df is None:
            raise ValueError("Pass regions=<DataFrame> or variant_df to derive regions.")
        regions = regions_from_variant_df(variant_df, gap=region_gap)
    codes = [p.upper() for p in populations]
    unknown = [c for c in codes if c not in _CODE_TO_SAMN]
    if unknown:
        raise ValueError(f"Unknown ALFA population code(s) {unknown}; valid: {sorted(_CODE_TO_SAMN)}")
    wanted_pos: Optional[set] = None
    if variant_df is not None:
        wanted_pos = set(zip((normalize_chrom(c) for c in variant_df["chrom"]),
                             pd.to_numeric(variant_df["pos"]).astype("int64")))

    rows: list[dict] = []
    n_dropped = 0
    with pysam.VariantFile(path) as vf:
        present = set(vf.header.samples)
        samns = {}
        for c in codes:
            s = _CODE_TO_SAMN[c]
            if s in present:
                samns[c] = s
            else:
                logger.write(f"    ** ALFA sample {s} ({c}) not in VCF; skipped")
        for chrom, rec in iter_regions(vf, regions, logger=logger):
            if rec.chrom.startswith("NC_") and rec.chrom not in GRCH38_REFSEQ_TO_CHROM:
                n_dropped += 1
                continue
            alts = rec.alts or ()
            if not alts:
                continue
            if wanted_pos is not None and (chrom, int(rec.pos)) not in wanted_pos:
                continue
            for i, alt in enumerate(alts):
                row = {"chrom": chrom, "pos": int(rec.pos), "ref": rec.ref, "alt": alt,
                       "rsid": rec.id if rec.id and str(rec.id).startswith("rs") else None}
                for c, s in samns.items():
                    fmt = rec.samples[s]
                    an = fmt.get("AN")
                    ac = fmt.get("AC")
                    an = np.nan if an is None else float(an)
                    if isinstance(ac, (tuple, list)):
                        ac = ac[i] if i < len(ac) else None
                    ac = np.nan if ac is None else float(ac)
                    key = c.lower()
                    row[f"an_{key}"] = an
                    row[f"ac_{key}"] = ac
                    row[f"af_{key}"] = ac / an if (np.isfinite(an) and an > 0) else np.nan
                rows.append(row)
    if n_dropped:
        logger.write(f"    ** dropped {n_dropped} ALFA records on non-GRCh38 accessions")
    if not rows:
        raise ValueError("No ALFA records matched the requested regions.")
    df = pd.DataFrame(rows)
    meta = {"file": os.path.basename(path), "populations": codes,
            "population_names": {c: ALFA_POPULATION_NAMES[c] for c in codes},
            "n_dropped_non_grch38": n_dropped}
    return FrequencyTable(df, source="alfa", metadata=meta)
