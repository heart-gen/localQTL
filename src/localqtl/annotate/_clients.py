"""
Lazy imports and small shared helpers for the optional annotation clients.

Neither ``requests`` nor the AlphaGenome SDK is a hard dependency; importing
:mod:`localqtl.annotate` never imports them.  The accessors here mirror
:func:`localqtl.utils._try_get_cupy` / :func:`localqtl.freq._vcf._try_get_pysam`.
"""
from __future__ import annotations

import logging
import re
from typing import Optional

import numpy as np
import pandas as pd

__all__ = [
    "requests_available",
    "alphagenome_available",
    "parse_variant_key",
    "variant_keys_for",
    "strip_gene_version",
]

logger = logging.getLogger(__name__)

_REQUESTS_HINT = 'requests is required for the IGVF Catalog client: pip install "localqtl[annotate-live]"'
_ALPHAGENOME_HINT = ('the alphagenome SDK is required for live Atlas queries: '
                     'pip install "localqtl[annotate-live]" (non-commercial use only; '
                     'see https://deepmind.google.com/science/alphagenome)')

_REQUESTS_AVAILABLE: Optional[bool] = None
_ALPHAGENOME_AVAILABLE: Optional[bool] = None


def _try_get_requests(force: bool = True):
    global _REQUESTS_AVAILABLE
    if _REQUESTS_AVAILABLE is False and not force:
        return None
    try:
        import requests
    except Exception as e:  # pragma: no cover - depends on environment
        _REQUESTS_AVAILABLE = False
        if force:
            raise ImportError(_REQUESTS_HINT) from e
        logger.warning(_REQUESTS_HINT)
        return None
    _REQUESTS_AVAILABLE = True
    return requests


def requests_available() -> bool:
    return _try_get_requests(force=False) is not None


def _try_get_alphagenome(force: bool = True):
    """Return ``(atlas, genome, variant_scorers)`` from the AlphaGenome SDK."""
    global _ALPHAGENOME_AVAILABLE
    if _ALPHAGENOME_AVAILABLE is False and not force:
        return None
    try:
        from alphagenome.atlas import atlas
        from alphagenome.data import genome
        from alphagenome.models import variant_scorers
    except Exception as e:  # pragma: no cover - depends on environment
        _ALPHAGENOME_AVAILABLE = False
        if force:
            raise ImportError(_ALPHAGENOME_HINT) from e
        logger.warning(_ALPHAGENOME_HINT)
        return None
    _ALPHAGENOME_AVAILABLE = True
    return atlas, genome, variant_scorers


def alphagenome_available() -> bool:
    return _try_get_alphagenome(force=False) is not None


# ``chr22:36201698:A>C`` (AlphaGenome) and ``chr22_36201698_A_C_b38`` (GTEx) forms.
_AG_VARIANT_RE = re.compile(
    r"^(?P<chrom>[^:\s]+):(?P<pos>\d+):(?P<ref>[ACGTN]+)>(?P<alt>[ACGTN]+)$", re.IGNORECASE)


def parse_variant_key(variant_ids) -> pd.DataFrame:
    """
    ``chrom, pos, ref, alt`` parsed from variant IDs shaped like
    ``chr22:36201698:A>C`` (AlphaGenome) or ``chr22_36201698_A_C_b38`` /
    ``22:36201698:A:C`` (GTEx / tensorQTL).  Unparsable IDs give NaN.
    """
    from ..freq.table import normalize_chrom
    from ..genotypeio import _VARIANT_ID_PATTERN

    ids = pd.Index(pd.Series(list(variant_ids)).astype(str))
    chrom, pos, ref, alt = [], [], [], []
    for vid in ids:
        m = _AG_VARIANT_RE.match(vid) or _VARIANT_ID_PATTERN.match(vid)
        if m is None:
            chrom.append(None); pos.append(np.nan); ref.append(None); alt.append(None)
            continue
        chrom.append(normalize_chrom(m.group("chrom")))
        pos.append(int(m.group("pos")))
        ref.append(m.group("ref").upper())
        alt.append(m.group("alt").upper())
    return pd.DataFrame({"chrom": chrom, "pos": pos, "ref": ref, "alt": alt}, index=ids)


def variant_keys_for(variant_ids, variant_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    ``chrom, pos, ref, alt`` for ``variant_ids``: looked up in ``variant_df``
    when given (the mapping results were produced with), otherwise parsed from
    the IDs themselves.  Indexed by the input IDs, duplicates kept.
    """
    ids = pd.Index(pd.Series(list(variant_ids)).astype(str))
    if variant_df is None:
        return parse_variant_key(ids)
    from ..freq.table import ensure_variant_keys
    keys = ensure_variant_keys(variant_df, require_alleles=False)
    keys.index = keys.index.astype(str)
    out = keys.reindex(ids)
    missing = out["chrom"].isna()
    if missing.any():  # IDs absent from variant_df: fall back to parsing
        parsed = parse_variant_key(ids[missing])
        out.loc[missing, ["chrom", "pos", "ref", "alt"]] = parsed.to_numpy()
    return out


def strip_gene_version(genes) -> pd.Index:
    """``ENSG00000103494.12`` -> ``ENSG00000103494`` (other IDs unchanged)."""
    s = pd.Series(list(genes), dtype=object).astype(str)
    return pd.Index(s.str.replace(r"^(ENS[A-Z]*G\d+)\.\d+$", r"\1", regex=True))
