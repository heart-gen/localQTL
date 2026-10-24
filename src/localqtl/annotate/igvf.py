"""
IGVF Catalog regulatory-link comparison (planned; API stubs).

Compare ancestry-specific molecular-QTL calls with independently supported
regulatory links from the IGVF Catalog (https://catalog.igvf.org):
variant->gene edges (eQTL / spliceQTL from the eQTL Catalogue and AFGR,
CRISPR screens, Variant-EFFECTS) and element->gene edges (CRISPR screens,
ENCODE-rE2G, scE2G, Perturb-seq), queried through the public REST API or read
from downloaded BEDPE prediction sets.
"""
from __future__ import annotations

from typing import Optional, Sequence

import pandas as pd

__all__ = [
    "IGVF_CATALOG_BASE",
    "IGVF_VARIANT_GENE_METHODS",
    "IGVF_ELEMENT_GENE_METHODS",
    "IGVF_SOURCES",
    "IGVF_MAX_REGION_BP",
    "query_variant_gene_links",
    "query_element_gene_links",
    "read_bedpe_links",
    "compare_with_regulatory_links",
    "summarize_link_concordance",
]

IGVF_CATALOG_BASE = "https://api.catalogkg.igvf.org/api"
#: ``method`` values of ``GET /variants/genes``.
IGVF_VARIANT_GENE_METHODS = ("CRISPR screen", "Variant-EFFECTS", "eQTL", "spliceQTL")
#: ``method`` values of ``GET /genomic-elements/genes`` / ``/enhancer-gene-predictions``.
IGVF_ELEMENT_GENE_METHODS = ("CRISPR screen", "ENCODE-rE2G", "Perturb-seq", "scE2G")
#: ``source`` values accepted by the variant->gene endpoint.
IGVF_SOURCES = ("AFGR", "EBI", "IGVF")
#: Maximum ``region`` span accepted by the API.
IGVF_MAX_REGION_BP = 10_000
IGVF_PAGE_LIMIT = 500

_PLANNED = "localqtl.annotate.igvf is planned for a later release."


def query_variant_gene_links(
        lead_df: pd.DataFrame,
        *,
        methods: Optional[Sequence[str]] = None,
        sources: Optional[Sequence[str]] = None,
        biosample_term: Optional[str] = None,
        page_size: int = IGVF_PAGE_LIMIT,
        session=None,
        cache_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    ``GET /variants/genes`` for every lead variant (by rsID, SPDI or a
    ≤10 kb region around the position).

    Returns ``variant_id, spdi, rsid, gene_id, gene_name, method, source,
    biosample_term, effect_size, neg_log10_pvalue, label``.
    """
    raise NotImplementedError(_PLANNED)


def query_element_gene_links(
        regions_df: pd.DataFrame,
        *,
        methods: Optional[Sequence[str]] = None,
        sources: Optional[Sequence[str]] = None,
        biosample_term: Optional[str] = None,
        page_size: int = IGVF_PAGE_LIMIT,
        session=None,
) -> pd.DataFrame:
    """
    ``GET /genomic-elements/genes`` for ``chrom,start,end`` regions (split
    into ≤10 kb chunks).  Returns ``chrom, start, end, element_id, gene_id,
    gene_name, score, method, source, biosample_term``.
    """
    raise NotImplementedError(_PLANNED)


def read_bedpe_links(path: str, *, gene_col: Optional[str] = None,
                     score_col: Optional[str] = None) -> pd.DataFrame:
    """Read an IGVF / ENCODE element->gene BEDPE prediction set into the element-link layout."""
    raise NotImplementedError(_PLANNED)


def compare_with_regulatory_links(
        results_df: pd.DataFrame,
        links_df: pd.DataFrame,
        variant_df: pd.DataFrame,
        *,
        gene_col: str = "phenotype_id",
        max_distance: int = 0,
        per_ancestry: bool = True,
        alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Flag QTL pairs whose variant lies in (or within ``max_distance`` of) an
    element linked to the same gene, or that has a direct variant->gene link.

    Adds ``n_links, same_gene_support, best_method, best_source,
    distance_to_element`` and, for dosage results, ``supported_anc{k}``
    (``pval_anc{k} < alpha`` and same-gene support).
    """
    raise NotImplementedError(_PLANNED)


def summarize_link_concordance(df: pd.DataFrame, *, by: Sequence[str] = ("ancestry",)) -> pd.DataFrame:
    """Tidy concordance summary (fraction supported, by ancestry / method / source)."""
    raise NotImplementedError(_PLANNED)
