"""
External molecular-effect and regulatory-link annotation of QTL results.

- :mod:`localqtl.annotate.alphagenome` — AlphaGenome Atlas variant-effect
  scores for lead variants (live client with an API key, or a previously
  exported tidy-score file), joined onto results by (variant, gene).
- :mod:`localqtl.annotate.igvf` — IGVF Catalog variant->gene and
  element->gene regulatory links (REST API, cached tables or BEDPE files),
  with concordance metrics for ancestry-specific QTL calls.

Importing this package never imports the optional clients (``requests``, the
``alphagenome`` SDK); install them with ``pip install "localqtl[annotate-live]"``.
"""
from . import alphagenome, igvf
from .alphagenome import (alphagenome_available, annotate_with_alphagenome, read_atlas_scores,
                          score_lead_variants)
from .igvf import (compare_with_regulatory_links, query_element_gene_links,
                   query_variant_gene_links, read_bedpe_links, read_links, requests_available,
                   summarize_link_concordance)

__all__ = [
    "alphagenome",
    "igvf",
    # AlphaGenome Atlas
    "alphagenome_available",
    "score_lead_variants",
    "read_atlas_scores",
    "annotate_with_alphagenome",
    # IGVF Catalog
    "requests_available",
    "query_variant_gene_links",
    "query_element_gene_links",
    "read_bedpe_links",
    "read_links",
    "compare_with_regulatory_links",
    "summarize_link_concordance",
]
