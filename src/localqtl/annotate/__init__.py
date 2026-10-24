"""
External molecular-effect and regulatory-link annotation of QTL results.

Planned modules (API defined now, implementations in a later round):

- :mod:`localqtl.annotate.alphagenome` — AlphaGenome Atlas variant-effect
  scores for lead variants (live client with an API key, or a tidy-scores
  file), joined onto results by (variant, gene).
- :mod:`localqtl.annotate.igvf` — IGVF Catalog variant->gene and
  element->gene regulatory links (REST API, or BEDPE files), with
  concordance metrics for ancestry-specific QTL calls.

Importing this package never imports the optional clients.
"""
from . import alphagenome, igvf

__all__ = ["alphagenome", "igvf"]
