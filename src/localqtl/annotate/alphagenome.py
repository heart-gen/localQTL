"""
AlphaGenome Atlas annotation (planned; API stubs).

Targeted molecular-effect validation of lead xQTL variants: query the
precomputed AlphaGenome Atlas (``alphagenome.atlas.atlas.create(api_key)`` ->
``AtlasClient.query_variants(...)``) for gene-level RNA-seq / splicing /
chromatin effect scores in matched tissues, or read a previously exported
``variant_scorers.tidy_scores`` table, and join the scores onto nominal /
dosage results so that the direction and magnitude of ancestry-specific
effects can be compared with the model's prediction.

The AlphaGenome API and Atlas data are free for non-commercial use only; see
https://deepmind.google.com/science/alphagenome for the terms.
"""
from __future__ import annotations

from typing import Optional, Sequence

import pandas as pd

__all__ = [
    "ALPHAGENOME_API_KEY_ENV",
    "ALPHAGENOME_DEFAULT_SCORERS",
    "ATLAS_SCORE_COLUMNS",
    "score_lead_variants",
    "read_atlas_scores",
    "annotate_with_alphagenome",
]

#: Environment variable read for the API key when ``api_key`` is not given.
ALPHAGENOME_API_KEY_ENV = "ALPHAGENOME_API_KEY"
#: Output types requested by default (gene-level expression effects).
ALPHAGENOME_DEFAULT_SCORERS = ("RNA_SEQ",)
#: Columns of the tidy score table (``variant_scorers.tidy_scores`` layout).
ATLAS_SCORE_COLUMNS = (
    "variant_id", "gene_id", "gene_name", "output_type", "variant_scorer",
    "ontology_curie", "biosample_name", "raw_score", "quantile_score",
)

_PLANNED = "localqtl.annotate.alphagenome is planned for a later release."


def score_lead_variants(
        lead_df: pd.DataFrame,
        *,
        api_key: Optional[str] = None,
        client=None,
        scorers: Sequence[str] = ALPHAGENOME_DEFAULT_SCORERS,
        ontology_terms: Optional[Sequence[str]] = None,
        gene_ids: Optional[Sequence[str]] = None,
        variant_cols: Sequence[str] = ("chrom", "pos", "ref", "alt"),
        batch_size: int = 100,
) -> pd.DataFrame:
    """
    Query the AlphaGenome Atlas for every row of ``lead_df``.

    Parameters
    ----------
    lead_df : DataFrame
        One row per variant with ``variant_cols`` (GRCh38) and optionally
        ``gene_id`` to restrict the returned genes.
    api_key, client
        Either an API key (defaults to ``$ALPHAGENOME_API_KEY``) or an
        existing ``AtlasClient``.
    scorers : sequence of str
        Atlas scorer names / output types (``"RNA_SEQ"``, ``"SPLICE_SITES"``,
        ``"ATAC"``, ...).
    ontology_terms : sequence of str, optional
        UBERON / CL terms to keep (e.g. ``"UBERON:0001873"`` caudate nucleus).

    Returns
    -------
    DataFrame with :data:`ATLAS_SCORE_COLUMNS` (one row per variant x gene x
    scorer x track).
    """
    raise NotImplementedError(_PLANNED)


def read_atlas_scores(path: str) -> pd.DataFrame:
    """Read an exported tidy score table (Parquet/TSV) with :data:`ATLAS_SCORE_COLUMNS`."""
    raise NotImplementedError(_PLANNED)


def annotate_with_alphagenome(
        results_df: pd.DataFrame,
        scores_df: pd.DataFrame,
        *,
        gene_col: str = "phenotype_id",
        gene_id_col: str = "gene_id",
        agg: str = "max_abs",
) -> pd.DataFrame:
    """
    Join Atlas scores onto results by ``(variant_id, gene)``.

    Adds ``ag_raw_score``, ``ag_quantile_score``, ``ag_biosample``,
    ``ag_n_tracks`` and ``ag_sign_concordant`` (sign of ``slope`` / the
    strongest ``slope_anc{k}`` versus ``raw_score``).  ``agg`` selects the
    track kept per pair (``"max_abs"`` or ``"mean"``).
    """
    raise NotImplementedError(_PLANNED)
