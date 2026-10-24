"""
AlphaGenome Atlas annotation of lead xQTL variants.

Targeted molecular-effect validation: query the precomputed AlphaGenome Atlas
for gene-level RNA-seq / splicing / chromatin effect scores in matched
tissues, or read a previously exported tidy score table, and join the scores
onto nominal or dosage results so that the direction and magnitude of
(ancestry-specific) effects can be compared with the model's prediction.

The live client needs the ``alphagenome`` SDK and an API key
(``pip install "localqtl[annotate-live]"``); importing this module does not.
The AlphaGenome API and Atlas data are free for **non-commercial use only**;
see https://deepmind.google.com/science/alphagenome for the terms.

Typical use::

    from localqtl.annotate import alphagenome as ag

    scores = ag.score_lead_variants(lead_df, ontology_terms=["UBERON:0001873"],
                                    cache_path="atlas_scores.parquet")
    # or, without the SDK / an API key:
    scores = ag.read_atlas_scores("atlas_scores.parquet")

    annotated = ag.annotate_with_alphagenome(cis_df, scores, variant_df=variant_df)
"""
from __future__ import annotations

import logging
import os
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from ._clients import (_try_get_alphagenome, alphagenome_available, strip_gene_version,
                       variant_keys_for)

__all__ = [
    "ALPHAGENOME_API_KEY_ENV",
    "ALPHAGENOME_DEFAULT_SCORERS",
    "ATLAS_SCORE_COLUMNS",
    "alphagenome_available",
    "atlas_client",
    "score_lead_variants",
    "read_atlas_scores",
    "write_atlas_scores",
    "annotate_with_alphagenome",
]

logger = logging.getLogger(__name__)

#: Environment variable read for the API key when ``api_key`` is not given.
ALPHAGENOME_API_KEY_ENV = "ALPHAGENOME_API_KEY"
#: Output types requested by default (gene-level expression effects).
ALPHAGENOME_DEFAULT_SCORERS = ("RNA_SEQ",)
#: Columns required of a tidy score table (``variant_scorers.tidy_scores``
#: returns these and more, e.g. ``gene_type``, ``track_name``, ``gtex_tissue``).
ATLAS_SCORE_COLUMNS = (
    "variant_id", "gene_id", "gene_name", "output_type", "variant_scorer",
    "ontology_curie", "biosample_name", "raw_score", "quantile_score",
)
_AGG_METHODS = ("max_abs", "mean")


def atlas_client(api_key: Optional[str] = None, **kwargs):
    """
    Create an ``AtlasClient`` (``alphagenome.atlas.atlas.create``).

    ``api_key`` defaults to ``$ALPHAGENOME_API_KEY``.  Extra keyword arguments
    (``timeout``, ``address``) are passed through.
    """
    atlas, _genome, _scorers = _try_get_alphagenome()
    key = api_key or os.environ.get(ALPHAGENOME_API_KEY_ENV)
    if not key:
        raise ValueError(
            f"No AlphaGenome API key: pass api_key= or set ${ALPHAGENOME_API_KEY_ENV}. "
            "Keys are issued at https://deepmind.google.com/science/alphagenome (non-commercial use)."
        )
    return atlas.create(key, **kwargs)


def _variant_frame(lead_df: pd.DataFrame, variant_cols: Sequence[str],
                   variant_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    """``chrom, pos, ref, alt`` for the rows of ``lead_df`` (deduplicated, ordered)."""
    cols = list(variant_cols)
    if all(c in lead_df.columns for c in cols):
        keys = lead_df[cols].copy()
        keys.columns = ["chrom", "pos", "ref", "alt"]
        from ..freq.table import normalize_chrom
        keys["chrom"] = [normalize_chrom(c) for c in keys["chrom"].to_numpy()]
        keys["pos"] = pd.to_numeric(keys["pos"], errors="coerce")
        for c in ("ref", "alt"):
            keys[c] = keys[c].astype(str).str.upper()
    elif "variant_id" in lead_df.columns:
        keys = variant_keys_for(lead_df["variant_id"], variant_df).reset_index(drop=True)
    else:
        raise ValueError(
            f"lead_df needs {cols} or a 'variant_id' column (with variant_df, or IDs that carry the alleles).")
    keys = keys.dropna(subset=["chrom", "pos", "ref", "alt"])
    keys["pos"] = keys["pos"].astype("int64")
    return keys.drop_duplicates(["chrom", "pos", "ref", "alt"]).reset_index(drop=True)


def score_lead_variants(
        lead_df: pd.DataFrame,
        *,
        api_key: Optional[str] = None,
        client=None,
        scorers: Sequence[str] = ALPHAGENOME_DEFAULT_SCORERS,
        ontology_terms: Optional[Sequence[str]] = None,
        gene_ids: Optional[Sequence[str]] = None,
        gene_names: Optional[Sequence[str]] = None,
        variant_cols: Sequence[str] = ("chrom", "pos", "ref", "alt"),
        variant_df: Optional[pd.DataFrame] = None,
        batch_size: int = 100,
        progress_bar: bool = False,
        cache_path: Optional[str] = None,
        logger_: Optional[logging.Logger] = None,
) -> pd.DataFrame:
    """
    Query the AlphaGenome Atlas for every variant of ``lead_df``.

    Parameters
    ----------
    lead_df : DataFrame
        One row per variant, either with ``variant_cols`` (GRCh38) or with a
        ``variant_id`` column whose IDs carry the alleles
        (``chr1_123_A_G_b38``, ``chr1:123:A>C``) or that can be resolved
        through ``variant_df``.
    api_key, client
        An API key (default ``$ALPHAGENOME_API_KEY``) or an existing
        ``AtlasClient``.
    scorers : sequence of str
        Atlas scorer / output-type names (``"RNA_SEQ"``, ``"SPLICE_SITES"``,
        ``"ATAC"``, ...).
    ontology_terms : sequence of str, optional
        UBERON / CL / EFO terms to keep (``"UBERON:0001873"`` caudate nucleus).
    gene_ids, gene_names : sequence of str, optional
        Restrict the returned genes.
    batch_size : int
        Variants per Atlas call.
    cache_path : str, optional
        Write the tidy table here (Parquet or TSV) as it is returned, so that
        later runs can use :func:`read_atlas_scores` without an API key.

    Returns
    -------
    DataFrame
        Tidy scores, one row per variant x gene x scorer x track, with at
        least :data:`~localqtl.annotate.alphagenome.ATLAS_SCORE_COLUMNS`.
    """
    log = logger_ or logger
    atlas_mod, genome, variant_scorers = _try_get_alphagenome()
    del atlas_mod
    keys = _variant_frame(lead_df, variant_cols, variant_df)
    if keys.empty:
        return pd.DataFrame(columns=list(ATLAS_SCORE_COLUMNS))
    if client is None:
        client = atlas_client(api_key)

    variants = [
        genome.Variant(chromosome=str(r.chrom), position=int(r.pos),
                       reference_bases=str(r.ref), alternate_bases=str(r.alt))
        for r in keys.itertuples(index=False)
    ]
    log.info("AlphaGenome Atlas: %d variants, scorers=%s", len(variants), list(scorers))

    frames = []
    step = max(1, int(batch_size))
    for start in range(0, len(variants), step):
        batch = variants[start:start + step]
        result = client.query_variants(
            batch, requested_scorers=list(scorers), ontology_terms=ontology_terms,
            gene_ids=gene_ids, gene_names=gene_names, progress_bar=progress_bar,
        )
        # tidy_scores takes a nested sequence of AnnData (one inner sequence per call)
        tidy = variant_scorers.tidy_scores([list(result.values())])
        if tidy is not None and len(tidy):
            frames.append(tidy)
    if not frames:
        out = pd.DataFrame(columns=list(ATLAS_SCORE_COLUMNS))
    else:
        out = pd.concat(frames, ignore_index=True)
        for col in ATLAS_SCORE_COLUMNS:
            if col not in out.columns:
                out[col] = np.nan
    if cache_path:
        write_atlas_scores(out, cache_path)
        log.info("AlphaGenome Atlas: wrote %d score rows to %s", len(out), cache_path)
    return out


def write_atlas_scores(scores_df: pd.DataFrame, path: str) -> str:
    """Write a tidy score table to Parquet (``.parquet``) or TSV; returns ``path``."""
    df = scores_df.copy()
    # tidy_scores keeps rich objects (Variant, Interval) that Parquet cannot hold
    for col in df.columns:
        if df[col].dtype == object and len(df) and not isinstance(df[col].iloc[0], (str, bytes, type(None))):
            df[col] = df[col].astype(str)
    if str(path).endswith(".parquet"):
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, sep="\t", index=False)
    return path


def read_atlas_scores(path: str) -> pd.DataFrame:
    """
    Read an exported tidy score table (Parquet or TSV, optionally gzipped).

    The table must carry :data:`~localqtl.annotate.alphagenome.ATLAS_SCORE_COLUMNS`; this is the offline
    fallback for :func:`score_lead_variants`.
    """
    if str(path).endswith(".parquet"):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path, sep="\t")
    missing = [c for c in ("variant_id", "raw_score") if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is not a tidy AlphaGenome score table (missing {missing}); "
                         f"expected columns include {list(ATLAS_SCORE_COLUMNS)}")
    for col in ATLAS_SCORE_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    return df


def annotate_with_alphagenome(
        results_df: pd.DataFrame,
        scores_df: pd.DataFrame,
        *,
        variant_df: Optional[pd.DataFrame] = None,
        gene_col: str = "phenotype_id",
        gene_id_col: str = "gene_id",
        agg: str = "max_abs",
        slope_col: str = "slope",
        output_type: Optional[str] = None,
        ontology_terms: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """
    Join Atlas scores onto QTL results by (variant, gene).

    Variants are matched on ``(chrom, pos, ref, alt)``: the score table's
    ``variant_id`` is parsed (``chr22:36201698:A>C``), and the results'
    ``variant_id`` is resolved through ``variant_df`` when given, otherwise
    parsed the same way.  Gene IDs are compared without their version suffix,
    so ``ENSG00000103494.12`` matches ``ENSG00000103494``.

    Parameters
    ----------
    results_df : DataFrame
        Output of ``map_nominal`` / ``map_permutations`` / the dosage scan,
        with ``variant_id`` and ``gene_col``.
    scores_df : DataFrame
        Tidy Atlas scores (:func:`score_lead_variants` or
        :func:`read_atlas_scores`).
    agg : {"max_abs", "mean"}
        Which track to keep per (variant, gene): the one with the largest
        ``|raw_score|``, or the mean over tracks.
    slope_col : str
        Effect-size column used for the sign comparison; for the dosage model
        ``slope`` carries the ancestry with the strongest effect.
    output_type, ontology_terms
        Optional filters applied to ``scores_df`` first.

    Returns
    -------
    DataFrame
        ``results_df`` with ``ag_raw_score``, ``ag_quantile_score``,
        ``ag_biosample``, ``ag_n_tracks`` and ``ag_sign_concordant``
        (NaN where the pair has no Atlas score).
    """
    if agg not in _AGG_METHODS:
        raise ValueError(f"agg must be one of {_AGG_METHODS}")
    if "variant_id" not in results_df.columns:
        raise ValueError("results_df must have a 'variant_id' column.")
    if gene_col not in results_df.columns:
        raise ValueError(f"results_df must have the gene column {gene_col!r}.")
    out = results_df.copy()
    new_cols = ["ag_raw_score", "ag_quantile_score", "ag_biosample", "ag_n_tracks", "ag_sign_concordant"]

    scores = scores_df
    if output_type is not None and "output_type" in scores.columns:
        scores = scores[scores["output_type"].astype(str) == str(output_type)]
    if ontology_terms is not None and "ontology_curie" in scores.columns:
        scores = scores[scores["ontology_curie"].astype(str).isin([str(t) for t in ontology_terms])]
    if scores is None or scores.empty:
        for c in new_cols:
            out[c] = np.nan
        out["ag_n_tracks"] = 0
        return out

    s_keys = variant_keys_for(scores["variant_id"], None).reset_index(drop=True)
    s = pd.DataFrame({
        "chrom": s_keys["chrom"].to_numpy(), "pos": s_keys["pos"].to_numpy(),
        "ref": s_keys["ref"].to_numpy(), "alt": s_keys["alt"].to_numpy(),
        "gene": strip_gene_version(scores[gene_id_col].fillna("")).to_numpy()
        if gene_id_col in scores.columns else "",
        "raw_score": pd.to_numeric(scores["raw_score"], errors="coerce").to_numpy(),
        "quantile_score": pd.to_numeric(scores.get("quantile_score", np.nan), errors="coerce").to_numpy()
        if "quantile_score" in scores.columns else np.nan,
        "biosample": (scores["biosample_name"].astype(str).to_numpy()
                      if "biosample_name" in scores.columns else ""),
    }).dropna(subset=["chrom", "pos", "raw_score"])
    if s.empty:
        for c in new_cols:
            out[c] = np.nan
        out["ag_n_tracks"] = 0
        return out
    s["pos"] = s["pos"].astype("int64")

    grp = s.groupby(["chrom", "pos", "ref", "alt", "gene"], sort=False)
    if agg == "max_abs":
        idx = grp["raw_score"].apply(lambda v: v.abs().idxmax())
        best = s.loc[idx.to_numpy()].copy()
        best["ag_n_tracks"] = grp.size().to_numpy()
        best = best.rename(columns={"raw_score": "ag_raw_score", "quantile_score": "ag_quantile_score",
                                    "biosample": "ag_biosample"})
    else:
        best = grp.agg(ag_raw_score=("raw_score", "mean"),
                       ag_quantile_score=("quantile_score", "mean"),
                       ag_n_tracks=("raw_score", "size")).reset_index()
        best["ag_biosample"] = "mean"
    best = best[["chrom", "pos", "ref", "alt", "gene", "ag_raw_score", "ag_quantile_score",
                 "ag_biosample", "ag_n_tracks"]]

    r_keys = variant_keys_for(out["variant_id"], variant_df).reset_index(drop=True)
    left = pd.DataFrame({
        "chrom": r_keys["chrom"].to_numpy(), "pos": r_keys["pos"].to_numpy(),
        "ref": r_keys["ref"].to_numpy(), "alt": r_keys["alt"].to_numpy(),
        "gene": strip_gene_version(out[gene_col]).to_numpy(),
    })
    left["pos"] = pd.to_numeric(left["pos"], errors="coerce")
    merged = left.merge(best, on=["chrom", "pos", "ref", "alt", "gene"], how="left")

    out["ag_raw_score"] = merged["ag_raw_score"].to_numpy()
    out["ag_quantile_score"] = merged["ag_quantile_score"].to_numpy()
    out["ag_biosample"] = merged["ag_biosample"].to_numpy()
    out["ag_n_tracks"] = merged["ag_n_tracks"].fillna(0).astype("int64").to_numpy()
    if slope_col in out.columns:
        slope = pd.to_numeric(out[slope_col], errors="coerce").to_numpy()
        score = merged["ag_raw_score"].to_numpy()
        concordant = pd.array(np.sign(slope) == np.sign(score), dtype="boolean")
        concordant[np.isnan(slope) | np.isnan(score) | (score == 0)] = pd.NA
        out["ag_sign_concordant"] = concordant
    else:
        out["ag_sign_concordant"] = pd.array([pd.NA] * len(out), dtype="boolean")
    return out
