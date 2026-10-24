"""
IGVF Catalog regulatory-link comparison.

Compare ancestry-specific molecular-QTL calls with independently supported
regulatory links from the IGVF Catalog (https://catalog.igvf.org):
variant->gene edges (eQTL / spliceQTL from the eQTL Catalogue and AFGR, CRISPR
screens, Variant-EFFECTS) and element->gene edges (CRISPR screens,
ENCODE-rE2G, scE2G, Perturb-seq), queried through the public REST API or read
from downloaded BEDPE prediction sets.

The REST client needs ``requests`` (``pip install "localqtl[annotate-live]"``);
importing this module does not.  Note that the Catalog reports **0-based**
variant positions (its ``NC_000015.10:28125242:C:T`` is 1-based 28125243);
every position returned here is converted to the 1-based convention used by
``variant_df``.

Typical use::

    from localqtl.annotate import igvf

    links = igvf.query_variant_gene_links(lead_df, variant_df=variant_df,
                                          methods=["eQTL"], cache_path="igvf_links.parquet")
    elements = igvf.query_element_gene_links(regions_df, methods=["ENCODE-rE2G"])
    compared = igvf.compare_with_regulatory_links(cis_df, links, variant_df)
    summary = igvf.summarize_link_concordance(compared)
"""
from __future__ import annotations

import logging
import time
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from ._clients import _try_get_requests, requests_available, strip_gene_version, variant_keys_for

__all__ = [
    "IGVF_CATALOG_BASE",
    "IGVF_VARIANT_GENE_METHODS",
    "IGVF_ELEMENT_GENE_METHODS",
    "IGVF_SOURCES",
    "IGVF_MAX_REGION_BP",
    "IGVF_PAGE_LIMIT",
    "VARIANT_LINK_COLUMNS",
    "ELEMENT_LINK_COLUMNS",
    "requests_available",
    "query_variant_gene_links",
    "query_element_gene_links",
    "read_links",
    "write_links",
    "read_bedpe_links",
    "compare_with_regulatory_links",
    "summarize_link_concordance",
]

logger = logging.getLogger(__name__)

IGVF_CATALOG_BASE = "https://api.catalogkg.igvf.org/api"
#: ``method`` values of ``GET /variants/genes``.
IGVF_VARIANT_GENE_METHODS = ("CRISPR screen", "Variant-EFFECTS", "eQTL", "spliceQTL")
#: ``method`` values of ``GET /genomic-elements/genes``.
IGVF_ELEMENT_GENE_METHODS = ("CRISPR screen", "ENCODE-rE2G", "Perturb-seq", "scE2G")
#: ``source`` values accepted by the variant->gene endpoint.
IGVF_SOURCES = ("AFGR", "EBI", "IGVF")
#: Maximum ``region`` span accepted by the API (larger spans are rejected).
IGVF_MAX_REGION_BP = 10_000
#: Maximum ``limit`` per page.
IGVF_PAGE_LIMIT = 500

#: Columns of a variant->gene link table.
VARIANT_LINK_COLUMNS = (
    "variant_id", "chrom", "pos", "ref", "alt", "gene_id", "gene_name", "method", "source",
    "biosample_term", "biological_context", "effect_size", "p_value", "neg_log10_pvalue", "label",
)
#: Columns of an element->gene link table.
ELEMENT_LINK_COLUMNS = (
    "chrom", "start", "end", "element_id", "gene_id", "gene_name", "score", "method", "source",
    "biosample_term", "biological_context", "p_value", "neg_log10_pvalue", "significant",
)


# ---------------------------------------------------------------------------
# REST helpers
# ---------------------------------------------------------------------------
def _strip_collection(value) -> Optional[str]:
    """``genes/ENSG001`` -> ``ENSG001``; ``ontology_terms/CL_0000546`` -> ``CL_0000546``."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    s = str(value)
    return s.split("/", 1)[1] if "/" in s else s


def _get_json(session, url: str, params: dict, *, timeout: float, retries: int, backoff: float):
    """GET with retries on 429 / 5xx; raises for other errors."""
    last = None
    for attempt in range(max(1, int(retries))):
        resp = session.get(url, params=params, timeout=timeout)
        if resp.status_code == 200:
            return resp.json()
        last = resp
        if resp.status_code in (429, 500, 502, 503, 504) and attempt + 1 < retries:
            time.sleep(backoff * (2 ** attempt))
            continue
        break
    detail = ""
    try:
        detail = f": {last.json().get('message', '')}" if last is not None else ""
    except Exception:
        detail = f": {last.text[:200]}" if last is not None else ""
    raise RuntimeError(f"IGVF Catalog request failed ({last.status_code if last else '?'}) "
                       f"for {url} {params}{detail}")


def _paged_rows(session, url: str, params: dict, *, page_size: int, max_pages: int,
                timeout: float, retries: int, backoff: float, delay: float) -> list:
    """Collect every page of a Catalog endpoint (``page`` is 0-based)."""
    rows: list = []
    limit = min(int(page_size), IGVF_PAGE_LIMIT)
    for page in range(int(max_pages)):
        payload = _get_json(session, url, {**params, "page": page, "limit": limit},
                            timeout=timeout, retries=retries, backoff=backoff)
        if isinstance(payload, dict):  # error payloads are objects, data pages are arrays
            raise RuntimeError(f"Unexpected IGVF Catalog response for {url} {params}: {payload}")
        rows.extend(payload)
        if len(payload) < limit:
            break
        if delay:
            time.sleep(delay)
    else:
        logger.warning("IGVF Catalog: stopped after max_pages=%d for %s %s", max_pages, url, params)
    return rows


def _make_session(session=None):
    if session is not None:
        return session
    requests = _try_get_requests()
    s = requests.Session()
    s.headers.update({"Accept": "application/json", "User-Agent": "localqtl"})
    return s


def _region_strings(regions: pd.DataFrame) -> list[str]:
    """``chr:start-end`` strings of at most :data:`~localqtl.annotate.igvf.IGVF_MAX_REGION_BP`, 1-based inclusive input."""
    out = []
    for chrom, start, end in regions[["chrom", "start", "end"]].itertuples(index=False):
        start, end = int(start), int(end)
        for s in range(start, end + 1, IGVF_MAX_REGION_BP):
            e = min(s + IGVF_MAX_REGION_BP - 1, end)
            out.append(f"{chrom}:{max(s - 1, 0)}-{e}")
    return out


def _variant_regions(lead_df: pd.DataFrame, variant_df: Optional[pd.DataFrame],
                     pad: int = 1) -> tuple[pd.DataFrame, pd.DataFrame]:
    """``(keys, regions)``: per-variant keys (1-based) and merged ≤10 kb query regions."""
    from ..freq._vcf import merge_intervals
    if "variant_id" not in lead_df.columns:
        raise ValueError("lead_df must have a 'variant_id' column.")
    keys = variant_keys_for(lead_df["variant_id"], variant_df)
    keys = keys.dropna(subset=["chrom", "pos"]).copy()
    keys["pos"] = keys["pos"].astype("int64")
    if keys.empty:
        return keys, pd.DataFrame(columns=["chrom", "start", "end"])
    reg = pd.DataFrame({"chrom": keys["chrom"].to_numpy(),
                        "start": np.maximum(keys["pos"].to_numpy() - pad, 1),
                        "end": keys["pos"].to_numpy() + pad})
    return keys, merge_intervals(reg, gap=IGVF_MAX_REGION_BP // 2)


def _parse_catalog_variant(spdi) -> tuple[Optional[str], float, Optional[str], Optional[str]]:
    """``variants/NC_000015.10:28125242:C:T`` -> ``("chr15", 28125243, "C", "T")`` (1-based)."""
    from ..freq.table import GRCH38_REFSEQ_TO_CHROM, normalize_chrom
    s = _strip_collection(spdi)
    if not s:
        return None, np.nan, None, None
    parts = s.split(":")
    if len(parts) < 4:
        return None, np.nan, None, None
    acc, pos, ref, alt = parts[0], parts[1], parts[2], parts[3]
    chrom = GRCH38_REFSEQ_TO_CHROM.get(acc)
    chrom = f"chr{chrom}" if chrom else normalize_chrom(acc)
    try:
        pos1 = int(pos) + 1       # the Catalog reports 0-based positions
    except ValueError:
        return chrom, np.nan, None, None
    return chrom, pos1, (ref or None), (alt or None)


# ---------------------------------------------------------------------------
# queries
# ---------------------------------------------------------------------------
def query_variant_gene_links(
        lead_df: pd.DataFrame,
        *,
        variant_df: Optional[pd.DataFrame] = None,
        methods: Optional[Sequence[str]] = None,
        sources: Optional[Sequence[str]] = None,
        biosample_term: Optional[str] = None,
        min_neg_log10_pvalue: Optional[float] = None,
        exact_variant_match: bool = True,
        page_size: int = IGVF_PAGE_LIMIT,
        max_pages: int = 100,
        session=None,
        base_url: str = IGVF_CATALOG_BASE,
        timeout: float = 60.0,
        retries: int = 3,
        backoff: float = 1.0,
        delay: float = 0.0,
        cache_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    ``GET /variants/genes`` for the lead variants of ``lead_df``.

    The Catalog is queried by genomic region (at most 10 kb per request;
    positions around the lead variants are merged into as few regions as
    possible), then rows are matched back to the input variants.

    Parameters
    ----------
    lead_df : DataFrame
        Must have ``variant_id``; alleles come from ``variant_df`` or from IDs
        that carry them.
    methods, sources : sequence of str, optional
        Filters, e.g. ``["eQTL", "spliceQTL"]`` (:data:`~localqtl.annotate.igvf.IGVF_VARIANT_GENE_METHODS`)
        and ``["EBI", "AFGR"]`` (:data:`~localqtl.annotate.igvf.IGVF_SOURCES`). Each value is a separate
        request.
    biosample_term : str, optional
        Ontology term of the biosample (e.g. ``"UBERON_0006631"``).
    min_neg_log10_pvalue : float, optional
        Server-side ``neg_log10_pvalue >= x`` filter.
    exact_variant_match : bool
        Keep only links whose variant matches an input variant on
        ``(chrom, pos, ref, alt)``; with ``False`` every link in the queried
        regions is returned (``variant_id`` is then the Catalog's SPDI).
    cache_path : str, optional
        Write the normalised table here (Parquet or TSV) for offline reuse
        with :func:`read_links`.

    Returns
    -------
    DataFrame with :data:`~localqtl.annotate.igvf.VARIANT_LINK_COLUMNS`.
    """
    keys, regions = _variant_regions(lead_df, variant_df)
    if regions.empty:
        return pd.DataFrame(columns=list(VARIANT_LINK_COLUMNS))
    sess = _make_session(session)
    url = f"{base_url.rstrip('/')}/variants/genes"
    base_params: dict = {}
    if biosample_term:
        base_params["biosample_term"] = biosample_term
    if min_neg_log10_pvalue is not None:
        base_params["neg_log10_pvalue"] = f"gte:{min_neg_log10_pvalue}"
    combos = [{}]
    if methods:
        combos = [{**c, "method": m} for c in combos for m in methods]
    if sources:
        combos = [{**c, "source": s} for c in combos for s in sources]

    region_strings = _region_strings(regions)
    logger.info("IGVF Catalog: %d variant->gene region queries (%d filter combinations)",
                len(region_strings), len(combos))
    rows: list = []
    for region in region_strings:
        for combo in combos:
            rows.extend(_paged_rows(sess, url, {**base_params, **combo, "region": region},
                                    page_size=page_size, max_pages=max_pages, timeout=timeout,
                                    retries=retries, backoff=backoff, delay=delay))
    out = _normalise_variant_rows(rows)
    if exact_variant_match and not out.empty:
        out = _match_to_input(out, keys)
    if cache_path:
        write_links(out, cache_path)
    return out


def _normalise_variant_rows(rows: list) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=list(VARIANT_LINK_COLUMNS))
    raw = pd.DataFrame(rows)
    gene = raw.get("gene")
    gene_id, gene_name = [], []
    for g in (gene if gene is not None else []):
        if isinstance(g, dict):
            gene_id.append(_strip_collection(g.get("_id")))
            gene_name.append(g.get("name"))
        else:
            gene_id.append(_strip_collection(g))
            gene_name.append(None)
    parsed = [_parse_catalog_variant(v) for v in raw.get("sequence_variant", pd.Series([None] * len(raw)))]
    out = pd.DataFrame({
        "variant_id": [_strip_collection(v) for v in raw.get("sequence_variant", pd.Series([None] * len(raw)))],
        "chrom": [p[0] for p in parsed],
        "pos": [p[1] for p in parsed],
        "ref": [p[2] for p in parsed],
        "alt": [p[3] for p in parsed],
        "gene_id": gene_id or [None] * len(raw),
        "gene_name": gene_name or [None] * len(raw),
        "method": raw.get("method"),
        "source": raw.get("source"),
        "biosample_term": [_strip_collection(t) for t in raw.get("biosample_term", pd.Series([None] * len(raw)))],
        "biological_context": raw.get("biological_context"),
        "effect_size": pd.to_numeric(raw.get("effect_size"), errors="coerce"),
        "p_value": pd.to_numeric(raw.get("p_value"), errors="coerce"),
        "neg_log10_pvalue": pd.to_numeric(raw.get("neg_log10_pvalue"), errors="coerce"),
        "label": raw.get("label"),
    })
    return out.drop_duplicates().reset_index(drop=True)


def _match_to_input(links: pd.DataFrame, keys: pd.DataFrame) -> pd.DataFrame:
    """Replace the Catalog SPDI with the input ``variant_id`` for matching variants."""
    k = keys.reset_index().rename(columns={keys.index.name or "index": "input_id"})
    k.columns = ["input_id", "chrom", "pos", "ref", "alt"]
    both = links.merge(k, on=["chrom", "pos"], how="inner", suffixes=("", "_in"))
    allele_ok = ((both["ref"].astype(str).str.upper() == both["ref_in"].astype(str).str.upper())
                 & (both["alt"].astype(str).str.upper() == both["alt_in"].astype(str).str.upper()))
    unknown = both["ref_in"].isna() | both["alt_in"].isna()
    both = both[allele_ok | unknown].copy()
    both["variant_id"] = both["input_id"]
    return both[list(VARIANT_LINK_COLUMNS)].drop_duplicates().reset_index(drop=True)


def query_element_gene_links(
        regions_df: pd.DataFrame,
        *,
        methods: Optional[Sequence[str]] = None,
        sources: Optional[Sequence[str]] = None,
        biosample_term: Optional[str] = None,
        page_size: int = IGVF_PAGE_LIMIT,
        max_pages: int = 100,
        session=None,
        base_url: str = IGVF_CATALOG_BASE,
        timeout: float = 60.0,
        retries: int = 3,
        backoff: float = 1.0,
        delay: float = 0.0,
        cache_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    ``GET /genomic-elements/genes`` for ``chrom, start, end`` regions (1-based
    inclusive; split into ≤10 kb requests), with ``verbose=true`` so that the
    element coordinates and gene identifiers come back in full.

    Returns a DataFrame with :data:`~localqtl.annotate.igvf.ELEMENT_LINK_COLUMNS`.
    """
    need = {"chrom", "start", "end"}
    if not need <= set(regions_df.columns):
        raise ValueError(f"regions_df must have columns {sorted(need)}")
    sess = _make_session(session)
    url = f"{base_url.rstrip('/')}/genomic-elements/genes"
    base_params: dict = {"verbose": "true"}
    if biosample_term:
        base_params["biosample_term"] = biosample_term
    combos = [{}]
    if methods:
        combos = [{**c, "method": m} for c in combos for m in methods]
    if sources:
        combos = [{**c, "source": s} for c in combos for s in sources]

    region_strings = _region_strings(regions_df)
    logger.info("IGVF Catalog: %d element->gene region queries", len(region_strings))
    rows: list = []
    for region in region_strings:
        for combo in combos:
            rows.extend(_paged_rows(sess, url, {**base_params, **combo, "region": region},
                                    page_size=page_size, max_pages=max_pages, timeout=timeout,
                                    retries=retries, backoff=backoff, delay=delay))
    out = _normalise_element_rows(rows)
    if cache_path:
        write_links(out, cache_path)
    return out


def _normalise_element_rows(rows: list) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=list(ELEMENT_LINK_COLUMNS))
    from ..freq.table import normalize_chrom
    recs = []
    for r in rows:
        el = r.get("genomic_element")
        gene = r.get("gene")
        if isinstance(el, dict):
            chrom, start, end = el.get("chr"), el.get("start"), el.get("end")
            element_id = _strip_collection(el.get("_id") or el.get("name"))
        else:
            chrom = start = end = None
            element_id = _strip_collection(el)
        if isinstance(gene, dict):
            gene_id, gene_name = _strip_collection(gene.get("_id")), gene.get("name")
        else:
            gene_id, gene_name = _strip_collection(gene), None
        score = r.get("score")
        if score is None:
            score = r.get("log2FC", r.get("effect_size"))
        recs.append({
            "chrom": normalize_chrom(chrom) if chrom else None,
            "start": start, "end": end, "element_id": element_id,
            "gene_id": gene_id, "gene_name": gene_name,
            "score": score, "method": r.get("method"), "source": r.get("source"),
            "biosample_term": _strip_collection(r.get("biosample_term")),
            "biological_context": r.get("biological_context"),
            "p_value": r.get("p_value"), "neg_log10_pvalue": r.get("neg_log10_pvalue"),
            "significant": r.get("significant"),
        })
    out = pd.DataFrame.from_records(recs, columns=list(ELEMENT_LINK_COLUMNS))
    for c in ("start", "end"):
        out[c] = pd.to_numeric(out[c], errors="coerce").astype("Int64")
    for c in ("score", "p_value", "neg_log10_pvalue"):
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out.drop_duplicates().reset_index(drop=True)


# ---------------------------------------------------------------------------
# file inputs
# ---------------------------------------------------------------------------
def write_links(links_df: pd.DataFrame, path: str) -> str:
    """Write a link table to Parquet (``.parquet``) or TSV; returns ``path``."""
    if str(path).endswith(".parquet"):
        links_df.to_parquet(path, index=False)
    else:
        links_df.to_csv(path, sep="\t", index=False)
    return path


def read_links(path: str) -> pd.DataFrame:
    """Read a link table written by :func:`~localqtl.annotate.igvf.write_links` (or the ``cache_path`` of a query)."""
    if str(path).endswith(".parquet"):
        return pd.read_parquet(path)
    return pd.read_csv(path, sep="\t")


def read_bedpe_links(path: str, *, gene_col: Optional[str] = None, score_col: Optional[str] = None,
                     method: Optional[str] = None, source: Optional[str] = None) -> pd.DataFrame:
    """
    Read an element->gene BEDPE prediction set (IGVF / ENCODE-rE2G, scE2G) into
    the :data:`~localqtl.annotate.igvf.ELEMENT_LINK_COLUMNS` layout.

    The first anchor is taken as the regulatory element and the second as the
    gene (its TSS).  Gene identifiers come from ``gene_col`` when given,
    otherwise from a ``TargetGene``/``gene``/``name`` column (``name`` may be
    ``element|gene`` or ``element_gene``), and the score from ``score_col``
    or the BEDPE ``score`` column.  Files with and without a header are
    accepted; ``.gz`` is handled by pandas.
    """
    from ..freq.table import normalize_chrom
    head = pd.read_csv(path, sep="\t", nrows=1, header=None, dtype=str)
    # a header row has a non-numeric second field (data rows carry start1)
    second = str(head.iloc[0, 1]) if head.shape[1] > 1 else ""
    has_header = not second.strip().lstrip("+-").isdigit()
    if has_header:
        df = pd.read_csv(path, sep="\t")
        df.columns = [str(c).lstrip("#") for c in df.columns]
    else:
        names = ["chrom1", "start1", "end1", "chrom2", "start2", "end2", "name", "score",
                 "strand1", "strand2"]
        df = pd.read_csv(path, sep="\t", header=None)
        df = df.iloc[:, :len(names)]
        df.columns = names[:df.shape[1]]
    cols = {c.lower(): c for c in df.columns}
    c1 = cols.get("chrom1", cols.get("chr1"))
    if c1 is None:
        raise ValueError(f"{path} does not look like BEDPE (no chrom1/chr1 column)")
    s1, e1 = cols.get("start1"), cols.get("end1")

    gene_src = gene_col or cols.get("targetgene") or cols.get("gene") or cols.get("gene_id") or cols.get("name")
    if gene_src is None:
        raise ValueError(f"{path}: no gene column; pass gene_col=")
    genes = df[gene_src].astype(str)
    if gene_src == cols.get("name"):
        genes = genes.str.rsplit(r"|", n=1).str[-1].str.rsplit("_", n=1).str[-1]
    score_src = score_col or cols.get("score")
    out = pd.DataFrame({
        "chrom": [normalize_chrom(c) for c in df[c1].to_numpy()],
        "start": pd.to_numeric(df[s1], errors="coerce").astype("Int64") + 1,  # BED is 0-based
        "end": pd.to_numeric(df[e1], errors="coerce").astype("Int64"),
        "element_id": (df[cols["name"]].astype(str) if "name" in cols
                       else df[c1].astype(str) + ":" + df[s1].astype(str) + "-" + df[e1].astype(str)),
        "gene_id": genes.to_numpy(),
        "gene_name": genes.to_numpy(),
        "score": pd.to_numeric(df[score_src], errors="coerce") if score_src else np.nan,
        "method": method, "source": source,
        "biosample_term": None, "biological_context": None,
        "p_value": np.nan, "neg_log10_pvalue": np.nan, "significant": pd.NA,
    })
    return out[list(ELEMENT_LINK_COLUMNS)]


# ---------------------------------------------------------------------------
# comparison
# ---------------------------------------------------------------------------
def _ancestry_pval_columns(results_df: pd.DataFrame) -> list[str]:
    import re
    pat = re.compile(r"^pval_anc(\d+)$")
    return sorted([c for c in results_df.columns if pat.match(c)],
                  key=lambda c: int(pat.match(c).group(1)))


def compare_with_regulatory_links(
        results_df: pd.DataFrame,
        links_df: pd.DataFrame,
        variant_df: Optional[pd.DataFrame] = None,
        *,
        gene_col: str = "phenotype_id",
        max_distance: int = 0,
        per_ancestry: bool = True,
        alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Flag QTL pairs that have independent regulatory support for the same gene.

    ``links_df`` may be variant-level (:data:`~localqtl.annotate.igvf.VARIANT_LINK_COLUMNS`, matched on
    ``chrom, pos`` and the alleles when both sides have them) or element-level
    (:data:`~localqtl.annotate.igvf.ELEMENT_LINK_COLUMNS`, the variant must fall inside the element
    extended by ``max_distance`` on both sides).

    Adds ``n_links``, ``same_gene_support``, ``best_method``, ``best_source``,
    ``distance_to_element`` (0 inside the element, NaN for variant-level
    links) and, with ``per_ancestry`` and a dosage result table,
    ``supported_anc{k}`` (``pval_anc{k} < alpha`` and same-gene support).
    """
    if "variant_id" not in results_df.columns:
        raise ValueError("results_df must have a 'variant_id' column.")
    if gene_col not in results_df.columns:
        raise ValueError(f"results_df must have the gene column {gene_col!r}.")
    out = results_df.copy()
    keys = variant_keys_for(out["variant_id"], variant_df).reset_index(drop=True)
    left = pd.DataFrame({
        "_row": np.arange(len(out)),
        "chrom": keys["chrom"].to_numpy(),
        "pos": pd.to_numeric(keys["pos"], errors="coerce").to_numpy(),
        "ref": keys["ref"].to_numpy(),
        "alt": keys["alt"].to_numpy(),
        "gene": strip_gene_version(out[gene_col]).to_numpy(),
    })

    n_links = np.zeros(len(out), dtype="int64")
    best_method = np.array([None] * len(out), dtype=object)
    best_source = np.array([None] * len(out), dtype=object)
    distance = np.full(len(out), np.nan)

    if links_df is not None and len(links_df):
        links = links_df.copy()
        links["gene"] = strip_gene_version(links["gene_id"].fillna("")).to_numpy()
        element_level = {"start", "end"} <= set(links.columns)
        if element_level:
            m = left.merge(links, on=["chrom", "gene"], how="inner", suffixes=("", "_link"))
            start = pd.to_numeric(m["start"], errors="coerce") - int(max_distance)
            end = pd.to_numeric(m["end"], errors="coerce") + int(max_distance)
            inside = (m["pos"] >= start) & (m["pos"] <= end)
            m = m[inside].copy()
            el_start = pd.to_numeric(m["start"], errors="coerce")
            el_end = pd.to_numeric(m["end"], errors="coerce")
            m["_distance"] = np.maximum.reduce([
                np.zeros(len(m)), (el_start - m["pos"]).to_numpy(dtype=float),
                (m["pos"] - el_end).to_numpy(dtype=float)])
            rank_col = "score"
        else:
            m = left.merge(links, on=["chrom", "pos", "gene"], how="inner", suffixes=("", "_link"))
            if len(m) and {"ref", "alt"} <= set(links.columns):
                same = ((m["ref"].astype(str).str.upper() == m["ref_link"].astype(str).str.upper())
                        & (m["alt"].astype(str).str.upper() == m["alt_link"].astype(str).str.upper()))
                unknown = m["ref"].isna() | m["ref_link"].isna()
                m = m[same | unknown].copy()
            m["_distance"] = 0.0
            rank_col = "neg_log10_pvalue"
        if len(m):
            counts = m.groupby("_row").size()
            n_links[counts.index.to_numpy()] = counts.to_numpy()
            rank = pd.to_numeric(m[rank_col], errors="coerce") if rank_col in m.columns else pd.Series(np.nan, index=m.index)
            m = m.assign(_rank=rank.fillna(-np.inf)).sort_values("_rank", ascending=False)
            top = m.drop_duplicates("_row")
            rows = top["_row"].to_numpy()
            best_method[rows] = top["method"].to_numpy()
            best_source[rows] = top["source"].to_numpy()
            dmin = m.groupby("_row")["_distance"].min()
            distance[dmin.index.to_numpy()] = dmin.to_numpy()

    out["n_links"] = n_links
    out["same_gene_support"] = pd.array(n_links > 0, dtype="boolean")
    out["best_method"] = best_method
    out["best_source"] = best_source
    out["distance_to_element"] = distance
    if per_ancestry:
        for col in _ancestry_pval_columns(out):
            k = col[len("pval_anc"):]
            pv = pd.to_numeric(out[col], errors="coerce").to_numpy()
            sup = pd.array((pv < alpha) & (n_links > 0), dtype="boolean")
            sup[np.isnan(pv)] = pd.NA
            out[f"supported_anc{k}"] = sup
    return out


def summarize_link_concordance(df: pd.DataFrame, *, by: Sequence[str] = ("ancestry",),
                               alpha: float = 0.05, pval_col: str = "pval_nominal") -> pd.DataFrame:
    """
    Tidy concordance summary of :func:`compare_with_regulatory_links` output.

    With the default ``by=("ancestry",)`` one row per ancestry is returned from
    the ``supported_anc{k}`` / ``pval_anc{k}`` columns (plus a pooled ``all``
    row from ``pval_col``); any other ``by`` groups the table by those columns
    (e.g. ``("best_method",)``).  Columns: ``n``, ``n_significant``,
    ``n_supported``, ``frac_supported`` (of the significant pairs),
    ``frac_any_link`` and ``mean_links``.
    """
    rows = []

    def _row(label_fields: dict, sub: pd.DataFrame, pv_col: str, sup_col: Optional[str]):
        pv = pd.to_numeric(sub[pv_col], errors="coerce") if pv_col in sub.columns else pd.Series(dtype=float)
        sig = (pv < alpha).fillna(False).to_numpy() if len(pv) else np.zeros(len(sub), dtype=bool)
        links = pd.to_numeric(sub.get("n_links", 0), errors="coerce").fillna(0).to_numpy()
        if sup_col is not None and sup_col in sub.columns:
            supported = sub[sup_col].fillna(False).to_numpy().astype(bool)
        else:
            supported = sig & (links > 0)
        n_sig = int(sig.sum())
        rows.append({**label_fields, "n": int(len(sub)), "n_significant": n_sig,
                     "n_supported": int(supported.sum()),
                     "frac_supported": float(supported.sum() / n_sig) if n_sig else np.nan,
                     "frac_any_link": float((links > 0).mean()) if len(sub) else np.nan,
                     "mean_links": float(links.mean()) if len(sub) else np.nan})

    if tuple(by) == ("ancestry",):
        for col in _ancestry_pval_columns(df):
            k = col[len("pval_anc"):]
            _row({"ancestry": f"anc{k}"}, df, col, f"supported_anc{k}")
        _row({"ancestry": "all"}, df, pval_col, None)
    else:
        for keys, sub in df.groupby(list(by), dropna=False, sort=False):
            keys = keys if isinstance(keys, tuple) else (keys,)
            _row(dict(zip(by, keys)), sub, pval_col, None)
    return pd.DataFrame(rows)
