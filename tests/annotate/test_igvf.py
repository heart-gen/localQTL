import numpy as np
import pandas as pd
import pytest

from localqtl.annotate import igvf


class _Resp:
    def __init__(self, payload, status=200):
        self._payload, self.status_code = payload, status
        self.text = str(payload)

    def json(self):
        return self._payload


class FakeSession:
    """Returns ``rows`` on page 0 of every request and records the calls."""

    def __init__(self, rows, status=200, pages=1):
        self.rows, self.status, self.pages, self.calls = rows, status, pages, []

    def get(self, url, params=None, timeout=None):
        self.calls.append((url, dict(params or {})))
        if self.status != 200:
            return _Resp({"message": "boom"}, self.status)
        return _Resp(self.rows if params.get("page", 0) < self.pages else [])


VARIANT_ROWS = [
    {"gene": "genes/ENSG00000104044", "sequence_variant": "variants/NC_000015.10:28125242:C:T",
     "effect_size": 1.41, "neg_log10_pvalue": 11.9, "p_value": 1.3e-12, "method": "eQTL",
     "source": "EBI", "biological_context": "Th2 cell",
     "biosample_term": "ontology_terms/CL_0000546", "label": "eQTL"},
    {"gene": "genes/ENSG00000128731", "sequence_variant": "variants/NC_000015.10:28125242:C:T",
     "effect_size": 0.9, "neg_log10_pvalue": 13.0, "p_value": 1e-13, "method": "eQTL",
     "source": "EBI", "biological_context": "Th17",
     "biosample_term": "ontology_terms/CL_0000899", "label": "eQTL"},
]
ELEMENT_ROWS = [
    {"genomic_element": {"_id": "CRISPR_chr16_53761342_53761642", "chr": "chr16",
                         "start": 53761342, "end": 53761642},
     "gene": {"name": "RPGRIP1L", "_id": "ENSG00000103494"}, "method": "Perturb-seq",
     "source": "IGVF", "biosample_term": "ontology_terms/EFO_0022614",
     "biological_context": "HCASMC", "log2FC": 0.12, "p_value": 0.392,
     "neg_log10_pvalue": 0.407, "significant": False},
]


def test_constants_and_module_imports_without_requests():
    assert igvf.IGVF_CATALOG_BASE.startswith("https://api.catalogkg.igvf.org")
    assert "ENCODE-rE2G" in igvf.IGVF_ELEMENT_GENE_METHODS
    assert "eQTL" in igvf.IGVF_VARIANT_GENE_METHODS
    assert igvf.IGVF_MAX_REGION_BP == 10_000 and igvf.IGVF_PAGE_LIMIT == 500
    assert isinstance(igvf.requests_available(), bool)


def test_variant_gene_query_normalises_rows_and_converts_positions():
    """The Catalog reports 0-based positions; the client returns 1-based ones."""
    session = FakeSession(VARIANT_ROWS)
    lead = pd.DataFrame({"variant_id": ["chr15_28125243_C_T_b38"]})
    out = igvf.query_variant_gene_links(lead, session=session, methods=["eQTL"])
    assert list(out.columns) == list(igvf.VARIANT_LINK_COLUMNS)
    assert out["pos"].tolist() == [28125243, 28125243]        # 28125242 + 1
    assert out["variant_id"].unique().tolist() == ["chr15_28125243_C_T_b38"]
    assert out["gene_id"].tolist() == ["ENSG00000104044", "ENSG00000128731"]
    assert out["biosample_term"].tolist() == ["CL_0000546", "CL_0000899"]
    assert out["chrom"].unique().tolist() == ["chr15"] and out["ref"].unique().tolist() == ["C"]
    url, params = session.calls[0]
    assert url.endswith("/variants/genes") and params["method"] == "eQTL"
    chrom, span = params["region"].split(":")
    start, end = (int(x) for x in span.split("-"))
    assert chrom == "chr15" and end - start <= igvf.IGVF_MAX_REGION_BP


def test_variant_gene_query_filters_matching_and_paging():
    # a variant at another position gets no rows back when exact matching is on
    session = FakeSession(VARIANT_ROWS)
    other = pd.DataFrame({"variant_id": ["chr15_29999999_C_T_b38"]})
    assert igvf.query_variant_gene_links(other, session=session).empty
    # with exact_variant_match=False every link in the region is kept
    loose = igvf.query_variant_gene_links(other, session=FakeSession(VARIANT_ROWS),
                                          exact_variant_match=False)
    assert len(loose) == 2 and loose["variant_id"].iloc[0].startswith("NC_000015.10")
    # a mismatching alternate allele is dropped
    assert igvf.query_variant_gene_links(
        pd.DataFrame({"variant_id": ["chr15_28125243_C_A_b38"]}), session=FakeSession(VARIANT_ROWS)).empty
    # paging stops when a short page comes back; both filter values are requested
    paged = FakeSession(VARIANT_ROWS, pages=2)
    igvf.query_variant_gene_links(pd.DataFrame({"variant_id": ["chr15_28125243_C_T_b38"]}),
                                  session=paged, page_size=2, sources=["EBI", "AFGR"])
    assert [c[1]["page"] for c in paged.calls] == [0, 1, 2, 0, 1, 2]
    assert {c[1]["source"] for c in paged.calls} == {"EBI", "AFGR"}
    assert all(c[1]["limit"] == 2 for c in paged.calls)


def test_query_errors_and_region_splitting():
    with pytest.raises(RuntimeError, match="IGVF Catalog request failed"):
        igvf.query_variant_gene_links(pd.DataFrame({"variant_id": ["chr1_100_A_G_b38"]}),
                                      session=FakeSession([], status=400), retries=1)
    session = FakeSession(ELEMENT_ROWS)
    regions = pd.DataFrame({"chrom": ["chr16"], "start": [1], "end": [25_000]})
    igvf.query_element_gene_links(regions, session=session)
    spans = [tuple(int(x) for x in c[1]["region"].split(":")[1].split("-")) for c in session.calls]
    assert len(spans) == 3 and all(e - s <= igvf.IGVF_MAX_REGION_BP for s, e in spans)
    assert all(c[1]["verbose"] == "true" for c in session.calls)
    with pytest.raises(ValueError, match="regions_df"):
        igvf.query_element_gene_links(pd.DataFrame({"chrom": ["chr1"]}), session=session)


def test_element_gene_query_normalises_nested_objects():
    session = FakeSession(ELEMENT_ROWS)
    out = igvf.query_element_gene_links(
        pd.DataFrame({"chrom": ["chr16"], "start": [53761000], "end": [53762000]}), session=session)
    assert list(out.columns) == list(igvf.ELEMENT_LINK_COLUMNS)
    row = out.iloc[0]
    assert row["element_id"] == "CRISPR_chr16_53761342_53761642"
    assert row["gene_id"] == "ENSG00000103494" and row["gene_name"] == "RPGRIP1L"
    assert (row["chrom"], int(row["start"]), int(row["end"])) == ("chr16", 53761342, 53761642)
    assert row["score"] == pytest.approx(0.12)        # log2FC when no explicit score
    assert row["biosample_term"] == "EFO_0022614"


def test_link_table_round_trip_and_bedpe(tmp_path):
    links = igvf._normalise_element_rows(ELEMENT_ROWS)
    p = igvf.write_links(links, str(tmp_path / "links.parquet"))
    assert len(igvf.read_links(p)) == 1
    t = igvf.write_links(links, str(tmp_path / "links.tsv"))
    assert len(igvf.read_links(t)) == 1

    bedpe = tmp_path / "pred.bedpe"
    bedpe.write_text(
        "chr1\t900\t1100\tchr1\t5000\t5001\tE1|ENSG001\t0.8\t.\t.\n"
        "chr1\t4000\t4500\tchr1\t5000\t5001\tE2|ENSG001\t0.2\t.\t.\n")
    out = igvf.read_bedpe_links(str(bedpe), method="ENCODE-rE2G", source="ENCODE")
    assert list(out.columns) == list(igvf.ELEMENT_LINK_COLUMNS)
    assert out["gene_id"].tolist() == ["ENSG001", "ENSG001"]
    assert out["start"].tolist() == [901, 4001] and out["end"].tolist() == [1100, 4500]
    assert out["score"].tolist() == [0.8, 0.2] and out["method"].unique().tolist() == ["ENCODE-rE2G"]

    headed = tmp_path / "pred_header.bedpe"
    headed.write_text("#chrom1\tstart1\tend1\tchrom2\tstart2\tend2\tname\tscore\tTargetGene\n"
                      "chr1\t900\t1100\tchr1\t5000\t5001\tE1\t0.8\tENSG001\n")
    out2 = igvf.read_bedpe_links(str(headed))
    assert out2["gene_id"].tolist() == ["ENSG001"] and out2["element_id"].tolist() == ["E1"]
    with pytest.raises(ValueError, match="BEDPE"):
        igvf.read_bedpe_links(str(tmp_path / "links.tsv"))


@pytest.fixture
def results_df():
    return pd.DataFrame({
        "phenotype_id": ["ENSG001.2", "ENSG001.2", "ENSG002", "ENSG003"],
        "variant_id": ["chr1_1000_A_G_b38", "chr1_5000_C_T_b38", "chr1_1000_A_G_b38",
                       "chr2_200_G_A_b38"],
        "pval_nominal": [1e-6, 0.4, 1e-3, 0.02],
        "pval_anc0": [1e-5, 0.3, 2e-3, 0.5],
        "pval_anc1": [0.2, 0.9, 1e-4, 0.01],
        "slope": [0.5, -0.1, 0.3, 0.2],
    })


@pytest.fixture
def element_links():
    return pd.DataFrame({
        "chrom": ["chr1", "chr1"], "start": [900, 4000], "end": [1100, 4500],
        "element_id": ["E1", "E2"], "gene_id": ["ENSG001", "ENSG001"], "gene_name": ["A", "A"],
        "score": [0.8, 0.2], "method": ["ENCODE-rE2G", "scE2G"], "source": ["IGVF", "IGVF"],
        "biosample_term": [None, None], "biological_context": [None, None],
        "p_value": [np.nan, np.nan], "neg_log10_pvalue": [np.nan, np.nan],
        "significant": [True, False],
    })


def test_compare_with_element_links(results_df, element_links):
    out = igvf.compare_with_regulatory_links(results_df, element_links)
    # row 0: variant inside E1 and the same gene (version suffix ignored)
    assert out["n_links"].tolist() == [1, 0, 0, 0]
    assert out["same_gene_support"].tolist() == [True, False, False, False]
    assert out.loc[0, "best_method"] == "ENCODE-rE2G" and out.loc[0, "distance_to_element"] == 0
    # ancestry support needs both a link and a significant per-ancestry p-value
    assert out["supported_anc0"].tolist() == [True, False, False, False]
    assert out["supported_anc1"].tolist() == [False, False, False, False]
    # widening the window brings the second variant within E2
    near = igvf.compare_with_regulatory_links(results_df, element_links, max_distance=600)
    assert near["n_links"].tolist() == [1, 1, 0, 0]
    assert near.loc[1, "distance_to_element"] == pytest.approx(500)
    empty = igvf.compare_with_regulatory_links(results_df, element_links.iloc[:0])
    assert empty["n_links"].tolist() == [0, 0, 0, 0] and not empty["same_gene_support"].any()


def test_compare_with_variant_links_and_validation(results_df):
    variant_links = pd.DataFrame({
        "variant_id": ["x"], "chrom": ["chr1"], "pos": [1000], "ref": ["A"], "alt": ["G"],
        "gene_id": ["ENSG002"], "gene_name": ["B"], "method": ["eQTL"], "source": ["EBI"],
        "biosample_term": [None], "biological_context": [None], "effect_size": [1.0],
        "p_value": [1e-9], "neg_log10_pvalue": [9.0], "label": ["eQTL"],
    })
    out = igvf.compare_with_regulatory_links(results_df, variant_links)
    assert out["n_links"].tolist() == [0, 0, 1, 0]
    assert out.loc[2, "best_method"] == "eQTL" and out.loc[2, "distance_to_element"] == 0
    # the allele must agree
    mismatched = variant_links.assign(alt=["T"])
    assert igvf.compare_with_regulatory_links(results_df, mismatched)["n_links"].sum() == 0
    no_per_anc = igvf.compare_with_regulatory_links(results_df, variant_links, per_ancestry=False)
    assert "supported_anc0" not in no_per_anc.columns
    with pytest.raises(ValueError, match="variant_id"):
        igvf.compare_with_regulatory_links(results_df.drop(columns="variant_id"), variant_links)
    with pytest.raises(ValueError, match="gene column"):
        igvf.compare_with_regulatory_links(results_df, variant_links, gene_col="missing")


def test_summarize_link_concordance(results_df, element_links):
    compared = igvf.compare_with_regulatory_links(results_df, element_links)
    summary = igvf.summarize_link_concordance(compared)
    assert summary["ancestry"].tolist() == ["anc0", "anc1", "all"]
    anc0 = summary.set_index("ancestry").loc["anc0"]
    assert anc0["n"] == 4 and anc0["n_significant"] == 2 and anc0["n_supported"] == 1
    assert anc0["frac_supported"] == pytest.approx(0.5)
    assert anc0["frac_any_link"] == pytest.approx(0.25)
    by_method = igvf.summarize_link_concordance(compared, by=("best_method",))
    assert set(by_method.columns) >= {"best_method", "n", "frac_supported"}
    assert by_method.set_index("best_method").loc["ENCODE-rE2G", "n_supported"] == 1


@pytest.mark.live
def test_live_variant_gene_query():
    lead = pd.DataFrame({"variant_id": ["chr15_28125243_C_T_b38"]})
    out = igvf.query_variant_gene_links(lead, methods=["eQTL"], page_size=100)
    assert not out.empty
    assert list(out.columns) == list(igvf.VARIANT_LINK_COLUMNS)
    assert out["pos"].unique().tolist() == [28125243]
    assert out["gene_id"].str.startswith("ENSG").all()


@pytest.mark.live
def test_live_element_gene_query():
    regions = pd.DataFrame({"chrom": ["chr16"], "start": [53761000], "end": [53762000]})
    out = igvf.query_element_gene_links(regions, page_size=50)
    assert not out.empty and out["gene_id"].str.startswith("ENSG").all()
    assert (out["end"] >= out["start"]).all()
