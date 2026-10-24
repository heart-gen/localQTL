import types

import numpy as np
import pandas as pd
import pytest

from localqtl.annotate import alphagenome as ag


@pytest.fixture
def scores_df():
    return pd.DataFrame({
        "variant_id": ["chr1:100:A>G", "chr1:100:A>G", "chr1:200:C>T", "chr1:100:A>G"],
        "gene_id": ["ENSG001.3", "ENSG001.3", "ENSG002", "ENSG009"],
        "gene_name": ["A", "A", "B", "Z"],
        "output_type": ["RNA_SEQ", "RNA_SEQ", "RNA_SEQ", "ATAC"],
        "variant_scorer": ["GeneMaskLFC"] * 4,
        "ontology_curie": ["UBERON:1", "UBERON:2", "UBERON:1", "UBERON:1"],
        "biosample_name": ["caudate", "cortex", "caudate", "caudate"],
        "raw_score": [0.8, -0.2, -1.5, 2.0],
        "quantile_score": [0.9, 0.4, 0.95, 0.99],
    })


@pytest.fixture
def results_df():
    return pd.DataFrame({
        "phenotype_id": ["ENSG001", "ENSG002", "ENSG003"],
        "variant_id": ["chr1_100_A_G_b38", "chr1_200_C_T_b38", "chr1_300_G_A_b38"],
        "slope": [0.5, 0.4, -0.1],
    })


def test_module_imports_without_the_sdk():
    assert ag.ALPHAGENOME_API_KEY_ENV == "ALPHAGENOME_API_KEY"
    assert "RNA_SEQ" in ag.ALPHAGENOME_DEFAULT_SCORERS
    assert {"variant_id", "gene_id", "raw_score", "quantile_score"} <= set(ag.ATLAS_SCORE_COLUMNS)
    assert isinstance(ag.alphagenome_available(), bool)


def test_atlas_client_requires_a_key(monkeypatch):
    monkeypatch.setattr(ag, "_try_get_alphagenome", lambda *a, **k: (object(), object(), object()))
    monkeypatch.delenv("ALPHAGENOME_API_KEY", raising=False)
    with pytest.raises(ValueError, match="API key"):
        ag.atlas_client()


def test_annotate_matches_on_variant_and_gene(scores_df, results_df):
    out = ag.annotate_with_alphagenome(results_df, scores_df)
    # the gene version suffix in the score table is ignored; max |raw_score| track is kept
    assert out.loc[0, "ag_raw_score"] == pytest.approx(0.8)
    assert out.loc[0, "ag_biosample"] == "caudate"
    assert out.loc[0, "ag_n_tracks"] == 2            # two tracks for (chr1:100 A>G, ENSG001)
    assert out.loc[1, "ag_raw_score"] == pytest.approx(-1.5)
    assert np.isnan(out.loc[2, "ag_raw_score"]) and out.loc[2, "ag_n_tracks"] == 0
    # sign concordance: slope>0 with score>0 is True, slope>0 with score<0 is False, no score is NA
    assert out["ag_sign_concordant"].tolist()[:2] == [True, False]
    assert pd.isna(out["ag_sign_concordant"].iloc[2])
    assert out["ag_sign_concordant"].dtype == "boolean"


def test_annotate_aggregation_and_filters(scores_df, results_df):
    mean = ag.annotate_with_alphagenome(results_df, scores_df, agg="mean")
    assert mean.loc[0, "ag_raw_score"] == pytest.approx((0.8 - 0.2) / 2)
    assert mean.loc[0, "ag_biosample"] == "mean"
    only_atac = ag.annotate_with_alphagenome(results_df, scores_df, output_type="ATAC")
    assert only_atac["ag_n_tracks"].sum() == 0       # the ATAC row is for another gene
    one_tissue = ag.annotate_with_alphagenome(results_df, scores_df, ontology_terms=["UBERON:2"])
    assert one_tissue.loc[0, "ag_raw_score"] == pytest.approx(-0.2)
    empty = ag.annotate_with_alphagenome(results_df, scores_df.iloc[:0])
    assert empty["ag_n_tracks"].tolist() == [0, 0, 0]
    with pytest.raises(ValueError, match="agg"):
        ag.annotate_with_alphagenome(results_df, scores_df, agg="median")
    with pytest.raises(ValueError, match="variant_id"):
        ag.annotate_with_alphagenome(results_df.drop(columns="variant_id"), scores_df)


def test_annotate_resolves_variant_ids_through_variant_df(scores_df):
    variant_df = pd.DataFrame({"chrom": ["chr1", "chr1"], "pos": [100, 200],
                               "ref": ["A", "C"], "alt": ["G", "T"]}, index=["v1", "v2"])
    res = pd.DataFrame({"phenotype_id": ["ENSG001", "ENSG002"], "variant_id": ["v1", "v2"],
                        "slope": [0.5, 0.4]})
    out = ag.annotate_with_alphagenome(res, scores_df, variant_df=variant_df)
    assert out["ag_raw_score"].tolist() == pytest.approx([0.8, -1.5])


def test_score_table_round_trip(tmp_path, scores_df):
    p = ag.write_atlas_scores(scores_df, str(tmp_path / "scores.parquet"))
    back = ag.read_atlas_scores(p)
    assert len(back) == len(scores_df) and "raw_score" in back.columns
    t = ag.write_atlas_scores(scores_df, str(tmp_path / "scores.tsv"))
    assert len(ag.read_atlas_scores(t)) == len(scores_df)
    bad = tmp_path / "bad.tsv"
    pd.DataFrame({"a": [1]}).to_csv(bad, sep="\t", index=False)
    with pytest.raises(ValueError, match="tidy AlphaGenome score table"):
        ag.read_atlas_scores(str(bad))


def test_score_lead_variants_batches_and_tidies(monkeypatch, tmp_path):
    """The Atlas call is mocked: this checks variant construction, batching and the tidy join."""
    calls = []

    class FakeVariant:
        def __init__(self, chromosome, position, reference_bases, alternate_bases):
            self.chromosome, self.position = chromosome, position
            self.reference_bases, self.alternate_bases = reference_bases, alternate_bases

    class FakeClient:
        def query_variants(self, variants, *, requested_scorers, ontology_terms=None,
                           gene_ids=None, gene_names=None, progress_bar=True):
            calls.append((list(variants), tuple(requested_scorers), ontology_terms))
            return {"RNA_SEQ": f"anndata-for-{len(variants)}"}

    def fake_tidy(scores):
        inner = list(scores)[0]
        return pd.DataFrame({
            "variant_id": [f"chr1:{100 + i}:A>G" for i in range(len(inner))],
            "gene_id": ["ENSG001"], "gene_name": ["A"], "output_type": ["RNA_SEQ"],
            "variant_scorer": ["GeneMaskLFC"], "ontology_curie": ["UBERON:1"],
            "biosample_name": ["caudate"], "raw_score": [0.5], "quantile_score": [0.9],
        })

    genome = types.SimpleNamespace(Variant=FakeVariant)
    scorers_mod = types.SimpleNamespace(tidy_scores=fake_tidy)
    monkeypatch.setattr(ag, "_try_get_alphagenome", lambda *a, **k: (None, genome, scorers_mod))

    lead = pd.DataFrame({"variant_id": [f"chr1_{100 + i}_A_G_b38" for i in range(5)]})
    cache = str(tmp_path / "scores.parquet")
    out = ag.score_lead_variants(lead, client=FakeClient(), batch_size=2,
                                 ontology_terms=["UBERON:0001873"], cache_path=cache)
    assert [len(c[0]) for c in calls] == [2, 2, 1]            # batched
    assert calls[0][1] == ("RNA_SEQ",) and calls[0][2] == ["UBERON:0001873"]
    first = calls[0][0][0]
    assert (first.chromosome, first.position, first.reference_bases, first.alternate_bases) == ("chr1", 100, "A", "G")
    assert len(out) == 3 and set(ag.ATLAS_SCORE_COLUMNS) <= set(out.columns)
    assert len(ag.read_atlas_scores(cache)) == 3
    # explicit chrom/pos/ref/alt columns are accepted too
    lead2 = pd.DataFrame({"chrom": ["1"], "pos": [100], "ref": ["a"], "alt": ["g"]})
    ag.score_lead_variants(lead2, client=FakeClient())
    assert (calls[-1][0][0].chromosome, calls[-1][0][0].ref if hasattr(calls[-1][0][0], "ref")
            else calls[-1][0][0].reference_bases) == ("chr1", "A")
    with pytest.raises(ValueError, match="variant_id"):
        ag.score_lead_variants(pd.DataFrame({"x": [1]}), client=FakeClient())


@pytest.mark.live
def test_live_atlas_query_requires_key_and_sdk():
    pytest.importorskip("alphagenome")
    lead = pd.DataFrame({"variant_id": ["chr22_36201698_A_C_b38"]})
    out = ag.score_lead_variants(lead, scorers=["RNA_SEQ"])
    assert set(ag.ATLAS_SCORE_COLUMNS) <= set(out.columns)
