import numpy as np
import pandas as pd
import pytest
from scipy import stats as st

from localqtl import post
from tests.conftest import make_cohort_with_chroms, ols_fit


@pytest.fixture
def cohort():
    return make_cohort_with_chroms(n=120)


def test_calculate_afc_recovers_planted_fold_change(cohort):
    d = cohort
    rng = d["rng"]
    g = d["genotype_df"].loc["v1_0"].to_numpy(np.float64)
    # aFC = 1 (log2): expression doubles per alt haplotype -> log(count) = log(2) * g/2 + noise
    counts = np.exp(np.log(200) + np.log(2) * g / 2 + 0.05 * rng.normal(size=len(g)))
    counts_df = pd.DataFrame([counts], index=["gene1_0"], columns=d["samples"])
    assoc = pd.DataFrame({"gene_id": ["gene1_0"], "variant_id": ["v1_0"]})
    out = post.calculate_afc(assoc, counts_df, d["genotype_df"], covariates_df=d["covariates_df"],
                             device="cpu", verbose=False)
    assert list(out.columns) == ["gene_id", "variant_id", "afc", "afc_se"]
    assert abs(out["afc"].iloc[0] - 1.0) < 0.1 and out["afc_se"].iloc[0] > 0
    # several variants per gene are fitted jointly; half_min imputation path runs
    assoc2 = pd.DataFrame({"gene_id": ["gene1_0", "gene1_0"], "variant_id": ["v1_0", "v1_5"]})
    counts_df.iloc[0, 0] = 0.0
    out2 = post.calculate_afc(assoc2, counts_df, d["genotype_df"], imputation="half_min", device="cpu", verbose=False)
    assert len(out2) == 2 and np.isfinite(out2["afc"]).all()


def test_trc_standard_mode_matches_ols(cohort):
    import torch
    d = cohort
    g = d["genotype_df"].loc[["v1_0", "v1_1"]].to_numpy(np.float32)
    counts = np.exp(1 + 0.3 * g[0] + 0.1 * d["rng"].normal(size=g.shape[1])).astype(np.float32)
    t, b, se = post.trc(torch.tensor(g), torch.tensor(counts), mode="standard")
    bb, ss, _ = ols_fit(np.log1p(counts), (g[0] / 2)[:, None])
    np.testing.assert_allclose(b[0].item(), bb[0], rtol=1e-4)
    np.testing.assert_allclose(se[0].item(), ss[0], rtol=1e-4)


def test_calculate_replication_matches_ols(cohort):
    d = cohort
    res = pd.DataFrame({"variant_id": ["v1_0", "v1_3", "v2_0"]}, index=["gene1_0", "gene1_1", "gene2_0"])
    pi1, rep = post.calculate_replication(res, d["genotype_df"], d["phenotype_df"], d["covariates_df"],
                                          compute_pi1=True, device="cpu")
    assert list(rep.columns) == ['phenotype_id', 'variant_id', 'ma_samples', 'ma_count', 'af', 'pval_nominal',
                                 'slope', 'slope_se']
    assert np.isfinite(pi1) and 0 <= pi1 <= 1
    y = d["phenotype_df"].loc["gene1_0"].to_numpy(np.float64)
    g = d["genotype_df"].loc["v1_0"].to_numpy(np.float64)
    b, se, dof = ols_fit(y, np.column_stack([g, d["covariates_df"].values]))
    np.testing.assert_allclose(rep.slope.iloc[0], b[0], rtol=1e-4)
    np.testing.assert_allclose(rep.slope_se.iloc[0], se[0], rtol=1e-4)
    np.testing.assert_allclose(rep.pval_nominal.iloc[0], 2 * st.t.sf(abs(b[0] / se[0]), dof), rtol=1e-4)
    # phenotype-specific covariate
    paired = pd.DataFrame(d["rng"].normal(size=(1, len(d["samples"]))), index=["gene1_0"], columns=d["samples"])
    rep2 = post.calculate_replication(res, d["genotype_df"], d["phenotype_df"], d["covariates_df"],
                                      paired_covariate_df=paired, device="cpu")
    b, se, dof = ols_fit(y, np.column_stack([g, d["covariates_df"].values, paired.values[0]]))
    np.testing.assert_allclose(rep2.slope.iloc[0], b[0], rtol=1e-4)
    np.testing.assert_allclose(rep2.pval_nominal.iloc[0], 2 * st.t.sf(abs(b[0] / se[0]), dof), rtol=1e-4)
    np.testing.assert_allclose(rep2.slope.iloc[1], rep.slope.iloc[1], rtol=1e-6)  # unpaired phenotype unchanged
    # interaction
    i_s = pd.Series(d["rng"].normal(size=len(d["samples"])), index=d["samples"])
    rep3 = post.calculate_replication(res, d["genotype_df"], d["phenotype_df"], d["covariates_df"],
                                      interaction_s=i_s, device="cpu")
    b, se, dof = ols_fit(y, np.column_stack([g, i_s.values, g * i_s.values, d["covariates_df"].values]))
    np.testing.assert_allclose([rep3.b_g.iloc[0], rep3.b_i.iloc[0], rep3.b_gi.iloc[0]], b[:3], rtol=1e-3)
    np.testing.assert_allclose(rep3.pval_gi.iloc[0], 2 * st.t.sf(abs(b[2] / se[2]), dof), rtol=1e-3)
    # no covariates
    rep4 = post.calculate_replication(res, d["genotype_df"], d["phenotype_df"], None, device="cpu")
    b, se, dof = ols_fit(y, g[:, None])
    np.testing.assert_allclose(rep4.slope.iloc[0], b[0], rtol=1e-4)


def test_annotate_genes(tmp_path):
    gtf = tmp_path / "genes.gtf"
    gtf.write_text(
        "#comment\n"
        'chr1\tHAVANA\tgene\t1000\t5000\t.\t+\t.\tgene_id "gene1_0"; gene_name "ALPHA"; gene_type "protein_coding";\n'
        'chr1\tHAVANA\ttranscript\t1000\t5000\t.\t+\t.\tgene_id "gene1_0"; transcript_id "t1";\n'
        'chr2\tHAVANA\tgene\t200\t900\t.\t-\t.\tgene_id "gene2_0"; gene_name "BETA";\n'
    )
    gene_df = pd.DataFrame({
        "num_var": [5, 3], "beta_shape1": [1.0, 1.1], "beta_shape2": [20.0, 25.0], "true_dof": [50.0, 51.0],
        "pval_true_dof": [1e-3, 0.2], "variant_id": ["v1_0", "v2_0"], "start_distance": [10, -20],
        "end_distance": [10, -20], "ma_samples": [20, 30], "ma_count": [25, 35], "af": [0.2, 0.3],
        "pval_nominal": [1e-3, 0.2], "slope": [0.5, -0.1], "slope_se": [0.1, 0.1], "pval_perm": [1e-3, 0.3],
        "pval_beta": [1e-3, 0.3], "qval": [0.01, 0.5], "pval_nominal_threshold": [1e-3, np.nan],
    }, index=["gene1_0", "gene2_0"])
    lookup = pd.DataFrame({"rs_id": ["rs1", "rs2"]}, index=["v1_0", "v2_0"])
    out = post.annotate_genes(gene_df, str(gtf), lookup_df=lookup, verbose=False)
    assert list(out.columns[:5]) == ["gene_name", "gene_chr", "gene_start", "gene_end", "strand"]
    assert out.loc["gene1_0", "gene_name"] == "ALPHA" and out.loc["gene2_0", "strand"] == "-"
    assert out.loc["gene1_0", "rs_id"] == "rs1" and out.columns[-1] == "pval_nominal_threshold"
    with pytest.raises(KeyError):
        post.annotate_genes(gene_df.rename(index={"gene1_0": "missing"}), str(gtf), verbose=False)
