import numpy as np
import pandas as pd
import pytest
from scipy import stats as st

from localqtl import trans
from tests.conftest import make_cohort_with_chroms, make_random_panel, ols_fit


@pytest.fixture
def cohort():
    return make_cohort_with_chroms()


def test_map_trans_sparse_matches_ols(cohort):
    d = cohort
    out = trans.map_trans(d["genotype_df"], d["phenotype_df"], d["covariates_df"], return_sparse=True,
                          pval_threshold=1.0, maf_threshold=0, return_r2=True, device="cpu", verbose=False)
    assert list(out.columns) == ["variant_id", "phenotype_id", "pval", "b", "b_se", "r2", "af"]
    assert len(out) == d["genotype_df"].shape[0] * d["phenotype_df"].shape[0]
    y = d["phenotype_df"].loc["gene1_0"].to_numpy(np.float64)
    g = d["genotype_df"].loc["v1_0"].to_numpy(np.float64)
    b, se, dof = ols_fit(y, np.column_stack([g, d["covariates_df"].values]))
    row = out[(out.variant_id == "v1_0") & (out.phenotype_id == "gene1_0")].iloc[0]
    np.testing.assert_allclose(row.b, b[0], rtol=1e-4)
    np.testing.assert_allclose(row.b_se, se[0], rtol=1e-4)
    np.testing.assert_allclose(row.pval, 2 * st.t.sf(abs(b[0] / se[0]), dof), rtol=1e-4)
    # thresholding keeps the significant planted pair only when requested
    sig = trans.map_trans(d["genotype_df"], d["phenotype_df"], d["covariates_df"], pval_threshold=1e-3,
                          maf_threshold=0, device="cpu", verbose=False)
    assert ((sig.variant_id == "v1_0") & (sig.phenotype_id == "gene1_0")).any()
    assert (sig.pval < 1e-3).all()


def test_map_trans_dense_and_no_covariates(cohort):
    d = cohort
    pval_df, b_df, b_se_df, af_s = trans.map_trans(d["genotype_df"], d["phenotype_df"], None, return_sparse=False,
                                                   maf_threshold=0, device="cpu", verbose=False)
    assert pval_df.shape == (d["genotype_df"].shape[0], d["phenotype_df"].shape[0])
    y = d["phenotype_df"].loc["gene2_0"].to_numpy(np.float64)
    g = d["genotype_df"].loc["v2_0"].to_numpy(np.float64)
    b, se, dof = ols_fit(y, g[:, None])
    np.testing.assert_allclose(b_df.loc["v2_0", "gene2_0"], b[0], rtol=1e-4)
    np.testing.assert_allclose(pval_df.loc["v2_0", "gene2_0"], 2 * st.t.sf(abs(b[0] / se[0]), dof), rtol=1e-4)
    np.testing.assert_allclose(af_s.loc["v2_0"], g.sum() / (2 * len(g)), rtol=1e-6)


def test_map_trans_maf_filter_and_logp(cohort):
    d = cohort
    gdf = d["genotype_df"].copy()
    gdf.loc["v1_0"] = 0.0
    gdf.loc["v1_0", d["samples"][0]] = 1.0  # MAF ~ 0.8%
    out = trans.map_trans(gdf, d["phenotype_df"], None, pval_threshold=1.0, maf_threshold=0.05, device="cpu",
                          verbose=False, logp=True)
    assert "v1_0" not in set(out.variant_id)
    assert (out.pval >= 0).all()  # -log10(p)


def test_map_trans_interaction_matches_ols(cohort):
    d = cohort
    i_s = pd.Series(d["rng"].normal(size=len(d["samples"])), index=d["samples"])
    out = trans.map_trans(d["genotype_df"], d["phenotype_df"], d["covariates_df"], interaction_s=i_s,
                          pval_threshold=1.0, maf_threshold=0, device="cpu", verbose=False)
    assert list(out.columns) == ['variant_id', 'phenotype_id', 'pval_g', 'b_g', 'b_g_se', 'pval_i', 'b_i', 'b_i_se',
                                 'pval_gi', 'b_gi', 'b_gi_se', 'af']
    y = d["phenotype_df"].loc["gene1_1"].to_numpy(np.float64)
    g = d["genotype_df"].loc["v1_3"].to_numpy(np.float64)
    b, se, dof = ols_fit(y, np.column_stack([g, i_s.values, g * i_s.values, d["covariates_df"].values]))
    row = out[(out.variant_id == "v1_3") & (out.phenotype_id == "gene1_1")].iloc[0]
    np.testing.assert_allclose([row.b_g, row.b_i, row.b_gi], b[:3], rtol=1e-3)
    np.testing.assert_allclose([row.b_g_se, row.b_i_se, row.b_gi_se], se[:3], rtol=1e-3)
    np.testing.assert_allclose(row.pval_gi, 2 * st.t.sf(abs(b[2] / se[2]), dof), rtol=1e-3)
    dense = trans.map_trans(d["genotype_df"], d["phenotype_df"], d["covariates_df"], interaction_s=i_s,
                            return_sparse=False, maf_threshold=0, device="cpu", verbose=False)
    assert len(dense) == 6 and dense[2].shape[0] == d["genotype_df"].shape[0]
    np.testing.assert_allclose(dense[2].loc["v1_3", "gene1_1"], row.pval_gi, rtol=1e-5)


def test_map_trans_with_local_ancestry_covariates(cohort):
    """y ~ g + h + C per variant, h = local-ancestry count at the variant."""
    d = cohort
    panel, admix = make_random_panel(d["variant_df"], d["samples"], seed=9)
    out = trans.map_trans(d["genotype_df"], d["phenotype_df"], d["covariates_df"], haplotypes=panel,
                          variant_df=d["variant_df"], pval_threshold=1.0, maf_threshold=0, device="cpu",
                          verbose=False, return_r2=True)
    assert len(out) == d["genotype_df"].shape[0] * d["phenotype_df"].shape[0]
    k = list(d["variant_df"].index).index("v2_2")
    y = d["phenotype_df"].loc["gene1_0"].to_numpy(np.float64)
    g = d["genotype_df"].loc["v2_2"].to_numpy(np.float64)
    h = admix[k, :, 0].astype(np.float64)
    b, se, dof = ols_fit(y, np.column_stack([g, h, d["covariates_df"].values]))
    row = out[(out.variant_id == "v2_2") & (out.phenotype_id == "gene1_0")].iloc[0]
    np.testing.assert_allclose(row.b, b[0], rtol=1e-4)
    np.testing.assert_allclose(row.b_se, se[0], rtol=1e-4)
    np.testing.assert_allclose(row.pval, 2 * st.t.sf(abs(b[0] / se[0]), dof), rtol=1e-4)
    with pytest.raises(ValueError, match="variant_df"):
        trans.map_trans(d["genotype_df"], d["phenotype_df"], haplotypes=panel, device="cpu", verbose=False)


def test_filter_cis_and_generator(cohort):
    d = cohort
    out = trans.map_trans(d["genotype_df"], d["phenotype_df"], None, pval_threshold=1.0, maf_threshold=0,
                          device="cpu", verbose=False)
    kept = trans.filter_cis(out, d["phenotype_pos_df"], d["variant_df"], window=10_000)
    # every chr1 phenotype x chr1 variant pair is within 10 kb -> only cross-chromosome pairs remain
    chrom_v = d["variant_df"]["chrom"]
    chrom_p = d["phenotype_pos_df"]["chr"]
    assert (kept.variant_id.map(chrom_v).to_numpy() != kept.phenotype_id.map(chrom_p).to_numpy()).all()
    assert len(kept) == 8 * 1 + 6 * 2
    chr_s = d["variant_df"]["chrom"]
    ggt = trans.GenotypeGeneratorTrans(d["genotype_df"], batch_size=5, chr_s=chr_s)
    assert ggt.chroms == ["chr1", "chr2"]
    batches = list(ggt.generate_data(chrom="chr1"))
    assert sum(len(b[1]) for b in batches) == 8 and all(chr_s[b[1]].eq("chr1").all() for b in batches)
    with pytest.raises(ValueError, match="contiguous"):
        trans.GenotypeGeneratorTrans(d["genotype_df"], chr_s=pd.Series(["chr1", "chr2"] * 7, index=chr_s.index))


def test_trans_permutations_and_apply(cohort):
    d = cohort
    res = trans.map_trans_permutations(d["genotype_df"], d["covariates_df"], nperms=40, maf_threshold=0, seed=1,
                                       device="cpu", verbose=False)
    assert list(res.index) == ['num_samples', 'dof', 'beta_shape1', 'beta_shape2', 'true_dof', 'minp_true_dof',
                               'minp_empirical']
    assert res['dof'] == len(d["samples"]) - 2 - 2 and len(res['minp_empirical']) == 40
    # reproducible with the same seed
    res2 = trans.map_trans_permutations(d["genotype_df"], d["covariates_df"], nperms=40, maf_threshold=0, seed=1,
                                        device="cpu", verbose=False)
    np.testing.assert_allclose(res['minp_empirical'], res2['minp_empirical'])
    pairs = trans.map_trans(d["genotype_df"], d["phenotype_df"], d["covariates_df"], pval_threshold=1.0,
                            maf_threshold=0, return_r2=True, device="cpu", verbose=False)
    out = trans.apply_permutations(res, pairs.copy())
    for c in ["beta_shape1", "beta_shape2", "true_dof", "pval_true_dof", "pval_perm", "pval_beta"]:
        assert c in out.columns
    assert ((out.pval_perm > 0) & (out.pval_perm <= 1)).all()
    # smaller nominal p -> smaller (or equal) permutation p
    o = out.sort_values("pval")
    assert (np.diff(o.pval_perm.to_numpy()) >= 0).all()
    # per-chromosome null (leave-one-chromosome-out)
    res_chr = trans.map_trans_permutations(d["genotype_df"], d["covariates_df"], nperms=30, maf_threshold=0,
                                           chr_s=d["variant_df"]["chrom"], seed=2, device="cpu", verbose=False)
    assert list(res_chr.index) == ["chr1", "chr2"]
    out2 = trans.apply_permutations(res_chr, pairs.copy(), phenotype_pos_df=d["phenotype_pos_df"])
    assert out2["phenotype_chr"].isin(["chr1", "chr2"]).all() and out2["pval_beta"].between(0, 1).all()
    with pytest.raises(ValueError, match="r2"):
        trans.apply_permutations(res, pairs.drop(columns="r2"))
