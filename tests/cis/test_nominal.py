import numpy as np
import pandas as pd
import pytest

from src.localqtl.cis.nominal import map_nominal
from src.localqtl.cis.postproc import annotate_ancestry_difference, annotate_with_susie
from src.localqtl.utils import SimpleLogger


@pytest.fixture
def covariates_df(toy_data):
    return pd.DataFrame(
        {
            "cov1": [0.1, 0.2, 0.3, 0.4],
            "cov2": [1.0, 0.5, -0.5, -1.0],
        },
        index=toy_data["samples"],
    )


def test_map_nominal_with_covariate_interaction_runs(toy_data, covariates_df):
    result = map_nominal(
        genotype_df=toy_data["genotype_df"],
        variant_df=toy_data["variant_df"],
        phenotype_df=toy_data["phenotype_df"],
        phenotype_pos_df=toy_data["phenotype_pos_df"],
        covariates_df=covariates_df,
        device="cpu",
        nperm=None,
        out_dir=None,
        logger=SimpleLogger(verbose=False),
        interaction_covariate="cov1",
    )

    assert not result.empty
    assert result["pval_nominal"].notna().any()


def test_covariate_interaction_canonical_columns(toy_data, covariates_df):
    """Verify the canonical b_g/b_i/b_gi columns are present and non-NaN."""
    result = map_nominal(
        genotype_df=toy_data["genotype_df"],
        variant_df=toy_data["variant_df"],
        phenotype_df=toy_data["phenotype_df"],
        phenotype_pos_df=toy_data["phenotype_pos_df"],
        covariates_df=covariates_df,
        device="cpu",
        nperm=None,
        out_dir=None,
        logger=SimpleLogger(verbose=False),
        interaction_covariate="cov1",
    )

    canonical = ["b_g", "se_g", "tstat_g", "pval_g",
                 "b_i", "se_i", "tstat_i", "pval_i",
                 "b_gi", "se_gi", "tstat_gi", "pval_gi"]
    for col in canonical:
        assert col in result.columns, f"Missing column: {col}"
        assert result[col].notna().any(), f"Column {col} is all NaN"

    # Standard errors should be positive
    for col in ["se_g", "se_i", "se_gi"]:
        assert (result[col].dropna() > 0).all(), f"{col} has non-positive values"

    # P-values should be in [0, 1]
    for col in ["pval_g", "pval_i", "pval_gi"]:
        vals = result[col].dropna()
        assert (vals >= 0).all() and (vals <= 1).all(), f"{col} out of [0,1]"


def test_ancestry_interaction_joint_model(toy_hap_data_2anc):
    """Test that ancestry interaction mode produces per-ancestry columns."""
    d = toy_hap_data_2anc
    result = map_nominal(
        genotype_df=d["genotype_df"],
        variant_df=d["variant_df"],
        phenotype_df=d["phenotype_df"],
        phenotype_pos_df=d["phenotype_pos_df"],
        haplotypes=d["haplotypes"],
        loci_df=d["loci_df"],
        device="cpu",
        nperm=None,
        out_dir=None,
        logger=SimpleLogger(verbose=False),
        ancestry_model="interaction",
    )

    assert not result.empty
    # Should have per-ancestry interaction columns
    for anc in range(2):
        for prefix in ("slope_gxh_", "slope_se_gxh_", "tstat_gxh_", "pval_gxh_"):
            col = f"{prefix}anc{anc}"
            assert col in result.columns, f"Missing column: {col}"


def test_annotate_ancestry_difference(toy_hap_data_2anc):
    """Test the difference-of-effects annotation."""
    d = toy_hap_data_2anc
    result = map_nominal(
        genotype_df=d["genotype_df"],
        variant_df=d["variant_df"],
        phenotype_df=d["phenotype_df"],
        phenotype_pos_df=d["phenotype_pos_df"],
        haplotypes=d["haplotypes"],
        loci_df=d["loci_df"],
        device="cpu",
        nperm=None,
        out_dir=None,
        logger=SimpleLogger(verbose=False),
        ancestry_model="interaction",
    )

    annotated = annotate_ancestry_difference(result, n_ancestries=2)
    # Should have difference columns for anc0 vs anc1
    assert "beta_diff_anc0_anc1" in annotated.columns
    assert "pval_diff_anc0_anc1" in annotated.columns
    assert "z_diff_anc0_anc1" in annotated.columns

    # P-values should be in [0, 1] where not NaN
    pvals = annotated["pval_diff_anc0_anc1"].dropna()
    if len(pvals) > 0:
        assert (pvals >= 0).all() and (pvals <= 1).all()


def test_map_nominal_no_interaction(toy_data):
    """Baseline: nominal scan without any interaction works."""
    result = map_nominal(
        genotype_df=toy_data["genotype_df"],
        variant_df=toy_data["variant_df"],
        phenotype_df=toy_data["phenotype_df"],
        phenotype_pos_df=toy_data["phenotype_pos_df"],
        device="cpu",
        nperm=None,
        out_dir=None,
        logger=SimpleLogger(verbose=False),
    )

    assert not result.empty
    assert "slope" in result.columns
    assert "pval_nominal" in result.columns
    # No interaction columns should be present
    assert "b_g" not in result.columns
    assert "slope_gxh_anc0" not in result.columns


def test_annotate_with_susie_collapses_duplicate_cs_membership():
    nominal_df = pd.DataFrame(
        {
            "phenotype_id": ["geneA", "geneA", "geneB"],
            "variant_id": ["var1", "var2", "var3"],
        }
    )
    susie_summary = pd.DataFrame(
        {
            "phenotype_id": ["geneA", "geneA", "geneA"],
            "variant_id": ["var1", "var1", "var2"],
            "pip": [0.7, 0.7, 0.2],
            "cs_id": ["2", "1", "3"],
        }
    )

    annotated = annotate_with_susie(nominal_df, susie_summary)

    assert annotated.loc[0, "susie_pip"] == pytest.approx(0.7)
    assert annotated.loc[0, "susie_cs"] == "1,2"
    assert annotated.loc[1, "susie_cs"] == "3"
    assert pd.isna(annotated.loc[2, "susie_pip"])
    assert annotated.loc[2, "susie_cs"] is None


def test_ancestry_interaction_with_panel(toy_panel_2anc, toy_hap_data_2anc):
    """LocalAncestryPanel input (no loci_df) runs the interaction model end to end
    and matches the legacy array input built from the same counts."""
    d = toy_panel_2anc
    common = dict(genotype_df=d["genotype_df"], variant_df=d["variant_df"],
                  phenotype_df=d["phenotype_df"], phenotype_pos_df=d["phenotype_pos_df"],
                  device="cpu", nperm=None, out_dir=None, logger=SimpleLogger(verbose=False),
                  ancestry_model="interaction")
    res_panel = map_nominal(haplotypes=d["panel"], **common)
    assert not res_panel.empty and "pval_gxh_anc0" in res_panel.columns

    H = d["admix"][:, :, [0]].astype(float)             # legacy path: first channel, loci == variants
    res_array = map_nominal(haplotypes=H, loci_df=d["panel"].loci_df, **common)
    num = [c for c in res_panel.columns if res_panel[c].dtype.kind == "f"]
    np.testing.assert_allclose(res_panel[num].to_numpy(), res_array[num].to_numpy(), rtol=1e-5, equal_nan=True)


def _dosage_common(d, **extra):
    return dict(genotype_df=d["genotype_df"], variant_df=d["variant_df"],
                phenotype_df=d["phenotype_df"], phenotype_pos_df=d["phenotype_pos_df"],
                device="cpu", nperm=None, out_dir=None, logger=SimpleLogger(verbose=False), **extra)


def test_dosage_model_columns_and_hard_call_dosages(toy_panel_2anc):
    """ancestry_model='dosage' on a hard-call panel: per-ancestry, joint and
    pairwise-difference columns, finite where estimable."""
    from localqtl.cis.nominal import _dosage_columns

    d = toy_panel_2anc
    res = map_nominal(haplotypes=d["panel"], ancestry_model="dosage", **_dosage_common(d))
    assert not res.empty
    cols = _dosage_columns(2)
    assert cols == ["slope_anc0", "slope_se_anc0", "tstat_anc0", "pval_anc0",
                    "slope_anc1", "slope_se_anc1", "tstat_anc1", "pval_anc1",
                    "fstat_joint", "pval_joint", "z_diff_anc0_anc1", "pval_diff_anc0_anc1",
                    "an_anc0", "ac_anc0", "af_anc0", "ma_count_anc0",
                    "an_anc1", "ac_anc1", "af_anc1", "ma_count_anc1"]
    assert all(c in res.columns for c in cols)
    assert "slope_gxh_anc0" not in res.columns
    # lead columns are the ancestry with the largest |t|
    best = np.where(res["tstat_anc0"].abs() >= res["tstat_anc1"].abs(), res["slope_anc0"], res["slope_anc1"])
    np.testing.assert_allclose(res["slope"].to_numpy(), best, rtol=1e-6)
    assert ((res["pval_joint"] >= 0) & (res["pval_joint"] <= 1)).all()
    assert ((res["pval_diff_anc0_anc1"] >= 0) & (res["pval_diff_anc0_anc1"] <= 1)).all()


def test_dosage_model_homogeneous_ancestry_matches_main_model():
    """With every haplotype assigned to ancestry 0, D_0 == G and the ancestry-0
    effect equals the plain genotype slope of the main model (with a covariate,
    so both designs are centred and the constant local-ancestry column vanishes)."""
    from rfmix_reader import from_legacy
    from localqtl.haplotypeio import LocalAncestryPanel

    rng = np.random.default_rng(3)
    n, m = 16, 6
    samples = [f"S{i}" for i in range(n)]
    vids = [f"v{k}" for k in range(m)]
    variant_df = pd.DataFrame({"chrom": "1", "pos": np.arange(1, m + 1) * 100}, index=vids)
    genotype_df = pd.DataFrame(rng.integers(0, 3, size=(m, n)).astype(np.float32), index=vids, columns=samples)
    phenotype_df = pd.DataFrame(rng.normal(size=(2, n)).astype(np.float32), index=["gA", "gB"], columns=samples)
    phenotype_pos_df = pd.DataFrame({"chr": "1", "start": [250, 450], "end": [250, 450]}, index=["gA", "gB"])
    covariates_df = pd.DataFrame({"c1": rng.normal(size=n)}, index=samples)

    loci = pd.DataFrame({"chromosome": "1", "physical_position": variant_df["pos"].values})
    g_anc = pd.DataFrame({"sample_id": samples, "P0": 1.0, "P1": 0.0, "chrom": "1"})
    admix = np.zeros((m, n, 2), dtype=np.int8)
    admix[:, :, 0] = 2
    panel = LocalAncestryPanel(from_legacy(loci, g_anc, admix))

    common = dict(genotype_df=genotype_df, variant_df=variant_df, phenotype_df=phenotype_df,
                  phenotype_pos_df=phenotype_pos_df, covariates_df=covariates_df, device="cpu",
                  nperm=None, out_dir=None, logger=SimpleLogger(verbose=False))
    res_d = map_nominal(haplotypes=panel, ancestry_model="dosage", **common)
    res_m = map_nominal(**common)
    merged = res_d.merge(res_m, on=["phenotype_id", "variant_id"], suffixes=("_d", "_m"))
    assert len(merged) == len(res_d) == 2 * m
    np.testing.assert_allclose(merged["slope_anc0"], merged["slope_m"], rtol=1e-3, atol=1e-5)
    # ancestry 1 carries no alleles: its column is the (regularised) zero
    np.testing.assert_allclose(merged["slope_anc1"], 0.0, atol=1e-3)


def test_dosage_model_posterior_panel_runs_and_differs_from_hard_calls(toy_panel_2anc_posterior):
    d = toy_panel_2anc_posterior
    res_post = map_nominal(haplotypes=d["panel_post"], ancestry_model="dosage", **_dosage_common(d))
    res_hard = map_nominal(haplotypes=d["panel_hard"], ancestry_model="dosage", **_dosage_common(d))
    assert len(res_post) == len(res_hard) > 0
    assert np.isfinite(res_post["slope_anc0"]).any()
    # soft posteriors change the design, so the estimates move
    assert not np.allclose(res_post["slope_anc0"].to_numpy(), res_hard["slope_anc0"].to_numpy())


def test_dosage_model_argument_validation(toy_panel_2anc, covariates_df):
    from localqtl.cis.permutations import map_permutations

    d = toy_panel_2anc
    with pytest.raises(ValueError, match="requires haplotypes"):
        map_nominal(ancestry_model="dosage", **_dosage_common(d))
    with pytest.raises(ValueError, match="nperm"):
        map_nominal(haplotypes=d["panel"], ancestry_model="dosage",
                    **{**_dosage_common(d), "nperm": 5})
    with pytest.raises(ValueError, match="interaction_covariate"):
        map_nominal(haplotypes=d["panel"], ancestry_model="dosage", covariates_df=covariates_df,
                    interaction_covariate=covariates_df.columns[0], **_dosage_common(d))
    with pytest.raises(ValueError, match="Unknown ancestry_model"):
        map_nominal(haplotypes=d["panel"], ancestry_model="bogus", **_dosage_common(d))
    with pytest.raises(ValueError, match="only implemented for map_nominal"):
        map_permutations(genotype_df=d["genotype_df"], variant_df=d["variant_df"],
                         phenotype_df=d["phenotype_df"], phenotype_pos_df=d["phenotype_pos_df"],
                         haplotypes=d["panel"], ancestry_model="dosage", device="cpu", nperm=4,
                         logger=SimpleLogger(verbose=False))


# ---------------------------------------------------------------------------
# Phase 0 regression tests: intercept without covariates, GxC + local ancestry,
# ancestry_model whitelist, map_nominal_loci window handling.
# ---------------------------------------------------------------------------

def _ols(X, y):
    """Plain numpy OLS with intercept -> (beta, se, t) for column 1 of X (after intercept)."""
    X1 = np.column_stack([np.ones(len(y)), X])
    XtX_inv = np.linalg.inv(X1.T @ X1)
    beta = XtX_inv @ X1.T @ y
    resid = y - X1 @ beta
    dof = len(y) - X1.shape[1]
    sigma2 = resid @ resid / dof
    se = np.sqrt(np.diag(XtX_inv) * sigma2)
    return beta, se, beta / se, dof


def _random_cohort(seed=0, n=40, m=6):
    rng = np.random.default_rng(seed)
    samples = [f"S{i}" for i in range(n)]
    G = rng.integers(0, 3, size=(m, n)).astype(np.float32)
    variant_ids = [f"v{i}_chr1_{100 + 10 * i}" for i in range(m)]
    genotype_df = pd.DataFrame(G, index=variant_ids, columns=samples)
    variant_df = pd.DataFrame({"chrom": "chr1", "pos": [100 + 10 * i for i in range(m)]}, index=variant_ids)
    y = 0.4 * G[0] + rng.normal(size=n)
    phenotype_df = pd.DataFrame([y], index=["geneA"], columns=samples).astype(np.float32)
    phenotype_pos_df = pd.DataFrame({"chr": ["chr1"], "start": [120], "end": [120]}, index=["geneA"])
    return dict(genotype_df=genotype_df, variant_df=variant_df, phenotype_df=phenotype_df,
                phenotype_pos_df=phenotype_pos_df, samples=samples, G=G, y=y, rng=rng)


def test_no_covariates_matches_intercept_ols():
    d = _random_cohort()
    res = map_nominal(genotype_df=d["genotype_df"], variant_df=d["variant_df"],
                      phenotype_df=d["phenotype_df"], phenotype_pos_df=d["phenotype_pos_df"],
                      covariates_df=None, device="cpu", nperm=None, out_dir=None,
                      window=1000, logger=SimpleLogger(verbose=False))
    res = res.set_index("variant_id")
    for i, vid in enumerate(d["genotype_df"].index):
        beta, se, t, dof = _ols(d["G"][i], d["y"])
        np.testing.assert_allclose(res.loc[vid, "slope"], beta[1], rtol=1e-4)
        np.testing.assert_allclose(res.loc[vid, "slope_se"], se[1], rtol=1e-3)
        np.testing.assert_allclose(res.loc[vid, "tstat"], t[1], rtol=1e-3)


def test_no_covariates_equals_constant_covariate():
    d = _random_cohort(seed=1)
    common = dict(genotype_df=d["genotype_df"], variant_df=d["variant_df"],
                  phenotype_df=d["phenotype_df"], phenotype_pos_df=d["phenotype_pos_df"],
                  device="cpu", nperm=None, out_dir=None, window=1000,
                  logger=SimpleLogger(verbose=False))
    ones = pd.DataFrame({"intercept": np.ones(len(d["samples"]))}, index=d["samples"])
    a = map_nominal(covariates_df=None, **common).set_index("variant_id")
    b = map_nominal(covariates_df=ones, **common).set_index("variant_id")
    for col in ("slope", "slope_se", "tstat", "pval_nominal"):
        np.testing.assert_allclose(a[col].to_numpy(), b.loc[a.index, col].to_numpy(), rtol=1e-4)


def _random_panel_cohort(seed=3, n=60, m=6):
    """Random cohort plus an in-memory two-ancestry LocalAncestryPanel on its variants."""
    from rfmix_reader import from_legacy
    from localqtl.haplotypeio import LocalAncestryPanel
    d = _random_cohort(seed=seed, n=n, m=m)
    var_df = d["variant_df"]
    loci = pd.DataFrame({"chromosome": var_df["chrom"].astype(str).values,
                         "physical_position": var_df["pos"].values})
    g_anc = pd.DataFrame({"sample_id": d["samples"], "P0": 0.5, "P1": 0.5, "chrom": "1"})
    codes = d["rng"].integers(0, 2, size=(m, n, 2))
    admix = np.stack([(codes == a).sum(axis=-1) for a in range(2)], axis=-1).astype(np.int8)
    d["panel"] = LocalAncestryPanel(from_legacy(loci, g_anc, admix))
    d["admix"] = admix
    return d


def test_gxc_with_haplotypes_includes_ancestry_main_effects():
    """GxC with a panel fits y ~ 1 + g + c + g*c + h (h = local-ancestry count)."""
    d = _random_panel_cohort()
    cov = pd.DataFrame({"c": d["rng"].normal(size=len(d["samples"]))}, index=d["samples"])
    res = map_nominal(genotype_df=d["genotype_df"], variant_df=d["variant_df"],
                      phenotype_df=d["phenotype_df"], phenotype_pos_df=d["phenotype_pos_df"],
                      covariates_df=cov, haplotypes=d["panel"], interaction_covariate="c",
                      device="cpu", nperm=None, out_dir=None, window=1000,
                      logger=SimpleLogger(verbose=False)).set_index("variant_id")
    assert "b_gi" in res.columns and len(res) == len(d["genotype_df"])
    c = cov["c"].to_numpy()
    c = (c - c.mean()) / c.std(ddof=0)          # prepare_interaction_covariate z-scores (ddof=0)
    for i, vid in enumerate(d["genotype_df"].index):
        g = d["G"][i].astype(np.float64)
        h = d["admix"][i, :, 0].astype(np.float64)
        beta, se, t, dof = _ols(np.column_stack([g, c, g * c, h]), d["y"])
        np.testing.assert_allclose(res.loc[vid, "b_g"], beta[1], rtol=1e-3, atol=1e-5)
        np.testing.assert_allclose(res.loc[vid, "b_gi"], beta[3], rtol=1e-3, atol=1e-5)
        np.testing.assert_allclose(res.loc[vid, "se_gi"], se[3], rtol=2e-3, atol=1e-5)
        np.testing.assert_allclose(res.loc[vid, "tstat_gi"], t[3], rtol=2e-3, atol=1e-5)


def test_unknown_ancestry_model_rejected_everywhere(toy_data):
    from src.localqtl.cis.permutations import map_permutations
    from src.localqtl.cis.independent import map_independent
    common = dict(genotype_df=toy_data["genotype_df"], variant_df=toy_data["variant_df"],
                  phenotype_df=toy_data["phenotype_df"], phenotype_pos_df=toy_data["phenotype_pos_df"],
                  device="cpu", logger=SimpleLogger(verbose=False))
    with pytest.raises(ValueError, match="Unknown ancestry_model"):
        map_nominal(nperm=None, out_dir=None, ancestry_model="mian", **common)
    with pytest.raises(ValueError, match="Unknown ancestry_model"):
        map_permutations(nperm=3, ancestry_model="mian", **common)
    cis_df = pd.DataFrame({"phenotype_id": ["geneA"], "qval": [0.01], "pval_beta": [0.01]})
    with pytest.raises(ValueError, match="Unknown ancestry_model"):
        map_independent(cis_df=cis_df, nperm=3, ancestry_model="mian", **common)


def test_map_nominal_loci_uses_exact_window():
    from src.localqtl.cis.nominal import map_nominal_loci
    d = _random_cohort(seed=2, n=30, m=8)       # positions 100..170 step 10
    locus_df = pd.DataFrame({"phenotype_id": ["geneA", "geneA", "missing"],
                             "chr": ["chr1", "chr1", "chr1"],
                             "position": [120, 160, 120]})
    res = map_nominal_loci(locus_df, d["genotype_df"], d["variant_df"], d["phenotype_df"],
                           window=15, device="cpu", logger=SimpleLogger(verbose=False))
    assert set(res["phenotype_id"]) == {"geneA__locus0", "geneA__locus1"}
    for pid, center in (("geneA__locus0", 120), ("geneA__locus1", 160)):
        pos = res.loc[res["phenotype_id"] == pid, "variant_id"].str.split("_").str[-1].astype(int)
        assert pos.between(center - 15, center + 15).all()
        assert len(pos) == 3                      # 110/120/130 or 150/160/170
    # explicit start/end is used as-is (no extra padding)
    res2 = map_nominal_loci(pd.DataFrame({"phenotype_id": ["geneA"], "chr": ["chr1"],
                                          "start": [100], "end": [120]}),
                            d["genotype_df"], d["variant_df"], d["phenotype_df"],
                            device="cpu", logger=SimpleLogger(verbose=False))
    assert sorted(res2["variant_id"].str.split("_").str[-1].astype(int)) == [100, 110, 120]
    assert set(res2["phenotype_id"]) == {"geneA"}


# ---------------------------------------------------------------------------
# Phase 1: per-ancestry in-sample allele statistics and min_ancestry_ac mask
# ---------------------------------------------------------------------------

def _expected_anc_stats(G, L):
    """numpy reference: G (m,n) imputed dosages, L (m,n,K) ancestry counts."""
    an = L.sum(axis=1)
    ac = (G[:, :, None] * L * 0.5).sum(axis=1)
    af = np.where(an > 0, ac / np.where(an > 0, an, 1), np.nan)
    return an, ac, af, np.minimum(ac, an - ac)


def test_dosage_per_ancestry_af_matches_numpy_hard_calls(toy_panel_2anc):
    d = toy_panel_2anc
    res = map_nominal(haplotypes=d["panel"], ancestry_model="dosage", **_dosage_common(d))
    per_var = res.drop_duplicates("variant_id").set_index("variant_id")
    G = d["genotype_df"].to_numpy(np.float64)
    G = np.where(G == -9, np.nan, G)
    G = np.where(np.isnan(G), np.nanmean(G, axis=1, keepdims=True), G)
    L = d["admix"].astype(np.float64)                           # (variants, samples, 2)
    an, ac, af, mac = _expected_anc_stats(G, L)
    vidx = {v: i for i, v in enumerate(d["genotype_df"].index)}
    for vid in per_var.index:
        i = vidx[vid]
        for k in range(2):
            np.testing.assert_allclose(per_var.loc[vid, f"an_anc{k}"], an[i, k], rtol=1e-6)
            np.testing.assert_allclose(per_var.loc[vid, f"ac_anc{k}"], ac[i, k], rtol=1e-5)
            np.testing.assert_allclose(per_var.loc[vid, f"af_anc{k}"], af[i, k], rtol=1e-5)
            np.testing.assert_allclose(per_var.loc[vid, f"ma_count_anc{k}"], mac[i, k], rtol=1e-5)
        n = len(d["samples"])
        np.testing.assert_allclose(per_var.loc[vid, "an_anc0"] + per_var.loc[vid, "an_anc1"], 2 * n)
        np.testing.assert_allclose(per_var.loc[vid, "ac_anc0"] + per_var.loc[vid, "ac_anc1"],
                                   2 * n * per_var.loc[vid, "af"], rtol=1e-5)


def test_dosage_per_ancestry_af_posterior_panel(toy_panel_2anc_posterior):
    d = toy_panel_2anc_posterior
    res = map_nominal(haplotypes=d["panel_post"], ancestry_model="dosage", **_dosage_common(d))
    per_var = res.drop_duplicates("variant_id").set_index("variant_id")
    n = len(d["samples"])
    # expected ancestry counts: posterior summed over the two haplotypes
    L = d["posterior"].sum(axis=2).astype(np.float64)           # (variants, samples, 2)
    assert np.all(np.isfinite(per_var[["an_anc0", "an_anc1"]].to_numpy()))
    np.testing.assert_allclose(per_var["an_anc0"] + per_var["an_anc1"], 2 * n, rtol=1e-5)
    vidx = {v: i for i, v in enumerate(d["genotype_df"].index)}
    for vid in per_var.index:
        np.testing.assert_allclose(per_var.loc[vid, "an_anc0"], L[vidx[vid], :, 0].sum(), rtol=1e-5)
    # fractional counts are allowed
    assert not np.allclose(per_var["an_anc0"].to_numpy(), np.round(per_var["an_anc0"].to_numpy()))


def test_min_ancestry_ac_masks_only_that_ancestry(toy_panel_2anc):
    d = toy_panel_2anc
    base = map_nominal(haplotypes=d["panel"], ancestry_model="dosage", **_dosage_common(d))
    thr = float(np.nanmedian(np.concatenate([base["ma_count_anc0"], base["ma_count_anc1"]]))) + 1e-6
    res = map_nominal(haplotypes=d["panel"], ancestry_model="dosage", min_ancestry_ac=thr,
                      **_dosage_common(d))
    assert len(res) == len(base)                                  # rows are kept
    np.testing.assert_allclose(res["fstat_joint"], base["fstat_joint"], rtol=1e-6, equal_nan=True)
    for k in range(2):
        masked = res[f"ma_count_anc{k}"] < thr
        assert masked.any() and (~masked).any()
        assert res.loc[masked, f"slope_anc{k}"].isna().all()
        assert res.loc[masked, f"pval_anc{k}"].isna().all()
        np.testing.assert_allclose(res.loc[~masked, f"slope_anc{k}"], base.loc[~masked, f"slope_anc{k}"])
    either = (res["ma_count_anc0"] < thr) | (res["ma_count_anc1"] < thr)
    assert res.loc[either, "pval_diff_anc0_anc1"].isna().all()
    assert res.loc[~either, "pval_diff_anc0_anc1"].notna().all()
    # lead columns come from an unmasked ancestry when one remains
    one_masked = either & ~((res["ma_count_anc0"] < thr) & (res["ma_count_anc1"] < thr))
    if one_masked.any():
        keep_k = np.where(res.loc[one_masked, "ma_count_anc0"] < thr, 1, 0)
        expected = np.where(keep_k == 0, res.loc[one_masked, "slope_anc0"], res.loc[one_masked, "slope_anc1"])
        np.testing.assert_allclose(res.loc[one_masked, "slope"], expected)
    with pytest.raises(ValueError, match="min_ancestry_ac"):
        map_nominal(haplotypes=d["panel"], ancestry_model="main", min_ancestry_ac=1.0, **_dosage_common(d))


# ---------------------------------------------------------------------------
# tensorQTL parity: interaction_df, maf_threshold_interaction, paired covariates,
# top associations / eigenMT, best_only
# ---------------------------------------------------------------------------
def _interaction_cohort(seed=3, n=70, m=8):
    d = _random_cohort(seed=seed, n=n, m=m)
    rng = d["rng"]
    C = rng.normal(size=(n, 2))
    I = rng.normal(size=(n, 2))
    y = 0.3 * d["G"][2] + 0.25 * d["G"][2] * I[:, 1] + rng.normal(size=n)
    d["phenotype_df"] = pd.DataFrame([y], index=["geneA"], columns=d["samples"]).astype(np.float32)
    d["y"] = y
    d["covariates_df"] = pd.DataFrame(C, index=d["samples"], columns=["c1", "c2"])
    d["interaction_df"] = pd.DataFrame(I, index=d["samples"], columns=["age", "sex"])
    return d


def _fit(y, X):
    X1 = np.column_stack([np.ones(len(y)), X])
    b = np.linalg.lstsq(X1, y, rcond=None)[0]
    r = y - X1 @ b
    dof = len(y) - X1.shape[1]
    cov = (r @ r) / dof * np.linalg.inv(X1.T @ X1)
    return b[1:], np.sqrt(np.diag(cov))[1:], dof


def test_interaction_df_multiple_terms_matches_ols():
    from scipy import stats as st
    d = _interaction_cohort()
    res = map_nominal(d["genotype_df"], d["variant_df"], d["phenotype_df"], d["phenotype_pos_df"],
                      d["covariates_df"], interaction_df=d["interaction_df"], device="cpu", out_dir=None,
                      logger=SimpleLogger(verbose=False))
    for c in ["b_g", "se_g", "tstat_g", "pval_g", "b_age", "pval_age", "b_sex", "b_g-age", "se_g-age",
              "tstat_g-age", "pval_g-age", "b_g-sex", "pval_g-sex"]:
        assert c in res.columns
    assert "b_i" not in res.columns
    G, I, C = d["G"], d["interaction_df"].values, d["covariates_df"].values
    b, se, dof = _fit(d["y"], np.column_stack([G[2], I, G[2] * I[:, 0], G[2] * I[:, 1], C]))
    row = res[res.variant_id == d["variant_df"].index[2]].iloc[0]
    np.testing.assert_allclose([row.b_g, row.b_age, row.b_sex, row["b_g-age"], row["b_g-sex"]], b[:5], rtol=1e-3)
    np.testing.assert_allclose([row.se_g, row.se_age, row["se_g-sex"]], [se[0], se[1], se[4]], rtol=1e-3)
    np.testing.assert_allclose(row["pval_g-sex"], 2 * st.t.sf(abs(b[4] / se[4]), dof), rtol=1e-3)
    # lead columns = strongest g:i term
    k = np.argmax(np.abs([row["tstat_g-age"], row["tstat_g-sex"]]))
    assert row.slope == [row["b_g-age"], row["b_g-sex"]][k]
    top = res.attrs["top_assoc"]
    assert list(top.index) == ["geneA"] and "b_g-sex" in top.columns and "tests_emt" not in top.columns
    with pytest.raises(ValueError, match="either"):
        map_nominal(d["genotype_df"], d["variant_df"], d["phenotype_df"], d["phenotype_pos_df"],
                    d["covariates_df"], interaction_df=d["interaction_df"], interaction_covariate="c1",
                    device="cpu", out_dir=None, logger=SimpleLogger(verbose=False))


def test_interaction_single_term_with_covariates_matches_ols():
    """Regression test: g:i is built from the raw genotypes and i is residualized (Frisch-Waugh)."""
    from scipy import stats as st
    d = _interaction_cohort()
    sex = d["interaction_df"][["sex"]]
    res = map_nominal(d["genotype_df"], d["variant_df"], d["phenotype_df"], d["phenotype_pos_df"],
                      d["covariates_df"], interaction_df=sex, device="cpu", out_dir=None,
                      logger=SimpleLogger(verbose=False), run_eigenmt=True)
    G, s, C = d["G"], sex.values[:, 0], d["covariates_df"].values
    b, se, dof = _fit(d["y"], np.column_stack([G[2], s, G[2] * s, C]))
    row = res[res.variant_id == d["variant_df"].index[2]].iloc[0]
    np.testing.assert_allclose([row.b_g, row.b_i, row.b_gi], b[:3], rtol=1e-3)
    np.testing.assert_allclose([row.se_g, row.se_i, row.se_gi], se[:3], rtol=1e-3)
    np.testing.assert_allclose(row.pval_gi, 2 * st.t.sf(abs(b[2] / se[2]), dof), rtol=1e-3)
    top = res.attrs["top_assoc"]
    assert {"tests_emt", "pval_emt", "pval_adj_bh"} <= set(top.columns)
    assert top.loc["geneA", "tests_emt"] <= len(d["variant_df"])
    assert top.loc["geneA", "pval_emt"] == pytest.approx(min(top.loc["geneA", "tests_emt"] * top.loc["geneA", "pval_gi"], 1.0))
    # the legacy z-scored interaction_covariate gives the same t / p and a rescaled slope
    res2 = map_nominal(d["genotype_df"], d["variant_df"], d["phenotype_df"], d["phenotype_pos_df"],
                       d["covariates_df"], interaction_covariate=sex["sex"], device="cpu", out_dir=None,
                       logger=SimpleLogger(verbose=False))
    row2 = res2[res2.variant_id == d["variant_df"].index[2]].iloc[0]
    np.testing.assert_allclose(row2.pval_gi, row.pval_gi, rtol=1e-4)
    np.testing.assert_allclose(row2.b_gi / s.std(), row.b_gi, rtol=1e-3)


def test_maf_threshold_interaction_filters_both_halves():
    d = _interaction_cohort()
    gdf = d["genotype_df"].copy()
    sex = d["interaction_df"]["sex"]
    # variant polymorphic only in the upper half of `sex`
    upper = sex.sort_values(kind="mergesort").index[len(sex) // 2:]
    v = d["variant_df"].index[4]
    gdf.loc[v] = 0.0
    gdf.loc[v, upper[:10]] = 1.0
    kw = dict(device="cpu", out_dir=None, logger=SimpleLogger(verbose=False))
    res0 = map_nominal(gdf, d["variant_df"], d["phenotype_df"], d["phenotype_pos_df"], d["covariates_df"],
                       interaction_df=sex, maf_threshold_interaction=0.0, **kw)
    res1 = map_nominal(gdf, d["variant_df"], d["phenotype_df"], d["phenotype_pos_df"], d["covariates_df"],
                       interaction_df=sex, maf_threshold_interaction=0.05, **kw)
    assert v in set(res0.variant_id) and v not in set(res1.variant_id)
    assert len(res1) == len(res0) - 1


def test_paired_covariate_matches_ols_and_only_affects_listed_phenotypes():
    from scipy import stats as st
    d = _interaction_cohort()
    pdf = pd.concat([d["phenotype_df"], d["phenotype_df"].rename(index={"geneA": "geneB"})])
    pos = pd.concat([d["phenotype_pos_df"], d["phenotype_pos_df"].rename(index={"geneA": "geneB"})])
    paired = pd.DataFrame(d["rng"].normal(size=(1, len(d["samples"]))), index=["geneA"], columns=d["samples"])
    res = map_nominal(d["genotype_df"], d["variant_df"], pdf, pos, d["covariates_df"],
                      paired_covariate_df=paired, device="cpu", out_dir=None, logger=SimpleLogger(verbose=False))
    v = d["variant_df"].index[2]
    b, se, dof = _fit(d["y"], np.column_stack([d["G"][2], d["covariates_df"].values, paired.values[0]]))
    row = res[(res.variant_id == v) & (res.phenotype_id == "geneA")].iloc[0]
    np.testing.assert_allclose(row.slope, b[0], rtol=1e-4)
    np.testing.assert_allclose(row.slope_se, se[0], rtol=1e-4)
    np.testing.assert_allclose(row.pval_nominal, 2 * st.t.sf(abs(b[0] / se[0]), dof), rtol=1e-4)
    b2, se2, dof2 = _fit(d["y"], np.column_stack([d["G"][2], d["covariates_df"].values]))
    rowB = res[(res.variant_id == v) & (res.phenotype_id == "geneB")].iloc[0]
    np.testing.assert_allclose(rowB.slope, b2[0], rtol=1e-4)
    np.testing.assert_allclose(rowB.pval_nominal, 2 * st.t.sf(abs(b2[0] / se2[0]), dof2), rtol=1e-4)
    with pytest.raises(ValueError, match="sample"):
        map_nominal(d["genotype_df"], d["variant_df"], pdf, pos, d["covariates_df"],
                    paired_covariate_df=paired.iloc[:, ::-1], device="cpu", out_dir=None,
                    logger=SimpleLogger(verbose=False))
    with pytest.raises(ValueError, match="groups"):
        map_nominal(d["genotype_df"], d["variant_df"], pdf, pos, d["covariates_df"],
                    paired_covariate_df=paired, group_s=pd.Series({"geneA": "g", "geneB": "g"}),
                    device="cpu", out_dir=None, logger=SimpleLogger(verbose=False))


def test_top_assoc_file_and_best_only(tmp_path):
    d = _interaction_cohort()
    lg = SimpleLogger(verbose=False)
    out = map_nominal(d["genotype_df"], d["variant_df"], d["phenotype_df"], d["phenotype_pos_df"],
                      d["covariates_df"], interaction_df=d["interaction_df"][["sex"]], device="cpu",
                      out_dir=str(tmp_path), out_prefix="x.cis_qtl_pairs", run_eigenmt=True, logger=lg,
                      top_assoc_path=str(tmp_path / "x.cis_qtl_top_assoc.txt.gz"))
    assert out is None
    assert (tmp_path / "x.cis_qtl_pairs.chr1.parquet").exists()
    top = pd.read_csv(tmp_path / "x.cis_qtl_top_assoc.txt.gz", sep="\t", index_col=0)
    assert list(top.index) == ["geneA"] and {"pval_gi", "tests_emt", "pval_emt", "pval_adj_bh"} <= set(top.columns)
    pairs = pd.read_parquet(tmp_path / "x.cis_qtl_pairs.chr1.parquet")
    assert "b_gi" in pairs.columns and len(pairs) == len(d["variant_df"])
    # write_top=False returns the table instead
    top2 = map_nominal(d["genotype_df"], d["variant_df"], d["phenotype_df"], d["phenotype_pos_df"],
                       d["covariates_df"], interaction_df=d["interaction_df"][["sex"]], device="cpu",
                       out_dir=str(tmp_path / "b"), write_top=False, logger=lg)
    pd.testing.assert_frame_equal(top2.drop(columns=[c for c in top2.columns if c in ("tests_emt", "pval_emt", "pval_adj_bh")]),
                                  top.drop(columns=["tests_emt", "pval_emt", "pval_adj_bh"]), check_dtype=False, atol=1e-5)
    # best_only: no pair-level output
    top3 = map_nominal(d["genotype_df"], d["variant_df"], d["phenotype_df"], d["phenotype_pos_df"],
                       d["covariates_df"], interaction_df=d["interaction_df"][["sex"]], device="cpu",
                       out_dir=str(tmp_path / "c"), write_stats=False, logger=lg)
    assert top3 is None and not list((tmp_path / "c").glob("*.parquet"))
    assert (tmp_path / "c" / "cis_nominal.top_assoc.txt.gz").exists()
    top4 = map_nominal(d["genotype_df"], d["variant_df"], d["phenotype_df"], d["phenotype_pos_df"],
                       d["covariates_df"], interaction_df=d["interaction_df"][["sex"]], device="cpu",
                       out_dir=None, write_stats=False, logger=lg)
    assert isinstance(top4, pd.DataFrame) and list(top4.index) == ["geneA"]
    with pytest.raises(ValueError, match="interaction"):
        map_nominal(d["genotype_df"], d["variant_df"], d["phenotype_df"], d["phenotype_pos_df"],
                    d["covariates_df"], device="cpu", out_dir=None, write_stats=False, logger=lg)


def test_grouped_interaction_top_per_group():
    d = _interaction_cohort()
    pdf = pd.concat([d["phenotype_df"], d["phenotype_df"].rename(index={"geneA": "geneB"}) * 0.5])
    pos = pd.concat([d["phenotype_pos_df"], d["phenotype_pos_df"].rename(index={"geneA": "geneB"})])
    group_s = pd.Series({"geneA": "grp", "geneB": "grp"})
    res = map_nominal(d["genotype_df"], d["variant_df"], pdf, pos, d["covariates_df"], group_s=group_s,
                      interaction_df=d["interaction_df"][["sex"]], device="cpu", out_dir=None,
                      logger=SimpleLogger(verbose=False))
    top = res.attrs["top_assoc"]
    assert len(top) == 1 and top["group_id"].iloc[0] == "grp" and top["num_phenotypes"].iloc[0] == 2
