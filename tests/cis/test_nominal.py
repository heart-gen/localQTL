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
                    "fstat_joint", "pval_joint", "z_diff_anc0_anc1", "pval_diff_anc0_anc1"]
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
