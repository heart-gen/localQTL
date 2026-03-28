import numpy as np
import pandas as pd
import pytest

from src.localqtl.cis.nominal import map_nominal
from src.localqtl.cis.postproc import annotate_ancestry_difference
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
