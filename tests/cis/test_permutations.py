import numpy as np
import pandas as pd
import torch

from src.localqtl.cis.permutations import map_permutations
from src.localqtl.utils import SimpleLogger


def test_map_permutations_computes_empirical_stats(toy_data):
    torch.manual_seed(0)
    result = map_permutations(
        genotype_df=toy_data["genotype_df"],
        variant_df=toy_data["variant_df"],
        phenotype_df=toy_data["phenotype_df"],
        phenotype_pos_df=toy_data["phenotype_pos_df"],
        covariates_df=None,
        device="cpu",
        nperm=5,
        seed=123,
        logger=SimpleLogger(verbose=False),
    )

    assert {"phenotype_id", "variant_id", "pval_perm", "pval_beta", "true_dof"}.issubset(result.columns)
    expected_ids = {pid for pid in toy_data["phenotype_df"].index if pid != "geneConst"}
    assert set(result["phenotype_id"]) == expected_ids
    assert ((result["pval_perm"] >= 0.0) & (result["pval_perm"] <= 1.0)).all()
    assert np.isfinite(result[["slope", "slope_se", "tstat", "r2_nominal"]]).all().all()


def test_map_permutations_grouped_mode_returns_one_row_per_group(toy_data):
    torch.manual_seed(0)
    group_s = pd.Series(
        {
            "geneA": "grp1",
            "geneB": "grp1",
            "geneC": "grp2",
            "geneConst": "grp1",
        }
    )

    result = map_permutations(
        genotype_df=toy_data["genotype_df"],
        variant_df=toy_data["variant_df"],
        phenotype_df=toy_data["phenotype_df"],
        phenotype_pos_df=toy_data["phenotype_pos_df"],
        covariates_df=None,
        group_s=group_s,
        device="cpu",
        nperm=5,
        seed=123,
        logger=SimpleLogger(verbose=False),
    )

    assert {"group_id", "group_size", "phenotype_id", "variant_id", "pval_perm"}.issubset(result.columns)
    assert set(result["group_id"]) == {"grp1", "grp2"}


def test_map_permutations_with_covariate_interaction_runs(toy_data):
    covariates_df = pd.DataFrame(
        {
            "cov1": [0.1, 0.2, 0.3, 0.4],
            "cov2": [1.0, 0.5, -0.5, -1.0],
        },
        index=toy_data["samples"],
    )

    result = map_permutations(
        genotype_df=toy_data["genotype_df"],
        variant_df=toy_data["variant_df"],
        phenotype_df=toy_data["phenotype_df"],
        phenotype_pos_df=toy_data["phenotype_pos_df"],
        covariates_df=covariates_df,
        device="cpu",
        nperm=4,
        seed=123,
        logger=SimpleLogger(verbose=False),
        interaction_covariate="cov1",
    )

    assert not result.empty
    assert result["pval_nominal"].notna().any()
    assert result.shape[0] == 2
    sizes = dict(zip(result["group_id"], result["group_size"]))
    assert sizes["grp1"] == 2
    assert sizes["grp2"] == 1
    assert ((result["pval_perm"] >= 0.0) & (result["pval_perm"] <= 1.0)).all()
