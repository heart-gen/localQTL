import numpy as np
import pandas as pd
import torch
import pytest

from src.localqtl.cis.permutations import map_permutations
from src.localqtl.cis.nominal import map_nominal
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
    # one row per non-constant phenotype; no grouping was requested
    expected_ids = {pid for pid in toy_data["phenotype_df"].index if pid != "geneConst"}
    assert set(result["phenotype_id"]) == expected_ids
    assert "group_id" not in result.columns
    assert ((result["pval_perm"] >= 0.0) & (result["pval_perm"] <= 1.0)).all()


def test_lead_nominal_pval_matches_map_nominal_with_covariates(random_cohort):
    """Regression: the permutation core must count covariates in the residual dof."""
    from src.localqtl.cis.nominal import map_nominal
    d = random_cohort(seed=5, n=40, m=6)
    covariates_df = pd.DataFrame({"cov1": d["rng"].normal(size=40), "cov2": d["rng"].normal(size=40)},
                                 index=d["samples"])
    common = dict(genotype_df=d["genotype_df"], variant_df=d["variant_df"],
                  phenotype_df=d["phenotype_df"], phenotype_pos_df=d["phenotype_pos_df"],
                  covariates_df=covariates_df, device="cpu", window=1000,
                  logger=SimpleLogger(verbose=False))
    perm = map_permutations(nperm=3, seed=1, beta_approx=False, **common)
    nom = map_nominal(nperm=None, out_dir=None, **common).set_index(["phenotype_id", "variant_id"])
    for _, row in perm.iterrows():
        ref = nom.loc[(row["phenotype_id"], row["variant_id"])]
        np.testing.assert_allclose(row["pval_nominal"], ref["pval_nominal"], rtol=1e-4)
        np.testing.assert_allclose(row["slope"], ref["slope"], rtol=1e-4)


def test_paired_covariate_lead_matches_map_nominal(random_cohort):
    from scipy import stats as st
    d = random_cohort(seed=11, n=60, m=6)
    rng = d["rng"]
    cov = pd.DataFrame(rng.normal(size=(60, 2)), index=d["samples"], columns=["a", "b"])
    paired = pd.DataFrame(rng.normal(size=(1, 60)), index=["geneA"], columns=d["samples"])
    lg = SimpleLogger(verbose=False)
    perm = map_permutations(d["genotype_df"], d["variant_df"], d["phenotype_df"], d["phenotype_pos_df"], cov,
                            paired_covariate_df=paired, nperm=30, device="cpu", seed=1, logger=lg)
    nom = map_nominal(d["genotype_df"], d["variant_df"], d["phenotype_df"], d["phenotype_pos_df"], cov,
                      paired_covariate_df=paired, device="cpu", out_dir=None, logger=lg)
    r = perm.iloc[0]
    nr = nom[nom.variant_id == r.variant_id].iloc[0]
    np.testing.assert_allclose(r.slope, nr.slope, rtol=1e-5)
    np.testing.assert_allclose(r.pval_nominal, nr.pval_nominal, rtol=1e-4)
    assert r.variant_id == nom.sort_values("pval_nominal").variant_id.iloc[0]
    # and it is the OLS fit with the extra covariate
    g = d["genotype_df"].loc[r.variant_id].to_numpy(np.float64)
    X1 = np.column_stack([np.ones(60), g, cov.values, paired.values[0]])
    b = np.linalg.lstsq(X1, d["y"], rcond=None)[0]
    res = d["y"] - X1 @ b
    dof = 60 - X1.shape[1]
    se = np.sqrt(np.diag((res @ res) / dof * np.linalg.inv(X1.T @ X1)))
    np.testing.assert_allclose(r.slope, b[1], rtol=1e-4)
    np.testing.assert_allclose(r.pval_nominal, 2 * st.t.sf(abs(b[1] / se[1]), dof), rtol=1e-4)
    with pytest.raises(ValueError, match="groups"):
        map_permutations(d["genotype_df"], d["variant_df"], d["phenotype_df"], d["phenotype_pos_df"], cov,
                         paired_covariate_df=paired, group_s=pd.Series({"geneA": "g"}), nperm=10, device="cpu",
                         logger=lg)


def test_covariate_interaction_lead_matches_ols(random_cohort):
    from scipy import stats as st
    d = random_cohort(seed=12, n=60, m=6)
    rng = d["rng"]
    cov = pd.DataFrame(rng.normal(size=(60, 2)), index=d["samples"], columns=["a", "c"])
    perm = map_permutations(d["genotype_df"], d["variant_df"], d["phenotype_df"], d["phenotype_pos_df"], cov,
                            interaction_covariate="c", nperm=20, device="cpu", seed=1,
                            logger=SimpleLogger(verbose=False))
    r = perm.iloc[0]
    g = d["genotype_df"].loc[r.variant_id].to_numpy(np.float64)
    c = cov["c"].to_numpy()
    cz = (c - c.mean()) / c.std()
    X1 = np.column_stack([np.ones(60), g, cz, g * cz, cov[["a"]].values])
    b = np.linalg.lstsq(X1, d["y"], rcond=None)[0]
    res = d["y"] - X1 @ b
    dof = 60 - X1.shape[1]
    se = np.sqrt(np.diag((res @ res) / dof * np.linalg.inv(X1.T @ X1)))
    np.testing.assert_allclose(r.slope, b[3], rtol=1e-3)
    np.testing.assert_allclose(r.pval_nominal, 2 * st.t.sf(abs(b[3] / se[3]), dof), rtol=1e-3)
