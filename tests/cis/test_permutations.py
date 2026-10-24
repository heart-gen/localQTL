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


# ---------------------------------------------------------------------------
# ancestry-specific dosage model
# ---------------------------------------------------------------------------
def _dosage_cohort(seed=4, n=80, m=10, effect=0.8):
    """Random cohort with a two-ancestry panel and an effect on variant 3."""
    from tests.conftest import make_random_panel
    rng = np.random.default_rng(seed)
    samples = [f"S{i}" for i in range(n)]
    G = rng.integers(0, 3, size=(m, n)).astype(np.float32)
    gdf = pd.DataFrame(G, index=[f"v{i}" for i in range(m)], columns=samples)
    vdf = pd.DataFrame({"chrom": "chr1", "pos": 100 + 10 * np.arange(m)}, index=gdf.index)
    panel, admix = make_random_panel(vdf, samples, seed=seed + 1)
    y = effect * G[3] + rng.normal(size=n)
    pdf = pd.DataFrame([y], index=["geneA"], columns=samples).astype(np.float32)
    pos = pd.DataFrame({"chr": ["chr1"], "start": [140], "end": [140]}, index=["geneA"])
    cov = pd.DataFrame(rng.normal(size=(n, 2)), index=samples, columns=["c1", "c2"])
    return dict(genotype_df=gdf, variant_df=vdf, phenotype_df=pdf, phenotype_pos_df=pos,
                covariates_df=cov, panel=panel, admix=admix, samples=samples, rng=rng, G=G, y=y)


def test_dosage_permutations_match_the_nominal_dosage_scan():
    d = _dosage_cohort()
    lg = SimpleLogger(verbose=False)
    kw = dict(haplotypes=d["panel"], ancestry_model="dosage", device="cpu", window=1000, logger=lg)
    perm = map_permutations(d["genotype_df"], d["variant_df"], d["phenotype_df"],
                            d["phenotype_pos_df"], d["covariates_df"], nperm=200, seed=1, **kw)
    nom = map_nominal(d["genotype_df"], d["variant_df"], d["phenotype_df"], d["phenotype_pos_df"],
                      d["covariates_df"], out_dir=None, **kw)
    assert len(perm) == 1
    r = perm.iloc[0]
    lead = nom.loc[nom["pval_joint"].idxmin()]
    assert r["variant_id"] == lead["variant_id"] == "v3"
    # float32: the permutation path reaches F through a precomputed inverse,
    # the nominal path through a solve, so tiny p-values differ in the last digit
    np.testing.assert_allclose(r["pval_nominal"], lead["pval_joint"], rtol=1e-4)
    np.testing.assert_allclose(r["fstat_joint"], lead["fstat_joint"], rtol=1e-4)
    for k in (0, 1):
        np.testing.assert_allclose(r[f"slope_anc{k}"], lead[f"slope_anc{k}"], rtol=1e-4)
        np.testing.assert_allclose(r[f"af_anc{k}"], lead[f"af_anc{k}"], rtol=1e-5)
        np.testing.assert_allclose(r[f"an_anc{k}"], lead[f"an_anc{k}"], rtol=1e-5)
    from src.localqtl.cis.nominal import _dosage_columns
    assert set(_dosage_columns(2)) <= set(perm.columns)
    dof = int(r["true_dof"])
    np.testing.assert_allclose(r["r2_nominal"], 2 * r["fstat_joint"] / (2 * r["fstat_joint"] + dof),
                               rtol=1e-5)
    assert r["pval_perm"] < 0.05 and r["pval_beta"] < 0.05
    assert np.isnan(r["pval_true_dof"])      # no t-style dof re-estimation for an F test


def test_dosage_permutations_are_calibrated_under_the_null():
    """Phenotypes with no genotype effect give ~uniform permutation p-values."""
    from tests.conftest import make_random_panel
    rng = np.random.default_rng(99)
    n, m, P = 70, 12, 60
    samples = [f"S{i}" for i in range(n)]
    G = rng.integers(0, 3, size=(m, n)).astype(np.float32)
    gdf = pd.DataFrame(G, index=[f"v{i}" for i in range(m)], columns=samples)
    vdf = pd.DataFrame({"chrom": "chr1", "pos": 100 + 10 * np.arange(m)}, index=gdf.index)
    panel, _ = make_random_panel(vdf, samples, seed=3)
    Y = rng.normal(size=(P, n)).astype(np.float32)
    pdf = pd.DataFrame(Y, index=[f"g{i}" for i in range(P)], columns=samples)
    pos = pd.DataFrame({"chr": "chr1", "start": 150, "end": 150}, index=pdf.index)
    out = map_permutations(gdf, vdf, pdf, pos, None, haplotypes=panel, ancestry_model="dosage",
                           nperm=200, device="cpu", seed=2, window=1000,
                           logger=SimpleLogger(verbose=False))
    assert len(out) == P
    for col in ("pval_perm", "pval_beta"):
        p = out[col].to_numpy(float)
        p = p[np.isfinite(p)]
        assert p.size >= P - 2
        assert 0.35 < p.mean() < 0.65, f"{col} mean {p.mean()}"
        assert np.mean(p < 0.05) < 0.20, f"{col} tail {np.mean(p < 0.05)}"
    assert np.corrcoef(out["pval_perm"], out["pval_beta"])[0, 1] > 0.9


def test_dosage_permutations_min_ancestry_ac_and_validation():
    d = _dosage_cohort()
    lg = SimpleLogger(verbose=False)
    kw = dict(haplotypes=d["panel"], ancestry_model="dosage", device="cpu", window=1000,
              logger=lg, nperm=50, seed=1)
    masked = map_permutations(d["genotype_df"], d["variant_df"], d["phenotype_df"],
                              d["phenotype_pos_df"], d["covariates_df"],
                              min_ancestry_ac=1e9, **kw)
    r = masked.iloc[0]
    assert np.isnan(r["slope_anc0"]) and np.isnan(r["slope_anc1"])
    assert np.isnan(r["z_diff_anc0_anc1"])
    assert np.isfinite(r["fstat_joint"]) and np.isfinite(r["pval_perm"])
    assert np.isfinite(r["af_anc0"])        # allele statistics are still reported
    with pytest.raises(ValueError, match="min_ancestry_ac"):
        map_permutations(d["genotype_df"], d["variant_df"], d["phenotype_df"],
                         d["phenotype_pos_df"], d["covariates_df"], haplotypes=d["panel"],
                         ancestry_model="main", min_ancestry_ac=5, nperm=10, device="cpu", logger=lg)
    with pytest.raises(ValueError, match="dosage"):
        map_permutations(d["genotype_df"], d["variant_df"], d["phenotype_df"],
                         d["phenotype_pos_df"], d["covariates_df"], haplotypes=d["panel"],
                         ancestry_model="dosage", group_s=pd.Series({"geneA": "grp"}),
                         nperm=10, device="cpu", logger=lg)
    with pytest.raises(ValueError, match="dosage"):
        map_permutations(d["genotype_df"], d["variant_df"], d["phenotype_df"],
                         d["phenotype_pos_df"], d["covariates_df"], haplotypes=d["panel"],
                         ancestry_model="dosage", nperm=10, device="cpu", logger=lg,
                         paired_covariate_df=pd.DataFrame(
                             d["rng"].normal(size=(1, len(d["samples"]))),
                             index=["geneA"], columns=d["samples"]))
    with pytest.raises(ValueError, match="requires haplotypes"):
        map_permutations(d["genotype_df"], d["variant_df"], d["phenotype_df"],
                         d["phenotype_pos_df"], d["covariates_df"], ancestry_model="dosage",
                         nperm=10, device="cpu", logger=lg)
