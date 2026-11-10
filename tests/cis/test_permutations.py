import numpy as np
import pandas as pd
import torch
import pytest

from src.localqtl.cis.permutations import POOLED_LEAD_COLUMNS, map_permutations
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
    assert r["pval_perm"] < 0.05
    # the dosage scan's phenotype-level product is the empirical p (0.4.5): no Beta is fitted
    assert r["pval_fit_method"] == "empirical" and np.isnan(r["pval_beta"]) and np.isnan(r["beta_shape1"])
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
                           nperm=200, device="cpu", seed=2, window=1000, pval_fit_method="beta",
                           logger=SimpleLogger(verbose=False))
    assert len(out) == P and (out["pval_fit_method"] == "beta").all()
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
                         ancestry_model="dosage", nperm=10, device="cpu", logger=lg,
                         paired_covariate_df=pd.DataFrame(
                             d["rng"].normal(size=(1, len(d["samples"]))),
                             index=["geneA"], columns=d["samples"]))
    with pytest.raises(ValueError, match="requires haplotypes"):
        map_permutations(d["genotype_df"], d["variant_df"], d["phenotype_df"],
                         d["phenotype_pos_df"], d["covariates_df"], ancestry_model="dosage",
                         nperm=10, device="cpu", logger=lg)


def _dosage_group_cohort(seed=11, n=80, m=10, effect=0.9):
    """Cohort whose three phenotypes form one group; only ``geneB`` has an effect."""
    from tests.conftest import make_random_panel
    rng = np.random.default_rng(seed)
    samples = [f"S{i}" for i in range(n)]
    G = rng.integers(0, 3, size=(m, n)).astype(np.float32)
    gdf = pd.DataFrame(G, index=[f"v{i}" for i in range(m)], columns=samples)
    vdf = pd.DataFrame({"chrom": "chr1", "pos": 100 + 10 * np.arange(m)}, index=gdf.index)
    panel, _ = make_random_panel(vdf, samples, seed=seed + 1)
    ids = ["geneA", "geneB", "geneC"]
    Y = rng.normal(size=(3, n)).astype(np.float32)
    Y[1] += effect * G[3]
    pdf = pd.DataFrame(Y, index=ids, columns=samples)
    pos = pd.DataFrame({"chr": "chr1", "start": 140, "end": 140}, index=ids)
    cov = pd.DataFrame(rng.normal(size=(n, 2)), index=samples, columns=["c1", "c2"])
    return dict(genotype_df=gdf, variant_df=vdf, phenotype_df=pdf, phenotype_pos_df=pos,
                covariates_df=cov, panel=panel, samples=samples,
                group_s=pd.Series({i: "grp" for i in ids}))


def test_dosage_permutations_grouped_pick_the_best_phenotype_of_the_group():
    d = _dosage_group_cohort()
    lg = SimpleLogger(verbose=False)
    kw = dict(haplotypes=d["panel"], ancestry_model="dosage", device="cpu", window=1000, logger=lg)
    perm = map_permutations(d["genotype_df"], d["variant_df"], d["phenotype_df"],
                            d["phenotype_pos_df"], d["covariates_df"],
                            group_s=d["group_s"], nperm=200, seed=1, **kw)
    assert len(perm) == 1
    r = perm.iloc[0]
    assert r["group_id"] == "grp" and int(r["group_size"]) == 3
    assert r["phenotype_id"] == "geneB" and r["variant_id"] == "v3"

    # the reported lead row is the nominal dosage fit of that phenotype/variant
    nom = map_nominal(d["genotype_df"], d["variant_df"], d["phenotype_df"],
                      d["phenotype_pos_df"], d["covariates_df"], out_dir=None, **kw)
    lead = nom[(nom["phenotype_id"] == "geneB") & (nom["variant_id"] == "v3")].iloc[0]
    np.testing.assert_allclose(r["pval_nominal"], lead["pval_joint"], rtol=1e-4)
    np.testing.assert_allclose(r["fstat_joint"], lead["fstat_joint"], rtol=1e-4)
    for k in (0, 1):
        np.testing.assert_allclose(r[f"slope_anc{k}"], lead[f"slope_anc{k}"], rtol=1e-4)
        np.testing.assert_allclose(r[f"af_anc{k}"], lead[f"af_anc{k}"], rtol=1e-5)
    assert r["pval_perm"] < 0.05 and r["pval_fit_method"] == "empirical" and np.isnan(r["pval_beta"])
    assert np.isnan(r["pval_true_dof"])
    dof = int(r["true_dof"])
    np.testing.assert_allclose(r["r2_nominal"], 2 * r["fstat_joint"] / (2 * r["fstat_joint"] + dof),
                               rtol=1e-5)
    # the group's permutation null is a maximum over three phenotypes, so it
    # cannot be more permissive than the same phenotype mapped on its own
    solo = map_permutations(d["genotype_df"], d["variant_df"], d["phenotype_df"].loc[["geneB"]],
                            d["phenotype_pos_df"].loc[["geneB"]], d["covariates_df"],
                            nperm=200, seed=1, **kw).iloc[0]
    np.testing.assert_allclose(r["fstat_joint"], solo["fstat_joint"], rtol=1e-4)
    assert r["pval_perm"] >= solo["pval_perm"]


def test_dosage_permutations_grouped_mask_and_column_schema():
    d = _dosage_group_cohort()
    from src.localqtl.cis.nominal import _dosage_columns
    masked = map_permutations(d["genotype_df"], d["variant_df"], d["phenotype_df"],
                              d["phenotype_pos_df"], d["covariates_df"],
                              group_s=d["group_s"], haplotypes=d["panel"],
                              ancestry_model="dosage", device="cpu", window=1000,
                              nperm=50, seed=1, min_ancestry_ac=1e9,
                              logger=SimpleLogger(verbose=False))
    r = masked.iloc[0]
    assert set(_dosage_columns(2)) <= set(masked.columns)
    assert {"group_id", "group_size"}.issubset(masked.columns)
    assert np.isnan(r["slope_anc0"]) and np.isnan(r["z_diff_anc0_anc1"])
    assert np.isfinite(r["fstat_joint"]) and np.isfinite(r["af_anc0"])


def test_dosage_permutations_grouped_are_calibrated_under_the_null():
    """Groups of null phenotypes still give ~uniform permutation p-values."""
    from tests.conftest import make_random_panel
    rng = np.random.default_rng(7)
    n, m, n_groups, per_group = 70, 12, 40, 3
    samples = [f"S{i}" for i in range(n)]
    G = rng.integers(0, 3, size=(m, n)).astype(np.float32)
    gdf = pd.DataFrame(G, index=[f"v{i}" for i in range(m)], columns=samples)
    vdf = pd.DataFrame({"chrom": "chr1", "pos": 100 + 10 * np.arange(m)}, index=gdf.index)
    panel, _ = make_random_panel(vdf, samples, seed=8)
    ids, groups = [], {}
    for g in range(n_groups):
        for j in range(per_group):
            pid = f"g{g}_{j}"
            ids.append(pid)
            groups[pid] = f"grp{g}"
    Y = rng.normal(size=(len(ids), n)).astype(np.float32)
    pdf = pd.DataFrame(Y, index=ids, columns=samples)
    pos = pd.DataFrame({"chr": "chr1", "start": 150, "end": 150}, index=ids)
    out = map_permutations(gdf, vdf, pdf, pos, None, haplotypes=panel,
                           ancestry_model="dosage", group_s=pd.Series(groups),
                           nperm=200, device="cpu", seed=5, window=1000, pval_fit_method="beta",
                           logger=SimpleLogger(verbose=False))
    assert len(out) == n_groups and (out["pval_fit_method"] == "beta").all()
    for col in ("pval_perm", "pval_beta"):
        p = out[col].to_numpy(float)
        p = p[np.isfinite(p)]
        assert p.size >= n_groups - 2
        assert 0.35 < p.mean() < 0.65, f"{col} mean {p.mean()}"
        assert np.mean(p < 0.05) < 0.20, f"{col} tail {np.mean(p < 0.05)}"


def test_dosage_perm_chunk_fmax_sanitises_like_the_observed_statistic():
    """An overflowing null F must win the maximum, a NaN must never win it."""
    from src.localqtl import regression_kernels as rk

    F = torch.tensor([[1.0, 2.0],
                      [float("inf"), float("nan")],
                      [3.0, float("-inf")]])
    saved = rk._dosage_fstat_block
    rk._dosage_fstat_block = lambda ctx, Y: F
    try:
        out = rk.dosage_perm_chunk_fmax({}, torch.zeros(1, 2))
    finally:
        rk._dosage_fstat_block = saved
    # column 0 overflowed: it must stay the largest value, not collapse to 0
    assert out[0].item() > 3.0
    # column 1 is NaN at the variant that would otherwise win: the finite 2.0 does
    assert out[1].item() == pytest.approx(2.0)


def test_dosage_permutation_row_reports_a_single_joint_f():
    """``pval_nominal`` is the lead's joint p-value, so it must equal ``pval_joint``."""
    d = _dosage_cohort()
    perm = map_permutations(
        d["genotype_df"], d["variant_df"], d["phenotype_df"], d["phenotype_pos_df"],
        d["covariates_df"], nperm=100, seed=3, haplotypes=d["panel"],
        ancestry_model="dosage", device="cpu", window=1000,
        logger=SimpleLogger(verbose=False),
    )
    r = perm.iloc[0]
    assert r["pval_nominal"] == pytest.approx(r["pval_joint"], rel=1e-12)
    dof = int(r["true_dof"])
    f_from_r2 = r["r2_nominal"] * dof / (2 * (1 - r["r2_nominal"]))
    assert f_from_r2 == pytest.approx(r["fstat_joint"], rel=1e-6)


def test_dosage_permutation_grouped_row_reports_a_single_joint_f():
    d = _dosage_group_cohort()
    perm = map_permutations(
        d["genotype_df"], d["variant_df"], d["phenotype_df"], d["phenotype_pos_df"],
        d["covariates_df"], group_s=d["group_s"], nperm=100, seed=3,
        haplotypes=d["panel"], ancestry_model="dosage", device="cpu", window=1000,
        logger=SimpleLogger(verbose=False),
    )
    r = perm.iloc[0]
    assert r["pval_nominal"] == pytest.approx(r["pval_joint"], rel=1e-12)


# pooled-lead selection (0.4.4): the lead is the main model's, the test stays the dosage joint F,
# and the permutation null selects the lead the same way
# ---------------------------------------------------------------------------
def _perm(d, **kw):
    base = dict(haplotypes=d["panel"], device="cpu", window=1000, logger=SimpleLogger(verbose=False),
                nperm=200, seed=1)
    base.update(kw)
    return map_permutations(d["genotype_df"], d["variant_df"], d["phenotype_df"], d["phenotype_pos_df"],
                            d["covariates_df"], **base)


def test_pooled_lead_is_the_main_models_lead_and_the_row_keeps_the_joint_test():
    """``lead_selection="pooled"``: the lead is the variant the main model picks, the row's test is the
    dosage model's joint F at that variant, and the pooled statistics are reported as information."""
    d = _dosage_cohort()
    pooled = _perm(d, ancestry_model="dosage", lead_selection="pooled")
    joint = _perm(d, ancestry_model="dosage")
    main = _perm(d, ancestry_model="main")
    nom = map_nominal(d["genotype_df"], d["variant_df"], d["phenotype_df"], d["phenotype_pos_df"],
                      d["covariates_df"], out_dir=None, haplotypes=d["panel"], ancestry_model="dosage",
                      device="cpu", window=1000, logger=SimpleLogger(verbose=False))
    r, m = pooled.iloc[0], main.iloc[0]
    assert r["variant_id"] == m["variant_id"]
    at = nom.set_index("variant_id").loc[r["variant_id"]]
    # the dosage model's own joint test at the pooled lead, never the pooled p-value
    assert r["pval_nominal"] == pytest.approx(r["pval_joint"], rel=1e-12)
    np.testing.assert_allclose(r["pval_nominal"], at["pval_joint"], rtol=1e-4)
    np.testing.assert_allclose(r["fstat_joint"], at["fstat_joint"], rtol=1e-4)
    for k in (0, 1):
        np.testing.assert_allclose(r[f"slope_anc{k}"], at[f"slope_anc{k}"], rtol=1e-4)
    # the pooled statistics of the lead are the main model's row
    np.testing.assert_allclose(r["slope_pooled"], m["slope"], rtol=1e-3)
    np.testing.assert_allclose(r["tstat_pooled"], m["tstat"], rtol=1e-3)
    np.testing.assert_allclose(r["pval_nominal_pooled"], m["pval_nominal"], rtol=1e-3)
    np.testing.assert_allclose(r["r2_pooled"], m["r2_nominal"], rtol=1e-3)
    assert set(POOLED_LEAD_COLUMNS) <= set(pooled.columns)
    assert not set(POOLED_LEAD_COLUMNS) & set(joint.columns)
    assert r["pval_perm"] < 0.05 and r["pval_fit_method"] == "empirical" and np.isnan(r["pval_beta"])
    assert np.isnan(r["pval_true_dof"]) and int(r["true_dof"]) == int(joint.iloc[0]["true_dof"])
    # the default is the joint-lead scan, unchanged
    pd.testing.assert_frame_equal(_perm(d, ancestry_model="dosage", lead_selection="joint"), joint)


def _null_cohort(seed, n=70, m=12, P=60, panel_seed=3):
    from tests.conftest import make_random_panel
    rng = np.random.default_rng(seed)
    samples = [f"S{i}" for i in range(n)]
    G = rng.integers(0, 3, size=(m, n)).astype(np.float32)
    gdf = pd.DataFrame(G, index=[f"v{i}" for i in range(m)], columns=samples)
    vdf = pd.DataFrame({"chrom": "chr1", "pos": 100 + 10 * np.arange(m)}, index=gdf.index)
    panel, _ = make_random_panel(vdf, samples, seed=panel_seed)
    Y = rng.normal(size=(P, n)).astype(np.float32)
    pdf = pd.DataFrame(Y, index=[f"g{i}" for i in range(P)], columns=samples)
    pos = pd.DataFrame({"chr": "chr1", "start": 150, "end": 150}, index=pdf.index)
    return gdf, vdf, pdf, pos, panel


def test_pooled_lead_permutations_are_calibrated_under_the_null():
    gdf, vdf, pdf, pos, panel = _null_cohort(99)
    out = map_permutations(gdf, vdf, pdf, pos, None, haplotypes=panel, ancestry_model="dosage",
                           lead_selection="pooled", nperm=200, device="cpu", seed=2, window=1000,
                           pval_fit_method="beta", logger=SimpleLogger(verbose=False))
    assert len(out) == len(pdf) and (out["pval_fit_method"] == "beta").all()
    for col in ("pval_perm", "pval_beta"):
        p = out[col].to_numpy(float)
        p = p[np.isfinite(p)]
        assert p.size >= len(pdf) - 2
        assert 0.35 < p.mean() < 0.65, f"{col} mean {p.mean()}"
        assert np.mean(p < 0.05) < 0.20, f"{col} tail {np.mean(p < 0.05)}"
    assert np.corrcoef(out["pval_perm"], out["pval_beta"])[0, 1] > 0.9


def test_pooled_lead_null_is_selection_aware_where_the_joint_lead_null_is_conservative():
    """At a lead the pooled statistic picked, the joint F compared with the maximum F over the window (the
    joint-lead null) is conservative; compared with the F at each permutation's own pooled lead it is uniform."""
    from src.localqtl import regression_kernels as rk
    from src.localqtl.cis.common import ancestry_dosage_design
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    n, m, P, nperm = 60, 10, 150, 400
    G = torch.tensor(rng.integers(0, 3, size=(m, n)), dtype=torch.float32)
    H = torch.tensor(rng.integers(0, 3, size=(m, n, 1)), dtype=torch.float32)
    D, L = ancestry_dosage_design(G, H)
    D = D - D.mean(1, keepdim=True)
    L = L - L.mean(1, keepdim=True)
    Gc = D.sum(2)
    ctx_d = rk.prep_ctx_for_dosage_perm(D, L, k_eff=1)
    naive, aware = [], []
    for _ in range(P):
        y = torch.randn(n)
        y = y - y.mean()
        f_obs = torch.nan_to_num(rk.dosage_fstat(ctx_d, y), nan=-1.0)
        ctx_p, _, _, tt = rk.prep_ctx_for_perm(y, Gc, L, 1)
        ix = int(tt[:, 0].abs().argmax())
        f_at = float(f_obs[ix])
        perm = y[torch.argsort(torch.rand(n, nperm), dim=0)]          # (n, nperm) independent permutations
        f_max = rk.dosage_perm_chunk_fmax(ctx_d, perm)
        _, ix_p = rk.perm_chunk_r2_argmax(ctx_p, L, Gc, perm)
        f_sel = rk.dosage_perm_chunk_f_at(ctx_d, perm, ix_p)
        naive.append((float((f_max >= f_at).sum()) + 1) / (nperm + 1))
        aware.append((float((f_sel >= f_at).sum()) + 1) / (nperm + 1))
    naive, aware = np.array(naive), np.array(aware)
    assert naive.mean() > 0.58, naive.mean()
    assert 0.44 < aware.mean() < 0.56, aware.mean()
    assert np.mean(aware < 0.05) < 0.10
    assert np.mean(naive < 0.05) <= np.mean(aware < 0.05)


def test_pooled_lead_grouped_picks_the_main_models_group_lead():
    d = _dosage_group_cohort()
    kw = dict(group_s=d["group_s"], nperm=100, seed=3)
    pooled = _perm(d, ancestry_model="dosage", lead_selection="pooled", **kw)
    main = _perm(d, ancestry_model="main", **kw)
    assert len(pooled) == len(main) == 1
    r, m = pooled.iloc[0], main.iloc[0]
    assert (r["phenotype_id"], r["variant_id"]) == (m["phenotype_id"], m["variant_id"]) == ("geneB", "v3")
    assert r["pval_nominal"] == pytest.approx(r["pval_joint"], rel=1e-12)
    np.testing.assert_allclose(r["tstat_pooled"], m["tstat"], rtol=1e-3)
    assert r["group_size"] == 3 and r["pval_perm"] < 0.05
    assert set(POOLED_LEAD_COLUMNS) <= set(pooled.columns)


def test_pooled_lead_grouped_permutations_are_calibrated_under_the_null():
    from tests.conftest import make_random_panel
    rng = np.random.default_rng(7)
    n, m, n_groups, per_group = 70, 12, 40, 3
    samples = [f"S{i}" for i in range(n)]
    G = rng.integers(0, 3, size=(m, n)).astype(np.float32)
    gdf = pd.DataFrame(G, index=[f"v{i}" for i in range(m)], columns=samples)
    vdf = pd.DataFrame({"chrom": "chr1", "pos": 100 + 10 * np.arange(m)}, index=gdf.index)
    panel, _ = make_random_panel(vdf, samples, seed=8)
    ids, groups = [], {}
    for g in range(n_groups):
        for j in range(per_group):
            pid = f"g{g}_{j}"
            ids.append(pid)
            groups[pid] = f"grp{g}"
    Y = rng.normal(size=(len(ids), n)).astype(np.float32)
    pdf = pd.DataFrame(Y, index=ids, columns=samples)
    pos = pd.DataFrame({"chr": "chr1", "start": 150, "end": 150}, index=ids)
    out = map_permutations(gdf, vdf, pdf, pos, None, haplotypes=panel,
                           ancestry_model="dosage", lead_selection="pooled", group_s=pd.Series(groups),
                           nperm=200, device="cpu", seed=5, window=1000, pval_fit_method="beta",
                           logger=SimpleLogger(verbose=False))
    assert len(out) == n_groups and (out["pval_fit_method"] == "beta").all()
    for col in ("pval_perm", "pval_beta"):
        p = out[col].to_numpy(float)
        p = p[np.isfinite(p)]
        assert p.size >= n_groups - 2
        assert 0.35 < p.mean() < 0.65, f"{col} mean {p.mean()}"
        assert np.mean(p < 0.05) < 0.20, f"{col} tail {np.mean(p < 0.05)}"


def test_pooled_lead_with_ld_aware_reports_the_contrast_at_the_pooled_lead():
    d = _dosage_cohort()
    plain = _perm(d, ancestry_model="dosage", lead_selection="pooled")
    ld = _perm(d, ancestry_model="dosage", lead_selection="pooled", ld_aware=True, ld_aware_opts={"L": 3})
    assert ld.iloc[0]["variant_id"] == plain.iloc[0]["variant_id"]
    assert {"pval_diff_ld_anc0_anc1", "n_conditioned_ld", "cs_size_ld", "pip_ld"} <= set(ld.columns)
    assert 0 <= ld.iloc[0]["pval_diff_ld_anc0_anc1"] <= 1
    # the LD-aware pass runs after the selection and does not touch the test columns
    for col in ("pval_perm", "pval_beta", "pval_nominal", "tstat_pooled", "pval_nominal_pooled"):
        assert ld.iloc[0][col] == pytest.approx(plain.iloc[0][col], nan_ok=True)


def test_pooled_lead_validation():
    d = _dosage_cohort()
    with pytest.raises(ValueError, match="lead_selection"):
        _perm(d, ancestry_model="main", lead_selection="pooled")
    with pytest.raises(ValueError, match="lead_selection"):
        _perm(d, ancestry_model="dosage", lead_selection="best")


def test_dosage_perm_chunk_f_at_gathers_and_sanitises_like_the_observed_statistic():
    from src.localqtl import regression_kernels as rk

    F = torch.tensor([[1.0, 2.0],
                      [float("inf"), float("nan")],
                      [3.0, float("-inf")]])
    saved = rk._dosage_fstat_block
    rk._dosage_fstat_block = lambda ctx, Y: F
    try:
        at_1 = rk.dosage_perm_chunk_f_at({}, torch.zeros(1, 2), torch.tensor([1, 1]))
        at_20 = rk.dosage_perm_chunk_f_at({}, torch.zeros(1, 2), torch.tensor([2, 0]))
    finally:
        rk._dosage_fstat_block = saved
    # an overflowing null F stays more extreme than any observation; a NaN never counts
    assert at_1[0].item() > 3.0 and at_1[1].item() == -1.0
    assert at_20.tolist() == [3.0, 2.0]


def test_perm_chunk_r2_argmax_matches_perm_chunk_r2():
    from src.localqtl import regression_kernels as rk
    torch.manual_seed(1)
    m, n, q, c = 7, 40, 1, 5
    G, H, y, yp = torch.randn(m, n), torch.randn(m, n, q), torch.randn(n), torch.randn(n, c)
    ctx, *_ = rk.prep_ctx_for_perm(y, G, H, k_eff=1)
    r2_max, ix = rk.perm_chunk_r2_argmax(ctx, H, G, yp)
    torch.testing.assert_close(r2_max, rk.perm_chunk_r2(ctx, H, G, yp))
    assert ix.tolist() == rk._perm_chunk_r2_block(ctx, H, G, yp).argmax(dim=0).tolist()


# the observed scan and its permutation null are one procedure (review of 2026-11-09,
# docs/reviews/2026-11-09_permutation_inference.md): selection statistic, covariate adjustment,
# tie rule and test statistic, checked on the identity permutation and on exact proxies
# ---------------------------------------------------------------------------
class _FixedPermutations:
    """Stands in for ``PermutationStream``: yields the given index rows, ``(chunk, n)``."""

    def __init__(self, sel: torch.Tensor):
        self.sel = sel

    def iter_chunks(self, limit=None):
        yield self.sel if limit is None else self.sel[:limit]


def _residualised_dosage_design(seed=0, n=90, m=14, k=5, proxy=None, effect_at=None):
    """``(y, D, L, rez)`` as the permutation core builds them (``residualize_batch_dosage``): random genotypes,
    a collapsed two-ancestry panel and ``k`` covariates that explain part of ``y``.  ``proxy=(i, j)`` makes
    variant ``j`` an exact copy of variant ``i`` (genotype and local ancestry), a perfect-LD proxy on the same
    ancestry background; ``effect_at`` puts an effect on that variant."""
    from src.localqtl.cis.common import residualize_batch_dosage
    from src.localqtl.regression_kernels import Residualizer
    rng = np.random.default_rng(seed)
    G = rng.integers(0, 3, size=(m, n)).astype(np.float32)
    H = rng.integers(0, 3, size=(m, n, 1)).astype(np.float32)
    if proxy is not None:
        i, j = proxy
        G[j], H[j] = G[i], H[i]
    C = rng.normal(size=(n, k)).astype(np.float32)
    y = C @ rng.normal(size=k) + rng.normal(size=n)
    if effect_at is not None:
        y = y + 1.0 * G[effect_at]
    rez = Residualizer(torch.tensor(C))
    y_res, D, L = residualize_batch_dosage(torch.tensor(y.astype(np.float32)), torch.tensor(G), torch.tensor(H),
                                           rez, center=True, group=False)
    return y_res, D, L, rez


def test_pooled_lead_null_on_the_identity_permutation_is_the_observed_scan():
    """The null of ``lead_selection="pooled"`` applied to the unpermuted phenotype returns the observed scan's
    own numbers: the same pooled R^2 at every variant, the same lead, the same joint F at it.  One residualised
    design, one selection statistic, one test statistic, re-residualisation included."""
    from src.localqtl import regression_kernels as rk
    from src.localqtl.cis._permute import projectors_of
    from src.localqtl.cis.permutations import _pooled_lead_null, _pooled_lead_observed
    y, D, L, rez = _residualised_dosage_design(seed=3)
    n = int(y.shape[0])
    G_sum = D.sum(dim=2)
    ctx = rk.prep_ctx_for_dosage_perm(D, L, k_eff=rez.k_eff)
    f_obs = torch.nan_to_num(rk.dosage_fstat(ctx, y), nan=-1.0)
    ctx_p, r2_obs, *_ = _pooled_lead_observed(y, G_sum, L, rez.k_eff, None)
    ix_obs = int(r2_obs.argmax())
    # the selection statistic, variant by variant
    r2_null = rk._perm_chunk_r2_block(ctx_p, L, G_sum, y.reshape(-1, 1))[:, 0]
    torch.testing.assert_close(r2_null, r2_obs, rtol=1e-4, atol=1e-6)
    assert int(r2_null.argmax()) == ix_obs
    # the whole null path on the identity permutation
    identity = _FixedPermutations(torch.arange(n).unsqueeze(0))
    r2_sel, f_sel, f_win = _pooled_lead_null(ctx, ctx_p, L, G_sum, y, None, identity, 1, n, projectors_of(rez), None)
    assert tuple(r2_sel.shape) == tuple(f_sel.shape) == tuple(f_win.shape) == (1,)
    assert r2_sel.item() == pytest.approx(r2_obs[ix_obs].item(), rel=1e-4)
    assert f_sel.item() == pytest.approx(f_obs[ix_obs].item(), rel=1e-4)
    # the window maximum of the same block is the observed scan's maximum F, at least the F at the pooled lead
    assert f_win.item() == pytest.approx(torch.nan_to_num(f_obs, nan=-1.0).max().item(), rel=1e-4) and f_win.item() >= f_sel.item()


def test_pooled_lead_ties_follow_one_rule_and_cannot_move_the_null_statistic():
    """Ties.  Both paths return the first maximal index (``argmax`` for the observed scan, ``max(dim=0)`` for
    the null).  An exact tie in pooled R^2 needs the same genotype and the same local ancestry, hence the same
    ancestry dosages and the same joint F, so the statistic read at a tied lead does not depend on which member
    of the tie is taken, in the observed scan or in any permutation."""
    from src.localqtl import regression_kernels as rk
    from src.localqtl.cis._permute import freedman_lane, projectors_of
    from src.localqtl.cis.permutations import _pooled_lead_observed
    r2 = torch.tensor([0.1, 0.7, 0.7, 0.3])
    assert int(r2.argmax()) == 1
    assert r2.unsqueeze(1).expand(-1, 3).contiguous().max(dim=0).indices.tolist() == [1, 1, 1]

    i, j = 4, 9
    y, D, L, rez = _residualised_dosage_design(seed=5, proxy=(i, j), effect_at=i)
    n = int(y.shape[0])
    G_sum = D.sum(dim=2)
    ctx = rk.prep_ctx_for_dosage_perm(D, L, k_eff=rez.k_eff)
    ctx_p, r2_obs, *_ = _pooled_lead_observed(y, G_sum, L, rez.k_eff, None)
    f_obs = rk.dosage_fstat(ctx, y)
    assert int(r2_obs.argmax()) in (i, j)                       # the proxy pair is the observed lead
    assert r2_obs[i].item() == pytest.approx(r2_obs[j].item(), rel=1e-5)
    assert f_obs[i].item() == pytest.approx(f_obs[j].item(), rel=1e-4)

    g = torch.Generator().manual_seed(11)
    sel = torch.stack([torch.randperm(n, generator=g) for _ in range(256)])
    y_perm = freedman_lane(y[sel].T.contiguous(), projectors_of(rez))            # (n, 256)
    R2 = rk._perm_chunk_r2_block(ctx_p, L, G_sum, y_perm)
    F = rk._dosage_fstat_block(ctx, y_perm)
    torch.testing.assert_close(R2[i], R2[j], rtol=1e-4, atol=1e-7)
    torch.testing.assert_close(F[i], F[j], rtol=1e-3, atol=1e-5)
    _, ix = rk.perm_chunk_r2_argmax(ctx_p, L, G_sum, y_perm)
    assert ((ix == i) | (ix == j)).any()                        # some permutations select the tied pair
    other = torch.where(ix == i, torch.full_like(ix, j), torch.where(ix == j, torch.full_like(ix, i), ix))
    torch.testing.assert_close(rk.dosage_perm_chunk_f_at(ctx, y_perm, ix),
                               rk.dosage_perm_chunk_f_at(ctx, y_perm, other), rtol=1e-3, atol=1e-5)


@pytest.mark.parametrize("with_local_ancestry", [True, False])
def test_main_model_observed_r2_is_the_null_statistic(with_local_ancestry):
    """``ancestry_model="main"`` and ``tensorqtl_flavor`` compare an observed R^2 from the ridge-regularised
    OLS t statistic (``run_batch_regression``) with null maxima from the partial-correlation kernels.  Two code
    paths, one statistic: on the same vector they agree far inside the resolution of ``pval_perm``."""
    from src.localqtl import regression_kernels as rk
    from src.localqtl.cis.common import residualize_batch
    rng = np.random.default_rng(8)
    n, m, k = 90, 14, 5
    G = torch.tensor(rng.integers(0, 3, size=(m, n)).astype(np.float32))
    H = torch.tensor(rng.integers(0, 3, size=(m, n, 1)).astype(np.float32)) if with_local_ancestry else None
    C = rng.normal(size=(n, k)).astype(np.float32)
    y = torch.tensor((C @ rng.normal(size=k) + rng.normal(size=n)).astype(np.float32))
    rez = rk.Residualizer(torch.tensor(C))
    y_res, G_res, H_res = residualize_batch(y, G, H, rez, center=True)
    _, _, tstats = rk.run_batch_regression(y=y_res, G=G_res, H=H_res, k_eff=rez.k_eff, device="cpu")
    dof = n - rez.k_eff - (1 + (0 if H_res is None else int(H_res.shape[2])))
    t2 = tstats[:, 0].double().pow(2)
    r2_obs = (t2 / (t2 + dof)).to(torch.float32)
    y_col = y_res.reshape(-1, 1)
    if H_res is None:
        *_, r2_null_max = rk.run_batch_regression_with_permutations(
            y=y_res, G=G_res, H=None, y_perm=y_col, k_eff=rez.k_eff, device="cpu")
    else:
        ctx, *_ = rk.prep_ctx_for_perm(y_res, G_res, H_res, rez.k_eff)
        r2_null = rk._perm_chunk_r2_block(ctx, H_res, G_res, y_col)[:, 0]
        torch.testing.assert_close(r2_null, r2_obs, rtol=1e-3, atol=1e-6)
        assert int(r2_null.argmax()) == int(r2_obs.argmax())
        r2_null_max = rk.perm_chunk_r2(ctx, H_res, G_res, y_col)
    assert r2_null_max.item() == pytest.approx(r2_obs.max().item(), rel=1e-3)


@pytest.mark.parametrize("effect", [0.8, -0.8])
def test_pooled_lead_reports_the_main_models_standard_error_for_either_sign(effect):
    """``slope_pooled``, ``slope_se_pooled`` and ``tstat_pooled`` are one consistent triple and the main
    model's row, for a negative pooled effect as for a positive one."""
    d = _dosage_cohort(effect=effect)
    r = _perm(d, ancestry_model="dosage", lead_selection="pooled").iloc[0]
    m = _perm(d, ancestry_model="main").iloc[0]
    assert r["variant_id"] == m["variant_id"] == "v3" and np.sign(r["slope_pooled"]) == np.sign(effect)
    np.testing.assert_allclose(r["slope_pooled"], m["slope"], rtol=1e-3)
    np.testing.assert_allclose(r["slope_se_pooled"], m["slope_se"], rtol=1e-3)
    np.testing.assert_allclose(r["slope_pooled"] / r["slope_se_pooled"], r["tstat_pooled"], rtol=1e-4)


@pytest.mark.parametrize("effect", [1.0, -1.0])
def test_interaction_scans_report_a_consistent_standard_error_for_either_sign(effect):
    """The g x local-ancestry and the g x covariate permutation scans report ``slope_se`` with
    ``slope / slope_se == tstat`` whatever the sign of the interaction effect."""
    d = _dosage_cohort(seed=9, n=120)
    rng = np.random.default_rng(1)
    h0 = d["admix"][3, :, 0].astype(np.float64)
    cov = d["covariates_df"]
    cz = (cov["c2"] - cov["c2"].mean()) / cov["c2"].std(ddof=0)
    for kind, term in (("ancestry", d["G"][3] * h0), ("covariate", d["G"][3] * cz.to_numpy())):
        y = effect * term + rng.normal(size=len(h0))
        pdf = pd.DataFrame([y], index=["geneA"], columns=d["samples"]).astype(np.float32)
        kw = dict(device="cpu", window=1000, logger=SimpleLogger(verbose=False), nperm=50, seed=1)
        if kind == "ancestry":
            kw.update(haplotypes=d["panel"], ancestry_model="interaction")
        else:
            kw.update(interaction_covariate="c2")
        r = map_permutations(d["genotype_df"], d["variant_df"], pdf, d["phenotype_pos_df"], cov, **kw).iloc[0]
        assert r["variant_id"] == "v3" and np.sign(r["slope"]) == np.sign(effect), kind
        assert 0 < r["slope_se"] < 1, kind
        np.testing.assert_allclose(r["slope"] / r["slope_se"], r["tstat"], rtol=1e-4, err_msg=kind)
        if kind == "ancestry":
            np.testing.assert_allclose(r["slope_gxh_anc0"] / r["slope_se_gxh_anc0"], r["tstat_gxh_anc0"], rtol=1e-4)


@pytest.mark.parametrize("with_panel", [False, True])
def test_grouped_scan_reports_a_consistent_standard_error_for_a_negative_effect(with_panel):
    """The grouped scan takes its lead's slope, standard error and t from the permutation kernels:
    ``slope / slope_se == tstat`` for a negative effect, and the row is the ungrouped scan's row."""
    d = _dosage_group_cohort(effect=-0.9)
    kw = dict(device="cpu", window=1000, logger=SimpleLogger(verbose=False), nperm=100, seed=1)
    if with_panel:
        kw.update(haplotypes=d["panel"], ancestry_model="main")
    args = (d["genotype_df"], d["variant_df"], d["phenotype_df"], d["phenotype_pos_df"], d["covariates_df"])
    r = map_permutations(*args, group_s=d["group_s"], **kw).iloc[0]
    u = map_permutations(*args, **kw).set_index("phenotype_id").loc["geneB"]
    assert r["phenotype_id"] == "geneB" and r["variant_id"] == "v3" and r["slope"] < 0
    np.testing.assert_allclose(r["slope"] / r["slope_se"], r["tstat"], rtol=1e-4)
    np.testing.assert_allclose(r["slope_se"], u["slope_se"], rtol=1e-3)
