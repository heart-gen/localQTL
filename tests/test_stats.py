"""localqtl.stats: q-values from permutation results."""
import numpy as np
import pandas as pd
import pytest

from src.localqtl.stats import calculate_qvalues
from src.localqtl.utils import SimpleLogger


def _perm_table(n=400, n_signal=40, seed=0):
    rng = np.random.default_rng(seed)
    p = rng.uniform(size=n)
    p[:n_signal] = rng.uniform(0, 1e-4, size=n_signal)
    return pd.DataFrame({"phenotype_id": [f"g{i}" for i in range(n)], "pval_perm": p, "pval_beta": p.copy(),
                         "beta_shape1": 1.0, "beta_shape2": 50.0})


def test_calculate_qvalues_uses_pval_perm_where_the_beta_fit_failed():
    """A degenerate beta fit leaves ``pval_beta`` NaN (``_beta_mle_on_p``); that phenotype is tested with its
    empirical permutation p-value, not set to p = 1 and dropped from the discoveries."""
    lg = SimpleLogger(verbose=False)
    df = _perm_table()
    ref = calculate_qvalues(df.copy(), logger=lg)
    broken = df.copy()
    broken.loc[[0, 100], ["pval_beta", "beta_shape1", "beta_shape2"]] = np.nan     # one signal, one null phenotype
    out = calculate_qvalues(broken, logger=lg)
    # pval_perm equals pval_beta in this table, so the fallback reproduces the complete table's q-values
    np.testing.assert_allclose(out["qval"].to_numpy(), ref["qval"].to_numpy(), rtol=1e-12)
    assert out.loc[0, "qval"] <= 0.05 < out.loc[100, "qval"]
    # the stored p-values are untouched and the phenotype without beta shapes has no nominal threshold
    assert np.isnan(out.loc[0, "pval_beta"]) and np.isnan(out.loc[0, "pval_nominal_threshold"])
    assert np.isfinite(out.loc[1, "pval_nominal_threshold"])


def test_calculate_qvalues_without_pval_perm_keeps_the_conservative_fill():
    """With no ``pval_perm`` column to fall back on, a NaN ``pval_beta`` is treated as p = 1 (unchanged)."""
    df = _perm_table().drop(columns="pval_perm")
    df.loc[0, "pval_beta"] = np.nan
    out = calculate_qvalues(df, logger=SimpleLogger(verbose=False))
    assert out.loc[0, "qval"] > 0.05
    assert (out["qval"].iloc[1:40] <= 0.05).all()


# --------------------------------------------------------------------------- empirical null summaries (0.4.5)
from scipy import stats as _st

from src.localqtl.stats import (NULL_SUMMARY_LEVELS, NULL_SUMMARY_TOP, empirical_nominal_threshold,
                                summarise_null_p)


def test_summarise_null_p_keeps_the_smallest_values_exactly_and_the_quantiles_at_the_levels():
    rng = np.random.default_rng(1)
    p = rng.beta(1, 300, size=10_000)
    top, lv = summarise_null_p(p)
    srt = np.sort(p)
    assert top.shape == (NULL_SUMMARY_TOP,) and lv.shape == NULL_SUMMARY_LEVELS.shape
    np.testing.assert_allclose(top, srt[:NULL_SUMMARY_TOP].astype(np.float32))
    k = np.clip(np.ceil(NULL_SUMMARY_LEVELS * p.size).astype(int), 1, p.size) - 1
    np.testing.assert_allclose(lv, srt[k].astype(np.float32))
    assert np.all(np.diff(top) >= 0) and np.all(np.diff(lv) >= 0) and lv[-1] == np.float32(srt[-1])
    # a non-finite permutation statistic counts as the least extreme draw
    top2, _ = summarise_null_p(np.r_[p[:5], np.nan, np.inf], top=7)
    assert top2[-1] == 1.0 and top2[-2] == 1.0


def test_empirical_nominal_threshold_is_the_k_th_smallest_null_value_and_interpolates_beyond_the_top():
    rng = np.random.default_rng(2)
    n = 10_000
    p = np.sort(rng.beta(1, 300, size=n))
    top, lv = summarise_null_p(p)
    # exact within the stored top: k = floor(p* (n + 1))
    for p_star in (1.5e-4, 1e-3, 0.01, 0.0249):
        k = int(np.floor(p_star * (n + 1)))
        assert empirical_nominal_threshold(top, lv, n, p_star) == np.float32(p[k - 1])
    # beyond the top the log-interpolated quantile is within a few percent of the true order statistic
    for p_star in (0.05, 0.1, 0.3, 0.7):
        k = int(np.floor(p_star * (n + 1)))
        t = empirical_nominal_threshold(top, lv, n, p_star)
        assert abs(np.log(t / p[k - 1])) < 0.05, (p_star, t, p[k - 1])
    # a variant at the threshold has an empirical p of (k + 1) / (n + 1) against this null: p* to the resolution
    p_star = 0.004
    t = empirical_nominal_threshold(top, lv, n, p_star)
    assert ((p <= t).sum() + 1) / (n + 1) == pytest.approx(p_star, abs=1.5 / (n + 1))
    # below the resolution of the permutations there is no threshold; at or above 1 it is the largest null value
    assert np.isnan(empirical_nominal_threshold(top, lv, n, 0.5 / (n + 1)))
    assert empirical_nominal_threshold(top, lv, n, 1.0) == np.float32(p[-1])
    assert np.isnan(empirical_nominal_threshold(top, lv, 0, 0.01)) and np.isnan(empirical_nominal_threshold(top, lv, n, np.nan))


def _null_summary(ids, nperm, seed=0, shape2=300.0):
    rng = np.random.default_rng(seed)
    rows = []
    for pid in ids:
        p_max = rng.beta(1, shape2, size=nperm)
        top, lv = summarise_null_p(p_max)
        rows.append({"phenotype_id": pid, "group_id": None, "nperm": nperm, "K": 2, "dof": 100,
                     "levels": NULL_SUMMARY_LEVELS.astype(np.float32), "null_lead_p_top": top, "null_lead_p_levels": lv,
                     "null_max_p_top": top, "null_max_p_levels": lv})
    return pd.DataFrame(rows)


def test_calculate_qvalues_reads_the_fit_method_per_row_and_the_empirical_threshold_from_the_null_summary():
    """Two products in one table: Beta rows keep the Beta quantile, empirical rows take their q-values from
    pval_perm and their nominal threshold from the stored window-minimum null at the shared cutoff p*."""
    lg = SimpleLogger(verbose=False)
    df = _perm_table()
    df["pval_fit_method"] = "beta"
    emp = df.index[200:]                                       # the second half becomes empirical
    df.loc[emp, "pval_fit_method"] = "empirical"
    df.loc[emp, ["pval_beta", "beta_shape1", "beta_shape2"]] = np.nan
    df.loc[emp[:20], "pval_perm"] = np.linspace(1e-4, 3e-4, 20)          # 20 empirical signals
    summary = _null_summary(df["phenotype_id"], nperm=10_000)
    ref = calculate_qvalues(df.copy(), logger=lg)              # no summary: the empirical rows have no threshold
    assert np.isnan(ref.loc[emp, "pval_nominal_threshold"]).all() and np.isfinite(ref.loc[:39, "pval_nominal_threshold"]).all()
    out = calculate_qvalues(df.copy(), logger=lg, null_summary=summary)
    # q-values use pval_beta on the Beta rows and pval_perm on the empirical rows: the 20 planted empirical signals pass
    np.testing.assert_allclose(out["qval"], ref["qval"])
    assert (out.loc[emp[:20], "qval"] <= 0.05).all()
    assert (out.loc[emp[40:], "qval"][out.loc[emp[40:], "pval_perm"] > 0.1] > 0.05).all()
    # the shared cutoff p*: between the largest significant and the smallest non-significant phenotype-level p
    p_level = np.where(out["pval_fit_method"] == "beta", out["pval_beta"], out["pval_perm"])
    sig = out["qval"].to_numpy() <= 0.05
    p_star = 0.5 * (p_level[sig].max() + p_level[~sig].min())
    # Beta rows: beta.ppf(p*, a, b); empirical rows: the k-th smallest stored null value
    np.testing.assert_allclose(out.loc[:39, "pval_nominal_threshold"], _st.beta.ppf(p_star, 1.0, 50.0), rtol=1e-6)
    s = summary.set_index("phenotype_id")
    for pid in out.loc[emp[:3], "phenotype_id"]:
        row = s.loc[pid]
        assert out.set_index("phenotype_id").loc[pid, "pval_nominal_threshold"] == pytest.approx(
            empirical_nominal_threshold(row["null_max_p_top"], row["null_max_p_levels"], 10_000, p_star))
    # the summary table itself is parquet-writable (list columns) and reads back to the same thresholds
    import tempfile, os
    with tempfile.TemporaryDirectory() as d:
        summary.to_parquet(os.path.join(d, "ns.parquet"), index=False)
        back = pd.read_parquet(os.path.join(d, "ns.parquet"))
    out2 = calculate_qvalues(df.copy(), logger=lg, null_summary=back)
    np.testing.assert_allclose(out2["pval_nominal_threshold"].to_numpy(), out["pval_nominal_threshold"].to_numpy())


def test_calculate_qvalues_infers_the_method_of_tables_written_before_0_4_5():
    """No ``pval_fit_method`` column: a finite ``pval_beta`` is the Beta product, NaN the empirical one."""
    lg = SimpleLogger(verbose=False)
    df = _perm_table()
    ref = calculate_qvalues(df.copy(), logger=lg)
    df2 = df.copy()
    df2["pval_fit_method"] = "beta"
    out = calculate_qvalues(df2, logger=lg)
    np.testing.assert_allclose(out["qval"], ref["qval"])
    np.testing.assert_allclose(out["pval_nominal_threshold"], ref["pval_nominal_threshold"])
    legacy_empirical = df.drop(columns=["pval_beta", "beta_shape1", "beta_shape2"])
    out3 = calculate_qvalues(legacy_empirical, logger=lg)
    assert (out3["qval"].iloc[:40] <= 0.05).all() and np.isnan(out3["pval_nominal_threshold"]).all()


def test_calculate_qvalues_accepts_a_single_lambda():
    """``qvalue_lambda=0.5`` (the CLI's ``--qvalue_lambda``) is Storey's pi0 at that lambda; before 0.4.5 the scalar
    reached py_qvalue's grid-only estimator and raised."""
    lg = SimpleLogger(verbose=False)
    df = _perm_table()
    out = calculate_qvalues(df.copy(), logger=lg, qvalue_lambda=0.5)
    p = df["pval_beta"].to_numpy()
    pi0 = min(1.0, np.mean(p >= 0.5) / 0.5)
    from py_qvalue import qvalue
    np.testing.assert_allclose(out["qval"].to_numpy(), qvalue(p, pi0=pi0)["qvalues"])
    grid = calculate_qvalues(df.copy(), logger=lg, qvalue_lambda=[0.4, 0.5, 0.6, 0.7, 0.8])
    assert np.isfinite(grid["qval"]).all()
    with pytest.raises(ValueError, match="qvalue_lambda"):
        calculate_qvalues(df.copy(), logger=lg, qvalue_lambda=1.0)
