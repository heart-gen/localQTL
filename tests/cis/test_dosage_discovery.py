"""The dosage scan's two inference products end to end (0.4.5): phenotype-level discovery from the selected-lead
permutation null (pval_perm, the empirical reference; no Beta), and variant-level extraction of joint associations
from the window-maximum null (pval_nominal_threshold read from the stored null summary, pairs compared on pval_joint)."""
import numpy as np
import pandas as pd
import pytest

from src.localqtl.cis.nominal import map_nominal
from src.localqtl.cis.permutations import map_permutations
from src.localqtl.cis.postproc import get_significant_pairs, pair_pval_column_of
from src.localqtl.stats import NULL_SUMMARY_LEVELS, calculate_qvalues, empirical_nominal_threshold, null_summary_of
from src.localqtl.utils import SimpleLogger

LG = SimpleLogger(verbose=False)


def _cohort(seed=11, n=120, m=12, n_signal=4, n_null=36, effect=0.9):
    """Twelve variants in LD blocks of three; the first ``n_signal`` phenotypes have an effect on the lead variant of
    their block, the rest are pure noise.  Returns the frames map_nominal / map_permutations take."""
    from tests.conftest import make_random_panel
    rng = np.random.default_rng(seed)
    samples = [f"S{i}" for i in range(n)]
    hap = rng.integers(0, 2, size=(m // 3, n, 2))                    # one haplotype pair per block
    G = np.empty((m, n), dtype=np.float32)
    for b in range(m // 3):
        g = hap[b].sum(-1).astype(np.float32)
        for j in range(3):                                           # three variants in LD within the block
            flip = rng.random(n) < 0.12 * j
            G[3 * b + j] = np.where(flip, 2 - g, g)
    gdf = pd.DataFrame(G, index=[f"v{i}" for i in range(m)], columns=samples)
    vdf = pd.DataFrame({"chrom": "chr1", "pos": 100 + 10 * np.arange(m)}, index=gdf.index)
    panel, _ = make_random_panel(vdf, samples, seed=seed + 1)
    ids = [f"sig{i}" for i in range(n_signal)] + [f"null{i}" for i in range(n_null)]
    Y = rng.normal(size=(len(ids), n)).astype(np.float32)
    for i in range(n_signal):
        Y[i] += effect * G[3 * (i % (m // 3))]
    pdf = pd.DataFrame(Y, index=ids, columns=samples)
    pos = pd.DataFrame({"chr": "chr1", "start": 150, "end": 150}, index=ids)
    cov = pd.DataFrame(rng.normal(size=(n, 2)), index=samples, columns=["c1", "c2"])
    return dict(genotype_df=gdf, variant_df=vdf, phenotype_df=pdf, phenotype_pos_df=pos, covariates_df=cov,
                panel=panel, ids=ids, n_signal=n_signal)


def _scan(d, tmp_path, lead_selection, nperm=200, seed=3):
    kw = dict(haplotypes=d["panel"], ancestry_model="dosage", device="cpu", window=1000, logger=LG)
    map_nominal(d["genotype_df"], d["variant_df"], d["phenotype_df"], d["phenotype_pos_df"], d["covariates_df"],
                out_dir=str(tmp_path), out_prefix="nom", write_stats=True, **kw)
    perm = map_permutations(d["genotype_df"], d["variant_df"], d["phenotype_df"], d["phenotype_pos_df"],
                            d["covariates_df"], nperm=nperm, seed=seed, lead_selection=lead_selection, **kw)
    return perm, str(tmp_path / "nom.chr1.parquet")


@pytest.mark.parametrize("lead_selection", ["pooled", "joint"])
def test_dosage_discovery_end_to_end(tmp_path, lead_selection):
    d = _cohort()
    perm, nominal = _scan(d, tmp_path, lead_selection)
    assert len(perm) == len(d["ids"])
    # product 1: the empirical p is the phenotype-level statistic; the Beta columns are blank
    assert (perm["pval_fit_method"] == "empirical").all()
    assert perm[["pval_beta", "beta_shape1", "beta_shape2"]].isna().all().all()
    # the stored null: one row per phenotype, both nulls on the (K, dof) p scale, sorted, max <= lead draw by draw
    ns = null_summary_of(perm)
    assert list(ns["phenotype_id"]) == list(perm["phenotype_id"]) and (ns["nperm"] == 200).all()
    for _, row in ns.iterrows():
        top_lead, top_max = np.asarray(row["null_lead_p_top"]), np.asarray(row["null_max_p_top"])
        assert top_lead.shape == top_max.shape == (200,)           # nperm <= NULL_SUMMARY_TOP: the whole null, sorted
        assert np.all(np.diff(top_lead) >= 0) and np.all(np.diff(top_max) >= 0)
        assert np.all(top_max <= top_lead + 1e-7)                   # the window minimum is at most the p at any lead
        np.testing.assert_allclose(row["levels"], NULL_SUMMARY_LEVELS.astype(np.float32))
        if lead_selection == "joint":
            np.testing.assert_array_equal(top_lead, top_max)        # one null when the lead is the window maximum
    # the empirical p of the table is the one the stored lead null gives
    for _, r in perm.iterrows():
        row = ns.set_index("phenotype_id").loc[r["phenotype_id"]]
        p_obs = r["pval_nominal"]
        k = int((np.asarray(row["null_lead_p_top"], dtype=np.float64) <= p_obs * (1 + 1e-4)).sum())
        assert abs(r["pval_perm"] - (k + 1) / 201) <= 2.5 / 201, (r["phenotype_id"], r["pval_perm"], k)
    # q-values and thresholds
    res = calculate_qvalues(perm, fdr=0.05, logger=LG, qvalue_lambda=0.5)
    sig = res["qval"] <= 0.05
    found = set(res.loc[sig, "phenotype_id"])
    assert set(d["ids"][:d["n_signal"]]) <= found                   # every planted signal is discovered
    assert len([pid for pid in found if pid.startswith("null")]) <= 2   # at most a chance null or two among 36
    # product 2: the nominal threshold is the k-th smallest window-minimum null p at the shared cutoff p*
    p_level = res["pval_perm"].to_numpy(dtype=float)
    p_star = 0.5 * (p_level[sig].max() + p_level[~sig].min())
    k = int(np.floor(p_star * 201))
    for _, r in res[sig].iterrows():
        row = ns.set_index("phenotype_id").loc[r["phenotype_id"]]
        expect = np.asarray(row["null_max_p_top"])[k - 1]
        assert r["pval_nominal_threshold"] == pytest.approx(expect)
        assert r["pval_nominal_threshold"] == pytest.approx(empirical_nominal_threshold(
            row["null_max_p_top"], row["null_max_p_levels"], 200, p_star))
    # pair extraction on the joint-F p-value of every variant
    assert pair_pval_column_of(res) == "pval_joint"
    pairs = get_significant_pairs(res, {"chr1": nominal}, fdr=0.05, logger=LG, pair_pval_col="pval_joint")
    assert not pairs.empty and set(pairs["phenotype_id"]) <= found
    thr = res.set_index("phenotype_id")["pval_nominal_threshold"]
    assert (pairs["pval_joint"].to_numpy() <= pairs["phenotype_id"].map(thr).to_numpy() * (1 + 1e-6)).all()
    nom = pd.read_parquet(nominal)
    for pid in found:
        got = set(pairs.loc[pairs["phenotype_id"] == pid, "variant_id"])
        want = set(nom.loc[(nom["phenotype_id"] == pid) & (nom["pval_joint"] <= thr[pid]), "variant_id"])
        assert got == want                                          # exactly the variants below the phenotype's threshold
        assert res.set_index("phenotype_id").loc[pid, "variant_id"] in got    # the lead itself is among them
    # no pair of a phenotype below the FDR, and no variant above its phenotype's threshold
    assert not (set(nom["phenotype_id"]) - found) & set(pairs["phenotype_id"])


def test_fit_method_beta_keeps_the_legacy_product_and_the_same_empirical_p(tmp_path):
    d = _cohort(n_signal=2, n_null=4)
    kw = dict(haplotypes=d["panel"], ancestry_model="dosage", device="cpu", window=1000, logger=LG, nperm=200, seed=5,
              lead_selection="pooled")
    args = (d["genotype_df"], d["variant_df"], d["phenotype_df"], d["phenotype_pos_df"], d["covariates_df"])
    emp = map_permutations(*args, **kw)
    beta = map_permutations(*args, pval_fit_method="beta", **kw)
    assert (beta["pval_fit_method"] == "beta").all() and np.isfinite(beta["pval_beta"]).all()
    # the same permutations: pval_perm, leads and every non-Beta column identical; the Beta is the only addition
    for col in [c for c in emp.columns if c not in ("pval_beta", "beta_shape1", "beta_shape2", "pval_fit_method")]:
        pd.testing.assert_series_equal(emp[col], beta[col], check_names=False)
    # the null summary is stored either way, so the empirical threshold is available to both products
    pd.testing.assert_frame_equal(null_summary_of(emp).drop(columns="levels"), null_summary_of(beta).drop(columns="levels"))
    # q-values: the Beta table uses pval_beta, the empirical table pval_perm; thresholds from the Beta vs the null
    rb = calculate_qvalues(beta.copy(), fdr=0.5, logger=LG, qvalue_lambda=0.5)
    re_ = calculate_qvalues(emp.copy(), fdr=0.5, logger=LG, qvalue_lambda=0.5)
    assert (rb["qval"] <= 0.5).sum() >= 2 and (re_["qval"] <= 0.5).sum() >= 2
    assert np.isfinite(rb.loc[rb["qval"] <= 0.5, "pval_nominal_threshold"]).all()
    assert np.isfinite(re_.loc[re_["qval"] <= 0.5, "pval_nominal_threshold"]).all()
    with pytest.raises(ValueError, match="beta_approx"):
        map_permutations(*args, pval_fit_method="beta", beta_approx=False, **kw)
    with pytest.raises(ValueError, match="pval_fit_method"):
        map_permutations(*args, pval_fit_method="gpd", **kw)


def test_correlation_scans_keep_the_beta_product_by_default_and_report_the_method():
    d = _cohort(n_signal=1, n_null=3)
    args = (d["genotype_df"], d["variant_df"], d["phenotype_df"], d["phenotype_pos_df"], d["covariates_df"])
    main = map_permutations(*args, haplotypes=d["panel"], ancestry_model="main", device="cpu", window=1000, logger=LG,
                            nperm=100, seed=2)
    assert (main["pval_fit_method"] == "beta").all() and np.isfinite(main["pval_beta"]).all()
    assert "null_summary" not in main.attrs
    off = map_permutations(*args, haplotypes=d["panel"], ancestry_model="main", device="cpu", window=1000, logger=LG,
                           nperm=100, seed=2, beta_approx=False)
    assert (off["pval_fit_method"] == "empirical").all() and off["pval_beta"].isna().all()
    pd.testing.assert_series_equal(off["pval_perm"], main["pval_perm"])
