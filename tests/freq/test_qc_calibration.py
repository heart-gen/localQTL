import numpy as np
import pandas as pd
import pytest

from localqtl.freq import (
    FrequencyTable,
    ancestry_allele_frequencies, ancestry_labels, anc_columns_to_labels,
    compare_allele_frequencies, summarize_af_concordance, annotate_with_frequency,
    genomic_control_lambda, qq_table, calibration_by_frequency, calibration_by_frequency_dosage,
    run_frequency_benchmark,
)
from localqtl.cis import map_nominal
from localqtl.utils import SimpleLogger


# ---------------------------------------------------------------- in-sample AF

def test_ancestry_allele_frequencies_matches_numpy_and_dosage_scan(toy_panel_2anc):
    d = toy_panel_2anc
    labels = ancestry_labels(d["panel"])
    out = ancestry_allele_frequencies(d["genotype_df"], d["variant_df"], d["panel"])
    assert list(out.index) == list(d["genotype_df"].index)
    G = d["genotype_df"].to_numpy(np.float64)
    L = d["admix"].astype(np.float64)
    valid = (G != -9)
    for i, vid in enumerate(out.index):
        v = valid[i]
        for k, lab in enumerate(labels):
            an = L[i, v, k].sum()
            ac = (G[i, v] * L[i, v, k] * 0.5).sum()
            np.testing.assert_allclose(out.loc[vid, f"an_anc_{lab}"], an)
            np.testing.assert_allclose(out.loc[vid, f"ac_anc_{lab}"], ac, rtol=1e-6)
        assert out.loc[vid, "n_called"] == v.sum()
    # variants without missing genotypes agree with the dosage-scan columns
    res = map_nominal(genotype_df=d["genotype_df"], variant_df=d["variant_df"],
                      phenotype_df=d["phenotype_df"], phenotype_pos_df=d["phenotype_pos_df"],
                      haplotypes=d["panel"], ancestry_model="dosage", device="cpu", nperm=None,
                      out_dir=None, logger=SimpleLogger(verbose=False))
    res = anc_columns_to_labels(res.drop_duplicates("variant_id").set_index("variant_id"), labels)
    for vid in res.index:
        if valid[list(out.index).index(vid)].all():
            for lab in labels:
                np.testing.assert_allclose(res.loc[vid, f"af_anc_{lab}"], out.loc[vid, f"af_anc_{lab}"], rtol=1e-5)
    assert "z_diff_anc_" in "".join(res.columns)   # pairwise columns renamed too


def test_ancestry_allele_frequencies_posterior_panel(toy_panel_2anc_posterior):
    d = toy_panel_2anc_posterior
    out = ancestry_allele_frequencies(d["genotype_df"], d["variant_df"], d["panel_post"])
    labels = ancestry_labels(d["panel_post"])
    an = out[[f"an_anc_{l}" for l in labels]].to_numpy()
    np.testing.assert_allclose(an.sum(axis=1), 2 * out["n_called"].to_numpy(), rtol=1e-5)


def test_ancestry_allele_frequencies_rejects_sample_mismatch(toy_panel_2anc):
    d = toy_panel_2anc
    bad = d["genotype_df"].rename(columns={"S1": "ZZ"})
    with pytest.raises(ValueError, match="select_samples"):
        ancestry_allele_frequencies(bad, d["variant_df"], d["panel"])


# ------------------------------------------------------------------- QC

def test_compare_allele_frequencies_recovers_truth_and_flags(sim_cohort):
    d = sim_cohort(seed=1, n=200, m=60)
    labels = ancestry_labels(d["panel"])
    insample = ancestry_allele_frequencies(d["genotype_df"], d["variant_df"], d["panel"])
    with pytest.raises(ValueError, match="ancestry_map is required"):
        compare_allele_frequencies(insample, d["table"], {})
    amap = {labels[0]: "afr", labels[1]: "nfe"}
    cmp_df = compare_allele_frequencies(insample, d["table"], amap)
    a0 = labels[0]
    assert {f"af_in_{a0}", "af_ref_afr", f"af_diff_{a0}", f"z_{a0}", f"discordant_{a0}",
            "af_ref_expected", "af_diff_pooled", "discordant", "match_type"} <= set(cmp_df.columns)
    ok = cmp_df[f"af_diff_{a0}"].notna()
    assert ok.sum() > 40
    r = np.corrcoef(cmp_df.loc[ok, f"af_in_{a0}"], cmp_df.loc[ok, "af_ref_afr"])[0, 1]
    assert r > 0.9
    assert cmp_df["discordant"].mean() < 0.15
    # inject a flipped reference row and a wildly wrong one
    tdf = d["table"].df.copy()
    flip_ix = tdf.index[0]
    tdf.loc[flip_ix, ["ref", "alt"]] = tdf.loc[flip_ix, ["alt", "ref"]].to_numpy()
    tdf.loc[tdf.index[1], ["af_afr", "af_nfe"]] = [0.99, 0.99]
    tdf.loc[tdf.index[1], ["ac_afr", "ac_nfe"]] = [1980, 4950]
    cmp2 = compare_allele_frequencies(insample, FrequencyTable(tdf, source="t"), amap)
    assert bool(cmp2.iloc[0]["allele_flipped"])
    # flipped orientation is corrected, so the diff stays small
    assert abs(cmp2.iloc[0][f"af_diff_{a0}"]) < 0.2
    assert bool(cmp2.iloc[1]["discordant"])
    summ = summarize_af_concordance(cmp2)
    assert set(summ["ancestry"]) == set(labels)
    allrow = summ[(summ["ancestry"] == a0) & (summ["maf_bin"] == "all")].iloc[0]
    assert allrow["n"] > 40 and allrow["pearson_r"] > 0.85 and 0 < allrow["frac_discordant"] < 0.2


def test_annotate_with_frequency_merges_vectorised(sim_cohort):
    d = sim_cohort(seed=2, n=40, m=20)
    labels = ancestry_labels(d["panel"])
    res = map_nominal(genotype_df=d["genotype_df"], variant_df=d["variant_df"],
                      phenotype_df=d["phenotype_df"], phenotype_pos_df=d["phenotype_pos_df"],
                      haplotypes=d["panel"], ancestry_model="dosage", device="cpu", nperm=None,
                      out_dir=None, window=100_000, logger=SimpleLogger(verbose=False))
    amap = {labels[0]: "afr", labels[1]: "nfe"}
    out = annotate_with_frequency(res, d["variant_df"], d["table"], ancestry_map=amap, ancestry_labels=labels)
    assert len(out) == len(res) and list(out.index) == list(res.index)
    assert {"af_ref_afr", "an_ref_afr", "af_ref_nfe", "af_ref_anc0", "af_ref_anc1", "match_type"} <= set(out.columns)
    assert out["missing_in_reference"].sum() == 0
    truth = pd.Series(d["p_afr"], index=d["variant_df"].index)
    np.testing.assert_allclose(out["af_ref_afr"].to_numpy(), truth.loc[out["variant_id"]].to_numpy())
    with pytest.raises(ValueError, match="not in variant_df"):
        annotate_with_frequency(res, d["variant_df"].iloc[:5], d["table"], groups=["afr"])


# ------------------------------------------------------------ calibration

def test_calibration_metrics_on_uniform_null():
    rng = np.random.default_rng(0)
    p = rng.uniform(size=20_000)
    lam = genomic_control_lambda(p)
    assert 0.95 < lam < 1.05
    qq = qq_table(p, n_points=50)
    assert len(qq) <= 50 and np.all(np.diff(qq["expected_neglog10"]) <= 0)
    df = pd.DataFrame({"pval_nominal": p, "af": rng.uniform(0, 1, size=p.size)})
    tab = calibration_by_frequency(df, bins=(0, 0.05, 0.25, 0.5))
    assert list(tab["bin"])[0] == "all" and len(tab) == 4
    for _, row in tab.iterrows():
        assert 0.85 < row["lambda_gc"] < 1.15
        assert 0.035 < row["frac_p_lt_0.05"] < 0.065
        assert row["kolmogorov_p"] > 1e-4
    # small bins report NaN
    tiny = calibration_by_frequency(df.iloc[:10], bins=(0, 0.5), min_n=20)
    assert tiny["lambda_gc"].isna().all()


def test_calibration_by_frequency_dosage_layout():
    rng = np.random.default_rng(1)
    n = 5000
    df = pd.DataFrame({
        "pval_anc0": rng.uniform(size=n), "af_anc0": rng.uniform(size=n),
        "pval_anc1": rng.uniform(size=n), "af_anc1": rng.uniform(size=n),
        "pval_joint": rng.uniform(size=n), "af": rng.uniform(size=n),
        "ma_count_anc0": rng.integers(0, 50, size=n), "ma_count_anc1": rng.integers(0, 50, size=n),
    })
    tab = calibration_by_frequency_dosage(df, ancestry_labels=["AFR", "EUR"], bins=(0, 0.1, 0.5))
    assert set(tab["ancestry"]) == {"AFR", "EUR", "joint"}
    assert ((tab["lambda_gc"] > 0.8) & (tab["lambda_gc"] < 1.2)).all()
    mac = calibration_by_frequency_dosage(df, by="ma_count_anc", bins=(0, 10, 25, 50), fold=False,
                                          include_joint=False)
    assert set(mac["ancestry"]) == {"anc0", "anc1"}
    with pytest.raises(ValueError, match="pval_anc"):
        calibration_by_frequency_dosage(df[["pval_joint", "af"]])


# -------------------------------------------------------------- benchmark

def test_run_frequency_benchmark_end_to_end(sim_cohort):
    d = sim_cohort(seed=3, n=30, m=40)
    labels = ancestry_labels(d["panel"])
    amap = {labels[0]: "afr", labels[1]: "nfe"}
    res = run_frequency_benchmark(
        d["genotype_df"], d["variant_df"], d["phenotype_df"], d["phenotype_pos_df"],
        d["panel"], d["table"], amap, window=100_000, n_null=1, seed=0, device="cpu",
        drop_discordant=True, bins=(0, 0.1, 0.5), min_ancestry_ac=2,
    )
    assert res.n_variants_in == 40 and 0 < res.n_variants_kept <= 40
    assert not res.nominal_df.empty and "af_ref_anc0" in res.nominal_df.columns
    assert set(res.calibration_df["ancestry"]) == set(labels) | {"joint"}
    assert res.null_df is not None and "null_replicate" in res.null_df.columns
    assert set(res.calibration_null_df["ancestry"]) == set(labels) | {"joint"}
    assert res.flags["ancestry_map"] == amap
    assert set(res.af_summary["ancestry"]) == set(labels)
