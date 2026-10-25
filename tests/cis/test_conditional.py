import numpy as np
import pandas as pd
import pytest

from src.localqtl.cis.conditional import map_conditional_dosage
from src.localqtl.utils import SimpleLogger

LG = SimpleLogger(verbose=False)


def _two_signal_cohort(seed=17, n=120, m=8):
    """Cohort where v2 acts equally on both ancestries and v5 only on ancestry 0."""
    from tests.conftest import make_random_panel
    rng = np.random.default_rng(seed)
    samples = [f"S{i}" for i in range(n)]
    G = rng.integers(0, 3, size=(m, n)).astype(np.float32)
    gdf = pd.DataFrame(G, index=[f"v{i}" for i in range(m)], columns=samples)
    vdf = pd.DataFrame({"chrom": "chr1", "pos": 100 + 10 * np.arange(m)}, index=gdf.index)
    panel, admix = make_random_panel(vdf, samples, seed=seed + 1)
    H0 = admix[:, :, 0].astype(np.float64)          # ancestry-0 count per variant/sample
    y = (0.6 * G[2] + 0.9 * G[5] * H0[5] / 2.0 + rng.normal(size=n)).astype(np.float32)
    pdf = pd.DataFrame([y], index=["geneA"], columns=samples)
    pos = pd.DataFrame({"chr": ["chr1"], "start": [150], "end": [150]}, index=["geneA"])
    cov = pd.DataFrame(rng.normal(size=(n, 2)), index=samples, columns=["c1", "c2"])
    signals = pd.DataFrame({"phenotype_id": ["geneA", "geneA"],
                            "variant_id": ["v2", "v5"], "rank": [1, 2]})
    return dict(genotype_df=gdf, variant_df=vdf, phenotype_df=pdf, phenotype_pos_df=pos,
                covariates_df=cov, panel=panel, samples=samples, signals=signals,
                G=G, H0=H0, y=y)


def _reference_fit(d, target, others, condition_on):
    """numpy OLS of y ~ 1 + cov + <conditioning> + D_0 + D_1 + L_adj."""
    from tests.conftest import ols_fit
    G, H0 = d["G"], d["H0"]
    ix = {v: int(v[1:]) for v in d["genotype_df"].index}
    cols = [d["covariates_df"].to_numpy(np.float64)]
    for o in others:
        j = ix[o]
        if condition_on == "genotype":
            cols.append(G[j][:, None].astype(np.float64))
        elif condition_on == "dosage":
            cols.append(np.column_stack([G[j] * H0[j] / 2.0, G[j] * (2.0 - H0[j]) / 2.0]))
    t = ix[target]
    n_cond = sum(c.shape[1] for c in cols[1:])
    cols.append(np.column_stack([G[t] * H0[t] / 2.0, G[t] * (2.0 - H0[t]) / 2.0]))  # D_0, D_1
    cols.append(H0[t][:, None])                                                     # L_adj
    X = np.column_stack(cols)
    b, se, dof = ols_fit(d["y"].astype(np.float64), X)
    first = d["covariates_df"].shape[1] + n_cond
    return b[first:first + 2], se[first:first + 2], dof


@pytest.mark.parametrize("condition_on", ["genotype", "dosage", "none"])
def test_conditional_dosage_matches_numpy_ols(condition_on):
    d = _two_signal_cohort()
    out = map_conditional_dosage(
        d["genotype_df"], d["variant_df"], d["phenotype_df"], d["phenotype_pos_df"],
        d["signals"], d["panel"], d["covariates_df"], condition_on=condition_on,
        device="cpu", logger=LG,
    )
    assert list(out["variant_id"]) == ["v2", "v5"]
    assert list(out["phenotype_id"]) == ["geneA", "geneA"]
    assert list(out["rank"]) == [1, 2]
    expected_cond = 0 if condition_on == "none" else 1
    assert (out["n_conditioned"] == expected_cond).all()

    for _, row in out.iterrows():
        others = [v for v in ("v2", "v5") if v != row["variant_id"]]
        b_ref, se_ref, dof_ref = _reference_fit(d, row["variant_id"], others, condition_on)
        assert int(row["dof"]) == dof_ref
        for k in (0, 1):
            np.testing.assert_allclose(row[f"slope_anc{k}"], b_ref[k], rtol=2e-3, atol=2e-3)
            np.testing.assert_allclose(row[f"slope_se_anc{k}"], se_ref[k], rtol=2e-3, atol=2e-3)


def test_conditional_dosage_recovers_the_ancestry_specific_signal():
    d = _two_signal_cohort()
    out = map_conditional_dosage(
        d["genotype_df"], d["variant_df"], d["phenotype_df"], d["phenotype_pos_df"],
        d["signals"], d["panel"], d["covariates_df"], device="cpu", logger=LG,
    ).set_index("variant_id")
    # v5 was simulated to act only on ancestry-0 haplotypes
    assert out.loc["v5", "slope_anc0"] > out.loc["v5", "slope_anc1"]
    assert out.loc["v5", "pval_diff_anc0_anc1"] < 0.05
    assert out.loc["v5", "pval_anc0"] < 0.05
    # v2 acts equally on both, so the difference test should not fire
    assert out.loc["v2", "pval_diff_anc0_anc1"] > 0.05
    assert out.loc["v2", "pval_joint"] < 0.05
    stats = out.loc["v2", ["af_anc0", "af_anc1", "an_anc0", "ma_count_anc0"]].astype(float)
    assert np.isfinite(stats).all()


def test_conditional_dosage_conditioning_mode_changes_the_degrees_of_freedom():
    d = _two_signal_cohort()
    kw = dict(device="cpu", logger=LG)
    none = map_conditional_dosage(d["genotype_df"], d["variant_df"], d["phenotype_df"],
                                  d["phenotype_pos_df"], d["signals"], d["panel"],
                                  d["covariates_df"], condition_on="none", **kw)
    geno = map_conditional_dosage(d["genotype_df"], d["variant_df"], d["phenotype_df"],
                                  d["phenotype_pos_df"], d["signals"], d["panel"],
                                  d["covariates_df"], condition_on="genotype", **kw)
    dos = map_conditional_dosage(d["genotype_df"], d["variant_df"], d["phenotype_df"],
                                 d["phenotype_pos_df"], d["signals"], d["panel"],
                                 d["covariates_df"], condition_on="dosage", **kw)
    # one conditioned signal costs 1 dof pooled, K=2 dof ancestry-resolved
    assert (none["dof"] - geno["dof"] == 1).all()
    assert (none["dof"] - dos["dof"] == 2).all()
    assert not np.allclose(geno["slope_anc0"], none["slope_anc0"])


def test_conditional_dosage_masking_and_validation():
    d = _two_signal_cohort()
    masked = map_conditional_dosage(
        d["genotype_df"], d["variant_df"], d["phenotype_df"], d["phenotype_pos_df"],
        d["signals"], d["panel"], d["covariates_df"], min_ancestry_ac=1e9,
        device="cpu", logger=LG,
    )
    assert masked["slope_anc0"].isna().all() and masked["z_diff_anc0_anc1"].isna().all()
    assert np.isfinite(masked["fstat_joint"]).all() and np.isfinite(masked["af_anc0"]).all()

    base = (d["genotype_df"], d["variant_df"], d["phenotype_df"], d["phenotype_pos_df"])
    with pytest.raises(ValueError, match="condition_on"):
        map_conditional_dosage(*base, d["signals"], d["panel"], condition_on="nope",
                               device="cpu", logger=LG)
    with pytest.raises(ValueError, match="requires haplotypes"):
        map_conditional_dosage(*base, d["signals"], None, device="cpu", logger=LG)
    with pytest.raises(ValueError, match="variant_id"):
        map_conditional_dosage(*base, d["signals"].drop(columns=["variant_id"]), d["panel"],
                               device="cpu", logger=LG)
    with pytest.raises(ValueError, match="not in genotype_df"):
        map_conditional_dosage(*base, pd.DataFrame({"phenotype_id": ["geneA"],
                                                    "variant_id": ["nope"]}),
                               d["panel"], device="cpu", logger=LG)
    with pytest.raises(ValueError, match="not in phenotype_df"):
        map_conditional_dosage(*base, pd.DataFrame({"phenotype_id": ["nope"],
                                                    "variant_id": ["v2"]}),
                               d["panel"], device="cpu", logger=LG)


def _lexicographic_trap_cohort(seed=23, n=100, m=12):
    """
    Two phenotypes whose signal IDs sort differently from their positions.

    With ``v0..v11`` the lexicographic order puts ``v10``/``v11`` before
    ``v2``, so ordering the sub-frames by variant ID leaves ``pos``
    non-monotonic and the generator's bisect returns the wrong range.  Gene
    starts sit away from the signals so the reported distances are checked
    against the gene, not against the span of its signals.
    """
    from tests.conftest import make_random_panel
    rng = np.random.default_rng(seed)
    samples = [f"S{i}" for i in range(n)]
    G = rng.integers(0, 3, size=(m, n)).astype(np.float32)
    gdf = pd.DataFrame(G, index=[f"v{i}" for i in range(m)], columns=samples)
    vdf = pd.DataFrame({"chrom": "chr1", "pos": 100 + 10 * np.arange(m)}, index=gdf.index)
    panel, admix = make_random_panel(vdf, samples, seed=seed + 1)
    H0 = admix[:, :, 0].astype(np.float64)
    yA = (0.7 * G[2] + 0.5 * G[10] + rng.normal(size=n)).astype(np.float32)
    yB = (0.8 * G[11] + rng.normal(size=n)).astype(np.float32)
    pdf = pd.DataFrame([yA, yB], index=["geneA", "geneB"], columns=samples)
    pos = pd.DataFrame({"chr": ["chr1", "chr1"], "start": [150, 400], "end": [250, 460]},
                       index=["geneA", "geneB"])
    signals = pd.DataFrame({"phenotype_id": ["geneA", "geneA", "geneB"],
                            "variant_id": ["v2", "v10", "v11"], "rank": [1, 2, 1]})
    return dict(genotype_df=gdf, variant_df=vdf, phenotype_df=pdf, phenotype_pos_df=pos,
                panel=panel, signals=signals, G=G, H0=H0)


def test_conditional_dosage_keeps_signals_whose_ids_sort_unlike_their_positions():
    d = _lexicographic_trap_cohort()
    out = map_conditional_dosage(
        d["genotype_df"], d["variant_df"], d["phenotype_df"], d["phenotype_pos_df"],
        d["signals"], d["panel"], device="cpu", logger=LG,
    )
    # every signal is fitted; none is silently dropped as "unmapped"
    assert set(zip(out["phenotype_id"], out["variant_id"])) == {
        ("geneA", "v2"), ("geneA", "v10"), ("geneB", "v11")}
    a = out[out["phenotype_id"] == "geneA"]
    assert (a["n_conditioned"] == 1).all()          # each conditioned on the other
    assert (out[out["phenotype_id"] == "geneB"]["n_conditioned"] == 0).all()


def test_conditional_dosage_distances_are_measured_from_the_phenotype():
    d = _lexicographic_trap_cohort()
    out = map_conditional_dosage(
        d["genotype_df"], d["variant_df"], d["phenotype_df"], d["phenotype_pos_df"],
        d["signals"], d["panel"], device="cpu", logger=LG,
    ).set_index("variant_id")
    pos = d["variant_df"]["pos"]
    # geneA spans [150, 250]; geneB spans [400, 460]
    assert int(out.loc["v2", "start_distance"]) == int(pos["v2"]) - 150      # 120 - 150
    assert int(out.loc["v2", "end_distance"]) == int(pos["v2"]) - 250
    assert int(out.loc["v10", "start_distance"]) == int(pos["v10"]) - 150    # 200 - 150
    assert int(out.loc["v11", "start_distance"]) == int(pos["v11"]) - 400
    assert int(out.loc["v11", "end_distance"]) == int(pos["v11"]) - 460


def test_conditional_dosage_empty_result_keeps_the_full_column_schema():
    d = _two_signal_cohort()
    gdf = d["genotype_df"].copy()
    gdf.loc[["v2", "v5"]] = 1.0                     # monomorphic: both signals drop out
    out = map_conditional_dosage(
        gdf, d["variant_df"], d["phenotype_df"], d["phenotype_pos_df"],
        d["signals"], d["panel"], d["covariates_df"], device="cpu", logger=LG,
    )
    assert len(out) == 0
    full = map_conditional_dosage(
        d["genotype_df"], d["variant_df"], d["phenotype_df"], d["phenotype_pos_df"],
        d["signals"], d["panel"], d["covariates_df"], device="cpu", logger=LG,
    )
    assert list(out.columns) == list(full.columns)
    # the documented dosage columns are indexable on an empty result
    for col in ("rank", "slope_anc0", "pval_joint", "pval_diff_anc0_anc1", "af_anc1"):
        assert col in out.columns
        assert len(out[col]) == 0
