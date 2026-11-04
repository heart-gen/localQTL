"""The LD-aware ancestry contrast (localqtl.cis.ld_aware) and its two entry points."""
import numpy as np
import pandas as pd
import pytest
import torch

from src.localqtl.cis import ld_aware as LD
from src.localqtl.cis import map_conditional_dosage, map_permutations
from src.localqtl.cis.common import residualize_batch_dosage
from src.localqtl.regression_kernels import Residualizer
from src.localqtl.utils import SimpleLogger

LG = SimpleLogger(verbose=False)
TAG, CAUSAL = 0, 1


def _ld_cohort(seed=0, n=300, m=8, beta=1.0, rho=(0.95, 0.2), f=0.4, n_pheno=1, n_cov=2):
    """
    Haplotype-level cohort with ancestry-specific LD: variant CAUSAL carries a shared effect ``beta``
    in both ancestries and variant TAG copies its allele with probability ``rho[k]`` on ancestry-k
    haplotypes, so the marginal effect of TAG differs by ancestry although the causal effect does not.
    The remaining variants are independent noise.  Local ancestry is constant over the window.
    """
    from rfmix_reader import from_legacy
    from src.localqtl.haplotypeio import LocalAncestryPanel
    rng = np.random.default_rng(seed)
    samples = [f"S{i}" for i in range(n)]
    anc = rng.integers(0, 2, size=(n, 2))                                     # haplotype ancestries
    a_c = (rng.random((n, 2)) < f).astype(np.float32)
    copy = rng.random((n, 2)) < np.where(anc == 0, rho[0], rho[1])
    a_t = np.where(copy, a_c, (rng.random((n, 2)) < f).astype(np.float32)).astype(np.float32)
    G = rng.integers(0, 3, size=(m, n)).astype(np.float32)
    G[TAG] = a_t.sum(1)
    G[CAUSAL] = a_c.sum(1)
    gdf = pd.DataFrame(G, index=[f"v{i}" for i in range(m)], columns=samples)
    vdf = pd.DataFrame({"chrom": "chr1", "pos": 100 + 10 * np.arange(m)}, index=gdf.index)
    admix = np.stack([np.repeat((anc == k).sum(1)[None, :], m, axis=0) for k in range(2)], axis=-1).astype(np.int8)
    loci = pd.DataFrame({"chromosome": vdf["chrom"].astype(str).values, "physical_position": vdf["pos"].values})
    g_anc = pd.DataFrame({"sample_id": samples, "P0": 0.5, "P1": 0.5, "chrom": "1"})
    panel = LocalAncestryPanel(from_legacy(loci, g_anc, admix))
    cov = pd.DataFrame(rng.normal(size=(n, n_cov)), index=samples, columns=[f"c{i}" for i in range(n_cov)])
    Y = []
    for _ in range(n_pheno):
        Y.append(beta * G[CAUSAL] + 0.3 * cov.to_numpy().sum(1) + rng.normal(size=n))
    pdf = pd.DataFrame(np.asarray(Y, dtype=np.float32), index=[f"g{i}" for i in range(n_pheno)], columns=samples)
    pos = pd.DataFrame({"chr": "chr1", "start": 120, "end": 120}, index=pdf.index)
    H0 = admix[:, :, 0].astype(np.float64)
    return dict(genotype_df=gdf, variant_df=vdf, phenotype_df=pdf, phenotype_pos_df=pos, covariates_df=cov,
                panel=panel, samples=samples, G=G, H0=H0, admix=admix, y=np.asarray(Y[0], dtype=np.float64))


def _numpy_conditional(d, target, cond, condition_on="genotype"):
    """Full conditional OLS y ~ 1 + cov + <cond columns> + D_0(target) + D_1(target) + L_adj(target):
    returns (a - b, se(a - b), dof)."""
    G, H0 = d["G"], d["H0"]
    cols = [np.ones(len(d["y"])), d["covariates_df"].to_numpy(np.float64)]
    for j in cond:
        if condition_on == "genotype":
            cols.append(G[j].astype(np.float64)[:, None])
        else:
            cols.append(np.column_stack([G[j] * H0[j] / 2.0, G[j] * (2.0 - H0[j]) / 2.0]))
    t = target
    cols.append(np.column_stack([G[t] * H0[t] / 2.0, G[t] * (2.0 - H0[t]) / 2.0]))
    cols.append(H0[t][:, None])
    X = np.column_stack([c if c.ndim == 2 else c[:, None] for c in cols])
    y = d["y"]
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    r = y - X @ b
    dof = len(y) - np.linalg.matrix_rank(X)
    cov = (r @ r) / dof * np.linalg.pinv(X.T @ X)
    i0 = X.shape[1] - 3
    c = np.zeros(X.shape[1]); c[i0] = 1.0; c[i0 + 1] = -1.0
    return float(c @ b), float(np.sqrt(c @ cov @ c)), int(dof)


def _torch_inputs(d, device="cpu"):
    G_t = torch.as_tensor(d["G"], dtype=torch.float32)
    H_t = torch.as_tensor(d["admix"].astype(np.float32))
    y_t = torch.as_tensor(d["phenotype_df"].iloc[0].to_numpy(np.float32))
    rez = Residualizer(torch.as_tensor(d["covariates_df"].to_numpy(np.float32)))
    y_r, D_r, L_r = residualize_batch_dosage(y_t, G_t, H_t, rez, center=True, group=False)
    (G_r,) = rez.transform(G_t, center=True)
    return G_t, H_t, rez, y_r, D_r, L_r, G_r


@pytest.mark.parametrize("condition_on", ["genotype", "dosage"])
def test_ld_aware_contrast_matches_the_full_conditional_ols(condition_on):
    d = _ld_cohort(seed=3)
    G_t, H_t, rez, y_r, D_r, L_r, G_r = _torch_inputs(d)
    X = LD.conditioning_columns(G_t, H_t, rez, [CAUSAL], condition_on=condition_on, G_resid=G_r)
    out = LD.ld_aware_contrast(y_r, D_r[TAG:TAG + 1], L_r[TAG:TAG + 1], G_r[TAG], X, k_eff=rez.k_eff, device="cpu")
    diff, se, dof = _numpy_conditional(d, TAG, [CAUSAL], condition_on)
    assert out["n_conditioned_ld"] == (1 if condition_on == "genotype" else 2)
    assert out["rank_ld"] == out["n_conditioned_ld"] and out["dof_ld"] == dof
    np.testing.assert_allclose(out["slope_ld_anc0"] - out["slope_ld_anc1"], diff, rtol=2e-3, atol=2e-4)
    np.testing.assert_allclose(out["z_diff_ld_anc0_anc1"], diff / se, rtol=2e-3, atol=2e-3)
    assert 0 <= out["pval_diff_ld_anc0_anc1"] <= 1 and np.isfinite(out["pval_joint_ld"])


def test_ld_aware_contrast_without_conditioning_is_the_marginal_contrast_and_a_duplicate_lead_is_dropped():
    d = _ld_cohort(seed=5)
    G_t, H_t, rez, y_r, D_r, L_r, G_r = _torch_inputs(d)
    marg = LD.ld_aware_contrast(y_r, D_r[TAG:TAG + 1], L_r[TAG:TAG + 1], G_r[TAG], None, k_eff=rez.k_eff, device="cpu")
    diff, se, dof = _numpy_conditional(d, TAG, [])
    np.testing.assert_allclose(marg["z_diff_ld_anc0_anc1"], diff / se, rtol=2e-3, atol=2e-3)
    assert marg["n_conditioned_ld"] == 0 and marg["rank_ld"] == 0 and marg["dof_ld"] == dof
    # a conditioning variant identical to the lead carries nothing beyond the lead's own genotype
    dup = LD.ld_aware_contrast(y_r, D_r[TAG:TAG + 1], L_r[TAG:TAG + 1], G_r[TAG], G_r[TAG].reshape(-1, 1),
                               k_eff=rez.k_eff, device="cpu")
    assert dup["n_conditioned_ld"] == 1 and dup["rank_ld"] == 0 and dup["dof_ld"] == dof
    np.testing.assert_allclose(dup["z_diff_ld_anc0_anc1"], marg["z_diff_ld_anc0_anc1"], rtol=1e-5)
    assert np.isfinite([dup[c] for c in LD.ld_columns(2, with_cs=False)]).all()


def test_conditioning_on_the_causal_variant_removes_the_tagging_leak():
    """The S8 mechanism: a shared causal effect tagged with ancestry-specific LD makes the marginal contrast at the
    tag fire; conditioned on the causal variant's pooled genotype the contrast is null."""
    marginal, conditional = [], []
    for seed in range(20):
        d = _ld_cohort(seed=100 + seed)
        G_t, H_t, rez, y_r, D_r, L_r, G_r = _torch_inputs(d)
        m = LD.ld_aware_contrast(y_r, D_r[TAG:TAG + 1], L_r[TAG:TAG + 1], G_r[TAG], None, k_eff=rez.k_eff, device="cpu")
        X = LD.conditioning_columns(G_t, H_t, rez, [CAUSAL], condition_on="genotype", G_resid=G_r)
        c = LD.ld_aware_contrast(y_r, D_r[TAG:TAG + 1], L_r[TAG:TAG + 1], G_r[TAG], X, k_eff=rez.k_eff, device="cpu")
        marginal.append(m["pval_diff_ld_anc0_anc1"])
        conditional.append(c["pval_diff_ld_anc0_anc1"])
    assert np.mean(np.asarray(marginal) < 0.05) >= 0.8
    assert np.mean(np.asarray(conditional) < 0.05) <= 0.25


def test_resolve_ld_options_and_credible_sets():
    o = LD.resolve_ld_options(None)
    assert o["condition_on"] == "genotype" and o["susie"]["L"] == 10 and o["susie"]["coverage"] == 0.95
    o = LD.resolve_ld_options({"condition_on": "dosage", "L": 3, "susie": {"coverage": 0.9}})
    assert o["condition_on"] == "dosage" and o["susie"]["L"] == 3 and o["susie"]["coverage"] == 0.9
    with pytest.raises(ValueError, match="condition_on"):
        LD.resolve_ld_options({"condition_on": "none"})
    with pytest.raises(ValueError, match="unknown"):
        LD.resolve_ld_options({"nope": 1})
    d = _ld_cohort(seed=11, n=400)
    G_t, H_t, rez, y_r, D_r, L_r, G_r = _torch_inputs(d)
    cs = LD.credible_sets(y_r, G_r, L=5)
    assert cs["converged"] and len(cs["pip"]) == G_t.shape[0]
    assert any(CAUSAL in set(int(i) for i in arr) for arr in cs["cs"])      # the causal variant is fine-mapped
    empty = LD.credible_sets(y_r, G_r[:0])
    assert empty["cs"] == [] and len(empty["pip"]) == 0


def test_lead_ld_contrast_accepts_credible_sets_as_tensors():
    """SuSiE returns the credible-set indices (and may return the PIPs) as tensors on the compute device."""
    d = _ld_cohort(seed=17)
    G_t, H_t, rez, y_r, D_r, L_r, G_r = _torch_inputs(d)
    cs_t = {"cs": [torch.tensor([TAG, CAUSAL])], "pip": torch.tensor([0.4, 0.6] + [0.0] * (G_t.shape[0] - 2), dtype=torch.float64),
            "converged": True}
    cs_n = {"cs": [np.array([TAG, CAUSAL])], "pip": np.array([0.4, 0.6] + [0.0] * (G_t.shape[0] - 2)), "converged": True}
    a = LD.lead_ld_contrast(y_r, D_r, L_r, G_t, H_t, rez, TAG, k_eff=rez.k_eff, cs=cs_t, G_resid=G_r, device="cpu")
    b = LD.lead_ld_contrast(y_r, D_r, L_r, G_t, H_t, rez, TAG, k_eff=rez.k_eff, cs=cs_n, G_resid=G_r, device="cpu")
    assert a == b and a["n_conditioned_ld"] == 1 and a["cs_size_ld"] == 2 and a["pip_ld"] == 0.4


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU"))])
def test_susie_purity_subsamples_a_large_credible_set_on_its_device(device):
    """susie_get_cs hands _get_purity the member indices as a tensor on the compute device; sets larger than 100
    members are subsampled (the first GPU run of the LD-aware scan failed there)."""
    from src.localqtl.susie import _get_purity
    torch.manual_seed(0); np.random.seed(0)
    X = torch.randn(60, 150, device=device)
    out = _get_purity(torch.arange(150, device=device), X, None)
    assert len(out) == 3 and all(np.isfinite(out)) and 0 <= out[0] <= out[1] <= 1
    small = _get_purity(torch.tensor([0, 1, 2], device=device), X, None)
    assert len(small) == 3 and all(np.isfinite(small))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the device path of the LD-aware scan needs a GPU")
def test_map_permutations_ld_aware_on_cuda_matches_cpu():
    d = _ld_cohort(seed=23, n=200)
    args = (d["genotype_df"], d["variant_df"], d["phenotype_df"], d["phenotype_pos_df"], d["covariates_df"])
    cpu = map_permutations(*args, **_perm_kw(d, ld_aware=True, nperm=30))
    gpu = map_permutations(*args, **_perm_kw(d, ld_aware=True, nperm=30, device="cuda"))
    assert gpu.iloc[0]["variant_id"] == cpu.iloc[0]["variant_id"]
    for c in ("pval_diff_ld_anc0_anc1", "z_diff_ld_anc0_anc1", "n_conditioned_ld", "cs_size_ld", "pip_ld", "dof_ld"):
        np.testing.assert_allclose(gpu.iloc[0][c], cpu.iloc[0][c], rtol=1e-3, atol=1e-5)


def _perm_kw(d, **extra):
    kw = dict(haplotypes=d["panel"], ancestry_model="dosage", device="cpu", window=1000, logger=LG, nperm=100, seed=1)
    kw.update(extra)
    return kw


def test_map_permutations_ld_aware_reports_the_conditional_contrast_at_the_lead():
    d = _ld_cohort(seed=21, n=400)
    base = map_permutations(d["genotype_df"], d["variant_df"], d["phenotype_df"], d["phenotype_pos_df"],
                            d["covariates_df"], **_perm_kw(d))
    out = map_permutations(d["genotype_df"], d["variant_df"], d["phenotype_df"], d["phenotype_pos_df"],
                           d["covariates_df"], **_perm_kw(d, ld_aware=True))
    assert set(LD.ld_columns(2)) <= set(out.columns) and not set(LD.ld_columns(2)) & set(base.columns)
    r = out.iloc[0]
    # the marginal columns are untouched by the LD-aware pass
    for c in ("variant_id", "pval_perm", "pval_beta", "slope_anc0", "pval_diff_anc0_anc1"):
        assert r[c] == base.iloc[0][c]
    assert r["variant_id"] in ("v0", "v1") and r["n_cs_ld"] >= 1 and r["cs_size_ld"] >= 1
    assert 0 <= r["pip_ld"] <= 1 and np.isfinite(r["pval_diff_ld_anc0_anc1"]) and r["dof_ld"] <= r["true_dof"]
    if r["n_conditioned_ld"] == 0:                     # the lead is its credible set: the conditional is the marginal
        np.testing.assert_allclose(r["z_diff_ld_anc0_anc1"], r["z_diff_anc0_anc1"], rtol=1e-4)
    else:
        assert r["rank_ld"] >= 1 and r["dof_ld"] == r["true_dof"] - r["rank_ld"]
    for c in ("n_conditioned_ld", "rank_ld", "dof_ld", "n_cs_ld", "cs_size_ld"):
        assert out[c].dtype.kind == "i"


@pytest.mark.parametrize("condition_on", ["genotype", "dosage"])
def test_lead_ld_contrast_at_a_tagging_lead_is_calibrated_where_the_marginal_is_not(condition_on):
    """
    The per-lead routine of the permutation scan (credible sets fitted on the window, every other
    member conditioned on) at a lead that tags the causal variant: across cohorts the marginal
    contrast fires, the LD-aware contrast does not.  The index is forced because a typed causal
    variant beats its own tag on the joint statistic in this small window, so the scan's lead
    would be the causal variant itself (see the untyped case below for a scan-level tag lead).
    """
    marg, ld, conditioned = [], [], []
    for seed in range(20):
        d = _ld_cohort(seed=300 + seed, n=400, rho=(0.98, 0.15))
        G_t, H_t, rez, y_r, D_r, L_r, G_r = _torch_inputs(d)
        m = LD.ld_aware_contrast(y_r, D_r[TAG:TAG + 1], L_r[TAG:TAG + 1], G_r[TAG], None, k_eff=rez.k_eff, device="cpu")
        r = LD.lead_ld_contrast(y_r, D_r, L_r, G_t, H_t, rez, TAG, k_eff=rez.k_eff, condition_on=condition_on,
                                susie=LD.SUSIE_DEFAULTS, G_resid=G_r, device="cpu")
        marg.append(m["pval_diff_ld_anc0_anc1"])
        ld.append(r["pval_diff_ld_anc0_anc1"])
        conditioned.append(r["n_conditioned_ld"] >= 1 and r["n_cs_ld"] >= 1)
        assert r["rank_ld"] <= r["n_conditioned_ld"] and r["dof_ld"] == m["dof_ld"] - r["rank_ld"]
    assert np.mean(conditioned) >= 0.8                    # the causal variant is fine-mapped and conditioned on
    assert np.mean(np.asarray(marg) < 0.05) >= 0.8
    assert np.mean(np.asarray(ld) < 0.05) <= 0.25


def test_map_permutations_ld_aware_with_an_untyped_causal_variant_reports_the_marginal_contrast():
    """
    When the causal variant is not typed (the benchmark's tagging design) the scan's lead is the
    tag and nothing in its credible set carries the shared effect, so the LD-aware contrast has
    nothing to condition on and equals the marginal one: the column cannot repair a leak whose
    source is absent from the data, and says so through ``n_conditioned_ld == 0``.
    """
    d = _ld_cohort(seed=321, n=400, rho=(0.98, 0.15))
    keep = [v for v in d["genotype_df"].index if v != "v1"]
    out = map_permutations(d["genotype_df"].loc[keep], d["variant_df"].loc[keep], d["phenotype_df"],
                           d["phenotype_pos_df"], d["covariates_df"], **_perm_kw(d, ld_aware=True, nperm=50))
    r = out.iloc[0]
    assert r["variant_id"] == "v0" and r["n_cs_ld"] >= 1 and r["cs_size_ld"] >= 1 and r["pip_ld"] > 0.5
    assert r["n_conditioned_ld"] == 0 and r["rank_ld"] == 0 and r["dof_ld"] == r["true_dof"]
    np.testing.assert_allclose(r["z_diff_ld_anc0_anc1"], r["z_diff_anc0_anc1"], rtol=1e-4)
    np.testing.assert_allclose(r["pval_diff_ld_anc0_anc1"], r["pval_diff_anc0_anc1"], rtol=1e-4)


def test_map_permutations_ld_aware_grouped_and_validation():
    d = _ld_cohort(seed=31, n=200, n_pheno=2)
    group_s = pd.Series(["gA", "gA"], index=d["phenotype_df"].index)
    out = map_permutations(d["genotype_df"], d["variant_df"], d["phenotype_df"], d["phenotype_pos_df"],
                           d["covariates_df"], group_s=group_s, **_perm_kw(d, ld_aware=True))
    assert len(out) == 1 and set(LD.ld_columns(2)) <= set(out.columns)
    assert np.isfinite(out.iloc[0]["pval_diff_ld_anc0_anc1"]) and out.iloc[0]["n_cs_ld"] >= 0
    with pytest.raises(ValueError, match="ld_aware"):
        map_permutations(d["genotype_df"], d["variant_df"], d["phenotype_df"], d["phenotype_pos_df"],
                         d["covariates_df"], **_perm_kw(d, ld_aware=True, ancestry_model="main"))
    with pytest.raises(ValueError, match="condition_on"):
        map_permutations(d["genotype_df"], d["variant_df"], d["phenotype_df"], d["phenotype_pos_df"],
                         d["covariates_df"], **_perm_kw(d, ld_aware=True, ld_aware_opts={"condition_on": "none"}))


@pytest.mark.parametrize("condition_on", ["genotype", "dosage"])
def test_conditional_dosage_with_credible_sets_adds_the_ld_aware_columns(condition_on):
    d = _ld_cohort(seed=41)
    signals = pd.DataFrame({"phenotype_id": ["g0"], "variant_id": ["v0"], "rank": [1]})
    cs = pd.DataFrame({"phenotype_id": ["g0", "g0", "g0"], "variant_id": ["v0", "v1", "v1"], "cs_id": ["1", "1", "1"]})
    base = map_conditional_dosage(d["genotype_df"], d["variant_df"], d["phenotype_df"], d["phenotype_pos_df"],
                                  signals, d["panel"], d["covariates_df"], condition_on=condition_on, device="cpu", logger=LG)
    out = map_conditional_dosage(d["genotype_df"], d["variant_df"], d["phenotype_df"], d["phenotype_pos_df"],
                                 signals, d["panel"], d["covariates_df"], condition_on=condition_on,
                                 credible_sets_df=cs, device="cpu", logger=LG)
    assert len(out) == 1 and set(LD.ld_columns(2, with_cs=False)) | {"n_conditioned_cs"} <= set(out.columns)
    r = out.iloc[0]
    for c in ("slope_anc0", "pval_diff_anc0_anc1", "dof", "n_conditioned"):           # the standard columns are unchanged
        assert r[c] == base.iloc[0][c]
    assert r["n_conditioned_cs"] == 1 and r["n_conditioned_ld"] == (1 if condition_on == "genotype" else 2)
    diff, se, dof = _numpy_conditional(d, TAG, [CAUSAL], condition_on)
    np.testing.assert_allclose(r["z_diff_ld_anc0_anc1"], diff / se, rtol=2e-3, atol=2e-3)
    assert r["dof_ld"] == dof
    with pytest.raises(ValueError, match="condition_on"):
        map_conditional_dosage(d["genotype_df"], d["variant_df"], d["phenotype_df"], d["phenotype_pos_df"],
                               signals, d["panel"], d["covariates_df"], condition_on="none", credible_sets_df=cs,
                               device="cpu", logger=LG)
    with pytest.raises(ValueError, match="credible-set variants"):
        map_conditional_dosage(d["genotype_df"], d["variant_df"], d["phenotype_df"], d["phenotype_pos_df"],
                               signals, d["panel"], d["covariates_df"],
                               credible_sets_df=pd.DataFrame({"phenotype_id": ["g0"], "variant_id": ["nope"]}),
                               device="cpu", logger=LG)


def test_conditional_dosage_credible_sets_combine_with_other_signals():
    """Two signals plus a credible set: the LD-aware block holds the other signal and the set, each once."""
    d = _ld_cohort(seed=43)
    signals = pd.DataFrame({"phenotype_id": ["g0", "g0"], "variant_id": ["v0", "v5"], "rank": [1, 2]})
    cs = pd.DataFrame({"phenotype_id": ["g0"] * 3, "variant_id": ["v0", "v1", "v5"]})
    out = map_conditional_dosage(d["genotype_df"], d["variant_df"], d["phenotype_df"], d["phenotype_pos_df"],
                                 signals, d["panel"], d["covariates_df"], credible_sets_df=cs, device="cpu", logger=LG)
    out = out.set_index("variant_id")
    assert out.loc["v0", "n_conditioned"] == 1 and out.loc["v0", "n_conditioned_cs"] == 2 and out.loc["v0", "n_conditioned_ld"] == 2
    assert out.loc["v5", "n_conditioned_cs"] == 2 and out.loc["v5", "n_conditioned_ld"] == 2
    diff, se, dof = _numpy_conditional(d, TAG, [CAUSAL, 5])
    np.testing.assert_allclose(out.loc["v0", "z_diff_ld_anc0_anc1"], diff / se, rtol=2e-3, atol=2e-3)
    assert out.loc["v0", "dof_ld"] == dof
