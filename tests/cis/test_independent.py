import numpy as np
import pytest
import torch
import pandas as pd

from src.localqtl.cis.independent import map_independent
from src.localqtl.cis.permutations import map_permutations
from src.localqtl.utils import SimpleLogger


def test_map_independent_identifies_forward_backward_hits(toy_data):
    torch.manual_seed(0)
    base = map_permutations(
        genotype_df=toy_data["genotype_df"],
        variant_df=toy_data["variant_df"],
        phenotype_df=toy_data["phenotype_df"],
        phenotype_pos_df=toy_data["phenotype_pos_df"],
        covariates_df=None,
        device="cpu",
        nperm=4,
        seed=321,
        logger=SimpleLogger(verbose=False),
    )

    cis_df = base.copy()
    cis_df["qval"] = np.linspace(0.01, 0.03, len(cis_df))

    result = map_independent(
        genotype_df=toy_data["genotype_df"],
        variant_df=toy_data["variant_df"],
        cis_df=cis_df,
        phenotype_df=toy_data["phenotype_df"],
        phenotype_pos_df=toy_data["phenotype_pos_df"],
        covariates_df=None,
        device="cpu",
        nperm=3,
        seed=123,
        logger=SimpleLogger(verbose=False),
    )

    assert not result.empty
    assert {"phenotype_id", "variant_id", "rank", "pval_beta"}.issubset(result.columns)
    # ranks should start at 1 for each phenotype
    grouped = result.groupby("phenotype_id")["rank"]
    for _, ranks in grouped:
        assert ranks.iloc[0] == 1


def test_interaction_stage2_extends_permutations(monkeypatch, toy_hap_data_2anc):
    class FakeStream:
        total_generated = 0

        def __init__(self, n_samples, nperm, device, chunk_size, seed=None, mode="generator"):
            self.n_samples = int(n_samples)
            self.nperm = int(nperm)
            self.chunk_size = max(1, int(chunk_size))
            self.device = torch.device(device) if not isinstance(device, torch.device) else device

        def iter_chunks(self, limit=None):
            total = self.nperm if limit is None else int(limit)
            produced = 0
            while produced < total:
                count = min(self.chunk_size, total - produced)
                base = torch.arange(self.n_samples, device=self.device, dtype=torch.long)
                chunk = base.repeat(count, 1)
                FakeStream.total_generated += count
                produced += count
                yield chunk

    monkeypatch.setattr("src.localqtl.cis.independent.PermutationStream", FakeStream)

    toy = toy_hap_data_2anc
    cis_df = pd.DataFrame({
        "phenotype_id": ["geneA"],
        "variant_id": [toy["variant_df"].index[0]],
        "pval_beta": [0.5],
        "qval": [0.01],
    })

    map_independent(
        genotype_df=toy["genotype_df"],
        variant_df=toy["variant_df"],
        cis_df=cis_df,
        phenotype_df=toy["phenotype_df"],
        phenotype_pos_df=toy["phenotype_pos_df"],
        covariates_df=None,
        haplotypes=toy["haplotypes"],
        loci_df=toy["loci_df"],
        device="cpu",
        nperm=6,
        nperm_stage1=2,
        seed=123,
        logger=SimpleLogger(verbose=False),
        ancestry_model="interaction",
    )

    assert FakeStream.total_generated == 6


def test_interaction_permutation_chunks_shared_across_ancestries(monkeypatch, toy_hap_data_3anc):
    from src.localqtl.cis import independent as indep_mod

    calls = []
    orig = indep_mod.perm_chunk_r2

    def wrapped(ctx, H, G, y_perm, mixed_precision=None):
        sig = hash(y_perm.detach().cpu().numpy().tobytes())
        calls.append(sig)
        return orig(ctx, H, G, y_perm, mixed_precision=mixed_precision)

    monkeypatch.setattr("src.localqtl.cis.independent.perm_chunk_r2", wrapped)

    toy = toy_hap_data_3anc
    cis_df = pd.DataFrame({
        "phenotype_id": ["geneA"],
        "variant_id": [toy["variant_df"].index[0]],
        "pval_beta": [0.5],
        "qval": [0.01],
    })

    map_independent(
        genotype_df=toy["genotype_df"],
        variant_df=toy["variant_df"],
        cis_df=cis_df,
        phenotype_df=toy["phenotype_df"],
        phenotype_pos_df=toy["phenotype_pos_df"],
        covariates_df=None,
        haplotypes=toy["haplotypes"],
        loci_df=toy["loci_df"],
        device="cpu",
        nperm=4,
        nperm_stage1=2,
        seed=456,
        perm_chunk=2,
        logger=SimpleLogger(verbose=False),
        ancestry_model="interaction",
    )

    pH = 2  # toy_hap_data_3anc -> H_resid is trimmed to K-1 = 2
    assert len(calls) >= pH
    assert len(calls) % pH == 0
    for i in range(0, len(calls), pH):
        block = calls[i:i + pH]
        assert all(sig == block[0] for sig in block)


# ---------------------------------------------------------------------------
# phenotype-specific (paired) covariates
# ---------------------------------------------------------------------------
def _two_signal_cohort(seed=12, n=150, m=14):
    """One phenotype with two independent cis signals and a strong paired confounder."""
    rng = np.random.default_rng(seed)
    samples = [f"S{i}" for i in range(n)]
    G = rng.integers(0, 3, size=(m, n)).astype(np.float32)
    gdf = pd.DataFrame(G, index=[f"v{i}" for i in range(m)], columns=samples)
    vdf = pd.DataFrame({"chrom": "chr1", "pos": 100 + 10 * np.arange(m)}, index=gdf.index)
    paired_vec = rng.normal(size=n)
    y = 1.0 * G[0] + 0.9 * G[5] + 2.0 * paired_vec + rng.normal(size=n) * 0.5
    pdf = pd.DataFrame([y], index=["geneA"], columns=samples).astype(np.float32)
    pos = pd.DataFrame({"chr": ["chr1"], "start": [170], "end": [170]}, index=["geneA"])
    cov = pd.DataFrame(rng.normal(size=(n, 2)), index=samples, columns=["c1", "c2"])
    paired = pd.DataFrame([paired_vec], index=["geneA"], columns=samples)
    return dict(genotype_df=gdf, variant_df=vdf, phenotype_df=pdf, phenotype_pos_df=pos,
                covariates_df=cov, paired=paired, G=G, y=y, samples=samples, rng=rng)


def test_map_independent_paired_covariate_matches_ols():
    """The conditional (rank >= 2) fits include the phenotype-specific covariate."""
    from tests.conftest import ols_fit
    d = _two_signal_cohort()
    lg = SimpleLogger(verbose=False)
    kw = dict(window=1000, device="cpu", logger=lg, seed=1)
    perm = map_permutations(d["genotype_df"], d["variant_df"], d["phenotype_df"],
                            d["phenotype_pos_df"], d["covariates_df"],
                            paired_covariate_df=d["paired"], nperm=200, **kw)
    cis_df = perm.copy()
    cis_df["qval"] = 0.001
    ind = map_independent(d["genotype_df"], d["variant_df"], cis_df, d["phenotype_df"],
                          d["phenotype_pos_df"], d["covariates_df"],
                          paired_covariate_df=d["paired"], nperm=200, **kw)
    assert len(ind) >= 2 and list(ind["rank"]) == list(range(1, len(ind) + 1))
    seed_v, second = ind["variant_id"].iloc[0], ind.iloc[1]
    ids = list(d["genotype_df"].index)
    X = np.column_stack([d["G"][ids.index(second["variant_id"])], d["G"][ids.index(seed_v)],
                         d["covariates_df"].values, d["paired"].values[0]])
    b, se, _ = ols_fit(d["y"], X)
    np.testing.assert_allclose(second["slope"], b[0], rtol=1e-3)
    np.testing.assert_allclose(second["slope_se"], se[0], rtol=1e-3)
    # dropping the paired covariate changes the conditional estimate materially
    b_no, _, _ = ols_fit(d["y"], np.column_stack([
        d["G"][ids.index(second["variant_id"])], d["G"][ids.index(seed_v)],
        d["covariates_df"].values]))
    assert abs(b_no[0] - b[0]) > 0.1


def test_map_independent_paired_covariate_validation():
    d = _two_signal_cohort()
    lg = SimpleLogger(verbose=False)
    cis_df = pd.DataFrame({"phenotype_id": ["geneA"], "variant_id": ["v0"],
                           "pval_beta": [1e-12], "qval": [1e-12]})
    base = dict(genotype_df=d["genotype_df"], variant_df=d["variant_df"], cis_df=cis_df,
                phenotype_df=d["phenotype_df"], phenotype_pos_df=d["phenotype_pos_df"],
                covariates_df=d["covariates_df"], nperm=20, window=1000, device="cpu", logger=lg)
    with pytest.raises(ValueError, match="not in\n?.*phenotype_df|phenotypes not in"):
        map_independent(paired_covariate_df=d["paired"].rename(index={"geneA": "ghost"}), **base)
    with pytest.raises(ValueError, match="groups"):
        map_independent(paired_covariate_df=d["paired"],
                        group_s=pd.Series({"geneA": "grp"}), **base)
    with pytest.raises(ValueError, match="samples"):
        map_independent(paired_covariate_df=d["paired"].iloc[:, :5], **base)
