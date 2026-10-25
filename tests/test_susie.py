import numpy as np
import torch
import pytest

from src.localqtl.susie import (
    susie,
    susie_get_pip,
    susie_get_cs,
    get_x_attributes,
)


@pytest.fixture
def synthetic_causal():
    """Synthetic data with one known causal variant (index 3)."""
    torch.manual_seed(42)
    np.random.seed(42)
    n, p = 200, 20
    X = torch.randn(n, p)
    causal_idx = 3
    beta_true = torch.zeros(p)
    beta_true[causal_idx] = 2.0
    y = X @ beta_true + 0.5 * torch.randn(n)
    return X, y, causal_idx


@pytest.fixture
def synthetic_two_causal():
    """Synthetic data with two causal variants (indices 2, 7)."""
    torch.manual_seed(99)
    np.random.seed(99)
    n, p = 300, 25
    X = torch.randn(n, p)
    beta_true = torch.zeros(p)
    beta_true[2] = 1.5
    beta_true[7] = -1.8
    y = X @ beta_true + 0.5 * torch.randn(n)
    return X, y, [2, 7]


class TestSuSiE:
    def test_basic_runs(self, synthetic_causal):
        X, y, _ = synthetic_causal
        res = susie(X, y, L=5, max_iter=50)
        assert 'alpha' in res
        assert 'pip' in res
        assert 'sets' in res
        assert 'converged' in res
        assert 'elbo' in res

    def test_output_shapes(self, synthetic_causal):
        X, y, _ = synthetic_causal
        n, p = X.shape
        L = 5
        res = susie(X, y, L=L, max_iter=50)
        assert res['alpha'].shape == (L, p)
        assert res['mu'].shape == (L, p)
        assert len(res['pip']) == p

    def test_pip_highest_at_causal(self, synthetic_causal):
        """PIP should be highest at the true causal variant."""
        X, y, causal_idx = synthetic_causal
        res = susie(X, y, L=5, max_iter=100)
        pip = res['pip']
        assert np.argmax(pip) == causal_idx

    def test_pip_range(self, synthetic_causal):
        X, y, _ = synthetic_causal
        res = susie(X, y, L=5, max_iter=50)
        pip = res['pip']
        assert (pip >= 0).all() and (pip <= 1).all()

    def test_converges(self, synthetic_causal):
        X, y, _ = synthetic_causal
        res = susie(X, y, L=5, max_iter=200, tol=1e-3)
        assert res['converged']

    def test_elbo_monotone(self, synthetic_causal):
        """ELBO should be non-decreasing (within tolerance)."""
        X, y, _ = synthetic_causal
        res = susie(X, y, L=5, max_iter=100,
                    estimate_residual_variance=False)
        elbo = res['elbo']
        # Allow small numerical decreases (fp32 accumulation)
        diffs = np.diff(elbo)
        assert (diffs >= -1e-3).all(), f"ELBO decreased: {diffs.min()}"


class TestSuSiEGetCS:
    def test_credible_set_contains_causal(self, synthetic_causal):
        """At least one credible set should contain the true causal."""
        X, y, causal_idx = synthetic_causal
        res = susie(X, y, L=5, max_iter=100)
        cs = res['sets']
        if cs['cs'] is not None:
            found = any(causal_idx in indices for indices in cs['cs'].values())
            assert found, f"Causal index {causal_idx} not in any CS"

    def test_no_cs_when_no_signal(self):
        """Pure noise should yield no credible sets (or very few)."""
        torch.manual_seed(123)
        n, p = 200, 20
        X = torch.randn(n, p)
        y = torch.randn(n)
        res = susie(X, y, L=3, max_iter=50)
        # With pure noise, expect no CS or empty CS
        cs = res['sets']
        if cs['cs'] is not None:
            assert len(cs['cs']) <= 1  # At most one spurious CS

    def test_two_causal_detected(self, synthetic_two_causal):
        """With two causal variants, SuSiE should find at least two CS."""
        X, y, causal_idxs = synthetic_two_causal
        res = susie(X, y, L=5, max_iter=200)
        cs = res['sets']
        if cs['cs'] is not None:
            # Each causal should be in some CS
            for ci in causal_idxs:
                found = any(ci in indices for indices in cs['cs'].values())
                assert found, f"Causal index {ci} not found in any CS"

    def test_dedup_keeps_first_unique_credible_sets(self):
        """Duplicate CS rows should not displace later unique rows."""
        alpha = torch.tensor([
            [0.9, 0.1, 0.0, 0.0],
            [0.9, 0.1, 0.0, 0.0],
            [0.0, 0.0, 0.8, 0.2],
        ], dtype=torch.float32)
        res = {
            "alpha": alpha,
            "V": torch.ones(alpha.shape[0], dtype=torch.float32),
            "null_index": 0,
        }

        cs = susie_get_cs(res, coverage=0.95, dedup=True)

        assert cs["cs"] is not None
        assert list(cs["cs"]) == ["L1", "L3"]


class TestSuSiEGetPIP:
    def test_pip_standalone(self, synthetic_causal):
        X, y, causal_idx = synthetic_causal
        res = susie(X, y, L=5, max_iter=100)
        pip = susie_get_pip(res)
        assert pip.shape[0] == X.shape[1]
        assert pip[causal_idx] > 0.5

    def test_pip_sums_reasonable(self, synthetic_causal):
        """Sum of PIPs should be roughly equal to number of causal effects."""
        X, y, _ = synthetic_causal
        res = susie(X, y, L=5, max_iter=100)
        pip = susie_get_pip(res)
        # With 1 causal, sum should be close to 1 (not exact)
        assert 0.5 < float(pip.sum()) < 5.0


class TestGetXAttributes:
    def test_basic(self):
        X = torch.randn(50, 10)
        attr = get_x_attributes(X)
        assert 'd' in attr
        assert attr['d'].shape == (10,)
        assert (attr['d'] > 0).all()

    def test_no_center_no_scale(self):
        X = torch.randn(50, 10)
        attr = get_x_attributes(X, center=False, scale=False)
        assert (attr['scaled_center'] == 0).all()
        assert (attr['scaled_scale'] == 1).all()


# ---------------------------------------------------------------------------
# phenotype-specific (paired) covariates
# ---------------------------------------------------------------------------
class TestPairedCovariate:
    @staticmethod
    def _cohort(seed=21, n=140, m=12):
        import pandas as pd
        rng = np.random.default_rng(seed)
        samples = [f"S{i}" for i in range(n)]
        G = rng.integers(0, 3, size=(m, n)).astype(np.float32)
        gdf = pd.DataFrame(G, index=[f"v{i}" for i in range(m)], columns=samples)
        vdf = pd.DataFrame({"chrom": "chr1", "pos": 100 + 10 * np.arange(m)}, index=gdf.index)
        paired_vec = rng.normal(size=n)
        y1 = 1.1 * G[3] + 2.5 * paired_vec + rng.normal(size=n) * 0.4
        y2 = 1.1 * G[7] + rng.normal(size=n) * 0.4
        pdf = pd.DataFrame([y1, y2], index=["geneA", "geneB"], columns=samples).astype(np.float32)
        pos = pd.DataFrame({"chr": "chr1", "start": [150, 150], "end": [150, 150]}, index=pdf.index)
        cov = pd.DataFrame(rng.normal(size=(n, 2)), index=samples, columns=["c1", "c2"])
        paired = pd.DataFrame([paired_vec], index=["geneA"], columns=samples)
        return dict(gdf=gdf, vdf=vdf, pdf=pdf, pos=pos, cov=cov, paired=paired)

    def test_matches_adding_the_column_to_the_covariates(self):
        """A paired covariate equals putting that column in covariates_df for that phenotype."""
        import pandas as pd
        from localqtl import susie
        from localqtl.utils import SimpleLogger
        d = self._cohort()
        lg = SimpleLogger(verbose=False)
        kw = dict(L=3, window=1000, device="cpu", max_iter=300, logger=lg)
        paired_run = susie.map(d["gdf"], d["vdf"], d["pdf"], d["pos"], d["cov"],
                               paired_covariate_df=d["paired"], **kw)
        cov_aug = pd.concat(
            [d["cov"], pd.Series(d["paired"].loc["geneA"], name="paired")], axis=1)
        manual = susie.map(d["gdf"], d["vdf"], d["pdf"].loc[["geneA"]], d["pos"].loc[["geneA"]],
                           cov_aug, **kw)
        a = paired_run[paired_run["phenotype_id"] == "geneA"].reset_index(drop=True)
        b = manual.reset_index(drop=True)
        assert len(a) and a["variant_id"].tolist() == b["variant_id"].tolist()
        np.testing.assert_allclose(a["pip"].to_numpy(), b["pip"].to_numpy(), rtol=1e-4, atol=1e-6)
        # the phenotype that is not listed is unaffected
        plain = susie.map(d["gdf"], d["vdf"], d["pdf"].loc[["geneB"]], d["pos"].loc[["geneB"]],
                          d["cov"], **kw).reset_index(drop=True)
        gb = paired_run[paired_run["phenotype_id"] == "geneB"].reset_index(drop=True)
        assert gb["variant_id"].tolist() == plain["variant_id"].tolist()
        np.testing.assert_allclose(gb["pip"].to_numpy(), plain["pip"].to_numpy(), rtol=1e-4, atol=1e-6)

    def test_validation(self):
        import pandas as pd
        import pytest
        from localqtl import susie
        from localqtl.utils import SimpleLogger
        d = self._cohort()
        kw = dict(L=2, window=1000, device="cpu", logger=SimpleLogger(verbose=False))
        with pytest.raises(ValueError, match="phenotypes not in"):
            susie.map(d["gdf"], d["vdf"], d["pdf"], d["pos"], d["cov"],
                      paired_covariate_df=d["paired"].rename(index={"geneA": "ghost"}), **kw)
        with pytest.raises(ValueError, match="samples"):
            susie.map(d["gdf"], d["vdf"], d["pdf"], d["pos"], d["cov"],
                      paired_covariate_df=d["paired"].iloc[:, :5], **kw)
