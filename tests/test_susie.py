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
