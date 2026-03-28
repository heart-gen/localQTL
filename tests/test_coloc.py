import numpy as np
import torch
import pytest

from src.localqtl.coloc import coloc, coloc_from_summary


@pytest.fixture
def shared_causal_data():
    """Two traits sharing a single causal variant (index 3)."""
    torch.manual_seed(42)
    np.random.seed(42)
    n, m = 200, 30
    G = torch.randn(m, n)
    causal = 3
    y1 = 2.0 * G[causal] + 0.5 * torch.randn(n)
    y2 = 1.5 * G[causal] + 0.5 * torch.randn(n)
    return G, y1, y2, causal


@pytest.fixture
def independent_causal_data():
    """Two traits with independent causal variants (indices 3 and 10)."""
    torch.manual_seed(99)
    np.random.seed(99)
    n, m = 200, 30
    G = torch.randn(m, n)
    y1 = 2.0 * G[3] + 0.5 * torch.randn(n)
    y2 = 2.0 * G[10] + 0.5 * torch.randn(n)
    return G, y1, y2


class TestColoc:
    def test_output_shape(self, shared_causal_data):
        G, y1, y2, _ = shared_causal_data
        pp = coloc(G, G, y1, y2)
        assert pp.shape == (5,)

    def test_posteriors_sum_to_one(self, shared_causal_data):
        G, y1, y2, _ = shared_causal_data
        pp = coloc(G, G, y1, y2)
        assert abs(pp.sum().item() - 1.0) < 1e-4

    def test_shared_causal_high_h4(self, shared_causal_data):
        """When traits share a causal variant, PP_H4 should be high."""
        G, y1, y2, _ = shared_causal_data
        pp = coloc(G, G, y1, y2)
        pp_h4 = pp[4].item()
        assert pp_h4 > 0.5, f"PP_H4 = {pp_h4}, expected > 0.5 for shared causal"

    def test_independent_causal_low_h4(self, independent_causal_data):
        """When traits have independent causals, PP_H4 should be low."""
        G, y1, y2 = independent_causal_data
        pp = coloc(G, G, y1, y2)
        pp_h4 = pp[4].item()
        pp_h3 = pp[3].item()
        # H3 (both associated, different causal) should dominate over H4
        assert pp_h3 > pp_h4, f"PP_H3={pp_h3} should exceed PP_H4={pp_h4}"

    def test_no_signal_h0_dominates(self):
        """Pure noise: H0 should dominate."""
        torch.manual_seed(7)
        n, m = 200, 20
        G = torch.randn(m, n)
        y1 = torch.randn(n)
        y2 = torch.randn(n)
        pp = coloc(G, G, y1, y2)
        pp_h0 = pp[0].item()
        assert pp_h0 > 0.5, f"PP_H0 = {pp_h0}, expected > 0.5 for noise"

    def test_posteriors_nonneg(self, shared_causal_data):
        G, y1, y2, _ = shared_causal_data
        pp = coloc(G, G, y1, y2)
        assert (pp >= 0).all()

    def test_cc_mode_runs(self, shared_causal_data):
        """Case-control mode should run without error."""
        G, y1, y2, _ = shared_causal_data
        pp = coloc(G, G, y1, y2, mode='cc')
        assert pp.shape == (5,)
        assert abs(pp.sum().item() - 1.0) < 1e-4


class TestColocFromSummary:
    def test_basic_output(self):
        np.random.seed(42)
        m = 50
        beta1 = np.random.randn(m) * 0.1
        se1 = np.abs(np.random.randn(m) * 0.05) + 0.01
        beta2 = np.random.randn(m) * 0.1
        se2 = np.abs(np.random.randn(m) * 0.05) + 0.01
        maf = np.random.uniform(0.05, 0.5, m)

        result = coloc_from_summary(beta1, se1, beta2, se2, maf, n1=1000, n2=1000)
        assert set(result.keys()) == {'PP_H0', 'PP_H1', 'PP_H2', 'PP_H3', 'PP_H4', 'nsnps'}
        pp_sum = sum(result[f'PP_H{i}'] for i in range(5))
        assert abs(pp_sum - 1.0) < 1e-6

    def test_shared_signal(self):
        """Construct summary stats with a shared strong signal."""
        np.random.seed(0)
        m = 30
        beta1 = np.random.randn(m) * 0.01
        se1 = np.full(m, 0.05)
        beta2 = np.random.randn(m) * 0.01
        se2 = np.full(m, 0.05)
        maf = np.full(m, 0.3)

        # Inject shared strong effect at index 5
        beta1[5] = 0.8
        beta2[5] = 0.6

        result = coloc_from_summary(beta1, se1, beta2, se2, maf, n1=1000, n2=1000)
        assert result['PP_H4'] > 0.5

    def test_nsnps(self):
        m = 20
        result = coloc_from_summary(
            np.zeros(m), np.ones(m), np.zeros(m), np.ones(m),
            np.full(m, 0.3), n1=100, n2=100)
        assert result['nsnps'] == m
