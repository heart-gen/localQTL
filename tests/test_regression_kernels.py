import numpy as np
import torch
import pytest

from src.localqtl.regression_kernels import (
    run_batch_interaction_regression,
    run_batch_regression,
)


@pytest.fixture
def interaction_data():
    """Small synthetic data for y ~ g + i + g*i validation."""
    torch.manual_seed(42)
    np.random.seed(42)
    m, n = 5, 30  # 5 variants, 30 samples
    G = torch.randn(m, n)
    interaction = torch.randn(n)
    # True model: y = 0.5*g + 0.3*i + 0.8*g*i + noise
    g0 = G[0]
    y = 0.5 * g0 + 0.3 * interaction + 0.8 * (g0 * interaction) + 0.1 * torch.randn(n)
    return y, G, interaction, m, n


def _ols_interaction_numpy(y, g, interaction):
    """Reference OLS for y ~ g + i + g*i using numpy (no intercept, matching kernel)."""
    n = len(y)
    gi = g * interaction
    X = np.column_stack([g, interaction, gi])
    XtX = X.T @ X
    Xty = X.T @ y
    betas = np.linalg.solve(XtX, Xty)
    y_hat = X @ betas
    resid = y - y_hat
    dof = n - X.shape[1]
    sigma2 = (resid @ resid) / dof
    XtX_inv = np.linalg.inv(XtX)
    ses = np.sqrt(np.diag(XtX_inv) * sigma2)
    tstats = betas / ses
    # Return: b_g, se_g, tstat_g, b_i, se_i, tstat_i, b_gi, se_gi, tstat_gi
    return betas[0], ses[0], tstats[0], betas[1], ses[1], tstats[1], betas[2], ses[2], tstats[2]


class TestRunBatchInteractionRegression:
    def test_basic_output_shape(self, interaction_data):
        y, G, interaction, m, n = interaction_data
        result = run_batch_interaction_regression(y, G, interaction, device="cpu")
        for key in ("b_g", "se_g", "tstat_g", "b_gi", "se_gi", "tstat_gi"):
            assert result[key].shape == (m,), f"{key} has wrong shape"
        assert isinstance(result["dof"], int)

    def test_matches_numpy_ols(self, interaction_data):
        """Verify kernel matches manual numpy OLS for each variant."""
        y, G, interaction, m, n = interaction_data
        result = run_batch_interaction_regression(y, G, interaction, device="cpu")

        y_np = y.numpy()
        i_np = interaction.numpy()

        for v in range(m):
            g_np = G[v].numpy()
            ref = _ols_interaction_numpy(y_np, g_np, i_np)
            b_g, se_g, t_g, b_i, se_i, t_i, b_gi, se_gi, t_gi = ref

            assert abs(result["b_g"][v].item() - b_g) < 1e-4, f"b_g mismatch at variant {v}"
            assert abs(result["se_g"][v].item() - se_g) < 1e-4, f"se_g mismatch at variant {v}"
            assert abs(result["tstat_g"][v].item() - t_g) < 1e-3, f"tstat_g mismatch at variant {v}"
            assert abs(result["b_i"][v].item() - b_i) < 1e-4, f"b_i mismatch at variant {v}"
            assert abs(result["b_gi"][v].item() - b_gi) < 1e-4, f"b_gi mismatch at variant {v}"
            assert abs(result["tstat_gi"][v].item() - t_gi) < 1e-3, f"tstat_gi mismatch at variant {v}"

    def test_dof_with_k_eff(self, interaction_data):
        y, G, interaction, m, n = interaction_data
        result_0 = run_batch_interaction_regression(y, G, interaction, k_eff=0, device="cpu")
        result_5 = run_batch_interaction_regression(y, G, interaction, k_eff=5, device="cpu")
        # DOF should decrease by 5
        assert result_0["dof"] - result_5["dof"] == 5

    def test_multi_interaction(self):
        """Test with multiple interaction terms (n, ni) where ni > 1."""
        torch.manual_seed(7)
        m, n, ni = 3, 20, 2
        y = torch.randn(n)
        G = torch.randn(m, n)
        interaction = torch.randn(n, ni)
        result = run_batch_interaction_regression(y, G, interaction, device="cpu")
        # b_i and b_gi should be (m, ni)
        assert result["b_i"].shape == (m, ni)
        assert result["b_gi"].shape == (m, ni)
        assert result["b_g"].shape == (m,)

    def test_ses_positive(self, interaction_data):
        y, G, interaction, m, n = interaction_data
        result = run_batch_interaction_regression(y, G, interaction, device="cpu")
        assert (result["se_g"] > 0).all()
        assert (result["se_i"] > 0).all()
        assert (result["se_gi"] > 0).all()

    def test_strong_interaction_signal(self):
        """When the true DGP has a strong g*i effect, tstat_gi should be large."""
        torch.manual_seed(0)
        m, n = 1, 100
        G = torch.randn(m, n)
        i_vec = torch.randn(n)
        y = 5.0 * (G[0] * i_vec) + 0.1 * torch.randn(n)
        result = run_batch_interaction_regression(y, G, i_vec, device="cpu")
        assert abs(result["tstat_gi"][0].item()) > 5.0


class TestRunBatchRegression:
    def test_basic_no_H(self):
        torch.manual_seed(1)
        m, n = 4, 20
        y = torch.randn(n)
        G = torch.randn(m, n)
        betas, ses, tstats = run_batch_regression(y, G, device="cpu")
        assert betas.shape == (m, 1)
        assert ses.shape == (m, 1)
        assert (ses > 0).all()

    def test_with_H(self):
        torch.manual_seed(2)
        m, n, pH = 4, 20, 2
        y = torch.randn(n)
        G = torch.randn(m, n)
        H = torch.randn(m, n, pH)
        betas, ses, tstats = run_batch_regression(y, G, H, device="cpu")
        assert betas.shape == (m, 1 + pH)
        assert tstats.shape == (m, 1 + pH)
