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


from src.localqtl.regression_kernels import run_batch_dosage_regression


def _ols_dosage_numpy(y, D, L):
    """Reference OLS for y ~ D_1..D_K + L_1..L_q (no intercept, matching the kernel)."""
    X = np.column_stack([D, L]) if L is not None and L.shape[1] else D
    K = D.shape[1]
    XtX_inv = np.linalg.inv(X.T @ X)
    betas = XtX_inv @ X.T @ y
    resid = y - X @ betas
    dof = len(y) - X.shape[1]
    sigma2 = resid @ resid / dof
    cov = XtX_inv * sigma2
    ses = np.sqrt(np.diag(cov))
    b = betas[:K]
    f = b @ np.linalg.solve(cov[:K, :K], b) / K
    return b, ses[:K], b / ses[:K], cov[:K, :K], f, dof


class TestRunBatchDosageRegression:
    @pytest.mark.parametrize("K", [2, 3])
    def test_matches_numpy_ols(self, K):
        torch.manual_seed(0)
        m, n = 4, 40
        G = torch.randint(0, 3, (m, n)).float()
        L = torch.distributions.Dirichlet(torch.ones(K)).sample((m, n)) * 2.0   # expected counts, sum 2
        D = G.unsqueeze(-1) * L / 2.0
        L_adj = L[:, :, :-1]
        beta = torch.linspace(0.2, 1.0, K)
        y = (D[1] @ beta) + 0.3 * torch.randn(n)
        res = run_batch_dosage_regression(y, D, L_adj, k_eff=0, device="cpu")
        assert res["betas"].shape == (m, K) and res["cov"].shape == (m, K, K)
        assert res["fstat"].shape == (m,) and res["dof"] == n - (2 * K - 1)
        for v in range(m):
            b, se, t, cov, f, dof = _ols_dosage_numpy(
                y.numpy().astype(np.float64), D[v].numpy().astype(np.float64),
                L_adj[v].numpy().astype(np.float64))
            np.testing.assert_allclose(res["betas"][v].numpy(), b, rtol=1e-3, atol=1e-4)
            np.testing.assert_allclose(res["ses"][v].numpy(), se, rtol=1e-3, atol=1e-4)
            np.testing.assert_allclose(res["cov"][v].numpy(), cov, rtol=2e-3, atol=1e-5)
            np.testing.assert_allclose(res["fstat"][v].item(), f, rtol=2e-3)
        # the simulated variant recovers its ancestry-specific effects
        np.testing.assert_allclose(res["betas"][1].numpy(), beta.numpy(), atol=0.25)

    def test_without_local_ancestry_columns_and_k_eff(self):
        torch.manual_seed(1)
        m, n, K = 3, 25, 2
        D = torch.rand(m, n, K)
        y = torch.randn(n)
        res = run_batch_dosage_regression(y, D, None, k_eff=3, device="cpu")
        assert res["dof"] == n - 3 - K
        b, se, *_ = _ols_dosage_numpy(y.numpy().astype(np.float64), D[0].numpy().astype(np.float64), None)
        np.testing.assert_allclose(res["betas"][0].numpy(), b, rtol=1e-3, atol=1e-4)
        assert (res["ses"] > 0).all()


def test_dosage_joint_f_block_refit_and_nested_ols_agree_on_a_residualised_design():
    """The permutation scan compares its null with the block estimator of the joint F (``dosage_fstat`` on the
    ctx of ``prep_ctx_for_dosage_perm``) and reports the refit (``run_batch_dosage_regression``).  With
    projected-out covariates and local-ancestry columns both are the F of the nested comparison
    ``y ~ 1 + C + L`` against ``y ~ 1 + C + L + D_1..D_K`` in float64, with ``(K, n - 1 - k - (2K - 1))``
    degrees of freedom."""
    from src.localqtl.cis.common import residualize_batch_dosage
    from src.localqtl.regression_kernels import Residualizer, dosage_fstat, prep_ctx_for_dosage_perm
    rng = np.random.default_rng(2)
    n, m, k, K = 120, 12, 6, 2
    G = rng.integers(0, 3, size=(m, n)).astype(np.float32)
    H = rng.integers(0, 3, size=(m, n, 1)).astype(np.float32)         # ancestry-0 haplotype count
    C = rng.normal(size=(n, k)).astype(np.float32)
    y = (C @ rng.normal(size=k) + 0.6 * G[2] + rng.normal(size=n)).astype(np.float32)
    rez = Residualizer(torch.tensor(C))
    y_res, D, L = residualize_batch_dosage(torch.tensor(y), torch.tensor(G), torch.tensor(H), rez,
                                           center=True, group=False)
    ctx = prep_ctx_for_dosage_perm(D, L, k_eff=rez.k_eff)
    res = run_batch_dosage_regression(y_res, D, L, k_eff=rez.k_eff, device="cpu")
    dof = n - 1 - k - (2 * K - 1)
    assert ctx["K"] == K and ctx["dof"] == res["dof"] == dof
    f_block = dosage_fstat(ctx, y_res)
    torch.testing.assert_close(f_block, res["fstat"], rtol=1e-3, atol=1e-5)

    def rss(X):
        r = y.astype(np.float64) - X @ np.linalg.lstsq(X, y.astype(np.float64), rcond=None)[0]
        return float(r @ r)

    base = np.column_stack([np.ones(n), C.astype(np.float64)])
    for v in range(m):
        l0 = H[v, :, 0].astype(np.float64)
        d = np.column_stack([G[v] * l0 / 2.0, G[v] * (2.0 - l0) / 2.0])
        rss0, rss1 = rss(np.column_stack([base, l0])), rss(np.column_stack([base, l0, d]))
        f_ref = ((rss0 - rss1) / K) / (rss1 / dof)
        np.testing.assert_allclose(res["fstat"][v].item(), f_ref, rtol=5e-3, atol=1e-4)
        np.testing.assert_allclose(f_block[v].item(), f_ref, rtol=5e-3, atol=1e-4)


@pytest.mark.parametrize("with_h", [False, True])
def test_single_predictor_kernels_report_the_ols_standard_error_whatever_the_sign(with_h):
    """The correlation kernels behind the interaction scans, the pooled lead and the permutation
    context (``run_batch_regression_with_permutations``, ``prep_ctx_for_perm``) return the OLS
    standard error for negative effects as for positive ones: ``slope / se == t``, and all three
    agree with ``run_batch_regression``.  (``se = |beta| / clamp(t, 1e-8)`` gave ``|beta| * 1e8``
    wherever the effect was negative.)"""
    from src.localqtl import regression_kernels as rk
    torch.manual_seed(7)
    n, m = 120, 8
    G = torch.randn(m, n); G = G - G.mean(dim=1, keepdim=True)
    H = None
    if with_h:
        H = torch.randn(m, n, 2); H = H - H.mean(dim=1, keepdim=True)
    y = torch.randn(n); y = y - y.mean()
    y = y - 0.6 * G[0] + 0.6 * G[1]                               # variant 0 negative, variant 1 positive
    b_ref, se_ref, t_ref = run_batch_regression(y=y, G=G, H=H, k_eff=1, device="cpu")
    kernels = [rk.run_batch_regression_with_permutations(y=y, G=G, H=H, y_perm=None, k_eff=1, device="cpu")[:3]]
    if with_h:
        kernels.append(rk.prep_ctx_for_perm(y, G, H, k_eff=1)[1:])
    for b, se, t in kernels:
        assert b[0, 0] < 0 < b[1, 0] and t[0, 0] < -3 and t[1, 0] > 3
        assert (se > 0).all() and torch.isfinite(se).all()
        torch.testing.assert_close(b[:, 0] / se[:, 0], t[:, 0], rtol=1e-4, atol=1e-5)
        torch.testing.assert_close(se[:, 0], se_ref[:, 0].to(se.dtype), rtol=1e-3, atol=1e-6)
        torch.testing.assert_close(b[:, 0], b_ref[:, 0].to(b.dtype), rtol=1e-3, atol=1e-5)


def test_single_predictor_kernel_standard_error_is_positive_at_a_zero_effect():
    """A predictor orthogonal to the phenotype has slope 0 and a positive standard error."""
    from src.localqtl import regression_kernels as rk
    n = 40
    g = torch.tensor([1.0, -1.0] * (n // 2))
    y = torch.tensor([1.0, 1.0, -1.0, -1.0] * (n // 4))            # g'y == 0 exactly
    b, se, t, _ = rk.run_batch_regression_with_permutations(y=y, G=g.unsqueeze(0), H=None, y_perm=None, k_eff=1,
                                                            device="cpu")
    assert b.item() == 0.0 and t.item() == 0.0
    assert se.item() == pytest.approx(float(np.sqrt(n / (n - 2) / n)), rel=1e-5)
