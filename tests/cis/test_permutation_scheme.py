"""Permutation scheme: Freedman-Lane (M P M y) by default, tensorQTL's permute-raw-then-residualise (M P y) under
tensorqtl_flavor; both checked against a numpy reference on the same permutation indices, the flavor one also
against tensorQTL itself when it is installed.  Plus the true_dof guard."""
import numpy as np
import pandas as pd
import pytest
import torch

from src.localqtl import stats as lstats
from src.localqtl.cis._permute import PermutationStream, compute_perm_r2_max, perm_source, projectors_of
from src.localqtl.cis.permutations import map_permutations
from src.localqtl.regression_kernels import Residualizer
from src.localqtl.utils import SimpleLogger


def _setup(seed=0, n=60, m=15, k=4):
    rng = np.random.default_rng(seed)
    G = rng.integers(0, 3, size=(m, n)).astype(np.float32)
    C = rng.normal(size=(n, k)).astype(np.float32)
    y = (C @ rng.normal(size=k) * 2.0 + rng.normal(size=n)).astype(np.float32)   # covariates explain most of y
    return G, C, y


def _indices(n, nperm, seed):
    stream = PermutationStream(n_samples=n, nperm=nperm, device=torch.device("cpu"), chunk_size=7, seed=seed, mode="generator")
    return torch.cat(list(stream.iter_chunks(nperm)), dim=0).numpy()


def _reference(G, C, y_src, perms):
    """max over variants of corr(M g, M P y_src)^2 per permutation, M the residual projector of [1, C]."""
    n = len(y_src)
    X = np.c_[np.ones(n), C.astype(np.float64)]
    M = np.eye(n) - X @ np.linalg.pinv(X)
    Gr = M @ G.astype(np.float64).T                                   # (n, m)
    Gn = Gr / np.linalg.norm(Gr, axis=0, keepdims=True)
    out = np.empty(len(perms))
    for i, sel in enumerate(perms):
        yp = M @ y_src.astype(np.float64)[sel]
        yp = yp / np.linalg.norm(yp)
        out[i] = np.max((Gn.T @ yp) ** 2)
    return out


def _scan(G, C, y, nperm, seed, flavor):
    rez = Residualizer(torch.tensor(C), tensorqtl_flavor=flavor)
    y_t = torch.tensor(y)
    y_res = rez.transform(y_t.unsqueeze(0), center=True)[0].squeeze(0)
    G_res = rez.transform(torch.tensor(G), center=True)[0]
    stream = PermutationStream(n_samples=len(y), nperm=nperm, device=torch.device("cpu"), chunk_size=7, seed=seed, mode="generator")
    _, _, _, _, r2 = compute_perm_r2_max(y_resid=y_res, G_resid=G_res, H_resid=None, k_eff=rez.k_eff, perm_stream=stream,
                                         projectors=projectors_of(rez), y_source=perm_source(y_t, flavor))
    return r2.numpy()


def test_default_is_freedman_lane_m_p_m_y():
    G, C, y = _setup()
    perms = _indices(len(y), 50, seed=11)
    X = np.c_[np.ones(len(y)), C]
    y_res = y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
    np.testing.assert_allclose(_scan(G, C, y, 50, 11, flavor=False), _reference(G, C, y_res, perms), rtol=2e-4, atol=2e-5)


def test_tensorqtl_flavor_is_m_p_y():
    G, C, y = _setup()
    perms = _indices(len(y), 50, seed=11)
    np.testing.assert_allclose(_scan(G, C, y, 50, 11, flavor=True), _reference(G, C, y, perms), rtol=2e-4, atol=2e-5)


def test_the_two_schemes_differ_when_covariates_explain_variance():
    G, C, y = _setup()
    a, b = _scan(G, C, y, 50, 11, flavor=False), _scan(G, C, y, 50, 11, flavor=True)
    assert np.abs(a - b).max() > 1e-3


def test_perm_source_contract():
    y = torch.arange(6, dtype=torch.float32)
    assert perm_source(y, False) is None and perm_source(None, True) is None
    s = perm_source(y, True)
    assert torch.allclose(s, y - y.mean())
    Y = torch.stack([y, 2 * y])
    assert torch.allclose(perm_source(Y, True).mean(dim=1), torch.zeros(2), atol=1e-6)


def test_tensorqtl_flavor_matches_tensorqtl_itself():
    tq = pytest.importorskip("tensorqtl.core")
    G, C, y = _setup()
    perms = _indices(len(y), 50, seed=11)
    rez_tq = tq.Residualizer(torch.tensor(C))
    perm_t = torch.tensor(y)[torch.tensor(perms)]                       # (perm, n): tensorQTL permutes the raw phenotype
    r2_tq = tq.calculate_corr(torch.tensor(G), perm_t, residualizer=rez_tq).pow(2).max(0)[0].numpy()
    np.testing.assert_allclose(_scan(G, C, y, 50, 11, flavor=True), r2_tq, rtol=2e-4, atol=2e-5)


@pytest.mark.parametrize("grouped", [False, True])
def test_map_permutations_runs_in_both_schemes(random_cohort, grouped):
    d = random_cohort(seed=3, n=40, m=8)
    cov = pd.DataFrame(np.random.default_rng(1).normal(size=(40, 3)), index=d["samples"], columns=["c1", "c2", "c3"])
    ph = pd.concat([d["phenotype_df"], d["phenotype_df"].rename(index={"geneA": "geneB"}) * 0.5 + 0.1], axis=0)
    pos = pd.concat([d["phenotype_pos_df"], d["phenotype_pos_df"].rename(index={"geneA": "geneB"})], axis=0)
    group_s = pd.Series({"geneA": "g1", "geneB": "g1"}) if grouped else None
    out = {}
    for flavor in (False, True):
        df = map_permutations(d["genotype_df"], d["variant_df"], ph, pos, covariates_df=cov, group_s=group_s, device="cpu",
                              nperm=40, seed=5, tensorqtl_flavor=flavor, logger=SimpleLogger(verbose=False))
        assert df["true_dof"].dtype.kind == "f" and np.isfinite(df["true_dof"]).all()
        assert ((df["pval_perm"] > 0) & (df["pval_perm"] <= 1)).all()
        out[flavor] = df
    # the nominal statistics do not depend on the scheme
    pd.testing.assert_frame_equal(out[False][["phenotype_id", "variant_id", "slope", "pval_nominal"]].reset_index(drop=True),
                                  out[True][["phenotype_id", "variant_id", "slope", "pval_nominal"]].reset_index(drop=True))


def test_true_dof_guard(monkeypatch):
    r2 = np.random.default_rng(0).beta(1, 200, size=500)
    monkeypatch.setattr(lstats.scipy.optimize, "newton", lambda *a, **k: np.log(1e14))
    assert lstats._solve_true_dof(r2, 180.0) == 180.0
    a, b, dof = lstats.fit_beta_parameters(r2, 180.0)
    assert dof == 180.0 and np.isfinite(a) and np.isfinite(b)
    assert lstats._accept_true_dof(150.0, 180.0) == 150.0 and lstats._accept_true_dof(np.nan, 180.0) == 180.0
