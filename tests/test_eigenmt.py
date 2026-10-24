import numpy as np
import pandas as pd
import torch

from localqtl import eigenmt
from tests.conftest import make_cohort_with_chroms


def test_compute_tests_independent_vs_duplicated():
    rng = np.random.default_rng(1)
    G = torch.tensor(rng.integers(0, 3, size=(50, 80)).astype(np.float32))
    m_indep = eigenmt.compute_tests(G)
    assert 40 <= m_indep <= 50
    Gd = torch.cat([G[:10]] * 5)
    assert eigenmt.compute_tests(Gd) < 50
    # several windows: sum over windows
    assert eigenmt.compute_tests(G, variant_window=20) >= eigenmt.compute_tests(G) - 5
    assert eigenmt.compute_tests(G[:0]) == 0


def test_find_num_eigs_and_bh():
    assert eigenmt.find_num_eigs([3.0, 1.0, 0.0], 4.0, var_thresh=0.74) == 1
    assert eigenmt.find_num_eigs([3.0, 1.0, 0.0], 4.0, var_thresh=0.99) == 2
    p = np.array([0.01, 0.04, 0.03, 0.5])
    adj = eigenmt.padjust_bh(p)
    # manual BH
    n = len(p); order = np.argsort(p); ranked = p[order] * n / np.arange(1, n + 1)
    expected = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n); out[order] = np.minimum(expected, 1)
    np.testing.assert_allclose(adj, out)


def test_run_eigenmt_per_phenotype():
    d = make_cohort_with_chroms()
    s = eigenmt.run_eigenmt(d["genotype_df"], d["variant_df"], d["phenotype_df"], d["phenotype_pos_df"],
                            window=10_000, device="cpu", verbose=False)
    assert isinstance(s, pd.Series) and set(s.index) == set(d["phenotype_df"].index)
    assert (s.loc[["gene1_0", "gene1_1"]] <= 8).all() and (s.loc["gene2_0"] <= 6) and (s > 0).all()
    i_s = pd.Series(d["rng"].normal(size=len(d["samples"])), index=d["samples"])
    s2 = eigenmt.run_eigenmt(d["genotype_df"], d["variant_df"], d["phenotype_df"], d["phenotype_pos_df"],
                             interaction_s=i_s, maf_threshold=0.05, window=10_000, device="cpu", verbose=False)
    assert (s2 <= s).all()
