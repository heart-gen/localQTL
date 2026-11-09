"""localqtl.stats: q-values from permutation results."""
import numpy as np
import pandas as pd

from src.localqtl.stats import calculate_qvalues
from src.localqtl.utils import SimpleLogger


def _perm_table(n=400, n_signal=40, seed=0):
    rng = np.random.default_rng(seed)
    p = rng.uniform(size=n)
    p[:n_signal] = rng.uniform(0, 1e-4, size=n_signal)
    return pd.DataFrame({"phenotype_id": [f"g{i}" for i in range(n)], "pval_perm": p, "pval_beta": p.copy(),
                         "beta_shape1": 1.0, "beta_shape2": 50.0})


def test_calculate_qvalues_uses_pval_perm_where_the_beta_fit_failed():
    """A degenerate beta fit leaves ``pval_beta`` NaN (``_beta_mle_on_p``); that phenotype is tested with its
    empirical permutation p-value, not set to p = 1 and dropped from the discoveries."""
    lg = SimpleLogger(verbose=False)
    df = _perm_table()
    ref = calculate_qvalues(df.copy(), logger=lg)
    broken = df.copy()
    broken.loc[[0, 100], ["pval_beta", "beta_shape1", "beta_shape2"]] = np.nan     # one signal, one null phenotype
    out = calculate_qvalues(broken, logger=lg)
    # pval_perm equals pval_beta in this table, so the fallback reproduces the complete table's q-values
    np.testing.assert_allclose(out["qval"].to_numpy(), ref["qval"].to_numpy(), rtol=1e-12)
    assert out.loc[0, "qval"] <= 0.05 < out.loc[100, "qval"]
    # the stored p-values are untouched and the phenotype without beta shapes has no nominal threshold
    assert np.isnan(out.loc[0, "pval_beta"]) and np.isnan(out.loc[0, "pval_nominal_threshold"])
    assert np.isfinite(out.loc[1, "pval_nominal_threshold"])


def test_calculate_qvalues_without_pval_perm_keeps_the_conservative_fill():
    """With no ``pval_perm`` column to fall back on, a NaN ``pval_beta`` is treated as p = 1 (unchanged)."""
    df = _perm_table().drop(columns="pval_perm")
    df.loc[0, "pval_beta"] = np.nan
    out = calculate_qvalues(df, logger=SimpleLogger(verbose=False))
    assert out.loc[0, "qval"] > 0.05
    assert (out["qval"].iloc[1:40] <= 0.05).all()
