"""
eigenMT (Davis et al., AJHG 2016): effective number of independent variants
per cis-window, used to correct interaction p-values in the top-association
output of :func:`localqtl.cis.map_nominal` (``run_eigenmt=True``).

Re-implementation adapted from tensorQTL ``eigenmt.py`` (BSD-3, Broad Institute).
"""
from __future__ import annotations

import time
from collections import OrderedDict

import numpy as np
import pandas as pd
import torch

from ._torch_utils import resolve_device
from .genotypeio import InputGeneratorCis
from .preproc import calculate_maf, filter_maf_interaction, impute_mean_and_filter, interaction_half_mask
from .utils import SimpleLogger

__all__ = [
    "lw_shrink",
    "find_num_eigs",
    "compute_tests",
    "run_eigenmt",
    "padjust_bh",
]


def lw_shrink(X_t: torch.Tensor):
    """
    Ledoit-Wolf shrunk covariance of ``X_t`` (samples x variants), or of a
    stack of windows ``(windows, samples, variants)``.  Returns
    ``(shrunk_cov_t, shrinkage_t)`` (adapted from scikit-learn).
    """
    device = X_t.device
    if X_t.dim() == 2:
        n_samples, n_features = X_t.shape
        X_t = X_t - X_t.mean(0)
        X2_t = X_t.pow(2)
        emp_cov_trace_sum = X2_t.sum() / n_samples
        delta_ = torch.mm(X_t.t(), X_t).pow(2).sum() / n_samples ** 2
        beta_ = torch.mm(X2_t.t(), X2_t).sum()
        beta = 1. / (n_features * n_samples) * (beta_ / n_samples - delta_)
        delta = delta_ - 1. * emp_cov_trace_sum ** 2 / n_features
        delta /= n_features
        beta = torch.min(beta, delta)
        shrinkage_t = torch.zeros((), device=device, dtype=X_t.dtype) if float(beta) == 0 else beta / delta
        emp_cov_t = torch.mm(X_t.t(), X_t) / n_samples
        mu_t = torch.trace(emp_cov_t) / n_features
        shrunk_cov_t = (1. - shrinkage_t) * emp_cov_t
        shrunk_cov_t.view(-1)[::n_features + 1] += shrinkage_t * mu_t
        return shrunk_cov_t, shrinkage_t
    # batched
    n_samples, n_features = X_t.shape[1:]
    X_t = X_t - X_t.mean(1, keepdim=True)
    X2_t = X_t.pow(2)
    emp_cov_trace_sum = X2_t.sum([1, 2]) / n_samples
    delta_ = torch.matmul(torch.transpose(X_t, 1, 2), X_t).pow(2).sum([1, 2]) / n_samples ** 2
    beta_ = torch.matmul(torch.transpose(X2_t, 1, 2), X2_t).sum([1, 2])
    beta = 1. / (n_features * n_samples) * (beta_ / n_samples - delta_)
    delta = delta_ - 1. * emp_cov_trace_sum ** 2 / n_features
    delta /= n_features
    beta = torch.min(beta, delta)
    shrinkage_t = torch.where(beta == 0, torch.zeros_like(beta), beta / delta)
    emp_cov_t = torch.matmul(torch.transpose(X_t, 1, 2), X_t) / n_samples
    mu_t = torch.diagonal(emp_cov_t, dim1=1, dim2=2).sum(1) / n_features
    shrunk_cov_t = (1 - shrinkage_t.reshape([-1, 1, 1])) * emp_cov_t
    diag = torch.diagonal(shrunk_cov_t, dim1=1, dim2=2)
    diag += (shrinkage_t * mu_t).unsqueeze(-1)
    return shrunk_cov_t, shrinkage_t


def find_num_eigs(eigenvalues, variance, var_thresh: float = 0.99) -> int:
    """Number of (largest) eigenvalues needed to explain ``var_thresh`` of ``variance``."""
    ev = np.sort(np.asarray(eigenvalues, dtype=np.float64))[::-1]
    running_sum = 0.0
    counter = 0
    while running_sum < variance * var_thresh and counter < ev.shape[0]:
        running_sum += ev[counter]
        counter += 1
    return int(counter)


def compute_tests(genotypes_t: torch.Tensor, var_thresh: float = 0.99, variant_window: int = 200) -> int:
    """Effective number of independent variants (``M_eff``) in ``genotypes_t`` (variants x samples)."""
    if genotypes_t.shape[0] == 0:
        return 0
    windows = torch.split(genotypes_t.float(), int(variant_window))
    eigenvalues = []
    if len(windows) > 1:
        shrunk_cov_t, _ = lw_shrink(torch.transpose(torch.stack(windows[:-1]), 1, 2))
        d = torch.diagonal(shrunk_cov_t, dim1=1, dim2=2).clamp_min(1e-12).pow(-0.5)
        shrunk_cor_t = shrunk_cov_t * d.unsqueeze(2) * d.unsqueeze(1)
        eigenvalues.extend(list(torch.linalg.eigvalsh(shrunk_cor_t).cpu().numpy()))
    shrunk_cov0_t, _ = lw_shrink(windows[-1].t())
    d0 = torch.diag(shrunk_cov0_t).clamp_min(1e-12).pow(-0.5)
    shrunk_cor0_t = shrunk_cov0_t * d0.unsqueeze(1) * d0.unsqueeze(0)
    eigenvalues.append(torch.linalg.eigvalsh(shrunk_cor0_t).cpu().numpy())

    m_eff = 0
    for ev, m in zip(eigenvalues, [w.shape[0] for w in windows]):
        ev = np.asarray(ev, dtype=np.float64).copy()
        ev[ev < 0] = 0
        m_eff += find_num_eigs(ev, m, var_thresh=var_thresh)
    return int(m_eff)


def run_eigenmt(genotype_df, variant_df, phenotype_df, phenotype_pos_df, interaction_s=None,
                maf_threshold: float = 0, var_thresh: float = 0.99, variant_window: int = 200,
                window: int = 1_000_000, device: str = "auto", verbose: bool = True,
                logger: SimpleLogger | None = None) -> pd.Series:
    """
    Standalone eigenMT correction: number of independent tests (``tests_emt``)
    for each phenotype's cis-window.  With ``interaction_s`` and
    ``maf_threshold > 0`` the interaction MAF filter (both halves of the
    interaction variable) is applied, as in ``map_nominal``.
    """
    dev = resolve_device(device)
    logger = logger or SimpleLogger(verbose=verbose)
    logger.write('eigenMT: estimating number of independent variants tested for each phenotype')
    logger.write(f'  * {phenotype_df.shape[1]} samples')
    logger.write(f'  * {phenotype_df.shape[0]} phenotypes')
    logger.write(f'  * {genotype_df.shape[0]} variants')

    interaction_mask_t = None
    if interaction_s is not None and maf_threshold > 0:
        interaction_mask_t = interaction_half_mask(interaction_s.loc[phenotype_df.columns].to_numpy()).to(dev)

    cols = pd.Index(genotype_df.columns.astype(str))
    genotype_ix = cols.get_indexer(pd.Index(phenotype_df.columns).astype(str))
    if (genotype_ix < 0).any():
        raise ValueError("phenotype samples missing from genotype_df")
    genotype_ix_t = torch.as_tensor(genotype_ix, device=dev)

    igc = InputGeneratorCis(genotype_df, variant_df, phenotype_df, phenotype_pos_df, window=window)
    start_time = time.time()
    m_eff = OrderedDict()
    for phenotype, genotypes, genotype_range, phenotype_id in igc.generate_data(verbose=verbose):
        genotypes_t = torch.tensor(np.asarray(genotypes), dtype=torch.float32, device=dev)[:, genotype_ix_t]
        genotypes_t, _, _ = impute_mean_and_filter(genotypes_t)
        if interaction_s is None:
            if maf_threshold > 0 and genotypes_t.shape[0] > 0:
                genotypes_t = genotypes_t[calculate_maf(genotypes_t) >= maf_threshold]
        else:
            genotypes_t, _ = filter_maf_interaction(genotypes_t, interaction_mask_t=interaction_mask_t,
                                                    maf_threshold_interaction=maf_threshold)
        m_eff[phenotype_id] = compute_tests(genotypes_t, var_thresh=var_thresh, variant_window=variant_window)
    logger.write(f'    time elapsed: {(time.time() - start_time) / 60:.2f} min')
    return pd.Series(m_eff, name='tests_emt', dtype=np.int64)


def padjust_bh(p) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values."""
    p = np.asarray(p, dtype=np.float64)
    if not np.all(np.isfinite(p)):
        raise ValueError('P values must be finite.')
    n = len(p)
    if n == 0:
        return p
    i = np.arange(n, 0, -1)
    o = np.argsort(p)[::-1]
    ro = np.argsort(o)
    return np.minimum(1, np.minimum.accumulate(np.float64(n) / i * p[o]))[ro]
