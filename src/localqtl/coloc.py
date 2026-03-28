# COLOC (Colocalisation Analysis)
#
# Approximate Bayes Factor (ABF) colocalization analysis.
# Reference: Giambartolomei et al., PLoS Genetics, 2014
#
# Port of https://github.com/broadinstitute/tensorqtl/blob/master/tensorqtl/coloc.py
# Adapted for localQTL: pure Python, uses localQTL device management.

import numpy as np
import pandas as pd
import scipy.stats as stats
import torch
import time

from ._torch_utils import resolve_device
from .regression_kernels import Residualizer
from .preproc import impute_mean_and_filter
from .utils import SimpleLogger

__all__ = [
    "coloc",
    "coloc_from_summary",
    "run_pairs",
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _logsumexp(x, dim=0):
    mmax, _ = torch.max(x, dim=dim, keepdim=True)
    return mmax + (x - mmax).exp().sum(dim, keepdim=True).log()


def _logdiff(x, y, dim=0):
    xmax, _ = torch.max(x, dim=dim, keepdim=True)
    ymax, _ = torch.max(y, dim=dim, keepdim=True)
    mmax = torch.max(xmax, ymax)
    return mmax + ((x - mmax).exp() - (y - mmax).exp()).log()


def _center_normalize(x, dim=1):
    """Center and L2-normalize along dim."""
    x = x - x.mean(dim=dim, keepdim=True)
    norm = x.norm(dim=dim, keepdim=True)
    norm = torch.clamp(norm, min=1e-12)
    return x / norm


def _calculate_corr(genotype_t, phenotype_t, residualizer=None, return_var=False):
    """Calculate correlation between (residualized) genotypes and phenotypes."""
    if residualizer is not None:
        (genotype_res_t,) = residualizer.transform(genotype_t, center=True)
        (phenotype_res_t,) = residualizer.transform(phenotype_t, center=True)
    else:
        genotype_res_t = genotype_t
        phenotype_res_t = phenotype_t

    if return_var:
        genotype_var_t = genotype_res_t.var(1)
        phenotype_var_t = phenotype_res_t.var(1)

    genotype_res_t = _center_normalize(genotype_res_t, dim=1)
    phenotype_res_t = _center_normalize(phenotype_res_t, dim=1)

    if return_var:
        return torch.mm(genotype_res_t, phenotype_res_t.t()), genotype_var_t, phenotype_var_t
    else:
        return torch.mm(genotype_res_t, phenotype_res_t.t())


def _calculate_maf(genotypes_t):
    """Minor allele frequency from dosage matrix (variants x samples)."""
    af = genotypes_t.sum(1) / (2 * genotypes_t.shape[1])
    return torch.minimum(af, 1 - af)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def coloc(genotypes1_t, genotypes2_t, phenotype1_t, phenotype2_t,
          residualizer1=None, residualizer2=None, mode='beta',
          p1=1e-4, p2=1e-4, p12=1e-5):
    """
    COLOC ABF: compute H0-H4 posterior probabilities from raw data.

    Parameters
    ----------
    genotypes1_t : torch.Tensor
        Genotypes for trait 1, shape (m, n1).
    genotypes2_t : torch.Tensor
        Genotypes for trait 2, shape (m, n2).
    phenotype1_t : torch.Tensor
        Phenotype 1, shape (n1,).
    phenotype2_t : torch.Tensor
        Phenotype 2, shape (n2,) or (k, n2) for k phenotypes.
    residualizer1 : Residualizer, optional
        Covariate residualizer for trait 1.
    residualizer2 : Residualizer, optional
        Covariate residualizer for trait 2.
    mode : str
        ``'beta'`` for beta-based ABF, ``'cc'`` for case-control (z-score based).
    p1, p2, p12 : float
        Prior probabilities for association with trait 1 only, trait 2 only,
        and both traits (shared causal variant).

    Returns
    -------
    torch.Tensor
        Posterior probabilities [PP_H0, PP_H1, PP_H2, PP_H3, PP_H4].
    """
    assert phenotype1_t.dim() == 1
    device = genotypes1_t.device

    # --- Trait 1 ---
    if mode == 'beta':
        r_nominal_t, genotype_var_t, phenotype_var_t = _calculate_corr(
            genotypes1_t, phenotype1_t.reshape(1, -1), residualizer1, return_var=True)
        r_nominal_t = r_nominal_t.squeeze()
        var_ratio_t = phenotype_var_t.reshape(1, -1) / genotype_var_t.reshape(-1, 1)
    else:
        r_nominal_t = _calculate_corr(
            genotypes1_t, phenotype1_t.reshape(1, -1), residualizer1, return_var=False).squeeze()
    r2_nominal_t = r_nominal_t.double().pow(2)

    dof1 = residualizer1.dof if residualizer1 is not None else phenotype1_t.shape[0] - 2

    if mode == 'beta':
        tstat2_t = r2_nominal_t * dof1 / (1 - r2_nominal_t)
        beta2_t = r2_nominal_t * var_ratio_t.squeeze()
        beta_var_t = beta2_t / tstat2_t
        var_prior = 0.0225 * phenotype_var_t
        r = var_prior / (var_prior + beta_var_t)
        l1 = 0.5 * ((1 - r).log() + r * tstat2_t)
    else:
        tstat_t = r_nominal_t * torch.sqrt(torch.tensor(dof1, dtype=torch.float64, device=device)
                                           / (1 - r2_nominal_t))
        p_vals = stats.t.cdf(-np.abs(tstat_t.cpu().numpy()), dof1)
        maf_t = _calculate_maf(genotypes1_t)
        N = phenotype1_t.shape[0]
        v = 1 / (2 * N * maf_t * (1 - maf_t))
        z2_t = torch.tensor(stats.norm.isf(p_vals) ** 2, dtype=torch.float64, device=device)
        r = 0.0225 / (0.0225 + v.double())
        l1 = 0.5 * ((1 - r).log() + r * z2_t)

    # --- Trait 2 ---
    if phenotype2_t.dim() == 1:
        num_phenotypes = 1
        num_samples = phenotype2_t.shape[0]
        phenotype2_t = phenotype2_t.reshape(1, -1)
    else:
        num_phenotypes, num_samples = phenotype2_t.shape

    if mode == 'beta':
        r_nominal_t, genotype_var_t, phenotype_var_t = _calculate_corr(
            genotypes2_t, phenotype2_t, residualizer2, return_var=True)
        r_nominal_t = r_nominal_t.squeeze()
        var_ratio_t = phenotype_var_t.reshape(1, -1) / genotype_var_t.reshape(-1, 1)
    else:
        r_nominal_t = _calculate_corr(
            genotypes2_t, phenotype2_t, residualizer2, return_var=False).squeeze()
    r2_nominal_t = r_nominal_t.double().pow(2)

    dof2 = residualizer2.dof if residualizer2 is not None else num_samples - 2

    if mode == 'beta':
        tstat2_t = r2_nominal_t * dof2 / (1 - r2_nominal_t)
        beta2_t = r2_nominal_t * var_ratio_t.squeeze()
        beta_var_t = beta2_t / tstat2_t
        var_prior = 0.0225 * phenotype_var_t
        r = var_prior / (var_prior + beta_var_t)
        l2 = 0.5 * ((1 - r).log() + r * tstat2_t)
    else:
        tstat_t = r_nominal_t * torch.sqrt(torch.tensor(dof2, dtype=torch.float64, device=device)
                                           / (1 - r2_nominal_t))
        p_vals = stats.t.cdf(-np.abs(tstat_t.cpu().numpy()), dof2)
        maf_t = _calculate_maf(genotypes2_t)
        v = 1 / (2 * num_samples * maf_t * (1 - maf_t))
        z2_t = torch.tensor(stats.norm.isf(p_vals) ** 2, dtype=torch.float64, device=device)
        r = 0.0225 / (0.0225 + v.double())
        if num_phenotypes > 1:
            r = r.reshape(-1, 1)
        l2 = 0.5 * ((1 - r).log() + r * z2_t)

    # --- Combine ABFs ---
    if num_phenotypes > 1:
        lsum = l1.reshape(-1, 1) + l2
        lh0_abf = torch.zeros([1, num_phenotypes], device=device, dtype=torch.float64)
        lh1_abf = np.log(p1) + _logsumexp(l1).repeat([1, num_phenotypes])
    else:
        lsum = l1 + l2
        lh0_abf = torch.zeros([1], device=device, dtype=torch.float64)
        lh1_abf = np.log(p1) + _logsumexp(l1)

    lh2_abf = np.log(p2) + _logsumexp(l2)
    lh3_abf = (np.log(p1) + np.log(p2)
               + _logdiff(_logsumexp(l1) + _logsumexp(l2), _logsumexp(lsum)))
    lh4_abf = np.log(p12) + _logsumexp(lsum)

    all_abf = torch.cat([lh0_abf, lh1_abf, lh2_abf, lh3_abf, lh4_abf])
    return (all_abf - _logsumexp(all_abf, dim=0)).exp().squeeze()


def coloc_from_summary(beta1, se1, beta2, se2, maf, n1, n2,
                       p1=1e-4, p2=1e-4, p12=1e-5):
    """
    COLOC ABF from pre-computed summary statistics.

    This avoids the need for raw genotype/phenotype data. Useful for
    colocalizing ancestry-aware results with external GWAS summary stats.

    Parameters
    ----------
    beta1, se1 : array-like
        Effect sizes and standard errors for trait 1, length m.
    beta2, se2 : array-like
        Effect sizes and standard errors for trait 2, length m.
    maf : array-like
        Minor allele frequencies, length m.
    n1, n2 : int
        Sample sizes for trait 1 and trait 2.
    p1, p2, p12 : float
        Prior probabilities.

    Returns
    -------
    dict
        Keys ``'PP_H0'`` .. ``'PP_H4'``, ``'nsnps'``.
    """
    beta1 = np.asarray(beta1, dtype=np.float64)
    se1 = np.asarray(se1, dtype=np.float64)
    beta2 = np.asarray(beta2, dtype=np.float64)
    se2 = np.asarray(se2, dtype=np.float64)
    maf = np.asarray(maf, dtype=np.float64)

    # Wakefield ABF: log ABF = 0.5 * [log(1 - r) + r * z^2]
    # where r = W / (W + V), W = prior variance, V = se^2, z = beta / se
    W = 0.04  # prior variance (COLOC default for quantitative traits)

    var1 = se1 ** 2
    z1 = beta1 / se1
    r1 = W / (W + var1)
    l1 = 0.5 * (np.log(1 - r1) + r1 * z1 ** 2)

    var2 = se2 ** 2
    z2 = beta2 / se2
    r2 = W / (W + var2)
    l2 = 0.5 * (np.log(1 - r2) + r2 * z2 ** 2)

    def _lse(x):
        m = x.max()
        return m + np.log(np.exp(x - m).sum())

    lsum = l1 + l2
    lh0 = 0.0
    lh1 = np.log(p1) + _lse(l1)
    lh2 = np.log(p2) + _lse(l2)
    lh3 = np.log(p1) + np.log(p2) + _lse(l1) + _lse(l2) - _lse(lsum)
    # logdiff: log(exp(a) - exp(b))
    a = _lse(l1) + _lse(l2)
    b = _lse(lsum)
    diff = max(np.exp(a) - np.exp(b), 1e-300)
    lh3 = np.log(p1) + np.log(p2) + np.log(diff)
    lh4 = np.log(p12) + _lse(lsum)

    all_lbf = np.array([lh0, lh1, lh2, lh3, lh4])
    denom = _lse(all_lbf)
    pp = np.exp(all_lbf - denom)

    return {
        f'PP_H{i}': float(pp[i]) for i in range(5)
    } | {'nsnps': len(beta1)}


def run_pairs(genotype_df, variant_df, phenotype1_df, phenotype2_df,
              phenotype_pos_df, covariates1_df=None, covariates2_df=None,
              p1=1e-4, p2=1e-4, p12=1e-5, mode='beta',
              maf_threshold=0, window=1000000,
              logger=None, verbose=True, device="cuda"):
    """
    Compute COLOC for all phenotype pairs across cis-windows.

    Parameters
    ----------
    genotype_df : pd.DataFrame
        Genotypes (variants x samples).
    variant_df : pd.DataFrame
        Variant metadata.
    phenotype1_df, phenotype2_df : pd.DataFrame
        Phenotypes for trait 1 and trait 2 (phenotypes x samples).
        Must have matching indices.
    phenotype_pos_df : pd.DataFrame
        Phenotype positions.
    covariates1_df, covariates2_df : pd.DataFrame, optional
        Covariates for each trait.
    mode : str
        ``'beta'`` or ``'cc'``.
    device : str
        ``"cuda"`` or ``"cpu"``.

    Returns
    -------
    pd.DataFrame
        Columns: ``pp_h0_abf`` .. ``pp_h4_abf``, indexed by phenotype.
    """
    from .genotypeio import InputGeneratorCis

    assert np.all(phenotype1_df.index == phenotype2_df.index)
    device = resolve_device(device)
    logger = logger or SimpleLogger(verbose=verbose)

    logger.write('Computing COLOC for all pairs of phenotypes')
    logger.write(f'  * {phenotype1_df.shape[0]} phenotypes')
    logger.write(f'  * phenotype group 1: {phenotype1_df.shape[1]} samples')
    logger.write(f'  * phenotype group 2: {phenotype2_df.shape[1]} samples')

    if covariates1_df is not None:
        logger.write(f'  * phenotype group 1: {covariates1_df.shape[1]} covariates')
        residualizer1 = Residualizer(
            torch.tensor(covariates1_df.values, dtype=torch.float32, device=device))
    else:
        residualizer1 = None

    if covariates2_df is not None:
        logger.write(f'  * phenotype group 2: {covariates2_df.shape[1]} covariates')
        residualizer2 = Residualizer(
            torch.tensor(covariates2_df.values, dtype=torch.float32, device=device))
    else:
        residualizer2 = None

    if maf_threshold > 0:
        logger.write(f'  * applying in-sample MAF >= {maf_threshold} filter')

    genotype1_ix = np.array([genotype_df.columns.tolist().index(i) for i in phenotype1_df.columns])
    genotype1_ix_t = torch.from_numpy(genotype1_ix).to(device)
    genotype2_ix = np.array([genotype_df.columns.tolist().index(i) for i in phenotype2_df.columns])
    genotype2_ix_t = torch.from_numpy(genotype2_ix).to(device)

    igc = InputGeneratorCis(
        genotype_df=genotype_df, variant_df=variant_df,
        phenotype_df=phenotype1_df, phenotype_pos_df=phenotype_pos_df,
        window=window)

    coloc_results = []
    start_time = time.time()
    logger.write('  * Computing pairwise colocalization')

    for batch in igc.generate_data():
        phenotype1, genotypes, genotype_range, phenotype_id = batch[:4]
        phenotype2 = phenotype2_df.loc[phenotype_id].values

        genotypes_t = torch.tensor(genotypes, dtype=torch.float32, device=device)
        genotypes_t, keep_mono, _ = impute_mean_and_filter(genotypes_t)

        genotypes1_t = genotypes_t[:, genotype1_ix_t]
        genotypes2_t = genotypes_t[:, genotype2_ix_t]

        if maf_threshold > 0:
            maf1_t = _calculate_maf(genotypes1_t)
            maf2_t = _calculate_maf(genotypes2_t)
            mask_t = (maf1_t >= maf_threshold) | (maf2_t >= maf_threshold)
            genotypes1_t = genotypes1_t[mask_t]
            genotypes2_t = genotypes2_t[mask_t]

        if genotypes1_t.shape[0] < 2:
            coloc_results.append(np.full(5, np.nan))
            continue

        phenotype1_t = torch.tensor(phenotype1, dtype=torch.float32, device=device)
        phenotype2_t = torch.tensor(phenotype2, dtype=torch.float32, device=device)

        coloc_t = coloc(genotypes1_t, genotypes2_t, phenotype1_t, phenotype2_t,
                        residualizer1=residualizer1, residualizer2=residualizer2,
                        p1=p1, p2=p2, p12=p12, mode=mode)
        coloc_results.append(coloc_t.cpu().numpy())

    logger.write(f'  Time elapsed: {(time.time() - start_time) / 60:.2f} min')
    logger.write('done.')

    coloc_df = pd.DataFrame(
        coloc_results,
        columns=[f'pp_h{i}_abf' for i in range(5)],
        index=phenotype1_df.index)
    return coloc_df
