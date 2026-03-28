# SuSiE (Sum of Single Effects) model
#
# References:
# [1] Wang et al., J. Royal Stat. Soc. B, 2020
#     https://rss.onlinelibrary.wiley.com/doi/full/10.1111/rssb.12388
#
# Port of https://github.com/broadinstitute/tensorqtl/blob/master/tensorqtl/susie.py
# Adapted for localQTL: pure Python, no R/rpy2, uses localQTL device management.

import torch
import numpy as np
import pandas as pd
from collections import defaultdict
import time

from ._torch_utils import resolve_device
from .regression_kernels import Residualizer
from .preproc import impute_mean_and_filter
from .utils import SimpleLogger

__all__ = [
    "susie",
    "susie_get_pip",
    "susie_get_cs",
    "map",
    "map_loci",
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def get_x_attributes(X_t, center=True, scale=True):
    """Compute column means and SDs for standardization."""
    cm_t = X_t.mean(0)
    csd_t = X_t.std(0, unbiased=True)
    csd_t[csd_t == 0] = 1

    if not center:
        cm_t = torch.zeros_like(cm_t)
    if not scale:
        csd_t = torch.ones_like(csd_t)

    x_std_t = (X_t - cm_t) / csd_t
    return {
        'd': (x_std_t * x_std_t).sum(0),
        'scaled_center': cm_t,
        'scaled_scale': csd_t,
    }


def _init_setup(n, p, L, scaled_prior_variance, varY, device,
                residual_variance=None, prior_weights=None, null_weight=None):
    if scaled_prior_variance < 0:
        raise ValueError('Scaled prior variance must be positive.')
    if residual_variance is None:
        residual_variance = varY
    if prior_weights is None:
        prior_weights = torch.full([p], 1 / p, dtype=torch.float32, device=device)
    else:
        prior_weights = prior_weights / prior_weights.sum()
    if len(prior_weights) != p:
        raise ValueError('Prior weights must have length p.')
    if p < L:
        L = p

    s = {
        'alpha': torch.full((L, p), 1 / p, device=device),
        'mu': torch.zeros((L, p), device=device),
        'mu2': torch.zeros((L, p), device=device),
        'Xr': torch.zeros(n, device=device),
        'KL': torch.full([L], float('nan'), device=device),
        'lbf': torch.full([L], float('nan'), device=device),
        'lbf_variable': torch.full([L, p], float('nan'), device=device),
        'sigma2': residual_variance,
        'V': scaled_prior_variance * varY,
        'pi': prior_weights,
    }
    s['null_index'] = 0 if null_weight is None else p
    return s


def _init_finalize(s, device):
    if s['V'].ndim == 0:
        s['V'] = torch.full([s['alpha'].shape[0]], s['V'], device=device)

    if s['sigma2'] <= 0:
        raise ValueError("residual variance 'sigma2' must be positive (is var(Y) zero?)")
    if not (s['V'] >= 0).all():
        raise ValueError("prior variance must be non-negative")

    L = s['alpha'].shape[0]
    s['KL'] = torch.full([L], float('nan'), device=device)
    s['lbf'] = torch.full([L], float('nan'), device=device)
    return s


def _compute_Xb(X_t, b_t, cm_t, csd_t):
    """Compute Xb with column-standardized X."""
    scaled_Xb_t = torch.mm(X_t, (b_t / csd_t).reshape(-1, 1)).squeeze()
    return scaled_Xb_t - (cm_t * b_t / csd_t).sum()


def _compute_Xty(X_t, y_t, cm_t, csd_t):
    ytX_t = torch.mm(y_t.T, X_t)
    scaled_Xty_t = ytX_t.T / csd_t.reshape(-1, 1)
    centered_scaled_Xty_t = scaled_Xty_t - cm_t.reshape(-1, 1) / csd_t.reshape(-1, 1) * y_t.sum()
    return centered_scaled_Xty_t.squeeze()


def _compute_MXt(M_t, X_t, xattr):
    return (torch.mm(M_t, (X_t / xattr['scaled_scale']).T)
            - torch.mm(M_t, (xattr['scaled_center'] / xattr['scaled_scale']).reshape(-1, 1)))


def _loglik(V, betahat, shat2, prior_weights):
    lbf = (torch.distributions.Normal(0, torch.sqrt(V + shat2)).log_prob(betahat)
           - torch.distributions.Normal(0, torch.sqrt(shat2)).log_prob(betahat))
    lbf[torch.isinf(shat2)] = 0
    maxlbf = lbf.max()
    return torch.log((torch.exp(lbf - maxlbf) * prior_weights).sum()) + maxlbf


def _optimize_prior_variance(betahat, shat2, prior_weights, alpha, post_mean2,
                             check_null_threshold=0):
    V = (alpha * post_mean2).sum()
    if (_loglik(0, betahat, shat2, prior_weights) + check_null_threshold
            >= _loglik(V, betahat, shat2, prior_weights)):
        V = 0
    return V


def _SER_posterior_e_loglik(X_t, xattr, Y_t, s2, Eb, Eb2):
    n = X_t.shape[0]
    return (-0.5 * n * torch.log(2 * np.pi * s2)
            - (0.5 / s2) * ((Y_t * Y_t).sum()
                             - 2 * (Y_t.squeeze() * _compute_Xb(X_t, Eb, xattr['scaled_center'], xattr['scaled_scale'])).sum()
                             + (xattr['d'] * Eb2).sum()))


def _single_effect_regression(Y_t, X_t, xattr, V, device, residual_variance=1,
                              prior_weights=None, check_null_threshold=0):
    Xty = _compute_Xty(X_t, Y_t, xattr['scaled_center'], xattr['scaled_scale'])
    betahat = (1 / xattr['d']) * Xty
    shat2 = residual_variance / xattr['d']
    if prior_weights is None:
        prior_weights = torch.full([X_t.shape[1]], 1 / X_t.shape[1], device=device)

    lbf = (torch.distributions.Normal(0, torch.sqrt(V + shat2)).log_prob(betahat)
           - torch.distributions.Normal(0, torch.sqrt(shat2)).log_prob(betahat))
    lbf[torch.isinf(shat2)] = 0
    maxlbf = lbf.max()
    w = torch.exp(lbf - maxlbf)
    w_weighted = w * prior_weights
    weighted_sum_w = w_weighted.sum()
    alpha = w_weighted / weighted_sum_w

    if V == 0:
        post_var = torch.zeros(xattr['d'].shape, device=device)
    else:
        post_var = (1 / V + xattr['d'] / residual_variance) ** (-1)

    post_mean = (1 / residual_variance) * post_var * Xty
    post_mean2 = post_var + post_mean ** 2
    lbf_model = maxlbf + torch.log(weighted_sum_w)
    loglik_val = lbf_model + torch.distributions.Normal(
        0, torch.sqrt(residual_variance)).log_prob(Y_t).sum()

    V = _optimize_prior_variance(betahat, shat2, prior_weights, alpha,
                                 post_mean2, check_null_threshold=check_null_threshold)
    return {
        'alpha': alpha, 'mu': post_mean, 'mu2': post_mean2,
        'lbf': lbf, 'lbf_model': lbf_model, 'V': V, 'loglik': loglik_val,
    }


def _update_each_effect(X_t, xattr, Y_t, s, device,
                        estimate_prior_variance=False, check_null_threshold=0):
    estimate_prior_method = 'EM' if estimate_prior_variance else 'none'
    L = s['alpha'].shape[0]
    for l in range(L):
        s['Xr'] = s['Xr'] - _compute_Xb(
            X_t, s['alpha'][l] * s['mu'][l], xattr['scaled_center'], xattr['scaled_scale'])
        R_t = Y_t - s['Xr'].reshape(-1, 1)
        res = _single_effect_regression(
            R_t, X_t, xattr, s['V'][l], device,
            residual_variance=s['sigma2'], prior_weights=s['pi'],
            check_null_threshold=check_null_threshold)
        s['mu'][l] = res['mu']
        s['alpha'][l] = res['alpha']
        s['mu2'][l] = res['mu2']
        s['V'][l] = res['V']
        s['lbf'][l] = res['lbf_model']
        s['lbf_variable'][l] = res['lbf']
        s['KL'][l] = (-res['loglik']
                      + _SER_posterior_e_loglik(X_t, xattr, R_t, s['sigma2'],
                                               res['alpha'] * res['mu'],
                                               res['alpha'] * res['mu2']))
        s['Xr'] = s['Xr'] + _compute_Xb(
            X_t, s['alpha'][l] * s['mu'][l], xattr['scaled_center'], xattr['scaled_scale'])
    return s


def _get_ER2(X_t, xattr, Y_t, s):
    Xr_L = _compute_MXt(s['alpha'] * s['mu'], X_t, xattr)
    postb2 = s['alpha'] * s['mu2']
    return (((Y_t.squeeze() - s['Xr']) ** 2).sum()
            - (Xr_L ** 2).sum()
            + (xattr['d'].reshape(-1, 1) * postb2.T).sum())


def _eloglik(X_t, xattr, Y_t, s):
    n = X_t.shape[0]
    return -(n / 2) * torch.log(2 * np.pi * s['sigma2']) - (1 / (2 * s['sigma2'])) * _get_ER2(X_t, xattr, Y_t, s)


def _get_objective(X_t, xattr, Y_t, s):
    return _eloglik(X_t, xattr, Y_t, s) - s['KL'].sum()


def _estimate_residual_variance(X_t, xattr, Y_t, s):
    return (1 / X_t.shape[0]) * _get_ER2(X_t, xattr, Y_t, s)


def _in_CS(res, coverage=0.9):
    o = torch.flip(res['alpha'].argsort(), [1])
    n = (torch.cumsum(torch.gather(res['alpha'], 1, o), 1) < coverage).sum(1) + 1
    result = torch.zeros(res['alpha'].shape, dtype=torch.bool, device=res['alpha'].device)
    for i in range(result.shape[0]):
        result[i, o[i][:n[i]]] = True
    return result


def _corrcoef(X_t):
    X0_t = X_t - X_t.mean(1, keepdim=True)
    c = torch.mm(X0_t, X0_t.T) / (X_t.shape[1] - 1)
    sd = torch.sqrt(torch.diag(c))
    c = c / sd[:, None] / sd[None, :]
    return torch.clamp(c, -1, 1)


def _get_purity(pos, X, Xcorr, squared=False, n=100):
    if len(pos) == 1:
        return np.ones(3)
    if len(pos) > n:
        pos = np.random.choice(pos, n, replace=False)
    if Xcorr is None:
        X_sub = X[:, pos]
        value = _corrcoef(X_sub.T).abs()
    else:
        value = Xcorr[pos][:, pos].abs()
    if squared:
        value = value ** 2
    return float(value.min()), float(value.mean()), float(value.median())


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def susie_get_pip(res, prune_by_cs=False, prior_tol=1e-9):
    """
    Compute posterior inclusion probability (PIP) for all variables.

    Parameters
    ----------
    res : dict
        Output of :func:`susie`.
    prune_by_cs : bool
        Whether to ignore single effects not in reported CS.
    prior_tol : float
        Filter out effects with estimated prior variance below this.

    Returns
    -------
    torch.Tensor
        Array of PIPs, length p.
    """
    alpha = res['alpha']
    if res['null_index'] > 0:
        alpha = alpha[:, :-res['null_index']]

    include_idx = torch.where(res['V'] > prior_tol)[0]

    if prune_by_cs:
        raise NotImplementedError("prune_by_cs not yet supported")

    if len(include_idx) > 0:
        alpha = alpha[include_idx]
    else:
        alpha = torch.zeros([1, alpha.shape[1]], device=alpha.device)

    return 1 - (1 - alpha).prod(0)


def susie_get_cs(res, X=None, Xcorr=None, coverage=0.95, min_abs_corr=0.5,
                 dedup=True, squared=False):
    """
    Extract credible sets from a SuSiE fit.

    Parameters
    ----------
    res : dict
        Output of :func:`susie`.
    X : torch.Tensor, optional
        Genotype matrix (n x p) for purity computation.
    Xcorr : torch.Tensor, optional
        Correlation matrix (p x p). Only one of X or Xcorr.
    coverage : float
        Target coverage for credible sets.
    min_abs_corr : float
        Minimum absolute pairwise correlation for a CS to be "pure".
    dedup : bool
        Remove duplicate credible sets.
    squared : bool
        Use squared correlations for purity.

    Returns
    -------
    dict
        ``'cs'``: dict mapping L-names to arrays of variant indices;
        ``'purity'``: DataFrame; ``'cs_index'``: array; ``'coverage'``: float.
    """
    device = res['alpha'].device
    if X is not None and Xcorr is not None:
        raise ValueError('Only one of X or Xcorr should be specified.')

    null_index = res['null_index']
    include_mask = res['V'] > 1e-9

    status = _in_CS(res, coverage=coverage)
    cs = [torch.where(i)[0] for i in status]
    include_mask = include_mask & torch.tensor(
        [len(i) > 0 for i in cs], dtype=torch.bool, device=device)

    if dedup:
        duplicated = torch.ones(status.shape[0], dtype=torch.bool, device=device)
        _, ix = status.unique(dim=0, return_inverse=True)
        duplicated[ix.unique()] = False
        include_mask = include_mask & ~duplicated

    if not include_mask.any():
        return {'cs': None, 'coverage': coverage}

    if Xcorr is None and X is None:
        cs_dict = {f'L{k + 1}': cs[k] for k, i in enumerate(include_mask) if i}
        return {'cs': cs_dict, 'coverage': coverage}

    cs = [cs[k] for k, i in enumerate(include_mask) if i]
    purity = []
    for i in range(len(cs)):
        if null_index > 0 and null_index in cs[i]:
            purity.append([-9, -9, -9])
        else:
            purity.append(_get_purity(cs[i], X, Xcorr, squared=squared))

    cols = (['min_sq_corr', 'mean_sq_corr', 'median_sq_corr'] if squared
            else ['min_abs_corr', 'mean_abs_corr', 'median_abs_corr'])
    purity = pd.DataFrame(purity, columns=cols)

    threshold = min_abs_corr ** 2 if squared else min_abs_corr
    is_pure = np.where(purity.values[:, 0] >= threshold)[0]
    if len(is_pure) > 0:
        include_idx = torch.where(include_mask)[0]
        cs = [cs[k] for k in is_pure]
        purity = purity.iloc[is_pure]
        rownames = [f'L{i + 1}' for i in include_idx[is_pure]]
        purity.index = rownames
        ordering = purity.values[:, 0].argsort()[::-1]
        return {
            'cs': {rownames[i]: cs[i].numpy() for i in ordering},
            'purity': purity.iloc[ordering],
            'cs_index': include_idx[is_pure[ordering]].cpu().numpy(),
            'coverage': coverage,
        }
    else:
        return {'cs': None, 'coverage': coverage}


def susie(X_t, y_t, L=10, scaled_prior_variance=0.2,
          residual_variance=None, prior_weights=None, null_weight=None,
          standardize=True, intercept=True,
          estimate_residual_variance=True, estimate_prior_variance=True,
          check_null_threshold=0, prior_tol=1e-9,
          residual_variance_upperbound=np.inf,
          coverage=0.95, min_abs_corr=0.5,
          max_iter=100, tol=0.001, verbose=False):
    """
    Fit the SuSiE model via Iterative Bayesian Stepwise Selection (IBSS).

    Parameters
    ----------
    X_t : torch.Tensor
        Genotype matrix (n x p), samples x variants.
    y_t : torch.Tensor
        Phenotype vector (n x 1) or (n,).
    L : int
        Maximum number of causal effects.
    scaled_prior_variance : float
        Prior variance scaled by var(y).
    coverage : float
        Target coverage for credible sets.
    min_abs_corr : float
        Minimum purity for credible sets.
    max_iter : int
        Maximum IBSS iterations.
    tol : float
        Convergence tolerance on ELBO.

    Returns
    -------
    dict
        SuSiE fit object with keys: ``'alpha'``, ``'mu'``, ``'mu2'``,
        ``'pip'``, ``'sets'``, ``'converged'``, ``'elbo'``, ``'niter'``,
        ``'lbf_variable'``, ``'sigma2'``, ``'V'``, ``'intercept'``, ``'fitted'``.
    """
    device = X_t.device
    if y_t.dim() == 1:
        y_t = y_t.unsqueeze(-1)

    n, p = X_t.shape
    mean_y = y_t.mean()

    if intercept:
        y_t = y_t - mean_y

    xattr = get_x_attributes(X_t, center=intercept, scale=standardize)

    s = _init_setup(n, p, L, scaled_prior_variance, y_t.var(unbiased=True),
                    device, residual_variance=residual_variance,
                    prior_weights=prior_weights, null_weight=null_weight)
    s = _init_finalize(s, device)

    elbo = torch.full([max_iter + 1], float('nan'), device=device)
    elbo[0] = float('-inf')

    for i in range(1, max_iter + 1):
        s = _update_each_effect(X_t, xattr, y_t, s, device,
                                estimate_prior_variance=estimate_prior_variance,
                                check_null_threshold=0)
        elbo[i] = _get_objective(X_t, xattr, y_t, s)
        if verbose:
            print(f'Objective (iter {i}): {elbo[i]}')
        if (elbo[i] - elbo[i - 1]) < tol:
            s['converged'] = True
            break
        if estimate_residual_variance:
            s['sigma2'] = _estimate_residual_variance(X_t, xattr, y_t, s)
            if s['sigma2'] > residual_variance_upperbound:
                s['sigma2'] = residual_variance_upperbound

    s['elbo'] = elbo[1:i + 1].cpu().numpy()
    s['niter'] = i

    if 'converged' not in s:
        s['converged'] = False

    if intercept:
        s['intercept'] = (mean_y
                          - (xattr['scaled_center']
                             * ((s['alpha'] * s['mu']).sum(0)
                                / xattr['scaled_scale'])).sum())
        s['fitted'] = s['Xr'] + mean_y
    else:
        s['intercept'] = 0
        s['fitted'] = s['Xr']
    s['fitted'] = s['fitted'].squeeze()

    s['lbf_variable'] = s['lbf_variable'].cpu().numpy()

    if coverage is not None and min_abs_corr is not None:
        s['sets'] = susie_get_cs(s, coverage=coverage, X=X_t, min_abs_corr=min_abs_corr)
        s['pip'] = susie_get_pip(s, prune_by_cs=False, prior_tol=prior_tol).cpu().numpy()

    return s


# ---------------------------------------------------------------------------
# High-level mapping API
# ---------------------------------------------------------------------------

def map(genotype_df, variant_df, phenotype_df, phenotype_pos_df,
        covariates_df=None, L=10, scaled_prior_variance=0.2,
        estimate_residual_variance=True, estimate_prior_variance=True,
        tol=1e-3, coverage=0.95, min_abs_corr=0.5,
        summary_only=True, maf_threshold=0, max_iter=200, window=1000000,
        logger=None, verbose=True, device="cuda"):
    """
    Run SuSiE fine-mapping for all phenotypes across cis-windows.

    Parameters
    ----------
    genotype_df : pd.DataFrame
        Genotypes (variants x samples).
    variant_df : pd.DataFrame
        Variant metadata (index=variant_id, columns include 'chrom', 'pos').
    phenotype_df : pd.DataFrame
        Phenotypes (phenotypes x samples).
    phenotype_pos_df : pd.DataFrame
        Phenotype positions (index=phenotype_id, columns 'chr', 'start', 'end').
    covariates_df : pd.DataFrame, optional
        Covariates (samples x covariates).
    summary_only : bool
        If True, return only the summary DataFrame. If False, also return
        full SuSiE outputs per phenotype.
    device : str
        ``"cuda"`` or ``"cpu"``.

    Returns
    -------
    summary_df : pd.DataFrame or list
        Summary of credible sets across all phenotypes.
    susie_res : dict (only if ``summary_only=False``)
        Full SuSiE output per phenotype.
    """
    from .genotypeio import InputGeneratorCis

    device = resolve_device(device)
    logger = logger or SimpleLogger(verbose=verbose)

    logger.write('SuSiE fine-mapping')
    logger.write(f'  * {phenotype_df.shape[1]} samples')
    logger.write(f'  * {phenotype_df.shape[0]} phenotypes')
    if covariates_df is not None:
        logger.write(f'  * {covariates_df.shape[1]} covariates')
    logger.write(f'  * {variant_df.shape[0]} variants')
    logger.write(f'  * cis-window: \u00B1{window:,}')
    if maf_threshold > 0:
        logger.write(f'  * applying in-sample MAF >= {maf_threshold} filter')

    residualizer = None
    if covariates_df is not None:
        residualizer = Residualizer(
            torch.tensor(covariates_df.values, dtype=torch.float32, device=device))

    igc = InputGeneratorCis(
        genotype_df=genotype_df, variant_df=variant_df,
        phenotype_df=phenotype_df, phenotype_pos_df=phenotype_pos_df,
        window=window)

    start_time = time.time()
    logger.write('  * fine-mapping')
    copy_keys = ['pip', 'sets', 'converged', 'elbo', 'niter', 'lbf_variable']
    susie_summary = []
    susie_res = {} if not summary_only else None

    for k, batch in enumerate(igc.generate_data(), 1):
        phenotype, genotypes, v_idx, phenotype_id = batch[:4]

        genotypes_t = torch.tensor(genotypes, dtype=torch.float32, device=device)
        genotypes_t, keep_mono, _ = impute_mean_and_filter(genotypes_t)
        v_idx = v_idx[keep_mono.detach().cpu().numpy()]

        if genotypes_t.shape[0] == 0:
            logger.write(f'WARNING: skipping {phenotype_id} (no valid variants)')
            continue

        variant_ids = variant_df.index[v_idx]

        if maf_threshold > 0:
            af = genotypes_t.sum(1) / (2 * genotypes_t.shape[1])
            maf = torch.minimum(af, 1 - af)
            mask = maf >= maf_threshold
            if mask.any():
                genotypes_t = genotypes_t[mask]
                variant_ids = variant_ids[mask.cpu().numpy()]
            if genotypes_t.shape[0] == 0:
                continue

        phenotype_t = torch.tensor(phenotype, dtype=torch.float32, device=device)
        if phenotype_t.dim() == 1:
            phenotype_t = phenotype_t.unsqueeze(0)

        if residualizer is not None:
            (genotypes_res_t,) = residualizer.transform(genotypes_t, center=True)
            (phenotype_res_t,) = residualizer.transform(phenotype_t, center=True)
        else:
            genotypes_res_t = genotypes_t
            phenotype_res_t = phenotype_t

        res = susie(
            genotypes_res_t.T, phenotype_res_t.T, L=L,
            scaled_prior_variance=scaled_prior_variance,
            coverage=coverage, min_abs_corr=min_abs_corr,
            estimate_residual_variance=estimate_residual_variance,
            estimate_prior_variance=estimate_prior_variance,
            tol=tol, max_iter=max_iter)

        af_t = genotypes_t.sum(1) / (2 * genotypes_t.shape[1])
        res['pip'] = pd.DataFrame(
            {'pip': res['pip'], 'af': af_t.cpu().numpy()},
            index=variant_ids)

        if res['sets']['cs'] is not None and res['converged']:
            for c in sorted(res['sets']['cs'], key=lambda x: int(x.replace('L', ''))):
                cs_indices = res['sets']['cs'][c]
                p = res['pip'].iloc[cs_indices].copy().reset_index()
                p.columns = ['variant_id', 'pip', 'af']
                p['cs_id'] = c.replace('L', '')
                p.insert(0, 'phenotype_id', phenotype_id)
                susie_summary.append(p)
            res['lbf_variable'] = res['lbf_variable'][res['sets']['cs_index']]

        if susie_res is not None:
            susie_res[phenotype_id] = {k_: res[k_] for k_ in copy_keys if k_ in res}

    logger.write(f'  Time elapsed: {(time.time() - start_time) / 60:.2f} min')
    logger.write('done.')

    if susie_summary:
        susie_summary = pd.concat(susie_summary, axis=0).reset_index(drop=True)
    else:
        susie_summary = pd.DataFrame()

    if summary_only:
        return susie_summary
    else:
        if susie_res:
            drop_ids = [k_ for k_ in susie_res
                        if susie_res[k_].get('sets', {}).get('cs') is None]
            for k_ in drop_ids:
                del susie_res[k_]
        return susie_summary, susie_res


def map_loci(locus_df, genotype_df, variant_df, phenotype_df,
             covariates_df=None, **kwargs):
    """
    Run SuSiE fine-mapping on specific phenotype-locus pairs.

    Parameters
    ----------
    locus_df : pd.DataFrame
        Columns: ``['phenotype_id', 'chr', 'position']`` or
        ``['phenotype_id', 'chr', 'start', 'end']``.
    genotype_df, variant_df, phenotype_df, covariates_df
        As in :func:`map`.

    Returns
    -------
    summary_df : pd.DataFrame
    susie_outputs : dict
    """
    window = kwargs.pop('window', 1_000_000)
    locus_df = locus_df.rename(columns={'position': 'pos'}).copy()

    num_loci = defaultdict(int)
    locus_ix = []
    for pid in locus_df['phenotype_id']:
        num_loci[pid] += 1
        locus_ix.append(num_loci[pid])
    locus_df['locus'] = locus_ix

    if 'start' in locus_df and 'end' in locus_df:
        pos_df = locus_df[['phenotype_id', 'chr', 'start', 'end']]
    else:
        locus_df['start'] = np.maximum(locus_df['pos'] - window, 1)
        locus_df['end'] = locus_df['pos'] + window
        pos_df = locus_df[['phenotype_id', 'chr', 'start', 'end']]

    summary_dfs = []
    all_res = {}
    nmax = locus_df['locus'].max()
    for i in np.arange(1, nmax + 1):
        m = locus_df['locus'] == i
        pids = locus_df.loc[m, 'phenotype_id']
        chunk_summary, chunk_res = map(
            genotype_df, variant_df,
            phenotype_df.loc[pids],
            pos_df[m].set_index('phenotype_id'),
            covariates_df, summary_only=False, window=window, **kwargs)
        if len(chunk_summary) > 0:
            chunk_summary.insert(1, 'locus', i)
            summary_dfs.append(chunk_summary)
            all_res |= chunk_res

    summary_df = pd.concat(summary_dfs).reset_index(drop=True) if summary_dfs else pd.DataFrame()
    return summary_df, all_res
