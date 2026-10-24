"""
Post-processing with tensorQTL's names (``tensorqtl.post`` equivalent):
q-values, significant pairs, allelic fold change (aFC), replication and gene
annotation.

Adapted from tensorQTL ``post.py`` / ``mixqtl.py`` (BSD-3, Broad Institute).
"""
from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import torch
from py_qvalue import pi0est

from ._torch_utils import resolve_device
from .cis.postproc import get_significant_pairs
from .preproc import allele_stats
from .regression_kernels import Residualizer
from .stats import calculate_qvalues, get_t_pval

__all__ = [
    "calculate_qvalues",
    "get_significant_pairs",
    "linreg",
    "trc",
    "calculate_afc",
    "calculate_replication",
    "annotate_genes",
]


# ---------------------------------------------------------------------------
# small linear algebra helpers
# ---------------------------------------------------------------------------
def linreg(X_t: torch.Tensor, y_t: torch.Tensor, dtype=torch.float64):
    """OLS of ``y_t`` (n,) on ``X_t`` (n, p) (must include an intercept column). Returns ``(b, b_se)``."""
    X = X_t.to(dtype)
    y = y_t.to(dtype)
    n, p = X.shape
    XtX_inv = torch.linalg.pinv(X.T @ X)
    b = XtX_inv @ (X.T @ y)
    r = y - X @ b
    dof = max(n - p, 1)
    sigma2 = (r @ r) / dof
    b_se = torch.sqrt(torch.clamp(torch.diag(XtX_inv) * sigma2, min=1e-30))
    return b, b_se


def _impute_mean_rows(G_t: torch.Tensor) -> torch.Tensor:
    """Mean-impute ``-9`` / non-finite entries per row (keeps every row)."""
    miss = (G_t == -9) | ~torch.isfinite(G_t)
    if not miss.any():
        return G_t
    n_obs = (~miss).sum(1, keepdim=True).clamp_min(1)
    mu = torch.where(miss, torch.zeros_like(G_t), G_t).sum(1, keepdim=True) / n_obs
    return torch.where(miss, mu, G_t)


def trc(genotypes_t: torch.Tensor, counts_t: torch.Tensor, covariates_t: torch.Tensor | None = None,
        select_covariates: bool = True, count_threshold: float = 0, imputation: str = "offset",
        mode: str = "multi"):
    """
    mixQTL total-read-count (trc) model (Liang et al., 2021).

    ``genotypes_t`` dosages (variants x samples); ``counts_t`` size-factor
    normalised read counts (samples); ``covariates_t`` (samples x covariates)
    **with an intercept in the first column**.  Zeros are handled by
    ``log(counts + 1)`` (``imputation='offset'``) or half the minimum non-zero
    count (``'half_min'``).  ``mode='multi'`` fits all variants jointly and
    returns ``(b, b_se)`` on the natural-log scale (per haplotype: genotypes
    are divided by two); ``mode='standard'`` fits each variant separately and
    returns ``(tstat, b, b_se)``.
    """
    nonzero_t = counts_t != 0
    if imputation == "offset":
        log_counts_t = counts_t.log1p()
    elif imputation == "half_min":
        log_counts_t = counts_t.clone()
        log_counts_t[~nonzero_t] = log_counts_t[nonzero_t].min() / 2
        log_counts_t = log_counts_t.log()
    else:
        raise ValueError("imputation must be 'offset' or 'half_min'")

    if covariates_t is not None:
        if select_covariates:
            b_t, b_se_t = linreg(covariates_t[nonzero_t, :], log_counts_t[nonzero_t], dtype=torch.float32)
            tstat_t = b_t / b_se_t
            m = tstat_t.abs() > 2
            m[0] = True  # keep intercept
            sel_covariates_t = covariates_t[:, m]
        else:
            sel_covariates_t = covariates_t
        # Regress covariates out of the non-zero counts, keep zeros (mixQTL convention).
        y_t = log_counts_t.clone()
        if sel_covariates_t.shape[1] > 1:
            residualizer = Residualizer(sel_covariates_t[nonzero_t, 1:])
            y_t[nonzero_t] = residualizer.transform(log_counts_t[nonzero_t].reshape(1, -1), center=True)[0].squeeze(0)
        else:
            y_t[nonzero_t] = log_counts_t[nonzero_t] - log_counts_t[nonzero_t].mean()
        y_t[~nonzero_t] = log_counts_t[~nonzero_t] if imputation == "offset" else y_t[~nonzero_t]
    else:
        y_t = log_counts_t

    m_t = counts_t >= count_threshold
    if mode == "standard":
        out_b, out_se = [], []
        for k in range(genotypes_t.shape[0]):
            X = torch.stack([torch.ones(int(m_t.sum()), device=genotypes_t.device), genotypes_t[k, m_t] / 2], 1)
            b, se = linreg(X, y_t[m_t], dtype=torch.float32)
            out_b.append(b[1]); out_se.append(se[1])
        b = torch.stack(out_b); se = torch.stack(out_se)
        return b / se, b, se
    if mode.startswith("multi"):
        X_t = torch.cat([torch.ones([int(m_t.sum()), 1], device=genotypes_t.device, dtype=genotypes_t.dtype),
                         genotypes_t[:, m_t].T / 2], dim=1)
        b_t, b_se_t = linreg(X_t, y_t[m_t], dtype=torch.float32)
        return b_t[1:], b_se_t[1:]
    raise ValueError("mode must be 'standard' or 'multi'")


# ---------------------------------------------------------------------------
# aFC
# ---------------------------------------------------------------------------
def calculate_afc(assoc_df: pd.DataFrame, counts_df: pd.DataFrame, genotype_df, variant_df=None,
                  covariates_df: pd.DataFrame | None = None, select_covariates: bool = True,
                  group: str = "gene_id", imputation: str = "offset", count_threshold: float = 0,
                  device: str = "auto", verbose: bool = True) -> pd.DataFrame:
    """
    Allelic fold change (aFC, log2) for variant-gene pairs using the mixQTL
    total-read-count model.

    ``assoc_df`` must have ``variant_id`` and the ``group`` column (gene /
    phenotype id); several variants per gene are fitted jointly.
    ``counts_df`` holds size-factor normalised counts (genes x samples).
    ``genotype_df`` is a dosage DataFrame (variants x samples) or a
    :class:`~localqtl.pgen.PgenReader`.  Returns ``assoc_df`` with ``afc`` and
    ``afc_se`` columns (genes whose fit fails are dropped with a warning).
    """
    dev = resolve_device(device)
    is_reader = not isinstance(genotype_df, pd.DataFrame)
    sample_ids = list(counts_df.columns)
    if is_reader:
        reader_samples = [str(s) for s in genotype_df.sample_ids]
        genotype_ix = np.array([reader_samples.index(str(s)) for s in sample_ids])
    else:
        gcols = pd.Index(genotype_df.columns.astype(str))
        genotype_ix = gcols.get_indexer(pd.Index(sample_ids).astype(str))
        if (genotype_ix < 0).any():
            raise ValueError("counts_df samples missing from genotype_df")
    genotype_ix_t = torch.as_tensor(genotype_ix, device=dev)

    covariates_t = None
    if covariates_df is not None:
        cov = covariates_df.loc[sample_ids]
        covariates_t = torch.tensor(np.c_[np.ones(len(sample_ids)), cov.values], dtype=torch.float32, device=dev)

    afc_df = []
    n = assoc_df[group].nunique()
    for k, (phenotype_id, gdf) in enumerate(assoc_df.groupby(group, sort=False), 1):
        if verbose and (k % 10 == 0 or k == n):
            print(f"\rCalculating aFC for {group.replace('_id', '')} {k}/{n}", end='' if k != n else None, flush=True)
        if phenotype_id not in counts_df.index:
            print(f'WARNING: {phenotype_id} not in counts_df; skipping')
            continue
        counts_t = torch.tensor(counts_df.loc[phenotype_id].values, dtype=torch.float32, device=dev)
        vids = gdf['variant_id'].tolist()
        if is_reader:
            g = genotype_df.read_list(vids, dosages=True)
            g = g.detach().cpu().numpy() if torch.is_tensor(g) else np.asarray(g)
        else:
            g = genotype_df.loc[vids].to_numpy(dtype=np.float32)
        genotypes_t = torch.tensor(g, dtype=torch.float32, device=dev)[:, genotype_ix_t]
        genotypes_t = _impute_mean_rows(genotypes_t)
        try:
            b, b_se = trc(genotypes_t, counts_t, covariates_t=covariates_t,
                          select_covariates=select_covariates, count_threshold=count_threshold,
                          imputation=imputation, mode='multi')
            gdf = gdf.copy()
            gdf['afc'] = b.cpu().numpy() * np.log2(np.e)
            gdf['afc_se'] = b_se.cpu().numpy() * np.log2(np.e)
            afc_df.append(gdf)
        except Exception as e:  # pragma: no cover - defensive, mirrors tensorQTL
            print(f'WARNING: aFC calculation failed for {phenotype_id} ({e})')
    if not afc_df:
        out = assoc_df.iloc[0:0].copy()
        out['afc'] = np.nan; out['afc_se'] = np.nan
        return out
    return pd.concat(afc_df)


# ---------------------------------------------------------------------------
# replication
# ---------------------------------------------------------------------------
def calculate_replication(res_df: pd.DataFrame, genotypes, phenotype_df: pd.DataFrame,
                          covariates_df: pd.DataFrame | None = None,
                          paired_covariate_df: pd.DataFrame | None = None,
                          interaction_s: pd.Series | None = None, compute_pi1: bool = False,
                          lambda_qvalue=None, logp: bool = False, device: str = "auto"):
    """
    Test the variant-phenotype pairs of ``res_df`` (``variant_id`` column,
    phenotype IDs as index) in another dataset: nominal association of each
    pair, optionally with an interaction term and phenotype-specific paired
    covariates (``paired_covariate_df``: phenotypes x samples).

    Returns ``rep_df`` with ``phenotype_id, variant_id, ma_samples, ma_count,
    af, pval_nominal, slope, slope_se`` (interaction: ``pval_g, b_g, b_g_se,
    pval_i, b_i, b_i_se, pval_gi, b_gi, b_gi_se``); with ``compute_pi1`` the
    tuple ``(pi1, rep_df)`` where ``pi1 = 1 - pi0`` of the replication
    p-values (Storey).
    """
    dev = resolve_device(device)
    sample_ids = list(phenotype_df.columns)
    if paired_covariate_df is not None:
        if covariates_df is None:
            raise ValueError("paired_covariate_df requires covariates_df")
        if not paired_covariate_df.columns.equals(phenotype_df.columns):
            raise ValueError("paired_covariate_df must be phenotypes x samples with the phenotype samples")
    if covariates_df is not None and not covariates_df.index.equals(phenotype_df.columns):
        covariates_df = covariates_df.loc[sample_ids]

    vids = res_df['variant_id'].tolist()
    if isinstance(genotypes, pd.DataFrame):
        g = genotypes.loc[vids]
        gcols = pd.Index(g.columns.astype(str))
        genotype_ix = gcols.get_indexer(pd.Index(sample_ids).astype(str))
        g = g.to_numpy(dtype=np.float32)
    else:  # PgenReader-like
        arr = genotypes.read_list(vids, dosages=False)
        g = arr.detach().cpu().numpy() if torch.is_tensor(arr) else np.asarray(arr)
        rs = [str(s) for s in genotypes.sample_ids]
        genotype_ix = np.array([rs.index(str(s)) for s in sample_ids])
    if (np.asarray(genotype_ix) < 0).any():
        raise ValueError("phenotype samples missing from genotypes")
    genotypes_t = torch.tensor(g, dtype=torch.float32, device=dev)[:, torch.as_tensor(genotype_ix, device=dev)]
    genotypes_t = _impute_mean_rows(genotypes_t)
    af_t, ma_samples_t, ma_count_t = allele_stats(genotypes_t)
    phenotypes_t = torch.tensor(phenotype_df.loc[res_df.index].values, dtype=torch.float32, device=dev)
    n = phenotypes_t.shape[1]

    residualizer = Residualizer(torch.tensor(covariates_df.values, dtype=torch.float32, device=dev)) \
        if covariates_df is not None else None
    k_eff = residualizer.k_eff if residualizer is not None else 1

    if interaction_s is None:
        if paired_covariate_df is None:
            if residualizer is not None:
                g_res = residualizer.transform(genotypes_t, center=True)[0]
                p_res = residualizer.transform(phenotypes_t, center=True)[0]
            else:
                g_res = genotypes_t - genotypes_t.mean(1, keepdim=True)
                p_res = phenotypes_t - phenotypes_t.mean(1, keepdim=True)
            dof = np.full(len(res_df), n - k_eff - 1, dtype=np.float64)
        else:
            g_res = torch.zeros_like(genotypes_t)
            p_res = torch.zeros_like(phenotypes_t)
            dof = np.empty(len(res_df), dtype=np.float64)
            for k, pid in enumerate(res_df.index):
                if pid in paired_covariate_df.index:
                    C = np.c_[covariates_df.values, paired_covariate_df.loc[pid, sample_ids].to_numpy()]
                    irez = Residualizer(torch.tensor(C, dtype=torch.float32, device=dev))
                else:
                    irez = residualizer
                g_res[[k]] = irez.transform(genotypes_t[[k]], center=True)[0]
                p_res[[k]] = irez.transform(phenotypes_t[[k]], center=True)[0]
                dof[k] = n - irez.k_eff - 1
        gvar = g_res.var(1); pvar = p_res.var(1)
        std_ratio = torch.sqrt(pvar / gvar.clamp_min(1e-30))
        gn = g_res / torch.linalg.norm(g_res, dim=1, keepdim=True).clamp_min(1e-12)
        pn = p_res / torch.linalg.norm(p_res, dim=1, keepdim=True).clamp_min(1e-12)
        r = (gn * pn).sum(1).double().cpu().numpy()
        r2 = np.clip(r ** 2, 0, 1 - 1e-12)
        tstat = np.sign(r) * np.sqrt(dof * r2 / (1 - r2))
        slope = r * std_ratio.double().cpu().numpy()
        slope_se = np.abs(slope) / np.maximum(np.abs(tstat), 1e-12)
        pval = get_t_pval(tstat, dof, log10=logp)
        rep_df = pd.DataFrame({
            'phenotype_id': res_df.index.to_numpy(), 'variant_id': res_df['variant_id'].to_numpy(),
            'ma_samples': ma_samples_t.cpu().numpy(), 'ma_count': ma_count_t.cpu().numpy(),
            'af': af_t.cpu().numpy(), 'pval_nominal': pval, 'slope': slope.astype(np.float32),
            'slope_se': slope_se.astype(np.float32),
        })
    else:
        if paired_covariate_df is not None:
            raise NotImplementedError("Paired covariates are not yet supported for interactions")
        i_t = torch.tensor(interaction_s.loc[sample_ids].to_numpy(dtype=np.float32), device=dev).unsqueeze(0)
        ng = genotypes_t.shape[0]
        gi_t = genotypes_t * i_t
        if residualizer is not None:
            g0, gi0, p0, i0 = residualizer.transform(genotypes_t, gi_t, phenotypes_t, i_t, center=True)
        else:
            g0 = genotypes_t - genotypes_t.mean(1, keepdim=True)
            gi0 = gi_t - gi_t.mean(1, keepdim=True)
            p0 = phenotypes_t - phenotypes_t.mean(1, keepdim=True)
            i0 = i_t - i_t.mean()
        X_t = torch.stack([g0, i0.expand(ng, -1), gi0], 2)                         # ng x ns x 3
        XtX = torch.matmul(X_t.transpose(1, 2), X_t)
        Xinv = torch.linalg.inv(XtX)
        b_t = (torch.matmul(Xinv, X_t.transpose(1, 2)) * p0.unsqueeze(1)).sum(2)   # ng x 3
        r_t = (X_t * b_t.unsqueeze(1)).sum(2) - p0
        dof = n - k_eff - 3
        rss_t = (r_t * r_t).sum(1)
        b_se_t = torch.sqrt(torch.diagonal(Xinv, dim1=1, dim2=2) * rss_t.unsqueeze(-1) / dof)
        tstat = (b_t.double() / b_se_t.double()).cpu().numpy()
        pval = get_t_pval(tstat, dof, log10=logp)
        b = b_t.cpu().numpy(); b_se = b_se_t.cpu().numpy()
        rep_df = pd.DataFrame({
            'phenotype_id': res_df.index.to_numpy(), 'variant_id': res_df['variant_id'].to_numpy(),
            'ma_samples': ma_samples_t.cpu().numpy(), 'ma_count': ma_count_t.cpu().numpy(),
            'af': af_t.cpu().numpy(),
            'pval_g': pval[:, 0], 'b_g': b[:, 0], 'b_g_se': b_se[:, 0],
            'pval_i': pval[:, 1], 'b_i': b[:, 1], 'b_i_se': b_se[:, 1],
            'pval_gi': pval[:, 2], 'b_gi': b[:, 2], 'b_gi_se': b_se[:, 2],
        })
        pval = pval[:, 2]

    if compute_pi1:
        try:
            p_in = np.asarray(pval, dtype=np.float64)
            if logp:
                p_in = 10 ** (-p_in)
            lam = None if lambda_qvalue is None else np.atleast_1d(lambda_qvalue)
            pi1 = float(np.clip(1 - float(pi0est(p_in, lambda_=lam)['pi0']), 0.0, 1.0))
        except Exception:
            pi1 = np.nan
        return pi1, rep_df
    return rep_df


# ---------------------------------------------------------------------------
# gene annotation
# ---------------------------------------------------------------------------
def _parse_gtf_genes(annotation_gtf: str) -> dict:
    import gzip
    opener = gzip.open if str(annotation_gtf).endswith('.gz') else open
    gene_dict = {}
    with opener(annotation_gtf, 'rt') as gtf:
        for row in gtf:
            if row.startswith('#'):
                continue
            row = row.rstrip('\n').split('\t')
            if len(row) < 9 or row[2] != 'gene':
                continue
            attr = {}
            for item in row[8].split(';'):
                item = item.strip()
                if not item:
                    continue
                parts = item.replace('"', '').split(None, 1)
                if len(parts) == 2:
                    attr[parts[0]] = parts[1]
            gid = attr.get('gene_id')
            if gid is None:
                continue
            gene_dict[gid] = [attr.get('gene_name', gid), row[0], int(row[3]), int(row[4]), row[6]]
    return gene_dict


def annotate_genes(gene_df: pd.DataFrame, annotation_gtf: str, lookup_df: pd.DataFrame | None = None,
                   verbose: bool = True) -> pd.DataFrame:
    """
    Add gene (``gene_name, gene_chr, gene_start, gene_end, strand`` from a GTF)
    and optional variant annotations (``lookup_df`` indexed by ``variant_id``)
    to phenotype-level output (``map_permutations`` + ``calculate_qvalues``;
    phenotype IDs as index, or ``group_id`` column for grouped results).
    Columns are ordered like tensorQTL's ``annotate_genes``; columns that are
    absent are skipped.
    """
    if verbose:
        print('[' + datetime.now().strftime("%b %d %H:%M:%S") + '] Adding gene and variant annotations', flush=True)
        print('  * parsing GTF', flush=True)
    gene_dict = _parse_gtf_genes(annotation_gtf)
    if verbose:
        print('  * annotating genes', flush=True)
    keys = gene_df['group_id'] if 'group_id' in gene_df else gene_df.index
    missing = [k for k in keys if k not in gene_dict]
    if missing:
        raise KeyError(f"{len(missing)} gene IDs not in GTF (first: {missing[:5]})")
    gene_info = pd.DataFrame([gene_dict[i] for i in keys],
                             columns=['gene_name', 'gene_chr', 'gene_start', 'gene_end', 'strand'],
                             index=gene_df.index)
    out = pd.concat([gene_info, gene_df], axis=1)
    col_order = ['gene_name', 'gene_chr', 'gene_start', 'gene_end', 'strand',
                 'num_var', 'beta_shape1', 'beta_shape2', 'true_dof', 'true_df', 'pval_true_dof',
                 'pval_true_df', 'variant_id']
    if 'tss_distance' in out:
        col_order += ['tss_distance']
    else:
        col_order += ['start_distance', 'end_distance']
    if lookup_df is not None:
        if verbose:
            print('  * adding variant annotations from lookup table', flush=True)
        out = out.join(lookup_df, on='variant_id')
        col_order += list(lookup_df.columns)
    col_order += ['ma_samples', 'ma_count', 'af', 'pval_nominal', 'slope', 'slope_se', 'pval_perm', 'pval_beta']
    if 'group_id' in out:
        col_order += ['group_id', 'group_size']
    col_order += ['qval', 'pval_nominal_threshold']
    present = [c for c in col_order if c in out.columns]
    rest = [c for c in out.columns if c not in present]
    if verbose:
        print('done.', flush=True)
    return out[present + rest]
