"""
trans-QTL mapping (tensorQTL ``trans`` module equivalent).

Adapted from tensorQTL
(https://github.com/broadinstitute/tensorqtl/blob/master/tensorqtl/trans.py).

Additions over tensorQTL:

* ``haplotypes``: a :class:`~localqtl.haplotypeio.LocalAncestryPanel` adds the
  per-variant local-ancestry counts as main-effect covariates
  (``y ~ g + h_1..h_{K-1}``); this requires ``variant_df`` (variant positions)
  and processes the genotypes chromosome by chromosome.
* explicit ``device`` selection and a rank-0 residualizer when no covariates
  are given (every design is centred, i.e. fitted with an intercept).
"""
from __future__ import annotations

import time
from collections import OrderedDict

import numpy as np
import pandas as pd
import scipy.stats as stats
import torch

from .genotypeio import background, print_progress
from .preproc import (allele_stats, filter_by_maf, filter_maf_interaction,
                      impute_mean_and_filter, interaction_half_mask)
from .stats import fit_beta_parameters, get_t_pval, pval_from_corr_r2
from .utils import SimpleLogger
from .cis.common import residualize_matrix_with_covariates, _stack_channels, _unstack_channels
from ._torch_utils import resolve_device

__all__ = [
    "GenotypeGeneratorTrans",
    "map_trans",
    "filter_cis",
    "map_trans_permutations",
    "map_permutations",
    "apply_permutations",
]


# ---------------------------------------------------------------------------
# Genotype generator
# ---------------------------------------------------------------------------
class GenotypeGeneratorTrans(object):
    """
    Generator for iterating over all variants (trans-scan), in batches of
    ``batch_size`` variants; with ``chr_s`` (``pd.Series`` variant -> chrom,
    same index as ``genotype_df``) batches never cross a chromosome and
    ``generate_data(chrom=...)`` restricts to one chromosome.

    Generates ``(genotype array (2D), variant ID array)``.
    """
    def __init__(self, genotype_df, batch_size=50000, chr_s=None):
        self.genotype_df = genotype_df
        self.batch_size = int(batch_size)
        n = self.genotype_df.shape[0]
        self.num_batches = int(np.ceil(n / self.batch_size))
        self.batch_indexes = [[i * self.batch_size, (i + 1) * self.batch_size] for i in range(self.num_batches)]
        if self.batch_indexes:
            self.batch_indexes[-1][1] = n
        self.chroms = None
        self.chr_batch_indexes = None
        if chr_s is not None:
            if not chr_s.index.equals(genotype_df.index):
                raise ValueError("chr_s must be indexed like genotype_df (variant IDs, same order).")
            chr_vals = np.asarray(chr_s.values).astype(str)
            chroms, chr_ix = np.unique(chr_vals, return_index=True)
            s = np.argsort(chr_ix)
            self.chroms = list(chroms[s])
            chr_ix = list(chr_ix[s]) + [n]
            self.chr_batch_indexes = {}
            for k, c in enumerate(self.chroms):
                size = chr_ix[k + 1] - chr_ix[k]
                nb = int(np.ceil(size / self.batch_size))
                bi = [[chr_ix[k] + i * self.batch_size, chr_ix[k] + (i + 1) * self.batch_size] for i in range(nb)]
                bi[-1][1] = chr_ix[k + 1]
                self.chr_batch_indexes[c] = bi
            # chromosomes must be contiguous blocks
            blocks = int((chr_vals[1:] != chr_vals[:-1]).sum()) + 1
            if blocks != len(self.chroms):
                raise ValueError("Variants must be sorted so that each chromosome is a contiguous block.")

    def __len__(self):
        return self.num_batches

    @background(max_prefetch=6)
    def generate_data(self, chrom=None, verbose=False, enum_start=1):
        """Generate batches from genotype data"""
        if chrom is None:
            batch_indexes = self.batch_indexes
            num_batches = self.num_batches
        else:
            batch_indexes = self.chr_batch_indexes[str(chrom)]
            num_batches = int(np.sum([len(i) for i in self.chr_batch_indexes.values()]))
        for k, i in enumerate(batch_indexes, enum_start):
            if verbose:
                print_progress(k, num_batches, 'batch')
            g = self.genotype_df.values[i[0]:i[1]]
            ix = self.genotype_df.index[i[0]:i[1]]
            yield g, ix


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _genotype_sample_index(genotype_df: pd.DataFrame, sample_ids) -> np.ndarray:
    cols = pd.Index(genotype_df.columns.astype(str))
    ix = cols.get_indexer(pd.Index(sample_ids).astype(str))
    if (ix < 0).any():
        missing = list(pd.Index(sample_ids)[ix < 0][:5])
        raise ValueError(f"{int((ix < 0).sum())} phenotype samples missing from genotype_df (first: {missing})")
    return ix


class _PanelHapSource:
    """Serve per-variant local-ancestry main-effect columns for trans batches."""

    def __init__(self, panel, variant_df: pd.DataFrame, device: torch.device):
        from .haplotypeio import _fill_missing_along_loci, _is_panel
        if not _is_panel(panel):
            raise ValueError("map_trans accepts local ancestry only as a LocalAncestryPanel "
                             "(RFMixReader / FlareReader / open_panel).")
        self.panel = panel
        self.device = device
        self._fill = _fill_missing_along_loci
        self.variant_df = variant_df
        self.chrom = variant_df["chrom"].astype(str).to_numpy()
        self.pos = variant_df["pos"].to_numpy(dtype=np.int64)
        self.geno2hap = np.full(len(variant_df), -1, dtype=np.int64)
        for c in pd.unique(self.chrom):
            m = np.flatnonzero(self.chrom == c)
            self.geno2hap[m] = panel.locus_index(c, self.pos[m])
        self.mapped = self.geno2hap >= 0
        self.n_unmapped = int((~self.mapped).sum())
        self._chrom = None
        self._counts_t = None
        # channels: the dosage of every ancestry but the last (collinear with the intercept)
        self.n_channels = panel.n_channels if panel.n_channels == 1 else panel.n_channels - 1

    def block(self, chrom: str, row_idx: np.ndarray) -> torch.Tensor:
        """``(m, n, pH)`` float32 counts for genotype rows ``row_idx`` (all on ``chrom``)."""
        chrom = str(chrom)
        if self._chrom != chrom:
            counts = self._fill(self.panel.chrom_counts(chrom))
            t = torch.as_tensor(np.asarray(counts, dtype=np.float32), device=self.device)
            if t.shape[2] > 1:
                t = t[:, :, :-1]
            self._counts_t, self._chrom = t.contiguous(), chrom
        hap_idx = torch.as_tensor(self.geno2hap[row_idx], dtype=torch.long, device=self.device)
        H = torch.index_select(self._counts_t, 0, hap_idx)
        H = torch.nan_to_num(H, nan=0.0)
        return H


def _project_out_H(G: torch.Tensor, H: torch.Tensor, Y: torch.Tensor):
    """
    Per-variant Frisch–Waugh projection of ``G`` (m, n) and ``Y`` (P, n) on the
    per-variant nuisance columns ``H`` (m, n, pH) (all centred / residualized).
    Returns ``Gy_resid`` (m, P), ``G_resid2`` (m,), ``y_resid2`` (m, P).
    """
    EPS = 1e-8
    pH = int(H.shape[2])
    S = torch.einsum("mnp,mnq->mpq", H, H)
    I = torch.eye(pH, device=H.device, dtype=S.dtype).expand_as(S)
    L, info = torch.linalg.cholesky_ex(S)
    if (info != 0).any():
        L = torch.linalg.cholesky(S + 1e-6 * I)
    Hy = torch.einsum("mnp,qn->mpq", H, Y)                        # (m, pH, P)
    Hg = torch.einsum("mnp,mn->mp", H, G)                          # (m, pH)
    alpha_y = torch.cholesky_solve(Hy, L)                          # (m, pH, P)
    alpha_g = torch.cholesky_solve(Hg.unsqueeze(-1), L).squeeze(-1)
    G2 = (G * G).sum(1) + EPS
    Y2 = (Y * Y).sum(1) + EPS                                      # (P,)
    GY = G @ Y.T                                                   # (m, P)
    G_resid2 = G2 - (Hg * alpha_g).sum(1)
    y_resid2 = Y2.unsqueeze(0) - (Hy * alpha_y).sum(1)
    Gy_resid = GY - torch.einsum("mp,mpq->mq", Hg, alpha_y)
    return Gy_resid, G_resid2.clamp_min(EPS), y_resid2.clamp_min(EPS)


def _corr_stats(G_resid: torch.Tensor, Y_resid: torch.Tensor, H_resid: torch.Tensor | None):
    """
    Correlation ``r`` (m, P) between residualized genotypes and phenotypes,
    plus the std ratio needed for slopes (``b = r * sd_y / sd_g``) and the
    residual variances used after projecting per-variant ``H`` columns.
    """
    EPS = 1e-8
    if H_resid is None:
        gn = torch.linalg.norm(G_resid, dim=1).clamp_min(EPS)
        yn = torch.linalg.norm(Y_resid, dim=1).clamp_min(EPS)
        r = (G_resid / gn.unsqueeze(1)) @ (Y_resid / yn.unsqueeze(1)).T
        std_ratio = (yn.unsqueeze(0) / gn.unsqueeze(1))
        return r, std_ratio
    Gy, G2, y2 = _project_out_H(G_resid, H_resid, Y_resid)
    r = Gy / torch.sqrt(G2.unsqueeze(1) * y2)
    std_ratio = torch.sqrt(y2 / G2.unsqueeze(1))
    return r, std_ratio


def _resolve_haps(haplotypes, variant_df, genotype_df, device):
    if haplotypes is None:
        return None, None
    if variant_df is None:
        raise ValueError("variant_df (chrom, pos) is required when haplotypes are given.")
    if not genotype_df.index.equals(variant_df.index):
        raise ValueError("genotype_df and variant_df must share the same variant index/order.")
    src = _PanelHapSource(haplotypes, variant_df, device)
    chr_s = pd.Series(variant_df["chrom"].astype(str).to_numpy(), index=genotype_df.index)
    return src, chr_s


# ---------------------------------------------------------------------------
# trans mapping
# ---------------------------------------------------------------------------
def map_trans(genotype_df, phenotype_df, covariates_df=None, interaction_s=None, *,
              haplotypes=None, variant_df=None,
              return_sparse=True, pval_threshold=1e-5, maf_threshold=0.05,
              alleles=2, return_r2=False, batch_size=20000, maf_threshold_interaction=None,
              logp=False, device="auto", logger=None, verbose=True):
    """
    Run trans-QTL mapping (all variants x all phenotypes).

    Parameters mirror tensorQTL's ``trans.map_trans``.  ``interaction_s`` adds
    the ``y ~ g + i + g:i`` model (single interaction).  ``haplotypes``
    (a LocalAncestryPanel) adds local-ancestry main effects per variant.

    Outputs (``return_sparse=True``):
      DataFrame ``variant_id, phenotype_id, pval, b, b_se, [r2,] af``
      (interaction: ``variant_id, phenotype_id, pval_g, b_g, b_g_se, pval_i,
      b_i, b_i_se, pval_gi, b_gi, b_gi_se, af``, thresholded on ``pval_gi``).
    Outputs (``return_sparse=False``):
      ``pval_df, b_df, b_se_df, af_s`` (interaction: ``pval_g_df, pval_i_df,
      pval_gi_df, af_s, ma_samples_s, ma_count_s``), variants x phenotypes.
    """
    dev = resolve_device(device)
    device_str = dev.type
    logger = logger or SimpleLogger(verbose=verbose)

    n_samples = phenotype_df.shape[1]
    logger.write('trans-QTL mapping')
    logger.write(f'  * device: {device_str}')
    logger.write(f'  * {n_samples} samples')
    logger.write(f'  * {phenotype_df.shape[0]} phenotypes')
    if covariates_df is not None:
        if not phenotype_df.columns.equals(covariates_df.index):
            raise ValueError("covariates_df must be indexed by the phenotype samples (same order).")
        logger.write(f'  * {covariates_df.shape[1]} covariates')
    logger.write(f'  * {genotype_df.shape[0]} variants')
    if interaction_s is not None:
        logger.write('  * including interaction term')
    if haplotypes is not None:
        logger.write('  * including local ancestry main effects')

    genotype_ix_t = torch.as_tensor(_genotype_sample_index(genotype_df, phenotype_df.columns), device=dev)
    hap_src, chr_s = _resolve_haps(haplotypes, variant_df, genotype_df, dev)
    if hap_src is not None and hap_src.n_unmapped:
        logger.write(f'    ** {hap_src.n_unmapped} variants without a local-ancestry locus will be skipped')
    pH = hap_src.n_channels if hap_src is not None else 0

    Y = torch.tensor(phenotype_df.values, dtype=torch.float32, device=dev)
    Y_resid, rez = residualize_matrix_with_covariates(Y, covariates_df, device_str)
    k_eff = rez.k_eff

    ggt = GenotypeGeneratorTrans(genotype_df, batch_size=batch_size, chr_s=chr_s)
    chrom_iter = [None] if chr_s is None else list(ggt.chroms)

    def _batches():
        for c in chrom_iter:
            for g, ids in ggt.generate_data(chrom=c, verbose=verbose):
                yield c, g, ids

    def _prep_batch(c, g, ids):
        """impute, MAF-filter, residualize; returns tensors or None."""
        G_t = torch.tensor(np.asarray(g), dtype=torch.float32, device=dev)[:, genotype_ix_t]
        ids = np.asarray(ids)
        if hap_src is not None:
            rows = genotype_df.index.get_indexer(ids)
            keep = hap_src.mapped[rows]
            if not keep.all():
                G_t, ids, rows = G_t[torch.as_tensor(keep, device=dev)], ids[keep], rows[keep]
            if G_t.shape[0] == 0:
                return None
        G_t, keep_mono, _ = impute_mean_and_filter(G_t)
        if G_t.shape[0] == 0:
            return None
        km = keep_mono.detach().cpu().numpy()
        ids = ids[km]
        if hap_src is not None:
            rows = rows[km]
        if maf_threshold and maf_threshold > 0:
            keep_maf, _ = filter_by_maf(G_t, maf_threshold, ploidy=alleles)
            if keep_maf.sum().item() == 0:
                return None
            G_t = G_t[keep_maf]
            kmf = keep_maf.detach().cpu().numpy()
            ids = ids[kmf]
            if hap_src is not None:
                rows = rows[kmf]
        af_t, ma_samples_t, ma_count_t = allele_stats(G_t, ploidy=alleles)
        H_t = None
        if hap_src is not None:
            H_t = hap_src.block(c, rows)
        return G_t, ids, af_t, ma_samples_t, ma_count_t, H_t

    start_time = time.time()
    n_variants = 0

    if interaction_s is None:
        dof = max(n_samples - int(k_eff) - 1 - pH, 1)
        if return_sparse:
            tstat_threshold = -stats.t.ppf(pval_threshold / 2, dof)
            r_threshold = tstat_threshold / np.sqrt(dof + tstat_threshold ** 2)
        res = []
        for c, g, ids in _batches():
            prepped = _prep_batch(c, g, ids)
            if prepped is None:
                continue
            G_t, ids, af_t, _, _, H_t = prepped
            n_variants += G_t.shape[0]
            if H_t is None:
                (G_resid,) = rez.transform(G_t, center=True)
                H_resid = None
            else:
                m, n, p = H_t.shape
                G_resid, Hs = rez.transform(G_t, _stack_channels(H_t), center=True)
                H_resid = _unstack_channels(Hs, m, n, p)
            r_t, std_ratio_t = _corr_stats(G_resid, Y_resid, H_resid)
            del G_resid, H_resid
            if return_sparse:
                mask = r_t.abs() >= r_threshold
                ix_t = mask.nonzero(as_tuple=False)
                if ix_t.shape[0] == 0:
                    continue
                r_sel = r_t[mask].double()
                r2_sel = r_sel.pow(2).clamp(max=1 - 1e-12)
                t_sel = r_sel * torch.sqrt(dof / (1 - r2_sel))
                b_sel = r_sel * std_ratio_t[mask].double()
                b_se_sel = (b_sel / t_sel).abs()
                ix = ix_t.cpu().numpy()
                res.append(pd.DataFrame({
                    "variant_id": ids[ix[:, 0]],
                    "phenotype_id": phenotype_df.index.to_numpy()[ix[:, 1]],
                    "tstat": t_sel.cpu().numpy(),
                    "b": b_sel.float().cpu().numpy(),
                    "b_se": b_se_sel.float().cpu().numpy(),
                    "r2": r2_sel.float().cpu().numpy(),
                    "af": af_t[ix_t[:, 0]].cpu().numpy(),
                }))
            else:
                r_d = r_t.double()
                r2_d = r_d.pow(2).clamp(max=1 - 1e-12)
                t_d = r_d * torch.sqrt(dof / (1 - r2_d))
                b_d = r_d * std_ratio_t.double()
                b_se_d = (b_d / t_d).abs()
                res.append([ids, t_d.cpu().numpy(), b_d.float().cpu().numpy(),
                            b_se_d.float().cpu().numpy(), af_t.cpu().numpy()])
        logger.write(f'    elapsed time: {(time.time() - start_time) / 60:.2f} min')
        if maf_threshold and maf_threshold > 0:
            logger.write(f'  * {n_variants} variants passed MAF >= {maf_threshold} filtering')

        if return_sparse:
            cols = ['variant_id', 'phenotype_id', 'pval', 'b', 'b_se', 'r2', 'af']
            if not res:
                pval_df = pd.DataFrame(columns=cols)
            else:
                pval_df = pd.concat(res, ignore_index=True)
                pval_df.insert(2, 'pval', get_t_pval(pval_df['tstat'].to_numpy(np.float64), dof, log10=logp))
                pval_df = pval_df.drop(columns='tstat')[cols]
            if not return_r2:
                pval_df = pval_df.drop(columns='r2')
            logger.write('done.')
            return pval_df
        if not res:
            empty = pd.DataFrame(columns=phenotype_df.index)
            return empty, empty.copy(), empty.copy(), pd.Series(dtype=np.float32, name='af')
        variant_ids = pd.Index(np.concatenate([i[0] for i in res]), name='variant_id')
        pval_df = pd.DataFrame(get_t_pval(np.concatenate([i[1] for i in res]), dof, log10=logp),
                               index=variant_ids, columns=phenotype_df.index)
        b_df = pd.DataFrame(np.concatenate([i[2] for i in res]), index=variant_ids, columns=phenotype_df.index)
        b_se_df = pd.DataFrame(np.concatenate([i[3] for i in res]), index=variant_ids, columns=phenotype_df.index)
        af_s = pd.Series(np.concatenate([i[4] for i in res]), index=variant_ids, name='af')
        logger.write('done.')
        return pval_df, b_df, b_se_df, af_s

    # ---- interaction model: y ~ g + i + g:i (+ H) -------------------------
    if isinstance(interaction_s, pd.DataFrame):
        if interaction_s.shape[1] != 1:
            raise NotImplementedError('trans-QTL mapping supports a single interaction term.')
        interaction_s = interaction_s.iloc[:, 0]
    interaction_s = interaction_s.loc[phenotype_df.columns]
    i_np = interaction_s.to_numpy(dtype=np.float32)
    if maf_threshold_interaction is None:
        maf_threshold_interaction = maf_threshold
    interaction_mask_t = interaction_half_mask(i_np).to(dev) if maf_threshold_interaction > 0 else None
    i_t = torch.as_tensor(i_np, device=dev)
    (i0_t,) = rez.transform(i_t.unsqueeze(0), center=True)           # (1, n)
    nb = 3 + pH
    dof = max(n_samples - int(k_eff) - nb, 1)
    if return_sparse:
        tstat_threshold = -stats.t.ppf(pval_threshold / 2, dof)
    Y2 = (Y_resid * Y_resid).sum(1)                                   # (P,)
    out = []
    for c, g, ids in _batches():
        G_t = torch.tensor(np.asarray(g), dtype=torch.float32, device=dev)[:, genotype_ix_t]
        ids = np.asarray(ids)
        rows = genotype_df.index.get_indexer(ids) if hap_src is not None else None
        if hap_src is not None:
            keep = hap_src.mapped[rows]
            G_t, ids, rows = G_t[torch.as_tensor(keep, device=dev)], ids[keep], rows[keep]
        if G_t.shape[0] == 0:
            continue
        G_t, keep_mono, _ = impute_mean_and_filter(G_t)
        if G_t.shape[0] == 0:
            continue
        km = keep_mono.detach().cpu().numpy()
        ids = ids[km]
        rows = rows[km] if rows is not None else None
        G_t, mask_t = filter_maf_interaction(G_t, interaction_mask_t, maf_threshold_interaction)
        if G_t.shape[0] == 0:
            continue
        mk = mask_t.detach().cpu().numpy()
        ids = ids[mk]
        rows = rows[mk] if rows is not None else None
        n_variants += G_t.shape[0]
        af_t, ma_samples_t, ma_count_t = allele_stats(G_t, ploidy=alleles)
        m = G_t.shape[0]
        GI = G_t * i_t.unsqueeze(0)
        mats = [G_t, GI]
        if hap_src is not None:
            H_t = hap_src.block(c, rows)
            mats.append(_stack_channels(H_t))
        resid = rez.transform(*mats, center=True)
        g0, gi0 = resid[0], resid[1]
        cols = [g0.unsqueeze(-1), i0_t.expand(m, -1).unsqueeze(-1), gi0.unsqueeze(-1)]
        if hap_src is not None:
            cols.append(_unstack_channels(resid[2], m, n_samples, pH))
        X = torch.cat(cols, dim=2)                                    # (m, n, nb)
        XtX = X.transpose(1, 2) @ X
        eps = (1e-6 * torch.diagonal(XtX, dim1=1, dim2=2).abs().mean(1, keepdim=True) + 1e-8).unsqueeze(-1)
        XtX = XtX + eps * torch.eye(nb, device=dev, dtype=XtX.dtype).unsqueeze(0)
        XtY = torch.einsum("mnb,qn->mbq", X, Y_resid)                 # (m, nb, P)
        XtX_inv = torch.linalg.inv(XtX)
        B = XtX_inv @ XtY                                             # (m, nb, P)
        rss = (Y2.unsqueeze(0) - (XtY * B).sum(1)).clamp_min(1e-12)   # (m, P)
        diag = torch.diagonal(XtX_inv, dim1=1, dim2=2)                # (m, nb)
        SE = torch.sqrt(diag.unsqueeze(-1) * rss.unsqueeze(1) / dof)  # (m, nb, P)
        T = (B.double() / SE.double()).float()
        if return_sparse:
            mask = T[:, 2, :].abs() >= tstat_threshold
            ix_t = mask.nonzero(as_tuple=False)
            if ix_t.shape[0] == 0:
                continue
            ix = ix_t.cpu().numpy()
            Tm = T[ix_t[:, 0], :, ix_t[:, 1]].cpu().numpy()           # (k, nb)
            Bm = B[ix_t[:, 0], :, ix_t[:, 1]].cpu().numpy()
            Sm = SE[ix_t[:, 0], :, ix_t[:, 1]].cpu().numpy()
            out.append(pd.DataFrame({
                "variant_id": ids[ix[:, 0]],
                "phenotype_id": phenotype_df.index.to_numpy()[ix[:, 1]],
                "tstat_g": Tm[:, 0], "b_g": Bm[:, 0], "b_g_se": Sm[:, 0],
                "tstat_i": Tm[:, 1], "b_i": Bm[:, 1], "b_i_se": Sm[:, 1],
                "tstat_gi": Tm[:, 2], "b_gi": Bm[:, 2], "b_gi_se": Sm[:, 2],
                "af": af_t[ix_t[:, 0]].cpu().numpy(),
            }))
        else:
            out.append([ids, T[:, :3, :].cpu().numpy(), af_t.cpu().numpy(),
                        ma_samples_t.cpu().numpy(), ma_count_t.cpu().numpy()])
    logger.write(f'    time elapsed: {(time.time() - start_time) / 60:.2f} min')
    if return_sparse:
        cols = ['variant_id', 'phenotype_id', 'pval_g', 'b_g', 'b_g_se', 'pval_i', 'b_i', 'b_i_se',
                'pval_gi', 'b_gi', 'b_gi_se', 'af']
        if not out:
            return pd.DataFrame(columns=cols)
        df = pd.concat(out, ignore_index=True)
        for k in ("g", "i", "gi"):
            df[f"pval_{k}"] = get_t_pval(df[f"tstat_{k}"].to_numpy(np.float64), dof, log10=logp)
        return df[cols]
    if not out:
        empty = pd.DataFrame(columns=phenotype_df.index)
        return (empty, empty.copy(), empty.copy(), pd.Series(dtype=np.float32, name='af'),
                pd.Series(dtype=np.int32, name='ma_samples'), pd.Series(dtype=np.int32, name='ma_count'))
    variant_ids = pd.Index(np.concatenate([i[0] for i in out]), name='variant_id')
    tstat = np.concatenate([i[1] for i in out])                       # (M, 3, P)
    pval = get_t_pval(tstat, dof, log10=logp)
    pval_g_df = pd.DataFrame(pval[:, 0, :], index=variant_ids, columns=phenotype_df.index)
    pval_i_df = pd.DataFrame(pval[:, 1, :], index=variant_ids, columns=phenotype_df.index)
    pval_gi_df = pd.DataFrame(pval[:, 2, :], index=variant_ids, columns=phenotype_df.index)
    af_s = pd.Series(np.concatenate([i[2] for i in out]), index=variant_ids, name='af').astype(np.float32)
    ma_samples_s = pd.Series(np.concatenate([i[3] for i in out]), index=variant_ids, name='ma_samples').astype(np.int32)
    ma_count_s = pd.Series(np.concatenate([i[4] for i in out]), index=variant_ids, name='ma_count').astype(np.int32)
    return pval_g_df, pval_i_df, pval_gi_df, af_s, ma_samples_s, ma_count_s


def filter_cis(pairs_df, phenotype_pos_df, variant_df, window=5_000_000):
    """
    Drop pairs where the variant lies within ``window`` of the phenotype
    (``pos``, or ``[start, end]``) on the same chromosome (cis pairs).
    """
    pos = phenotype_pos_df.copy()
    if 'pos' in pos.columns:
        pos = pos.rename(columns={'pos': 'start'})
        pos['end'] = pos['start']
    ph = pos.reindex(pairs_df['phenotype_id'].to_numpy())
    va = variant_df.reindex(pairs_df['variant_id'].to_numpy())
    same_chr = ph['chr'].astype(str).to_numpy() == va['chrom'].astype(str).to_numpy()
    vpos = pd.to_numeric(va['pos']).to_numpy()
    in_cis = same_chr & (vpos >= pd.to_numeric(ph['start']).to_numpy() - window) \
        & (vpos <= pd.to_numeric(ph['end']).to_numpy() + window)
    in_cis = np.where(np.isnan(vpos) | ph['chr'].isna().to_numpy(), False, in_cis)
    return pairs_df.loc[~in_cis]


# ---------------------------------------------------------------------------
# permutations
# ---------------------------------------------------------------------------
def map_trans_permutations(genotype_df, covariates_df=None, permutations=None, *,
                           phenotype_df=None, chr_s=None, nperms=10000, maf_threshold=0.05,
                           batch_size=20000, haplotypes=None, variant_df=None, logger=None,
                           seed=None, device="auto", verbose=True):
    """
    Genome-wide permutation null for trans-QTL mapping (tensorQTL
    ``trans.map_permutations``).  Random normal "phenotypes" (the quantiles of
    the standard normal, shuffled per permutation) are correlated with every
    variant and the maximum ``r2`` per permutation is recorded.  This assumes
    the real phenotypes are (inverse-)normal transformed.

    Parameters
    ----------
    covariates_df : DataFrame, optional (samples x covariates)
        Defines the samples; pass ``phenotype_df`` instead when there are no
        covariates (its columns define the samples).
    permutations : ndarray (nperms x samples), optional
        Pre-generated null phenotypes.
    chr_s : Series (variant -> chromosome), optional
        Split the null by chromosome and compute a leave-one-chromosome-out
        maximum (so that a phenotype on chromosome ``c`` is calibrated against
        trans variants only).  Returns a DataFrame indexed by chromosome.
    haplotypes, variant_df
        Local-ancestry main effects, as in :func:`map_trans`.

    Returns
    -------
    pandas.Series or pandas.DataFrame
        Without ``chr_s`` a Series with ``num_samples``, ``dof``,
        ``beta_shape1``, ``beta_shape2``, ``true_dof``, ``minp_true_dof`` and
        ``minp_empirical``; with ``chr_s`` a DataFrame with one row per
        chromosome.  Both are consumed by :func:`apply_permutations`.
    """
    dev = resolve_device(device)
    logger = logger or SimpleLogger(verbose=verbose)
    if covariates_df is not None:
        sample_ids = pd.Index(covariates_df.index)
    elif phenotype_df is not None:
        sample_ids = pd.Index(phenotype_df.columns)
    else:
        raise ValueError("Provide covariates_df (samples x covariates) or phenotype_df to define the samples.")
    genotype_ix_t = torch.as_tensor(_genotype_sample_index(genotype_df, sample_ids), device=dev)
    n_samples = len(sample_ids)
    n_variants = genotype_df.shape[0]

    logger.write('trans-QTL mapping (permutations)')
    logger.write(f'  * {n_samples} samples')
    logger.write(f'  * {0 if covariates_df is None else covariates_df.shape[1]} covariates')
    logger.write(f'  * {n_variants} variants')

    if permutations is None:
        q = stats.norm.ppf(np.arange(1, n_samples + 1) / (n_samples + 1))
        permutations = np.tile(q, [nperms, 1])
        rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()
        for i in range(nperms):
            rng.shuffle(permutations[i, :])
    else:
        permutations = np.asarray(permutations)
        if permutations.shape[1] != n_samples:
            raise ValueError("permutations must be (nperms x samples).")
        nperms = permutations.shape[0]
        logger.write(f'  * {nperms} permutations')

    Pm = torch.tensor(permutations, dtype=torch.float32, device=dev)
    P_resid, rez = residualize_matrix_with_covariates(Pm, covariates_df, dev.type)
    k_eff = rez.k_eff

    hap_src, hap_chr_s = _resolve_haps(haplotypes, variant_df, genotype_df, dev)
    pH = hap_src.n_channels if hap_src is not None else 0
    dof = max(n_samples - int(k_eff) - 1 - pH, 1)
    if chr_s is None and hap_chr_s is not None:
        chr_s = hap_chr_s
        split_chr = False
    else:
        split_chr = chr_s is not None
    ggt = GenotypeGeneratorTrans(genotype_df, batch_size=batch_size, chr_s=chr_s)

    def _batch_max_r2(c, g, ids, max_r2_t):
        G_t = torch.tensor(np.asarray(g), dtype=torch.float32, device=dev)[:, genotype_ix_t]
        ids = np.asarray(ids)
        rows = None
        if hap_src is not None:
            rows = genotype_df.index.get_indexer(ids)
            keep = hap_src.mapped[rows]
            G_t, rows = G_t[torch.as_tensor(keep, device=dev)], rows[keep]
            if G_t.shape[0] == 0:
                return 0
        G_t, keep_mono, _ = impute_mean_and_filter(G_t)
        if G_t.shape[0] == 0:
            return 0
        if rows is not None:
            rows = rows[keep_mono.detach().cpu().numpy()]
        if maf_threshold and maf_threshold > 0:
            keep_maf, _ = filter_by_maf(G_t, maf_threshold)
            if keep_maf.sum().item() == 0:
                return 0
            G_t = G_t[keep_maf]
            if rows is not None:
                rows = rows[keep_maf.detach().cpu().numpy()]
        if hap_src is None:
            (G_resid,) = rez.transform(G_t, center=True)
            H_resid = None
        else:
            H_t = hap_src.block(c, rows)
            m, n, p = H_t.shape
            G_resid, Hs = rez.transform(G_t, _stack_channels(H_t), center=True)
            H_resid = _unstack_channels(Hs, m, n, p)
        r_t, _ = _corr_stats(G_resid, P_resid, H_resid)
        r2 = torch.nan_to_num(r_t.pow(2), nan=0.0)
        mx, _ = r2.max(0)
        torch.maximum(mx, max_r2_t, out=max_r2_t)
        return int(G_t.shape[0])

    start_time = time.time()
    n_kept = 0
    if split_chr:
        chr_max_r2 = OrderedDict()
        for chrom in ggt.chroms:
            max_r2_t = torch.zeros(nperms, device=dev)
            for g, ids in ggt.generate_data(chrom=chrom, verbose=verbose):
                n_kept += _batch_max_r2(chrom, g, ids, max_r2_t)
            chr_max_r2[chrom] = max_r2_t.cpu().numpy().astype(np.float64)
        logger.write(f'    time elapsed: {(time.time() - start_time) / 60:.2f} min')
        if maf_threshold and maf_threshold > 0:
            logger.write(f'  * {n_kept} variants passed MAF >= {maf_threshold} filtering')
        chr_max_r2 = pd.DataFrame(chr_max_r2)
        # leave-one-chromosome-out maximum
        max_r2 = OrderedDict()
        for c in chr_max_r2.columns:
            others = [x for x in chr_max_r2.columns if x != c]
            max_r2[c] = chr_max_r2[others].max(1) if others else chr_max_r2[c]
        max_r2 = pd.DataFrame(max_r2)
        rows = []
        for c in max_r2.columns:
            a, b, tdof, minp = fit_beta_parameters(max_r2[c].to_numpy(), dof, return_minp=True)
            rows.append({
                'num_samples': n_samples, 'dof': dof,
                'beta_shape1': a, 'beta_shape2': b, 'true_dof': tdof,
                'minp_true_dof': minp,
                'minp_empirical': pval_from_corr_r2(max_r2[c].to_numpy(), dof),
            })
        return pd.DataFrame(rows, index=pd.Index(list(max_r2.columns), name='chr'))

    max_r2_t = torch.zeros(nperms, device=dev)
    chroms = [None] if chr_s is None else list(ggt.chroms)
    for c in chroms:
        for g, ids in ggt.generate_data(chrom=c, verbose=verbose):
            n_kept += _batch_max_r2(c, g, ids, max_r2_t)
    logger.write(f'    time elapsed: {(time.time() - start_time) / 60:.2f} min')
    if maf_threshold and maf_threshold > 0:
        logger.write(f'  * {n_kept} variants passed MAF >= {maf_threshold} filtering')
    max_r2 = max_r2_t.cpu().numpy().astype(np.float64)
    minp_empirical = pval_from_corr_r2(max_r2, dof)
    a, b, tdof, minp = fit_beta_parameters(max_r2, dof, tol=1e-4, return_minp=True)
    return pd.Series([n_samples, dof, a, b, tdof, minp, minp_empirical],
                     index=['num_samples', 'dof', 'beta_shape1', 'beta_shape2', 'true_dof',
                            'minp_true_dof', 'minp_empirical'])


# tensorQTL name within this module
map_permutations = map_trans_permutations


def apply_permutations(res, pairs_df, phenotype_pos_df=None):
    """
    Attach permutation-based significance to the sparse output of
    :func:`map_trans` (requires ``return_r2=True``).

    ``res`` is the output of :func:`map_trans_permutations`; for the
    per-chromosome DataFrame form ``pairs_df`` needs a ``phenotype_chr``
    column or ``phenotype_pos_df`` must be given.  Adds ``beta_shape1,
    beta_shape2, true_dof, pval_true_dof, pval_perm, pval_beta`` (in place
    and returned).
    """
    if 'r2' not in pairs_df.columns:
        raise ValueError("pairs_df needs an 'r2' column: run map_trans(..., return_r2=True).")
    if isinstance(res, pd.Series):
        nperms = len(res['minp_true_dof'])
        for k in ['beta_shape1', 'beta_shape2', 'true_dof']:
            pairs_df[k] = res[k]
        pairs_df['pval_true_dof'] = pval_from_corr_r2(pairs_df['r2'].to_numpy(np.float64), float(res['true_dof']))
        minp = np.sort(np.asarray(res['minp_empirical'], dtype=np.float64))
        pairs_df['pval_perm'] = (np.searchsorted(minp, pairs_df['pval'].to_numpy(np.float64), side='right') + 1) / (nperms + 1)
        pairs_df['pval_beta'] = stats.beta.cdf(pairs_df['pval_true_dof'], pairs_df['beta_shape1'], pairs_df['beta_shape2'])
        return pairs_df
    if isinstance(res, pd.DataFrame):
        if 'phenotype_chr' not in pairs_df.columns:
            if phenotype_pos_df is None:
                raise ValueError("pairs_df needs 'phenotype_chr' or pass phenotype_pos_df.")
            pairs_df['phenotype_chr'] = phenotype_pos_df['chr'].astype(str).reindex(pairs_df['phenotype_id'].to_numpy()).to_numpy()
        chrs = pairs_df['phenotype_chr'].astype(str).to_numpy()
        missing = set(np.unique(chrs)) - set(res.index.astype(str))
        if missing:
            raise ValueError(f"No permutation null for phenotype chromosome(s): {sorted(missing)}")
        res = res.copy(); res.index = res.index.astype(str)
        for k in ['beta_shape1', 'beta_shape2', 'true_dof']:
            pairs_df[k] = res.loc[chrs, k].to_numpy()
        pairs_df['pval_true_dof'] = pval_from_corr_r2(pairs_df['r2'].to_numpy(np.float64), pairs_df['true_dof'].to_numpy(np.float64))
        pval_perm = np.empty(len(pairs_df))
        pvals = pairs_df['pval'].to_numpy(np.float64)
        for c in np.unique(chrs):
            minp = np.sort(np.asarray(res.loc[c, 'minp_empirical'], dtype=np.float64))
            m = chrs == c
            pval_perm[m] = (np.searchsorted(minp, pvals[m], side='right') + 1) / (len(minp) + 1)
        pairs_df['pval_perm'] = pval_perm
        pairs_df['pval_beta'] = stats.beta.cdf(pairs_df['pval_true_dof'], pairs_df['beta_shape1'], pairs_df['beta_shape2'])
        return pairs_df
    raise TypeError("res must be the Series or DataFrame returned by map_trans_permutations.")
