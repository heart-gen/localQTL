"""
Functions for reading dosages from PLINK pgen files based on the Pgenlib Python API:
https://github.com/chrchang/plink-ng/blob/master/2.0/Python/python_api.txt

This was adapted from tensorQTL:
https://github.com/broadinstitute/tensorqtl/blob/master/tensorqtl/pgen.py
"""

import os
import torch
import bisect
import numpy as np
import pandas as pd
import pgenlib as pg

__all__ = [
    "read_pvar",
    "read_psam",
    "PgenReader",
    "get_chunk_size",
    "generate_paired_chunks",
]


def read_pvar(pvar_path):
    """Read pvar file as pd.DataFrame"""
    return pd.read_csv(
        pvar_path, sep='\t', comment='#',
        names=['chrom', 'pos', 'id', 'ref', 'alt', 'qual', 'filter', 'info'],
        dtype={'chrom':str, 'pos':np.int32, 'id':str, 'ref':str, 'alt':str,
               'qual':str, 'filter':str, 'info':str}
    )


def read_psam(psam_path):
    """Read psam file as pd.DataFrame"""
    psam_df = pd.read_csv(psam_path, sep='\t', index_col=0)
    psam_df.index = psam_df.index.astype(str)
    return psam_df


def _impute_mean(genotypes, missing_code=-9):
    """Impute missing genotypes (-9) with per-variant mean."""
    arr = genotypes.astype(np.float32, copy=True)
    mask = arr == float(missing_code)

    if arr.ndim == 1:
        if mask.all():
            arr.fill(0.0)
        elif mask.any():
            arr[mask] = arr[~mask].mean(dtype=np.float32)
    else:
        # Compute per-row means excluding missing entries
        valid = ~mask
        counts = valid.sum(axis=1, keepdims=True)
        sums = np.where(valid, arr, 0.0).sum(axis=1, keepdims=True)
        means = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
        row_means = means.squeeze(1)
        if row_means.ndim == 0:
            row_means = np.array([row_means], dtype=arr.dtype)
        arr[mask] = row_means[np.nonzero(mask)[0]]

    return arr


class PgenReader(object):
    """
    Slim PGEN reader class for PLINK2 .pgen/.psam/.pvar files.
    Optimized for QTL mapping with DataFrame compatibility.

    Requires pgenlib: https://github.com/chrchang/plink-ng/tree/master/2.0/Python
    """
    def __init__(self, plink_prefix, select_samples=None, impute=True,
                 dtype=np.float32, device="cpu"):
        self.pvar_df = (
            pd.read_parquet(f"{plink_prefix}.pvar.parquet")
            if os.path.exists(f"{plink_prefix}.pvar.parquet")
            else read_pvar(f"{plink_prefix}.pvar")
        )
        self.psam_df = read_psam(f"{plink_prefix}.psam")
        self.pgen_file = f"{plink_prefix}.pgen"

        # variant metadata
        self.num_variants = self.pvar_df.shape[0]
        self.variant_ids = self.pvar_df['id'].tolist()
        self.variant_idx = {vid: i for i, vid in enumerate(self.variant_ids)}

        self.sample_ids = self.psam_df.index.tolist()
        self.sample_idxs = None
        if select_samples is not None:
            self.set_samples(select_samples)

        # ``alt`` is the counted (dosage) allele: pgenlib reports ALT-allele counts.
        variant_df = self.pvar_df.set_index('id')[['chrom', 'pos', 'ref', 'alt']].copy()
        variant_df['index'] = np.arange(len(variant_df))
        self.variant_df = variant_df
        self.variant_dfs = {c:g[['pos', 'index']]
                            for c,g in variant_df.groupby('chrom', sort=False)}

        self.impute = impute
        self.dtype = dtype
        self.device = self._get_device(device)

    def _get_device(self, device):
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device

    def set_samples(self, sample_ids=None):
        """Restrict to a subset of samples (order preserved)."""
        sample_idxs = [self.sample_ids.index(i) for i in sample_ids]
        self.sample_ids = list(sample_ids)
        self.sample_idxs = np.array(sample_idxs, dtype=np.uint32)

    def get_range(self, chrom, start=None, end=None):
        """
        Get variant indexes corresponding to region specified as 'chr:start-end', or as chr, start, end.
        Return [start,end] indexes for variants in a region.
        """
        vpos = self.variant_dfs[chrom]["pos"].values
        lb = bisect.bisect_left(vpos, start) if start else 0
        ub = bisect.bisect_right(vpos, end) if end else vpos.shape[0]
        if lb < ub:
            return self.variant_dfs[chrom]['index'].values[[lb, ub - 1]]
        return []

    def read_list(self, variant_ids, dosages=False, to_torch=False):
        """Read list of variants as numpy or torch."""
        vix = [self.variant_idx[v] for v in variant_ids]
        nvar, nsamp = len(vix), len(self.sample_ids)
        arr = np.zeros((nvar, nsamp),
                       dtype=np.float32 if dosages else np.int8)

        with pg.PgenReader(self.pgen_file.encode(),
                           sample_subset=self.sample_idxs) as r:
            if dosages:
                r.read_dosages_list(np.array(vix, dtype=np.uint32), arr)
            else:
                r.read_list(np.array(vix, dtype=np.uint32), arr)

        if not dosages and self.impute:
            arr = _impute_mean(arr)

        arr = arr.astype(self.dtype, copy=False)

        if to_torch or self.device == "cuda":
            return torch.as_tensor(arr, device=self.device)
        return arr

    def read_range(
        self,
        start_idx,
        end_idx,
        impute_mean=True,
        dtype=None,
        dosages=False,
        to_torch=False,
    ):
        """Read contiguous block of variants by index."""
        nvar = end_idx - start_idx + 1
        nsamp = len(self.sample_ids)
        arr = np.zeros((nvar, nsamp),
                       dtype=np.float32 if dosages else np.int8)

        with pg.PgenReader(self.pgen_file.encode(),
                           sample_subset=self.sample_idxs) as r:
            if dosages:
                r.read_dosages_range(start_idx, end_idx + 1, arr)
            else:
                r.read_range(start_idx, end_idx + 1, arr)

        if not dosages and self.impute and impute_mean:
            arr = _impute_mean(arr)

        target_dtype = dtype if dtype is not None else self.dtype
        arr = arr.astype(target_dtype, copy=False)

        if to_torch or self.device == "cuda":
            return torch.as_tensor(arr, device=self.device)
        return arr

    def load_genotypes(self):
        """Load all genotypes (hardcalls) as DataFrame [variants x samples]."""
        arr = self.read_range(0, self.num_variants - 1, dosages=False)
        return pd.DataFrame(arr, index=self.variant_ids, columns=self.sample_ids)

    def load_dosages(self):
        """Load all dosages as DataFrame [variants x samples]."""
        arr = self.read_range(0, self.num_variants - 1, dosages=True)
        return pd.DataFrame(arr, index=self.variant_ids, columns=self.sample_ids)

    def read(self, variant_id, dosages=False, to_torch=False):
        """Read a single variant by ID (genotypes or dosages)."""
        arr = self.read_list([variant_id], dosages=dosages, to_torch=to_torch)
        if isinstance(arr, torch.Tensor):
            return arr.squeeze(0)  # (samples,)
        return arr[0]

    def read_alleles(self, variant_id, to_torch=False):
        """Read alleles for a single variant (2 per sample)."""
        vidx = self.variant_idx[variant_id]
        nsamp = len(self.sample_ids)
        alleles = np.zeros(2 * nsamp, dtype=np.int32)

        with pg.PgenReader(self.pgen_file.encode(),
                           sample_subset=self.sample_idxs) as r:
            r.read_alleles(np.array(vidx, dtype=np.uint32), alleles)

        alleles = alleles.reshape(nsamp, 2)

        if to_torch or self.device == "cuda":
            return torch.as_tensor(alleles, device=self.device)
        return pd.DataFrame(alleles, index=self.sample_ids,
                            columns=["allele1", "allele2"])


def get_chunk_size(memory_gb, samples):
    """Number of variants (int8 genotypes) that fit in ``memory_gb`` for ``samples`` samples."""
    return int(memory_gb * 1024 ** 3 // int(samples))


def generate_paired_chunks(pgr, phenotype_df, phenotype_pos_df, chunk_size, window=1_000_000,
                           dosages=False, verbose=True):
    """
    Generate paired genotype-phenotype chunks for data sets whose genotypes do
    not fit in memory (tensorQTL ``generate_paired_chunks``).

    ``chunk_size`` is the maximum number of variants loaded at once (at least
    the largest cis-window), or ``'chr'`` for one chunk per chromosome.
    Phenotypes must be sorted by chromosome and position.  Yields
    ``(genotype_df, variant_df, phenotype_df_chunk, phenotype_pos_df_chunk, chunk_index)``;
    ``variant_df`` carries ``chrom, pos, ref, alt``.
    """
    from .genotypeio import get_cis_ranges

    variant_df = pgr.variant_df[['chrom', 'pos', 'ref', 'alt']]
    keep = phenotype_pos_df['chr'].astype(str).isin([str(c) for c in pgr.variant_dfs])
    if (~keep).any():
        print(f"    ** dropping {int((~keep).sum())} phenotypes on chrs. without genotypes")
    phenotype_pos_df = phenotype_pos_df.loc[keep]
    phenotype_df = phenotype_df.loc[phenotype_pos_df.index]
    cis_ranges, drop_ids = get_cis_ranges(phenotype_pos_df, pgr.variant_dfs, window, verbose=verbose)
    if drop_ids:
        print(f"    ** dropping {len(drop_ids)} phenotypes without variants in cis-window")
        phenotype_pos_df = phenotype_pos_df.drop(drop_ids)
        phenotype_df = phenotype_df.drop(drop_ids)
    range_df = pd.DataFrame(cis_ranges, index=['start', 'end']).T.loc[phenotype_pos_df.index]
    range_df = range_df.join(phenotype_pos_df['chr'])
    if not (np.diff(range_df['start'].to_numpy()) >= 0).all():
        raise ValueError("Phenotypes must be sorted by chromosome and position for chunked processing.")

    if chunk_size == 'chr':
        chrlen_s = range_df['chr'].value_counts(sort=False).loc[pd.unique(range_df['chr'])]
        start_ixs = [0] + chrlen_s.cumsum().tolist()
    else:
        chunk_size = int(chunk_size)
        max_cis_var = int((range_df['end'] - range_df['start'] + 1).max())
        if max_cis_var > chunk_size:
            raise ValueError(f"Max. chunk size must be at least largest cis-window ({max_cis_var})")
        start_ixs = [0]
        while start_ixs[-1] < range_df.shape[0]:
            end_ix = bisect.bisect_left(range_df['end'].values, range_df['start'].values[start_ixs[-1]] + chunk_size)
            if end_ix <= start_ixs[-1]:
                end_ix = start_ixs[-1] + 1
            start_ixs.append(end_ix)
        start_ixs[-1] = range_df.shape[0]

    nchunks = len(start_ixs) - 1
    for ci in range(nchunks):
        if verbose:
            print(f"Processing genotype-phenotype chunk {ci + 1}/{nchunks}")
        ix = slice(start_ixs[ci], start_ixs[ci + 1])
        chunk_df = range_df.iloc[ix]
        v0, v1 = int(chunk_df['start'].values[0]), int(chunk_df['end'].values[-1])
        arr = pgr.read_range(v0, v1, impute_mean=False, dosages=dosages,
                             dtype=np.float32 if dosages else np.int8)
        if torch.is_tensor(arr):
            arr = arr.detach().cpu().numpy()
        var_df = variant_df.iloc[v0:v1 + 1]
        gt_df = pd.DataFrame(arr, index=var_df.index, columns=pgr.sample_ids)
        yield gt_df, var_df, phenotype_df.iloc[ix], phenotype_pos_df.iloc[ix], ci
