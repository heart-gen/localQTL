# Functions for reading dosages from PLINK pgen files based on the Pgenlib Python API:
# https://github.com/chrchang/plink-ng/blob/master/2.0/Python/python_api.txt

import os
import torch
import bisect
import numpy as np
import pandas as pd
import pgenlib as pg


def read_pvar(pvar_path):
    """Read pvar file as pd.DataFrame"""
    return pd.read_csv(pvar_path, sep='\t', comment='#',
                       names=['chrom', 'pos', 'id', 'ref', 'alt', 'qual', 'filter', 'info'],
                       dtype={'chrom':str, 'pos':np.int32, 'id':str, 'ref':str, 'alt':str,
                              'qual':str, 'filter':str, 'info':str})


def read_psam(psam_path):
    """Read psam file as pd.DataFrame"""
    psam_df = pd.read_csv(psam_path, sep='\t', index_col=0)
    psam_df.index = psam_df.index.astype(str)
    return psam_df


def _impute_mean(genotypes, missing_code=-9):
    """Impute missing genotypes (-9) with per-variant mean"""
    m = genotypes == missing_code
    if genotypes.ndim == 1:
        if m.any():
            genotypes[m] = genotypes[~m].mean()
    else:  # genotypes.ndim == 2
        row_means = np.where(m.any(1), genotypes.clip(min=0).mean(1), 0)
        genotypes[m] = np.take(row_means, np.nonzero(m)[0])
    return genotypes


class PgenReader(object):
    """
    Class for reading genotype data from PLINK 2 pgen files

    To generate the pgen/psam/pvar files from a VCF, run
        plink2 --vcf ${vcf_file} --output-chr chrM --out ${plink_prefix}
    To use dosages, run:
        plink2 --vcf ${vcf_file} 'dosage=DS' --output-chr chrM --out ${plink_prefix}

    Requires pgenlib: https://github.com/chrchang/plink-ng/tree/master/2.0/Python
    """
    def __init__(self, plink_prefix, select_samples=None, impute=True,
                 dtype=np.float32, device="cpu"):
        """
        plink_prefix: prefix to PLINK pgen,psam,pvar files
        select_samples: specify a subset of samples
        """
        self.pvar_df = (
            pd.read_parquet(f"{plink_prefix}.pvar.parquet")
            if os.path.exists(f"{plink_prefix}.pvar.parquet")
            else read_pvar(f"{plink_prefix}.pvar")
        )
        self.psam_df = read_psam(f"{plink_prefix}.psam")
        self.pgen_file = f"{plink_prefix}.pgen"

        self.num_variants = self.pvar_df.shape[0]
        self.variant_ids = self.pvar_df['id'].tolist()
        self.variant_idx_dict = {i:k for k,i in enumerate(self.variant_ids)}

        self.sample_ids = self.psam_df.index.tolist()
        self.sample_idxs = None
        if select_samples is not None:
            self.set_samples(select_samples)

        self.variant_df = self.pvar_df.set_index('id')[['chrom', 'pos']].copy()
        self.variant_df['index'] = np.arange(len(self.variant_df))
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
        self.sample_ids = sample_ids
        self.sample_idxs = sample_idxs

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

    def read_range(self, start_idx, end_idx, impute_mean=True, dtype=np.float32):
        """Read genotypes for range of variants as 0,1,2,-9; impute missing values (-9) to mean (default)."""
        genotypes = read_range(self.pgen_file, start_idx, end_idx, sample_subset=self.sample_idxs,
                               dtype=np.int8).astype(dtype)
        if impute_mean:
            _impute_mean(genotypes)
        return pd.DataFrame(genotypes, index=self.variant_ids[start_idx:end_idx+1], columns=self.sample_ids)

    def read(self, variant_id, impute_mean=True, dtype=np.float32):
        """Read genotypes for an individual variant as 0,1,2,-9; impute missing values (-9) to mean (default)."""
        variant_idx = self.variant_idx_dict[variant_id]
        genotypes = read(self.pgen_file, variant_idx, sample_subset=self.sample_idxs,
                         dtype=np.int8).astype(dtype)
        if impute_mean:
            _impute_mean(genotypes)
        return pd.Series(genotypes, index=self.sample_ids, name=variant_id)

    def read_region(self, region, start_pos=None, end_pos=None, impute_mean=True, dtype=np.float32):
        """Read genotypes for variants in a genomic region as 0,1,2,-9; impute missing values (-9) to mean (default)."""
        r = self.get_range(region, start_pos, end_pos)
        if len(r) > 0:
            return self.read_range(*r, impute_mean=impute_mean, dtype=dtype)

    def read_dosages(self, variant_id, dtype=np.float32):
        variant_idx = self.variant_idx_dict[variant_id]
        dosages = read_dosages(self.pgen_file, variant_idx, sample_subset=self.sample_idxs, dtype=dtype)
        return pd.Series(dosages, index=self.sample_ids, name=variant_id)

    def read_dosages_list(self, variant_ids, dtype=np.float32):
        variant_idxs = [self.variant_idx_dict[i] for i in variant_ids]
        dosages = read_dosages_list(self.pgen_file, variant_idxs, sample_subset=self.sample_idxs, dtype=dtype)
        return pd.DataFrame(dosages, index=variant_ids, columns=self.sample_ids)

    def read_dosages_range(self, start_idx, end_idx, dtype=np.float32):
        dosages = read_dosages_range(self.pgen_file, start_idx, end_idx, sample_subset=self.sample_idxs, dtype=dtype)
        return pd.DataFrame(dosages, index=self.variant_ids[start_idx:end_idx+1], columns=self.sample_ids)

    def read_dosages_region(self, region, start_pos=None, end_pos=None, dtype=np.float32):
        r = self.get_range(region, start_pos, end_pos)
        if len(r) > 0:
            return self.read_dosages_range(*r, dtype=dtype)

    def read_alleles(self, variant_id):
        variant_idx = self.variant_idx_dict[variant_id]
        alleles = read_alleles(self.pgen_file, variant_idx, sample_subset=self.sample_idxs)
        s1 = pd.Series(alleles[::2],  index=self.sample_ids, name=variant_id)
        s2 = pd.Series(alleles[1::2], index=self.sample_ids, name=variant_id)
        return s1, s2

    def read_alleles_list(self, variant_ids):
        variant_idxs = [self.variant_idx_dict[i] for i in variant_ids]
        alleles = read_alleles_list(self.pgen_file, variant_idxs, sample_subset=self.sample_idxs)
        df1 = pd.DataFrame(alleles[:,::2],  index=variant_ids, columns=self.sample_ids)
        df2 = pd.DataFrame(alleles[:,1::2], index=variant_ids, columns=self.sample_ids)
        return df1, df2

    def read_alleles_range(self, start_idx, end_idx):
        alleles = read_alleles_range(self.pgen_file, start_idx, end_idx, sample_subset=self.sample_idxs)
        df1 = pd.DataFrame(alleles[:,::2],  index=self.variant_ids[start_idx:end_idx+1], columns=self.sample_ids)
        df2 = pd.DataFrame(alleles[:,1::2], index=self.variant_ids[start_idx:end_idx+1], columns=self.sample_ids)
        return df1, df2

    def read_alleles_region(self, region, start_pos=None, end_pos=None):
        r = self.get_range(region, start_pos, end_pos)
        if len(r) > 0:
            return self.read_alleles_range(*r)
        else:
            return None, None

    def load_genotypes(self):
        """Load all genotypes as np.int8, without imputing missing values."""
        genotypes = read_range(self.pgen_file, 0, self.num_variants-1, sample_subset=self.sample_idxs)
        return pd.DataFrame(genotypes, index=self.variant_ids, columns=self.sample_ids)

    def load_dosages(self):
        """Load all dosages."""
        return self.read_dosages_range(0, self.num_variants-1)

    def load_alleles(self):
        """Load all alleles."""
        return self.read_alleles_range(0, self.num_variants-1)

    def get_pairwise_ld(self, id1, id2, r2=True, dtype=np.float32):
        """Compute pairwise LD (R2) between (lists of) variants"""
        if isinstance(id1, str) and isinstance(id2, str):
            g1 = self.read(id1, dtype=dtype)
            g2 = self.read(id2, dtype=dtype)
            g1 -= g1.mean()
            g2 -= g2.mean()
            if r2:
                r = (g1 * g2).sum()**2 / ( (g1**2).sum() * (g2**2).sum() )
            else:
                r = (g1 * g2).sum() / np.sqrt( (g1**2).sum() * (g2**2).sum() )
        elif isinstance(id1, str):
            g1 = self.read(id1, dtype=dtype)
            g2 = self.read_list(id2, dtype=dtype)
            g1 -= g1.mean()
            g2 -= g2.values.mean(1, keepdims=True)
            if r2:
                r = (g1 * g2).sum(1)**2 / ( (g1**2).sum() * (g2**2).sum(1) )
            else:
                r = (g1 * g2).sum(1) / np.sqrt( (g1**2).sum() * (g2**2).sum(1) )
        elif isinstance(id2, str):
            g1 = self.read_list(id1, dtype=dtype)
            g2 = self.read(id2, dtype=dtype)
            g1 -= g1.values.mean(1, keepdims=True)
            g2 -= g2.mean()
            if r2:
                r = (g1 * g2).sum(1)**2 / ( (g1**2).sum(1) * (g2**2).sum() )
            else:
                r = (g1 * g2).sum(1) / np.sqrt( (g1**2).sum(1) * (g2**2).sum() )
        else:
            assert len(id1) == len(id2)
            g1 = self.read_list(id1, dtype=dtype).values
            g2 = self.read_list(id2, dtype=dtype).values
            g1 -= g1.mean(1, keepdims=True)
            g2 -= g2.mean(1, keepdims=True)
            if r2:
                r = (g1 * g2).sum(1) ** 2 / ( (g1**2).sum(1) * (g2**2).sum(1) )
            else:
                r = (g1 * g2).sum(1) / np.sqrt( (g1**2).sum(1) * (g2**2).sum(1) )
        return r

    def get_ld_matrix(self, variant_ids, dtype=np.float32):
        g = self.read_list(variant_ids, dtype=dtype).values
        return pd.DataFrame(np.corrcoef(g), index=variant_ids, columns=variant_ids)


def load_dosages_df(plink_prefix, select_samples=None):
    """
    Load dosages for all variants and all/selected samples as a dataframe.

    Parameters
    ----------
    plink_prefix : str
        Prefix to .pgen/.psam/.pvar files
    select_samples : array_like
        List of sample IDs to select. Default: all samples.

    Returns
    -------
    dosages_df : pd.DataFrame (variants x samples)
        Genotype dosages for the selected samples.
    """
    p = Pgen(plink_prefix, select_samples=select_samples)
    return p.load_dosages_df()
