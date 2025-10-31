"""
This script was adapted from tensorQTL `genotypeio.py`:
https://github.com/broadinstitute/tensorqtl/blob/master/tensorqtl/genotypeio.py
"""
import numpy as np
import pandas as pd
import os, gzip, sys
import threading, queue, bisect
from pandas_plink import read_plink

sys.path.insert(1, os.path.dirname(__file__))
from .phenotypeio import read_phenotype_bed

try:
    import pgen
except ImportError as e:
    pgen = None

__all__ = [
    "BackgroundGenerator",
    "background",
    "print_progress",
    "PlinkReader",
    "load_genotypes",
    "get_cis_ranges",
    "InputGeneratorCis",
]

class BackgroundGenerator(threading.Thread):
    def __init__(self, generator, max_prefetch=10):
        threading.Thread.__init__(self)
        self.queue = queue.Queue(max_prefetch)
        self.generator = generator
        self.daemon = True
        self.start()

    def run(self):
        try:
            for item in self.generator:
                self.queue.put(item)
        except Exception as exception:
            self.queue.put(exception)
        self.queue.put(None)

    def next(self):
        next_item = self.queue.get()
        if next_item is None:
            self.join()
            raise StopIteration
        if isinstance(next_item, Exception):
            self.join()
            raise next_item
        return next_item

    def __next__(self):
        return self.next()

    def __iter__(self):
        return self

class background:
    def __init__(self, max_prefetch=10):
        self.max_prefetch = max_prefetch
    def __call__(self, gen):
        def bg_generator(*args, **kwargs):
            return BackgroundGenerator(gen(*args, **kwargs), max_prefetch=self.max_prefetch)
        return bg_generator


def print_progress(k, n, entity):
    s = f'\r    processing {entity} {k}/{n}'
    if k == n:
        s += '\n'
    sys.stdout.write(s)
    sys.stdout.flush()


def _impute_mean(g, missing=-9, verbose=False):
    """Impute rows to mean (in place)"""
    if not g.dtype in [np.float32, np.float64]:
        raise ValueError('Input dtype must be np.float32 or np.float64')
    n = 0
    for i in np.where((g == missing).any(1))[0]:
        ix = g[i] == missing
        g[i][ix] = np.mean(g[i][~ix])
        n += 1
    if verbose and n > 0:
        print(f'    imputed at least 1 sample in {n}/{g.shape[0]} sites')


class PlinkReader(object):
    def __init__(self, plink_prefix_path, select_samples=None, include_variants=None,
                 exclude_variants=None, exclude_chrs=None, verbose=True, dtype=np.int8):
        """
        Class for reading genotypes from PLINK bed files

        plink_prefix_path: prefix to PLINK bed,bim,fam files
        select_samples: specify a subset of samples

        Notes:
          Use this command to convert a VCF to PLINK format:
            plink2 --make-bed \
                --output-chr chrM \
                --vcf ${plink_prefix_path}.vcf.gz \
                --out ${plink_prefix_path}

            If using plink v1, the --keep-allele-order flag must be included.

          Uses read_plink from pandas_plink.
        """

        self.bim, self.fam, self.bed = read_plink(plink_prefix_path, verbose=verbose)
        self.bed = 2 - self.bed  # flip allele order: PLINK uses REF as effect allele
        if dtype == np.int8:
            self.bed[np.isnan(self.bed)] = -9  # convert missing (NaN) to -9 for int8
        self.bed = self.bed.astype(dtype, copy=False)
        self.sample_ids = self.fam['iid'].tolist()
        if select_samples is not None:
            ix = [self.sample_ids.index(i) for i in select_samples]
            self.fam = self.fam.loc[ix]
            self.bed = self.bed[:,ix]
            self.sample_ids = self.fam['iid'].tolist()
        if include_variants is not None:
            m = self.bim['snp'].isin(include_variants).values
            self.bed = self.bed[m,:]
            self.bim = self.bim[m]
            self.bim.reset_index(drop=True, inplace=True)
            self.bim['i'] = self.bim.index
        if exclude_variants is not None:
            m = ~self.bim['snp'].isin(exclude_variants).values
            self.bed = self.bed[m,:]
            self.bim = self.bim[m]
            self.bim.reset_index(drop=True, inplace=True)
            self.bim['i'] = self.bim.index
        if exclude_chrs is not None:
            m = ~self.bim['chrom'].isin(exclude_chrs).values
            self.bed = self.bed[m,:]
            self.bim = self.bim[m]
            self.bim.reset_index(drop=True, inplace=True)
            self.bim['i'] = self.bim.index
        self.n_samples = self.fam.shape[0]
        self.chrs = list(self.bim['chrom'].unique())
        self.variant_pos = {i:g['pos'] for i,g in self.bim.set_index('snp')[['chrom', 'pos']].groupby('chrom')}
        self.variant_pos_dict = self.bim.set_index('snp')['pos'].to_dict()

    def get_region_index(self, region_str, return_pos=False):
        s = region_str.split(':')
        chrom = s[0]
        c = self.bim[self.bim['chrom'] == chrom]
        if len(s) > 1:
            start, end = s[1].split('-')
            start = int(start)
            end = int(end)
            c = c[(c['pos'] >= start) & (c['pos'] <= end)]
        if return_pos:
            return c['i'].values, c.set_index('snp')['pos']
        else:
            return c['i'].values

    def get_region(self, region_str, sample_ids=None, impute=False, verbose=False, dtype=np.int8):
        """Get genotypes for a region defined by 'chr:start-end' or 'chr'"""
        ix, pos_s = self.get_region_index(region_str, return_pos=True)
        g = self.bed[ix, :].compute().astype(dtype)
        if sample_ids is not None:
            ix = [self.sample_ids.index(i) for i in sample_ids]
            g = g[:, ix]
        if impute:
            _impute_mean(g, verbose=verbose)
        return g, pos_s

    def get_genotypes(self, variant_ids, sample_ids=None, impute=False, verbose=False, dtype=np.int8):
        """Load genotypes for selected variant IDs"""
        c = self.bim[self.bim['snp'].isin(variant_ids)]
        g = self.bed[c.i.values, :].compute().astype(dtype)
        if sample_ids is not None:
            ix = [self.sample_ids.index(i) for i in sample_ids]
            g = g[:, ix]
        if impute:
            _impute_mean(g, verbose=verbose)
        return g, c.set_index('snp')['pos']

    def get_genotype(self, variant_id, sample_ids=None, impute=False, verbose=False, dtype=np.int8):
        """Load genotypes for a single variant ID as pd.Series"""
        g,_ = self.get_genotypes([variant_id], sample_ids=sample_ids, impute=impute, verbose=verbose, dtype=dtype)
        if sample_ids is None:
            return pd.Series(g[0], index=self.fam['iid'], name=variant_id)
        else:
            return pd.Series(g[0], index=sample_ids, name=variant_id)

    def load_genotypes(self):
        """Load all genotypes into memory, as pd.DataFrame"""
        return pd.DataFrame(self.bed.compute(), index=self.bim['snp'], columns=self.fam['iid'])


def load_genotypes(genotype_path, select_samples=None, dosages=False):
    """Load all genotypes into a dataframe"""
    if all([os.path.exists(f"{genotype_path}.{ext}") for ext in ['pgen', 'psam', 'pvar']]):
        if pgen is None:
            raise ImportError('Pgenlib must be installed to use PLINK 2 pgen/psam/pvar files.')
        pgr = pgen.PgenReader(genotype_path, select_samples=select_samples)
        variant_df = pgr.pvar_df.set_index('id')[['chrom', 'pos']]
        if dosages:
            genotype_df = pgr.load_dosages()
        else:
            genotype_df = pgr.load_genotypes()
    elif all([os.path.exists(f"{genotype_path}.{ext}") for ext in ['bed', 'bim', 'fam']]):
        pr = PlinkReader(genotype_path, select_samples=select_samples, dtype=np.int8)
        genotype_df = pr.load_genotypes()
        variant_df = pr.bim.set_index('snp')[['chrom', 'pos']]
    elif genotype_path.endswith(('.bed.parquet', '.bed.gz', '.bed')):
        genotype_df, variant_df = read_phenotype_bed(genotype_path)
        assert variant_df.columns[1] == 'pos', "The BED file must define a single position for each variant, with start + 1 == end."
        variant_df.columns = ['chrom', 'pos']
    elif genotype_path.endswith('.parquet'):
        genotype_df = pd.read_parquet(genotype_path)
        variant_df = None
    elif genotype_path.endswith('.gz'):
        with gzip.open(genotype_path, 'rt') as f:
            header = f.readline().strip().split('\t')
        dtypes = {i:np.float32 for i in header}
        dtypes[header[0]] = str
        genotype_df = pd.read_csv(genotype_path, sep='\t', index_col=0, dtype=dtypes)
        variant_df = None
    else:
        raise ValueError(f"Failed to load genotypes from {genotype_path}. Supported formats: pgen/psam/pvar, bed/bim/fam, parquet, tsv.gz")
    return genotype_df, variant_df


def get_cis_ranges(phenotype_pos_df, chr_variant_dfs, window, verbose=True):
    """

    start, end indexes (inclusive)
    """
    # check phenotypes & calculate genotype ranges
    # get genotype indexes corresponding to cis-window of each phenotype
    if 'pos' in phenotype_pos_df:
        phenotype_pos_df = phenotype_pos_df.rename(columns={'pos':'start'})
        phenotype_pos_df['end'] = phenotype_pos_df['start']
    phenotype_pos_dict = phenotype_pos_df.to_dict(orient='index')

    drop_ids = []
    cis_ranges = {}
    n = len(phenotype_pos_df)
    for k, phenotype_id in enumerate(phenotype_pos_df.index, 1):
        if verbose and (k % 1000 == 0 or k == n):
            print(f'\r  * checking phenotypes: {k}/{n}',  end='' if k != n else None)

        pos = phenotype_pos_dict[phenotype_id]
        chrom = pos['chr']
        m = len(chr_variant_dfs[chrom]['pos'].values)
        lb = bisect.bisect_left(chr_variant_dfs[chrom]['pos'].values, pos['start'] - window)
        ub = bisect.bisect_right(chr_variant_dfs[chrom]['pos'].values, pos['end'] + window)
        if lb != ub:
            r = chr_variant_dfs[chrom]['index'].values[[lb, ub - 1]]
        else:
            r = []

        if len(r) > 0:
            cis_ranges[phenotype_id] = r
        else:
            drop_ids.append(phenotype_id)

    return cis_ranges, drop_ids


class InputGeneratorCis(object):
    """
    Base input generator for cis-mapping.
    Subclasses can extend by overriding `_postprocess_batch` to add haplotypes.

    Inputs:
      genotype_df:      genotype DataFrame (genotypes x samples)
      variant_df:       DataFrame mapping variant_id (index) to chrom, pos
      phenotype_df:     phenotype DataFrame (phenotypes x samples)
      phenotype_pos_df: DataFrame defining position of each phenotype, with columns ['chr', 'pos'] or ['chr', 'start', 'end']
      window:           cis-window; selects variants within +- cis-window from 'pos' (e.g., TSS for gene-based features)
                        or within [start-window, end+window] if 'start' and 'end' are present in phenotype_pos_df

    Generates: phenotype array, genotype array (2D), cis-window indices, phenotype ID
    """
    def __init__(self, genotype_df, variant_df, phenotype_df, phenotype_pos_df,
                 group_s=None, window=1_000_000, **kwargs):
        self.genotype_df = genotype_df
        self.variant_df = variant_df.copy()
        self.phenotype_df = phenotype_df
        self.phenotype_pos_df = phenotype_pos_df
        self.group_s = group_s
        self.window = window

        # core preprocessing
        self._validate_inputs()
        self._filter_phenotypes()
        self._drop_constant_phenotypes()
        self._calculate_cis_ranges()

    # Protected methods (overridable)
    def _validate_inputs(self):
        assert (self.genotype_df.index == self.variant_df.index).all(), \
            "Genotype and variant DataFrames must share index order"
        assert (self.phenotype_df.index == self.phenotype_df.index.unique()).all(), \
            "Phenotype index must be unique"
        self.variant_df['index'] = np.arange(self.variant_df.shape[0])
        self.n_samples = self.phenotype_df.shape[1]
        
    def _filter_phenotypes(self):
        variant_chrs = self.variant_df['chrom'].unique()
        phenotype_chrs = self.phenotype_pos_df['chr'].unique()
        self.chrs = [c for c in phenotype_chrs if c in variant_chrs]
        mask = self.phenotype_pos_df['chr'].isin(self.chrs)
        if any(~mask):
            print(f'    ** dropping {sum(~mask)} phenotypes on chrs. without genotypes')
        self.phenotype_df = self.phenotype_df.loc[mask]
        self.phenotype_pos_df = self.phenotype_pos_df.loc[mask]

    def _drop_constant_phenotypes(self):
        m = np.all(self.phenotype_df.values == self.phenotype_df.values[:, [0]], axis=1)
        if m.any():
            print(f'    ** dropping {m.sum()} constant phenotypes')
            self.phenotype_df = self.phenotype_df.loc[~m]
            self.phenotype_pos_df = self.phenotype_pos_df.loc[~m]
        if len(self.phenotype_df) == 0:
            raise ValueError("No phenotypes remain after filters.")
    
    def _calculate_cis_ranges(self):
        self.chr_variant_dfs = {c: g[['pos', 'index']]
                                for c, g in self.variant_df.groupby('chrom')}
        self.cis_ranges, drop_ids = get_cis_ranges(
            self.phenotype_pos_df, self.chr_variant_dfs, self.window
        )
        if drop_ids:
            print(f"    ** dropping {len(drop_ids)} phenotypes without variants in cis-window")
            self.phenotype_df = self.phenotype_df.drop(drop_ids)
            self.phenotype_pos_df = self.phenotype_pos_df.drop(drop_ids)
        if 'pos' in self.phenotype_pos_df:
            self.phenotype_start = self.phenotype_pos_df['pos'].to_dict()
            self.phenotype_end = self.phenotype_start
        else:
            self.phenotype_start = self.phenotype_pos_df['start'].to_dict()
            self.phenotype_end = self.phenotype_pos_df['end'].to_dict()
        self.n_phenotypes = self.phenotype_df.shape[0]
        if self.group_s is not None:
            self.group_s = self.group_s.loc[self.phenotype_df.index].copy()
            self.n_groups = self.group_s.unique().shape[0]

    # Hook for subclasses
    def _postprocess_batch(self, batch):
        return batch

    # Data generator
    @background(max_prefetch=6)
    def generate_data(self, chrom=None, verbose=False):
        """
        Generate batches from genotype data.
        """
        if chrom is None:
            phenotype_ids = self.phenotype_df.index
            chr_offset = 0
        else:
            phenotype_ids = self.phenotype_pos_df[self.phenotype_pos_df['chr'] == chrom].index
            if self.group_s is None:
                offset_dict = {c: i for c, i in zip(*np.unique(self.phenotype_pos_df['chr'],
                                                               return_index=True))}
            else:
                offset_dict = {c: i for c, i in zip(
                    *np.unique(self.phenotype_pos_df['chr'][self.group_s.drop_duplicates().index],
                               return_index=True))}
            chr_offset = offset_dict[chrom]

        index_dict = {pid: i for i, pid in enumerate(self.phenotype_df.index)}

        if self.group_s is None:
            for k, pid in enumerate(phenotype_ids, chr_offset+1):
                if verbose:
                    print_progress(k, self.n_phenotypes, 'phenotype')
                p = self.phenotype_df.values[index_dict[pid]]
                r = self.cis_ranges[pid]
                batch = (
                    p,
                    self.genotype_df.values[r[0]:r[-1]+1],
                    np.arange(r[0], r[-1]+1),
                    pid
                )
                yield self._postprocess_batch(batch)
        else:
            gdf = self.group_s[phenotype_ids].groupby(self.group_s, sort=False)
            for k, (group_id, g) in enumerate(gdf, chr_offset+1):
                if verbose:
                    print_progress(k, self.n_groups, 'phenotype group')
                assert np.all([self.cis_ranges[g.index[0]][0] == self.cis_ranges[i][0] and
                               self.cis_ranges[g.index[0]][1] == self.cis_ranges[i][1]
                               for i in g.index[1:]])
                group_pids = g.index.tolist()
                p = self.phenotype_df.values[[index_dict[i] for i in group_pids]]
                r = self.cis_ranges[g.index[0]]
                batch = (
                    p,
                    self.genotype_df.values[r[0]:r[-1]+1],
                    np.arange(r[0],r[-1]+1),
                    group_pids,
                    group_id
                )
                yield self._postprocess_batch(batch)
