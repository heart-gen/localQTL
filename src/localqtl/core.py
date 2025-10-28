## Adapted from tensorqtl: https://github.com/broadinstitute/tensorqtl/blob/master/tensorqtl/core.py
import sys
import torch
import numpy as np

class SimpleLogger(object):
    def __init__(self, logfile=None, verbose=True):
        self.console = sys.stdout
        self.verbose = verbose
        if logfile is not None:
            self.log = open(logfile, 'w')
        else:
            self.log = None

    def write(self, message):
        if self.verbose:
            self.console.write(message+'\n')
        if self.log is not None:
            self.log.write(message+'\n')
            self.log.flush()

#------------------------------------------------------------------------------
#  Core classes/functions for mapping associations on GPU
#------------------------------------------------------------------------------
def calculate_maf(genotype_t, alleles=2):
    """Calculate minor allele frequency"""
    af_t = genotype_t.sum(1) / (alleles * genotype_t.shape[1])
    return torch.where(af_t > 0.5, 1 - af_t, af_t)


def get_allele_stats(genotype_t):
    """Returns allele frequency, minor allele samples, and minor allele counts (row-wise)."""
    # allele frequency
    n2 = 2 * genotype_t.shape[1]
    af_t = genotype_t.sum(1) / n2
    # minor allele samples and counts
    ix_t = af_t <= 0.5
    m = genotype_t > 0.5
    a = m.sum(1).int()
    b = (genotype_t < 1.5).sum(1).int()
    ma_samples_t = torch.where(ix_t, a, b)
    a = (genotype_t * m.float()).sum(1).int()
    # a = (genotype_t * m.float()).sum(1).round().int()  # round for missing/imputed genotypes
    ma_count_t = torch.where(ix_t, a, n2-a)
    return af_t, ma_samples_t, ma_count_t


def filter_maf(genotypes_t, variant_ids, maf_threshold, alleles=2):
    """Calculate MAF and filter genotypes that don't pass threshold"""
    af_t = genotypes_t.sum(1) / (alleles * genotypes_t.shape[1])
    maf_t = torch.where(af_t > 0.5, 1 - af_t, af_t)
    if maf_threshold > 0:
        mask_t = maf_t >= maf_threshold
        genotypes_t = genotypes_t[mask_t]
        variant_ids = variant_ids[mask_t.cpu().numpy().astype(bool)]
        af_t = af_t[mask_t]
    return genotypes_t, variant_ids, af_t


def impute_mean(genotypes_t, missing=-9):
    """Impute missing genotypes to mean"""
    m = genotypes_t == missing
    ix = torch.nonzero(m, as_tuple=True)[0]
    if len(ix) > 0:
        a = genotypes_t.sum(1)
        b = m.sum(1).float()
        mu = (a - missing*b) / (genotypes_t.shape[1] - b)
        genotypes_t[m] = mu[ix]

    return N_t / torch.sqrt(torch.pow(N_t, 2).sum(dim=dim, keepdim=True))
