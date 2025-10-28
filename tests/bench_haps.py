#!/usr/bin/env python3
"""
Benchmark & correctness test for cis mapping WITH HAPLOTYPES:
- functional core (_run_nominal_core) if available
- SimpleCisMapper.map_nominal (using InputGeneratorCisWithHaps)

Scales over (#variants in cis-window, #phenotypes, #ancestries) and reports timings.
Also checks functional vs OO numerical agreement when the functional core is available.

Usage:
  python tests/bench_haps.py \
      --variants 2000,8000 \
      --phenotypes 50,200 \
      --ancestries 2,3 \
      --samples 256 \
      --covars 6 \
      --device auto \
      --csv bench_haps_results.csv
"""

import argparse
import inspect
import os
import sys
import time
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch

# --- repo import path ---
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from localqtl.regression_kernels import Residualizer
from localqtl.cis import SimpleCisMapper  # functional path probed dynamically below
from localqtl.haplotypeio import InputGeneratorCisWithHaps
from localqtl.genotypeio import InputGeneratorCis  # used only for signature probing


def make_synthetic_data(
    m_variants: int,
    n_samples: int,
    n_pheno: int,
    n_covars: int,
    chrom: str = "1",
    region_start: int = 1_000_000,
    region_step: int = 50,
    window: int = 2_000_000,
    seed: int = 1337,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, int]:
    """
    Create synthetic genotype/phenotype/covariate data. Every phenotype's cis-window covers all variants.
    Returns: genotype_df, variant_df, phenotype_df, phenotype_pos_df, covariates_df, window
    """
    rng = np.random.default_rng(seed)

    # Samples
    samples = [f"S{i:03d}" for i in range(n_samples)]

    # Variants (positions evenly spaced)
    variant_ids = [f"{chrom}_{region_start + i*region_step}_A_G" for i in range(m_variants)]
    positions = np.array([region_start + i*region_step for i in range(m_variants)], dtype=np.int32)
    variant_df = pd.DataFrame(
        {"chrom": chrom, "pos": positions},
        index=pd.Index(variant_ids, name="variant_id"),
    )

    # Genotypes in {0,1,2} with random MAF; inject ~1% -9 to test imputation
    maf = rng.uniform(0.05, 0.45, size=m_variants)
    G = np.vstack([rng.binomial(2, p, size=n_samples) for p in maf]).astype(np.float32)
    miss_mask = rng.random(G.shape) < 0.01
    G[miss_mask] = -9
    genotype_df = pd.DataFrame(G, index=variant_df.index, columns=samples)

    # Phenotypes
    Y = rng.normal(size=(n_pheno, n_samples)).astype(np.float32)
    phenotype_ids = [f"PHEN{i:04d}" for i in range(n_pheno)]
    phenotype_df = pd.DataFrame(Y, index=phenotype_ids, columns=samples)

    # Positions (single pos per phenotype)
    pheno_pos = region_start + (m_variants // 2) * region_step
    phenotype_pos_df = pd.DataFrame(
        {"chr": chrom, "pos": pheno_pos},
        index=pd.Index(phenotype_ids, name="phenotype_id"),
    )

    # Covariates
    C = rng.normal(size=(n_samples, n_covars)).astype(np.float32)
    covariates_df = pd.DataFrame(C, index=samples, columns=[f"cov{i}" for i in range(n_covars)])

    return genotype_df, variant_df, phenotype_df, phenotype_pos_df, covariates_df, window


def make_synthetic_haplotypes(
    m_variants: int,
    n_samples: int,
    n_ancestries: int,
    seed: int = 1337,
    frac_missing: float = 0.02,
) -> np.ndarray:
    """
    Create one-hot local ancestry calls per (variant, sample) over K ancestries.
    Shape: (m_variants, n_samples, n_ancestries).
    A small fraction set to NaN to exercise on-the-fly interpolation in InputGeneratorCisWithHaps.
    """
    rng = np.random.default_rng(seed)
    # Choose one ancestry index per (variant, sample)
    anc_ix = rng.integers(low=0, high=n_ancestries, size=(m_variants, n_samples))
    H = np.zeros((m_variants, n_samples, n_ancestries), dtype=np.float32)
    rows = np.arange(m_variants)[:, None]
    cols = np.arange(n_samples)[None, :]
    H[rows, cols, anc_ix] = 1.0

    if frac_missing > 0:
        mask = rng.random(size=(m_variants, n_samples, 1)) < frac_missing
        H = H.astype(np.float32)
        H[mask.repeat(n_ancestries, axis=2)] = np.nan  # will be interpolated/rounded

    return H


def try_functional_core_with_haps(
    genotype_df, variant_df, phenotype_df, phenotype_pos_df, covariates_df, haplotypes, device: str
):
    """
    Try to call localqtl.cis._run_nominal_core using an InputGeneratorCisWithHaps.
    If not available, return (None, None) to indicate 'skipped'.
    """
    import localqtl.cis as cis_mod
    if not hasattr(cis_mod, "_run_nominal_core"):
        return None, "functional core (_run_nominal_core) not exported; skipping."

    rez = Residualizer(torch.tensor(covariates_df.values, dtype=torch.float32).to(device))
    ig_h = InputGeneratorCisWithHaps(
        genotype_df, variant_df, phenotype_df, phenotype_pos_df,
        ha
