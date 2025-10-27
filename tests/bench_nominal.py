#!/usr/bin/env python3
"""
Benchmark & correctness test for cis mapping:
- map_cis_nominal (functional)
- SimpleCisMapper.map_nominal (OO)

Scales over (#variants in cis-window, #phenotypes) and reports timings.
Also checks functional vs OO numerical agreement when H=None.

Usage:
  python tests/bench_nominal.py \
      --variants 2000,8000,20000 \
      --phenotypes 50,200 \
      --samples 256 \
      --covars 6 \
      --device auto \
      --csv bench_results.csv
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

# Adjust this path if your package lives elsewhere
sys.path.append("src")

# Imports from your codebase
from cis import map_cis_nominal, SimpleCisMapper
from genotypeio import InputGeneratorCis


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
    Create a small synthetic dataset where every phenotype's cis-window covers all variants.
    Returns: genotype_df, variant_df, phenotype_df, phenotype_pos_df, covariates_df, window
    """
    rng = np.random.default_rng(seed)

    # Samples
    samples = [f"S{i:03d}" for i in range(n_samples)]

    # Variants and positions
    variant_ids = [f"{chrom}_{region_start + i*region_step}_A_G" for i in range(m_variants)]
    positions = np.array([region_start + i*region_step for i in range(m_variants)], dtype=np.int32)
    variant_df = pd.DataFrame(
        {"chrom": chrom, "pos": positions},
        index=pd.Index(variant_ids, name="variant_id"),
    )

    # Genotypes in {0,1,2} with random MAF; inject a few -9 to test imputation
    maf = rng.uniform(0.05, 0.45, size=m_variants)
    G = np.vstack([rng.binomial(2, p, size=n_samples) for p in maf]).astype(np.float32)
    # sprinkle ~1% missing
    miss_mask = rng.random(G.shape) < 0.01
    G[miss_mask] = -9
    genotype_df = pd.DataFrame(G, index=variant_df.index, columns=samples)

    # Phenotypes (centered-ish)
    Y = rng.normal(size=(n_pheno, n_samples)).astype(np.float32)
    phenotype_ids = [f"PHEN{i:04d}" for i in range(n_pheno)]
    phenotype_df = pd.DataFrame(Y, index=phenotype_ids, columns=samples)

    # Phenotype positions: one per gene near the middle of the region
    pheno_pos = region_start + (m_variants // 2) * region_step
    phenotype_pos_df = pd.DataFrame(
        {"chr": chrom, "pos": pheno_pos},
        index=pd.Index(phenotype_ids, name="phenotype_id"),
    )

    # Covariates (samples x covars)
    C = rng.normal(size=(n_samples, n_covars)).astype(np.float32)
    covariates_df = pd.DataFrame(C, index=samples, columns=[f"cov{i}" for i in range(n_covars)])

    return genotype_df, variant_df, phenotype_df, phenotype_pos_df, covariates_df, window


def call_simple_mapper_robust(
    genotype_df, variant_df, phenotype_df, phenotype_pos_df, covariates_df, device: str
) -> pd.DataFrame:
    """
    Instantiates SimpleCisMapper regardless of whether it expects a 'genotype_reader'
    or builds its own generator internally. Returns mapper.map_nominal() DataFrame.
    """
    sig = inspect.signature(SimpleCisMapper.__init__).parameters
    # Older API: (genotype_reader, phenotype_df, phenotype_pos_df, ...)
    if "genotype_reader" in sig:
        ig = InputGeneratorCis(
            genotype_df, variant_df, phenotype_df, phenotype_pos_df, window=2_000_000
        )
        mapper = SimpleCisMapper(
            genotype_reader=ig,
            phenotype_df=phenotype_df,
            phenotype_pos_df=phenotype_pos_df,
            covariates_df=covariates_df,
            device=device,
        )
    else:
        # Newer API (assumes SimpleCisMapper builds the generator itself)
        mapper = SimpleCisMapper(
            genotype_df=genotype_df,
            variant_df=variant_df,
            phenotype_df=phenotype_df,
            phenotype_pos_df=phenotype_pos_df,
            covariates_df=covariates_df,
            device=device,
        )
    return mapper.map_nominal()


def compare_results(df_func: pd.DataFrame, df_class: pd.DataFrame, tol=1e-5) -> pd.Series:
    """
    Align and compare outputs from functional and OO APIs.
    Returns a Series of max |diff| for key numeric columns.
    """
    # Standardize column names that might differ
    left = df_func.copy()
    right = df_class.copy()

    # Key for join
    join_cols = ["phenotype_id", "variant_id"]

    # Ensure required columns exist
    for need in join_cols:
        if need not in left.columns or need not in right.columns:
            raise ValueError(f"Missing '{need}' in outputs to compare.")

    merged = left.merge(right, on=join_cols, suffixes=("_f", "_c"))

    metrics = []
    for col in ["beta", "se", "tstat"]:
        cf = f"{col}_f"
        cc = f"{col}_c"
        if cf in merged.columns and cc in merged.columns:
            diff = np.abs(merged[cf].to_numpy() - merged[cc].to_numpy())
            metrics.append((col, diff.max() if diff.size else 0.0))

    out = pd.Series(dict(metrics))
    if (out > tol).any():
        print("[WARN] Differences exceed tolerance:")
        print(out)
    else:
        print(f"[OK] Functional vs OO agree within tol={tol}. Max abs diffs: {out.to_dict()}")
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", type=str, default="2000,8000",
                        help="Comma-separated list of #variants per window.")
    parser.add_argument("--phenotypes", type=str, default="50,200",
                        help="Comma-separated list of #phenotypes.")
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--covars", type=int, default=6)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--csv", type=str, default="")
    args = parser.parse_args()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    var_list = [int(x) for x in args.variants.split(",") if x.strip()]
    phe_list = [int(x) for x in args.phenotypes.split(",") if x.strip()]

    records: List[dict] = []

    for m in var_list:
        for p in phe_list:
            print(f"\n=== Bench: m={m} variants, p={p} phenotypes ===")
            geno, var_df, pheno, pheno_pos, covs, window = make_synthetic_data(
                m_variants=m,
                n_samples=args.samples,
                n_pheno=p,
                n_covars=args.covars,
                window=2_000_000,
                seed=42,
            )

            # Functional API
            t0 = time.perf_counter()
            df_func = map_cis_nominal(
                genotype_df=geno,
                variant_df=var_df,
                phenotype_df=pheno,
                phenotype_pos_df=pheno_pos,
                covariates_df=covs,
                window=window,
                nperm=None,
                device=device,
            )
            t1 = time.perf_counter()
            t_func = t1 - t0
            print(f"map_cis_nominal: {t_func:.3f} s, rows={len(df_func):,}")

            # OO API
            t0 = time.perf_counter()
            df_class = call_simple_mapper_robust(
                genotype_df=geno,
                variant_df=var_df,
                phenotype_df=pheno,
                phenotype_pos_df=pheno_pos,
                covariates_df=covs,
                device=device,
            )
            t1 = time.perf_counter()
            t_class = t1 - t0
            print(f"SimpleCisMapper.map_nominal: {t_class:.3f} s, rows={len(df_class):,}")

            # Compare numerics
            try:
                diffs = compare_results(df_func, df_class, tol=1e-5)
                max_beta = float(diffs.get("beta", np.nan))
                max_se = float(diffs.get("se", np.nan))
                max_t = float(diffs.get("tstat", np.nan))
            except Exception as e:
                print(f"[WARN] Comparison skipped due to: {e}")
                max_beta = max_se = max_t = np.nan

            records.append(
                dict(
                    variants=m,
                    phenotypes=p,
                    samples=args.samples,
                    covars=args.covars,
                    device=device,
                    time_map_cis_nominal_sec=t_func,
                    time_simple_mapper_sec=t_class,
                    rows_func=len(df_func),
                    rows_class=len(df_class),
                    max_abs_diff_beta=max_beta,
                    max_abs_diff_se=max_se,
                    max_abs_diff_tstat=max_t,
                )
            )

    bench_df = pd.DataFrame.from_records(records).sort_values(["variants", "phenotypes"])
    print("\n=== Timing summary (seconds) ===")
    print(
        bench_df[
            [
                "variants",
                "phenotypes",
                "samples",
                "device",
                "time_map_cis_nominal_sec",
                "time_simple_mapper_sec",
                "rows_func",
                "rows_class",
            ]
        ].to_string(index=False)
    )

    if args.csv:
        bench_df.to_csv(args.csv, index=False)
        print(f"\nSaved CSV: {args.csv}")


if __name__ == "__main__":
    main()
