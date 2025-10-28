#!/usr/bin/env python3
"""
Benchmark & correctness test for cis mapping WITH HAPLOTYPES:
- functional map_nominal (with haplotypes)
- CisMapper.map_nominal (OO API with haplotypes)

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

# Repo import path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from localqtl.cis import CisMapper, map_nominal

def make_synthetic_data(
        m_variants: int, n_samples: int, n_pheno: int, n_covars: int,
        chrom: str = "1", region_start: int = 1_000_000, region_step: int = 50,
        window: int = 2_000_000, seed: int = 1337,
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
        m_variants: int, n_samples: int, n_ancestries: int, seed: int = 1337,
        frac_missing: float = 0.02,
) -> np.ndarray:
    """
    Create one-hot local ancestry calls per (variant, sample) over K ancestries.
    Shape: (m_variants, n_samples, n_ancestries).
    A small fraction set to NaN to exercise on-the-fly interpolation in InputGeneratorCisWithHaps.
    """
    rng = np.random.default_rng(seed)
    anc_ix = rng.integers(low=0, high=n_ancestries, size=(m_variants, n_samples))
    H = np.zeros((m_variants, n_samples, n_ancestries), dtype=np.float32)
    rows = np.arange(m_variants)[:, None]
    cols = np.arange(n_samples)[None, :]
    H[rows, cols, anc_ix] = 1.0

    if frac_missing > 0:
        mask = rng.random(size=(m_variants, n_samples, 1)) < frac_missing
        H = H.astype(np.float32)
        H[mask.repeat(n_ancestries, axis=2)] = np.nan # will be interpolated/rounded

    return H


def map_nominal_with_haps(
        genotype_df, variant_df, phenotype_df, phenotype_pos_df, covariates_df,
        haplotypes, window: int, device: str
):
    """
    Try to call map_nominal with haplotypes if the function signature supports it.
    Falls back to calling without haplotypes and returns a note.
    """
    sig = inspect.signature(map_nominal).parameters
    kwargs = dict(
        genotype_df=genotype_df,
        variant_df=variant_df,
        phenotype_df=phenotype_df,
        phenotype_pos_df=phenotype_pos_df,
        covariates_df=covariates_df,
        window=window,
        nperm=None,
        device=device,
    )
    supported_h_keys = [k for k in ("haplotypes", "H", "haplotype_reader") if k in sig]
    if haplotypes is not None and len(supported_h_keys) > 0:
        # Prefer explicit "haplotypes" or "H"
        if "haplotypes" in supported_h_keys:
            kwargs["haplotypes"] = haplotypes
            note = ""
        elif "H" in supported_h_keys:
            kwargs["H"] = haplotypes
            note = ""
        else:
            note = "functional API exposes haplotype_reader but no direct haplotype array; ran without H"
    else:
        note = "functional API does not accept H; ran without H"
    
    t0 = time.perf_counter()
    df = map_nominal(**kwargs)
    t1 = time.perf_counter()
    return df, (t1 - t0), note


def call_mapper_with_haps(
    genotype_df, variant_df, phenotype_df, phenotype_pos_df, covariates_df,
    haplotypes, device: str, loci_df=None
):
    """
    Build a CisMapper with haplotypes and run map_nominal().
    """
    sig = inspect.signature(CisMapper.__init__).parameters
    # Newer API (cis.py provided): accepts haplotypes (and optional loci_df)
    if "haplotypes" in sig:
        mapper = CisMapper(
            genotype_df=genotype_df,
            variant_df=variant_df,
            phenotype_df=phenotype_df,
            phenotype_pos_df=phenotype_pos_df,
            covariates_df=covariates_df,
            haplotypes=haplotypes,
            loci_df=loci_df,
            device=device,
        )
    else:
        # Very old fallback (unlikely for your tree). Runs without H.
        mapper = CisMapper(
             genotype_df=genotype_df,
             variant_df=variant_df,
             phenotype_df=phenotype_df,
             phenotype_pos_df=phenotype_pos_df,
             covariates_df=covariates_df,
             device=device,
         )
    t0 = time.perf_counter()
    df_class = mapper.map_nominal()
    t1 = time.perf_counter()
    return df_class, (t1 - t0)


def compare_results(df_func: pd.DataFrame, df_class: pd.DataFrame, tol=1e-5) -> pd.Series:
    """
    Align and compare outputs from functional and OO APIs (for genotype effect columns).
    Returns a Series of max |diff| for ['beta','se','tstat'].
    """
    left = df_func.copy()
    right = df_class.copy()

    join_cols = ["phenotype_id", "variant_id"]
    for need in join_cols:
        if need not in left.columns or need not in right.columns:
            raise ValueError(f"Missing '{need}' in outputs to compare.")

    merged = left.merge(right, on=join_cols, suffixes=("_f", "_c"))

    metrics = []
    for col in ["beta", "se", "tstat"]:
        cf, cc = f"{col}_f", f"{col}_c"
        if cf in merged.columns and cc in merged.columns:
            diff = np.abs(merged[cf].to_numpy() - merged[cc].to_numpy())
            metrics.append((col, diff.max() if diff.size else 0.0))

    out = pd.Series(dict(metrics))
    if (out > tol).any():
        print("[WARN] Functional vs OO diffs exceed tol:", out.to_dict())
    else:
        print(f"[OK] Functional vs OO agree within tol={tol}. Max abs diffs: {out.to_dict()}")
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", type=str, default="2000,8000",
                        help="Comma-separated list of #variants per window.")
    parser.add_argument("--phenotypes", type=str, default="50,200",
                        help="Comma-separated list of #phenotypes.")
    parser.add_argument("--ancestries", type=str, default="3",
                        help="Comma-separated list of #ancestries (K).")
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--covars", type=int, default=6)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--csv", type=str, default="")
    args = parser.parse_args()

    device = "cuda" if (args.device == "auto" and torch.cuda.is_available()) else (args.device if args.device != "auto" else "cpu")

    var_list = [int(x) for x in args.variants.split(",") if x.strip()]
    phe_list = [int(x) for x in args.phenotypes.split(",") if x.strip()]
    anc_list = [int(x) for x in args.ancestries.split(",") if x.strip()]

    records: List[dict] = []

    for m in var_list:
        for p in phe_list:
            for k in anc_list:
                print(f"\n=== Bench HAPS: m={m} variants, p={p} phenotypes, K={k} ancestries ===")
                geno, var_df, pheno, pheno_pos, covs, window = make_synthetic_data(
                    m_variants=m,
                    n_samples=args.samples,
                    n_pheno=p,
                    n_covars=args.covars,
                    window=2_000_000,
                    seed=42,
                )
                H = make_synthetic_haplotypes(
                    m_variants=m,
                    n_samples=args.samples,
                    n_ancestries=k,
                    seed=1234,
                    frac_missing=0.02,
                )

                # Functional API
                func_time = np.nan
                max_beta = max_se = max_t = np.nan
                func_note = None
                df_func = None

                df_func, t_func, note = map_nominal_with_haps(
                    genotype_df=geno,
                    variant_df=var_df,
                    phenotype_df=pheno,
                    phenotype_pos_df=pheno_pos,
                    covariates_df=covs,
                    haplotypes=H,
                    window=window,
                    device=device,
                )
                ran_with_H = (note == "")
                print(f"map_nominal: {t_func:.3f} s, rows={len(df_func):,}"
                      + ("" if ran_with_H else f"  [NOTE: {note}]"))

                # OO API
                df_class, cls_time = call_mapper_with_haps(
                    geno, var_df, pheno, pheno_pos, covs, H, device,
                    loci_df=None  # synthetic H already aligned to variant order
                )
                print(f"CisMapper.map_nominal (with H): {cls_time:.3f} s, rows={len(df_class):,}")

                # Compare numerics only
                max_beta = max_se = max_t = np.nan
                if ran_with_H:
                    try:
                        diffs = compare_results(df_func, df_class, tol=1e-5)
                        max_beta = float(diffs.get("beta", np.nan))
                        max_se = float(diffs.get("se", np.nan))
                        max_t = float(diffs.get("tstat", np.nan))
                    except Exception as e:
                        print(f"[WARN] Comparison failed: {e}")

                records.append(
                    dict(
                        variants=m,
                        phenotypes=p,
                        ancestries=k,
                        samples=args.samples,
                        covars=args.covars,
                        device=device,
                        time_map_cis_nominal_sec=t_func,
                        time_simple_mapper_sec=cls_time,
                        rows_func=len(df_func),
                        rows_class=len(df_class),
                        ran_functional_with_H=bool(ran_with_H),
                        max_abs_diff_beta=max_beta,
                        max_abs_diff_se=max_se,
                        max_abs_diff_tstat=max_t,
                        note=note,
                    )
                )

    bench_df = pd.DataFrame.from_records(records).sort_values(["variants", "phenotypes", "ancestries"])
    print("\n=== Timing summary (seconds) ===")
    cols = [
        "variants", "phenotypes", "ancestries", "samples", "device",
        "time_map_cis_nominal_sec", "time_simple_mapper_sec",
        "rows_func", "rows_class", "ran_functional_with_H"
    ]
    print(bench_df[cols].to_string(index=False))

    if args.csv:
        bench_df.to_csv(args.csv, index=False)
        print(f"\nSaved CSV: {args.csv}")


if __name__ == "__main__":
    main()
