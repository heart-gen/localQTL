#!/usr/bin/env python3
"""
Benchmark & correctness test for conditionally independent cis-QTLs:
- functional map_independent(...)
- CisMapper(...).map_independent(...)

Flow:
  1) Generate synthetic data (+ optional haplotypes).
  2) Run map_permutations to get per-phenotype top hits with pval_beta.
  3) Compute q-values (BH) → cis_df['qval'].
  4) Run independent mapping with both APIs and compare.

Usage:
  python tests/bench_independent.py \
      --variants 4000 \
      --phenotypes 200 \
      --samples 256 \
      --covars 6 \
      --nperm 4000 \
      --fdr 0.05 \
      --maf 0.00 \
      --ancestries 0 \
      --device auto \
      --csv bench_independent_results.csv
"""

import argparse
import os
import sys
import time
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import torch

# --- Repo import path (adjust if needed) ---
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from localqtl.cis import (
    map_permutations,
    map_independent,
    CisMapper,
)

# ----------------------------
# Synthetic data generators
# ----------------------------
def make_synthetic_data(
    m_variants: int,
    n_samples: int,
    n_pheno: int,
    n_covars: int,
    chrom: str = "1",
    region_start: int = 1_000_000,
    region_step: int = 50,
    window: int = 1_000_000,
    seed: int = 1337,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, int]:
    """
    Create synthetic genotype / phenotype / covariate data.
    Every phenotype's cis-window covers all variants.
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

    # Genotypes with random MAF; sprinkle ~1% missing as -9
    maf = rng.uniform(0.05, 0.45, size=m_variants)
    G = np.vstack([rng.binomial(2, p, size=n_samples) for p in maf]).astype(np.float32)
    miss_mask = rng.random(G.shape) < 0.01
    G[miss_mask] = -9
    genotype_df = pd.DataFrame(G, index=variant_df.index, columns=samples)

    # Phenotypes ~ N(0,1)
    Y = rng.normal(size=(n_pheno, n_samples)).astype(np.float32)
    phenotype_ids = [f"PHEN{i:04d}" for i in range(n_pheno)]
    phenotype_df = pd.DataFrame(Y, index=phenotype_ids, columns=samples)

    # One position per phenotype (midpoint)
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
    One-hot local ancestry per (variant, sample) over K ancestries.
    Shape: (m_variants, n_samples, K).
    A small fraction set to NaN to exercise interpolation paths.
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
        H[mask.repeat(n_ancestries, axis=2)] = np.nan

    return H


# ----------------------------
# Utilities
# ----------------------------
def bh_qvalues(pvals: np.ndarray) -> np.ndarray:
    """
    Benjamini–Hochberg FDR control (q-values) for 1-D p-value array.
    """
    p = np.asarray(pvals, dtype=float)
    n = p.size
    order = np.argsort(p)
    ranks = np.arange(1, n + 1, dtype=float)
    p_sorted = p[order]
    q_sorted = p_sorted * n / ranks
    # monotone non-increasing when traversed from end
    q_sorted = np.minimum.accumulate(q_sorted[::-1])[::-1]
    q = np.empty_like(q_sorted)
    q[order] = np.clip(q_sorted, 0.0, 1.0)
    return q


def compare_independent_results(df_func: pd.DataFrame, df_class: pd.DataFrame, tol: float = 1e-4) -> dict:
    """
    Compare sets of (phenotype_id, variant_id). Also compute diffs on overlaps for ['beta','se','tstat'].
    """
    key = ["phenotype_id", "variant_id"]
    left_keys = set(map(tuple, df_func[key].to_numpy()))
    right_keys = set(map(tuple, df_class[key].to_numpy()))

    inter = left_keys & right_keys
    only_left = left_keys - right_keys
    only_right = right_keys - left_keys

    # numeric diffs on intersection
    if inter:
        mleft = df_func.set_index(key)
        mright = df_class.set_index(key)
        common_ix = pd.MultiIndex.from_tuples(sorted(list(inter)), names=key)
        merged = pd.concat(
            [mleft.loc[common_ix], mright.loc[common_ix]],
            axis=1,
            keys=["f", "c"]
        )
        diffs = {}
        for col in ["beta", "se", "tstat", "pval_beta", "pval_nominal", "pval_perm"]:
            try:
                vf = merged[("f", col)].to_numpy(dtype=float)
                vc = merged[("c", col)].to_numpy(dtype=float)
                d = np.nanmax(np.abs(vf - vc))
                diffs[col] = float(d)
            except Exception:
                pass
    else:
        diffs = {}

    return {
        "n_func": len(left_keys),
        "n_class": len(right_keys),
        "n_intersect": len(inter),
        "n_only_func": len(only_left),
        "n_only_class": len(only_right),
        "numeric_max_abs_diffs": diffs,
        "agree_within_tol": all(d <= tol for d in diffs.values()) if diffs else True,
    }


# ----------------------------
# Main bench
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", type=int, default=4000)
    ap.add_argument("--phenotypes", type=int, default=200)
    ap.add_argument("--samples", type=int, default=256)
    ap.add_argument("--covars", type=int, default=6)
    ap.add_argument("--nperm", type=int, default=4000)
    ap.add_argument("--fdr", type=float, default=0.05)
    ap.add_argument("--maf", type=float, default=0.00)
    ap.add_argument("--ancestries", type=int, default=0, help="0 disables haplotypes; otherwise K ancestries")
    ap.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--csv", type=str, default="")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--random_tiebreak", action="store_true", default=False)
    args = ap.parse_args()

    # Device
    device = (
        "cuda" if (args.device == "auto" and torch.cuda.is_available())
        else (args.device if args.device != "auto" else "cpu")
    )

    # Seeds (reset before each major call too)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Build synthetic data
    geno, var_df, pheno, pheno_pos, covs, window = make_synthetic_data(
        m_variants=args.variants,
        n_samples=args.samples,
        n_pheno=args.phenotypes,
        n_covars=args. covars,
        window=1_000_000,
        seed=args.seed,
    )

    H = None
    if args.ancestries and args.ancestries > 0:
        H = make_synthetic_haplotypes(
            m_variants=args.variants,
            n_samples=args.samples,
            n_ancestries=args.ancestries,
            seed=args.seed,
            frac_missing=0.02,
        )

    # 1) map_permutations → cis_df with q-values
    print("\n=== Step 1: map_permutations to produce cis_df (with q-values) ===")
    np.random.seed(args.seed); torch.manual_seed(args.seed)
    t0 = time.perf_counter()
    cis = map_permutations(
        genotype_df=geno, variant_df=var_df, phenotype_df=pheno,
        phenotype_pos_df=pheno_pos, covariates_df=covs,
        haplotypes=H, loci_df=None, group_s=None,
        maf_threshold=args.maf, window=window,
        nperm=args.nperm, device=device, beta_approx=True,
        logger=None, verbose=True,
    )
    p_time = time.perf_counter() - t0
    print(f"map_permutations: {p_time:.3f} s, rows={len(cis):,}")

    # Compute BH q-values on pval_beta (tensorQTL uses gene-level empirical p)
    if "pval_beta" not in cis.columns:
        raise RuntimeError("map_permutations output must contain 'pval_beta'.")
    cis = cis.copy()
    cis["qval"] = bh_qvalues(cis["pval_beta"].to_numpy())
    print(f"  computed BH q-values; {np.sum(cis['qval'] <= args.fdr):,} significant at FDR ≤ {args.fdr:g}")

    # 2) map_independent (functional)
    print("\n=== Step 2: map_independent (functional) ===")
    np.random.seed(args.seed); torch.manual_seed(args.seed)
    t0 = time.perf_counter()
    ind_func = map_independent(
        genotype_df=geno, variant_df=var_df, cis_df=cis,
        phenotype_df=pheno, phenotype_pos_df=pheno_pos,
        covariates_df=covs, haplotypes=H, loci_df=None, group_s=None,
        maf_threshold=args.maf, fdr=args.fdr, fdr_col="qval",
        nperm=args.nperm, window=window, missing=-9.0,
        random_tiebreak=args.random_tiebreak, device=device,
        beta_approx=True, logger=None, verbose=True,
    )
    t_func = time.perf_counter() - t0
    print(f"map_independent (functional): {t_func:.3f} s, rows={len(ind_func):,}")

    # 3) CisMapper(...).map_independent
    print("\n=== Step 3: CisMapper(...).map_independent ===")
    mapper = CisMapper(
        genotype_df=geno, variant_df=var_df, phenotype_df=pheno,
        phenotype_pos_df=pheno_pos, covariates_df=covs,
        group_s=None, haplotypes=H, loci_df=None,
        device=device, window=window, maf_threshold=args.maf,
        logger=None, verbose=True,
    )
    np.random.seed(args.seed); torch.manual_seed(args.seed)
    t0 = time.perf_counter()
    ind_class = mapper.map_independent(
        cis_df=cis, fdr=args.fdr, fdr_col="qval", nperm=args.nperm,
        maf_threshold=args.maf, random_tiebreak=args.random_tiebreak,
        seed=args.seed, logp=False, missing_val=-9.0,
    )
    t_class = time.perf_counter() - t0
    print(f"CisMapper.map_independent: {t_class:.3f} s, rows={len(ind_class):,}")

    # 4) Compare results (set overlap + numeric diffs on overlaps)
    print("\n=== Comparison: functional vs class ===")
    cmp = compare_independent_results(ind_func, ind_class, tol=1e-4)
    print(f"  #calls: func={cmp['n_func']}, class={cmp['n_class']}")
    print(f"  overlap={cmp['n_intersect']}, only_func={cmp['n_only_func']}, only_class={cmp['n_only_class']}")
    if cmp["numeric_max_abs_diffs"]:
        print("  numeric max |diff| on overlaps:", cmp["numeric_max_abs_diffs"])
        print(f"  agree_within_tol={cmp['agree_within_tol']}")

    # Summarize & optionally save CSV
    out = pd.DataFrame([{
        "variants": args.variants,
        "phenotypes": args.phenotypes,
        "samples": args.samples,
        "covars": args.covars,
        "nperm": args.nperm,
        "fdr": args.fdr,
        "maf": args.maf,
        "ancestries": args.ancestries,
        "device": device,
        "time_map_permutations_sec": p_time,
        "time_map_independent_func_sec": t_func,
        "time_map_independent_class_sec": t_class,
        "rows_independent_func": len(ind_func),
        "rows_independent_class": len(ind_class),
        "n_pairs_func": cmp["n_func"],
        "n_pairs_class": cmp["n_class"],
        "n_pairs_intersect": cmp["n_intersect"],
        "n_pairs_only_func": cmp["n_only_func"],
        "n_pairs_only_class": cmp["n_only_class"],
        "agree_within_tol": cmp["agree_within_tol"],
        **{f"maxdiff_{k}": v for k, v in cmp["numeric_max_abs_diffs"].items()},
    }])

    print("\n=== Timing summary (seconds) ===")
    print(out[[
        "variants", "phenotypes", "samples", "nperm", "ancestries", "device",
        "time_map_permutations_sec",
        "time_map_independent_func_sec", "time_map_independent_class_sec",
        "rows_independent_func", "rows_independent_class",
        "n_pairs_intersect", "n_pairs_only_func", "n_pairs_only_class",
        "agree_within_tol",
    ]].to_string(index=False))

    if args.csv:
        out.to_csv(args.csv, index=False)
        print(f"\nSaved CSV: {args.csv}")


if __name__ == "__main__":
    main()
