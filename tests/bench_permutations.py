#!/usr/bin/env python3
"""
Permutation mapping bench & sanity checks:

Runs both functional and OO APIs for:
  (A) WITHOUT haplotypes
  (B) WITH haplotypes (K ancestries)

For each setting, it:
  - times map_permutations()
  - compares functional vs OO columns (beta, se, tstat, r2_nominal, pval_nominal, pval_perm, pval_beta)
  - checks basic sanity (p-values within (0,1], correlations between empirical and Beta-approx)

Usage:
  python tests/bench_permutations.py \
      --variants 2000,8000 \
      --phenotypes 50,200 \
      --ancestries 2,3 \
      --samples 256 \
      --covars 6 \
      --nperm 200 \
      --maf 0.00 \
      --device auto \
      --csv bench_perm_results.csv

Notes:
- If you see NameError about `k_eff` inside the library, set it in the relevant core before calling
  run_batch_regression_with_permutations (e.g., k_eff = rez.Q_t.shape[1] if rez else 0).
"""

import argparse
import os
import sys
import time
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import torch

# Repo import path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from localqtl.cis import map_permutations, CisMapper


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
    window: int = 2_000_000,
    seed: int = 1337,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, int]:
    rng = np.random.default_rng(seed)

    samples = [f"S{i:03d}" for i in range(n_samples)]

    # Variants & positions
    variant_ids = [f"{chrom}_{region_start + i * region_step}_A_G" for i in range(m_variants)]
    positions = np.array([region_start + i * region_step for i in range(m_variants)], dtype=np.int32)
    variant_df = pd.DataFrame(
        {"chrom": chrom, "pos": positions},
        index=pd.Index(variant_ids, name="variant_id"),
    )

    # Genotypes (with a bit of missing coded as -9)
    maf = rng.uniform(0.05, 0.45, size=m_variants)
    G = np.vstack([rng.binomial(2, p, size=n_samples) for p in maf]).astype(np.float32)
    miss_mask = rng.random(G.shape) < 0.01
    G[miss_mask] = -9
    genotype_df = pd.DataFrame(G, index=variant_df.index, columns=samples)

    # Phenotypes
    Y = rng.normal(size=(n_pheno, n_samples)).astype(np.float32)
    phenotype_ids = [f"PHEN{i:04d}" for i in range(n_pheno)]
    phenotype_df = pd.DataFrame(Y, index=phenotype_ids, columns=samples)

    # All phenotypes in the middle of the region (so every cis-window covers all variants)
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
    seed: int = 13,
    frac_missing: float = 0.02,
) -> np.ndarray:
    """
    One-hot local ancestry calls per (variant, sample) over K ancestries.
    Shape: (m_variants, n_samples, n_ancestries). Some entries set to NaN to
    exercise interpolation/imputation in the InputGenerator.
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
# Comparisons & sanity checks
# ----------------------------
def compare_perm_results(
    df_func: pd.DataFrame, df_class: pd.DataFrame, tol: float = 1e-6
) -> pd.Series:
    """
    Compare functional vs OO outputs for permutation mapping.
    Returns max|diff| per numeric column of interest.
    """
    left = df_func.copy()
    right = df_class.copy()

    join_cols = ["phenotype_id"]
    for need in join_cols:
        if need not in left.columns or need not in right.columns:
            raise ValueError(f"Missing '{need}' in outputs to compare.")

    merged = left.merge(right, on=join_cols, suffixes=("_f", "_c"))

    numeric_cols = ["beta", "se", "tstat", "r2_nominal", "pval_nominal", "pval_perm", "pval_beta"]
    out = {}
    for col in numeric_cols:
        cf, cc = f"{col}_f", f"{col}_c"
        if cf in merged.columns and cc in merged.columns:
            a = merged[cf].to_numpy()
            b = merged[cc].to_numpy()
            mask = np.isfinite(a) & np.isfinite(b)
            diff = np.max(np.abs(a[mask] - b[mask])) if np.any(mask) else np.nan
            out[col] = float(diff)

    s = pd.Series(out)
    if np.nanmax(s.values) > tol:
        print("[WARN] Functional vs OO diffs exceed tol:", s.to_dict())
    else:
        print(f"[OK] Functional vs OO agree within tol={tol}. Max abs diffs: {s.to_dict()}")
    return s


def sanity_checks(df: pd.DataFrame, label: str) -> dict:
    """
    Basic validity: p-values in (0,1], r2_nominal in [0,1), pval_beta ~ pval_perm.
    Returns small dict of summary metrics.
    """
    res = {}
    if "pval_perm" in df and "pval_beta" in df:
        # correlation on log scale to reduce tail compression effects
        p1 = df["pval_perm"].to_numpy()
        p2 = df["pval_beta"].to_numpy()
        mask = np.isfinite(p1) & np.isfinite(p2) & (p1 > 0) & (p2 > 0)
        if np.any(mask):
            x = -np.log10(p1[mask])
            y = -np.log10(p2[mask])
            # Pearson on logs (simple, dependency-free)
            if x.std() > 0 and y.std() > 0:
                corr = float(np.corrcoef(x, y)[0, 1])
            else:
                corr = np.nan
            res["corr_logp_beta_vs_emp"] = corr

    def frac_in_open01(a: np.ndarray) -> float:
        m = np.isfinite(a)
        if not np.any(m):
            return np.nan
        return float(np.mean((a[m] > 0.0) & (a[m] <= 1.0)))

    for col in ["pval_nominal", "pval_perm", "pval_beta"]:
        if col in df:
            res[f"valid_{col}"] = frac_in_open01(df[col].to_numpy())

    if "r2_nominal" in df:
        r2 = df["r2_nominal"].to_numpy()
        m = np.isfinite(r2)
        res["valid_r2_nominal"] = float(np.mean((r2[m] >= 0.0) & (r2[m] < 1.0))) if np.any(m) else np.nan

    print(f"[SANITY:{label}] {res}")
    return res


# ----------------------------
# Main
# ----------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variants", type=str, default="2000,8000")
    p.add_argument("--phenotypes", type=str, default="50,200")
    p.add_argument("--ancestries", type=str, default="3")
    p.add_argument("--samples", type=int, default=256)
    p.add_argument("--covars", type=int, default=6)
    p.add_argument("--nperm", type=int, default=200)
    p.add_argument("--maf", type=float, default=0.0)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--beta_approx", action="store_true", help="Also compute Beta-approx p-values.")
    p.add_argument("--csv", type=str, default="")
    args = p.parse_args()

    device = "cuda" if (args.device == "auto" and torch.cuda.is_available()) else (args.device if args.device != "auto" else "cpu")
    var_list = [int(x) for x in args.variants.split(",") if x.strip()]
    phe_list = [int(x) for x in args.phenotypes.split(",") if x.strip()]
    anc_list = [int(x) for x in args.ancestries.split(",") if x.strip()]

    records: List[dict] = []

    for m in var_list:
        for p_ in phe_list:
            # Base synthetic data (no H)
            geno, var_df, pheno, pheno_pos, covs, window = make_synthetic_data(
                m_variants=m, n_samples=args.samples, n_pheno=p_, n_covars=args.covars,
                window=1_000_000, seed=13,
            )

            # ----------------- WITHOUT HAPS -----------------
            print(f"\n=== Perm bench (NO H): m={m}, p={p_} ===")

            # Functional
            try:
                t0 = time.perf_counter()
                df_func = map_permutations(
                    genotype_df=geno, variant_df=var_df, phenotype_df=pheno,
                    phenotype_pos_df=pheno_pos, covariates_df=covs,
                    maf_threshold=args.maf, window=window, nperm=args.nperm,
                    device=device, beta_approx=args.beta_approx,
                )
                t_func = time.perf_counter() - t0
                print(f"map_permutations (no H): {t_func:.3f} s, rows={len(df_func):,}")
            except Exception as e:
                print(f"[ERROR] functional map_permutations (no H) failed: {e}")
                df_func, t_func = pd.DataFrame(), np.nan

            # OO
            try:
                mapper = CisMapper(
                    genotype_df=geno, variant_df=var_df, phenotype_df=pheno,
                    phenotype_pos_df=pheno_pos, covariates_df=covs,
                    device=device, window=window, maf_threshold=args.maf,
                )
                t0 = time.perf_counter()
                df_class = mapper.map_permutations(nperm=args.nperm, beta_approx=args.beta_approx)
                t_class = time.perf_counter() - t0
                print(f"CisMapper.map_permutations (no H): {t_class:.3f} s, rows={len(df_class):,}")
            except Exception as e:
                print(f"[ERROR] OO map_permutations (no H) failed: {e}")
                df_class, t_class = pd.DataFrame(), np.nan

            # Compare + sanity
            if not df_func.empty and not df_class.empty:
                diffs_noH = compare_perm_results(df_func, df_class, tol=1e-6)
                sanity_noH_func = sanity_checks(df_func, "noH:func")
                sanity_noH_class = sanity_checks(df_class, "noH:class")
            else:
                diffs_noH = pd.Series({})
                sanity_noH_func = sanity_noH_class = {}

            records.append(
                dict(
                    mode="no_haps", variants=m, phenotypes=p_, samples=args.samples,
                    covars=args.covars, ancestries=np.nan, nperm=args.nperm, maf=args.maf,
                    device=device, time_functional_sec=t_func, time_class_sec=t_class,
                    rows_func=len(df_func), rows_class=len(df_class),
                    maxdiff_beta=float(diffs_noH.get("beta", np.nan)) if not diffs_noH.empty else np.nan,
                    maxdiff_se=float(diffs_noH.get("se", np.nan)) if not diffs_noH.empty else np.nan,
                    maxdiff_t=float(diffs_noH.get("tstat", np.nan)) if not diffs_noH.empty else np.nan,
                    maxdiff_r2=float(diffs_noH.get("r2_nominal", np.nan)) if not diffs_noH.empty else np.nan,
                    maxdiff_p=float(diffs_noH.get("pval_nominal", np.nan)) if not diffs_noH.empty else np.nan,
                    maxdiff_pperm=float(diffs_noH.get("pval_perm", np.nan)) if not diffs_noH.empty else np.nan,
                    corr_logp_beta_vs_emp_func=float(sanity_noH_func.get("corr_logp_beta_vs_emp", np.nan)) if sanity_noH_func else np.nan,
                    corr_logp_beta_vs_emp_class=float(sanity_noH_class.get("corr_logp_beta_vs_emp", np.nan)) if sanity_noH_class else np.nan,
                )
            )

            # ----------------- WITH HAPS -----------------
            for k in anc_list:
                print(f"\n=== Perm bench (WITH H): m={m}, p={p_}, K={k} ===")
                H = make_synthetic_haplotypes(
                    m_variants=m, n_samples=args.samples, n_ancestries=k, seed=13, frac_missing=0.02
                )

                # Functional (with H)
                try:
                    t0 = time.perf_counter()
                    df_func_h = map_permutations(
                        genotype_df=geno, variant_df=var_df, phenotype_df=pheno,
                        phenotype_pos_df=pheno_pos, covariates_df=covs, haplotypes=H,
                        loci_df=None, maf_threshold=args.maf, window=window, nperm=args.nperm,
                        device=device, beta_approx=args.beta_approx,
                    )
                    t_func_h = time.perf_counter() - t0
                    print(f"map_permutations (with H): {t_func_h:.3f} s, rows={len(df_func_h):,}")
                except Exception as e:
                    print(f"[ERROR] functional map_permutations (with H) failed: {e}")
                    df_func_h, t_func_h = pd.DataFrame(), np.nan

                # OO (with H)
                try:
                    mapper_h = CisMapper(
                        genotype_df=geno, variant_df=var_df, phenotype_df=pheno,
                        phenotype_pos_df=pheno_pos, covariates_df=covs,
                        haplotypes=H, loci_df=None, device=device, window=window,
                        maf_threshold=args.maf,
                    )
                    t0 = time.perf_counter()
                    df_class_h = mapper_h.map_permutations(nperm=args.nperm, beta_approx=args.beta_approx)
                    t_class_h = time.perf_counter() - t0
                    print(f"CisMapper.map_permutations (with H): {t_class_h:.3f} s, rows={len(df_class_h):,}")
                except Exception as e:
                    print(f"[ERROR] OO map_permutations (with H) failed: {e}")
                    df_class_h, t_class_h = pd.DataFrame(), np.nan

                # Compare + sanity
                if not df_func_h.empty and not df_class_h.empty:
                    diffs_H = compare_perm_results(df_func_h, df_class_h, tol=1e-6)
                    sanity_H_func = sanity_checks(df_func_h, "withH:func")
                    sanity_H_class = sanity_checks(df_class_h, "withH:class")
                else:
                    diffs_H = pd.Series({})
                    sanity_H_func = sanity_H_class = {}

                records.append(
                    dict(
                        mode="with_haps", variants=m, phenotypes=p_, samples=args.samples,
                        covars=args.covars, ancestries=k, nperm=args.nperm, maf=args.maf,
                        device=device, time_functional_sec=t_func_h, time_class_sec=t_class_h,
                        rows_func=len(df_func_h), rows_class=len(df_class_h),
                        maxdiff_beta=float(diffs_H.get("beta", np.nan)) if not diffs_H.empty else np.nan,
                        maxdiff_se=float(diffs_H.get("se", np.nan)) if not diffs_H.empty else np.nan,
                        maxdiff_t=float(diffs_H.get("tstat", np.nan)) if not diffs_H.empty else np.nan,
                        maxdiff_r2=float(diffs_H.get("r2_nominal", np.nan)) if not diffs_H.empty else np.nan,
                        maxdiff_p=float(diffs_H.get("pval_nominal", np.nan)) if not diffs_H.empty else np.nan,
                        maxdiff_pperm=float(diffs_H.get("pval_perm", np.nan)) if not diffs_H.empty else np.nan,
                        corr_logp_beta_vs_emp_func=float(sanity_H_func.get("corr_logp_beta_vs_emp", np.nan)) if sanity_H_func else np.nan,
                        corr_logp_beta_vs_emp_class=float(sanity_H_class.get("corr_logp_beta_vs_emp", np.nan)) if sanity_H_class else np.nan,
                    )
                )

    bench_df = pd.DataFrame.from_records(records).sort_values(
        ["mode", "variants", "phenotypes", "ancestries"]
    )

    print("\n=== Timing summary (seconds) ===")
    cols = [
        "mode", "variants", "phenotypes", "ancestries", "samples", "nperm", "maf", "device",
        "time_functional_sec", "time_class_sec", "rows_func", "rows_class",
        "maxdiff_beta", "maxdiff_se", "maxdiff_t", "maxdiff_r2",
        "maxdiff_p", "maxdiff_pperm",
        "corr_logp_beta_vs_emp_func", "corr_logp_beta_vs_emp_class",
    ]
    print(bench_df[cols].to_string(index=False))

    if args.csv:
        bench_df.to_csv(args.csv, index=False)
        print(f"\nSaved CSV: {args.csv}")


if __name__ == "__main__":
    main()
