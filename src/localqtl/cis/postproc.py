## Adapted from tensorqtl: https://github.com/broadinstitute/tensorqtl/blob/master/tensorqtl/post.py
from __future__ import annotations
import os, glob
import pandas as pd
from typing import Mapping, Optional, List, Union

import pyarrow as pa

from ..utils import SimpleLogger

__all__ = [
    "get_significant_pairs",
    "annotate_ancestry_difference",
    "annotate_with_susie",
    "annotate_with_coloc",
]

def _chrom_sort_key(ch: str) -> tuple:
    """Robust chromosome sort: chr1..chr22, X=23, Y=24, MT/M=25, else lexicographic fallback."""
    s = os.path.basename(ch)
    s = s.replace(".parquet", "")
    s = s[3:] if s.lower().startswith("chr") else s
    mapping = {"x": 23, "y": 24, "mt": 25, "m": 25, "xy": 24}
    t = s.lower()
    if t in mapping: return (0, mapping[t])
    try:
        return (0, int(s))
    except ValueError:
        return (1, s)


def get_significant_pairs(res_df: pd.DataFrame,
                          nominal_files: Union[str, Mapping[str, str]],
                          group_s: Optional[pd.Series] = None,
                          fdr: float = 0.05,
                          columns: Optional[List[str]] = None,
                          logger: Optional[SimpleLogger] = None) -> pd.DataFrame:
    """
    Collect all significant variant–phenotype pairs from per-chromosome nominal Parquet outputs,
    using phenotype-wise nominal thresholds computed by `calculate_qvalues`.

    Parameters
    ----------
    res_df : DataFrame (output of map_permutations + calculate_qvalues)
      Must contain columns: ['qval', 'pval_nominal_threshold'] and either:
        - ungrouped: phenotype-level rows with ['phenotype_id']
        - grouped:   group-level rows with ['group_id'] (pass group_s mapping)
    nominal_files : str or Mapping[str, str]
      Glob pattern (e.g., '/path/nominal/chr*.parquet') OR dict {chrom: path}.
      Each Parquet must contain at least: ['phenotype_id','pval_nominal'] (+ whatever you want to keep).
    group_s : pd.Series, optional
      Mapping phenotype_id -> group_id (required only for grouped mode).
    fdr : float
      FDR cutoff used to select phenotypes/groups.
    columns : list[str], optional
      Columns to read from Parquet for speed. If None, read all.
    """
    lg = logger or SimpleLogger()
    lg.write("Parsing significant variant–phenotype pairs from nominal Parquet files")

    if "qval" not in res_df.columns:
        raise ValueError("res_df must contain 'qval' (run calculate_qvalues first).")

    grouped = group_s is not None
    if grouped and "group_id" not in res_df.columns:
        raise ValueError("Grouped mode: res_df must contain 'group_id'.")

    # Select the seed set and thresholds
    if grouped:
        keep = res_df.loc[res_df["qval"] < fdr, ["group_id", "pval_nominal_threshold"]].copy()
        keep = keep.dropna(subset=["pval_nominal_threshold"])
        if keep.empty:
            lg.write("  * No groups pass FDR; returning empty DataFrame.")
            return pd.DataFrame()
        keep = keep.drop_duplicates("group_id").set_index("group_id")
        if group_s is None:
            raise ValueError("group_s must be provided for grouped mode.")
        if group_s.index.name != "phenotype_id":
            group_s.index.name = "phenotype_id"
        threshold_lookup = keep["pval_nominal_threshold"].to_dict()
        target_groups = set(keep.index)
        lg.write(f"  * Groups passing FDR: {len(target_groups)}")
    else:
        if "phenotype_id" not in res_df.columns or "pval_nominal_threshold" not in res_df.columns:
            raise ValueError("Ungrouped mode: res_df must contain 'phenotype_id' and 'pval_nominal_threshold'.")
        keep = res_df.loc[res_df["qval"] < fdr, ["phenotype_id", "pval_nominal_threshold"]].copy()
        keep = keep.dropna(subset=["pval_nominal_threshold"]).drop_duplicates("phenotype_id").set_index("phenotype_id")
        if keep.empty:
            lg.write("  * No phenotypes pass FDR; returning empty DataFrame.")
            return pd.DataFrame()
        threshold_lookup = keep["pval_nominal_threshold"].to_dict()
        target_phens = set(keep.index)
        lg.write(f"  * Phenotypes passing FDR: {len(target_phens)}")

    # Expand nominal_files glob/dict
    if isinstance(nominal_files, str):
        paths = {os.path.basename(p).split(".")[-2]: p for p in glob.glob(nominal_files)}
    else:
        paths = dict(nominal_files)
    if not paths:
        raise ValueError("No nominal Parquet files found.")

    # Read & filter
    default_cols = ["phenotype_id", "variant_id", "pval_nominal",
                    "start_distance", "end_distance"]
    wanted_cols = columns or default_cols
    out = []
    chroms = sorted(paths.keys(), key=_chrom_sort_key)
    for i, chrom in enumerate(chroms, 1):
        lg.write(f"  * reading {chrom} ({i}/{len(chroms)})")
        try:
            df_nom = pd.read_parquet(paths[chrom], columns=wanted_cols, engine="pyarrow")
        except pa.lib.ArrowInvalid:
            if columns is not None:
                raise
            schema_cols = set(pa.parquet.ParquetFile(paths[chrom]).schema.names)
            present = [c for c in default_cols if c in schema_cols]
            missing_required = {c for c in ["phenotype_id", "variant_id", "pval_nominal"] if c not in schema_cols}
            if missing_required:
                missing = ", ".join(sorted(missing_required))
                raise ValueError(f"Nominal file {paths[chrom]} is missing required columns: {missing}")
            df_nom = pd.read_parquet(paths[chrom], columns=present, engine="pyarrow")

        # Early filter: keep only relevant phenotypes
        if grouped:
            df_nom["group_id"] = df_nom["phenotype_id"].map(group_s)
            df_nom = df_nom[df_nom["group_id"].isin(target_groups)]
            if df_nom.empty:
                continue
            # apply phenotype->group threshold
            thr = df_nom["group_id"].map(threshold_lookup)
        else:
            df_nom = df_nom[df_nom["phenotype_id"].isin(target_phens)]
            if df_nom.empty:
                continue
            thr = df_nom["phenotype_id"].map(threshold_lookup)

        # Thresholding (use < to include exact boundary)
        mask = df_nom["pval_nominal"] < thr
        if mask.any():
            out.append(df_nom.loc[mask].copy())

    if not out:
        lg.write("  * No nominal pairs pass thresholds; returning empty DataFrame.")
        return pd.DataFrame()

    signif = pd.concat(out, axis=0, ignore_index=True)

    # Join back the q-values / thresholds for convenience
    if grouped:
        signif = signif.merge(
            keep.rename(columns={"pval_nominal_threshold": "pval_nominal_threshold_group"}),
            left_on="group_id", right_index=True, how="left"
        )
    else:
        signif = signif.merge(
            keep.rename(columns={"pval_nominal_threshold": "pval_nominal_threshold_pheno"}),
            left_on="phenotype_id", right_index=True, how="left"
        )

    lg.write(f"  * Final significant pairs: {len(signif):,}")
    return signif


def annotate_ancestry_difference(
    nominal_df: pd.DataFrame,
    n_ancestries: int = 2,
) -> pd.DataFrame:
    """
    Add columns testing H0: beta_anc_i = beta_anc_j for all ancestry pairs.

    Expects columns ``slope_gxh_anc{k}`` and ``slope_se_gxh_anc{k}`` for
    k in 0..n_ancestries-1 (produced by ``map_nominal`` with
    ``ancestry_model='interaction'``).

    New columns per pair (i, j):
      - ``beta_diff_anc{i}_anc{j}``
      - ``se_diff_anc{i}_anc{j}``
      - ``z_diff_anc{i}_anc{j}``
      - ``pval_diff_anc{i}_anc{j}``

    Parameters
    ----------
    nominal_df : DataFrame
        Output of ``map_nominal`` with ancestry interaction columns.
    n_ancestries : int
        Number of ancestry channels (K). Only K-1 columns are expected to be
        non-NaN (the last ancestry is dropped for identifiability).

    Returns
    -------
    DataFrame
        Input with new difference-of-effects columns appended.
    """
    import numpy as np
    from ..stats import test_beta_difference

    df = nominal_df.copy()
    for i in range(n_ancestries):
        for j in range(i + 1, n_ancestries):
            col_b_i = f"slope_gxh_anc{i}"
            col_se_i = f"slope_se_gxh_anc{i}"
            col_b_j = f"slope_gxh_anc{j}"
            col_se_j = f"slope_se_gxh_anc{j}"

            if not all(c in df.columns for c in (col_b_i, col_se_i, col_b_j, col_se_j)):
                continue

            z, pval = test_beta_difference(
                df[col_b_i].values, df[col_se_i].values,
                df[col_b_j].values, df[col_se_j].values,
            )
            suffix = f"anc{i}_anc{j}"
            df[f"beta_diff_{suffix}"] = (df[col_b_i] - df[col_b_j]).astype(np.float32)
            df[f"se_diff_{suffix}"] = np.sqrt(
                df[col_se_i] ** 2 + df[col_se_j] ** 2
            ).astype(np.float32)
            df[f"z_diff_{suffix}"] = z.astype(np.float32)
            df[f"pval_diff_{suffix}"] = pval.astype(np.float32)
    return df


def annotate_with_susie(
    nominal_df: pd.DataFrame,
    susie_summary: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add PIP and credible set membership columns to nominal results.

    Parameters
    ----------
    nominal_df : DataFrame
        Output of ``map_nominal``. Must have ``phenotype_id`` and ``variant_id``.
    susie_summary : DataFrame
        Output of ``susie.map()``. Must have ``phenotype_id``, ``variant_id``,
        ``pip``, and ``cs_id``.

    Returns
    -------
    DataFrame
        Input with ``susie_pip`` and ``susie_cs`` columns appended.
    """
    df = nominal_df.copy()

    if susie_summary.empty:
        df['susie_pip'] = float('nan')
        df['susie_cs'] = None
        return df

    # Build lookup: (phenotype_id, variant_id) -> (pip, cs_id)
    pip_map = susie_summary.set_index(['phenotype_id', 'variant_id'])['pip']
    cs_map = susie_summary.set_index(['phenotype_id', 'variant_id'])['cs_id']

    key = list(zip(df['phenotype_id'], df['variant_id']))
    df['susie_pip'] = [pip_map.get(k, float('nan')) for k in key]
    df['susie_cs'] = [cs_map.get(k, None) for k in key]
    return df


def annotate_with_coloc(
    perm_df: pd.DataFrame,
    coloc_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add COLOC H4 posterior probability to permutation-level results.

    Parameters
    ----------
    perm_df : DataFrame
        Output of ``map_permutations``. Must have ``phenotype_id`` as index or column.
    coloc_df : DataFrame
        Output of ``coloc.run_pairs()``. Must have ``pp_h4_abf`` column,
        indexed by phenotype.

    Returns
    -------
    DataFrame
        Input with ``coloc_pp_h4`` column appended.
    """
    df = perm_df.copy()
    if coloc_df.empty:
        df['coloc_pp_h4'] = float('nan')
        return df

    if 'phenotype_id' in df.columns:
        df['coloc_pp_h4'] = df['phenotype_id'].map(coloc_df['pp_h4_abf']).values
    else:
        # phenotype_id is the index
        df['coloc_pp_h4'] = df.index.map(coloc_df['pp_h4_abf']).values
    return df
