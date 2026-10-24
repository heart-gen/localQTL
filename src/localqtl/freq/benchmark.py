"""
End-to-end ancestry-frequency QC benchmark: compare in-sample per-ancestry
allele frequencies with a reference table, optionally drop discordant
variants, run the ancestry-specific dosage scan, and summarise p-value
calibration by frequency and ancestry (with a permutation-null companion).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from ..utils import SimpleLogger
from .calibration import DEFAULT_AF_BINS, calibration_by_frequency_dosage
from .insample import ancestry_allele_frequencies, ancestry_labels
from .qc import compare_allele_frequencies, summarize_af_concordance, annotate_with_frequency
from .table import FrequencyTable, resolve_ancestry_map

__all__ = ["BenchmarkResult", "run_frequency_benchmark"]


@dataclass
class BenchmarkResult:
    af_comparison: pd.DataFrame
    af_summary: pd.DataFrame
    nominal_df: pd.DataFrame
    calibration_df: pd.DataFrame
    null_df: Optional[pd.DataFrame] = None
    calibration_null_df: Optional[pd.DataFrame] = None
    n_variants_in: int = 0
    n_variants_kept: int = 0
    flags: dict = field(default_factory=dict)

    def write(self, out_dir: str, prefix: str = "freq_benchmark") -> dict[str, str]:
        """Write every table as Parquet (summaries as TSV) under ``out_dir``."""
        os.makedirs(out_dir, exist_ok=True)
        paths = {}
        for name, df, ext in (("af_comparison", self.af_comparison, "parquet"),
                              ("af_summary", self.af_summary, "tsv"),
                              ("nominal", self.nominal_df, "parquet"),
                              ("calibration", self.calibration_df, "tsv"),
                              ("null", self.null_df, "parquet"),
                              ("calibration_null", self.calibration_null_df, "tsv")):
            if df is None:
                continue
            p = os.path.join(out_dir, f"{prefix}.{name}.{ext}")
            if ext == "parquet":
                df.to_parquet(p)
            else:
                df.to_csv(p, sep="\t", index=False)
            paths[name] = p
        return paths


def _permute_samples(df: pd.DataFrame, perm: np.ndarray, axis: str) -> pd.DataFrame:
    """Permute sample labels so y (and covariates) are decoupled from genotypes."""
    if axis == "columns":
        out = df.copy()
        out.columns = df.columns.to_numpy()[perm]
        return out[df.columns]
    out = df.copy()
    out.index = df.index.to_numpy()[perm]
    return out.loc[df.index]


def run_frequency_benchmark(
        genotype_df: pd.DataFrame,
        variant_df: pd.DataFrame,
        phenotype_df: pd.DataFrame,
        phenotype_pos_df: pd.DataFrame,
        panel,
        freq_table: FrequencyTable,
        ancestry_map: Mapping[str, str],
        *,
        covariates_df: Optional[pd.DataFrame] = None,
        window: int = 1_000_000,
        maf_threshold: float = 0.0,
        min_ancestry_ac: float = 0.0,
        drop_discordant: bool = False,
        drop_flipped: bool = False,
        drop_ambiguous: bool = False,
        n_null: int = 1,
        seed: int = 0,
        device: str = "cpu",
        bins: Sequence[float] = DEFAULT_AF_BINS,
        calibration_by: str = "af_anc",
        min_an: float = 20.0,
        logger: Optional[SimpleLogger] = None,
        **map_nominal_kwargs,
) -> BenchmarkResult:
    """
    Run the ancestry-frequency QC benchmark.

    Steps: in-sample per-ancestry AF -> comparison with ``freq_table`` via
    ``ancestry_map`` -> optional variant exclusion -> dosage-model
    :func:`~localqtl.cis.map_nominal` -> reference-AF annotation ->
    :func:`~localqtl.freq.calibration.calibration_by_frequency_dosage`.  The
    null repeats the scan ``n_null`` times with sample labels permuted
    (phenotypes and covariates together) so that calibration under
    no association can be compared bin by bin.

    ``calibration_by`` is ``"af_anc"`` (in-sample), ``"af_ref_anc"`` (reference
    frequency of the mapped group) or ``"ma_count_anc"`` (pass integer ``bins``).
    """
    from ..cis.nominal import map_nominal

    logger = logger or SimpleLogger(verbose=False)
    labels = ancestry_labels(panel)
    amap = resolve_ancestry_map(ancestry_map, labels, freq_table)

    insample = ancestry_allele_frequencies(genotype_df, variant_df, panel)
    cmp_df = compare_allele_frequencies(insample, freq_table, amap, min_an=min_an)
    summary = summarize_af_concordance(cmp_df)

    drop = np.zeros(len(cmp_df), dtype=bool)
    if drop_discordant:
        drop |= cmp_df["discordant"].to_numpy(dtype=bool)
    if drop_flipped:
        drop |= cmp_df["allele_flipped"].to_numpy(dtype=bool)
    if drop_ambiguous:
        drop |= cmp_df["strand_ambiguous"].to_numpy(dtype=bool)
    keep_ids = cmp_df.index[~drop]
    g_use = genotype_df.loc[keep_ids]
    v_use = variant_df.loc[keep_ids]
    logger.write(f"  * frequency QC: keeping {len(keep_ids):,} / {len(cmp_df):,} variants")

    common = dict(genotype_df=g_use, variant_df=v_use, phenotype_pos_df=phenotype_pos_df,
                  haplotypes=panel, window=window, maf_threshold=maf_threshold, device=device,
                  out_dir=None, nperm=None, ancestry_model="dosage",
                  min_ancestry_ac=min_ancestry_ac, logger=logger, **map_nominal_kwargs)
    nominal = map_nominal(phenotype_df=phenotype_df, covariates_df=covariates_df, **common)
    nominal = annotate_with_frequency(nominal, v_use, freq_table, ancestry_map=amap,
                                      ancestry_labels=labels)
    fold = calibration_by != "ma_count_anc"
    calib = calibration_by_frequency_dosage(nominal, ancestry_labels=labels, by=calibration_by,
                                            bins=bins, fold=fold)

    null_df = calib_null = None
    if n_null and n_null > 0:
        rng = np.random.default_rng(seed)
        parts = []
        for i in range(int(n_null)):
            perm = rng.permutation(phenotype_df.shape[1])
            y_perm = _permute_samples(phenotype_df, perm, "columns")
            c_perm = _permute_samples(covariates_df, perm, "index") if covariates_df is not None else None
            res = map_nominal(phenotype_df=y_perm, covariates_df=c_perm, **common)
            res.insert(0, "null_replicate", i)
            parts.append(res)
        null_df = pd.concat(parts, ignore_index=True)
        null_df = annotate_with_frequency(null_df, v_use, freq_table, ancestry_map=amap,
                                          ancestry_labels=labels)
        calib_null = calibration_by_frequency_dosage(null_df, ancestry_labels=labels,
                                                     by=calibration_by, bins=bins, fold=fold)

    flags = {"drop_discordant": drop_discordant, "drop_flipped": drop_flipped,
             "drop_ambiguous": drop_ambiguous, "n_dropped": int(drop.sum()),
             "ancestry_map": dict(amap), "freq_source": freq_table.source}
    return BenchmarkResult(af_comparison=cmp_df, af_summary=summary, nominal_df=nominal,
                           calibration_df=calib, null_df=null_df, calibration_null_df=calib_null,
                           n_variants_in=len(cmp_df), n_variants_kept=len(keep_ids), flags=flags)
