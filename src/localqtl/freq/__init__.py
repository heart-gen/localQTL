"""
External allele frequencies and ancestry-aware frequency QC.

- :class:`FrequencyTable`: standardised ``(chrom, pos, ref, alt)``-keyed
  ``af_/ac_/an_<group>`` table (GRCh38) with allele-aware :meth:`lookup`.
- Importers: :func:`from_gnomad_vcf`, :func:`from_alfa_vcf` (tabix via the
  optional ``annotate`` extra) and :func:`from_bcftools_query_tsv` (no deps).
- In-sample per-ancestry frequencies: :func:`ancestry_allele_frequencies`.
- QC: :func:`compare_allele_frequencies`, :func:`summarize_af_concordance`,
  :func:`annotate_with_frequency`.
- Calibration: :func:`calibration_by_frequency`,
  :func:`calibration_by_frequency_dosage`; :func:`run_frequency_benchmark`.

Importing this package does not import ``pysam``.
"""
from .table import (
    GRCH38_REFSEQ_TO_CHROM,
    CHROM_TO_GRCH38_REFSEQ,
    FrequencyTable,
    read_frequency_table,
    resolve_ancestry_map,
    normalize_chrom,
    ensure_variant_keys,
)
from .insample import ancestry_allele_frequencies, ancestry_labels, anc_columns_to_labels
from .gnomad import GNOMAD_V4_GROUPS, GNOMAD_V4_EXTRA_FIELDS, from_gnomad_vcf, inspect_gnomad_header
from .alfa import ALFA_POPULATIONS, ALFA_POPULATION_NAMES, ALFA_URL, from_alfa_vcf
from .bcftools import from_bcftools_query_tsv, regions_to_bed, regions_to_refseq_bed
from ._vcf import pysam_available, regions_from_variant_df, regions_from_phenotype_pos
from .qc import compare_allele_frequencies, summarize_af_concordance, annotate_with_frequency
from .calibration import (
    genomic_control_lambda,
    qq_table,
    calibration_by_frequency,
    calibration_by_frequency_dosage,
)
from .benchmark import BenchmarkResult, run_frequency_benchmark

__all__ = [
    "GRCH38_REFSEQ_TO_CHROM", "CHROM_TO_GRCH38_REFSEQ",
    "FrequencyTable", "read_frequency_table", "resolve_ancestry_map",
    "normalize_chrom", "ensure_variant_keys",
    "ancestry_allele_frequencies", "ancestry_labels", "anc_columns_to_labels",
    "GNOMAD_V4_GROUPS", "GNOMAD_V4_EXTRA_FIELDS", "from_gnomad_vcf", "inspect_gnomad_header",
    "ALFA_POPULATIONS", "ALFA_POPULATION_NAMES", "ALFA_URL", "from_alfa_vcf",
    "from_bcftools_query_tsv", "regions_to_bed", "regions_to_refseq_bed",
    "pysam_available", "regions_from_variant_df", "regions_from_phenotype_pos",
    "compare_allele_frequencies", "summarize_af_concordance", "annotate_with_frequency",
    "genomic_control_lambda", "qq_table", "calibration_by_frequency",
    "calibration_by_frequency_dosage",
    "BenchmarkResult", "run_frequency_benchmark",
]
