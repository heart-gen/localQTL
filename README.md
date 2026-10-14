# localQTL

localQTL is a **pure-Python** library for local-ancestry-aware xQTL mapping that
lets researchers run end-to-end analyses on large cohorts **without any R/rpy2
dependencies**. It preserves the familiar 
[tensorQTL](https://github.com/broadinstitute/tensorqtl) data model with **GPU-first
execution paths**, flexible genotype loaders, and streaming outputs for large-scale
workflows, while adding ancestry-aware use cases.

## Features

- **GPU-accelerated workflows** powered by PyTorch, with automatic fallbacks to CPU
  execution when CUDA is unavailable.
- **Modular cis-QTL mapping API** exposed through functional helpers such as
  `map_nominal`, `map_permutations`, and `map_independent`, or via the convenience
  wrapper `CisMapper`.
- **Multiple genotype backends** including PLINK (BED/BIM/FAM), PLINK 2 (PGEN/PSAM/PVAR),
  and BED/Parquet inputs, with helpers to stream data in manageable windows.
- **Local ancestry integration** by pairing genotype windows with haplotype panels
  produced by RFMix through `RFMixReader` and `InputGeneratorCisWithHaps`.
- **Parquet streaming sinks** that make it easy to materialise association statistics
  without loading the entire result set in memory.
- **Pure-Python statistics (no R/rpy2 required)**: tensorQTL's `rfunc` calls have been refactored to scipy.stats for p-values and q-values are computed with the Python port of Storey's `qvalue` (`py-qvalue`).
- **Fine-mapping and colocalization helpers** via `localqtl.susie`,
  `localqtl.coloc`, and cis post-processing utilities that annotate nominal or
  permutation outputs with credible sets and COLOC posterior support.

## Installation

Install the latest release from PyPI:

```bash
pip install localqtl            # CPU / PyTorch only
pip install "localqtl[gpu]"     # + CuPy and cuDF (CUDA 12 wheels)
```

The core depends on PyTorch, numpy/scipy/pandas, pgenlib and rfmix-reader.
The `gpu` extra adds the CuPy and cuDF (RAPIDS) array and dataframe backends;
without it every helper falls back to numpy and pandas, and PyTorch still uses
the GPU for the regressions when one is visible.

The project uses [Poetry](https://python-poetry.org/) for dependency management.
Clone the repository and install the package into a virtual environment:

```bash
poetry install                  # CPU
poetry install --extras gpu     # with the CUDA backends
```

If you prefer `pip`, you can install the library in editable mode:

```bash
pip install -e ".[gpu]"
```

> **Note:** the `gpu` extra pins CUDA 12 wheels (`cupy-cuda12x`, `cudf-cu12`,
> `dask-cudf-cu12`); match them to the CUDA toolkit on your system.

## Quickstart

Below is a minimal example that runs a nominal cis-QTL scan against PLINK-formatted
genotypes and BED-formatted phenotypes. The example mirrors the data layout
expected by tensorQTL, so **existing preprocessing pipelines can be reused**.

### Standard mapping (tensorQTL equivalent)
```python
from localqtl import PlinkReader, read_phenotype_bed
from localqtl.cis import map_nominal

# Load genotypes and variant metadata
plink = PlinkReader("data/genotypes")
genotype_df = plink.load_genotypes()
variant_df = plink.bim.set_index("snp")[["chrom", "pos"]]

# Load phenotypes (BED-style) and their genomic coordinates
phenotype_df, phenotype_pos_df = read_phenotype_bed("data/phenotypes.bed")

# Optional: load covariates as a DataFrame indexed by sample IDs
covariates_df = None

results = map_nominal(
    genotype_df=genotype_df,
    variant_df=variant_df,
    phenotype_df=phenotype_df,
    phenotype_pos_df=phenotype_pos_df,
    covariates_df=covariates_df,
    window=1_000_000,         # ±1 Mb cis window
    maf_threshold=0.01,       # filter on in-sample MAF
    device="auto",            # picks CUDA when available, otherwise CPU
    out_prefix="cis_nominal", # default prefix
    return_df=True            # default is False, parquet streamed sink
)

print(results.head())
```

### Ancestry-specific dosage model (posterior-aware)

Hard local-ancestry calls discard the uncertainty of the assignment (a
haplotype with P(AFR) = 0.65, P(EUR) = 0.35 is simply "AFR"). With
`ancestry_model="dosage"`, `map_nominal` instead estimates one regulatory
effect per ancestry from the expected ancestry-specific alternate-allele
dosage. For sample *i* with alternate-allele dosage `G_i` and expected
ancestry counts `L_ik = sum_h P(Z_ih = k)` (hard calls give 0/1/2):

```
D_ik = G_i * L_ik / 2                      (sum_k D_ik = G_i; phase unknown)
y_i  = sum_k beta_k D_ik + sum_{k<K} gamma_k L_ik + covariates + e_i
```

The `L` columns adjust for local-ancestry main effects. Output columns per
ancestry are `slope_anc{k}`, `slope_se_anc{k}`, `tstat_anc{k}`,
`pval_anc{k}`; `fstat_joint` / `pval_joint` test all `beta_k = 0` jointly;
`z_diff_anc{i}_anc{j}` / `pval_diff_anc{i}_anc{j}` test `beta_i = beta_j`
using the full covariance of the ancestry betas. The standard `slope`,
`slope_se`, `tstat`, `pval_nominal` columns carry the ancestry with the
strongest effect.

```python
panel = RFMixReader("data/rfmix/", source="fb", keep_posteriors=True,
                    cache_dir="data/rfmix/la_cache_fb/", use_posterior=True,
                    select_samples=list(genotype_df.columns))
dosage_df = mapper.map_nominal(ancestry_model="dosage")   # mapper built with haplotypes=panel
```

With a hard-call panel (`.msp.tsv`) the same model reduces to the classic
ancestry-specific genotype regression. Note that substituting posterior
expectations for the unknown haplotype ancestries does not propagate that
uncertainty into the standard errors; joint-likelihood or multiple-imputation
estimation is future work. `ancestry_model="dosage"` is currently available
for `map_nominal` only (no permutations, no `interaction_covariate`).

### Covariate interaction scans

`map_nominal`, `map_permutations`, `map_independent`, and `CisMapper` also
support sample-level genotype-by-covariate interaction models through the
`interaction_covariate` argument. Pass either a covariate-column name, a
`pandas.Series` indexed by sample ID, or a 1D array aligned to the sample
order.

```python
from localqtl.cis import map_nominal

results = map_nominal(
    genotype_df=genotype_df,
    variant_df=variant_df,
    phenotype_df=phenotype_df,
    phenotype_pos_df=phenotype_pos_df,
    covariates_df=covariates_df,
    interaction_covariate="age",
    device="auto",
    out_dir=None,
)

print(results[["variant_id", "pval_nominal", "pval_g", "pval_i", "pval_gi"]].head())
```

This emits the standard nominal columns plus canonical joint-model statistics
for `y ~ g + i + g:i`: `b_g`, `b_i`, `b_gi`, their standard errors, t-stats,
and p-values. `interaction_covariate` cannot be combined with
`ancestry_model="interaction"`.

For analyses that combine nominal scans, permutations, and independent signal
calling, the `CisMapper` class offers a thin object-oriented façade:

```python
from localqtl.cis import CisMapper

mapper = CisMapper(
    genotype_df=genotype_df,
    variant_df=variant_df,
    phenotype_df=phenotype_df,
    phenotype_pos_df=phenotype_pos_df,
    covariates_df=covariates_df,
    window=500_000,
    maf_threshold=0.01,
    out_dir="./",
    out_prefix="cis_nominal",
)

mapper.map_nominal(nperm=0)
perm_df = mapper.map_permutations(nperm=1_000, beta_approx=True)
perm_df = mapper.calculate_qvalues(perm_df, fdr=0.05)
lead_df = mapper.map_independent(cis_df=perm_df, fdr=0.05)
```

### Local ancestry-aware mapping

localQTL can incorporate local ancestry (RFMix, FLARE or haptools output read
by [rfmix-reader](https://github.com/heart-gen/rfmix_reader)) so that cis-xQTL
tests are performed with ancestry-aware genotype inputs. Pass a
**`LocalAncestryPanel`** as `haplotypes` to `CisMapper`; the ancestry-aware
input generator then holds one chromosome of ancestry counts in memory at a
time and maps every SNP to the RFMix segment that covers it, so memory does
not grow with the genome. (A plain `(loci, samples, ancestries)` array with a
`loci_df` is still accepted.)

```python
from localqtl import PlinkReader, read_phenotype_bed
from localqtl.haplotypeio import RFMixReader, open_panel
from localqtl.cis import CisMapper

plink = PlinkReader("data/genotypes")
genotype_df = plink.load_genotypes()  # samples as columns, variants as rows
variant_df = plink.bim.set_index("snp")[["chrom", "pos"]]

phenotype_df, phenotype_pos_df = read_phenotype_bed("data/phenotypes.bed")
covariates_df = None  # optional

# Local ancestry from RFMix (.msp.tsv segments), aligned to the genotype samples.
# cache_dir: parsed once into a Zarr cache, reopened instantly afterwards.
panel = RFMixReader(
    "data/rfmix/",                              # directory / file / path prefix of the RFMix output
    select_samples=list(genotype_df.columns),   # same samples, same order as the genotypes
    cache_dir="data/rfmix/la_cache/",
)
# or, for an existing cache:  panel = open_panel("data/rfmix/la_cache/", select_samples=...)
# or FLARE output:            panel = FlareReader("data/flare/", select_samples=...)

# For two ancestries the panel exposes one channel (the other is 2 - first);
# panel.ancestries, panel.chromosomes, panel.g_anc (global ancestry) are available.

# Keep RFMix's forward-backward posteriors (.fb.tsv) to serve posterior-expected
# ancestry counts  E[count_k] = sum_h P(Z_h = k)  instead of hard calls:
# panel = RFMixReader("data/rfmix/", source="fb", keep_posteriors=True,
#                     cache_dir="data/rfmix/la_cache_fb/", use_posterior=True,
#                     select_samples=list(genotype_df.columns))

# Ancestry-aware mapping
mapper = CisMapper(
    genotype_df=genotype_df,
    variant_df=variant_df,
    phenotype_df=phenotype_df,
    phenotype_pos_df=phenotype_pos_df,
    covariates_df=covariates_df,
    haplotypes=panel,      # <-- enable local ancestry-aware mode
    window=1_000_000,
    maf_threshold=0.01,
    device="auto",
    out_dir="./",
    out_prefix="cis_nominal",
)

# Run nominal scans and permutations as usual; ancestry-awareness is automatic
mapper.map_nominal(nperm=0)
perm_df = mapper.map_permutations(nperm=1_000, beta_approx=True)

# FDR using the pure-Python q-value port (no R/rpy2 needed)
perm_df = mapper.calculate_qvalues(perm_df, fdr=0.05)

# Identify conditionally independent signals
lead_df = mapper.map_independent(cis_df=perm_df, fdr=0.05)

print(lead_df.head())
```

### Fine-mapping and colocalization

The branch also adds SuSiE fine-mapping and COLOC colocalization helpers for
downstream interpretation of cis hits:

```python
from localqtl import coloc, susie

susie_summary = susie.map(
    genotype_df=genotype_df,
    variant_df=variant_df,
    phenotype_df=phenotype_df,
    phenotype_pos_df=phenotype_pos_df,
    covariates_df=covariates_df,
    device="auto",
)

coloc_df = coloc.run_pairs(
    genotype_df=genotype_df,
    variant_df=variant_df,
    phenotype1_df=phenotype_df,
    phenotype2_df=external_trait_df,
    phenotype_pos_df=phenotype_pos_df,
    covariates1_df=covariates_df,
    device="auto",
)
```

`susie.map()` returns a summary table of credible-set membership with per-variant
PIPs. `coloc.run_pairs()` returns one row per phenotype with posterior support
for hypotheses `H0` through `H4`, including shared-signal support in
`pp_h4_abf`.

To annotate downstream cis outputs:

```python
from localqtl.cis.postproc import annotate_with_coloc, annotate_with_susie

nominal_annotated = annotate_with_susie(results, susie_summary)
perm_annotated = annotate_with_coloc(perm_df, coloc_df)
```

**Input expectations**

* The panel exposes `(variants, samples, channels)` int8 ancestry counts, one chromosome at a time (`panel.chrom_counts(chrom)`); with ≥3 ancestries every ancestry is a channel, with 2 ancestries a single channel is used (the other is `2 - first`).
* Each genotype SNP is matched to the RFMix segment that covers it (`panel.locus_index(chrom, positions)`; `locus_method="nearest"` with `nearest_tolerance` for locus-level sources); SNPs no segment covers are dropped from the ancestry-aware model.
* Sample order in `genotype_df`, `phenotype_df`, and `covariates_df` should match `H` (reindex as shown).
* The cis-mapping helpers default to ``preload_haplotypes=True`` so the current chromosome's counts are staged as an int8 tensor on the requested device (GPU or CPU) and windows are sliced there. Override this flag when working under strict memory constraints.

## Testing

Run the test suite with:

```bash
poetry run pytest
```

This exercises the core cis-QTL mapping routines using small synthetic datasets.
