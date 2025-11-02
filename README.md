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

## Installation

Install the latest release from PyPI:

```bash
pip install localqtl
```

The project uses [Poetry](https://python-poetry.org/) for dependency management.
Clone the repository and install the package into a virtual environment:

```bash
poetry install
```

If you prefer `pip`, you can install the library in editable mode after exporting
Poetry's dependency specification:

```bash
pip install -e .
```

> **Note:** GPU acceleration relies on PyTorch, CuPy, and cuDF. Make sure you use
> versions that match the CUDA toolkit available on your system. The versions in
> `pyproject.toml` target CUDA 12.

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
)

mapper.map_nominal(nperm=1_000)
perm_df = mapper.map_permutations(nperm=1_000, beta_approx=True)
perm_df = mapper.calculate_qvalues(perm_df, fdr=0.05)
lead_df = mapper.map_independent(cis_df=perm_df, fdr=0.05)
```

### Local ancestry-aware mapping

localQTL can incorporate local ancestry (e.g., from RFMix) so that cis-xQTL tests are performed with ancestry-aware genotype inputs. To enable this, pass **`haplotypes`** (per-variant × per-sample × per-ancestry dosages) and **`loci_df`** (variant positions corresponding to the haplotype tensor) to `CisMapper`. When these are provided, `CisMapper` switches to the ancestry-aware input generator under the hood.

```python
from localqtl import PlinkReader, read_phenotype_bed
from localqtl.haplotypeio import RFMixReader
from localqtl.cis import CisMapper

plink = PlinkReader("data/genotypes")
genotype_df = plink.load_genotypes()  # samples as columns, variants as rows
variant_df = plink.bim.set_index("snp")[["chrom", "pos"]]

phenotype_df, phenotype_pos_df = read_phenotype_bed("data/phenotypes.bed")
covariates_df = None  # optional

# Local ancestry from RFMix
rfmix = RFMixReader(
    prefix_path="data/rfmix/prefix",   # directory with per-chrom outputs + fb.tsv
    binary_path="data/rfmix",          # where prebuilt binaries live (if used)
    verbose=True
)

# Materialize ancestry haplotypes into memory (NumPy array)
# Shape: (variants, samples, ancestries) for >2 ancestries.
# For 2 ancestries, the reader exposes the ancestry channel selected internally.
H = rfmix.load_haplotypes()
loci_df = rfmix.loci_df  # DataFrame with ['chrom','pos', 'ancestry', 'hap', 'index'] (indexed by 'hap')

# Align samples to the RFMix order (recommended)
# Keep only intersection and put everything in the same order as rfmix.sample_ids.
keep = [sid for sid in rfmix.sample_ids if sid in genotype_df.columns]
genotype_df = genotype_df[keep]
phenotype_df = phenotype_df[keep]
if covariates_df is not None:
    covariates_df = covariates_df.loc[keep]

# (Optional) Ensure chromosome dtype matches between variant_df and loci_df
variant_df["chrom"] = variant_df["chrom"].astype(str)
loci_df = loci_df.copy()
loci_df["chrom"] = loci_df["chrom"].astype(str)

# Ancestry-aware mapping
mapper = CisMapper(
    genotype_df=genotype_df,
    variant_df=variant_df,
    phenotype_df=phenotype_df,
    phenotype_pos_df=phenotype_pos_df,
    covariates_df=covariates_df,
    haplotypes=H,          # <-- enable local ancestry-aware mode
    loci_df=loci_df,       # <-- positions that align to H
    window=1_000_000,
    device="auto",
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

**Input expectations**

* `H` (from `RFMixReader.load_haplotypes()`) is ancestry dosages with shape `(variants, samples, ancestries)` for ≥3 ancestries; for 2 ancestries the reader exposes the configured channel.
* `loci_df` rows correspond 1:1 to `H`'s and be joinable (by `chrom`/`pos`) to `variant_df` used for genotypes.
* Sample order in `genotype_df`, `phenotype_df`, and `covariates_df` should match `rfmix.sample_ids` (reindex as shown).
* The cis-mapping helpers default to ``preload_haplotypes=True`` so ancestry blocks are staged as contiguous tensors on the requested device (GPU or CPU). Override this flag when working under strict memory constraints.

## Testing

Run the test suite with:

```bash
poetry run pytest
```

This exercises the core cis-QTL mapping routines using small synthetic datasets.
