# localQTL

localQTL is a Python library for cis-QTL mapping that keeps the familiar data model
from [tensorQTL](https://github.com/broadinstitute/tensorqtl) while adding GPU-first
execution paths, flexible genotype loaders, and support for local ancestry-aware
analyses. It is designed for researchers who need an end-to-end toolkit for
pre-processing, running nominal and permutation scans, and summarising independent
cis signals on large cohorts.

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

## Installation

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
expected by tensorQTL, so existing preprocessing pipelines can be reused.

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
    window=1_000_000,      # ±1 Mb cis window
    maf_threshold=0.01,    # filter on in-sample MAF
    device="auto",         # picks CUDA when available, otherwise CPU
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

nominal_df = mapper.map_nominal(nperm=1_000)
perm_df = mapper.map_permutations(nperm=1_000, beta_approx=True)
perm_df = mapper.calculate_qvalues(perm_df, fdr=0.05)
lead_df = mapper.map_independent(cis_df=perm_df, fdr=0.05)
```

## Project structure

```
src/localqtl/
├── cis/                   # Nominal, permutation, and independent cis-QTL mappers
├── genotypeio.py          # PLINK/PGEN readers and windowed input generators
├── haplotypeio.py         # RFMix haplotype loader and ancestry-aware generators
├── phenotypeio.py         # Utilities for tensorQTL-compatible BED phenotypes
├── preproc.py             # Genotype imputation, filtering, and QC helpers
├── regression_kernels.py  # Batched regression routines executed on CPU or GPU
└── utils.py               # Logging helpers and device utilities
```

## Testing

Run the test suite with:

```bash
poetry run pytest
```

This exercises the core cis-QTL mapping routines using small synthetic datasets.
