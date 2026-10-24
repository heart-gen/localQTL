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
- **External validation of lead variants**: AlphaGenome Atlas molecular-effect
  scores and IGVF Catalog regulatory links (`localqtl.annotate`), as live
  clients or from exported tables.
- **tensorQTL feature parity**: trans-QTL mapping (`localqtl.trans`), eigenMT
  (`localqtl.eigenmt`), multi-term `interaction_df` scans with top-association
  output, phenotype-specific `paired_covariate_df`, aFC / replication / gene
  annotation (`localqtl.post`), tabix VCF readers, chunked pgen processing and a
  `localqtl` command line with tensorQTL's modes and output file names.

## Installation

Install the latest release from PyPI:

```bash
pip install localqtl              # CPU / PyTorch only
pip install "localqtl[gpu]"       # + CuPy and cuDF (CUDA 12 wheels)
pip install "localqtl[annotate]"       # + pysam, for tabix access to gnomAD / ALFA frequency VCFs
pip install "localqtl[annotate-live]"  # + requests, for the IGVF Catalog REST client
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

Release notes are in [CHANGELOG.md](CHANGELOG.md).

### Upgrading from 0.1

Version 0.2.0 changes numeric output in three places; re-run affected scans
rather than mixing results across versions.

- **Standard errors and degrees of freedom.** The batched kernels now count the
  intercept in the degrees of freedom (`dof = n - k_eff - p`, with `k_eff`
  including the centring step). Standard errors from 0.1 were too small by
  `sqrt((n - k - 1) / (n - k - 2))`; t-statistics and p-values shift accordingly.
- **Runs without covariates** now fit an intercept (every matrix is centred).
  In 0.1 they were fitted through the origin.
- **Interaction scans** (`interaction_covariate`, and the new `interaction_df`)
  now equal a full OLS fit of `y ~ 1 + g + i + g:i + C`. In 0.1 the `g:i`
  column was built from the residualized genotype, which biased the
  interaction coefficients whenever covariates were given.

Other things to know: `variant_df` from the PLINK readers carries `ref`/`alt`
with `alt` the counted allele; the dosage-model Parquet files have additional
per-ancestry count columns after the pairwise-difference block (read columns by
name); `read_phenotype_bed` no longer lowercases sample IDs; `ancestry_model`
must be one of `main`, `interaction`, `dosage`. The command line writes
tensorQTL column names (`true_df`, `b_gi_se`, ...) by default; the Python API
keeps `true_dof`, `se_gi`, ... and `localqtl.compat` converts between the two.

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
strongest effect. Each row also reports the in-sample per-ancestry allele
statistics `an_anc{k}` (allele number), `ac_anc{k}` (count of the alt allele
on ancestry-*k* haplotypes), `af_anc{k}` and `ma_count_anc{k}` (minor-allele
count within the ancestry; fractional for posterior-expected counts). Pass
`min_ancestry_ac=<count>` to mask (`NaN`) the effect estimates and pairwise
differences of ancestries with fewer minor alleles than that in-sample; the
variant row, pooled columns and `fstat_joint` are kept.

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

`map_nominal`, `map_permutations`, `map_independent`, and `CisMapper` support
genotype-by-covariate interaction models. `interaction_df` (tensorQTL's
argument: a DataFrame of samples x terms, or a Series) fits
`y ~ g + i_1..i_k + g:i_1..g:i_k`; the legacy `interaction_covariate` takes one
z-scored term (column name, Series or array).

```python
from localqtl.cis import map_nominal

results = map_nominal(
    genotype_df=genotype_df,
    variant_df=variant_df,
    phenotype_df=phenotype_df,
    phenotype_pos_df=phenotype_pos_df,
    covariates_df=covariates_df,
    interaction_df=covariates_df[["age"]],   # one or several terms
    maf_threshold_interaction=0.05,          # MAF >= 5 % in both halves of the term
    run_eigenmt=True,                        # tests_emt / pval_emt / pval_adj_bh on the top hits
    device="auto",
    out_dir=None,
)
top_df = results.attrs["top_assoc"]          # one row per phenotype (written to
                                             # <out_prefix>.top_assoc.txt.gz with out_dir)
print(results[["variant_id", "pval_nominal", "pval_g", "pval_i", "pval_gi"]].head())
```

With one term the output carries `b_g`, `b_i`, `b_gi` plus their standard
errors (`se_*`), t-statistics and p-values; with several terms the variable
names are substituted (`b_age`, `pval_g-age`, ...). The `slope` /
`pval_nominal` columns report the strongest `g:i` term. The `i` and `g:i`
columns are residualized against the covariates like `g` (Frisch–Waugh), so the
estimates equal a full OLS fit of `y ~ 1 + g + i + g:i + C`. Pass
`write_stats=False` (tensorQTL `--best_only`) to keep only the top
associations. `interaction_df` cannot be combined with
`ancestry_model="interaction"`; with a local-ancestry panel the ancestry counts
enter as main effects.

Phenotype-specific covariates (one extra covariate per phenotype, e.g. a local
PEER factor) are passed as `paired_covariate_df` (phenotypes x samples) to
`map_nominal` and `map_permutations`; phenotypes absent from it use the shared
covariates only.

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

### Allele metadata

`PgenReader.variant_df` and `PlinkReader.variant_df` (and `load_genotypes`)
now carry `chrom, pos, ref, alt`, where **`alt` is the counted (dosage)
allele**: pgenlib reports ALT counts, and for PLINK 1 files the `2 - bed` flip
makes the dosage count `.bim` column 5 (A1, which is ALT for
`plink2 --make-bed` / `plink --keep-allele-order`). The `af` column in every
result is the in-sample frequency of that counted allele. Variant tables that
only have `chrom`/`pos` keep working; `genotypeio.alleles_from_variant_ids`
parses GTEx-style `chr1_123_A_G_b38` IDs when needed.

### Ancestry-aware allele-frequency QC (gnomAD / ALFA)

`localqtl.freq` compares in-sample (per-ancestry) allele frequencies with
external references and summarises p-value calibration by frequency bin and
ancestry. References are held in a `FrequencyTable` keyed by
`(chrom, pos, ref, alt)` on GRCh38 with `af_<group>`, `ac_<group>`,
`an_<group>` columns, and are joined allele-aware (swapped ref/alt rows are
re-oriented; A/T and C/G sites are flagged as strand-ambiguous).

```python
from localqtl import freq

# gnomAD v4.1 sites VCFs (tabix; reads only the regions around your variants;
# the gnomAD local-ancestry tract AF fields are detected from the header)
gnomad = freq.from_gnomad_vcf("gnomad.genomes.v4.1.sites.chr*.vcf.bgz",
                              variant_df=variant_df, groups=("afr", "nfe"))
# NCBI ALFA (one GRCh38 VCF; populations are BioSample columns)
alfa = freq.from_alfa_vcf("freq.vcf.gz", variant_df=variant_df, populations=("AFR", "AFA", "EUR"))
# or, without pysam, read `bcftools query -H` output (recipes in localqtl.freq.bcftools)
table = freq.from_bcftools_query_tsv("gnomad.chr22.freq.tsv")
table.write("gnomad_chr22.freq.parquet")

# panel labels -> reference groups: always explicit, no defaults
ancestry_map = {"AFR": "afr", "EUR": "nfe"}

insample = freq.ancestry_allele_frequencies(genotype_df, variant_df, panel)
cmp_df = freq.compare_allele_frequencies(insample, gnomad, ancestry_map)
summary = freq.summarize_af_concordance(cmp_df)          # r, |diff|, % discordant per MAF bin

dosage_df = freq.annotate_with_frequency(dosage_df, variant_df, gnomad,
                                         ancestry_map=ancestry_map, ancestry_labels=list(panel.ancestries))
calib = freq.calibration_by_frequency_dosage(dosage_df, ancestry_labels=list(panel.ancestries),
                                             by="af_ref_anc")  # lambda_GC, frac p<0.05, KS per bin

# everything at once, with a permutation null for comparison
bench = freq.run_frequency_benchmark(genotype_df, variant_df, phenotype_df, phenotype_pos_df,
                                     panel, gnomad, ancestry_map, drop_discordant=True, n_null=1)
bench.write("qc/")
```

Only GRCh38 tables are accepted; strand flips are flagged, never corrected. The
gnomAD local-ancestry-informed release covers bi-allelic SNPs with MAF > 0.1 %
in the African/African American and Admixed American groups.

### trans-QTL mapping, eigenMT and post-processing (tensorQTL parity)

`localqtl.trans` mirrors tensorQTL's `trans` module: `map_trans` (sparse or
dense output, optional single `interaction_s`), `filter_cis`,
`map_trans_permutations` (genome-wide null of shuffled normal phenotypes, per
chromosome with `chr_s`) and `apply_permutations`. A `LocalAncestryPanel`
passed as `haplotypes` (with `variant_df`) adds each variant's local-ancestry
counts as main-effect covariates.

```python
from localqtl import trans, eigenmt, post

pairs_df = trans.map_trans(genotype_df, phenotype_df, covariates_df,
                           haplotypes=panel, variant_df=variant_df,
                           pval_threshold=1e-5, maf_threshold=0.05, return_r2=True)
pairs_df = trans.filter_cis(pairs_df, phenotype_pos_df, variant_df, window=5_000_000)
null = trans.map_trans_permutations(genotype_df, covariates_df, nperms=10_000, seed=1)
pairs_df = trans.apply_permutations(null, pairs_df)        # pval_perm, pval_beta

tests_emt = eigenmt.run_eigenmt(genotype_df, variant_df, phenotype_df, phenotype_pos_df)
afc_df = post.calculate_afc(assoc_df, counts_df, genotype_df, covariates_df=covariates_df)
pi1, rep_df = post.calculate_replication(cis_df.set_index("phenotype_id"), genotype_df2,
                                         phenotype_df2, covariates_df2, compute_pi1=True)
annotated = post.annotate_genes(cis_df.set_index("phenotype_id"), "gencode.v39.genes.gtf")
```

Other parity helpers: `genotypeio.get_vcf_region` / `get_vcf_variants`
(tabix VCF slices through pysam or the `tabix` binary),
`pgen.generate_paired_chunks` (genotype/phenotype chunks for cohorts whose
genotypes do not fit in memory), and `localqtl.compat` to translate column
names (`true_dof` <-> `true_df`, `se_gi` <-> `b_gi_se`, ...).

### Command line

`python -m localqtl` (or the `localqtl` console script) accepts tensorQTL's
arguments and writes tensorQTL's file names, plus local-ancestry options:

```bash
localqtl geno pheno.bed.gz prefix --mode cis --covariates covariates.txt --permutations 10000
localqtl geno pheno.bed.gz prefix --mode cis_nominal --covariates covariates.txt \
        --interaction age.txt --maf_threshold_interaction 0.05 --cis_output prefix.cis_qtl.txt.gz
localqtl geno pheno.bed.gz prefix --mode cis_independent --cis_output prefix.cis_qtl.txt.gz
localqtl geno pheno.bed.gz prefix --mode trans --pval_threshold 1e-5 --return_r2
# local ancestry (RFMix prefix, FLARE prefix or an rfmix_reader Zarr cache)
localqtl geno pheno.bed.gz prefix --mode cis_nominal --haplotypes rfmix/out --la_source msp \
        --ancestry_model dosage --min_ancestry_ac 10
```

Modes: `cis` (`<prefix>.cis_qtl.txt.gz`), `cis_nominal`
(`<prefix>.cis_qtl_pairs.<chr>.parquet`, `<prefix>.cis_qtl_top_assoc.txt.gz`
for interaction scans, `<prefix>.cis_qtl.signif_pairs.parquet` with
`--cis_output`), `cis_independent` (`<prefix>.cis_independent_qtl.txt.gz`),
`cis_susie` / `trans_susie` (`<prefix>.SuSiE_summary.parquet`,
`<prefix>.SuSiE.pickle`) and `trans` (`<prefix>.trans_qtl_pairs.parquet` or
`.txt.gz`; dense `--return_dense` outputs). `--chunk_size N|chr` streams PLINK 2
genotypes in chunks. Output columns use tensorQTL's spelling unless
`--localqtl_names` is given; `--device`, `--perm_chunk`, `--haplotypes`,
`--la_format`, `--la_source`, `--la_cache_dir`, `--use_posterior`,
`--ancestry_model` and `--min_ancestry_ac` are localQTL additions.

### External validation: AlphaGenome Atlas and IGVF Catalog

`localqtl.annotate` compares lead xQTL variants with external evidence. Both
clients work live or from files, so a cached table reproduces an analysis
without network access or an API key.

**AlphaGenome Atlas** supplies predicted molecular effects per variant and
gene. The live client needs the SDK (`pip install alphagenome`, non-commercial
licence) and an API key in `$ALPHAGENOME_API_KEY`.

```python
from localqtl.annotate import alphagenome as ag

scores = ag.score_lead_variants(lead_df, ontology_terms=["UBERON:0001873"],  # caudate nucleus
                                scorers=["RNA_SEQ"], cache_path="atlas_scores.parquet")
scores = ag.read_atlas_scores("atlas_scores.parquet")        # offline rerun

annotated = ag.annotate_with_alphagenome(dosage_df, scores, variant_df=variant_df)
```

Variants are matched on `(chrom, pos, ref, alt)` and genes without their
version suffix. Each row gains `ag_raw_score`, `ag_quantile_score`,
`ag_biosample`, `ag_n_tracks` and `ag_sign_concordant`, the last comparing the
sign of the measured `slope` (the strongest ancestry in the dosage model) with
the predicted effect.

**IGVF Catalog** supplies independently measured regulatory links: variant to
gene (eQTL, spliceQTL, CRISPR screens, Variant-EFFECTS) and element to gene
(ENCODE-rE2G, scE2G, Perturb-seq, CRISPR screens).

```python
from localqtl.annotate import igvf

links = igvf.query_variant_gene_links(lead_df, variant_df=variant_df,
                                      methods=["eQTL"], cache_path="igvf_links.parquet")
elements = igvf.query_element_gene_links(regions_df, methods=["ENCODE-rE2G"])
elements = igvf.read_bedpe_links("ENCODE-rE2G.bedpe", method="ENCODE-rE2G")  # bulk download

compared = igvf.compare_with_regulatory_links(dosage_df, elements, variant_df, max_distance=0)
summary = igvf.summarize_link_concordance(compared)   # per ancestry: frac_supported
```

The Catalog is queried by region (at most 10 kb per request, so positions are
merged into as few requests as possible) and reports **0-based** positions,
which the client converts to the 1-based convention of `variant_df`. Results
gain `n_links`, `same_gene_support`, `best_method`, `best_source`,
`distance_to_element` and, for the dosage model, `supported_anc{k}` (that
ancestry is significant *and* the pair has independent support).

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
* ``af`` in every output is the in-sample frequency of the counted (``alt``) allele, pooled over samples; ``af_anc{k}`` (dosage model) is the same quantity within ancestry *k*.
* The cis-mapping helpers default to ``preload_haplotypes=True`` so the current chromosome's counts are staged as an int8 tensor on the requested device (GPU or CPU) and windows are sliced there. Override this flag when working under strict memory constraints.

## Testing

Run the test suite with:

```bash
poetry run pytest
```

This exercises the cis- and trans-QTL mapping routines, the frequency QC
helpers and the command line (via toy PLINK 2 files) using small synthetic
datasets; the regression kernels are checked against numpy OLS.
