localQTL user guide
===================

This guide expands on :mod:`localqtl`'s cis-mapping utilities with curated
recipes for running nominal scans, permutations, and ancestry-aware analyses.
It mirrors much of the project's top-level onboarding guidance so that the
details are available inside the rendered documentation.

.. seealso::
   For background on design motivations and roadmap highlights, refer to the
   `README overview <../README.md>`_.

Overview
--------

``localqtl`` is a pure-Python toolkit that keeps the
`tensorQTL <https://github.com/broadinstitute/tensorqtl>`_ data model while
adding GPU-first execution paths and local ancestry-aware inputs. The
functional API and helper classes combine familiar cis-QTL primitives with
utilities for loading haplotypes, aligning samples, and streaming results to
Parquet sinks. Beyond the cis-mapping entry points the codebase ships device
selection helpers, structured loggers, and streaming writers that mirror the
defaults used by the CLI examples and automated tests.

Features
--------

The library exposes the same functionality outlined in the README while adding
documentation-only call-outs for common production workflows:

- **GPU-accelerated pipelines** via ``torch``, with automatic fallbacks to CPU
  when CUDA devices are unavailable (``localqtl.utils.pick_device`` drives
  device selection). See the `feature summary <../README.md#features>`_ for a
  concise checklist.
- **Modular cis-QTL mapping helpers** such as
  :func:`localqtl.cis.map_nominal`, :func:`localqtl.cis.map_permutations`, and
  :func:`localqtl.cis.map_independent`, plus the orchestration-friendly
  :class:`localqtl.cis.CisMapper`.
- **Flexible genotype backends** powered by
  :class:`localqtl.genotypeio.PlinkReader` and :class:`localqtl.pgen.PgenReader`,
  with streaming sinks in :mod:`localqtl.iosinks`.
- **Local ancestry-aware inputs** via :class:`localqtl.haplotypeio.RFMixReader`
  and :class:`localqtl.haplotypeio.InputGeneratorCisWithHaps`.
- **Pure-Python statistics** (``scipy`` + ``py-qvalue``) that remove ``rpy2``
  requirements and play nicely with containerised deployments. The
  :mod:`localqtl.stats` module exposes the q-value helpers used by the high-level
  APIs.
- **tensorQTL feature parity**: :mod:`localqtl.trans` (trans-QTL mapping and
  permutation null), :mod:`localqtl.eigenmt`, multi-term ``interaction_df``
  scans with top-association output, ``paired_covariate_df``,
  :mod:`localqtl.post` (aFC, replication, gene annotation), tabix VCF readers,
  chunked pgen processing and the ``localqtl`` command line.

Installation
------------

Follow the repository's setup instructions—summarised here for convenience.
For additional platform guidance see the
`installation section of the README <../README.md#installation>`_.

Poetry
~~~~~~

.. code-block:: bash

   poetry install                 # CPU / PyTorch only
   poetry install --extras gpu    # + CuPy and cuDF (CUDA 12 wheels)
   poetry install --extras annotate  # + pysam for gnomAD / ALFA tabix importers

Pip
~~~

Install the published package directly from PyPI when you do not need the
editable source checkout; the ``gpu`` extra adds the CuPy / cuDF backends
(everything falls back to numpy and pandas without them):

.. code-block:: bash

   pip install localqtl
   pip install "localqtl[gpu]"
   pip install "localqtl[annotate]"

To work against the latest repository state, fall back to an editable install:

.. code-block:: bash

   pip install -e .

Quick Start
-----------

Nominal scans and permutations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This mirrors the tensorQTL-style workflow shown in the
`README quickstart <../README.md#quickstart>`_ while clarifying module paths for
autodoc cross-references:

.. code-block:: python

   from localqtl.genotypeio import PlinkReader
   from localqtl.phenotypeio import read_phenotype_bed
   from localqtl.cis import map_nominal, map_permutations

   plink = PlinkReader("data/genotypes")
   genotype_df = plink.load_genotypes()
   variant_df = plink.bim.set_index("snp")[["chrom", "pos"]]

   phenotype_df, phenotype_pos_df = read_phenotype_bed("data/phenotypes.bed")
   covariates_df = None

   nominal_df = map_nominal(
       genotype_df=genotype_df,
       variant_df=variant_df,
       phenotype_df=phenotype_df,
       phenotype_pos_df=phenotype_pos_df,
       covariates_df=covariates_df,
       window=1_000_000,
       device="auto",
       return_df=True,
   )

   # Returning a DataFrame skips the per-chromosome Parquet writers. Leave
   # ``return_df`` as ``False`` to mirror tensorQTL's on-disk outputs.

   perm_df = map_permutations(
       genotype_df=genotype_df,
       variant_df=variant_df,
       phenotype_df=phenotype_df,
       phenotype_pos_df=phenotype_pos_df,
       covariates_df=covariates_df,
       window=1_000_000,
       nperm=1_000,
       device="auto",
       perm_chunk=4096,
       seed=42,
   )

``map_nominal``, ``map_permutations``, and ``map_independent`` accept a
``preload_haplotypes`` flag that defaults to ``True``. When ancestry dosages
are provided, enabling the flag stages the current chromosome's haplotype
counts as a contiguous int8 tensor on the target device (GPU or CPU) and
slices per-gene windows there. Disable it if the additional memory pressure
is a concern.

The helper trio also mirrors tensorQTL's "tensor-friendly" outputs through the
``tensorqtl_flavor`` switch. Setting the flag to ``True`` swaps Parquet streaming
for tensorQTL-compatible TSV/Parquet naming conventions and column layouts.

Covariate interaction models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The cis-mapping entry points support genotype-by-covariate interaction tests.
``interaction_df`` (tensorQTL's argument; a DataFrame of samples x terms or a
Series) fits ``y ~ g + i_1..i_k + g:i_1..g:i_k``; the legacy
``interaction_covariate`` takes a single z-scored term (covariate column name,
:class:`pandas.Series` or aligned 1D array).

.. code-block:: python

   interaction_df = map_nominal(
       genotype_df=genotype_df,
       variant_df=variant_df,
       phenotype_df=phenotype_df,
       phenotype_pos_df=phenotype_pos_df,
       covariates_df=covariates_df,
       interaction_df=covariates_df[["age"]],
       maf_threshold_interaction=0.05,
       run_eigenmt=True,
       device="auto",
       out_dir=None,
   )
   top_df = interaction_df.attrs["top_assoc"]

   interaction_df[[
       "variant_id", "pval_nominal", "pval_g", "pval_i", "pval_gi"
   ]].head()

With one term the result includes the standard nominal columns plus the
joint-model coefficients ``b_g``, ``b_i``, ``b_gi`` with standard errors
(``se_*``), t-statistics and p-values; with several terms the variable names
are substituted (``b_age``, ``pval_g-age``, ...).  ``i`` and ``g:i`` are
residualized against the covariates like ``g`` (Frisch–Waugh), so the fit
equals a full OLS of ``y ~ 1 + g + i + g:i + C``.  The top association per
phenotype (or group) is collected with optional eigenMT correction
(``tests_emt``, ``pval_emt``, ``pval_adj_bh``) and written to
``<out_prefix>.top_assoc.txt.gz`` (``top_assoc_path``); ``write_stats=False``
skips the pair-level output.  ``interaction_df`` is mutually exclusive with
``ancestry_model="interaction"``.

``paired_covariate_df`` (phenotypes x samples) adds one phenotype-specific
covariate to the shared covariates in :func:`localqtl.cis.map_nominal` and
:func:`localqtl.cis.map_permutations`.

trans-QTL mapping and tensorQTL post-processing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:mod:`localqtl.trans` mirrors tensorQTL's ``trans`` module
(:func:`localqtl.trans.map_trans`, :func:`localqtl.trans.filter_cis`,
:func:`localqtl.trans.map_trans_permutations`,
:func:`localqtl.trans.apply_permutations`); a
:class:`~localqtl.haplotypeio.LocalAncestryPanel` passed as ``haplotypes``
(with ``variant_df``) adds per-variant local-ancestry main effects.
:mod:`localqtl.eigenmt` estimates the effective number of tests per
cis-window, and :mod:`localqtl.post` provides
:func:`~localqtl.post.calculate_afc` (mixQTL total-read-count model),
:func:`~localqtl.post.calculate_replication` and
:func:`~localqtl.post.annotate_genes`.

.. code-block:: python

   from localqtl import trans, eigenmt, post

   pairs_df = trans.map_trans(genotype_df, phenotype_df, covariates_df,
                              pval_threshold=1e-5, maf_threshold=0.05, return_r2=True)
   pairs_df = trans.filter_cis(pairs_df, phenotype_pos_df, variant_df, window=5_000_000)
   null = trans.map_trans_permutations(genotype_df, covariates_df, nperms=10_000, seed=1)
   pairs_df = trans.apply_permutations(null, pairs_df)

   tests_emt = eigenmt.run_eigenmt(genotype_df, variant_df, phenotype_df, phenotype_pos_df)

Command line
~~~~~~~~~~~~

``python -m localqtl`` / ``localqtl`` accepts tensorQTL's arguments and modes
(``cis``, ``cis_nominal``, ``cis_independent``, ``cis_susie``, ``trans``,
``trans_susie``) and writes tensorQTL's output file names
(``<prefix>.cis_qtl.txt.gz``, ``<prefix>.cis_qtl_pairs.<chr>.parquet``,
``<prefix>.cis_qtl_top_assoc.txt.gz``, ``<prefix>.cis_independent_qtl.txt.gz``,
``<prefix>.trans_qtl_pairs.parquet`` ...).  Local ancestry is enabled with
``--haplotypes`` (RFMix / FLARE output or an rfmix_reader cache),
``--ancestry_model`` and ``--min_ancestry_ac``; ``--device``,
``--perm_chunk`` and ``--localqtl_names`` are the other localQTL additions.

.. code-block:: bash

   localqtl geno pheno.bed.gz prefix --mode cis --covariates covariates.txt
   localqtl geno pheno.bed.gz prefix --mode cis_nominal --covariates covariates.txt \
           --interaction age.txt --cis_output prefix.cis_qtl.txt.gz
   localqtl geno pheno.bed.gz prefix --mode trans --haplotypes rfmix/out

CisMapper orchestration
~~~~~~~~~~~~~~~~~~~~~~~

Use :class:`localqtl.cis.CisMapper` when combining nominal scans, permutations,
q-value corrections, and conditional analyses:

.. code-block:: python

   from localqtl.cis import CisMapper

   mapper = CisMapper(
       genotype_df=genotype_df,
       variant_df=variant_df,
       phenotype_df=phenotype_df,
       phenotype_pos_df=phenotype_pos_df,
       covariates_df=covariates_df,
       window=500_000,
   )

   mapper.map_nominal(nperm=0)
   perm_df = mapper.map_permutations(nperm=1_000, beta_approx=True)
   perm_df = mapper.calculate_qvalues(perm_df, fdr=0.05)
   lead_df = mapper.map_independent(cis_df=perm_df, fdr=0.05)

   # ``nperm`` defaults to ``None`` (alias ``0``) to mirror tensorQTL's
   # permutation-free nominal scans.

   # Persist independent hits using the streaming Parquet sink employed under
   # the hood by the functional APIs.
   lead_df.to_parquet("lead_hits.parquet", compression="snappy")

Local ancestry workflows
------------------------

localQTL integrates RFMix-style ancestry dosages by swapping in ancestry-aware
input generators. Consult the
`local ancestry guidance <../README.md#local-ancestry-aware-mapping>`_ in the
README for additional context.

Haplotype loading
~~~~~~~~~~~~~~~~~

.. code-block:: python

   from localqtl.haplotypeio import RFMixReader, open_panel

   panel = RFMixReader(
       "data/rfmix/",                             # RFMix .msp.tsv output (+ .rfmix.Q)
       select_samples=list(genotype_df.columns),  # same samples, same order as the genotypes
       cache_dir="data/rfmix/la_cache/",          # parsed once into a Zarr cache
   )
   # existing cache: panel = open_panel("data/rfmix/la_cache/", select_samples=...)

   panel.chrom_counts("chr21")                   # (variants, samples, channels) int8 for one chromosome
   panel.locus_index("chr21", positions)         # covering segment per SNP, -1 = none

The panel is passed as ``haplotypes`` to ``CisMapper`` / ``map_nominal``; the
generator keeps one chromosome in memory at a time.  With two ancestries a
single channel is exposed (the second is ``2 - first``).

Posterior-expected ancestry counts
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

RFMix's forward-backward posteriors (``.fb.tsv``) can be kept in the cache and
served as expected ancestry counts ``E[count_k] = sum_h P(Z_h = k)`` (float32,
NaN = no call) instead of hard calls:

.. code-block:: python

   panel = RFMixReader(
       "data/rfmix/", source="fb", keep_posteriors=True,   # cache_dir is required
       cache_dir="data/rfmix/la_cache_fb/", use_posterior=True,
       select_samples=list(genotype_df.columns),
   )
   panel.chrom_counts("chr21")      # float32 expected counts
   panel.chrom_posterior("chr21")   # (variants, samples, 2, ancestries) per-haplotype posteriors

``use_posterior="auto"`` uses posteriors when the Dataset has them and hard
calls otherwise; ``open_panel`` and ``FlareReader`` accept the same argument.

Ancestry-specific dosage model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``ancestry_model="dosage"`` estimates one regulatory effect per ancestry from
the expected ancestry-specific alternate-allele dosage.  For sample *i* with
alternate-allele dosage :math:`G_i` and expected ancestry counts
:math:`L_{ik} = \sum_h P(Z_{ih} = k)`:

.. math::

   D_{ik} = G_i \, L_{ik} / 2, \qquad
   y_i = \sum_k \beta_k D_{ik} + \sum_{k<K} \gamma_k L_{ik}
         + \boldsymbol{\gamma}^{T}\mathbf{C}_i + \epsilon_i

so that :math:`\sum_k D_{ik} = G_i` (the phase of the alternate allele is
unknown) and the :math:`L` columns adjust for local-ancestry main effects.

.. code-block:: python

   dosage_df = map_nominal(
       genotype_df=genotype_df, variant_df=variant_df,
       phenotype_df=phenotype_df, phenotype_pos_df=phenotype_pos_df,
       covariates_df=covariates_df, haplotypes=panel,
       ancestry_model="dosage", out_dir=None,
   )
   dosage_df[["slope_anc0", "pval_anc0", "slope_anc1", "pval_anc1",
              "pval_joint", "pval_diff_anc0_anc1"]]

Per ancestry: ``slope_anc{k}``, ``slope_se_anc{k}``, ``tstat_anc{k}``,
``pval_anc{k}``.  ``fstat_joint`` / ``pval_joint`` test all
:math:`\beta_k = 0`; ``z_diff_anc{i}_anc{j}`` / ``pval_diff_anc{i}_anc{j}``
test :math:`\beta_i = \beta_j` using the covariance of the ancestry betas.
The standard ``slope`` / ``pval_nominal`` columns carry the ancestry with the
strongest effect.  Every row also reports the in-sample per-ancestry allele
statistics ``an_anc{k}`` (allele number :math:`\sum_i L_{ik}`), ``ac_anc{k}``
(expected alt-allele count on ancestry-*k* haplotypes :math:`\sum_i D_{ik}`),
``af_anc{k}`` and ``ma_count_anc{k}`` (minor-allele count within the ancestry;
fractional for posterior-expected counts).  ``min_ancestry_ac=<count>`` masks
(``NaN``) the effect estimates and pairwise differences of ancestries with
fewer in-sample minor alleles than that, keeping the row, the pooled columns
and ``fstat_joint``.  With a hard-call panel the model reduces to the classic
ancestry-specific genotype regression.  Substituting posterior expectations
does not propagate the assignment uncertainty into the standard errors
(joint-likelihood / multiple-imputation estimation is future work), and the
model is currently available for ``map_nominal`` only.

Sample alignment and mapping
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # the panel was built with select_samples=list(genotype_df.columns), so the
   # ancestry samples already match the genotype columns (order included);
   # restrict the other tables to the same samples
   keep = list(genotype_df.columns)
   phenotype_df = phenotype_df[keep]
   if covariates_df is not None:
       covariates_df = covariates_df.loc[keep]

   mapper = CisMapper(
       genotype_df=genotype_df,
       variant_df=variant_df,
       phenotype_df=phenotype_df,
       phenotype_pos_df=phenotype_pos_df,
       covariates_df=covariates_df,
       haplotypes=panel,
       window=1_000_000,
       device="auto",
   )

   mapper.map_nominal(nperm=0)
   perm_df = mapper.map_permutations(nperm=1_000, beta_approx=True)
   perm_df = mapper.calculate_qvalues(perm_df, fdr=0.05)
   lead_df = mapper.map_independent(cis_df=perm_df, fdr=0.05)

External allele frequencies and ancestry QC
-------------------------------------------

:mod:`localqtl.freq` compares in-sample (per-ancestry) allele frequencies
with external references and summarises p-value calibration by frequency bin
and ancestry.

Allele metadata
~~~~~~~~~~~~~~~

:attr:`localqtl.pgen.PgenReader.variant_df`, :attr:`localqtl.genotypeio.PlinkReader.variant_df`
and :func:`localqtl.genotypeio.load_genotypes` carry ``chrom, pos, ref, alt``
where ``alt`` is the counted (dosage) allele: pgenlib reports ALT counts, and
for PLINK 1 files the ``2 - bed`` flip makes the dosage count ``.bim`` column 5
(A1; ALT for ``plink2 --make-bed`` / ``plink --keep-allele-order``).  The
``af`` column of every result is the in-sample frequency of that allele.
Tables with only ``chrom``/``pos`` keep working;
:func:`localqtl.genotypeio.alleles_from_variant_ids` parses GTEx-style
``chr1_123_A_G_b38`` IDs.

Frequency tables
~~~~~~~~~~~~~~~~

A :class:`localqtl.freq.FrequencyTable` is keyed by ``(chrom, pos, ref, alt)``
on GRCh38 and holds ``af_<group>``, ``ac_<group>``, ``an_<group>`` columns.
:meth:`~localqtl.freq.FrequencyTable.lookup` joins it onto a ``variant_df``
allele-aware: exact matches first, then swapped ``ref``/``alt`` rows
(re-oriented as ``1 - af``, ``an - ac``), with ``match_type``,
``allele_flipped``, ``strand_ambiguous`` (A/T, C/G) and
``missing_in_reference`` flags.  Only GRCh38 tables are accepted; strand flips
are flagged, never corrected.

.. code-block:: python

   from localqtl import freq

   # gnomAD v4.1 sites VCFs (tabix; only the regions around your variants are
   # read; local-ancestry tract AF/AC/AN fields are detected from the header
   # and exposed as groups ``lai_<suffix>``)
   gnomad = freq.from_gnomad_vcf("gnomad.genomes.v4.1.sites.chr*.vcf.bgz",
                                 variant_df=variant_df, groups=("afr", "nfe"))
   # NCBI ALFA: one GRCh38 VCF whose sample columns are populations
   alfa = freq.from_alfa_vcf("freq.vcf.gz", variant_df=variant_df,
                             populations=("AFR", "AFA", "EUR"))
   # without pysam: ``bcftools query -H`` output (recipes in localqtl.freq.bcftools)
   table = freq.from_bcftools_query_tsv("gnomad.chr22.freq.tsv")
   table.write("gnomad_chr22.freq.parquet")

Valid gnomAD group keys are listed in :data:`localqtl.freq.GNOMAD_V4_GROUPS`
(``afr, ami, amr, asj, eas, fin, mid, nfe, remaining, sas``) and ALFA codes in
:data:`localqtl.freq.ALFA_POPULATION_NAMES` (``EUR, AFO, EAS, AFA, LAC, LEN,
OAS, SAS, OTR, AFR, ASN, TOT``).  The gnomAD local-ancestry-informed release
covers bi-allelic SNPs with MAF > 0.1 % in the African/African American
(AFR/EUR tracts) and Admixed American (AFR/EUR/AMR tracts) groups.

Ancestry-frequency QC
~~~~~~~~~~~~~~~~~~~~~

The mapping from panel ancestry labels to reference groups is **always
explicit**; there are no defaults.

.. code-block:: python

   ancestry_map = {"AFR": "afr", "EUR": "nfe"}        # panel label -> table group

   insample = freq.ancestry_allele_frequencies(genotype_df, variant_df, panel)
   cmp_df = freq.compare_allele_frequencies(insample, gnomad, ancestry_map)
   summary = freq.summarize_af_concordance(cmp_df)

``compare_allele_frequencies`` adds, per ancestry, ``af_in``, ``af_ref``,
``af_diff``, ``logodds_diff``, a two-proportion ``z`` and a ``discordant`` flag
(``|diff| > 0.10`` or ``|z| > 4`` by default), the admixture-weighted expected
pooled frequency, and the lookup flags.  ``summarize_af_concordance`` gives
Pearson/Spearman *r*, mean absolute difference and the discordant, flipped and
missing fractions per reference-MAF bin.

Calibration benchmark
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   dosage_df = freq.annotate_with_frequency(
       dosage_df, variant_df, gnomad, ancestry_map=ancestry_map,
       ancestry_labels=list(panel.ancestries))                 # adds af_ref_<g>, af_ref_anc{k}
   calib = freq.calibration_by_frequency_dosage(
       dosage_df, ancestry_labels=list(panel.ancestries), by="af_ref_anc")

   bench = freq.run_frequency_benchmark(
       genotype_df, variant_df, phenotype_df, phenotype_pos_df, panel, gnomad,
       ancestry_map, drop_discordant=True, n_null=1)
   bench.write("qc/")

:func:`~localqtl.freq.calibration_by_frequency` reports, per frequency bin,
the genomic-control :math:`\lambda`, the fraction of p-values below each
``alpha``, the median p-value and a Kolmogorov-Smirnov uniformity test; the
dosage variant does so for each ``pval_anc{k}`` (binned by in-sample
``af_anc{k}``, reference ``af_ref_anc{k}`` or ``ma_count_anc{k}``) and for
``pval_joint``.  :func:`~localqtl.freq.run_frequency_benchmark` chains the
comparison, optional exclusion of discordant / flipped / ambiguous variants,
the dosage scan and the calibration tables, and repeats the scan on
sample-permuted phenotypes as a null.

Planned annotation modules
~~~~~~~~~~~~~~~~~~~~~~~~~~

:mod:`localqtl.annotate.alphagenome` (AlphaGenome Atlas molecular-effect scores
for lead variants; non-commercial API) and :mod:`localqtl.annotate.igvf` (IGVF
Catalog variant->gene and element->gene regulatory links) define their public
signatures and constants now; the implementations follow in a later release.

Advanced topics
---------------

Beyond core cis-mapping, ``localqtl`` ships LD utilities that can seed
fine-mapping or replication analyses:

.. code-block:: python

   from localqtl.finemap import get_pairwise_ld, get_ld_matrix
   from localqtl.pgen import PgenReader

   pgr = PgenReader("data/genotypes")
   ld_value, variant_ids = get_pairwise_ld(pgr, "rs123", "rs456")
   ld_matrix, variant_ids = get_ld_matrix(pgr, anchor_variant="rs123", window=1_000_000)

For additional preprocessing helpers, explore :mod:`localqtl.preproc` (MAF
filters) and :mod:`localqtl.utils` (CUDA detection utilities).

SuSiE fine-mapping
------------------

The :mod:`localqtl.susie` module ports tensorQTL's SuSiE workflow to pure
Python. It supports low-level model fitting through
:func:`localqtl.susie.susie` and high-level cis-window scanning through
:func:`localqtl.susie.map` and :func:`localqtl.susie.map_loci`.

Genome-wide cis windows
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from localqtl import susie

   susie_summary = susie.map(
       genotype_df=genotype_df,
       variant_df=variant_df,
       phenotype_df=phenotype_df,
       phenotype_pos_df=phenotype_pos_df,
       covariates_df=covariates_df,
       L=10,
       coverage=0.95,
       min_abs_corr=0.5,
       device="auto",
   )

   susie_summary.head()

The summary output contains ``phenotype_id``, ``variant_id``, posterior
inclusion probabilities in ``pip``, allele frequency in ``af``, and credible
set membership in ``cs_id``.

Targeted loci
~~~~~~~~~~~~~

.. code-block:: python

   locus_df = phenotype_pos_df.reset_index()[["phenotype_id", "chr"]].copy()
   locus_df["position"] = (
       phenotype_pos_df["start"].to_numpy() + phenotype_pos_df["end"].to_numpy()
   ) // 2

   locus_summary, locus_res = susie.map_loci(
       locus_df=locus_df,
       genotype_df=genotype_df,
       variant_df=variant_df,
       phenotype_df=phenotype_df,
       covariates_df=covariates_df,
       device="auto",
   )

Use ``map_loci`` when each phenotype should be fine-mapped only in specific
windows rather than across its full cis span.

COLOC colocalization
--------------------

The :mod:`localqtl.coloc` module provides Approximate Bayes Factor
colocalization from raw data or summary statistics.

Raw-data phenotype pairs
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from localqtl import coloc

   coloc_df = coloc.run_pairs(
       genotype_df=genotype_df,
       variant_df=variant_df,
       phenotype1_df=phenotype_df,
       phenotype2_df=trait2_df,
       phenotype_pos_df=phenotype_pos_df,
       covariates1_df=covariates_df,
       covariates2_df=trait2_covariates_df,
       mode="beta",
       device="auto",
   )

   coloc_df[["pp_h3_abf", "pp_h4_abf"]].head()

This returns one row per phenotype with posterior support for hypotheses
``H0`` through ``H4``. ``pp_h4_abf`` is the shared-causal-variant posterior and
is the typical downstream summary statistic.

Summary-statistics interface
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   coloc_stats = coloc.coloc_from_summary(
       beta1=eqtl_beta,
       se1=eqtl_se,
       beta2=gwas_beta,
       se2=gwas_se,
       maf=maf,
       n1=eqtl_n,
       n2=gwas_n,
   )

Use :func:`localqtl.coloc.coloc_from_summary` when only marginal effect sizes
and standard errors are available.

Post-processing helpers
-----------------------

The branch adds lightweight helpers in :mod:`localqtl.cis.postproc` for
attaching fine-mapping and colocalization results to existing cis outputs.

.. code-block:: python

   from localqtl.cis.postproc import annotate_with_coloc, annotate_with_susie

   nominal_annotated = annotate_with_susie(nominal_df, susie_summary)
   perm_annotated = annotate_with_coloc(perm_df, coloc_df)

``annotate_with_susie`` appends ``susie_pip`` and ``susie_cs`` columns keyed by
``(phenotype_id, variant_id)``. When a variant belongs to multiple credible
sets, ``susie_cs`` stores a comma-separated list of set IDs.
``annotate_with_coloc`` appends a phenotype-level ``coloc_pp_h4`` column to
permutation results.

Device, logging, and reproducibility helpers
--------------------------------------------

:mod:`localqtl.utils` centralises the lightweight utilities that the mapping
pipelines consume internally:

.. code-block:: python

   from localqtl.utils import gpu_available, pick_device, SimpleLogger, NullLogger, subseed

   if not gpu_available():
       print("Falling back to CPU execution")

   device = pick_device("auto")  # returns "cuda" when a GPU is visible, else "cpu"
   logger = SimpleLogger(verbose=True, timestamps=True)

   with logger.time_block("residualizing phenotypes"):
       ...

   # Derive deterministic sub-seeds when distributing work across processes
   worker_seed = subseed(base=1234, key="worker-0")

``SimpleLogger`` mirrors the console logging behaviour of tensorQTL while adding
context managers for timing blocks. ``NullLogger`` provides a drop-in silent
replacement. ``pick_device`` implements the ``"auto"`` device semantics used by
the cis-mapping functions and :class:`localqtl.cis.CisMapper`.

Streaming output sinks
----------------------

The cis-mapping helpers stream association statistics through reusable Parquet
writes that are also exposed as public classes:

.. code-block:: python

   import pyarrow as pa
   from localqtl.iosinks import ParquetSink, AsyncParquetSink, RowGroupBuffer

   schema = pa.schema([
       ("phenotype_id", pa.string()),
       ("variant_id", pa.string()),
       ("pval_beta", pa.float64()),
   ])

   with ParquetSink("cis_hits.parquet", schema=schema) as sink:
       sink.write({"phenotype_id": ["geneA"], "variant_id": ["rs1"], "pval_beta": [1e-8]})

   # Or overlap Arrow compression + filesystem writes with GPU/CPU work
   async_sink = AsyncParquetSink("cis_hits_async.parquet", schema=schema)
   async_sink.write({"phenotype_id": ["geneA"], "variant_id": ["rs1"], "pval_beta": [1e-8]})
   async_sink.close()

``RowGroupBuffer`` batches rows before flushing them through ``ParquetSink``.
The streaming classes share the exact code paths used by
:func:`localqtl.cis.map_nominal` and :func:`localqtl.cis.map_permutations` when
``return_df=False``.

Statistics helpers
------------------

:mod:`localqtl.stats` contains the numerically stable routines that power the
beta approximation used during permutation scans:

.. code-block:: python

   from localqtl.stats import calculate_qvalues, beta_approx_pval, pval_from_corr_r2

   perm_df = mapper.map_permutations(nperm=1_000)
   perm_df = calculate_qvalues(perm_df, fdr=0.05)

   # Inspect the beta-approximation metadata already materialised by map_permutations
   beta_meta = perm_df[["pval_beta", "beta_shape1", "beta_shape2", "true_dof"]].head()

   # If you capture raw permutation maxima (e.g., via a custom hook), you can
   # recompute the beta approximation with the standalone helper:
   #   raw_perm_r2 = np.load("permutation_r2.npy")
   #   p_beta, a_hat, b_hat, dof_est, p_true = beta_approx_pval(
   #       r2_perm=raw_perm_r2,
   #       r2_true=perm_df["r2_nominal"].max(),
   #       dof_init=perm_df["true_dof"].iloc[0],
   #   )

   # Convert R^2 back to a two-sided p-value for downstream filtering
   p_from_r2 = pval_from_corr_r2(r2=0.25, dof=perm_df["true_dof"].iloc[0])

API reference
-------------

Core mapping API
~~~~~~~~~~~~~~~~

* :func:`localqtl.cis.map_nominal`
* :func:`localqtl.cis.map_permutations`
* :func:`localqtl.cis.map_independent`
* :func:`localqtl.cis.map_nominal_loci`
* :class:`localqtl.cis.CisMapper`
* :mod:`localqtl.cis`

trans-QTL mapping, eigenMT and post-processing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* :func:`localqtl.trans.map_trans`
* :func:`localqtl.trans.filter_cis`
* :func:`localqtl.trans.map_trans_permutations`
* :func:`localqtl.trans.apply_permutations`
* :class:`localqtl.trans.GenotypeGeneratorTrans`
* :func:`localqtl.eigenmt.run_eigenmt`
* :func:`localqtl.eigenmt.compute_tests`
* :func:`localqtl.eigenmt.padjust_bh`
* :func:`localqtl.post.calculate_afc`
* :func:`localqtl.post.calculate_replication`
* :func:`localqtl.post.annotate_genes`
* :func:`localqtl.compat.to_tensorqtl_columns`
* :func:`localqtl.compat.from_tensorqtl_columns`
* :func:`localqtl.cli.main`
* :mod:`localqtl.trans`
* :mod:`localqtl.eigenmt`
* :mod:`localqtl.post`
* :mod:`localqtl.compat`

Allele frequencies and QC
~~~~~~~~~~~~~~~~~~~~~~~~~

* :class:`localqtl.freq.FrequencyTable`
* :func:`localqtl.freq.read_frequency_table`
* :func:`localqtl.freq.from_gnomad_vcf`
* :func:`localqtl.freq.from_alfa_vcf`
* :func:`localqtl.freq.from_bcftools_query_tsv`
* :func:`localqtl.freq.ancestry_allele_frequencies`
* :func:`localqtl.freq.compare_allele_frequencies`
* :func:`localqtl.freq.summarize_af_concordance`
* :func:`localqtl.freq.annotate_with_frequency`
* :func:`localqtl.freq.calibration_by_frequency`
* :func:`localqtl.freq.calibration_by_frequency_dosage`
* :func:`localqtl.freq.run_frequency_benchmark`
* :mod:`localqtl.freq`
* :mod:`localqtl.annotate`

Genotype and phenotype I/O
~~~~~~~~~~~~~~~~~~~~~~~~~~

* :class:`localqtl.genotypeio.PlinkReader`
* :func:`localqtl.genotypeio.alleles_from_variant_ids`
* :func:`localqtl.genotypeio.get_vcf_region`
* :func:`localqtl.genotypeio.get_vcf_variants`
* :class:`localqtl.genotypeio.InputGeneratorCis`
* :class:`localqtl.haplotypeio.RFMixReader`
* :class:`localqtl.haplotypeio.InputGeneratorCisWithHaps`
* :func:`localqtl.phenotypeio.read_phenotype_bed`
* :class:`localqtl.pgen.PgenReader`
* :func:`localqtl.pgen.generate_paired_chunks`
* :class:`localqtl.iosinks.ParquetSink`
* :class:`localqtl.iosinks.AsyncParquetSink`
* :class:`localqtl.iosinks.RowGroupBuffer`
* :mod:`localqtl.genotypeio`
* :mod:`localqtl.haplotypeio`
* :mod:`localqtl.phenotypeio`
* :mod:`localqtl.pgen`
* :mod:`localqtl.iosinks`

Execution utilities
~~~~~~~~~~~~~~~~~~~

* :class:`localqtl.utils.SimpleLogger`
* :class:`localqtl.utils.NullLogger`
* :func:`localqtl.utils.gpu_available`
* :func:`localqtl.utils.pick_device`
* :func:`localqtl.utils.subseed`
* :mod:`localqtl.utils`

Statistics and regression
~~~~~~~~~~~~~~~~~~~~~~~~~

* :func:`localqtl.stats.calculate_qvalues`
* :func:`localqtl.stats.beta_approx_pval`
* :func:`localqtl.stats.fit_beta_parameters`
* :func:`localqtl.stats.pval_from_corr_r2`
* :func:`localqtl.stats.get_t_pval`
* :mod:`localqtl.stats`
* :class:`localqtl.regression_kernels.Residualizer`
* :func:`localqtl.regression_kernels.run_batch_regression`
* :func:`localqtl.regression_kernels.run_batch_regression_with_permutations`
* :func:`localqtl.regression_kernels.perm_chunk_r2`
* :func:`localqtl.regression_kernels.prep_ctx_for_perm`
* :mod:`localqtl.regression_kernels`

Testing
-------

Run the automated test suite to validate GPU/CPU parity and verify the
regression harnesses:

.. code-block:: bash

   poetry run pytest

These tests exercise nominal scans, permutations, and ancestry-aware code paths
on synthetic datasets. For context about the lightweight fixtures used in CI,
see the `testing section of the README <../README.md#testing>`_.
