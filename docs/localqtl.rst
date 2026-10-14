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

Pip
~~~

Install the published package directly from PyPI when you do not need the
editable source checkout; the ``gpu`` extra adds the CuPy / cuDF backends
(everything falls back to numpy and pandas without them):

.. code-block:: bash

   pip install localqtl
   pip install "localqtl[gpu]"

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

The cis-mapping entry points also support genotype-by-covariate interaction
tests by passing ``interaction_covariate``. Accepted forms are a covariate
column name in ``covariates_df``, a :class:`pandas.Series` indexed by sample
IDs, or a 1D array-like already aligned to the sample order.

.. code-block:: python

   interaction_df = map_nominal(
       genotype_df=genotype_df,
       variant_df=variant_df,
       phenotype_df=phenotype_df,
       phenotype_pos_df=phenotype_pos_df,
       covariates_df=covariates_df,
       interaction_covariate="age",
       device="auto",
       return_df=True,
   )

   interaction_df[[
       "variant_id", "pval_nominal", "pval_g", "pval_i", "pval_gi"
   ]].head()

In this mode the result includes the standard nominal columns plus the
joint-model coefficients ``b_g``, ``b_i``, ``b_gi`` and their standard errors,
t-statistics, and p-values. ``interaction_covariate`` is mutually exclusive
with ``ancestry_model="interaction"``.

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
strongest effect.  With a hard-call panel the model reduces to the classic
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
* :class:`localqtl.cis.CisMapper`
* :mod:`localqtl.cis`

Genotype and phenotype I/O
~~~~~~~~~~~~~~~~~~~~~~~~~~

* :class:`localqtl.genotypeio.PlinkReader`
* :class:`localqtl.genotypeio.InputGeneratorCis`
* :class:`localqtl.haplotypeio.RFMixReader`
* :class:`localqtl.haplotypeio.InputGeneratorCisWithHaps`
* :func:`localqtl.phenotypeio.read_phenotype_bed`
* :class:`localqtl.pgen.PgenReader`
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
