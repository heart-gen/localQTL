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

   poetry install

Pip
~~~

Install the published package directly from PyPI when you do not need the
editable source checkout:

.. code-block:: bash

   pip install localqtl

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
are provided, enabling the flag stages haplotypes as contiguous tensors on the
target device (GPU or CPU). Disable it if the additional memory pressure is a
concern.

The helper trio also mirrors tensorQTL's "tensor-friendly" outputs through the
``tensorqtl_flavor`` switch. Setting the flag to ``True`` swaps Parquet streaming
for tensorQTL-compatible TSV/Parquet naming conventions and column layouts.

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

   from localqtl.haplotypeio import RFMixReader

   rfmix = RFMixReader(
       prefix_path="data/rfmix/prefix",
       binary_path="data/rfmix",
       verbose=True,
   )

   H = rfmix.load_haplotypes()
   loci_df = rfmix.loci_df

Sample alignment and mapping
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   keep = [sid for sid in rfmix.sample_ids if sid in genotype_df.columns]
   genotype_df = genotype_df[keep]
   phenotype_df = phenotype_df[keep]
   if covariates_df is not None:
       covariates_df = covariates_df.loc[keep]

   variant_df["chrom"] = variant_df["chrom"].astype(str)
   loci_df = loci_df.copy()
   loci_df["chrom"] = loci_df["chrom"].astype(str)

   mapper = CisMapper(
       genotype_df=genotype_df,
       variant_df=variant_df,
       phenotype_df=phenotype_df,
       phenotype_pos_df=phenotype_pos_df,
       covariates_df=covariates_df,
       haplotypes=H,
       loci_df=loci_df,
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
