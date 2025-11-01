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
Parquet sinks.

Features
--------

The library exposes the same functionality outlined in the README while adding
documentation-only call-outs for common production workflows:

- **GPU-accelerated pipelines** via ``torch``, with automatic fallbacks to CPU
  when CUDA devices are unavailable. See the
  `feature summary <../README.md#features>`_ for a concise checklist.
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
  requirements and play nicely with containerised deployments.

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
   )

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

API reference
-------------

.. currentmodule:: localqtl

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   map_nominal
   map_permutations
   map_independent
   CisMapper
   PlinkReader
   RFMixReader
   PgenReader
   InputGeneratorCis
   InputGeneratorCisWithHaps
   read_phenotype_bed
   gpu_available
   cis
   genotypeio
   haplotypeio
   iosinks
   phenotypeio
   preproc
   finemap
   stats
   utils

Testing
-------

Run the automated test suite to validate GPU/CPU parity and verify the
regression harnesses:

.. code-block:: bash

   poetry run pytest

These tests exercise nominal scans, permutations, and ancestry-aware code paths
on synthetic datasets. For context about the lightweight fixtures used in CI,
see the `testing section of the README <../README.md#testing>`_.
