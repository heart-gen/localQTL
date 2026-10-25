"""
Top-level package exports for :mod:`localqtl`.

This module re-exports the most commonly used classes/functions so that they
remain accessible from ``localqtl`` after the project was reorganized into a
package with dedicated submodules (``cis``, ``preproc``, ``finemap`` …).
"""
import os

_IS_RTD = os.environ.get("READTHEDOCS") == "True"
if _IS_RTD:
    os.environ.setdefault("LOCALQTL_NO_GPU", "1")

from . import annotate
from . import cis
from . import coloc
from . import compat
from . import eigenmt
from . import freq
from . import finemap
from . import genotypeio
from . import haplotypeio
from . import iosinks
from . import pgen
from . import phenotypeio
from . import post
from . import preproc
from . import regression_kernels
from . import stats
from . import susie
from . import trans
from . import utils

from .cis import (CisMapper, map_conditional_dosage, map_independent, map_nominal,
                  map_nominal_loci, map_permutations)
from .genotypeio import InputGeneratorCis, PlinkReader, get_vcf_region, get_vcf_variants
from .trans import GenotypeGeneratorTrans, apply_permutations, filter_cis, map_trans, map_trans_permutations
from .eigenmt import run_eigenmt
from .post import annotate_genes, calculate_afc, calculate_replication
from .stats import calculate_qvalues
from .haplotypeio import FlareReader, InputGeneratorCisWithHaps, LocalAncestryPanel, RFMixReader, open_panel
from .freq import FrequencyTable, ancestry_allele_frequencies, annotate_with_frequency
from .phenotypeio import read_phenotype_bed
from .pgen import PgenReader
from .utils import gpu_available

__all__ = [
    "map_nominal",
    "map_nominal_loci",
    "map_permutations",
    "map_independent",
    "map_conditional_dosage",
    "CisMapper",
    "map_trans",
    "map_trans_permutations",
    "apply_permutations",
    "filter_cis",
    "GenotypeGeneratorTrans",
    "run_eigenmt",
    "calculate_qvalues",
    "calculate_afc",
    "calculate_replication",
    "annotate_genes",
    "get_vcf_region",
    "get_vcf_variants",
    "read_phenotype_bed",
    "gpu_available",
    "PlinkReader",
    "RFMixReader",
    "FlareReader",
    "LocalAncestryPanel",
    "open_panel",
    "PgenReader",
    "InputGeneratorCis",
    "InputGeneratorCisWithHaps",
    "FrequencyTable",
    "ancestry_allele_frequencies",
    "annotate_with_frequency",
    "annotate",
    "cis",
    "coloc",
    "compat",
    "eigenmt",
    "finemap",
    "freq",
    "phenotypeio",
    "genotypeio",
    "haplotypeio",
    "iosinks",
    "pgen",
    "post",
    "preproc",
    "regression_kernels",
    "stats",
    "susie",
    "trans",
    "utils",
]
