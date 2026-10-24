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
from . import freq
from . import finemap
from . import genotypeio
from . import haplotypeio
from . import iosinks
from . import pgen
from . import phenotypeio
from . import preproc
from . import regression_kernels
from . import stats
from . import susie
from . import utils

from .cis import CisMapper, map_independent, map_nominal, map_nominal_loci, map_permutations
from .genotypeio import InputGeneratorCis, PlinkReader
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
    "CisMapper",
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
    "finemap",
    "freq",
    "phenotypeio",
    "genotypeio",
    "haplotypeio",
    "iosinks",
    "pgen",
    "preproc",
    "regression_kernels",
    "stats",
    "susie",
    "utils",
]
