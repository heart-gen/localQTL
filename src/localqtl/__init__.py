"""Top-level package exports for :mod:`localqtl`.

This module re-exports the most commonly used classes/functions so that they
remain accessible from ``localqtl`` after the project was reorganized into a
package with dedicated submodules (``cis``, ``preproc``, ``finemap`` …).
"""

from . import cis
from . import finemap
from . import genotypeio
from . import haplotypeio
from . import iosinks
from . import pgen
from . import phenotypeio
from . import preproc
from . import regression_kernels
from . import stats
from . import utils

from .cis import CisMapper, map_independent, map_nominal, map_permutations
from .genotypeio import InputGeneratorCis, PlinkReader
from .haplotypeio import InputGeneratorCisWithHaps, RFMixReader
from .phenotypeio import read_phenotype_bed
from .pgen import PgenReader
from .utils import gpu_available

__all__ = [
    "map_nominal",
    "map_permutations",
    "map_independent",
    "CisMapper",
    "read_phenotype_bed",
    "gpu_available",
    "PlinkReader",
    "RFMixReader",
    "PgenReader",
    "InputGeneratorCis",
    "InputGeneratorCisWithHaps",
    "cis",
    "finemap",
    "phenotypeio",
    "genotypeio",
    "haplotypeio",
    "iosinks",
    "pgen",
    "preproc",
    "regression_kernels",
    "stats",
    "utils",
]
