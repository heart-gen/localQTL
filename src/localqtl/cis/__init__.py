from .nominal import map_nominal, map_nominal_loci
from .permutations import map_permutations
from .independent import map_independent
from .conditional import map_conditional_dosage
from .mapper import CisMapper

__all__ = [
    "map_nominal",
    "map_nominal_loci",
    "map_permutations",
    "map_independent",
    "map_conditional_dosage",
    "CisMapper"
]
