"""
Deprecated models kept for backward compatibility.
"""

from .broken_powerlaw import BrokenPowerLawMLR
from .multi_metallicity import MultiMetallicityFitter
from .nonparametric import NonParametricMLR
from .sampling import BinarySystem, BinaryPopulation

__all__ = [
    "BrokenPowerLawMLR",
    "MultiMetallicityFitter",
    "NonParametricMLR",
    "BinarySystem",
    "BinaryPopulation",
]
