"""
Core (backwards-compatible) exports for the binary_masses package.

Implementation lives in smaller, focused modules:
- `binary_masses/nonparametric.py`
- `binary_masses/broken_powerlaw.py`
- `binary_masses/polynomial.py`
- `binary_masses/multi_metallicity.py`
"""

from .jax_utils import _configure_jax_gpu_fallback
from .nonparametric import NonParametricMLR
from .broken_powerlaw import BrokenPowerLawMLR
from .polynomial import (
    PolyMassAbsgModel,
    PolynomialMLR,
    IsochroneMassModel,
    DifferencePolyMLR,
)
from .multi_metallicity import MultiMetallicityFitter

__all__ = [
    "_configure_jax_gpu_fallback",
    "NonParametricMLR",
    "BrokenPowerLawMLR",
    "PolyMassAbsgModel",
    "PolynomialMLR",
    "IsochroneMassModel",
    "DifferencePolyMLR",
    "MultiMetallicityFitter",
]
