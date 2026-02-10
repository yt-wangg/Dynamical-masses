"""
Core (backwards-compatible) exports for the binary_masses package.

Implementation lives in smaller, focused modules:
- `binary_masses/polynomial.py`
- `binary_masses/differencepoly_feh.py`
- `binary_masses/isochrone_grid.py`
- Deprecated models live under `binary_masses/depracated/`
"""

from .jax_utils import _configure_jax_gpu_fallback
from .polynomial import (
    PolyMassAbsgModel,
    PolynomialMLR,
    IsochroneMassModel,
    DifferencePolyMLR,
)
from .differencepoly_feh import (
    IsochroneMassSurfaceModel,
    DifferencePolyFehMassAbsgModel,
    DifferencePolyFehMLR,
)
from .isochrone_grid import (
    load_interpolated_isochrone_data,
    load_interpolated_mass_curve,
    load_interpolated_mass_grid,
)

__all__ = [
    "_configure_jax_gpu_fallback",
    "PolyMassAbsgModel",
    "PolynomialMLR",
    "IsochroneMassModel",
    "DifferencePolyMLR",
    "IsochroneMassSurfaceModel",
    "DifferencePolyFehMassAbsgModel",
    "DifferencePolyFehMLR",
    "load_interpolated_mass_grid",
    "load_interpolated_mass_curve",
    "load_interpolated_isochrone_data",
]
