"""
Bayesian Binary Mass Estimation

A Python package for estimating dynamical masses of binary stars using
Bayesian inference with Rice distribution uncertainties.

Authors: Yutong Wang and collaborators
License: MIT
"""

from .core import (
    PolynomialMLR,
    IsochroneMassModel,
    DifferencePolyMLR,
    IsochroneMassSurfaceModel,
    DifferencePolyFehMassAbsgModel,
    DifferencePolyFehMLR,
    load_interpolated_mass_grid,
    load_interpolated_mass_curve,
    load_interpolated_isochrone_data,
)

__version__ = "1.0.0"
__author__ = "Yutong Wang"

__all__ = [
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
