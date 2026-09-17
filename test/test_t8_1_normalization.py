#!/usr/bin/env python3
"""T8.1 dynamics-density normalization unit tests.

The mixture observed in u-space must integrate to one for every total mass:

    p(u | m) = (1/sqrt(m)) * [ (1-f) p0(u/sqrt(m)) + f q(u/sqrt(m)) ],

with s = sqrt(Mtot/Msun), w = u/s, both components densities in w-space
carrying the same 1/s Jacobian.  The T8.1 revision divides the good basis by
its raw support integral (0.9978600946) so each component integrates to one
exactly; previously the good component integrated to 0.99786, which made the
mixture weight a non-physical basis weight (a constant, mass-independent
0.2 percent tilt absorbed by the sampled f, with no effect on masses).

The third test documents the missing-Jacobian failure mode audited in the
public Hwang+2023 likelihood (outlier term lacking 1/s): its integral is
(1-f) + f*sqrt(m), which varies with mass and biases recovered masses high.
"""

import sys
from pathlib import Path

import numpy as np
from scipy.integrate import quad

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from binary_masses.hierarchical_metallicity import (  # noqa: E402
    RICE_GOOD_BASIS_RAW_INTEGRAL,
    RICE_GOOD_SUPPORT,
    _rice_component_log_density,
    rice_good_raw,
    rice_outlier_normalization,
)


def _mixture_log_density(u, sqrt_m, f):
    w = u / sqrt_m
    log_good = _rice_component_log_density(w, component="good", support_max=RICE_GOOD_SUPPORT)
    log_bad = _rice_component_log_density(w, component="bad", support_max=RICE_GOOD_SUPPORT)
    good, bad = np.exp(log_good), np.exp(log_bad)
    return ((1.0 - f) * good + f * bad) / sqrt_m


def test_good_basis_unit_integral():
    integral, _ = quad(rice_good_raw, 0.0, RICE_GOOD_SUPPORT, limit=400)
    assert abs(integral - 1.0) < 1e-9, integral


def test_raw_integral_constant_matches_raw_shape():
    raw_shape = lambda w: (
        rice_good_raw(w) * RICE_GOOD_BASIS_RAW_INTEGRAL
    )
    expected, _ = quad(raw_shape, 0.0, RICE_GOOD_SUPPORT, limit=400)
    assert abs(expected - RICE_GOOD_BASIS_RAW_INTEGRAL) < 1e-9, expected


def test_mixture_unit_integral_over_u():
    for sqrt_m in (0.25, 0.5, 1.0, 1.5, 2.5):
        for f in (0.0, 0.2, 0.5):
            integral, _ = quad(
                lambda u: _mixture_log_density(u, sqrt_m, f),
                0.0, RICE_GOOD_SUPPORT * sqrt_m, limit=400,
            )
            assert abs(integral - 1.0) < 1e-9, (sqrt_m, f, integral)


def test_outlier_component_unit_integral():
    density = lambda w: np.exp(
        _rice_component_log_density(w, component="bad", support_max=RICE_GOOD_SUPPORT)
    )
    integral, _ = quad(density, 0.0, RICE_GOOD_SUPPORT, limit=400)
    assert abs(integral - 1.0) < 1e-9, integral


def test_missing_jacobian_form_is_not_normalized():
    """The audited Hwang+2023 failure mode, for contrast.

    With the outlier term stripped of its 1/s Jacobian the u-space integral
    becomes (1-f) + f*sqrt(m) instead of one, so the optimizer can inflate
    the likelihood by increasing the mass.
    """
    for sqrt_m in (0.5, 1.0, 2.0):
        f = 0.2

        def broken_density(u):
            w = u / sqrt_m
            good = np.exp(
                _rice_component_log_density(w, component="good", support_max=RICE_GOOD_SUPPORT)
            )
            bad = np.exp(
                _rice_component_log_density(w, component="bad", support_max=RICE_GOOD_SUPPORT)
            )
            return (1 - f) * good / sqrt_m + f * bad

        integral, _ = quad(broken_density, 0.0, RICE_GOOD_SUPPORT * sqrt_m, limit=400)
        assert abs(integral - ((1 - f) + f * sqrt_m)) < 1e-9, (sqrt_m, integral)
        assert integral != 1.0


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"{name}: OK")
    print("All T8.1 normalization tests passed.")
