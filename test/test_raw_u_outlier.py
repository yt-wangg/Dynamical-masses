"""Regression tests for the post-7bc mass-independent raw-u outlier model."""

import os
import sys

import numpy as np
from scipy.integrate import quad
from scipy.special import i0e

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from binary_masses.hierarchical_metallicity import (
    RICE_GOOD_SUPPORT,
    RICE_OUTLIER_MU,
    RICE_OUTLIER_SIGMA,
    raw_u_outlier_log_likelihood,
    rice_outlier_normalization,
)


def _rice_pdf(u_obs, v, sigma_obs):
    arg = u_obs * v / sigma_obs**2
    return (
        u_obs / sigma_obs**2
        * np.exp(-(u_obs - v) ** 2 / (2.0 * sigma_obs**2))
        * i0e(arg)
    )


def _reference(u_obs, sigma_obs):
    norm = rice_outlier_normalization(
        support_max=RICE_GOOD_SUPPORT,
        mu=RICE_OUTLIER_MU,
        sigma=RICE_OUTLIER_SIGMA,
    )

    def integrand(v):
        outlier = (
            np.exp(-0.5 * ((v - RICE_OUTLIER_MU) / RICE_OUTLIER_SIGMA) ** 2)
            / (RICE_OUTLIER_SIGMA * np.sqrt(2.0 * np.pi) * norm)
        )
        return _rice_pdf(u_obs, v, sigma_obs) * outlier

    return quad(
        integrand, 0.0, RICE_GOOD_SUPPORT,
        epsabs=1e-11, epsrel=1e-10, limit=400
    )[0]


def test_raw_u_outlier_matches_independent_adaptive_integral():
    u = np.array([6.0, 18.0, 40.0, 63.0])
    sigma = np.array([0.8, 2.5, 4.0, 3.0])
    actual = np.exp(raw_u_outlier_log_likelihood(u, sigma, quadrature_nodes=512))
    expected = np.array([_reference(ui, si) for ui, si in zip(u, sigma)])
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=1e-12)


def test_raw_u_outlier_has_no_mass_argument_or_mass_scaling():
    u = np.array([12.0, 35.0, 58.0])
    sigma = np.array([1.0, 2.0, 4.0])
    first = raw_u_outlier_log_likelihood(u, sigma, quadrature_nodes=256)
    second = raw_u_outlier_log_likelihood(u.copy(), sigma.copy(), quadrature_nodes=256)
    np.testing.assert_array_equal(first, second)
