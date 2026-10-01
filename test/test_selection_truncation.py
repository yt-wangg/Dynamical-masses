"""Tests for the u_obs/sigma_u > cut truncation normalization."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from binary_masses.hierarchical_metallicity import (
    RICE_GOOD_B,
    RICE_GOOD_C,
    RICE_GOOD_SUPPORT,
    RICE_GOOD_UC,
    RICE_OUTLIER_MU,
    RICE_OUTLIER_SIGMA,
    selection_pass_log_tables,
)

GRID = np.geomspace(0.25, 2.5, 7)


def _sample_good(rng, n):
    w = np.linspace(0.0, RICE_GOOD_SUPPORT, 40001)
    p = w * np.exp(-RICE_GOOD_B * w**2 - np.exp((w - RICE_GOOD_UC) / RICE_GOOD_C))
    c = np.cumsum(p)
    c /= c[-1]
    return np.interp(rng.random(n), c, w)


def _sample_bad(rng, n):
    out = np.empty(0)
    while out.size < n:
        x = rng.normal(RICE_OUTLIER_MU, RICE_OUTLIER_SIGMA, 2 * n)
        out = np.concatenate([out, x[(x >= 0) & (x <= RICE_GOOD_SUPPORT)]])
    return out[:n]


def test_pass_probability_matches_monte_carlo():
    rng = np.random.default_rng(0)
    sigma = np.array([0.5, 3.0, 12.0])
    log_good, log_bad = selection_pass_log_tables(sigma, GRID, cut=3.0)
    n = 400_000
    for i, sig in enumerate(sigma):
        for l in (0, 3, 6):
            v = GRID[l] * _sample_good(rng, n)
            u = np.hypot(v + sig * rng.normal(size=n), sig * rng.normal(size=n))
            assert abs(np.exp(log_good[i, l]) - np.mean(u > 3.0 * sig)) < 4e-3
        v = _sample_bad(rng, n)
        u = np.hypot(v + sig * rng.normal(size=n), sig * rng.normal(size=n))
        assert abs(np.exp(log_bad[i]) - np.mean(u > 3.0 * sig)) < 4e-3


def test_limits():
    sigma = np.array([1e-3, 50.0])
    log_good, log_bad = selection_pass_log_tables(sigma, GRID, cut=3.0)
    assert np.all(np.exp(log_good[0]) > 0.999)  # tiny noise: nothing is cut
    assert np.all(np.diff(log_good, axis=1) >= -1e-9)  # pass prob grows with mass
    assert np.all(log_good <= 1e-9) and np.all(log_bad <= 1e-9)
    tiny_cut, _ = selection_pass_log_tables(sigma, GRID, cut=1e-3)
    assert np.all(np.exp(tiny_cut) > 0.999)
