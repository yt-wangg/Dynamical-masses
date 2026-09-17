"""Focused tests for the M_G-window score diagnostic helpers.

Covers the failure paths the b0017cf audit identified: out-of-domain
interpolation must return -inf, responsibilities must be NaN-free at
non-finite nodes, posterior parameters must agree on draw counts, and the
grouped summary must conserve the total score.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import jax.numpy as jnp

from binary_masses.hierarchical_metallicity import lookup_interpolate
from examples.diagnose_mg_window_mass_score import (
    assert_group_conservation,
    check_draw_consistency,
    safe_responsibilities,
)


def test_lookup_interpolate_is_linear_inside_domain():
    grid = jnp.asarray(np.array([1.0, 2.0, 3.0, 4.0]))
    table = jnp.asarray(np.array([np.linspace(0.0, 3.0, 4)] * 2))
    values = jnp.asarray(np.array([[1.5], [2.25]]))
    out = np.asarray(lookup_interpolate(values, grid, table))
    assert np.allclose(out[:, 0], [0.5, 1.25])


def test_lookup_interpolate_returns_minus_inf_outside_domain():
    grid = jnp.asarray(np.array([1.0, 2.0, 3.0, 4.0]))
    table = jnp.asarray(np.array([np.linspace(0.0, 3.0, 4)] * 3))
    values = jnp.asarray(np.array([[0.999], [4.0001], [2.0]]))
    out = np.asarray(lookup_interpolate(values, grid, table))
    assert not np.isfinite(out[0, 0]) and out[0, 0] == -np.inf
    assert not np.isfinite(out[1, 0]) and out[1, 0] == -np.inf
    assert np.isclose(out[2, 0], 1.0)


def test_safe_responsibilities_are_nan_free_at_non_finite_nodes():
    log_a = jnp.asarray(np.array([-np.inf, -1.0]))
    log_b = jnp.asarray(np.array([-np.inf, -3.0]))
    log_cond = jnp.asarray(np.array([-np.inf, np.logaddexp(-1.0, -3.0)]))
    r_good, r_bad = safe_responsibilities(log_a, log_b, log_cond)
    r_good, r_bad = np.asarray(r_good), np.asarray(r_bad)
    assert np.all(np.isfinite(r_good)) and np.all(np.isfinite(r_bad))
    assert r_good[0] == 0.0 and r_bad[0] == 0.0
    assert np.isclose(r_good[1] + r_bad[1], 1.0)
    assert np.isclose(r_good[1], np.exp(-1.0 - np.logaddexp(-1.0, -3.0)))


def test_check_draw_consistency_rejects_mismatched_chains():
    consistent = {"c0": np.zeros(200), "a": np.zeros((200, 3))}
    check_draw_consistency(consistent)
    mismatched = {"c0": np.zeros(200), "a": np.zeros((180, 3))}
    with pytest.raises(ValueError, match="draw counts"):
        check_draw_consistency(mismatched)


def _summary(total, category_sums, responsibility_sums, quartile_sums):
    def block(value):
        return {"S10_linear_total": value}

    return {
        "total": block(total),
        "by_category": {name: block(value) for name, value in category_sums.items()},
        "by_outlier_responsibility": {
            name: block(value) for name, value in responsibility_sums.items()
        },
        "by_u_ratio_quartile": {
            name: block(value) for name, value in quartile_sums.items()
        },
    }


def test_group_conservation_accepts_consistent_summary():
    summary = _summary(
        5.7,
        {"both": 2.5, "brighter_only": 0.5, "fainter_only": 2.7, "none": 0.0},
        {"p_out<0.5": -7.25, "p_out>=0.5": 12.95},
        {"Q1": -15.3, "Q2": -7.6, "Q3": 4.3, "Q4": 24.3},
    )
    assert_group_conservation(summary)


def test_group_conservation_rejects_inconsistent_summary():
    summary = _summary(
        5.7,
        {"both": 2.5, "brighter_only": 0.5, "fainter_only": 2.7, "none": 1.0},
        {"p_out<0.5": -7.25, "p_out>=0.5": 12.95},
        {"Q1": -15.3, "Q2": -7.6, "Q3": 4.3, "Q4": 24.3},
    )
    with pytest.raises(AssertionError, match="Window-membership"):
        assert_group_conservation(summary)
