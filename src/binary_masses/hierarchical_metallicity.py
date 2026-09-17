"""T8 hierarchical metallicity calibration and dynamical-MLR model.

This module is intentionally independent of :mod:`differencepoly_feh`.  Stage
one combines two component metallicity measurements and two CMD positions on a
fixed, global PARSEC metallicity grid.  Stage two consumes the resulting
per-system grid probabilities and fits a PARSEC-relative, hard-monotone tensor
B-spline mass surface.  The older three-knot prototype remains available only
for reproducing T7 results.

The public classes in this file are deliberately narrow test interfaces; the
existing DifferencePolyFehMLR implementation is not modified or subclassed.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy.special import expit, gammaln, logsumexp, ndtr
from scipy.integrate import quad
from scipy.special import i0e

from .differencepoly_feh import IsochroneMassSurfaceModel
from .jax_utils import _configure_jax_gpu_fallback


DEFAULT_Z_GRID = np.linspace(-1.0, 0.6, 81, dtype=np.float64)
DEFAULT_MG_KNOTS = np.array([3.5, 8.5, 13.5], dtype=np.float64)
DEFAULT_T8_CMD_MIN_TEFF = 4000.0
DEFAULT_T8_KNOT_X = np.array(
    [3.5, 3.5, 3.5, 3.5, 5.5, 7.5, 9.5, 11.5, 13.5, 13.5, 13.5, 13.5],
    dtype=np.float64,
)
DEFAULT_T8_KNOT_Z = np.array(
    [-1.0, -1.0, -1.0, -1.0, 0.6, 0.6, 0.6, 0.6], dtype=np.float64
)

# Plug-in calibration from FINAL_FEH_JCAPS_STUDENT_T_WORKFLOW_2026-09-10.md.
# The pair-difference fit determines only the relative magnitude trend and the
# measurement-shape parameters; the absolute offset is inferred below from CMD.
JCAPS_BIAS_MG_KNOTS = np.array([3.5, 6.0, 8.5, 11.0, 13.5], dtype=np.float64)
JCAPS_BIAS_VALUES = np.array(
    [
        0.11212849827583429,
        0.04987760289606098,
        0.0,
        -0.3103054232731345,
        -0.5332038779443041,
    ],
    dtype=np.float64,
)
JCAPS_EXTRA_SCALE = 0.04646502063846824
JCAPS_STUDENT_DF = 2.5236494464371377
RICE_GOOD_A = 5.434e-3
RICE_GOOD_B = 2.544e-3
RICE_GOOD_UC = 35.67
RICE_GOOD_C = 3.100
RICE_GOOD_SUPPORT = 80.0
RICE_OUTLIER_MU = 40.0
RICE_OUTLIER_SIGMA = 13.0
# Integral of the raw good basis over its support.  The T8.1 revision divides
# the good basis by this value so each mixture component integrates to one.
RICE_GOOD_BASIS_RAW_INTEGRAL = 0.9978600946
# The T8.1 normalization makes f a branch probability of the DEFINED mixture;
# calling it a physical contamination fraction additionally requires the
# contamination and selection model to match reality.
MIXTURE_SEMANTICS = (
    "normalized mixture probability; physical interpretation depends on the "
    "contamination and selection model"
)


def set_good_shape_constants(*, B, C, uc, A=None):
    """Override the fixed good-basis shape constants for shape-sensitivity runs.

    After T8.1 the density is normalized to unit integral, so ``A`` is absorbed
    by the normalization and only (B, C, uc) move the shape.  The raw support
    integral is recomputed for the new constants.  Overrides must be applied
    before the dynamics lookup is built; lookup metadata records the constants
    and schema validation rejects any later mismatch.
    """
    global RICE_GOOD_A, RICE_GOOD_B, RICE_GOOD_UC, RICE_GOOD_C, RICE_GOOD_BASIS_RAW_INTEGRAL
    if not (B > 0 and C > 0 and uc > 0):
        raise ValueError("Good-shape constants B, C, uc must be positive.")
    if A is None:
        A = RICE_GOOD_A
    raw = quad(
        lambda w: A * w * np.exp(-(B * w * w + np.exp((w - uc) / C))),
        0.0, RICE_GOOD_SUPPORT, limit=400,
    )[0]
    if not (raw > 0 and np.isfinite(raw)):
        raise ValueError("The overridden good basis has no usable normalization.")
    RICE_GOOD_A = float(A)
    RICE_GOOD_B = float(B)
    RICE_GOOD_UC = float(uc)
    RICE_GOOD_C = float(C)
    RICE_GOOD_BASIS_RAW_INTEGRAL = float(raw)


def rice_outlier_normalization(*, support_max=RICE_GOOD_SUPPORT, mu=RICE_OUTLIER_MU, sigma=RICE_OUTLIER_SIGMA):
    """Normalization of the fixed outlier Gaussian on ``[0, support_max]``."""
    return float(ndtr((float(support_max) - float(mu)) / float(sigma)) - ndtr(-float(mu) / float(sigma)))


def rice_good_raw(tilde_u):
    """Good-component density in ``w`` space, normalized to unit support integral (T8.1)."""
    tilde_u = np.asarray(tilde_u, dtype=np.float64)
    return (
        RICE_GOOD_A
        / RICE_GOOD_BASIS_RAW_INTEGRAL
        * tilde_u
        * np.exp(-(RICE_GOOD_B * tilde_u**2 + np.exp((tilde_u - RICE_GOOD_UC) / RICE_GOOD_C)))
        * ((tilde_u > 0) & (tilde_u <= RICE_GOOD_SUPPORT))
    )


def _rice_component_log_density(value, *, component, support_max, outlier_u0=RICE_OUTLIER_MU, outlier_sigma=RICE_OUTLIER_SIGMA):
    value = np.asarray(value, dtype=np.float64)
    if component == "good":
        result = (
            np.log(RICE_GOOD_A)
            - np.log(RICE_GOOD_BASIS_RAW_INTEGRAL)
            + np.log(np.maximum(value, np.finfo(np.float64).tiny))
            - RICE_GOOD_B * value**2
            - np.exp((value - RICE_GOOD_UC) / RICE_GOOD_C)
        )
        return np.where((value > 0.0) & (value <= support_max), result, -np.inf)
    if component == "bad":
        norm = rice_outlier_normalization(support_max=support_max, mu=outlier_u0, sigma=outlier_sigma)
        result = (
            -0.5 * ((value - float(outlier_u0)) / float(outlier_sigma)) ** 2
            - np.log(float(outlier_sigma) * np.sqrt(2.0 * np.pi) * norm)
        )
        return np.where((value >= 0.0) & (value <= support_max), result, -np.inf)
    raise ValueError("component must be 'good' or 'bad'.")


def _scaled_piecewise_log_quad(log_integrand, boundaries):
    """Integrate exp(log_integrand) without losing sub-floor references."""
    boundaries = np.unique(np.asarray(boundaries, dtype=np.float64))
    total = 0.0
    for lower, upper in zip(boundaries[:-1], boundaries[1:]):
        if not upper > lower:
            continue
        probes = np.linspace(lower, upper, 65)
        log_scale = float(np.max(log_integrand(probes)))
        if not np.isfinite(log_scale):
            continue
        scaled = quad(
            lambda coordinate: float(np.exp(log_integrand(coordinate) - log_scale)),
            float(lower), float(upper), epsabs=1e-12, epsrel=1e-10, limit=200,
        )[0]
        total += float(np.exp(log_scale) * scaled)
    return total


def rice_component_reference_integral(u_obs, u_sigma, sqrt_mtot, *, component="good", support_max=RICE_GOOD_SUPPORT, outlier_u0=RICE_OUTLIER_MU, outlier_sigma=RICE_OUTLIER_SIGMA):
    """Reference adaptive Rice integral used by lookup convergence checks."""
    u_obs, u_sigma, sqrt_mtot = map(float, (u_obs, u_sigma, sqrt_mtot))
    if u_obs <= 0 or u_sigma <= 0 or sqrt_mtot <= 0:
        raise ValueError("u_obs, u_sigma, and sqrt_mtot must be strictly positive.")
    def log_integrand(v):
        v = np.asarray(v, dtype=np.float64)
        argument = u_obs * v / u_sigma**2
        # Since u_obs and v are non-negative, abs(argument)=argument and the
        # two large exponential terms combine exactly to -(u_obs-v)^2/(2s^2).
        # Use that form to avoid catastrophic cancellation for narrow kernels.
        log_rice = (
            np.log(u_obs / u_sigma**2)
            - (u_obs - v) ** 2 / (2 * u_sigma**2)
            + np.log(i0e(argument))
        )
        return (
            log_rice
            + _rice_component_log_density(
                v / sqrt_mtot, component=component, support_max=support_max,
                outlier_u0=outlier_u0, outlier_sigma=outlier_sigma
            )
            - np.log(sqrt_mtot)
        )
    upper = sqrt_mtot * float(support_max)
    split_velocity = np.clip(
        u_obs + u_sigma * np.asarray([-14, -10, -6, -3, 0, 3, 6, 10, 14]),
        0.0, upper,
    )
    return _scaled_piecewise_log_quad(
        log_integrand, np.concatenate(([0.0], split_velocity, [upper]))
    )


def rice_component_reference_integral_tilde(u_obs, u_sigma, sqrt_mtot, *, component="good", support_max=RICE_GOOD_SUPPORT, outlier_u0=RICE_OUTLIER_MU, outlier_sigma=RICE_OUTLIER_SIGMA):
    """The same reference integral in ``tilde_u`` coordinates (no Jacobian)."""
    u_obs, u_sigma, sqrt_mtot = map(float, (u_obs, u_sigma, sqrt_mtot))
    def log_integrand(tilde):
        tilde = np.asarray(tilde, dtype=np.float64)
        velocity = sqrt_mtot * tilde
        argument = u_obs * velocity / u_sigma**2
        log_rice = (
            np.log(u_obs / u_sigma**2)
            - (u_obs - velocity) ** 2 / (2 * u_sigma**2)
            + np.log(i0e(argument))
        )
        return log_rice + _rice_component_log_density(
            tilde, component=component, support_max=support_max,
            outlier_u0=outlier_u0, outlier_sigma=outlier_sigma
        )
    split_tilde = np.clip(
        (u_obs + u_sigma * np.asarray([-14, -10, -6, -3, 0, 3, 6, 10, 14]))
        / sqrt_mtot,
        0.0, float(support_max),
    )
    return _scaled_piecewise_log_quad(
        log_integrand,
        np.concatenate(([0.0], split_tilde, [float(support_max)])),
    )


def compare_rice_integrations(*, u_obs, u_sigma, sqrt_mtot, velocity_sigma_extent=10.0, velocity_quadrature_nodes=64):
    """Compare local-window quadrature with adaptive reference integrals."""
    rows = np.asarray([0], dtype=np.int64)
    lookup = DynamicsLikelihoodLookup.precompute(
        row_indices=rows, u=np.asarray([u_obs]), u_sigma=np.asarray([u_sigma]),
        sqrt_mtot_min=float(sqrt_mtot), sqrt_mtot_max=float(sqrt_mtot) * (1 + 1e-8),
        sqrt_mtot_points=2, velocity_sigma_extent=velocity_sigma_extent,
        velocity_quadrature_nodes=velocity_quadrature_nodes, system_chunk=1,
    )
    result = {}
    for component in ("good", "bad"):
        reference = rice_component_reference_integral(u_obs, u_sigma, sqrt_mtot, component=component)
        approx = float(np.exp(lookup.log_good[0, 0] if component == "good" else lookup.log_bad[0, 0]))
        result[component] = {"reference": reference, "quadrature": approx, "abs_error": abs(approx - reference)}
    return result


def assess_rice_lookup_convergence(
    *,
    row_indices,
    u,
    u_sigma,
    sqrt_mtot_min=0.25,
    sqrt_mtot_max=2.5,
    sample_systems=32,
    scale_points=16,
    velocity_sigma_extent=10.0,
    velocity_quadrature_nodes=64,
    tolerance_log_likelihood=1e-3,
    floor=1e-30,
    outlier_u0=RICE_OUTLIER_MU,
    outlier_sigma=RICE_OUTLIER_SIGMA,
):
    """Check the adopted local Rice quadrature against stricter references.

    Systems are selected deterministically across the observed fractional-error
    range.  The production ``K=10, R=64`` table is compared with ``K=14,
    R=128`` and with adaptive integrations in both integration coordinates.
    Floor hits are classified explicitly.  A floor is safe only when the
    float64 adaptive reference is itself no larger than the stored floor;
    otherwise the production table has hidden a resolvable likelihood and the
    convergence gate fails.
    """
    rows = np.asarray(row_indices, dtype=np.int64)
    u = _as_native_f64(u, name="u", ndim=1)
    u_sigma = _as_native_f64(u_sigma, name="u_sigma", ndim=1)
    if rows.shape != u.shape or u.shape != u_sigma.shape:
        raise ValueError("row_indices, u, and u_sigma must have identical shapes.")
    count = min(int(sample_systems), rows.size)
    if count < 1 or int(scale_points) < 2:
        raise ValueError("Convergence checks require systems and at least two scale nodes.")
    order = np.argsort(u_sigma / u, kind="stable")
    selected = np.unique(
        order[np.linspace(0, rows.size - 1, count, dtype=np.int64)]
    )
    kwargs = {
        "row_indices": rows[selected],
        "u": u[selected],
        "u_sigma": u_sigma[selected],
        "sqrt_mtot_min": float(sqrt_mtot_min),
        "sqrt_mtot_max": float(sqrt_mtot_max),
        "sqrt_mtot_points": int(scale_points),
        "system_chunk": max(1, selected.size),
        "floor": float(floor),
        "outlier_u0": float(outlier_u0),
        "outlier_sigma": float(outlier_sigma),
    }
    reference_extent = max(14.0, float(velocity_sigma_extent) + 4.0)
    reference_nodes = max(128, 2 * int(velocity_quadrature_nodes))
    adopted = DynamicsLikelihoodLookup.precompute(
        **kwargs,
        velocity_sigma_extent=float(velocity_sigma_extent),
        velocity_quadrature_nodes=int(velocity_quadrature_nodes),
    )
    stricter = DynamicsLikelihoodLookup.precompute(
        **kwargs,
        velocity_sigma_extent=reference_extent,
        velocity_quadrature_nodes=reference_nodes,
    )
    scales = adopted.sqrt_mtot_grid
    component_results = {}
    passed = True
    for component, adopted_log, stricter_log in (
        ("good", adopted.log_good, stricter.log_good),
        ("outlier", adopted.log_bad, stricter.log_bad),
    ):
        reference_v = np.empty_like(adopted_log, dtype=np.float64)
        reference_tilde = np.empty_like(adopted_log, dtype=np.float64)
        reference_name = "good" if component == "good" else "bad"
        for local_row, source_row in enumerate(selected):
            for scale_index, scale in enumerate(scales):
                reference_v[local_row, scale_index] = rice_component_reference_integral(
                    u[source_row], u_sigma[source_row], scale, component=reference_name,
                    outlier_u0=outlier_u0, outlier_sigma=outlier_sigma,
                )
                reference_tilde[local_row, scale_index] = (
                    rice_component_reference_integral_tilde(
                        u[source_row],
                        u_sigma[source_row],
                        scale,
                        component=reference_name,
                        outlier_u0=outlier_u0, outlier_sigma=outlier_sigma,
                    )
                )
        floor_value = float(floor)
        floor_reference_tolerance = floor_value * (1.0 + 1e-6)
        floor_log_tolerance = np.log(floor_value) + 1e-5
        adopted_at_floor = adopted_log <= floor_log_tolerance
        stricter_at_floor = stricter_log <= floor_log_tolerance
        reference_at_floor = reference_v <= floor_reference_tolerance
        tilde_at_floor = reference_tilde <= floor_reference_tolerance
        unsafe = (
            (adopted_at_floor != reference_at_floor)
            | (stricter_at_floor != reference_at_floor)
            | (reference_at_floor != tilde_at_floor)
        )
        safe_floor = adopted_at_floor & stricter_at_floor & reference_at_floor & tilde_at_floor
        valid = (
            ~adopted_at_floor
            & ~stricter_at_floor
            & ~reference_at_floor
            & ~tilde_at_floor
        )

        def locations(mask):
            locations_out = []
            for local_row, scale_index in np.argwhere(mask):
                locations_out.append(
                    {
                        "row_index": int(rows[selected[local_row]]),
                        "sqrt_mtot": float(scales[scale_index]),
                    }
                )
            return locations_out

        errors = {
            "adopted_vs_k14_r128_max_abs_log": 0.0,
            "adopted_vs_adaptive_max_abs_log": 0.0,
            "k14_r128_vs_adaptive_max_abs_log": 0.0,
            "adaptive_coordinate_max_abs_log": 0.0,
            "non_floor_comparisons": int(np.count_nonzero(valid)),
            "safe_floor_count": int(np.count_nonzero(safe_floor)),
            "safe_floor_locations": locations(safe_floor),
            "unsafe_floor_count": int(np.count_nonzero(unsafe)),
            "unsafe_floor_locations": locations(unsafe),
        }
        if np.any(valid):
            log_reference_v = np.log(reference_v[valid])
            log_reference_tilde = np.log(reference_tilde[valid])
            errors.update(
                {
                    "adopted_vs_k14_r128_max_abs_log": float(
                        np.max(np.abs(adopted_log[valid] - stricter_log[valid]))
                    ),
                    "adopted_vs_adaptive_max_abs_log": float(
                        np.max(np.abs(adopted_log[valid] - log_reference_v))
                    ),
                    "k14_r128_vs_adaptive_max_abs_log": float(
                        np.max(np.abs(stricter_log[valid] - log_reference_v))
                    ),
                    "adaptive_coordinate_max_abs_log": float(
                        np.max(np.abs(log_reference_v - log_reference_tilde))
                    ),
                }
            )
        component_passed = all(
            value <= float(tolerance_log_likelihood)
            for key, value in errors.items()
            if key.endswith("_max_abs_log")
        ) and errors["unsafe_floor_count"] == 0
        errors["passed"] = bool(component_passed)
        component_results[component] = errors
        passed &= component_passed
    return {
        "passed": bool(passed),
        "tolerance_max_abs_log_likelihood": float(tolerance_log_likelihood),
        "source_data_digest": array_digest(rows, u, u_sigma),
        "floor": float(floor),
        "outlier_u0": float(outlier_u0),
        "outlier_sigma": float(outlier_sigma),
        "sample_systems_requested": int(sample_systems),
        "sample_system_count": int(selected.size),
        "scale_points_requested": int(scale_points),
        "selected_row_indices": rows[selected].tolist(),
        "selected_row_indices_digest": array_digest(rows[selected]),
        "scale_grid": scales.tolist(),
        "adopted": {
            "velocity_sigma_extent": float(velocity_sigma_extent),
            "quadrature_nodes": int(velocity_quadrature_nodes),
        },
        "stricter": {
            "velocity_sigma_extent": reference_extent,
            "quadrature_nodes": reference_nodes,
        },
        "components": component_results,
    }


def _as_native_f64(value, *, name: str, ndim: Optional[int] = None) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if ndim is not None and array.ndim != ndim:
        raise ValueError(f"{name} must have ndim={ndim}; got shape {array.shape}.")
    return array


def trapezoid_weights(grid: Sequence[float]) -> np.ndarray:
    """Return integration weights for a strictly increasing one-dimensional grid."""
    grid = _as_native_f64(grid, name="grid", ndim=1)
    if grid.size < 2 or np.any(np.diff(grid) <= 0):
        raise ValueError("grid must contain at least two strictly increasing values.")
    weights = np.empty_like(grid)
    weights[0] = 0.5 * (grid[1] - grid[0])
    weights[-1] = 0.5 * (grid[-1] - grid[-2])
    weights[1:-1] = 0.5 * (grid[2:] - grid[:-2])
    return weights


def color_uncertainty_from_flux_snr(bp_flux_over_error, rp_flux_over_error) -> np.ndarray:
    """Propagate Gaia BP/RP flux S/N values into a BP-RP magnitude error."""
    bp_snr = _as_native_f64(bp_flux_over_error, name="bp_flux_over_error")
    rp_snr = _as_native_f64(rp_flux_over_error, name="rp_flux_over_error")
    bp_snr, rp_snr = np.broadcast_arrays(bp_snr, rp_snr)
    if np.any(~np.isfinite(bp_snr)) or np.any(~np.isfinite(rp_snr)):
        raise ValueError("BP/RP flux-over-error values must be finite.")
    if np.any(bp_snr <= 0) or np.any(rp_snr <= 0):
        raise ValueError("BP/RP flux-over-error values must be strictly positive.")
    magnitude_factor = 2.5 / np.log(10.0)
    return magnitude_factor * np.sqrt(bp_snr ** -2 + rp_snr ** -2)


def student_t_logpdf_np(x, *, df: float, loc, scale) -> np.ndarray:
    """Small vectorized Student-t log density used by posterior post-processing."""
    x = np.asarray(x, dtype=np.float64)
    loc = np.asarray(loc, dtype=np.float64)
    scale = np.asarray(scale, dtype=np.float64)
    scale = np.maximum(scale, 1e-12)
    standardized = (x - loc) / scale
    constant = gammaln((df + 1.0) / 2.0) - gammaln(df / 2.0)
    constant -= 0.5 * np.log(df * np.pi) + np.log(scale)
    return constant - 0.5 * (df + 1.0) * np.log1p(standardized**2 / df)


def jcaps_magnitude_bias(absg) -> np.ndarray:
    """Return the fixed T6d JCAPS relative metallicity bias in dex."""
    return np.interp(
        np.asarray(absg, dtype=np.float64),
        JCAPS_BIAS_MG_KNOTS,
        JCAPS_BIAS_VALUES,
    )


def normalize_log_weights(log_weights: np.ndarray, axis: int = -1) -> np.ndarray:
    """Normalize log weights, raising if an entire row is non-finite."""
    log_weights = np.asarray(log_weights, dtype=np.float64)
    normalizer = logsumexp(log_weights, axis=axis, keepdims=True)
    if np.any(~np.isfinite(normalizer)):
        raise ValueError("At least one log-weight row has no finite support.")
    return np.exp(log_weights - normalizer)


def marginalize_dynamics(log_conditional: np.ndarray, z_probabilities: np.ndarray) -> np.ndarray:
    """Marginalize a dynamics-only log likelihood over normalized P(Z)."""
    log_conditional = np.asarray(log_conditional, dtype=np.float64)
    z_probabilities = np.asarray(z_probabilities, dtype=np.float64)
    if log_conditional.shape != z_probabilities.shape:
        raise ValueError(
            "log_conditional and z_probabilities must have identical shapes; "
            f"got {log_conditional.shape} and {z_probabilities.shape}."
        )
    if np.any(z_probabilities < 0):
        raise ValueError("z_probabilities cannot be negative.")
    sums = np.sum(z_probabilities, axis=-1)
    if not np.allclose(sums, 1.0, atol=1e-6):
        raise ValueError("Every z_probabilities row must sum to one.")
    return logsumexp(np.log(np.maximum(z_probabilities, 1e-300)) + log_conditional, axis=-1)


def array_digest(*arrays) -> str:
    """Stable SHA-256 digest for row/data validation across model stages."""
    digest = hashlib.sha256()
    for value in arrays:
        array = np.ascontiguousarray(value)
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(array.view(np.uint8))
    return digest.hexdigest()


class IsochroneColorSurfaceModel:
    """Bilinear PARSEC surfaces for BP-RP color and, when available, Teff."""

    def __init__(self, absg_grid, mh_grid, color_grid, *, teff_grid=None):
        self.absg_grid_np = _as_native_f64(absg_grid, name="absg_grid", ndim=1)
        self.mh_grid_np = _as_native_f64(mh_grid, name="mh_grid", ndim=1)
        self.color_grid_np = _as_native_f64(color_grid, name="color_grid", ndim=2)
        if self.color_grid_np.shape != (self.mh_grid_np.size, self.absg_grid_np.size):
            raise ValueError(
                "color_grid must have shape (n_mh, n_absg); got "
                f"{self.color_grid_np.shape}."
            )
        if np.any(np.diff(self.absg_grid_np) <= 0) or np.any(np.diff(self.mh_grid_np) <= 0):
            raise ValueError("absg_grid and mh_grid must be strictly increasing.")
        if np.any(~np.isfinite(self.color_grid_np)):
            raise ValueError("color_grid must contain only finite values.")
        self.teff_grid_np = None
        if teff_grid is not None:
            self.teff_grid_np = _as_native_f64(teff_grid, name="teff_grid", ndim=2)
            if self.teff_grid_np.shape != self.color_grid_np.shape:
                raise ValueError(
                    "teff_grid must have the same shape as color_grid; got "
                    f"{self.teff_grid_np.shape}."
                )
            if np.any(~np.isfinite(self.teff_grid_np)) or np.any(self.teff_grid_np <= 0):
                raise ValueError("teff_grid must contain only finite positive values.")
        self._jax_cache = None

    @classmethod
    def from_parsec_csv(
        cls,
        csv_path,
        *,
        absg_grid: Optional[Sequence[float]] = None,
        logage_min: float = 9.3,
        logage_max: float = 10.0,
        absg_min: float = 3.5,
        absg_max: float = 13.5,
    ) -> "IsochroneColorSurfaceModel":
        """Build the color surface in memory using the existing grid selection."""
        try:
            from astropy.table import Table
            from scipy.interpolate import UnivariateSpline
        except ImportError as exc:  # pragma: no cover - project dependencies
            raise ImportError("Astropy and SciPy are required to read the PARSEC table.") from exc

        table = Table.read(str(csv_path))
        required = {"logAge", "Gmag", "MH", "G_BPmag", "G_RPmag", "logTe"}
        missing = sorted(required.difference(table.colnames))
        if missing:
            raise ValueError(f"PARSEC table is missing columns: {missing}.")

        logage = np.asarray(table["logAge"], dtype=float)
        gmag = np.asarray(table["Gmag"], dtype=float)
        mask = (
            (logage > float(logage_min))
            & (logage < float(logage_max))
            & (gmag > 3.0)
            & (gmag < 15.0)
        )
        table = table[mask]
        if len(table) == 0:
            raise ValueError("No PARSEC rows remain after the age/magnitude selection.")

        if absg_grid is None:
            absg_grid = np.linspace(absg_min, absg_max, 500)
        absg_grid = _as_native_f64(absg_grid, name="absg_grid", ndim=1)
        if absg_grid[0] < absg_min - 1e-12 or absg_grid[-1] > absg_max + 1e-12:
            raise ValueError("absg_grid lies outside the requested color-surface domain.")

        mh_values = np.asarray(table["MH"], dtype=float)
        mh_grid = np.unique(mh_values)
        curves = []
        teff_curves = []
        for mh in mh_grid:
            selected = np.isclose(mh_values, mh)
            g = np.asarray(table["Gmag"][selected], dtype=float)
            color = (
                np.asarray(table["G_BPmag"][selected], dtype=float)
                - np.asarray(table["G_RPmag"][selected], dtype=float)
            )
            logte = np.asarray(table["logTe"][selected], dtype=float)
            order = np.argsort(g)
            g, color, logte = g[order], color[order], logte[order]
            unique_g, inverse = np.unique(g, return_inverse=True)
            mean_color = np.array(
                [np.mean(color[inverse == index]) for index in range(unique_g.size)]
            )
            mean_logte = np.array(
                [np.mean(logte[inverse == index]) for index in range(unique_g.size)]
            )
            if unique_g.size < 2:
                raise ValueError(f"Too few PARSEC points at MH={mh:+.2f}.")
            degree = min(3, unique_g.size - 1)
            spline = UnivariateSpline(unique_g, mean_color, k=degree, s=0.001, ext="const")
            curves.append(np.asarray(spline(absg_grid), dtype=np.float64))
            # Linear temperature interpolation avoids cubic overshoot near the
            # temperature threshold used to select trusted CMD constraints.
            interpolated_logte = np.interp(
                absg_grid,
                unique_g,
                mean_logte,
                left=mean_logte[0],
                right=mean_logte[-1],
            )
            teff_curves.append(np.power(10.0, interpolated_logte))
        return cls(
            absg_grid,
            mh_grid,
            np.vstack(curves),
            teff_grid=np.vstack(teff_curves),
        )

    def color_from_absg_mh(self, absg, mh) -> np.ndarray:
        absg = np.asarray(absg, dtype=np.float64)
        mh = np.asarray(mh, dtype=np.float64)
        absg, mh = np.broadcast_arrays(absg, mh)
        absg_c = np.clip(absg, self.absg_grid_np[0], self.absg_grid_np[-1])
        mh_c = np.clip(mh, self.mh_grid_np[0], self.mh_grid_np[-1])
        ia1 = np.clip(np.searchsorted(self.absg_grid_np, absg_c, side="right"), 1, self.absg_grid_np.size - 1)
        ia0 = ia1 - 1
        im1 = np.clip(np.searchsorted(self.mh_grid_np, mh_c, side="right"), 1, self.mh_grid_np.size - 1)
        im0 = im1 - 1
        a0, a1 = self.absg_grid_np[ia0], self.absg_grid_np[ia1]
        z0, z1 = self.mh_grid_np[im0], self.mh_grid_np[im1]
        wa = (absg_c - a0) / np.maximum(a1 - a0, 1e-12)
        wz = (mh_c - z0) / np.maximum(z1 - z0, 1e-12)
        c00 = self.color_grid_np[im0, ia0]
        c01 = self.color_grid_np[im0, ia1]
        c10 = self.color_grid_np[im1, ia0]
        c11 = self.color_grid_np[im1, ia1]
        return (c00 * (1.0 - wa) + c01 * wa) * (1.0 - wz) + (
            c10 * (1.0 - wa) + c11 * wa
        ) * wz

    def teff_from_absg_mh(self, absg, mh) -> np.ndarray:
        """Interpolate PARSEC effective temperature in K at ``(M_G, [M/H])``."""
        if self.teff_grid_np is None:
            raise ValueError("This color surface does not include a PARSEC temperature grid.")
        absg = np.asarray(absg, dtype=np.float64)
        mh = np.asarray(mh, dtype=np.float64)
        absg, mh = np.broadcast_arrays(absg, mh)
        absg_c = np.clip(absg, self.absg_grid_np[0], self.absg_grid_np[-1])
        mh_c = np.clip(mh, self.mh_grid_np[0], self.mh_grid_np[-1])
        ia1 = np.clip(
            np.searchsorted(self.absg_grid_np, absg_c, side="right"),
            1,
            self.absg_grid_np.size - 1,
        )
        ia0 = ia1 - 1
        im1 = np.clip(
            np.searchsorted(self.mh_grid_np, mh_c, side="right"),
            1,
            self.mh_grid_np.size - 1,
        )
        im0 = im1 - 1
        a0, a1 = self.absg_grid_np[ia0], self.absg_grid_np[ia1]
        z0, z1 = self.mh_grid_np[im0], self.mh_grid_np[im1]
        wa = (absg_c - a0) / np.maximum(a1 - a0, 1e-12)
        wz = (mh_c - z0) / np.maximum(z1 - z0, 1e-12)
        t00 = self.teff_grid_np[im0, ia0]
        t01 = self.teff_grid_np[im0, ia1]
        t10 = self.teff_grid_np[im1, ia0]
        t11 = self.teff_grid_np[im1, ia1]
        return (t00 * (1.0 - wa) + t01 * wa) * (1.0 - wz) + (
            t10 * (1.0 - wa) + t11 * wa
        ) * wz

    def color_from_absg_mh_jax(self, absg, mh):
        import jax.numpy as jnp

        if self._jax_cache is None:
            self._jax_cache = (
                jnp.asarray(self.absg_grid_np),
                jnp.asarray(self.mh_grid_np),
                jnp.asarray(self.color_grid_np),
            )
        absg_grid, mh_grid, color_grid = self._jax_cache
        absg, mh = jnp.broadcast_arrays(jnp.asarray(absg), jnp.asarray(mh))
        absg_c = jnp.clip(absg, absg_grid[0], absg_grid[-1])
        mh_c = jnp.clip(mh, mh_grid[0], mh_grid[-1])
        ia1 = jnp.clip(jnp.searchsorted(absg_grid, absg_c, side="right"), 1, absg_grid.size - 1)
        ia0 = ia1 - 1
        im1 = jnp.clip(jnp.searchsorted(mh_grid, mh_c, side="right"), 1, mh_grid.size - 1)
        im0 = im1 - 1
        a0, a1 = absg_grid[ia0], absg_grid[ia1]
        z0, z1 = mh_grid[im0], mh_grid[im1]
        wa = (absg_c - a0) / jnp.maximum(a1 - a0, 1e-12)
        wz = (mh_c - z0) / jnp.maximum(z1 - z0, 1e-12)
        c00 = color_grid[im0, ia0]
        c01 = color_grid[im0, ia1]
        c10 = color_grid[im1, ia0]
        c11 = color_grid[im1, ia1]
        return (c00 * (1.0 - wa) + c01 * wa) * (1.0 - wz) + (
            c10 * (1.0 - wa) + c11 * wa
        ) * wz


@dataclass(frozen=True)
class MetallicityPosteriorGrid:
    """Portable stage-one output consumed by the MLR stage."""

    row_indices: np.ndarray
    z_grid: np.ndarray
    probabilities: np.ndarray
    z_quantiles: np.ndarray
    bad_probabilities: np.ndarray
    metadata: Mapping[str, object]

    def validate(self) -> None:
        rows = np.asarray(self.row_indices)
        grid = np.asarray(self.z_grid)
        probs = np.asarray(self.probabilities)
        if rows.ndim != 1 or grid.ndim != 1 or probs.shape != (rows.size, grid.size):
            raise ValueError("MetallicityPosteriorGrid has inconsistent array shapes.")
        if np.any(probs < 0) or not np.allclose(probs.sum(axis=1), 1.0, atol=2e-5):
            raise ValueError("Metallicity posterior rows must be non-negative and normalized.")
        if np.asarray(self.z_quantiles).shape != (rows.size, 3):
            raise ValueError("z_quantiles must have shape (n_systems, 3).")
        if np.asarray(self.bad_probabilities).shape != (rows.size, 2):
            raise ValueError("bad_probabilities must have shape (n_systems, 2).")

    def save(self, path) -> Path:
        self.validate()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            row_indices=np.asarray(self.row_indices, dtype=np.int64),
            z_grid=np.asarray(self.z_grid, dtype=np.float64),
            probabilities=np.asarray(self.probabilities, dtype=np.float32),
            z_quantiles=np.asarray(self.z_quantiles, dtype=np.float32),
            bad_probabilities=np.asarray(self.bad_probabilities, dtype=np.float32),
            metadata_json=np.asarray(json.dumps(dict(self.metadata), sort_keys=True)),
        )
        return path

    @classmethod
    def load(cls, path) -> "MetallicityPosteriorGrid":
        with np.load(path, allow_pickle=False) as saved:
            result = cls(
                row_indices=np.asarray(saved["row_indices"], dtype=np.int64),
                z_grid=np.asarray(saved["z_grid"], dtype=np.float64),
                probabilities=np.asarray(saved["probabilities"], dtype=np.float64),
                z_quantiles=np.asarray(saved["z_quantiles"], dtype=np.float64),
                bad_probabilities=np.asarray(saved["bad_probabilities"], dtype=np.float64),
                metadata=json.loads(str(saved["metadata_json"].item())),
            )
        result.validate()
        return result


class HierarchicalMetallicityCalibrator:
    """Stage-one T6d JCAPS observation model plus a warm-PARSEC CMD anchor.

    The relative magnitude bias, per-star extra Student-t scale, and degrees of
    freedom are fixed to the pair-difference calibration documented in T6d.
    Only the absolute JCAPS-minus-PARSEC offset, CMD scatter, and population
    prior weights are inferred here.
    """

    def __init__(
        self,
        color_surface: IsochroneColorSurfaceModel,
        *,
        z_grid: Sequence[float] = DEFAULT_Z_GRID,
        population_components: int = 6,
        cmd_min_teff: Optional[float] = DEFAULT_T8_CMD_MIN_TEFF,
        bias_mg_knots: Sequence[float] = JCAPS_BIAS_MG_KNOTS,
        bias_values: Sequence[float] = JCAPS_BIAS_VALUES,
        extra_scale: float = JCAPS_EXTRA_SCALE,
        student_df: float = JCAPS_STUDENT_DF,
    ):
        self.color_surface = color_surface
        self.z_grid = _as_native_f64(z_grid, name="z_grid", ndim=1)
        self.grid_weights = trapezoid_weights(self.z_grid)
        self.cmd_min_teff = None if cmd_min_teff is None else float(cmd_min_teff)
        if self.cmd_min_teff is not None and self.cmd_min_teff <= 0:
            raise ValueError("cmd_min_teff must be positive when supplied.")
        self.bias_mg_knots = _as_native_f64(
            bias_mg_knots, name="bias_mg_knots", ndim=1
        )
        self.bias_values = _as_native_f64(bias_values, name="bias_values", ndim=1)
        if self.bias_mg_knots.shape != self.bias_values.shape:
            raise ValueError("bias_mg_knots and bias_values must have identical shapes.")
        if np.any(np.diff(self.bias_mg_knots) <= 0):
            raise ValueError("bias_mg_knots must be strictly increasing.")
        self.extra_scale = float(extra_scale)
        self.student_df = float(student_df)
        if self.extra_scale <= 0 or self.student_df <= 0:
            raise ValueError("extra_scale and student_df must be positive.")
        self.population_components = int(population_components)
        if self.population_components < 2:
            raise ValueError("population_components must be at least two.")
        self.population_centers = np.linspace(
            self.z_grid[0], self.z_grid[-1], self.population_components
        )
        self.population_sigma = (
            self.z_grid[-1] - self.z_grid[0]
        ) / self.population_components
        self.sampler = None
        self.posterior_samples = None
        self._data_set = False
        self.selected_draw_indices = None
        self.posterior_draw_seed = None

    def bias_from_absg(self, absg) -> np.ndarray:
        """Interpolate the fixed T6d relative metallicity bias in dex."""
        return np.interp(
            np.asarray(absg, dtype=np.float64),
            self.bias_mg_knots,
            self.bias_values,
        )

    def observation_model_metadata(self) -> Dict[str, object]:
        """Return the fixed T6d plug-in parameters in a serializable form."""
        return {
            "model": "t8_jcaps_student_t_independent_members",
            "sampled_parameters": ["delta_z", "s_C", "population_weights"],
            "delta_z_prior": {"distribution": "Normal", "loc_dex": 0.0, "scale_dex": 0.3},
            "s_C_prior": {"distribution": "HalfNormal", "scale_mag": 0.1},
            "cmd_min_parsec_teff_k": float(self.cmd_min_teff) if self.cmd_min_teff is not None else None,
            "metallicity_grid": self.z_grid.tolist(),
            "metallicity_grid_weights": self.grid_weights.tolist(),
            "population_components": self.population_components,
            "population_centers_dex": self.population_centers.tolist(),
            "population_sigma_dex": float(self.population_sigma),
            "population_prior": "Dirichlet(1,...,1) over truncated Gaussian bases",
            "bias_mg_knots": self.bias_mg_knots.tolist(),
            "bias_values_dex": self.bias_values.tolist(),
            "extra_scale_dex_per_star": self.extra_scale,
            "student_t_df": self.student_df,
            "bias_anchor": "b_G(M_G=8.5)=0",
            "source": "FINAL_FEH_JCAPS_STUDENT_T_WORKFLOW_2026-09-10.md",
        }

    def set_data(
        self,
        *,
        row_indices,
        absg,
        feh_observed,
        feh_sigma,
        color_observed,
        color_sigma,
    ) -> None:
        self.row_indices = np.asarray(row_indices, dtype=np.int64)
        self.absg = _as_native_f64(absg, name="absg", ndim=2)
        self.feh_observed = _as_native_f64(feh_observed, name="feh_observed", ndim=2)
        self.feh_sigma = _as_native_f64(feh_sigma, name="feh_sigma", ndim=2)
        self.color_observed = _as_native_f64(color_observed, name="color_observed", ndim=2)
        self.color_sigma = _as_native_f64(color_sigma, name="color_sigma", ndim=2)
        expected = (self.row_indices.size, 2)
        for name in ("absg", "feh_observed", "feh_sigma", "color_observed", "color_sigma"):
            if getattr(self, name).shape != expected:
                raise ValueError(
                    f"{name} must have shape {expected}; got {getattr(self, name).shape}."
                )
        stacked = np.column_stack(
            [self.absg, self.feh_observed, self.feh_sigma, self.color_observed, self.color_sigma]
        )
        if np.any(~np.isfinite(stacked)):
            raise ValueError("Stage-one arrays must contain only finite values.")
        if np.any(self.feh_sigma <= 0) or np.any(self.color_sigma <= 0):
            raise ValueError("All metallicity and color uncertainties must be positive.")
        if np.any(self.absg < self.color_surface.absg_grid_np[0]) or np.any(
            self.absg > self.color_surface.absg_grid_np[-1]
        ):
            raise ValueError("Absolute magnitudes lie outside the color-surface domain.")

        self.fixed_bias = self.bias_from_absg(self.absg)
        self.cmd_mask = np.ones(expected, dtype=bool)
        self.cmd_minimum_parsec_teff = np.full(expected, np.nan, dtype=np.float64)
        if self.cmd_min_teff is not None and self.color_surface.teff_grid_np is None:
            raise ValueError(
                "A PARSEC temperature grid is required when the T8 CMD "
                f"threshold ({self.cmd_min_teff:g} K) is enabled."
            )
        if self.cmd_min_teff is not None:
            # The mask is fixed per observed component. Requiring the complete
            # latent-Z support to be warm prevents parameter-dependent removal
            # of a likelihood term from favoring cold metallicity grid points.
            parsec_teff = self.color_surface.teff_from_absg_mh(
                self.absg[:, :, None], self.z_grid[None, None, :]
            )
            self.cmd_minimum_parsec_teff = np.min(parsec_teff, axis=2)
            self.cmd_mask = self.cmd_minimum_parsec_teff >= self.cmd_min_teff
            if not np.any(self.cmd_mask):
                raise ValueError(
                    "No component remains in the CMD likelihood after applying "
                    f"PARSEC Teff >= {self.cmd_min_teff:g} K over the metallicity grid."
                )
        self._data_set = True

    def set_data_from_table(
        self, *, table, row_indices, absg, color_observed, color_sigma
    ) -> None:
        """Set data while enforcing the four uncorrected JCAPS columns."""
        required = (
            "feh_jcaps_1", "feh_jcaps_2",
            "jc_sigma_m_h_1", "jc_sigma_m_h_2",
        )
        names = set(getattr(table, "colnames", ()))
        if not names and isinstance(table, Mapping):
            names = set(table)
        missing = [name for name in required if name not in names]
        if missing:
            raise ValueError(
                "T8 requires raw JCAPS columns; missing columns: "
                f"{missing}. Corrected metallicity columns are not valid substitutes."
            )
        feh = np.column_stack([np.asarray(table[name], dtype=np.float64) for name in required[:2]])
        sigma = np.column_stack([np.asarray(table[name], dtype=np.float64) for name in required[2:]])
        self.set_data(
            row_indices=row_indices, absg=absg, feh_observed=feh,
            feh_sigma=sigma, color_observed=color_observed, color_sigma=color_sigma,
        )

    def _population_component_density_np(self) -> np.ndarray:
        standardized = (
            self.z_grid[:, None] - self.population_centers[None, :]
        ) / self.population_sigma
        norm = ndtr(
            (self.z_grid[-1] - self.population_centers) / self.population_sigma
        )
        norm -= ndtr(
            (self.z_grid[0] - self.population_centers) / self.population_sigma
        )
        return np.exp(-0.5 * standardized**2) / (
            np.sqrt(2.0 * np.pi) * self.population_sigma * norm[None, :]
        )

    def log_joint_numpy(self, params: Mapping[str, np.ndarray]) -> np.ndarray:
        """Evaluate the T6d+CMD per-system/per-grid log joint density."""
        if not self._data_set:
            raise ValueError("Call set_data() first.")
        z_value = params.get("delta_z", params.get("z_offset"))
        scatter_value = params.get("s_C", params.get("cmd_scatter"))
        if z_value is None or scatter_value is None:
            raise KeyError("log_joint_numpy requires delta_z and s_C.")
        z_offset = float(np.asarray(z_value))
        cmd_scatter = float(np.asarray(scatter_value))
        population_weights = np.asarray(params["population_weights"], dtype=float)
        population_density = self._population_component_density_np() @ population_weights
        log_joint = np.log(self.grid_weights)[None, :] + np.log(population_density)[None, :]
        log_joint = np.broadcast_to(
            log_joint, (self.row_indices.size, self.z_grid.size)
        ).copy()
        for component in range(2):
            loc = (
                z_offset
                + self.z_grid[None, :]
                + self.fixed_bias[:, component, None]
            )
            scale = np.sqrt(
                self.feh_sigma[:, component, None] ** 2 + self.extra_scale**2
            )
            log_metal = student_t_logpdf_np(
                self.feh_observed[:, component, None],
                df=self.student_df,
                loc=loc,
                scale=scale,
            )
            predicted_color = self.color_surface.color_from_absg_mh(
                self.absg[:, component, None], self.z_grid[None, :]
            )
            color_scale = np.sqrt(
                self.color_sigma[:, component, None] ** 2 + cmd_scatter**2
            )
            log_color = student_t_logpdf_np(
                self.color_observed[:, component, None],
                df=4.0,
                loc=predicted_color,
                scale=color_scale,
            )
            log_joint += log_metal + np.where(
                self.cmd_mask[:, component, None], log_color, 0.0
            )
        return log_joint

    def _build_numpyro_model(self):
        import jax.numpy as jnp
        from jax.scipy.special import logsumexp as jax_logsumexp
        from jax.scipy.special import ndtr as jax_ndtr
        import numpyro
        import numpyro.distributions as dist

        z_grid = jnp.asarray(self.z_grid)
        log_grid_weights = jnp.log(jnp.asarray(self.grid_weights))
        centers = jnp.asarray(self.population_centers)
        sigma_pop = float(self.population_sigma)
        standardized = (z_grid[:, None] - centers[None, :]) / sigma_pop
        component_norm = jax_ndtr((float(self.z_grid[-1]) - centers) / sigma_pop)
        component_norm -= jax_ndtr((float(self.z_grid[0]) - centers) / sigma_pop)
        component_density = jnp.exp(-0.5 * standardized**2) / (
            jnp.sqrt(2.0 * jnp.pi) * sigma_pop * component_norm[None, :]
        )
        cmd_mask = jnp.asarray(self.cmd_mask)
        fixed_bias = jnp.asarray(self.fixed_bias)
        extra_scale = float(self.extra_scale)
        student_df = float(self.student_df)

        def model(absg, feh_observed, feh_sigma, color_observed, color_sigma):
            delta_z = numpyro.sample("delta_z", dist.Normal(0.0, 0.3))
            s_C = numpyro.sample("s_C", dist.HalfNormal(0.1))
            population_weights = numpyro.sample(
                "population_weights", dist.Dirichlet(jnp.ones(self.population_components))
            )
            population_density = component_density @ population_weights
            log_joint = log_grid_weights[None, :] + jnp.log(population_density)[None, :]
            for component in range(2):
                loc = delta_z + z_grid[None, :] + fixed_bias[:, component, None]
                scale = jnp.sqrt(feh_sigma[:, component, None] ** 2 + extra_scale**2)
                log_metal = dist.StudentT(
                    student_df, loc=loc, scale=scale
                ).log_prob(feh_observed[:, component, None])
                predicted_color = self.color_surface.color_from_absg_mh_jax(
                    absg[:, component, None], z_grid[None, :]
                )
                effective_color_sigma = jnp.sqrt(
                    color_sigma[:, component, None] ** 2 + s_C**2
                )
                log_color = dist.StudentT(
                    4.0, loc=predicted_color, scale=effective_color_sigma
                ).log_prob(color_observed[:, component, None])
                log_joint += log_metal + jnp.where(
                    cmd_mask[:, component, None], log_color, 0.0
                )
            log_likelihood = jax_logsumexp(log_joint, axis=1)
            numpyro.factor("observations", jnp.sum(log_likelihood))
            # Compatibility aliases are deterministic, not additional sampled
            # global parameters; the T8 vector remains (delta_z, s_C, omega).
            numpyro.deterministic("z_offset", delta_z)
            numpyro.deterministic("cmd_scatter", s_C)

        return model

    def run_mcmc(
        self,
        *,
        num_warmup: int = 1000,
        num_samples: int = 1000,
        num_chains: int = 4,
        seed: int = 42,
        progress_bar: bool = True,
        **nuts_kwargs,
    ):
        if not self._data_set:
            raise ValueError("Call set_data() before run_mcmc().")
        jax = _configure_jax_gpu_fallback()
        import jax.numpy as jnp
        from numpyro.infer import MCMC, NUTS

        model = self._build_numpyro_model()
        kernel = NUTS(model, **nuts_kwargs)
        mcmc = MCMC(
            kernel,
            num_warmup=int(num_warmup),
            num_samples=int(num_samples),
            num_chains=int(num_chains),
            progress_bar=bool(progress_bar),
        )
        mcmc.run(
            jax.random.PRNGKey(int(seed)),
            jnp.asarray(self.absg),
            jnp.asarray(self.feh_observed),
            jnp.asarray(self.feh_sigma),
            jnp.asarray(self.color_observed),
            jnp.asarray(self.color_sigma),
            extra_fields=("diverging", "energy", "potential_energy", "num_steps", "accept_prob"),
        )
        self.sampler = mcmc
        self.posterior_samples = {
            name: np.asarray(values) for name, values in mcmc.get_samples().items()
        }
        return mcmc

    def posterior_grid(
        self,
        *,
        max_draws: int = 128,
        system_chunk: int = 256,
        seed: int = 0,
        metadata: Optional[Mapping[str, object]] = None,
    ) -> MetallicityPosteriorGrid:
        """Average each system's T6d+CMD P(Z) over global posterior draws."""
        if self.posterior_samples is None:
            raise ValueError("Run run_mcmc() before posterior_grid().")
        sample_count = next(iter(self.posterior_samples.values())).shape[0]
        draw_count = min(int(max_draws), sample_count)
        if draw_count < 1:
            raise ValueError("max_draws must be positive.")
        draw_indices = np.linspace(0, sample_count - 1, draw_count, dtype=int)
        if draw_count < sample_count:
            rng = np.random.default_rng(seed)
            draw_indices = np.sort(
                rng.choice(sample_count, size=draw_count, replace=False)
            )
        self.selected_draw_indices = np.asarray(draw_indices, dtype=np.int64)
        self.posterior_draw_seed = int(seed)
        samples = {
            name: values[draw_indices] for name, values in self.posterior_samples.items()
        }
        delta_samples = samples.get("delta_z", samples.get("z_offset"))
        scatter_samples = samples.get("s_C", samples.get("cmd_scatter"))
        if delta_samples is None or scatter_samples is None:
            raise KeyError("Posterior samples must contain delta_z and s_C.")
        component_density = self._population_component_density_np()
        n_systems, n_grid = self.row_indices.size, self.z_grid.size
        mean_probabilities = np.empty((n_systems, n_grid), dtype=np.float64)

        for start in range(0, n_systems, int(system_chunk)):
            stop = min(start + int(system_chunk), n_systems)
            batch = stop - start
            population_density = np.einsum(
                "qk,dk->dq", component_density, samples["population_weights"]
            )
            log_joint = (
                np.log(self.grid_weights)[None, None, :]
                + np.log(population_density)[:, None, :]
            )
            log_joint = np.broadcast_to(
                log_joint, (draw_count, batch, n_grid)
            ).copy()
            for component in range(2):
                loc = (
                    delta_samples[:, None, None]
                    + self.z_grid[None, None, :]
                    + self.fixed_bias[start:stop, component][None, :, None]
                )
                effective_sigma = np.sqrt(
                    self.feh_sigma[start:stop, component][None, :, None] ** 2
                    + self.extra_scale**2
                )
                observed = self.feh_observed[
                    start:stop, component
                ][None, :, None]
                log_metal = student_t_logpdf_np(
                    observed,
                    df=self.student_df,
                    loc=loc,
                    scale=effective_sigma,
                )
                predicted_color = self.color_surface.color_from_absg_mh(
                    self.absg[start:stop, component, None], self.z_grid[None, :]
                )[None, :, :]
                effective_color_sigma = np.sqrt(
                    self.color_sigma[
                        start:stop, component
                    ][None, :, None] ** 2
                    + scatter_samples[:, None, None] ** 2
                )
                log_color = student_t_logpdf_np(
                    self.color_observed[
                        start:stop, component
                    ][None, :, None],
                    df=4.0,
                    loc=predicted_color,
                    scale=effective_color_sigma,
                )
                log_joint += log_metal + np.where(
                    self.cmd_mask[start:stop, component][None, :, None],
                    log_color,
                    0.0,
                )
            probabilities = normalize_log_weights(log_joint, axis=2)
            mean_probabilities[start:stop] = np.mean(probabilities, axis=0)

        cumulative = np.cumsum(mean_probabilities, axis=1)
        cumulative /= cumulative[:, -1, None]
        quantiles = np.empty((n_systems, 3), dtype=np.float64)
        for index, target in enumerate((0.16, 0.50, 0.84)):
            quantiles[:, index] = np.array(
                [
                    np.interp(target, cumulative[row], self.z_grid)
                    for row in range(n_systems)
                ]
            )
        result = MetallicityPosteriorGrid(
            row_indices=self.row_indices.copy(),
            z_grid=self.z_grid.copy(),
            probabilities=mean_probabilities,
            z_quantiles=quantiles,
            # Compatibility placeholder: the T6d observation model has no
            # good/bad mixture state, so both probabilities are fixed to zero.
            bad_probabilities=np.zeros((n_systems, 2), dtype=np.float64),
            metadata={
                **self.observation_model_metadata(),
                "selected_draw_count": int(draw_count),
                "selected_draw_seed": int(seed),
                "selected_draw_indices": self.selected_draw_indices.tolist(),
                "selected_draw_indices_digest": array_digest(self.selected_draw_indices),
                "forbidden_metallicity_columns": [
                    "jc_m_h_fit_1", "jc_m_h_fit_2", "jc_m_h_fit_cal_1", "jc_m_h_fit_cal_2",
                ],
                **({} if metadata is None else dict(metadata)),
            },
        )
        result.validate()
        return result


@dataclass(frozen=True)
class DynamicsLikelihoodLookup:
    """Per-system dynamics likelihood tabulated against sqrt(total mass).

    ``log_good[j, l]`` and ``log_bad[j, l]`` contain the accurately integrated
    likelihood for system ``j`` at ``sqrt_mtot_grid[l]``.  The outlier shape is
    fixed when the table is constructed; only its mixture fraction remains a
    free parameter in the MLR fit.
    """

    row_indices: np.ndarray
    sqrt_mtot_grid: np.ndarray
    log_good: np.ndarray
    log_bad: np.ndarray
    metadata: Mapping[str, object]

    def validate(self) -> None:
        rows = np.asarray(self.row_indices)
        grid = np.asarray(self.sqrt_mtot_grid)
        good = np.asarray(self.log_good)
        bad = np.asarray(self.log_bad)
        if rows.ndim != 1 or grid.ndim != 1:
            raise ValueError("Lookup row_indices and sqrt_mtot_grid must be one-dimensional.")
        if grid.size < 2 or np.any(~np.isfinite(grid)) or np.any(np.diff(grid) <= 0):
            raise ValueError("sqrt_mtot_grid must be finite and strictly increasing.")
        expected = (rows.size, grid.size)
        if good.shape != expected or bad.shape != expected:
            raise ValueError(
                f"Lookup tables must have shape {expected}; got {good.shape} and {bad.shape}."
            )
        if np.any(~np.isfinite(good)) or np.any(~np.isfinite(bad)):
            raise ValueError("Lookup log likelihoods must be finite.")

    def validate_t8_schema(self) -> None:
        """Reject legacy/incomplete tables before they can enter T8 MLR."""
        required = {
            "model", "schema", "data_digest", "sqrt_mtot_min", "sqrt_mtot_max",
            "sqrt_mtot_points", "velocity_quadrature_nodes", "velocity_sigma_extent",
            "tilde_u_max", "outlier_u0", "outlier_sigma", "outlier_normalization",
            "normalization_version", "good_basis_normalized", "jacobian", "interpolation",
            "convergence_check", "migration_applied", "scale_grid",
            "outlier_support", "outlier_normalization_value", "C_out",
            "C_good", "good_basis_support", "good_basis_constant",
            "good_basis_quadratic", "good_basis_turnover", "good_basis_scale",
            "good_basis_support_integral", "scale_definition",
            "integration_coordinate", "row_order", "floor",
            "velocity_units",
            "mixture_weight_semantics",
        }
        missing = sorted(required.difference(self.metadata))
        if missing:
            raise ValueError(
                "Dynamics lookup metadata is incomplete for T8; rebuild lookup. "
                f"Missing: {missing}"
            )
        if self.metadata.get("model") != "dynamics_likelihood_lookup_t8":
            raise ValueError("Legacy dynamics lookup schema is not accepted; rebuild T8 lookup.")
        if self.metadata.get("schema") != "t8c-rice-lookup-v1":
            raise ValueError("Unexpected T8 lookup schema version; rebuild the lookup.")
        if self.metadata.get("jacobian") != "1/s":
            raise ValueError("Dynamics lookup does not record the required 1/s Jacobian.")
        if self.metadata.get("interpolation") != "linear_log_likelihood_in_raw_s":
            raise ValueError("Dynamics lookup interpolation convention is not the T8 convention.")
        if self.metadata.get("normalization_version") != "finite_support_exact_t8_1":
            raise ValueError("Dynamics lookup normalization version is not the T8.1 convention.")
        if self.metadata.get("outlier_normalization") != "Phi((U-mu)/sigma)-Phi(-mu/sigma)":
            raise ValueError("Dynamics lookup outlier normalization convention is invalid.")
        if self.metadata.get("velocity_units") != "km s^-1 sqrt(AU)":
            raise ValueError("Dynamics lookup velocity units are incompatible with T8.")
        if not np.isfinite(float(self.metadata.get("floor", np.nan))) or float(self.metadata["floor"]) <= 0:
            raise ValueError("Dynamics lookup floor must be finite and positive.")
        if not bool(self.metadata.get("good_basis_normalized")):
            raise ValueError("T8.1 requires the normalized good basis; rebuild lookup.")
        if bool(self.metadata.get("migration_applied")):
            raise ValueError("Migrated dynamics lookup tables are not accepted in T8; rebuild lookup.")
        if self.metadata.get("scale_grid") != "geometric":
            raise ValueError("T8 requires a geometric sqrt(total-mass) lookup grid.")
        if self.metadata.get("scale_definition") != "s=sqrt(Mtot/Msun)":
            raise ValueError("Dynamics lookup mass-scale definition is incompatible with T8.")
        if self.metadata.get("integration_coordinate") != "v=s*tilde_u":
            raise ValueError("Dynamics lookup integration coordinate is incompatible with T8.")
        if self.metadata.get("row_order") != "row_indices order":
            raise ValueError("Dynamics lookup row-order convention is incompatible with T8.")
        legacy_mix = "physical outlier fraction (T8.1 exact component normalization)"
        if self.metadata.get("mixture_weight_semantics") not in (MIXTURE_SEMANTICS, legacy_mix):
            raise ValueError("Dynamics lookup mixture-weight semantics are missing or incompatible.")
        convergence = self.metadata.get("convergence_check")
        if not isinstance(convergence, Mapping) or not bool(convergence.get("passed")):
            raise ValueError("T8 lookup lacks a passing convergence check; rebuild lookup.")
        selected_rows = np.asarray(convergence.get("selected_row_indices", []), dtype=np.int64)
        requested_systems = int(convergence.get("sample_systems_requested", -1))
        actual_systems = int(convergence.get("sample_system_count", -1))
        requested_scale_points = int(convergence.get("scale_points_requested", -1))
        # Older T8 files omit the experiment marker and are interpreted as the
        # frozen baseline. Experimental lookups carry their explicit shape.
        expected_outlier_u0 = float(self.metadata.get("outlier_u0", RICE_OUTLIER_MU))
        expected_outlier_sigma = float(self.metadata.get("outlier_sigma", RICE_OUTLIER_SIGMA))
        if not np.isfinite(expected_outlier_u0) or not np.isfinite(expected_outlier_sigma) or expected_outlier_sigma <= 0:
            raise ValueError("Outlier center and positive width must be finite.")
        if not bool(self.metadata.get("experimental_outlier_sensitivity", False)) and (
            expected_outlier_u0 != RICE_OUTLIER_MU or expected_outlier_sigma != RICE_OUTLIER_SIGMA
        ):
            raise ValueError("Nonbaseline outlier shape requires an explicit experimental marker.")
        if bool(self.metadata.get("experimental_outlier_sensitivity", False)) and (
            "outlier_u0" not in convergence or "outlier_sigma" not in convergence
        ):
            raise ValueError("Experimental lookup convergence record lacks its outlier shape.")
        if (
            convergence.get("source_data_digest") != self.metadata.get("data_digest")
            or not np.isclose(float(convergence.get("outlier_u0", expected_outlier_u0)), expected_outlier_u0, rtol=0.0, atol=0.0)
            or not np.isclose(float(convergence.get("outlier_sigma", expected_outlier_sigma)), expected_outlier_sigma, rtol=0.0, atol=0.0)
            or not np.isclose(
                float(convergence.get("floor", np.nan)), float(self.metadata["floor"]),
                rtol=0.0, atol=0.0,
            )
            or requested_systems < 1
            or actual_systems < 1
            or actual_systems > min(requested_systems, len(self.row_indices))
            or selected_rows.size != actual_systems
            or np.unique(selected_rows).size != actual_systems
            or not np.all(np.isin(selected_rows, np.asarray(self.row_indices, dtype=np.int64)))
            or convergence.get("selected_row_indices_digest") != array_digest(selected_rows)
            or requested_scale_points < 2
        ):
            raise ValueError("Lookup convergence record is not bound to the saved T8 data/settings.")
        fixed_values = (
            ("tilde_u_max", RICE_GOOD_SUPPORT),
            ("outlier_u0", expected_outlier_u0),
            ("outlier_sigma", expected_outlier_sigma),
        )
        if any(
            not np.isclose(float(self.metadata.get(name, np.nan)), expected)
            for name, expected in fixed_values
        ):
            raise ValueError("T8 lookup component support/shape is incompatible; rebuild lookup.")
        for name in ("outlier_support", "good_basis_support"):
            if not np.array_equal(
                np.asarray(self.metadata.get(name), dtype=np.float64),
                np.asarray([0.0, RICE_GOOD_SUPPORT]),
            ):
                raise ValueError("T8 lookup component support is incompatible; rebuild lookup.")
        expected_good = {
            "good_basis_constant": RICE_GOOD_A,
            "good_basis_quadratic": RICE_GOOD_B,
            "good_basis_turnover": RICE_GOOD_UC,
            "good_basis_scale": RICE_GOOD_C,
            "good_basis_support_integral": 1.0,
            "C_good": 1.0,
        }
        if any(
            not np.isclose(float(self.metadata.get(name, np.nan)), expected, rtol=0.0, atol=1e-12)
            for name, expected in expected_good.items()
        ):
            raise ValueError("T8 lookup raw-good component constants are incompatible; rebuild lookup.")
        expected_outlier_norm = rice_outlier_normalization(
            mu=expected_outlier_u0, sigma=expected_outlier_sigma
        )
        if not np.isclose(
            float(self.metadata.get("outlier_normalization_value", np.nan)),
            expected_outlier_norm,
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError("T8 lookup outlier normalization value is invalid; rebuild lookup.")
        if not np.isclose(
            float(self.metadata.get("C_out", np.nan)), expected_outlier_norm,
            rtol=0.0, atol=1e-12,
        ):
            raise ValueError("T8 lookup C_out is invalid; rebuild lookup.")
        grid = np.asarray(self.sqrt_mtot_grid, dtype=np.float64)
        expected_grid = np.geomspace(
            float(self.metadata["sqrt_mtot_min"]),
            float(self.metadata["sqrt_mtot_max"]),
            int(self.metadata["sqrt_mtot_points"]),
        )
        if not np.allclose(grid, expected_grid, rtol=0.0, atol=1e-14):
            raise ValueError("Saved lookup grid does not match its T8 metadata; rebuild lookup.")
        adopted = convergence.get("adopted", {})
        checked_grid = np.asarray(convergence.get("scale_grid", []), dtype=np.float64)
        if (
            int(adopted.get("quadrature_nodes", -1))
            != int(self.metadata["velocity_quadrature_nodes"])
            or not np.isclose(
                float(adopted.get("velocity_sigma_extent", np.nan)),
                float(self.metadata["velocity_sigma_extent"]), rtol=0.0, atol=0.0,
            )
            or checked_grid.size < 2
            or checked_grid.size != requested_scale_points
            or not np.isclose(checked_grid[0], grid[0], rtol=0.0, atol=1e-14)
            or not np.isclose(checked_grid[-1], grid[-1], rtol=0.0, atol=1e-14)
        ):
            raise ValueError("Lookup convergence check does not cover the saved T8 table settings.")

    def validate_for(self, *, row_indices, u, u_sigma) -> None:
        """Verify that a saved table belongs to the exact stage-two data."""
        self.validate()
        self.validate_t8_schema()
        rows = np.asarray(row_indices, dtype=np.int64)
        u = np.asarray(u, dtype=np.float64)
        u_sigma = np.asarray(u_sigma, dtype=np.float64)
        if not np.array_equal(rows, np.asarray(self.row_indices, dtype=np.int64)):
            raise ValueError("Dynamics lookup row indices do not match the selected data.")
        expected_digest = array_digest(rows, u, u_sigma)
        if self.metadata.get("data_digest") != expected_digest:
            raise ValueError("Dynamics lookup u/u_sigma digest does not match the selected data.")

    def save(self, path) -> Path:
        self.validate()
        self.validate_t8_schema()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Deliberately uncompressed: these smooth float32 tables are large, and
        # fast server-side save/load is more useful than modest compression.
        np.savez(
            path,
            row_indices=np.asarray(self.row_indices, dtype=np.int64),
            sqrt_mtot_grid=np.asarray(self.sqrt_mtot_grid, dtype=np.float64),
            log_good=np.asarray(self.log_good, dtype=np.float32),
            log_bad=np.asarray(self.log_bad, dtype=np.float32),
            metadata_json=np.asarray(json.dumps(dict(self.metadata), sort_keys=True)),
        )
        return path

    @classmethod
    def load(cls, path) -> "DynamicsLikelihoodLookup":
        with np.load(path, allow_pickle=False) as saved:
            result = cls(
                row_indices=np.asarray(saved["row_indices"], dtype=np.int64),
                sqrt_mtot_grid=np.asarray(saved["sqrt_mtot_grid"], dtype=np.float64),
                log_good=np.asarray(saved["log_good"], dtype=np.float32),
                log_bad=np.asarray(saved["log_bad"], dtype=np.float32),
                metadata=json.loads(str(saved["metadata_json"].item())),
            )
        result.validate()
        result.validate_t8_schema()
        return result

    def interpolate_numpy(self, sqrt_mtot, *, component: str = "good") -> np.ndarray:
        """Row-wise linear interpolation in log likelihood for diagnostics."""
        values = np.asarray(sqrt_mtot, dtype=np.float64)
        if values.ndim != 2 or values.shape[0] != len(self.row_indices):
            raise ValueError("sqrt_mtot must have shape (n_systems, n_evaluations).")
        table = self.log_good if component == "good" else self.log_bad
        if component not in {"good", "bad"}:
            raise ValueError("component must be 'good' or 'bad'.")
        if np.any(values < self.sqrt_mtot_grid[0]) or np.any(values > self.sqrt_mtot_grid[-1]):
            raise ValueError("Requested sqrt(total mass) lies outside the lookup domain.")
        output = np.empty_like(values)
        for row in range(values.shape[0]):
            output[row] = np.interp(values[row], self.sqrt_mtot_grid, table[row])
        return output

    @classmethod
    def precompute(
        cls,
        *,
        row_indices,
        u,
        u_sigma,
        sqrt_mtot_min: float = 0.25,
        sqrt_mtot_max: float = 2.5,
        sqrt_mtot_points: int = 1024,
        velocity_quadrature_nodes: int = 64,
        velocity_sigma_extent: float = 10.0,
        tilde_u_max: float = 80.0,
        outlier_u0: float = 40.0,
        outlier_sigma: float = 13.0,
        outlier_sensitivity: bool = False,
        system_chunk: int = 256,
        floor: float = 1e-30,
    ) -> "DynamicsLikelihoodLookup":
        """Precompute accurate Rice-convolved likelihood tables on a GPU/CPU.

        The expensive integral is expressed in observed velocity ``v``.  Each
        system gets Gauss--Legendre nodes over ``u +/- velocity_sigma_extent *
        u_sigma`` (clipped at zero).  These local nodes resolve a narrow Rice
        kernel; the old 64 nodes were instead spread over the full [0, 80]
        normalized-velocity interval and could completely miss that kernel.
        """
        rows = np.asarray(row_indices, dtype=np.int64)
        u = _as_native_f64(u, name="u", ndim=1)
        u_sigma = _as_native_f64(u_sigma, name="u_sigma", ndim=1)
        if rows.shape != u.shape or u.shape != u_sigma.shape:
            raise ValueError("row_indices, u, and u_sigma must have identical shapes.")
        if np.any(~np.isfinite(u)) or np.any(~np.isfinite(u_sigma)):
            raise ValueError("u and u_sigma must be finite.")
        if np.any(u <= 0) or np.any(u_sigma <= 0):
            raise ValueError("u and u_sigma must be strictly positive.")
        if not (0 < sqrt_mtot_min < sqrt_mtot_max):
            raise ValueError("Require 0 < sqrt_mtot_min < sqrt_mtot_max.")
        if (
            int(sqrt_mtot_points) < 2
            or int(velocity_quadrature_nodes) < 8
            or velocity_sigma_extent <= 0
            or tilde_u_max <= 0
        ):
            raise ValueError("Lookup grid sizes and spacings must be positive.")
        if outlier_sigma <= 0 or system_chunk < 1:
            raise ValueError("outlier_sigma and system_chunk must be positive.")

        jax = _configure_jax_gpu_fallback()
        import jax.numpy as jnp
        from jax.scipy.special import i0e
        from jax.scipy.special import ndtr as jax_ndtr

        sqrt_grid = np.geomspace(
            float(sqrt_mtot_min), float(sqrt_mtot_max), int(sqrt_mtot_points)
        ).astype(np.float64)
        sqrt_jax = jnp.asarray(sqrt_grid, dtype=jnp.float32)
        gl_x, gl_w = np.polynomial.legendre.leggauss(int(velocity_quadrature_nodes))
        gl_x_jax = jnp.asarray(gl_x, dtype=jnp.float32)
        gl_w_jax = jnp.asarray(gl_w, dtype=jnp.float32)
        outlier_norm = jax_ndtr(
            (float(tilde_u_max) - float(outlier_u0)) / float(outlier_sigma)
        ) - jax_ndtr(-float(outlier_u0) / float(outlier_sigma))

        def calculate_chunk(u_chunk, sigma_chunk):
            lower_window = jnp.maximum(
                0.0, u_chunk - float(velocity_sigma_extent) * sigma_chunk
            )
            upper_window = u_chunk + float(velocity_sigma_extent) * sigma_chunk
            # Split exactly at the transformed component-support boundary
            # v=s*U.  Masking a single Gauss-Legendre interval at that boundary
            # introduces a discontinuity and converges extremely slowly,
            # especially for the outlier component.
            upper = jnp.minimum(
                upper_window[:, None],
                sqrt_jax[None, :] * float(tilde_u_max),
            )
            lower = jnp.minimum(lower_window[:, None], upper)
            midpoint = 0.5 * (lower + upper)
            half_width = 0.5 * (upper - lower)
            velocity = (
                midpoint[:, None, :]
                + half_width[:, None, :] * gl_x_jax[None, :, None]
            )
            quadrature_weight = (
                half_width[:, None, :] * gl_w_jax[None, :, None]
            )
            observed = u_chunk[:, None, None]
            sigma = sigma_chunk[:, None, None]
            sigma_sq = sigma**2
            argument = observed * velocity / sigma_sq
            log_rice = (
                jnp.log(observed)
                - jnp.log(sigma_sq)
                - (observed - velocity) ** 2 / (2.0 * sigma_sq)
                + jnp.log(i0e(argument) + 1e-30)
            )
            kernel = jnp.exp(log_rice) * quadrature_weight

            tilde_u = velocity / sqrt_jax[None, None, :]
            tilde_for_eval = jnp.clip(tilde_u, 1e-12, float(tilde_u_max))
            log_good_basis = (
                jnp.log(RICE_GOOD_A)
                - jnp.log(RICE_GOOD_BASIS_RAW_INTEGRAL)
                + jnp.log(tilde_for_eval)
                - RICE_GOOD_B * tilde_for_eval**2
                - jnp.exp((tilde_for_eval - RICE_GOOD_UC) / RICE_GOOD_C)
                - jnp.log(sqrt_jax[None, None, :])
            )
            good_basis = jnp.where(
                (tilde_u > 0.0) & (tilde_u <= float(tilde_u_max)),
                jnp.exp(log_good_basis),
                0.0,
            )
            log_bad_basis = (
                -0.5 * ((tilde_for_eval - float(outlier_u0)) / float(outlier_sigma)) ** 2
                - jnp.log(float(outlier_sigma) * jnp.sqrt(2.0 * jnp.pi))
                - jnp.log(outlier_norm)
                - jnp.log(sqrt_jax[None, None, :])
            )
            bad_basis = jnp.where(
                (tilde_u >= 0.0) & (tilde_u <= float(tilde_u_max)),
                jnp.exp(log_bad_basis),
                0.0,
            )
            return (
                jnp.sum(kernel * good_basis, axis=1),
                jnp.sum(kernel * bad_basis, axis=1),
            )

        calculate_chunk = jax.jit(calculate_chunk)
        n_systems = rows.size
        log_good = np.empty((n_systems, sqrt_grid.size), dtype=np.float32)
        log_bad = np.empty_like(log_good)
        devices = ", ".join(str(device) for device in jax.devices())
        print(
            "Precomputing dynamics lookup: "
            f"{n_systems} systems x {sqrt_grid.size} mass nodes, "
            f"{int(velocity_quadrature_nodes)} local velocity nodes on {devices}."
        )
        for start in range(0, n_systems, int(system_chunk)):
            stop = min(start + int(system_chunk), n_systems)
            count = stop - start
            u_chunk = np.empty(int(system_chunk), dtype=np.float32)
            sigma_chunk = np.empty(int(system_chunk), dtype=np.float32)
            u_chunk[:count] = u[start:stop]
            sigma_chunk[:count] = u_sigma[start:stop]
            if count < int(system_chunk):
                u_chunk[count:] = u[start]
                sigma_chunk[count:] = u_sigma[start]
            good_chunk, bad_chunk = calculate_chunk(
                jnp.asarray(u_chunk), jnp.asarray(sigma_chunk)
            )
            good_chunk = np.asarray(jax.device_get(good_chunk))[:count]
            bad_chunk = np.asarray(jax.device_get(bad_chunk))[:count]
            log_good[start:stop] = np.log(np.maximum(good_chunk, float(floor)))
            log_bad[start:stop] = np.log(np.maximum(bad_chunk, float(floor)))
            print(f"  lookup systems {stop}/{n_systems}", flush=True)

        metadata = {
            "model": "dynamics_likelihood_lookup_t8",
            "schema": "t8c-rice-lookup-v1",
            "data_digest": array_digest(rows, u, u_sigma),
            "sqrt_mtot_min": float(sqrt_mtot_min),
            "sqrt_mtot_max": float(sqrt_mtot_max),
            "sqrt_mtot_points": int(sqrt_mtot_points),
            "scale_grid": "geometric",
            "velocity_quadrature_nodes": int(velocity_quadrature_nodes),
            "velocity_sigma_extent": float(velocity_sigma_extent),
            "tilde_u_max": float(tilde_u_max),
            "outlier_u0": float(outlier_u0),
            "outlier_sigma": float(outlier_sigma),
            "experimental_outlier_sensitivity": bool(outlier_sensitivity),
            "outlier_support": [0.0, float(tilde_u_max)],
            "outlier_normalization": "Phi((U-mu)/sigma)-Phi(-mu/sigma)",
            "outlier_normalization_value": float(rice_outlier_normalization(support_max=tilde_u_max, mu=outlier_u0, sigma=outlier_sigma)),
            "C_out": float(rice_outlier_normalization(support_max=tilde_u_max, mu=outlier_u0, sigma=outlier_sigma)),
            "C_good": 1.0,
            "normalization_version": "finite_support_exact_t8_1",
            "good_basis_normalized": True,
            "good_basis_raw_support_integral": RICE_GOOD_BASIS_RAW_INTEGRAL,
            "good_basis_support": [0.0, float(tilde_u_max)],
            "good_basis_constant": float(RICE_GOOD_A),
            "good_basis_quadratic": float(RICE_GOOD_B),
            "good_basis_turnover": float(RICE_GOOD_UC),
            "good_basis_scale": float(RICE_GOOD_C),
            "good_basis_support_integral": 1.0,
            "jacobian": "1/s",
            "velocity_units": "km s^-1 sqrt(AU)",
            "scale_definition": "s=sqrt(Mtot/Msun)",
            "integration_coordinate": "v=s*tilde_u",
            "interpolation": "linear_log_likelihood_in_raw_s",
            "row_order": "row_indices order",
            "mixture_weight_semantics": MIXTURE_SEMANTICS,
            "floor": float(floor),
            "migration_applied": False,
        }
        result = cls(
            row_indices=rows,
            sqrt_mtot_grid=sqrt_grid,
            log_good=log_good,
            log_bad=log_bad,
            metadata=metadata,
        )
        result.validate()
        return result


class DynamicsLikelihoodShapeStack:
    """Precomputed Rice lookups over a 3-axis grid of good-shape constants.

    The axes are (log B, log uc, log C) with three nodes each; stack member k
    corresponds to flat index k = ib + 3*iu + 9*ic.  The T8.2 MLR stage
    samples (log B, log uc, log C) inside the node box and mixes the member
    tables with trilinear weights (linear in probability), so the
    shape-mass degeneracy enters the posterior explicitly instead of being
    absorbed by a fixed calibration.
    """

    SCHEMA = "t8d2-rice-shapestack-v1"
    AXIS_NAMES = ("log_b", "log_uc", "log_c")

    def __init__(self, *, row_indices, sqrt_mtot_grid, log_good_stack, log_bad_stack,
                 axes, node_constants, metadata):
        self.row_indices = np.asarray(row_indices, dtype=np.int64)
        self.sqrt_mtot_grid = np.asarray(sqrt_mtot_grid, dtype=np.float64)
        self.log_good_stack = np.asarray(log_good_stack, dtype=np.float32)
        self.log_bad_stack = np.asarray(log_bad_stack, dtype=np.float32)
        self.axes = {name: np.asarray(values, dtype=np.float64) for name, values in axes.items()}
        self.node_constants = node_constants
        self.metadata = dict(metadata)
        self.validate()

    @property
    def n_nodes(self) -> int:
        return int(self.log_good_stack.shape[0])

    def validate(self) -> None:
        n = self.row_indices.size
        n_s = self.sqrt_mtot_grid.size
        for name in self.AXIS_NAMES:
            axis = self.axes.get(name)
            if axis is None or axis.shape != (3,) or np.any(np.diff(axis) <= 0):
                raise ValueError(f"Shape-stack axis {name} must hold three strictly increasing nodes.")
        expected = 27
        if self.log_good_stack.shape != (expected, n, n_s) or self.log_bad_stack.shape != (expected, n, n_s):
            raise ValueError(
                "Shape-stack tables must have shape (27, n_systems, n_s); got "
                f"{self.log_good_stack.shape} and {self.log_bad_stack.shape} for n={n}, n_s={n_s}."
            )
        if len(self.node_constants) != expected:
            raise ValueError("Shape-stack node constant records do not match the 27 members.")

    @classmethod
    def build(cls, *, row_indices, u, u_sigma, axes,
              sqrt_mtot_min=0.25, sqrt_mtot_max=2.5, sqrt_mtot_points=512,
              velocity_quadrature_nodes=64, velocity_sigma_extent=10.0,
              outlier_u0=40.0, outlier_sigma=13.0, system_chunk=128,
              outlier_sensitivity=False, progress=True):
        """Precompute the 27-member stack, restoring the ambient constants afterwards."""
        rows = np.asarray(row_indices, dtype=np.int64)
        axes = {name: np.asarray(values, dtype=np.float64) for name, values in axes.items()}
        global RICE_GOOD_A, RICE_GOOD_B, RICE_GOOD_UC, RICE_GOOD_C, RICE_GOOD_BASIS_RAW_INTEGRAL
        original = (RICE_GOOD_A, RICE_GOOD_B, RICE_GOOD_UC, RICE_GOOD_C, RICE_GOOD_BASIS_RAW_INTEGRAL)
        log_good_stack = np.empty((27, rows.size, int(sqrt_mtot_points)), dtype=np.float32)
        log_bad_stack = np.empty_like(log_good_stack)
        node_constants = []
        try:
            k = 0
            for lb in axes["log_b"]:
                for luc in axes["log_uc"]:
                    for lc in axes["log_c"]:
                        B, C, uc = float(np.exp(lb)), float(np.exp(lc)), float(np.exp(luc))
                        set_good_shape_constants(B=B, C=C, uc=uc)
                        lookup = DynamicsLikelihoodLookup.precompute(
                            row_indices=rows, u=u, u_sigma=u_sigma,
                            sqrt_mtot_min=sqrt_mtot_min, sqrt_mtot_max=sqrt_mtot_max,
                            sqrt_mtot_points=sqrt_mtot_points,
                            velocity_quadrature_nodes=velocity_quadrature_nodes,
                            velocity_sigma_extent=velocity_sigma_extent,
                            outlier_u0=outlier_u0, outlier_sigma=outlier_sigma,
                            outlier_sensitivity=outlier_sensitivity,
                            system_chunk=system_chunk,
                        )
                        log_good_stack[k] = lookup.log_good
                        log_bad_stack[k] = lookup.log_bad
                        node_constants.append({
                            "index": k, "log_b": float(lb), "log_uc": float(luc), "log_c": float(lc),
                            "B": B, "uc": uc, "C": C,
                            "good_basis_constant": float(RICE_GOOD_A),
                            "raw_support_integral": float(RICE_GOOD_BASIS_RAW_INTEGRAL),
                        })
                        if progress:
                            print(f"  shape stack node {k + 1}/27: B={B:.4g} uc={uc:.4g} C={C:.4g}")
                        k += 1
        finally:
            (RICE_GOOD_A, RICE_GOOD_B, RICE_GOOD_UC, RICE_GOOD_C, RICE_GOOD_BASIS_RAW_INTEGRAL) = original
        sqrt_grid = np.asarray(lookup.sqrt_mtot_grid, dtype=np.float64)
        metadata = {
            "model": "dynamics_likelihood_shapestack_t8",
            "schema": cls.SCHEMA,
            "data_digest": array_digest(rows, u, u_sigma),
            "sqrt_mtot_min": float(sqrt_mtot_min), "sqrt_mtot_max": float(sqrt_mtot_max),
            "sqrt_mtot_points": int(sqrt_mtot_points),
            "velocity_quadrature_nodes": int(velocity_quadrature_nodes),
            "velocity_sigma_extent": float(velocity_sigma_extent),
            "outlier_u0": float(outlier_u0), "outlier_sigma": float(outlier_sigma),
            "outlier_support": [0.0, 80.0],
            "normalization_version": "finite_support_exact_t8_1",
            "jacobian": "1/s",
            "row_order": "row_indices order",
        }
        return cls(
            row_indices=rows, sqrt_mtot_grid=sqrt_grid,
            log_good_stack=log_good_stack, log_bad_stack=log_bad_stack,
            axes=axes, node_constants=node_constants, metadata=metadata,
        )

    def validate_for(self, *, row_indices, u, u_sigma) -> None:
        self.validate()
        rows = np.asarray(row_indices, dtype=np.int64)
        u = np.asarray(u, dtype=np.float64)
        u_sigma = np.asarray(u_sigma, dtype=np.float64)
        if not np.array_equal(rows, self.row_indices):
            raise ValueError("Dynamics shape-stack row indices do not match the selected data.")
        if self.metadata.get("data_digest") != array_digest(rows, u, u_sigma):
            raise ValueError("Dynamics shape-stack u/u_sigma digest does not match the selected data.")

    def save(self, path) -> Path:
        self.validate()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            row_indices=self.row_indices,
            sqrt_mtot_grid=self.sqrt_mtot_grid,
            log_good_stack=self.log_good_stack,
            log_bad_stack=self.log_bad_stack,
            axis_log_b=self.axes["log_b"],
            axis_log_uc=self.axes["log_uc"],
            axis_log_c=self.axes["log_c"],
            node_constants_json=np.asarray(json.dumps(self.node_constants, sort_keys=True)),
            metadata_json=np.asarray(json.dumps(dict(self.metadata), sort_keys=True)),
        )
        return path

    @classmethod
    def load(cls, path) -> "DynamicsLikelihoodShapeStack":
        with np.load(path, allow_pickle=False) as saved:
            return cls(
                row_indices=np.asarray(saved["row_indices"], dtype=np.int64),
                sqrt_mtot_grid=np.asarray(saved["sqrt_mtot_grid"], dtype=np.float64),
                log_good_stack=np.asarray(saved["log_good_stack"], dtype=np.float32),
                log_bad_stack=np.asarray(saved["log_bad_stack"], dtype=np.float32),
                axes={
                    "log_b": np.asarray(saved["axis_log_b"], dtype=np.float64),
                    "log_uc": np.asarray(saved["axis_log_uc"], dtype=np.float64),
                    "log_c": np.asarray(saved["axis_log_c"], dtype=np.float64),
                },
                node_constants=json.loads(str(saved["node_constants_json"])),
                metadata=json.loads(str(saved["metadata_json"])),
            )


def shape_trilinear_weights_numpy(values, axes):
    """Corner indices and trilinear weights on the 3-node shape axes (numpy)."""
    corners = np.empty(8, dtype=np.int64)
    weights = np.empty(8, dtype=np.float64)
    cells, fracs = [], []
    for name, value in zip(("log_b", "log_uc", "log_c"), values):
        n0, n1, n2 = axes[name]
        cell = 1 if value > n1 else 0
        lo, hi = (n1, n2) if cell else (n0, n1)
        frac = float(np.clip((value - lo) / (hi - lo), 0.0, 1.0))
        cells.append(cell); fracs.append(frac)
    for c_b in (0, 1):
        for c_u in (0, 1):
            for c_c in (0, 1):
                idx = (cells[0] + c_b) + 3 * (cells[1] + c_u) + 9 * (cells[2] + c_c)
                corners[c_b * 4 + c_u * 2 + c_c] = idx
                weights[c_b * 4 + c_u * 2 + c_c] = (
                    (fracs[0] if c_b else 1 - fracs[0])
                    * (fracs[1] if c_u else 1 - fracs[1])
                    * (fracs[2] if c_c else 1 - fracs[2])
                )
    return corners, weights


def _bspline_basis_numpy(values, knots, degree):
    """Evaluate the nonnegative, partition-of-unity B-spline basis."""
    x = np.asarray(values, dtype=np.float64)
    t = np.asarray(knots, dtype=np.float64)
    n = t.size - int(degree) - 1
    if n < 1:
        raise ValueError("Knot vector is too short for the requested degree.")
    xc = np.clip(x, t[0], t[-1])
    # Degree-zero half-open intervals, with the right endpoint assigned to the
    # final basis function.  This convention is important for the solar point
    # and for the upper knot boundary.
    basis = np.stack(
        [((xc >= t[i]) & (xc < t[i + 1])).astype(np.float64) for i in range(t.size - 1)],
        axis=-1,
    )
    basis = np.where((xc == t[-1])[..., None], np.eye(1, t.size - 1, t.size - 2)[0], basis)
    for order in range(1, int(degree) + 1):
        terms = []
        for i in range(t.size - order - 1):
            left_den = t[i + order] - t[i]
            right_den = t[i + order + 1] - t[i + 1]
            left = np.zeros_like(xc) if left_den == 0 else (xc - t[i]) / left_den * basis[..., i]
            right = np.zeros_like(xc) if right_den == 0 else (t[i + order + 1] - xc) / right_den * basis[..., i + 1]
            terms.append(left + right)
        basis = np.stack(terms, axis=-1)
    endpoint_basis = np.zeros_like(basis[..., :n])
    endpoint_basis[..., -1] = 1.0
    return np.where((xc == t[-1])[..., None], endpoint_basis, basis[..., :n])


def _bspline_basis_jax(values, knots, degree):
    import jax.numpy as jnp

    x = jnp.asarray(values)
    t = jnp.asarray(knots)
    n = int(t.shape[0]) - int(degree) - 1
    xc = jnp.clip(x, t[0], t[-1])
    basis = jnp.stack(
        [((xc >= t[i]) & (xc < t[i + 1])).astype(x.dtype) for i in range(int(t.shape[0]) - 1)],
        axis=-1,
    )
    endpoint = jnp.arange(int(t.shape[0]) - 1) == int(t.shape[0]) - 2
    basis = jnp.where((xc == t[-1])[..., None], endpoint, basis)
    for order in range(1, int(degree) + 1):
        terms = []
        for i in range(int(t.shape[0]) - order - 1):
            left_den = t[i + order] - t[i]
            right_den = t[i + order + 1] - t[i + 1]
            safe_left_den = jnp.where(left_den == 0, 1.0, left_den)
            safe_right_den = jnp.where(right_den == 0, 1.0, right_den)
            left = (
                (left_den != 0)
                * (xc - t[i])
                / safe_left_den
                * basis[..., i]
            )
            right = (
                (right_den != 0)
                * (t[i + order + 1] - xc)
                / safe_right_den
                * basis[..., i + 1]
            )
            terms.append(left + right)
        basis = jnp.stack(terms, axis=-1)
    endpoint_basis = jnp.zeros_like(basis[..., :n]).at[..., -1].set(1.0)
    return jnp.where((xc == t[-1])[..., None], endpoint_basis, basis[..., :n])


def lookup_interpolate(values, sqrt_grid, table):
    """Row-wise linear interpolation of a lookup table in sqrt(total mass).

    Shared by the NumPyro likelihood and diagnostics so both use the same
    domain convention: values outside [sqrt_grid[0], sqrt_grid[-1]] return
    ``-inf`` instead of extrapolating.
    """
    import jax.numpy as jnp

    sqrt_grid = jnp.asarray(sqrt_grid)
    upper = jnp.clip(jnp.searchsorted(sqrt_grid, values, side="right"), 1, sqrt_grid.size - 1)
    lower = upper - 1
    yl = jnp.take_along_axis(table, lower, axis=1)
    yu = jnp.take_along_axis(table, upper, axis=1)
    xl, xu = sqrt_grid[lower], sqrt_grid[upper]
    out = yl + (values - xl) / (xu - xl) * (yu - yl)
    return jnp.where((values >= sqrt_grid[0]) & (values <= sqrt_grid[-1]), out, -jnp.inf)


class MonotoneTensorSplineMLR:
    """T8b hard-monotone PARSEC-relative tensor-product MLR.

    The sampled coordinates are unconstrained raw variables.  The final
    coefficient surface is built recursively so every valid coefficient
    satisfies the required magnitude and metallicity inequalities.
    """

    def __init__(
        self,
        mass_surface: IsochroneMassSurfaceModel,
        *,
        knots_x: Sequence[float] = DEFAULT_T8_KNOT_X,
        knots_z: Sequence[float] = DEFAULT_T8_KNOT_Z,
        degree_x: int = 3,
        degree_z: int = 3,
        tau_D: float = 0.10,
        tau_x: float = 0.05,
        tau_Z: float = 0.05,
        tau_xZ: float = 0.05,
        solar_anchor_mean: float = 0.0,
        solar_anchor_sigma: float = 0.01 / np.log(10.0),
    ):
        self.mass_surface = mass_surface
        self.dynamics_shape_stack = None
        self.knots_x = _as_native_f64(knots_x, name="knots_x", ndim=1)
        self.knots_z = _as_native_f64(knots_z, name="knots_z", ndim=1)
        self.degree_x, self.degree_z = int(degree_x), int(degree_z)
        self.K_x = self.knots_x.size - self.degree_x - 1
        self.K_Z = self.knots_z.size - self.degree_z - 1
        if self.K_x != 8 or self.K_Z != 4:
            raise ValueError("T8 defaults require 8 x-direction and 4 Z-direction basis functions.")
        if np.any(np.diff(self.knots_x) < 0) or np.any(np.diff(self.knots_z) < 0):
            raise ValueError("B-spline knot vectors must be nondecreasing.")
        if not (
            np.all(self.knots_x[: self.degree_x + 1] == self.knots_x[0])
            and np.all(self.knots_x[-self.degree_x - 1 :] == self.knots_x[-1])
            and np.all(self.knots_z[: self.degree_z + 1] == self.knots_z[0])
            and np.all(self.knots_z[-self.degree_z - 1 :] == self.knots_z[-1])
        ):
            raise ValueError("T8 requires open-clamped knot vectors in both dimensions.")
        if self.knots_x[0] != self.mass_surface.absg_grid_np[0] or self.knots_x[-1] != self.mass_surface.absg_grid_np[-1]:
            raise ValueError("knots_x must cover the mass-surface M_G domain exactly.")
        if self.knots_z[0] != self.mass_surface.mh_grid_np[0] or self.knots_z[-1] != self.mass_surface.mh_grid_np[-1]:
            raise ValueError("knots_z must cover the mass-surface metallicity domain exactly.")
        self.tau_D, self.tau_x, self.tau_Z, self.tau_xZ = map(float, (tau_D, tau_x, tau_Z, tau_xZ))
        self.solar_anchor_mean = float(solar_anchor_mean)
        self.solar_anchor_sigma = float(solar_anchor_sigma)
        if min(self.tau_D, self.tau_x, self.tau_Z, self.tau_xZ, self.solar_anchor_sigma) <= 0:
            raise ValueError("All T8 MLR scales must be positive.")
        self.greville_x = np.array([np.mean(self.knots_x[i + 1:i + self.degree_x + 1]) for i in range(self.K_x)])
        self.greville_z = np.array([np.mean(self.knots_z[i + 1:i + self.degree_z + 1]) for i in range(self.K_Z)])
        self._fit_parsec_projection()
        self.param_names = ["c0", "a", "b", "r", "log_lambda_x", "log_lambda_z", "f_outlier"]
        self.n_params = 1 + (self.K_Z - 1) + (self.K_x - 1) + (self.K_x - 1) * (self.K_Z - 1) + 3
        self.sampler = None
        self.posterior_samples = None
        self._data_set = False

    @property
    def sample_parameter_names(self):
        return ("c0", "a", "b", "r", "log_lambda_x", "log_lambda_z", "f_outlier")

    def _fit_parsec_projection(self):
        x = np.linspace(self.knots_x[0], self.knots_x[-1], 501)
        z = np.linspace(self.knots_z[0], self.knots_z[-1], 81)
        bx = _bspline_basis_numpy(x, self.knots_x, self.degree_x)
        bz = _bspline_basis_numpy(z, self.knots_z, self.degree_z)
        # Keep the design rows in the same (Z, M_G) order as ``gp`` below.
        # A transposed row order silently fits the right values at the wrong
        # coordinates while still returning a full-rank least-squares result.
        design = np.einsum("zh,xi->zxih", bz, bx).reshape(
            -1, self.K_x * self.K_Z
        )
        gp = np.log10(self.mass_surface.mass_from_absg_mh(x[None, :], z[:, None]))
        coeff, _, rank, singular = np.linalg.lstsq(design, gp.reshape(-1), rcond=None)
        self.parsec_projection = coeff.reshape(self.K_x, self.K_Z)
        self.Pi = self.parsec_projection
        projected = (design @ coeff).reshape(gp.shape)
        # T8 notation: epsilon_P = g_P^* - g_P.
        error = projected - gp
        self.projection_grid_x, self.projection_grid_z = x, z
        self.projection_error_grid = error
        self.epsilon_P_grid = error
        self.projection_metadata = {
            "projection": "ordinary_least_squares_fixed_parsec",
            "projection_grid_shape": [501, 81],
            "projection_grid_digest": array_digest(x, z, gp),
            "projection_rank": int(rank),
            "projection_condition_number": float(singular[0] / singular[-1]),
            "epsilon_P_rms_dex": float(np.sqrt(np.mean(error**2))),
            "epsilon_P_max_abs_dex": float(np.max(np.abs(error))),
        }

    @staticmethod
    def _softplus(value):
        return np.logaddexp(0.0, np.asarray(value, dtype=np.float64))

    def lambda_references(self) -> Tuple[float, float]:
        """Return separate PARSEC-derived raw increment scales for x and Z."""
        dx = self.parsec_projection[:-1] - self.parsec_projection[1:]
        dz = self.parsec_projection[:, 1:] - self.parsec_projection[:, :-1]
        fallback = 0.02 / float(self._softplus(0.0))
        positive_x = dx[dx > 0]
        positive_z = dz[dz > 0]
        reference_x = (
            float(np.median(positive_x) / self._softplus(0.0))
            if positive_x.size
            else fallback
        )
        reference_z = (
            float(np.median(positive_z) / self._softplus(0.0))
            if positive_z.size
            else fallback
        )
        return reference_x, reference_z

    def lambda_reference(self):
        """Compatibility scalar; new code should use :meth:`lambda_references`."""
        return float(np.sqrt(np.prod(self.lambda_references())))

    @staticmethod
    def _isotonic_increasing(values):
        """Unweighted pool-adjacent-violators projection onto increasing values."""
        values = np.asarray(values, dtype=np.float64)
        levels = []
        counts = []
        for value in values:
            levels.append(float(value))
            counts.append(1)
            while len(levels) >= 2 and levels[-2] > levels[-1]:
                count = counts[-2] + counts[-1]
                pooled = (
                    counts[-2] * levels[-2] + counts[-1] * levels[-1]
                ) / count
                levels[-2:] = [pooled]
                counts[-2:] = [count]
        return np.concatenate(
            [np.full(count, level, dtype=np.float64) for level, count in zip(levels, counts)]
        )

    def _monotone_projection_for_initialization(self):
        """Alternating isotonic projection used only to initialize NUTS."""
        target = np.asarray(self.parsec_projection, dtype=np.float64).copy()
        for _ in range(200):
            previous = target.copy()
            for h in range(self.K_Z):
                target[:, h] = -self._isotonic_increasing(-target[:, h])
            for i in range(self.K_x):
                target[i, :] = self._isotonic_increasing(target[i, :])
            if np.max(np.abs(target - previous)) < 1e-12:
                break
        anchor_value = float(self.g_from_theta(4.67, 0.0, target))
        target += self.solar_anchor_mean - anchor_value
        return target

    @staticmethod
    def _inverse_softplus(value):
        value = np.maximum(np.asarray(value, dtype=np.float64), 1e-8)
        return np.where(value > 20.0, value, np.log(np.expm1(value)))

    def initial_raw_parameters(self):
        """Return a feasible, PARSEC-near raw state for sampler initialization.

        The alternating isotonic projection is not part of the likelihood or
        prior.  It only avoids starting NUTS at a strongly penalized surface
        that violates the hard T8 coefficient order.
        """
        target = self._monotone_projection_for_initialization()
        lambda_x, lambda_z = self.lambda_references()
        # Keep initialization in the strict interior of both transforms.  A
        # raw bound of four corresponds to sigmoid fractions [0.018, 0.982]
        # and prevents flat isotonic blocks from becoming saturated tails.
        raw_bound = 4.0
        a = np.clip(
            self._inverse_softplus(np.diff(target[0, :]) / lambda_z),
            -raw_bound, raw_bound,
        )
        b = np.clip(
            self._inverse_softplus((target[:-1, 0] - target[1:, 0]) / lambda_x),
            -raw_bound, raw_bound,
        )
        params = {
            "c0": float(target[0, 0]),
            "a": np.asarray(a, dtype=np.float64),
            "b": np.asarray(b, dtype=np.float64),
            "r": np.zeros((self.K_x - 1, self.K_Z - 1), dtype=np.float64),
            "log_lambda_x": float(np.log(lambda_x)),
            "log_lambda_z": float(np.log(lambda_z)),
            "f_outlier": 0.2,
        }
        if self.dynamics_shape_stack is not None:
            axes = self.dynamics_shape_stack.axes
            params.update({
                "log_good_shape_b": float(np.mean(axes["log_b"])),
                "log_good_shape_uc": float(np.mean(axes["log_uc"])),
                "log_good_shape_c": float(np.mean(axes["log_c"])),
            })
        # Invert each diagonal recursion against the target, using the already
        # reconstructed neighbours as its exact feasible bounds.
        for total in range(2, self.K_x + self.K_Z - 1):
            theta = self.theta_from_raw(params)
            for i in range(1, self.K_x):
                h = total - i
                if 1 <= h < self.K_Z:
                    lo, hi = theta[i, h - 1], theta[i - 1, h]
                    fraction = (target[i, h] - lo) / max(hi - lo, 1e-12)
                    fraction = float(
                        np.clip(fraction, expit(-raw_bound), expit(raw_bound))
                    )
                    params["r"][i - 1, h - 1] = np.log(fraction / (1.0 - fraction))
        # c0 translates the complete recursively constructed surface, so this
        # final correction makes the requested solar anchor exact at start.
        theta = self.theta_from_raw(params)
        params["c0"] += self.solar_anchor_mean - float(
            self.g_from_theta(4.67, 0.0, theta)
        )
        return params

    def theta_from_raw(self, params: Mapping[str, np.ndarray]) -> np.ndarray:
        c0 = float(np.asarray(params["c0"]))
        a = np.asarray(params["a"], dtype=np.float64).reshape(self.K_Z - 1)
        b = np.asarray(params["b"], dtype=np.float64).reshape(self.K_x - 1)
        r = np.asarray(params["r"], dtype=np.float64).reshape(self.K_x - 1, self.K_Z - 1)
        lam_x, lam_z = np.exp(float(np.asarray(params["log_lambda_x"]))), np.exp(float(np.asarray(params["log_lambda_z"])))
        theta = np.zeros((self.K_x, self.K_Z), dtype=np.float64)
        theta[0, 0] = c0
        for h in range(1, self.K_Z):
            theta[0, h] = theta[0, h - 1] + lam_z * self._softplus(a[h - 1])
        for i in range(1, self.K_x):
            theta[i, 0] = theta[i - 1, 0] - lam_x * self._softplus(b[i - 1])
        for total in range(2, self.K_x + self.K_Z - 1):
            for i in range(1, self.K_x):
                h = total - i
                if 1 <= h < self.K_Z:
                    lo, hi = theta[i, h - 1], theta[i - 1, h]
                    theta[i, h] = lo + (hi - lo) * expit(r[i - 1, h - 1])
        return theta

    def _theta_from_raw_jax(self, c0, a, b, r, log_lambda_x, log_lambda_z):
        import jax.numpy as jnp
        from jax.nn import sigmoid
        theta = jnp.zeros((self.K_x, self.K_Z)).at[0, 0].set(c0)
        lx, lz = jnp.exp(log_lambda_x), jnp.exp(log_lambda_z)
        for h in range(1, self.K_Z):
            theta = theta.at[0, h].set(theta[0, h - 1] + lz * jnp.logaddexp(0.0, a[h - 1]))
        for i in range(1, self.K_x):
            theta = theta.at[i, 0].set(theta[i - 1, 0] - lx * jnp.logaddexp(0.0, b[i - 1]))
        for total in range(2, self.K_x + self.K_Z - 1):
            for i in range(1, self.K_x):
                h = total - i
                if 1 <= h < self.K_Z:
                    lo, hi = theta[i, h - 1], theta[i - 1, h]
                    theta = theta.at[i, h].set(lo + (hi - lo) * sigmoid(r[i - 1, h - 1]))
        return theta

    def g_from_theta(self, absg, mh, theta):
        bx = _bspline_basis_numpy(absg, self.knots_x, self.degree_x)
        bz = _bspline_basis_numpy(mh, self.knots_z, self.degree_z)
        return np.einsum("...i,...h,ih->...", bx, bz, theta)

    def g_parsec_projection(self, absg, mh):
        return self.g_from_theta(absg, mh, self.parsec_projection)

    def g_from_raw(self, absg, mh, params):
        return self.g_from_theta(absg, mh, self.theta_from_raw(params))

    def mass_from_absg_mh(self, absg, mh, params):
        return np.power(10.0, self.g_from_raw(absg, mh, params))

    def spacing_aware_penalties(self, D):
        """Return spacing-normalized differences with units of dex.

        Greville spacings are used to form the non-uniform finite-difference
        derivatives.  Multiplication by the corresponding uniform reference
        spacing restores the ordinary finite-difference scale, so the fixed
        ``tau_*`` hyperparameters retain their documented dex interpretation.
        """
        D = np.asarray(D, dtype=np.float64)
        hx = np.diff(self.greville_x)
        hz = np.diff(self.greville_z)
        href_x = (self.greville_x[-1] - self.greville_x[0]) / (self.K_x - 1)
        href_z = (self.greville_z[-1] - self.greville_z[0]) / (self.K_Z - 1)
        slope_x = np.diff(D, axis=0) / hx[:, None]
        slope_z = np.diff(D, axis=1) / hz[None, :]
        d2x = (
            2.0
            * np.diff(slope_x, axis=0)
            / (hx[:-1, None] + hx[1:, None])
            * href_x**2
        )
        d2z = (
            2.0
            * np.diff(slope_z, axis=1)
            / (hz[None, :-1] + hz[None, 1:])
            * href_z**2
        )
        dxz = (
            D[1:, 1:] - D[1:, :-1] - D[:-1, 1:] + D[:-1, :-1]
        ) / (hx[:, None] * hz[None, :]) * href_x * href_z
        return d2x, d2z, dxz

    def log_prior_terms(self, params):
        theta = self.theta_from_raw(params)
        D = theta - self.parsec_projection
        d2x, d2z, dxz = self.spacing_aware_penalties(D)
        return {
            "D_shrinkage": -0.5 * np.sum((D / self.tau_D) ** 2),
            "D_smoothness_x": -0.5 * np.sum((d2x / self.tau_x) ** 2),
            "D_smoothness_Z": -0.5 * np.sum((d2z / self.tau_Z) ** 2),
            "D_smoothness_xZ": -0.5 * np.sum((dxz / self.tau_xZ) ** 2),
        }

    def correction_log10(self, absg, mh, params):
        return self.g_from_raw(absg, mh, params) - self.g_parsec_projection(absg, mh)

    def set_data(self, *, row_indices, u, u_sigma, absg, metallicity_grid,
                 dynamics_lookup=None, dynamics_shape_stack=None):
        metallicity_grid.validate()
        posterior_model = str(metallicity_grid.metadata.get("model", ""))
        if not (posterior_model.startswith("t8") or "_t8_" in posterior_model):
            raise ValueError("The MLR stage requires a T8 metallicity posterior; rebuild calibration.")
        if (dynamics_lookup is None) == (dynamics_shape_stack is None):
            raise ValueError(
                "Provide exactly one dynamics source: a fixed lookup or a shape stack."
            )
        self.row_indices = np.asarray(row_indices, dtype=np.int64)
        if not np.array_equal(self.row_indices, metallicity_grid.row_indices):
            raise ValueError("Stage-two row indices do not match the metallicity posterior file.")
        self.u, self.u_sigma = _as_native_f64(u, name="u", ndim=1), _as_native_f64(u_sigma, name="u_sigma", ndim=1)
        self.absg = _as_native_f64(absg, name="absg", ndim=2)
        n = self.row_indices.size
        if self.u.shape != (n,) or self.u_sigma.shape != (n,) or self.absg.shape != (n, 2):
            raise ValueError("Stage-two arrays have inconsistent shapes.")
        if np.any(~np.isfinite(np.column_stack([self.u, self.u_sigma, self.absg]))):
            raise ValueError("Stage-two arrays must contain only finite values.")
        if np.any(self.u <= 0) or np.any(self.u_sigma <= 0):
            raise ValueError("u and u_sigma must be strictly positive.")
        self.dynamics_shape_stack = dynamics_shape_stack
        if dynamics_shape_stack is not None:
            dynamics_shape_stack.validate_for(row_indices=self.row_indices, u=self.u, u_sigma=self.u_sigma)
            self.dynamics_lookup = None
            self.sqrt_mtot_grid = np.asarray(dynamics_shape_stack.sqrt_mtot_grid, dtype=np.float64)
            self.log_good_lookup = None
            self.log_bad_lookup = None
        else:
            dynamics_lookup.validate_for(row_indices=self.row_indices, u=self.u, u_sigma=self.u_sigma)
            self.dynamics_lookup = dynamics_lookup
            self.sqrt_mtot_grid = np.asarray(dynamics_lookup.sqrt_mtot_grid, dtype=np.float64)
            self.log_good_lookup = np.asarray(dynamics_lookup.log_good, dtype=np.float32)
            self.log_bad_lookup = np.asarray(dynamics_lookup.log_bad, dtype=np.float32)
        self.z_grid = np.asarray(metallicity_grid.z_grid, dtype=np.float64)
        self.z_probabilities = np.asarray(metallicity_grid.probabilities, dtype=np.float64)
        if self.z_probabilities.shape != (n, self.z_grid.size):
            raise ValueError("Metallicity probability array has the wrong shape.")
        if np.any(self.absg < self.knots_x[0]) or np.any(self.absg > self.knots_x[-1]):
            raise ValueError("Observed component magnitudes lie outside the MLR knot domain.")
        self._data_set = True

    def _build_numpyro_model(self):
        import jax.numpy as jnp
        from jax.scipy.special import logsumexp as jax_logsumexp
        import numpyro
        import numpyro.distributions as dist
        z_grid, sqrt_grid = jnp.asarray(self.z_grid), jnp.asarray(self.sqrt_mtot_grid)
        # These basis values depend only on fixed observations/grid locations,
        # so precompute them outside the NumPyro trace.
        bx1 = jnp.asarray(
            _bspline_basis_numpy(self.absg[:, 0], self.knots_x, self.degree_x)
        )
        bx2 = jnp.asarray(
            _bspline_basis_numpy(self.absg[:, 1], self.knots_x, self.degree_x)
        )
        bz = jnp.asarray(_bspline_basis_numpy(self.z_grid, self.knots_z, self.degree_z))
        anchor_bx = jnp.asarray(
            _bspline_basis_numpy(4.67, self.knots_x, self.degree_x)
        )
        anchor_bz = jnp.asarray(
            _bspline_basis_numpy(0.0, self.knots_z, self.degree_z)
        )

        def interp(values, table):
            return lookup_interpolate(values, sqrt_grid, table)

        if self.dynamics_shape_stack is not None:
            stack_axes = {
                name: jnp.asarray(self.dynamics_shape_stack.axes[name])
                for name in DynamicsLikelihoodShapeStack.AXIS_NAMES
            }

            def effective_shape_tables(log_b, log_uc, log_c, stack_good, stack_bad):
                """Trilinear mix of the 27 member tables at sampled (log B, log uc, log C)."""
                fracs, cells = [], []
                for axis_value, name in zip((log_b, log_uc, log_c), DynamicsLikelihoodShapeStack.AXIS_NAMES):
                    n0, n1, n2 = stack_axes[name]
                    cell = (axis_value > n1).astype(jnp.int32)
                    lo = jnp.where(cell == 1, n1, n0)
                    hi = jnp.where(cell == 1, n2, n1)
                    fracs.append(jnp.clip((axis_value - lo) / (hi - lo), 0.0, 1.0))
                    cells.append(cell)
                corner_log_weights = []
                corner_indices = []
                for c_b in (0, 1):
                    for c_u in (0, 1):
                        for c_c in (0, 1):
                            corner_indices.append(
                                (cells[0] + c_b) + 3 * (cells[1] + c_u) + 9 * (cells[2] + c_c)
                            )
                            w = (
                                (fracs[0] if c_b else 1.0 - fracs[0])
                                * (fracs[1] if c_u else 1.0 - fracs[1])
                                * (fracs[2] if c_c else 1.0 - fracs[2])
                            )
                            corner_log_weights.append(jnp.log(jnp.maximum(w, 1e-30)))
                indices = jnp.asarray(jnp.stack(corner_indices))
                log_w = jnp.asarray(jnp.stack(corner_log_weights))
                good = jax_logsumexp(log_w[:, None, None] + stack_good[indices], axis=0)
                bad = jax_logsumexp(log_w[:, None, None] + stack_bad[indices], axis=0)
                return good, bad

        def model(absg, z_probabilities, log_good_lookup, log_bad_lookup):
            if self.dynamics_shape_stack is not None:
                axis_b, axis_uc, axis_c = (
                    stack_axes["log_b"], stack_axes["log_uc"], stack_axes["log_c"]
                )
                log_b = numpyro.sample(
                    "log_good_shape_b", dist.Uniform(float(axis_b[0]), float(axis_b[-1]))
                )
                log_uc = numpyro.sample(
                    "log_good_shape_uc", dist.Uniform(float(axis_uc[0]), float(axis_uc[-1]))
                )
                log_c = numpyro.sample(
                    "log_good_shape_c", dist.Uniform(float(axis_c[0]), float(axis_c[-1]))
                )
                numpyro.deterministic("good_shape_B", jnp.exp(log_b))
                numpyro.deterministic("good_shape_uc", jnp.exp(log_uc))
                numpyro.deterministic("good_shape_C", jnp.exp(log_c))
            c0 = numpyro.sample("c0", dist.Normal(float(self.parsec_projection[0, 0]), 0.10))
            a = numpyro.sample("a", dist.Normal(0.0, 1.0).expand([self.K_Z - 1]))
            b = numpyro.sample("b", dist.Normal(0.0, 1.0).expand([self.K_x - 1]))
            r = numpyro.sample("r", dist.Normal(0.0, 1.0).expand([self.K_x - 1, self.K_Z - 1]))
            lambda_ref_x, lambda_ref_z = self.lambda_references()
            log_lambda_x = numpyro.sample(
                "log_lambda_x", dist.Normal(np.log(lambda_ref_x), 0.7)
            )
            log_lambda_z = numpyro.sample(
                "log_lambda_z", dist.Normal(np.log(lambda_ref_z), 0.7)
            )
            f_outlier = numpyro.sample("f_outlier", dist.Beta(3.0, 12.0))
            theta = self._theta_from_raw_jax(c0, a, b, r, log_lambda_x, log_lambda_z)
            D = theta - jnp.asarray(self.parsec_projection)
            hx = jnp.asarray(np.diff(self.greville_x))
            hz = jnp.asarray(np.diff(self.greville_z))
            href_x = float((self.greville_x[-1] - self.greville_x[0]) / (self.K_x - 1))
            href_z = float((self.greville_z[-1] - self.greville_z[0]) / (self.K_Z - 1))
            slope_x = jnp.diff(D, axis=0) / hx[:, None]
            slope_z = jnp.diff(D, axis=1) / hz[None, :]
            d2x = 2.0 * jnp.diff(slope_x, axis=0) / (hx[:-1, None] + hx[1:, None]) * href_x**2
            d2z = 2.0 * jnp.diff(slope_z, axis=1) / (hz[None, :-1] + hz[None, 1:]) * href_z**2
            dxz = (D[1:, 1:] - D[1:, :-1] - D[:-1, 1:] + D[:-1, :-1]) / (hx[:, None] * hz[None, :]) * href_x * href_z
            numpyro.factor("D_shrinkage", -0.5 * jnp.sum((D / self.tau_D) ** 2))
            numpyro.factor("D_smoothness", -0.5 * (jnp.sum((d2x / self.tau_x) ** 2) + jnp.sum((d2z / self.tau_Z) ** 2) + jnp.sum((dxz / self.tau_xZ) ** 2)))
            gp1 = jnp.einsum("ni,qh,ih->nq", bx1, bz, theta)
            gp2 = jnp.einsum("ni,qh,ih->nq", bx2, bz, theta)
            m1 = jnp.power(10.0, gp1)
            m2 = jnp.power(10.0, gp2)
            sqrt_mtot = jnp.sqrt(jnp.maximum(m1 + m2, 1e-12))
            if self.dynamics_shape_stack is not None:
                log_good, log_bad = effective_shape_tables(
                    log_b, log_uc, log_c, log_good_lookup, log_bad_lookup
                )
            else:
                log_good = interp(sqrt_mtot, log_good_lookup)
                log_bad = interp(sqrt_mtot, log_bad_lookup)
            log_conditional = jnp.logaddexp(jnp.log1p(-f_outlier) + log_good, jnp.log(f_outlier) + log_bad)
            numpyro.factor("dynamics", jnp.sum(jax_logsumexp(jnp.log(jnp.maximum(z_probabilities, 1e-30)) + log_conditional, axis=1)))
            solar_g = jnp.einsum("i,h,ih->", anchor_bx, anchor_bz, theta)
            numpyro.deterministic("solar_anchor_g", solar_g)
            numpyro.factor(
                "solar_anchor",
                dist.Normal(self.solar_anchor_mean, self.solar_anchor_sigma).log_prob(
                    solar_g
                ),
            )
        return model

    def baseline_dynamics_log_likelihood(self, f_outlier=0.2):
        if not self._data_set:
            raise ValueError("Call set_data() before evaluating the baseline likelihood.")
        if self.dynamics_shape_stack is not None:
            raise ValueError("Baseline lookup likelihood is not defined for shape-stack runs.")
        mh = self.z_grid[None, :]
        sqrt_mtot = np.sqrt(self.mass_surface.mass_from_absg_mh(self.absg[:, 0, None], mh) + self.mass_surface.mass_from_absg_mh(self.absg[:, 1, None], mh))
        good = self.dynamics_lookup.interpolate_numpy(sqrt_mtot, component="good")
        bad = self.dynamics_lookup.interpolate_numpy(sqrt_mtot, component="bad")
        return marginalize_dynamics(np.logaddexp(np.log1p(-f_outlier) + good, np.log(f_outlier) + bad), self.z_probabilities)

    def run_mcmc(self, *, num_warmup=1000, num_samples=1000, num_chains=4, seed=43, progress_bar=True, **nuts_kwargs):
        if not self._data_set:
            raise ValueError("Call set_data() before run_mcmc().")
        jax = _configure_jax_gpu_fallback()
        import jax.numpy as jnp
        from numpyro.infer import MCMC, NUTS
        from numpyro.infer.initialization import init_to_value
        from numpyro.infer.util import log_density
        init_numpy = self.initial_raw_parameters()
        init = {name: jnp.asarray(value) for name, value in init_numpy.items()}
        initial_theta = self.theta_from_raw(init_numpy)
        initial_g1 = self.g_from_theta(
            self.absg[:, 0, None], self.z_grid[None, :], initial_theta
        )
        initial_g2 = self.g_from_theta(
            self.absg[:, 1, None], self.z_grid[None, :], initial_theta
        )
        initial_s = np.sqrt(10.0**initial_g1 + 10.0**initial_g2)
        lookup_min, lookup_max = self.sqrt_mtot_grid[[0, -1]]
        if np.any(initial_s < lookup_min) or np.any(initial_s > lookup_max):
            raise ValueError(
                "T8 feasible initialization lies outside the dynamics lookup domain: "
                f"initial=[{np.min(initial_s):.6g},{np.max(initial_s):.6g}], "
                f"lookup=[{lookup_min:.6g},{lookup_max:.6g}]. Rebuild a wider lookup."
            )
        model = self._build_numpyro_model()
        if self.dynamics_shape_stack is not None:
            model_args = (
                jnp.asarray(self.absg), jnp.asarray(self.z_probabilities),
                jnp.asarray(self.dynamics_shape_stack.log_good_stack),
                jnp.asarray(self.dynamics_shape_stack.log_bad_stack),
            )
        else:
            model_args = (
                jnp.asarray(self.absg), jnp.asarray(self.z_probabilities),
                jnp.asarray(self.log_good_lookup), jnp.asarray(self.log_bad_lookup),
            )

        def initial_potential(parameters):
            return -log_density(model, model_args, {}, parameters)[0]

        potential, gradient = jax.value_and_grad(initial_potential)(init)
        gradient_leaves = jax.tree_util.tree_leaves(gradient)
        if not np.isfinite(float(potential)) or not all(
            np.all(np.isfinite(np.asarray(leaf))) for leaf in gradient_leaves
        ):
            raise ValueError("T8 MLR initialization has a non-finite potential or gradient.")
        projection_delta = initial_theta - self.parsec_projection
        self.initialization_metadata = {
            "method": "alternating_isotonic_projection_then_exact_recursion_inverse",
            "likelihood_role": "initialization_only",
            "potential_energy": float(potential),
            "gradient_finite": True,
            "initial_sqrt_mtot_min": float(np.min(initial_s)),
            "initial_sqrt_mtot_max": float(np.max(initial_s)),
            "lookup_sqrt_mtot_min": float(lookup_min),
            "lookup_sqrt_mtot_max": float(lookup_max),
            "solar_anchor_g": float(self.g_from_theta(4.67, 0.0, initial_theta)),
            "theta_minus_projection_rms_dex": float(np.sqrt(np.mean(projection_delta**2))),
            "theta_minus_projection_max_abs_dex": float(np.max(np.abs(projection_delta))),
        }
        strategy = nuts_kwargs.pop("init_strategy", init_to_value(values=init))
        mcmc = MCMC(NUTS(model, init_strategy=strategy, **nuts_kwargs), num_warmup=int(num_warmup), num_samples=int(num_samples), num_chains=int(num_chains), progress_bar=bool(progress_bar))
        mcmc.run(jax.random.PRNGKey(int(seed)), jnp.asarray(self.absg), jnp.asarray(self.z_probabilities), jnp.asarray(self.log_good_lookup), jnp.asarray(self.log_bad_lookup), extra_fields=("diverging", "energy", "potential_energy", "num_steps", "accept_prob"))
        self.sampler = mcmc
        self.posterior_samples = {name: np.asarray(values) for name, values in mcmc.get_samples().items()}
        return mcmc

    def correction_grid(self, *, absg_grid=None, z_grid=None, max_draws=512):
        if self.posterior_samples is None:
            raise ValueError("Run run_mcmc() before correction_grid().")
        x = np.linspace(self.knots_x[0], self.knots_x[-1], 201) if absg_grid is None else np.asarray(absg_grid, dtype=float)
        z = np.array([-1.0, -0.5, 0.0, 0.3, 0.6]) if z_grid is None else np.asarray(z_grid, dtype=float)
        n = self.posterior_samples["c0"].shape[0]
        idx = np.linspace(0, n - 1, min(int(max_draws), n), dtype=np.int64)
        shape = (idx.size, z.size, x.size)
        delta_star = np.empty(shape)
        mass = np.empty(shape)
        parsec = self.mass_surface.mass_from_absg_mh(x[None, :], z[:, None])
        gparsec = np.log10(parsec)
        for q, j in enumerate(idx):
            params = {name: self.posterior_samples[name][j] for name in ("c0", "a", "b", "r", "log_lambda_x", "log_lambda_z")}
            g = self.g_from_raw(x[None, :], z[:, None], params)
            delta_star[q] = g - self.g_parsec_projection(x[None, :], z[:, None])
            mass[q] = 10.0 ** g
        epsilon = self.g_parsec_projection(x[None, :], z[:, None]) - gparsec
        delta_p = delta_star + epsilon[None, :, :]
        percentiles = np.array([16.0, 50.0, 84.0])
        violation_x = np.diff(mass, axis=2) > 1e-12
        violation_z = np.diff(mass, axis=1) < -1e-12
        violation_count = np.count_nonzero(violation_x) + np.count_nonzero(violation_z)
        cell_count = violation_x.size + violation_z.size
        draw_violation = np.any(violation_x, axis=(1, 2)) | np.any(
            violation_z, axis=(1, 2)
        )
        return {
            "absg_grid": x,
            "z_grid": z,
            "percentiles": percentiles,
            "parsec_mass": parsec,
            "delta_star": delta_star,
            "epsilon_P": epsilon,
            "delta_P": delta_p,
            "mass": mass,
            "correction_percent": 100.0 * (10.0 ** delta_p - 1.0),
            "mass_percentiles": np.percentile(mass, percentiles, axis=0),
            "correction_percentiles": np.percentile(
                100.0 * (10.0 ** delta_p - 1.0), percentiles, axis=0
            ),
            "monotone_cell_violation_fraction": np.asarray(
                violation_count / max(cell_count, 1)
            ),
            "monotone_draw_violation_fraction": np.asarray(
                np.mean(draw_violation)
            ),
            "monotone_x_cell_violation_fraction": np.asarray(
                np.mean(violation_x)
            ),
            "monotone_z_cell_violation_fraction": np.asarray(
                np.mean(violation_z)
            ),
        }


class ThreeKnotMetallicityMLR:
    """Stage-two PARSEC-relative MLR with two three-knot correction curves."""

    def __init__(
        self,
        mass_surface: IsochroneMassSurfaceModel,
        *,
        mg_knots: Sequence[float] = DEFAULT_MG_KNOTS,
    ):
        self.mass_surface = mass_surface
        self.mg_knots = _as_native_f64(mg_knots, name="mg_knots", ndim=1)
        if self.mg_knots.shape != (3,) or np.any(np.diff(self.mg_knots) <= 0):
            raise ValueError("mg_knots must contain exactly three increasing values.")
        self.sampler = None
        self.posterior_samples = None
        self._data_set = False

    def correction_log10(self, absg, mh, f0_knots, fz_knots) -> np.ndarray:
        absg = np.asarray(absg, dtype=np.float64)
        mh = np.asarray(mh, dtype=np.float64)
        absg, mh = np.broadcast_arrays(absg, mh)
        f0 = np.interp(absg, self.mg_knots, np.asarray(f0_knots, dtype=float))
        fz = np.interp(absg, self.mg_knots, np.asarray(fz_knots, dtype=float))
        return f0 + mh * fz

    def mass_from_absg_mh(self, absg, mh, f0_knots, fz_knots) -> np.ndarray:
        baseline = self.mass_surface.mass_from_absg_mh(absg, mh)
        return baseline * np.power(10.0, self.correction_log10(absg, mh, f0_knots, fz_knots))

    def set_data(
        self,
        *,
        row_indices,
        u,
        u_sigma,
        absg,
        metallicity_grid: MetallicityPosteriorGrid,
        dynamics_lookup: DynamicsLikelihoodLookup,
    ) -> None:
        metallicity_grid.validate()
        self.row_indices = np.asarray(row_indices, dtype=np.int64)
        if not np.array_equal(self.row_indices, metallicity_grid.row_indices):
            raise ValueError("Stage-two row indices do not match the metallicity posterior file.")
        self.u = _as_native_f64(u, name="u", ndim=1)
        self.u_sigma = _as_native_f64(u_sigma, name="u_sigma", ndim=1)
        self.absg = _as_native_f64(absg, name="absg", ndim=2)
        n = self.row_indices.size
        if self.u.shape != (n,) or self.u_sigma.shape != (n,) or self.absg.shape != (n, 2):
            raise ValueError("Stage-two arrays have inconsistent shapes.")
        if np.any(~np.isfinite(np.column_stack([self.u, self.u_sigma, self.absg]))):
            raise ValueError("Stage-two arrays must contain only finite values.")
        if np.any(self.u <= 0) or np.any(self.u_sigma <= 0):
            raise ValueError("u and u_sigma must be strictly positive.")
        dynamics_lookup.validate_for(
            row_indices=self.row_indices, u=self.u, u_sigma=self.u_sigma
        )
        self.dynamics_lookup = dynamics_lookup
        self.sqrt_mtot_grid = np.asarray(
            dynamics_lookup.sqrt_mtot_grid, dtype=np.float64
        )
        self.log_good_lookup = np.asarray(dynamics_lookup.log_good, dtype=np.float32)
        self.log_bad_lookup = np.asarray(dynamics_lookup.log_bad, dtype=np.float32)
        self.z_grid = np.asarray(metallicity_grid.z_grid, dtype=np.float64)
        self.z_probabilities = np.asarray(metallicity_grid.probabilities, dtype=np.float64)
        if self.z_probabilities.shape != (n, self.z_grid.size):
            raise ValueError("Metallicity probability array has the wrong shape.")
        self._data_set = True

    def _build_numpyro_model(self):
        import jax.numpy as jnp
        from jax.scipy.special import logsumexp as jax_logsumexp
        import numpyro
        import numpyro.distributions as dist

        mg_knots = jnp.asarray(self.mg_knots)
        z_grid = jnp.asarray(self.z_grid)
        sqrt_grid = jnp.asarray(self.sqrt_mtot_grid)

        def interpolate_lookup(sqrt_mtot, log_table):
            """Interpolate each system's table at every metallicity node."""
            upper = jnp.searchsorted(sqrt_grid, sqrt_mtot, side="right")
            upper = jnp.clip(upper, 1, sqrt_grid.size - 1)
            lower = upper - 1
            y_lower = jnp.take_along_axis(log_table, lower, axis=1)
            y_upper = jnp.take_along_axis(log_table, upper, axis=1)
            x_lower = sqrt_grid[lower]
            x_upper = sqrt_grid[upper]
            weight = (sqrt_mtot - x_lower) / (x_upper - x_lower)
            interpolated = y_lower + weight * (y_upper - y_lower)
            in_domain = (sqrt_mtot >= sqrt_grid[0]) & (sqrt_mtot <= sqrt_grid[-1])
            return jnp.where(in_domain, interpolated, -jnp.inf)

        def model(absg, z_probabilities, log_good_lookup, log_bad_lookup):
            f0_knots = numpyro.sample("f0_knots", dist.Normal(0.0, 0.10).expand([3]))
            fz_knots = numpyro.sample("fz_knots", dist.Normal(0.0, 0.15).expand([3]))
            numpyro.factor(
                "f0_smoothness",
                dist.Normal(0.0, 0.05).log_prob(f0_knots[0] - 2.0 * f0_knots[1] + f0_knots[2]),
            )
            numpyro.factor(
                "fz_smoothness",
                dist.Normal(0.0, 0.05).log_prob(fz_knots[0] - 2.0 * fz_knots[1] + fz_knots[2]),
            )
            f_outlier = numpyro.sample("f_outlier", dist.Beta(3.0, 12.0))

            absg1 = absg[:, 0, None]
            absg2 = absg[:, 1, None]
            mh = z_grid[None, :]
            f0_1 = jnp.interp(absg1, mg_knots, f0_knots)
            f0_2 = jnp.interp(absg2, mg_knots, f0_knots)
            fz_1 = jnp.interp(absg1, mg_knots, fz_knots)
            fz_2 = jnp.interp(absg2, mg_knots, fz_knots)
            m1 = self.mass_surface.mass_from_absg_mh_jax(absg1, mh) * jnp.power(
                10.0, f0_1 + mh * fz_1
            )
            m2 = self.mass_surface.mass_from_absg_mh_jax(absg2, mh) * jnp.power(
                10.0, f0_2 + mh * fz_2
            )
            sqrt_mtot = jnp.sqrt(jnp.maximum(m1 + m2, 1e-12))
            log_good = interpolate_lookup(sqrt_mtot, log_good_lookup)
            log_bad = interpolate_lookup(sqrt_mtot, log_bad_lookup)
            log_conditional = jnp.logaddexp(
                jnp.log1p(-f_outlier) + log_good,
                jnp.log(f_outlier) + log_bad,
            )
            log_likelihood = jax_logsumexp(
                jnp.log(jnp.maximum(z_probabilities, 1e-30)) + log_conditional,
                axis=1,
            )
            numpyro.factor("dynamics", jnp.sum(log_likelihood))

            anchor_f0 = jnp.interp(jnp.asarray(4.67), mg_knots, f0_knots)
            anchor_baseline = self.mass_surface.mass_from_absg_mh_jax(
                jnp.asarray(4.67), jnp.asarray(0.0)
            )
            anchor_mass = anchor_baseline * jnp.power(10.0, anchor_f0)
            numpyro.deterministic("anchor_mass", anchor_mass)
            numpyro.factor(
                "solar_anchor", dist.Normal(1.0, 0.01).log_prob(anchor_mass)
            )

        return model

    def baseline_dynamics_log_likelihood(self, f_outlier: float = 0.2) -> np.ndarray:
        """Dynamics-only log likelihood per system at zero MLR correction."""
        if not self._data_set:
            raise ValueError("Call set_data() before evaluating the baseline likelihood.")
        if not 0.0 < float(f_outlier) < 1.0:
            raise ValueError("f_outlier must lie strictly between zero and one.")
        mh = self.z_grid[None, :]
        m1 = self.mass_surface.mass_from_absg_mh(self.absg[:, 0, None], mh)
        m2 = self.mass_surface.mass_from_absg_mh(self.absg[:, 1, None], mh)
        sqrt_mtot = np.sqrt(np.maximum(m1 + m2, 1e-12))
        log_good = self.dynamics_lookup.interpolate_numpy(sqrt_mtot, component="good")
        log_bad = self.dynamics_lookup.interpolate_numpy(sqrt_mtot, component="bad")
        log_conditional = np.logaddexp(
            np.log1p(-float(f_outlier)) + log_good,
            np.log(float(f_outlier)) + log_bad,
        )
        return marginalize_dynamics(log_conditional, self.z_probabilities)

    def run_mcmc(
        self,
        *,
        num_warmup: int = 1000,
        num_samples: int = 1000,
        num_chains: int = 4,
        seed: int = 43,
        progress_bar: bool = True,
        **nuts_kwargs,
    ):
        if not self._data_set:
            raise ValueError("Call set_data() before run_mcmc().")
        jax = _configure_jax_gpu_fallback()
        import jax.numpy as jnp
        from numpyro.infer import MCMC, NUTS
        from numpyro.infer.initialization import init_to_value

        baseline_log_likelihood = self.baseline_dynamics_log_likelihood(f_outlier=0.2)
        if np.any(~np.isfinite(baseline_log_likelihood)):
            invalid = np.flatnonzero(~np.isfinite(baseline_log_likelihood))
            raise ValueError(
                "The zero-correction PARSEC baseline has non-finite dynamics likelihood "
                f"for {invalid.size} systems; first row indices: "
                f"{self.row_indices[invalid[:10]].tolist()}."
            )
        print(
            "Zero-correction baseline dynamics log likelihood: "
            f"{np.sum(baseline_log_likelihood):.6g}."
        )

        # NumPyro's default init_to_uniform operates in an O(1) unconstrained
        # range.  That is far wider than these 0.10--0.15 dex MLR priors and can
        # initialize every metallicity node outside the finite lookup domain.
        # Start instead from the scientifically natural PARSEC baseline.
        init_strategy = nuts_kwargs.pop(
            "init_strategy",
            init_to_value(
                values={
                    "f0_knots": jnp.zeros(3),
                    "fz_knots": jnp.zeros(3),
                    "f_outlier": jnp.asarray(0.2),
                }
            ),
        )

        model = self._build_numpyro_model()
        mcmc = MCMC(
            NUTS(model, init_strategy=init_strategy, **nuts_kwargs),
            num_warmup=int(num_warmup),
            num_samples=int(num_samples),
            num_chains=int(num_chains),
            progress_bar=bool(progress_bar),
        )
        mcmc.run(
            jax.random.PRNGKey(int(seed)),
            jnp.asarray(self.absg),
            jnp.asarray(self.z_probabilities),
            jnp.asarray(self.log_good_lookup),
            jnp.asarray(self.log_bad_lookup),
            extra_fields=("diverging", "energy", "potential_energy", "num_steps", "accept_prob"),
        )
        self.sampler = mcmc
        self.posterior_samples = {
            name: np.asarray(values) for name, values in mcmc.get_samples().items()
        }
        return mcmc

    def correction_grid(
        self,
        *,
        absg_grid: Optional[Sequence[float]] = None,
        z_grid: Optional[Sequence[float]] = None,
        max_draws: int = 512,
    ) -> Dict[str, np.ndarray]:
        if self.posterior_samples is None:
            raise ValueError("Run run_mcmc() before correction_grid().")
        if absg_grid is None:
            absg_grid = np.linspace(self.mg_knots[0], self.mg_knots[-1], 201)
        if z_grid is None:
            z_grid = np.array([-1.0, -0.5, 0.0, 0.3, 0.6])
        absg_grid = np.asarray(absg_grid, dtype=float)
        z_grid = np.asarray(z_grid, dtype=float)
        n_available = self.posterior_samples["f0_knots"].shape[0]
        count = min(max_draws, n_available)
        # Even spacing covers every chain after NumPyro flattens grouped draws;
        # taking [:count] silently selected only chain 0 in long multi-chain runs.
        draw_indices = np.linspace(0, n_available - 1, count, dtype=np.int64)
        f0_samples = self.posterior_samples["f0_knots"][draw_indices]
        fz_samples = self.posterior_samples["fz_knots"][draw_indices]
        correction = np.empty((count, z_grid.size, absg_grid.size), dtype=np.float64)
        mass = np.empty_like(correction)
        baseline = self.mass_surface.mass_from_absg_mh(
            absg_grid[None, :], z_grid[:, None]
        )
        for draw in range(count):
            delta = self.correction_log10(
                absg_grid[None, :], z_grid[:, None], f0_samples[draw], fz_samples[draw]
            )
            correction[draw] = 100.0 * (np.power(10.0, delta) - 1.0)
            mass[draw] = baseline * np.power(10.0, delta)
        percentiles = np.array([16.0, 50.0, 84.0])
        correction_pct = np.percentile(correction, percentiles, axis=0)
        mass_pct = np.percentile(mass, percentiles, axis=0)
        monotone_cells = np.diff(mass, axis=2) > 0.0
        return {
            "absg_grid": absg_grid,
            "z_grid": z_grid,
            "percentiles": percentiles,
            "parsec_mass": baseline,
            "correction_percent": correction_pct,
            "mass": mass_pct,
            "monotone_cell_violation_fraction": np.asarray(np.mean(monotone_cells)),
            "monotone_draw_violation_fraction": np.asarray(
                np.mean(np.any(monotone_cells, axis=(1, 2)))
            ),
        }


def grouped_mcmc_payload(sampler) -> Dict[str, np.ndarray]:
    """Return chain-aware posterior and NUTS diagnostic arrays for NPZ output."""
    posterior = sampler.get_samples(group_by_chain=True)
    diagnostics = sampler.get_extra_fields(group_by_chain=True)
    payload = {f"posterior__{key}": np.asarray(value) for key, value in posterior.items()}
    payload.update({f"sample_stats__{key}": np.asarray(value) for key, value in diagnostics.items()})
    if "num_steps" in diagnostics:
        steps = np.asarray(diagnostics["num_steps"])
        payload["sample_stats__tree_depth"] = np.ceil(np.log2(steps + 1)).astype(np.int16)
    return payload


def simulate_mock_dataset(
    color_surface: IsochroneColorSurfaceModel,
    mass_surface: IsochroneMassSurfaceModel,
    *,
    n_systems: int = 300,
    seed: int = 123,
) -> Tuple[Dict[str, np.ndarray], Dict[str, object]]:
    """Generate a compact T6d+CMD end-to-end recovery data set."""
    rng = np.random.default_rng(seed)
    n = int(n_systems)
    z_true = np.clip(rng.normal(-0.3, 0.35, n), -1.0, 0.6)
    absg1 = rng.uniform(4.0, 10.5, n)
    absg2 = np.clip(absg1 + rng.uniform(0.3, 2.5, n), 4.0, 13.3)
    absg = np.column_stack([absg1, absg2])
    z_offset, cmd_scatter = 0.10, 0.035
    z_sigma = rng.uniform(0.05, 0.18, size=(n, 2))
    z_mean = z_offset + z_true[:, None] + jcaps_magnitude_bias(absg)
    z_observed = z_mean + rng.standard_t(JCAPS_STUDENT_DF, size=(n, 2)) * np.sqrt(
        z_sigma**2 + JCAPS_EXTRA_SCALE**2
    )
    color_sigma = rng.uniform(0.002, 0.015, size=(n, 2))
    color_mean = color_surface.color_from_absg_mh(absg, z_true[:, None])
    color = color_mean + rng.standard_t(4, size=(n, 2)) * np.sqrt(
        color_sigma**2 + cmd_scatter**2
    )
    f0_knots = np.array([0.015, 0.0, -0.015])
    fz_knots = np.array([-0.06, 0.02, 0.08])
    mlr = ThreeKnotMetallicityMLR(mass_surface)
    m1 = mlr.mass_from_absg_mh(absg1, z_true, f0_knots, fz_knots)
    m2 = mlr.mass_from_absg_mh(absg2, z_true, f0_knots, fz_knots)
    tilde_grid = np.linspace(0.01, 80.0, 20000)
    density = 5.434e-3 * tilde_grid * np.exp(
        -(2.544e-3 * tilde_grid**2 + np.exp((tilde_grid - 35.67) / 3.100))
    )
    density /= np.sum(density)
    tilde_u = rng.choice(tilde_grid, size=n, p=density)
    u_true = np.sqrt(m1 + m2) * tilde_u
    u_sigma = np.maximum(0.5, 0.08 * u_true)
    u = np.sqrt((u_true + rng.normal(0.0, u_sigma)) ** 2 + rng.normal(0.0, u_sigma) ** 2)
    data = {
        "row_indices": np.arange(n, dtype=np.int64),
        "absg": absg,
        "feh_observed": z_observed,
        "feh_sigma": z_sigma,
        "color_observed": color,
        "color_sigma": color_sigma,
        "u": u,
        "u_sigma": u_sigma,
        "z_true": z_true,
    }
    truth = {
        "z_offset": z_offset,
        "cmd_scatter": cmd_scatter,
        "bias_mg_knots": JCAPS_BIAS_MG_KNOTS.tolist(),
        "bias_values": JCAPS_BIAS_VALUES.tolist(),
        "extra_scale": JCAPS_EXTRA_SCALE,
        "student_df": JCAPS_STUDENT_DF,
        "f0_knots": f0_knots.tolist(),
        "fz_knots": fz_knots.tolist(),
    }
    return data, truth
