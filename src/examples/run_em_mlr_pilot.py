#!/usr/bin/env python3
"""Alternating-MAP exploration for the T8 mass--velocity model.

The run uses one fixed selection of the formal T8 metallicity posterior and
alternates two conditional MAP updates:

* the hard-monotone MLR parameters, and
* the good-shape parameters (inside the finite T8.2 shape box) plus the raw-u
  outlier fraction.

All updates use the same marginal observation likelihood as T8.2: Rice
measurement convolution, a normal/outlier mixture, and a sum over the saved
metallicity posterior.  This is an alternating MAP calculation, not a full
posterior or a strict EM implementation.
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import os
from pathlib import Path
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from scipy.special import i0e, ndtr


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from binary_masses import hierarchical_metallicity as hm  # noqa: E402
import run_hierarchical_metallicity_test as workflow  # noqa: E402


T8_POSTERIOR_NAME = "latent_metallicity_weights_t8.npz"
T8_SHAPESTACK_NAME = "dynamics_likelihood_shapestack_t8.npz"
T8_MLR_NAME = "mlr_mcmc_t8.npz"

SHAPE_AXES = {
    "log_b": np.array([np.log(0.7e-3), np.log(1.52e-3), np.log(3.3e-3)]),
    "log_uc": np.array([np.log(24.0), np.log(32.5), np.log(44.0)]),
    "log_c": np.array([np.log(2.5), np.log(5.81), np.log(13.5)]),
}
SHAPE_STARTS = {
    "default": np.log([0.002544, 35.67, 3.1]),
    "S2": np.log([0.00103, 40.98, 11.53]),
}
PHYSICAL_SHAPE_LOG_CENTER = SHAPE_STARTS["default"].copy()
DEFAULT_SHAPE_PRIOR_SIGMA = np.log([2.0, 1.3, 2.0])
SHAPE_NAMES = ("log_b", "log_uc", "log_c")
OUTLIER_MU = 40.0
OUTLIER_SIGMA = 13.0
OUTLIER_MAX = 80.0
OUTLIER_FLOOR = 1e-30
# c0(1) + a(3) + b(7) + r(21) + log-lambda_x/z(2); f is separate.
MLR_DIM = 34
FULL_DIM = 38
F_INDEX = 34
SHAPE_OFFSET = 35
REPRESENTATIVE_MG = np.array([5.0, 7.0, 9.0, 11.0, 13.0])


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True,
                        help="Directory holding the frozen T8 metallicity posterior.")
    parser.add_argument("--initial-mlr", type=Path, default=None,
                        help="MLR posterior used only for the common starting state; defaults to --baseline.")
    parser.add_argument("--shape-stack", type=Path, default=None,
                        help="Formal T8.2 27-node shape stack.")
    parser.add_argument("--dynamics-lookup", type=Path, default=None,
                        help="Validated full-sample T8 dynamics lookup used to define the mass grid and data rows in direct mode.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-systems", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260919)
    parser.add_argument("--max-iterations", type=int, default=10)
    parser.add_argument("--block-maxiter", type=int, default=80)
    parser.add_argument("--tol", type=float, default=2e-4,
                        help="Deprecated absolute tolerance retained for CLI compatibility.")
    parser.add_argument(
        "--shape-evaluation", choices=("direct", "linear_stack"), default="direct",
        help="Good-shape likelihood used by alternating MAP. direct evaluates the continuous Rice model; linear_stack is a faster diagnostic surrogate.",
    )
    parser.add_argument("--direct-nodes", type=int, default=12,
                        help="Gauss-Legendre order per direct Rice integration interval.")
    parser.add_argument("--direct-chunk", type=int, default=16,
                        help="System chunk size for direct JAX Rice integration.")
    parser.add_argument("--objective-rtol", type=float, default=1e-7,
                        help="Relative full-cycle log-posterior improvement threshold.")
    parser.add_argument("--shape-tol", type=float, default=1e-3,
                        help="Maximum absolute change in log(B), log(uc), log(C) for convergence.")
    parser.add_argument("--mass-rtol", type=float, default=1e-3,
                        help="Maximum relative MLR mass change at representative M_G points for convergence.")
    parser.add_argument("--stable-cycles", type=int, default=2,
                        help="Number of consecutive stable full cycles required for convergence.")
    parser.add_argument("--shape-prior-sigma-log-b", type=float,
                        default=float(DEFAULT_SHAPE_PRIOR_SIGMA[0]),
                        help="Physics-centered Gaussian prior sigma for log(B); default ln(2).")
    parser.add_argument("--shape-prior-sigma-log-uc", type=float,
                        default=float(DEFAULT_SHAPE_PRIOR_SIGMA[1]),
                        help="Physics-centered Gaussian prior sigma for log(uc); default ln(1.3).")
    parser.add_argument("--shape-prior-sigma-log-c", type=float,
                        default=float(DEFAULT_SHAPE_PRIOR_SIGMA[2]),
                        help="Physics-centered Gaussian prior sigma for log(C); default ln(2).")
    parser.add_argument("--data-subset", type=Path, default=None,
                        help="Optional previously saved selected row positions.")
    return parser.parse_args()


def outlier_log_likelihood(u, sigma, nodes=512):
    """Raw-u truncated-normal density convolved with the Rice likelihood."""
    u = np.asarray(u, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    x, weights = np.polynomial.legendre.leggauss(int(nodes))
    lo = np.minimum(np.maximum(u - 14.0 * sigma, 0.0), OUTLIER_MAX)
    hi = np.minimum(u + 14.0 * sigma, OUTLIER_MAX)
    half = (hi - lo) / 2.0
    v = (hi + lo)[:, None] / 2.0 + half[:, None] * x
    s = sigma[:, None]
    obs = u[:, None]
    log_rice = np.log(obs / s**2) - (obs - v) ** 2 / (2.0 * s**2)
    log_rice += np.log(i0e(obs * v / s**2))
    norm = ndtr((OUTLIER_MAX - OUTLIER_MU) / OUTLIER_SIGMA) - ndtr(-OUTLIER_MU / OUTLIER_SIGMA)
    log_bad = (
        -0.5 * ((v - OUTLIER_MU) / OUTLIER_SIGMA) ** 2
        - np.log(OUTLIER_SIGMA * np.sqrt(2.0 * np.pi) * norm)
    )
    integral = half * np.sum(weights * np.exp(log_rice + log_bad), axis=1)
    return np.log(np.maximum(integral, OUTLIER_FLOOR))


def validate_outlier_integral(u, sigma, seed):
    """Compare the local quadrature to the independent reference integral."""
    rng = np.random.default_rng(seed)
    take = np.sort(rng.choice(len(u), size=min(32, len(u)), replace=False))
    local = outlier_log_likelihood(u[take], sigma[take])
    reference = np.array([
        hm.rice_component_reference_integral(float(ui), float(si), 1.0, component="bad")
        for ui, si in zip(u[take], sigma[take])
    ])
    error = float(np.max(np.abs(local - np.log(np.maximum(reference, OUTLIER_FLOOR)))))
    if error >= 1e-3:
        raise RuntimeError(f"Raw-u outlier quadrature check failed: max |dlogL|={error:.3g}")
    return {"n_checked": int(take.size), "max_abs_log_error": error, "pass": True}


def subset_posterior(posterior, positions):
    """Build a validated posterior object after selecting fixed row positions."""
    metadata = dict(posterior.metadata)
    metadata.update({
        "pilot_subset": True,
        "pilot_subset_size": int(len(positions)),
        "pilot_subset_positions": np.asarray(positions, dtype=np.int64).tolist(),
    })
    return hm.MetallicityPosteriorGrid(
        row_indices=np.asarray(posterior.row_indices)[positions],
        z_grid=np.asarray(posterior.z_grid),
        probabilities=np.asarray(posterior.probabilities)[positions],
        z_quantiles=np.asarray(posterior.z_quantiles)[positions],
        bad_probabilities=np.asarray(posterior.bad_probabilities)[positions],
        metadata=metadata,
    )


def subset_shape_stack(stack, positions, rows, u, u_sigma, log_bad):
    """Keep only selected systems and replace the bad component by raw-u L_out."""
    metadata = dict(stack.metadata)
    metadata.update({
        "data_digest": hm.array_digest(rows, u, u_sigma),
        "row_order": "selected formal posterior row_indices order",
        "outlier_coordinate": "raw_u",
        "outlier_mass_dependent": False,
        "outlier_source": "Rice convolution of TN(40,13,[0,80])",
    })
    bad = np.broadcast_to(log_bad[None, :, None],
                          (27, len(rows), stack.sqrt_mtot_grid.size)).astype(np.float32)
    return hm.DynamicsLikelihoodShapeStack(
        row_indices=rows,
        sqrt_mtot_grid=stack.sqrt_mtot_grid,
        log_good_stack=np.asarray(stack.log_good_stack[:, positions, :]),
        log_bad_stack=bad,
        axes=stack.axes,
        node_constants=stack.node_constants,
        metadata=metadata,
    )


def params_to_vector(params, f, shape_logs):
    vector = np.empty(FULL_DIM, dtype=np.float64)
    vector[0] = float(np.asarray(params["c0"]))
    vector[1:4] = np.asarray(params["a"], dtype=np.float64).reshape(3)
    vector[4:11] = np.asarray(params["b"], dtype=np.float64).reshape(7)
    vector[11:32] = np.asarray(params["r"], dtype=np.float64).reshape(21)
    vector[32] = float(np.asarray(params["log_lambda_x"]))
    vector[33] = float(np.asarray(params["log_lambda_z"]))
    vector[F_INDEX] = float(f)
    vector[SHAPE_OFFSET:] = np.asarray(shape_logs, dtype=np.float64)
    return vector


def vector_to_params(vector):
    vector = np.asarray(vector, dtype=np.float64)
    return {
        "c0": vector[0],
        "a": vector[1:4],
        "b": vector[4:11],
        "r": vector[11:32].reshape(7, 3),
        "log_lambda_x": vector[32],
        "log_lambda_z": vector[33],
        "f_outlier": vector[F_INDEX],
    }


def build_objective(mlr, args):
    """Build JAX value/gradient functions for the constrained joint MAP.

    ``numpyro.infer.util.log_density`` expects unconstrained values for
    transformed sample sites.  The optimizer here uses bounded physical
    values for ``f`` and the three shape logs, so the target is written out
    explicitly to avoid accidentally optimizing a coordinate-transformed
    density.
    """
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from jax.scipy.special import gammaln, logsumexp

    z_prob = jnp.asarray(mlr.z_probabilities, dtype=jnp.float64)
    z_grid = jnp.asarray(mlr.z_grid, dtype=jnp.float64)
    sqrt_grid = jnp.asarray(mlr.sqrt_mtot_grid, dtype=jnp.float64)
    bx1 = jnp.asarray(hm._bspline_basis_numpy(mlr.absg[:, 0], mlr.knots_x, mlr.degree_x), dtype=jnp.float64)
    bx2 = jnp.asarray(hm._bspline_basis_numpy(mlr.absg[:, 1], mlr.knots_x, mlr.degree_x), dtype=jnp.float64)
    bz = jnp.asarray(hm._bspline_basis_numpy(mlr.z_grid, mlr.knots_z, mlr.degree_z), dtype=jnp.float64)
    stack_mode = args.shape_evaluation == "linear_stack"
    if stack_mode:
        log_good_stack = jnp.asarray(mlr.dynamics_shape_stack.log_good_stack, dtype=jnp.float64)
        log_bad_stack = jnp.asarray(mlr.dynamics_shape_stack.log_bad_stack, dtype=jnp.float64)
        axes = {name: jnp.asarray(mlr.dynamics_shape_stack.axes[name], dtype=jnp.float64)
                for name in SHAPE_NAMES}
    parsec_projection = jnp.asarray(mlr.parsec_projection, dtype=jnp.float64)
    hx = jnp.asarray(np.diff(mlr.greville_x), dtype=jnp.float64)
    hz = jnp.asarray(np.diff(mlr.greville_z), dtype=jnp.float64)
    href_x = float((mlr.greville_x[-1] - mlr.greville_x[0]) / (mlr.K_x - 1))
    href_z = float((mlr.greville_z[-1] - mlr.greville_z[0]) / (mlr.K_Z - 1))
    lambda_ref_x, lambda_ref_z = mlr.lambda_references()
    absg_grid = jnp.asarray(mlr.absg, dtype=jnp.float64)
    normal_const = np.log(np.sqrt(2.0 * np.pi))
    beta_const = float(gammaln(3.0) + gammaln(12.0) - gammaln(15.0))
    shape_prior_center = jnp.asarray(PHYSICAL_SHAPE_LOG_CENTER, dtype=jnp.float64)
    shape_prior_sigma = jnp.asarray(
        [
            args.shape_prior_sigma_log_b,
            args.shape_prior_sigma_log_uc,
            args.shape_prior_sigma_log_c,
        ],
        dtype=jnp.float64,
    )
    shape_prior_normalization = jnp.sum(
        jnp.log(shape_prior_sigma) + np.log(np.sqrt(2.0 * np.pi))
    )
    u_observed = jnp.asarray(mlr.u, dtype=jnp.float64)
    u_sigma = jnp.asarray(mlr.u_sigma, dtype=jnp.float64)
    if mlr.raw_u_outlier_log_likelihood is not None:
        raw_bad_np = np.asarray(mlr.raw_u_outlier_log_likelihood, dtype=np.float64)
    elif mlr.dynamics_shape_stack is not None:
        raw_bad_np = np.asarray(mlr.dynamics_shape_stack.log_bad_stack[0, :, 0], dtype=np.float64)
    else:
        raise ValueError("A raw-u outlier likelihood is required for EM direct mode.")
    raw_bad = jnp.asarray(raw_bad_np, dtype=jnp.float64)

    gl_x_np, gl_w_np = np.polynomial.legendre.leggauss(int(args.direct_nodes))
    norm_x_np, norm_w_np = np.polynomial.legendre.leggauss(256)
    norm_x_np = 0.5 * (norm_x_np + 1.0) * 80.0
    norm_w_np = 40.0 * norm_w_np
    gl_x = jnp.asarray(gl_x_np, dtype=jnp.float64)
    gl_w = jnp.asarray(gl_w_np, dtype=jnp.float64)
    norm_x = jnp.asarray(norm_x_np, dtype=jnp.float64)
    norm_w = jnp.asarray(norm_w_np, dtype=jnp.float64)
    shape_offsets = jnp.asarray(np.array([-8, -4, -2, 0, 2, 4, 8], dtype=np.float64))
    rice_offsets = jnp.asarray(np.array([-30, -14, -10, -6, -3, 0, 3, 6, 10, 14, 30], dtype=np.float64))
    direct_chunk = int(args.direct_chunk)
    n_systems = int(mlr.u.size)
    n_pad = int(np.ceil(n_systems / direct_chunk) * direct_chunk)

    def direct_shape_tables(shape, sqrt_mtot):
        log_b, log_uc, log_c = shape
        log_norm = logsumexp(
            jnp.log(float(hm.RICE_GOOD_A))
            + jnp.log(norm_w)
            + jnp.log(norm_x)
            - jnp.exp(log_b) * norm_x**2
            - jnp.exp((norm_x - jnp.exp(log_uc)) / jnp.exp(log_c))
        )
        sqrt_pad = jnp.pad(sqrt_mtot, ((0, n_pad - n_systems), (0, 0)), constant_values=1.0)
        u_pad = jnp.pad(u_observed, (0, n_pad - n_systems), constant_values=1.0)
        sigma_pad = jnp.pad(u_sigma, (0, n_pad - n_systems), constant_values=1.0)
        s_chunks = sqrt_pad.reshape((-1, direct_chunk, sqrt_mtot.shape[1]))
        u_chunks = u_pad.reshape((-1, direct_chunk))
        sigma_chunks = sigma_pad.reshape((-1, direct_chunk))

        def integrate_chunk(chunk_args):
            s_chunk, u_chunk, sigma_chunk = chunk_args
            upper = s_chunk * 80.0
            shape_boundaries = s_chunk[..., None] * (
                jnp.exp(log_uc) + jnp.exp(log_c) * shape_offsets
            )
            rice_boundaries = (
                u_chunk[:, None, None] + sigma_chunk[:, None, None] * rice_offsets
            )
            rice_boundaries = jnp.broadcast_to(
                rice_boundaries, (s_chunk.shape[0], s_chunk.shape[1], rice_offsets.size)
            )
            interior = jnp.concatenate((shape_boundaries, rice_boundaries), axis=-1)
            n_interior = int(interior.shape[-1])
            boundary_eps = 1e-8
            safe_upper = upper - boundary_eps * (n_interior + 1)
            interior = jnp.clip(interior, boundary_eps, safe_upper[..., None])
            interior += boundary_eps * jnp.arange(n_interior, dtype=jnp.float64)
            boundaries = jnp.concatenate(
                (jnp.zeros_like(upper[..., None]), jnp.sort(interior, axis=-1), upper[..., None]),
                axis=-1,
            )
            lower = boundaries[..., :-1]
            upper_interval = boundaries[..., 1:]
            midpoint = 0.5 * (lower + upper_interval)
            half_width = 0.5 * (upper_interval - lower)
            velocity = midpoint[..., None] + half_width[..., None] * gl_x
            observed = u_chunk[:, None, None, None]
            sigma = sigma_chunk[:, None, None, None]
            argument = observed * velocity / sigma**2
            log_rice = (
                jnp.log(observed) - 2.0 * jnp.log(sigma)
                - (observed - velocity) ** 2 / (2.0 * sigma**2)
                + jnp.log(jax.scipy.special.i0e(argument) + 1e-30)
            )
            scale = s_chunk[..., None, None]
            tilde_u = velocity / scale
            tilde_eval = jnp.clip(tilde_u, 1e-12, 80.0)
            log_good = (
                jnp.log(float(hm.RICE_GOOD_A)) - log_norm + jnp.log(tilde_eval)
                - jnp.exp(log_b) * tilde_eval**2
                - jnp.exp((tilde_eval - jnp.exp(log_uc)) / jnp.exp(log_c))
                - jnp.log(scale)
            )
            log_quad = jnp.log(jnp.maximum(half_width, 1e-300))[..., None] + jnp.log(gl_w)
            valid = (tilde_u > 0.0) & (tilde_u <= 80.0)
            log_interval = logsumexp(
                log_rice + jnp.where(valid, log_good, -jnp.inf) + log_quad, axis=-1
            )
            return logsumexp(log_interval, axis=-1)

        good = jax.lax.map(integrate_chunk, (s_chunks, u_chunks, sigma_chunks))
        good = good.reshape((n_pad, sqrt_mtot.shape[1]))[:n_systems]
        bad = raw_bad[:, None] + jnp.zeros_like(good)
        return good, bad

    def linear_stack_shape_tables(shape, sqrt_mtot):
        cells, fracs = [], []
        for axis_value, name in zip(shape, SHAPE_NAMES):
            axis = axes[name]
            cell = (axis_value > axis[1]).astype(jnp.int32)
            lo = jnp.where(cell == 1, axis[1], axis[0])
            hi = jnp.where(cell == 1, axis[2], axis[1])
            t = jnp.clip((axis_value - lo) / (hi - lo), 0.0, 1.0)
            cells.append(cell)
            fracs.append(t)
        logs_good, log_weights = [], []
        for cb in (0, 1):
            for cu in (0, 1):
                for cc in (0, 1):
                    idx = 9 * (cells[0] + cb) + 3 * (cells[1] + cu) + (cells[2] + cc)
                    log_weights.append(jnp.log(jnp.maximum(
                        (fracs[0] if cb else 1.0 - fracs[0])
                        * (fracs[1] if cu else 1.0 - fracs[1])
                        * (fracs[2] if cc else 1.0 - fracs[2]), 1e-30)))
                    logs_good.append(hm.lookup_interpolate(sqrt_mtot, sqrt_grid, log_good_stack[idx]))
        weights = jnp.stack(log_weights)[:, None, None]
        good = logsumexp(weights + jnp.stack(logs_good), axis=0)
        bad = raw_bad[:, None] + jnp.zeros_like(good)
        return good, bad

    def shape_tables(shape, sqrt_mtot):
        if args.shape_evaluation == "direct":
            return direct_shape_tables(shape, sqrt_mtot)
        return linear_stack_shape_tables(shape, sqrt_mtot)

    def log_posterior_and_likelihood(v):
        c0 = v[0]
        a = v[1:4]
        b = v[4:11]
        r = v[11:32].reshape((7, 3))
        log_lambda_x, log_lambda_z = v[32], v[33]
        f = v[F_INDEX]
        shape = v[SHAPE_OFFSET:SHAPE_OFFSET + 3]
        theta = mlr._theta_from_raw_jax(c0, a, b, r, log_lambda_x, log_lambda_z)
        D = theta - parsec_projection
        slope_x = jnp.diff(D, axis=0) / hx[:, None]
        slope_z = jnp.diff(D, axis=1) / hz[None, :]
        d2x = 2.0 * jnp.diff(slope_x, axis=0) / (hx[:-1, None] + hx[1:, None]) * href_x**2
        d2z = 2.0 * jnp.diff(slope_z, axis=1) / (hz[None, :-1] + hz[None, 1:]) * href_z**2
        dxz = (D[1:, 1:] - D[1:, :-1] - D[:-1, 1:] + D[:-1, :-1]) / (hx[:, None] * hz[None, :]) * href_x * href_z
        log_prior = (
            -0.5 * ((c0 - float(mlr.parsec_projection[0, 0])) / 0.10) ** 2
            - np.log(0.10) - normal_const
            - 0.5 * jnp.sum(a**2) - 0.5 * jnp.sum(b**2) - 0.5 * jnp.sum(r**2)
            - (3 + 7 + 21) * normal_const
            - 0.5 * ((log_lambda_x - np.log(lambda_ref_x)) / 0.7) ** 2
            - 0.5 * ((log_lambda_z - np.log(lambda_ref_z)) / 0.7) ** 2
            - 2.0 * np.log(np.sqrt(2.0 * np.pi)) - np.log(0.7)
            + (3.0 - 1.0) * jnp.log(f) + (12.0 - 1.0) * jnp.log1p(-f) - beta_const
            - 0.5 * jnp.sum((D / mlr.tau_D) ** 2)
            - 0.5 * (jnp.sum((d2x / mlr.tau_x) ** 2)
                     + jnp.sum((d2z / mlr.tau_Z) ** 2)
                     + jnp.sum((dxz / mlr.tau_xZ) ** 2))
        )
        gp1 = jnp.einsum("ni,qh,ih->nq", bx1, bz, theta)
        gp2 = jnp.einsum("ni,qh,ih->nq", bx2, bz, theta)
        sqrt_mtot = jnp.sqrt(jnp.maximum(jnp.power(10.0, gp1) + jnp.power(10.0, gp2), 1e-12))
        log_good, log_bad = shape_tables(shape, sqrt_mtot)
        log_mix = jnp.logaddexp(jnp.log1p(-f) + log_good, jnp.log(f) + log_bad)
        log_likelihood = jnp.sum(logsumexp(jnp.log(jnp.maximum(z_prob, 1e-30)) + log_mix, axis=1))
        anchor_bx = jnp.asarray(hm._bspline_basis_numpy(4.67, mlr.knots_x, mlr.degree_x), dtype=jnp.float64)
        anchor_bz = jnp.asarray(hm._bspline_basis_numpy(0.0, mlr.knots_z, mlr.degree_z), dtype=jnp.float64)
        solar_g = jnp.einsum("i,h,ih->", anchor_bx, anchor_bz, theta)
        solar_anchor = -0.5 * ((solar_g - mlr.solar_anchor_mean) / mlr.solar_anchor_sigma) ** 2 - np.log(mlr.solar_anchor_sigma) - normal_const
        # Physics-informed calibration prior. The existing shape box remains a
        # hard physical/numerical support constraint enforced by L-BFGS-B.
        # The Gaussian center is the orbit-model-derived default shape.
        shape_log_prior = (
            -0.5 * jnp.sum(((shape - shape_prior_center) / shape_prior_sigma) ** 2)
            - shape_prior_normalization
        )
        joint = log_likelihood + log_prior + solar_anchor + shape_log_prior
        return joint, log_likelihood

    def log_posterior(v):
        return log_posterior_and_likelihood(v)[0]

    def log_likelihood_only(v):
        return log_posterior_and_likelihood(v)[1]

    value_grad = jax.jit(jax.value_and_grad(lambda v: -log_posterior(v)))

    def objective(v):
        value, grad = value_grad(jnp.asarray(v, dtype=jnp.float64))
        value, grad = float(value), np.asarray(grad, dtype=np.float64)
        if not np.isfinite(value) or not np.all(np.isfinite(grad)):
            return 1e100, np.zeros_like(np.asarray(v, dtype=np.float64))
        return value, grad

    logpost_jit = jax.jit(log_posterior)
    loglike_jit = jax.jit(log_likelihood_only)

    def logpost(v):
        return float(logpost_jit(jnp.asarray(v, dtype=jnp.float64)))

    def loglike(v):
        return float(loglike_jit(jnp.asarray(v, dtype=jnp.float64)))

    return objective, logpost, loglike


def summarize_objective(mlr, vector, logpost, loglike):
    """Return the exact optimizer target and likelihood/prior decomposition."""
    params = vector_to_params(vector)
    params["f_outlier"] = float(vector[F_INDEX])
    mh = mlr.z_grid[None, :]
    m1 = mlr.mass_from_absg_mh(mlr.absg[:, 0, None], mh, params)
    m2 = mlr.mass_from_absg_mh(mlr.absg[:, 1, None], mh, params)
    sqrt_mtot = np.sqrt(np.maximum(m1 + m2, 1e-12))
    log_likelihood = float(loglike(vector))
    joint = float(logpost(vector))
    return {
        "log_posterior": joint,
        "log_likelihood": log_likelihood,
        "log_prior": joint - log_likelihood,
        "shape_B": float(np.exp(vector[SHAPE_OFFSET])),
        "shape_uc": float(np.exp(vector[SHAPE_OFFSET + 1])),
        "shape_C": float(np.exp(vector[SHAPE_OFFSET + 2])),
        "f_outlier": float(vector[F_INDEX]),
        "lookup_min": float(np.min(sqrt_mtot)),
        "lookup_max": float(np.max(sqrt_mtot)),
    }


def interpolate_lookup_numpy(values, grid, table):
    values = np.asarray(values, dtype=np.float64)
    upper = np.clip(np.searchsorted(grid, values), 1, len(grid) - 1)
    lower = upper - 1
    xl, xu = grid[lower], grid[upper]
    yl = np.take_along_axis(np.asarray(table), lower, axis=1)
    yu = np.take_along_axis(np.asarray(table), upper, axis=1)
    out = yl + (values - xl) / (xu - xl) * (yu - yl)
    return np.where((values >= grid[0]) & (values <= grid[-1]), out, -np.inf)


def interpolate_shape_numpy(values, grid, stack, axes, sqrt_mtot):
    """Linear trilinear probability mixture used by the stack diagnostic mode."""
    values = np.asarray(values, dtype=np.float64)
    cells, fracs = [], []
    for value, name in zip(values, SHAPE_NAMES):
        axis = np.asarray(axes[name], dtype=np.float64)
        cell = int(value > axis[1])
        lo, hi = (axis[1], axis[2]) if cell else (axis[0], axis[1])
        t = np.clip((value - lo) / (hi - lo), 0.0, 1.0)
        cells.append(cell)
        fracs.append(t)
    out = np.full_like(sqrt_mtot, -np.inf, dtype=np.float64)
    pieces = []
    weights = []
    for cb in (0, 1):
        for cu in (0, 1):
            for cc in (0, 1):
                idx = 9 * (cells[0] + cb) + 3 * (cells[1] + cu) + (cells[2] + cc)
                weight = ((fracs[0] if cb else 1.0 - fracs[0])
                          * (fracs[1] if cu else 1.0 - fracs[1])
                          * (fracs[2] if cc else 1.0 - fracs[2]))
                pieces.append(interpolate_lookup_numpy(sqrt_mtot, grid, stack[idx]))
                weights.append(weight)
    # The stack stores log densities; T8.2 mixes in probability space.
    pieces = np.stack(pieces)
    weights = np.asarray(weights)[:, None, None]
    return np.log(np.sum(weights * np.exp(pieces), axis=0))


def initial_vector(initial_mlr_path, shape_name):
    with np.load(initial_mlr_path, allow_pickle=False) as saved:
        params = {
            "c0": np.median(saved["posterior__c0"]),
            "a": np.median(saved["posterior__a"].reshape(-1, 3), axis=0),
            "b": np.median(saved["posterior__b"].reshape(-1, 7), axis=0),
            "r": np.median(saved["posterior__r"].reshape(-1, 7, 3), axis=0),
            "log_lambda_x": np.median(saved["posterior__log_lambda_x"]),
            "log_lambda_z": np.median(saved["posterior__log_lambda_z"]),
        }
        f = float(np.median(saved["posterior__f_outlier"]))
    return params_to_vector(params, f, SHAPE_STARTS[shape_name]), f


def optimize_block(objective, current, indices, bounds, maxiter):
    current = np.asarray(current, dtype=np.float64)
    start = current[indices].copy()
    # Optimize dimensionless displacements.  The raw T8 recursion mixes c0,
    # softplus logits, and log-scales, so scaling improves the numerical line
    # search while leaving the target in the original constrained coordinates.
    scales = np.full(len(indices), 0.10, dtype=np.float64)
    for j, index in enumerate(indices):
        if index == 0:
            scales[j] = 0.02
    x0 = np.zeros(len(indices), dtype=np.float64)

    def fun(block):
        candidate = current.copy()
        candidate[indices] = start + scales * block
        value, grad = objective(candidate)
        return value, grad[indices] * scales

    displacement_bounds = None
    if bounds is not None:
        displacement_bounds = [
            ((float(lo) - start[j]) / scales[j], (float(hi) - start[j]) / scales[j])
            for j, (lo, hi) in enumerate(bounds)
        ]
    result = minimize(fun, x0, jac=True, method="L-BFGS-B", bounds=displacement_bounds,
                      options={"maxiter": int(maxiter), "ftol": 1e-10,
                               "gtol": 1e-6, "maxls": 30})
    candidate = current.copy()
    candidate[indices] = start + scales * result.x
    return candidate, result


def representative_masses(mlr, vector):
    params = vector_to_params(vector)
    params["f_outlier"] = float(vector[F_INDEX])
    values = 10.0 ** mlr.g_from_raw(
        REPRESENTATIVE_MG[:, None], np.zeros((1, len(REPRESENTATIVE_MG))), params
    )
    # g_from_raw broadcasts the two inputs; use a direct scalar loop to keep
    # the output unambiguous (one mass at each requested M_G).
    return [float(mlr.mass_from_absg_mh(mg, 0.0, params)) for mg in REPRESENTATIVE_MG]


def run_trajectory(mlr, objective, logpost, loglike, start_vector, name, args):
    current = np.asarray(start_vector, dtype=np.float64).copy()
    trace = []
    partial_path = args.output / f"trace_{name}.partial.jsonl"
    partial_stream = partial_path.open("w", encoding="utf-8")
    initial = summarize_objective(mlr, current, logpost, loglike)
    initial.update({"run": name, "iteration": 0, "block": "initial",
                    "objective_change": 0.0, "accepted": True})
    initial["representative_masses"] = representative_masses(mlr, current)
    trace.append(initial)
    partial_stream.write(json.dumps(initial) + "\n")
    partial_stream.flush()
    print(f"EM-like run {name}: start, N={mlr.u.size}, log-posterior={initial['log_posterior']:.6g}", flush=True)
    mlr_indices = np.arange(MLR_DIM, dtype=np.int64)
    shape_indices = np.arange(F_INDEX, FULL_DIM, dtype=np.int64)
    shape_bounds = [(float(SHAPE_AXES[n][0]), float(SHAPE_AXES[n][-1])) for n in SHAPE_NAMES]
    shape_bounds = [(1e-5, 1.0 - 1e-5)] + shape_bounds
    previous_target = float(initial["log_posterior"])
    previous_masses = np.asarray(initial["representative_masses"], dtype=np.float64)
    previous_shape = current[SHAPE_OFFSET:SHAPE_OFFSET + 3].copy()
    statuses = []
    stable_count = 0
    for iteration in range(1, int(args.max_iterations) + 1):
        cycle_start_target = previous_target
        cycle_start_masses = previous_masses.copy()
        cycle_start_shape = previous_shape.copy()

        candidate, result_mlr = optimize_block(
            objective, current, mlr_indices, None, args.block_maxiter
        )
        candidate_target = float(logpost(candidate))
        if np.isfinite(candidate_target) and candidate_target >= previous_target - 1e-6:
            current = candidate
            previous_target = candidate_target
            accepted_mlr = True
        else:
            accepted_mlr = False

        candidate, result_shape = optimize_block(
            objective, current, shape_indices, shape_bounds, args.block_maxiter
        )
        candidate_target = float(logpost(candidate))
        if np.isfinite(candidate_target) and candidate_target >= previous_target - 1e-6:
            current = candidate
            previous_target = candidate_target
            accepted_shape = True
        else:
            accepted_shape = False

        current_masses = np.asarray(representative_masses(mlr, current), dtype=np.float64)
        current_shape = current[SHAPE_OFFSET:SHAPE_OFFSET + 3].copy()
        cycle_delta = float(previous_target - cycle_start_target)
        objective_rel_change = cycle_delta / (1.0 + abs(cycle_start_target))
        shape_max_change = float(np.max(np.abs(current_shape - cycle_start_shape)))
        mass_max_rel_change = float(np.max(np.abs(current_masses / cycle_start_masses - 1.0)))
        stable = (
            accepted_mlr and accepted_shape
            and objective_rel_change < float(args.objective_rtol)
            and shape_max_change < float(args.shape_tol)
            and mass_max_rel_change < float(args.mass_rtol)
        )
        stable_count = stable_count + 1 if stable else 0

        row = summarize_objective(mlr, current, logpost, loglike)
        row.update({
            "run": name, "iteration": iteration, "block": "full_cycle",
            "objective_change": cycle_delta,
            "objective_relative_change": float(objective_rel_change),
            "shape_log_max_change": shape_max_change,
            "mass_max_relative_change": mass_max_rel_change,
            "stable_cycle": bool(stable),
            "stable_cycle_count": int(stable_count),
            "accepted": bool(accepted_mlr and accepted_shape),
            "mlr_optimizer_success": bool(result_mlr.success),
            "shape_optimizer_success": bool(result_shape.success),
            "mlr_optimizer_message": str(result_mlr.message),
            "shape_optimizer_message": str(result_shape.message),
        })
        row["representative_masses"] = current_masses.tolist()
        trace.append(row)
        partial_stream.write(json.dumps(row) + "\n")
        partial_stream.flush()
        print(
            f"EM-like run {name}: iteration {iteration}, "
            f"dlogpost={cycle_delta:.4g}, B={row['shape_B']:.5g}, "
            f"uc={row['shape_uc']:.5g}, C={row['shape_C']:.5g}, "
            f"max dM/M={mass_max_rel_change:.3g}", flush=True,
        )
        statuses.append({
            "iteration": iteration,
            "mlr_success": bool(result_mlr.success),
            "shape_success": bool(result_shape.success),
            "accepted_mlr": accepted_mlr,
            "accepted_shape": accepted_shape,
            "full_cycle_delta": cycle_delta,
            "objective_relative_change": float(objective_rel_change),
            "shape_log_max_change": shape_max_change,
            "mass_max_relative_change": mass_max_rel_change,
            "stable_cycle": bool(stable),
            "stable_cycle_count": int(stable_count),
        })
        previous_masses = current_masses
        previous_shape = current_shape
        if stable_count >= int(args.stable_cycles):
            break
    partial_stream.close()
    return current, trace, statuses


def parsec_masses(mlr):
    return np.array([float(mlr.mass_surface.mass_from_absg_mh(mg, 0.0))
                     for mg in REPRESENTATIVE_MG])


def write_outputs(args, mlr, results, baseline_vector, validation, selection_meta, stack_meta):
    args.output.mkdir(parents=True, exist_ok=True)
    serializable = {
        "settings": {
            "algorithm": "alternating conditional MAP",
            "posterior_scope": f"formal T8 metallicity posterior, fixed {int(selection_meta['n_selected'])}-system selection",
            "outlier": "raw-u TN(40,13,[0,80]) with Rice convolution; mass independent",
            "shape_evaluation": args.shape_evaluation,
            "good_shape_likelihood": (
                "continuous direct Rice quadrature; no shape stack"
                if args.dynamics_lookup is not None
                else "27-node shape stack (linear trilinear interpolation only in linear_stack mode)"
            ),
            "direct_quadrature": {"nodes_per_interval": int(args.direct_nodes), "system_chunk": int(args.direct_chunk)},
            "convergence": {
                "objective_relative_tolerance": float(args.objective_rtol),
                "shape_log_max_change_tolerance": float(args.shape_tol),
                "mass_max_relative_change_tolerance": float(args.mass_rtol),
                "required_consecutive_stable_cycles": int(args.stable_cycles),
            },
            "shape_box": {k: v.tolist() for k, v in SHAPE_AXES.items()},
            "shape_prior": {
                "type": "Gaussian in log parameters within the hard shape box",
                "center_B_uc_C": np.exp(PHYSICAL_SHAPE_LOG_CENTER).tolist(),
                "sigma_log_b_log_uc_log_c": [
                    float(args.shape_prior_sigma_log_b),
                    float(args.shape_prior_sigma_log_uc),
                    float(args.shape_prior_sigma_log_c),
                ],
                "default_interpretation": "1-sigma factor 2 in B and C; factor 1.3 in uc",
            },
            "solar_anchor": {"M_G": 4.67, "mh": 0.0, "log10_mass": 0.0},
            "uncertainty": "No posterior intervals; a later joint sampler is required.",
        },
        "selection": selection_meta,
        "validation": validation,
        "stack": stack_meta,
        "formal_fixed_shape_B_initial": [0.002544, 35.67, 3.1],
        "formal_fixed_shape_initial_state_note": "The formal fixed-shape posterior-median MLR is used as the common initial state; it is not a fixed-shape MAP optimized under this EM-like objective.",
        "results": {},
    }
    baseline_masses = representative_masses(mlr, baseline_vector)
    parsec = parsec_masses(mlr)
    for name, payload in results.items():
        vector, trace, statuses = payload
        final = trace[-1]
        serializable["results"][name] = {
            "converged": bool(final.get("stable_cycle_count", 0) >= int(args.stable_cycles)),
            "iterations": int(final["iteration"]),
            "final": {k: v for k, v in final.items() if k != "representative_masses"},
            "final_shape": [final["shape_B"], final["shape_uc"], final["shape_C"]],
            "final_representative_masses": final["representative_masses"],
            "final_residual_percent": (100.0 * (np.asarray(final["representative_masses"]) / parsec - 1.0)).tolist(),
            "relative_to_fixed_shape_initial_state_percent": (100.0 * (np.asarray(final["representative_masses"]) / baseline_masses - 1.0)).tolist(),
            "statuses": statuses,
        }
        with (args.output / f"trace_{name}.jsonl").open("w", encoding="utf-8") as stream:
            for row in trace:
                stream.write(json.dumps(row) + "\n")
    serializable["comparison"] = {
        "M_G": REPRESENTATIVE_MG.tolist(),
        "parsec_mass": parsec.tolist(),
        "fixed_shape_initial_state_mass": baseline_masses,
    }
    (args.output / "summary.json").write_text(json.dumps(serializable, indent=2), encoding="utf-8")
    np.savez_compressed(args.output / "map_states.npz",
                        **{f"{name}_vector": payload[0] for name, payload in results.items()},
                        fixed_shape_initial_state_vector=baseline_vector,
                        M_G=REPRESENTATIVE_MG,
                        parsec_mass=parsec)
    plot_results(args.output, results, mlr, baseline_vector, parsec, int(selection_meta["n_selected"]))


def plot_results(output, results, mlr, baseline_vector, parsec, n_systems):
    grid = np.linspace(3.5, 13.5, 300)
    mh = 0.0
    colors = {"default": "tab:blue", "S2": "tab:orange"}
    fig, axes = plt.subplots(3, 2, figsize=(12, 12), constrained_layout=True)
    for name, payload in results.items():
        final_vector, trace, _ = payload
        color = colors[name]
        vals = [float(mlr.mass_from_absg_mh(x, mh, vector_to_params(final_vector))) for x in grid]
        axes[0, 0].plot(grid, vals, color=color, lw=1.5, label=f"{name} final")
        axes[0, 1].plot(grid, 100.0 * (np.asarray(vals) /
                                        np.asarray([mlr.mass_surface.mass_from_absg_mh(x, mh) for x in grid]) - 1.0),
                        color=color, lw=1.5, label=f"{name} final")
        # Trajectory at solar metallicity.
        iterations = [row["iteration"] for row in trace]
        axes[1, 0].plot(iterations, [row["shape_B"] for row in trace], color=color, marker="o", ms=3, label=name)
        axes[1, 1].plot(iterations, [row["shape_uc"] for row in trace], color=color, marker="o", ms=3, label=name)
        axes[2, 0].plot(iterations, [row["shape_C"] for row in trace], color=color, marker="o", ms=3, label=name)
        axes[2, 1].plot(iterations, [row["f_outlier"] for row in trace], color=color, marker="o", ms=3, label=name)
    parsec_grid = np.asarray([float(mlr.mass_surface.mass_from_absg_mh(x, 0.0)) for x in grid])
    formal_grid = np.asarray([float(mlr.mass_from_absg_mh(x, 0.0, vector_to_params(baseline_vector))) for x in grid])
    axes[0, 0].plot(grid, parsec_grid, "--", color="0.35", label="PARSEC")
    axes[0, 0].plot(grid, formal_grid, ":", color="k", label="Fixed-shape initial state")
    axes[0, 1].plot(grid, np.zeros_like(grid), "--", color="0.35")
    axes[0, 1].plot(grid, 100.0 * (formal_grid / parsec_grid - 1.0), ":", color="k")
    axes[0, 0].set_ylabel(r"$M$ [$M_\odot$] at [M/H]=0")
    axes[0, 1].set_ylabel(r"$100(M/M_{\rm PARSEC}-1)$ [%]")
    axes[0, 0].set_title(f"Final MLR on fixed {int(n_systems):,}-system sample")
    axes[0, 1].set_title("Final MLR residual relative to PARSEC")
    axes[1, 0].set_ylabel("B")
    axes[1, 1].set_ylabel(r"$u_c$")
    axes[2, 0].set_ylabel("C")
    axes[2, 1].set_ylabel(r"$f_{\rm outlier}$")
    for ax in axes.flat:
        ax.grid(alpha=0.2)
        ax.set_xlabel("iteration" if ax in axes[1:, :].flat else r"$M_G$")
    for ax in axes.flat:
        ax.legend(fontsize=8)
    fig.savefig(output / "em_mlr_pilot.png", dpi=180)
    plt.close(fig)


def main():
    args = parse_args()
    prior_sigmas = np.array([
        args.shape_prior_sigma_log_b,
        args.shape_prior_sigma_log_uc,
        args.shape_prior_sigma_log_c,
    ], dtype=np.float64)
    if np.any(~np.isfinite(prior_sigmas)) or np.any(prior_sigmas <= 0):
        raise ValueError("Shape-prior log sigmas must be finite and strictly positive.")
    if args.direct_nodes < 4 or args.direct_chunk < 1:
        raise ValueError("Direct quadrature requires --direct-nodes >= 4 and --direct-chunk >= 1.")
    if args.shape_evaluation == "direct":
        if args.dynamics_lookup is None and args.shape_stack is None:
            raise ValueError("Direct mode requires --dynamics-lookup or the legacy --shape-stack.")
        if args.dynamics_lookup is not None and args.shape_stack is not None:
            raise ValueError("Provide only one of --dynamics-lookup and --shape-stack in direct mode.")
    elif args.shape_stack is None:
        raise ValueError("linear_stack mode requires --shape-stack.")
    if args.stable_cycles < 1:
        raise ValueError("--stable-cycles must be positive.")
    if min(args.objective_rtol, args.shape_tol, args.mass_rtol) <= 0:
        raise ValueError("Convergence tolerances must be strictly positive.")
    args.output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    started = time.monotonic()
    posterior = hm.MetallicityPosteriorGrid.load(args.baseline / T8_POSTERIOR_NAME)
    arrays = workflow.filter_data(
        workflow.load_real_data(args.data), max_systems=None,
        seed=args.seed, fixed_rows=posterior.row_indices,
    )
    workflow._require_t8_posterior_metadata(posterior, mock=False, arrays=arrays)
    full_lookup = None
    if args.dynamics_lookup is not None:
        full_lookup = hm.DynamicsLikelihoodLookup.load(args.dynamics_lookup)
        if args.max_systems < len(posterior.row_indices):
            raise ValueError(
                "Direct mode with --dynamics-lookup requires the full formal sample: "
                f"--max-systems >= {len(posterior.row_indices)}."
            )
        if not np.array_equal(full_lookup.row_indices, posterior.row_indices):
            raise ValueError("Dynamics lookup rows do not match the formal metallicity posterior rows.")
        if args.data_subset is not None:
            raise ValueError("--data-subset is not supported with a full-sample --dynamics-lookup.")
        positions = np.arange(len(posterior.row_indices), dtype=np.int64)
    elif args.data_subset is not None and args.data_subset.exists():
        subset_info = np.load(args.data_subset, allow_pickle=False)
        positions = np.asarray(subset_info["positions"], dtype=np.int64)
    else:
        count = min(int(args.max_systems), len(posterior.row_indices))
        positions = np.sort(np.random.default_rng(args.seed).choice(
            len(posterior.row_indices), size=count, replace=False
        )).astype(np.int64)
    if positions.size != min(int(args.max_systems), len(posterior.row_indices)):
        raise ValueError("Saved subset does not match --max-systems.")
    subset_arrays = {key: np.asarray(value)[positions] for key, value in arrays.items()}
    subset = subset_posterior(posterior, positions)
    subset.validate()
    subset_shape_positions = np.searchsorted(posterior.row_indices, subset.row_indices)
    if not np.array_equal(posterior.row_indices[subset_shape_positions], subset.row_indices):
        raise ValueError("Selected formal row indices are not found in posterior order.")
    np.savez_compressed(args.output / "selected_subset.npz",
                        positions=positions, row_indices=subset.row_indices)
    selection_meta = {"seed": int(args.seed), "n_formal": int(len(posterior.row_indices)),
                      "n_selected": int(len(positions)), "positions_file": "selected_subset.npz"}

    validation = validate_outlier_integral(subset_arrays["u"], subset_arrays["u_sigma"], args.seed)
    log_bad = outlier_log_likelihood(subset_arrays["u"], subset_arrays["u_sigma"])
    if full_lookup is not None:
        dynamics_source = full_lookup
        stack = None
        stack_meta = {
            "source": str(args.dynamics_lookup),
            "source_type": "validated full-sample dynamics lookup; used for mass grid and row/data validation",
            "n_nodes": 0,
            "sqrt_mtot_range": [float(full_lookup.sqrt_mtot_grid[0]), float(full_lookup.sqrt_mtot_grid[-1])],
            "interpolation": "continuous direct Rice quadrature for good shape; lookup good table is not read by the direct objective",
        }
    else:
        stack_full = hm.DynamicsLikelihoodShapeStack.load(args.shape_stack)
        if not np.array_equal(stack_full.row_indices, posterior.row_indices):
            raise ValueError("T8.2 formal shape stack rows do not match formal metallicity posterior rows.")
        stack = subset_shape_stack(stack_full, subset_shape_positions, subset.row_indices,
                                   subset_arrays["u"], subset_arrays["u_sigma"], log_bad)
        dynamics_source = None
        stack_meta = {
            "source": str(args.shape_stack),
            "source_type": "formal T8.2 27-node shape stack",
            "n_nodes": int(stack.n_nodes),
            "sqrt_mtot_range": [float(stack.sqrt_mtot_grid[0]), float(stack.sqrt_mtot_grid[-1])],
            "axes": {key: value.tolist() for key, value in stack.axes.items()},
            "interpolation": "linear trilinear probability interpolation in finite 3-node log-shape box (diagnostic fallback only)",
        }
    mass_surface, _ = workflow.build_surfaces()
    mlr = workflow._make_t8_mlr(mass_surface, argparse.Namespace(
        mlr_knot_x=workflow.T8_MLR_DEFAULT_KNOT_X,
        mlr_knot_z=workflow.T8_MLR_DEFAULT_KNOT_Z,
        mlr_degree_x=3, mlr_degree_z=3, mlr_tau_d=0.10, mlr_tau_x=0.05,
        mlr_tau_z=0.05, mlr_tau_xz=0.05, mlr_solar_mean=0.0,
        mlr_solar_sigma=0.01 / np.log(10.0),
    ))
    mlr.set_data(row_indices=subset.row_indices, u=subset_arrays["u"],
                 u_sigma=subset_arrays["u_sigma"], absg=subset_arrays["absg"],
                 metallicity_grid=subset, dynamics_lookup=dynamics_source,
                 dynamics_shape_stack=stack, raw_u_outlier_log_likelihood=log_bad)
    objective, logpost, loglike = build_objective(mlr, args)
    initial_mlr_path = args.initial_mlr or (args.baseline / T8_MLR_NAME)
    baseline_vector, _ = initial_vector(initial_mlr_path, "default")
    probe_value, probe_grad = objective(baseline_vector)
    probe_plus = baseline_vector.copy()
    probe_plus[0] += 1e-3
    probe_plus_value, _ = objective(probe_plus)
    probe_candidate, probe_result = optimize_block(
        objective, baseline_vector, np.arange(MLR_DIM, dtype=np.int64), None, 20
    )
    (args.output / "objective_probe.json").write_text(json.dumps({
        "baseline_objective": float(probe_value),
        "baseline_gradient_norm_mlr": float(np.linalg.norm(probe_grad[:MLR_DIM])),
        "baseline_gradient_max_mlr": float(np.max(np.abs(probe_grad[:MLR_DIM]))),
        "c0_plus_1e-3_objective": float(probe_plus_value),
        "c0_finite_difference": float(probe_plus_value - probe_value),
        "probe_optimizer_success": bool(probe_result.success),
        "probe_optimizer_message": str(probe_result.message),
        "probe_optimizer_nit": int(probe_result.nit),
        "probe_optimizer_fun": float(probe_result.fun),
        "probe_candidate_objective": float(objective(probe_candidate)[0]),
        "probe_candidate_mlr_max_change": float(np.max(np.abs(probe_candidate[:MLR_DIM] - baseline_vector[:MLR_DIM]))),
    }, indent=2), encoding="utf-8")
    results = {}
    for name in ("default", "S2"):
        vector, trace, statuses = run_trajectory(
            mlr, objective, logpost, loglike, baseline_vector.copy() if name == "default"
            else params_to_vector(vector_to_params(baseline_vector), baseline_vector[F_INDEX], SHAPE_STARTS[name]),
            name, args,
        )
        results[name] = (vector, trace, statuses)
    write_outputs(args, mlr, results, baseline_vector, validation, selection_meta, stack_meta)
    elapsed = time.monotonic() - started
    summary = json.loads((args.output / "summary.json").read_text())
    summary["elapsed_seconds"] = elapsed
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
