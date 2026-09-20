#!/usr/bin/env python3
"""Run one independent [M/H]-bin T8.2 joint-shape MCMC.

The two start labels are initialization choices for the same posterior:
``parsec`` uses the default T8 shape and ``s2`` uses the original S2 shape.
The normal component uses direct Rice quadrature, or a probability mixture
of tabulated shapes when --shape-stack-unpacked is supplied. The outlier
component is the raw-u TN(40, 13, [0,80])
Rice convolution and is independent of mass and metallicity.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "src" / "examples"))

from binary_masses import hierarchical_metallicity as hm  # noqa: E402
import run_em_mlr_pilot as em  # noqa: E402
import run_hierarchical_metallicity_test as workflow  # noqa: E402


BOUNDARIES = np.array([-1.0, -0.5, 0.0, 0.3, 0.6], dtype=np.float64)
START_SHAPES = {
    "parsec": (0.002544, 35.67, 3.1),
    "s2": (0.00103, 40.98, 11.53),
}
DIRECT_SHAPE_OFFSETS = (-8.0, -4.0, -2.0, 0.0, 2.0, 4.0, 8.0)
DIRECT_RICE_OFFSETS = (-30.0, -14.0, -10.0, -6.0, -3.0, 0.0, 3.0, 6.0, 10.0, 14.0, 30.0)
VALIDATION_SHAPES = {
    **START_SHAPES,
    "em_parsec": (0.0019698189981769507, 33.53931429506461, 2.5),
    "em_s2": (0.0007, 28.59638752813399, 2.5),
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--posterior", type=Path, required=True)
    parser.add_argument("--nodes-json", type=Path, required=True)
    parser.add_argument(
        "--shape-stack-unpacked", type=Path, default=None,
        help="Directory containing memory-mapped formal stack .npy members.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bin-index", type=int, required=True, choices=range(4))
    parser.add_argument("--start", choices=tuple(START_SHAPES), required=True)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--chains", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260920)
    parser.add_argument(
        "--direct-nodes", type=int, default=32,
        help="Gauss-Legendre order per adaptive velocity interval.",
    )
    parser.add_argument("--direct-chunk", type=int, default=32)
    parser.add_argument("--initial-jitter", type=float, default=0.03)
    parser.add_argument(
        "--skip-validation", action="store_true",
        help="Skip the expensive adaptive-reference check after a one-time validation.",
    )
    parser.add_argument(
        "--max-systems", type=int, default=None,
        help="Optional deterministic random subsample for a resource pilot; never use for production.",
    )
    parser.add_argument("--progress-bar", action="store_true")
    return parser.parse_args()


def load_selected_arrays(data_path, posterior_path):
    arrays = workflow.load_real_data(data_path)
    posterior = hm.MetallicityPosteriorGrid.load(posterior_path)
    arrays = workflow.filter_data(
        arrays,
        max_systems=None,
        seed=0,
        fixed_rows=posterior.row_indices,
    )
    if not np.array_equal(arrays["row_indices"], posterior.row_indices):
        raise ValueError("Formal data rows and metallicity posterior rows are not aligned.")
    return arrays, posterior


def make_dummy_shape_stack(arrays, nodes_json):
    metadata = json.loads(Path(nodes_json).read_text(encoding="utf-8"))
    axes = {
        name: np.asarray(values, dtype=np.float64)
        for name, values in metadata["axes"].items()
    }
    nodes = metadata["nodes"]
    if len(nodes) != 27:
        raise ValueError("The formal T8.2 shape metadata must contain 27 nodes.")
    sqrt_min, sqrt_max = 0.25, 2.5
    sqrt_grid = np.array([sqrt_min, sqrt_max], dtype=np.float64)
    rows = np.asarray(arrays["row_indices"], dtype=np.int64)
    n = rows.size
    digest = hm.array_digest(rows, arrays["u"], arrays["u_sigma"])
    stack = hm.DynamicsLikelihoodShapeStack(
        row_indices=rows,
        sqrt_mtot_grid=sqrt_grid,
        log_good_stack=np.zeros((27, n, 2), dtype=np.float32),
        log_bad_stack=np.zeros((27, n, 2), dtype=np.float32),
        axes=axes,
        node_constants=nodes,
        metadata={
            "model": "t8d2-direct-shape-mcmc",
            "schema": "t8d2-rice-shapestack-v1",
            "data_digest": digest,
            "row_order": "row_indices order",
            "source_nodes_json": str(Path(nodes_json).resolve()),
            "sqrt_mtot_min": sqrt_min,
            "sqrt_mtot_max": sqrt_max,
            "evaluation": "continuous analytic good-shape Rice quadrature",
        },
    )
    return stack


def make_formal_shape_stack(arrays, nodes_json, unpacked_dir):
    """Load only the selected rows of the formal stack from .npy memmaps."""
    metadata = json.loads(Path(nodes_json).read_text(encoding="utf-8"))
    axes = {
        name: np.asarray(values, dtype=np.float64)
        for name, values in metadata["axes"].items()
    }
    nodes = metadata["nodes"]
    root = Path(unpacked_dir)
    stack_rows = np.load(root / "row_indices.npy", mmap_mode="r")
    target_rows = np.asarray(arrays["row_indices"], dtype=np.int64)
    positions = np.searchsorted(stack_rows, target_rows)
    if np.any(positions >= stack_rows.size) or not np.array_equal(
        np.asarray(stack_rows)[positions], target_rows
    ):
        raise ValueError("Formal stack rows do not cover the selected bin.")
    sqrt_grid = np.asarray(np.load(root / "sqrt_mtot_grid.npy", mmap_mode="r"))
    good_memmap = np.load(root / "log_good_stack.npy", mmap_mode="r")
    bad_memmap = np.load(root / "log_bad_stack.npy", mmap_mode="r")
    good = np.asarray(good_memmap[:, positions, :])
    bad = np.asarray(bad_memmap[:, positions, :])
    digest = hm.array_digest(target_rows, arrays["u"], arrays["u_sigma"])
    return hm.DynamicsLikelihoodShapeStack(
        row_indices=target_rows,
        sqrt_mtot_grid=sqrt_grid,
        log_good_stack=good,
        log_bad_stack=bad,
        axes=axes,
        node_constants=nodes,
        metadata={
            "model": "t8d2-formal-stack-linear-shape-mcmc",
            "schema": "t8d2-rice-shapestack-v1",
            "data_digest": digest,
            "row_order": "selected bin row_indices order",
            "source_nodes_json": str(Path(nodes_json).resolve()),
            "source_stack_unpacked": str(root.resolve()),
            "evaluation": "formal node stack with linear log-parameter interpolation",
        },
    )


def direct_good_loglik_numpy(u_obs, u_sigma, sqrt_mtot, shape, *, nodes=12):
    """Independent piecewise GL implementation matching the JAX evaluator."""
    B, uc, C = map(float, shape)
    gl_x, gl_w = np.polynomial.legendre.leggauss(int(nodes))
    norm_x, norm_w = np.polynomial.legendre.leggauss(256)
    norm_x = 0.5 * (norm_x + 1.0) * 80.0
    norm_w = 40.0 * norm_w
    raw_norm = np.sum(
        norm_w * 5.434e-3 * norm_x
        * np.exp(-B * norm_x**2 - np.exp((norm_x - uc) / C))
    )
    from scipy.special import i0e, logsumexp
    shape_offsets = np.array([-8, -4, -2, 0, 2, 4, 8], dtype=np.float64)
    rice_offsets = np.array([-30, -14, -10, -6, -3, 0, 3, 6, 10, 14, 30], dtype=np.float64)
    upper = float(sqrt_mtot) * 80.0
    boundaries = np.concatenate((
        np.array([0.0, upper]),
        float(sqrt_mtot) * (uc + C * shape_offsets),
        float(u_obs) + float(u_sigma) * rice_offsets,
    ))
    boundaries = np.unique(np.clip(boundaries, 0.0, upper))
    interval_log_integrals = []
    for lower, upper_interval in zip(boundaries[:-1], boundaries[1:]):
        if upper_interval <= lower:
            continue
        midpoint = 0.5 * (lower + upper_interval)
        half_width = 0.5 * (upper_interval - lower)
        velocity = midpoint + half_width * gl_x
        argument = float(u_obs) * velocity / float(u_sigma) ** 2
        log_rice = (
            np.log(float(u_obs) / float(u_sigma) ** 2)
            - (float(u_obs) - velocity) ** 2 / (2.0 * float(u_sigma) ** 2)
            + np.log(i0e(argument) + 1e-30)
        )
        tilde = velocity / float(sqrt_mtot)
        valid = (tilde > 0.0) & (tilde <= 80.0)
        log_good = (
            np.log(5.434e-3) - np.log(raw_norm) + np.log(np.maximum(tilde, 1e-300))
            - B * tilde**2 - np.exp((tilde - uc) / C) - np.log(float(sqrt_mtot))
        )
        log_values = np.where(valid, log_rice + log_good, -np.inf)
        interval_log_integrals.append(
            np.log(half_width) + logsumexp(np.log(gl_w) + log_values)
        )
    return float(logsumexp(interval_log_integrals))


def validate_direct_quadrature(arrays, output, nodes):
    rng = np.random.default_rng(20260919)
    take = np.sort(rng.choice(arrays["u"].size, size=min(8, arrays["u"].size), replace=False))
    scales = np.array([0.45, 0.65, 0.9, 1.2, 1.6], dtype=np.float64)
    records = {}
    max_error = 0.0
    for name, shape in VALIDATION_SHAPES.items():
        hm.set_good_shape_constants(B=shape[0], uc=shape[1], C=shape[2])
        errors = []
        reference_underflow = 0
        for row in take:
            for scale in scales:
                reference_value = hm.rice_component_reference_integral(
                    float(arrays["u"][row]), float(arrays["u_sigma"][row]),
                    float(scale), component="good",
                )
                # The adaptive reference is returned in linear space and can
                # underflow for an extremely narrow, many-sigma tail.  Such a
                # value cannot provide a meaningful log-error reference.
                if not (reference_value > 0.0 and np.isfinite(reference_value)):
                    reference_underflow += 1
                    continue
                reference = np.log(reference_value)
                approx = direct_good_loglik_numpy(
                    arrays["u"][row], arrays["u_sigma"][row], scale, shape, nodes=nodes
                )
                errors.append(abs(approx - reference))
        records[name] = {
            "n_systems": int(take.size),
            "n_comparisons": int(len(errors)),
            "n_reference_underflow": int(reference_underflow),
            "scales": scales.tolist(),
            "max_abs_log_error": float(np.max(errors)) if errors else None,
            "p95_abs_log_error": float(np.quantile(errors, 0.95)) if errors else None,
            "rms_abs_log_error": float(np.sqrt(np.mean(np.square(errors)))) if errors else None,
        }
        if errors:
            max_error = max(max_error, records[name]["max_abs_log_error"])
    if max_error >= 1e-3:
        raise RuntimeError(f"Direct good-shape quadrature failed: max |dlogL|={max_error:.4g}")
    result = {"pass": True, "max_abs_log_error": max_error, "shapes": records}
    (output / "direct_quadrature_validation.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result


def build_initial_values(mlr, start):
    B, uc, C = START_SHAPES[start]
    values = mlr.initial_raw_parameters()
    values.update({
        "log_good_shape_b": np.log(B),
        "log_good_shape_uc": np.log(uc),
        "log_good_shape_c": np.log(C),
    })
    return values


def main():
    args = parse_args()
    started = time.monotonic()
    args.output.mkdir(parents=True, exist_ok=True)
    arrays, posterior = load_selected_arrays(args.data, args.posterior)
    z_median = np.asarray(posterior.z_quantiles)[:, 1]
    bin_index = np.digitize(z_median, BOUNDARIES[1:-1], right=False)
    keep = bin_index == int(args.bin_index)
    if not np.any(keep):
        raise ValueError("Requested metallicity bin is empty.")
    full_bin_count = int(keep.sum())
    if args.max_systems is not None:
        if args.max_systems < 1:
            raise ValueError("--max-systems must be positive.")
        if args.max_systems < full_bin_count:
            candidate = np.flatnonzero(keep)
            rng = np.random.default_rng(int(args.seed) + 817)
            chosen = np.sort(rng.choice(candidate, size=int(args.max_systems), replace=False))
            keep = np.zeros_like(keep, dtype=bool)
            keep[chosen] = True
    arrays = {key: np.asarray(value)[keep] for key, value in arrays.items()}
    posterior = hm.MetallicityPosteriorGrid(
        row_indices=posterior.row_indices[keep],
        z_grid=posterior.z_grid.copy(),
        probabilities=posterior.probabilities[keep],
        z_quantiles=posterior.z_quantiles[keep],
        bad_probabilities=posterior.bad_probabilities[keep],
        metadata={**posterior.metadata, "bin_index": int(args.bin_index),
                  "bin_boundaries": BOUNDARIES.tolist(), "bin_assignment": "median latent Z"},
    )
    if args.shape_stack_unpacked is None:
        stack = make_dummy_shape_stack(arrays, args.nodes_json)
    else:
        stack = make_formal_shape_stack(arrays, args.nodes_json, args.shape_stack_unpacked)
    mass_surface, _ = workflow.build_surfaces()
    mlr = workflow._make_t8_mlr(mass_surface, argparse.Namespace(
        mlr_knot_x=workflow.T8_MLR_DEFAULT_KNOT_X,
        mlr_knot_z=workflow.T8_MLR_DEFAULT_KNOT_Z,
        mlr_degree_x=3, mlr_degree_z=3, mlr_tau_d=0.10, mlr_tau_x=0.05,
        mlr_tau_z=0.05, mlr_tau_xz=0.05, mlr_solar_mean=0.0,
        mlr_solar_sigma=0.01 / np.log(10.0),
    ))
    mlr.set_data(
        row_indices=arrays["row_indices"], u=arrays["u"], u_sigma=arrays["u_sigma"],
        absg=arrays["absg"], metallicity_grid=posterior, dynamics_shape_stack=stack,
    )
    mlr.shape_stack_mode = "linear" if args.shape_stack_unpacked is not None else "direct"
    mlr.direct_velocity_quadrature_nodes = int(args.direct_nodes)
    mlr.direct_shape_chunk = int(args.direct_chunk)
    mlr.raw_u_outlier_log_likelihood = em.outlier_log_likelihood(
        arrays["u"], arrays["u_sigma"], nodes=512
    )
    validation = None
    if not args.skip_validation:
        validation = validate_direct_quadrature(
            arrays, args.output, int(args.direct_nodes)
        )
    initial = build_initial_values(mlr, args.start)
    (args.output / "run_config.json").write_text(json.dumps({
        "algorithm": "joint MLR plus continuous good-shape MCMC",
        "shape_evaluation": "formal stack with linear log-parameter interpolation" if args.shape_stack_unpacked is not None else "continuous piecewise Rice quadrature",
        "start": args.start,
        "start_shape_B_uc_C": START_SHAPES[args.start],
        "bin_index": int(args.bin_index),
        "bin_boundaries": BOUNDARIES.tolist(),
        "n_systems": int(keep.sum()),
        "full_bin_count": full_bin_count,
        "pilot_subsample": args.max_systems is not None and int(keep.sum()) < full_bin_count,
        "formal_posterior": str(args.posterior.resolve()),
        "assignment": "median latent [M/H] only; full 81-node posterior retained",
        "outlier": "raw-u TN(40,13,[0,80]) Rice convolution; mass-independent",
        "shape_box": {name: np.asarray(values).tolist() for name, values in stack.axes.items()},
        "formal_stack_unpacked": str(args.shape_stack_unpacked.resolve()) if args.shape_stack_unpacked is not None else None,
        "direct_velocity_quadrature_nodes": int(args.direct_nodes),
        "direct_velocity_shape_offsets": DIRECT_SHAPE_OFFSETS,
        "direct_velocity_rice_offsets": DIRECT_RICE_OFFSETS,
        "direct_shape_chunk": int(args.direct_chunk),
        "validation_skipped": bool(args.skip_validation),
        "initial_jitter_scale": float(args.initial_jitter),
        "validation": validation,
    }, indent=2), encoding="utf-8")
    # Expose independent CPU devices so NumPyro can run the requested chains
    # in parallel.  This must happen before the first JAX import in the MCMC
    # path; the data preparation above is NumPy-only.
    import numpyro
    numpyro.set_host_device_count(int(args.chains))
    sampler = mlr.run_mcmc(
        num_warmup=int(args.warmup), num_samples=int(args.samples),
        num_chains=int(args.chains), seed=int(args.seed),
        progress_bar=bool(args.progress_bar), initial_values=initial,
        initial_jitter_scale=float(args.initial_jitter), target_accept_prob=0.95,
    )
    workflow.save_grouped_mcmc(sampler, args.output / "mlr_mcmc_t8.npz")
    workflow.save_posterior_summary(mlr.posterior_samples, args.output / "mlr_summary_t8.csv")
    workflow.save_sampler_diagnostics(sampler, args.output / "mlr_diagnostics_t8.json")
    elapsed = time.monotonic() - started
    (args.output / "run_timing.json").write_text(json.dumps({
        "elapsed_seconds": elapsed,
        "n_systems": int(keep.sum()), "chains": int(args.chains),
        "warmup": int(args.warmup), "samples": int(args.samples),
    }, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "n_systems": int(keep.sum()),
                      "elapsed_seconds": elapsed}, indent=2), flush=True)


if __name__ == "__main__":
    main()
