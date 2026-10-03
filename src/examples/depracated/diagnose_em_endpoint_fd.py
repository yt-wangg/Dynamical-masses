#!/usr/bin/env python3
"""Independent finite-difference diagnostic for an EM-like endpoint.

This script rebuilds the same full-sample direct objective used by
``run_em_mlr_pilot.py`` and evaluates, without fitting, selected finite
differences at the saved default and S2 endpoints.  The B coordinate is
handled specially because the saved solution is on its lower hard bound:
the report contains both the feasible inward slope and an exploratory slope
just below the bound.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np


HERE = Path(__file__).resolve()
REPO_ROOT = HERE.parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))


def load_pilot_module():
    path = HERE.with_name("run_em_mlr_pilot.py")
    spec = importlib.util.spec_from_file_location("run_em_mlr_pilot_fd", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_exact_objective(pilot, args):
    """Mirror the direct full-sample setup in pilot.main, without fitting."""
    import run_hierarchical_metallicity_test as workflow
    from binary_masses import hierarchical_metallicity as hm

    posterior = hm.MetallicityPosteriorGrid.load(args.baseline / pilot.T8_POSTERIOR_NAME)
    arrays = workflow.filter_data(
        workflow.load_real_data(args.data), max_systems=None,
        seed=args.seed, fixed_rows=posterior.row_indices,
    )
    try:
        workflow._require_t8_posterior_metadata(posterior, mock=False, arrays=arrays)
    except ValueError as exc:
        # The polish continuation accepted this exact FITS after independently
        # checking its SHA-256, because a derived-array digest changed across
        # the OneDrive copy.  Reuse that same narrowly scoped fallback.
        if "input-data digest does not match" not in str(exc):
            raise
        args.initial_map_states = args.map_states
        pilot.validate_resume_input_file(args, posterior, arrays)
    lookup = hm.DynamicsLikelihoodLookup.load(args.dynamics_lookup)
    if not np.array_equal(lookup.row_indices, posterior.row_indices):
        raise ValueError("Dynamics lookup rows do not match posterior rows.")
    positions = np.arange(len(posterior.row_indices), dtype=np.int64)
    subset_arrays = {key: np.asarray(value)[positions] for key, value in arrays.items()}
    subset = pilot.subset_posterior(posterior, positions)
    subset.validate()
    log_bad = pilot.outlier_log_likelihood(
        subset_arrays["u"], subset_arrays["u_sigma"]
    )
    mass_surface, _ = workflow.build_surfaces()
    mlr = workflow._make_t8_mlr(mass_surface, __import__("argparse").Namespace(
        mlr_knot_x=workflow.T8_MLR_DEFAULT_KNOT_X,
        mlr_knot_z=workflow.T8_MLR_DEFAULT_KNOT_Z,
        mlr_degree_x=3, mlr_degree_z=3, mlr_tau_d=0.10, mlr_tau_x=0.05,
        mlr_tau_z=0.05, mlr_tau_xz=0.05, mlr_solar_mean=0.0,
        mlr_solar_sigma=0.01 / np.log(10.0),
    ))
    mlr.set_data(
        row_indices=subset.row_indices,
        u=subset_arrays["u"],
        u_sigma=subset_arrays["u_sigma"],
        absg=subset_arrays["absg"],
        metallicity_grid=subset,
        dynamics_lookup=lookup,
        dynamics_shape_stack=None,
        raw_u_outlier_log_likelihood=log_bad,
    )
    objective, logpost, loglike = pilot.build_objective(mlr, args)
    return objective, logpost, loglike, len(positions)


def evaluate(objective, vector):
    value, grad = objective(np.asarray(vector, dtype=np.float64))
    return float(value), np.asarray(grad, dtype=np.float64)


def fd_row(objective, vector, name, index, h_y, f0, g0, scale, lower=None):
    x = np.asarray(vector, dtype=np.float64)
    h = float(scale * h_y)
    plus = x.copy()
    minus = x.copy()
    plus[index] += h
    minus[index] -= h
    fp, _ = evaluate(objective, plus)
    fm, _ = evaluate(objective, minus)
    central = (fp - fm) / (2.0 * h)
    forward = (fp - f0) / h
    backward = (f0 - fm) / h
    return {
        "name": name,
        "index": int(index),
        "h_y": float(h_y),
        "raw_step": float(h),
        "optimizer_scale": float(scale),
        "x": float(x[index]),
        "lower": None if lower is None else float(lower),
        "f0": float(f0),
        "f_plus": float(fp),
        "f_minus": float(fm),
        "fd_central": float(central),
        "fd_forward": float(forward),
        "fd_backward": float(backward),
        "jax_gradient": float(g0[index]),
        "jax_gradient_scaled": float(g0[index] * scale),
        "central_minus_jax": float(central - g0[index]),
        "central_scaled": float(central * scale),
        "central_scaled_minus_jax_scaled": float(central * scale - g0[index] * scale),
        "relative_central_error": float(
            abs(central * scale - g0[index] * scale) / max(1e-12, abs(g0[index] * scale))
        ),
        "below_lower_exploratory": bool(lower is not None and x[index] - h < lower),
    }


def b_inward_row(objective, vector, f0, g0, h_y, scale, lower, do_outward):
    """Feasible one-sided B derivative, plus one exploratory point below bound."""
    index = 35
    h = float(scale * h_y)
    x = np.asarray(vector, dtype=np.float64)
    p1, p2 = x.copy(), x.copy()
    p1[index] += h
    p2[index] += 2.0 * h
    f1, _ = evaluate(objective, p1)
    f2, _ = evaluate(objective, p2)
    # Second-order forward derivative at the lower boundary.
    d1 = (f1 - f0) / h
    d2 = (-3.0 * f0 + 4.0 * f1 - f2) / (2.0 * h)
    outward_h_y = 0.003
    fm = None
    if do_outward:
        pm = x.copy()
        pm[index] -= scale * outward_h_y
        fm, _ = evaluate(objective, pm)
    return {
        "endpoint_coordinate": "log_b",
        "index": index,
        "h_y": float(h_y),
        "raw_step": h,
        "optimizer_scale": float(scale),
        "x": float(x[index]),
        "lower": float(lower),
        "f0": float(f0),
        "f_plus_h": float(f1),
        "f_plus_2h": float(f2),
        "f_minus_outward": None if fm is None else float(fm),
        "inward_forward_scaled": float(d1 * scale),
        "inward_second_order_scaled": float(d2 * scale),
        "jax_gradient_scaled": float(g0[index] * scale),
        "inward_second_order_minus_jax_scaled": float(d2 * scale - g0[index] * scale),
        "outward_slope_scaled": None if fm is None else float((f0 - fm) / (scale * outward_h_y) * scale),
        "outward_h_y": outward_h_y,
        "below_lower_exploratory": bool(x[index] - scale * outward_h_y < lower),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--dynamics-lookup", type=Path, required=True)
    parser.add_argument("--map-states", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260919)
    parser.add_argument("--direct-nodes", type=int, default=12)
    parser.add_argument("--direct-chunk", type=int, default=128)
    parser.add_argument("--endpoint", choices=("default", "S2", "both"), default="both")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    pilot = load_pilot_module()
    args.shape_evaluation = "direct"
    args.shape_prior_sigma_log_b = float(pilot.DEFAULT_SHAPE_PRIOR_SIGMA[0])
    args.shape_prior_sigma_log_uc = float(pilot.DEFAULT_SHAPE_PRIOR_SIGMA[1])
    args.shape_prior_sigma_log_c = float(pilot.DEFAULT_SHAPE_PRIOR_SIGMA[2])
    print("Building exact full-sample objective...", flush=True)
    objective, logpost, loglike, n_systems = build_exact_objective(pilot, args)

    with np.load(args.map_states, allow_pickle=False) as saved:
        vectors = {
            "default": np.asarray(saved["default_vector"], dtype=np.float64),
            "S2": np.asarray(saved["S2_vector"], dtype=np.float64),
        }
    if args.endpoint != "both":
        vectors = {args.endpoint: vectors[args.endpoint]}

    bounds = {
        "f_outlier": (1e-5, 1.0 - 1e-5),
        "log_b": (float(pilot.SHAPE_AXES["log_b"][0]), float(pilot.SHAPE_AXES["log_b"][2])),
        "log_uc": (float(pilot.SHAPE_AXES["log_uc"][0]), float(pilot.SHAPE_AXES["log_uc"][2])),
        "log_c": (float(pilot.SHAPE_AXES["log_c"][0]), float(pilot.SHAPE_AXES["log_c"][2])),
    }
    names = {0: "c0", 32: "log_lambda_x", 33: "log_lambda_z", 34: "f_outlier",
             35: "log_b", 36: "log_uc", 37: "log_c"}
    fd_steps = (0.03, 0.01, 0.003)
    all_rows = []
    endpoint_summary = {}
    source_summary = json.loads((args.map_states.parent / "summary.json").read_text(encoding="utf-8"))
    for endpoint, vector in vectors.items():
        f0, g0 = evaluate(objective, vector)
        expected_f0 = -float(source_summary["results"][endpoint]["final"]["log_posterior"])
        repeated = np.array([evaluate(objective, vector)[0] for _ in range(5)])
        scaled = np.abs(g0[:pilot.MLR_DIM] * pilot.optimizer_scales(np.arange(pilot.MLR_DIM)))
        # Five representative coordinates: the largest MLR gradient, c0,
        # log B, the largest of f/log uc/log C, and the largest lambda scale.
        selected = [int(np.argmax(scaled)), 0, 35]
        shape_or_f = [34, 36, 37]
        selected.append(int(shape_or_f[np.argmax([scaled[i] if i < pilot.MLR_DIM else
                                                  abs(g0[i] * pilot.optimizer_scales(np.array([i]))[0])
                                                  for i in shape_or_f])]))
        lambda_indices = [32, 33]
        selected.append(int(lambda_indices[np.argmax([scaled[i] for i in lambda_indices])]))
        unique = list(dict.fromkeys(selected))
        for index in unique:
            name = names.get(index, f"v{index}")
            lower = None
            if index == 34:
                lower = bounds["f_outlier"][0]
            elif index == 35:
                lower = bounds["log_b"][0]
            elif index == 36:
                lower = bounds["log_uc"][0]
            elif index == 37:
                lower = bounds["log_c"][0]
            if index == 35:
                continue
            scale = float(pilot.optimizer_scales(np.array([index], dtype=np.int64))[0])
            for h_y in fd_steps:
                row = fd_row(objective, vector, name, index, h_y, f0, g0, scale, lower)
                row["endpoint"] = endpoint
                all_rows.append(row)
        b_scale = float(pilot.optimizer_scales(np.array([35], dtype=np.int64))[0])
        b_rows = []
        for h_y in fd_steps:
            row = b_inward_row(objective, vector, f0, g0, h_y, b_scale,
                               bounds["log_b"][0], do_outward=(h_y == fd_steps[-1]))
            row["endpoint"] = endpoint
            b_rows.append(row)
            all_rows.append(row)
        endpoint_summary[endpoint] = {
            "n_systems": int(n_systems),
            "objective_negative_log_posterior": f0,
            "source_objective_negative_log_posterior": expected_f0,
            "source_objective_difference": float(f0 - expected_f0),
            "source_objective_match": bool(abs(f0 - expected_f0) <= 1e-6),
            "log_posterior": float(logpost(vector)),
            "log_likelihood": float(loglike(vector)),
            "jax_gradient_max_abs": float(np.max(np.abs(g0))),
            "jax_gradient_max_abs_mlr": float(np.max(np.abs(g0[:pilot.MLR_DIM]))),
            "scaled_gradient_max_abs_mlr": float(np.max(scaled)),
            "selected_indices": unique,
            "selected_names": [names.get(i, f"v{i}") for i in unique],
            "f0_repeated_values": repeated.tolist(),
            "f0_repeat_range": float(np.ptp(repeated)),
            "f0_ulp": float(np.spacing(f0)),
            "shape": {
                "B": float(np.exp(vector[35])),
                "uc": float(np.exp(vector[36])),
                "C": float(np.exp(vector[37])),
                "log_b": float(vector[35]),
                "log_uc": float(vector[36]),
                "log_c": float(vector[37]),
            },
            "jax_gradients_selected": {
                names.get(i, f"v{i}"): float(g0[i]) for i in unique
            },
            "b_inward_rows": b_rows,
        }

    with (args.output / "finite_difference_rows.csv").open("w", newline="", encoding="utf-8") as stream:
        fieldnames = sorted({key for row in all_rows for key in row})
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    report = {
        "objective": "negative joint log posterior from run_em_mlr_pilot.build_objective",
        "data": str(args.data.resolve()),
        "baseline": str(args.baseline.resolve()),
        "dynamics_lookup": str(args.dynamics_lookup.resolve()),
        "map_states": str(args.map_states.resolve()),
        "n_systems": int(n_systems),
        "finite_difference_steps_log_coordinates": list(fd_steps),
        "shape_bounds": {key: list(value) for key, value in bounds.items()},
        "endpoints": endpoint_summary,
        "rows": all_rows,
    }
    (args.output / "finite_difference_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(endpoint_summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
