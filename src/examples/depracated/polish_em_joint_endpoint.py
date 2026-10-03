#!/usr/bin/env python3
"""Joint L-BFGS-B polishing of the direct full-sample EM endpoints."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve()
REPO_ROOT = HERE.parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def masses(mlr, vector, mg, mh):
    params = mlr.vector_to_params(vector) if hasattr(mlr, "vector_to_params") else None
    if params is None:
        params = {
            "c0": vector[0], "a": vector[1:4], "b": vector[4:11],
            "r": vector[11:32].reshape(7, 3),
            "log_lambda_x": vector[32], "log_lambda_z": vector[33],
            "f_outlier": vector[34],
        }
    return np.asarray([[float(mlr.mass_from_absg_mh(x, z, params)) for z in mh] for x in mg])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--baseline", type=Path, required=True)
    p.add_argument("--dynamics-lookup", type=Path, required=True)
    p.add_argument("--map-states", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seed", type=int, default=20260919)
    p.add_argument("--direct-nodes", type=int, default=12)
    p.add_argument("--direct-chunk", type=int, default=128)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    args.shape_evaluation = "direct"
    fd = load_module("diagnose_em_endpoint_fd_polish", HERE.with_name("diagnose_em_endpoint_fd.py"))
    pilot = fd.load_pilot_module()
    args.shape_prior_sigma_log_b = float(pilot.DEFAULT_SHAPE_PRIOR_SIGMA[0])
    args.shape_prior_sigma_log_uc = float(pilot.DEFAULT_SHAPE_PRIOR_SIGMA[1])
    args.shape_prior_sigma_log_c = float(pilot.DEFAULT_SHAPE_PRIOR_SIGMA[2])
    captured = {}
    original_build = pilot.build_objective

    def capture_build(mlr, build_args):
        captured["mlr"] = mlr
        return original_build(mlr, build_args)

    pilot.build_objective = capture_build
    print("Building exact full-sample objective...", flush=True)
    objective, logpost, loglike, n_systems = fd.build_exact_objective(pilot, args)
    mlr = captured["mlr"]
    with np.load(args.map_states, allow_pickle=False) as saved:
        starts = {name: np.asarray(saved[f"{name}_vector"], dtype=np.float64)
                  for name in ("default", "S2")}
    source_summary = json.loads((args.map_states.parent / "summary.json").read_text(encoding="utf-8"))
    source_shape_box = source_summary["settings"]["shape_box"]
    bounds = [(-np.inf, np.inf)] * pilot.MLR_DIM + [
        (1e-5, 1.0 - 1e-5),
        tuple(float(x) for x in source_shape_box["log_b"])[::2],
        tuple(float(x) for x in source_shape_box["log_uc"])[::2],
        tuple(float(x) for x in source_shape_box["log_c"])[::2],
    ]
    shape_bounds = bounds[pilot.MLR_DIM:]
    indices = np.arange(pilot.FULL_DIM, dtype=np.int64)
    records = {}
    vectors = {}
    for name, start in starts.items():
        start_obj, _ = objective(start)
        expected = -float(source_summary["results"][name]["final"]["log_posterior"])
        if not np.isfinite(start_obj) or abs(start_obj - expected) > 1e-6:
            raise ValueError(
                f"{name} starting objective mismatch: rebuilt={start_obj:.12g}, "
                f"saved={expected:.12g}, tolerance=1e-6")
        print(f"{name}: start objective={start_obj:.12g}", flush=True)
        candidate, result = pilot.optimize_block(
            objective, start, indices, bounds, maxiter=120, label=f"joint-{name}",
            ftol=1e-14, gtol=1e-6, maxls=50,
        )
        candidate_obj, candidate_grad = objective(candidate)
        candidate_start_delta = float(candidate_obj - start_obj)
        accepted = bool(np.isfinite(candidate_obj) and candidate_start_delta <= 5e-11)
        final = candidate if accepted else start.copy()
        final_obj, final_grad = objective(final)
        metrics = pilot.projected_gradient_metrics(objective, final, shape_bounds)
        records[name] = {
            "n_systems": int(n_systems),
            "source_objective_negative_log_posterior": expected,
            "starting_negative_log_posterior": float(start_obj),
            "starting_log_posterior": float(-start_obj),
            "starting_source_difference": float(start_obj - expected),
            "candidate_negative_log_posterior": float(candidate_obj),
            "candidate_minus_start_negative_log_posterior": candidate_start_delta,
            "final_negative_log_posterior": float(final_obj),
            "final_log_posterior": float(-final_obj),
            "objective_non_decreasing": accepted,
            "retained_start_on_failure": bool(not accepted),
            "optimizer_success": bool(result.success),
            "optimizer_message": str(result.message),
            "status": int(result.status),
            "nit": int(result.nit), "nfev": int(result.nfev),
            "final_projected_gradient": metrics,
            "candidate_max_abs_gradient": float(np.max(np.abs(candidate_grad))),
            "final_B": float(np.exp(final[35])), "final_uc": float(np.exp(final[36])),
            "final_C": float(np.exp(final[37])), "final_f": float(final[34]),
            "formal_convergence": bool(metrics["finite"] and metrics["mlr_inf_norm"] <= 1e-6
                                       and metrics["shape_inf_norm"] <= 1e-6),
        }
        vectors[name] = final
        print(f"{name}: final log posterior={-final_obj:.12g}; {result.message}", flush=True)

    mg_rep = np.array([5.0, 7.0, 9.0, 11.0, 13.0])
    mh_grid = np.array([-1.0, -0.5, 0.0, 0.3, 0.6])
    mg_grid = np.array([3.5, 6.0, 8.5, 11.0, 13.5])
    cont40 = np.asarray(starts["default"])
    ref_rep = masses(mlr, cont40, mg_rep, np.array([0.0]))[:, 0]
    ref_grid = masses(mlr, cont40, mg_grid, mh_grid)
    comparison = {}
    for name, vector in vectors.items():
        rep = masses(mlr, vector, mg_rep, np.array([0.0]))[:, 0]
        grid = masses(mlr, vector, mg_grid, mh_grid)
        comparison[name] = {
            "representative_mg": mg_rep.tolist(),
            "representative_mh": 0.0,
            "representative_percent_change_vs_cont40": (100.0 * (rep / ref_rep - 1.0)).tolist(),
            "multi_mg": mg_grid.tolist(), "multi_mh": mh_grid.tolist(),
            "multi_percent_change_vs_cont40": (100.0 * (grid / ref_grid - 1.0)).tolist(),
        }
    cross = vectors["default"] - vectors["S2"]
    report = {
        "objective": "negative joint log posterior from diagnose_em_endpoint_fd.build_exact_objective",
        "data": str(args.data.resolve()), "baseline": str(args.baseline.resolve()),
        "dynamics_lookup": str(args.dynamics_lookup.resolve()), "map_states": str(args.map_states.resolve()),
        "n_systems": int(n_systems), "settings": {"maxiter": 120, "maxls": 50, "ftol": 1e-14, "gtol": 1e-6},
        "results": records, "comparison_vs_cont40": comparison,
        "cross_start_difference": {"max_abs_vector_difference": float(np.max(np.abs(cross))),
                                    "l2_vector_difference": float(np.linalg.norm(cross))},
    }
    np.savez_compressed(args.output / "joint_polish_vectors.npz", **vectors,
                        default_start=starts["default"], S2_start=starts["S2"])
    (args.output / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
