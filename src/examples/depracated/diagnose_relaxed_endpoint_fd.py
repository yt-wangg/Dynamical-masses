#!/usr/bin/env python3
"""Finite-difference precision check at the relaxed-B default endpoint."""

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


def load_old_diagnostic():
    path = HERE.with_name("diagnose_em_endpoint_fd.py")
    spec = importlib.util.spec_from_file_location("diagnose_em_endpoint_fd_base", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fd_row(base, objective, vector, name, index, h_y, f0, g0, scale):
    x = np.asarray(vector, dtype=np.float64)
    h = float(scale * h_y)
    plus, minus = x.copy(), x.copy()
    plus[index] += h
    minus[index] -= h
    fp, _ = base.evaluate(objective, plus)
    fm, _ = base.evaluate(objective, minus)
    central = (fp - fm) / (2.0 * h)
    return {
        "name": name, "index": int(index), "h_y": float(h_y),
        "raw_step": h, "optimizer_scale": float(scale), "x": float(x[index]),
        "f0": float(f0), "f_plus": float(fp), "f_minus": float(fm),
        "fd_central": float(central), "jax_gradient": float(g0[index]),
        "fd_central_scaled": float(central * scale),
        "jax_gradient_scaled": float(g0[index] * scale),
        "central_scaled_minus_jax_scaled": float(central * scale - g0[index] * scale),
        "relative_central_error": float(abs(central - g0[index]) / max(1e-12, abs(g0[index]))),
        "curvature_span_from_smallest": None,
    }


def main():
    p = argparse.ArgumentParser()
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
    base = load_old_diagnostic()
    pilot = base.load_pilot_module()
    args.shape_evaluation = "direct"
    # The relaxed endpoint is allowed below the historical direct-mode bound.
    args.direct_min_b = 1e-5
    args.shape_prior_sigma_log_b = float(pilot.DEFAULT_SHAPE_PRIOR_SIGMA[0])
    args.shape_prior_sigma_log_uc = float(pilot.DEFAULT_SHAPE_PRIOR_SIGMA[1])
    args.shape_prior_sigma_log_c = float(pilot.DEFAULT_SHAPE_PRIOR_SIGMA[2])
    args.initial_map_states = args.map_states
    print("Building exact full-sample objective...", flush=True)
    objective, logpost, loglike, n_systems = base.build_exact_objective(pilot, args)
    with np.load(args.map_states, allow_pickle=False) as saved:
        vector = np.asarray(saved["default_vector"], dtype=np.float64)
    f0, g0 = base.evaluate(objective, vector)
    summary = json.loads((args.map_states.parent / "summary.json").read_text(encoding="utf-8"))
    expected = -float(summary["results"]["default"]["final"]["log_posterior"])
    if abs(f0 - expected) > 1e-6:
        raise RuntimeError(f"Reconstructed objective differs from saved summary by {f0 - expected:.6g}")
    scales = pilot.optimizer_scales(np.arange(pilot.MLR_DIM))
    scaled = g0[:pilot.MLR_DIM] * scales
    max_index = int(np.argmax(np.abs(scaled)))
    names = {0: "c0", 32: "log_lambda_x", 33: "log_lambda_z"}
    selected = [max_index, 0, 32]
    if max_index == 32:
        selected.append(33)
    selected = list(dict.fromkeys(selected))
    steps = (0.03, 0.01, 0.003, 0.001)
    rows = []
    for index in selected:
        name = names.get(index, f"mlr_{index}")
        scale = float(pilot.optimizer_scales(np.array([index], dtype=np.int64))[0])
        coord_rows = [fd_row(base, objective, vector, name, index, h, f0, g0, scale) for h in steps]
        small = coord_rows[-1]["fd_central_scaled"]
        for row in coord_rows:
            row["curvature_span_from_smallest"] = float(row["fd_central_scaled"] - small)
            row["endpoint"] = "default"
        rows.extend(coord_rows)
    repeated = np.array([base.evaluate(objective, vector)[0] for _ in range(5)])
    report = {
        "objective": "negative joint log posterior from run_em_mlr_pilot.build_objective",
        "data": str(args.data.resolve()), "baseline": str(args.baseline.resolve()),
        "dynamics_lookup": str(args.dynamics_lookup.resolve()), "map_states": str(args.map_states.resolve()),
        "n_systems": int(n_systems), "endpoint": "default", "steps_scaled_coordinate": list(steps),
        "selected_indices": selected, "selected_names": [names.get(i, f"mlr_{i}") for i in selected],
        "max_abs_scaled_gradient_mlr_index": max_index,
        "max_abs_scaled_gradient_mlr_name": names.get(max_index, f"mlr_{max_index}"),
        "max_abs_scaled_gradient_mlr": float(np.abs(scaled[max_index])),
        "objective_negative_log_posterior": float(f0), "source_objective_negative_log_posterior": expected,
        "source_objective_difference": float(f0 - expected), "source_objective_match": bool(abs(f0 - expected) <= 1e-6),
        "log_posterior": float(logpost(vector)), "log_likelihood": float(loglike(vector)),
        "f0_repeated_values": repeated.tolist(), "f0_repeat_range": float(np.ptp(repeated)),
        "f0_ulp": float(np.spacing(f0)), "jax_gradients_selected": {names.get(i, f"mlr_{i}"): float(g0[i]) for i in selected},
        "rows": rows,
    }
    with (args.output / "finite_difference_rows.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = sorted({key for row in rows for key in row})
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    (args.output / "finite_difference_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
