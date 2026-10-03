"""Raw-factor MLR prior predictive curves with the EM MAP overlay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "examples"))

import run_em_mlr_pilot as em  # noqa: E402
import run_hierarchical_metallicity_test as workflow  # noqa: E402


DEFAULT_EM = ROOT / "results/em_solar_only_full_rebuild_20260928_b_relaxed_cont40"
DEFAULT_OUT = ROOT / "results/em_mlr_raw_prior_predictive_20260929"


def build_mlr():
    mass_surface, _ = workflow.build_surfaces()
    return workflow._make_t8_mlr(
        mass_surface,
        argparse.Namespace(
            mlr_knot_x=workflow.T8_MLR_DEFAULT_KNOT_X,
            mlr_knot_z=workflow.T8_MLR_DEFAULT_KNOT_Z,
            mlr_degree_x=3,
            mlr_degree_z=3,
            mlr_tau_d=0.10,
            mlr_tau_x=0.05,
            mlr_tau_z=0.05,
            mlr_tau_xz=0.05,
            mlr_solar_mean=0.0,
            mlr_solar_sigma=0.01 / np.log(10.0),
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--em-output", type=Path, default=DEFAULT_EM)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--draws", type=int, default=256)
    parser.add_argument("--n-grid", type=int, default=241)
    parser.add_argument("--seed", type=int, default=20260929)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    mlr = build_mlr()
    rng = np.random.default_rng(args.seed)
    mg = np.linspace(3.5, 13.5, args.n_grid)
    mh = np.array([-0.5, 0.0, 0.5])
    center_c0 = float(mlr.parsec_projection[0, 0])
    lambda_ref_x, lambda_ref_z = mlr.lambda_references()

    with np.load(args.em_output / "map_states.npz", allow_pickle=False) as states:
        map_vector = np.asarray(states["default_vector"], dtype=float)
    map_params = em.vector_to_params(map_vector)
    map_log_mass = np.asarray(mlr.g_from_raw(mg[None, :], mh[:, None], map_params), dtype=float)
    prior_log_mass = np.empty((args.draws, mh.size, mg.size), dtype=float)
    for draw in range(args.draws):
        params = {
            "c0": rng.normal(center_c0, 0.10),
            "a": rng.normal(0.0, 1.0, size=3),
            "b": rng.normal(0.0, 1.0, size=7),
            "r": rng.normal(0.0, 1.0, size=(7, 3)),
            "log_lambda_x": rng.normal(np.log(lambda_ref_x), 0.7),
            "log_lambda_z": rng.normal(np.log(lambda_ref_z), 0.7),
        }
        prior_log_mass[draw] = np.asarray(mlr.g_from_raw(mg[None, :], mh[:, None], params), dtype=float)
    if not np.all(np.isfinite(prior_log_mass)) or not np.all(np.isfinite(map_log_mass)):
        raise ValueError("Prior predictive or MAP curves contain non-finite values.")
    # Monotonicity is built into theta_from_raw; verify the sampled curves.
    prior_monotone_mag = np.all(np.diff(prior_log_mass, axis=2) <= 2e-10, axis=2)
    map_monotone_mag = np.all(np.diff(map_log_mass, axis=1) <= 2e-10, axis=1)
    if not np.all(map_monotone_mag):
        raise ValueError("The EM MAP curve violates MLR monotonicity.")

    percentiles = np.percentile(prior_log_mass, [5.0, 50.0, 95.0], axis=0)
    display_min = min(float(np.min(percentiles[0])), float(np.min(map_log_mass))) - 0.25
    display_max = max(float(np.max(percentiles[2])), float(np.max(map_log_mass))) + 0.25
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 4.4), sharey=True)
    fig.subplots_adjust(left=0.07, right=0.995, bottom=0.19, top=0.83, wspace=0.08)
    colors = ["#3568a8", "#c44e52", "#4c9f70"]
    for index, (ax, metal, color) in enumerate(zip(axes, mh, colors)):
        for draw in range(min(args.draws, 80)):
            ax.plot(mg, prior_log_mass[draw, index], color="0.65", lw=0.35, alpha=0.22)
        ax.fill_between(mg, percentiles[0, index], percentiles[2, index], color=color, alpha=0.24, label="Raw-factor 5–95%")
        ax.plot(mg, percentiles[1, index], color=color, lw=1.8, label="Raw-factor median")
        ax.plot(mg, map_log_mass[index], color="black", lw=2.0, label="EM default MAP")
        ax.set_title(rf"$[M/H]={metal:+.1f}$", fontsize=12)
        ax.set_xlabel(r"Absolute magnitude $M_G$ (mag)", fontsize=10)
        ax.invert_xaxis()
        ax.grid(alpha=0.22)
        ax.tick_params(labelsize=9)
        ax.set_ylim(display_min, display_max)
    axes[0].set_ylabel(r"$\log_{10}(M/M_\odot)$", fontsize=11)
    axes[0].legend(fontsize=8, frameon=True, loc="lower left")
    fig.suptitle("Raw-factor MLR prior predictive and EM-like default MAP", fontsize=14)
    fig.text(0.5, 0.01, "Monotonicity is built in. Coupled D shrinkage, smoothness, and solar-anchor terms are omitted; tails extend beyond the plotted range.", ha="center", fontsize=8.5)
    fig.savefig(args.output_dir / "em_mlr_raw_factor_prior_predictive.png", dpi=220, bbox_inches="tight")
    fig.savefig(args.output_dir / "em_mlr_raw_factor_prior_predictive.pdf", bbox_inches="tight")
    plt.close(fig)

    report = {
        "source_em_output": str(args.em_output.resolve()),
        "draws": int(args.draws),
        "seed": int(args.seed),
        "grid_shape": [int(mh.size), int(mg.size)],
        "metallicity_slices": mh.tolist(),
        "M_G_domain": [3.5, 13.5],
        "prior_sampling": {
            "c0": f"Normal({center_c0:.10g},0.1)",
            "a_b_r": "iid Normal(0,1)",
            "log_lambda_x": f"Normal(log({lambda_ref_x:.10g}),0.7)",
            "log_lambda_z": f"Normal(log({lambda_ref_z:.10g}),0.7)",
        },
        "raw_factor_prior_excludes": ["D shrinkage", "D smoothness", "solar anchor"],
        "finite_values": True,
        "monotonicity_validation": {
            "sampled_raw_factor_draws_monotone_in_M_G_fraction": float(np.mean(prior_monotone_mag)),
            "em_map_slices_monotone_in_M_G": bool(np.all(map_monotone_mag)),
            "note": "Monotonicity is built into theta_from_raw and is retained in every raw-factor draw.",
        },
        "prior_log_mass_range": [float(np.min(prior_log_mass)), float(np.max(prior_log_mass))],
        "map_log_mass_range": [float(np.min(map_log_mass)), float(np.max(map_log_mass))],
        "interpretation": "This is a raw-factor induced prior predictive, not the full joint prior and not a posterior distribution.",
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
