"""Plot the EM-like default MAP MLR mass surface and PARSEC correction."""

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
DEFAULT_OUT = ROOT / "results/em_mlr_surface_2d_20260929"


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
    parser.add_argument("--nx", type=int, default=241)
    parser.add_argument("--nz", type=int, default=161)
    args = parser.parse_args()
    if args.nx < 21 or args.nz < 21:
        raise ValueError("Require nx and nz >= 21.")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary = json.loads((args.em_output / "summary.json").read_text())
    with np.load(args.em_output / "map_states.npz", allow_pickle=False) as states:
        vector = np.asarray(states["default_vector"], dtype=float)
    params = em.vector_to_params(vector)
    shape_from_map = np.exp(vector[em.SHAPE_OFFSET : em.SHAPE_OFFSET + 3])
    shape_from_summary = np.asarray(summary["results"]["default"]["final_shape"], dtype=float)
    if not np.allclose(shape_from_map, shape_from_summary, rtol=2e-8, atol=2e-10):
        raise ValueError("Default shape in map_states.npz does not match summary.json.")

    mlr = build_mlr()
    mg = np.linspace(3.5, 13.5, args.nx)
    mh = np.linspace(-1.0, 0.6, args.nz)
    inferred_mass = np.asarray(mlr.mass_from_absg_mh(mg[None, :], mh[:, None], params), dtype=float)
    parsec_mass = np.asarray(mlr.mass_surface.mass_from_absg_mh(mg[None, :], mh[:, None]), dtype=float)
    correction_percent = 100.0 * (inferred_mass / parsec_mass - 1.0)
    if not np.all(np.isfinite(inferred_mass)) or not np.all(np.isfinite(parsec_mass)):
        raise ValueError("Mass surface contains non-finite values.")
    if np.any(inferred_mass <= 0.0) or np.any(parsec_mass <= 0.0):
        raise ValueError("Mass surface contains non-positive values.")
    mass_dx_max = float(np.max(np.diff(inferred_mass, axis=1)))
    mass_dz_min = float(np.min(np.diff(inferred_mass, axis=0)))
    if mass_dx_max > 2e-10 or mass_dz_min < -2e-10:
        raise ValueError("Inferred mass surface violates expected monotonicity.")

    representative_mg = np.asarray(em.REPRESENTATIVE_MG, dtype=float)
    representative_mass = np.asarray(
        mlr.mass_from_absg_mh(representative_mg, 0.0, params), dtype=float
    )
    summary_mass = np.asarray(summary["results"]["default"]["final_representative_masses"], dtype=float)
    representative_error = float(np.max(np.abs(representative_mass - summary_mass)))
    if representative_error > 2e-9:
        raise ValueError(f"Representative-mass mismatch: {representative_error:.3e}")

    np.savez_compressed(
        args.output_dir / "em_mlr_surface_grid.npz",
        M_G=mg,
        M_H=mh,
        mass_msun=inferred_mass,
        parsec_mass_msun=parsec_mass,
        correction_percent=correction_percent,
    )

    X, Z = np.meshgrid(mg, mh)
    plt.rcParams.update({"font.size": 11, "axes.grid": False})
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 4.4), constrained_layout=True)
    mass_image = axes[0].pcolormesh(X, Z, inferred_mass, shading="auto", cmap="viridis")
    correction_limit = max(abs(float(np.nanmin(correction_percent))), abs(float(np.nanmax(correction_percent))))
    correction_image = axes[1].pcolormesh(
        X, Z, correction_percent, shading="auto", cmap="coolwarm",
        vmin=-correction_limit, vmax=correction_limit,
    )
    axes[0].set_title("EM-like default MAP mass", fontsize=13)
    axes[1].set_title("Correction relative to PARSEC", fontsize=13)
    axes[0].set_ylabel(r"Metallicity $[M/H]$ (dex)", fontsize=12)
    for ax in axes:
        ax.set_xlabel(r"Absolute magnitude $M_G$ (mag)", fontsize=12)
        ax.tick_params(labelsize=10)
        ax.set_xlim(3.5, 13.5)
        ax.set_ylim(-1.0, 0.6)
    fig.colorbar(mass_image, ax=axes[0], label=r"Inferred mass ($M_\odot$)")
    fig.colorbar(correction_image, ax=axes[1], label=r"$100\,(M_{\rm EM}/M_{\rm PARSEC}-1)$ (%)")
    fig.savefig(args.output_dir / "em_mlr_surface_2d.png", dpi=220, bbox_inches="tight")
    fig.savefig(args.output_dir / "em_mlr_surface_2d.pdf", bbox_inches="tight")
    plt.close(fig)

    report = {
        "source_em_output": str(args.em_output.resolve()),
        "endpoint": "default",
        "grid_shape": [int(args.nz), int(args.nx)],
        "domain": {"M_G": [3.5, 13.5], "M_H": [-1.0, 0.6]},
        "shape_B_uc_C": shape_from_summary.tolist(),
        "f_outlier": float(vector[em.F_INDEX]),
        "converged": bool(summary["results"]["default"]["converged"]),
        "representative_M_G": representative_mg.tolist(),
        "representative_mass_msun": representative_mass.tolist(),
        "summary_representative_mass_msun": summary_mass.tolist(),
        "representative_max_abs_error": representative_error,
        "mass_range_msun": [float(np.min(inferred_mass)), float(np.max(inferred_mass))],
        "parsec_mass_range_msun": [float(np.min(parsec_mass)), float(np.max(parsec_mass))],
        "correction_percent_range": [float(np.min(correction_percent)), float(np.max(correction_percent))],
        "monotonicity_checks": {
            "max_positive_dM_dMG": mass_dx_max,
            "min_dM_dMH": mass_dz_min,
            "expected": "mass non-increasing with M_G and non-decreasing with [M/H]",
        },
        "interpretation": "EM-like default MAP surface only; no fixed T8 comparison and no refit.",
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
