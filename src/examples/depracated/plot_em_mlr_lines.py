"""Plot EM-like MAP and PARSEC MLR slices in screenshot-style line panels."""

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
DEFAULT_OUT = ROOT / "results/em_mlr_lines_20260929"
MH_SLICES = np.array([-1.0, -0.5, 0.0, 0.3, 0.6])


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
    parser.add_argument("--n-grid", type=int, default=401)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary = json.loads((args.em_output / "summary.json").read_text())
    with np.load(args.em_output / "map_states.npz", allow_pickle=False) as states:
        vector = np.asarray(states["default_vector"], dtype=float)
    params = em.vector_to_params(vector)
    mlr = build_mlr()
    mg = np.linspace(13.5, 3.5, args.n_grid)
    em_mass = np.asarray(mlr.mass_from_absg_mh(mg[None, :], MH_SLICES[:, None], params), dtype=float)
    parsec_mass = np.asarray(mlr.mass_surface.mass_from_absg_mh(mg[None, :], MH_SLICES[:, None]), dtype=float)
    correction = 100.0 * (em_mass / parsec_mass - 1.0)
    if not np.all(np.isfinite(em_mass)) or not np.all(np.isfinite(parsec_mass)):
        raise ValueError("Mass curves contain non-finite values.")
    if np.any(em_mass <= 0.0) or np.any(parsec_mass <= 0.0):
        raise ValueError("Mass curves contain non-positive values.")

    representative_mg = np.asarray(em.REPRESENTATIVE_MG, dtype=float)
    representative_mass = np.asarray(mlr.mass_from_absg_mh(representative_mg, 0.0, params), dtype=float)
    summary_mass = np.asarray(summary["results"]["default"]["final_representative_masses"], dtype=float)
    representative_error = float(np.max(np.abs(representative_mass - summary_mass)))
    if representative_error > 2e-9:
        raise ValueError(f"Representative-mass mismatch: {representative_error:.3e}")
    solar_anchor_mass = float(mlr.mass_from_absg_mh(4.67, 0.0, params))

    colors = ["#3568a8", "#4c9f70", "#c44e52", "#d27a28", "#7b5ea7"]
    labels = [r"[M/H] = %.1f" % value for value in MH_SLICES]
    plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": 0.22})
    fig, (ax_mass, ax_corr) = plt.subplots(
        2, 1, figsize=(7.6, 6.8), sharex=True,
        gridspec_kw={"height_ratios": [2.0, 1.15], "hspace": 0.16},
    )
    for index, (color, label) in enumerate(zip(colors, labels)):
        ax_mass.plot(mg, em_mass[index], color=color, lw=2.2, label=label)
        ax_mass.plot(mg, parsec_mass[index], color=color, lw=1.6, ls="--")
        ax_corr.plot(mg, correction[index], color=color, lw=2.0)
    ax_mass.plot(4.67, 1.0, marker="*", markersize=12, color="black", markeredgecolor="white", markeredgewidth=0.7, label="Sun (reference)")
    ax_corr.axhline(0.0, color="0.3", lw=1.0)
    ax_mass.set_yscale("log")
    ax_mass.set_ylabel(r"Mass ($M_\odot$)", fontsize=13)
    ax_mass.set_title("EM-like default MAP MLR and PARSEC mass relations", fontsize=14)
    ax_corr.set_ylabel(r"$100\,(M_{\rm EM}/M_{\rm PARSEC}-1)$ (%)", fontsize=12)
    ax_corr.set_xlabel(r"Absolute magnitude $M_G$ (mag)", fontsize=13)
    for ax in (ax_mass, ax_corr):
        ax.set_xlim(13.5, 3.5)
        ax.tick_params(axis="both", labelsize=10)
    color_handles = [
        plt.Line2D([], [], color=color, lw=2.2, label=label)
        for color, label in zip(colors, labels)
    ]
    color_handles.append(
        plt.Line2D([], [], color="black", marker="*", linestyle="None", markersize=10, label="Sun (reference)")
    )
    ax_mass.add_artist(ax_mass.legend(handles=color_handles, loc="upper left", fontsize=8.5, frameon=True, ncol=2))
    style_handle = plt.Line2D([], [], color="0.2", lw=2.0, label="EM-like MAP")
    parsec_handle = plt.Line2D([], [], color="0.2", lw=1.6, ls="--", label="Original PARSEC")
    ax_mass.add_artist(ax_mass.legend(handles=[style_handle, parsec_handle], loc="lower right", fontsize=9, frameon=True))
    fig.tight_layout()
    fig.savefig(args.output_dir / "em_mlr_lines.png", dpi=220, bbox_inches="tight")
    fig.savefig(args.output_dir / "em_mlr_lines.pdf", bbox_inches="tight")
    plt.close(fig)

    report = {
        "source_em_output": str(args.em_output.resolve()),
        "endpoint": "default",
        "M_H_slices": MH_SLICES.tolist(),
        "M_G_range_reversed": [13.5, 3.5],
        "n_grid": int(args.n_grid),
        "shape_B_uc_C": np.exp(vector[em.SHAPE_OFFSET : em.SHAPE_OFFSET + 3]).tolist(),
        "converged": bool(summary["results"]["default"]["converged"]),
        "representative_mass_max_abs_error": representative_error,
        "representative_mass_msun": representative_mass.tolist(),
        "summary_representative_mass_msun": summary_mass.tolist(),
        "solar_anchor": {"M_G": 4.67, "M_H": 0.0, "EM_mass_msun": solar_anchor_mass, "target_msun": 1.0, "absolute_error": abs(solar_anchor_mass - 1.0)},
        "mass_range_msun": [float(np.min(em_mass)), float(np.max(em_mass))],
        "correction_range_percent": [float(np.min(correction)), float(np.max(correction))],
        "finite_positive_checks": True,
        "interpretation": "EM-like default MAP point estimate only; solid curves are EM-like, dashed curves are original PARSEC, and no uncertainty band or refit is shown.",
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
