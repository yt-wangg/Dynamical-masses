#!/usr/bin/env python3
"""Export the provisional EM-like MLR at several metallicities.

This plotting-only script reconstructs the same hard-monotone MLR and PARSEC
mass surface used by ``run_em_mlr_pilot.py``.  It reads the saved MAP vectors,
evaluates them on a common ``(M_G, [M/H])`` grid, and writes presentation
figures plus a long-form CSV.  It never runs an optimizer or a sampler.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from run_em_mlr_pilot import vector_to_params  # noqa: E402
import run_hierarchical_metallicity_test as workflow  # noqa: E402


DEFAULT_RESULT = REPO_ROOT / "results" / "em_solar_only_full_rebuild_20260927_em_continuation_run2"
DEFAULT_OUTPUT = REPO_ROOT / "results" / "em_mlr_multimetallicity_provisional_20260928"
METALLICITIES = np.array([-0.5, -0.25, 0.0, 0.25, 0.5], dtype=float)


def make_mlr(mass_surface):
    """Construct the exact MLR parameterization used by the EM pilot."""
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


def load_vectors(result_dir: Path):
    map_path = result_dir / "map_states.npz"
    if not map_path.exists():
        raise FileNotFoundError(f"Missing saved MAP states: {map_path}")
    with np.load(map_path, allow_pickle=False) as saved:
        required = {"default_vector", "S2_vector"}
        missing = required - set(saved.files)
        if missing:
            raise ValueError(f"map_states.npz is missing: {sorted(missing)}")
        return {name: np.asarray(saved[f"{name}_vector"], dtype=float) for name in ("default", "S2")}


def evaluate(result_dir: Path, output_dir: Path, *, nx: int = 401):
    summary_path = result_dir / "summary.json"
    source_summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    converged = all(
        source_summary.get("results", {}).get(name, {}).get("converged") is True
        for name in ("default", "S2")
    )
    mass_surface, _ = workflow.build_surfaces()
    mlr = make_mlr(mass_surface)
    vectors = load_vectors(result_dir)
    x = np.linspace(3.5, 13.5, int(nx), dtype=float)

    curves = {}
    for model_name, vector in vectors.items():
        params = vector_to_params(vector)
        curves[model_name] = {
            "mass": np.asarray(
                [mlr.mass_from_absg_mh(x, mh, params) for mh in METALLICITIES], dtype=float
            ),
        }
    parsec = np.asarray(
        [mass_surface.mass_from_absg_mh(x, mh) for mh in METALLICITIES], dtype=float
    )
    for model_name in curves:
        curves[model_name]["residual_percent"] = 100.0 * (
            curves[model_name]["mass"] / parsec - 1.0
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "em_mlr_multimetallicity.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["model", "mh_dex", "M_G_mag", "mass_msun", "parsec_mass_msun", "residual_percent"])
        for model_name in ("default", "S2"):
            for iz, mh in enumerate(METALLICITIES):
                for ix, mg in enumerate(x):
                    writer.writerow([
                        model_name,
                        f"{mh:.8g}",
                        f"{mg:.8g}",
                        f"{curves[model_name]['mass'][iz, ix]:.12g}",
                        f"{parsec[iz, ix]:.12g}",
                        f"{curves[model_name]['residual_percent'][iz, ix]:.12g}",
                    ])

    colors = {"default": "#2479ad", "S2": "#d9792b"}
    labels = {"default": "Default-shape start", "S2": "S2-shape start"}
    fig, axes = plt.subplots(
        2, len(METALLICITIES), figsize=(17.5, 7.2), sharex=True,
        sharey="row", constrained_layout=True,
        gridspec_kw={"height_ratios": [1.45, 1.0]},
    )
    for iz, mh in enumerate(METALLICITIES):
        ax_mass, ax_res = axes[:, iz]
        for model_name in ("default", "S2"):
            ax_mass.plot(x, curves[model_name]["mass"][iz], color=colors[model_name],
                         lw=2.0, label=labels[model_name])
            ax_res.plot(x, curves[model_name]["residual_percent"][iz], color=colors[model_name],
                        lw=2.0, label=labels[model_name])
        ax_mass.plot(x, parsec[iz], "--", color="0.35", lw=1.7, label="PARSEC")
        ax_res.axhline(0.0, ls="--", color="0.35", lw=1.2)
        ax_mass.set_title(rf"[M/H] = {mh:+.2f}")
        ax_res.set_xlabel(r"$M_G$ [mag]")
        ax_mass.set_xlim(x[0], x[-1])
        ax_mass.set_xticks([4, 7, 10, 13])
        ax_res.set_xticks([4, 7, 10, 13])
        ax_mass.grid(alpha=0.2)
        ax_res.grid(alpha=0.2)
    axes[0, 0].set_ylabel(r"Mass [$M_\odot$]")
    axes[1, 0].set_ylabel(r"$100(M/M_{\rm PARSEC}-1)$ [%]")
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 1.03))
    status_line = (
        "Converged alternating-MAP endpoints; no posterior intervals"
        if converged else
        "Provisional MAP endpoints; no posterior intervals; convergence not certified"
    )
    fig.suptitle("EM-like MLR across metallicity slices\n" + status_line, fontsize=14)
    png_path = output_dir / "em_mlr_multimetallicity_parsec.png"
    pdf_path = output_dir / "em_mlr_multimetallicity_parsec.pdf"
    fig.savefig(png_path, dpi=190, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    metadata = {
        "status": "converged_map_evaluation" if converged else "provisional_saved_map_evaluation",
        "description": "Plot-only evaluation of saved alternating-MAP vectors; no optimizer or sampler was run.",
        "source_result_dir": str(result_dir),
        "source_map_states": str(result_dir / "map_states.npz"),
        "models": ["default", "S2"],
        "metallicity_slices_mh_dex": METALLICITIES.tolist(),
        "M_G_domain_mag": [float(x[0]), float(x[-1])],
        "n_M_G_points": int(x.size),
        "mass_surface": "workflow.build_surfaces() and official mass_surface.mass_from_absg_mh",
        "mlr_evaluation": "MonotoneTensorSplineMLR.mass_from_absg_mh reconstructed with run_em_mlr_pilot.vector_to_params",
        "output_files": [csv_path.name, png_path.name, pdf_path.name],
        "source_convergence": {
            name: source_summary.get("results", {}).get(name, {}).get("converged")
            for name in ("default", "S2")
        },
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return x, curves, parsec, metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--nx", type=int, default=401)
    args = parser.parse_args()
    x, curves, parsec, metadata = evaluate(args.result_dir.resolve(), args.output_dir.resolve(), nx=args.nx)
    print(json.dumps({
        "output_dir": str(args.output_dir.resolve()),
        "n_mg": int(x.size),
        "metallicity_slices": METALLICITIES.tolist(),
        "source_convergence": metadata["source_convergence"],
        "max_abs_residual_percent": {
            name: float(np.max(np.abs(curves[name]["residual_percent"])))
            for name in curves
        },
        "max_default_s2_relative_difference": float(
            np.max(np.abs(curves["default"]["mass"] / curves["S2"]["mass"] - 1.0))
        ),
        "parsec_finite": bool(np.all(np.isfinite(parsec))),
    }, indent=2))


if __name__ == "__main__":
    main()
