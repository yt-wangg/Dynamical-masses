#!/usr/bin/env python3
"""Compare two fixed-good-shape T8 MLR posterior grids.

This is deliberately a post-processing script.  It reads completed MLR
products and never launches a fit.  The two products should differ only in
the fixed good-component shape (for example, the T8 baseline and Hwang+2024).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BASELINE = ROOT / "results/t8_raw_u_solar_only_recalibrated_baseline_20260924"
DEFAULT_HWANG = ROOT / "results/t8_raw_u_solar_only_recalibrated_hwang_20260924"
DEFAULT_OUTPUT = ROOT / "results/t8_hwang_shape_recalibrated_pair_20260924"


def _load_result(result_dir: Path) -> dict:
    grid_path = result_dir / "mlr_derived_grid_t8.npz"
    model_path = result_dir / "mlr_model.json"
    if not grid_path.is_file() or not model_path.is_file():
        raise FileNotFoundError(
            f"Missing completed MLR products in {result_dir}. Expected "
            "mlr_derived_grid_t8.npz and mlr_model.json; run the Hwang fit "
            "on the server first."
        )
    with np.load(grid_path) as saved:
        grid = {key: saved[key] for key in saved.files}
    model = json.loads(model_path.read_text())
    lookup_metadata_path = result_dir / "mlr_lookup_metadata_t8.json"
    if not lookup_metadata_path.is_file():
        raise FileNotFoundError(f"Missing {lookup_metadata_path}; cannot verify matched lookup inputs.")
    lookup_metadata = json.loads(lookup_metadata_path.read_text())
    diagnostics = {}
    for name in ("mlr_diagnostics_t8.json", "mlr_diagnostics.json"):
        path = result_dir / name
        if path.is_file():
            diagnostics = json.loads(path.read_text())
            break
    return {"dir": result_dir, "grid": grid, "model": model,
            "lookup_metadata": lookup_metadata, "diagnostics": diagnostics}


def _shape(model: dict) -> dict:
    return model.get("good_shape_constants", {})


def _validate_pair(base: dict, hwang: dict) -> None:
    bgrid, hgrid = base["grid"], hwang["grid"]
    for key in ("absg_grid", "z_grid"):
        if key not in bgrid or key not in hgrid or not np.array_equal(bgrid[key], hgrid[key]):
            raise ValueError(f"The two MLR products do not have identical {key}.")
    for key in ("mass_percentiles", "percentiles"):
        if key not in bgrid or key not in hgrid or bgrid[key].shape != hgrid[key].shape:
            raise ValueError(f"The two MLR products do not have compatible {key}.")
    if not np.array_equal(bgrid["percentiles"], hgrid["percentiles"]) or not np.array_equal(
        bgrid["percentiles"], [16.0, 50.0, 84.0]
    ):
        raise ValueError("The two MLR products need the same 16/50/84 mass percentiles.")
    if not np.array_equal(bgrid["parsec_mass"], hgrid["parsec_mass"]):
        raise ValueError("The PARSEC reference grid differs between runs.")

    bm, hm = base["model"], hwang["model"]
    for key in ("schema", "n_systems", "outlier_coordinate", "outlier_mass_dependent", "metallicity_posterior_source"):
        if bm.get(key) != hm.get(key):
            raise ValueError(f"Model field {key!r} differs between the two products.")
    if bm.get("outlier_shape") != hm.get("outlier_shape"):
        raise ValueError("The outlier model differs; this comparison is intended to isolate good shape.")
    for key in ("knot_x", "knot_z", "degree_x", "degree_z", "solar_anchor"):
        if bm.get(key) != hm.get(key):
            raise ValueError(f"Model field {key!r} differs between the two products.")
    if _shape(bm) == _shape(hm):
        raise ValueError("The two products have identical good-shape constants.")
    expected_shapes = ((0.002544, 35.67, 3.10), (0.00224, 36.09, 3.85))
    for model, expected in zip((bm, hm), expected_shapes):
        actual = _shape(model)
        if not all(np.isclose(actual.get(key, np.nan), value, rtol=0, atol=1e-10)
                   for key, value in zip(("B", "uc", "C"), expected)):
            raise ValueError(f"Unexpected fixed good shape: {actual}.")
    bl, hl = base["lookup_metadata"], hwang["lookup_metadata"]
    for key in ("data_digest", "sqrt_mtot_min", "sqrt_mtot_max", "sqrt_mtot_points",
                "velocity_quadrature_nodes", "velocity_sigma_extent", "normalization_version",
                "outlier_u0", "outlier_sigma", "outlier_normalization_value"):
        if bl.get(key) != hl.get(key):
            raise ValueError(f"Lookup field {key!r} differs between the two products.")


def _interp_at(values: np.ndarray, x: np.ndarray, target: np.ndarray) -> np.ndarray:
    return np.asarray([np.interp(target, x, row) for row in values])


def _plot(base: dict, hwang: dict, output: Path, selected_mg: np.ndarray) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    bg, hg = base["grid"], hwang["grid"]
    z_index = int(np.flatnonzero(bg["z_grid"] == 0.0)[0])
    x = bg["absg_grid"]
    bp = bg["mass_percentiles"][:, z_index, :]
    hp = hg["mass_percentiles"][:, z_index, :]
    parsec = bg["parsec_mass"][z_index]
    # Rows are 16th, 50th, 84th percentiles.
    ratio_pct = 100.0 * (hp[1] / bp[1] - 1.0)

    fig, (ax_mass, ax_parsec, ax_ratio) = plt.subplots(
        3, 1, figsize=(8.6, 9.0), sharex=True,
        gridspec_kw={"height_ratios": [2.0, 1.0, 1.0], "hspace": 0.10},
    )
    colors = {"baseline": "#1769aa", "hwang": "#d95f02"}
    ax_mass.plot(x, parsec, color="0.35", lw=1.5, ls=":", label="PARSEC reference")
    ax_mass.plot(x, bp[1], color=colors["baseline"], lw=2.2, label="Baseline good shape")
    ax_mass.fill_between(x, bp[0], bp[2], color=colors["baseline"], alpha=0.18)
    ax_mass.plot(x, hp[1], color=colors["hwang"], lw=2.2, ls="--", label="Hwang et al. (2024) good shape")
    ax_mass.fill_between(x, hp[0], hp[2], color=colors["hwang"], alpha=0.18)
    ax_mass.set_ylabel(r"Inferred mass [$M_\odot$]")
    ax_mass.set_title("Fixed-good-shape MLR refits at [M/H] = 0")
    ax_mass.grid(alpha=0.22)
    ax_mass.legend(frameon=False, loc="best")
    ax_mass.text(
        0.03, 0.06,
        "Same solar-only metallicity posterior and raw-u outlier\n"
        "Shading: 16–84% posterior interval",
        transform=ax_mass.transAxes, fontsize=8.5, va="bottom",
        bbox={"facecolor": "white", "alpha": 0.88, "edgecolor": "none"},
    )

    ax_parsec.axhline(0.0, color="0.25", lw=1)
    for mass_percentiles, color, line_style in (
        (bp, colors["baseline"], "-"), (hp, colors["hwang"], "--")
    ):
        residual = 100.0 * (mass_percentiles / parsec[None, :] - 1.0)
        ax_parsec.plot(x, residual[1], color=color, lw=2.0, ls=line_style)
        ax_parsec.fill_between(x, residual[0], residual[2], color=color, alpha=0.14)
    ax_parsec.set_ylabel("Mass / PARSEC\n− 1 [%]")
    ax_parsec.grid(alpha=0.22)

    ax_ratio.axhline(0.0, color="0.25", lw=1)
    ax_ratio.plot(x, ratio_pct, color=colors["hwang"], lw=2.0)
    ax_ratio.set_xlabel(r"Absolute magnitude $M_G$")
    ax_ratio.set_ylabel("Hwang / baseline\n− 1 [%]")
    ax_ratio.grid(alpha=0.22)
    fig.savefig(output / "hwang_shape_pair_comparison.png", dpi=220, bbox_inches="tight")
    fig.savefig(output / "hwang_shape_pair_comparison.pdf", bbox_inches="tight")
    plt.close(fig)

    bp_selected = _interp_at(bp, x, selected_mg)
    hp_selected = _interp_at(hp, x, selected_mg)
    parsec_selected = np.interp(selected_mg, x, parsec)

    payload = {
        "baseline_dir": str(base["dir"]),
        "hwang_dir": str(hwang["dir"]),
        "z_value": 0.0,
        "selected_M_G": selected_mg.tolist(),
        "baseline_good_shape_constants": _shape(base["model"]),
        "hwang_good_shape_constants": _shape(hwang["model"]),
        "baseline_mass_percentiles_at_selected_M_G": bp_selected.tolist(),
        "hwang_mass_percentiles_at_selected_M_G": hp_selected.tolist(),
        "baseline_over_parsec_median_percent": (100.0 * (bp_selected[1] / parsec_selected - 1.0)).tolist(),
        "hwang_over_parsec_median_percent": (100.0 * (hp_selected[1] / parsec_selected - 1.0)).tolist(),
        "hwang_over_baseline_median_percent": (100.0 * (hp_selected[1] / bp_selected[1] - 1.0)).tolist(),
        "baseline_diagnostics": base["diagnostics"],
        "hwang_diagnostics": hwang["diagnostics"],
        "comparison_scope": "Separate posterior summaries; no paired posterior draws or uncertainty on the difference.",
    }
    (output / "summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--hwang", type=Path, default=DEFAULT_HWANG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    base = _load_result(args.baseline)
    hwang = _load_result(args.hwang)
    _validate_pair(base, hwang)
    selected_mg = np.array([5.0, 7.0, 9.0, 11.0])
    _plot(base, hwang, args.output, selected_mg)
    print(f"Wrote paired comparison to {args.output}")


if __name__ == "__main__":
    main()
