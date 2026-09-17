#!/usr/bin/env python3
"""Plot the fitted T8 monotone tensor B-spline MLR surface.

This is a plotting/validation-only script.  It reconstructs the saved model
from the posterior and metadata files and never runs a fit or MCMC sampler.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from binary_masses.hierarchical_metallicity import (  # noqa: E402
    IsochroneMassSurfaceModel,
    MonotoneTensorSplineMLR,
)


DEFAULT_RESULT_DIR = REPO_ROOT / "results" / "hierarchical_metallicity_t8_20260913"
DEFAULT_OUTPUT = "mlr_surface_2d_t8.png"
SLICE_Z = np.array([-1.0, -0.5, 0.0, 0.3, 0.6], dtype=float)


def build_mass_surface() -> IsochroneMassSurfaceModel:
    """Build the same in-domain PARSEC mass surface used by T8 fitting."""
    full = IsochroneMassSurfaceModel.from_interpolated_mass_data(
        REPO_ROOT / "data" / "interpolated_mass_data", mass_min=0.05
    )
    mask = full.absg_grid_np < 13.5 - 1e-12
    x_grid = np.concatenate([full.absg_grid_np[mask], [13.5]])
    mass_grid = np.column_stack(
        [
            full.mass_grid_np[:, mask],
            np.array(
                [full.mass_from_absg_mh(13.5, z) for z in full.mh_grid_np]
            ),
        ]
    )
    return IsochroneMassSurfaceModel(
        x_grid, full.mh_grid_np, mass_grid, mass_min=0.05
    )


def load_model(result_dir: Path):
    metadata = json.loads((result_dir / "mlr_model.json").read_text(encoding="utf-8"))
    mass_surface = build_mass_surface()
    mlr = MonotoneTensorSplineMLR(
        mass_surface,
        knots_x=np.asarray(metadata["knot_x"], dtype=float),
        knots_z=np.asarray(metadata["knot_z"], dtype=float),
        degree_x=int(metadata["degree_x"]),
        degree_z=int(metadata["degree_z"]),
        tau_D=float(metadata["penalty_scales_dex"]["tau_D"]),
        tau_x=float(metadata["penalty_scales_dex"]["tau_x"]),
        tau_Z=float(metadata["penalty_scales_dex"]["tau_Z"]),
        tau_xZ=float(metadata["penalty_scales_dex"]["tau_xZ"]),
        solar_anchor_mean=float(metadata["solar_anchor"]["mean_log10_mass"]),
        solar_anchor_sigma=float(metadata["solar_anchor"]["sigma_log10_mass"]),
    )
    with np.load(result_dir / "mlr_mcmc_t8.npz", allow_pickle=False) as saved:
        samples = {
            key[len("posterior__") :]: np.asarray(saved[key]).reshape(
                (-1,) + saved[key].shape[2:]
            )
            for key in saved.files
            if key.startswith("posterior__") and not key.endswith("solar_anchor_g")
        }
    required = {"c0", "a", "b", "r", "log_lambda_x", "log_lambda_z"}
    missing = required - set(samples)
    if missing:
        raise ValueError(f"Posterior is missing required MLR parameters: {sorted(missing)}")
    return mlr, samples


def evaluate_draws(mlr, samples, x, z, max_draws):
    n_total = samples["c0"].shape[0]
    n_eval = min(int(max_draws), n_total)
    indices = np.linspace(0, n_total - 1, n_eval, dtype=np.int64)
    g = np.empty((n_eval, z.size, x.size), dtype=float)
    delta = np.empty_like(g)
    raw_parsec = np.log10(mlr.mass_surface.mass_from_absg_mh(x[None, :], z[:, None]))
    parsec_projection = mlr.g_parsec_projection(x[None, :], z[:, None])
    for out_i, draw_i in enumerate(indices):
        params = {name: samples[name][draw_i] for name in (
            "c0", "a", "b", "r", "log_lambda_x", "log_lambda_z"
        )}
        g_draw = mlr.g_from_raw(x[None, :], z[:, None], params)
        if not np.all(np.isfinite(g_draw)):
            raise ValueError(f"Non-finite final log-mass values in draw {draw_i}.")
        if np.any(np.diff(g_draw, axis=1) > 2e-11):
            raise ValueError(f"Magnitude monotonicity failed in draw {draw_i}.")
        if np.any(np.diff(g_draw, axis=0) < -2e-11):
            raise ValueError(f"Metallicity monotonicity failed in draw {draw_i}.")
        g[out_i] = g_draw
        delta[out_i] = g_draw - raw_parsec
    return indices, g, delta, parsec_projection


def validate_saved_slices(mlr, samples, result_dir: Path, indices: np.ndarray) -> float:
    """Check exact agreement with the saved 201-point, five-Z derived grid."""
    saved = np.load(result_dir / "mlr_derived_grid_t8.npz", allow_pickle=False)
    x = np.asarray(saved["absg_grid"], dtype=float)
    z = np.asarray(saved["z_grid"], dtype=float)
    if not np.array_equal(z, SLICE_Z):
        raise ValueError(f"Saved derived grid has unexpected Z slices: {z}")
    _, g, delta, _ = evaluate_draws(mlr, samples, x, z, len(indices))
    # evaluate_draws selects by count; saved correction_grid uses the same
    # linspace selection, so the order and draw set are identical.
    mass_pct = np.percentile(10.0**g, [16.0, 50.0, 84.0], axis=0)
    corr_pct = np.percentile(100.0 * (10.0**delta - 1.0), [16.0, 50.0, 84.0], axis=0)
    mass_err = float(np.max(np.abs(mass_pct - saved["mass_percentiles"])))
    corr_err = float(np.max(np.abs(corr_pct - saved["correction_percentiles"])))
    max_err = max(mass_err, corr_err)
    if max_err > 2e-10:
        raise ValueError(
            f"Saved-grid mismatch: max mass error={mass_err:.3e}, "
            f"max correction error={corr_err:.3e}."
        )
    return max_err


def make_plot(x, z, g, delta, output: Path):
    mass_pct = np.percentile(10.0**g, [16.0, 50.0, 84.0], axis=0)
    log_mass_median = np.percentile(g, 50.0, axis=0)
    corr_pct = np.percentile(100.0 * (10.0**delta - 1.0), [16.0, 50.0, 84.0], axis=0)
    correction_median = corr_pct[1]
    correction_half_width = 0.5 * (corr_pct[2] - corr_pct[0])
    X, Z = np.meshgrid(x, z)
    solar_x, solar_z = 4.67, 0.0
    fig, axes = plt.subplots(1, 3, figsize=(17.4, 5.5), constrained_layout=True)
    cmap_mass = "viridis"
    im0 = axes[0].pcolormesh(X, Z, log_mass_median, shading="auto", cmap=cmap_mass)
    mass_levels = np.linspace(np.nanmin(mass_pct[1]), np.nanmax(mass_pct[1]), 8)
    contour = axes[0].contour(
        X, Z, 10.0**log_mass_median, levels=mass_levels,
        colors="white", linewidths=0.55, alpha=0.8,
    )
    axes[0].clabel(contour, inline=True, fontsize=8, fmt=r"%.2f $M_\odot$")
    fig.colorbar(im0, ax=axes[0], label=r"Posterior median $\log_{10}(M/M_\odot)$")
    axes[0].set_title("Posterior median final mass")

    limit = max(abs(float(np.nanmin(correction_median))), abs(float(np.nanmax(correction_median))), 1.0)
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
    im1 = axes[1].pcolormesh(X, Z, correction_median, shading="auto", cmap="coolwarm", norm=norm)
    cmin, cmax = np.nanmin(correction_median), np.nanmax(correction_median)
    if cmin <= 0.0 <= cmax:
        axes[1].contour(X, Z, correction_median, levels=[0.0], colors="black", linewidths=0.9)
    fig.colorbar(im1, ax=axes[1], label="Correction relative to raw PARSEC (%)")
    axes[1].set_title("Posterior median scientific correction")

    im2 = axes[2].pcolormesh(X, Z, correction_half_width, shading="auto", cmap="magma")
    fig.colorbar(im2, ax=axes[2], label="Half 16–84% width (%)")
    axes[2].set_title("Correction uncertainty")

    for ax in axes:
        ax.plot(solar_x, solar_z, marker="*", markersize=11, color="black", markeredgecolor="white", markeredgewidth=0.8)
        ax.set_xlabel(r"$M_G$ (mag)")
        ax.set_ylabel(r"$[M/H]$ (dex)")
        ax.set_xlim(float(x[0]), float(x[-1]))
        ax.set_ylim(float(z[0]), float(z[-1]))
        ax.grid(color="white", alpha=0.18, linewidth=0.5)
    axes[2].text(
        solar_x + 0.18, solar_z + 0.07, "Solar anchor", color="white", fontsize=9,
        bbox={"facecolor": "black", "alpha": 0.45, "edgecolor": "none", "pad": 1.5},
    )
    fig.suptitle("T8 monotone tensor B-spline MLR surface", fontsize=14)
    fig.savefig(output, dpi=190, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT_DIR)
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-draws", type=int, default=512)
    parser.add_argument("--nx", type=int, default=241)
    parser.add_argument("--nz", type=int, default=161)
    args = parser.parse_args()
    result_dir = args.result_dir.resolve()
    mlr, samples = load_model(result_dir)
    x = np.linspace(float(mlr.knots_x[0]), float(mlr.knots_x[-1]), int(args.nx))
    z = np.linspace(float(mlr.knots_z[0]), float(mlr.knots_z[-1]), int(args.nz))
    if not (np.all(np.isfinite(x)) and np.all(np.isfinite(z))):
        raise ValueError("Evaluation grid contains non-finite values.")
    if x[0] < mlr.knots_x[0] or x[-1] > mlr.knots_x[-1] or z[0] < mlr.knots_z[0] or z[-1] > mlr.knots_z[-1]:
        raise ValueError("Evaluation grid leaves the fitted knot domain.")
    indices, g, delta, _ = evaluate_draws(mlr, samples, x, z, args.max_draws)
    if not np.all(np.isfinite(delta)):
        raise ValueError("Scientific correction contains non-finite values.")
    slice_error = validate_saved_slices(mlr, samples, result_dir, indices)
    output = result_dir / args.output
    make_plot(x, z, g, delta, output)
    print(json.dumps({
        "output": str(output),
        "posterior_draws_total": int(samples["c0"].shape[0]),
        "posterior_draws_evaluated": int(indices.size),
        "grid_shape_z_x": [int(z.size), int(x.size)],
        "domain_mg": [float(x[0]), float(x[-1])],
        "domain_mh": [float(z[0]), float(z[-1])],
        "saved_slice_max_abs_error": float(slice_error),
        "monotonicity_validation": "passed for every evaluated draw on dense grid",
    }, indent=2))


if __name__ == "__main__":
    main()
