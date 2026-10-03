"""Compare observed raw-u data with fixed and EM-like predictive mixtures.

This is a point-estimate predictive visualization only.  It uses the saved fixed
T8 posterior medians and the saved EM-like default MAP state; it never fits a
model.  Both models use the same selected systems, metallicity draws, and
Rice-noise draws within each predictive repetition.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "examples"))

import run_hierarchical_metallicity_test as workflow  # noqa: E402
from binary_masses.hierarchical_metallicity import MonotoneTensorSplineMLR  # noqa: E402


DEFAULT_DATA = ROOT / "data/jd_msms_single_bic_1kpc_filtered_cmdcut_cutb_jcaps_err_mhcal_good.fits"
DEFAULT_POSTERIOR = ROOT / "results/solar_bg466_zero_recalibrated_20260924/latent_metallicity_weights_t8.npz"
DEFAULT_EM = ROOT / "results/em_solar_only_full_rebuild_20260928_b_relaxed_cont40"
DEFAULT_FIXED = ROOT / "results/t8_raw_u_solar_only_recalibrated_baseline_20260924"
DEFAULT_OUT = ROOT / "results/em_fixed_observed_u_predictive_comparison_20260929"
OUTLIER_MU = 40.0
OUTLIER_SIGMA = 13.0
OUTLIER_MAX = 80.0


def build_mlr(result_dir: Path) -> tuple[MonotoneTensorSplineMLR, dict]:
    metadata = json.loads((result_dir / "mlr_model.json").read_text())
    mass_surface, _ = workflow.build_surfaces()
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
    return mlr, metadata


def posterior_medians(result_dir: Path) -> tuple[dict, float, int]:
    with np.load(result_dir / "mlr_mcmc_t8.npz", allow_pickle=False) as saved:
        params = {}
        for name in ("c0", "a", "b", "r", "log_lambda_x", "log_lambda_z"):
            values = np.asarray(saved[f"posterior__{name}"])
            params[name] = np.median(values.reshape((-1,) + values.shape[2:]), axis=0)
        f_values = np.asarray(saved["posterior__f_outlier"])
        f_outlier = float(np.median(f_values.reshape(-1)))
        n_draws = int(f_values.size)
    return params, f_outlier, n_draws


def em_params(em_dir: Path) -> tuple[dict, tuple[float, float, float], float, dict]:
    summary = json.loads((em_dir / "summary.json").read_text())
    with np.load(em_dir / "map_states.npz", allow_pickle=False) as states:
        vector = np.asarray(states["default_vector"], dtype=float)
        shape = tuple(float(v) for v in np.exp(vector[35:38]))
    params = {
        "c0": vector[0],
        "a": vector[1:4],
        "b": vector[4:11],
        "r": vector[11:32].reshape(7, 3),
        "log_lambda_x": vector[32],
        "log_lambda_z": vector[33],
    }
    return params, shape, float(vector[34]), {
        "converged": bool(summary["results"]["default"]["converged"]),
        "summary_shape": summary["results"]["default"]["final_shape"],
        "map_shape": list(shape),
        "f_outlier": float(vector[34]),
    }


def sample_truncated_normal(rng: np.random.Generator, size: int) -> np.ndarray:
    """Sample TN(40,13,[0,80]) by rejection, retaining exact support."""
    values = np.empty(size, dtype=float)
    filled = 0
    while filled < size:
        proposal = rng.normal(OUTLIER_MU, OUTLIER_SIGMA, size - filled + 256)
        accepted = proposal[(proposal >= 0.0) & (proposal <= OUTLIER_MAX)]
        take = min(accepted.size, size - filled)
        values[filled : filled + take] = accepted[:take]
        filled += take
    return values


def sample_good(uniforms: np.ndarray, shape: tuple[float, float, float]) -> np.ndarray:
    grid = np.linspace(0.0, OUTLIER_MAX, 40001)
    B, uc, C = shape
    basis = np.where(grid >= 0.0, grid * np.exp(-B * grid**2 - np.exp((grid - uc) / C)), 0.0)
    cdf = np.empty_like(grid)
    cdf[0] = 0.0
    cdf[1:] = np.cumsum(0.5 * (basis[1:] + basis[:-1]) * np.diff(grid))
    cdf /= cdf[-1]
    return np.interp(uniforms, cdf, grid)


def rice_draw(v: np.ndarray, sigma: np.ndarray, normal_x: np.ndarray, normal_y: np.ndarray) -> np.ndarray:
    return np.sqrt((v + sigma * normal_x) ** 2 + (sigma * normal_y) ** 2)


def predictive_densities(
    *,
    mlr: MonotoneTensorSplineMLR,
    params: dict,
    shape: tuple[float, float, float],
    f_outlier: float,
    absg: np.ndarray,
    u_sigma: np.ndarray,
    probabilities: np.ndarray,
    z_grid: np.ndarray,
    common_draws: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    edges: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    n = len(u_sigma)
    good_counts = np.zeros(edges.size - 1, dtype=np.float64)
    bad_counts = np.zeros(edges.size - 1, dtype=np.float64)
    max_mass = np.zeros(n, dtype=float)
    min_mass = np.full(n, np.inf, dtype=float)
    for z_draw, good_uniforms, bad_v, noise_x, noise_y in common_draws:
        m1 = np.asarray(mlr.mass_from_absg_mh(absg[:, 0], z_draw, params), dtype=float)
        m2 = np.asarray(mlr.mass_from_absg_mh(absg[:, 1], z_draw, params), dtype=float)
        mass = m1 + m2
        max_mass = np.maximum(max_mass, mass)
        min_mass = np.minimum(min_mass, mass)
        good_v = np.sqrt(mass) * sample_good(good_uniforms, shape)
        good_u = rice_draw(good_v, u_sigma, noise_x, noise_y)
        # Reuse the same noise realization for the outlier branch where possible.
        bad_u = rice_draw(bad_v, u_sigma, noise_x, noise_y)
        good_counts += np.histogram(good_u, bins=edges)[0]
        bad_counts += np.histogram(bad_u, bins=edges)[0]
    repetitions = len(common_draws)
    scale = 1.0 / (repetitions * n * np.diff(edges))
    good_density = good_counts * scale
    bad_density = bad_counts * scale
    total_density = (1.0 - f_outlier) * good_density + f_outlier * bad_density
    return good_density, bad_density, total_density, {
        "sample_count_per_component": int(repetitions * n),
        "mass_min": float(np.min(min_mass)),
        "mass_max": float(np.max(max_mass)),
        "good_density_integral_on_plot_range": float(np.sum(good_density * width_from_edges(edges))),
        "bad_density_integral_on_plot_range": float(np.sum(bad_density * width_from_edges(edges))),
    }


def width_from_edges(edges: np.ndarray) -> np.ndarray:
    return np.diff(edges)


def make_common_draws(
    probabilities: np.ndarray,
    z_grid: np.ndarray,
    n: int,
    repetitions: int,
    rng: np.random.Generator,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Create common random numbers for the fixed and variable predictions."""
    z_cdf = np.cumsum(probabilities / probabilities.sum(axis=1, keepdims=True), axis=1)
    draws = []
    for _ in range(repetitions):
        z_draw = z_grid[np.sum(z_cdf < rng.random(n)[:, None], axis=1).clip(0, z_grid.size - 1)]
        draws.append((z_draw, rng.random(n), sample_truncated_normal(rng, n), rng.normal(size=n), rng.normal(size=n)))
    return draws


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--posterior", type=Path, default=DEFAULT_POSTERIOR)
    parser.add_argument("--em-output", type=Path, default=DEFAULT_EM)
    parser.add_argument("--fixed-output", type=Path, default=DEFAULT_FIXED)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--repetitions", type=int, default=128)
    parser.add_argument("--bin-width", type=float, default=2.0)
    parser.add_argument("--x-max", type=float, default=180.0)
    parser.add_argument("--seed", type=int, default=20260929)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.repetitions < 8 or args.bin_width <= 0.0 or args.x_max <= 80.0:
        raise ValueError("Require repetitions >= 8, positive bin-width, and x-max > 80.")

    arrays = workflow.filter_data(
        workflow.load_real_data(args.data), max_systems=None, seed=0,
        fixed_rows=np.load(args.em_output / "selected_subset.npz")["row_indices"],
    )
    with np.load(args.posterior, allow_pickle=False) as posterior:
        posterior_rows = np.asarray(posterior["row_indices"], dtype=np.int64)
        probabilities = np.asarray(posterior["probabilities"], dtype=float)
        z_grid = np.asarray(posterior["z_grid"], dtype=float)
    selected_rows = np.asarray(arrays["row_indices"], dtype=np.int64)
    if not np.array_equal(selected_rows, posterior_rows):
        raise ValueError("Selected FITS rows are not aligned with the metallicity posterior rows.")

    mlr, fixed_metadata = build_mlr(args.fixed_output)
    fixed_params, fixed_f, n_fixed_draws = posterior_medians(args.fixed_output)
    em_params_values, em_shape, em_f, em_metadata = em_params(args.em_output)
    fixed_shape = tuple(float(fixed_metadata["good_shape_constants"][key]) for key in ("B", "uc", "C"))
    edges = np.arange(0.0, args.x_max + args.bin_width, args.bin_width)
    common_draws = make_common_draws(
        probabilities, z_grid, len(selected_rows), args.repetitions, np.random.default_rng(args.seed)
    )
    fixed_good, fixed_bad, fixed_total, fixed_sim = predictive_densities(
        mlr=mlr, params=fixed_params, shape=fixed_shape, f_outlier=fixed_f,
        absg=arrays["absg"], u_sigma=arrays["u_sigma"], probabilities=probabilities,
        z_grid=z_grid, common_draws=common_draws, edges=edges,
    )
    em_good, em_bad, em_total, em_sim = predictive_densities(
        mlr=mlr, params=em_params_values, shape=em_shape, f_outlier=em_f,
        absg=arrays["absg"], u_sigma=arrays["u_sigma"], probabilities=probabilities,
        z_grid=z_grid, common_draws=common_draws, edges=edges,
    )
    centers = 0.5 * (edges[1:] + edges[:-1])
    observed = np.histogram(arrays["u"], bins=edges, density=True)[0]
    fixed_residual = observed - fixed_total
    em_residual = observed - em_total
    plt.rcParams.update({"font.size": 12, "axes.grid": True, "grid.alpha": 0.22})
    fig, (ax, residual_ax) = plt.subplots(
        2, 1, figsize=(7.4, 6.4), sharex=True,
        gridspec_kw={"height_ratios": [3.0, 1.15], "hspace": 0.08},
    )
    ax.step(centers, observed, where="mid", color="0.25", lw=1.5, label="Observed raw $u$")
    ax.plot(centers, fixed_total, color="#3568a8", lw=2.5, label="Our fixed model total")
    ax.plot(centers, em_total, color="#c44e52", lw=2.7, label="Variable model total")
    ax.plot(centers, fixed_f * fixed_bad, color="#3568a8", lw=1.8, ls="--", label=f"Our fixed weighted outlier ($f={fixed_f:.3f}$)")
    ax.plot(centers, em_f * em_bad, color="#c44e52", lw=1.8, ls=":", label=f"Variable weighted outlier ($f={em_f:.3f}$)")
    ax.set_xlim(0.0, args.x_max)
    ax.set_ylabel("Density", fontsize=13)
    ax.set_title("Observed raw-u distribution and predictive models", fontsize=14)
    ax.tick_params(axis="both", labelsize=11)
    ax.legend(loc="upper right", fontsize=9, frameon=True)
    residual_ax.axhline(0.0, color="0.25", lw=1.0)
    residual_ax.plot(centers, fixed_residual, color="#3568a8", lw=1.8, label="Observed − fixed")
    residual_ax.plot(centers, em_residual, color="#c44e52", lw=1.8, label="Observed − variable")
    residual_ax.set_xlabel(r"Observed relative velocity $u$ (km s$^{-1}$ AU$^{1/2}$)", fontsize=13)
    residual_ax.set_ylabel("Residual density", fontsize=12)
    residual_ax.tick_params(axis="both", labelsize=11)
    residual_ax.legend(loc="upper right", fontsize=9, frameon=True)
    fig.tight_layout()
    fig.savefig(args.output_dir / "observed_u_em_fixed_predictive_comparison.png", dpi=240, bbox_inches="tight")
    fig.savefig(args.output_dir / "observed_u_em_fixed_predictive_comparison.pdf", bbox_inches="tight")
    plt.close(fig)

    width = np.diff(edges)
    tail = edges[:-1] >= 60.0
    report = {
        "n_systems": int(selected_rows.size),
        "row_alignment": {
            "selected_subset_rows_equal_posterior_rows": True,
            "first_row": int(selected_rows[0]),
            "last_row": int(selected_rows[-1]),
        },
        "simulation": {
            "seed": int(args.seed),
            "common_random_numbers_between_models": True,
            "repetitions": int(args.repetitions),
            "samples_per_component": int(args.repetitions * selected_rows.size),
            "bin_width": float(args.bin_width),
            "x_range": [float(edges[0]), float(edges[-1])],
            "observed_raw_u_range": [float(np.min(arrays["u"])), float(np.max(arrays["u"]))],
        },
        "fixed_model": {
            "shape": list(fixed_shape),
            "f_outlier_median": fixed_f,
            "posterior_draws": n_fixed_draws,
            "simulation": fixed_sim,
        },
        "em_like_default_map": {
            "shape": list(em_shape),
            "f_outlier": em_f,
            "converged": em_metadata["converged"],
            "simulation": em_sim,
        },
        "density_checks": {
            "observed_histogram_integral_on_plot_range": float(np.sum(observed * width)),
            "fixed_total_integral_on_plot_range": float(np.sum(fixed_total * width)),
            "em_total_integral_on_plot_range": float(np.sum(em_total * width)),
            "fixed_weighted_outlier_integral_on_plot_range": float(np.sum(fixed_f * fixed_bad * width)),
            "em_weighted_outlier_integral_on_plot_range": float(np.sum(em_f * em_bad * width)),
            "fixed_residual_max_abs": float(np.max(np.abs(fixed_residual))),
            "em_residual_max_abs": float(np.max(np.abs(em_residual))),
            "fixed_residual_rms": float(np.sqrt(np.mean(fixed_residual**2))),
            "em_residual_rms": float(np.sqrt(np.mean(em_residual**2))),
        },
        "fraction_above_u_60": {
            "observed": float(np.mean(arrays["u"] >= 60.0)),
            "fixed_point_prediction": float(np.sum(fixed_total[tail] * width[tail])),
            "variable_point_prediction": float(np.sum(em_total[tail] * width[tail])),
        },
        "interpretation": "Raw-u predictive comparison using saved point estimates; no refit or posterior uncertainty band. Intrinsic good support is [0,80], while Rice-predicted observed u may exceed 80.",
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
