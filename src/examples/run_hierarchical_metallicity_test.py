#!/usr/bin/env python3
"""Run the independent minimal hierarchical metallicity/MLR experiment."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys
from typing import Dict, Mapping, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from binary_masses.hierarchical_metallicity import (  # noqa: E402
    HierarchicalMetallicityCalibrator,
    IsochroneColorSurfaceModel,
    IsochroneMassSurfaceModel,
    MetallicityPosteriorGrid,
    ThreeKnotMetallicityMLR,
    color_uncertainty_from_flux_snr,
    grouped_mcmc_payload,
    simulate_mock_dataset,
)


INPUT_COLUMNS = {
    "feh_observed": ["jc_m_h_fit_1", "jc_m_h_fit_2"],
    "feh_sigma": ["jc_sigma_m_h_cal_1", "jc_sigma_m_h_cal_2"],
    "color_observed": ["bp_rp0_1", "bp_rp0_2"],
    "absg": ["absg1", "absg2"],
    "bp_snr": ["phot_bp_mean_flux_over_error1", "phot_bp_mean_flux_over_error2"],
    "rp_snr": ["phot_rp_mean_flux_over_error1", "phot_rp_mean_flux_over_error2"],
}


def _stack_columns(table: Table, names) -> np.ndarray:
    missing = [name for name in names if name not in table.colnames]
    if missing:
        raise ValueError(f"Input table is missing columns: {missing}.")
    return np.column_stack([np.asarray(table[name], dtype=np.float64) for name in names])


def load_real_data(data_path: Path) -> Dict[str, np.ndarray]:
    table = Table.read(data_path)
    arrays = {key: _stack_columns(table, names) for key, names in INPUT_COLUMNS.items()}
    arrays["color_sigma"] = color_uncertainty_from_flux_snr(
        arrays.pop("bp_snr"), arrays.pop("rp_snr")
    )
    if "u" in table.colnames and "u_sigma" in table.colnames:
        arrays["u"] = np.asarray(table["u"], dtype=np.float64)
        arrays["u_sigma"] = np.asarray(table["u_sigma"], dtype=np.float64)
    else:
        required = ["pmra1", "pmra2", "pmdec1", "pmdec2", "parallax1", "sep_AU", "dpm_over_error"]
        missing = [name for name in required if name not in table.colnames]
        if missing:
            raise ValueError(f"Cannot construct u; missing columns: {missing}.")
        velocity = 4.74 * np.sqrt(
            (np.asarray(table["pmra2"]) - np.asarray(table["pmra1"])) ** 2
            + (np.asarray(table["pmdec2"]) - np.asarray(table["pmdec1"])) ** 2
        ) / np.asarray(table["parallax1"])
        arrays["u"] = velocity * np.sqrt(np.asarray(table["sep_AU"]))
        arrays["u_sigma"] = arrays["u"] / np.asarray(table["dpm_over_error"])
    arrays["row_indices"] = np.arange(len(table), dtype=np.int64)
    return arrays


def filter_data(
    arrays: Mapping[str, np.ndarray],
    *,
    max_systems: Optional[int],
    seed: int,
    fixed_rows: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    n_total = len(arrays["row_indices"])
    finite = np.ones(n_total, dtype=bool)
    for key in ("absg", "feh_observed", "feh_sigma", "color_observed", "color_sigma"):
        finite &= np.all(np.isfinite(arrays[key]), axis=1)
    finite &= np.isfinite(arrays["u"]) & np.isfinite(arrays["u_sigma"])
    finite &= np.all(arrays["feh_sigma"] > 0, axis=1)
    finite &= np.all(arrays["color_sigma"] > 0, axis=1)
    finite &= (arrays["u"] > 0) & (arrays["u_sigma"] > 0)
    in_magnitude_domain = np.all((arrays["absg"] >= 3.5) & (arrays["absg"] <= 13.5), axis=1)
    valid = finite & in_magnitude_domain
    valid_positions = np.flatnonzero(valid)
    print(
        f"Data selection: {n_total} input, {np.sum(~finite)} invalid, "
        f"{np.sum(finite & ~in_magnitude_domain)} outside M_G=[3.5,13.5], "
        f"{valid_positions.size} retained."
    )

    if fixed_rows is not None:
        fixed_rows = np.asarray(fixed_rows, dtype=np.int64)
        row_to_position = {int(row): int(pos) for pos, row in enumerate(arrays["row_indices"])}
        try:
            positions = np.array([row_to_position[int(row)] for row in fixed_rows], dtype=int)
        except KeyError as exc:
            raise ValueError(f"Saved row index {exc.args[0]} is absent from the input data.") from exc
        if np.any(~valid[positions]):
            bad_rows = fixed_rows[~valid[positions]][:10]
            raise ValueError(f"Saved rows no longer pass input validation: {bad_rows.tolist()}.")
    else:
        positions = valid_positions
        if max_systems is not None and positions.size > int(max_systems):
            positions = np.sort(
                np.random.default_rng(seed).choice(positions, size=int(max_systems), replace=False)
            )
            print(f"Subsampled {positions.size} systems with seed={seed}.")
    return {key: np.asarray(value)[positions] for key, value in arrays.items()}


def build_surfaces():
    mass_surface = IsochroneMassSurfaceModel.from_interpolated_mass_data(
        REPO_ROOT / "data" / "interpolated_mass_data", mass_min=0.05
    )
    color_surface = IsochroneColorSurfaceModel.from_parsec_csv(
        REPO_ROOT / "data" / "PARSEC_logAge_6to10_0p5_MH_n1to0p6_0p2.csv",
        absg_grid=np.linspace(3.5, 13.5, 500),
    )
    return mass_surface, color_surface


def save_grouped_mcmc(sampler, path: Path) -> None:
    np.savez_compressed(path, **grouped_mcmc_payload(sampler))


def _flatten_samples(samples: Mapping[str, np.ndarray]):
    for name, values in samples.items():
        values = np.asarray(values)
        if values.ndim == 1:
            yield name, values
        else:
            for index in np.ndindex(values.shape[1:]):
                label = name + "[" + ",".join(str(value) for value in index) + "]"
                yield label, values[(slice(None),) + index]


def save_posterior_summary(samples: Mapping[str, np.ndarray], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["parameter", "mean", "sd", "q05", "q16", "q50", "q84", "q95"])
        for name, values in _flatten_samples(samples):
            q05, q16, q50, q84, q95 = np.percentile(values, [5, 16, 50, 84, 95])
            writer.writerow(
                [name, np.mean(values), np.std(values, ddof=1), q05, q16, q50, q84, q95]
            )


def save_sampler_diagnostics(sampler, path: Path) -> None:
    fields = sampler.get_extra_fields(group_by_chain=True)
    diagnostics = {
        "num_divergences": int(np.sum(np.asarray(fields.get("diverging", 0)))),
        "mean_accept_prob": float(np.mean(np.asarray(fields.get("accept_prob", np.nan)))),
        "max_num_steps": int(np.max(np.asarray(fields.get("num_steps", 0)))),
        "num_chains": int(sampler.num_chains),
        "num_samples_per_chain": int(sampler.num_samples),
    }
    path.write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")


def save_latent_summary(posterior: MetallicityPosteriorGrid, path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["row_index", "z_q16", "z_q50", "z_q84", "bad_probability_1", "bad_probability_2"])
        for row, quantile, bad in zip(
            posterior.row_indices, posterior.z_quantiles, posterior.bad_probabilities
        ):
            writer.writerow([int(row), *quantile.tolist(), *bad.tolist()])


def plot_calibration(
    calibrator: HierarchicalMetallicityCalibrator,
    posterior: MetallicityPosteriorGrid,
    output_dir: Path,
) -> None:
    samples = calibrator.posterior_samples
    z = np.linspace(-1.0, 0.6, 150)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for mg, color in zip((5.0, 8.5, 12.0), ("tab:blue", "tab:green", "tab:red")):
        correction = (
            samples["z_offset"][:, None]
            + (samples["z_scale"][:, None] - 1.0) * z[None, :]
            + samples["z_mag_slope"][:, None] * (mg - 8.5) / 5.0
        )
        lo, med, hi = np.percentile(correction, [16, 50, 84], axis=0)
        axes[0].fill_between(z, lo, hi, color=color, alpha=0.18)
        axes[0].plot(z, med, color=color, label=fr"$M_G={mg:g}$")
    axes[0].axhline(0, color="0.4", linestyle="--", linewidth=1)
    axes[0].set(xlabel="latent PARSEC [M/H]", ylabel="observed minus latent [M/H] (dex)")
    axes[0].legend(frameon=False)

    axes[1].hist(posterior.z_quantiles[:, 1], bins=50, histtype="step", density=True, label="posterior median Z")
    axes[1].hist(calibrator.feh_observed.ravel(), bins=70, histtype="step", density=True, label="observed components")
    axes[1].set(xlabel="[M/H]", ylabel="density")
    axes[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "metallicity_systematics.png", dpi=180)
    plt.close(fig)

    median_z = posterior.z_quantiles[:, 1]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for component, color in enumerate(("tab:blue", "tab:orange")):
        predicted = calibrator.color_surface.color_from_absg_mh(
            calibrator.absg[:, component], median_z
        )
        residual = calibrator.color_observed[:, component] - predicted
        axes[0].scatter(
            calibrator.absg[:, component], residual, s=3, alpha=0.12, color=color,
            label=f"component {component + 1}",
        )
        axes[1].hist(residual, bins=70, histtype="step", density=True, color=color)
    axes[0].axhline(0, color="0.4", linestyle="--", linewidth=1)
    axes[0].set(xlabel="$M_G$", ylabel="BP-RP residual (mag)")
    axes[0].legend(frameon=False, markerscale=3)
    axes[1].set(xlabel="BP-RP residual (mag)", ylabel="density")
    fig.tight_layout()
    fig.savefig(output_dir / "cmd_residuals.png", dpi=180)
    plt.close(fig)


def plot_mlr_grid(grid: Mapping[str, np.ndarray], output_dir: Path) -> None:
    absg, z_values = grid["absg_grid"], grid["z_grid"]
    correction, mass = grid["correction_percent"], grid["mass"]
    cmap = plt.get_cmap("viridis")
    norm = plt.Normalize(z_values.min(), z_values.max())
    fig, (ax_mass, ax_corr) = plt.subplots(2, 1, figsize=(8, 9), sharex=True)
    for index, z in enumerate(z_values):
        color = cmap(norm(z))
        ax_mass.fill_between(absg, mass[0, index], mass[2, index], color=color, alpha=0.15)
        ax_mass.plot(absg, mass[1, index], color=color)
        ax_corr.fill_between(absg, correction[0, index], correction[2, index], color=color, alpha=0.15)
        ax_corr.plot(absg, correction[1, index], color=color, label=f"{z:+.1f}")
    ax_mass.set_yscale("log")
    ax_mass.set_ylabel(r"Mass ($M_\odot$)")
    ax_mass.invert_xaxis()
    ax_corr.axhline(0, color="0.4", linestyle="--", linewidth=1)
    ax_corr.set(xlabel="$M_G$", ylabel="Correction relative to PARSEC (%)")
    ax_corr.legend(title="[M/H]", frameon=False, ncol=3)
    fig.tight_layout()
    fig.savefig(output_dir / "mlr_correction.png", dpi=180)
    plt.close(fig)


def save_mlr_grid_csv(grid: Mapping[str, np.ndarray], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["absg", "mh", "correction_p16_percent", "correction_p50_percent", "correction_p84_percent", "mass_p16", "mass_p50", "mass_p84"]
        )
        for iz, z in enumerate(grid["z_grid"]):
            for ix, mg in enumerate(grid["absg_grid"]):
                writer.writerow(
                    [mg, z, *grid["correction_percent"][:, iz, ix], *grid["mass"][:, iz, ix]]
                )


def recovery_report(samples: Mapping[str, np.ndarray], truth: Mapping[str, object], names) -> Dict[str, object]:
    result = {}
    for name in names:
        if name not in samples or name not in truth:
            continue
        draw = np.asarray(samples[name])
        expected = np.asarray(truth[name])
        lo, median, hi = np.percentile(draw, [5, 50, 95], axis=0)
        result[name] = {
            "truth": expected.tolist(),
            "median": np.asarray(median).tolist(),
            "inside_90_percent_interval": np.asarray((expected >= lo) & (expected <= hi)).tolist(),
            "sign_recovered": np.asarray(np.sign(median) == np.sign(expected)).tolist(),
        }
    return result


def run_calibration(
    arrays: Mapping[str, np.ndarray],
    color_surface: IsochroneColorSurfaceModel,
    output_dir: Path,
    args,
    truth: Optional[Mapping[str, object]],
) -> MetallicityPosteriorGrid:
    calibrator = HierarchicalMetallicityCalibrator(color_surface)
    calibrator.set_data(
        row_indices=arrays["row_indices"],
        absg=arrays["absg"],
        feh_observed=arrays["feh_observed"],
        feh_sigma=arrays["feh_sigma"],
        color_observed=arrays["color_observed"],
        color_sigma=arrays["color_sigma"],
    )
    print("Running stage-one metallicity calibration...")
    sampler = calibrator.run_mcmc(
        num_warmup=args.warmup,
        num_samples=args.samples,
        num_chains=args.chains,
        seed=args.seed,
        target_accept_prob=args.target_accept,
    )
    sampler.print_summary()
    metadata = {
        "model": "hierarchical_metallicity_minimal_v1",
        "feh_columns": INPUT_COLUMNS["feh_observed"],
        "feh_sigma_columns": INPUT_COLUMNS["feh_sigma"],
        "color_columns": INPUT_COLUMNS["color_observed"],
        "absg_columns": INPUT_COLUMNS["absg"],
        "input_path": str(args.data),
        "seed": int(args.seed),
        "mock": bool(args.mock),
        "n_systems": int(len(arrays["row_indices"])),
    }
    posterior = calibrator.posterior_grid(
        max_draws=args.posterior_grid_draws, seed=args.seed, metadata=metadata
    )
    posterior.save(output_dir / "latent_metallicity_weights.npz")
    save_grouped_mcmc(sampler, output_dir / "calibration_mcmc.npz")
    save_posterior_summary(calibrator.posterior_samples, output_dir / "calibration_summary.csv")
    save_sampler_diagnostics(sampler, output_dir / "calibration_diagnostics.json")
    save_latent_summary(posterior, output_dir / "latent_metallicity_summary.csv")
    plot_calibration(calibrator, posterior, output_dir)
    if truth is not None:
        report = recovery_report(
            calibrator.posterior_samples,
            truth,
            ["z_offset", "z_scale", "z_mag_slope", "z_extra_scatter", "bad_fraction", "cmd_scatter"],
        )
        (output_dir / "calibration_recovery.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
    return posterior


def run_mlr(
    arrays: Mapping[str, np.ndarray],
    posterior: MetallicityPosteriorGrid,
    mass_surface: IsochroneMassSurfaceModel,
    output_dir: Path,
    args,
    truth: Optional[Mapping[str, object]],
) -> None:
    metadata = posterior.metadata
    if bool(metadata.get("mock", False)) != bool(args.mock):
        raise ValueError(
            "The saved metallicity posterior mock/real-data mode does not match this run."
        )
    if metadata.get("feh_columns") != INPUT_COLUMNS["feh_observed"]:
        raise ValueError("Metallicity posterior was created with different observed columns.")
    if metadata.get("feh_sigma_columns") != INPUT_COLUMNS["feh_sigma"]:
        raise ValueError("Metallicity posterior was created with different uncertainty columns.")
    mlr = ThreeKnotMetallicityMLR(
        mass_surface, quadrature_nodes=args.velocity_quadrature_nodes
    )
    mlr.set_data(
        row_indices=arrays["row_indices"],
        u=arrays["u"],
        u_sigma=arrays["u_sigma"],
        absg=arrays["absg"],
        metallicity_grid=posterior,
    )
    print("Running stage-two dynamical MLR correction...")
    sampler = mlr.run_mcmc(
        num_warmup=args.warmup,
        num_samples=args.samples,
        num_chains=args.chains,
        seed=args.seed + 1,
        target_accept_prob=args.target_accept,
    )
    sampler.print_summary()
    grid = mlr.correction_grid()
    np.savez_compressed(output_dir / "mlr_correction_grid.npz", **grid)
    save_grouped_mcmc(sampler, output_dir / "mlr_mcmc.npz")
    save_posterior_summary(mlr.posterior_samples, output_dir / "mlr_summary.csv")
    save_sampler_diagnostics(sampler, output_dir / "mlr_diagnostics.json")
    save_mlr_grid_csv(grid, output_dir / "mlr_correction_grid.csv")
    plot_mlr_grid(grid, output_dir)
    monotonicity = {
        "cell_violation_fraction": float(grid["monotone_cell_violation_fraction"]),
        "draw_violation_fraction": float(grid["monotone_draw_violation_fraction"]),
        "definition": "fraction with increasing mass between adjacent M_G grid points",
    }
    (output_dir / "mlr_monotonicity.json").write_text(
        json.dumps(monotonicity, indent=2), encoding="utf-8"
    )
    if truth is not None:
        report = recovery_report(mlr.posterior_samples, truth, ["f0_knots", "fz_knots"])
        (output_dir / "mlr_recovery.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("calibration", "mlr", "all"), default="all")
    parser.add_argument(
        "--data",
        type=Path,
        default=REPO_ROOT / "data" / "jd_msms_single_bic_1kpc_filtered_cmdcut_cutb_jcaps_err_mhcal_good.fits",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "results" / "hierarchical_metallicity_minimal",
    )
    parser.add_argument("--quick", action="store_true", help="Use 2,000 systems, 200 warmup, 300 draws, and one chain.")
    parser.add_argument("--mock", action="store_true", help="Use a generated recovery data set instead of the real FITS table.")
    parser.add_argument("--mock-systems", type=int, default=300)
    parser.add_argument("--max-systems", type=int, default=None)
    parser.add_argument("--warmup", type=int, default=None)
    parser.add_argument("--samples", type=int, default=None)
    parser.add_argument("--chains", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20260819)
    parser.add_argument("--posterior-grid-draws", type=int, default=128)
    parser.add_argument("--velocity-quadrature-nodes", type=int, default=64)
    parser.add_argument("--target-accept", type=float, default=0.9)
    args = parser.parse_args()
    if args.quick:
        if args.max_systems is None:
            args.max_systems = 2000
        args.warmup = 200 if args.warmup is None else args.warmup
        args.samples = 300 if args.samples is None else args.samples
        args.chains = 1 if args.chains is None else args.chains
    else:
        args.warmup = 1000 if args.warmup is None else args.warmup
        args.samples = 1000 if args.samples is None else args.samples
        args.chains = 4 if args.chains is None else args.chains
    if min(args.warmup, args.samples, args.chains, args.posterior_grid_draws) < 1:
        parser.error("warmup, samples, chains, and posterior-grid-draws must be positive.")
    return args


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    mass_surface, color_surface = build_surfaces()
    truth = None
    if args.mock:
        arrays, truth = simulate_mock_dataset(
            color_surface, mass_surface, n_systems=args.mock_systems, seed=args.seed
        )
        (args.output_dir / "mock_truth.json").write_text(
            json.dumps(truth, indent=2), encoding="utf-8"
        )
    else:
        arrays = load_real_data(args.data)

    posterior_path = args.output_dir / "latent_metallicity_weights.npz"
    if args.stage in {"calibration", "all"}:
        arrays = filter_data(
            arrays, max_systems=args.max_systems, seed=args.seed
        )
        posterior = run_calibration(
            arrays, color_surface, args.output_dir, args, truth
        )
    else:
        posterior = MetallicityPosteriorGrid.load(posterior_path)
        arrays = filter_data(
            arrays,
            max_systems=None,
            seed=args.seed,
            fixed_rows=posterior.row_indices,
        )

    if args.stage in {"mlr", "all"}:
        run_mlr(arrays, posterior, mass_surface, args.output_dir, args, truth)
    print(f"Done. Outputs: {args.output_dir}")


if __name__ == "__main__":
    main()
