#!/usr/bin/env python3
"""Diagnose dynamical outlier responsibilities in the both-warm CMD sample."""

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
from astropy.table import Table
from scipy.special import logsumexp
from scipy.stats import ks_2samp, spearmanr


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from binary_masses.hierarchical_metallicity import (  # noqa: E402
    DynamicsLikelihoodLookup,
    IsochroneColorSurfaceModel,
    IsochroneMassSurfaceModel,
    MetallicityPosteriorGrid,
)


DEFAULT_RESULT = (
    REPO_ROOT
    / "results"
    / "hierarchical_metallicity_t7_teff4730_bothcmd1726_mg3p5_6p34"
)
DEFAULT_POSTERIOR = (
    REPO_ROOT
    / "results"
    / "hierarchical_metallicity_t7_teff4730"
    / "latent_metallicity_weights.npz"
)
DEFAULT_LOOKUP = (
    REPO_ROOT
    / "results"
    / "hierarchical_metallicity_minimal_lookup"
    / "dynamics_likelihood_lookup.npz"
)
DEFAULT_DATA = (
    REPO_ROOT
    / "data"
    / "jd_msms_single_bic_1kpc_filtered_cmdcut_cutb_jcaps_err_mhcal_good.fits"
)


def interpolate_rows(grid, table, values):
    """Vectorized row-wise linear interpolation."""
    upper = np.searchsorted(grid, values, side="right")
    upper = np.clip(upper, 1, len(grid) - 1)
    lower = upper - 1
    x0 = grid[lower]
    x1 = grid[upper]
    row = np.arange(values.shape[0])[:, None]
    y0 = table[row, lower]
    y1 = table[row, upper]
    weight = (values - x0) / (x1 - x0)
    result = y0 + weight * (y1 - y0)
    if np.any(values < grid[0]) or np.any(values > grid[-1]):
        raise ValueError("A posterior mass evaluation lies outside the lookup grid.")
    return result


def posterior_bad_responsibility(
    *,
    absg,
    z_grid,
    z_probabilities,
    mass_surface,
    lookup,
    f0_samples,
    fz_samples,
    f_outlier_samples,
    mg_knots,
):
    """Average p(bad | dynamics, Z posterior, MLR draw) over MLR draws."""
    baseline1 = mass_surface.mass_from_absg_mh(
        absg[:, 0, None], z_grid[None, :]
    )
    baseline2 = mass_surface.mass_from_absg_mh(
        absg[:, 1, None], z_grid[None, :]
    )
    log_z_probability = np.log(np.maximum(z_probabilities, 1e-300))
    mean_responsibility = np.zeros(absg.shape[0], dtype=np.float64)

    for f0, fz, f_outlier in zip(f0_samples, fz_samples, f_outlier_samples):
        f01 = np.interp(absg[:, 0], mg_knots, f0)[:, None]
        f02 = np.interp(absg[:, 1], mg_knots, f0)[:, None]
        fz1 = np.interp(absg[:, 0], mg_knots, fz)[:, None]
        fz2 = np.interp(absg[:, 1], mg_knots, fz)[:, None]
        mass1 = baseline1 * 10.0 ** (f01 + z_grid[None, :] * fz1)
        mass2 = baseline2 * 10.0 ** (f02 + z_grid[None, :] * fz2)
        sqrt_total_mass = np.sqrt(mass1 + mass2)
        log_good = interpolate_rows(
            lookup.sqrt_mtot_grid, lookup.log_good, sqrt_total_mass
        )
        log_bad = interpolate_rows(
            lookup.sqrt_mtot_grid, lookup.log_bad, sqrt_total_mass
        )
        log_good_joint = (
            log_z_probability + np.log1p(-f_outlier) + log_good
        )
        log_bad_joint = log_z_probability + np.log(f_outlier) + log_bad
        log_evidence = logsumexp(
            np.stack(
                [
                    logsumexp(log_good_joint, axis=1),
                    logsumexp(log_bad_joint, axis=1),
                ]
            ),
            axis=0,
        )
        mean_responsibility += np.exp(
            logsumexp(log_bad_joint, axis=1) - log_evidence
        )

    return mean_responsibility / len(f_outlier_samples)


def quantile_bin_summary(x, y, bins=8):
    finite = np.isfinite(x) & np.isfinite(y)
    x = np.asarray(x[finite], dtype=float)
    y = np.asarray(y[finite], dtype=float)
    edges = np.unique(np.quantile(x, np.linspace(0.0, 1.0, bins + 1)))
    centers, means, errors, counts = [], [], [], []
    for index, (low, high) in enumerate(zip(edges[:-1], edges[1:])):
        selected = (x >= low) & (x < high)
        if index == len(edges) - 2:
            selected = (x >= low) & (x <= high)
        if not np.any(selected):
            continue
        centers.append(np.median(x[selected]))
        means.append(np.mean(y[selected]))
        errors.append(np.std(y[selected], ddof=1) / np.sqrt(np.sum(selected)))
        counts.append(np.sum(selected))
    return tuple(np.asarray(value) for value in (centers, means, errors, counts))


def empirical_cdf(values):
    values = np.sort(np.asarray(values, dtype=float))
    return values, np.arange(1, len(values) + 1) / len(values)


def paired_max(table, first, second):
    return np.maximum(np.asarray(table[first], float), np.asarray(table[second], float))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--metallicity-posterior", type=Path, default=DEFAULT_POSTERIOR)
    parser.add_argument("--dynamics-lookup", type=Path, default=DEFAULT_LOOKUP)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--posterior-draws", type=int, default=128)
    args = parser.parse_args()

    model_metadata = json.loads(
        (args.result_dir / "mlr_model.json").read_text(encoding="utf-8")
    )
    if int(model_metadata["n_systems"]) != 1726:
        raise ValueError("This diagnostic expects the 1,726-system both-warm fit.")
    mg_knots = np.asarray(model_metadata["mg_knots"], dtype=float)

    posterior = MetallicityPosteriorGrid.load(args.metallicity_posterior)
    table_all = Table.read(args.data)
    table = table_all[posterior.row_indices]
    absg_all = np.column_stack(
        [np.asarray(table["absg1"], float), np.asarray(table["absg2"], float)]
    )
    color_surface = IsochroneColorSurfaceModel.from_parsec_csv(
        REPO_ROOT / "data" / "PARSEC_logAge_6to10_0p5_MH_n1to0p6_0p2.csv",
        absg_grid=np.linspace(3.5, 13.5, 500),
    )
    parsec_teff = color_surface.teff_from_absg_mh(
        absg_all[:, :, None], posterior.z_grid[None, None, :]
    )
    component_warm = np.min(parsec_teff, axis=2) >= 4730.0
    keep = np.all(component_warm, axis=1)
    if int(np.sum(keep)) != 1726:
        raise ValueError(f"Expected 1,726 both-warm systems; found {np.sum(keep)}.")

    table = table[keep]
    absg = absg_all[keep]
    z_probabilities = posterior.probabilities[keep]
    z_quantiles = posterior.z_quantiles[keep]
    z_median = z_quantiles[:, 1]

    lookup_all = DynamicsLikelihoodLookup.load(args.dynamics_lookup)
    if not np.array_equal(lookup_all.row_indices, posterior.row_indices):
        raise ValueError("Full lookup and metallicity posterior rows do not match.")
    lookup = DynamicsLikelihoodLookup(
        row_indices=lookup_all.row_indices[keep],
        sqrt_mtot_grid=lookup_all.sqrt_mtot_grid,
        log_good=lookup_all.log_good[keep],
        log_bad=lookup_all.log_bad[keep],
        metadata=lookup_all.metadata,
    )

    with np.load(args.result_dir / "mlr_mcmc.npz", allow_pickle=False) as saved:
        f0_all = np.asarray(saved["posterior__f0_knots"]).reshape(-1, 3)
        fz_all = np.asarray(saved["posterior__fz_knots"]).reshape(-1, 3)
        f_outlier_all = np.asarray(saved["posterior__f_outlier"]).reshape(-1)
    draw_count = min(int(args.posterior_draws), len(f_outlier_all))
    if draw_count < 1:
        raise ValueError("--posterior-draws must be positive.")
    draw_indices = np.linspace(0, len(f_outlier_all) - 1, draw_count, dtype=int)

    mass_surface = IsochroneMassSurfaceModel.from_interpolated_mass_data(
        REPO_ROOT / "data" / "interpolated_mass_data", mass_min=0.05
    )
    responsibility = posterior_bad_responsibility(
        absg=absg,
        z_grid=posterior.z_grid,
        z_probabilities=z_probabilities,
        mass_surface=mass_surface,
        lookup=lookup,
        f0_samples=f0_all[draw_indices],
        fz_samples=fz_all[draw_indices],
        f_outlier_samples=f_outlier_all[draw_indices],
        mg_knots=mg_knots,
    )

    u = np.asarray(table["u"], float)
    parsec_total_mass = mass_surface.mass_from_absg_mh(
        absg[:, 0], z_median
    ) + mass_surface.mass_from_absg_mh(absg[:, 1], z_median)
    normalized_u = u / np.sqrt(parsec_total_mass)

    indicator_specs = [
        (
            "R_chance_align",
            np.asarray(table["R_chance_align"], float),
            "Chance-alignment probability",
            "log",
        ),
        ("max_ruwe", paired_max(table, "ruwe1", "ruwe2"), "Maximum RUWE", "log"),
        (
            "max_ipd_multi_peak",
            paired_max(table, "ipd_frac_multi_peak1", "ipd_frac_multi_peak2"),
            "Maximum IPD multi-peak fraction",
            "symlog",
        ),
        (
            "max_excess_noise_sig",
            paired_max(
                table, "astrometric_excess_noise_sig1", "astrometric_excess_noise_sig2"
            ),
            "Maximum excess-noise significance",
            "symlog",
        ),
        (
            "dpm_over_error",
            np.asarray(table["dpm_over_error"], float),
            "Relative-motion S/N",
            "log",
        ),
        (
            "sep_arcsec",
            np.asarray(table["sep_arcsec"], float),
            "Angular separation (arcsec)",
            "log",
        ),
    ]

    colors = ("tab:blue", "tab:green", "tab:red")
    z_edges = np.quantile(z_median, [0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0])
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for index, (low, high, color) in enumerate(
        zip(z_edges[:-1], z_edges[1:], colors)
    ):
        selected = (z_median >= low) & (z_median < high)
        if index == 2:
            selected = (z_median >= low) & (z_median <= high)
        label = f"{low:+.2f} <= [M/H] < {high:+.2f}"
        raw_x, raw_y = empirical_cdf(u[selected])
        norm_x, norm_y = empirical_cdf(normalized_u[selected])
        axes[0, 0].plot(raw_x, raw_y, color=color, label=label)
        axes[0, 1].plot(norm_x, norm_y, color=color, label=label)
    axes[0, 0].set(
        xscale="log", xlabel=r"Observed $u$", ylabel="Empirical cumulative fraction"
    )
    axes[0, 1].set(
        xscale="log",
        xlabel=r"$u/\sqrt{M_{\rm PARSEC,tot}}$",
        ylabel="Empirical cumulative fraction",
    )
    axes[0, 0].legend(frameon=False, fontsize=9)
    scatter = axes[1, 0].scatter(
        z_median,
        normalized_u,
        c=responsibility,
        s=10,
        alpha=0.55,
        cmap="magma",
        vmin=0.0,
        vmax=1.0,
    )
    axes[1, 0].set(
        xlabel="Posterior median [M/H]",
        ylabel=r"$u/\sqrt{M_{\rm PARSEC,tot}}$",
        yscale="log",
    )
    colorbar = fig.colorbar(scatter, ax=axes[1, 0])
    colorbar.set_label("Posterior bad-component responsibility")
    center, mean, error, count = quantile_bin_summary(
        z_median, responsibility, bins=8
    )
    axes[1, 1].errorbar(center, mean, yerr=error, fmt="o-", capsize=3)
    axes[1, 1].axhline(
        np.mean(f_outlier_all), color="0.4", linestyle="--", label="Global mixture weight"
    )
    axes[1, 1].set(
        xlabel="Posterior median [M/H]",
        ylabel="Mean posterior bad-component responsibility",
        ylim=(-0.02, 1.02),
    )
    axes[1, 1].legend(frameon=False)
    fig.suptitle("Dynamical velocity and mixture responsibility by metallicity")
    fig.tight_layout()
    fig.savefig(args.result_dir / "outlier_velocity_metallicity.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    rows = []
    for axis, (name, values, label, x_scale) in zip(axes.ravel(), indicator_specs):
        finite = np.isfinite(values) & np.isfinite(responsibility)
        rho, p_value = spearmanr(values[finite], responsibility[finite])
        z_rho, z_p_value = spearmanr(values[finite], z_median[finite])
        center, mean, error, count = quantile_bin_summary(
            values, responsibility, bins=8
        )
        axis.scatter(values[finite], responsibility[finite], s=4, alpha=0.08, color="0.35")
        axis.errorbar(center, mean, yerr=error, fmt="o-", color="tab:blue", capsize=3)
        if x_scale == "log" and np.all(values[finite] > 0):
            axis.set_xscale("log")
        elif x_scale == "symlog":
            positive = values[finite & (values > 0)]
            linear_threshold = 0.1 if positive.size == 0 else max(0.1, np.percentile(positive, 10))
            axis.set_xscale("symlog", linthresh=linear_threshold)
        axis.set(xlabel=label, ylabel="Mean bad responsibility", ylim=(-0.02, 1.02))
        axis.set_title(fr"Spearman $\rho={rho:+.2f}$")
        rows.append(
            {
                "indicator": name,
                "n_finite": int(np.sum(finite)),
                "spearman_rho": float(rho),
                "spearman_p_value": float(p_value),
                "spearman_with_metallicity_rho": float(z_rho),
                "spearman_with_metallicity_p_value": float(z_p_value),
                "lowest_bin_mean_responsibility": float(mean[0]),
                "highest_bin_mean_responsibility": float(mean[-1]),
            }
        )
    fig.suptitle("Posterior bad-component responsibility versus quality indicators")
    fig.tight_layout()
    fig.savefig(args.result_dir / "outlier_quality_indicators.png", dpi=180)
    plt.close(fig)

    with (args.result_dir / "outlier_indicator_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    np.savez_compressed(
        args.result_dir / "outlier_diagnostic_arrays.npz",
        row_indices=posterior.row_indices[keep],
        z_median=z_median,
        u=u,
        parsec_total_mass=parsec_total_mass,
        normalized_u=normalized_u,
        bad_responsibility=responsibility,
        **{name: values for name, values, _, _ in indicator_specs},
    )
    low_z = z_median <= z_edges[1]
    high_z = z_median >= z_edges[2]
    z_bad_rho, z_bad_p = spearmanr(z_median, responsibility)
    z_normu_rho, z_normu_p = spearmanr(z_median, normalized_u)
    raw_u_ks = ks_2samp(u[low_z], u[high_z])
    normalized_u_ks = ks_2samp(normalized_u[low_z], normalized_u[high_z])
    summary = {
        "n_systems": int(len(z_median)),
        "posterior_draws_used": int(draw_count),
        "mean_global_outlier_weight": float(np.mean(f_outlier_all)),
        "mean_posterior_bad_responsibility": float(np.mean(responsibility)),
        "fraction_bad_responsibility_above_half": float(np.mean(responsibility > 0.5)),
        "metallicity_diagnostics": {
            "spearman_z_vs_bad_responsibility": float(z_bad_rho),
            "spearman_z_vs_bad_responsibility_p_value": float(z_bad_p),
            "spearman_z_vs_parsec_normalized_u": float(z_normu_rho),
            "spearman_z_vs_parsec_normalized_u_p_value": float(z_normu_p),
            "low_z_tertile_mean_bad_responsibility": float(np.mean(responsibility[low_z])),
            "high_z_tertile_mean_bad_responsibility": float(np.mean(responsibility[high_z])),
            "low_vs_high_z_raw_u_ks_statistic": float(raw_u_ks.statistic),
            "low_vs_high_z_raw_u_ks_p_value": float(raw_u_ks.pvalue),
            "low_vs_high_z_normalized_u_ks_statistic": float(normalized_u_ks.statistic),
            "low_vs_high_z_normalized_u_ks_p_value": float(normalized_u_ks.pvalue),
        },
        "indicator_correlations": rows,
    }
    (args.result_dir / "outlier_diagnostic_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
