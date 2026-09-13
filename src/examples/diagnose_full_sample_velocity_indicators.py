#!/usr/bin/env python3
"""Directly diagnose observed velocity tails versus Gaia quality indicators.

This is deliberately independent of the dynamical MLR fit: it uses only the
full stage-one row alignment, observed ``u``, the stage-one latent-Z median,
and the PARSEC mass surface.  No MLR posterior or outlier responsibility is
read by this script.
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
from astropy.table import Table
from scipy.stats import spearmanr


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from binary_masses.differencepoly_feh import IsochroneMassSurfaceModel  # noqa: E402
from binary_masses.hierarchical_metallicity import MetallicityPosteriorGrid  # noqa: E402


DEFAULT_POSTERIOR = REPO_ROOT / "results" / "hierarchical_metallicity_t7_teff4730" / "latent_metallicity_weights.npz"
DEFAULT_DATA = REPO_ROOT / "data" / "jd_msms_single_bic_1kpc_filtered_cmdcut_cutb_jcaps_err_mhcal_good.fits"
DEFAULT_OUTPUT = REPO_ROOT / "results" / "full_sample_velocity_indicators"


def paired_max(table, first, second):
    return np.maximum(np.asarray(table[first], float), np.asarray(table[second], float))


def paired_min(table, first, second):
    return np.minimum(np.asarray(table[first], float), np.asarray(table[second], float))


def quantile_groups(values, n_bins=10):
    """Return quantile groups without splitting identical indicator values."""
    values = np.asarray(values, float)
    finite = np.isfinite(values)
    labels = np.full(values.shape, -1, dtype=int)
    edges = np.unique(
        np.quantile(values[finite], np.linspace(0.0, 1.0, n_bins + 1))
    )
    if edges.size < 2:
        return labels, []
    inner_edges = edges[1:-1]
    labels[finite] = np.searchsorted(inner_edges, values[finite], side="left")
    group_labels = []
    for group in range(edges.size - 1):
        low, high = edges[group], edges[group + 1]
        group_labels.append(f"{low:.3g}–{high:.3g}")
    return labels, group_labels


def indicator_groups(name, values):
    """Use interpretable tied groups for discrete indicators, quantiles otherwise."""
    values = np.asarray(values, float)
    finite = np.isfinite(values)
    unique, counts = np.unique(values[finite], return_counts=True)
    if unique.size < 2 or np.max(counts) / np.sum(finite) > 0.98:
        return np.full(values.shape, -1, dtype=int), [], "insufficient_dynamic_range"
    if name == "max_ipd_frac_multi_peak":
        labels = np.full(values.shape, -1, dtype=int)
        labels[finite & (values == 0)] = 0
        labels[finite & (values == 1)] = 1
        labels[finite & (values >= 2)] = 2
        return labels, ["0", "1", ">=2"], "discrete_0_1_ge2"
    labels, group_labels = quantile_groups(values)
    return labels, group_labels, "quantile_groups_preserving_ties"


def bootstrap_quantiles(values, rng, n_boot=500):
    values = np.asarray(values, float)
    if values.size < 2:
        return np.full(3, np.nan), np.full(3, np.nan), np.full(3, np.nan)
    qs = np.array([0.50, 0.90, 0.95])
    observed = np.quantile(values, qs)
    draws = rng.integers(0, values.size, size=(n_boot, values.size))
    boot = np.quantile(values[draws], qs, axis=1)
    lower, upper = np.quantile(boot, [0.025, 0.975], axis=1)
    return observed, lower, upper


def bootstrap_fraction(flags, rng, n_boot=500):
    flags = np.asarray(flags, bool)
    if flags.size < 2:
        return np.nan, np.nan, np.nan
    observed = np.mean(flags)
    draws = rng.integers(0, flags.size, size=(n_boot, flags.size))
    boot = np.mean(flags[draws], axis=1)
    lower, upper = np.quantile(boot, [0.025, 0.975])
    return observed, lower, upper


def conditional_tail_rank(log_u, bright_mg, delta_m, n_mg=5, n_delta=4):
    """Mark the top 10% raw-u systems within 5x4 CMD-composition cells."""
    finite = np.isfinite(log_u) & np.isfinite(bright_mg) & np.isfinite(delta_m)
    mg_edges = np.unique(
        np.quantile(bright_mg[finite], np.linspace(0, 1, n_mg + 1))
    )
    if mg_edges.size < n_mg + 1:
        raise ValueError("Bright-component M_G has too few distinct values.")
    delta_edges = np.quantile(delta_m[finite], np.linspace(0, 1, n_delta + 1))
    delta_edges = np.unique(delta_edges)
    if delta_edges.size < n_delta + 1:
        raise ValueError("CMD magnitude-difference coordinate has too few distinct values.")
    cell = np.full(log_u.shape, -1, dtype=int)
    high = np.zeros(log_u.shape, dtype=bool)
    for i in range(n_mg):
        mg_mask = (bright_mg >= mg_edges[i]) & (bright_mg <= mg_edges[i + 1] if i == n_mg - 1 else bright_mg < mg_edges[i + 1])
        for j in range(n_delta):
            d_mask = (delta_m >= delta_edges[j]) & (delta_m <= delta_edges[j + 1] if j == n_delta - 1 else delta_m < delta_edges[j + 1])
            selected = finite & mg_mask & d_mask
            cell[selected] = i * n_delta + j
            if np.sum(selected) >= 10:
                selected_indices = np.flatnonzero(selected)
                order = selected_indices[np.argsort(log_u[selected], kind="mergesort")]
                n_high = int(np.ceil(0.10 * order.size))
                high[order[-n_high:]] = True
    counts = [int(np.sum(cell == index)) for index in range(n_mg * n_delta)]
    return high, cell, {
        "bright_mg_edges": mg_edges.tolist(),
        "delta_m_edges": delta_edges.tolist(),
        "n_cells": n_mg * n_delta,
        "cell_counts": counts,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--metallicity-posterior", type=Path, default=DEFAULT_POSTERIOR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=20260912)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    posterior = MetallicityPosteriorGrid.load(args.metallicity_posterior)
    table_all = Table.read(args.data)
    if len(posterior.row_indices) != 14876:
        raise ValueError(f"Expected 14,876 stage-one systems, found {len(posterior.row_indices)}.")
    if np.unique(posterior.row_indices).size != len(posterior.row_indices):
        raise ValueError("Posterior row_indices contain duplicates.")
    if np.any(posterior.row_indices < 0) or np.any(posterior.row_indices >= len(table_all)):
        raise ValueError("Posterior row_indices are outside the FITS table.")
    table = table_all[posterior.row_indices]
    required = ["u", "absg1", "absg2", "ruwe1", "ruwe2", "ipd_frac_multi_peak1", "ipd_frac_multi_peak2", "astrometric_excess_noise_sig1", "astrometric_excess_noise_sig2", "R_chance_align", "sep_arcsec"]
    missing = [name for name in required if name not in table.colnames]
    if missing:
        raise ValueError(f"Input FITS is missing required columns: {missing}")

    mass_surface = IsochroneMassSurfaceModel.from_interpolated_mass_data(REPO_ROOT / "data" / "interpolated_mass_data", mass_min=0.05)
    absg = np.column_stack([np.asarray(table["absg1"], float), np.asarray(table["absg2"], float)])
    z_median = np.asarray(posterior.z_quantiles[:, 1], float)
    if np.any(absg < mass_surface.absg_grid_np[0]) or np.any(absg > mass_surface.absg_grid_np[-1]):
        raise ValueError("PARSEC mass evaluation would require M_G extrapolation.")
    if np.any(z_median < mass_surface.mh_grid_np[0]) or np.any(z_median > mass_surface.mh_grid_np[-1]):
        raise ValueError("PARSEC mass evaluation would require metallicity extrapolation.")
    total_mass = mass_surface.mass_from_absg_mh(absg[:, 0], z_median) + mass_surface.mass_from_absg_mh(absg[:, 1], z_median)
    u = np.asarray(table["u"], float)
    normalized_u = u / np.sqrt(total_mass)
    if np.any(~np.isfinite(u)) or np.any(u <= 0):
        raise ValueError("Observed u must be finite and positive for all selected systems.")
    if np.any(~np.isfinite(total_mass)) or np.any(total_mass <= 0):
        raise ValueError("PARSEC total masses must be finite and positive.")
    log_u = np.log(u)
    bright_mg = np.min(absg, axis=1)
    delta_m = np.abs(absg[:, 0] - absg[:, 1])
    high_tail, cmd_cell, cmd_meta = conditional_tail_rank(
        log_u, bright_mg, delta_m
    )

    indicators = {
        "max_ruwe": (paired_max(table, "ruwe1", "ruwe2"), "Max RUWE"),
        "max_ipd_frac_multi_peak": (paired_max(table, "ipd_frac_multi_peak1", "ipd_frac_multi_peak2"), "Max IPD multi-peak"),
        "max_astrometric_excess_noise_sig": (paired_max(table, "astrometric_excess_noise_sig1", "astrometric_excess_noise_sig2"), "Max AEN significance"),
        "R_chance_align": (np.asarray(table["R_chance_align"], float), "Chance-alignment probability (context)"),
        "sep_arcsec": (np.asarray(table["sep_arcsec"], float), "Angular separation (context)"),
    }
    if "fidelity_v2_1" in table.colnames and "fidelity_v2_2" in table.colnames:
        indicators["min_fidelity_v2"] = (paired_min(table, "fidelity_v2_1", "fidelity_v2_2"), "Minimum fidelity_v2")
    elif "fidelity_v2" in table.colnames:
        indicators["fidelity_v2"] = (np.asarray(table["fidelity_v2"], float), "fidelity_v2")

    indicator_notes = {}
    plotted_indicators = {}
    for name, (values, label) in indicators.items():
        labels, group_labels, grouping = indicator_groups(name, values)
        indicator_notes[name] = {
            "grouping": grouping,
            "n_unique": int(np.unique(values[np.isfinite(values)]).size),
            "group_labels": group_labels,
        }
        if group_labels:
            plotted_indicators[name] = (values, label, labels, group_labels)

    rng = np.random.default_rng(args.seed)
    quantile_rows, tail_rows = [], []
    for name, (values, label, labels, group_labels) in plotted_indicators.items():
        rho, p_value = spearmanr(values[np.isfinite(values)], normalized_u[np.isfinite(values)])
        for group_index, group_label in enumerate(group_labels):
            selected = labels == group_index
            if not np.any(selected):
                continue
            q, qlow, qhigh = bootstrap_quantiles(normalized_u[selected], rng)
            tail_selected = selected & (cmd_cell >= 0)
            tail_p, tail_low, tail_high = bootstrap_fraction(
                high_tail[tail_selected], rng
            )
            common = {
                "indicator": name,
                "label": label,
                "group": group_index + 1,
                "group_label": group_label,
                "n": int(np.sum(selected)),
                "median_indicator": float(np.nanmedian(values[selected])),
            }
            quantile_rows.append({
                **common,
                "q50_normalized_u": float(q[0]),
                "q50_ci_low": float(qlow[0]),
                "q50_ci_high": float(qhigh[0]),
                "q90_normalized_u": float(q[1]),
                "q90_ci_low": float(qlow[1]),
                "q90_ci_high": float(qhigh[1]),
                "q95_normalized_u": float(q[2]),
                "q95_ci_low": float(qlow[2]),
                "q95_ci_high": float(qhigh[2]),
                "spearman_rho": float(rho),
                "spearman_p_value": float(p_value),
            })
            tail_rows.append({
                **common,
                "tail_n_cmd": int(np.sum(tail_selected)),
                "tail_k_cmd": int(np.sum(high_tail[tail_selected])),
                "conditional_high_tail_fraction": float(tail_p),
                "tail_ci_low": float(tail_low),
                "tail_ci_high": float(tail_high),
            })

    with (args.output_dir / "velocity_indicator_summary.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = list(quantile_rows[0]) + [f for f in tail_rows[0] if f not in quantile_rows[0]]
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        for qrow, trow in zip(quantile_rows, tail_rows): writer.writerow({**qrow, **{k: v for k, v in trow.items() if k not in qrow}})

    ncols = 3
    nrows = int(np.ceil(len(plotted_indicators) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(14.5, 4.4 * nrows), squeeze=False)
    for panel, (axis, (name, (_, label, _, group_labels))) in enumerate(zip(axes.ravel(), plotted_indicators.items())):
        rows = [r for r in quantile_rows if r["indicator"] == name]
        x = np.arange(1, len(rows) + 1)
        for key, low_key, high_key, color in [
            ("q50_normalized_u", "q50_ci_low", "q50_ci_high", "tab:blue"),
            ("q90_normalized_u", "q90_ci_low", "q90_ci_high", "tab:orange"),
            ("q95_normalized_u", "q95_ci_low", "q95_ci_high", "tab:red"),
        ]:
            y = np.asarray([r[key] for r in rows])
            low = np.asarray([r[low_key] for r in rows])
            high = np.asarray([r[high_key] for r in rows])
            axis.errorbar(x, y, yerr=np.vstack([y - low, high - y]), marker="o", ms=3, lw=1, capsize=2, label=key.replace("_normalized_u", "").upper(), color=color)
        axis.set(xlabel="Indicator group (low to high)", title=label)
        if panel % ncols == 0:
            axis.set_ylabel(r"$u/\sqrt{M_{\rm PARSEC,tot}}$")
        tick_labels = [f"{text}\n(n={row['n']})" for text, row in zip(group_labels, rows)]
        axis.set_xticks(x, tick_labels, rotation=35, ha="right", fontsize=7)
        axis.grid(alpha=0.2)
    for axis in axes.ravel()[len(plotted_indicators):]:
        axis.set_visible(False)
    axes.ravel()[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Observed normalized-velocity quantiles versus quality indicators")
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.12, top=0.91, wspace=0.25, hspace=0.58)
    fig.savefig(args.output_dir / "velocity_quantiles_vs_indicators.png", dpi=180); plt.close(fig)

    fig, axes = plt.subplots(nrows, ncols, figsize=(14.5, 4.4 * nrows), squeeze=False)
    for panel, (axis, (name, (_, label, _, group_labels))) in enumerate(zip(axes.ravel(), plotted_indicators.items())):
        rows = [r for r in tail_rows if r["indicator"] == name]
        x = np.arange(1, len(rows) + 1)
        y = np.asarray([r["conditional_high_tail_fraction"] for r in rows])
        low = np.asarray([r["tail_ci_low"] for r in rows])
        high = np.asarray([r["tail_ci_high"] for r in rows])
        axis.errorbar(x, y, yerr=np.vstack([y - low, high - y]), marker="o", ms=3, lw=1, capsize=2, color="tab:purple")
        axis.axhline(0.10, color="0.45", ls="--", lw=1)
        axis.set(xlabel="Indicator group (low to high)", title=label, ylim=(0, 0.50))
        if panel % ncols == 0:
            axis.set_ylabel("Within-CMD-cell top-10% fraction")
        tick_labels = [f"{text}\n(n={row['n']})" for text, row in zip(group_labels, rows)]
        axis.set_xticks(x, tick_labels, rotation=35, ha="right", fontsize=7)
        axis.grid(alpha=0.2)
    for axis in axes.ravel()[len(plotted_indicators):]:
        axis.set_visible(False)
    fig.suptitle("Conditional high-velocity tail versus quality indicators")
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.12, top=0.91, wspace=0.25, hspace=0.58)
    fig.savefig(args.output_dir / "conditional_velocity_tail_vs_indicators.png", dpi=180); plt.close(fig)

    metadata = {
        "n_systems": int(len(table)), "expected_n_systems": 14876, "row_alignment": "FITS rows indexed by latent_metallicity_weights.npz row_indices", "posterior_path": str(args.metallicity_posterior), "data_path": str(args.data), "normalized_velocity": "u / sqrt(total uncorrected PARSEC mass at stage-one posterior median Z)", "cmd_tail_definition": "top 10% of raw log(u) within 5 bright-component-MG quantile bins x 4 absolute-delta-MG quantile bins", "cmd_cells": cmd_meta, "indicator_missingness": {name: int(np.sum(~np.isfinite(vals))) for name, (vals, _) in indicators.items()}, "indicator_grouping": indicator_notes, "excluded_inputs": ["mlr_mcmc.npz", "f_Z", "f_outlier", "dynamics_likelihood_lookup.npz", "dpm_over_error"], "seed": args.seed,
    }
    (args.output_dir / "velocity_indicator_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
