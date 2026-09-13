#!/usr/bin/env python3
"""Direct RUWE--metallicity diagnostic for the complete stage-one sample."""

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
from binary_masses.hierarchical_metallicity import MetallicityPosteriorGrid  # noqa: E402

DEFAULT_POSTERIOR = REPO_ROOT / "results" / "hierarchical_metallicity_t7_teff4730" / "latent_metallicity_weights.npz"
DEFAULT_DATA = REPO_ROOT / "data" / "jd_msms_single_bic_1kpc_filtered_cmdcut_cutb_jcaps_err_mhcal_good.fits"
DEFAULT_OUTPUT = REPO_ROOT / "results" / "full_sample_ruwe_metallicity"
N_SYSTEMS = 14876
N_BINS = 10
N_BOOT = 400


def rank_bins(values, n_bins=N_BINS):
    finite = np.isfinite(values)
    labels = np.full(len(values), -1, dtype=int)
    positions = np.flatnonzero(finite)[np.argsort(values[finite], kind="mergesort")]
    labels[positions] = np.minimum(n_bins - 1, np.arange(len(positions)) * n_bins // len(positions))
    return labels


def cmd_high_tail(ruwe, absg):
    """Flag the top 10% RUWE within 5 bright-MG x 4 delta-MG cells."""
    bright_mg = np.min(absg, axis=1)
    delta_mg = np.abs(absg[:, 0] - absg[:, 1])
    finite = np.isfinite(ruwe) & np.isfinite(bright_mg) & np.isfinite(delta_mg)
    mg_edges = np.quantile(bright_mg[finite], np.linspace(0, 1, 6))
    dm_edges = np.quantile(delta_mg[finite], np.linspace(0, 1, 5))
    high = np.zeros(len(ruwe), dtype=bool)
    cell = np.full(len(ruwe), -1, dtype=int)
    for i in range(5):
        im = (bright_mg >= mg_edges[i]) & (bright_mg <= mg_edges[i + 1] if i == 4 else bright_mg < mg_edges[i + 1])
        for j in range(4):
            idm = (delta_mg >= dm_edges[j]) & (delta_mg <= dm_edges[j + 1] if j == 3 else delta_mg < dm_edges[j + 1])
            selected = finite & im & idm
            cell[selected] = 4 * i + j
            if np.sum(selected) >= 10:
                high[selected] = ruwe[selected] >= np.quantile(ruwe[selected], 0.90)
    return high, cell, {
        "n_cells": 20,
        "bright_mg_edges": mg_edges.tolist(),
        "delta_mg_edges": dm_edges.tolist(),
        "cell_counts": [int(np.sum(cell == index)) for index in range(20)],
    }


def draw_z(rng, z_grid, probabilities, indices):
    probs = probabilities[indices]
    cdf = np.cumsum(probs, axis=1)
    cdf[:, -1] = 1.0
    return z_grid[np.sum(rng.random(len(indices))[:, None] > cdf, axis=1)]


def summarize(z, ruwe, high_cmd, cell, edges):
    labels = np.searchsorted(edges, z, side="right") - 1
    labels = np.clip(labels, 0, N_BINS - 1)
    rows = []
    for b in range(N_BINS):
        selected = (labels == b) & np.isfinite(ruwe)
        controlled = selected & (cell >= 0)
        values = ruwe[selected]
        rows.append({
            "z_bin": b + 1,
            "z_low": float(edges[b]), "z_high": float(edges[b + 1]),
            "n": int(np.sum(selected)), "z_median": float(np.median(z[selected])),
            "ruwe_median": float(np.quantile(values, 0.50)),
            "ruwe_q90": float(np.quantile(values, 0.90)), "ruwe_q95": float(np.quantile(values, 0.95)),
            "fraction_ruwe_gt_1p4": float(np.mean(values > 1.4)),
            "n_cmd_controlled": int(np.sum(controlled)),
            "fraction_cmd_high_ruwe": float(np.mean(high_cmd[controlled])) if np.any(controlled) else np.nan,
        })
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--metallicity-posterior", type=Path, default=DEFAULT_POSTERIOR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=20260912)
    parser.add_argument("--bootstrap", type=int, default=N_BOOT)
    args = parser.parse_args()
    if args.bootstrap < 300 or args.bootstrap > 500:
        raise ValueError("--bootstrap must be between 300 and 500.")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    posterior = MetallicityPosteriorGrid.load(args.metallicity_posterior)
    table_all = Table.read(args.data)
    if len(posterior.row_indices) != N_SYSTEMS:
        raise ValueError(f"Expected {N_SYSTEMS} posterior systems, found {len(posterior.row_indices)}.")
    if np.unique(posterior.row_indices).size != N_SYSTEMS or np.any(posterior.row_indices < 0) or np.any(posterior.row_indices >= len(table_all)):
        raise ValueError("latent posterior row_indices are not a unique valid FITS alignment.")
    table = table_all[posterior.row_indices]
    required = ["ruwe1", "ruwe2", "absg1", "absg2"]
    missing = [name for name in required if name not in table.colnames]
    if missing:
        raise ValueError(f"Input FITS is missing required columns: {missing}")

    ruwe1, ruwe2 = np.asarray(table["ruwe1"], float), np.asarray(table["ruwe2"], float)
    ruwe = np.maximum(ruwe1, ruwe2)
    absg = np.column_stack([np.asarray(table["absg1"], float), np.asarray(table["absg2"], float)])
    z_median = np.asarray(posterior.z_quantiles[:, 1], float)
    finite = np.isfinite(ruwe) & np.isfinite(z_median) & np.all(np.isfinite(absg), axis=1) & (ruwe > 0)
    if np.sum(finite) != N_SYSTEMS:
        raise ValueError(f"Expected all {N_SYSTEMS} systems to have finite RUWE/Z/CMD values; found {np.sum(finite)}.")
    high_cmd, cell, cmd_meta = cmd_high_tail(ruwe, absg)
    z_edges = np.quantile(z_median, np.linspace(0, 1, N_BINS + 1))
    observed_rows = summarize(z_median, ruwe, high_cmd, cell, z_edges)

    rng = np.random.default_rng(args.seed)
    bootstrap_rows = [[[] for _ in range(N_BINS)] for _ in range(5)]
    bootstrap_rho = []
    # Per replicate: system bootstrap plus a posterior-grid draw of every Z.
    for _ in range(args.bootstrap):
        indices = rng.integers(0, N_SYSTEMS, size=N_SYSTEMS)
        z_draw = draw_z(rng, posterior.z_grid, posterior.probabilities, indices)
        boot_edges = np.quantile(z_draw, np.linspace(0, 1, N_BINS + 1))
        rows = summarize(z_draw, ruwe[indices], high_cmd[indices], cell[indices], boot_edges)
        bootstrap_rho.append(spearmanr(z_draw, np.log(ruwe[indices])).statistic)
        for b, row in enumerate(rows):
            bootstrap_rows[0][b].append(row["ruwe_median"])
            bootstrap_rows[1][b].append(row["ruwe_q90"])
            bootstrap_rows[2][b].append(row["ruwe_q95"])
            bootstrap_rows[3][b].append(row["fraction_ruwe_gt_1p4"])
            bootstrap_rows[4][b].append(row["fraction_cmd_high_ruwe"])

    for row_index, row in enumerate(observed_rows):
        for key, values in zip(["ruwe_median", "ruwe_q90", "ruwe_q95", "fraction_ruwe_gt_1p4", "fraction_cmd_high_ruwe"], bootstrap_rows):
            array = np.asarray(values[row_index], float)
            row[f"{key}_ci_low"] = float(np.nanpercentile(array, 2.5))
            row[f"{key}_ci_high"] = float(np.nanpercentile(array, 97.5))

    with (args.output_dir / "ruwe_metallicity_summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(observed_rows[0])); writer.writeheader(); writer.writerows(observed_rows)

    x = np.array([row["z_median"] for row in observed_rows])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for key, label, color in [("ruwe_median", "Median RUWE", "tab:blue"), ("ruwe_q90", "RUWE 90th percentile", "tab:orange"), ("ruwe_q95", "RUWE 95th percentile", "tab:red")]:
        y = np.array([r[key] for r in observed_rows]); lo = np.array([r[f"{key}_ci_low"] for r in observed_rows]); hi = np.array([r[f"{key}_ci_high"] for r in observed_rows])
        axes[0].plot(x, y, "o-", ms=3, lw=1, color=color, label=label); axes[0].fill_between(x, lo, hi, color=color, alpha=0.12)
    axes[0].set(yscale="log", xlabel="Stage-one posterior median [M/H]", ylabel="Maximum RUWE", title="RUWE distribution by metallicity")
    y = np.array([r["fraction_ruwe_gt_1p4"] for r in observed_rows]); lo = np.array([r["fraction_ruwe_gt_1p4_ci_low"] for r in observed_rows]); hi = np.array([r["fraction_ruwe_gt_1p4_ci_high"] for r in observed_rows])
    axes[1].plot(x, y, "o-", color="tab:green", ms=3, lw=1, label="max RUWE > 1.4"); axes[1].fill_between(x, lo, hi, color="tab:green", alpha=0.12)
    y2 = np.array([r["fraction_cmd_high_ruwe"] for r in observed_rows]); lo2 = np.array([r["fraction_cmd_high_ruwe_ci_low"] for r in observed_rows]); hi2 = np.array([r["fraction_cmd_high_ruwe_ci_high"] for r in observed_rows])
    axes[1].plot(x, y2, "s--", color="tab:purple", ms=3, lw=1, label="CMD-controlled top-10% RUWE"); axes[1].fill_between(x, lo2, hi2, color="tab:purple", alpha=0.12)
    axes[1].set(xlabel="Stage-one posterior median [M/H]", ylabel="Fraction", title="High-RUWE incidence by metallicity", ylim=(0, 0.35))
    for axis in axes: axis.grid(alpha=0.2); axis.legend(frameon=False, fontsize=8)
    fig.suptitle("Direct RUWE--metallicity diagnostic (14,876 systems)"); fig.tight_layout(); fig.savefig(args.output_dir / "ruwe_vs_metallicity.png", dpi=180); plt.close(fig)

    faint_source = np.where(ruwe1 >= ruwe2, 1, 2)
    fainter_member = np.where(absg[:, 0] >= absg[:, 1], 1, 2)
    comes_from_fainter = faint_source == fainter_member
    low = z_median <= z_edges[1]; high = z_median >= z_edges[-2]
    rho_draws = np.asarray(bootstrap_rho)
    metadata = {
        "n_systems": N_SYSTEMS,
        "posterior_path": str(args.metallicity_posterior),
        "data_path": str(args.data),
        "row_alignment": "FITS rows indexed by latent_metallicity_weights.npz row_indices",
        "z_coordinate": "stage-one posterior median [M/H]",
        "indicator": "max(ruwe1, ruwe2)",
        "bootstrap_replicates": int(args.bootstrap),
        "bootstrap_seed": int(args.seed),
        "bootstrap_scheme": "system bootstrap plus one posterior-grid Z draw per sampled system",
        "missingness": {
            "ruwe1": int(np.sum(~np.isfinite(ruwe1))),
            "ruwe2": int(np.sum(~np.isfinite(ruwe2))),
            "z_median": int(np.sum(~np.isfinite(z_median))),
        },
        "cmd_control": cmd_meta,
        "spearman_z_vs_log_max_ruwe": {
            "posterior_median_z_rho": float(spearmanr(z_median, np.log(ruwe)).statistic),
            "bootstrap_posterior_draw_rho_median": float(np.median(rho_draws)),
            "bootstrap_posterior_draw_rho_16_84": np.quantile(rho_draws, [0.16, 0.84]).tolist(),
        },
        "low_vs_high_z": {
            "low_z_ruwe_q95": float(np.quantile(ruwe[low], 0.95)),
            "high_z_ruwe_q95": float(np.quantile(ruwe[high], 0.95)),
            "q95_ratio_high_over_low": float(np.quantile(ruwe[high], 0.95) / np.quantile(ruwe[low], 0.95)),
            "low_z_fraction_gt_1p4": float(np.mean(ruwe[low] > 1.4)),
            "high_z_fraction_gt_1p4": float(np.mean(ruwe[high] > 1.4)),
            "fraction_gt_1p4_difference_high_minus_low": float(np.mean(ruwe[high] > 1.4) - np.mean(ruwe[low] > 1.4)),
            "low_z_cmd_high_ruwe_fraction": float(np.mean(high_cmd[low])),
            "high_z_cmd_high_ruwe_fraction": float(np.mean(high_cmd[high])),
            "cmd_high_ruwe_difference_high_minus_low": float(np.mean(high_cmd[high]) - np.mean(high_cmd[low])),
            "low_z_max_ruwe_from_fainter_fraction": float(np.mean(comes_from_fainter[low])),
            "high_z_max_ruwe_from_fainter_fraction": float(np.mean(comes_from_fainter[high])),
        },
        "max_ruwe_from_fainter_member_fraction": float(np.mean(comes_from_fainter)),
        "max_ruwe_source_spearman_with_z": float(spearmanr(z_median, comes_from_fainter.astype(float)).statistic),
        "excluded_inputs": ["corrected metallicity columns", "mlr_mcmc.npz", "f_Z", "f_outlier", "dynamics_likelihood_lookup.npz"],
    }
    (args.output_dir / "ruwe_metallicity_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
