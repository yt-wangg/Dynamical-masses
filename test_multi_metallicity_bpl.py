#!/usr/bin/env python3
"""
Test script for fitting a broken power-law mass-luminosity relation
using the local mock multi-metallicity dataset.

This runs the parametric BrokenPowerLawMLR on a filtered subset of the
mock binaries, reports basic error metrics against the ground truth
masses, and writes a comparison plot to the results directory.
"""

import os
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.table import Table

from binary_masses import BrokenPowerLawMLR


def main():
    data_path = "data/mock_data_efuncu_unifm_mh_Jsu_n10k_obserr.fits"
    np.random.seed(42)

    # Load and filter data
    data = Table.read(data_path)
    data = data[data["u"] > 0]
    data = data[data["absg1"] >= 3]
    data = data[data["absg2"] >= 3]
    data = data[data["feh"] >= -1]
    data = data[data["feh"] <= 0.6]

    # Bin by Fe/H (equal-width)
    feh_min, feh_max = -1.0, 0.6
    n_bins = 2
    feh_edges = np.linspace(feh_min, feh_max, n_bins + 1)

    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)

    for i in range(n_bins):
        bin_mask = (data["feh"] >= feh_edges[i]) & (data["feh"] < feh_edges[i + 1] if i < n_bins - 1 else data["feh"] <= feh_edges[i + 1])
        bin_data = data[bin_mask]
        if len(bin_data) == 0:
            print(f"Skipping FeH bin {i} (empty).")
            continue

        # Use a manageable subset per bin
        np.random.seed(42 + i)
        subset_idx = np.random.choice(len(bin_data), size=min(400, len(bin_data)), replace=False)
        bin_data = bin_data[subset_idx]

        print(f"\nFeH bin {i} [{feh_edges[i]:.2f}, {feh_edges[i+1]:.2f}]: {len(bin_data)} systems")

        fitter = BrokenPowerLawMLR(uncertainty_model="rice")
        fitter.set_data(
            u_values=np.array(bin_data["u"]),
            u_sigma_values=np.array(bin_data["u_sigma"]),
            absg1_values=np.array(bin_data["absg1"]),
            absg2_values=np.array(bin_data["absg2"]),
        )

        fitter.run_numpyro(
            num_warmup=300,
            num_samples=400,
            num_chains=1,
            seed=42 + i,
            int_umax=50,
            int_du=0.1,
        )

        params_median = np.median(fitter.samples, axis=0)
        print("  Posterior median params (first few):", params_median[:4])

        pred_m1 = fitter.mass_from_absg(np.array(bin_data["absg1"]), params_median)
        pred_m2 = fitter.mass_from_absg(np.array(bin_data["absg2"]), params_median)
        true_m1 = np.array(bin_data["m1"])
        true_m2 = np.array(bin_data["m2"])

        pred_masses = np.concatenate([pred_m1, pred_m2])
        true_masses = np.concatenate([true_m1, true_m2])

        abs_err = np.abs(pred_masses - true_masses)
        mae = abs_err.mean()
        rmse = np.sqrt(np.mean((pred_masses - true_masses) ** 2))
        med_frac_err = np.median(abs_err / true_masses) * 100

        print(
            f"  Errors: MAE={mae:.3f} Msun, RMSE={rmse:.3f} Msun, median |Δ|/true={med_frac_err:.1f}%"
        )

        # Plot per bin
        mass_grid = np.logspace(np.log10(0.05), np.log10(3.0), 200)
        n_plot_samples = min(300, len(fitter.samples))
        plot_indices = np.random.choice(len(fitter.samples), size=n_plot_samples, replace=False)

        absg_samples = np.zeros((n_plot_samples, len(mass_grid)))
        for j, idx in enumerate(plot_indices):
            absg_samples[j] = fitter.absg_from_mass(mass_grid, fitter.samples[idx])

        median_absg = np.median(absg_samples, axis=0)
        lower_absg = np.percentile(absg_samples, 16, axis=0)
        upper_absg = np.percentile(absg_samples, 84, axis=0)

        gt_masses = np.concatenate([true_m1, true_m2])
        gt_absg = np.concatenate([np.array(bin_data["absg1"]), np.array(bin_data["absg2"])])

        plt.figure(figsize=(9, 7))
        plt.fill_between(mass_grid, lower_absg, upper_absg, color="skyblue", alpha=0.4, label="Fit 16-84%")
        plt.plot(mass_grid, median_absg, color="blue", linewidth=2, label="Fit median")
        for bp in fitter.break_points:
            plt.axvline(bp, color="gray", linestyle="--", alpha=0.5)
        plt.scatter(gt_masses, gt_absg, s=12, color="orange", alpha=0.6, label="Ground truth components")
        plt.xscale("log")
        plt.xlabel("Mass [$M_\\odot$]")
        plt.ylabel("$M_G$ [mag]")
        plt.gca().invert_yaxis()
        plt.legend()
        plt.title(f"BPL MLR vs Ground Truth (FeH bin {i})")
        plt.tight_layout()

        out_path = os.path.join(results_dir, f"broken_powerlaw_mlr_comparison_fehbin{i}.png")
        plt.savefig(out_path, dpi=200)
        plt.close()
        print(f"  Saved {out_path}")


if __name__ == "__main__":
    main()
