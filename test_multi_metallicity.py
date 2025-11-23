#!/usr/bin/env python3
"""
Test script for multi-metallicity fitting using local data.
"""

import numpy as np
from astropy.table import Table
from binary_masses import MultiMetallicityFitter
import os
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Load test data
data_path = 'data/mock_data_efuncu_unifm_mh_Jsu_n10k_obserr.fits'
data = Table.read(data_path)

# Filter data
data = data[data['u'] > 0]
data = data[data['absg1'] >= 3]
data = data[data['absg2'] >= 3]
data = data[data['feh'] >= -1]
data = data[data['feh'] <= 0.6]

# Use small subset for quick test
np.random.seed(42)
indices = np.random.choice(len(data), size=500, replace=False)
data = data[indices]

print(f"Loaded {len(data)} systems")

# Initialize fitter
fitter = MultiMetallicityFitter(
    n_absg_bins=5,
    absg_min=4.0,
    absg_max=12.0,
    uncertainty_model='rice'
)

# Bin by metallicity
binned_data = fitter.bin_data_by_metallicity(
    data,
    feh_column='feh',
    n_feh_bins=2,
    feh_min=-1,
    feh_max=0.6,
    equal_frequency=False
)

print(f"Created {len(binned_data)} metallicity bins")
for bin_idx, bin_data in binned_data.items():
    print(f"  Bin {bin_idx}: {len(bin_data)} systems")

# Run fitting (reduced parameters for quick test)
print("\nRunning MCMC fitting (this may take a few minutes)...")
fitter.fit_all_bins(
    binned_data,
    gamma=np.inf,
    num_warmup=200,
    num_samples=300,
    num_chains=1,
    mass_min=0.05,
    mass_max=2.0,
    seed=42,
)

print("\nFitting completed!")

# Print summary
for bin_idx, f in fitter.fitters.items():
    feh_center = fitter.feh_bin_centers[bin_idx]
    mean_masses = np.mean(f.samples, axis=0)
    print(f"Bin {bin_idx} ([Fe/H]={feh_center:.2f}): mean masses = {mean_masses[:3]}...")

# Compare best-fit mass-luminosity relation against ground truth component masses
print("\nGround-truth comparison per metallicity bin:")
for bin_idx, f in fitter.fitters.items():
    bin_data = binned_data[bin_idx]

    # Use median posterior mass in each absg bin as the best-fit curve
    best_fit_mass_bins = np.median(f.samples, axis=0)

    # Predict component masses from absg using the fitted relation
    pred_m1 = np.interp(bin_data['absg1'], f.absg_bins, best_fit_mass_bins)
    pred_m2 = np.interp(bin_data['absg2'], f.absg_bins, best_fit_mass_bins)

    true_masses = np.concatenate([bin_data['m1'], bin_data['m2']])
    pred_masses = np.concatenate([pred_m1, pred_m2])

    abs_err = np.abs(pred_masses - true_masses)
    mae = abs_err.mean()
    rmse = np.sqrt(np.mean((pred_masses - true_masses) ** 2))
    med_frac_err = np.median(abs_err / true_masses) * 100

    feh_center = fitter.feh_bin_centers[bin_idx]
    print(
        f"  Bin {bin_idx} ([Fe/H]={feh_center:.2f}): "
        f"MAE={mae:.3f} Msun, RMSE={rmse:.3f} Msun, median |Δ|/true={med_frac_err:.1f}%"
    )

# Plot comparison of fitted mass-luminosity relation vs. ground truth
results_dir = "results"
os.makedirs(results_dir, exist_ok=True)
print(f"\nSaving mass-luminosity comparison plots to '{results_dir}'...")

for bin_idx, f in fitter.fitters.items():
    bin_data = binned_data[bin_idx]
    absg_range = np.linspace(f.absg_min, f.absg_max, 400)

    # Compute percentile envelopes for the fitted relation
    fitted_mass_samples = np.array([
        np.interp(absg_range, f.absg_bins, mass_bins) for mass_bins in f.samples
    ])
    lower, median, upper = np.percentile(fitted_mass_samples, [16, 50, 84], axis=0)

    # Ground-truth masses
    true_absg = np.concatenate([np.array(bin_data["absg1"]), np.array(bin_data["absg2"])])
    true_mass = np.concatenate([np.array(bin_data["m1"]), np.array(bin_data["m2"])])

    plt.figure(figsize=(8, 6))
    plt.fill_between(absg_range, lower, upper, color="skyblue", alpha=0.4, label="Fit 16-84%")
    plt.plot(absg_range, median, color="blue", linewidth=2, label="Fit median")
    plt.scatter(true_absg, true_mass, s=10, color="orange", alpha=0.6, label="Ground truth")
    plt.xlabel(r"$M_G$ [mag]")
    plt.ylabel(r"Mass [$M_\odot$]")
    plt.title(f"Mass-Luminosity: Bin {bin_idx} ([Fe/H]={fitter.feh_bin_centers[bin_idx]:.2f})")
    plt.legend()
    plt.tight_layout()

    out_path = os.path.join(results_dir, f"mass_luminosity_comparison_bin{bin_idx}.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"  Saved {out_path}")
