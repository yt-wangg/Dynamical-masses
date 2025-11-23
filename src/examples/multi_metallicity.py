#!/usr/bin/env python3
"""
Multi-Metallicity Fitting Example

This example demonstrates how to use the MultiMetallicityFitter
to fit mass-luminosity relations for different metallicity bins.

Author: Yutong Wang
"""

import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table
import sys
import os

# Add the package to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from binary_masses import MultiMetallicityFitter


def generate_mock_data_with_metallicity(n_samples=5000):
    """
    Generate mock binary data with metallicity-dependent mass-luminosity relations.

    Parameters
    ----------
    n_samples : int
        Number of binary systems to generate

    Returns
    -------
    data : astropy.table.Table
        Mock binary data with metallicity
    """
    np.random.seed(42)

    # Generate metallicities (roughly solar neighborhood distribution)
    feh = np.random.normal(-0.1, 0.2, n_samples)
    feh = np.clip(feh, -1.0, 0.5)

    # Generate true masses
    m1_true = np.random.uniform(0.1, 1.5, n_samples)
    m2_true = np.random.uniform(0.1, 1.5, n_samples)
    m_tot = m1_true + m2_true

    # Metallicity-dependent mass-luminosity relation
    def mass_to_absg(mass, feh):
        # Base relation (solar metallicity)
        absg_solar = 4.43 - 2.5 * np.log10(np.maximum(mass**4, 1e-10))
        # Metallicity effect: metal-rich stars are brighter at fixed mass
        feh_offset = -0.3 * feh  # dex
        return absg_solar + feh_offset

    # Generate absolute G magnitudes
    absg1_true = mass_to_absg(m1_true, feh) + np.random.normal(0, 0.1, n_samples)
    absg2_true = mass_to_absg(m2_true, feh) + np.random.normal(0, 0.1, n_samples)

    # Generate u values using the empirical distribution
    from binary_masses.core import PU8Sampler
    sampler = PU8Sampler()
    tilde_u = sampler.sample(n_samples, method='inverse_transform')
    u_true = tilde_u * np.sqrt(m_tot)

    # Add measurement noise
    noise_level = 0.1
    u_obs = u_true + np.random.normal(0, noise_level * u_true)
    u_sigma = noise_level * u_true * np.ones_like(u_obs)

    # Ensure positive u values
    u_obs = np.maximum(u_obs, 1e-6)

    # Create data table
    data = Table()
    data['u'] = u_obs
    data['u_sigma'] = u_sigma
    data['absg1'] = absg1_true
    data['absg2'] = absg2_true
    data['feh'] = feh
    data['m1_true'] = m1_true
    data['m2_true'] = m2_true

    # Filter data
    mask = (data['absg1'] >= 3) & (data['absg1'] <= 14) & \
           (data['absg2'] >= 3) & (data['absg2'] <= 14)
    data = data[mask]

    print(f"Generated {len(data)} mock binary systems with metallicity")
    print(f"Metallicity range: [{np.min(feh):.2f}, {np.max(feh):.2f}]")

    return data


def true_mass_relations():
    """Generate true mass-luminosity relations for different metallicities."""
    absg_range = np.linspace(3, 14, 1000)

    def mass_relation(absg, feh):
        return 10**((4.43 - absg + 0.3 * feh) / 10)

    # Define relations for different metallicities
    relations = {
        -0.5: lambda absg: mass_relation(absg, -0.5),  # Metal-poor
        -0.1: lambda absg: mass_relation(absg, -0.1),  # Solar-like
        0.2:  lambda absg: mass_relation(absg, 0.2),   # Metal-rich
    }

    return relations, absg_range


def main():
    """Main multi-metallicity fitting routine."""
    print("=" * 60)
    print("Multi-Metallicity Binary Mass Fitting Example")
    print("=" * 60)

    # Generate mock data
    print("\n1. Generating mock data with metallicity...")
    data = generate_mock_data_with_metallicity(n_samples=8000)

    # Initialize multi-metallicity fitter
    print("\n2. Initializing multi-metallicity fitter...")
    multi_fitter = MultiMetallicityFitter(
        n_absg_bins=10,
        absg_min=3.0,
        absg_max=14.0,
        uncertainty_model='rice'  # Use Rice distribution
    )

    # Bin data by metallicity
    print("\n3. Binning data by metallicity...")
    binned_data = multi_fitter.bin_data_by_metallicity(
        data,
        feh_column='feh',
        n_feh_bins=3,
        feh_min=-0.8,
        feh_max=0.4,
        equal_frequency=True  # Equal number of stars per bin
    )

    # Run fitting for all bins
    print("\n4. Fitting all metallicity bins...")
    print("   This may take several minutes...")
    multi_fitter.fit_all_bins(
        binned_data,
        gamma=np.inf,  # No regularization
        num_warmup=800,   # Reduced for demo
        num_samples=1500,  # Reduced for demo
        num_chains=2,      # Reduced for demo
        mass_min=0.05,
        mass_max=2.0,
        seed=42,
        int_du=0.1,       # Faster integration
        int_umax=50       # Reasonable range
    )

    # Create comprehensive plots
    print("\n5. Creating comprehensive plots...")
    os.makedirs('results', exist_ok=True)

    # Plot individual results for each bin
    multi_fitter.plot_all_results(output_dir='results')

    # Get true mass relations for comparison
    true_relations, absg_range = true_mass_relations()

    # Create custom comparison plot
    fig, ax = plt.subplots(figsize=(12, 8))

    # Color map for different metallicity bins
    from matplotlib import cm
    colors = cm.viridis(np.linspace(0, 1, len(multi_fitter.fitters)))

    for (bin_idx, fitter), color in zip(multi_fitter.fitters.items(), colors):
        # Get median and credible intervals
        num_samples = min(1000, len(fitter.samples))
        indices = np.random.choice(len(fitter.samples), size=num_samples, replace=False)
        resampled_data = fitter.samples[indices]

        fitted_mass_samples = np.array([
            np.interp(absg_range, fitter.absg_bins, mass_bins)
            for mass_bins in resampled_data
        ])

        percentiles = np.percentile(fitted_mass_samples, [16, 50, 84], axis=0)
        lower, median, upper = percentiles[0], percentiles[1], percentiles[2]

        # Plot fitted relation
        feh_center = multi_fitter.feh_bin_centers[bin_idx]
        label = f'[Fe/H] = {feh_center:.2f} (fit)'
        ax.fill_between(absg_range, lower, upper, color=color, alpha=0.2)
        ax.plot(absg_range, median, color=color, label=label, linewidth=2)

        # Plot true relation (closest metallicity)
        feh_centers = list(true_relations.keys())
        closest_feh = feh_centers[np.argmin(np.abs(np.array(feh_centers) - feh_center))]
        true_func = true_relations[closest_feh]
        true_mass = true_func(absg_range)

        true_label = f'[Fe/H] = {closest_feh:.1f} (true)'
        ax.plot(absg_range, true_mass, color=color, linestyle='--',
               linewidth=1.5, alpha=0.7, label=true_label)

    ax.set_xlim(3, 14)
    ax.set_xlabel('$M_{\\mathrm{G}}$ [mag]', fontsize=14)
    ax.set_ylabel('Mass [$M_{\\odot}$]', fontsize=14)
    ax.legend(fontsize=9, loc='best', ncol=2)
    ax.set_title('Multi-Metallicity Mass-Luminosity Relations', fontsize=16)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('results/multi_metallicity_comparison.png', dpi=300)
    plt.close(fig)

    # Save samples
    multi_fitter.save_all_samples(output_dir='results', prefix='multi_feh_samples')

    # Print summary
    print("\n6. Summary Statistics:")
    for bin_idx, fitter in multi_fitter.fitters.items():
        feh_center = multi_fitter.feh_bin_centers[bin_idx]
        mean_masses = np.mean(fitter.samples, axis=0)
        print(f"   Bin {bin_idx} ([Fe/H]={feh_center:.2f}):")
        print(f"     Mean masses: {mean_masses[:5]}...")
        print(f"     Number of systems: {len(binned_data[bin_idx])}")

    print("\n7. Results saved to 'results/' directory:")
    print("   - Individual corner and fit plots for each metallicity bin")
    print("   - multi_metallicity_comparison.png")
    print("   - Posterior samples: multi_feh_samples_*.txt")

    print("\n" + "=" * 60)
    print("Multi-metallicity example completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()