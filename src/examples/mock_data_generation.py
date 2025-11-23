#!/usr/bin/env python3
"""
Mock Data Generation Example

This example demonstrates how to generate realistic mock binary populations
using the binary sampling utilities.

Author: Yutong Wang
"""

import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table
import sys
import os

# Add the package to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from binary_masses import BinaryPopulation, PU8Sampler
import astropy.units as u


def generate_realistic_binary_population(n_binaries=10000):
    """
    Generate a realistic binary star population.

    Parameters
    ----------
    n_binaries : int
        Number of binary systems to generate

    Returns
    -------
    pop : BinaryPopulation
        Generated binary population
    data : astropy.table.Table
        Tabulated observable properties
    """
    print(f"Generating {n_binaries} binary systems...")

    # Create binary population with realistic distributions
    pop = BinaryPopulation(
        n_binaries=n_binaries,
        # Öpik's law: flat distribution in log(a)
        semi_major_axis_params=(0.1, 1000, -1.0),  # a in AU
        # Eccentricity distribution (thermal for hard binaries)
        eccentricity_alpha=0.0,
        # Power-law total mass distribution
        total_mass=np.random.uniform(0.2, 3.0, n_binaries) * u.Msun,
        # Distance distribution (local neighborhood)
        distance=np.random.exponential(200, n_binaries) * u.pc,
        face_on=False  # Random orientations
    )

    # Convert to observable quantities
    data = Table()
    data['distance'] = pop.distance
    data['total_mass'] = pop.total_mass

    # Convert projected quantities to u parameter
    # u = v_tan * sqrt(s_proj) where v_tan is tangential velocity, s_proj is projected separation
    projected_separation_au = pop.projected_separation.to(u.AU).value
    projected_velocity_kms = pop.projected_velocity.to(u.km/u.s).value

    # Calculate u parameter (has units of km/s * sqrt(AU))
    data['u'] = projected_velocity_kms * np.sqrt(projected_separation_au)

    # Generate photometric properties (simplified)
    # Split total mass into primary and secondary
    q = np.random.uniform(0.1, 1.0, n_binaries)  # Mass ratio
    m1 = pop.total_mass.value / (1 + q)
    m2 = q * m1

    # Simple mass-luminosity relation for main sequence stars
    def mass_to_absolute_g_mass(mass):
        """Convert mass to absolute G magnitude."""
        # Approximate relation for main sequence stars
        if mass < 0.43:
            Mg = 5.5 - 10 * np.log10(mass)
        elif mass < 1.0:
            Mg = 4.0 - 4.5 * np.log10(mass)
        elif mass < 2.0:
            Mg = 4.43 - 2.5 * np.log10(mass**4)
        else:
            Mg = 3.0 - 2.0 * np.log10(mass)
        return Mg

    # Vectorized mass to magnitude conversion
    m1_mag = np.array([mass_to_absolute_g_mass(m) for m in m1])
    m2_mag = np.array([mass_to_absolute_g_mass(m) for m in m2])

    # Add some scatter
    m1_mag += np.random.normal(0, 0.1, n_binaries)
    m2_mag += np.random.normal(0, 0.1, n_binaries)

    data['m1'] = m1
    data['m2'] = m2
    data['absg1'] = m1_mag
    data['absg2'] = m2_mag
    data['mass_ratio'] = q

    # Add realistic measurement uncertainties
    # Distance uncertainties (Gaia-like)
    data['distance_error'] = 0.01 * data['distance'].value + 0.001 * u.pc

    # Magnitude uncertainties
    data['absg1_error'] = 0.02 + 0.001 * np.maximum(0, data['absg1'] - 10)
    data['absg2_error'] = 0.02 + 0.001 * np.maximum(0, data['absg2'] - 10)

    # Velocity uncertainties (depends on magnitude)
    v_error_base = 0.1  # km/s for bright stars
    data['u_error'] = v_error_base * (1 + 0.1 * np.maximum(0, data['absg1'] - 12))

    # Add measurement noise to u
    data['u_obs'] = data['u'] + np.random.normal(0, data['u_error'], n_binaries)
    data['u_sigma'] = data['u_error']

    # Ensure positive u values
    data['u_obs'] = np.maximum(data['u_obs'], 1e-6)

    # Add metallicity (solar neighborhood distribution)
    data['feh'] = np.random.normal(-0.1, 0.2, n_binaries)
    data['feh'] = np.clip(data['feh'], -1.0, 0.5)

    print(f"Generated population statistics:")
    print(f"  Distance range: {np.min(data['distance']):.1f} - {np.max(data['distance']):.1f} pc")
    print(f"  Mass range: {np.min(m1):.2f} - {np.max(m1):.2f} Msun")
    print(f"  Mass ratio range: {np.min(q):.2f} - {np.max(q):.2f}")
    print(f"  u parameter range: {np.min(data['u']):.2f} - {np.max(data['u']):.2f}")

    return pop, data


def validate_mock_data(data):
    """
    Validate and visualize the mock data distributions.

    Parameters
    ----------
    data : astropy.table.Table
        Mock binary data
    """
    print("\nGenerating validation plots...")

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    # Distance distribution
    axes[0].hist(data['distance'].value, bins=50, alpha=0.7, edgecolor='black')
    axes[0].set_xlabel('Distance [pc]')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Distance Distribution')
    axes[0].grid(True, alpha=0.3)

    # Mass distribution
    axes[1].hist(data['m1'], bins=50, alpha=0.7, label='Primary', edgecolor='black')
    axes[1].hist(data['m2'], bins=50, alpha=0.7, label='Secondary', edgecolor='black')
    axes[1].set_xlabel('Mass [M☉]')
    axes[1].set_ylabel('Count')
    axes[1].set_title('Mass Distribution')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Mass ratio distribution
    axes[2].hist(data['mass_ratio'], bins=50, alpha=0.7, edgecolor='black')
    axes[2].set_xlabel('Mass Ratio (M₂/M₁)')
    axes[2].set_ylabel('Count')
    axes[2].set_title('Mass Ratio Distribution')
    axes[2].grid(True, alpha=0.3)

    # u parameter distribution
    axes[3].hist(data['u'], bins=50, alpha=0.7, edgecolor='black')
    axes[3].set_xlabel('u Parameter [km/s √AU]')
    axes[3].set_ylabel('Count')
    axes[3].set_title('u Parameter Distribution')
    axes[3].grid(True, alpha=0.3)

    # Magnitude distribution
    axes[4].hist(data['absg1'], bins=50, alpha=0.7, label='Primary', edgecolor='black')
    axes[4].hist(data['absg2'], bins=50, alpha=0.7, label='Secondary', edgecolor='black')
    axes[4].set_xlabel('Absolute G Magnitude')
    axes[4].set_ylabel('Count')
    axes[4].set_title('Absolute Magnitude Distribution')
    axes[4].legend()
    axes[4].grid(True, alpha=0.3)

    # Metallicity distribution
    axes[5].hist(data['feh'], bins=50, alpha=0.7, edgecolor='black')
    axes[5].set_xlabel('[Fe/H]')
    axes[5].set_ylabel('Count')
    axes[5].set_title('Metallicity Distribution')
    axes[5].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('results/mock_data_validation.png', dpi=300)
    plt.close(fig)

    # Mass-magnitude relation
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(data['absg1'], data['m1'], alpha=0.5, s=1, label='Primary')
    ax.scatter(data['absg2'], data['m2'], alpha=0.5, s=1, label='Secondary')
    ax.set_xlabel('Absolute G Magnitude')
    ax.set_ylabel('Mass [M☉]')
    ax.set_title('Mass-Luminosity Relation')
    ax.legend()
    ax.invert_xaxis()  # Brighter stars have smaller magnitudes
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('results/mock_data_mass_luminosity.png', dpi=300)
    plt.close(fig)

    print("Validation plots saved to 'results/' directory")


def main():
    """Main mock data generation routine."""
    print("=" * 60)
    print("Mock Binary Population Generation Example")
    print("=" * 60)

    # Create output directory
    os.makedirs('results', exist_ok=True)

    # Generate mock population
    print("\n1. Generating mock binary population...")
    pop, data = generate_realistic_binary_population(n_binaries=5000)

    # Validate the mock data
    print("\n2. Validating mock data...")
    validate_mock_data(data)

    # Save the data
    print("\n3. Saving mock data...")
    data.write('results/mock_binary_population.fits', overwrite=True)
    data.write('results/mock_binary_population.csv', overwrite=True)

    # Filter data for typical analysis cuts
    print("\n4. Applying typical analysis cuts...")
    mask = (
        (data['absg1'] >= 4) & (data['absg1'] <= 12) &
        (data['absg2'] >= 4) & (data['absg2'] <= 12) &
        (data['u_obs'] > 0) &
        (data['distance'].value <= 500)  # Nearby sample
    )
    filtered_data = data[mask]

    print(f"   Original sample: {len(data)} systems")
    print(f"   After cuts: {len(filtered_data)} systems")

    # Save filtered data
    filtered_data.write('results/mock_binary_population_filtered.fits', overwrite=True)

    # Test PU8 sampler
    print("\n5. Testing PU8 sampler...")
    sampler = PU8Sampler()
    samples = sampler.sample(n_samples=10000, method='inverse_transform')

    # Plot PU8 distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(samples, bins=100, density=True, alpha=0.7, label='Samples', edgecolor='black')

    # Plot theoretical curve
    tilde_u_range = np.linspace(0, 50, 1000)
    pdf_values = sampler.func_pu_8(torch.tensor(tilde_u_range, dtype=torch.float32)).numpy()
    ax.plot(tilde_u_range, pdf_values, 'r-', linewidth=2, label='Theory')

    ax.set_xlabel('ũ = u/√m_tot')
    ax.set_ylabel('Probability Density')
    ax.set_title('PU8 Distribution Sampling Validation')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('results/pu8_distribution_validation.png', dpi=300)
    plt.close(fig)

    print("\n6. Summary:")
    print(f"   Generated {len(data)} mock binary systems")
    print(f"   Applied cuts left {len(filtered_data)} systems")
    print(f"   All validation plots saved to 'results/' directory")
    print(f"   Data files:")
    print(f"     - mock_binary_population.fits (raw data)")
    print(f"     - mock_binary_population_filtered.fits (analysis-ready)")
    print(f"     - mock_binary_population.csv (human-readable)")

    print("\n" + "=" * 60)
    print("Mock data generation completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    # Import torch for PU8 sampling
    try:
        import torch
    except ImportError:
        print("Warning: torch not available, skipping PU8 sampling validation")
        torch = None
    main()