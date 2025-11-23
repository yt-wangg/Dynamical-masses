#!/usr/bin/env python3
"""
Basic Fitting Example

This example demonstrates how to use the NonParametricPosteriorPlotter
to fit a mass-luminosity relation for a single metallicity bin.

Author: Yutong Wang
"""

import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table
import sys
import os

# Add the package to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from binary_masses import NonParametricPosteriorPlotter


def generate_mock_data(n_samples=5000, noise_level=0.1):
    """
    Generate mock binary data for testing.

    Parameters
    ----------
    n_samples : int
        Number of binary systems to generate
    noise_level : float
        Relative noise level for u uncertainties

    Returns
    -------
    data : astropy.table.Table
        Mock binary data
    """
    np.random.seed(42)

    # Generate true masses (realistic range)
    m1_true = np.random.uniform(0.1, 1.5, n_samples)
    m2_true = np.random.uniform(0.1, 1.5, n_samples)
    m_tot = m1_true + m2_true

    # Generate absolute G magnitudes using a simple mass-luminosity relation
    # L ∝ M^4 for main sequence stars, M_G ∝ -2.5*log10(L)
    def mass_to_absg(mass):
        return 4.43 - 2.5 * np.log10(np.maximum(mass**4, 1e-10))

    absg1_true = mass_to_absg(m1_true) + np.random.normal(0, 0.1, n_samples)
    absg2_true = mass_to_absg(m2_true) + np.random.normal(0, 0.1, n_samples)

    # Generate u values using the empirical distribution
    from binary_masses.core import PU8Sampler
    sampler = PU8Sampler()

    # Sample tilde_u = u/sqrt(m_tot)
    tilde_u = sampler.sample(n_samples, method='inverse_transform')
    u_true = tilde_u * np.sqrt(m_tot)

    # Add measurement noise
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
    data['m1_true'] = m1_true
    data['m2_true'] = m2_true

    # Filter data
    mask = (data['absg1'] >= 4) & (data['absg1'] <= 12) & \
           (data['absg2'] >= 4) & (data['absg2'] <= 12)
    data = data[mask]

    print(f"Generated {len(data)} mock binary systems")
    return data


def main():
    """Main fitting routine."""
    print("=" * 60)
    print("Basic Binary Mass Fitting Example")
    print("=" * 60)

    # Generate mock data
    print("\n1. Generating mock data...")
    data = generate_mock_data(n_samples=3000, noise_level=0.1)

    # Initialize fitter
    print("\n2. Initializing fitter...")
    plotter = NonParametricPosteriorPlotter(
        n_bins=8,
        absg_min=4.0,
        absg_max=12.0,
        uncertainty_model='rice'  # Use Rice distribution
    )

    # Set data
    print("3. Setting data...")
    plotter.set_data(
        u_values=data['u'],
        u_sigma_values=data['u_sigma'],
        absg1_values=data['absg1'],
        absg2_values=data['absg2'],
        gamma=10.0  # Moderate regularization
    )

    # Run inference
    print("\n4. Running Bayesian inference...")
    print("   This may take a few minutes...")
    mcmc = plotter.run_numpyro(
        num_warmup=500,   # Reduced for demo
        num_samples=1000,  # Reduced for demo
        num_chains=2,      # Reduced for demo
        mass_min=0.05,
        mass_max=2.0,
        seed=42,
        int_du=0.1,       # Faster integration
        int_umax=50       # Reasonable range
    )

    # Plot results
    print("\n5. Creating plots...")
    os.makedirs('results', exist_ok=True)

    # Plot corner plot
    plotter.plot_results(output_dir='results', output_suffix='_basic')

    # Plot fitting results
    def true_mass_relation(absg):
        """True mass-luminosity relation used for mock data."""
        return 10**((4.43 - absg) / 10)

    plotter.plot_fitting_results(
        true_mass_func=true_mass_relation,
        output_dir='results',
        output_suffix='_basic'
    )

    # Print summary statistics
    print("\n6. Summary Statistics:")
    print(f"   Number of samples per parameter: {len(plotter.samples)}")
    print(f"   Parameter means: {np.mean(plotter.samples, axis=0)}")
    print(f"   Parameter stds: {np.std(plotter.samples, axis=0)}")

    print("\n7. Results saved to 'results/' directory")
    print("   - corner_plot_basic.png")
    print("   - nonparametric_fit_8bins_rice_basic.png")

    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()