#!/usr/bin/env python3
"""
Basic Fitting Example

This example demonstrates how to use `binary_masses.depracated.NonParametricMLR`
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

from binary_masses.depracated import NonParametricMLR


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

    # Generate u values by sampling tilde_u = u/sqrt(m_tot) from the empirical p(u)
    # used by the model. For this example we use simple rejection sampling.
    def sample_tilde_u_pu8(n, u_max=80.0, rng=None):
        rng = np.random.default_rng() if rng is None else rng
        model = NonParametricMLR()
        grid = np.linspace(0.0, u_max, 4000)
        pdf_grid = model.func_pu_8(grid)
        pdf_max = float(np.max(pdf_grid)) * 1.05

        samples = []
        while len(samples) < n:
            proposal = rng.uniform(0.0, u_max, size=max(1000, n // 5))
            accept_u = rng.uniform(0.0, pdf_max, size=proposal.shape[0])
            accept = accept_u < model.func_pu_8(proposal)
            samples.extend(proposal[accept].tolist())

        return np.array(samples[:n])

    tilde_u = sample_tilde_u_pu8(n_samples)
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
    data['m1'] = m1_true
    data['m2'] = m2_true

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
    fitter = NonParametricMLR(
        n_bins=8,
        absg_min=4.0,
        absg_max=12.0,
        mass_min=0.05,
        mass_max=2.0,
        uncertainty_model='rice'  # Use Rice distribution
    )

    # Set data
    print("3. Setting data...")
    fitter.set_data(
        u_values=data['u'],
        u_sigma_values=data['u_sigma'],
        absg1_values=data['absg1'],
        absg2_values=data['absg2'],
    )

    # Run inference
    print("\n4. Running Bayesian inference...")
    print("   This may take a few minutes...")
    mcmc = fitter.run_numpyro(
        num_warmup=500,   # Reduced for demo
        num_samples=1000,  # Reduced for demo
        num_chains=2,      # Reduced for demo
        gamma=10.0,       # Moderate regularization
        seed=42,
        int_du=0.1,       # Faster integration
        int_umax=50       # Reasonable range
    )

    # Plot results
    print("\n5. Creating plots...")
    os.makedirs('results', exist_ok=True)

    # Plot corner plot
    fitter.plot_results(output_dir='results', output_suffix='_basic')

    # Plot fitting results (includes truth scatter from the mock data table)
    fitter.plot_fitting_results(data=data, output_dir='results', output_suffix='_basic')

    # Print summary statistics
    print("\n6. Summary Statistics:")
    print(f"   Number of samples: {len(fitter.samples)}")
    print(f"   Sample means: {np.mean(fitter.samples, axis=0)}")
    print(f"   Sample stds: {np.std(fitter.samples, axis=0)}")

    print("\n7. Results saved to 'results/' directory")
    print("   - corner_basic.png")
    print("   - fit_basic.png")

    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
