#!/usr/bin/env python3
"""
Test script for Broken Power Law model with plotting functionality from core.py
"""

import numpy as np
import os
from astropy.table import Table
import sys

# Add the package to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from binary_masses import BrokenPowerLawMLR

def main():
    print("=" * 70)
    print("Testing Broken Power Law Model with Plotting")
    print("=" * 70)

    # Import data
    print("\n1. Loading mock data...")
    tag = 'efuncu_unifm_mh_Jsu_n10k_obserr'
    data_path = 'data/mock_data_' + tag + '.fits'
    
    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
        return
    
    data = Table.read(data_path)
    
    # Filter data
    data = data[data['u'] > 0]
    data = data[data['absg1'] >= 3]
    data = data[data['absg2'] >= 3]
    data = data[data['feh'] >= -1]
    data = data[data['feh'] <= 0.6]
    
    # Use subset for faster testing
    indices = np.random.choice(len(data), size=min(3000, len(data)), replace=False)
    data = data[indices]
    print(f"   Using {len(data)} systems for testing")

    # Set output directory
    output_dir = 'results/bpl_test'
    os.makedirs(output_dir, exist_ok=True)
    print(f"   Results will be saved to: {output_dir}")

    # Test with single break point
    print("\n2. Initializing Broken Power Law model...")
    break_points = np.array([0.45])
    print(f"   Break points: {break_points} M_sun")
    
    bpl_fitter = BrokenPowerLawMLR(
        break_points=break_points,
        mass_min=0.08,
        mass_max=1.5,
        absg_min=3.0,
        absg_max=14.0,
        uncertainty_model='gaussian',
        f_outlier=0,
        outlier_u0=30,
        outlier_sigma=15
    )

    # Prepare data
    print("\n3. Setting data for fitting...")
    bpl_fitter.set_data(
        u_values=data['u'],
        u_sigma_values=data['u_sigma'],
        absg1_values=data['absg1'],
        absg2_values=data['absg2']
    )

    # Run inference
    print("\n4. Running Bayesian inference (MCMC)...")
    print("   This may take several minutes...")
    bpl_fitter.run_numpyro(
        num_warmup=500,
        num_samples=2000,
        num_chains=2,
        seed=42,
        a_prior_range=(-20, 10),
        b_prior_range=(-100, 5)
    )

    # Plot results using core.py plotting functions
    print("\n5. Generating plots...")
    
    # Corner plot
    print("   - Creating corner plot...")
    bpl_fitter.plot_results(
        output_dir=output_dir,
        output_suffix='_test'
    )
    
    # Fitting results plot
    print("   - Creating fitting results plot...")
    bpl_fitter.plot_fitting_results(
        data=data,
        output_dir=output_dir,
        output_suffix='_test'
    )

    # Print summary
    print("\n6. Model Summary:")
    if hasattr(bpl_fitter, 'print_summary'):
        bpl_fitter.print_summary()
    else:
        print(f"   Number of segments: {bpl_fitter.n_segments}")
        print(f"   Break points: {bpl_fitter.break_points}")
        if bpl_fitter.samples is not None:
            median_params = np.median(bpl_fitter.samples, axis=0)
            print(f"   Median parameters: {median_params}")

    print("\n" + "=" * 70)
    print("Test completed successfully!")
    print(f"Output files saved in: {output_dir}")
    print("=" * 70)

if __name__ == "__main__":
    main()
