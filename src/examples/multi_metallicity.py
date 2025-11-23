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


def main():
    """Main multi-metallicity fitting routine."""
    print("=" * 60)
    print("Multi-Metallicity Binary Mass Fitting Example")
    print("=" * 60)

    # Import data
    print("\n1. Importing data with metallicity...")
    tag = 'efuncu_unifm_mh_Jsu_n10k_obserr'
    data_path = '/Volumes/WANG/postgraduate/4.2/Dyn/mock_data_'+tag+'.fits'
    data = Table.read(data_path)

    # Prepare data subset
    data = data[data['u'] > 0]
    data = data[data['absg1']>=3]
    data = data[data['absg2']>=3]
    data = data[data['feh']>=-1]  # focus on higher metallicity
    data = data[data['feh']<=0.6]

    # Using a smaller subset for faster grid search
    indices = np.random.choice(len(data), size=5000, replace=False)
    data = data[indices]

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
        feh_min=-1,
        feh_max=0.6,
        equal_frequency=False  # Equal number of stars per bin
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
    )

    # Plot results
    multi_fitter.plot_all_results(output_dir='/Users/ytwang/Library/CloudStorage/OneDrive-Personal/Files/postgraduate/PyProjects/Dyn/Validation/V3_results')
    multi_fitter.plot_comparison(output_path='/Users/ytwang/Library/CloudStorage/OneDrive-Personal/Files/postgraduate/PyProjects/Dyn/Validation/V3_results/mass_absg_comparison_Jsumh_rice_poly_n5k.png')
    
    # Save samples
    multi_fitter.save_all_samples(output_dir='/Users/ytwang/Library/CloudStorage/OneDrive-Personal/Files/postgraduate/PyProjects/Dyn/Validation/V3_results', prefix='posterior_samples_n5k_Jsu_rice_poly')

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