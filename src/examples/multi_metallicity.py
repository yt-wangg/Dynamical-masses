#!/usr/bin/env python3
"""
Multi-Metallicity Fitting Example

This example demonstrates how to use the MultiMetallicityFitter
to fit mass-luminosity relations for different metallicity bins.

Author: Yutong Wang
"""

import numpy as np
from astropy.table import Table
import sys
import os

# Add the package to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.binary_masses import MultiMetallicityFitter


def test_nonparametric_model(data, output_dir,
                             n_absg_bins=10, absg_min=3.0, absg_max=14.0,
                             uncertainty_model='rice',
                             feh_column='feh', n_feh_bins=3, feh_min=-1, feh_max=0.6, equal_frequency=False,
                             gamma=np.inf, num_warmup=800, num_samples=1500, num_chains=2,
                             mass_min=0.08, mass_max=1.5, seed=42):
    """Test the non-parametric model using the new MultiMetallicityFitter."""
    print("\n" + "="*50)
    print("TESTING NON-PARAMETRIC MODEL")
    print("="*50)

    # Initialize multi-metallicity fitter with non-parametric model
    print("1. Initializing non-parametric multi-metallicity fitter...")
    multi_fitter = MultiMetallicityFitter(
        model_type='nonparametric',
        n_absg_bins=n_absg_bins,
        absg_min=absg_min,
        absg_max=absg_max,
        uncertainty_model=uncertainty_model,  # Use Rice distribution
        f_outlier=0,  # No outliers
        outlier_u0=30,
        outlier_sigma=15
    )

    # Bin data by metallicity
    print("2. Binning data by metallicity...")
    binned_data = multi_fitter.bin_data_by_metallicity(
        data,
        feh_column=feh_column,
        n_feh_bins=n_feh_bins,
        feh_min=feh_min,
        feh_max=feh_max,
        equal_frequency=equal_frequency  # Equal number of stars per bin
    )

    # Run fitting for all bins
    print("3. Fitting all metallicity bins (non-parametric)...")
    print("   This may take several minutes...")
    multi_fitter.fit_all_bins(
        binned_data,
        gamma=gamma,  # No regularization
        num_warmup=num_warmup,   # Reduced for demo
        num_samples=num_samples,  # Reduced for demo
        num_chains=num_chains,      # Reduced for demo
        mass_min=mass_min,
        mass_max=mass_max,
        seed=seed,
    )

    # Save samples
    print("4. Saving non-parametric samples...")
    multi_fitter.save_all_samples(output_dir=output_dir, prefix=f'nonparametric_{uncertainty_model}_samples')

    # Plot results
    print("5. Plotting non-parametric results...")
    multi_fitter.plot_all_results(binned_data=binned_data, output_dir=output_dir)
    multi_fitter.plot_comparison(output_path=os.path.join(output_dir, f'nonparametric_{uncertainty_model}_uncertainty_{n_feh_bins}fehbins_comparison.png'), data=binned_data)

    # Print summary
    print("\n6. Non-parametric Summary Statistics:")
    for bin_idx, fitter in multi_fitter.fitters.items():
        feh_center = multi_fitter.feh_bin_centers[bin_idx]
        mean_masses = np.mean(fitter.samples, axis=0)
        print(f"   Bin {bin_idx} ([Fe/H]={feh_center:.2f}):")
        print(f"     Mean masses: {mean_masses[:5]}...")
        print(f"     Number of systems: {len(binned_data[bin_idx])}")

    return multi_fitter


def test_broken_powerlaw_model(data, output_dir,
                               n_segments=3, absg_min=3.0, absg_max=14.0,
                               uncertainty_model='rice',
                               feh_column='feh', n_feh_bins=1, feh_min=-1, feh_max=0.6, equal_frequency=False,
                               num_warmup=800, num_samples=2500, num_chains=2,
                               seed=40):
    """Test the broken power law model using the new MultiMetallicityFitter."""
    print("\n" + "="*50)
    print("TESTING BROKEN POWER LAW MODEL")
    print("="*50)

    # Define break points for the broken power law
    print(f"1. Using broken power law with : {n_segments} segments")

    # Initialize multi-metallicity fitter with broken power law model
    print("2. Initializing broken power law multi-metallicity fitter...")
    multi_fitter = MultiMetallicityFitter(
        model_type='broken_powerlaw',
        absg_min=absg_min,
        absg_max=absg_max,
        n_segments=n_segments,
        uncertainty_model=uncertainty_model,
        f_outlier=0,  # Allow 10% outliers
        outlier_u0=30,
        outlier_sigma=15
    )

    # Bin data by metallicity
    print("3. Binning data by metallicity...")
    binned_data = multi_fitter.bin_data_by_metallicity(
        data,
        feh_column=feh_column,
        n_feh_bins=n_feh_bins,
        feh_min=feh_min,
        feh_max=feh_max,
        equal_frequency=equal_frequency  # Equal number of stars per bin
    )

    # Run fitting for all bins
    print("4. Fitting all metallicity bins (broken power law)...")
    print("   This may take several minutes...")
    multi_fitter.fit_all_bins(
        binned_data,
        num_warmup=num_warmup,   # Reduced for demo
        num_samples=num_samples,  # Reduced for demo
        num_chains=num_chains,      # Reduced for demo
        seed=seed,
        a_prior_range=(-15,15), b_prior_range=(-80,5),
    )

    # Save samples
    print("5. Saving broken power law samples...")
    multi_fitter.save_all_samples(output_dir=output_dir, prefix=f'broken_powerlaw_{uncertainty_model}_samples')

    # Plot results
    print("6. Plotting broken power law results...")
    multi_fitter.plot_all_results(binned_data=binned_data, output_dir=output_dir)
    multi_fitter.plot_comparison(output_path=os.path.join(output_dir, f'broken_powerlaw_{uncertainty_model}_uncertainty_{n_feh_bins}fehbins_comparison.png'), data=binned_data)

    # Print summary
    print("\n7. Broken Power Law Summary Statistics:")
    for bin_idx, fitter in multi_fitter.fitters.items():
        feh_center = multi_fitter.feh_bin_centers[bin_idx]
        print(f"   Bin {bin_idx} ([Fe/H]={feh_center:.2f}):")
        print(f"     Number of systems: {len(binned_data[bin_idx])}")
        # Print model-specific summary if available
        if hasattr(fitter, 'print_summary'):
            try:
                fitter.print_summary()
            except:
                pass  # Skip if print_summary fails

    return multi_fitter




def main():
    """Main multi-metallicity fitting routine."""
    print("=" * 70)
    print("Multi-Metallicity Binary Mass Fitting Example")
    print("Testing Both Non-Parametric and Broken Power Law Models")
    print("=" * 70)

    # Import data
    print("\n1. Importing data with metallicity...")
    tag = 'efuncu_unifm_mh_Jsu_n10k_obserr'
    data_path = 'data/mock_data_'+tag+'.fits'
    data = Table.read(data_path)

    # Prepare data subset
    data = data[data['u'] > 0]
    data = data[data['absg1']>=3]
    data = data[data['absg2']>=3]
    data = data[data['feh']>=-1]  # focus on higher metallicity
    data = data[data['feh']<=0.6]

    # Using a smaller subset for faster testing
    indices = np.random.choice(len(data), size=5000, replace=False)
    data = data[indices]
    print(f"   Using {len(data)} systems for testing")

    # Set output directory
    output_dir = 'results/multi_metallicity'
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n2. Results will be saved to: {output_dir}")

    # Test non-parametric model
    # nonparametric_fitter = test_nonparametric_model(data, output_dir,
    #                                                 n_absg_bins=10, absg_min=3.0, absg_max=14.0,
    #                                                 uncertainty_model='gaussian',
    #                                                 feh_column='feh', n_feh_bins=1, feh_min=-1, feh_max=0.6, equal_frequency=False,
    #                                                 gamma=np.inf, num_warmup=500, num_samples=3000, num_chains=2,
    #                                                 mass_min=0.01, mass_max=1.5, seed=4)
    

    # Test broken power law model
    broken_powerlaw_models = test_broken_powerlaw_model(data, output_dir,
                                                        n_segments=2, absg_min=3.0, absg_max=14.0,
                                                        uncertainty_model='gaussian',
                                                        feh_column='feh', n_feh_bins=1, feh_min=-1, feh_max=0.6, equal_frequency=False,
                                                        num_warmup=500, num_samples=2000, num_chains=2,
                                                        seed=8)

    # Final summary
    # print("\n" + "=" * 70)
    # print("FINAL SUMMARY")
    # print("=" * 70)
    # print(f"✓ Non-parametric model test completed")
    # print(f"  - Number of metallicity bins: {len(nonparametric_fitter.fitters)}")

    print(f"\n✓ Broken power law model test completed")
    # print(f"  - Number of metallicity bins fitted: {len(broken_powerlaw_models)}")

    print(f"\nGenerated files in {output_dir}:")
    print(f"  Non-parametric:")
    print(f"    - nonparametric_*.png")
    print(f"    - nonparametric_samples_*.txt")
    print(f"    - nonparametric_multi_feh_comparison.png")
    print(f"  Broken Power Law:")
    print(f"    - broken_powerlaw_feh_*_results.png")
    print(f"    - broken_powerlaw_feh_*_samples.txt")
    print(f"    - broken_powerlaw_multi_feh_comparison.png")

    print("\n" + "=" * 70)
    print("Multi-metallicity example completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()