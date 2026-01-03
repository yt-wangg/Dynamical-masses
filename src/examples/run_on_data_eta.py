#!/usr/bin/env python3
"""
Test outlier_kappa values for Multi-Metallicity Fitting

This script tests different outlier_kappa values (10, 50, 200) to see
how they affect the fitting results.

Author: Yutong Wang
"""

import numpy as np
from astropy.table import Table
import sys
import os

# Add the package to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from binary_masses import MultiMetallicityFitter


def test_nonparametric_model(data, output_dir, outlier_kappa,
                             iso_data=None,
                             n_absg_bins=10, absg_min=3.0, absg_max=14.0,
                             uncertainty_model='rice',
                             feh_column='feh', n_feh_bins=3, feh_min=-1, feh_max=0.6, equal_frequency=False,
                             gamma=np.inf, num_warmup=800, num_samples=1500, num_chains=2,
                             mass_min=0.05, mass_max=1.5,
                             seed=42, fit_outlier_params=True, iso_colname_dict={'absg':'absg', 'mass':'mass', 'feh':'MH'}):
    """Test the non-parametric model using the new MultiMetallicityFitter."""
    print("\n" + "="*50)
    print(f"TESTING NON-PARAMETRIC MODEL (kappa={outlier_kappa})")
    print("="*50)

    # Initialize multi-metallicity fitter with non-parametric model
    print("1. Initializing non-parametric multi-metallicity fitter...")
    multi_fitter = MultiMetallicityFitter(
        model_type='nonparametric',
        n_absg_bins=n_absg_bins,
        absg_min=absg_min,
        absg_max=absg_max,
        mass_min=mass_min,
        mass_max=mass_max,
        uncertainty_model=uncertainty_model,  # Use Rice distribution
        f_outlier=0.1,  # Allow 10% outliers
        outlier_u0=30,
        outlier_sigma=15,
        outlier_kappa=outlier_kappa,
        fit_outlier_params=fit_outlier_params
    )

    # Override _build_suffix method to include kappa value
    original_build_suffix = multi_fitter._build_suffix
    def build_suffix_with_kappa(bin_idx=None):
        base_suffix = original_build_suffix(bin_idx)
        # Replace '_outlierfixed' with '_outlierfixed_kappa{outlier_kappa}'
        if '_outlierfixed' in base_suffix:
            return base_suffix.replace('_outlierfixed', f'_outlierfixed_kappa{outlier_kappa}')
        return base_suffix + f'_kappa{outlier_kappa}'

    multi_fitter._build_suffix = build_suffix_with_kappa

    # Bin data by metallicity
    print("2. Binning data by metallicity...")
    if iso_data is not None:
        binned_data, binned_iso = multi_fitter.bin_data_by_metallicity(
            data, iso_data=iso_data,
            feh_column=feh_column,
            n_feh_bins=n_feh_bins,
            feh_min=feh_min,
            feh_max=feh_max,
            equal_frequency=equal_frequency,  # Equal number of stars per bin
            iso_colname_dict=iso_colname_dict
        )
    else:
        binned_data = multi_fitter.bin_data_by_metallicity(
            data, iso_data=iso_data,
            feh_column=feh_column,
            n_feh_bins=n_feh_bins,
            feh_min=feh_min,
            feh_max=feh_max,
            equal_frequency=equal_frequency  # Equal number of stars per bin
        )
        binned_iso = None

    # Run fitting for all bins
    print("3. Fitting all metallicity bins (non-parametric)...")
    print("   This may take several minutes...")
    multi_fitter.fit_all_bins(
        binned_data,
        gamma=gamma,  # No regularization
        outlier_kappa=outlier_kappa,
        num_warmup=num_warmup,   # Reduced for demo
        num_samples=num_samples,  # Reduced for demo
        num_chains=num_chains,      # Reduced for demo
        seed=seed,
    )

    # Save samples with kappa in filename
    print("4. Saving non-parametric samples...")
    multi_fitter.save_all_samples(output_dir=output_dir, prefix=f'mcmc_kappa{outlier_kappa}')

    # Plot results with kappa in filename (now handled by overridden _build_suffix)
    print("5. Plotting non-parametric results...")
    multi_fitter.plot_all_results(binned_data=binned_data, binned_iso=binned_iso, iso_colname_dict=iso_colname_dict,
                                 output_dir=output_dir)
    multi_fitter.plot_comparison(output_dir=output_dir, data=binned_data, iso_data=binned_iso,
                                iso_colname_dict=iso_colname_dict)

    # Print summary
    print("\n6. Non-parametric Summary Statistics:")
    for bin_idx, fitter in multi_fitter.fitters.items():
        feh_center = multi_fitter.feh_bin_centers[bin_idx]
        mean_masses = np.mean(fitter.samples, axis=0)
        print(f"   Bin {bin_idx} ([Fe/H]={feh_center:.2f}):")
        print(f"     Mean masses: {mean_masses[:5]}...")
        print(f"     Number of systems: {len(binned_data[bin_idx])}")

    return multi_fitter


def main():
    """Main testing routine for different outlier_kappa values."""
    print("=" * 70)
    print("Testing outlier_kappa values for Non-Parametric Model")
    print("Values to test: [10, 50, 200]")
    print("=" * 70)

    # Import data
    print("\n1. Importing data with metallicity...")
    data_path = 'data/jd_single_1kpc_filtered.fits'
    data = Table.read(data_path)

    # Use the pre-resampled interpolated isochrone grid (fast, already filtered)
    from binary_masses import load_interpolated_isochrone_data

    iso_data = load_interpolated_isochrone_data("data/interpolated_mass_data")
    iso_colname_dict = {"absg": "absg", "mass": "mass", "feh": "MH"}

    # Prepare data subset
    data['v'] = 4.74 * np.sqrt((data['pmra2']-data['pmra1'])**2 +
                                    (data['pmdec2']-data['pmdec1'])**2
                                    ) / data['parallax1']
    data['u'] = data['v'] * np.sqrt(data['sep_AU'])
    data['u_sigma'] = data['u'] / data['dpm_over_error']

    print(f"   Using {len(data)} systems for testing")

    # Test different outlier_kappa values
    kappa_values = [50, 200, 500]

    for kappa in kappa_values:
        print(f"\n" + "="*70)
        print(f"Testing outlier_kappa = {kappa}")
        print("="*70)

        # Set output directory for this kappa value
        output_dir = f'eta_test/kappa{kappa}'
        os.makedirs(output_dir, exist_ok=True)
        print(f"Results will be saved to: {output_dir}")

        # Test non-parametric model with current kappa
        try:
            nonparametric_fitter = test_nonparametric_model(data, output_dir,
                                                           n_absg_bins=10, absg_min=3, absg_max=14.0,
                                                           uncertainty_model='rice',
                                                           feh_column='feh_jcaps_1', n_feh_bins=1, feh_min=-1.5, feh_max=0.6, equal_frequency=False,
                                                           gamma=np.inf, num_warmup=500, num_samples=2000, num_chains=2,
                                                           mass_min=0.05, mass_max=1.5,
                                                           seed=14, fit_outlier_params=True, iso_data=iso_data,
                                                           iso_colname_dict=iso_colname_dict, outlier_kappa=kappa)

            print(f"✓ Successfully completed test for kappa={kappa}")

        except Exception as e:
            print(f"✗ Error testing kappa={kappa}: {e}")
            continue

    # Final summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"✓ Tested outlier_kappa values: {kappa_values}")
    print(f"✓ Results saved in eta_test/ directory with kappa-specific subfolders")

    print("\nGenerated files structure:")
    print("  eta_test/")
    for kappa in kappa_values:
        print(f"    kappa{kappa}/")
        print(f"      - mcmc_kappa{kappa}_nonparametric_*.txt")
        print(f"      - *_kappa{kappa}.png")

    print("\n" + "=" * 70)
    print("outlier_kappa testing completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
