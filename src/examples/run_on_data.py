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
from binary_masses import MultiMetallicityFitter, DifferencePolyMLR, IsochroneMassModel


def test_nonparametric_model(data, output_dir,
                             iso_data = None, 
                             n_absg_bins=10, absg_min=3.0, absg_max=14.0,
                             uncertainty_model='rice',
                             feh_column='feh', n_feh_bins=3, feh_min=-1, feh_max=0.6, equal_frequency=False,
                             gamma=np.inf, outlier_kappa=None, outlier_kappa_scale=0.05, num_warmup=800, num_samples=1500, num_chains=2,
                             mass_min=0.05, mass_max=1.5, 
                             seed=42, fit_outlier_params=False, iso_colname_dict = {'absg':'absg', 'mass':'mass', 'feh':'MH'}):
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
        mass_min=mass_min,
        mass_max=mass_max,
        uncertainty_model=uncertainty_model,  # Use Rice distribution
        f_outlier=0.1,  # Allow 10% outliers
        outlier_u0=30,
        outlier_sigma=15,
        outlier_kappa=outlier_kappa,
        outlier_kappa_scale=outlier_kappa_scale,
        fit_outlier_params=fit_outlier_params
    )

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
        outlier_kappa_scale=outlier_kappa_scale,
        num_warmup=num_warmup,   # Reduced for demo
        num_samples=num_samples,  # Reduced for demo
        num_chains=num_chains,      # Reduced for demo
        seed=seed,
    )

    # Save samples
    print("4. Saving non-parametric samples...")
    multi_fitter.save_all_samples(output_dir=output_dir, prefix='mcmc')

    # Plot results
    print("5. Plotting non-parametric results...")
    multi_fitter.plot_all_results(binned_data=binned_data, binned_iso=binned_iso, iso_colname_dict=iso_colname_dict, output_dir=output_dir)
    multi_fitter.plot_comparison(output_dir=output_dir, data=binned_data, iso_data=binned_iso, iso_colname_dict=iso_colname_dict)

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
                               outlier_kappa=None, outlier_kappa_scale=0.05, seed=40, fit_outlier_params=False, iso_data=None, iso_colname_dict = {'absg':'absg', 'mass':'mass', 'feh':'MH'}):
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
        f_outlier=0.1,  # Allow 10% outliers
        outlier_u0=30,
        outlier_sigma=15,
        outlier_kappa=outlier_kappa,
        outlier_kappa_scale=outlier_kappa_scale,
        fit_outlier_params=fit_outlier_params
    )

    # Bin data by metallicity
    print("3. Binning data by metallicity...")
    if iso_data is not None:
        binned_data, binned_iso = multi_fitter.bin_data_by_metallicity(
            data,
            iso_data=iso_data,
            feh_column=feh_column,
            n_feh_bins=n_feh_bins,
            feh_min=feh_min,
            feh_max=feh_max,
            equal_frequency=equal_frequency,  # Equal number of stars per bin
            iso_colname_dict=iso_colname_dict
        )
    else:
        binned_data = multi_fitter.bin_data_by_metallicity(
            data,
            feh_column=feh_column,
            n_feh_bins=n_feh_bins,
            feh_min=feh_min,
            feh_max=feh_max,
            equal_frequency=equal_frequency  # Equal number of stars per bin
        )
        binned_iso = None

    # Run fitting for all bins
    print("4. Fitting all metallicity bins (broken power law)...")
    print("   This may take several minutes...")
    multi_fitter.fit_all_bins(
        binned_data,
        outlier_kappa=outlier_kappa,
        outlier_kappa_scale=outlier_kappa_scale,
        num_warmup=num_warmup,   # Reduced for demo
        num_samples=num_samples,  # Reduced for demo
        num_chains=num_chains,      # Reduced for demo
        seed=seed,
        a_prior_range=(-15,15), b_prior_range=(-80,5),
    )

    # Save samples
    print("5. Saving broken power law samples...")
    multi_fitter.save_all_samples(output_dir=output_dir, prefix='mcmc')

    # Plot results
    print("6. Plotting broken power law results...")
    multi_fitter.plot_all_results(binned_data=binned_data, binned_iso=binned_iso, iso_colname_dict=iso_colname_dict, output_dir=output_dir)
    multi_fitter.plot_comparison(output_dir=output_dir, data=binned_data, iso_data=binned_iso, iso_colname_dict=iso_colname_dict)

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


def test_polynomial_model(data, output_dir,
                          order=3, absg_min=3.0, absg_max=14.0,
                          uncertainty_model='rice',
                          feh_column='feh', n_feh_bins=1, feh_min=-1, feh_max=0.6, equal_frequency=False,
                          num_warmup=800, num_samples=2000, num_chains=2,
                          mass_min=0.05, mass_max=2.0,
                          poly_deriv_penalty_strength=10.0, poly_coeff_prior_scale=5.0,
                          outlier_kappa=None, outlier_kappa_scale=0.05, seed=42, fit_outlier_params=False, iso_data=None, iso_colname_dict = {'absg':'absg', 'mass':'mass', 'feh':'MH'}):
    """Test the polynomial model using the new MultiMetallicityFitter."""
    print("\n" + "="*50)
    print("TESTING POLYNOMIAL MODEL")
    print("="*50)

    print(f"1. Using polynomial of order {order}")

    print("2. Initializing polynomial multi-metallicity fitter...")
    multi_fitter = MultiMetallicityFitter(
        model_type='polynomial',
        absg_min=absg_min,
        absg_max=absg_max,
        mass_min=mass_min,
        mass_max=mass_max,
        uncertainty_model=uncertainty_model,
        f_outlier=0.1,
        outlier_u0=30,
        outlier_sigma=15,
        outlier_kappa=outlier_kappa,
        outlier_kappa_scale=outlier_kappa_scale,
        poly_order=order,
        poly_deriv_penalty_strength=poly_deriv_penalty_strength,
        poly_coeff_prior_scale=poly_coeff_prior_scale,
        fit_outlier_params=fit_outlier_params
    )

    print("3. Binning data by metallicity...")
    if iso_data is not None:
        binned_data, binned_iso = multi_fitter.bin_data_by_metallicity(
            data,
            iso_data=iso_data,
            feh_column=feh_column,
            n_feh_bins=n_feh_bins,
            feh_min=feh_min,
            feh_max=feh_max,
            equal_frequency=equal_frequency,
            iso_colname_dict=iso_colname_dict
        )
    else:
        binned_data = multi_fitter.bin_data_by_metallicity(
            data,
            feh_column=feh_column,
            n_feh_bins=n_feh_bins,
            feh_min=feh_min,
            feh_max=feh_max,
            equal_frequency=equal_frequency
        )
        binned_iso = None

    print("4. Fitting all metallicity bins (polynomial)...")
    print("   This may take several minutes...")
    multi_fitter.fit_all_bins(
        binned_data,
        outlier_kappa=outlier_kappa,
        outlier_kappa_scale=outlier_kappa_scale,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        seed=seed,
    )

    print("5. Saving polynomial samples...")
    multi_fitter.save_all_samples(output_dir=output_dir, prefix='mcmc')

    print("6. Plotting polynomial results...")
    multi_fitter.plot_all_results(binned_data=binned_data, binned_iso=binned_iso, iso_colname_dict=iso_colname_dict, output_dir=output_dir)
    multi_fitter.plot_comparison(output_dir=output_dir, data=binned_data, iso_data=binned_iso, iso_colname_dict=iso_colname_dict)

    print("\n7. Polynomial Summary Statistics:")
    for bin_idx, fitter in multi_fitter.fitters.items():
        feh_center = multi_fitter.feh_bin_centers[bin_idx]
        print(f"   Bin {bin_idx} ([Fe/H]={feh_center:.2f}]):")
        print(f"     Number of systems: {len(binned_data[bin_idx])}")

    return multi_fitter


def test_differencepoly_model(
    data,
    iso_data,
    output_dir,
    order=3,
    absg_min=3.0,
    absg_max=14.0,
    iso_absg_bin_width=0.05,
    uncertainty_model="rice",
    feh_column="feh",
    n_feh_bins=1,
    feh_min=-1,
    feh_max=0.6,
    equal_frequency=False,
    num_warmup=800,
    num_samples=2000,
    num_chains=2,
    mass_min=0.05,
    mass_max=2.0,
    outlier_kappa=None,
    outlier_kappa_scale=0.05,
    seed=42,
    fit_outlier_params=False,
    iso_colname_dict={"absg": "absg", "mass": "mass", "feh": "MH"},
):
    """
    Test the DifferencePolyMLR model (polynomial residual relative to an isochrone).

    Fits, per metallicity bin:
        log10 m_dyn(M_G) = log10 m_iso(M_G) + P(M_G)
    """
    if iso_data is None:
        raise ValueError("DifferencePolyMLR requires `iso_data`.")

    def _bin_isochrone_mass_by_absg(iso_table, absg_col, mass_col, bin_width, absg_min_val, absg_max_val):
        absg = np.asarray(iso_table[absg_col], dtype=float)
        mass = np.asarray(iso_table[mass_col], dtype=float)

        finite = np.isfinite(absg) & np.isfinite(mass)
        absg = absg[finite]
        mass = mass[finite]

        if absg.size == 0:
            raise ValueError("Isochrone data has no finite (absg, mass) points after filtering.")

        if bin_width <= 0:
            raise ValueError("iso_absg_bin_width must be > 0.")

        edges = np.arange(absg_min_val, absg_max_val + bin_width, bin_width)
        if edges.size < 2:
            raise ValueError("Invalid binning configuration for isochrone absg binning.")

        centers = edges[:-1] + 0.5 * bin_width
        bin_index = np.digitize(absg, edges) - 1

        mean_mass = np.full(centers.shape[0], np.nan, dtype=float)
        for i in range(centers.shape[0]):
            in_bin = bin_index == i
            if np.any(in_bin):
                mean_mass[i] = np.nanmean(mass[in_bin])

        valid = np.isfinite(mean_mass)
        return centers[valid], mean_mass[valid]

    def _plot_differencepoly_comparison(
        fitters_by_bin,
        binned_iso_raw,
        binned_iso_curve,
        feh_bin_edges,
        absg_min_val,
        absg_max_val,
        mass_min_val,
        mass_max_val,
        output_dir_val,
        n_feh_bins_val,
        outlier_tag_val,
        uncertainty_model_val,
        iso_col_dict,
    ):
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm

        if not fitters_by_bin:
            return

        os.makedirs(output_dir_val, exist_ok=True)
        out_path = os.path.join(
            output_dir_val,
            f"feh_comparison_differencepoly_unc-{uncertainty_model_val}_feh{n_feh_bins_val}bins_{outlier_tag_val}.png",
        )

        fig, ax = plt.subplots(figsize=(12, 8))
        colors = cm.viridis(np.linspace(0, 1, len(fitters_by_bin)))
        absg_range = np.linspace(absg_min_val, absg_max_val, 1000)

        for (bin_idx, fitter), color in zip(fitters_by_bin.items(), colors):
            if fitter.fit_outlier_params:
                coeff_samples = fitter.samples[:, :-3]
            else:
                coeff_samples = fitter.samples

            all_masses = fitter.mass_from_absg(absg_range, coeff_samples)
            lower_m, _, upper_m = np.percentile(all_masses, [16, 50, 84], axis=0)

            median_params = np.median(coeff_samples, axis=0)
            best_fit = fitter.mass_from_absg(absg_range, median_params)

            label = f"[Fe/H]=[{feh_bin_edges[bin_idx]:.2f}, {feh_bin_edges[bin_idx+1]:.2f}]"
            ax.fill_between(absg_range, lower_m, upper_m, color=color, alpha=0.3)
            ax.plot(absg_range, best_fit, color=color, linestyle="-.", linewidth=2, label=label)

            # Raw isochrone scatter in the background
            # if binned_iso_raw is not None:
            #     iso_bin = binned_iso_raw.get(bin_idx)
            #     if iso_bin is not None:
            #         try:
            #             ax.plot(
            #                 np.array(iso_bin[iso_col_dict["absg"]]),
            #                 np.array(iso_bin[iso_col_dict["mass"]]),
            #                 color=color,
            #                 linestyle=":",
            #                 linewidth=2.5,
            #                 alpha=1,
            #                 zorder=0,
            #                 label=label
            #             )
            #         except Exception:
            #             pass

            # Binned isochrone curve used in the model (for reference)
            if binned_iso_curve is not None and bin_idx in binned_iso_curve:
                label_iso = f"[Fe/H]=[{feh_bin_edges[bin_idx]:.2f}, {feh_bin_edges[bin_idx+1]:.2f}] Isochrone"
                try:
                    iso_curve = binned_iso_curve[bin_idx]
                    ax.plot(
                        iso_curve["absg"],
                        iso_curve["mass"],
                        color=color,
                        linewidth=2,
                        alpha=0.8,
                        linestyle=":",
                        label=label_iso,
                        zorder=0,
                    )
                except Exception:
                    pass

        ax.set_yscale("log")
        ax.set_xlim(absg_min_val, absg_max_val)
        ax.set_ylim(mass_min_val, mass_max_val)
        ax.set_xlabel("$M_{\\mathrm{G}}$ [mag]", fontsize=14)
        ax.set_ylabel("Mass [$M_{\\odot}$]", fontsize=14)
        ax.invert_xaxis()
        ax.legend(fontsize=10, loc="best")
        ax.set_title(
            f"Difference-Polynomial MLR across metallicities ({uncertainty_model_val.capitalize()} uncertainty)",
            fontsize=16,
        )
        plt.tight_layout()
        plt.savefig(out_path, dpi=300)
        plt.close(fig)
        print(f"Saved comparison plot to {out_path}")

    print("\n" + "=" * 50)
    print("TESTING DIFFERENCE-POLYNOMIAL MODEL")
    print("=" * 50)
    print(f"1. Using residual polynomial of order {order}")
    print(f"   Isochrone pre-binning: ΔM_G={iso_absg_bin_width:.3f} mag, mean mass per bin")

    # Reuse the binning utility (including isochrone binning)
    binning = MultiMetallicityFitter(
        model_type="polynomial",
        absg_min=absg_min,
        absg_max=absg_max,
        mass_min=mass_min,
        mass_max=mass_max,
        uncertainty_model=uncertainty_model,
        outlier_kappa=outlier_kappa,
        outlier_kappa_scale=outlier_kappa_scale,
        fit_outlier_params=fit_outlier_params,
        poly_order=order,
    )

    print("2. Binning data (and isochrone) by metallicity...")
    binned_data, binned_iso = binning.bin_data_by_metallicity(
        data,
        iso_data=iso_data,
        feh_column=feh_column,
        n_feh_bins=n_feh_bins,
        feh_min=feh_min,
        feh_max=feh_max,
        equal_frequency=equal_frequency,
        iso_colname_dict=iso_colname_dict,
    )

    fitters = {}
    iso_curve_by_bin = {}
    outlier_tag = "outlierfit" if fit_outlier_params else "outlierfixed"

    print("3. Fitting each metallicity bin (difference-poly)...")
    for bin_idx in sorted(binned_data.keys()):
        bin_data = binned_data[bin_idx]
        iso_bin = binned_iso[bin_idx]

        feh_center = binning.feh_bin_centers[bin_idx] if binning.feh_bin_centers is not None else float("nan")
        print(f"   - Bin {bin_idx} ([Fe/H]={feh_center:+.2f}): N={len(bin_data)}")

        absg_grid, mass_grid = _bin_isochrone_mass_by_absg(
            iso_bin,
            absg_col=iso_colname_dict["absg"],
            mass_col=iso_colname_dict["mass"],
            bin_width=iso_absg_bin_width,
            absg_min_val=absg_min,
            absg_max_val=absg_max,
        )
        if absg_grid.size < 2:
            raise ValueError(
                f"Isochrone bin {bin_idx} produced <2 binned points; "
                "increase isochrone sample size or increase iso_absg_bin_width."
            )

        iso_model = IsochroneMassModel(
            absg_min=absg_min,
            absg_max=absg_max,
            absg_grid=absg_grid,
            mass_grid=mass_grid,
            mass_min=mass_min,
        )
        iso_curve_by_bin[bin_idx] = {"absg": absg_grid, "mass": mass_grid}

        fitter = DifferencePolyMLR(
            order=order,
            isochrone_model=iso_model,
            mass_min=mass_min,
            mass_max=mass_max,
            absg_min=absg_min,
            absg_max=absg_max,
            uncertainty_model=uncertainty_model,
            f_outlier=0.1,
            outlier_u0=30,
            outlier_sigma=15,
            outlier_kappa=outlier_kappa,
            outlier_kappa_scale=outlier_kappa_scale,
            fit_outlier_params=fit_outlier_params,
            deriv_penalty_strength=0.0,
        )

        fitter.set_data(
            u_values=bin_data["u"],
            u_sigma_values=bin_data["u_sigma"],
            absg1_values=bin_data["absg1"],
            absg2_values=bin_data["absg2"],
            outlier_kappa=outlier_kappa,
            outlier_kappa_scale=outlier_kappa_scale,
        )

        fitter.run_numpyro(
            num_warmup=num_warmup,
            num_samples=num_samples,
            num_chains=num_chains,
            seed=seed + int(bin_idx),
        )

        suffix = f"_model-differencepoly_unc-{uncertainty_model}_feh{n_feh_bins}bins_{outlier_tag}_bin{bin_idx}_feh{feh_center:+.2f}"

        print("   - Saving samples and plots...")
        np.savetxt(os.path.join(output_dir, f"mcmc{suffix}.txt"), fitter.samples)
        fitter.plot_results(output_dir=output_dir, output_suffix=suffix)
        fitter.plot_fitting_results(
            data=bin_data,
            output_dir=output_dir,
            output_suffix=suffix,
            isochrone_data_feh=iso_bin,
            iso_colname_dict=iso_colname_dict,
            isochrone_curve_data_feh={
                iso_colname_dict["absg"]: absg_grid,
                iso_colname_dict["mass"]: mass_grid,
            },
        )

        fitters[bin_idx] = fitter

    _plot_differencepoly_comparison(
        fitters_by_bin=fitters,
        binned_iso_raw=binned_iso,
        binned_iso_curve=iso_curve_by_bin,
        feh_bin_edges=binning.feh_bin_edges,
        absg_min_val=absg_min,
        absg_max_val=absg_max,
        mass_min_val=mass_min,
        mass_max_val=mass_max,
        output_dir_val=output_dir,
        n_feh_bins_val=n_feh_bins,
        outlier_tag_val=outlier_tag,
        uncertainty_model_val=uncertainty_model,
        iso_col_dict=iso_colname_dict,
    )

    return fitters




def main():
    """Main multi-metallicity fitting routine."""
    print("=" * 70)
    print("Multi-Metallicity Binary Mass Fitting Example")
    print("Testing Both Non-Parametric and Broken Power Law Models")
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

    # Using a smaller subset for faster testing
    # indices = np.random.choice(len(data), size=1000, replace=False)
    # data = data[indices]
    print(f"   Using {len(data)} systems for testing")

    # Set output directory
    output_dir = 'results/data_nonparam_prior_kappac0p3'
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n2. Results will be saved to: {output_dir}")

    # Test non-parametric model with isochrone mass prior
    nonparametric_fitter = test_nonparametric_model(data, output_dir, iso_data=iso_data,
                                                    n_absg_bins=10, absg_min=3.5, absg_max=14.0,
                                                    uncertainty_model='rice',
                                                    feh_column='feh_jcaps_1', n_feh_bins=3, feh_min=-1, feh_max=0.6, equal_frequency=False,
                                                    gamma=0, num_warmup=500, num_samples=3000, num_chains=2,
                                                    mass_min=0.05, mass_max=1.5,
                                                    seed=32, fit_outlier_params=True, outlier_kappa_scale=0.4, iso_colname_dict=iso_colname_dict)
    

    # Test broken power law model
    # broken_powerlaw_models = test_broken_powerlaw_model(data, output_dir,
    #                                                     n_segments=2, absg_min=3.0, absg_max=14.0,
    #                                                     uncertainty_model='gaussian',
    #                                                     feh_column='feh_jcaps_1', n_feh_bins=1, feh_min=-1.5, feh_max=0.6, equal_frequency=False,
    #                                                     num_warmup=500, num_samples=2000, num_chains=2,
    #                                                     seed=8,
    #                                                     fit_outlier_params=True, iso_data=iso_data, iso_colname_dict=iso_colname_dict)

    # Test polynomial model
    # polynomial_models = test_polynomial_model(data, output_dir,
    #                                           order=3, absg_min=3, absg_max=14.0,
    #                                           uncertainty_model='rice',
    #                                           feh_column='feh_jcaps_1', n_feh_bins=4, feh_min=-1.5, feh_max=0.6, equal_frequency=False,
    #                                           num_warmup=500, num_samples=3000, num_chains=2,
    #                                           mass_min=0.05, mass_max=1.5,
    #                                           poly_deriv_penalty_strength=10.0, poly_coeff_prior_scale=5.0,
    #                                           seed=14, fit_outlier_params=True, iso_data=iso_data, iso_colname_dict=iso_colname_dict)

    # Test difference-polynomial model (residual relative to isochrone)
    # differencepoly_models = test_differencepoly_model(data, iso_data=iso_data, output_dir=output_dir, order=2, 
    #                                                   absg_min=3.5, absg_max=14.0, iso_absg_bin_width=0.25, uncertainty_model="rice", feh_column="feh_jcaps_1", n_feh_bins=3, feh_min=-1, feh_max=0.6,
    #                                                   equal_frequency=False, 
    #                                                   num_warmup=500, num_samples=3000, num_chains=2,
    #                                                   mass_min=0.05, mass_max=1.5,
    #                                                   seed=34, fit_outlier_params=True, outlier_kappa_scale=0.35, iso_colname_dict=iso_colname_dict,)

    # Final summary
    # print("\n" + "=" * 70)
    # print("FINAL SUMMARY")
    # print("=" * 70)
    # print(f"✓ Non-parametric model test completed")
    # print(f"  - Number of metallicity bins: {len(nonparametric_fitter.fitters)}")

    print(f"\n✓ Broken power law model test completed")
    # print(f"  - Number of metallicity bins fitted: {len(broken_powerlaw_models)}")

    print(f"\nGenerated files in {output_dir}:")
    print("  - mcmc_<model>_unc-<unc>_feh<nbin>bins_<outlier>_bin*.txt")
    print("  - corner_<model>_unc-<unc>_feh<nbin>bins_<outlier>_bin*.png")
    print("  - fit_<model>_unc-<unc>_feh<nbin>bins_<outlier>_bin*.png")
    print("  - feh_comparison_<model>_unc-<unc>_feh<nbin>bins_<outlier>.png")

    print("\n" + "=" * 70)
    print("Multi-metallicity example completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
