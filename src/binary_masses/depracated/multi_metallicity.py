"""Fit mass-luminosity relations across metallicity bins."""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from .nonparametric import NonParametricMLR
from .broken_powerlaw import BrokenPowerLawMLR
from ..polynomial import PolynomialMLR

class MultiMetallicityFitter:
    """
    Wrapper class to fit multiple metallicity bins.

    This class provides a convenient interface for fitting mass-luminosity
    relations in different metallicity bins using either non-parametric
    or broken power-law parameterizations.
    """

    def __init__(self, model_type='nonparametric', n_absg_bins=10, absg_min=4.0, absg_max=12.0, mass_min=0.05, mass_max=2.0,
                 uncertainty_model='rice', f_outlier=0, outlier_u0=30, outlier_sigma=15, outlier_kappa=None,
                 outlier_kappa_scale=0.05,
                 n_segments=3, fit_outlier_params=False,
                 poly_order=3, poly_pivot=None,
                 poly_deriv_penalty_strength=10.0, poly_coeff_prior_scale=5.0):

        """
        Initialize the multi-metallicity fitter.

        Parameters
        ----------
        model_type : str
            Type of model to use: 'nonparametric' or 'broken_powerlaw' (default: 'nonparametric')
        n_absg_bins : int
            Number of bins to split the absg range into (only for nonparametric model)
        absg_min : float
            Minimum absg value (only for nonparametric model)
        absg_max : float
            Maximum absg value (only for nonparametric model)
        uncertainty_model : str
            Uncertainty model to use: 'rice' or 'gaussian' (default: 'rice')
        f_outlier : float
            Fraction of outliers (default: 0)
        outlier_u0 : float
            Outlier distribution center (default: 30)
        outlier_sigma : float
            Outlier distribution width (default: 15)
        outlier_kappa : float
            Optional fixed concentration for the Beta prior on f_outlier; if None, use sample-size scaling.
        outlier_kappa_scale : float
            Scaling factor c for kappa = c * N when outlier_kappa is None (default: 0.05)
        n_segments : int
            Number of segments for broken power law model (default: 3)
        fit_outlier_params : bool
            Whether to fit outlier parameters in MCMC (default: True)
        """
        self.model_type = model_type.lower()
        if self.model_type not in ['nonparametric', 'broken_powerlaw', 'polynomial']:
            raise ValueError("model_type must be 'nonparametric', 'broken_powerlaw' or 'polynomial'")

        self.n_absg_bins = n_absg_bins
        self.absg_min = absg_min
        self.absg_max = absg_max
        self.mass_min = mass_min
        self.mass_max = mass_max
        self.uncertainty_model = uncertainty_model
        self.f_outlier = f_outlier
        self.f_good = 1 - f_outlier
        self.outlier_u0 = outlier_u0
        self.outlier_sigma = outlier_sigma
        self.outlier_kappa = outlier_kappa
        self.outlier_kappa_scale = outlier_kappa_scale

        # Set default break points for broken power law
        self.n_segments = n_segments
        self.fit_outlier_params = fit_outlier_params

        # Polynomial model settings
        self.poly_order = poly_order
        self.poly_pivot = poly_pivot
        self.poly_deriv_penalty_strength = poly_deriv_penalty_strength
        self.poly_coeff_prior_scale = poly_coeff_prior_scale

        self.fitters = {}  # Dictionary to store fitters for each metallicity bin
        self.feh_bin_edges = None
        self.feh_bin_centers = None

    def _n_feh_bins(self):
        """Return number of metallicity bins if available."""
        return len(self.feh_bin_centers) if self.feh_bin_centers is not None else 'unknown'

    def _outlier_tag(self):
        """Human-friendly tag indicating whether outliers are fitted."""
        return 'outlierfit' if self.fit_outlier_params else 'outlierfixed'

    def _build_suffix(self, bin_idx=None):
        """
        Build a standardized metadata suffix shared by all outputs.

        Includes model, uncertainty model, metallicity bin count, outlier flag,
        and (optionally) the specific metallicity bin information.
        """
        n_bins = self._n_feh_bins()
        suffix = f'_model-{self.model_type}_unc-{self.uncertainty_model}_feh{n_bins}bins_{self._outlier_tag()}'
        if bin_idx is not None and self.feh_bin_centers is not None:
            suffix += f'_bin{bin_idx}_feh{self.feh_bin_centers[bin_idx]:+.2f}'
        return suffix

    def bin_data_by_metallicity(self, data, feh_column='feh', n_feh_bins=5,
                                iso_data = None, iso_colname_dict = {'absg':'absg', 'mass':'mass', 'feh':'MH'},
                                feh_min=None, feh_max=None, equal_frequency=False):
        """
        Bin the data by metallicity using either equal-width or equal-frequency binning.

        Parameters
        ----------
        data : astropy.table.Table or dict
            Data containing metallicity and other columns
        feh_column : str
            Name of the metallicity column
        n_feh_bins : int
            Number of metallicity bins
        feh_min, feh_max : float
            Metallicity range (if None, use data range)
        equal_frequency : bool
            If True, use equal-frequency binning; if False, use equal-width binning

        Returns
        -------
        binned_data : dict
            Dictionary with keys as bin indices and values as data subsets
        """
        # Extract feh values based on data type
        if hasattr(data, 'colnames'):  # astropy.Table
            feh_values = data[feh_column]
            is_table = True
        elif isinstance(data, dict):  # dict
            feh_values = np.array(data[feh_column])
            is_table = False
        else:
            raise ValueError("data must be astropy.Table or dict")

        if feh_min is None:
            feh_min = np.min(feh_values)
        if feh_max is None:
            feh_max = np.max(feh_values)

        if equal_frequency:
            # Equal-frequency binning: bins with approximately equal number of stars
            sorted_indices = np.argsort(feh_values)
            sorted_feh = feh_values[sorted_indices]
            bin_size = len(feh_values) // n_feh_bins
            bin_edges = np.zeros(n_feh_bins+1)

            bin_edges[0] = feh_min
            bin_edges[-1] = feh_max

            for i in range(n_feh_bins-1):
                start_idx = (i+1) * bin_size
                bin_edges[i+1] = sorted_feh[start_idx]
        else:
            # Equal-width binning: bins with equal metallicity intervals
            bin_edges = np.linspace(feh_min, feh_max, n_feh_bins + 1)

        self.feh_bin_edges = bin_edges
        self.feh_bin_centers = (self.feh_bin_edges[:-1] + self.feh_bin_edges[1:]) / 2

        binned_data = {}
        for i in range(n_feh_bins):
            mask = (feh_values >= self.feh_bin_edges[i]) & (feh_values < self.feh_bin_edges[i+1])
            if i == n_feh_bins - 1:  # Include upper edge in last bin
                mask = (feh_values >= self.feh_bin_edges[i]) & (feh_values <= self.feh_bin_edges[i+1])

            # Handle both Table and dict data types
            if is_table:
                binned_data[i] = data[mask]
            else:
                binned_data[i] = {key: values[mask] if isinstance(values, np.ndarray) else np.array(values)[mask]
                                 for key, values in data.items()}

            star_count = len(binned_data[i]) if is_table else len(binned_data[i]['feh'])
            print(f"Metallicity bin {i} ([{self.feh_bin_edges[i]:.2f}, {self.feh_bin_edges[i+1]:.2f}]): {star_count} stars")

        if iso_data is not None:
            binned_iso = {}
            # Bin isochrone data similarly
            if hasattr(iso_data, 'colnames'):
                iso_feh_values = iso_data[iso_colname_dict['feh']]
                is_iso_table = True
            elif isinstance(iso_data, dict):
                iso_feh_values = np.array(iso_data[iso_colname_dict['feh']])
                is_iso_table = False
            else:
                raise ValueError("iso_data must be astropy.Table or dict")

            for i in range(n_feh_bins):
                mask = (iso_feh_values >= self.feh_bin_edges[i]) & (iso_feh_values < self.feh_bin_edges[i+1])
                if i == n_feh_bins - 1:
                    mask = (iso_feh_values >= self.feh_bin_edges[i]) & (iso_feh_values <= self.feh_bin_edges[i+1])

                if is_iso_table:
                    binned_iso[i] = iso_data[mask]
                else:
                    binned_iso[i] = {key: values[mask] if isinstance(values, np.ndarray) else np.array(values)[mask]
                                     for key, values in iso_data.items()}
            return binned_data, binned_iso
        
        else:
            return binned_data

    def fit_all_bins(self, binned_data, u_column='u', u_sigma_column='u_sigma',
                    absg1_column='absg1', absg2_column='absg2',
                    a_prior_range=(-1,20), b_prior_range=(-50,5), gamma=np.inf, outlier_kappa=None,
                    outlier_kappa_scale=None, num_warmup=1000, num_samples=2000, num_chains=4, seed=None):
        """
        Fit all metallicity bins.

        Parameters
        ----------
        binned_data : dict
            Dictionary with bin indices as keys and data as values
        u_column, u_sigma_column, absg1_column, absg2_column : str
            Column names in the data
        gamma : float
            Regularization parameter (only for nonparametric model)
        outlier_kappa : float
            Concentration parameter for the Beta prior on f_outlier
        outlier_kappa_scale : float
            Scaling factor c when computing kappa = c * N if outlier_kappa is None
        mass_min, mass_max : float
            Mass bounds
        num_warmup, num_samples, num_chains : int
            NumPyro parameters
        seed : int
            Random seed
        """
        for bin_idx, data in binned_data.items():
            if len(data) == 0:
                print(f"Skipping metallicity bin {bin_idx} (no data)")
                continue

            print(f"\n{'='*60}")
            print(f"Fitting metallicity bin {bin_idx} using {self.model_type} model")
            print(f"[Fe/H] range: [{self.feh_bin_edges[bin_idx]:.2f}, {self.feh_bin_edges[bin_idx+1]:.2f}]")
            print(f"Number of stars: {len(data)}")
            print(f"{'='*60}\n")

            # Extract data
            u_values = data[u_column].data
            u_sigma_values = data[u_sigma_column].data if u_sigma_column in data.colnames else None
            absg1_values = data[absg1_column].data
            absg2_values = data[absg2_column].data

            if self.model_type == 'nonparametric':
                # Create non-parametric fitter for this bin
                effective_kappa = self.outlier_kappa if outlier_kappa is None else outlier_kappa
                effective_kappa_scale = self.outlier_kappa_scale if outlier_kappa_scale is None else outlier_kappa_scale
                true_mass_data = None
                if hasattr(data, 'colnames'):
                    if {'m1', 'm2', 'absg1', 'absg2'}.issubset(set(data.colnames)):
                        true_mass_data = data
                elif isinstance(data, dict):
                    if all(key in data for key in ['m1', 'm2', 'absg1', 'absg2']):
                        true_mass_data = data

                fitter = NonParametricMLR(
                    n_bins=self.n_absg_bins,
                    absg_min=self.absg_min,
                    absg_max=self.absg_max,
                    mass_min=self.mass_min,
                    mass_max=self.mass_max,
                    uncertainty_model=self.uncertainty_model,
                    f_outlier=self.f_outlier,
                    outlier_u0=self.outlier_u0,
                    outlier_sigma=self.outlier_sigma,
                    outlier_kappa=effective_kappa,
                    outlier_kappa_scale=effective_kappa_scale,
                    fit_outlier_params=self.fit_outlier_params
                )

                # Set data
                fitter.set_data(
                    u_values=u_values,
                    u_sigma_values=u_sigma_values,
                    absg1_values=absg1_values,
                    absg2_values=absg2_values,
                    gamma=gamma,
                    outlier_kappa=effective_kappa,
                    outlier_kappa_scale=effective_kappa_scale,
                    true_mass_data=true_mass_data
                )

                # Run fitting
                fitter.run_numpyro(
                    num_warmup=num_warmup,
                    num_samples=num_samples,
                    num_chains=num_chains,
                    seed=seed
                )

            elif self.model_type == 'broken_powerlaw':
                # Create broken power-law fitter for this bin
                effective_kappa = self.outlier_kappa if outlier_kappa is None else outlier_kappa
                effective_kappa_scale = self.outlier_kappa_scale if outlier_kappa_scale is None else outlier_kappa_scale
                fitter = BrokenPowerLawMLR(
                    n_segments=self.n_segments,
                    absg_min=self.absg_min,
                    absg_max=self.absg_max,
                    uncertainty_model=self.uncertainty_model,
                    f_outlier=self.f_outlier,
                    outlier_u0=self.outlier_u0,
                    outlier_sigma=self.outlier_sigma,
                    outlier_kappa=effective_kappa,
                    outlier_kappa_scale=effective_kappa_scale,
                    fit_outlier_params=self.fit_outlier_params
                )

                # Set data
                fitter.set_data(
                    u_values=u_values,
                    u_sigma_values=u_sigma_values,
                    absg1_values=absg1_values,
                    absg2_values=absg2_values,
                    outlier_kappa=effective_kappa,
                    outlier_kappa_scale=effective_kappa_scale
                )

                # Run fitting
                fitter.run_numpyro(
                    num_warmup=num_warmup,
                    num_samples=num_samples,
                    num_chains=num_chains,
                    seed=seed,
                    a_prior_range=a_prior_range, b_prior_range=b_prior_range,
                )

            elif self.model_type == 'polynomial':
                effective_kappa = self.outlier_kappa if outlier_kappa is None else outlier_kappa
                effective_kappa_scale = self.outlier_kappa_scale if outlier_kappa_scale is None else outlier_kappa_scale
                fitter = PolynomialMLR(
                    order=self.poly_order,
                    absg_min=self.absg_min,
                    absg_max=self.absg_max,
                    mass_min=self.mass_min,
                    mass_max=self.mass_max,
                    pivot=self.poly_pivot,
                    uncertainty_model=self.uncertainty_model,
                    f_outlier=self.f_outlier,
                    outlier_u0=self.outlier_u0,
                    outlier_sigma=self.outlier_sigma,
                    outlier_kappa=effective_kappa,
                    outlier_kappa_scale=effective_kappa_scale,
                    fit_outlier_params=self.fit_outlier_params,
                    deriv_penalty_strength=self.poly_deriv_penalty_strength,
                    coeff_prior_scale=self.poly_coeff_prior_scale,
                )

                fitter.set_data(
                    u_values=u_values,
                    u_sigma_values=u_sigma_values,
                    absg1_values=absg1_values,
                    absg2_values=absg2_values,
                    outlier_kappa=effective_kappa,
                    outlier_kappa_scale=effective_kappa_scale
                )

                fitter.run_numpyro(
                    num_warmup=num_warmup,
                    num_samples=num_samples,
                    num_chains=num_chains,
                    seed=seed,
                )

            # Store fitter
            self.fitters[bin_idx] = fitter

    def plot_all_results(self, binned_data, output_dir='.', binned_iso=None, iso_colname_dict = {'absg':'absg', 'mass':'mass', 'feh':'MH'}):
        """
        Plot results for all metallicity bins.

        Parameters
        ----------
        output_dir : str
            Output directory for plots
        data : astropy.Table or dict, optional
            Input data containing 'm1', 'm2', 'absg1', 'absg2' columns for true mass scatter points
            If provided, will filter data by metallicity bin for each plot
        """
        os.makedirs(output_dir, exist_ok=True)

        for bin_idx, fitter in self.fitters.items():
            suffix = self._build_suffix(bin_idx)
            bin_data = binned_data[bin_idx] if binned_data is not None else None
            bin_iso = binned_iso[bin_idx] if binned_iso is not None else None

            if bin_data is not None and len(bin_data) == 0:
                print(f"Skipping metallicity bin {bin_idx} (no data)")
                continue

            if self.model_type == 'nonparametric':
                # Non-parametric model plotting
                fitter.plot_results(output_dir=output_dir, output_suffix=suffix)
                fitter.plot_fitting_results(data=bin_data, isochrone_data_feh=bin_iso, output_dir=output_dir, output_suffix=suffix, iso_colname_dict=iso_colname_dict)

            elif self.model_type == 'broken_powerlaw':
                # Broken power law model plotting
                fitter.plot_results(output_dir=output_dir, output_suffix=suffix)
                fitter.plot_fitting_results(data=bin_data, isochrone_data_feh=bin_iso, output_dir=output_dir, output_suffix=suffix, iso_colname_dict=iso_colname_dict)

            elif self.model_type == 'polynomial':
                fitter.plot_results(output_dir=output_dir, output_suffix=suffix)
                fitter.plot_fitting_results(data=bin_data, isochrone_data_feh=bin_iso, output_dir=output_dir, output_suffix=suffix, iso_colname_dict=iso_colname_dict)
    
    def plot_comparison(self, output_path=None, data=None, iso_data=None, output_dir='.', iso_colname_dict = {'absg':'absg', 'mass':'mass', 'feh':'MH'}):
        """
        Plot comparison of mass-luminosity relations across all metallicity bins.

        Parameters
        ----------
        output_path : str, optional
            Output file path. If None, a standardized path will be created in output_dir.
        output_dir : str
            Directory to write plots when output_path is None.
        data : binned_data : dict
        true_mass_funcs : dict or callable
            True mass functions for each bin (for validation)
        """
        if len(self.fitters) == 0:
            print("No fitted results to plot")
            return

        if output_path is None:
            os.makedirs(output_dir, exist_ok=True)
            n_bins = self._n_feh_bins()
            filename = f'feh_comparison_{self.model_type}_unc-{self.uncertainty_model}_feh{n_bins}bins_{self._outlier_tag()}.png'
            output_path = os.path.join(output_dir, filename)
        else:
            output_parent = os.path.dirname(output_path)
            if output_parent:
                os.makedirs(output_parent, exist_ok=True)

        fig, ax = plt.subplots(figsize=(12, 8))

        # Color map for different metallicity bins
        colors = cm.viridis(np.linspace(0, 1, len(self.fitters)))
        
        for (bin_idx, fitter), color in zip(self.fitters.items(), colors):

            if self.model_type == 'nonparametric':
                # Non-parametric model: plot credible intervals at bin centers
                # Extract only mass bin parameters (exclude outlier parameters if present)
                if hasattr(fitter, 'fit_outlier_params') and fitter.fit_outlier_params:
                    mass_bins_samples = fitter.samples[:, :fitter.n_bins]
                else:
                    mass_bins_samples = fitter.samples

                percentiles = np.percentile(mass_bins_samples, [16, 50, 84], axis=0)
                lower, median, upper = percentiles[0], percentiles[1], percentiles[2]
                absg_centers = fitter.absg_bins
                yerr = np.vstack((median - lower, upper - median))

                label = f'[Fe/H]=[{self.feh_bin_edges[bin_idx]:.2f}, {self.feh_bin_edges[bin_idx+1]:.2f}]'
                # add edge color for better visibility
                ax.errorbar(
                    absg_centers,
                    median,
                    yerr=yerr,
                    fmt='o',
                    markeredgecolor='black',
                    markeredgewidth=1.5, 
                    markersize=15,
                    color=color,
                    ecolor=color,
                    elinewidth=4,
                    capsize=0,
                    label=label,
                )

            elif self.model_type == 'broken_powerlaw':
                # Broken power law model: plot magnitude vs mass
                absg_range = np.linspace(fitter.absg_min, fitter.absg_max, 1000)  # absg range?

                # Extract only main model parameters (exclude outlier parameters if present)
                if hasattr(fitter, 'fit_outlier_params') and fitter.fit_outlier_params:
                    main_params_samples = fitter.samples[:, :-3]  # Exclude 3 outlier params
                else:
                    main_params_samples = fitter.samples

                all_masses = fitter.mass_from_absg(absg_range, main_params_samples)
                lower_mags, median_mags, upper_mags = np.percentile(all_masses, [16, 50, 84], axis=0)

                # Best-fit line
                median_params = np.median(main_params_samples, axis=0)
                best_fit_mags = fitter.mass_from_absg(absg_range, median_params)

                # Plot
                label = f'[Fe/H]=[{self.feh_bin_edges[bin_idx]:.2f}, {self.feh_bin_edges[bin_idx+1]:.2f}]'
                ax.fill_between(absg_range, lower_mags, upper_mags, color=color, alpha=0.2)
                ax.plot(absg_range, best_fit_mags, color=color, linestyle='-.', linewidth=3, label=label)

                # Add vertical lines for break points
                for bp in fitter.break_points:
                    ax.axhline(y=bp, color='gray', linestyle='--', alpha=0.5)

            elif self.model_type == 'polynomial':
                absg_range = np.linspace(fitter.poly_model.absg_min, fitter.poly_model.absg_max, 1000)

                # Extract only polynomial coefficients (exclude outlier parameters if present)
                if hasattr(fitter, 'fit_outlier_params') and fitter.fit_outlier_params:
                    coeff_samples = fitter.samples[:, :-3]  # Exclude 3 outlier params
                else:
                    coeff_samples = fitter.samples

                all_masses = fitter.mass_from_absg(absg_range, coeff_samples)
                lower_masses, median_masses, upper_masses = np.percentile(all_masses, [16, 50, 84], axis=0)

                median_params = np.median(coeff_samples, axis=0)
                best_fit_masses = fitter.mass_from_absg(absg_range, median_params)

                label = f'[Fe/H]=[{self.feh_bin_edges[bin_idx]:.2f}, {self.feh_bin_edges[bin_idx+1]:.2f}]'
                ax.fill_between(absg_range, lower_masses, upper_masses, color=color, alpha=0.2)
                ax.plot(absg_range, best_fit_masses, color=color, linestyle='-.', linewidth=3, label=label)

            # Plot data scatter points for this metallicity bin
            if data is not None:
                bin_data = data[bin_idx]
                try:
                    # Extract data columns
                    if hasattr(bin_data, 'colnames'):  # astropy.Table
                        m1 = np.array(bin_data['m1'])
                        m2 = np.array(bin_data['m2'])
                        absg1 = np.array(bin_data['absg1'])
                        absg2 = np.array(bin_data['absg2'])
                    elif isinstance(bin_data, dict):  # dict
                        m1 = np.array(bin_data['m1'])
                        m2 = np.array(bin_data['m2'])
                        absg1 = np.array(bin_data['absg1'])
                        absg2 = np.array(bin_data['absg2'])
                    else:
                        raise ValueError("data must be astropy.Table or dict")

                    # Plot both primary and secondary masses with low opacity
                    ax.scatter(absg1, m1, color=color, s=5, alpha=0.5, zorder=0)
                    ax.scatter(absg2, m2, color=color, s=5, alpha=0.5, zorder=0)
                except (KeyError, AttributeError) as e:
                    print(f"Could not extract masses from data for bin {bin_idx}: {e}")

            if iso_data is not None:
                bin_iso = iso_data[bin_idx]
                try:
                    if hasattr(bin_iso, 'colnames'):
                        iso_absg = np.array(bin_iso[iso_colname_dict['absg']])
                        iso_mass = np.array(bin_iso[iso_colname_dict['mass']])
                    elif isinstance(bin_iso, dict):
                        iso_absg = np.array(bin_iso[iso_colname_dict['absg']])
                        iso_mass = np.array(bin_iso[iso_colname_dict['mass']])
                    else:
                        raise ValueError("iso_data must be astropy.Table or dict")

                    ax.scatter(iso_absg, iso_mass, color=color, s=5, alpha=0.5, zorder=0)
                except (KeyError, AttributeError) as e:
                    print(f"Could not extract isochrone data for bin {bin_idx}: {e}")

        ax.set_yscale('log')
        ax.set_xlim(fitter.absg_min, fitter.absg_max)
        ax.set_ylim(fitter.mass_min * 0.8, fitter.mass_max * 1.2)
        ax.set_xlabel('$M_{\\mathrm{G}}$ [mag]', fontsize=14)
        ax.set_ylabel('Mass [$M_{\\odot}$]', fontsize=14)
        ax.invert_xaxis()  # Astronomical magnitude convention
        
        # Title and legend
        ax.legend(fontsize=10, loc='best')
        if self.model_type == 'nonparametric':
            ax.set_title(f'Mass-Absg Relations for Different Metallicities ({self.uncertainty_model.capitalize()} Distribution)', fontsize=16)
        else:
            ax.set_title(f'Broken Power Law MLR for Different Metallicities ({self.uncertainty_model.capitalize()} Distribution)', fontsize=16)

        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close(fig)
        print(f"Saved comparison plot to {output_path}")

    def save_all_samples(self, output_dir='.', prefix='mcmc'):
        """
        Save posterior samples for all bins.

        Parameters
        ----------
        output_dir : str
            Output directory for sample files
        prefix : str
            Prefix for output filenames
        """
        os.makedirs(output_dir, exist_ok=True)
        n_bins = self._n_feh_bins()
        outlier_tag = self._outlier_tag()

        for bin_idx, fitter in self.fitters.items():
            feh_tag = f'_bin{bin_idx}_feh{self.feh_bin_centers[bin_idx]:+.2f}' if self.feh_bin_centers is not None else f'_bin{bin_idx}'
            output_path = os.path.join(
                output_dir,
                f'{prefix}_{self.model_type}_unc-{self.uncertainty_model}_feh{n_bins}bins_{outlier_tag}{feh_tag}.txt'
            )
            np.savetxt(output_path, fitter.samples)
            print(f"Saved samples for bin {bin_idx} to {output_path}")

        return
