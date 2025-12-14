"""
Core Bayesian inference module for binary mass estimation

This module implements the main classes for estimating dynamical masses of binary stars
using Bayesian inference with Rice distribution uncertainties.
"""

import os
import numpy as np
import torch
from scipy import special
from scipy.stats import norm
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from typing import Union


def _configure_jax_gpu_fallback():
    """
    Configure JAX to prefer GPU but fall back to CPU if GPU is not available.

    Returns
    -------
    jax : module
        The configured JAX module
    """
    import warnings
    import sys
    import importlib

    # Try GPU first, but fall back to CPU if not available
    try:
        # Clear any existing JAX platform preference to allow auto-detection
        if 'JAX_PLATFORMS' in os.environ:
            del os.environ['JAX_PLATFORMS']

        # Try to import jax with auto-detection
        import jax
        devices = jax.devices()

        # If no GPU devices are available, switch to CPU
        if not any(device.platform == 'gpu' for device in devices):
            warnings.warn("No GPU devices available, switching to CPU backend")
            os.environ['JAX_PLATFORMS'] = 'cpu'
            # Need to reload modules to pick up the platform change
            importlib.reload(jax)

    except Exception as e:
        warnings.warn(f"Failed to initialize JAX GPU backend: {e}. Using CPU backend")
        os.environ['JAX_PLATFORMS'] = 'cpu'
        # Clear any cached imports and retry with CPU
        modules_to_remove = [mod for mod in sys.modules.keys() if mod.startswith('jax')]
        for mod in modules_to_remove:
            if mod in sys.modules:
                del sys.modules[mod]
        import jax

    return jax



class NonParametricMLR:
    """
    Non-parametric Bayesian inference for mass-luminosity relations.

    Uses NumPyro with NUTS sampler for posterior inference of mass-luminosity
    relations in bins of absolute magnitude. Supports Rice or Gaussian uncertainty
    models for the u parameter.
    """

    def __init__(self, n_bins=10, absg_min=4.0, absg_max=12.0, mass_min=0.01, mass_max=1.5, uncertainty_model='rice',
                 f_outlier=0, outlier_u0=30, outlier_sigma=15, outlier_kappa=50.0, fit_outlier_params=False):
        """
        Initialize the non-parametric plotter.

        Parameters
        ----------
        n_bins : int
            Number of bins to split the absg range into
        absg_min : float
            Minimum absg value
        absg_max : float
            Maximum absg value
        mass_min : float
            Minimum mass value for priors
        mass_max : float    
            Maximum mass value for priors
        uncertainty_model : str
            Uncertainty model to use: 'rice' or 'gaussian' (default: 'rice')
        f_outlier : float
            Initial fraction of outliers (default: 0)
        outlier_u0 : float
            Initial outlier distribution center (default: 30)
        outlier_sigma : float
            Initial outlier distribution width (default: 15)
        fit_outlier_params : bool
            Whether to fit outlier parameters in MCMC (True) or keep them fixed (False)
        """
        # Parameters to be fit in MCMC (with initial values as defaults)
        self.fit_outlier_params = fit_outlier_params
        self.f_outlier_init = f_outlier
        self.p_epsilon = 1e-10
        self.m_epsilon = 1e-10
        self.outlier_u0_init = outlier_u0
        self.outlier_sigma_init = outlier_sigma
        self.outlier_kappa_init = outlier_kappa

        # Non-parametric bin settings
        self.n_bins = n_bins
        self.absg_min = absg_min
        self.absg_max = absg_max
        self.absg_bins = np.linspace(absg_min, absg_max, n_bins)
        self.mass_min = mass_min
        self.mass_max = mass_max

        # Uncertainty model selection
        if uncertainty_model not in ['rice', 'gaussian']:
            raise ValueError("uncertainty_model must be 'rice' or 'gaussian'")
        self.uncertainty_model = uncertainty_model

        # Store observed u values (will be set when fitting data)
        self.u_values = None
        self.u_sigma_values = None
        self.absg1_values = None
        self.absg2_values = None
        self.sampler = None
        self.norm_factor = None
        self.gamma = np.inf

        # Storage for results
        self.results = None
        self.samples = None

    def func_pu_8(self, tilde_u, A=4.95e-3, B=2.24e-3, C=3.85, u0=36.09):
        """
        Version 8 of p(u)
        The probability distribution for the normalized parameter ũ = u/√mtot

        Parameters
        ----------
        tilde_u : array_like
            Normalized u values
        A, B, C, u0 : float
            Distribution parameters

        Returns
        -------
        array_like : probability density values
        """
        return A * tilde_u * np.exp(-1 * (
            B * tilde_u**2 + np.exp((tilde_u - u0) / C)
        ))

    def absg_to_mass(self, mass_bins, absg):
        """
        Convert absolute G magnitude to mass using bin values and interpolation.

        Parameters
        ----------
        mass_bins : array_like
            Mass values at each absg bin (length = n_bins)
        absg : array_like or float
            Absolute G magnitude values to convert

        Returns
        -------
        mass : array_like or float
            Interpolated mass values
        """
        mass = np.interp(absg, self.absg_bins, mass_bins)
        return mass

    def rice_distribution(self, u_obs, u_true, sigma):
        """
        Rice distribution PDF (replaces Gaussian for positive-definite measurements).

        p(u_obs | u_true, sigma) = (u_obs/sigma^2) * exp[-(u_obs^2 + u_true^2)/(2*sigma^2)] * I_0(u_obs*u_true/sigma^2)

        Parameters
        ----------
        u_obs : array_like
            Observed u values
        u_true : array_like
            True u values
        sigma : array_like
            Uncertainty in u

        Returns
        -------
        rice_pdf : array_like
            Rice distribution probability density
        """
        # Avoid division by zero
        sigma = np.maximum(sigma, 1e-10)
        sigma_sq = sigma ** 2

        # Pre-factor: u_obs / sigma^2
        prefactor = u_obs / sigma_sq

        # Exponential term: exp[-(u_obs^2 + u_true^2) / (2*sigma^2)]
        exp_term = np.exp(-(u_obs**2 + u_true**2) / (2 * sigma_sq))

        # Bessel function argument: u_obs * u_true / sigma^2
        bessel_arg = u_obs * u_true / sigma_sq

        # Modified Bessel function I_0
        i0_values = special.i0(bessel_arg)

        # Complete Rice distribution
        rice_pdf = prefactor * exp_term * i0_values

        return rice_pdf

    def gaussian(self, x, mu, sigma, norm_factor=None):
        """
        Standard Gaussian function (kept for backwards compatibility).

        Parameters
        ----------
        x : array_like
            Input values
        mu : array_like
            Mean values
        sigma : array_like
            Standard deviation
        norm_factor : float, optional
            Normalization factor

        Returns
        -------
        array_like : Gaussian probability density
        """
        if norm_factor is None:
            coeff = 1.0 / (sigma * np.sqrt(2 * np.pi))
            exponent = -0.5 * ((x - mu) / sigma)**2
            return coeff * np.exp(exponent)
        else:
            coeff = 1.0 / (sigma * np.sqrt(2 * np.pi) * norm_factor)
            exponent = -0.5 * ((x - mu) / sigma)**2
            return coeff * np.exp(exponent)

    def set_data(self, u_values=None, u_sigma_values=None,
                absg1_values=None, absg2_values=None, gamma=None, outlier_kappa=None):
        """
        Set the observed values for inference.

        Parameters
        ----------
        u_values : array_like
            Observed u values
        u_sigma_values : array_like, optional
            Uncertainties in u values
        absg1_values, absg2_values : array_like
            Absolute G magnitudes of primary and secondary stars
        gamma : float, optional
            Regularization parameter
        outlier_kappa : float, optional
            Concentration parameter for the Beta prior on f_outlier (default: 50.0)
        """
        self.u_values = np.array(u_values)
        self.absg1_values = np.array(absg1_values)
        self.absg2_values = np.array(absg2_values)

        if u_sigma_values is None:
            self.u_sigma_values = None
            self.norm_factor = np.ones(len(u_values))
        else:
            self.u_sigma_values = np.array(u_sigma_values)

            if self.uncertainty_model == 'rice':
                # Rice distribution is naturally defined for u >= 0, no truncation correction needed
                self.norm_factor = np.ones(len(u_values))
                print("Using RICE uncertainty (norm_factor = 1.0, no truncation correction)")
            elif self.uncertainty_model == 'gaussian':
                # Gaussian needs truncation correction for u >= 0
                self.norm_factor = 1 - norm.cdf(0, loc=self.u_values, scale=self.u_sigma_values)
                print(f"Using GAUSSIAN uncertainty (norm_factor computed for truncation at u=0)")
                print(f"  Mean norm_factor: {np.mean(self.norm_factor):.4f}")

        # Set hyper-parameter for regularization
        if gamma is not None:
            self.gamma = gamma
        else:
            self.gamma = np.inf  # No regularization by default

        self.outlier_kappa = self.outlier_kappa_init if outlier_kappa is None else outlier_kappa

    def run_numpyro(self, num_warmup=1000, num_samples=2000, num_chains=4,
                    int_umax=80., int_du=0.02,
                    seed=None, use_dense_mass=False, **kwargs):
        """
        Run HMC sampling using NumPyro with chosen uncertainty model (Rice or Gaussian).

        Performance tips:
        -----------------
        - Increase int_du (e.g., 0.05-0.1) for faster integration at slight accuracy cost
        - Decrease int_umax (e.g., 50-60) if your data has small u values
        - Use num_chains=2-4 for parallel sampling (automatically parallelized by JAX)
        - Set use_dense_mass=True for better adaptation (slower warmup, faster sampling)

        Parameters
        ----------
        num_warmup, num_samples : int
            Number of warmup and sampling steps
        num_chains : int
            Number of parallel chains
        mass_min, mass_max : float
            Mass bounds for priors
        int_umax, int_du : float
            Integration parameters for uncertainty handling
        seed : int, optional
            Random seed
        use_dense_mass : bool
            Whether to use dense mass matrix adaptation
        **kwargs
            Additional arguments passed to NUTS kernel

        Returns
        -------
        mcmc : numpyro.infer.MCMC
            The fitted MCMC sampler
        """
        try:
            # Configure JAX to prefer GPU but fall back to CPU
            jax = _configure_jax_gpu_fallback()
            import jax.numpy as jnp
            from jax.scipy.special import i0e  # Modified Bessel function I_0 * exp(-|x|)
            import numpyro
            import numpyro.distributions as dist
            from numpyro.distributions import constraints
            from numpyro.infer import MCMC, NUTS

            # Print device information
            devices = jax.devices()
            print(f"JAX devices: {devices}")
            print(f"Device count: {jax.local_device_count()}")
            print(f"JAX backend: {devices[0].platform if devices else 'None'}")

            # Log GPU status
            gpu_devices = [d for d in devices if d.platform == 'gpu']
            cpu_devices = [d for d in devices if d.platform == 'cpu']

            if gpu_devices:
                print(f"✓ Using {len(gpu_devices)} GPU device(s) for acceleration")
            else:
                print(f"⚠ Using CPU backend ({len(cpu_devices)} device(s))")
        except ImportError:
            raise ImportError("numpyro is not installed. Install it with: pip install numpyro")

        if self.u_values is None:
            raise ValueError("No data set. Use set_data() first.")

        print(f"Running numpyro with {self.n_bins} bins, hyper-parameter for regularisation gamma={self.gamma}...")
        print(f"*** USING {self.uncertainty_model.upper()} UNCERTAINTY for u_obs uncertainty ***")

        # Performance info
        if self.u_sigma_values is not None:
            n_integration_pts = int(int_umax / int_du)
            print(f"Integration grid: {n_integration_pts} points (int_du={int_du}, int_umax={int_umax})")
            print(f"  → To speed up: increase int_du (e.g., 0.05-0.1)")
        print(f"Sampling: {num_chains} chain(s) × ({num_warmup} warmup + {num_samples} samples)")

        # Convert data to JAX arrays
        u_values_jax = jnp.array(self.u_values)
        absg1_values_jax = jnp.array(self.absg1_values)
        absg2_values_jax = jnp.array(self.absg2_values)

        if self.u_sigma_values is not None:
            u_sigma_values_jax = jnp.array(self.u_sigma_values)
        else:
            u_sigma_values_jax = None

        norm_factor_jax = jnp.array(self.norm_factor)
        absg_bins_jax = jnp.array(self.absg_bins)
        
        # Pre-compute integration grid if using uncertainties
        if self.u_sigma_values is not None:
            int_ulist_jax = jnp.arange(0., int_umax, int_du)

        # JAX-compatible helper functions
        def func_pu_8_jax(tilde_u, A=4.95e-3, B=2.24e-3, C=3.85, u0=36.09):
            """JAX version of p(u) function"""
            return A * tilde_u * jnp.exp(-1 * (
                B * tilde_u**2 + jnp.exp((tilde_u - u0) / C)
            ))

        def gaussian_jax(x, mu, sigma):
            """
            JAX version of Gaussian distribution:
            p(x | mu, sigma) = (1/(sigma*sqrt(2*pi))) * exp[-(x-mu)^2/(2*sigma^2)]
            """
            x, mu, sigma = jnp.broadcast_arrays(x, mu, jnp.maximum(sigma, 1e-10))
            coeff = 1.0 / (sigma * jnp.sqrt(2 * jnp.pi))
            exponent = -0.5 * ((x - mu) / sigma)**2
            return coeff * jnp.exp(exponent)

        def rice_distribution_jax(u_obs, u_true, sigma):
            """
            JAX version of Rice distribution with numerical stability:
            p(u_obs | u_true, sigma) = (u_obs/sigma^2) * exp[-(u_obs^2 + u_true^2)/(2*sigma^2)] * I_0(u_obs*u_true/sigma^2)

            Uses log-space calculation for numerical stability when Bessel argument is large.
            """
            u_obs, u_true, sigma = jnp.broadcast_arrays(
                u_obs, u_true, jnp.maximum(sigma, 1e-10)
            )
            sigma_sq = sigma**2

            bessel_arg = u_obs * u_true / sigma_sq

            # For large arguments, I_0(x) ≈ exp(x)/sqrt(2*pi*x)
            # Use log-space to avoid overflow
            log_prefactor = jnp.log(u_obs + 1e-100) - jnp.log(sigma_sq)
            log_exp_term = -(u_obs**2 + u_true**2) / (2 * sigma_sq)

            # Use i0e(x) = I_0(x) * exp(-|x|) for numerical stability
            log_bessel = jnp.log(i0e(bessel_arg) + 1e-100) + jnp.abs(bessel_arg)

            log_rice_pdf = log_prefactor + log_exp_term + log_bessel
            rice_pdf = jnp.exp(log_rice_pdf)

            # Clamp to avoid numerical issues
            return jnp.maximum(rice_pdf, 1e-100)
        
        def outlier_gaussian_jax(tilde_u, outlier_u0, outlier_sigma):

            """
            JAX version of outlier Gaussian distribution for negative u values.
            """
            sigma = jnp.maximum(outlier_sigma, 1e-10)
            coeff = 1.0 / (sigma * jnp.sqrt(2 * jnp.pi))
            exponent = -0.5 * ((tilde_u - outlier_u0) / sigma)**2
            return coeff * jnp.exp(exponent)

        def likelihood_single_u_jax(u, sqrt_mtot, u_sigma=None, norm_factor=None,
                                    f_outlier=None, f_good=None, outlier_u0=None, outlier_sigma=None):
            """
            JAX version of likelihood for a single u value with chosen uncertainty model.

            Implements the integral:
            (1/sqrt(m_tot)) * ∫ p(ũ) * P(u_obs | ũ, σ) dũ

            where ũ = u_true/sqrt(m_tot) is the integration variable.
            """
            # Use provided values or fall back to initial values
            if f_outlier is None:
                f_outlier = self.f_outlier_init
            if outlier_u0 is None:
                outlier_u0 = self.outlier_u0_init
            if outlier_sigma is None:
                outlier_sigma = self.outlier_sigma_init
            f_good = 1.0 - f_outlier

            # Check for positive u and sqrt_mtot, if not, remove contribution
            valid = (u > 0) & (sqrt_mtot > 0)

            u = jnp.where(valid, u, 1e-10)
            sqrt_mtot = jnp.where(valid, sqrt_mtot, 1e-10)

            tilde_u = u / sqrt_mtot

            if u_sigma is None:
                # No uncertainty: direct evaluation
                good_component = (1.0 / sqrt_mtot) * func_pu_8_jax(tilde_u)
            else:
                # Ensure positive sigma
                u_sigma = jnp.maximum(u_sigma, 1e-10)

                # Expand dimensions to broadcast across integration grid
                u_obs_grid = (u / sqrt_mtot)[:, None]  # shape (N, 1)
                sigma_grid = (u_sigma / sqrt_mtot)[:, None]  # shape (N, 1)
                integration_grid = int_ulist_jax[None, :]  # shape (1, M)

                # With uncertainty: integrate over the distribution
                if self.uncertainty_model == 'rice':
                    # Rice distribution: p(u_obs | u_true, sigma)
                    uncertainty_dist = rice_distribution_jax(
                        u_obs_grid,        # u_obs (fixed observed value)
                        integration_grid,  # ũ (integration variable, true values)
                        sigma_grid         # σ
                    )
                elif self.uncertainty_model == 'gaussian':
                    # Gaussian distribution: p(u_obs | u_true, sigma) = N(u_obs; u_true, sigma)
                    uncertainty_dist = gaussian_jax(
                        u_obs_grid,        # u_obs (fixed observed value)
                        integration_grid,  # μ = ũ (integration variable, mean = true values)
                        sigma_grid         # σ
                    )

                # Integrand: (1/sqrt(m_tot)) * p(ũ) * P(u_obs | ũ, σ)
                integrand = (
                    (1.0 / sqrt_mtot[:, None])
                    * func_pu_8_jax(int_ulist_jax)[None, :]
                    * uncertainty_dist
                    * int_du
                )

                # Ensure norm_factor is positive
                norm_factor = jnp.maximum(norm_factor, 1e-10)
                good_component = jnp.sum(integrand, axis=1) / norm_factor + self.p_epsilon / (int_umax / int_du)
                outlier_component = outlier_gaussian_jax(tilde_u, outlier_u0, outlier_sigma) / norm_factor

            total_prob = f_good * good_component + f_outlier * outlier_component + self.p_epsilon

            # Clamp to reasonable range to avoid log(0)
            total_prob = jnp.clip(total_prob, 1e-100, 1e10)
            return total_prob

        # Define the NumPyro model with regularization
        def model(u_values, u_sigma_values, absg1_values, absg2_values, norm_factor,
                  absg_bins,
                  gamma=None):

            # Prior: uniform for each mass bin
            ndim = self.n_bins
            mass_min = self.mass_min
            mass_max = self.mass_max
            mass_bins = numpyro.sample('mass_bins',
                                      dist.Uniform(mass_min, mass_max).expand([ndim]))

            # Smoothness regularization
            if ndim > 2:
                second_diffs = (mass_bins[2:] - 2*mass_bins[1:-1] + mass_bins[:-2])/mass_bins[1:-1]
                gamma = self.gamma if gamma is None else gamma
                epsilon_reg = 10**(-gamma)
                smoothness_penalty = -epsilon_reg * jnp.sum(second_diffs**2)
                numpyro.factor('smoothness', smoothness_penalty)

            # Predict masses for each star based on prior mass-absg relation
            m1 = jnp.interp(absg1_values, absg_bins, mass_bins)
            m2 = jnp.interp(absg2_values, absg_bins, mass_bins)
            mtot = m1 + m2
            sqrt_mtot = jnp.sqrt(mtot)

            # Priors for outlier model parameters (conditional) with informative Beta/TruncatedNormal prior
            if self.fit_outlier_params:
                # ---- priors for contamination model
                pi = 0.01          # prior mean outlier rate (1%)
                kappa = self.outlier_kappa  # prior concentration (shrink strength)
                alpha = pi * kappa
                beta = (1 - pi) * kappa
                f_outlier = numpyro.sample("f_outlier", dist.Beta(alpha, beta))
                f_good = 1.0 - f_outlier

                # Tail separation thresholds (choose once; can be data-driven)
                u_tail_min = jnp.quantile(u_values / sqrt_mtot, 0.6)
                u_tail_max = int_umax

                # Outlier mean restricted to the tail region
                outlier_u0 = numpyro.sample(
                    "outlier_u0",
                    dist.TruncatedNormal(low=u_tail_min, high=u_tail_max, loc=u_tail_min+5.0, scale=5.0)
                )

                # Outlier sigma strictly large
                rho_sigma = numpyro.sample("rho_sigma", dist.Normal(0.0, 1.0))
                sigma_min_out = 10.0  # set relative to your main-component width in tilde_u
                outlier_sigma = sigma_min_out * (1.0 + jax.nn.softplus(rho_sigma))
                numpyro.deterministic("outlier_sigma", outlier_sigma)
            else:
                # Use fixed initial values
                f_outlier = self.f_outlier_init
                f_good = 1.0 - f_outlier
                outlier_u0 = self.outlier_u0_init
                outlier_sigma = self.outlier_sigma_init

            # Calculate log likelihood for each data point
            log_likelihood_vec = jnp.log(likelihood_single_u_jax(u_values, sqrt_mtot, u_sigma_values, norm_factor=norm_factor,
                                                                f_outlier=f_outlier, f_good=f_good,
                                                                outlier_u0=outlier_u0, outlier_sigma=outlier_sigma))

            # Check for numerical issues and clamp
            log_likelihood_sum = jnp.sum(log_likelihood_vec)
            log_likelihood_sum = jnp.where(
                jnp.isfinite(log_likelihood_sum),
                log_likelihood_sum,
                -1e10  # Large negative value if not finite
            )

            numpyro.factor('obs', log_likelihood_sum)

        # Run MCMC
        nuts_kernel = NUTS(model, dense_mass=use_dense_mass, **kwargs)
        mcmc = MCMC(
            nuts_kernel,
            num_warmup=num_warmup,
            num_samples=num_samples,
            num_chains=num_chains,
            progress_bar=True
        )

        rng_key = jax.random.PRNGKey(seed if seed is not None else 42)
        mcmc.run(rng_key, u_values_jax, u_sigma_values_jax,
                absg1_values_jax, absg2_values_jax,
                norm_factor_jax, absg_bins_jax)

        # Store results
        self.sampler = mcmc
        samples_dict = mcmc.get_samples()

        # Extract all samples including outlier parameters if they were fitted
        if self.fit_outlier_params:
            # Combine mass_bins with outlier parameters
            mass_bins_samples = np.array(samples_dict['mass_bins'])
            f_outlier_samples = np.array(samples_dict['f_outlier'])
            outlier_u0_samples = np.array(samples_dict['outlier_u0'])
            outlier_sigma_samples = np.array(samples_dict['outlier_sigma'])
            
            self.samples = np.column_stack([
                mass_bins_samples,
                f_outlier_samples,
                outlier_u0_samples,
                outlier_sigma_samples
            ])
            
            # Update parameter names to include outlier parameters
            self.param_names = [f'Mass bin {i+1}\n(absg={self.absg_bins[i]:.1f})' 
                            for i in range(self.n_bins)]
            self.param_names.extend(['f_outlier', 'outlier_u0', 'outlier_sigma'])
        else:
            self.samples = np.array(samples_dict['mass_bins'])

        self.results = type('obj', (object,), {
            'samples': self.samples,
            'logz': None,
            'logzerr': None
        })()

        mcmc.print_summary()

        return mcmc

    def plot_results(self, labels=None, truths=None, output_dir='', output_suffix=''):
        """
        Plot sampling results using corner plot.

        Parameters
        ----------
        labels : list, optional
            Labels for each parameter
        truths : list, optional
            True values for comparison
        output_dir : str
            Output directory for plots
        output_suffix : str
            Suffix for output filenames
        """
        if labels is None:
            labels = [f'Mass bin {i+1}\n(absg={self.absg_bins[i]:.1f})'
                     for i in range(self.n_bins)]
            # Add outlier parameter labels if they were fitted
            if self.fit_outlier_params:
                labels.extend(['f_outlier', 'outlier_u0', 'outlier_sigma'])
        try:
            import corner
            fig = corner.corner(self.samples, labels=labels, truths=truths,
                               truth_color='salmon', show_titles=True)
            plt.tight_layout()
            outlier_tag = 'outlierfit' if self.fit_outlier_params else 'outlierfixed'
            metadata_suffix = output_suffix if output_suffix else f'_model-nonparametric_unc-{self.uncertainty_model}_outlier-{outlier_tag}'
            filename = f'corner{metadata_suffix}.png'
            plt.savefig(f'{output_dir}/{filename}', dpi=300)
            plt.close(fig)
        except ImportError:
            print("corner package not installed. Install it with: pip install corner")

    def plot_fitting_results(self, data=None, output_dir='', output_suffix='', isochrone_data_feh=None, iso_colname_dict={'absg':'Gmag', 'mass':'mass', 'feh':'MH'}):
        """
        Plot the fitting results with credible region.

        Parameters
        ----------
        data : astropy.Table or dict, optional
            Input data containing 'm1', 'm2', 'absg1', 'absg2' columns for true mass scatter points
        truths : array_like, optional
            True mass values for bins
        true_mass_func : callable, optional
            True mass-luminosity relation function
        output_dir : str
            Output directory for plots
        output_suffix : str
            Suffix for output filenames
        """
        if self.samples is None or self.samples.size == 0:
            print("No samples available for plotting fitting results.")
            return

        # Extract only mass bin parameters for plotting (exclude outlier parameters)
        mass_bins_samples = self.samples[:, :self.n_bins]

        num_samples = min(1000, len(mass_bins_samples))
        indices = np.random.choice(len(mass_bins_samples), size=num_samples, replace=False)
        resampled_data = mass_bins_samples[indices]

        percentiles = np.percentile(mass_bins_samples, [16, 50, 84], axis=0)
        lower, median, upper = percentiles[0], percentiles[1], percentiles[2]

        # Making fill_between like step function needs extra points
        # bin_edges = np.concatenate(([self.absg_bins[0] - (self.absg_bins[1] - self.absg_bins[0]) / 2],
        #                            self.absg_bins,
        #                            [self.absg_bins[-1] + (self.absg_bins[-1] - self.absg_bins[-2]) / 2]))
        # lower = np.concatenate(([lower[0]], lower, [lower[-1]]))
        # upper = np.concatenate(([upper[0]], upper, [upper[-1]]))

        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot sample fits as step functions
        for i in range(min(400, len(resampled_data))):
            ax.step(self.absg_bins, resampled_data[i], color='gray', alpha=0.1, linewidth=1, zorder=1, where='mid')

        # Plot credible region as step function
        ax.fill_between(self.absg_bins, lower, upper, color='orange', step='mid',
                       alpha=0.3, label='1 sigma', zorder=2)

        # Plot median fit as step function
        ax.step(self.absg_bins, median, color='orange',
               label=f'Median ({self.uncertainty_model.capitalize()})', linewidth=2, zorder=3, where='mid')

        # Plot true masses from data if provided
        if data is not None:
            try:
                # Extract data columns
                if hasattr(data, 'colnames'):  # astropy.Table
                    m1 = np.array(data['m1'])
                    m2 = np.array(data['m2'])
                    absg1 = np.array(data['absg1'])
                    absg2 = np.array(data['absg2'])
                elif isinstance(data, dict):  # dict
                    m1 = np.array(data['m1'])
                    m2 = np.array(data['m2'])
                    absg1 = np.array(data['absg1'])
                    absg2 = np.array(data['absg2'])
                else:
                    raise ValueError("data must be astropy.Table or dict with 'm1', 'm2', 'absg1', 'absg2' columns")

                # Plot both primary and secondary masses
                ax.scatter(absg1, m1, color='black', s=5, alpha=0.5,
                          label='Truth', zorder=0)
                ax.scatter(absg2, m2, color='black', s=5, alpha=0.5,
                          zorder=0)
            except (KeyError, AttributeError) as e:
                print(f"Could not extract true masses from data: {e}")

        if isochrone_data_feh is not None:
            col = iso_colname_dict
            # Plot isochrone data if provided
            ax.scatter(isochrone_data_feh[col['absg']], isochrone_data_feh[col['mass']], 
                        color='black', s=5, alpha=0.5,label='Isochrone', zorder=0)


        ax.set_xlim(self.absg_min, self.absg_max)
        ax.set_xlabel('$M_{\\mathrm{G}}$ [mag]', fontsize=12)
        ax.set_ylabel('Mass [$M_{\\odot}$]', fontsize=12)
        ax.set_yscale('log')
        ax.invert_xaxis()
        ax.legend(fontsize=10)
        title = f'Non-parametric fit with {self.uncertainty_model.capitalize()} uncertainty ({self.n_bins} bins)'
        plt.title(title, fontsize=14)
        plt.tight_layout()
        outlier_tag = 'outlierfit' if self.fit_outlier_params else 'outlierfixed'
        metadata_suffix = output_suffix if output_suffix else f'_model-nonparametric_unc-{self.uncertainty_model}_outlier-{outlier_tag}'
        filename = f'fit{metadata_suffix}.png'
        plt.savefig(f'{output_dir}/{filename}', dpi=300)
        plt.close(fig)


class BrokenPowerLawMLR:
    """
    Parametric Bayesian inference for mass-luminosity relations using broken power-law.

    The mass-luminosity relation is parameterized as:
        log10(M/M_sun) = alpha_0 + beta_0 * (M_G - M_G0)  for M_G <= M_break_1
        log10(M/M_sun) = alpha_1 + beta_1 * (M_G - M_break_1)  for M_break_1 < M_G <= M_break_2
        ...

    Continuity is enforced at break points:
        alpha_{i+1} = alpha_i + (beta_i - beta_{i+1}) * (M_break_i - M_G0)

    where break magnitudes are sampled with physical constraints.

    Uses NumPyro with NUTS sampler for posterior inference.
    """

    def __init__(self, n_segments=3, mass_min=0.08, mass_max=1.0, absg_min=4.0, absg_max=12.0,
                 uncertainty_model='rice',
                 f_outlier=0, outlier_u0=30, outlier_sigma=15, outlier_kappa=50.0, fit_outlier_params=False):
        """
        Initialize the broken power-law MLR fitter.

        Parameters
        ----------
        n_segments : int, optional
            Number of power-law segments (default: 3)
        uncertainty_model : str
            Uncertainty model: 'rice' or 'gaussian' (default: 'rice')
        f_outlier : float
            Initial fraction of outliers (default: 0)
        outlier_u0 : float
            Initial outlier distribution center
        outlier_sigma : float
            Initial outlier distribution width
        fit_outlier_params : bool
            Whether to fit outlier parameters in MCMC (True) or keep them fixed (False)
        """

        self.n_segments = n_segments  # Number of power-law segments

        # Uncertainty model selection
        if uncertainty_model not in ['rice', 'gaussian']:
            raise ValueError("uncertainty_model must be 'rice' or 'gaussian'")
        self.uncertainty_model = uncertainty_model

        # Outlier model parameters (to be fit in MCMC)
        self.fit_outlier_params = fit_outlier_params
        self.f_outlier_init = f_outlier
        self.p_epsilon = 1e-10
        self.outlier_u0_init = outlier_u0
        self.outlier_sigma_init = outlier_sigma
        self.outlier_kappa_init = outlier_kappa
        self.mass_min = mass_min
        self.mass_max = mass_max
        self.absg_min = absg_min
        self.absg_max = absg_max

        # Data storage
        self.u_values = None
        self.u_sigma_values = None
        self.absg1_values = None
        self.absg2_values = None
        self.norm_factor = None

        # Results storage
        self.sampler = None
        self.samples = None
        self.results = None

        # Parameter names: a_0 (first intercept), b_i (slopes), M_break_i (break point masses)
        # For n_segments: 1 intercept + n_segments slopes + (n_segments-1) break points
        self.param_names = ['a_0']  # First intercept
        for i in range(self.n_segments):
            self.param_names.append(f'b_{i}')  # slopes for each segment
        for i in range(self.n_segments - 1):
            self.param_names.append(f'M_break_{i}')  # break point masses

    def _split_params(self, params):
        """Split parameters [a_0, b_0, ..., b_{n-1}, M_break_0, ..., M_break_{n-2}] into components."""
        # Use JAX-compatible operations to avoid tracer conversion errors
        import jax.numpy as jnp

        # Convert to JAX array if not already, ensure at least 1D
        params = jnp.atleast_1d(params)
        a_0 = params[0]  # First intercept
        slopes = params[1:1+self.n_segments]  # b_0, b_1, ..., b_{n-1}
        break_points = params[1+self.n_segments:]  # M_break_0, ..., M_break_{n-2}

        return a_0, slopes, break_points

    def _compute_all_intercepts(self, a_0, slopes, break_points):
        """
        Compute all intercepts analytically from a_0, slopes, and break points.
        Ensures continuity at break points by construction.

        Parameters
        ----------
        a_0 : float
            First intercept (for segment 0)
        slopes : array_like
            Slopes [b_0, b_1, ..., b_{n-1}]
        break_points : array_like
            Break point masses [M_break_0, ..., M_break_{n-2}]

        Returns
        -------
        intercepts : array_like
            All intercepts [a_0, a_1, ..., a_{n-1}]
        """
        # Use JAX-compatible operations to avoid tracer conversion errors
        import jax.numpy as jnp

        intercepts = [a_0]

        for i in range(len(break_points)):
            # At break point i: a_i + b_i * log10(M_break_i) = a_{i+1} + b_{i+1} * log10(M_break_i)
            # Solve for a_{i+1}: a_{i+1} = a_i + (b_i - b_{i+1}) * log10(M_break_i)
            a_i = intercepts[-1]
            b_i = slopes[i]
            b_ip1 = slopes[i+1]
            M_break_i = break_points[i]

            a_ip1 = a_i + (b_i - b_ip1) * jnp.log10(M_break_i)
            intercepts.append(a_ip1)

        return jnp.array(intercepts)

    # def _compute_break_points(self, intercepts, slopes):
    #     """
    #     Compute break points from segment intersections.
    #     Ensures continuity by construction.

    #     Parameters
    #     ----------
    #     intercepts : array_like
    #         Intercepts a_i for each segment [a_0, a_1, ..., a_{K-1}]
    #     slopes : array_like
    #         Slopes b_i for each segment [b_0, b_1, ..., b_{K-1}]

    #     Returns
    #     -------
    #     break_mags : array_like
    #         Break magnitudes where segments intersect [M_break_1, ..., M_break_{K-1}]
    #         Length = n_segments - 1
    #     """
    #     break_mags = []
    #     for i in range(len(intercepts) - 1):
    #         # Intersection point: a_i + b_i * log10(M) = a_{i+1} + b_{i+1} * log10(M)
    #         # Solve for M: log10(M) = (a_i - a_{i+1}) / (b_{i+1} - b_i)
    #         log10_mass_break = (intercepts[i] - intercepts[i+1]) / (slopes[i+1] - slopes[i])

    #         # Convert to magnitude using the first segment equation
    #         break_mag = intercepts[i] + slopes[i] * log10_mass_break
    #         break_mags.append(break_mag)

    #     return np.array(break_mags)

    # def _validate_continuity(self, params):
    #     """
    #     Validate that the broken power-law is continuous at break points.

    #     Parameters
    #     ----------
    #     params : array_like
    #         Flattened parameters [a_0, b_0, a_1, b_1, ...]

    #     Returns
    #     -------
    #     is_continuous : bool
    #         True if continuous within tolerance
    #     """
    #     intercepts, slopes = self._split_params(params)
    #     break_mags = self._compute_break_points(intercepts, slopes)

    #     # Check continuity at each break point
    #     tolerance = 1e-10
    #     for i, break_mag in enumerate(break_mags):
    #         # Compute log10(mass) from both adjacent segments at break point
    #         log10_mass_1 = (break_mag - intercepts[i]) / slopes[i]
    #         log10_mass_2 = (break_mag - intercepts[i+1]) / slopes[i+1]

    #         if abs(log10_mass_1 - log10_mass_2) > tolerance:
    #             return False

    #     return True

    def absg_from_mass(self, mass, params):
        """
        Compute absolute G magnitude from mass using broken power-law.

        M_G = a_i + b_i * log10(M/M_sun) for segment i

        Parameters
        ----------
        mass : array_like
            Mass values in solar masses
        params : array_like
            Flattened parameters [a_0, b_0, ..., b_{n-1}, M_break_0, ..., M_break_{n-2}]
            Can be 1D (single parameter set) or 2D (multiple parameter sets)

        Returns
        -------
        absg : array_like
            Absolute G magnitudes
            If params is 2D, returns shape (n_params, len(mass))
            If params is 1D, returns shape (len(mass),)
        """
        mass = np.atleast_1d(mass)
        log_mass = np.log10(mass)

        # Handle both 1D and 2D parameter arrays
        if params.ndim == 1:
            # Single parameter set
            params = params.reshape(1, -1)
            single_param_set = True
        else:
            single_param_set = False

        n_param_sets = params.shape[0]
        absg_values = np.zeros((n_param_sets, len(mass)))

        # Process each parameter set
        for param_idx in range(n_param_sets):
            a_0, slopes, break_points = self._split_params(params[param_idx])

            # Compute all intercepts analytically
            intercepts = self._compute_all_intercepts(a_0, slopes, break_points)

            # Build mass boundaries: [mass_min, break_points, mass_max]
            boundaries = break_points

            absg = np.zeros_like(mass)
            for i in range(self.n_segments):
                a_i = intercepts[i]
                b_i = slopes[i]
                # Use <= for the last segment to include mass_max
                if i==0:
                    mask = (mass < boundaries[i])
                elif i == self.n_segments - 1:
                    mask = (mass >= boundaries[i])
                else:
                    mask = (mass >= boundaries[i]) & (mass < boundaries[i+1])
                absg[mask] = a_i + b_i * log_mass[mask]

            absg_values[param_idx] = absg

        # Return appropriate shape based on input
        if single_param_set:
            return absg_values[0]  # Return 1D array for single parameter set
        else:
            return absg_values  # Return 2D array for multiple parameter sets

    def mass_from_absg(self, absg, params):
        """
        Compute mass from absolute G magnitude using broken power-law.

        M = 10^((M_G - a_i) / b_i) for segment i

        This requires iterative solving since we don't know which segment
        a given M_G falls into until we know the mass.

        Parameters
        ----------
        absg : array_like
            Absolute G magnitude values
        params : array_like
            Flattened parameters [a_0, b_0, ..., b_{n-1}, M_break_0, ..., M_break_{n-2}]
            Can be 1D (single parameter set) or 2D (multiple parameter sets)

        Returns
        -------
        mass : array_like
            Mass values in solar masses
            If params is 2D, returns shape (n_params, len(absg))
            If params is 1D, returns shape (len(absg),)
        """
        absg = np.atleast_1d(absg)

        # Handle both 1D and 2D parameter arrays
        if params.ndim == 1:
            # Single parameter set
            params = params.reshape(1, -1)
            single_param_set = True
        else:
            single_param_set = False

        n_param_sets = params.shape[0]
        masses = np.zeros((n_param_sets, len(absg)))

        # Process each parameter set
        for param_idx in range(n_param_sets):
            a_0, slopes, break_points = self._split_params(params[param_idx])

            # Compute all intercepts analytically
            intercepts = self._compute_all_intercepts(a_0, slopes, break_points)

            # Compute magnitude boundaries (where segments intersect in magnitude space)
            # NOTE: mag_boundaries are in DECREASING order (brighter stars have smaller magnitude)
            if len(break_points) > 0:
                mag_boundaries = []
                for i, break_point in enumerate(break_points):
                    # Magnitude at break point using segment i
                    mag_break = intercepts[i] + slopes[i] * np.log10(break_point)
                    mag_boundaries.append(mag_break)
                mag_boundaries = np.array(mag_boundaries)
            else:
                mag_boundaries = np.array([])

            # Build magnitude boundaries for segment selection: [absg_max, mag_boundaries, absg_min]
            # Note: mag_boundaries are already in decreasing order due to the physics
            mag_boundaries = np.array(mag_boundaries)

            mass = np.zeros_like(absg)
            # For each segment, compute mass and check if magnitude falls in this segment's range
            # Note: mag_boundaries_full is in DECREASING order [absg_max, ..., absg_min]
            for i in range(self.n_segments):
                a_i = intercepts[i]
                b_i = slopes[i]

                # Compute mass for this segment: M = 10^((M_G - a) / b)
                mass_candidate = 10**((absg - a_i) / b_i)

                # Check if magnitude falls within this segment's magnitude range
                # Handle different cases based on number of segments and boundaries
                if self.n_segments == 1:
                    # Single segment: all magnitudes belong to this segment
                    mask = np.ones_like(absg, dtype=bool)
                elif len(break_points) == 0:
                    # Multiple segments but no break points (shouldn't happen, but handle gracefully)
                    if i == 0:
                        mask = np.ones_like(absg, dtype=bool)
                    else:
                        mask = np.zeros_like(absg, dtype=bool)
                elif i == 0:
                    # First segment: absg > mag_boundaries[0]
                    mask = (absg > mag_boundaries[0])
                elif i == self.n_segments - 1:
                    # Last segment: absg <= mag_boundaries[i-1]
                    mask = (absg <= mag_boundaries[i-1])
                else:
                    # Other segments: mag_boundaries[i-1] > absg >= mag_boundaries[i]
                    mask = (absg < mag_boundaries[i-1]) & (absg >= mag_boundaries[i])
                mass[mask] = mass_candidate[mask]

            masses[param_idx] = mass

        # Return appropriate shape based on input
        if single_param_set:
            return masses[0]  # Return 1D array for single parameter set
        else:
            return masses  # Return 2D array for multiple parameter sets

    def mass_from_absg_jax(self, absg, params):
        """
        JAX-compatible version of mass_from_absg for use in NumPyro model.

        Parameters
        ----------
        absg : jax array
            Absolute G magnitude values
        params : jax array
            Flattened parameters [a_0, b_0, ..., b_{n-1}, M_break_0, ..., M_break_{n-2}]

        Returns
        -------
        mass : jax array
            Mass values in solar masses
        """
        import jax.numpy as jnp
        import jax

        a_0, slopes, break_points = self._split_params(params)
        n_segments = self.n_segments

        # Compute all intercepts analytically using JAX-compatible method
        intercepts = self._compute_all_intercepts(a_0, slopes, break_points)
        break_points_jax = break_points

        # Compute magnitude boundaries (where segments intersect in magnitude space)
        if len(break_points) > 0:
            mag_boundaries = jnp.array([
                intercepts[i] + slopes[i] * jnp.log10(break_points_jax[i])
                for i in range(len(break_points))
            ])
        else:
            mag_boundaries = jnp.array([])

        # Initialize with zeros
        mass = jnp.zeros_like(absg)

        # For each segment, compute mass and check if magnitude falls in this segment's range
        for i in range(n_segments):
            a_i = intercepts[i]
            b_i = slopes[i]

            # Compute mass for this segment: M = 10^((M_G - a) / b)
            mass_candidate = jnp.power(10.0, (absg - a_i) / b_i)

            # Check if magnitude falls within this segment's magnitude range
            if i == 0:
                in_segment = (absg > mag_boundaries[i])
            elif i == n_segments - 1:
                # Last segment includes upper bound
                in_segment = (absg <= mag_boundaries[i])
            else:
                # Other segments use exclusive upper bound
                in_segment = (absg < mag_boundaries[i]) & (absg >= mag_boundaries[i+1])

            # Update mass where this segment applies
            mass = jnp.where(in_segment, mass_candidate, mass)

        return mass

    def func_pu_8(self, tilde_u, A=4.95e-3, B=2.24e-3, C=3.85, u0=36.09):
        """
        The probability distribution for the normalized parameter ũ = u/√mtot
        """
        return A * tilde_u * np.exp(-1 * (
            B * tilde_u**2 + np.exp((tilde_u - u0) / C)
        ))

    def set_data(self, u_values, u_sigma_values, absg1_values, absg2_values, outlier_kappa=None):
        """
        Set the data for fitting.

        Parameters
        ----------
        u_values : array_like
            Observed u values
        u_sigma_values : array_like
            Uncertainties in u values
        absg1_values : array_like
            Primary star absolute G magnitudes
        absg2_values : array_like
            Secondary star absolute G magnitudes
        outlier_kappa : float, optional
            Concentration parameter for the Beta prior on f_outlier (default: 50.0)
        """
        self.u_values = np.array(u_values)
        self.u_sigma_values = np.array(u_sigma_values)
        self.absg1_values = np.array(absg1_values)
        self.absg2_values = np.array(absg2_values)

        # Pre-compute normalization factor
        int_umax = 100
        int_du = 0.02
        int_ulist = np.arange(0.01, int_umax, int_du)
        self.norm_factor = np.sum(self.func_pu_8(int_ulist) * int_du)
        self.outlier_kappa = self.outlier_kappa_init if outlier_kappa is None else outlier_kappa

    def run_numpyro(self, num_warmup=1000, num_samples=2000, num_chains=4,
                    seed=None,
                    int_umax=80, int_du=0.02, use_dense_mass=False,
                    a_prior_range=None, b_prior_range=None,
                    **kwargs):
        """
        Run NumPyro MCMC inference for broken power-law parameters.

        Parameters
        ----------
        num_warmup : int
            Number of warmup samples
        num_samples : int
            Number of posterior samples
        num_chains : int
            Number of MCMC chains
            mass_min, mass_max : float
            Mass bounds for clamping
        seed : int, optional
            Random seed
        int_umax : float
            Upper limit for u integration
        int_du : float
            Integration step size
        use_dense_mass : bool
            Use dense mass matrix in NUTS
        a_prior_range : tuple or None
            Optional Uniform(low, high) prior range for M_G at the first break point.
            If None (default), use a weakly-informative Normal with mean/sigma given
            by a_prior_mu_sigma.
        b_prior_range : tuple or None
            Optional Uniform(low, high) prior range for slopes. If None (default),
            use a truncated Normal with mean/sigma given by b_prior_mu_sigma and
            hard bounds b_prior_bounds.
        a_prior_mu_sigma : tuple
            Mean and sigma for the Normal prior on mag_at_bp0 when a_prior_range is None.
        b_prior_mu_sigma : tuple
            Mean and sigma for the (truncated) Normal prior on slopes when b_prior_range is None.
        b_prior_bounds : tuple
            Low/high bounds for the truncated Normal slope prior.
        **kwargs : dict
            Additional arguments passed to NUTS

        Returns
        -------
        mcmc : numpyro.infer.MCMC
            MCMC object with results
        """
        # Configure JAX to prefer GPU but fall back to CPU
        jax = _configure_jax_gpu_fallback()
        import jax.numpy as jnp
        import numpyro
        import numpyro.distributions as dist
        from numpyro.distributions import constraints
        from numpyro.infer import MCMC, NUTS
        from scipy.special import i0 as bessel_i0

        if self.u_values is None:
            raise ValueError("Data not set. Call set_data() first.")

        # Parameter names already set in __init__, no need to redefine

        # Convert data to JAX arrays
        u_values_jax = jnp.array(self.u_values)
        u_sigma_values_jax = jnp.array(self.u_sigma_values)
        absg1_values_jax = jnp.array(self.absg1_values)
        absg2_values_jax = jnp.array(self.absg2_values)
        norm_factor_jax = jnp.array(self.norm_factor)

        # Integration grid
        int_ulist_jax = jnp.arange(0.01, int_umax, int_du)

        # Define JAX-compatible functions
        def func_pu_8_jax(tilde_u, A=4.95e-3, B=2.24e-3, C=3.85, u0=36.09):
            return A * tilde_u * jnp.exp(-1 * (
                B * tilde_u**2 + jnp.exp((tilde_u - u0) / C)
            ))

        def rice_distribution_jax(x, nu, sigma):
            """Rice distribution PDF using JAX."""
            x_safe = jnp.maximum(x, 1e-10)
            nu_safe = jnp.maximum(nu, 1e-10)
            sigma_safe = jnp.maximum(sigma, 1e-10)
            sigma2_safe = sigma_safe**2

            bessel_arg = x_safe * nu_safe / sigma2_safe
            bessel_arg_clipped = jnp.minimum(bessel_arg, 700)
            log_i0 = bessel_arg_clipped

            log_pdf = (jnp.log(x_safe) - 2*jnp.log(sigma_safe)
                      - (x_safe**2 + nu_safe**2)/(2*sigma2_safe) + log_i0)
            return jnp.exp(log_pdf)

        def gaussian_jax(x, mu, sigma):
            """Gaussian distribution PDF using JAX."""
            return jnp.exp(-0.5 * ((x - mu) / sigma)**2) / (sigma * jnp.sqrt(2 * jnp.pi))

        def outlier_gaussian_jax(x, mu, sigma):
            """Outlier Gaussian distribution."""
            return jnp.exp(-0.5 * ((x - mu) / sigma)**2) / (sigma * jnp.sqrt(2 * jnp.pi))

        def mass_from_absg_jax_inner(absg, params_inner):
            """Compute mass from M_G using broken power-law with enforced continuity.

            For each M_G value, compute mass from each segment's equation and
            select the one where the M_G falls in that segment's magnitude range.
            """
            import jax.numpy as jnp

            intercepts, slopes, break_points_inner = params_inner

            absg_2d = absg[:, None]          # (N, 1)
            a_2d = intercepts[None, :]       # (1, n_seg)
            b_2d = slopes[None, :]           # (1, n_seg)
            n_seg = intercepts.shape[0]

            # Mass from each segment's equation: M = 10^((M_G - a) / b)
            mass_all = jnp.power(10.0, (absg_2d - a_2d) / b_2d)  # (N, n_seg)

            # Single segment: no boundaries needed, just evaluate everywhere
            if n_seg == 1:
                mass = mass_all[:, 0]
                mass = jnp.where(mass > 0, mass, self.mass_min)
                return mass

            # Compute magnitude boundaries (where segments intersect in magnitude space)
            if len(break_points_inner) > 0:
                mag_boundaries = jnp.array([
                    intercepts[i] + slopes[i] * jnp.log10(break_points_inner[i])
                    for i in range(len(break_points_inner))
                ])  # shape (n_seg-1,)
            else:
                # Should not happen if n_seg > 1, but keep a safe fallback
                mag_boundaries = jnp.array([])

            # Build per-segment magnitude bounds without using absg_min / absg_max:
            #   seg 0: (-inf, mag_boundaries[0])
            #   seg i (1..n_seg-2): [mag_boundaries[i-1], mag_boundaries[i])
            #   seg last: [mag_boundaries[-1], +inf)
            mag_low = jnp.concatenate([jnp.array([-jnp.inf]), mag_boundaries])   # (n_seg,)
            mag_high = jnp.concatenate([mag_boundaries, jnp.array([jnp.inf])])   # (n_seg,)

            absg_expanded = absg_2d                     # (N, 1)
            mag_low_2d = mag_low[None, :]               # (1, n_seg)
            mag_high_2d = mag_high[None, :]             # (1, n_seg)

            # Multiple segments: last segment inclusive, others half-open [low, high)
            in_segment_all = (absg_expanded >= mag_low_2d) & (absg_expanded < mag_high_2d)
            # For +inf upper bound, < and <= are equivalent, but we keep last inclusive for consistency
            in_segment = in_segment_all

            # Select valid mass (sum works since only one should be valid due to continuity)
            mass = jnp.sum(mass_all * in_segment, axis=1)

            # Fallback for edge cases where no segment matched
            mass = jnp.where(mass > 0, mass, self.mass_min)
            return mass

        def likelihood_single_u_jax(u, sqrt_mtot, u_sigma, norm_factor,
                                    f_outlier=None, f_good=None, outlier_u0=None, outlier_sigma=None):
            """Compute likelihood for observed u values."""
            # Use provided values or fall back to initial values
            if f_outlier is None:
                f_outlier = self.f_outlier_init
            if outlier_u0 is None:
                outlier_u0 = self.outlier_u0_init
            if outlier_sigma is None:
                outlier_sigma = self.outlier_sigma_init
            f_good = 1.0 - f_outlier

            valid = (u > 0) & (sqrt_mtot > 0)

            u = jnp.where(valid, u, 1e-10)
            sqrt_mtot = jnp.where(valid, sqrt_mtot, 1e-10)

            tilde_u = u / sqrt_mtot

            if u_sigma is None:
                # No uncertainty: direct evaluation
                good_component = (1.0 / sqrt_mtot) * func_pu_8_jax(tilde_u)
            else:
                # Ensure positive sigma
                u_sigma = jnp.maximum(u_sigma, 1e-10)
                
                u_obs_grid = (u / sqrt_mtot)[:, None]  # shape (N, 1)
                sigma_grid = (u_sigma / sqrt_mtot)[:, None]  # shape (N, 1)
                integration_grid = int_ulist_jax[None, :]  # shape (1, M)

                # With uncertainty: integrate over the distribution
                if self.uncertainty_model == 'rice':
                    # Rice distribution: p(u_obs | u_true, sigma)
                    uncertainty_dist = rice_distribution_jax(
                        u_obs_grid,        # u_obs (fixed observed value)
                        integration_grid,  # ũ (integration variable, true values)
                        sigma_grid         # σ
                    )
                elif self.uncertainty_model == 'gaussian':
                    # Gaussian distribution: p(u_obs | u_true, sigma) = N(u_obs; u_true, sigma)
                    uncertainty_dist = gaussian_jax(
                        u_obs_grid,        # u_obs (fixed observed value)
                        integration_grid,  # μ = ũ (integration variable, mean = true values)
                        sigma_grid         # σ
                    )
                else:
                    raise ValueError("Invalid uncertainty model.")

                integrand = (
                    (1.0 / sqrt_mtot[:, None])
                    * func_pu_8_jax(int_ulist_jax)[None, :]
                    * uncertainty_dist
                    * int_du
                )

                norm_factor = jnp.maximum(norm_factor, 1e-10)
                good_component = jnp.sum(integrand, axis=1) / norm_factor + self.p_epsilon / (int_umax / int_du)
            
            outlier_component = outlier_gaussian_jax(tilde_u, outlier_u0, outlier_sigma) / norm_factor

            total_prob = f_good * good_component + f_outlier * outlier_component + self.p_epsilon
            total_prob = jnp.clip(total_prob, 1e-100, 1e10)
            return total_prob

        # Define the NumPyro model
        def model(u_values, u_sigma_values, absg1_values, absg2_values,
                  norm_factor):

            n_segments = self.n_segments

            # Prior on a_0 (first intercept)
            if a_prior_range is not None:
                a_0 = numpyro.sample('a_0', dist.Uniform(a_prior_range[0], a_prior_range[1]))
            else:
                raise ValueError("Have to specify a_prior_range for a_0 in broken power-law model.")

            # Prior on slopes with physical constraints (slopes should be negative for stellar MLR)
            slopes = []
            if b_prior_range is not None:
                # Ensure b_prior_range covers only negative slopes
                b_low = max(b_prior_range[0], -80.0)
                b_high = min(b_prior_range[1], 0.0)
                for i in range(n_segments):
                    slopes.append(numpyro.sample(f'b_{i}', dist.Uniform(b_low, b_high)))
            else:
                raise ValueError("Have to specify b_prior_range for slopes in broken power-law model.")
            slopes = jnp.stack(slopes)

            # Prior on break points (should be within mass range and ordered)
            if n_segments > 1:
                break_points = []
                # First break point
                bp_min = self.mass_min
                bp_max = self.mass_max / (n_segments - 1)  # Simple partitioning
                break_points.append(numpyro.sample(f'M_break_0', dist.Uniform(bp_min, bp_max)))

                # Subsequent break points
                for i in range(1, n_segments - 1):
                    bp_min = break_points[i-1]  # Must be larger than previous break point
                    bp_max = self.mass_max * (i + 1) / (n_segments - 1)
                    break_points.append(numpyro.sample(f'M_break_{i}', dist.Uniform(bp_min, bp_max)))

                break_points = jnp.array(break_points)
            else:
                break_points = jnp.array([])  # No break points for single segment

            # Compute all intercepts analytically to ensure continuity
            intercepts = jnp.array([a_0])
            for i in range(len(break_points)):
                a_i = intercepts[i]
                b_i = slopes[i]
                b_ip1 = slopes[i+1]
                M_break_i = break_points[i]
                a_ip1 = a_i + (b_i - b_ip1) * jnp.log10(M_break_i)
                intercepts = jnp.concatenate([intercepts, jnp.array([a_ip1])])

            # Compute masses from magnitudes using our modified method
            # Create parameter array in the format expected by our JAX function
            params_jax = (jnp.array([a_0]), jnp.array(slopes), jnp.array(break_points))

            # Use the JAX version of mass_from_absg
            m1 = mass_from_absg_jax_inner(absg1_values, params_jax)
            m2 = mass_from_absg_jax_inner(absg2_values, params_jax)
            mtot = m1 + m2
            sqrt_mtot = jnp.sqrt(mtot)

            # Priors for outlier model parameters (conditional) with informative Beta/TruncatedNormal prior
            if self.fit_outlier_params:
                pi = 0.01
                kappa = self.outlier_kappa
                alpha = pi * kappa
                beta_param = (1 - pi) * kappa
                f_outlier = numpyro.sample("f_outlier", dist.Beta(alpha, beta_param))
                f_good = 1.0 - f_outlier

                # Tail separation thresholds (choose once; can be data-driven)
                u_tail_min = jnp.quantile(u_values / sqrt_mtot, 0.6)
                u_tail_max = int_umax

                # Outlier mean restricted to the tail region
                outlier_u0 = numpyro.sample(
                    "outlier_u0",
                    dist.TruncatedNormal(low=u_tail_min, high=u_tail_max, loc=u_tail_min+5.0, scale=5.0)
                )
                
                rho_sigma = numpyro.sample("rho_sigma", dist.Normal(0.0, 1.0))
                sigma_min_out = 10.0
                outlier_sigma = sigma_min_out * (1.0 + jax.nn.softplus(rho_sigma))
                numpyro.deterministic("outlier_sigma", outlier_sigma)
            else:
                f_outlier = self.f_outlier_init
                f_good = 1.0 - f_outlier
                outlier_u0 = self.outlier_u0_init
                outlier_sigma = self.outlier_sigma_init

            # Calculate log likelihood
            log_likelihood_vec = jnp.log(likelihood_single_u_jax(
                u_values, sqrt_mtot, u_sigma_values, norm_factor,
                f_outlier=f_outlier, f_good=f_good,
                outlier_u0=outlier_u0, outlier_sigma=outlier_sigma
            ))

            log_likelihood_sum = jnp.sum(log_likelihood_vec)
            log_likelihood_sum = jnp.where(
                jnp.isfinite(log_likelihood_sum),
                log_likelihood_sum,
                -1e10
            )

            numpyro.factor('obs', log_likelihood_sum)

        # Run MCMC
        nuts_kernel = NUTS(model, dense_mass=use_dense_mass, **kwargs)
        mcmc = MCMC(
            nuts_kernel,
            num_warmup=num_warmup,
            num_samples=num_samples,
            num_chains=num_chains,
            progress_bar=True
        )

        rng_key = jax.random.PRNGKey(seed if seed is not None else 42)
        mcmc.run(rng_key, u_values_jax, u_sigma_values_jax,
                 absg1_values_jax, absg2_values_jax,
                 norm_factor_jax)

        # Store results
        self.sampler = mcmc
        samples_dict = mcmc.get_samples()

        # Reconstruct samples as array [n_samples, 1 + n_segments + (n_segments-1) + outlier_params]
        # Format: [a_0, b_0, b_1, ..., b_{n-1}, M_break_0, ..., M_break_{n-2}, f_outlier, outlier_u0, outlier_sigma]
        n_samples_total = len(samples_dict['a_0'])

        # Extract a_0
        a_0_samples = np.array(samples_dict['a_0'])

        # Extract slopes
        slope_samples = np.column_stack([np.array(samples_dict[f'b_{i}']) for i in range(self.n_segments)])

        # Extract break points
        if self.n_segments > 1:
            break_point_samples = np.column_stack([np.array(samples_dict[f'M_break_{i}']) for i in range(self.n_segments - 1)])
            # Reconstruct all intercepts for compatibility
            intercept_samples = np.zeros((n_samples_total, self.n_segments))
            for i in range(n_samples_total):
                a_0 = a_0_samples[i]
                slopes_i = slope_samples[i]
                break_points_i = break_point_samples[i] if self.n_segments > 1 else np.array([])
                intercepts = self._compute_all_intercepts(a_0, slopes_i, break_points_i)
                intercept_samples[i] = intercepts

            # Combine main parameters
            main_samples = np.column_stack([a_0_samples, slope_samples, break_point_samples])
        else:
            # Single segment case
            main_samples = np.column_stack([a_0_samples, slope_samples])
            intercept_samples = a_0_samples.reshape(-1, 1)

        # Include outlier parameters if they were fitted
        if self.fit_outlier_params:
            f_outlier_samples = np.array(samples_dict['f_outlier'])
            outlier_u0_samples = np.array(samples_dict['outlier_u0'])
            outlier_sigma_samples = np.array(samples_dict['outlier_sigma'])

            # Combine all parameters
            self.samples = np.column_stack([
                main_samples, f_outlier_samples, outlier_u0_samples, outlier_sigma_samples
            ])

            # Update parameter names to include outlier parameters
            self.param_names = self.param_names + ['f_outlier', 'outlier_u0', 'outlier_sigma']
        else:
            self.samples = main_samples

        self.results = type('obj', (object,), {
            'samples': self.samples,
            'logz': None,
            'logzerr': None
        })()

        # Get median parameters for plotting
        median_a_0 = np.median(a_0_samples, axis=0)
        median_slopes = np.median(slope_samples, axis=0)

        if self.n_segments > 1:
            median_break_points = np.median(break_point_samples, axis=0)
            # Compute all intercepts from median parameters
            median_intercepts = self._compute_all_intercepts(median_a_0, median_slopes, median_break_points)
            self.break_points = median_break_points
        else:
            median_break_points = np.array([])
            median_intercepts = np.array([median_a_0])
            self.break_points = np.array([])

        mcmc.print_summary()

        return mcmc

    def plot_results(self, labels=None, truths=None, output_dir='', output_suffix=''):
        """
        Plot sampling results using corner plot.

        Parameters
        ----------
        labels : list, optional
            Labels for each parameter
        truths : list, optional
            True values for comparison
        output_dir : str
            Output directory for plots
        output_suffix : str
            Suffix for output filenames
        """
        if self.samples is None:
            print("No samples available. Run inference first.")
            return

        if labels is None:
            labels = self.param_names

        try:
            import corner
            fig = corner.corner(self.samples, labels=labels, truths=truths,
                               truth_color='salmon', show_titles=True)
            plt.tight_layout()
            os.makedirs(output_dir, exist_ok=True) if output_dir else None
            outlier_tag = 'outlierfit' if self.fit_outlier_params else 'outlierfixed'
            metadata_suffix = output_suffix if output_suffix else f'_model-brokenpowerlaw_unc-{self.uncertainty_model}_outlier-{outlier_tag}'
            filename = f'corner{metadata_suffix}.png'
            plt.savefig(f'{output_dir}/{filename}', dpi=300)
            plt.close(fig)
        except ImportError:
            print("corner package not installed. Install it with: pip install corner")

    def plot_fitting_results(self, data=None, output_dir='', output_suffix='', isochrone_data_feh=None, iso_colname_dict={'absg':'absg', 'mass':'mass', 'feh':'MH'}):
        """
        Plot the fitting results with credible region.

        Parameters
        ----------
        data : astropy.Table or dict, optional
            Input data containing 'm1', 'm2', 'absg1', 'absg2' columns for true mass scatter points
        truths : array_like, optional
            True mass values for comparison (not applicable for parametric model)
        true_mass_func : callable, optional
            True mass-luminosity relation function
        output_dir : str
            Output directory for plots
        output_suffix : str
            Suffix for output filenames
        """
        if self.samples is None or self.samples.size == 0:
            print("No samples available for plotting fitting results.")
            return

        # Extract only main model parameters for plotting (exclude outlier parameters)
        if self.fit_outlier_params:
            # Exclude last 3 outlier parameters (f_outlier, outlier_u0, outlier_sigma)
            main_params_samples = self.samples[:, :-3]
        else:
            main_params_samples = self.samples

        # Convert masses to M_G values for the plot (we'll invert later)
        num_samples = min(1000, len(main_params_samples))
        indices = np.random.choice(len(main_params_samples), size=num_samples, replace=False)
        subsamples = main_params_samples[indices]

        # Sample masses for plotting (use M_G range similar to non-parametric)
        absg_range = np.linspace(self.absg_min, self.absg_max, 1000)

        # Compute M_G for each posterior sample
        all_masses = self.mass_from_absg(absg_range, main_params_samples)
        lower_masses, median_masses, upper_masses = np.percentile(all_masses, [16, 50, 84], axis=0)

        # Best-fit line
        median_params = np.median(main_params_samples, axis=0)
        best_fit_masses = self.mass_from_absg(absg_range, median_params)

        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))
        # Plot sample fits
        for subsample in subsamples:
            ax.plot(absg_range, self.mass_from_absg(absg_range, subsample), color='gray', alpha=0.1, linewidth=1, zorder=1)

        # Plot credible region
        ax.fill_between(absg_range, lower_masses, upper_masses, color='orange',
                       alpha=0.3, label='1 sigma', zorder=2)
        ax.plot(absg_range, median_masses, color='orange',
               label='Median', ls='-', linewidth=1, alpha=0.5, zorder=3)

        # Plot best fit
        ax.plot(absg_range, best_fit_masses, color='#148dde',
               label='Best-Fit params', ls='-.', linewidth=3, zorder=3)

        # Plot true masses from data if provided
        if data is not None:
            try:
                # Extract data columns
                if hasattr(data, 'colnames'):  # astropy.Table
                    m1 = np.array(data['m1'])
                    m2 = np.array(data['m2'])
                    absg1 = np.array(data['absg1'])
                    absg2 = np.array(data['absg2'])
                elif isinstance(data, dict):  # dict
                    m1 = np.array(data['m1'])
                    m2 = np.array(data['m2'])
                    absg1 = np.array(data['absg1'])
                    absg2 = np.array(data['absg2'])
                else:
                    raise ValueError("data must be astropy.Table or dict with 'm1', 'm2', 'absg1', 'absg2' columns")

                # Plot both primary and secondary masses
                ax.scatter(absg1, m1, color='black', s=5, alpha=0.5,
                          label='Truth', zorder=0)
                ax.scatter(absg2, m2, color='black', s=5, alpha=0.5,
                            zorder=0)
            except (KeyError, AttributeError) as e:
                print(f"Could not extract true masses from data: {e}")

        if isochrone_data_feh is not None:
            col = iso_colname_dict
            # Plot isochrone data if provided
            ax.scatter(isochrone_data_feh[col['absg']], isochrone_data_feh[col['mass']], 
                        color='black', s=5, alpha=0.5,label='Isochrone', zorder=0)

        ax.set_xlabel('$M_{\\mathrm{G}}$ [mag]', fontsize=12)
        ax.set_ylabel('Mass [$M_{\\odot}$]', fontsize=12)
        ax.set_xlim(self.absg_min, self.absg_max)
        ax.set_ylim(0.05, 1.1)
        ax.set_yscale('log')
        ax.legend(fontsize=10)
        ax.invert_xaxis()
        
        title = f'Broken Power-Law fit with {self.uncertainty_model.capitalize()} uncertainty ({self.n_segments} segments)'
        plt.title(title, fontsize=14)
        plt.tight_layout()
        os.makedirs(output_dir, exist_ok=True) if output_dir else None
        outlier_tag = 'outlierfit' if self.fit_outlier_params else 'outlierfixed'
        metadata_suffix = output_suffix if output_suffix else f'_model-brokenpowerlaw_unc-{self.uncertainty_model}_outlier-{outlier_tag}'
        filename = f'fit{metadata_suffix}.png'
        plt.savefig(f'{output_dir}/{filename}', dpi=300)
        plt.close(fig)


class PolyMassAbsgModel:
    """
    Global low-order polynomial model for the mass–M_G relation.

        log10(M / Msun) = sum_{i=0}^order c_i * x^i

    where x is a rescaled version of the absolute G magnitude M_G,
    mapped from [absg_min, absg_max] to [-1, 1]. This rescaling greatly
    improves numerical conditioning compared to using raw magnitudes or
    a simple pivot.

    Parameters
    ----------
    order : int
        Polynomial order (typically 2–3 is enough).
    absg_min, absg_max : float
        Magnitude range you care about. Used for the x-rescaling and
        for the derivative-penalty grid.
    mass_min : float
        Lower floor for the mass (in Msun), used as a safety fallback.
    pivot : float or None
        Kept for backwards compatibility but not used in the current
        implementation (x-rescaling replaces it).
    deriv_penalty_strength : float
        Strength of the monotonicity penalty in log-prob units.
        Larger values enforce d log10(M) / d M_G <= 0 more strongly.
    """

    def __init__(self,
                 order: int = 3,
                 absg_min: float = -1.0,
                 absg_max: float = 15.0,
                 mass_min: float = 0.05,
                 pivot: Union[float, None] = None,
                 deriv_penalty_strength: float = 2.0):
        self.order = int(order)
        self.n_params = self.order + 1  # c_0,...,c_order
        self.absg_min = float(absg_min)
        self.absg_max = float(absg_max)
        self.mass_min = float(mass_min)
        # Pivot kept for API compatibility but no longer used explicitly
        self.pivot = (absg_min + absg_max) / 2.0 if pivot is None else float(pivot)
        self.deriv_penalty_strength = float(deriv_penalty_strength)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _split_params(self, params):
        """For API compatibility with the broken-power-law class.

        Here ``params`` is just the polynomial coefficients c_0,...,c_order.
        """
        return params  # (order+1,)

    def _absg_to_x_np(self, absg):
        """Map M_G in [absg_min, absg_max] -> x in [-1, 1] (NumPy)."""
        return 2.0 * (absg - self.absg_min) / (self.absg_max - self.absg_min) - 1.0

    def _absg_to_x_jax(self, absg):
        """Map M_G in [absg_min, absg_max] -> x in [-1, 1] (JAX)."""
        import jax.numpy as jnp
        return 2.0 * (absg - self.absg_min) / (self.absg_max - self.absg_min) - 1.0

    # ------------------------------------------------------------------
    # NumPy version: mass_from_absg
    # ------------------------------------------------------------------
    def mass_from_absg(self, absg, params):
        """NumPy version of mass_from_absg.

        Parameters
        ----------
        absg : array_like
            Absolute G magnitudes.
        params : array_like, shape (order+1,)
            Polynomial coefficients [c_0, ..., c_order].

        Returns
        -------
        mass : ndarray
            Mass in solar masses, same shape as absg.
        """
        absg = np.asarray(absg)
        coeffs = np.asarray(self._split_params(params))

        # Rescaled magnitude x in [-1, 1]
        x = self._absg_to_x_np(absg)
        powers = np.stack([x ** i for i in range(self.order + 1)], axis=-1)  # (..., order+1)
        log10_m = np.sum(coeffs * powers, axis=-1)
        mass = np.power(10.0, log10_m)

        # Safety floor
        mass = np.where(mass > 0.0, mass, self.mass_min)
        return mass

    # ------------------------------------------------------------------
    # JAX versions
    # ------------------------------------------------------------------
    def mass_from_absg_jax(self, absg, params):
        """JAX-compatible mass_from_absg for use in NumPyro.

        Parameters
        ----------
        absg : jax.numpy.ndarray
            Absolute G magnitudes.
        params : jax.numpy.ndarray, shape (order+1,)
            Polynomial coefficients [c_0, ..., c_order].

        Returns
        -------
        mass : jax.numpy.ndarray
            Mass in solar masses.
        """
        import jax.numpy as jnp

        coeffs = self._split_params(params)  # (order+1,)

        # Rescaled magnitude x in [-1, 1]
        x = self._absg_to_x_jax(absg)
        # Build [1, x, x^2, ...] along last axis
        powers = jnp.stack([jnp.power(x, i) for i in range(self.order + 1)], axis=-1)
        log10_m = jnp.sum(coeffs * powers, axis=-1)
        mass = jnp.power(10.0, log10_m)

        # Safety floor
        mass = jnp.where(mass > 0.0, mass, self.mass_min)
        return mass

    def derivative_penalty_jax(self, params, n_grid: int = 64):
        """Compute a soft penalty enforcing d log10(M) / d M_G <= 0.

        The derivative is evaluated over [absg_min, absg_max] using a grid
        in M_G and the chain rule with the rescaled variable x. This
        returns a *log-probability* penalty that you can plug into
        NumPyro via::

            numpyro.factor("poly_monotonicity",
                           model.derivative_penalty_jax(params))

        Parameters
        ----------
        params : jax.numpy.ndarray, shape (order+1,)
            Polynomial coefficients [c_0, ..., c_order].
        n_grid : int
            Number of grid points in M_G over which to evaluate the derivative.

        Returns
        -------
        penalty : jax.numpy.ndarray (scalar)
            Log-prob penalty (typically negative or zero).
        """
        import jax.numpy as jnp

        coeffs = self._split_params(params)  # (order+1,)

        if self.order == 0:
            # Constant polynomial: derivative is zero -> no penalty
            return jnp.array(0.0)

        # Grid in M_G, then convert to x
        absg_grid = jnp.linspace(self.absg_min, self.absg_max, n_grid)
        x_grid = self._absg_to_x_jax(absg_grid)

        # Derivative of polynomial in x:
        # p(x) = Σ c_i x^i  =>  p'(x) = Σ_{i>=1} i * c_i x^{i-1}
        dlog10m_dx = jnp.zeros_like(x_grid)
        for i in range(1, self.order + 1):
            dlog10m_dx = dlog10m_dx + i * coeffs[i] * jnp.power(x_grid, i - 1)

        # Chain rule: d log10 M / d M_G = d log10 M / d x * d x / d M_G
        dx_dabsg = 2.0 / (self.absg_max - self.absg_min)
        dlog10m_dabsg = dlog10m_dx * dx_dabsg

        # Enforce d log10(M)/d M_G <= 0  (mass decreases with increasing M_G)
        # -> penalise positive derivatives
        violations = jnp.maximum(dlog10m_dabsg, 0.0)

        # Quadratic penalty, averaged over the grid
        mean_sq_violation = jnp.mean(violations ** 2)

        # Convert to log-prob penalty; negative (or zero if no violation)
        penalty = -self.deriv_penalty_strength * mean_sq_violation
        return penalty



class PolynomialMLR:
    """
    Parametric Bayesian inference for mass-luminosity relations using
    a single low-order polynomial in log10(M) as a function of M_G.

    This mirrors the API of NonParametricMLR and BrokenPowerLawMLR:
    - set_data(u_values, u_sigma_values, absg1_values, absg2_values)
    - run_numpyro(...) to sample posterior coefficients
    - mass_from_absg(absg, params) helper for plotting
    - plot_results / plot_fitting_results for quick visualization
    """

    def __init__(self, order=3, mass_min=0.05, mass_max=2.0,
                 absg_min=-1.0, absg_max=15.0, pivot=None,
                 uncertainty_model='rice',
                 f_outlier=0, outlier_u0=30, outlier_sigma=15, outlier_kappa=50.0,
                 fit_outlier_params=False, deriv_penalty_strength=10.0,
                 coeff_prior_scale=5.0):
        if uncertainty_model not in ['rice', 'gaussian']:
            raise ValueError("uncertainty_model must be 'rice' or 'gaussian'")

        self.uncertainty_model = uncertainty_model
        self.mass_min = mass_min
        self.mass_max = mass_max
        self.absg_min = absg_min
        self.absg_max = absg_max
        self.poly_model = PolyMassAbsgModel(
            order=order,
            absg_min=absg_min,
            absg_max=absg_max,
            mass_min=mass_min,
            pivot=pivot,
            deriv_penalty_strength=deriv_penalty_strength
        )

        # Outlier handling matches other MLR classes
        self.fit_outlier_params = fit_outlier_params
        self.f_outlier_init = f_outlier
        self.p_epsilon = 1e-10
        self.outlier_u0_init = outlier_u0
        self.outlier_sigma_init = outlier_sigma
        self.outlier_kappa_init = outlier_kappa

        # Data placeholders
        self.u_values = None
        self.u_sigma_values = None
        self.absg1_values = None
        self.absg2_values = None
        self.norm_factor = None

        # Results placeholders
        self.sampler = None
        self.samples = None
        self.results = None

        self.param_names = [f'c_{i}' for i in range(self.poly_model.n_params)]
        self.coeff_prior_scale = coeff_prior_scale

    def func_pu_8(self, tilde_u, A=4.95e-3, B=2.24e-3, C=3.85, u0=36.09):
        """
        The probability distribution for the normalized parameter ũ = u/√mtot
        """
        return A * tilde_u * np.exp(-1 * (
            B * tilde_u**2 + np.exp((tilde_u - u0) / C)
        ))

    def set_data(self, u_values, u_sigma_values=None, absg1_values=None, absg2_values=None, outlier_kappa=None):
        """
        Set observational data.
        """
        self.u_values = np.array(u_values)
        self.absg1_values = np.array(absg1_values)
        self.absg2_values = np.array(absg2_values)

        if u_sigma_values is None:
            self.u_sigma_values = None
            self.norm_factor = np.ones(len(u_values))
        else:
            self.u_sigma_values = np.array(u_sigma_values)

            if self.uncertainty_model == 'rice':
                self.norm_factor = np.ones(len(u_values))
            elif self.uncertainty_model == 'gaussian':
                self.norm_factor = 1 - norm.cdf(0, loc=self.u_values, scale=self.u_sigma_values)

        self.outlier_kappa = self.outlier_kappa_init if outlier_kappa is None else outlier_kappa

    def mass_from_absg(self, absg, params):
        """
        Convenience wrapper around PolyMassAbsgModel.mass_from_absg that
        supports both single and multiple parameter sets.
        """
        absg = np.asarray(absg)
        params = np.asarray(params)
        single = params.ndim == 1
        if single:
            params = params.reshape(1, -1)

        masses = np.array([self.poly_model.mass_from_absg(absg, p) for p in params])
        return masses[0] if single else masses

    def run_numpyro(self, num_warmup=1000, num_samples=2000, num_chains=4,
                    int_umax=80., int_du=0.02, seed=None, use_dense_mass=False,
                    coeff_prior_scale=None, **kwargs):
        """
        Run HMC sampling for polynomial MLR coefficients.
        """
        try:
            # Configure JAX to prefer GPU but fall back to CPU
            jax = _configure_jax_gpu_fallback()
            import jax.numpy as jnp
            from jax.scipy.special import i0e  # Modified Bessel function I_0 * exp(-|x|)
            import numpyro
            import numpyro.distributions as dist
            from numpyro.distributions import constraints
            from numpyro.infer import MCMC, NUTS
        except ImportError as exc:
            raise ImportError("numpyro is not installed. Install it with: pip install numpyro") from exc

        if self.u_values is None:
            raise ValueError("No data set. Use set_data() first.")

        # Use provided prior scale or default
        coeff_prior_scale = coeff_prior_scale or self.coeff_prior_scale

        u_values_jax = jnp.array(self.u_values)
        absg1_values_jax = jnp.array(self.absg1_values)
        absg2_values_jax = jnp.array(self.absg2_values)
        norm_factor_jax = jnp.array(self.norm_factor)

        if self.u_sigma_values is not None:
            u_sigma_values_jax = jnp.array(self.u_sigma_values)
            int_ulist_jax = jnp.arange(0., int_umax, int_du)
        else:
            u_sigma_values_jax = None
            int_ulist_jax = None

        # Shared helper functions (mirrors NonParametricMLR)
        def func_pu_8_jax(tilde_u, A=4.95e-3, B=2.24e-3, C=3.85, u0=36.09):
            return A * tilde_u * jnp.exp(-1 * (
                B * tilde_u**2 + jnp.exp((tilde_u - u0) / C)
            ))

        def gaussian_jax(x, mu, sigma):
            x, mu, sigma = jnp.broadcast_arrays(x, mu, jnp.maximum(sigma, 1e-10))
            coeff = 1.0 / (sigma * jnp.sqrt(2 * jnp.pi))
            exponent = -0.5 * ((x - mu) / sigma)**2
            return coeff * jnp.exp(exponent)

        def rice_distribution_jax(u_obs, u_true, sigma):
            u_obs, u_true, sigma = jnp.broadcast_arrays(
                u_obs, u_true, jnp.maximum(sigma, 1e-10)
            )
            sigma_sq = sigma**2

            bessel_arg = u_obs * u_true / sigma_sq

            log_prefactor = jnp.log(u_obs + 1e-100) - jnp.log(sigma_sq)
            log_exp_term = -(u_obs**2 + u_true**2) / (2 * sigma_sq)
            log_bessel = jnp.log(i0e(bessel_arg) + 1e-100) + jnp.abs(bessel_arg)

            log_rice_pdf = log_prefactor + log_exp_term + log_bessel
            rice_pdf = jnp.exp(log_rice_pdf)
            return jnp.maximum(rice_pdf, 1e-100)

        def outlier_gaussian_jax(tilde_u, outlier_u0, outlier_sigma):
            sigma = jnp.maximum(outlier_sigma, 1e-10)
            coeff = 1.0 / (sigma * jnp.sqrt(2 * jnp.pi))
            exponent = -0.5 * ((tilde_u - outlier_u0) / sigma)**2
            return coeff * jnp.exp(exponent)

        def likelihood_single_u_jax(u, sqrt_mtot, u_sigma=None, norm_factor=None,
                                    f_outlier=None, f_good=None, outlier_u0=None, outlier_sigma=None):
            if f_outlier is None:
                f_outlier = self.f_outlier_init
            if outlier_u0 is None:
                outlier_u0 = self.outlier_u0_init
            if outlier_sigma is None:
                outlier_sigma = self.outlier_sigma_init
            f_good = 1.0 - f_outlier if f_good is None else f_good

            valid = (u > 0) & (sqrt_mtot > 0)
            u = jnp.where(valid, u, 1e-10)
            sqrt_mtot = jnp.where(valid, sqrt_mtot, 1e-10)

            tilde_u = u / sqrt_mtot

            if u_sigma is None:
                good_component = (1.0 / sqrt_mtot) * func_pu_8_jax(tilde_u)
                norm = jnp.ones_like(good_component)
            else:
                u_sigma = jnp.maximum(u_sigma, 1e-10)

                u_obs_grid = (u / sqrt_mtot)[:, None]
                sigma_grid = (u_sigma / sqrt_mtot)[:, None]
                integration_grid = int_ulist_jax[None, :]

                if self.uncertainty_model == 'rice':
                    uncertainty_dist = rice_distribution_jax(
                        u_obs_grid,
                        integration_grid,
                        sigma_grid
                    )
                elif self.uncertainty_model == 'gaussian':
                    uncertainty_dist = gaussian_jax(
                        u_obs_grid,
                        integration_grid,
                        sigma_grid
                    )

                integrand = (
                    (1.0 / sqrt_mtot[:, None])
                    * func_pu_8_jax(int_ulist_jax)[None, :]
                    * uncertainty_dist
                    * int_du
                )

                norm = jnp.maximum(norm_factor, 1e-10)
                good_component = jnp.sum(integrand, axis=1) / norm + self.p_epsilon / (int_umax / int_du)

            outlier_component = outlier_gaussian_jax(tilde_u, outlier_u0, outlier_sigma) / jnp.maximum(norm, 1e-10)

            total_prob = f_good * good_component + f_outlier * outlier_component + self.p_epsilon
            total_prob = jnp.clip(total_prob, 1e-100, 1e10)
            return total_prob

        def model(u_values, u_sigma_values, absg1_values, absg2_values, norm_factor):
            coeffs = numpyro.sample(
                'coeffs',
                dist.Normal(0.0, coeff_prior_scale).expand([self.poly_model.n_params])
            )

            if self.poly_model.deriv_penalty_strength > 0:
                numpyro.factor("poly_monotonicity", self.poly_model.derivative_penalty_jax(coeffs))

            m1 = self.poly_model.mass_from_absg_jax(absg1_values, coeffs)
            m2 = self.poly_model.mass_from_absg_jax(absg2_values, coeffs)
            mtot = m1 + m2
            sqrt_mtot = jnp.sqrt(mtot)

            if self.fit_outlier_params:
                pi = 0.01
                kappa = self.outlier_kappa
                alpha = pi * kappa
                beta_param = (1 - pi) * kappa
                f_outlier = numpyro.sample("f_outlier", dist.Beta(alpha, beta_param))          
                f_good = 1.0 - f_outlier
                
                # Tail separation thresholds (choose once; can be data-driven)
                u_tail_min = jnp.quantile(u_values / sqrt_mtot, 0.6)
                u_tail_max = int_umax

                # Outlier mean restricted to the tail region
                outlier_u0 = numpyro.sample(
                    "outlier_u0",
                    dist.TruncatedNormal(low=u_tail_min, high=u_tail_max, loc=u_tail_min+5.0, scale=5.0)
                )
                
                rho_sigma = numpyro.sample("rho_sigma", dist.Normal(0.0, 1.0))
                sigma_min_out = 10.0
                outlier_sigma = sigma_min_out * (1.0 + jax.nn.softplus(rho_sigma))
                numpyro.deterministic("outlier_sigma", outlier_sigma)
            else:
                f_outlier = self.f_outlier_init
                f_good = 1.0 - f_outlier
                outlier_u0 = self.outlier_u0_init
                outlier_sigma = self.outlier_sigma_init

            log_likelihood_vec = jnp.log(likelihood_single_u_jax(
                u_values, sqrt_mtot, u_sigma_values, norm_factor,
                f_outlier=f_outlier, f_good=f_good,
                outlier_u0=outlier_u0, outlier_sigma=outlier_sigma
            ))

            log_likelihood_sum = jnp.sum(log_likelihood_vec)
            log_likelihood_sum = jnp.where(
                jnp.isfinite(log_likelihood_sum),
                log_likelihood_sum,
                -1e10
            )

            numpyro.factor('obs', log_likelihood_sum)

        nuts_kernel = NUTS(model, dense_mass=use_dense_mass, **kwargs)
        mcmc = MCMC(
            nuts_kernel,
            num_warmup=num_warmup,
            num_samples=num_samples,
            num_chains=num_chains,
            progress_bar=True
        )

        rng_key = jax.random.PRNGKey(seed if seed is not None else 42)
        mcmc.run(rng_key, u_values_jax, u_sigma_values_jax,
                 absg1_values_jax, absg2_values_jax,
                 norm_factor_jax)

        self.sampler = mcmc
        samples_dict = mcmc.get_samples()

        # Extract coefficient samples
        coeffs_samples = np.array(samples_dict['coeffs'])

        # Include outlier parameters if they were fitted
        if self.fit_outlier_params:
            f_outlier_samples = np.array(samples_dict['f_outlier'])
            outlier_u0_samples = np.array(samples_dict['outlier_u0'])
            outlier_sigma_samples = np.array(samples_dict['outlier_sigma'])

            # Combine all parameters
            self.samples = np.column_stack([
                coeffs_samples, f_outlier_samples, outlier_u0_samples, outlier_sigma_samples
            ])

            # Update parameter names to include outlier parameters
            self.param_names = self.param_names + ['f_outlier', 'outlier_u0', 'outlier_sigma']
        else:
            self.samples = coeffs_samples

        self.results = type('obj', (object,), {
            'samples': self.samples,
            'logz': None,
            'logzerr': None
        })()

        mcmc.print_summary()
        return mcmc

    def plot_results(self, output_dir='.', output_suffix=''):
        """Corner plot for polynomial coefficients."""
        if self.samples is None or self.samples.size == 0:
            print("No samples available for plotting.")
            return

        os.makedirs(output_dir, exist_ok=True)
        labels = [f'$c_{i}$' for i in range(self.poly_model.n_params)]

        # Add outlier parameter labels if they were fitted
        if self.fit_outlier_params:
            labels.extend(['f_outlier', 'outlier_u0', 'outlier_sigma'])

        try:
            import corner
            fig = corner.corner(self.samples, labels=labels, show_titles=True)
            plt.tight_layout()
            outlier_tag = 'outlierfit' if self.fit_outlier_params else 'outlierfixed'
            metadata_suffix = output_suffix if output_suffix else f'_model-polynomial_unc-{self.uncertainty_model}_outlier-{outlier_tag}'
            filename = f'corner{metadata_suffix}.png'
            plt.savefig(f'{output_dir}/{filename}', dpi=300)
            plt.close(fig)
        except ImportError:
            print("corner package not installed. Install it with: pip install corner")

    def plot_fitting_results(self, data=None, output_dir='', output_suffix='', isochrone_data_feh=None, iso_colname_dict={'absg':'absg', 'mass':'mass', 'feh':'MH'}):
        """Plot posterior predictive mass-luminosity relation."""
        if self.samples is None or self.samples.size == 0:
            print("No samples available for plotting fitting results.")
            return

        # Extract only polynomial coefficients for plotting (exclude outlier parameters)
        if self.fit_outlier_params:
            # Exclude last 3 outlier parameters (f_outlier, outlier_u0, outlier_sigma)
            coeff_samples = self.samples[:, :-3]
        else:
            coeff_samples = self.samples

        num_samples = min(1000, len(coeff_samples))
        indices = np.random.choice(len(coeff_samples), size=num_samples, replace=False)
        subsamples = coeff_samples[indices]

        absg_range = np.linspace(self.poly_model.absg_min, self.poly_model.absg_max, 1000)
        all_masses = self.mass_from_absg(absg_range, coeff_samples)
        lower_masses, median_masses, upper_masses = np.percentile(all_masses, [16, 50, 84], axis=0)

        median_params = np.median(coeff_samples, axis=0)
        best_fit_masses = self.mass_from_absg(absg_range, median_params)

        fig, ax = plt.subplots(figsize=(10, 6))
        for subsample in subsamples:
            ax.plot(absg_range, self.mass_from_absg(absg_range, subsample), color='gray', alpha=0.1, linewidth=1, zorder=1)

        ax.fill_between(absg_range, lower_masses, upper_masses, color='orange',
                        alpha=0.3, label='1 sigma', zorder=2)
        ax.plot(absg_range, median_masses, color='orange',
                label='Median', ls='-', linewidth=1, alpha=0.5, zorder=3)
        ax.plot(absg_range, best_fit_masses, color='#148dde',
                label='Best-Fit params', ls='-.', linewidth=3, zorder=3)

        if data is not None:
            try:
                if hasattr(data, 'colnames'):
                    m1 = np.array(data['m1'])
                    m2 = np.array(data['m2'])
                    absg1 = np.array(data['absg1'])
                    absg2 = np.array(data['absg2'])
                elif isinstance(data, dict):
                    m1 = np.array(data['m1'])
                    m2 = np.array(data['m2'])
                    absg1 = np.array(data['absg1'])
                    absg2 = np.array(data['absg2'])
                else:
                    raise ValueError("data must be astropy.Table or dict with 'm1', 'm2', 'absg1', 'absg2' columns")

                ax.scatter(absg1, m1, color='black', s=5, alpha=0.5,
                           label='Truth', zorder=0)
                ax.scatter(absg2, m2, color='black', s=5, alpha=0.5,
                           zorder=0)
            except (KeyError, AttributeError) as e:
                print(f"Could not extract true masses from data: {e}")

        if isochrone_data_feh is not None:
            col = iso_colname_dict
            # Plot isochrone data if provided
            ax.scatter(isochrone_data_feh[col['absg']], isochrone_data_feh[col['mass']], 
                        color='black', s=5, alpha=0.5,label='Isochrone', zorder=0)


        ax.set_xlabel('$M_{\\mathrm{G}}$ [mag]', fontsize=12)
        ax.set_ylabel('Mass [$M_{\\odot}$]', fontsize=12)
        ax.set_xlim(self.poly_model.absg_min, self.poly_model.absg_max)
        ax.set_ylim(self.mass_min * 0.8, self.mass_max * 1.2)
        ax.set_yscale('log')
        ax.legend(fontsize=10)
        ax.invert_xaxis()

        title = f'Polynomial fit (order={self.poly_model.order}) with {self.uncertainty_model.capitalize()} uncertainty'
        plt.title(title, fontsize=14)
        plt.tight_layout()
        os.makedirs(output_dir, exist_ok=True) if output_dir else None
        outlier_tag = 'outlierfit' if self.fit_outlier_params else 'outlierfixed'
        metadata_suffix = output_suffix if output_suffix else f'_model-polynomial_unc-{self.uncertainty_model}_outlier-{outlier_tag}'
        filename = f'fit{metadata_suffix}.png'
        plt.savefig(f'{output_dir}/{filename}', dpi=300)
        plt.close(fig)


class MultiMetallicityFitter:
    """
    Wrapper class to fit multiple metallicity bins.

    This class provides a convenient interface for fitting mass-luminosity
    relations in different metallicity bins using either non-parametric
    or broken power-law parameterizations.
    """

    def __init__(self, model_type='nonparametric', n_absg_bins=10, absg_min=4.0, absg_max=12.0, mass_min=0.05, mass_max=2.0,
                 uncertainty_model='rice', f_outlier=0, outlier_u0=30, outlier_sigma=15, outlier_kappa=50.0,
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
                    num_warmup=1000, num_samples=2000, num_chains=4, seed=None):
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
                    fit_outlier_params=self.fit_outlier_params
                )

                # Set data
                fitter.set_data(
                    u_values=u_values,
                    u_sigma_values=u_sigma_values,
                    absg1_values=absg1_values,
                    absg2_values=absg2_values,
                    gamma=gamma,
                    outlier_kappa=effective_kappa
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
                fitter = BrokenPowerLawMLR(
                    n_segments=self.n_segments,
                    absg_min=self.absg_min,
                    absg_max=self.absg_max,
                    uncertainty_model=self.uncertainty_model,
                    f_outlier=self.f_outlier,
                    outlier_u0=self.outlier_u0,
                    outlier_sigma=self.outlier_sigma,
                    outlier_kappa=effective_kappa,
                    fit_outlier_params=self.fit_outlier_params
                )

                # Set data
                fitter.set_data(
                    u_values=u_values,
                    u_sigma_values=u_sigma_values,
                    absg1_values=absg1_values,
                    absg2_values=absg2_values,
                    outlier_kappa=effective_kappa
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
                    fit_outlier_params=self.fit_outlier_params,
                    deriv_penalty_strength=self.poly_deriv_penalty_strength,
                    coeff_prior_scale=self.poly_coeff_prior_scale,
                )

                fitter.set_data(
                    u_values=u_values,
                    u_sigma_values=u_sigma_values,
                    absg1_values=absg1_values,
                    absg2_values=absg2_values,
                    outlier_kappa=effective_kappa
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
