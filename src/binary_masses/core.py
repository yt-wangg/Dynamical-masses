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


class PU8Sampler:
    """
    Sampler for the p(ũ) distribution (version 8).

    The probability distribution for the normalized parameter ũ = u/√mtot
    """

    def __init__(self, A=4.95e-3, B=2.24e-3, C=3.85, u0=36.09):
        """
        Initialize the sampler with distribution parameters.

        Parameters
        ----------
        A, B, C, u0 : float
            Distribution parameters for the p(ũ) function
        """
        self.A = A
        self.B = B
        self.C = C
        self.u0 = u0

    def func_pu_8(self, tilde_u):
        """
        The probability distribution for the normalized parameter ũ = u/√mtot

        Parameters
        ----------
        tilde_u : array_like
            Normalized u values

        Returns
        -------
        array_like : probability density values
        """
        return self.A * tilde_u * torch.exp(-1 * (
            self.B * tilde_u**2 + torch.exp((tilde_u - self.u0) / self.C)
        ))

    def find_sampling_bounds(self, search_range=(0, 50), num_points=1000):
        """
        Find reasonable bounds for sampling by analyzing the distribution.

        Parameters
        ----------
        search_range : tuple
            Search range for ũ values
        num_points : int
            Number of points to use for finding bounds

        Returns
        -------
        tuple : (u_min, u_max, M) where M is the maximum PDF value
        """
        tilde_u_range = torch.linspace(search_range[0], search_range[1], num_points)
        pdf_values = self.func_pu_8(tilde_u_range)

        # Find where PDF becomes negligible (e.g., < 1e-6 of max value)
        max_pdf = torch.max(pdf_values)
        threshold = max_pdf * 1e-6

        valid_indices = pdf_values > threshold
        if torch.any(valid_indices):
            u_min = tilde_u_range[valid_indices][0].item()
            u_max = tilde_u_range[valid_indices][-1].item()
        else:
            u_min, u_max = search_range

        # Find maximum value for rejection sampling
        M = max_pdf.item()

        return u_min, u_max, M

    def rejection_sample(self, n_samples=1000, u_min=None, u_max=None, M=None):
        """
        Sample from the distribution using rejection sampling.

        Parameters
        ----------
        n_samples : int
            Number of samples to generate
        u_min, u_max, M : float, optional
            Sampling bounds and maximum PDF value (computed if None)

        Returns
        -------
        array : samples from the distribution
        """
        if u_min is None or u_max is None or M is None:
            u_min, u_max, M = self.find_sampling_bounds()

        samples = []
        n_attempts = 0
        max_attempts = n_samples * 100  # Prevent infinite loops

        while len(samples) < n_samples and n_attempts < max_attempts:
            # Generate candidate from uniform distribution
            u_candidate = torch.rand(1) * (u_max - u_min) + u_min

            # Generate uniform random number for acceptance test
            y = torch.rand(1) * M

            # Accept if y <= f(u_candidate)
            if y <= self.func_pu_8(u_candidate):
                samples.append(u_candidate.item())

            n_attempts += 1

        if len(samples) < n_samples:
            print(f"Warning: Only generated {len(samples)} samples out of {n_samples} requested")

        return np.array(samples)

    def inverse_transform_sample(self, n_samples=1000, u_min=None, u_max=None, num_cdf_points=10000):
        """
        Sample using inverse transform sampling (numerical approximation).

        Parameters
        ----------
        n_samples : int
            Number of samples to generate
        u_min, u_max : float, optional
            Sampling bounds (computed if None)
        num_cdf_points : int
            Number of points for CDF computation

        Returns
        -------
        array : samples from the distribution
        """
        if u_min is None or u_max is None:
            u_min, u_max, _ = self.find_sampling_bounds()

        # Create grid for numerical CDF computation
        u_grid = torch.linspace(u_min, u_max, num_cdf_points)
        pdf_values = self.func_pu_8(u_grid)

        # Numerical integration to get CDF
        du = (u_max - u_min) / (num_cdf_points - 1)
        cdf_values = torch.cumsum(pdf_values * du, dim=0)
        cdf_values = cdf_values / cdf_values[-1]  # Normalize to make it a proper CDF

        # Generate uniform random numbers
        uniform_samples = torch.rand(n_samples)

        # Interpolate to find corresponding u values
        samples = np.interp(uniform_samples.numpy(), cdf_values.numpy(), u_grid.numpy())

        return samples

    def sample(self, n_samples=1000, method='rejection'):
        """
        Main sampling function.

        Parameters
        ----------
        n_samples : int
            Number of samples to generate
        method : str
            Sampling method: 'rejection' or 'inverse_transform'

        Returns
        -------
        array : samples from the distribution
        """
        if method == 'rejection':
            return self.rejection_sample(n_samples)
        elif method == 'inverse_transform':
            return self.inverse_transform_sample(n_samples)
        else:
            raise ValueError("Method must be 'rejection' or 'inverse_transform'")


class NonParametricMLR:
    """
    Non-parametric Bayesian inference for mass-luminosity relations.

    Uses NumPyro with NUTS sampler for posterior inference of mass-luminosity
    relations in bins of absolute magnitude. Supports Rice or Gaussian uncertainty
    models for the u parameter.
    """

    def __init__(self, n_bins=10, absg_min=4.0, absg_max=12.0, uncertainty_model='rice',
                 f_outlier=0, outlier_u0=30, outlier_sigma=15):
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
        uncertainty_model : str
            Uncertainty model to use: 'rice' or 'gaussian' (default: 'rice')
        """
        # Fixed parameters as specified
        self.f_outlier = f_outlier
        self.f_good = 1 - self.f_outlier 
        self.p_epsilon = 1e-10
        self.m_epsilon = 1e-10
        self.outlier_u0 = outlier_u0
        self.outlier_sigma = outlier_sigma

        # Non-parametric bin settings
        self.n_bins = n_bins
        self.absg_min = absg_min
        self.absg_max = absg_max
        self.absg_bins = np.linspace(absg_min, absg_max, n_bins)

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
                absg1_values=None, absg2_values=None, gamma=None):
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

    def run_numpyro(self, num_warmup=1000, num_samples=2000, num_chains=4,
                    mass_min=0.01, mass_max=2, int_umax=80., int_du=0.02,
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
            import jax
            import jax.numpy as jnp
            from jax.scipy.special import i0e  # Modified Bessel function I_0 * exp(-|x|)
            import numpyro
            import numpyro.distributions as dist
            from numpyro.infer import MCMC, NUTS
            print(f"JAX devices: {jax.devices()}")
            print(f"Device count: {jax.local_device_count()}")
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

        def likelihood_single_u_jax(u, sqrt_mtot, u_sigma=None, norm_factor=None):
            """
            JAX version of likelihood for a single u value with chosen uncertainty model.

            Implements the integral:
            (1/sqrt(m_tot)) * ∫ p(ũ) * P(u_obs | ũ, σ) dũ

            where ũ = u_true/sqrt(m_tot) is the integration variable.
            """
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
                outlier_component = outlier_gaussian_jax(tilde_u, self.outlier_u0, self.outlier_sigma) / norm_factor

            total_prob = self.f_good * good_component + self.f_outlier * outlier_component + self.p_epsilon

            # Clamp to reasonable range to avoid log(0)
            total_prob = jnp.clip(total_prob, 1e-100, 1e10)
            return total_prob

        # Define the NumPyro model with regularization
        def model(u_values, u_sigma_values, absg1_values, absg2_values, norm_factor, 
                  absg_bins, 
                  gamma=None):

            # Prior: uniform for each mass bin
            ndim = self.n_bins
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

            # Calculate log likelihood for each data point
            log_likelihood_vec = jnp.log(likelihood_single_u_jax(u_values, sqrt_mtot, u_sigma_values, norm_factor=norm_factor))

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
        try:
            import corner
            fig = corner.corner(self.samples, labels=labels, truths=truths,
                               truth_color='salmon', show_titles=True)
            plt.tight_layout()
            plt.savefig(f'{output_dir}/corner_plot_nonparam_{self.uncertainty_model}{output_suffix}.png', dpi=300)
            plt.close(fig)
        except ImportError:
            print("corner package not installed. Install it with: pip install corner")

    def plot_fitting_results(self, data=None, output_dir='', output_suffix=''):
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

        num_samples = min(1000, len(self.samples))
        indices = np.random.choice(len(self.samples), size=num_samples, replace=False)
        resampled_data = self.samples[indices]

        percentiles = np.percentile(self.samples, [16, 50, 84], axis=0)
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


        ax.set_xlim(self.absg_min, self.absg_max)
        ax.set_xlabel('$M_{\\mathrm{G}}$ [mag]', fontsize=12)
        ax.set_ylabel('Mass [$M_{\\odot}$]', fontsize=12)
        ax.set_yscale('log')
        ax.invert_xaxis()
        ax.legend(fontsize=10)
        title = f'Non-parametric fit with {self.uncertainty_model.capitalize()} uncertainty ({self.n_bins} bins)'
        plt.title(title, fontsize=14)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/nonparametric_fit_{self.n_bins}bins_{self.uncertainty_model}{output_suffix}.png', dpi=300)
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
                 f_outlier=0, outlier_u0=30, outlier_sigma=15):
        """
        Initialize the broken power-law MLR fitter.

        Parameters
        ----------
        n_segments : int, optional
            Number of power-law segments (default: 3)
        uncertainty_model : str
            Uncertainty model: 'rice' or 'gaussian' (default: 'rice')
        f_outlier : float
            Fraction of outliers (default: 0)
        outlier_u0 : float
            Outlier distribution center
        outlier_sigma : float
            Outlier distribution width
        """

        self.n_segments = n_segments  # Number of power-law segments

        # Uncertainty model selection
        if uncertainty_model not in ['rice', 'gaussian']:
            raise ValueError("uncertainty_model must be 'rice' or 'gaussian'")
        self.uncertainty_model = uncertainty_model

        # Outlier model parameters
        self.f_outlier = f_outlier
        self.f_good = 1 - f_outlier
        self.p_epsilon = 1e-10
        self.outlier_u0 = outlier_u0
        self.outlier_sigma = outlier_sigma
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

        # Parameter names: a_i, b_i for each segment (intercepts and slopes)
        # Break points are calculated from intersections: M_break_i = (a_i - a_{i+1}) / (b_{i+1} - b_i)
        self.param_names = []
        for i in range(self.n_segments):
            self.param_names.extend([f'a_{i}', f'b_{i}'])

    def _split_params(self, params):
        """Split flattened [a0.., b0..] into intercept and slope arrays."""
        params = np.atleast_1d(params)
        intercepts = params[0::2]
        slopes = params[1::2]
        return intercepts, slopes

    def _compute_break_points(self, intercepts, slopes):
        """
        Compute break points from segment intersections.
        Ensures continuity by construction.

        Parameters
        ----------
        intercepts : array_like
            Intercepts a_i for each segment [a_0, a_1, ..., a_{K-1}]
        slopes : array_like
            Slopes b_i for each segment [b_0, b_1, ..., b_{K-1}]

        Returns
        -------
        break_mags : array_like
            Break magnitudes where segments intersect [M_break_1, ..., M_break_{K-1}]
            Length = n_segments - 1
        """
        break_mags = []
        for i in range(len(intercepts) - 1):
            # Intersection point: a_i + b_i * log10(M) = a_{i+1} + b_{i+1} * log10(M)
            # Solve for M: log10(M) = (a_i - a_{i+1}) / (b_{i+1} - b_i)
            log10_mass_break = (intercepts[i] - intercepts[i+1]) / (slopes[i+1] - slopes[i])

            # Convert to magnitude using the first segment equation
            break_mag = intercepts[i] + slopes[i] * log10_mass_break
            break_mags.append(break_mag)

        return np.array(break_mags)

    def _validate_continuity(self, params):
        """
        Validate that the broken power-law is continuous at break points.

        Parameters
        ----------
        params : array_like
            Flattened parameters [a_0, b_0, a_1, b_1, ...]

        Returns
        -------
        is_continuous : bool
            True if continuous within tolerance
        """
        intercepts, slopes = self._split_params(params)
        break_mags = self._compute_break_points(intercepts, slopes)

        # Check continuity at each break point
        tolerance = 1e-10
        for i, break_mag in enumerate(break_mags):
            # Compute log10(mass) from both adjacent segments at break point
            log10_mass_1 = (break_mag - intercepts[i]) / slopes[i]
            log10_mass_2 = (break_mag - intercepts[i+1]) / slopes[i+1]

            if abs(log10_mass_1 - log10_mass_2) > tolerance:
                return False

        return True

    def absg_from_mass(self, mass, params):
        """
        Compute absolute G magnitude from mass using broken power-law.

        M_G = a_i + b_i * log10(M/M_sun) for segment i

        Parameters
        ----------
        mass : array_like
            Mass values in solar masses
        params : array_like
            Flattened parameters [a_0, b_0, a_1, b_1, ...]
            Can be 1D (single parameter set) or 2D (multiple parameter sets)

        Returns
        -------
        absg : array_like
            Absolute G magnitudes
            If params is 2D, returns shape (n_params, len(mass))
            If params is 1D, returns shape (len(mass),)
        """
        mass = np.atleast_1d(mass)
        mass = np.clip(mass, self.mass_min, self.mass_max)  # Clip (limit) the values in an array.
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

        # Build segment boundaries: [mass_min, break_points, mass_max]
        # boundaries = np.concatenate([[self.mass_min], self.break_points, [self.mass_max]])

        # Process each parameter set
        for param_idx in range(n_param_sets):
            intercepts, slopes = self._split_params(params[param_idx])
            absg = np.zeros_like(mass)
            break_points = 10**((intercepts[1:] - intercepts[:-1]) / (slopes[:-1] - slopes[1:]))
            boundaries = np.concatenate([[self.mass_min], break_points, [self.mass_max]])

            for i in range(self.n_segments):
                a_i = intercepts[i]
                b_i = slopes[i]
                # Use <= for the last segment to include mass_max
                if i == self.n_segments - 1:
                    mask = (mass >= boundaries[i]) & (mass <= boundaries[i+1])
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
            Flattened parameters [a_0, b_0, a_1, b_1, ...]
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

        # Build segment boundaries
        # boundaries = np.concatenate([[self.mass_min], self.break_points, [self.mass_max]])

        # Process each parameter set
        for param_idx in range(n_param_sets):
            intercepts, slopes = self._split_params(params[param_idx])
            mass = np.zeros_like(absg)

            # Compute break points (mass values where segments intersect)
            # At break point: a_i + b_i * log10(M) = a_{i+1} + b_{i+1} * log10(M)
            # Solving: log10(M) = (a_i - a_{i+1}) / (b_{i+1} - b_i)
            break_points = 10**((intercepts[:-1] - intercepts[1:]) / (slopes[1:] - slopes[:-1]))

            # Compute magnitude boundaries (where segments intersect in magnitude space)
            if len(break_points) > 0:
                mag_boundaries = []
                for i, break_point in enumerate(break_points):
                    # Magnitude at break point using segment i
                    mag_break = intercepts[i] + slopes[i] * np.log10(break_point)
                    mag_boundaries.append(mag_break)
                mag_boundaries = np.array(mag_boundaries)
            else:
                mag_boundaries = np.array([])

            # Build magnitude boundaries for segment selection: [absg_min, break_mags, absg_max]
            mag_boundaries_full = np.concatenate([[self.absg_min], mag_boundaries, [self.absg_max]])

            # For each segment, compute mass and check if magnitude falls in this segment's range
            for i in range(self.n_segments):
                a_i = intercepts[i]
                b_i = slopes[i]

                # Compute mass for this segment: M = 10^((M_G - a) / b)
                mass_candidate = 10**((absg - a_i) / b_i)

                # Check if magnitude falls within this segment's magnitude range
                # Use <= for upper bound to include edge values
                if i == self.n_segments - 1:
                    mask = (absg >= mag_boundaries_full[i]) & (absg <= mag_boundaries_full[i+1])
                else:
                    mask = (absg >= mag_boundaries_full[i]) & (absg < mag_boundaries_full[i+1])
                mass[mask] = mass_candidate[mask]

            masses[param_idx] = np.clip(mass, self.mass_min, self.mass_max)

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
            Flattened parameters [a_0, b_0, a_1, b_1, ...]
        break_points : jax array
            Break point masses

        Returns
        -------
        mass : jax array
            Mass values in solar masses
        """
        import jax.numpy as jnp

        intercepts, slopes = self._split_params(params)
        n_segments = self.n_segments

        # Compute break points (mass values where segments intersect)
        # At break point: a_i + b_i * log10(M) = a_{i+1} + b_{i+1} * log10(M)
        # Solving: log10(M) = (a_i - a_{i+1}) / (b_{i+1} - b_i)
        break_points = jnp.array(10**((intercepts[:-1] - intercepts[1:]) / (slopes[1:] - slopes[:-1])))

        # Compute magnitude boundaries (where segments intersect in magnitude space)
        if len(break_points) > 0:
            mag_boundaries = jnp.array([
                intercepts[i] + slopes[i] * jnp.log10(break_points[i])
                for i in range(len(break_points))
            ])
        else:
            mag_boundaries = jnp.array([])

        # Build magnitude boundaries for segment selection: [absg_min, break_mags, absg_max]
        mag_boundaries_full = jnp.concatenate([
            jnp.array([self.absg_min]),
            mag_boundaries,
            jnp.array([self.absg_max])
        ])

        # Initialize with zeros
        mass = jnp.zeros_like(absg)

        # For each segment, compute mass and check if magnitude falls in this segment's range
        for i in range(n_segments):
            a_i = intercepts[i]
            b_i = slopes[i]

            # Compute mass for this segment: M = 10^((M_G - a) / b)
            mass_candidate = jnp.power(10.0, (absg - a_i) / b_i)

            # Check if magnitude falls within this segment's magnitude range
            if i == n_segments - 1:
                # Last segment includes upper bound
                in_segment = (absg >= mag_boundaries_full[i]) & (absg <= mag_boundaries_full[i+1])
            else:
                # Other segments use exclusive upper bound
                in_segment = (absg >= mag_boundaries_full[i]) & (absg < mag_boundaries_full[i+1])

            # Update mass where this segment applies
            mass = jnp.where(in_segment, mass_candidate, mass)

        # Clamp to reasonable range
        mass = jnp.clip(mass, self.mass_min, self.mass_max)

        return mass

    def func_pu_8(self, tilde_u, A=4.95e-3, B=2.24e-3, C=3.85, u0=36.09):
        """
        The probability distribution for the normalized parameter ũ = u/√mtot
        """
        return A * tilde_u * np.exp(-1 * (
            B * tilde_u**2 + np.exp((tilde_u - u0) / C)
        ))

    def set_data(self, u_values, u_sigma_values, absg1_values, absg2_values):
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
        import jax
        import jax.numpy as jnp
        import numpyro
        import numpyro.distributions as dist
        from numpyro.infer import MCMC, NUTS
        from scipy.special import i0 as bessel_i0

        if self.u_values is None:
            raise ValueError("Data not set. Call set_data() first.")

        self.param_names = []
        for i in range(self.n_segments):
            self.param_names.extend([f'a_{i}', f'b_{i}'])

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

        def mass_from_absg_jax_inner(absg, intercepts, slopes, break_points_inner):
            """Compute mass from M_G using broken power-law with enforced continuity.

            For each M_G value, compute mass from each segment's equation and
            select the one where the M_G falls in that segment's magnitude range.
            """
            absg_2d = absg[:, None]  # (N, 1)
            a_2d = intercepts[None, :]  # (1, n_seg)
            b_2d = slopes[None, :]  # (1, n_seg)

            # Mass from each segment's equation: M = 10^((M_G - a) / b)
            mass_all = jnp.power(10.0, (absg_2d - a_2d) / b_2d)  # (N, n_seg)

            # Compute magnitude boundaries (where segments intersect in magnitude space)
            if len(break_points_inner) > 0:
                mag_boundaries = jnp.array([
                    intercepts[i] + slopes[i] * jnp.log10(break_points_inner[i])
                    for i in range(len(break_points_inner))
                ])
            else:
                mag_boundaries = jnp.array([])

            # Build magnitude boundaries for segment selection: [absg_min, break_mags, absg_max]
            mag_boundaries_full = jnp.concatenate([
                jnp.array([self.absg_min]),
                mag_boundaries,
                jnp.array([self.absg_max])
            ])

            # Check if M_G falls in each segment's magnitude range
            absg_expanded = absg_2d  # (N, 1)
            mag_low = mag_boundaries_full[:-1][None, :]  # (1, n_seg)
            mag_high = mag_boundaries_full[1:][None, :]   # (1, n_seg)

            # Last segment includes upper bound, others use exclusive
            if len(intercepts) == 1:
                # Single segment case
                in_segment = (absg_expanded >= mag_low) & (absg_expanded <= mag_high)
            else:
                # Multiple segments: last segment inclusive, others exclusive
                in_segment_all = (absg_expanded >= mag_low) & (absg_expanded < mag_high)
                # Fix last segment to include upper bound
                in_segment_last = (absg_expanded >= mag_low[:, -1:]) & (absg_expanded <= mag_high[:, -1:])
                in_segment = jnp.concatenate([in_segment_all[:, :-1], in_segment_last], axis=1)

            # Select valid mass (sum works since only one should be valid due to continuity)
            mass = jnp.sum(mass_all * in_segment, axis=1)

            # Fallback for edge cases where no segment matched
            mass = jnp.where(mass > 0, mass, self.mass_min)
            return jnp.clip(mass, self.mass_min, self.mass_max)

        def likelihood_single_u_jax(u, sqrt_mtot, u_sigma, norm_factor):
            """Compute likelihood for observed u values."""
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
            
            outlier_component = outlier_gaussian_jax(tilde_u, self.outlier_u0, self.outlier_sigma) / norm_factor

            total_prob = self.f_good * good_component + self.f_outlier * outlier_component + self.p_epsilon
            total_prob = jnp.clip(total_prob, 1e-100, 1e10)
            return total_prob

        # Define the NumPyro model
        def model(u_values, u_sigma_values, absg1_values, absg2_values,
                  norm_factor):

            n_segments = self.n_segments

            # Prior on b with physical constraints (slopes should be negative for stellar MLR)
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

            # Prior on a with reasonable ranges
            intercepts = []
            if a_prior_range is not None:
                for i in range(n_segments):
                    intercepts.append(numpyro.sample(f'a_{i}', dist.Uniform(a_prior_range[0], a_prior_range[1])))
            else:
                raise ValueError("Have to specify a_prior_range for intercepts in broken power-law model.")
            intercepts = jnp.stack(intercepts)

            # Ensure physically valid break points (should be within mass range)
            # At break point: a_i + b_i * log10(M) = a_{i+1} + b_{i+1} * log10(M)
            # Solving: log10(M) = (a_i - a_{i+1}) / (b_{i+1} - b_i)
            break_points = jnp.array(10**((intercepts[:-1] - intercepts[1:]) / (slopes[1:] - slopes[:-1])))

            # Add constraint to ensure break points are within valid range
            for i, break_point in enumerate(break_points):
                numpyro.factor(f'break_point_{i}_valid',
                               jnp.where((break_point >= self.mass_min) & (break_point <= self.mass_max),
                                        0.0, -jnp.inf))

            # Compute break points for mass_from_absg_jax_inner (same formula)
            break_points_inner = jnp.array(10**((intercepts[:-1] - intercepts[1:]) / (slopes[1:] - slopes[:-1])))

            # Compute masses from magnitudes
            m1 = mass_from_absg_jax_inner(absg1_values, intercepts, slopes, break_points_inner)
            m2 = mass_from_absg_jax_inner(absg2_values, intercepts, slopes, break_points_inner)
            mtot = m1 + m2
            sqrt_mtot = jnp.sqrt(mtot)

            # Calculate log likelihood
            log_likelihood_vec = jnp.log(likelihood_single_u_jax(
                u_values, sqrt_mtot, u_sigma_values, norm_factor
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

        # Reconstruct samples as array [n_samples, 2 * n_segments] with continuous intercepts
        slope_samples = np.column_stack([np.array(samples_dict[f'b_{i}']) for i in range(self.n_segments)])
        n_samples_total = len(slope_samples)
        intercept_samples = np.column_stack([np.array(samples_dict[f'a_{i}']) for i in range(self.n_segments)])

        self.samples = np.zeros((n_samples_total, 2 * self.n_segments))
        for i in range(self.n_segments):
            self.samples[:, 2*i] = intercept_samples[:, i]
            self.samples[:, 2*i + 1] = slope_samples[:, i]

        self.results = type('obj', (object,), {
            'samples': self.samples,
            'logz': None,
            'logzerr': None
        })()

        # Compute break points from median parameters for plotting
        median_intercepts = np.median(intercept_samples, axis=0)
        median_slopes = np.median(slope_samples, axis=0)
        # Fixed formula: log10(M) = (a_i - a_{i+1}) / (b_{i+1} - b_i)
        self.break_points = 10**((median_intercepts[:-1] - median_intercepts[1:]) /
                                (median_slopes[1:] - median_slopes[:-1]))

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
            plt.savefig(f'{output_dir}/corner_plot_broken_powerlaw_{self.uncertainty_model}{output_suffix}.png', dpi=300)
            plt.close(fig)
        except ImportError:
            print("corner package not installed. Install it with: pip install corner")

    def plot_fitting_results(self, data=None, output_dir='', output_suffix=''):
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

        # Convert masses to M_G values for the plot (we'll invert later)
        num_samples = min(1000, len(self.samples))
        indices = np.random.choice(len(self.samples), size=num_samples, replace=False)
        subsamples = self.samples[indices]

        # Sample masses for plotting (use M_G range similar to non-parametric)
        absg_range = np.linspace(self.absg_min, self.absg_max, 1000)

        # Compute M_G for each posterior sample
        all_masses = self.mass_from_absg(absg_range, self.samples)
        lower_masses, median_masses, upper_masses = np.percentile(all_masses, [16, 50, 84], axis=0)

        # Best-fit line
        median_params = np.median(self.samples, axis=0)
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

        ax.set_xlabel('$M_{\\mathrm{G}}$ [mag]', fontsize=12)
        ax.set_ylabel('Mass [$M_{\\odot}$]', fontsize=12)
        ax.set_xlim(self.absg_min, self.absg_max)
        ax.set_yscale('log')
        ax.legend(fontsize=10)
        ax.invert_xaxis()
        
        title = f'Broken Power-Law fit with {self.uncertainty_model.capitalize()} uncertainty ({self.n_segments} segments)'
        plt.title(title, fontsize=14)
        plt.tight_layout()
        os.makedirs(output_dir, exist_ok=True) if output_dir else None
        plt.savefig(f'{output_dir}/broken_powerlaw_fit_{self.n_segments}segments_{self.uncertainty_model}{output_suffix}.png', dpi=300)
        plt.close(fig)


class MultiMetallicityFitter:
    """
    Wrapper class to fit multiple metallicity bins.

    This class provides a convenient interface for fitting mass-luminosity
    relations in different metallicity bins using either non-parametric
    or broken power-law parameterizations.
    """

    def __init__(self, model_type='nonparametric', n_absg_bins=10, absg_min=4.0, absg_max=12.0,
                 uncertainty_model='rice', f_outlier=0, outlier_u0=30, outlier_sigma=15,
                 n_segments=3):

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
        """
        self.model_type = model_type.lower()
        if self.model_type not in ['nonparametric', 'broken_powerlaw']:
            raise ValueError("model_type must be 'nonparametric' or 'broken_powerlaw'")

        self.n_absg_bins = n_absg_bins
        self.absg_min = absg_min
        self.absg_max = absg_max
        self.uncertainty_model = uncertainty_model
        self.f_outlier = f_outlier
        self.f_good = 1 - f_outlier
        self.outlier_u0 = outlier_u0
        self.outlier_sigma = outlier_sigma

        # Set default break points for broken power law
        self.n_segments = n_segments

        self.fitters = {}  # Dictionary to store fitters for each metallicity bin
        self.feh_bin_edges = None
        self.feh_bin_centers = None

    def bin_data_by_metallicity(self, data, feh_column='feh', n_feh_bins=5,
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

        return binned_data

    def fit_all_bins(self, binned_data, u_column='u', u_sigma_column='u_sigma',
                    absg1_column='absg1', absg2_column='absg2',
                    a_prior_range=(-1,20), b_prior_range=(-50,5), gamma=np.inf, mass_min=0.01, mass_max=2,
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
                fitter = NonParametricMLR(
                    n_bins=self.n_absg_bins,
                    absg_min=self.absg_min,
                    absg_max=self.absg_max,
                    uncertainty_model=self.uncertainty_model,
                    f_outlier=self.f_outlier,
                    outlier_u0=self.outlier_u0,
                    outlier_sigma=self.outlier_sigma
                )

                # Set data
                fitter.set_data(
                    u_values=u_values,
                    u_sigma_values=u_sigma_values,
                    absg1_values=absg1_values,
                    absg2_values=absg2_values,
                    gamma=gamma
                )

                # Run fitting
                fitter.run_numpyro(
                    num_warmup=num_warmup,
                    num_samples=num_samples,
                    num_chains=num_chains,
                    mass_min=mass_min,
                    mass_max=mass_max,
                    seed=seed
                )

            elif self.model_type == 'broken_powerlaw':
                # Create broken power-law fitter for this bin
                fitter = BrokenPowerLawMLR(
                    n_segments=self.n_segments,
                    absg_min=self.absg_min,
                    absg_max=self.absg_max,
                    uncertainty_model=self.uncertainty_model,
                    f_outlier=self.f_outlier,
                    outlier_u0=self.outlier_u0,
                    outlier_sigma=self.outlier_sigma
                )

                # Set data
                fitter.set_data(
                    u_values=u_values,
                    u_sigma_values=u_sigma_values,
                    absg1_values=absg1_values,
                    absg2_values=absg2_values
                )

                # Run fitting
                fitter.run_numpyro(
                    num_warmup=num_warmup,
                    num_samples=num_samples,
                    num_chains=num_chains,
                    seed=seed,
                    a_prior_range=a_prior_range, b_prior_range=b_prior_range,
                )

            # Store fitter
            self.fitters[bin_idx] = fitter

    def plot_all_results(self, binned_data, output_dir='.', data=None):
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
            suffix = f'_fehbin{bin_idx}'
            data = binned_data[bin_idx] if binned_data is not None else None

            if len(data) == 0:
                print(f"Skipping metallicity bin {bin_idx} (no data)")
                continue

            if self.model_type == 'nonparametric':
                # Non-parametric model plotting
                fitter.plot_results(output_dir=output_dir, output_suffix=suffix)
                fitter.plot_fitting_results(data=data, output_dir=output_dir, output_suffix=suffix)

            elif self.model_type == 'broken_powerlaw':
                # Broken power law model plotting
                fitter.plot_results(output_dir=output_dir, output_suffix=suffix)
                fitter.plot_fitting_results(data=data, output_dir=output_dir, output_suffix=suffix)

    def plot_comparison(self, output_path='mass_absg_comparison.png',
                       data=None):
        """
        Plot comparison of mass-luminosity relations across all metallicity bins.

        Parameters
        ----------
        output_path : str
            Output file path
        data : binned_data : dict
        true_mass_funcs : dict or callable
            True mass functions for each bin (for validation)
        """
        if len(self.fitters) == 0:
            print("No fitted results to plot")
            return

        fig, ax = plt.subplots(figsize=(12, 8))

        # Color map for different metallicity bins
        colors = cm.viridis(np.linspace(0, 1, len(self.fitters)))
        
        for (bin_idx, fitter), color in zip(self.fitters.items(), colors):

            if self.model_type == 'nonparametric':
                # Non-parametric model: plot step-like mass vs magnitude
                # Create absg bin edges for step plotting
                percentiles = np.percentile(fitter.samples, [16, 50, 84], axis=0)
                lower, median, upper = percentiles[0], percentiles[1], percentiles[2]

                # Plot step-like function
                label = f'[Fe/H]=[{self.feh_bin_edges[bin_idx]:.2f}, {self.feh_bin_edges[bin_idx+1]:.2f}]'
                ax.fill_between(fitter.absg_bins, lower, upper, color=color, alpha=0.2, step='mid')
                ax.step(fitter.absg_bins, median, color=color, label=label, linewidth=2, where='mid')

            elif self.model_type == 'broken_powerlaw':
                # Broken power law model: plot magnitude vs mass
                absg_range = np.linspace(fitter.absg_min, fitter.absg_max, 1000)  # absg range?
                
                all_masses = fitter.mass_from_absg(absg_range, fitter.samples)
                lower_mags, median_mags, upper_mags = np.percentile(all_masses, [16, 50, 84], axis=0)

                # Best-fit line
                median_params = np.median(fitter.samples, axis=0)
                best_fit_mags = fitter.mass_from_absg(absg_range, median_params)

                # Plot
                label = f'[Fe/H]=[{self.feh_bin_edges[bin_idx]:.2f}, {self.feh_bin_edges[bin_idx+1]:.2f}]'
                ax.fill_between(absg_range, lower_mags, upper_mags, color=color, alpha=0.2)
                ax.plot(absg_range, best_fit_mags, color=color, linestyle='-.', linewidth=3, label=label)

                # Add vertical lines for break points
                for bp in fitter.break_points:
                    ax.axhline(y=bp, color='gray', linestyle='--', alpha=0.5)

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


        ax.set_yscale('log')
        ax.set_xlim(fitter.absg_min, fitter.absg_max)
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

    def save_all_samples(self, output_dir='.', prefix='samples'):
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

        for bin_idx, fitter in self.fitters.items():
            output_path = os.path.join(
                output_dir,
                f'{prefix}_fehbin{bin_idx}_feh{self.feh_bin_centers[bin_idx]:.2f}.txt'
            )
            np.savetxt(output_path, fitter.samples)
            print(f"Saved samples for bin {bin_idx} to {output_path}")

        return