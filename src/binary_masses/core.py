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


class NonParametricPosteriorPlotter:
    """
    Non-parametric Bayesian inference for mass-luminosity relations.

    Uses NumPyro with NUTS sampler for posterior inference of mass-luminosity
    relations in bins of absolute magnitude. Supports Rice or Gaussian uncertainty
    models for the u parameter.
    """

    def __init__(self, n_bins=10, absg_min=4.0, absg_max=12.0, uncertainty_model='rice'):
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
        self.f_outlier = 0
        self.p_epsilon = 1e-10
        self.m_epsilon = 1e-10

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
                print("Using RICE distribution (norm_factor = 1.0, no truncation correction)")
            elif self.uncertainty_model == 'gaussian':
                # Gaussian needs truncation correction for u >= 0
                self.norm_factor = 1 - norm.cdf(0, loc=self.u_values, scale=self.u_sigma_values)
                print(f"Using GAUSSIAN distribution (norm_factor computed for truncation at u=0)")
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
        print(f"*** USING {self.uncertainty_model.upper()} DISTRIBUTION for u_obs uncertainty ***")

        # Performance info
        if self.u_sigma_values is not None:
            n_integration_pts = int(int_umax / int_du)
            print(f"Integration grid: {n_integration_pts} points (int_du={int_du}, int_umax={int_umax})")
            print(f"  → To speed up: increase int_du (e.g., 0.05-0.1) or reduce int_umax (e.g., 50-60)")
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
            sigma = jnp.maximum(sigma, 1e-10)
            coeff = 1.0 / (sigma * jnp.sqrt(2 * jnp.pi))
            exponent = -0.5 * ((x - mu) / sigma)**2
            return coeff * jnp.exp(exponent)

        def rice_distribution_jax(u_obs, u_true, sigma):
            """
            JAX version of Rice distribution with numerical stability:
            p(u_obs | u_true, sigma) = (u_obs/sigma^2) * exp[-(u_obs^2 + u_true^2)/(2*sigma^2)] * I_0(u_obs*u_true/sigma^2)

            Uses log-space calculation for numerical stability when Bessel argument is large.
            """
            sigma = jnp.maximum(sigma, 1e-10)
            sigma_sq = sigma ** 2

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

        def likelihood_single_u_jax(u, sqrt_mtot, u_sigma=None, norm_factor=None):
            """
            JAX version of likelihood for a single u value with chosen uncertainty model.

            Implements the integral:
            (1/sqrt(m_tot)) * ∫ p(ũ) * P(u_obs | ũ, σ) dũ

            where ũ = u_true/sqrt(m_tot) is the integration variable.
            """
            # Ensure positive values
            u = jnp.maximum(u, 1e-10)
            sqrt_mtot = jnp.maximum(sqrt_mtot, 1e-10)

            tilde_u = u / sqrt_mtot

            if u_sigma is None:
                # No uncertainty: direct evaluation
                good_component = (1.0 / sqrt_mtot) * func_pu_8_jax(tilde_u)
            else:
                # Ensure positive sigma
                u_sigma = jnp.maximum(u_sigma, 1e-10)

                # With uncertainty: integrate over the distribution
                if self.uncertainty_model == 'rice':
                    # Rice distribution: p(u_obs | u_true, sigma)
                    uncertainty_dist = rice_distribution_jax(
                        (u / sqrt_mtot),  # u_obs (fixed observed value)
                        int_ulist_jax,    # ũ (integration variable, true values)
                        (u_sigma / sqrt_mtot)  # σ
                    )
                elif self.uncertainty_model == 'gaussian':
                    # Gaussian distribution: p(u_obs | u_true, sigma) = N(u_obs; u_true, sigma)
                    uncertainty_dist = gaussian_jax(
                        (u / sqrt_mtot),  # u_obs (fixed observed value)
                        int_ulist_jax,    # μ = ũ (integration variable, mean = true values)
                        (u_sigma / sqrt_mtot)  # σ
                    )

                # Integrand: (1/sqrt(m_tot)) * p(ũ) * P(u_obs | ũ, σ)
                integrand = ((1.0 / sqrt_mtot) * func_pu_8_jax(int_ulist_jax) *
                           uncertainty_dist) * int_du

                # Ensure norm_factor is positive
                norm_factor = jnp.maximum(norm_factor, 1e-10)
                good_component = jnp.sum(integrand) / norm_factor + self.p_epsilon / (int_umax / int_du)

            total_prob = good_component + self.p_epsilon

            # Clamp to reasonable range to avoid log(0)
            total_prob = jnp.clip(total_prob, 1e-100, 1e10)
            return total_prob

        # Define the NumPyro model with regularization
        def model(u_values, u_sigma_values, absg1_values, absg2_values,
                norm_factor, absg_bins, gamma=None):
            ndim = self.n_bins

            # Prior: uniform for each mass bin
            mass_bins = numpyro.sample('mass_bins',
                                      dist.Uniform(mass_min, mass_max).expand([ndim]))

            # Smoothness regularization
            if ndim > 2:
                second_diffs = (mass_bins[2:] - 2*mass_bins[1:-1] + mass_bins[:-2])/mass_bins[1:-1]
                gamma = self.gamma if gamma is None else gamma
                epsilon_reg = 10**(-gamma)
                smoothness_penalty = -epsilon_reg * jnp.sum(second_diffs**2)
                numpyro.factor('smoothness', smoothness_penalty)

            # Vectorized likelihood calculation
            m1 = jnp.interp(absg1_values, absg_bins, mass_bins)
            m2 = jnp.interp(absg2_values, absg_bins, mass_bins)
            mtot = m1 + m2
            sqrt_mtot = jnp.sqrt(mtot)

            # Calculate log likelihood for each data point
            if u_sigma_values is None:
                tilde_u = u_values / sqrt_mtot
                pu = func_pu_8_jax(tilde_u)
                good_component = (1.0 / sqrt_mtot) * pu
                total_prob = good_component + self.p_epsilon
                log_likelihood_vec = jnp.log(jnp.maximum(total_prob, 1e-100))
            else:
                def compute_single_likelihood(i):
                    return jnp.log(likelihood_single_u_jax(
                        u_values[i], sqrt_mtot[i],
                        u_sigma_values[i], norm_factor[i]
                    ))

                log_likelihood_vec = jax.vmap(compute_single_likelihood)(
                    jnp.arange(len(u_values))
                )

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
            plt.savefig(f'{output_dir}/corner_plot_{output_suffix}.png', dpi=300)
            plt.close(fig)
        except ImportError:
            print("corner package not installed. Install it with: pip install corner")

    def plot_fitting_results(self, truths=None, true_mass_func=None, output_dir='', output_suffix=''):
        """
        Plot the fitting results with credible region.

        Parameters
        ----------
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

        absg_range = np.linspace(self.absg_min, self.absg_max, 1000)

        fitted_mass_samples = np.array([
            np.interp(absg_range, self.absg_bins, mass_bins)
            for mass_bins in resampled_data
        ])

        percentiles = np.percentile(fitted_mass_samples, [16, 50, 84], axis=0)
        lower, median, upper = percentiles[0], percentiles[1], percentiles[2]

        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot sample fits
        for i in range(min(400, len(resampled_data))):
            ax.plot(absg_range,
                   np.interp(absg_range, self.absg_bins, resampled_data[i]),
                   color='gray', alpha=0.1, linewidth=1, zorder=1)

        # Plot credible region
        ax.fill_between(absg_range, lower, upper, color='orange',
                       alpha=0.3, label='1 sigma', zorder=2)

        # Plot median fit
        ax.plot(absg_range, median, color='orange',
               label=f'Median Fit ({self.uncertainty_model.capitalize()})', ls='-', linewidth=2, zorder=3)

        # Plot bin centers
        median_masses = np.median(resampled_data, axis=0)
        ax.scatter(self.absg_bins, median_masses, color='red', s=50,
                  zorder=4, label='Bin centers')

        # Plot true relation if provided
        if true_mass_func is not None:
            true_mass = true_mass_func(absg_range)
            ax.plot(absg_range, true_mass, color='k', linestyle='--',
                   label='True relation', linewidth=2, zorder=5)
        elif truths is not None:
            ax.scatter(self.absg_bins, truths, color='green',
                      s=100, marker='x', linewidth=3,
                      label='True bin values', zorder=5)

        ax.set_xlim(self.absg_min, self.absg_max)
        ax.set_xlabel('$M_{\\mathrm{G}}$ [mag]', fontsize=12)
        ax.set_ylabel('Mass [$M_{\\odot}$]', fontsize=12)
        ax.legend(fontsize=10)
        title = f'Non-parametric fit with {self.uncertainty_model.capitalize()} distribution ({self.n_bins} bins)'
        plt.title(title, fontsize=14)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/nonparametric_fit_{self.n_bins}bins_{self.uncertainty_model}{output_suffix}.png', dpi=300)
        plt.close(fig)


class MultiMetallicityFitter:
    """
    Wrapper class to fit multiple metallicity bins.

    This class provides a convenient interface for fitting mass-luminosity
    relations in different metallicity bins.
    """

    def __init__(self, n_absg_bins=10, absg_min=4.0, absg_max=12.0, uncertainty_model='rice'):
        """
        Initialize the multi-metallicity fitter.

        Parameters
        ----------
        n_absg_bins : int
            Number of bins to split the absg range into
        absg_min : float
            Minimum absg value
        absg_max : float
            Maximum absg value
        uncertainty_model : str
            Uncertainty model to use: 'rice' or 'gaussian' (default: 'rice')
        """
        self.n_absg_bins = n_absg_bins
        self.absg_min = absg_min
        self.absg_max = absg_max
        self.uncertainty_model = uncertainty_model
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
        feh_values = data[feh_column]

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

            binned_data[i] = data[mask]
            print(f"Metallicity bin {i} ([{self.feh_bin_edges[i]:.2f}, {self.feh_bin_edges[i+1]:.2f}]): {len(binned_data[i])} stars")

        return binned_data

    def fit_all_bins(self, binned_data, u_column='u', u_sigma_column='u_sigma',
                    absg1_column='absg1', absg2_column='absg2',
                    gamma=np.inf, mass_min=0.01, mass_max=2,
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
            Regularization parameter
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
            print(f"Fitting metallicity bin {bin_idx}")
            print(f"[Fe/H] range: [{self.feh_bin_edges[bin_idx]:.2f}, {self.feh_bin_edges[bin_idx+1]:.2f}]")
            print(f"Number of stars: {len(data)}")
            print(f"{'='*60}\n")

            # Create fitter for this bin
            fitter = NonParametricPosteriorPlotter(
                n_bins=self.n_absg_bins,
                absg_min=self.absg_min,
                absg_max=self.absg_max,
                uncertainty_model=self.uncertainty_model
            )

            # Set data
            u_values = data[u_column].data
            u_sigma_values = data[u_sigma_column].data if u_sigma_column in data.colnames else None
            absg1_values = data[absg1_column].data
            absg2_values = data[absg2_column].data

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

            # Store fitter
            self.fitters[bin_idx] = fitter

    def plot_all_results(self, output_dir='.'):
        """
        Plot results for all metallicity bins.

        Parameters
        ----------
        output_dir : str
            Output directory for plots
        """
        os.makedirs(output_dir, exist_ok=True)

        for bin_idx, fitter in self.fitters.items():
            suffix = f'_fehbin{bin_idx}'
            fitter.plot_results(output_dir=output_dir, output_suffix=suffix)
            fitter.plot_fitting_results(output_dir=output_dir, output_suffix=suffix)

    def plot_comparison(self, output_path='mass_absg_comparison.png',
                       true_mass_funcs=None):
        """
        Plot comparison of mass-absg relations across all metallicity bins.

        Parameters
        ----------
        output_path : str
            Output file path
        true_mass_funcs : dict or callable
            True mass functions for each bin (for validation)
        """
        if len(self.fitters) == 0:
            print("No fitted results to plot")
            return

        fig, ax = plt.subplots(figsize=(12, 8))

        # Color map for different metallicity bins
        colors = cm.viridis(np.linspace(0, 1, len(self.fitters)))

        absg_range = np.linspace(self.absg_min, self.absg_max, 1000)

        for (bin_idx, fitter), color in zip(self.fitters.items(), colors):
            # Get median and credible intervals
            num_samples = min(1000, len(fitter.samples))
            indices = np.random.choice(len(fitter.samples), size=num_samples, replace=False)
            resampled_data = fitter.samples[indices]

            fitted_mass_samples = np.array([
                np.interp(absg_range, fitter.absg_bins, mass_bins)
                for mass_bins in resampled_data
            ])

            percentiles = np.percentile(fitted_mass_samples, [16, 50, 84], axis=0)
            lower, median, upper = percentiles[0], percentiles[1], percentiles[2]

            # Plot
            label = f'[Fe/H]=[{self.feh_bin_edges[bin_idx]:.2f}, {self.feh_bin_edges[bin_idx+1]:.2f}]'
            ax.fill_between(absg_range, lower, upper, color=color, alpha=0.2)
            ax.plot(absg_range, median, color=color, label=label, linewidth=2)

            # Plot true relation if provided
            if true_mass_funcs is not None:
                if callable(true_mass_funcs):
                    true_func = true_mass_funcs
                else:
                    true_func = true_mass_funcs.get(bin_idx)

                if true_func is not None:
                    true_mass = true_func(absg_range)
                    ax.plot(absg_range, true_mass, color=color,
                           linestyle='--', linewidth=1.5, alpha=0.7)

        ax.set_xlim(self.absg_min, self.absg_max)
        ax.set_xlabel('$M_{\\mathrm{G}}$ [mag]', fontsize=14)
        ax.set_ylabel('Mass [$M_{\\odot}$]', fontsize=14)
        ax.legend(fontsize=10, loc='best')
        ax.set_title(f'Mass-Absg Relations for Different Metallicities ({self.uncertainty_model.capitalize()} Distribution)', fontsize=16)
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