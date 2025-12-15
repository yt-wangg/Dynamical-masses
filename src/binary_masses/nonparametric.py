"""Non-parametric mass-luminosity relation (MLR) model."""

import os
import numpy as np
from scipy import special
from scipy.stats import norm
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from .jax_utils import _configure_jax_gpu_fallback

class NonParametricMLR:
    """
    Non-parametric Bayesian inference for mass-luminosity relations.

    Uses NumPyro with NUTS sampler for posterior inference of mass-luminosity
    relations in bins of absolute magnitude. Supports Rice or Gaussian uncertainty
    models for the u parameter.
    """

    def __init__(self, n_bins=10, absg_min=4.0, absg_max=12.0, mass_min=0.01, mass_max=1.5, uncertainty_model='rice',
                 f_outlier=0, outlier_u0=30, outlier_sigma=15, outlier_kappa=None, outlier_kappa_scale=0.05,
                 fit_outlier_params=False):
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
        self.outlier_kappa_scale = outlier_kappa_scale

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
        self.mass_prior_means_log10 = None
        self.mass_prior_sigma_log10 = 0.3

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

    def _extract_true_mass_data(self, data):
        """
        Extract true masses and absolute magnitudes from a table or dict.
        """
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
            raise ValueError("true_mass_data must be astropy.Table or dict with 'm1', 'm2', 'absg1', 'absg2' columns")
        return m1, m2, absg1, absg2

    def _compute_log_mass_bin_means(self, data):
        """
        Compute log10 mean masses in each absg bin from true masses if provided.
        """
        try:
            m1, m2, absg1, absg2 = self._extract_true_mass_data(data)
        except Exception as exc:
            print(f"Could not compute mass priors from true_mass_data: {exc}")
            return None

        masses = np.concatenate([m1, m2])
        absg = np.concatenate([absg1, absg2])

        valid = np.isfinite(masses) & np.isfinite(absg) & (masses > 0)
        masses = masses[valid]
        absg = absg[valid]

        if masses.size == 0:
            return None

        # Use bins centered at self.absg_bins with width set by spacing
        if len(self.absg_bins) > 1:
            absg_width = np.abs(self.absg_bins[1] - self.absg_bins[0])
        else:
            absg_width = (self.absg_max - self.absg_min) / max(self.n_bins, 1)
        half_width = 0.5 * absg_width

        log_means = np.full(self.n_bins, np.nan)
        for i in range(self.n_bins):
            center = self.absg_bins[i]
            lower = center - half_width
            upper = center + half_width
            bin_mask = (absg >= lower) & (absg <= upper)
            if np.any(bin_mask):
                log_means[i] = np.mean(np.log10(masses[bin_mask]))

        if np.all(~np.isfinite(log_means)):
            return None

        return log_means

    def set_data(self, u_values=None, u_sigma_values=None,
                absg1_values=None, absg2_values=None, gamma=None, outlier_kappa=None, outlier_kappa_scale=None,
                true_mass_data=None):
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
            Concentration parameter for the Beta prior on f_outlier. If None, computed as c * N.
        outlier_kappa_scale : float, optional
            Scaling factor c for kappa = c * N when outlier_kappa is not provided (default from initializer).
        true_mass_data : astropy.Table or dict, optional
            If provided, use the true masses per absg bin to build Gaussian log-mass priors
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

        scale = self.outlier_kappa_scale if outlier_kappa_scale is None else outlier_kappa_scale
        if outlier_kappa is not None:
            self.outlier_kappa = outlier_kappa
        elif self.outlier_kappa_init is not None:
            self.outlier_kappa = self.outlier_kappa_init
        else:
            self.outlier_kappa = scale * len(self.u_values)

        # Optional informative priors from true masses
        if true_mass_data is not None:
            self.mass_prior_means_log10 = self._compute_log_mass_bin_means(true_mass_data)
            if self.mass_prior_means_log10 is not None:
                print("Using Gaussian log10 mass priors per absg bin from provided true_mass_data (sigma=0.3 dex).")
            else:
                print("true_mass_data provided but no valid masses found; falling back to default priors.")
        else:
            self.mass_prior_means_log10 = None

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

        # Prepare informative priors from true masses if available
        if self.mass_prior_means_log10 is not None:
            prior_means_np = np.array(self.mass_prior_means_log10, dtype=float)
            prior_mask_np = np.isfinite(prior_means_np)
            n_prior_bins = int(np.sum(prior_mask_np))
            n_uniform_bins = self.n_bins - n_prior_bins
            mass_prior_means_log10_jax = jnp.array(prior_means_np)
            mass_prior_mask_jax = jnp.array(prior_mask_np)
        else:
            mass_prior_means_log10_jax = None
            mass_prior_mask_jax = None
            n_prior_bins = 0
            n_uniform_bins = self.n_bins

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

            ndim = self.n_bins
            mass_min = self.mass_min
            mass_max = self.mass_max

            # Prior for each mass bin
            if mass_prior_means_log10_jax is not None and mass_prior_mask_jax is not None:
                log_mass_min = jnp.log10(mass_min + 1e-12)
                log_mass_max = jnp.log10(mass_max + 1e-12)

                # Get indices for prior and uniform bins
                prior_indices = jnp.where(mass_prior_mask_jax, size=n_prior_bins)[0]
                uniform_indices = jnp.where(~mass_prior_mask_jax, size=n_uniform_bins)[0]

                # Initialize all mass bins
                mass_bins = jnp.full(ndim, (mass_min + mass_max) / 2.0)

                if n_prior_bins > 0:
                    prior_locs = mass_prior_means_log10_jax[prior_indices]
                    log_mass_prior = numpyro.sample(
                        'log_mass_bins_prior',
                        dist.TruncatedNormal(
                            low=log_mass_min,
                            high=log_mass_max,
                            loc=prior_locs,
                            scale=self.mass_prior_sigma_log10
                        )
                    )
                    # Set prior bins using index-based approach
                    mass_bins = mass_bins.at[prior_indices].set(jnp.power(10.0, log_mass_prior))

                if n_uniform_bins > 0:
                    mass_bins_uniform = numpyro.sample(
                        'mass_bins_uniform',
                        dist.Uniform(mass_min, mass_max).expand([n_uniform_bins])
                    )
                    # Set uniform bins using index-based approach
                    mass_bins = mass_bins.at[uniform_indices].set(mass_bins_uniform)

                mass_bins = numpyro.deterministic('mass_bins', mass_bins)
            else:
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


