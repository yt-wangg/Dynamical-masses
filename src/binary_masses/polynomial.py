"""Polynomial mass-luminosity relation (MLR) model."""

import os
from typing import Optional, Union

import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

from .jax_utils import _configure_jax_gpu_fallback

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
                 f_outlier=0, outlier_u0=30, outlier_sigma=15, outlier_kappa=None, outlier_kappa_scale=0.05,
                 fit_outlier_params=False, deriv_penalty_strength=10.0,
                 coeff_prior_scale=5.0,
                 param_truths: Optional[list] = None):
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
        self.outlier_kappa_scale = outlier_kappa_scale

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
        self.param_truths = param_truths

    def func_pu_8(self, tilde_u, A=5.434e-3, B=2.544e-3, C=3.100, u0=35.67):
        """
        The probability distribution for the normalized parameter ũ = u/√mtot
        phase selection considered
        """
        return A * tilde_u * np.exp(-1 * (
            B * tilde_u**2 + np.exp((tilde_u - u0) / C)
        ))

    def set_data(self, u_values, u_sigma_values=None, absg1_values=None, absg2_values=None, outlier_kappa=None, outlier_kappa_scale=None):
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

        scale = self.outlier_kappa_scale if outlier_kappa_scale is None else outlier_kappa_scale
        if outlier_kappa is not None:
            self.outlier_kappa = outlier_kappa
        elif self.outlier_kappa_init is not None:
            self.outlier_kappa = self.outlier_kappa_init
        else:
            self.outlier_kappa = scale * len(self.u_values)

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
        def func_pu_8_jax(tilde_u, A=5.434e-3, B=2.544e-3, C=3.100, u0=35.67):
            # phase selection considered
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

            from jax.scipy.special import erf as jax_erf
            outlier_gaussian_norm = 1. - 0.5 * (1 + jax_erf(-outlier_u0 / outlier_sigma / jnp.sqrt(2.)))
            outlier_component = outlier_gaussian_jax(tilde_u, outlier_u0, outlier_sigma) / jnp.maximum(outlier_gaussian_norm, 1e-10)

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
                pi = self.f_outlier_init
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
                outlier_sigma = numpyro.sample("outlier_sigma", dist.Normal(15.0, 2.0))
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
        if getattr(self, "param_names", None) and len(self.param_names) == self.samples.shape[1]:
            labels = list(self.param_names)
        else:
            labels = [f'$c_{i}$' for i in range(self.poly_model.n_params)]
            # Add outlier parameter labels if they were fitted
            if self.fit_outlier_params:
                labels.extend(['f_outlier', 'outlier_u0', 'outlier_sigma'])

        # Drop parameters with zero dynamic range to avoid corner errors.
        samples = self.samples
        keep_mask = np.ones(samples.shape[1], dtype=bool)
        param_truths = self.param_truths if self.param_truths is not None else None
        for idx in range(samples.shape[1]):
            col = samples[:, idx]
            if not np.isfinite(col).any():
                keep_mask[idx] = False
                continue
            if np.nanmax(col) == np.nanmin(col):
                keep_mask[idx] = False

        if not np.all(keep_mask):
            dropped = [labels[i] for i in range(len(labels)) if not keep_mask[i]]
            if dropped:
                print(f"Dropping constant parameters from corner plot: {', '.join(dropped)}")
            samples = samples[:, keep_mask]
            labels = [labels[i] for i in range(len(labels)) if keep_mask[i]]
            if param_truths is not None:
                param_truths = [param_truths[i] for i in range(len(param_truths)) if keep_mask[i]]

        try:
            import corner
            fig = corner.corner(samples, labels=labels, show_titles=True, truths=param_truths, truth_color='salmon')
            plt.tight_layout()
            outlier_tag = 'outlierfit' if self.fit_outlier_params else 'outlierfixed'
            metadata_suffix = output_suffix if output_suffix else f'_model-polynomial_unc-{self.uncertainty_model}_outlier-{outlier_tag}'
            filename = f'corner{metadata_suffix}.png'
            plt.savefig(f'{output_dir}/{filename}', dpi=300)
            plt.close(fig)
        except ImportError:
            print("corner package not installed. Install it with: pip install corner")

    def plot_fitting_results(
        self,
        data=None,
        output_dir="",
        output_suffix="",
        isochrone_data_feh=None,
        iso_colname_dict={"absg": "absg", "mass": "mass", "feh": "MH"},
        isochrone_curve_data_feh=None,
    ):
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

        col = iso_colname_dict

        if isochrone_curve_data_feh is not None:
            try:
                iso_absg_curve = np.array(isochrone_curve_data_feh[col["absg"]], dtype=float)
                iso_mass_curve = np.array(isochrone_curve_data_feh[col["mass"]], dtype=float)
                ok = np.isfinite(iso_absg_curve) & np.isfinite(iso_mass_curve)
                iso_absg_curve = iso_absg_curve[ok]
                iso_mass_curve = iso_mass_curve[ok]
                if iso_absg_curve.size > 1:
                    order = np.argsort(iso_absg_curve)
                    ax.plot(
                        iso_absg_curve[order],
                        iso_mass_curve[order],
                        color="black",
                        linewidth=1.5,
                        alpha=0.8,
                        linestyle="--",
                        label="Isochrone",
                        zorder=0,
                    )
            except Exception as e:
                print(f"Could not plot binned isochrone curve: {e}")

        # if isochrone_data_feh is not None:
        #     col = iso_colname_dict
        #     # Plot isochrone data if provided
        #     iso_scatter_label = "Isochrone (raw)" if isochrone_curve_data_feh is not None else "Isochrone"
        #     ax.scatter(isochrone_data_feh[col['absg']], isochrone_data_feh[col['mass']], 
        #                 color='black', s=5, alpha=0.5, label=iso_scatter_label, zorder=0)


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


class IsochroneMassModel:
    """
    Lightweight wrapper providing m_iso(M_G). Accepts either
    (1) a callable f(M_G)->mass, or
    (2) sample arrays (absg_grid, mass_grid) for interpolation.

    Provides NumPy and JAX versions.
    """

    def __init__(
        self,
        absg_min: float = -1.0,
        absg_max: float = 15.0,
        func=None,
        absg_grid=None,
        mass_grid=None,
        mass_min: float = 0.01,
    ):
        self.absg_min = float(absg_min)
        self.absg_max = float(absg_max)
        self.mass_min = float(mass_min)

        if func is not None and (absg_grid is not None or mass_grid is not None):
            raise ValueError("Provide either func or (absg_grid, mass_grid), not both.")

        self._func = func
        if func is None:
            if absg_grid is None or mass_grid is None:
                raise ValueError("Need both absg_grid and mass_grid when func is None.")
            self.absg_grid_np = np.asarray(absg_grid)
            self.mass_grid_np = np.asarray(mass_grid)
            order = np.argsort(self.absg_grid_np)
            self.absg_grid_np = self.absg_grid_np[order]
            self.mass_grid_np = self.mass_grid_np[order]
        else:
            self.absg_grid_np = None
            self.mass_grid_np = None

    def mass_from_absg(self, absg):
        absg = np.asarray(absg)
        if self._func is not None:
            mass = self._func(absg)
        else:
            mass = np.interp(
                absg,
                self.absg_grid_np,
                self.mass_grid_np,
                left=self.mass_grid_np[0],
                right=self.mass_grid_np[-1],
            )
        mass = np.where(mass > 0.0, mass, self.mass_min)
        return mass

    def mass_from_absg_jax(self, absg):
        import jax.numpy as jnp

        absg = jnp.asarray(absg)
        if self._func is not None:
            mass = self._func(absg)
        else:
            mass = jnp.interp(absg, jnp.asarray(self.absg_grid_np), jnp.asarray(self.mass_grid_np))
        mass = jnp.where(mass > 0.0, mass, self.mass_min)
        return mass

    @classmethod
    def from_interpolated_mass_data(
        cls,
        mh: float,
        data_dir: Optional[str] = None,
        *,
        mh_interpolation: str = "linear",
        absg_min: Optional[float] = None,
        absg_max: Optional[float] = None,
        mass_min: float = 0.01,
    ) -> "IsochroneMassModel":
        """Build an `IsochroneMassModel` from `data/interpolated_mass_data`.

        Parameters
        ----------
        mh : float
            Target metallicity [M/H] for the curve.
        data_dir : str, optional
            Directory containing `gmag_grid.npy` and `mass_interp_MH_*.npy`.
            If None, tries the repo-local default under `bayesian-binary-masses/data/`.
        mh_interpolation : {"linear","nearest"}
            How to handle MH values not exactly present in the grid.
        absg_min, absg_max : float, optional
            Magnitude bounds advertised by the model; defaults to the curve range.
        mass_min : float
            Safety floor applied by the model.
        """
        from .isochrone_grid import load_interpolated_mass_curve

        absg_grid, mass_grid = load_interpolated_mass_curve(
            mh=mh, data_dir=data_dir, mh_interpolation=mh_interpolation
        )
        if absg_min is None:
            absg_min = float(np.nanmin(absg_grid))
        if absg_max is None:
            absg_max = float(np.nanmax(absg_grid))

        return cls(
            absg_min=absg_min,
            absg_max=absg_max,
            absg_grid=absg_grid,
            mass_grid=mass_grid,
            mass_min=mass_min,
        )


class DifferencePolyMassAbsgModel(PolyMassAbsgModel):
    """
    Polynomial model for the **difference** relative to an isochrone:

        log10 M_dyn(M_G) = log10 M_iso(M_G) + sum_i c_i * x^i,

    where x is the rescaled M_G in [-1, 1] as in PolyMassAbsgModel.
    """

    def __init__(self, *args, isochrone_model: IsochroneMassModel, **kwargs):
        if isochrone_model is None:
            raise ValueError("DifferencePolyMassAbsgModel requires an `isochrone_model` (IsochroneMassModel).")
        super().__init__(*args, **kwargs)
        self.iso_model = isochrone_model

    def _log10_delta_np(self, absg, params):
        coeffs = np.asarray(self._split_params(params))
        x = self._absg_to_x_np(np.asarray(absg))
        powers = np.stack([x ** i for i in range(self.order + 1)], axis=-1)
        return np.sum(coeffs * powers, axis=-1)

    def _log10_delta_jax(self, absg, params):
        import jax.numpy as jnp

        coeffs = self._split_params(params)
        x = self._absg_to_x_jax(absg)
        powers = jnp.stack([jnp.power(x, i) for i in range(self.order + 1)], axis=-1)
        return jnp.sum(coeffs * powers, axis=-1)

    def mass_from_absg(self, absg, params):
        absg = np.asarray(absg)
        log10_delta = self._log10_delta_np(absg, params)
        m_iso = self.iso_model.mass_from_absg(absg)
        log10_m = np.log10(np.maximum(m_iso, self.mass_min)) + log10_delta
        mass = np.power(10.0, log10_m)
        mass = np.where(mass > 0.0, mass, self.mass_min)
        return mass

    def mass_from_absg_jax(self, absg, params):
        import jax.numpy as jnp

        log10_delta = self._log10_delta_jax(absg, params)
        m_iso = self.iso_model.mass_from_absg_jax(absg)
        log10_m = jnp.log10(jnp.maximum(m_iso, self.mass_min)) + log10_delta
        mass = jnp.power(10.0, log10_m)
        mass = jnp.where(mass > 0.0, mass, self.mass_min)
        return mass


class DifferencePolyMLR(PolynomialMLR):
    """
    Same interface as `PolynomialMLR`, but fits the *difference* between
    dynamical MLR and an input isochrone MLR:

        log10 m_dyn(M_G) = log10 m_iso(M_G) + P(M_G),

    with P a user-chosen low-order polynomial in rescaled M_G.
    """

    def __init__(self, order=3, isochrone_model=None, **kwargs):
        if isochrone_model is None:
            raise ValueError("DifferencePolyMLR requires `isochrone_model` (IsochroneMassModel).")
        super().__init__(order=order, **kwargs)
        self.poly_model = DifferencePolyMassAbsgModel(
            order=order,
            absg_min=self.absg_min,
            absg_max=self.absg_max,
            mass_min=self.mass_min,
            pivot=self.poly_model.pivot,
            deriv_penalty_strength=0.0,
            isochrone_model=isochrone_model,
        )
