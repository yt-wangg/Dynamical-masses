"""Broken power-law mass-luminosity relation (MLR) model."""

import os
import numpy as np
from scipy import special
from scipy.stats import norm
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from .jax_utils import _configure_jax_gpu_fallback

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
                 f_outlier=0, outlier_u0=30, outlier_sigma=15, outlier_kappa=None, outlier_kappa_scale=0.05,
                 fit_outlier_params=False):
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
        self.outlier_kappa_scale = outlier_kappa_scale
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

    def set_data(self, u_values, u_sigma_values, absg1_values, absg2_values, outlier_kappa=None, outlier_kappa_scale=None):
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
            Concentration parameter for the Beta prior on f_outlier. If None, computed as c * N.
        outlier_kappa_scale : float, optional
            Scaling factor c for kappa = c * N when outlier_kappa is not provided (default from initializer).
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
        scale = self.outlier_kappa_scale if outlier_kappa_scale is None else outlier_kappa_scale
        if outlier_kappa is not None:
            self.outlier_kappa = outlier_kappa
        elif self.outlier_kappa_init is not None:
            self.outlier_kappa = self.outlier_kappa_init
        else:
            self.outlier_kappa = scale * len(self.u_values)

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


