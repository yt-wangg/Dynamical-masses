"""Continuous-metallicity difference-polynomial MLR model.

This module implements a DifferencePoly-style model that:
- uses an isochrone baseline m_iso(M_G, [M/H]) from the interpolated grid, and
- fits a metallicity-dependent polynomial residual in log10 mass.

The core parameterization is:

    log10 m_dyn(M_G, MH) = log10 m_iso(M_G, MH) + Σ_i (a_i + b_i * z) x^i

where x is rescaled M_G in [-1, 1] and z is rescaled metallicity in [-1, 1].
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.stats import norm

from .jax_utils import _configure_jax_gpu_fallback
from .isochrone_grid import load_interpolated_mass_grid
from .polynomial import PolynomialMLR, PolyMassAbsgModel


class IsochroneMassSurfaceModel:
    """Bilinear-interpolated isochrone mass surface m_iso(M_G, MH)."""

    def __init__(
        self,
        absg_grid: np.ndarray,
        mh_grid: np.ndarray,
        mass_grid: np.ndarray,
        *,
        mass_min: float = 0.01,
    ):
        self.mass_min = float(mass_min)

        self.absg_grid_np = np.asarray(absg_grid, dtype=float).reshape(-1)
        self.mh_grid_np = np.asarray(mh_grid, dtype=float).reshape(-1)
        self.mass_grid_np = np.asarray(mass_grid, dtype=float)

        if self.mass_grid_np.shape != (self.mh_grid_np.size, self.absg_grid_np.size):
            raise ValueError(
                "mass_grid must have shape (n_mh, n_absg); got "
                f"{self.mass_grid_np.shape} vs ({self.mh_grid_np.size}, {self.absg_grid_np.size})"
            )

        # Ensure grids are sorted and mass_grid follows.
        absg_order = np.argsort(self.absg_grid_np)
        self.absg_grid_np = self.absg_grid_np[absg_order]
        self.mass_grid_np = self.mass_grid_np[:, absg_order]

        mh_order = np.argsort(self.mh_grid_np)
        self.mh_grid_np = self.mh_grid_np[mh_order]
        self.mass_grid_np = self.mass_grid_np[mh_order, :]

        self._absg_grid_jax = None
        self._mh_grid_jax = None
        self._mass_grid_jax = None

    @classmethod
    def from_interpolated_mass_data(
        cls,
        data_dir: Optional[str] = None,
        *,
        mass_min: float = 0.01,
    ) -> "IsochroneMassSurfaceModel":
        absg_grid, mh_grid, mass_grid = load_interpolated_mass_grid(data_dir=data_dir)
        return cls(absg_grid=absg_grid, mh_grid=mh_grid, mass_grid=mass_grid, mass_min=mass_min)

    def mass_from_absg_mh(self, absg, mh):
        """NumPy evaluation with bilinear interpolation and clipping."""
        absg = np.asarray(absg, dtype=float)
        mh = np.asarray(mh, dtype=float)
        absg, mh = np.broadcast_arrays(absg, mh)

        absg_c = np.clip(absg, self.absg_grid_np[0], self.absg_grid_np[-1])
        mh_c = np.clip(mh, self.mh_grid_np[0], self.mh_grid_np[-1])

        # Interpolate in absg for each MH curve -> (n_mh, *shape)
        mass_at_absg = np.stack(
            [np.interp(absg_c, self.absg_grid_np, self.mass_grid_np[i]) for i in range(self.mh_grid_np.size)],
            axis=0,
        )

        # Interpolate in MH elementwise.
        flat = mass_at_absg.reshape(self.mh_grid_np.size, -1)  # (K, P)
        out = np.empty(flat.shape[1], dtype=float)
        mh_flat = mh_c.reshape(-1)
        for j in range(flat.shape[1]):
            out[j] = np.interp(mh_flat[j], self.mh_grid_np, flat[:, j])
        out = out.reshape(absg_c.shape)

        out = np.where(out > 0.0, out, self.mass_min)
        return out

    def _ensure_jax_cache(self):
        if self._absg_grid_jax is not None:
            return
        import jax.numpy as jnp

        self._absg_grid_jax = jnp.asarray(self.absg_grid_np)
        self._mh_grid_jax = jnp.asarray(self.mh_grid_np)
        self._mass_grid_jax = jnp.asarray(self.mass_grid_np)

    def mass_from_absg_mh_jax(self, absg, mh):
        """JAX evaluation via bilinear interpolation on the (MH, M_G) grid."""
        import jax.numpy as jnp

        self._ensure_jax_cache()

        absg = jnp.asarray(absg)
        mh = jnp.asarray(mh)
        absg, mh = jnp.broadcast_arrays(absg, mh)

        absg_grid = self._absg_grid_jax
        mh_grid = self._mh_grid_jax
        mass_grid = self._mass_grid_jax

        absg_c = jnp.clip(absg, absg_grid[0], absg_grid[-1])
        mh_c = jnp.clip(mh, mh_grid[0], mh_grid[-1])

        ia1 = jnp.clip(jnp.searchsorted(absg_grid, absg_c, side="right"), 1, absg_grid.size - 1)
        ia0 = ia1 - 1
        im1 = jnp.clip(jnp.searchsorted(mh_grid, mh_c, side="right"), 1, mh_grid.size - 1)
        im0 = im1 - 1

        a0 = absg_grid[ia0]
        a1 = absg_grid[ia1]
        m0 = mh_grid[im0]
        m1 = mh_grid[im1]

        wa = (absg_c - a0) / jnp.maximum(a1 - a0, 1e-12)
        wm = (mh_c - m0) / jnp.maximum(m1 - m0, 1e-12)

        m00 = mass_grid[im0, ia0]
        m01 = mass_grid[im0, ia1]
        m10 = mass_grid[im1, ia0]
        m11 = mass_grid[im1, ia1]

        m0a = m00 * (1.0 - wa) + m01 * wa
        m1a = m10 * (1.0 - wa) + m11 * wa
        out = m0a * (1.0 - wm) + m1a * wm

        out = jnp.where(out > 0.0, out, self.mass_min)
        return out


class DifferencePolyFehMassAbsgModel(PolyMassAbsgModel):
    """Metallicity-dependent difference-polynomial mass model."""

    def __init__(
        self,
        *,
        order: int,
        absg_min: float,
        absg_max: float,
        mass_min: float,
        pivot: float,
        deriv_penalty_strength: float, # The penalty enforces d log10(M) / d M_G ≤ 0, \lambda_{M_G} in the paper.
        feh_monotone_strength: float,
        isochrone_surface_model: IsochroneMassSurfaceModel,
        feh_min: float,
        feh_max: float,
    ):
        if isochrone_surface_model is None:
            raise ValueError("DifferencePolyFehMassAbsgModel requires `isochrone_surface_model`.")
        if not (np.isfinite(feh_min) and np.isfinite(feh_max) and feh_max > feh_min):
            raise ValueError("Invalid (feh_min, feh_max) for metallicity scaling.")

        super().__init__(
            order=order,
            absg_min=absg_min,
            absg_max=absg_max,
            mass_min=mass_min,
            pivot=pivot,
            deriv_penalty_strength=deriv_penalty_strength,
        )
        self.feh_monotone_strength = float(feh_monotone_strength)

        self.iso_surface = isochrone_surface_model
        self.feh_min = float(feh_min)
        self.feh_max = float(feh_max)

        self.n_base_params = self.order + 1
        self.n_params = 2 * self.n_base_params  # (a_0..a_n, b_0..b_n)

    def _split_params(self, params):
        params = np.asarray(params)
        if params.shape[-1] != self.n_params:
            raise ValueError(f"Expected params with last dimension {self.n_params}, got {params.shape}")
        a = params[..., : self.n_base_params]
        b = params[..., self.n_base_params :]
        return a, b

    def _feh_to_z_np(self, feh):
        feh = np.asarray(feh, dtype=float)
        feh = np.clip(feh, self.feh_min, self.feh_max)
        return 2.0 * (feh - self.feh_min) / (self.feh_max - self.feh_min) - 1.0

    def _feh_to_z_jax(self, feh):
        import jax.numpy as jnp

        feh = jnp.asarray(feh)
        feh = jnp.clip(feh, self.feh_min, self.feh_max)
        return 2.0 * (feh - self.feh_min) / (self.feh_max - self.feh_min) - 1.0

    def _log10_delta_np(self, absg, feh, params):
        a, b = self._split_params(params)
        absg = np.asarray(absg, dtype=float)
        feh = np.asarray(feh, dtype=float)
        absg, feh = np.broadcast_arrays(absg, feh)

        z = self._feh_to_z_np(feh)[..., None]  # (..., 1)
        coeffs = a + b * z  # (..., n_base_params)

        x = self._absg_to_x_np(absg)
        powers = np.stack([x ** i for i in range(self.order + 1)], axis=-1)  # (..., n_base_params)
        return np.sum(coeffs * powers, axis=-1)

    def _log10_delta_jax(self, absg, feh, params):
        import jax.numpy as jnp

        a = params[: self.n_base_params]
        b = params[self.n_base_params :]

        absg = jnp.asarray(absg)
        feh = jnp.asarray(feh)
        absg, feh = jnp.broadcast_arrays(absg, feh)

        z = self._feh_to_z_jax(feh)[..., None]
        coeffs = a + b * z

        x = self._absg_to_x_jax(absg)
        powers = jnp.stack([jnp.power(x, i) for i in range(self.order + 1)], axis=-1)
        return jnp.sum(coeffs * powers, axis=-1)

    def mass_from_absg_feh(self, absg, feh, params):
        absg = np.asarray(absg, dtype=float)
        feh = np.asarray(feh, dtype=float)
        log10_delta = self._log10_delta_np(absg, feh, params)
        m_iso = self.iso_surface.mass_from_absg_mh(absg, feh)
        log10_m = np.log10(np.maximum(m_iso, self.mass_min)) + log10_delta
        mass = np.power(10.0, log10_m)
        mass = np.where(mass > 0.0, mass, self.mass_min)
        return mass

    def mass_from_absg_feh_jax(self, absg, feh, params):
        import jax.numpy as jnp

        log10_delta = self._log10_delta_jax(absg, feh, params)
        m_iso = self.iso_surface.mass_from_absg_mh_jax(absg, feh)
        log10_m = jnp.log10(jnp.maximum(m_iso, self.mass_min)) + log10_delta
        mass = jnp.power(10.0, log10_m)
        mass = jnp.where(mass > 0.0, mass, self.mass_min)
        return mass

    def derivative_penalty_jax(self, params, *, n_grid: int = 64, n_feh: int = 5):
        """Soft penalty enforcing d log10(M) / d M_G <= 0 for a small MH grid."""
        import jax.numpy as jnp

        if self.deriv_penalty_strength <= 0 and self.feh_monotone_strength <= 0:
            return jnp.array(0.0)

        absg_grid = jnp.linspace(self.absg_min, self.absg_max, n_grid)
        feh_grid = jnp.linspace(self.feh_min, self.feh_max, n_feh)

        absg_mesh = absg_grid[None, :]
        feh_mesh = feh_grid[:, None]
        mass = self.mass_from_absg_feh_jax(absg_mesh, feh_mesh, params)
        log10m = jnp.log10(jnp.maximum(mass, self.mass_min))

        dabsg = jnp.maximum(absg_grid[1] - absg_grid[0], 1e-12)
        dlog10m_dabsg = (log10m[:, 1:] - log10m[:, :-1]) / dabsg
        violations = jnp.maximum(dlog10m_dabsg, 0.0)

        mean_sq_violation = jnp.mean(violations ** 2)
        penalty = -self.deriv_penalty_strength * mean_sq_violation

        if self.feh_monotone_strength > 0:
            dfeh = jnp.maximum(feh_grid[1] - feh_grid[0], 1e-12)
            dlog10m_dfeh = (log10m[1:, :] - log10m[:-1, :]) / dfeh
            feh_violations = jnp.maximum(-dlog10m_dfeh, 0.0)
            mean_sq_feh_violation = jnp.mean(feh_violations ** 2)
            penalty = penalty - self.feh_monotone_strength * mean_sq_feh_violation

        return penalty


class DifferencePolyFehMLR(PolynomialMLR):
    """DifferencePolyMLR with continuous metallicity (one [M/H] per system)."""

    def __init__(
        self,
        *,
        order: int = 3,
        isochrone_surface_model: IsochroneMassSurfaceModel,
        feh_min: float,
        feh_max: float,
        feh_coeff_prior_scale: float = 1.0, # deriv_penalty_strength=10.0: strength of the penalty. Larger values enforce monotonicity more strongly (in log‑probability units).
        feh_monotone_strength: float = 0.0,
        monotone_n_feh: int = 5, # monotone_n_feh=5: number of metallicity grid points used when applying the penalty (i.e., how many [Fe/H] slices are checked). More points = more robust monotonicity across metallicity.
        **kwargs,
    ):
        super().__init__(order=order, **kwargs)

        self.feh_min = float(feh_min)
        self.feh_max = float(feh_max)
        self.feh_coeff_prior_scale = float(feh_coeff_prior_scale)
        self.feh_monotone_strength = float(feh_monotone_strength)
        self.monotone_n_feh = int(monotone_n_feh)

        if isochrone_surface_model is None:
            raise ValueError("DifferencePolyFehMLR requires `isochrone_surface_model`.")

        self.poly_model = DifferencePolyFehMassAbsgModel(
            order=order,
            absg_min=self.absg_min,
            absg_max=self.absg_max,
            mass_min=self.mass_min,
            pivot=self.poly_model.pivot,
            deriv_penalty_strength=self.poly_model.deriv_penalty_strength,
            feh_monotone_strength=self.feh_monotone_strength,
            isochrone_surface_model=isochrone_surface_model,
            feh_min=self.feh_min,
            feh_max=self.feh_max,
        )

        self.feh_values = None
        self.feh_plot = None  # used by mass_from_absg when feh is not provided

        self.param_names = (
            [f"a_{i}" for i in range(self.poly_model.n_base_params)]
            + [f"b_{i}" for i in range(self.poly_model.n_base_params)]
        )

    def set_data(
        self,
        u_values,
        u_sigma_values=None,
        absg1_values=None,
        absg2_values=None,
        feh_values=None,
        outlier_kappa=None,
        outlier_kappa_scale=None,
    ):
        if feh_values is None:
            raise ValueError("DifferencePolyFehMLR requires `feh_values` (one value per system).")

        self.u_values = np.array(u_values)
        self.absg1_values = np.array(absg1_values)
        self.absg2_values = np.array(absg2_values)
        self.feh_values = np.array(feh_values, dtype=float)

        if self.feh_values.shape[0] != self.u_values.shape[0]:
            raise ValueError("feh_values must have the same length as u_values.")

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

        if self.feh_plot is None and self.feh_values.size:
            self.feh_plot = float(np.nanmedian(self.feh_values))

    def mass_from_absg(self, absg, params, feh=None):
        """Mass evaluation helper for plotting at a chosen metallicity."""
        if feh is None:
            if self.feh_plot is None:
                raise ValueError("Provide `feh=...` or call set_data() first.")
            feh = self.feh_plot
        return self.mass_from_absg_feh(absg=absg, feh=feh, params=params)

    def mass_from_absg_feh(self, absg, feh, params):
        """Evaluate m_dyn(M_G, MH) for arbitrary MH (NumPy)."""
        absg = np.asarray(absg)
        feh = np.asarray(feh)
        params = np.asarray(params)
        single = params.ndim == 1
        if single:
            return self.poly_model.mass_from_absg_feh(absg, feh, params)
        return np.array([self.poly_model.mass_from_absg_feh(absg, feh, p) for p in params])

    def run_numpyro(
        self,
        *,
        num_warmup=1000,
        num_samples=2000,
        num_chains=4,
        int_umax=80.0,
        int_du=0.02,
        seed=None,
        use_dense_mass=False,
        coeff_prior_scale=None,
        feh_coeff_prior_scale=None,
        feh_b_mask=None,
        feh_b0_positive=False,
        **kwargs,
    ):
        """Run HMC sampling for (a_i, b_i) coefficients and optional outlier params."""
        try:
            jax = _configure_jax_gpu_fallback()
            import jax.numpy as jnp
            from jax.scipy.special import i0e
            import numpyro
            import numpyro.distributions as dist
            from numpyro.infer import MCMC, NUTS
        except ImportError as exc:
            raise ImportError("numpyro is not installed. Install it with: pip install numpyro") from exc

        if self.u_values is None:
            raise ValueError("No data set. Use set_data() first.")
        if self.feh_values is None:
            raise ValueError("No metallicity set. Provide feh_values via set_data().")

        coeff_prior_scale = coeff_prior_scale or self.coeff_prior_scale
        feh_coeff_prior_scale = feh_coeff_prior_scale or self.feh_coeff_prior_scale

        n_base = self.poly_model.n_base_params
        if feh_b_mask is None:
            feh_b_mask_list = [True] * n_base
        else:
            feh_b_mask = np.asarray(feh_b_mask, dtype=bool).reshape(-1)
            if feh_b_mask.shape[0] != n_base:
                raise ValueError(f"feh_b_mask must have length {n_base}.")
            feh_b_mask_list = [bool(x) for x in feh_b_mask]

        u_values_jax = jnp.array(self.u_values)
        absg1_values_jax = jnp.array(self.absg1_values)
        absg2_values_jax = jnp.array(self.absg2_values)
        feh_values_jax = jnp.array(self.feh_values)
        norm_factor_jax = jnp.array(self.norm_factor)

        if self.u_sigma_values is not None:
            u_sigma_values_jax = jnp.array(self.u_sigma_values)
            int_ulist_jax = jnp.arange(0.0, float(int_umax), float(int_du))
        else:
            u_sigma_values_jax = None
            int_ulist_jax = None

        def func_pu_8_jax(tilde_u, A=4.95e-3, B=2.24e-3, C=3.85, u0=36.09):
            return A * tilde_u * jnp.exp(-1 * (B * tilde_u**2 + jnp.exp((tilde_u - u0) / C)))

        def gaussian_jax(x, mu, sigma):
            x, mu, sigma = jnp.broadcast_arrays(x, mu, jnp.maximum(sigma, 1e-10))
            coeff = 1.0 / (sigma * jnp.sqrt(2 * jnp.pi))
            exponent = -0.5 * ((x - mu) / sigma) ** 2
            return coeff * jnp.exp(exponent)

        def rice_distribution_jax(u_obs, u_true, sigma):
            u_obs, u_true, sigma = jnp.broadcast_arrays(u_obs, u_true, jnp.maximum(sigma, 1e-10))
            sigma_sq = sigma**2
            bessel_arg = u_obs * u_true / sigma_sq
            log_prefactor = jnp.log(u_obs + 1e-100) - jnp.log(sigma_sq)
            log_exp_term = -(u_obs**2 + u_true**2) / (2 * sigma_sq)
            log_bessel = jnp.log(i0e(bessel_arg) + 1e-100) + jnp.abs(bessel_arg)
            return jnp.maximum(jnp.exp(log_prefactor + log_exp_term + log_bessel), 1e-100)

        def outlier_gaussian_jax(tilde_u, outlier_u0, outlier_sigma):
            sigma = jnp.maximum(outlier_sigma, 1e-10)
            coeff = 1.0 / (sigma * jnp.sqrt(2 * jnp.pi))
            exponent = -0.5 * ((tilde_u - outlier_u0) / sigma) ** 2
            return coeff * jnp.exp(exponent)

        def likelihood_single_u_jax(
            u,
            sqrt_mtot,
            u_sigma=None,
            norm_factor=None,
            *,
            f_outlier=None,
            outlier_u0=None,
            outlier_sigma=None,
        ):
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
                good_component = (1.0 / sqrt_mtot) * func_pu_8_jax(tilde_u)
            else:
                u_sigma = jnp.maximum(u_sigma, 1e-10)

                u_obs_grid = (u / sqrt_mtot)[:, None]
                sigma_grid = (u_sigma / sqrt_mtot)[:, None]
                integration_grid = int_ulist_jax[None, :]

                if self.uncertainty_model == "rice":
                    uncertainty_dist = rice_distribution_jax(u_obs_grid, integration_grid, sigma_grid)
                else:
                    uncertainty_dist = gaussian_jax(u_obs_grid, integration_grid, sigma_grid)

                integrand = (
                    (1.0 / sqrt_mtot[:, None])
                    * func_pu_8_jax(int_ulist_jax)[None, :]
                    * uncertainty_dist
                    * float(int_du)
                )
                norm_ = jnp.maximum(norm_factor, 1e-10)
                good_component = jnp.sum(integrand, axis=1) / norm_ + self.p_epsilon / (float(int_umax) / float(int_du))

            from jax.scipy.special import erf as jax_erf

            outlier_gaussian_norm = 1.0 - 0.5 * (1.0 + jax_erf(-outlier_u0 / outlier_sigma / jnp.sqrt(2.0)))
            outlier_component = outlier_gaussian_jax(tilde_u, outlier_u0, outlier_sigma) / jnp.maximum(
                outlier_gaussian_norm, 1e-10
            )

            total_prob = f_good * good_component + f_outlier * outlier_component + self.p_epsilon
            return jnp.clip(total_prob, 1e-100, 1e10)

        def model(u_values, u_sigma_values, absg1_values, absg2_values, feh_values, norm_factor):
            a = numpyro.sample("a", dist.Normal(0.0, float(coeff_prior_scale)).expand([n_base]))
            b_list = []
            for i in range(n_base):
                if not feh_b_mask_list[i]:
                    b_i = jnp.array(0.0)
                elif i == 0 and feh_b0_positive:
                    b_i = numpyro.sample(f"b_{i}", dist.HalfNormal(float(feh_coeff_prior_scale)))
                else:
                    b_i = numpyro.sample(f"b_{i}", dist.Normal(0.0, float(feh_coeff_prior_scale)))
                b_list.append(b_i)
            b = jnp.stack(b_list)
            coeffs = jnp.concatenate([a, b])
            numpyro.deterministic("coeffs", coeffs)

            if self.poly_model.deriv_penalty_strength > 0:
                numpyro.factor(
                    "mlr_monotonicity",
                    self.poly_model.derivative_penalty_jax(
                        coeffs, n_grid=64, n_feh=self.monotone_n_feh
                    ),
                )

            m1 = self.poly_model.mass_from_absg_feh_jax(absg1_values, feh_values, coeffs)
            m2 = self.poly_model.mass_from_absg_feh_jax(absg2_values, feh_values, coeffs)
            mtot = m1 + m2
            sqrt_mtot = jnp.sqrt(jnp.maximum(mtot, 1e-12))

            if self.fit_outlier_params:
                pi = self.f_outlier_init
                kappa = self.outlier_kappa
                alpha = pi * kappa
                beta_param = (1 - pi) * kappa
                f_outlier = numpyro.sample("f_outlier", dist.Beta(alpha, beta_param))

                u_tail_min = jnp.quantile(u_values / sqrt_mtot, 0.6)
                u_tail_max = float(int_umax)
                outlier_u0 = numpyro.sample(
                    "outlier_u0",
                    dist.TruncatedNormal(low=u_tail_min, high=u_tail_max, loc=u_tail_min + 5.0, scale=5.0),
                )
                outlier_sigma = numpyro.sample("outlier_sigma", dist.Normal(15.0, 2.0))
            else:
                f_outlier = self.f_outlier_init
                outlier_u0 = self.outlier_u0_init
                outlier_sigma = self.outlier_sigma_init

            log_likelihood_vec = jnp.log(
                likelihood_single_u_jax(
                    u_values,
                    sqrt_mtot,
                    u_sigma_values,
                    norm_factor,
                    f_outlier=f_outlier,
                    outlier_u0=outlier_u0,
                    outlier_sigma=outlier_sigma,
                )
            )
            log_likelihood_sum = jnp.sum(log_likelihood_vec)
            log_likelihood_sum = jnp.where(jnp.isfinite(log_likelihood_sum), log_likelihood_sum, -1e10)
            numpyro.factor("obs", log_likelihood_sum)

        nuts_kernel = NUTS(model, dense_mass=use_dense_mass, **kwargs)
        mcmc = MCMC(
            nuts_kernel,
            num_warmup=num_warmup,
            num_samples=num_samples,
            num_chains=num_chains,
            progress_bar=True,
        )

        rng_key = jax.random.PRNGKey(seed if seed is not None else 42)
        mcmc.run(
            rng_key,
            u_values_jax,
            u_sigma_values_jax,
            absg1_values_jax,
            absg2_values_jax,
            feh_values_jax,
            norm_factor_jax,
        )

        self.sampler = mcmc
        samples_dict = mcmc.get_samples()

        coeffs_samples = np.array(samples_dict["coeffs"])

        if self.fit_outlier_params:
            f_outlier_samples = np.array(samples_dict["f_outlier"])
            outlier_u0_samples = np.array(samples_dict["outlier_u0"])
            outlier_sigma_samples = np.array(samples_dict["outlier_sigma"])
            self.samples = np.column_stack([coeffs_samples, f_outlier_samples, outlier_u0_samples, outlier_sigma_samples])
            self.param_names = self.param_names + ["f_outlier", "outlier_u0", "outlier_sigma"]
        else:
            self.samples = coeffs_samples

        self.results = type("obj", (object,), {"samples": self.samples, "logz": None, "logzerr": None})()
        mcmc.print_summary()
        return mcmc
