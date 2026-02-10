"""Continuous-metallicity difference-polynomial MLR model.

This module implements DifferencePoly-style models that:
- use an isochrone baseline m_iso(M_G, [M/H]) from the interpolated grid, and
- fit a metallicity-dependent residual in log10 mass.

Supported parameterizations:

1) Linear-in-metallicity polynomial residual:

    log10 m_dyn(M_G, MH) = log10 m_iso(M_G, MH) + Σ_i (a_i + b_i * z) x^i

2) 2D quadratic perturbation (second order, but allow x-order 0/1/2):

    δ = a0 + b1*z + a1*x + b2*z^2 + a2*x^2 + c*z*x

    where the cross term can be free, tied to a2*b2, or omitted.

Notes on model choices
----------------------
Quadratic2D parameters (x = rescaled M_G, z = rescaled [Fe/H]):
    a0: constant offset in log10 mass.
    a1: linear-in-x trend (magnitude dependence).
    a2: quadratic-in-x curvature.
    b1: linear-in-z trend (metallicity dependence).
    b2: quadratic-in-z curvature.
    c_xy: cross term z * x (coupled magnitude–metallicity effect).

Quadratic2D order:
    order=0 -> fit a0 only
    order=1 -> fit a0 + a1 + b1
    order=2 -> fit a0 + a1 + a2 + b1 + b2
    (use quad_mode="full" to include c_xy or force all terms regardless of order)

Masking precedence for feh_model="quadratic2d":
    quad_mask (if provided) overrides order-based defaults.
    quad_mode (if provided) overrides order-based defaults.
    If neither is provided, order sets the default mask.

Cross-term choice (cross_mode):
    "free"    -> c_xy is an independent parameter
    "product" -> c_xy = a2 * b2 (coupled, no extra parameter)
    "none"    -> c_xy = 0 (no cross term)

Outlier handling (fit_outlier_params):
    False -> f_outlier, outlier_u0, outlier_sigma fixed to initial values
    True  -> these are sampled along with polynomial coefficients

where x is rescaled M_G in [-1, 1] and z is rescaled metallicity in [-1, 1].
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context

from scipy.stats import norm
from scipy.special import erf, i0e


def _predictive_metrics_worker(args):
    (
        coeff_samples,
        f_outlier_samples,
        outlier_u0_samples,
        outlier_sigma_samples,
        u,
        u_sigma,
        absg1,
        absg2,
        feh,
        norm_factor,
        int_ulist,
        func_ulist,
        uncertainty_model,
        int_umax,
        int_du,
        p_epsilon,
        data_chunk,
        poly_model,
    ) = args

    def rice_pdf(u_obs, u_true, sigma):
        sigma = np.maximum(sigma, 1e-10)
        sigma_sq = sigma**2
        bessel_arg = u_obs * u_true / sigma_sq
        log_prefactor = np.log(u_obs + 1e-100) - np.log(sigma_sq)
        log_exp_term = -(u_obs**2 + u_true**2) / (2 * sigma_sq)
        log_bessel = np.log(i0e(bessel_arg) + 1e-100) + np.abs(bessel_arg)
        return np.maximum(np.exp(log_prefactor + log_exp_term + log_bessel), 1e-100)

    def gaussian_pdf(x, mu, sigma):
        sigma = np.maximum(sigma, 1e-10)
        coeff = 1.0 / (sigma * np.sqrt(2 * np.pi))
        exponent = -0.5 * ((x - mu) / sigma) ** 2
        return coeff * np.exp(exponent)

    sum_probs = np.zeros(u.shape[0], dtype=float)
    sum_out = np.zeros(u.shape[0], dtype=float)

    for s in range(coeff_samples.shape[0]):
        theta = coeff_samples[s]
        f_out = f_outlier_samples[s]
        u0 = outlier_u0_samples[s]
        sig = outlier_sigma_samples[s]
        outlier_norm = 1.0 - 0.5 * (1.0 + erf(-u0 / np.maximum(sig, 1e-10) / np.sqrt(2.0)))
        outlier_norm = np.maximum(outlier_norm, 1e-10)
        outlier_ulist = None
        if u_sigma is not None:
            outlier_ulist = gaussian_pdf(int_ulist, u0, sig) / outlier_norm

        for d0 in range(0, u.shape[0], data_chunk):
            d1 = min(d0 + data_chunk, u.shape[0])
            u_c = u[d0:d1]
            absg1_c = absg1[d0:d1]
            absg2_c = absg2[d0:d1]
            feh_c = feh[d0:d1]
            norm_c = norm_factor[d0:d1]
            u_sigma_c = None if u_sigma is None else u_sigma[d0:d1]

            m1 = poly_model.mass_from_absg_feh(absg1_c, feh_c, theta)
            m2 = poly_model.mass_from_absg_feh(absg2_c, feh_c, theta)
            sqrt_mtot = np.sqrt(np.maximum(m1 + m2, 1e-12))
            tilde_u = u_c / sqrt_mtot

            if u_sigma_c is None:
                good = (1.0 / sqrt_mtot) * (
                    4.95e-3
                    * tilde_u
                    * np.exp(-1.0 * (2.24e-3 * tilde_u**2 + np.exp((tilde_u - 36.09) / 3.85)))
                )
                outlier = gaussian_pdf(tilde_u, u0, sig) / outlier_norm
            else:
                u_obs_grid = (u_c / sqrt_mtot)[:, None]
                sigma_grid = (u_sigma_c / sqrt_mtot)[:, None]
                integration_grid = int_ulist[None, :]

                if uncertainty_model == "rice":
                    unc = rice_pdf(u_obs_grid, integration_grid, sigma_grid)
                else:
                    unc = gaussian_pdf(u_obs_grid, integration_grid, sigma_grid)

                integrand = (1.0 / sqrt_mtot[:, None]) * func_ulist[None, :] * unc * float(int_du)
                outlier_integrand = outlier_ulist[None, :] * unc * float(int_du)
                norm_ = np.maximum(norm_c, 1e-10)
                good = np.sum(integrand, axis=1) / norm_ + p_epsilon / (int_umax / int_du)
                outlier = np.sum(outlier_integrand, axis=1) / norm_

            total = (1.0 - f_out) * good + f_out * outlier + p_epsilon
            total = np.clip(total, 1e-100, 1e10)
            sum_probs[d0:d1] += total
            sum_out[d0:d1] += (f_out * outlier) / total

    return sum_probs, sum_out, coeff_samples.shape[0]

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
        deriv_penalty_strength: float, # The penalty enforces d log10(M) / d M_G ≤ 0, i.e., mass should not increase with increasing magnitude.
        feh_monotone_strength: float,
        isochrone_surface_model: IsochroneMassSurfaceModel,
        feh_min: float,
        feh_max: float,
        feh_model: str = "quadratic2d",
        cross_mode: str = "product",
    ):
        if isochrone_surface_model is None:
            raise ValueError("DifferencePolyFehMassAbsgModel requires `isochrone_surface_model`.")
        if not (np.isfinite(feh_min) and np.isfinite(feh_max) and feh_max > feh_min):
            raise ValueError("Invalid (feh_min, feh_max) for metallicity scaling.")
        if feh_model not in {"linear", "quadratic2d"}:
            raise ValueError("feh_model must be 'linear' or 'quadratic2d'.")
        if cross_mode not in {"product", "free", "none"}:
            raise ValueError("cross_mode must be 'product', 'free', or 'none'.")
        if feh_model == "quadratic2d" and order not in {0, 1, 2}:
            raise ValueError("feh_model='quadratic2d' supports order in {0, 1, 2}.")

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

        self.feh_model = feh_model
        self.cross_mode = cross_mode

        if self.feh_model == "linear":
            self.n_base_params = self.order + 1
            self.n_params = 2 * self.n_base_params  # (a_0..a_n, b_0..b_n)
            self.param_names = (
                [f"a_{i}" for i in range(self.n_base_params)]
                + [f"b_{i}" for i in range(self.n_base_params)]
            )
        else:
            self.n_base_params = 3  # x^0, x^1, x^2 basis for the quadratic model
            if self.cross_mode == "free":
                self.n_params = 6
                self.param_names = ["a0", "b1", "a1", "b2", "a2", "c_xy"]
            else:
                self.n_params = 5
                self.param_names = ["a0", "b1", "a1", "b2", "a2"]

    def _split_params(self, params):
        params = np.asarray(params)
        if params.shape[-1] != self.n_params:
            raise ValueError(f"Expected params with last dimension {self.n_params}, got {params.shape}")
        if self.feh_model == "linear":
            a = params[..., : self.n_base_params]
            b = params[..., self.n_base_params :]
            return a, b
        return params

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
        absg = np.asarray(absg, dtype=float)
        feh = np.asarray(feh, dtype=float)
        absg, feh = np.broadcast_arrays(absg, feh)

        x = self._absg_to_x_np(absg)
        z = self._feh_to_z_np(feh)

        if self.feh_model == "linear":
            a, b = self._split_params(params)
            coeffs = a + b * z[..., None]  # (..., n_base_params)
            powers = np.stack([x ** i for i in range(self.order + 1)], axis=-1)  # (..., n_base_params)
            return np.sum(coeffs * powers, axis=-1)

        params = self._split_params(params)
        if self.cross_mode == "free":
            a0, b1, a1, b2, a2, c_xy = np.split(params, 6, axis=-1)
        else:
            a0, b1, a1, b2, a2 = np.split(params, 5, axis=-1)
            if self.cross_mode == "product":
                c_xy = a2 * b2
            else:
                c_xy = 0.0

        a0 = np.squeeze(a0, axis=-1)
        b1 = np.squeeze(b1, axis=-1)
        a1 = np.squeeze(a1, axis=-1)
        b2 = np.squeeze(b2, axis=-1)
        a2 = np.squeeze(a2, axis=-1)
        c_xy = np.squeeze(c_xy, axis=-1)

        return a0 + b1 * z + a1 * x + b2 * (z ** 2) + a2 * (x ** 2) + c_xy * z * x

    def _log10_delta_jax(self, absg, feh, params):
        import jax.numpy as jnp

        absg = jnp.asarray(absg)
        feh = jnp.asarray(feh)
        absg, feh = jnp.broadcast_arrays(absg, feh)

        x = self._absg_to_x_jax(absg)
        z = self._feh_to_z_jax(feh)

        if self.feh_model == "linear":
            a = params[: self.n_base_params]
            b = params[self.n_base_params :]
            coeffs = a + b * z[..., None]
            powers = jnp.stack([jnp.power(x, i) for i in range(self.order + 1)], axis=-1)
            return jnp.sum(coeffs * powers, axis=-1)

        if self.cross_mode == "free":
            a0, b1, a1, b2, a2, c_xy = jnp.split(params, 6)
        else:
            a0, b1, a1, b2, a2 = jnp.split(params, 5)
            if self.cross_mode == "product":
                c_xy = a2 * b2
            else:
                c_xy = 0.0

        return a0 + b1 * z + a1 * x + b2 * (z ** 2) + a2 * (x ** 2) + c_xy * z * x

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
        """Soft penalty enforcing d log10(M) / d M_G <= 0 for a small MH grid.

        feh_monotone_strength controls a *soft* monotonicity penalty in [Fe/H]:
        it penalizes negative d log10(M) / d [Fe/H]. Larger values enforce
        non-decreasing log10(M) with increasing [Fe/H] more strongly.

        Rule-of-thumb scale (log-prob units):
        - 0: off
        - 0.1–1: weak
        - 1–10: medium
        - 10–50: strong
        Approximate calibration: to make a mean negative slope of magnitude
        s (dex per dex) cost ~P log-prob units, set
            feh_monotone_strength ≈ P / s^2.
        Example: s=0.1 and P=1 -> strength ~100.
        """
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
        order: int = 2,
        isochrone_surface_model: IsochroneMassSurfaceModel,
        feh_min: Optional[float],
        feh_max: Optional[float],
        feh_coeff_prior_scale: float = 1.0, # deriv_penalty_strength=10.0: strength of the penalty. Larger values enforce monotonicity more strongly (in log‑probability units).
        feh_monotone_strength: float = 0.0,  # 0=off, 0.1–1 weak, 1–10 medium, 10–50 strong (log-prob units)
        monotone_n_feh: int = 5, # monotone_n_feh=5: number of metallicity grid points used when applying the penalty (i.e., how many [Fe/H] slices are checked). More points = more robust monotonicity across metallicity.
        feh_model: str = "quadratic2d",
        cross_mode: str = "product",
        **kwargs,
    ):
        absg_min_given = "absg_min" in kwargs
        absg_max_given = "absg_max" in kwargs
        absg_min_input = kwargs.pop("absg_min", None) if absg_min_given else None
        absg_max_input = kwargs.pop("absg_max", None) if absg_max_given else None

        self._auto_absg_min = absg_min_given and absg_min_input is None
        self._auto_absg_max = absg_max_given and absg_max_input is None

        super_kwargs = dict(kwargs)
        if absg_min_input is not None:
            super_kwargs["absg_min"] = absg_min_input
        if absg_max_input is not None:
            super_kwargs["absg_max"] = absg_max_input

        super().__init__(order=order, **super_kwargs)

        self.feh_min = None if feh_min is None else float(feh_min)
        self.feh_max = None if feh_max is None else float(feh_max)
        self._auto_feh_min = feh_min is None
        self._auto_feh_max = feh_max is None
        self.feh_coeff_prior_scale = float(feh_coeff_prior_scale)
        self.feh_monotone_strength = float(feh_monotone_strength)
        self.monotone_n_feh = int(monotone_n_feh)

        if isochrone_surface_model is None:
            raise ValueError("DifferencePolyFehMLR requires `isochrone_surface_model`.")

        self.iso_surface = isochrone_surface_model
        self.feh_model = feh_model
        self.cross_mode = cross_mode

        feh_min_init = self.feh_min
        feh_max_init = self.feh_max
        if feh_min_init is None and feh_max_init is None:
            feh_min_init, feh_max_init = -1.0, 0.5
        elif feh_min_init is None:
            feh_min_init = feh_max_init - 1.0
        elif feh_max_init is None:
            feh_max_init = feh_min_init + 1.0

        self.poly_model = DifferencePolyFehMassAbsgModel(
            order=order,
            absg_min=self.absg_min,
            absg_max=self.absg_max,
            mass_min=self.mass_min,
            pivot=self.poly_model.pivot,
            deriv_penalty_strength=self.poly_model.deriv_penalty_strength,
            feh_monotone_strength=self.feh_monotone_strength,
            isochrone_surface_model=self.iso_surface,
            feh_min=feh_min_init,
            feh_max=feh_max_init,
            feh_model=self.feh_model,
            cross_mode=self.cross_mode,
        )

        self.feh_values = None
        self.feh_plot = None  # used by mass_from_absg when feh is not provided

        self.param_names = list(self.poly_model.param_names)

    def _refresh_poly_model(self):
        """Rebuild the polynomial model after updating absg/feh ranges."""
        self.poly_model = DifferencePolyFehMassAbsgModel(
            order=self.poly_model.order,
            absg_min=self.absg_min,
            absg_max=self.absg_max,
            mass_min=self.mass_min,
            pivot=self.poly_model.pivot,
            deriv_penalty_strength=self.poly_model.deriv_penalty_strength,
            feh_monotone_strength=self.feh_monotone_strength,
            isochrone_surface_model=self.iso_surface,
            feh_min=self.feh_min,
            feh_max=self.feh_max,
            feh_model=self.feh_model,
            cross_mode=self.cross_mode,
        )
        self.param_names = list(self.poly_model.param_names)

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

        # Auto-derive normalization ranges from data (robust percentiles).
        need_refresh = False
        if self._auto_absg_min or self._auto_absg_max:
            absg_all = np.concatenate([self.absg1_values, self.absg2_values])
            absg_all = absg_all[np.isfinite(absg_all)]
            if absg_all.size == 0:
                raise ValueError("No finite absg values available to infer absg_min/absg_max.")
            p_lo, p_hi = np.nanpercentile(absg_all, [1.0, 99.0])
            if self._auto_absg_min:
                self.absg_min = float(p_lo)
                need_refresh = True
            if self._auto_absg_max:
                self.absg_max = float(p_hi)
                need_refresh = True
            if self.absg_max <= self.absg_min:
                raise ValueError(
                    f"Invalid absg range inferred from data: [{self.absg_min}, {self.absg_max}]."
                )

        if self._auto_feh_min or self._auto_feh_max:
            feh_all = self.feh_values[np.isfinite(self.feh_values)]
            if feh_all.size == 0:
                raise ValueError("No finite feh values available to infer feh_min/feh_max.")
            p_lo, p_hi = np.nanpercentile(feh_all, [1.0, 99.0])
            if self._auto_feh_min:
                self.feh_min = float(p_lo)
                need_refresh = True
            if self._auto_feh_max:
                self.feh_max = float(p_hi)
                need_refresh = True
            if self.feh_max <= self.feh_min:
                raise ValueError(
                    f"Invalid feh range inferred from data: [{self.feh_min}, {self.feh_max}]."
                )

        if need_refresh:
            self._refresh_poly_model()

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
        quad_mode=None,
        quad_mask=None,
        fit_outlier_params=None,
        save_predictive_metrics_path=None,
        save_predictive_metrics_kwargs=None,
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

        devices = jax.devices()
        gpu_devices = [device for device in devices if device.platform == "gpu"]
        if gpu_devices:
            gpu_names = ", ".join(getattr(device, "device_kind", str(device)) for device in gpu_devices)
            print(f"NumPyro: running on GPU backend ({gpu_names}).")

        coeff_prior_scale = coeff_prior_scale or self.coeff_prior_scale
        feh_coeff_prior_scale = feh_coeff_prior_scale or self.feh_coeff_prior_scale

        if fit_outlier_params is not None:
            self.fit_outlier_params = bool(fit_outlier_params)

        # Reset parameter names to the polynomial-only list before applying masks.
        self.param_names = list(self.poly_model.param_names)

        n_base = self.poly_model.n_base_params
        n_coeff = self.poly_model.n_params

        if self.poly_model.feh_model == "linear":
            if quad_mode is not None or quad_mask is not None:
                raise ValueError("quad_mode/quad_mask are only valid for feh_model='quadratic2d'.")
            if feh_b_mask is None:
                feh_b_mask_list = [True] * n_base
            else:
                feh_b_mask = np.asarray(feh_b_mask, dtype=bool).reshape(-1)
                if feh_b_mask.shape[0] != n_base:
                    raise ValueError(f"feh_b_mask must have length {n_base}.")
                feh_b_mask_list = [bool(x) for x in feh_b_mask]
        else:
            if quad_mode is not None and quad_mask is not None:
                raise ValueError("Provide only one of quad_mode or quad_mask.")

            if quad_mode is not None:
                mode = str(quad_mode).strip().lower()
                if mode in {"core", "a0a1b1", "linear"}:
                    quad_mask_list = [name in {"a0", "a1", "b1"} for name in self.param_names]
                elif mode in {"full", "all"}:
                    quad_mask_list = [True] * n_coeff
                else:
                    raise ValueError("quad_mode must be 'core'/'a0a1b1' or 'full'/'all'.")
            elif quad_mask is None:
                # Default: use `order` to include a-terms and matching b-terms.
                order = int(self.poly_model.order)
                if order <= 0:
                    keep = {"a0"}
                elif order == 1:
                    keep = {"a0", "a1", "b1"}
                else:
                    keep = {"a0", "a1", "a2", "b1", "b2"}
                quad_mask_list = [name in keep for name in self.param_names]
            else:
                if isinstance(quad_mask, dict):
                    quad_mask_list = [bool(quad_mask.get(name, True)) for name in self.param_names]
                else:
                    quad_mask = np.asarray(quad_mask, dtype=bool).reshape(-1)
                    if quad_mask.shape[0] != n_coeff:
                        raise ValueError(f"quad_mask must have length {n_coeff}.")
                    quad_mask_list = [bool(x) for x in quad_mask]

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

        def func_pu_8_jax(tilde_u, A=5.434e-3, B=2.544e-3, C=3.100, u0=35.67):
            # phase selection considered
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

            # valid = (u > 0) & (sqrt_mtot > 0)
            # u = jnp.where(valid, u, 1e-10)
            #  sqrt_mtot = jnp.where(valid, sqrt_mtot, 1e-10)

            valid = (u > 0) & (sqrt_mtot > 0)
            u = jnp.where(valid, u, 1e-10)
            sqrt_mtot = jnp.where(valid, sqrt_mtot, 1e-10)

            tilde_u_obs = u / sqrt_mtot

            from jax.scipy.special import erf as jax_erf
            outlier_gaussian_norm = 1.0 - 0.5 * (1.0 + jax_erf(-outlier_u0 / outlier_sigma / jnp.sqrt(2.0)))
            outlier_gaussian_norm = jnp.maximum(outlier_gaussian_norm, 1e-10)

            if u_sigma is None:
                good_component = (1.0 / sqrt_mtot) * func_pu_8_jax(tilde_u_obs)
                outlier_component = outlier_gaussian_jax(tilde_u_obs, outlier_u0, outlier_sigma) / outlier_gaussian_norm
            else:
                u_sigma = jnp.maximum(u_sigma, 1e-10)

                tilde_u_obs_grid = (u / sqrt_mtot)[:, None]
                tilde_u_sigma_grid = (u_sigma / sqrt_mtot)[:, None]
                tilde_u_grid = int_ulist_jax[None, :]

                if self.uncertainty_model == "rice":
                    uncertainty_dist = rice_distribution_jax(tilde_u_obs_grid, tilde_u_grid, tilde_u_sigma_grid)
                else:
                    uncertainty_dist = gaussian_jax(tilde_u_obs_grid, tilde_u_grid, tilde_u_sigma_grid)

                integrand = (
                    (1.0 / sqrt_mtot[:, None])
                    * func_pu_8_jax(tilde_u_grid)
                    * uncertainty_dist
                    * float(int_du)
                )
                outlier_integrand = (
                    (outlier_gaussian_jax(tilde_u_grid, outlier_u0, outlier_sigma) / outlier_gaussian_norm)
                    * uncertainty_dist
                    * float(int_du)
                )
                uncertainty_norm = jnp.maximum(norm_factor, 1e-10)
                good_component = jnp.sum(integrand, axis=1) / uncertainty_norm + self.p_epsilon / (float(int_umax) / float(int_du))
                outlier_component = jnp.sum(outlier_integrand, axis=1) / uncertainty_norm

            total_prob = f_good * good_component + f_outlier * outlier_component + self.p_epsilon
            return jnp.clip(total_prob, 1e-100, 1e10)

        def model(u_values, u_sigma_values, absg1_values, absg2_values, feh_values, norm_factor):
            if self.poly_model.feh_model == "linear":
                a = numpyro.sample(
                    "a",
                    dist.Uniform(-float(coeff_prior_scale), float(coeff_prior_scale)).expand([n_base]),
                )
                b_list = []
                for i in range(n_base):
                    if not feh_b_mask_list[i]:
                        b_i = jnp.array(0.0)
                    elif i == 0 and feh_b0_positive:
                        b_i = numpyro.sample(
                            f"b_{i}", dist.Uniform(0.0, float(feh_coeff_prior_scale))
                        )
                    else:
                        b_i = numpyro.sample(
                            f"b_{i}",
                            dist.Uniform(-float(feh_coeff_prior_scale), float(feh_coeff_prior_scale)),
                        )
                    b_list.append(b_i)
                b = jnp.stack(b_list)
                coeffs = jnp.concatenate([a, b])
            else:
                def sample_or_zero(name, scale, mask):
                    if not mask:
                        return jnp.array(0.0)
                    return numpyro.sample(
                        name, dist.Uniform(-float(scale), float(scale))
                    )

                a0 = sample_or_zero("a0", coeff_prior_scale, quad_mask_list[0])
                b1 = sample_or_zero("b1", feh_coeff_prior_scale, quad_mask_list[1])
                a1 = sample_or_zero("a1", coeff_prior_scale, quad_mask_list[2])
                b2 = sample_or_zero("b2", feh_coeff_prior_scale, quad_mask_list[3])
                a2 = sample_or_zero("a2", coeff_prior_scale, quad_mask_list[4])

                if self.poly_model.cross_mode == "free":
                    c_xy = sample_or_zero("c_xy", feh_coeff_prior_scale, quad_mask_list[5])
                    coeffs = jnp.stack([a0, b1, a1, b2, a2, c_xy])
                else:
                    coeffs = jnp.stack([a0, b1, a1, b2, a2])
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

            total_prob = likelihood_single_u_jax(
                u_values,
                sqrt_mtot,
                u_sigma_values,
                norm_factor,
                f_outlier=f_outlier,
                outlier_u0=outlier_u0,
                outlier_sigma=outlier_sigma,
            )
            log_likelihood_vec = jnp.log(total_prob)
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
        if save_predictive_metrics_path is not None:
            self.save_predictive_metrics(
                save_predictive_metrics_path,
                **(save_predictive_metrics_kwargs or {}),
            )
        return mcmc

    def compute_predictive_metrics(
        self,
        *,
        params_samples=None,
        int_umax=80.0,
        int_du=0.02,
        sample_limit=None,
        rng=None,
        data_chunk=2048,
        num_workers=0,
        sample_chunk=64,
        show_progress=False,
        f_outlier_init=None,
        outlier_u0_init=None,
        outlier_sigma_init=None,
        p_epsilon=None,
    ):
        """Compute per-point posterior predictive log-likelihood and outlier probability."""
        if self.u_values is None or self.absg1_values is None or self.absg2_values is None:
            raise ValueError("No data set. Use set_data() first.")
        if self.feh_values is None:
            raise ValueError("No metallicity set. Provide feh_values via set_data().")

        samples = self.samples if params_samples is None else np.asarray(params_samples)
        if samples is None or samples.size == 0:
            raise ValueError("No posterior samples available.")

        n_coeff = self.poly_model.n_params
        samples = np.atleast_2d(samples)
        if samples.shape[1] == n_coeff + 3:
            coeff_samples = samples[:, :n_coeff]
            f_outlier_samples = samples[:, n_coeff]
            outlier_u0_samples = samples[:, n_coeff + 1]
            outlier_sigma_samples = samples[:, n_coeff + 2]
        elif samples.shape[1] == n_coeff:
            f_outlier_init = self.f_outlier_init if f_outlier_init is None else f_outlier_init
            outlier_u0_init = self.outlier_u0_init if outlier_u0_init is None else outlier_u0_init
            outlier_sigma_init = self.outlier_sigma_init if outlier_sigma_init is None else outlier_sigma_init
            coeff_samples = samples
            f_outlier_samples = np.full(samples.shape[0], f_outlier_init, dtype=float)
            outlier_u0_samples = np.full(samples.shape[0], outlier_u0_init, dtype=float)
            outlier_sigma_samples = np.full(samples.shape[0], outlier_sigma_init, dtype=float)
        else:
            raise ValueError(
                f"Unexpected sample width: {samples.shape[1]} (expected {n_coeff} or {n_coeff + 3})"
            )

        if sample_limit is not None and samples.shape[0] > sample_limit:
            rng = np.random.default_rng(rng)
            idx = rng.choice(samples.shape[0], size=int(sample_limit), replace=False)
            coeff_samples = coeff_samples[idx]
            f_outlier_samples = f_outlier_samples[idx]
            outlier_u0_samples = outlier_u0_samples[idx]
            outlier_sigma_samples = outlier_sigma_samples[idx]

        p_epsilon = self.p_epsilon if p_epsilon is None else float(p_epsilon)

        u = np.asarray(self.u_values, dtype=float)
        u_sigma = None if self.u_sigma_values is None else np.asarray(self.u_sigma_values, dtype=float)
        absg1 = np.asarray(self.absg1_values, dtype=float)
        absg2 = np.asarray(self.absg2_values, dtype=float)
        feh = np.asarray(self.feh_values, dtype=float)

        if u_sigma is None:
            norm_factor = np.ones_like(u)
        elif self.uncertainty_model == "gaussian":
            norm_factor = 1.0 - norm.cdf(0.0, loc=u, scale=u_sigma)
        else:
            norm_factor = np.ones_like(u)

        int_ulist = np.arange(0.0, float(int_umax), float(int_du))
        func_ulist = 4.95e-3 * int_ulist * np.exp(
            -1.0 * (2.24e-3 * int_ulist**2 + np.exp((int_ulist - 36.09) / 3.85))
        )

        def rice_pdf(u_obs, u_true, sigma):
            sigma = np.maximum(sigma, 1e-10)
            sigma_sq = sigma**2
            bessel_arg = u_obs * u_true / sigma_sq
            log_prefactor = np.log(u_obs + 1e-100) - np.log(sigma_sq)
            log_exp_term = -(u_obs**2 + u_true**2) / (2 * sigma_sq)
            log_bessel = np.log(i0e(bessel_arg) + 1e-100) + np.abs(bessel_arg)
            return np.maximum(np.exp(log_prefactor + log_exp_term + log_bessel), 1e-100)

        def gaussian_pdf(x, mu, sigma):
            sigma = np.maximum(sigma, 1e-10)
            coeff = 1.0 / (sigma * np.sqrt(2 * np.pi))
            exponent = -0.5 * ((x - mu) / sigma) ** 2
            return coeff * np.exp(exponent)

        sum_probs = np.zeros(u.shape[0], dtype=float)
        sum_out = np.zeros(u.shape[0], dtype=float)
        n_samples = coeff_samples.shape[0]

        if num_workers and n_samples > sample_chunk:
            futures = []
            with ProcessPoolExecutor(max_workers=num_workers, mp_context=get_context("spawn")) as executor:
                for s0 in range(0, n_samples, sample_chunk):
                    s1 = min(s0 + sample_chunk, n_samples)
                    futures.append(
                        executor.submit(
                            _predictive_metrics_worker,
                            (
                                coeff_samples[s0:s1],
                                f_outlier_samples[s0:s1],
                                outlier_u0_samples[s0:s1],
                                outlier_sigma_samples[s0:s1],
                                u,
                                u_sigma,
                                absg1,
                                absg2,
                                feh,
                                norm_factor,
                                int_ulist,
                                func_ulist,
                                self.uncertainty_model,
                                int_umax,
                                int_du,
                                p_epsilon,
                                data_chunk,
                                self.poly_model,
                            ),
                        )
                    )
                for idx, future in enumerate(futures, start=1):
                    sum_probs_c, sum_out_c, n_part = future.result()
                    sum_probs += sum_probs_c
                    sum_out += sum_out_c
                    if show_progress:
                        done = min(idx * sample_chunk, n_samples)
                        print(f"predictive_metrics: {done}/{n_samples} samples processed")
        else:
            sum_probs, sum_out, _ = _predictive_metrics_worker(
                (
                    coeff_samples,
                    f_outlier_samples,
                    outlier_u0_samples,
                    outlier_sigma_samples,
                    u,
                    u_sigma,
                    absg1,
                    absg2,
                    feh,
                    norm_factor,
                    int_ulist,
                    func_ulist,
                    self.uncertainty_model,
                    int_umax,
                    int_du,
                    p_epsilon,
                    data_chunk,
                    self.poly_model,
                )
            )
            if show_progress:
                print(f"predictive_metrics: {n_samples}/{n_samples} samples processed")

        log_pred = np.log(sum_probs / n_samples)
        p_out = sum_out / n_samples
        return log_pred, p_out

    def save_predictive_metrics(
        self,
        output_path,
        *,
        params_samples=None,
        **kwargs,
    ):
        """Compute and save log_pred/p_out arrays to a .npz file."""
        log_pred, p_out = self.compute_predictive_metrics(
            params_samples=params_samples,
            **kwargs,
        )
        np.savez(output_path, log_pred=log_pred, p_out=p_out)
        return output_path
