"""Minimal two-stage hierarchical metallicity and dynamical-MLR model.

This module is intentionally independent of :mod:`differencepoly_feh`.  Stage
one combines two component metallicity measurements and two CMD positions on a
fixed, global PARSEC metallicity grid.  Stage two consumes the resulting
per-system grid probabilities and fits a small three-knot correction to the
PARSEC mass surface.

The public classes in this file are deliberately narrow test interfaces; the
existing DifferencePolyFehMLR implementation is not modified or subclassed.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy.special import gammaln, logsumexp, ndtr

from .differencepoly_feh import IsochroneMassSurfaceModel
from .jax_utils import _configure_jax_gpu_fallback


DEFAULT_Z_GRID = np.linspace(-1.0, 0.6, 81, dtype=np.float64)
DEFAULT_MG_KNOTS = np.array([3.5, 8.5, 13.5], dtype=np.float64)


def _as_native_f64(value, *, name: str, ndim: Optional[int] = None) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if ndim is not None and array.ndim != ndim:
        raise ValueError(f"{name} must have ndim={ndim}; got shape {array.shape}.")
    return array


def trapezoid_weights(grid: Sequence[float]) -> np.ndarray:
    """Return integration weights for a strictly increasing one-dimensional grid."""
    grid = _as_native_f64(grid, name="grid", ndim=1)
    if grid.size < 2 or np.any(np.diff(grid) <= 0):
        raise ValueError("grid must contain at least two strictly increasing values.")
    weights = np.empty_like(grid)
    weights[0] = 0.5 * (grid[1] - grid[0])
    weights[-1] = 0.5 * (grid[-1] - grid[-2])
    weights[1:-1] = 0.5 * (grid[2:] - grid[:-2])
    return weights


def color_uncertainty_from_flux_snr(bp_flux_over_error, rp_flux_over_error) -> np.ndarray:
    """Propagate Gaia BP/RP flux S/N values into a BP-RP magnitude error."""
    bp_snr = _as_native_f64(bp_flux_over_error, name="bp_flux_over_error")
    rp_snr = _as_native_f64(rp_flux_over_error, name="rp_flux_over_error")
    bp_snr, rp_snr = np.broadcast_arrays(bp_snr, rp_snr)
    if np.any(~np.isfinite(bp_snr)) or np.any(~np.isfinite(rp_snr)):
        raise ValueError("BP/RP flux-over-error values must be finite.")
    if np.any(bp_snr <= 0) or np.any(rp_snr <= 0):
        raise ValueError("BP/RP flux-over-error values must be strictly positive.")
    magnitude_factor = 2.5 / np.log(10.0)
    return magnitude_factor * np.sqrt(bp_snr ** -2 + rp_snr ** -2)


def student_t_logpdf_np(x, *, df: float, loc, scale) -> np.ndarray:
    """Small vectorized Student-t log density used by posterior post-processing."""
    x = np.asarray(x, dtype=np.float64)
    loc = np.asarray(loc, dtype=np.float64)
    scale = np.asarray(scale, dtype=np.float64)
    scale = np.maximum(scale, 1e-12)
    standardized = (x - loc) / scale
    constant = gammaln((df + 1.0) / 2.0) - gammaln(df / 2.0)
    constant -= 0.5 * np.log(df * np.pi) + np.log(scale)
    return constant - 0.5 * (df + 1.0) * np.log1p(standardized**2 / df)


def normalize_log_weights(log_weights: np.ndarray, axis: int = -1) -> np.ndarray:
    """Normalize log weights, raising if an entire row is non-finite."""
    log_weights = np.asarray(log_weights, dtype=np.float64)
    normalizer = logsumexp(log_weights, axis=axis, keepdims=True)
    if np.any(~np.isfinite(normalizer)):
        raise ValueError("At least one log-weight row has no finite support.")
    return np.exp(log_weights - normalizer)


def marginalize_dynamics(log_conditional: np.ndarray, z_probabilities: np.ndarray) -> np.ndarray:
    """Marginalize a dynamics-only log likelihood over normalized P(Z)."""
    log_conditional = np.asarray(log_conditional, dtype=np.float64)
    z_probabilities = np.asarray(z_probabilities, dtype=np.float64)
    if log_conditional.shape != z_probabilities.shape:
        raise ValueError(
            "log_conditional and z_probabilities must have identical shapes; "
            f"got {log_conditional.shape} and {z_probabilities.shape}."
        )
    if np.any(z_probabilities < 0):
        raise ValueError("z_probabilities cannot be negative.")
    sums = np.sum(z_probabilities, axis=-1)
    if not np.allclose(sums, 1.0, atol=1e-6):
        raise ValueError("Every z_probabilities row must sum to one.")
    return logsumexp(np.log(np.maximum(z_probabilities, 1e-300)) + log_conditional, axis=-1)


def array_digest(*arrays) -> str:
    """Stable SHA-256 digest for row/data validation across model stages."""
    digest = hashlib.sha256()
    for value in arrays:
        array = np.ascontiguousarray(value)
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(array.view(np.uint8))
    return digest.hexdigest()


class IsochroneColorSurfaceModel:
    """Bilinear PARSEC surface C(M_G, [M/H]) for de-reddened BP-RP color."""

    def __init__(self, absg_grid, mh_grid, color_grid):
        self.absg_grid_np = _as_native_f64(absg_grid, name="absg_grid", ndim=1)
        self.mh_grid_np = _as_native_f64(mh_grid, name="mh_grid", ndim=1)
        self.color_grid_np = _as_native_f64(color_grid, name="color_grid", ndim=2)
        if self.color_grid_np.shape != (self.mh_grid_np.size, self.absg_grid_np.size):
            raise ValueError(
                "color_grid must have shape (n_mh, n_absg); got "
                f"{self.color_grid_np.shape}."
            )
        if np.any(np.diff(self.absg_grid_np) <= 0) or np.any(np.diff(self.mh_grid_np) <= 0):
            raise ValueError("absg_grid and mh_grid must be strictly increasing.")
        if np.any(~np.isfinite(self.color_grid_np)):
            raise ValueError("color_grid must contain only finite values.")
        self._jax_cache = None

    @classmethod
    def from_parsec_csv(
        cls,
        csv_path,
        *,
        absg_grid: Optional[Sequence[float]] = None,
        logage_min: float = 9.3,
        logage_max: float = 10.0,
        absg_min: float = 3.5,
        absg_max: float = 13.5,
    ) -> "IsochroneColorSurfaceModel":
        """Build the color surface in memory using the existing grid selection."""
        try:
            from astropy.table import Table
            from scipy.interpolate import UnivariateSpline
        except ImportError as exc:  # pragma: no cover - project dependencies
            raise ImportError("Astropy and SciPy are required to read the PARSEC table.") from exc

        table = Table.read(str(csv_path))
        required = {"logAge", "Gmag", "MH", "G_BPmag", "G_RPmag"}
        missing = sorted(required.difference(table.colnames))
        if missing:
            raise ValueError(f"PARSEC table is missing columns: {missing}.")

        logage = np.asarray(table["logAge"], dtype=float)
        gmag = np.asarray(table["Gmag"], dtype=float)
        mask = (
            (logage > float(logage_min))
            & (logage < float(logage_max))
            & (gmag > 3.0)
            & (gmag < 15.0)
        )
        table = table[mask]
        if len(table) == 0:
            raise ValueError("No PARSEC rows remain after the age/magnitude selection.")

        if absg_grid is None:
            absg_grid = np.linspace(absg_min, absg_max, 500)
        absg_grid = _as_native_f64(absg_grid, name="absg_grid", ndim=1)
        if absg_grid[0] < absg_min - 1e-12 or absg_grid[-1] > absg_max + 1e-12:
            raise ValueError("absg_grid lies outside the requested color-surface domain.")

        mh_values = np.asarray(table["MH"], dtype=float)
        mh_grid = np.unique(mh_values)
        curves = []
        for mh in mh_grid:
            selected = np.isclose(mh_values, mh)
            g = np.asarray(table["Gmag"][selected], dtype=float)
            color = (
                np.asarray(table["G_BPmag"][selected], dtype=float)
                - np.asarray(table["G_RPmag"][selected], dtype=float)
            )
            order = np.argsort(g)
            g, color = g[order], color[order]
            unique_g, inverse = np.unique(g, return_inverse=True)
            mean_color = np.array(
                [np.mean(color[inverse == index]) for index in range(unique_g.size)]
            )
            if unique_g.size < 2:
                raise ValueError(f"Too few PARSEC points at MH={mh:+.2f}.")
            degree = min(3, unique_g.size - 1)
            spline = UnivariateSpline(unique_g, mean_color, k=degree, s=0.001, ext="const")
            curves.append(np.asarray(spline(absg_grid), dtype=np.float64))
        return cls(absg_grid, mh_grid, np.vstack(curves))

    def color_from_absg_mh(self, absg, mh) -> np.ndarray:
        absg = np.asarray(absg, dtype=np.float64)
        mh = np.asarray(mh, dtype=np.float64)
        absg, mh = np.broadcast_arrays(absg, mh)
        absg_c = np.clip(absg, self.absg_grid_np[0], self.absg_grid_np[-1])
        mh_c = np.clip(mh, self.mh_grid_np[0], self.mh_grid_np[-1])
        ia1 = np.clip(np.searchsorted(self.absg_grid_np, absg_c, side="right"), 1, self.absg_grid_np.size - 1)
        ia0 = ia1 - 1
        im1 = np.clip(np.searchsorted(self.mh_grid_np, mh_c, side="right"), 1, self.mh_grid_np.size - 1)
        im0 = im1 - 1
        a0, a1 = self.absg_grid_np[ia0], self.absg_grid_np[ia1]
        z0, z1 = self.mh_grid_np[im0], self.mh_grid_np[im1]
        wa = (absg_c - a0) / np.maximum(a1 - a0, 1e-12)
        wz = (mh_c - z0) / np.maximum(z1 - z0, 1e-12)
        c00 = self.color_grid_np[im0, ia0]
        c01 = self.color_grid_np[im0, ia1]
        c10 = self.color_grid_np[im1, ia0]
        c11 = self.color_grid_np[im1, ia1]
        return (c00 * (1.0 - wa) + c01 * wa) * (1.0 - wz) + (
            c10 * (1.0 - wa) + c11 * wa
        ) * wz

    def color_from_absg_mh_jax(self, absg, mh):
        import jax.numpy as jnp

        if self._jax_cache is None:
            self._jax_cache = (
                jnp.asarray(self.absg_grid_np),
                jnp.asarray(self.mh_grid_np),
                jnp.asarray(self.color_grid_np),
            )
        absg_grid, mh_grid, color_grid = self._jax_cache
        absg, mh = jnp.broadcast_arrays(jnp.asarray(absg), jnp.asarray(mh))
        absg_c = jnp.clip(absg, absg_grid[0], absg_grid[-1])
        mh_c = jnp.clip(mh, mh_grid[0], mh_grid[-1])
        ia1 = jnp.clip(jnp.searchsorted(absg_grid, absg_c, side="right"), 1, absg_grid.size - 1)
        ia0 = ia1 - 1
        im1 = jnp.clip(jnp.searchsorted(mh_grid, mh_c, side="right"), 1, mh_grid.size - 1)
        im0 = im1 - 1
        a0, a1 = absg_grid[ia0], absg_grid[ia1]
        z0, z1 = mh_grid[im0], mh_grid[im1]
        wa = (absg_c - a0) / jnp.maximum(a1 - a0, 1e-12)
        wz = (mh_c - z0) / jnp.maximum(z1 - z0, 1e-12)
        c00 = color_grid[im0, ia0]
        c01 = color_grid[im0, ia1]
        c10 = color_grid[im1, ia0]
        c11 = color_grid[im1, ia1]
        return (c00 * (1.0 - wa) + c01 * wa) * (1.0 - wz) + (
            c10 * (1.0 - wa) + c11 * wa
        ) * wz


@dataclass(frozen=True)
class MetallicityPosteriorGrid:
    """Portable stage-one output consumed by the MLR stage."""

    row_indices: np.ndarray
    z_grid: np.ndarray
    probabilities: np.ndarray
    z_quantiles: np.ndarray
    bad_probabilities: np.ndarray
    metadata: Mapping[str, object]

    def validate(self) -> None:
        rows = np.asarray(self.row_indices)
        grid = np.asarray(self.z_grid)
        probs = np.asarray(self.probabilities)
        if rows.ndim != 1 or grid.ndim != 1 or probs.shape != (rows.size, grid.size):
            raise ValueError("MetallicityPosteriorGrid has inconsistent array shapes.")
        if np.any(probs < 0) or not np.allclose(probs.sum(axis=1), 1.0, atol=2e-5):
            raise ValueError("Metallicity posterior rows must be non-negative and normalized.")
        if np.asarray(self.z_quantiles).shape != (rows.size, 3):
            raise ValueError("z_quantiles must have shape (n_systems, 3).")
        if np.asarray(self.bad_probabilities).shape != (rows.size, 2):
            raise ValueError("bad_probabilities must have shape (n_systems, 2).")

    def save(self, path) -> Path:
        self.validate()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            row_indices=np.asarray(self.row_indices, dtype=np.int64),
            z_grid=np.asarray(self.z_grid, dtype=np.float64),
            probabilities=np.asarray(self.probabilities, dtype=np.float32),
            z_quantiles=np.asarray(self.z_quantiles, dtype=np.float32),
            bad_probabilities=np.asarray(self.bad_probabilities, dtype=np.float32),
            metadata_json=np.asarray(json.dumps(dict(self.metadata), sort_keys=True)),
        )
        return path

    @classmethod
    def load(cls, path) -> "MetallicityPosteriorGrid":
        with np.load(path, allow_pickle=False) as saved:
            result = cls(
                row_indices=np.asarray(saved["row_indices"], dtype=np.int64),
                z_grid=np.asarray(saved["z_grid"], dtype=np.float64),
                probabilities=np.asarray(saved["probabilities"], dtype=np.float64),
                z_quantiles=np.asarray(saved["z_quantiles"], dtype=np.float64),
                bad_probabilities=np.asarray(saved["bad_probabilities"], dtype=np.float64),
                metadata=json.loads(str(saved["metadata_json"].item())),
            )
        result.validate()
        return result


class HierarchicalMetallicityCalibrator:
    """Stage-one shared-metallicity calibrator on a global fixed grid."""

    def __init__(
        self,
        color_surface: IsochroneColorSurfaceModel,
        *,
        z_grid: Sequence[float] = DEFAULT_Z_GRID,
        population_components: int = 6,
    ):
        self.color_surface = color_surface
        self.z_grid = _as_native_f64(z_grid, name="z_grid", ndim=1)
        self.grid_weights = trapezoid_weights(self.z_grid)
        self.population_components = int(population_components)
        if self.population_components < 2:
            raise ValueError("population_components must be at least two.")
        self.population_centers = np.linspace(
            self.z_grid[0], self.z_grid[-1], self.population_components
        )
        self.population_sigma = (self.z_grid[-1] - self.z_grid[0]) / self.population_components
        self.sampler = None
        self.posterior_samples = None
        self._data_set = False

    def set_data(
        self,
        *,
        row_indices,
        absg,
        feh_observed,
        feh_sigma,
        color_observed,
        color_sigma,
    ) -> None:
        self.row_indices = np.asarray(row_indices, dtype=np.int64)
        self.absg = _as_native_f64(absg, name="absg", ndim=2)
        self.feh_observed = _as_native_f64(feh_observed, name="feh_observed", ndim=2)
        self.feh_sigma = _as_native_f64(feh_sigma, name="feh_sigma", ndim=2)
        self.color_observed = _as_native_f64(color_observed, name="color_observed", ndim=2)
        self.color_sigma = _as_native_f64(color_sigma, name="color_sigma", ndim=2)
        expected = (self.row_indices.size, 2)
        for name in ("absg", "feh_observed", "feh_sigma", "color_observed", "color_sigma"):
            if getattr(self, name).shape != expected:
                raise ValueError(f"{name} must have shape {expected}; got {getattr(self, name).shape}.")
        stacked = np.column_stack(
            [self.absg, self.feh_observed, self.feh_sigma, self.color_observed, self.color_sigma]
        )
        if np.any(~np.isfinite(stacked)):
            raise ValueError("Stage-one arrays must contain only finite values.")
        if np.any(self.feh_sigma <= 0) or np.any(self.color_sigma <= 0):
            raise ValueError("All metallicity and color uncertainties must be positive.")
        if np.any(self.absg < self.color_surface.absg_grid_np[0]) or np.any(
            self.absg > self.color_surface.absg_grid_np[-1]
        ):
            raise ValueError("Absolute magnitudes lie outside the color-surface domain.")

        pooled = self.feh_observed.reshape(-1)
        self.bad_loc = float(np.median(pooled))
        mad = float(np.median(np.abs(pooled - self.bad_loc)))
        self.bad_scale = max(0.5, 1.4826 * mad)
        self._data_set = True

    def _population_component_density_np(self) -> np.ndarray:
        standardized = (self.z_grid[:, None] - self.population_centers[None, :]) / self.population_sigma
        norm = ndtr((self.z_grid[-1] - self.population_centers) / self.population_sigma)
        norm -= ndtr((self.z_grid[0] - self.population_centers) / self.population_sigma)
        return np.exp(-0.5 * standardized**2) / (
            np.sqrt(2.0 * np.pi) * self.population_sigma * norm[None, :]
        )

    def log_joint_numpy(self, params: Mapping[str, np.ndarray]) -> np.ndarray:
        """Evaluate per-system/per-grid log joint density for diagnostics/tests."""
        if not self._data_set:
            raise ValueError("Call set_data() first.")
        z_offset = float(np.asarray(params["z_offset"]))
        z_scale = float(np.asarray(params["z_scale"]))
        z_mag_slope = float(np.asarray(params["z_mag_slope"]))
        z_extra_scatter = float(np.asarray(params["z_extra_scatter"]))
        bad_fraction = float(np.asarray(params["bad_fraction"]))
        cmd_scatter = float(np.asarray(params["cmd_scatter"]))
        population_weights = np.asarray(params["population_weights"], dtype=float)
        population_density = self._population_component_density_np() @ population_weights
        log_joint = np.log(self.grid_weights)[None, :] + np.log(population_density)[None, :]
        for component in range(2):
            loc = (
                z_offset
                + z_scale * self.z_grid[None, :]
                + z_mag_slope * (self.absg[:, component, None] - 8.5) / 5.0
            )
            scale = np.sqrt(self.feh_sigma[:, component, None] ** 2 + z_extra_scatter**2)
            log_good = student_t_logpdf_np(
                self.feh_observed[:, component, None], df=4.0, loc=loc, scale=scale
            )
            log_bad = student_t_logpdf_np(
                self.feh_observed[:, component, None],
                df=2.0,
                loc=self.bad_loc,
                scale=self.bad_scale,
            )
            log_metal = np.logaddexp(
                np.log1p(-bad_fraction) + log_good,
                np.log(bad_fraction) + log_bad,
            )
            predicted_color = self.color_surface.color_from_absg_mh(
                self.absg[:, component, None], self.z_grid[None, :]
            )
            color_scale = np.sqrt(self.color_sigma[:, component, None] ** 2 + cmd_scatter**2)
            log_color = student_t_logpdf_np(
                self.color_observed[:, component, None],
                df=4.0,
                loc=predicted_color,
                scale=color_scale,
            )
            log_joint = log_joint + log_metal + log_color
        return log_joint

    def _build_numpyro_model(self):
        import jax.numpy as jnp
        from jax.scipy.special import logsumexp as jax_logsumexp
        from jax.scipy.special import ndtr as jax_ndtr
        import numpyro
        import numpyro.distributions as dist

        z_grid = jnp.asarray(self.z_grid)
        log_grid_weights = jnp.log(jnp.asarray(self.grid_weights))
        centers = jnp.asarray(self.population_centers)
        sigma_pop = float(self.population_sigma)
        standardized = (z_grid[:, None] - centers[None, :]) / sigma_pop
        component_norm = jax_ndtr((float(self.z_grid[-1]) - centers) / sigma_pop)
        component_norm -= jax_ndtr((float(self.z_grid[0]) - centers) / sigma_pop)
        component_density = jnp.exp(-0.5 * standardized**2) / (
            jnp.sqrt(2.0 * jnp.pi) * sigma_pop * component_norm[None, :]
        )
        bad_loc, bad_scale = float(self.bad_loc), float(self.bad_scale)

        def model(absg, feh_observed, feh_sigma, color_observed, color_sigma):
            z_offset = numpyro.sample("z_offset", dist.Normal(0.0, 0.3))
            z_scale = numpyro.sample("z_scale", dist.LogNormal(0.0, 0.25))
            z_mag_slope = numpyro.sample("z_mag_slope", dist.Normal(0.0, 0.3))
            z_extra_scatter = numpyro.sample("z_extra_scatter", dist.HalfNormal(0.3))
            bad_fraction = numpyro.sample("bad_fraction", dist.Beta(1.0, 4.0))
            cmd_scatter = numpyro.sample("cmd_scatter", dist.HalfNormal(0.1))
            population_weights = numpyro.sample(
                "population_weights", dist.Dirichlet(jnp.ones(self.population_components))
            )
            population_density = component_density @ population_weights
            log_joint = log_grid_weights[None, :] + jnp.log(population_density)[None, :]
            for component in range(2):
                loc = (
                    z_offset
                    + z_scale * z_grid[None, :]
                    + z_mag_slope * (absg[:, component, None] - 8.5) / 5.0
                )
                scale = jnp.sqrt(feh_sigma[:, component, None] ** 2 + z_extra_scatter**2)
                log_good = dist.StudentT(4.0, loc=loc, scale=scale).log_prob(
                    feh_observed[:, component, None]
                )
                log_bad = dist.StudentT(2.0, loc=bad_loc, scale=bad_scale).log_prob(
                    feh_observed[:, component, None]
                )
                log_metal = jnp.logaddexp(
                    jnp.log1p(-bad_fraction) + log_good,
                    jnp.log(bad_fraction) + log_bad,
                )
                predicted_color = self.color_surface.color_from_absg_mh_jax(
                    absg[:, component, None], z_grid[None, :]
                )
                effective_color_sigma = jnp.sqrt(
                    color_sigma[:, component, None] ** 2 + cmd_scatter**2
                )
                log_color = dist.StudentT(
                    4.0, loc=predicted_color, scale=effective_color_sigma
                ).log_prob(color_observed[:, component, None])
                log_joint = log_joint + log_metal + log_color
            log_likelihood = jax_logsumexp(log_joint, axis=1)
            numpyro.factor("observations", jnp.sum(log_likelihood))

        return model

    def run_mcmc(
        self,
        *,
        num_warmup: int = 1000,
        num_samples: int = 1000,
        num_chains: int = 4,
        seed: int = 42,
        progress_bar: bool = True,
        **nuts_kwargs,
    ):
        if not self._data_set:
            raise ValueError("Call set_data() before run_mcmc().")
        jax = _configure_jax_gpu_fallback()
        import jax.numpy as jnp
        from numpyro.infer import MCMC, NUTS

        model = self._build_numpyro_model()
        kernel = NUTS(model, **nuts_kwargs)
        mcmc = MCMC(
            kernel,
            num_warmup=int(num_warmup),
            num_samples=int(num_samples),
            num_chains=int(num_chains),
            progress_bar=bool(progress_bar),
        )
        mcmc.run(
            jax.random.PRNGKey(int(seed)),
            jnp.asarray(self.absg),
            jnp.asarray(self.feh_observed),
            jnp.asarray(self.feh_sigma),
            jnp.asarray(self.color_observed),
            jnp.asarray(self.color_sigma),
            extra_fields=("diverging", "energy", "potential_energy", "num_steps", "accept_prob"),
        )
        self.sampler = mcmc
        self.posterior_samples = {
            name: np.asarray(values) for name, values in mcmc.get_samples().items()
        }
        return mcmc

    def posterior_grid(
        self,
        *,
        max_draws: int = 128,
        system_chunk: int = 256,
        seed: int = 0,
        metadata: Optional[Mapping[str, object]] = None,
    ) -> MetallicityPosteriorGrid:
        """Average P(Z) and bad-component responsibilities over posterior draws."""
        if self.posterior_samples is None:
            raise ValueError("Run run_mcmc() before posterior_grid().")
        sample_count = next(iter(self.posterior_samples.values())).shape[0]
        draw_count = min(int(max_draws), sample_count)
        if draw_count < 1:
            raise ValueError("max_draws must be positive.")
        draw_indices = np.linspace(0, sample_count - 1, draw_count, dtype=int)
        if draw_count < sample_count:
            rng = np.random.default_rng(seed)
            draw_indices = np.sort(rng.choice(sample_count, size=draw_count, replace=False))
        samples = {name: values[draw_indices] for name, values in self.posterior_samples.items()}
        component_density = self._population_component_density_np()
        n_systems, n_grid = self.row_indices.size, self.z_grid.size
        mean_probabilities = np.empty((n_systems, n_grid), dtype=np.float64)
        mean_bad = np.empty((n_systems, 2), dtype=np.float64)

        for start in range(0, n_systems, int(system_chunk)):
            stop = min(start + int(system_chunk), n_systems)
            batch = stop - start
            population_density = np.einsum(
                "qk,dk->dq", component_density, samples["population_weights"]
            )
            log_joint = (
                np.log(self.grid_weights)[None, None, :]
                + np.log(population_density)[:, None, :]
            )
            log_joint = np.broadcast_to(log_joint, (draw_count, batch, n_grid)).copy()
            responsibilities = []
            for component in range(2):
                loc = (
                    samples["z_offset"][:, None, None]
                    + samples["z_scale"][:, None, None] * self.z_grid[None, None, :]
                    + samples["z_mag_slope"][:, None, None]
                    * (self.absg[start:stop, component][None, :, None] - 8.5)
                    / 5.0
                )
                effective_sigma = np.sqrt(
                    self.feh_sigma[start:stop, component][None, :, None] ** 2
                    + samples["z_extra_scatter"][:, None, None] ** 2
                )
                observed = self.feh_observed[start:stop, component][None, :, None]
                log_good = student_t_logpdf_np(observed, df=4.0, loc=loc, scale=effective_sigma)
                log_bad = student_t_logpdf_np(
                    observed, df=2.0, loc=self.bad_loc, scale=self.bad_scale
                )
                log_pi = np.log(samples["bad_fraction"][:, None, None])
                log_one_minus_pi = np.log1p(-samples["bad_fraction"][:, None, None])
                log_metal = np.logaddexp(log_one_minus_pi + log_good, log_pi + log_bad)
                responsibilities.append(np.exp(log_pi + log_bad - log_metal))
                predicted_color = self.color_surface.color_from_absg_mh(
                    self.absg[start:stop, component, None], self.z_grid[None, :]
                )[None, :, :]
                effective_color_sigma = np.sqrt(
                    self.color_sigma[start:stop, component][None, :, None] ** 2
                    + samples["cmd_scatter"][:, None, None] ** 2
                )
                log_color = student_t_logpdf_np(
                    self.color_observed[start:stop, component][None, :, None],
                    df=4.0,
                    loc=predicted_color,
                    scale=effective_color_sigma,
                )
                log_joint += log_metal + log_color
            probabilities = normalize_log_weights(log_joint, axis=2)
            mean_probabilities[start:stop] = np.mean(probabilities, axis=0)
            for component in range(2):
                conditional_bad = np.sum(probabilities * responsibilities[component], axis=2)
                mean_bad[start:stop, component] = np.mean(conditional_bad, axis=0)

        cumulative = np.cumsum(mean_probabilities, axis=1)
        cumulative /= cumulative[:, -1, None]
        quantiles = np.empty((n_systems, 3), dtype=np.float64)
        for index, target in enumerate((0.16, 0.50, 0.84)):
            quantiles[:, index] = np.array(
                [np.interp(target, cumulative[row], self.z_grid) for row in range(n_systems)]
            )
        result = MetallicityPosteriorGrid(
            row_indices=self.row_indices.copy(),
            z_grid=self.z_grid.copy(),
            probabilities=mean_probabilities,
            z_quantiles=quantiles,
            bad_probabilities=mean_bad,
            metadata={} if metadata is None else dict(metadata),
        )
        result.validate()
        return result


@dataclass(frozen=True)
class DynamicsLikelihoodLookup:
    """Per-system dynamics likelihood tabulated against sqrt(total mass).

    ``log_good[j, l]`` and ``log_bad[j, l]`` contain the accurately integrated
    likelihood for system ``j`` at ``sqrt_mtot_grid[l]``.  The outlier shape is
    fixed when the table is constructed; only its mixture fraction remains a
    free parameter in the MLR fit.
    """

    row_indices: np.ndarray
    sqrt_mtot_grid: np.ndarray
    log_good: np.ndarray
    log_bad: np.ndarray
    metadata: Mapping[str, object]

    def validate(self) -> None:
        rows = np.asarray(self.row_indices)
        grid = np.asarray(self.sqrt_mtot_grid)
        good = np.asarray(self.log_good)
        bad = np.asarray(self.log_bad)
        if rows.ndim != 1 or grid.ndim != 1:
            raise ValueError("Lookup row_indices and sqrt_mtot_grid must be one-dimensional.")
        if grid.size < 2 or np.any(~np.isfinite(grid)) or np.any(np.diff(grid) <= 0):
            raise ValueError("sqrt_mtot_grid must be finite and strictly increasing.")
        expected = (rows.size, grid.size)
        if good.shape != expected or bad.shape != expected:
            raise ValueError(
                f"Lookup tables must have shape {expected}; got {good.shape} and {bad.shape}."
            )
        if np.any(~np.isfinite(good)) or np.any(~np.isfinite(bad)):
            raise ValueError("Lookup log likelihoods must be finite.")

    def validate_for(self, *, row_indices, u, u_sigma) -> None:
        """Verify that a saved table belongs to the exact stage-two data."""
        self.validate()
        rows = np.asarray(row_indices, dtype=np.int64)
        u = np.asarray(u, dtype=np.float64)
        u_sigma = np.asarray(u_sigma, dtype=np.float64)
        if not np.array_equal(rows, np.asarray(self.row_indices, dtype=np.int64)):
            raise ValueError("Dynamics lookup row indices do not match the selected data.")
        expected_digest = array_digest(rows, u, u_sigma)
        if self.metadata.get("data_digest") != expected_digest:
            raise ValueError("Dynamics lookup u/u_sigma digest does not match the selected data.")

    def save(self, path) -> Path:
        self.validate()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Deliberately uncompressed: these smooth float32 tables are large, and
        # fast server-side save/load is more useful than modest compression.
        np.savez(
            path,
            row_indices=np.asarray(self.row_indices, dtype=np.int64),
            sqrt_mtot_grid=np.asarray(self.sqrt_mtot_grid, dtype=np.float64),
            log_good=np.asarray(self.log_good, dtype=np.float32),
            log_bad=np.asarray(self.log_bad, dtype=np.float32),
            metadata_json=np.asarray(json.dumps(dict(self.metadata), sort_keys=True)),
        )
        return path

    @classmethod
    def load(cls, path) -> "DynamicsLikelihoodLookup":
        with np.load(path, allow_pickle=False) as saved:
            result = cls(
                row_indices=np.asarray(saved["row_indices"], dtype=np.int64),
                sqrt_mtot_grid=np.asarray(saved["sqrt_mtot_grid"], dtype=np.float64),
                log_good=np.asarray(saved["log_good"], dtype=np.float32),
                log_bad=np.asarray(saved["log_bad"], dtype=np.float32),
                metadata=json.loads(str(saved["metadata_json"].item())),
            )
        result.validate()
        return result

    def interpolate_numpy(self, sqrt_mtot, *, component: str = "good") -> np.ndarray:
        """Row-wise linear interpolation in log likelihood for diagnostics."""
        values = np.asarray(sqrt_mtot, dtype=np.float64)
        if values.ndim != 2 or values.shape[0] != len(self.row_indices):
            raise ValueError("sqrt_mtot must have shape (n_systems, n_evaluations).")
        table = self.log_good if component == "good" else self.log_bad
        if component not in {"good", "bad"}:
            raise ValueError("component must be 'good' or 'bad'.")
        if np.any(values < self.sqrt_mtot_grid[0]) or np.any(values > self.sqrt_mtot_grid[-1]):
            raise ValueError("Requested sqrt(total mass) lies outside the lookup domain.")
        output = np.empty_like(values)
        for row in range(values.shape[0]):
            output[row] = np.interp(values[row], self.sqrt_mtot_grid, table[row])
        return output

    @classmethod
    def precompute(
        cls,
        *,
        row_indices,
        u,
        u_sigma,
        sqrt_mtot_min: float = 0.25,
        sqrt_mtot_max: float = 2.5,
        sqrt_mtot_points: int = 1024,
        velocity_quadrature_nodes: int = 64,
        velocity_sigma_extent: float = 10.0,
        tilde_u_max: float = 80.0,
        outlier_u0: float = 40.0,
        outlier_sigma: float = 13.0,
        system_chunk: int = 256,
        floor: float = 1e-30,
    ) -> "DynamicsLikelihoodLookup":
        """Precompute accurate Rice-convolved likelihood tables on a GPU/CPU.

        The expensive integral is expressed in observed velocity ``v``.  Each
        system gets Gauss--Legendre nodes over ``u +/- velocity_sigma_extent *
        u_sigma`` (clipped at zero).  These local nodes resolve a narrow Rice
        kernel; the old 64 nodes were instead spread over the full [0, 80]
        normalized-velocity interval and could completely miss that kernel.
        """
        rows = np.asarray(row_indices, dtype=np.int64)
        u = _as_native_f64(u, name="u", ndim=1)
        u_sigma = _as_native_f64(u_sigma, name="u_sigma", ndim=1)
        if rows.shape != u.shape or u.shape != u_sigma.shape:
            raise ValueError("row_indices, u, and u_sigma must have identical shapes.")
        if np.any(~np.isfinite(u)) or np.any(~np.isfinite(u_sigma)):
            raise ValueError("u and u_sigma must be finite.")
        if np.any(u <= 0) or np.any(u_sigma <= 0):
            raise ValueError("u and u_sigma must be strictly positive.")
        if not (0 < sqrt_mtot_min < sqrt_mtot_max):
            raise ValueError("Require 0 < sqrt_mtot_min < sqrt_mtot_max.")
        if (
            int(sqrt_mtot_points) < 2
            or int(velocity_quadrature_nodes) < 8
            or velocity_sigma_extent <= 0
            or tilde_u_max <= 0
        ):
            raise ValueError("Lookup grid sizes and spacings must be positive.")
        if outlier_sigma <= 0 or system_chunk < 1:
            raise ValueError("outlier_sigma and system_chunk must be positive.")

        jax = _configure_jax_gpu_fallback()
        import jax.numpy as jnp
        from jax.scipy.special import i0e
        from jax.scipy.special import ndtr as jax_ndtr

        sqrt_grid = np.geomspace(
            float(sqrt_mtot_min), float(sqrt_mtot_max), int(sqrt_mtot_points)
        ).astype(np.float64)
        sqrt_jax = jnp.asarray(sqrt_grid, dtype=jnp.float32)
        gl_x, gl_w = np.polynomial.legendre.leggauss(int(velocity_quadrature_nodes))
        gl_x_jax = jnp.asarray(gl_x, dtype=jnp.float32)
        gl_w_jax = jnp.asarray(gl_w, dtype=jnp.float32)
        outlier_norm = jax_ndtr(float(outlier_u0) / float(outlier_sigma))

        def calculate_chunk(u_chunk, sigma_chunk):
            lower = jnp.maximum(0.0, u_chunk - float(velocity_sigma_extent) * sigma_chunk)
            upper = u_chunk + float(velocity_sigma_extent) * sigma_chunk
            midpoint = 0.5 * (lower + upper)
            half_width = 0.5 * (upper - lower)
            velocity = midpoint[:, None] + half_width[:, None] * gl_x_jax[None, :]
            quadrature_weight = half_width[:, None] * gl_w_jax[None, :]
            observed = u_chunk[:, None]
            sigma = sigma_chunk[:, None]
            sigma_sq = sigma**2
            argument = observed * velocity / sigma_sq
            log_rice = (
                jnp.log(observed)
                - jnp.log(sigma_sq)
                - (observed**2 + velocity**2) / (2.0 * sigma_sq)
                + jnp.log(i0e(argument) + 1e-30)
                + jnp.abs(argument)
            )
            kernel = jnp.exp(log_rice) * quadrature_weight

            tilde_u = velocity[:, :, None] / sqrt_jax[None, None, :]
            tilde_for_eval = jnp.clip(tilde_u, 1e-12, float(tilde_u_max))
            log_good_basis = (
                jnp.log(5.434e-3)
                + jnp.log(tilde_for_eval)
                - 2.544e-3 * tilde_for_eval**2
                - jnp.exp((tilde_for_eval - 35.67) / 3.100)
                - jnp.log(sqrt_jax[None, None, :])
            )
            good_basis = jnp.where(
                (tilde_u > 0.0) & (tilde_u <= float(tilde_u_max)),
                jnp.exp(log_good_basis),
                0.0,
            )
            log_bad_basis = (
                -0.5 * ((tilde_for_eval - float(outlier_u0)) / float(outlier_sigma)) ** 2
                - jnp.log(float(outlier_sigma) * jnp.sqrt(2.0 * jnp.pi))
                - jnp.log(outlier_norm)
                - jnp.log(sqrt_jax[None, None, :])
            )
            bad_basis = jnp.where(
                (tilde_u >= 0.0) & (tilde_u <= float(tilde_u_max)),
                jnp.exp(log_bad_basis),
                0.0,
            )
            return (
                jnp.sum(kernel[:, :, None] * good_basis, axis=1),
                jnp.sum(kernel[:, :, None] * bad_basis, axis=1),
            )

        calculate_chunk = jax.jit(calculate_chunk)
        n_systems = rows.size
        log_good = np.empty((n_systems, sqrt_grid.size), dtype=np.float32)
        log_bad = np.empty_like(log_good)
        devices = ", ".join(str(device) for device in jax.devices())
        print(
            "Precomputing dynamics lookup: "
            f"{n_systems} systems x {sqrt_grid.size} mass nodes, "
            f"{int(velocity_quadrature_nodes)} local velocity nodes on {devices}."
        )
        for start in range(0, n_systems, int(system_chunk)):
            stop = min(start + int(system_chunk), n_systems)
            count = stop - start
            u_chunk = np.empty(int(system_chunk), dtype=np.float32)
            sigma_chunk = np.empty(int(system_chunk), dtype=np.float32)
            u_chunk[:count] = u[start:stop]
            sigma_chunk[:count] = u_sigma[start:stop]
            if count < int(system_chunk):
                u_chunk[count:] = u[start]
                sigma_chunk[count:] = u_sigma[start]
            good_chunk, bad_chunk = calculate_chunk(
                jnp.asarray(u_chunk), jnp.asarray(sigma_chunk)
            )
            good_chunk = np.asarray(jax.device_get(good_chunk))[:count]
            bad_chunk = np.asarray(jax.device_get(bad_chunk))[:count]
            log_good[start:stop] = np.log(np.maximum(good_chunk, float(floor)))
            log_bad[start:stop] = np.log(np.maximum(bad_chunk, float(floor)))
            print(f"  lookup systems {stop}/{n_systems}", flush=True)

        metadata = {
            "model": "dynamics_likelihood_lookup_local_gl_v2",
            "data_digest": array_digest(rows, u, u_sigma),
            "sqrt_mtot_min": float(sqrt_mtot_min),
            "sqrt_mtot_max": float(sqrt_mtot_max),
            "sqrt_mtot_points": int(sqrt_mtot_points),
            "velocity_quadrature_nodes": int(velocity_quadrature_nodes),
            "velocity_sigma_extent": float(velocity_sigma_extent),
            "tilde_u_max": float(tilde_u_max),
            "outlier_u0": float(outlier_u0),
            "outlier_sigma": float(outlier_sigma),
            "floor": float(floor),
        }
        result = cls(
            row_indices=rows,
            sqrt_mtot_grid=sqrt_grid,
            log_good=log_good,
            log_bad=log_bad,
            metadata=metadata,
        )
        result.validate()
        return result


class ThreeKnotMetallicityMLR:
    """Stage-two PARSEC-relative MLR with two three-knot correction curves."""

    def __init__(
        self,
        mass_surface: IsochroneMassSurfaceModel,
        *,
        mg_knots: Sequence[float] = DEFAULT_MG_KNOTS,
    ):
        self.mass_surface = mass_surface
        self.mg_knots = _as_native_f64(mg_knots, name="mg_knots", ndim=1)
        if self.mg_knots.shape != (3,) or np.any(np.diff(self.mg_knots) <= 0):
            raise ValueError("mg_knots must contain exactly three increasing values.")
        self.sampler = None
        self.posterior_samples = None
        self._data_set = False

    def correction_log10(self, absg, mh, f0_knots, fz_knots) -> np.ndarray:
        absg = np.asarray(absg, dtype=np.float64)
        mh = np.asarray(mh, dtype=np.float64)
        absg, mh = np.broadcast_arrays(absg, mh)
        f0 = np.interp(absg, self.mg_knots, np.asarray(f0_knots, dtype=float))
        fz = np.interp(absg, self.mg_knots, np.asarray(fz_knots, dtype=float))
        return f0 + mh * fz

    def mass_from_absg_mh(self, absg, mh, f0_knots, fz_knots) -> np.ndarray:
        baseline = self.mass_surface.mass_from_absg_mh(absg, mh)
        return baseline * np.power(10.0, self.correction_log10(absg, mh, f0_knots, fz_knots))

    def set_data(
        self,
        *,
        row_indices,
        u,
        u_sigma,
        absg,
        metallicity_grid: MetallicityPosteriorGrid,
        dynamics_lookup: DynamicsLikelihoodLookup,
    ) -> None:
        metallicity_grid.validate()
        self.row_indices = np.asarray(row_indices, dtype=np.int64)
        if not np.array_equal(self.row_indices, metallicity_grid.row_indices):
            raise ValueError("Stage-two row indices do not match the metallicity posterior file.")
        self.u = _as_native_f64(u, name="u", ndim=1)
        self.u_sigma = _as_native_f64(u_sigma, name="u_sigma", ndim=1)
        self.absg = _as_native_f64(absg, name="absg", ndim=2)
        n = self.row_indices.size
        if self.u.shape != (n,) or self.u_sigma.shape != (n,) or self.absg.shape != (n, 2):
            raise ValueError("Stage-two arrays have inconsistent shapes.")
        if np.any(~np.isfinite(np.column_stack([self.u, self.u_sigma, self.absg]))):
            raise ValueError("Stage-two arrays must contain only finite values.")
        if np.any(self.u <= 0) or np.any(self.u_sigma <= 0):
            raise ValueError("u and u_sigma must be strictly positive.")
        dynamics_lookup.validate_for(
            row_indices=self.row_indices, u=self.u, u_sigma=self.u_sigma
        )
        self.dynamics_lookup = dynamics_lookup
        self.sqrt_mtot_grid = np.asarray(
            dynamics_lookup.sqrt_mtot_grid, dtype=np.float64
        )
        self.log_good_lookup = np.asarray(dynamics_lookup.log_good, dtype=np.float32)
        self.log_bad_lookup = np.asarray(dynamics_lookup.log_bad, dtype=np.float32)
        self.z_grid = np.asarray(metallicity_grid.z_grid, dtype=np.float64)
        self.z_probabilities = np.asarray(metallicity_grid.probabilities, dtype=np.float64)
        if self.z_probabilities.shape != (n, self.z_grid.size):
            raise ValueError("Metallicity probability array has the wrong shape.")
        self._data_set = True

    def _build_numpyro_model(self):
        import jax.numpy as jnp
        from jax.scipy.special import logsumexp as jax_logsumexp
        import numpyro
        import numpyro.distributions as dist

        mg_knots = jnp.asarray(self.mg_knots)
        z_grid = jnp.asarray(self.z_grid)
        sqrt_grid = jnp.asarray(self.sqrt_mtot_grid)

        def interpolate_lookup(sqrt_mtot, log_table):
            """Interpolate each system's table at every metallicity node."""
            upper = jnp.searchsorted(sqrt_grid, sqrt_mtot, side="right")
            upper = jnp.clip(upper, 1, sqrt_grid.size - 1)
            lower = upper - 1
            y_lower = jnp.take_along_axis(log_table, lower, axis=1)
            y_upper = jnp.take_along_axis(log_table, upper, axis=1)
            x_lower = sqrt_grid[lower]
            x_upper = sqrt_grid[upper]
            weight = (sqrt_mtot - x_lower) / (x_upper - x_lower)
            interpolated = y_lower + weight * (y_upper - y_lower)
            in_domain = (sqrt_mtot >= sqrt_grid[0]) & (sqrt_mtot <= sqrt_grid[-1])
            return jnp.where(in_domain, interpolated, -jnp.inf)

        def model(absg, z_probabilities, log_good_lookup, log_bad_lookup):
            f0_knots = numpyro.sample("f0_knots", dist.Normal(0.0, 0.10).expand([3]))
            fz_knots = numpyro.sample("fz_knots", dist.Normal(0.0, 0.15).expand([3]))
            numpyro.factor(
                "f0_smoothness",
                dist.Normal(0.0, 0.05).log_prob(f0_knots[0] - 2.0 * f0_knots[1] + f0_knots[2]),
            )
            numpyro.factor(
                "fz_smoothness",
                dist.Normal(0.0, 0.05).log_prob(fz_knots[0] - 2.0 * fz_knots[1] + fz_knots[2]),
            )
            f_outlier = numpyro.sample("f_outlier", dist.Beta(3.0, 12.0))

            absg1 = absg[:, 0, None]
            absg2 = absg[:, 1, None]
            mh = z_grid[None, :]
            f0_1 = jnp.interp(absg1, mg_knots, f0_knots)
            f0_2 = jnp.interp(absg2, mg_knots, f0_knots)
            fz_1 = jnp.interp(absg1, mg_knots, fz_knots)
            fz_2 = jnp.interp(absg2, mg_knots, fz_knots)
            m1 = self.mass_surface.mass_from_absg_mh_jax(absg1, mh) * jnp.power(
                10.0, f0_1 + mh * fz_1
            )
            m2 = self.mass_surface.mass_from_absg_mh_jax(absg2, mh) * jnp.power(
                10.0, f0_2 + mh * fz_2
            )
            sqrt_mtot = jnp.sqrt(jnp.maximum(m1 + m2, 1e-12))
            log_good = interpolate_lookup(sqrt_mtot, log_good_lookup)
            log_bad = interpolate_lookup(sqrt_mtot, log_bad_lookup)
            log_conditional = jnp.logaddexp(
                jnp.log1p(-f_outlier) + log_good,
                jnp.log(f_outlier) + log_bad,
            )
            log_likelihood = jax_logsumexp(
                jnp.log(jnp.maximum(z_probabilities, 1e-30)) + log_conditional,
                axis=1,
            )
            numpyro.factor("dynamics", jnp.sum(log_likelihood))

            anchor_f0 = jnp.interp(jnp.asarray(4.67), mg_knots, f0_knots)
            anchor_baseline = self.mass_surface.mass_from_absg_mh_jax(
                jnp.asarray(4.67), jnp.asarray(0.0)
            )
            anchor_mass = anchor_baseline * jnp.power(10.0, anchor_f0)
            numpyro.deterministic("anchor_mass", anchor_mass)
            numpyro.factor(
                "solar_anchor", dist.Normal(1.0, 0.01).log_prob(anchor_mass)
            )

        return model

    def run_mcmc(
        self,
        *,
        num_warmup: int = 1000,
        num_samples: int = 1000,
        num_chains: int = 4,
        seed: int = 43,
        progress_bar: bool = True,
        **nuts_kwargs,
    ):
        if not self._data_set:
            raise ValueError("Call set_data() before run_mcmc().")
        jax = _configure_jax_gpu_fallback()
        import jax.numpy as jnp
        from numpyro.infer import MCMC, NUTS

        model = self._build_numpyro_model()
        mcmc = MCMC(
            NUTS(model, **nuts_kwargs),
            num_warmup=int(num_warmup),
            num_samples=int(num_samples),
            num_chains=int(num_chains),
            progress_bar=bool(progress_bar),
        )
        mcmc.run(
            jax.random.PRNGKey(int(seed)),
            jnp.asarray(self.absg),
            jnp.asarray(self.z_probabilities),
            jnp.asarray(self.log_good_lookup),
            jnp.asarray(self.log_bad_lookup),
            extra_fields=("diverging", "energy", "potential_energy", "num_steps", "accept_prob"),
        )
        self.sampler = mcmc
        self.posterior_samples = {
            name: np.asarray(values) for name, values in mcmc.get_samples().items()
        }
        return mcmc

    def correction_grid(
        self,
        *,
        absg_grid: Optional[Sequence[float]] = None,
        z_grid: Optional[Sequence[float]] = None,
        max_draws: int = 512,
    ) -> Dict[str, np.ndarray]:
        if self.posterior_samples is None:
            raise ValueError("Run run_mcmc() before correction_grid().")
        if absg_grid is None:
            absg_grid = np.linspace(self.mg_knots[0], self.mg_knots[-1], 201)
        if z_grid is None:
            z_grid = np.array([-1.0, -0.5, 0.0, 0.3, 0.6])
        absg_grid = np.asarray(absg_grid, dtype=float)
        z_grid = np.asarray(z_grid, dtype=float)
        n_available = self.posterior_samples["f0_knots"].shape[0]
        count = min(max_draws, n_available)
        # Even spacing covers every chain after NumPyro flattens grouped draws;
        # taking [:count] silently selected only chain 0 in long multi-chain runs.
        draw_indices = np.linspace(0, n_available - 1, count, dtype=np.int64)
        f0_samples = self.posterior_samples["f0_knots"][draw_indices]
        fz_samples = self.posterior_samples["fz_knots"][draw_indices]
        correction = np.empty((count, z_grid.size, absg_grid.size), dtype=np.float64)
        mass = np.empty_like(correction)
        baseline = self.mass_surface.mass_from_absg_mh(
            absg_grid[None, :], z_grid[:, None]
        )
        for draw in range(count):
            delta = self.correction_log10(
                absg_grid[None, :], z_grid[:, None], f0_samples[draw], fz_samples[draw]
            )
            correction[draw] = 100.0 * (np.power(10.0, delta) - 1.0)
            mass[draw] = baseline * np.power(10.0, delta)
        percentiles = np.array([16.0, 50.0, 84.0])
        correction_pct = np.percentile(correction, percentiles, axis=0)
        mass_pct = np.percentile(mass, percentiles, axis=0)
        monotone_cells = np.diff(mass, axis=2) > 0.0
        return {
            "absg_grid": absg_grid,
            "z_grid": z_grid,
            "percentiles": percentiles,
            "correction_percent": correction_pct,
            "mass": mass_pct,
            "monotone_cell_violation_fraction": np.asarray(np.mean(monotone_cells)),
            "monotone_draw_violation_fraction": np.asarray(
                np.mean(np.any(monotone_cells, axis=(1, 2)))
            ),
        }


def grouped_mcmc_payload(sampler) -> Dict[str, np.ndarray]:
    """Return chain-aware posterior and NUTS diagnostic arrays for NPZ output."""
    posterior = sampler.get_samples(group_by_chain=True)
    diagnostics = sampler.get_extra_fields(group_by_chain=True)
    payload = {f"posterior__{key}": np.asarray(value) for key, value in posterior.items()}
    payload.update({f"sample_stats__{key}": np.asarray(value) for key, value in diagnostics.items()})
    if "num_steps" in diagnostics:
        steps = np.asarray(diagnostics["num_steps"])
        payload["sample_stats__tree_depth"] = np.ceil(np.log2(steps + 1)).astype(np.int16)
    return payload


def simulate_mock_dataset(
    color_surface: IsochroneColorSurfaceModel,
    mass_surface: IsochroneMassSurfaceModel,
    *,
    n_systems: int = 300,
    seed: int = 123,
) -> Tuple[Dict[str, np.ndarray], Dict[str, object]]:
    """Generate a compact end-to-end recovery data set with known parameters."""
    rng = np.random.default_rng(seed)
    n = int(n_systems)
    z_true = np.clip(rng.normal(-0.3, 0.35, n), -1.0, 0.6)
    absg1 = rng.uniform(4.0, 10.5, n)
    absg2 = np.clip(absg1 + rng.uniform(0.3, 2.5, n), 4.0, 13.3)
    absg = np.column_stack([absg1, absg2])
    z_offset, z_scale, z_mag_slope = 0.10, 0.90, -0.12
    z_floor, bad_fraction, cmd_scatter = 0.10, 0.15, 0.035
    z_sigma = rng.uniform(0.05, 0.18, size=(n, 2))
    z_mean = z_offset + z_scale * z_true[:, None] + z_mag_slope * (absg - 8.5) / 5.0
    z_observed = z_mean + rng.standard_t(4, size=(n, 2)) * np.sqrt(z_sigma**2 + z_floor**2)
    bad = rng.random((n, 2)) < bad_fraction
    z_observed[bad] = -0.4 + 0.7 * rng.standard_t(2, size=np.sum(bad))
    color_sigma = rng.uniform(0.002, 0.015, size=(n, 2))
    color_mean = color_surface.color_from_absg_mh(absg, z_true[:, None])
    color = color_mean + rng.standard_t(4, size=(n, 2)) * np.sqrt(
        color_sigma**2 + cmd_scatter**2
    )
    f0_knots = np.array([0.015, 0.0, -0.015])
    fz_knots = np.array([-0.06, 0.02, 0.08])
    mlr = ThreeKnotMetallicityMLR(mass_surface)
    m1 = mlr.mass_from_absg_mh(absg1, z_true, f0_knots, fz_knots)
    m2 = mlr.mass_from_absg_mh(absg2, z_true, f0_knots, fz_knots)
    tilde_grid = np.linspace(0.01, 80.0, 20000)
    density = 5.434e-3 * tilde_grid * np.exp(
        -(2.544e-3 * tilde_grid**2 + np.exp((tilde_grid - 35.67) / 3.100))
    )
    density /= np.sum(density)
    tilde_u = rng.choice(tilde_grid, size=n, p=density)
    u_true = np.sqrt(m1 + m2) * tilde_u
    u_sigma = np.maximum(0.5, 0.08 * u_true)
    u = np.sqrt((u_true + rng.normal(0.0, u_sigma)) ** 2 + rng.normal(0.0, u_sigma) ** 2)
    data = {
        "row_indices": np.arange(n, dtype=np.int64),
        "absg": absg,
        "feh_observed": z_observed,
        "feh_sigma": z_sigma,
        "color_observed": color,
        "color_sigma": color_sigma,
        "u": u,
        "u_sigma": u_sigma,
        "z_true": z_true,
    }
    truth = {
        "z_offset": z_offset,
        "z_scale": z_scale,
        "z_mag_slope": z_mag_slope,
        "z_extra_scatter": z_floor,
        "bad_fraction": bad_fraction,
        "cmd_scatter": cmd_scatter,
        "f0_knots": f0_knots.tolist(),
        "fz_knots": fz_knots.tolist(),
    }
    return data, truth
