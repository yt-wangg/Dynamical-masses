"""Tests for the independent minimal hierarchical metallicity model."""

from __future__ import annotations

import os
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from scipy.integrate import quad
from scipy.special import i0e


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from binary_masses.differencepoly_feh import IsochroneMassSurfaceModel
from binary_masses.hierarchical_metallicity import (
    DynamicsLikelihoodLookup,
    HierarchicalMetallicityCalibrator,
    IsochroneColorSurfaceModel,
    MetallicityPosteriorGrid,
    ThreeKnotMetallicityMLR,
    array_digest,
    color_uncertainty_from_flux_snr,
    marginalize_dynamics,
    normalize_log_weights,
    simulate_mock_dataset,
    trapezoid_weights,
)


def simple_color_surface():
    absg = np.array([3.5, 8.5, 13.5])
    mh = np.array([-1.0, -0.2, 0.6])
    color = 0.2 * absg[None, :] + 0.4 * mh[:, None]
    return IsochroneColorSurfaceModel(absg, mh, color)


def simple_mass_surface():
    absg = np.array([3.5, 8.5, 13.5])
    mh = np.array([-1.0, -0.2, 0.6])
    base = np.array([1.2, 0.55, 0.15])
    mass = base[None, :] * (1.0 + 0.1 * mh[:, None])
    return IsochroneMassSurfaceModel(absg, mh, mass, mass_min=0.05)


def set_small_calibrator(surface):
    calibrator = HierarchicalMetallicityCalibrator(
        surface, z_grid=np.linspace(-1.0, 0.6, 9), population_components=3
    )
    absg = np.array([[5.0, 8.0], [7.0, 10.0], [9.0, 12.0]])
    true_z = np.array([-0.7, -0.2, 0.3])
    color = surface.color_from_absg_mh(absg, true_z[:, None])
    calibrator.set_data(
        row_indices=np.array([4, 8, 10]),
        absg=absg,
        feh_observed=np.column_stack([true_z + 0.02, true_z - 0.03]),
        feh_sigma=np.full((3, 2), 0.1),
        color_observed=color,
        color_sigma=np.full((3, 2), 0.01),
    )
    return calibrator


def simple_dynamics_lookup(rows, u, u_sigma):
    rows = np.asarray(rows, dtype=np.int64)
    u = np.asarray(u, dtype=np.float64)
    u_sigma = np.asarray(u_sigma, dtype=np.float64)
    grid = np.array([0.25, 0.75, 1.25, 2.5])
    row_term = np.arange(rows.size, dtype=float)[:, None] * 0.01
    log_good = -0.5 * (grid[None, :] - 1.2) ** 2 + row_term
    log_bad = -0.1 * (grid[None, :] - 1.0) ** 2 + row_term
    return DynamicsLikelihoodLookup(
        row_indices=rows,
        sqrt_mtot_grid=grid,
        log_good=log_good,
        log_bad=log_bad,
        metadata={"data_digest": array_digest(rows, u, u_sigma)},
    )


class HierarchicalMetallicityTests(unittest.TestCase):
    def test_trapezoid_weights_integrate_constant(self):
        grid = np.linspace(-1.0, 0.6, 81)
        weights = trapezoid_weights(grid)
        self.assertTrue(np.all(weights > 0))
        self.assertAlmostEqual(np.sum(weights), 1.6)

    def test_color_uncertainty_propagation(self):
        result = color_uncertainty_from_flux_snr(np.array([100.0]), np.array([200.0]))
        expected = (2.5 / np.log(10.0)) * np.sqrt(100.0**-2 + 200.0**-2)
        self.assertAlmostEqual(result[0], expected)
        with self.assertRaisesRegex(ValueError, "strictly positive"):
            color_uncertainty_from_flux_snr([0.0], [100.0])

    def test_color_surface_numpy_jax_and_boundary(self):
        surface = simple_color_surface()
        absg = np.array([2.0, 6.0, 15.0])
        mh = np.array([-2.0, 0.0, 1.0])
        numpy_value = surface.color_from_absg_mh(absg, mh)
        jax_value = np.asarray(surface.color_from_absg_mh_jax(absg, mh))
        self.assertTrue(np.allclose(numpy_value, jax_value, atol=1e-6))
        expected = 0.2 * np.clip(absg, 3.5, 13.5) + 0.4 * np.clip(mh, -1.0, 0.6)
        self.assertTrue(np.allclose(numpy_value, expected))

    def test_calibration_log_joint_is_finite_and_normalizable(self):
        calibrator = set_small_calibrator(simple_color_surface())
        log_joint = calibrator.log_joint_numpy(
            {
                "z_offset": 0.0,
                "z_scale": 1.0,
                "z_mag_slope": 0.0,
                "z_extra_scatter": 0.1,
                "bad_fraction": 0.15,
                "cmd_scatter": 0.03,
                "population_weights": np.ones(3) / 3.0,
            }
        )
        self.assertEqual(log_joint.shape, (3, 9))
        self.assertTrue(np.all(np.isfinite(log_joint)))
        normalized = normalize_log_weights(log_joint)
        self.assertTrue(np.allclose(normalized.sum(axis=1), 1.0))

    def test_stage_one_posterior_postprocessing(self):
        calibrator = set_small_calibrator(simple_color_surface())
        draws = 4
        calibrator.posterior_samples = {
            "z_offset": np.zeros(draws),
            "z_scale": np.ones(draws),
            "z_mag_slope": np.zeros(draws),
            "z_extra_scatter": np.full(draws, 0.1),
            "bad_fraction": np.full(draws, 0.15),
            "cmd_scatter": np.full(draws, 0.03),
            "population_weights": np.full((draws, 3), 1.0 / 3.0),
        }
        posterior = calibrator.posterior_grid(max_draws=draws, system_chunk=2)
        posterior.validate()
        self.assertEqual(posterior.probabilities.shape, (3, 9))
        self.assertTrue(np.all((posterior.bad_probabilities >= 0) & (posterior.bad_probabilities <= 1)))

    def test_posterior_grid_round_trip(self):
        probabilities = np.array([[0.25, 0.75], [0.6, 0.4]])
        posterior = MetallicityPosteriorGrid(
            row_indices=np.array([2, 9]),
            z_grid=np.array([-1.0, 0.6]),
            probabilities=probabilities,
            z_quantiles=np.array([[-0.8, 0.1, 0.5], [-0.9, -0.4, 0.3]]),
            bad_probabilities=np.array([[0.1, 0.2], [0.3, 0.4]]),
            metadata={"feh_columns": ["a", "b"]},
        )
        with tempfile.TemporaryDirectory() as directory:
            path = posterior.save(Path(directory) / "posterior.npz")
            loaded = MetallicityPosteriorGrid.load(path)
        self.assertTrue(np.array_equal(loaded.row_indices, posterior.row_indices))
        self.assertTrue(np.allclose(loaded.probabilities, probabilities))
        self.assertEqual(loaded.metadata, posterior.metadata)

    def test_dynamics_marginalization_is_dynamics_only(self):
        log_conditional = np.log(np.array([[0.2, 0.8], [0.7, 0.3]]))
        probabilities = np.array([[0.4, 0.6], [0.25, 0.75]])
        result = marginalize_dynamics(log_conditional, probabilities)
        expected = np.log(np.sum(np.exp(log_conditional) * probabilities, axis=1))
        self.assertTrue(np.allclose(result, expected))
        with self.assertRaises(TypeError):
            marginalize_dynamics(log_conditional, probabilities, np.ones_like(probabilities))

    def test_dynamics_lookup_round_trip_interpolation_and_digest(self):
        rows = np.array([2, 9])
        u = np.array([10.0, 12.0])
        u_sigma = np.array([1.0, 1.5])
        lookup = simple_dynamics_lookup(rows, u, u_sigma)
        lookup.validate_for(row_indices=rows, u=u, u_sigma=u_sigma)
        query = np.array([[0.5, 1.0], [0.8, 2.0]])
        expected = lookup.interpolate_numpy(query, component="good")
        with tempfile.TemporaryDirectory() as directory:
            loaded = DynamicsLikelihoodLookup.load(
                lookup.save(Path(directory) / "dynamics_lookup.npz")
            )
        self.assertTrue(np.allclose(loaded.interpolate_numpy(query), expected, atol=1e-6))
        with self.assertRaisesRegex(ValueError, "digest"):
            loaded.validate_for(row_indices=rows, u=u + 0.01, u_sigma=u_sigma)
        with self.assertRaisesRegex(ValueError, "outside"):
            loaded.interpolate_numpy(np.array([[0.1], [1.0]]))

    def test_local_velocity_lookup_precomputation_is_finite(self):
        lookup = DynamicsLikelihoodLookup.precompute(
            row_indices=np.array([1, 4]),
            u=np.array([10.0, 18.0]),
            u_sigma=np.array([0.2, 2.0]),
            sqrt_mtot_min=0.5,
            sqrt_mtot_max=2.0,
            sqrt_mtot_points=8,
            velocity_quadrature_nodes=16,
            system_chunk=2,
        )
        self.assertEqual(lookup.log_good.shape, (2, 8))
        self.assertTrue(np.all(np.isfinite(lookup.log_good)))
        self.assertTrue(np.all(np.isfinite(lookup.log_bad)))
        self.assertEqual(lookup.metadata["velocity_quadrature_nodes"], 16)

    def test_local_velocity_lookup_matches_adaptive_integral_for_narrow_kernel(self):
        u, u_sigma = 10.0, 0.2
        lookup = DynamicsLikelihoodLookup.precompute(
            row_indices=np.array([7]),
            u=np.array([u]),
            u_sigma=np.array([u_sigma]),
            sqrt_mtot_min=0.5,
            sqrt_mtot_max=2.0,
            sqrt_mtot_points=65,
            velocity_quadrature_nodes=64,
            system_chunk=1,
        )
        indices = np.array([8, 31, 57])
        sqrt_mass = lookup.sqrt_mtot_grid[indices]
        tabulated = np.exp(lookup.log_good[0, indices])

        def rice_pdf(observed, true_velocity, sigma):
            argument = observed * true_velocity / sigma**2
            return np.exp(
                np.log(observed / sigma**2)
                - (observed**2 + true_velocity**2) / (2.0 * sigma**2)
                + np.log(i0e(argument))
                + abs(argument)
            )

        def physical_pdf(tilde_u):
            return 5.434e-3 * tilde_u * np.exp(
                -(2.544e-3 * tilde_u**2 + np.exp((tilde_u - 35.67) / 3.100))
            )

        reference = np.array(
            [
                quad(
                    lambda velocity: rice_pdf(u, velocity, u_sigma)
                    * physical_pdf(velocity / scale)
                    / scale,
                    max(0.0, u - 12.0 * u_sigma),
                    u + 12.0 * u_sigma,
                    epsabs=1e-12,
                    epsrel=1e-10,
                )[0]
                for scale in sqrt_mass
            ]
        )
        self.assertTrue(np.allclose(tabulated, reference, rtol=2e-4, atol=1e-8))

    def test_three_knot_correction_and_row_validation(self):
        mlr = ThreeKnotMetallicityMLR(simple_mass_surface())
        f0 = np.array([0.0, 0.1, 0.2])
        fz = np.array([-0.1, 0.0, 0.1])
        correction = mlr.correction_log10(
            np.array([3.5, 8.5, 13.5]), np.array([0.5, 0.5, 0.5]), f0, fz
        )
        self.assertTrue(np.allclose(correction, f0 + 0.5 * fz))
        posterior = MetallicityPosteriorGrid(
            row_indices=np.array([1, 3]),
            z_grid=np.array([-1.0, 0.6]),
            probabilities=np.full((2, 2), 0.5),
            z_quantiles=np.zeros((2, 3)),
            bad_probabilities=np.zeros((2, 2)),
            metadata={},
        )
        with self.assertRaisesRegex(ValueError, "row indices"):
            mlr.set_data(
                row_indices=np.array([1, 4]),
                u=np.array([10.0, 12.0]),
                u_sigma=np.array([1.0, 1.0]),
                absg=np.array([[5.0, 7.0], [6.0, 8.0]]),
                metallicity_grid=posterior,
                dynamics_lookup=simple_dynamics_lookup(
                    np.array([1, 4]), np.array([10.0, 12.0]), np.array([1.0, 1.0])
                ),
            )

    def test_stage_two_correction_postprocessing(self):
        mlr = ThreeKnotMetallicityMLR(simple_mass_surface())
        mlr.posterior_samples = {
            "f0_knots": np.array([[0.0, 0.0, 0.0], [0.01, 0.0, -0.01]]),
            "fz_knots": np.array([[0.0, 0.0, 0.0], [-0.05, 0.0, 0.05]]),
        }
        grid = mlr.correction_grid(
            absg_grid=np.linspace(3.5, 13.5, 11), z_grid=[-1.0, 0.0, 0.6]
        )
        self.assertEqual(grid["correction_percent"].shape, (3, 3, 11))
        self.assertEqual(grid["mass"].shape, (3, 3, 11))
        self.assertTrue(np.all(np.isfinite(grid["correction_percent"])))

    def test_mock_generator_shapes(self):
        data, truth = simulate_mock_dataset(
            simple_color_surface(), simple_mass_surface(), n_systems=24, seed=7
        )
        self.assertEqual(data["absg"].shape, (24, 2))
        self.assertEqual(data["feh_observed"].shape, (24, 2))
        self.assertEqual(data["color_observed"].shape, (24, 2))
        self.assertEqual(data["u"].shape, (24,))
        self.assertTrue(np.all(data["u"] > 0))
        self.assertEqual(np.sign(truth["z_mag_slope"]), -1)

    def test_numpyro_models_execute_without_sampling(self):
        import jax.numpy as jnp
        from numpyro import handlers

        calibrator = set_small_calibrator(simple_color_surface())
        calibration_trace = handlers.trace(
            handlers.seed(calibrator._build_numpyro_model(), rng_seed=4)
        ).get_trace(
            jnp.asarray(calibrator.absg),
            jnp.asarray(calibrator.feh_observed),
            jnp.asarray(calibrator.feh_sigma),
            jnp.asarray(calibrator.color_observed),
            jnp.asarray(calibrator.color_sigma),
        )
        self.assertIn("observations", calibration_trace)

        z_grid = np.linspace(-1.0, 0.6, 9)
        posterior = MetallicityPosteriorGrid(
            row_indices=np.array([4, 8, 10]),
            z_grid=z_grid,
            probabilities=np.full((3, z_grid.size), 1.0 / z_grid.size),
            z_quantiles=np.zeros((3, 3)),
            bad_probabilities=np.zeros((3, 2)),
            metadata={},
        )
        mlr = ThreeKnotMetallicityMLR(simple_mass_surface())
        u = np.array([10.0, 12.0, 14.0])
        u_sigma = np.array([1.0, 1.2, 1.4])
        mlr.set_data(
            row_indices=posterior.row_indices,
            u=u,
            u_sigma=u_sigma,
            absg=calibrator.absg,
            metallicity_grid=posterior,
            dynamics_lookup=simple_dynamics_lookup(posterior.row_indices, u, u_sigma),
        )
        mlr_trace = handlers.trace(handlers.seed(mlr._build_numpyro_model(), rng_seed=5)).get_trace(
            jnp.asarray(mlr.absg),
            jnp.asarray(mlr.z_probabilities),
            jnp.asarray(mlr.log_good_lookup),
            jnp.asarray(mlr.log_bad_lookup),
        )
        self.assertIn("dynamics", mlr_trace)
        self.assertIn("solar_anchor", mlr_trace)


if __name__ == "__main__":
    unittest.main()
