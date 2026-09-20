"""Tests for the independent minimal hierarchical metallicity model."""

from __future__ import annotations

import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
from scipy.integrate import quad
from scipy.special import i0e


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "examples"))

from binary_masses.differencepoly_feh import IsochroneMassSurfaceModel
from binary_masses.hierarchical_metallicity import (
    DynamicsLikelihoodLookup,
    HierarchicalMetallicityCalibrator,
    IsochroneColorSurfaceModel,
    JCAPS_BIAS_MG_KNOTS,
    JCAPS_BIAS_VALUES,
    JCAPS_EXTRA_SCALE,
    JCAPS_STUDENT_DF,
    MetallicityPosteriorGrid,
    MonotoneTensorSplineMLR,
    ThreeKnotMetallicityMLR,
    _bspline_basis_numpy,
    array_digest,
    assess_rice_lookup_convergence,
    color_uncertainty_from_flux_snr,
    jcaps_magnitude_bias,
    marginalize_dynamics,
    normalize_log_weights,
    rice_good_raw,
    simulate_mock_dataset,
    trapezoid_weights,
    rice_outlier_normalization,
)
from run_hierarchical_metallicity_test import (
    INPUT_COLUMNS,
    T8_MODEL_ID,
    T8_WORKFLOW_ID,
    _require_t8_posterior_metadata,
    _require_t8_lookup_metadata,
)
from run_t8_holdout_validation import (
    evaluate_heldout,
    fit_parsec_outlier_posterior,
    posterior_predictive_log_likelihood,
    split_system_positions,
    subset_inputs,
)


def simple_color_surface():
    absg = np.array([3.5, 8.5, 13.5])
    mh = np.array([-1.0, -0.2, 0.6])
    color = 0.2 * absg[None, :] + 0.4 * mh[:, None]
    return IsochroneColorSurfaceModel(absg, mh, color)


def simple_color_temperature_surface():
    absg = np.array([3.5, 8.5, 13.5])
    mh = np.array([-1.0, -0.2, 0.6])
    color = 0.2 * absg[None, :] + 0.4 * mh[:, None]
    teff = np.broadcast_to(np.array([6200.0, 5000.0, 3800.0]), color.shape).copy()
    return IsochroneColorSurfaceModel(absg, mh, color, teff_grid=teff)


def simple_mass_surface():
    absg = np.array([3.5, 8.5, 13.5])
    mh = np.array([-1.0, -0.2, 0.6])
    base = np.array([1.2, 0.55, 0.15])
    mass = base[None, :] * (1.0 + 0.1 * mh[:, None])
    return IsochroneMassSurfaceModel(absg, mh, mass, mass_min=0.05)


def set_small_calibrator(surface):
    calibrator = HierarchicalMetallicityCalibrator(
        surface, z_grid=np.linspace(-1.0, 0.6, 9), population_components=3,
        cmd_min_teff=None,
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
    grid = np.geomspace(0.25, 2.5, 4)
    row_term = np.arange(rows.size, dtype=float)[:, None] * 0.01
    log_good = -0.5 * (grid[None, :] - 1.2) ** 2 + row_term
    log_bad = -0.1 * (grid[None, :] - 1.0) ** 2 + row_term
    return DynamicsLikelihoodLookup(
        row_indices=rows,
        sqrt_mtot_grid=grid,
        log_good=log_good,
        log_bad=log_bad,
        metadata={
            "model": "dynamics_likelihood_lookup_t8",
            "schema": "t8c-rice-lookup-v1",
            "data_digest": array_digest(rows, u, u_sigma),
            "sqrt_mtot_min": 0.25, "sqrt_mtot_max": 2.5, "sqrt_mtot_points": 4,
            "scale_grid": "geometric",
            "velocity_quadrature_nodes": 64, "velocity_sigma_extent": 10.0,
            "tilde_u_max": 80.0, "outlier_u0": 40.0, "outlier_sigma": 13.0,
            "outlier_support": [0.0, 80.0],
            "outlier_normalization": "Phi((U-mu)/sigma)-Phi(-mu/sigma)",
            "outlier_normalization_value": rice_outlier_normalization(),
            "C_out": rice_outlier_normalization(),
            "C_good": 1.0,
            "normalization_version": "finite_support_exact_t8_1",
            "good_basis_normalized": True,
            "good_basis_support": [0.0, 80.0],
            "good_basis_constant": 5.434e-3,
            "good_basis_quadratic": 2.544e-3,
            "good_basis_turnover": 35.67,
            "good_basis_scale": 3.100,
            "good_basis_support_integral": 1.0,
            "jacobian": "1/s",
            "scale_definition": "s=sqrt(Mtot/Msun)",
            "integration_coordinate": "v=s*tilde_u",
            "velocity_units": "km s^-1 sqrt(AU)",
            "interpolation": "linear_log_likelihood_in_raw_s",
            "row_order": "row_indices order",
            "mixture_weight_semantics": "physical outlier fraction (T8.1 exact component normalization)",
            "floor": 1e-30,
            "migration_applied": False,
            "convergence_check": {
                "passed": True,
                "source": "test fixture",
                "source_data_digest": array_digest(rows, u, u_sigma),
                "floor": 1e-30,
                "sample_systems_requested": 4,
                "sample_system_count": int(rows.size),
                "scale_points_requested": int(grid.size),
                "selected_row_indices": rows.tolist(),
                "selected_row_indices_digest": array_digest(rows),
                "scale_grid": grid.tolist(),
                "adopted": {"velocity_sigma_extent": 10.0, "quadrature_nodes": 64},
            },
        },
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

    def test_t6d_fixed_observation_parameters_and_bias_anchor(self):
        self.assertTrue(
            np.allclose(jcaps_magnitude_bias(JCAPS_BIAS_MG_KNOTS), JCAPS_BIAS_VALUES)
        )
        self.assertAlmostEqual(float(jcaps_magnitude_bias(8.5)), 0.0)
        self.assertAlmostEqual(JCAPS_EXTRA_SCALE, 0.04646502063846824)
        self.assertAlmostEqual(JCAPS_STUDENT_DF, 2.5236494464371377)

    def test_t8_delta_z_sign_matches_observed_minus_latent_metallicity(self):
        surface = simple_color_surface()
        calibrator = HierarchicalMetallicityCalibrator(
            surface, z_grid=np.array([-0.2, 0.0, 0.2]),
            population_components=3, cmd_min_teff=None,
        )
        calibrator.set_data(
            row_indices=np.array([1]),
            absg=np.array([[8.5, 8.5]]),
            feh_observed=np.array([[0.2, 0.2]]),
            feh_sigma=np.array([[0.03, 0.03]]),
            color_observed=np.array([[1.7, 1.7]]),
            color_sigma=np.array([[0.02, 0.02]]),
        )
        common = {"s_C": 0.03, "population_weights": np.ones(3) / 3.0}
        positive = calibrator.log_joint_numpy({**common, "delta_z": 0.2})[0, 1]
        negative = calibrator.log_joint_numpy({**common, "delta_z": -0.2})[0, 1]
        self.assertGreater(positive, negative)

    def test_cmd_temperature_cut_is_fixed_and_excludes_cool_color_likelihood(self):
        surface = simple_color_temperature_surface()
        self.assertGreater(float(surface.teff_from_absg_mh(5.0, 0.0)), 4800.0)
        self.assertLess(float(surface.teff_from_absg_mh(12.0, 0.0)), 4800.0)
        calibrator = HierarchicalMetallicityCalibrator(
            surface,
            z_grid=np.linspace(-1.0, 0.6, 9),
            population_components=3,
            cmd_min_teff=4730.0,
        )
        base_color = surface.color_from_absg_mh(np.array([[5.0, 12.0]]), 0.0)
        calibrator.set_data(
            row_indices=np.array([1]),
            absg=np.array([[5.0, 12.0]]),
            feh_observed=np.array([[0.0, 0.0]]),
            feh_sigma=np.array([[0.1, 0.1]]),
            color_observed=base_color,
            color_sigma=np.array([[0.01, 0.01]]),
        )
        self.assertTrue(np.array_equal(calibrator.cmd_mask, [[True, False]]))
        params = {
            "z_offset": 0.0,
            "cmd_scatter": 0.03,
            "population_weights": np.ones(3) / 3.0,
        }
        reference = calibrator.log_joint_numpy(params)
        calibrator.color_observed[0, 1] += 100.0
        self.assertTrue(np.allclose(calibrator.log_joint_numpy(params), reference))
        calibrator.color_observed[0, 0] += 1.0
        self.assertFalse(np.allclose(calibrator.log_joint_numpy(params), reference))

    def test_t8_default_cmd_threshold_is_4000_k(self):
        calibrator = HierarchicalMetallicityCalibrator(simple_color_temperature_surface())
        self.assertEqual(calibrator.cmd_min_teff, 4000.0)

    def test_t8_cmd_threshold_requires_temperature_grid(self):
        calibrator = HierarchicalMetallicityCalibrator(simple_color_surface())
        with self.assertRaisesRegex(ValueError, "temperature grid"):
            calibrator.set_data(
                row_indices=np.array([1]),
                absg=np.array([[5.0, 7.0]]),
                feh_observed=np.array([[0.0, 0.0]]),
                feh_sigma=np.array([[0.1, 0.1]]),
                color_observed=np.array([[1.0, 1.2]]),
                color_sigma=np.array([[0.01, 0.01]]),
            )

    def test_t8_table_input_never_substitutes_corrected_metallicity(self):
        calibrator = HierarchicalMetallicityCalibrator(simple_color_surface())
        corrected_only = {
            "jc_m_h_fit_cal_1": np.zeros(2),
            "jc_m_h_fit_cal_2": np.zeros(2),
            "jc_sigma_m_h_cal_1": np.ones(2),
            "jc_sigma_m_h_cal_2": np.ones(2),
        }
        with self.assertRaisesRegex(ValueError, "raw JCAPS columns"):
            calibrator.set_data_from_table(
                table=corrected_only,
                row_indices=np.arange(2),
                absg=np.full((2, 2), 6.0),
                color_observed=np.full((2, 2), 1.0),
                color_sigma=np.full((2, 2), 0.02),
            )

    def test_calibration_log_joint_is_finite_and_normalizable(self):
        calibrator = set_small_calibrator(simple_color_surface())
        log_joint = calibrator.log_joint_numpy(
            {
                "z_offset": 0.0,
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
            "cmd_scatter": np.full(draws, 0.03),
            "population_weights": np.full((draws, 3), 1.0 / 3.0),
        }
        posterior = calibrator.posterior_grid(max_draws=draws, system_chunk=2)
        posterior.validate()
        self.assertEqual(posterior.probabilities.shape, (3, 9))
        self.assertTrue(np.allclose(posterior.probabilities.sum(axis=1), 1.0))
        self.assertTrue(np.all(posterior.bad_probabilities == 0.0))
        self.assertEqual(posterior.metadata["selected_draw_count"], draws)
        self.assertEqual(posterior.metadata["selected_draw_seed"], 0)
        self.assertEqual(posterior.metadata["selected_draw_indices"], list(range(draws)))
        self.assertEqual(
            posterior.metadata["selected_draw_indices_digest"],
            array_digest(np.arange(draws, dtype=np.int64)),
        )

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
        invalid_metadata = {**dict(loaded.metadata), "migration_applied": True}
        migrated = DynamicsLikelihoodLookup(
            loaded.row_indices, loaded.sqrt_mtot_grid, loaded.log_good,
            loaded.log_bad, invalid_metadata,
        )
        with self.assertRaisesRegex(ValueError, "Migrated"):
            migrated.validate_t8_schema()

    def test_formal_profile_rejects_quick_lookup(self):
        rows = np.array([2, 9])
        u = np.array([10.0, 12.0])
        u_sigma = np.array([1.0, 1.5])
        lookup = simple_dynamics_lookup(rows, u, u_sigma)
        args = SimpleNamespace(
            quick=True,
            lookup_mass_points=4,
            lookup_velocity_nodes=64,
            lookup_sqrt_mass_min=0.25,
            lookup_sqrt_mass_max=2.5,
            lookup_sigma_extent=10.0,
        )
        _require_t8_lookup_metadata(lookup, args=args)
        args.quick = False
        with self.assertRaisesRegex(ValueError, "formal execution profile"):
            _require_t8_lookup_metadata(lookup, args=args)

    def test_t8_posterior_rejects_stale_input_digest(self):
        rows = np.array([2, 7])
        arrays = {
            "row_indices": rows,
            "feh_observed": np.array([[-0.2, -0.1], [0.1, 0.05]]),
            "feh_sigma": np.full((2, 2), 0.1),
            "absg": np.array([[5.0, 7.0], [6.0, 8.0]]),
            "color_observed": np.array([[1.0, 1.3], [1.1, 1.4]]),
            "color_sigma": np.full((2, 2), 0.02),
        }
        grid = np.linspace(-1.0, 0.6, 81)
        metadata = {
            "schema": "t8a-metallicity-posterior-v1",
            "workflow": T8_WORKFLOW_ID,
            "model": T8_MODEL_ID,
            "mock": True,
            "feh_columns": INPUT_COLUMNS["feh_observed"],
            "feh_sigma_columns": INPUT_COLUMNS["feh_sigma"],
            "cmd_min_parsec_teff_k": 4000.0,
            "posterior_grid_nodes": 81,
            "metallicity_grid": grid.tolist(),
            "metallicity_grid_weights": trapezoid_weights(grid).tolist(),
            "posterior_average_draw_count": 2,
            "selected_draw_count": 2,
            "posterior_draw_indices": [0, 1],
            "posterior_draw_indices_digest": array_digest(np.array([0, 1], dtype=np.int64)),
            "forbidden_metallicity_columns": [
                "jc_m_h_fit_1", "jc_m_h_fit_2", "jc_m_h_fit_cal_1",
                "jc_m_h_fit_cal_2", "jc_sigma_m_h_cal_1", "jc_sigma_m_h_cal_2",
            ],
            "input_data_digest": array_digest(
                arrays["row_indices"], arrays["feh_observed"], arrays["feh_sigma"],
                arrays["absg"], arrays["color_observed"], arrays["color_sigma"],
            ),
        }
        posterior = MetallicityPosteriorGrid(
            row_indices=rows,
            z_grid=grid,
            probabilities=np.full((2, 81), 1.0 / 81.0),
            z_quantiles=np.zeros((2, 3)),
            bad_probabilities=np.zeros((2, 2)),
            metadata=metadata,
        )
        _require_t8_posterior_metadata(posterior, mock=True, arrays=arrays)
        stale = {key: np.array(value, copy=True) for key, value in arrays.items()}
        stale["feh_observed"][0, 0] += 1e-5
        with self.assertRaisesRegex(ValueError, "digest"):
            _require_t8_posterior_metadata(posterior, mock=True, arrays=stale)

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
            return rice_good_raw(tilde_u)

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

    def test_rice_convergence_compares_coordinates_and_reports_floor_state(self):
        result = assess_rice_lookup_convergence(
            row_indices=np.array([3]),
            u=np.array([10.0]),
            u_sigma=np.array([0.5]),
            sqrt_mtot_min=0.5,
            sqrt_mtot_max=2.0,
            sample_systems=1,
            scale_points=2,
            velocity_quadrature_nodes=64,
        )
        self.assertTrue(result["passed"])
        for component in ("good", "outlier"):
            diagnostics = result["components"][component]
            self.assertLessEqual(diagnostics["adaptive_coordinate_max_abs_log"], 1e-3)
            self.assertEqual(diagnostics["unsafe_floor_count"], 0)
            self.assertIn("safe_floor_locations", diagnostics)
        floored = assess_rice_lookup_convergence(
            row_indices=np.array([3]),
            u=np.array([10.0]),
            u_sigma=np.array([0.5]),
            sqrt_mtot_min=0.5,
            sqrt_mtot_max=2.0,
            sample_systems=1,
            scale_points=2,
            velocity_quadrature_nodes=64,
            floor=1.0,
        )
        self.assertTrue(floored["passed"])
        self.assertEqual(floored["components"]["good"]["safe_floor_count"], 2)

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
            metadata={"model": "t8_jcaps_student_t_independent_members"},
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

    def test_t8_monotone_tensor_spline_parameter_count_and_surface_order(self):
        mlr = MonotoneTensorSplineMLR(simple_mass_surface())
        self.assertEqual(mlr.n_params, 35)
        rng = np.random.default_rng(42)
        params = {
            "c0": 0.0,
            "a": rng.normal(size=3),
            "b": rng.normal(size=7),
            "r": rng.normal(size=(7, 3)),
            "log_lambda_x": 0.0,
            "log_lambda_z": 0.0,
        }
        values = mlr.g_from_raw(
            np.linspace(3.5, 13.5, 101)[:, None],
            np.linspace(-1.0, 0.6, 31)[None, :],
            params,
        )
        self.assertLessEqual(float(np.max(np.diff(values, axis=0))), 1e-10)
        self.assertGreaterEqual(float(np.min(np.diff(values, axis=1))), -1e-10)

    def _real_parsec_mass_surface(self):
        surface = IsochroneMassSurfaceModel.from_interpolated_mass_data(
            Path(__file__).resolve().parent.parent / "data" / "interpolated_mass_data",
            mass_min=0.05,
        )
        # T8's clamped knot vector ends at M_G=13.5 while the packaged
        # interpolation grid extends to 14.0, so restrict the domain exactly
        # as the production runner does.
        x_mask = surface.absg_grid_np < 13.5 - 1e-12
        x_grid = np.concatenate([surface.absg_grid_np[x_mask], [13.5]])
        mass_grid = np.column_stack(
            [surface.mass_grid_np[:, x_mask],
             np.array([surface.mass_from_absg_mh(13.5, mh) for mh in surface.mh_grid_np])]
        )
        return IsochroneMassSurfaceModel(
            x_grid, surface.mh_grid_np, mass_grid, mass_min=0.05
        )

    def test_t8_real_parsec_surface_z_monotone_with_negative_corrections(self):
        """The final MLR on real PARSEC data keeps d log M / d Z >= 0 and
        d log M / d M_G <= 0 even when the PARSEC-relative correction is
        driven far below zero."""
        mlr = MonotoneTensorSplineMLR(self._real_parsec_mass_surface())
        x = np.linspace(3.5, 13.5, 201)
        z = np.linspace(-1.0, 0.6, 81)
        rng = np.random.default_rng(20260913)
        for _ in range(3):
            params = {
                "c0": rng.normal(),
                "a": rng.normal(scale=2.0, size=3),
                "b": rng.normal(scale=2.0, size=7),
                "r": rng.normal(scale=2.0, size=(7, 3)),
                "log_lambda_x": rng.normal(),
                "log_lambda_z": rng.normal(),
            }
            log_mass = mlr.g_from_raw(x[:, None], z[None, :], params)
            self.assertLessEqual(float(np.max(np.diff(log_mass, axis=0))), 1e-10)
            self.assertGreaterEqual(float(np.min(np.diff(log_mass, axis=1))), -1e-10)
            mass = mlr.mass_from_absg_mh(x[:, None], z[None, :], params)
            self.assertLessEqual(float(np.max(np.diff(mass, axis=0))), 0.0)
            self.assertGreaterEqual(float(np.min(np.diff(mass, axis=1))), 0.0)

        # Lowering c0 by delta lowers every theta entry by exactly delta, so
        # the PARSEC-relative correction can reach arbitrarily negative
        # values; shift just far enough to sit strictly below PARSEC
        # everywhere without depending on any absolute magnitude.
        initial_params = mlr.initial_raw_parameters()
        initial_correction = mlr.correction_log10(
            x[None, :], z[:, None], initial_params
        )
        shift = 0.5 + float(np.max(initial_correction))
        params = dict(initial_params)
        params["c0"] = params["c0"] - shift
        correction = mlr.correction_log10(x[None, :], z[:, None], params)
        self.assertTrue(
            np.allclose(correction, initial_correction - shift, atol=1e-12)
        )
        self.assertLess(float(np.max(correction)), 0.0)
        log_mass = mlr.g_from_raw(x[:, None], z[None, :], params)
        self.assertLessEqual(float(np.max(np.diff(log_mass, axis=0))), 1e-10)
        self.assertGreaterEqual(float(np.min(np.diff(log_mass, axis=1))), -1e-10)

    def test_t8_bspline_boundaries_projection_and_spacing_penalties(self):
        mlr = MonotoneTensorSplineMLR(simple_mass_surface())
        for values, knots, degree in (
            (np.array([3.5, 4.0, 13.5]), mlr.knots_x, mlr.degree_x),
            (np.array([-1.0, -0.2, 0.6]), mlr.knots_z, mlr.degree_z),
        ):
            basis = _bspline_basis_numpy(values, knots, degree)
            self.assertTrue(np.all(basis >= 0.0))
            self.assertTrue(np.allclose(np.sum(basis, axis=-1), 1.0))
            self.assertEqual(float(basis[-1, -1]), 1.0)

        x = np.linspace(3.5, 13.5, 31)
        z = np.linspace(-1.0, 0.6, 17)
        residual = mlr.g_parsec_projection(x[None, :], z[:, None]) - np.log10(
            mlr.mass_surface.mass_from_absg_mh(x[None, :], z[:, None])
        )
        self.assertLess(float(np.sqrt(np.mean(residual**2))), 0.1)
        self.assertIn("projection_grid_digest", mlr.projection_metadata)

        affine = 0.02 * mlr.greville_x[:, None] + 0.03 * mlr.greville_z[None, :]
        d2x, d2z, dxz = mlr.spacing_aware_penalties(affine)
        self.assertTrue(np.allclose(d2x, 0.0, atol=1e-12))
        self.assertTrue(np.allclose(d2z, 0.0, atol=1e-12))
        self.assertTrue(np.allclose(dxz, 0.0, atol=1e-12))

        mixed = np.sin(mlr.greville_x[:, None]) * np.cos(3.0 * mlr.greville_z[None, :])
        _, _, mixed_difference = mlr.spacing_aware_penalties(mixed)
        self.assertLess(float(np.min(mixed_difference)), 0.0)
        self.assertGreater(float(np.max(mixed_difference)), 0.0)

    def test_t8_feasible_initialization_is_monotone_and_solar_anchored(self):
        mlr = MonotoneTensorSplineMLR(simple_mass_surface())
        params = mlr.initial_raw_parameters()
        for value in params.values():
            self.assertTrue(np.all(np.isfinite(value)))
        raw_normal = np.concatenate([params["a"], params["b"], params["r"].ravel()])
        self.assertLessEqual(float(np.max(np.abs(raw_normal))), 4.0 + 1e-12)
        self.assertLess(float(np.sum(raw_normal**2)), 300.0)
        theta = mlr.theta_from_raw(params)
        self.assertLessEqual(float(np.max(np.diff(theta, axis=0))), 1e-10)
        self.assertGreaterEqual(float(np.min(np.diff(theta, axis=1))), -1e-10)
        self.assertAlmostEqual(float(mlr.g_from_theta(4.67, 0.0, theta)), 0.0, places=11)

    def test_t8_numpyro_trace_contains_all_35_sampled_scalars(self):
        import jax.numpy as jnp
        from numpyro import handlers

        rows = np.array([4, 8, 10])
        z_grid = np.linspace(-1.0, 0.6, 9)
        posterior = MetallicityPosteriorGrid(
            row_indices=rows,
            z_grid=z_grid,
            probabilities=np.full((rows.size, z_grid.size), 1.0 / z_grid.size),
            z_quantiles=np.zeros((rows.size, 3)),
            bad_probabilities=np.zeros((rows.size, 2)),
            metadata={"model": "t8_jcaps_student_t_independent_members"},
        )
        u = np.array([10.0, 12.0, 14.0])
        u_sigma = np.array([1.0, 1.2, 1.4])
        mlr = MonotoneTensorSplineMLR(simple_mass_surface())
        mlr.set_data(
            row_indices=rows,
            u=u,
            u_sigma=u_sigma,
            absg=np.array([[5.0, 7.0], [6.0, 8.0], [7.0, 9.0]]),
            metallicity_grid=posterior,
            dynamics_lookup=simple_dynamics_lookup(rows, u, u_sigma),
        )
        trace = handlers.trace(handlers.seed(mlr._build_numpyro_model(), rng_seed=9)).get_trace(
            jnp.asarray(mlr.absg),
            jnp.asarray(mlr.z_probabilities),
            jnp.asarray(mlr.log_good_lookup),
            jnp.asarray(mlr.log_bad_lookup),
        )
        shapes = {
            name: np.shape(trace[name]["value"])
            for name in mlr.sample_parameter_names
        }
        self.assertEqual(shapes["c0"], ())
        self.assertEqual(shapes["a"], (3,))
        self.assertEqual(shapes["b"], (7,))
        self.assertEqual(shapes["r"], (7, 3))
        self.assertEqual(shapes["log_lambda_x"], ())
        self.assertEqual(shapes["log_lambda_z"], ())
        self.assertEqual(shapes["f_outlier"], ())
        self.assertEqual(sum(max(1, int(np.prod(shape))) for shape in shapes.values()), 35)
        self.assertIn("dynamics", trace)
        self.assertIn("solar_anchor", trace)

    def test_t8_outlier_normalization_is_finite_support(self):
        self.assertAlmostEqual(rice_outlier_normalization(), 0.9979085073395224, places=12)

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
        self.assertEqual(grid["parsec_mass"].shape, (3, 11))
        self.assertTrue(np.all(np.isfinite(grid["correction_percent"])))
        self.assertTrue(np.all(np.isfinite(grid["parsec_mass"])))

    def test_t8_holdout_split_is_reproducible_and_system_disjoint(self):
        rows = np.arange(100, 200, dtype=np.int64)
        checked = rows[[2, 18, 36, 54, 72, 90]]
        train_a, test_a = split_system_positions(
            rows, test_fraction=0.20, seed=17, convergence_rows=checked
        )
        train_b, test_b = split_system_positions(
            rows, test_fraction=0.20, seed=17, convergence_rows=checked
        )
        self.assertTrue(np.array_equal(train_a, train_b))
        self.assertTrue(np.array_equal(test_a, test_b))
        self.assertEqual(train_a.size, 80)
        self.assertEqual(test_a.size, 20)
        self.assertEqual(np.intersect1d(train_a, test_a).size, 0)
        self.assertTrue(np.any(np.isin(rows[train_a], checked)))
        self.assertTrue(np.any(np.isin(rows[test_a], checked)))

    def test_t8_holdout_subset_preserves_lookup_validation(self):
        rows = np.arange(10, 20, dtype=np.int64)
        u = np.linspace(8.0, 17.0, rows.size)
        u_sigma = np.full(rows.size, 1.0)
        z_grid = np.linspace(-1.0, 0.6, 9)
        posterior = MetallicityPosteriorGrid(
            row_indices=rows,
            z_grid=z_grid,
            probabilities=np.full((rows.size, z_grid.size), 1.0 / z_grid.size),
            z_quantiles=np.zeros((rows.size, 3)),
            bad_probabilities=np.zeros((rows.size, 2)),
            metadata={"model": T8_MODEL_ID},
        )
        arrays = {
            "row_indices": rows,
            "u": u,
            "u_sigma": u_sigma,
            "absg": np.column_stack(
                [np.linspace(4.0, 9.0, rows.size), np.linspace(4.5, 9.5, rows.size)]
            ),
        }
        lookup = simple_dynamics_lookup(rows, u, u_sigma)
        selected_arrays, selected_posterior, selected_lookup = subset_inputs(
            arrays, posterior, lookup, np.array([0, 2, 5, 8]), label="unit_test"
        )
        self.assertTrue(np.array_equal(selected_arrays["row_indices"], rows[[0, 2, 5, 8]]))
        self.assertTrue(np.array_equal(selected_posterior.row_indices, rows[[0, 2, 5, 8]]))
        selected_lookup.validate_for(
            row_indices=selected_arrays["row_indices"],
            u=selected_arrays["u"],
            u_sigma=selected_arrays["u_sigma"],
        )

    def test_t8_heldout_likelihood_is_paired_by_draw_and_system(self):
        rows = np.arange(4, dtype=np.int64)
        u = np.linspace(9.0, 12.0, rows.size)
        u_sigma = np.full(rows.size, 1.0)
        z_grid = np.linspace(-1.0, 0.6, 9)
        posterior = MetallicityPosteriorGrid(
            row_indices=rows,
            z_grid=z_grid,
            probabilities=np.full((rows.size, z_grid.size), 1.0 / z_grid.size),
            z_quantiles=np.zeros((rows.size, 3)),
            bad_probabilities=np.zeros((rows.size, 2)),
            metadata={"model": T8_MODEL_ID},
        )
        arrays = {
            "row_indices": rows,
            "u": u,
            "u_sigma": u_sigma,
            "absg": np.array([[4.0, 4.5], [5.0, 5.5], [7.0, 7.5], [9.0, 9.5]]),
        }
        lookup = simple_dynamics_lookup(rows, u, u_sigma)
        model = MonotoneTensorSplineMLR(simple_mass_surface())
        initial = model.initial_raw_parameters()
        samples = {
            name: np.repeat(np.asarray(value)[None, ...], 3, axis=0)
            for name, value in initial.items()
        }
        samples["f_outlier"] = np.full(3, 0.15)
        result = evaluate_heldout(
            model, samples, arrays, posterior, lookup, max_draws=2
        )
        self.assertEqual(result["delta_log_likelihood"].shape, (2, 4))
        self.assertTrue(np.all(np.isfinite(result["parsec_log_likelihood"])))
        self.assertTrue(np.all(np.isfinite(result["mlr_log_likelihood"])))
        predictive = posterior_predictive_log_likelihood(
            np.log(np.array([[0.2, 0.8], [0.6, 0.4]]))
        )
        self.assertTrue(np.allclose(predictive, np.log([0.4, 0.6])))
        parsec_posterior = fit_parsec_outlier_posterior(
            model.mass_surface,
            arrays,
            posterior,
            lookup,
            quadrature_nodes=16,
        )
        self.assertAlmostEqual(float(np.sum(parsec_posterior["posterior_weights"])), 1.0)
        self.assertTrue(np.all(parsec_posterior["posterior_weights"] > 0.0))

    def test_mock_generator_shapes(self):
        data, truth = simulate_mock_dataset(
            simple_color_surface(), simple_mass_surface(), n_systems=24, seed=7
        )
        self.assertEqual(data["absg"].shape, (24, 2))
        self.assertEqual(data["feh_observed"].shape, (24, 2))
        self.assertEqual(data["color_observed"].shape, (24, 2))
        self.assertEqual(data["u"].shape, (24,))
        self.assertTrue(np.all(data["u"] > 0))
        self.assertAlmostEqual(truth["extra_scale"], JCAPS_EXTRA_SCALE)
        self.assertAlmostEqual(truth["student_df"], JCAPS_STUDENT_DF)

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
        self.assertIn("delta_z", calibration_trace)
        self.assertIn("s_C", calibration_trace)
        self.assertIn("z_offset", calibration_trace)
        self.assertIn("cmd_scatter", calibration_trace)
        self.assertIn("population_weights", calibration_trace)
        for removed_parameter in (
            "z_scale",
            "z_mag_slope",
            "z_extra_scatter",
            "bad_fraction",
        ):
            self.assertNotIn(removed_parameter, calibration_trace)

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
        baseline_log_likelihood = mlr.baseline_dynamics_log_likelihood()
        self.assertEqual(baseline_log_likelihood.shape, (3,))
        self.assertTrue(np.all(np.isfinite(baseline_log_likelihood)))
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
