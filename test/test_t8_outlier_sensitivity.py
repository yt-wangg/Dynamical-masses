"""Small numerical checks for the server outlier-shape experiment; no MCMC."""

from dataclasses import replace
from contextlib import redirect_stderr
import io
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from scipy.integrate import quad
from scipy.special import i0e, ndtr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "examples"))

from binary_masses.hierarchical_metallicity import (
    DynamicsLikelihoodLookup,
    assess_rice_lookup_convergence,
)
from run_hierarchical_metallicity_test import _require_t8_lookup_metadata, parse_args


class OutlierSensitivityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tables = {}
        for mu, sigma in ((40, 13), (30, 13), (40, 20), (30, 20)):
            common = dict(
                row_indices=np.array([3, 7]), u=np.array([18.0, 55.0]),
                u_sigma=np.array([0.7, 2.0]), sqrt_mtot_min=0.5,
                sqrt_mtot_max=2.0, velocity_sigma_extent=14,
                velocity_quadrature_nodes=128, outlier_u0=mu,
                outlier_sigma=sigma,
            )
            convergence = assess_rice_lookup_convergence(
                **common, sample_systems=4, scale_points=4,
            )
            table = DynamicsLikelihoodLookup.precompute(
                **common, sqrt_mtot_points=4, system_chunk=2,
                outlier_sensitivity=(mu, sigma) != (40, 13),
            )
            cls.tables[mu, sigma] = replace(
                table, metadata={**table.metadata, "convergence_check": convergence},
            )

    def test_all_shapes_match_independent_adaptive_integral(self):
        # Deliberately spell out the outlier density rather than calling the
        # production reference routine: catches stale default mu/sigma there.
        baseline_good = self.tables[40, 13].log_good
        for (mu, sigma), table in self.tables.items():
            with self.subTest(mu=mu, sigma=sigma):
                self.assertTrue(table.metadata["convergence_check"]["passed"])
                np.testing.assert_array_equal(table.log_good, baseline_good)
                norm = ndtr((80 - mu) / sigma) - ndtr(-mu / sigma)
                for row, (observed, error) in enumerate(((18., .7), (55., 2.))):
                    for column, scale in enumerate(table.sqrt_mtot_grid):
                        def density(velocity):
                            rice = observed / error**2 * np.exp(
                                -(observed - velocity)**2 / (2 * error**2)
                            ) * i0e(observed * velocity / error**2)
                            outlier = np.exp(-.5 * ((velocity / scale - mu) / sigma)**2)
                            return rice * outlier / (scale * sigma * np.sqrt(2*np.pi) * norm)
                        upper = 80 * scale
                        points = np.clip(observed + error*np.array([-10, -3, 0, 3, 10]), 0, upper)
                        reference = quad(density, 0, upper, points=np.unique(points),
                                         epsabs=1e-13, epsrel=1e-10)[0]
                        self.assertLess(abs(float(table.log_bad[row, column])-np.log(reference)), 1e-3)
        self.assertFalse(np.allclose(self.tables[40, 13].log_bad, self.tables[30, 20].log_bad))

    def test_save_load_and_requested_shape_mismatch(self):
        args = SimpleNamespace(
            quick=True, lookup_mass_points=4, lookup_velocity_nodes=128,
            lookup_sqrt_mass_min=.5, lookup_sqrt_mass_max=2., lookup_sigma_extent=14,
            outlier_sensitivity=True, outlier_u0=30., outlier_sigma=20.,
        )
        with tempfile.TemporaryDirectory() as directory:
            table = self.tables[30, 20]
            path = table.save(Path(directory) / "lookup.npz")
            loaded = DynamicsLikelihoodLookup.load(path)
            _require_t8_lookup_metadata(loaded, args=args)
            args.outlier_u0 = 40.
            with self.assertRaises(ValueError):
                _require_t8_lookup_metadata(loaded, args=args)

    def test_convergence_record_must_match_shape(self):
        table = self.tables[30, 20]
        wrong = replace(table, metadata={
            **table.metadata,
            "convergence_check": self.tables[40, 13].metadata["convergence_check"],
        })
        with self.assertRaises(ValueError):
            wrong.validate_t8_schema()
        missing_shape = dict(table.metadata["convergence_check"])
        missing_shape.pop("outlier_u0")
        with self.assertRaises(ValueError):
            replace(table, metadata={**table.metadata, "convergence_check": missing_shape}).validate_t8_schema()
        with self.assertRaises(ValueError):
            replace(table, metadata={**table.metadata, "experimental_outlier_sensitivity": False}).validate_t8_schema()

    def test_cli_opt_in_and_baseline_output_protection(self):
        with tempfile.TemporaryDirectory() as directory:
            baseline = Path(directory) / "baseline"
            baseline.mkdir()
            posterior = baseline / "latent_metallicity_weights_t8.npz"
            common = ["runner", "--stage", "lookup", "--outlier-u0", "30",
                      "--outlier-sigma", "20", "--metallicity-posterior", str(posterior)]
            with patch.object(sys, "argv", common), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    parse_args()
            with patch.object(sys, "argv", common + ["--outlier-sensitivity", "--output-dir", str(baseline)]), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    parse_args()
            with patch.object(sys, "argv", common + ["--outlier-sensitivity", "--output-dir", str(Path(directory) / "experiment")]):
                args = parse_args()
                self.assertEqual((args.outlier_u0, args.outlier_sigma), (30, 20))


if __name__ == "__main__":
    unittest.main()
