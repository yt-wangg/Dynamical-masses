"""
Test core functionality of the Bayesian binary mass estimation package.
"""

import numpy as np
import pytest
from astropy.table import Table

# Import the package modules
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from binary_masses.core import PU8Sampler, NonParametricPosteriorPlotter


class TestPU8Sampler:
    """Test the PU8Sampler class."""

    def test_initialization(self):
        """Test sampler initialization."""
        sampler = PU8Sampler()
        assert sampler.A == 4.95e-3
        assert sampler.B == 2.24e-3
        assert sampler.C == 3.85
        assert sampler.u0 == 36.09

    def test_custom_parameters(self):
        """Test initialization with custom parameters."""
        sampler = PU8Sampler(A=1.0, B=2.0, C=3.0, u0=40.0)
        assert sampler.A == 1.0
        assert sampler.B == 2.0
        assert sampler.C == 3.0
        assert sampler.u0 == 40.0

    def test_func_pu_8(self):
        """Test the p(u) function."""
        sampler = PU8Sampler()
        tilde_u = np.array([1.0, 10.0, 20.0])
        pdf_values = sampler.func_pu_8(tilde_u)

        assert len(pdf_values) == 3
        assert all(pdf_values >= 0)  # PDF should be non-negative
        assert all(np.isfinite(pdf_values))  # PDF should be finite

    def test_find_sampling_bounds(self):
        """Test finding sampling bounds."""
        sampler = PU8Sampler()
        u_min, u_max, M = sampler.find_sampling_bounds()

        assert u_min < u_max
        assert M > 0
        assert isinstance(u_min, float)
        assert isinstance(u_max, float)
        assert isinstance(M, float)

    def test_rejection_sample(self):
        """Test rejection sampling."""
        sampler = PU8Sampler()
        samples = sampler.rejection_sample(n_samples=100)

        assert len(samples) == 100
        assert all(samples >= 0)  # Should be positive
        assert all(np.isfinite(samples))

    def test_inverse_transform_sample(self):
        """Test inverse transform sampling."""
        sampler = PU8Sampler()
        samples = sampler.inverse_transform_sample(n_samples=100)

        assert len(samples) == 100
        assert all(samples >= 0)  # Should be positive
        assert all(np.isfinite(samples))

    def test_sample_method(self):
        """Test main sample method."""
        sampler = PU8Sampler()

        # Test rejection method
        samples1 = sampler.sample(n_samples=50, method='rejection')
        assert len(samples1) == 50

        # Test inverse_transform method
        samples2 = sampler.sample(n_samples=50, method='inverse_transform')
        assert len(samples2) == 50

        # Test invalid method
        with pytest.raises(ValueError):
            sampler.sample(n_samples=10, method='invalid')


class TestNonParametricPosteriorPlotter:
    """Test the NonParametricPosteriorPlotter class."""

    def test_initialization(self):
        """Test plotter initialization."""
        plotter = NonParametricPosteriorPlotter()

        assert plotter.n_bins == 10
        assert plotter.absg_min == 4.0
        assert plotter.absg_max == 12.0
        assert plotter.uncertainty_model == 'rice'
        assert len(plotter.absg_bins) == 10

    def test_custom_initialization(self):
        """Test initialization with custom parameters."""
        plotter = NonParametricPosteriorPlotter(
            n_bins=15,
            absg_min=2.0,
            absg_max=15.0,
            uncertainty_model='gaussian'
        )

        assert plotter.n_bins == 15
        assert plotter.absg_min == 2.0
        assert plotter.absg_max == 15.0
        assert plotter.uncertainty_model == 'gaussian'
        assert len(plotter.absg_bins) == 15

    def test_invalid_uncertainty_model(self):
        """Test initialization with invalid uncertainty model."""
        with pytest.raises(ValueError):
            NonParametricPosteriorPlotter(uncertainty_model='invalid')

    def test_func_pu_8(self):
        """Test the p(u) function."""
        plotter = NonParametricPosteriorPlotter()
        tilde_u = np.array([1.0, 10.0, 20.0])
        pdf_values = plotter.func_pu_8(tilde_u)

        assert len(pdf_values) == 3
        assert all(pdf_values >= 0)
        assert all(np.isfinite(pdf_values))

    def test_absg_to_mass(self):
        """Test magnitude to mass conversion."""
        plotter = NonParametricPosteriorPlotter()
        mass_bins = np.linspace(0.1, 2.0, 10)
        absg_values = np.array([5.0, 8.0, 11.0])

        masses = plotter.absg_to_mass(mass_bins, absg_values)

        assert len(masses) == 3
        assert all(masses >= 0.1)  # Should be within mass range
        assert all(masses <= 2.0)
        assert all(np.isfinite(masses))

    def test_rice_distribution(self):
        """Test Rice distribution."""
        plotter = NonParametricPosteriorPlotter()
        u_obs = np.array([1.0, 5.0, 10.0])
        u_true = np.array([2.0, 4.0, 8.0])
        sigma = np.array([0.5, 1.0, 2.0])

        rice_pdf = plotter.rice_distribution(u_obs, u_true, sigma)

        assert len(rice_pdf) == 3
        assert all(rice_pdf >= 0)
        assert all(np.isfinite(rice_pdf))

    def test_gaussian(self):
        """Test Gaussian distribution."""
        plotter = NonParametricPosteriorPlotter()
        x = np.array([0.0, 1.0, 2.0])
        mu = np.array([0.5, 1.5, 1.5])
        sigma = np.array([1.0, 1.0, 2.0])

        gauss_pdf = plotter.gaussian(x, mu, sigma)

        assert len(gauss_pdf) == 3
        assert all(gauss_pdf >= 0)
        assert all(np.isfinite(gauss_pdf))

    def test_set_data(self):
        """Test setting data."""
        plotter = NonParametricPosteriorPlotter()

        # Create test data
        u_values = np.array([1.0, 2.0, 3.0])
        u_sigma_values = np.array([0.1, 0.2, 0.3])
        absg1_values = np.array([5.0, 6.0, 7.0])
        absg2_values = np.array([6.0, 7.0, 8.0])

        # Test with uncertainties
        plotter.set_data(
            u_values=u_values,
            u_sigma_values=u_sigma_values,
            absg1_values=absg1_values,
            absg2_values=absg2_values,
            gamma=10.0
        )

        assert np.allclose(plotter.u_values, u_values)
        assert np.allclose(plotter.u_sigma_values, u_sigma_values)
        assert np.allclose(plotter.absg1_values, absg1_values)
        assert np.allclose(plotter.absg2_values, absg2_values)
        assert plotter.gamma == 10.0

        # Test without uncertainties
        plotter.set_data(
            u_values=u_values,
            u_sigma_values=None,
            absg1_values=absg1_values,
            absg2_values=absg2_values
        )

        assert plotter.u_sigma_values is None
        assert np.allclose(plotter.norm_factor, np.ones(3))


def create_mock_data(n_samples=100):
    """Create simple mock data for testing."""
    np.random.seed(42)

    # Simple mock data
    data = Table()
    data['u'] = np.random.uniform(5, 50, n_samples)
    data['u_sigma'] = np.random.uniform(0.5, 5, n_samples)
    data['absg1'] = np.random.uniform(4, 12, n_samples)
    data['absg2'] = np.random.uniform(4, 12, n_samples)

    return data


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])