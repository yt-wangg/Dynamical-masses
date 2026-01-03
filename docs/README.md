# Bayesian Binary Mass Estimation

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![DOI](https://img.shields.io/badge/DOI-10.5281/zenodo.xxxxxx-orange.svg)](https://doi.org/10.5281/zenodo.xxxxxx)

A Python package for estimating dynamical masses of binary stars using Bayesian inference with Rice distribution uncertainties. This package implements non-parametric fitting of mass-luminosity relations using NumPyro with NUTS sampling, supporting multi-metallicity analysis.

## Features

- **Non-parametric Bayesian inference** for mass-luminosity relations
- **Rice distribution** uncertainty model (naturally handles positive-definite measurements)
- **Multi-metallicity fitting** with automatic binning
- **GPU acceleration** through JAX (CUDA/Metal support)
- **Comprehensive visualization** with corner plots and fitting results
- **Modular design** for easy extension and customization

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/bayesian-binary-masses.git
cd bayesian-binary-masses

# Install the package and dependencies
pip install -r requirements.txt
pip install -e .
```

### Basic Usage

Download the example data: https://keeper.mpdl.mpg.de/f/1587c6531bf84155ad9a/?dl=1

```python
import numpy as np
from astropy.table import Table
from binary_masses import MultiMetallicityFitter
from binary_masses import load_interpolated_isochrone_data, IsochroneMassModel

# Load your binary star data
data = Table.read('your_binary_data.fits')

# Initialize the fitter
fitter = MultiMetallicityFitter(
    n_absg_bins=10,           # Number of magnitude bins
    absg_min=4.0,            # Minimum absolute G magnitude
    absg_max=12.0,           # Maximum absolute G magnitude
    uncertainty_model='rice'  # Use Rice distribution for uncertainties
)

# Bin data by metallicity
binned_data = fitter.bin_data_by_metallicity(
    data,
    feh_column='feh',        # Metallicity column name
    n_feh_bins=3,            # Number of metallicity bins
    equal_frequency=False    # Use equal-width binning
)

# (Optional) Load the pre-resampled isochrone grid for plotting / reference
iso_data = load_interpolated_isochrone_data('data/interpolated_mass_data')

# (Optional) Build a 1D isochrone mass model at a given [M/H] for DifferencePolyMLR
iso_model = IsochroneMassModel.from_interpolated_mass_data(mh=0.0, data_dir='data/interpolated_mass_data')

# Fit all metallicity bins
fitter.fit_all_bins(
    binned_data,
    gamma=np.inf,            # No regularization
    num_warmup=1000,         # Warmup samples
    num_samples=2000,        # Posterior samples
    num_chains=4,            # Parallel chains
    seed=42
)

# Plot and save results
fitter.plot_all_results(output_dir='results/')
fitter.plot_comparison(output_path='mass_luminosity_comparison.png')
fitter.save_all_samples(output_dir='results/', prefix='posterior_samples')
```

## Scientific Background

This package estimates stellar masses in wide binary systems using the **u-parameter**:

```
u = v × √s
```

where:
- `v` is the relative velocity between binary components
- `s` is the projected separation

The method relies on the empirical distribution of the normalized parameter:

```
ũ = u/√m_tot
```

where `m_tot` is the total binary mass. The likelihood model implements:

```
L = (1/√m_tot) × ∫ p(ũ) × P(u_obs | ũ, σ) dũ
```

where the integration accounts for measurement uncertainties using either:
- **Rice distribution** (recommended for positive-definite measurements)
- **Gaussian distribution** (traditional approach)

## Key Classes

### `NonParametricPosteriorPlotter`

Core class for single-metallicity fitting:

```python
from binary_masses import NonParametricPosteriorPlotter

# Initialize
plotter = NonParametricPosteriorPlotter(
    n_bins=10,
    absg_min=4.0,
    absg_max=12.0,
    uncertainty_model='rice'
)

# Set data
plotter.set_data(
    u_values=data['u'],
    u_sigma_values=data['u_sigma'],  # Optional
    absg1_values=data['absg1'],
    absg2_values=data['absg2'],
    gamma=10.0  # Regularization strength
)

# Run inference
mcmc = plotter.run_numpyro(
    num_warmup=1000,
    num_samples=2000,
    num_chains=4
)
```

### `MultiMetallicityFitter`

Convenient wrapper for multiple metallicity bins:

```python
from binary_masses import MultiMetallicityFitter

# Initialize with 10 magnitude bins
multi_fitter = MultiMetallicityFitter(
    n_absg_bins=10,
    absg_min=3.0,
    absg_max=14.0
)

# Automatically bin by metallicity and fit all
binned_data = multi_fitter.bin_data_by_metallicity(data, n_feh_bins=5)
multi_fitter.fit_all_bins(binned_data)
```

### `PU8Sampler`

Utility for sampling from the p(ũ) distribution:

```python
from binary_masses import PU8Sampler

sampler = PU8Sampler()
samples = sampler.sample(n_samples=1000, method='rejection')
```

## Data Format

Your data should contain the following columns:

| Column | Description | Type | Required |
|--------|-------------|------|----------|
| `u` | Velocity × √separation parameter | float | ✓ |
| `u_sigma` | Uncertainty in u (optional) | float | ✗ |
| `absg1` | Primary absolute G magnitude | float | ✓ |
| `absg2` | Secondary absolute G magnitude | float | ✓ |
| `feh` | Metallicity [Fe/H] | float | ✗ (for multi-metallicity) |

Example:

```python
import numpy as np
from astropy.table import Table

# Create example data
data = Table()
data['u'] = np.random.uniform(10, 100, 1000)
data['u_sigma'] = np.random.uniform(1, 5, 1000)
data['absg1'] = np.random.uniform(4, 12, 1000)
data['absg2'] = np.random.uniform(4, 12, 1000)
data['feh'] = np.random.uniform(-1, 0.5, 1000)
```

## Performance Tips

### GPU Acceleration
The package automatically detects and uses GPUs through JAX:

```python
import jax
print(f"Available devices: {jax.devices()}")
```

### Sampling Optimization
- **Faster integration**: Increase `int_du` (e.g., 0.05-0.1) or reduce `int_umax` (e.g., 50-60)
- **Better adaptation**: Set `use_dense_mass=True` (slower warmup, faster sampling)
- **Parallel chains**: Use multiple chains (automatically parallelized by JAX)

### Memory Management
For large datasets:
```python
# Use fewer chains to reduce memory usage
fitter.fit_all_bins(binned_data, num_chains=2)

# Reduce integration grid
plotter.run_numpyro(int_du=0.1, int_umax=50)
```

## Examples

The `examples/` directory contains comprehensive tutorials:

1. **`basic_fitting.py`** - Single-metallicity fitting
2. **`multi_metallicity.py`** - Multi-metallicity analysis
3. **`mock_data_generation.py`** - Generate synthetic binary populations
4. **`uncertainty_models.py`** - Compare Rice vs Gaussian uncertainties

Run examples:
```bash
cd examples
python basic_fitting.py
```

## Citation

If you use this package in your research, please cite:

```bibtex
@software{bayesian_binary_masses,
  title={Bayesian Binary Mass Estimation: Non-parametric mass-luminosity relations with Rice distribution uncertainties},
  author={Wang, Yutong and collaborators},
  year={2024},
  url={https://github.com/yourusername/bayesian-binary-masses},
  doi={10.5281/zenodo.xxxxxx}
}
```

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Setup

```bash
# Clone and install in development mode
git clone https://github.com/yourusername/bayesian-binary-masses.git
cd bayesian-binary-masses
pip install -e ".[dev]"

# Run tests
pytest tests/

# Code formatting
black src/
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built with [NumPyro](https://github.com/pyro-ppl/numpyro) for Bayesian inference
- Uses [JAX](https://github.com/google/jax) for high-performance computing
- Visualization with [Matplotlib](https://matplotlib.org/) and [Corner](https://corner.readthedocs.io/)

## Questions?

- Open an [issue](https://github.com/yourusername/bayesian-binary-masses/issues) for bug reports or feature requests
- Check the [documentation](https://bayesian-binary-masses.readthedocs.io/) for detailed API reference
- Contact: yutong.wang@yourinstitution.edu
