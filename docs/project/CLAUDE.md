# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Python package for **Bayesian dynamical mass estimation of binary stars** using non-parametric fitting with Rice distribution uncertainties. The package implements innovative statistical methods for estimating stellar masses from wide binary observations using the normalized velocity-separation parameter (ũ).

## Repository Structure

```
Dynamical-masses/
├── src/
│   └── binary_masses/
│       ├── __init__.py          # Package initialization and exports
│       ├── core.py              # Main Bayesian inference classes
│       └── sampling.py          # Binary population generation utilities
├── src/examples/                # Tutorial scripts demonstrating key features
├── tests/                       # Test suite using pytest
├── data/                        # Data files (example inputs, model outputs)
└── docs/                        # Documentation source files
```

## Core Architecture

### Key Classes and Their Roles

**`PU8Sampler`** ([core.py:17](src/binary_masses/core.py#L17))
- Implements the empirical p(ũ) distribution (func_pu_8)
- Uses rejection sampling for generating ũ samples
- Critical for likelihood calculations in Bayesian inference

**`NonParametricPosteriorPlotter`** ([core.py:150](src/binary_masses/core.py#L150))
- Main class for single-metallicity Bayesian fitting
- Integrates measurement uncertainties using Rice or Gaussian distributions
- Uses NumPyro with NUTS sampler for posterior inference
- Supports GPU acceleration through JAX

**`MultiMetallicityFitter`** ([core.py:450](src/binary_masses/core.py#L450))
- High-level interface for fitting multiple [Fe/H] bins
- Automates data binning by metallicity
- Coordinates multiple NonParametricPosteriorPlotter instances

**`BrokenPowerLawMLR`** ([core.py:650](src/binary_masses/core.py#L650))
- Parametric mass-luminosity relation model
- Provides alternative to non-parametric fitting
- Useful for comparisons and baseline models

### Scientific Foundation

The method relies on the **u-parameter**: `u = v × √s` (velocity × √separation)
- Normalized form: `ũ = u/√m_tot` follows empirical distribution p(ũ)
- Likelihood integrates over measurement uncertainties
- Rice distribution naturally handles positive-definite measurements

### Uncertainty Models

The package implements two uncertainty approaches:
- **Rice distribution** (recommended): Properly handles positive-definite measurements
- **Gaussian distribution**: Traditional approach for comparison

## Common Development Commands

### Installation and Setup
```bash
# Clone and install in development mode
git clone <repository-url>
cd Dynamical-masses
pip install -e ".[dev]"

# Install dependencies
pip install -r requirements.txt
```

### Testing
```bash
# Run all tests
pytest

# Run tests with coverage report
pytest --cov=binary_masses --cov-report=html

# Run specific test file
pytest tests/test_core.py

# Run tests for specific class
pytest tests/test_core.py::TestPU8Sampler
```

### Code Quality
```bash
# Format code with black
black src/ tests/

# Check code style with flake8
flake8 src/ tests/

# Type checking (if mypy is installed)
mypy src/binary_masses/
```

### Documentation
```bash
# Build documentation (if sphinx is set up)
cd docs/
make html

# Serve documentation locally
python -m http.server 8000 --directory _build/html
```

### Running Examples
```bash
cd src/examples/

# Basic single-metallicity fitting
python basic_fitting.py

# Multi-metallicity analysis
python multi_metallicity.py

# Mock data generation (if available)
python mock_data_generation.py
```

## Key Implementation Details

### Performance Optimization
- **GPU Support**: Automatic detection and use of JAX-compatible GPUs (CUDA/Metal)
- **Parallel Chains**: NumPyro automatically parallelizes MCMC chains
- **Integration Parameters**: `int_umax`, `int_du`, and `int_N` control numerical integration accuracy

### Memory Management
- Large datasets may require reducing `num_chains` to limit memory usage
- Integration grid size affects both accuracy and memory consumption
- Consider using `use_dense_mass=False` for very large parameter spaces

### Data Requirements
Essential columns for input data:
- `u`: velocity × √separation parameter (float)
- `absg1`, `absg2`: absolute G magnitudes (float)
- `u_sigma`: uncertainty in u (float, optional but recommended)
- `feh`: metallicity [Fe/H] (float, required for multi-metallicity)

### Model Configuration
Key parameters for fitting:
- `n_absg_bins`: number of magnitude bins (typically 8-15)
- `absg_min`, `absg_max`: magnitude range (usually 3-15)
- `gamma`: regularization strength (np.inf for no smoothing)
- `uncertainty_model`: 'rice' or 'gaussian'

## Development Workflow

### Adding New Features
1. Implement core functionality in appropriate class within `core.py`
2. Add corresponding tests in `tests/test_core.py`
3. Create example script in `src/examples/` if feature is user-facing
4. Update `__init__.py` exports if needed

### Testing Strategy
- Unit tests for individual methods and classes
- Integration tests for complete workflows
- Mock data generation for reproducible testing
- Performance tests for GPU/CPU compatibility

### Code Style
- Follow PEP 8 style guide
- Use type hints where possible
- Comprehensive docstrings following NumPy style
- Black formatting for consistent code style

## Dependencies

Core scientific packages:
- `numpy`, `scipy`: Numerical operations and statistics
- `astropy`: Astronomical data handling and tables
- `torch`: Neural network support (for legacy ML models)

Bayesian inference:
- `jax`: Automatic differentiation and GPU acceleration
- `numpyro`: Probabilistic programming with NUTS sampling

Visualization and utilities:
- `matplotlib`: Plotting and visualization
- `corner`: Corner plots for posterior distributions
- `tqdm`: Progress bars
- `h5py`: Data storage and access

## Troubleshooting Common Issues

### JAX/GPU Issues
- Verify JAX installation: `import jax; print(jax.devices())`
- GPU memory errors: reduce `num_chains` or batch size
- CUDA compatibility: ensure matching JAX/CUDA versions

### MCMC Convergence
- Increase `num_warmup` if chains don't converge
- Try `use_dense_mass=True` for better adaptation
- Check R-hat diagnostics in NumPyro output

### Integration Performance
- Adjust `int_du` (integration step size) for speed vs accuracy
- Reduce `int_umax` if u-values are small
- Monitor integration normalization factors

## Data Files and Paths

The package expects data in FITS format with standard astronomical column names. Example data files should be placed in the `data/` directory. For large datasets, consider:

- External data paths configured via environment variables
- Memory-mapped arrays for very large catalogs
- Chunked processing for datasets exceeding RAM capacity