# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Bayesian Binary Mass Estimation - A Python package for estimating dynamical masses of binary stars using Bayesian inference with Rice distribution uncertainties. Implements non-parametric fitting of mass-luminosity relations using NumPyro with NUTS sampling.

## Commands

### Installation
```bash
pip install -r requirements.txt
pip install -e .
pip install -e ".[dev]"  # With dev tools (pytest, black, flake8)
```

### Running Tests
```bash
pytest tests/                                    # All tests
pytest tests/test_core.py -v                     # Verbose
pytest tests/test_core.py::TestPU8Sampler -v     # Specific class
pytest tests/test_core.py::TestPU8Sampler::test_rejection_sample -v  # Single test
```

### Code Quality
```bash
black src/ tests/    # Format code
flake8 src/ tests/   # Lint
```

## Architecture

### Package Structure
```
src/binary_masses/
├── core.py       # Main inference classes (NonParametricPosteriorPlotter, MultiMetallicityFitter, PU8Sampler)
├── sampling.py   # Binary orbit utilities (BinarySystem, BinaryPopulation)
└── __init__.py   # Package exports

src/examples/
├── basic_fitting.py        # Single metallicity fitting example
└── multi_metallicity.py    # Multi-metallicity analysis example
```

### Core Classes

**NonParametricPosteriorPlotter** (core.py): Main Bayesian inference class using NumPyro + JAX
- `set_data()`: Configure input data (u values, magnitudes)
- `run_numpyro()`: Run MCMC inference with NUTS sampling
- `plot_results()`: Generate corner plots and visualizations
- Supports 'rice' (recommended) or 'gaussian' uncertainty models

**MultiMetallicityFitter** (core.py): Wrapper for fitting multiple metallicity bins
- `bin_data_by_metallicity()`: Equal-width or equal-frequency binning by [Fe/H]
- `fit_all_bins()`: Batch fit across metallicity bins
- `plot_comparison()`: Overlay comparison plots

**PU8Sampler** (core.py): Samples from the p(ũ) distribution where ũ = u/√m_tot
- `sample()`: Main interface with 'rejection' or 'inverse_transform' methods

### Key Scientific Concepts

The u-parameter: `u = v × √s` (velocity × √separation)
Normalized parameter: `ũ = u/√m_tot`

Likelihood model uses Rice distribution (for positive-definite measurements) or Gaussian distribution for measurement uncertainties.

### Data Format

Required columns (astropy.table.Table):
- `u`: Velocity × √separation parameter
- `absg1`, `absg2`: Primary/secondary absolute G magnitudes
- `u_sigma`: Uncertainty in u (optional, for Rice distribution)
- `feh`: Metallicity [Fe/H] (optional, for multi-metallicity fitting)

### Running Examples
```bash
cd src/examples
python basic_fitting.py        # Single metallicity fitting
python multi_metallicity.py    # Multi-metallicity analysis
```

## Dependencies

Python 3.8+ required. Core: numpy, scipy, matplotlib, astropy, torch, jax, jaxlib, numpyro, corner

GPU acceleration is automatic through JAX when available.
