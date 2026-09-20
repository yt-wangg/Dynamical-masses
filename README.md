# Bayesian Binary Masses

The current dynamical MLR workflow is **T8**, run through
[`src/examples/run_hierarchical_metallicity_test.py`](src/examples/run_hierarchical_metallicity_test.py).
See [T8 run instructions](docs/T8_SERVER_RUN.md) for calibration, fitting, and plotting.

T8 fits a PARSEC-relative mass surface with a shared metallicity posterior for
each wide binary. Over `3.5 <= M_G <= 13.5` and `-1.0 <= [M/H] <= 0.6`,
the final mass decreases or stays constant toward fainter magnitudes and
increases or stays constant toward higher metallicity. These constraints apply
to the final mass, while the correction relative to PARSEC may have either sign.
Fitted T8 curves are evaluated from `mlr_mcmc_t8.npz`.

## Joint velocity-shape and MLR fitting

The [T8.2 pipeline](docs/T82_JOINT_SHAPE_PIPELINE.md) fits independent [M/H]
bins from S2 and PARSEC initializations, with a mass-independent raw-u outlier
component. It includes full-sample MCMC commands, posterior comparison plots,
and the limitations of the saved runs.

## Legacy polynomial workflow

The instructions below describe
[`src/examples/run_on_data_feh_global.py`](src/examples/run_on_data_feh_global.py).
It fits a mass–absolute-magnitude relation with continuous metallicity
`[Fe/H]` to real Gaia wide-binary data, using NUTS sampling with JAX and
NumPyro.

This polynomial model allows unrestricted metallicity corrections. Its results
and the T7 three-knot fits are separate from the monotone T8 inference.

## Model overview

The main script uses `DifferencePolyFehMLR`:

- The isochrone mass grid in `data/interpolated_mass_data/` provides the
  baseline relation `m_iso(M_G, [M/H])`.
- A two-dimensional polynomial in absolute G magnitude and metallicity
  corrects `log10(mass)`.
- A Rice distribution models the observational uncertainty.
- The outlier fraction and outlier-distribution parameters can be fitted
  jointly with the mass relation.
- Posterior sampling uses NumPyro NUTS.
- JAX uses a GPU when one is available and supported, and otherwise uses the
  CPU.

The code passes the observed `[Fe/H]` directly to the isochrone `[M/H]` grid,
which assumes `[M/H] ≈ [Fe/H]`.

## Required repository files

The default configuration uses the following files included in the
repository:

```text
bayesian-binary-masses/
├── data/
│   ├── jd_msms_single_bic_1kpc_filtered_cutb_fehloss.fits
│   └── interpolated_mass_data/
│       ├── gmag_grid.npy
│       └── mass_interp_MH_*.npy
├── src/
│   ├── binary_masses/
│   │   ├── differencepoly_feh.py
│   │   ├── isochrone_grid.py
│   │   ├── jax_utils.py
│   │   └── polynomial.py
│   └── examples/
│       └── run_on_data_feh_global.py
└── requirements.txt
```

The script determines the repository root from its own location, so data
paths do not depend on the current working directory. Running it from the
`bayesian-binary-masses/` root is still recommended.

## Environment setup

An isolated Python 3.10 or 3.11 environment is recommended:

```bash
cd bayesian-binary-masses

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

To install `binary_masses` as an editable package, optionally run:

```bash
python -m pip install -e .
```

The main script adds `src/` to the Python path itself. Installing the package
is therefore optional when running only this script, but its dependencies
must still be installed.

### GPU and CPU selection

The code does not force a particular platform. It relies on JAX's automatic
device selection:

- If JAX detects a compatible GPU, the GPU is used.
- If no compatible GPU is available, the CPU is used.

At runtime, the script prints one of the following:

```text
NumPyro: running on GPU backend (...)
```

or:

```text
NumPyro: running on CPU backend (1 device(s)).
```

Check the devices visible to the current Python environment with:

```bash
python -c "import jax; print(jax.devices())"
```

Having an NVIDIA GPU installed does not by itself mean JAX can use it. A
compatible driver and GPU-enabled JAX installation are also required. The
installation procedure depends on the CUDA and operating-system versions;
refer to the
[official JAX installation guide](https://docs.jax.dev/en/latest/installation.html).
The script still works with CPU-only JAX, but sampling will usually take
considerably longer.

## Run the main script

From the repository root, run:

```bash
python src/examples/run_on_data_feh_global.py
```

The default workflow:

1. Reads `data/jd_msms_single_bic_1kpc_filtered_cutb_fehloss.fits`.
2. Keeps systems with `a_g_edhf_1 <= 0.1`.
3. Selects at most 5,000 systems using a fixed random seed.
4. Computes `v`, `u`, and `u_sigma` from proper motion, parallax, and projected
   separation.
5. Loads the continuous `[M/H]` isochrone mass surface.
6. Runs NumPyro MCMC.
7. Saves posterior samples, a corner plot, and a multi-metallicity
   mass–magnitude plot.

The default production configuration uses 800 warmup steps, 6,000 posterior
samples, one chain, and a dense mass matrix. Initial JAX compilation and the
full sampling run may take a long time.

### Quick smoke test

Before a production run, the complete workflow can be tested by temporarily
reducing the following arguments in the `test_differencepoly_feh_model(...)`
call near the bottom of
[`run_on_data_feh_global.py`](src/examples/run_on_data_feh_global.py):

```python
num_warmup=100,
num_samples=200,
use_dense_mass=False,
```

The 5,000-system subsample limit in the script can also be reduced
temporarily. Restore the production settings after confirming that data
loading, JAX compilation, sampling, and plotting all complete successfully.

## Input data requirements

The default FITS table must contain at least the following columns:

| Column | Purpose |
| --- | --- |
| `a_g_edhf_1` | Extinction selection |
| `pmra1`, `pmra2` | Right-ascension proper motions of both components |
| `pmdec1`, `pmdec2` | Declination proper motions of both components |
| `parallax1` | Parallax |
| `sep_AU` | Projected separation |
| `dpm_over_error` | Used to calculate `u_sigma` |
| `absg1`, `absg2` | Absolute G magnitudes of both components |
| `feh` | Preferred metallicity column |

If `feh` is unavailable, the script attempts to use `feh_jcaps_1`. It raises
an error before fitting if neither column exists.

The script calculates:

```text
v       = 4.74 × sqrt((pmra2-pmra1)² + (pmdec2-pmdec1)²) / parallax1
u       = v × sqrt(sep_AU)
u_sigma = u / dpm_over_error
```

To use another dataset, change `data_path` in `main()` and ensure that the
column names and units are consistent with these calculations.

## Main configuration

The main settings are in the `test_differencepoly_feh_model(...)` call near
the bottom of the script:

| Parameter | Current value | Purpose |
| --- | ---: | --- |
| `feh_min`, `feh_max` | `-1`, `0.6` | Metallicity range |
| `absg_min`, `absg_max` | `3.5`, `13.5` | Absolute G-magnitude range |
| `mass_min`, `mass_max` | `0.05`, `2.0` | Mass range in solar masses |
| `order` | `1` | Default term selection for the 2D correction |
| `uncertainty_model` | `"rice"` | Observational uncertainty model |
| `fit_outlier_params` | `True` | Fit the outlier parameters |
| `num_warmup` | `800` | Number of NUTS warmup steps |
| `num_samples` | `6000` | Number of posterior samples |
| `num_chains` | `1` | Number of MCMC chains |
| `use_dense_mass` | `True` | Use a dense mass matrix |
| `seed` | `33` | NumPyro random seed |
| `anchor_enabled` | `True` | Enable the solar-mass anchor |

The current solar anchor is:

```text
M_G = 4.67, [Fe/H] = 0.0, mass = 1.0 M_sun, sigma = 0.01
```

## Output

The default output directory is:

```text
results/data_diffpoly2d_anchor_cute/
```

The main output files are:

- `mcmc_*.txt`: posterior parameter samples;
- `corner*.png`: posterior corner plot;
- `fit_multifeh_*.png`: mass–magnitude relations and uncertainties at
  different `[Fe/H]` values.

Output filenames encode the uncertainty model, outlier configuration, model
order, metallicity range, and other important settings. Existing files with
the same names are overwritten, so back up production results when needed.

## Troubleshooting

### `ModuleNotFoundError`

Confirm that the intended environment is active, then run:

```bash
python -m pip install -r requirements.txt
```

Using `python -m pip` helps ensure that dependencies are installed for the
same Python interpreter that runs the script.

### A GPU is installed, but the script reports CPU

Run:

```bash
python -c "import jax; print(jax.devices())"
```

If this command lists only CPU devices, JAX in the current Python environment
has not detected the GPU. Check that the GPU driver, CUDA version, and JAX
installation are compatible.

### Out of memory or sampling is too slow

For a smoke test, reduce the input-system count, `num_warmup`, and
`num_samples`. Keep `num_chains=1`, and set `use_dense_mass=False` if needed.

### Isochrone data cannot be found

Confirm that these files exist:

```text
data/interpolated_mass_data/gmag_grid.npy
data/interpolated_mass_data/mass_interp_MH_*.npy
```

The main script passes an absolute path to the model, so starting the script
from another directory should not normally affect this path.

## License

This project is licensed under the MIT License. See [`LICENSE`](LICENSE).
