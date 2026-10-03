# Bayesian Binary Masses

This repository infers stellar mass--luminosity relations (MLRs) from wide
binaries using a hierarchical metallicity model and a Rice-convolved dynamical
likelihood.

The current primary workflow is the T8 pipeline implemented in
[`src/examples/run_hierarchical_metallicity_test.py`](src/examples/run_hierarchical_metallicity_test.py).
The older polynomial workflow in
[`src/examples/run_on_data_feh_global.py`](src/examples/run_on_data_feh_global.py)
is retained for reference and legacy comparisons, but it is no longer the main
MLR workflow.

## Current model state

Relative to the `7bc861a` baseline, the current T8 model includes several
important changes:

- the default outlier likelihood is defined directly in observed raw `u`, so
  it is independent of inferred binary mass;
- the legacy outlier defined in `u / sqrt(Mtot)` is retained explicitly for
  reproduction and sensitivity tests;
- both mixture components use exact finite-support normalization and consistent
  Jacobian accounting;
- the dynamics lookup has strict domain behavior rather than silent clamping;
- posterior diagnostics use all chains and include finite-value and
  consistency checks;
- the 27-node good-shape stack uses the corrected flat index
  `k = 9*i_B + 3*i_uc + i_C`;
- NumPyro initial values are transformed correctly into unconstrained sampler
  coordinates;
- the dynamics likelihood is truncated for the sample's `u_obs/sigma_u > 3`
  cut (see below); this is the main model, with the shape of `p(u~)` held fixed.

The repository also retains the experiments that motivated these changes:
(M_G)-window mass-score diagnostics, legacy-vs-raw-`u` outlier comparisons and
fixed good-shape sensitivity tests. Variable-shape code (joint good-shape + MLR
MCMC, metallicity-bin joint fits, alternating-MAP / EM-style optimization) is
no longer part of the main model and lives in
`src/examples/depracated/` and `test/depracated/`.

For the scientific history and interpretation of these changes, see
[`docs/model/POST_7BC_MODEL_IMPROVEMENTS.md`](docs/model/POST_7BC_MODEL_IMPROVEMENTS.md).

## Main MLR workflow

The standard T8 calculation has three restartable stages:

1. metallicity calibration;
2. dynamical Rice-likelihood lookup;
3. monotone PARSEC-relative MLR inference.

All three stages are handled by:

`src/examples/run_hierarchical_metallicity_test.py`

The default MLR stage uses the mass-independent raw-`u` outlier model.

### Environment

A typical setup is:

```bash
conda activate dyn
python -m pip install -e .
python -c 'import numpy, jax, numpyro; print(jax.devices())'
```

Set the input FITS file and output directory:

```bash
export DATA=data/jd_msms_single_bic_1kpc_filtered_cmdcut_cutb_jcaps_err_mhcal_good.fits
export OUT=results/hierarchical_metallicity_t8_formal
```

The T8 input table is expected to contain the uncorrected JCAPS metallicities
and errors used by the hierarchical calibration, including
`feh_jcaps_1`, `feh_jcaps_2`, `jc_sigma_m_h_1`, and
`jc_sigma_m_h_2`.

## 1. Smoke test

Before a formal run, test the complete pipeline on a tiny mock sample:

```bash
python -u src/examples/run_hierarchical_metallicity_test.py \
  --stage all \
  --mock \
  --mock-systems 32 \
  --quick \
  --output-dir results/t8_mock
```

This is a pipeline check, not a convergence test.

Useful focused checks are:

```bash
python -m py_compile \
  src/binary_masses/hierarchical_metallicity.py \
  src/examples/run_hierarchical_metallicity_test.py

python test/test_hierarchical_metallicity.py
python test/test_t8_1_normalization.py
python test/test_selection_truncation.py
python test/test_raw_u_outlier.py
```

## 2. Metallicity calibration

The first stage fits the hierarchical metallicity model and writes the
81-node latent-metallicity posterior for each binary:

```bash
python -u src/examples/run_hierarchical_metallicity_test.py \
  --stage calibration \
  --data "$DATA" \
  --output-dir "$OUT"
```

Important output:

`$OUT/latent_metallicity_weights_t8.npz`

This file is reused by the lookup and MLR stages.

## 3. Build the dynamical likelihood lookup

Build the fixed-good-shape Rice lookup:

```bash
python -u src/examples/run_hierarchical_metallicity_test.py \
  --stage lookup \
  --data "$DATA" \
  --output-dir "$OUT"
```

Important output:

`$OUT/dynamics_likelihood_lookup_t8.npz`

The lookup stage performs numerical convergence checks before accepting the
table. If the precision gate fails, inspect the saved convergence report and
increase the velocity quadrature range/nodes rather than bypassing the check.

## 4. Fit the standard MLR

Run the hard-monotone PARSEC-relative MLR with the current default
mass-independent raw-`u` outlier model:

```bash
python -u src/examples/run_hierarchical_metallicity_test.py \
  --stage mlr \
  --data "$DATA" \
  --output-dir "$OUT" \
  --outlier-coordinate raw_u
```

`--outlier-coordinate raw_u` is already the default; it is shown explicitly
here so that the model choice is visible in production command logs.

The main MLR outputs include:

- `mlr_mcmc_t8.npz`: grouped posterior samples;
- `mlr_summary_t8.csv`: posterior parameter summary;
- `mlr_diagnostics_t8.json`: sampler diagnostics;
- `mlr_derived_grid_t8.npz` and `mlr_derived_grid_t8.csv`: inferred MLR grid;
- `mlr_model.json`: full model/provenance metadata.

To regenerate plots from an existing MCMC file:

```bash
python -u src/examples/run_hierarchical_metallicity_test.py \
  --stage plot \
  --output-dir "$OUT"
```

### One-command alternative

After the smoke test, the three standard stages can also be run together:

```bash
python -u src/examples/run_hierarchical_metallicity_test.py \
  --stage all \
  --data "$DATA" \
  --output-dir "$OUT" \
  --outlier-coordinate raw_u
```

For long production runs, separate stages are usually preferable because they
are restartable and make intermediate validation easier.

## Standard production defaults

Without `--quick`, the T8 runner currently uses:

- 1000 warmup steps;
- 1000 posterior draws;
- 4 chains;
- 1024 lookup mass-scale nodes;
- 64 velocity quadrature nodes (the full sample passes the convergence gate
  with `--lookup-velocity-nodes 128 --lookup-sigma-extent 14`);
- selection truncation at `u/sigma_u > 3` (`--selection-cut 3`);
- target acceptance probability 0.9;
- an 8 x 4 monotone tensor-spline MLR;
- a solar anchor at (M_G=4.67), ([M/H]=0).

Override `--warmup`, `--samples`, `--chains`, and
`--target-accept` explicitly for formal runs when needed.

Do not use `--quick` for final inference: it reduces the sample size, MCMC
length, number of chains, and lookup resolution.

## Legacy outlier comparison

To reproduce the `7bc861a`-style mass-coupled outlier likelihood while
keeping the current metallicity calibration and fixed-good-shape lookup, run a
separate MLR output directory:

```bash
export OUT_LEGACY=results/t8_legacy_scaled_outlier

python -u src/examples/run_hierarchical_metallicity_test.py \
  --stage mlr \
  --data "$DATA" \
  --output-dir "$OUT_LEGACY" \
  --metallicity-posterior "$OUT/latent_metallicity_weights_t8.npz" \
  --dynamics-lookup "$OUT/dynamics_likelihood_lookup_t8.npz" \
  --outlier-coordinate scaled_tilde_u
```

This is a sensitivity/reproduction run, not the current default model.

The older dedicated script
[`src/examples/run_mass_independent_outlier.py`](src/examples/run_mass_independent_outlier.py)
is retained for the historical A/B experiment and validation workflow, but it
is no longer required for the standard raw-`u` MLR because raw-`u` is now
implemented directly in the core T8 runner.

## Fixed good-shape sensitivity

To test sensitivity to the normal (\tilde u) shape parameters (B),
(u_c), and (C), rebuild the dynamics lookup in a fresh directory and then
fit the MLR with the same shape override.

Example:

```bash
export SHAPE_OUT=results/t8_shape_sensitivity_example

python -u src/examples/run_hierarchical_metallicity_test.py \
  --stage lookup \
  --data "$DATA" \
  --output-dir "$SHAPE_OUT" \
  --metallicity-posterior "$OUT/latent_metallicity_weights_t8.npz" \
  --shape-sensitivity \
  --good-shape-b 0.00103 \
  --good-shape-uc 40.98 \
  --good-shape-c 11.53

python -u src/examples/run_hierarchical_metallicity_test.py \
  --stage mlr \
  --data "$DATA" \
  --output-dir "$SHAPE_OUT" \
  --metallicity-posterior "$OUT/latent_metallicity_weights_t8.npz" \
  --dynamics-lookup "$SHAPE_OUT/dynamics_likelihood_lookup_t8.npz" \
  --shape-sensitivity \
  --good-shape-b 0.00103 \
  --good-shape-uc 40.98 \
  --good-shape-c 11.53 \
  --outlier-coordinate raw_u
```

Use a new output directory for every shape experiment.

## Selection truncation (main model)

The catalogue keeps only pairs with `u_obs/sigma_u > 3`, so the observed `u`
are truncated at a per-system threshold. The likelihood of each pair is divided
by the probability of passing the cut,

```
L_j(m) / Pi_j(m),   Pi_j(m) = (1-f) Pi_good(m/sigma_j) + f Pi_out,
```

and the metallicity marginalization is `sum_q P_jq L_jq / Pi_jq`. This is on by
default for real data (`--selection-cut 3`); use `--no-selection-cut` for the
untruncated likelihood and `--selection-cut C` for another threshold. Mock data
default to no cut. With the cut modelled, the shape of `p(u~)` is kept fixed
instead of being fitted: letting it vary absorbed the selection into the shape
and gave masses 25--40 per cent too high at M_G ~ 6--9.

The full model, including the truncation, is described in
[`docs/t9_methods/T9_methods.pdf`](docs/t9_methods/T9_methods.pdf). A quick MAP
comparison with and without the cut on an existing calibration:

```bash
python -u src/examples/map_mlr_selection_check.py \
  --data "$DATA" --posterior "$OUT/latent_metallicity_weights_t8.npz" --cut 3.0
```

## Deprecated variable-shape workflows

The 27-node shape stack, `--sample-dynamics-shape`, the joint good-shape + MLR
MCMC, the metallicity-bin joint fits and the alternating-MAP / EM-style
experiments were removed from the library and runner. Their scripts are kept in
`src/examples/depracated/` for reference and need the code as of git tag
`pre-selection-truncation-main`; run them with
`PYTHONPATH=src/examples:src` from that tag.

## Diagnostic scripts

Useful diagnostic and comparison entry points include:

- `src/examples/diagnose_mg_window_mass_score.py`:
  local and finite-step likelihood pressure from changing masses in an
  (M_G) window;
- `src/examples/plot_mlr_residuals_parsec.py`:
  fixed-shape MLR residual comparisons;
- `src/examples/map_mlr_selection_check.py`:
  MAP comparison of the MLR with and without the selection truncation;
- `src/examples/compare_t8_relations_mg.py`:
  MLR comparison in absolute G magnitude;
- `src/examples/compare_t8_relations_bprp.py`:
  comparison in BP-RP color;
- `src/examples/compare_t8_mann2019_ks_local.py`:
  local (K_s)-band comparison.

## Recommended interpretation hierarchy

For scientific results, use this hierarchy:

1. standard T8 posterior with raw-`u` outlier, fixed good shape and the
   `u/sigma_u > 3` truncation;
2. untruncated, legacy-outlier and fixed-good-shape runs as controlled
   sensitivity tests.

Agreement between different starts or methods is useful evidence of numerical
stability, but it does not by itself establish the physical correctness of the
absolute mass scale.

## SLURM

The generic stage launcher is:

`scripts/run_t8_hierarchical_metallicity.slurm`

Example:

```bash
sbatch --partition=PARTITION --account=ACCOUNT --gres=gpu:1 \
  --export=ALL,STAGE=calibration,DATA="$DATA",OUTPUT_DIR="$OUT",CONDA_ENV=dyn \
  scripts/run_t8_hierarchical_metallicity.slurm
```

Repeat with `STAGE=lookup` and `STAGE=mlr`, or use `STAGE=all` after a
successful smoke test.

Additional runner options can be passed through `EXTRA_ARGS`.

## Legacy polynomial workflow

The older continuous-metallicity polynomial model remains available at:

`src/examples/run_on_data_feh_global.py`

It uses `DifferencePolyFehMLR` and is useful for historical comparison, but it
should not be confused with the current T8 hierarchical-metallicity +
monotone-spline MLR pipeline.

## Further documentation

- [Post-`7bc861a` model history](docs/model/POST_7BC_MODEL_IMPROVEMENTS.md)
- [T8 server run guide](docs/workflows/T8_SERVER_RUN.md)
- [T8.2 joint-shape pipeline](docs/archive/workflows/T82_JOINT_SHAPE_PIPELINE.md)
- [Alternating-MAP / EM-style workflow](docs/archive/workflows/EM_MLR_WORKFLOW.md)
- [Mass-independent outlier experiment plan](docs/validation/T8_MASS_INDEPENDENT_OUTLIER_GARCHING_PLAN.md)

## License

This project is licensed under the MIT License. See [`LICENSE`](LICENSE).
