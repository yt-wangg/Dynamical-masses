# T8 80/20 held-out validation

This validation reads the frozen full-sample T8 posterior, lookup, and MLR
trace, but never modifies that baseline directory. Binary systems are split
with a fixed random seed: 80% are used to refit the MLR and the remaining 20%
are used only for likelihood evaluation. The second training fit changes only
the solar-anchor prior width, from `0.01/ln(10)` dex to three times that value.
This is specifically a held-out test of the T8b dynamical MLR. It reuses the
frozen full-sample T8a metallicity posterior as a covariate posterior; T8a did
not use the held-out dynamical `u/u_sigma` measurements.

For a fair held-out reference, the uncorrected PARSEC model refits its own
`f_outlier` on the 80% training set with the same `Beta(3,12)` prior. Reported
predictive log likelihoods integrate over posterior uncertainty before taking
the logarithm. The PARSEC one-dimensional posterior integral first locates the
posterior mode and then uses 128 Gauss-Legendre nodes over a bounded interval
whose two tails are at least 30 log units below the mode.

## Files to upload

Upload these additional files together with the T8 baseline code:

```text
src/examples/run_t8_holdout_validation.py
scripts/run_t8_holdout_validation.slurm
docs/T8_HOLDOUT_VALIDATION.md
```

The server must already have the original input FITS and the complete frozen
baseline directory containing `latent_metallicity_weights_t8.npz`,
`dynamics_likelihood_lookup_t8.npz`, and `mlr_mcmc_t8.npz`.
No FITS or artifact SHA256 manifest is required. The runner still checks the
T8 model schema, system row order, and exact alignment of the dynamics lookup
with `u/u_sigma` so that training and test systems cannot be mixed accidentally.

## Smoke test

Run a small CPU or GPU smoke test first. It samples 64 systems and writes to a
separate disposable directory:

```bash
conda activate dyn
export DATA=/path/to/input.fits
export BASELINE_DIR=/path/to/results/hierarchical_metallicity_t8_20260913

python -u src/examples/run_t8_holdout_validation.py \
  --stage all --quick \
  --data "$DATA" \
  --baseline-dir "$BASELINE_DIR" \
  --output-dir /path/to/results/t8_holdout_smoke
```

This smoke run verifies the file chain only. Its short single-chain MCMC is not
a convergence or scientific test.

## Formal run

The stages can be submitted separately and safely resumed. Keep the same
`OUTPUT_DIR`, `--split-seed`, `--test-fraction`, and `--wide-solar-factor` for
all three jobs:

```bash
export OUTPUT_DIR=/path/to/results/hierarchical_metallicity_t8_20260913_holdout

python -u src/examples/run_t8_holdout_validation.py \
  --stage fit-default --data "$DATA" \
  --baseline-dir "$BASELINE_DIR" --output-dir "$OUTPUT_DIR"

python -u src/examples/run_t8_holdout_validation.py \
  --stage fit-wide --data "$DATA" \
  --baseline-dir "$BASELINE_DIR" --output-dir "$OUTPUT_DIR"

python -u src/examples/run_t8_holdout_validation.py \
  --stage evaluate --data "$DATA" \
  --baseline-dir "$BASELINE_DIR" --output-dir "$OUTPUT_DIR"
```

The defaults use the full cohort, the nearest-integer 80/20 split with seed `20260914`,
four chains, 1000 warmup iterations and 1000 retained draws per chain. The
wide-anchor fit uses three times the default solar-anchor standard deviation.

## SLURM

The generic launcher does not hard-code a cluster partition or account:

```bash
sbatch --partition=PARTITION --account=ACCOUNT --gres=gpu:1 \
  --export=ALL,STAGE=fit-default,DATA="$DATA",BASELINE_DIR="$BASELINE_DIR",OUTPUT_DIR="$OUTPUT_DIR",CONDA_ENV=dyn \
  scripts/run_t8_holdout_validation.slurm

sbatch --partition=PARTITION --account=ACCOUNT --gres=gpu:1 \
  --export=ALL,STAGE=fit-wide,DATA="$DATA",BASELINE_DIR="$BASELINE_DIR",OUTPUT_DIR="$OUTPUT_DIR",CONDA_ENV=dyn \
  scripts/run_t8_holdout_validation.slurm

sbatch --partition=PARTITION --account=ACCOUNT --gres=gpu:1 \
  --export=ALL,STAGE=evaluate,DATA="$DATA",BASELINE_DIR="$BASELINE_DIR",OUTPUT_DIR="$OUTPUT_DIR",CONDA_ENV=dyn \
  scripts/run_t8_holdout_validation.slurm
```

Do not submit the `evaluate` job until both fit jobs have finished. The launcher
checks for a GPU by default. Set `REQUIRE_GPU=0` only for a deliberate CPU run.

## Outputs

The main report is `holdout_validation_summary_t8.json`. Pointwise posterior
predictive scores are in `heldout_pointwise_t8.csv`; posterior-draw arrays are
in `heldout_loglikelihood_t8.npz`; the independently refit PARSEC mixture-weight
posterior is in `parsec_f_outlier_posterior_t8.npz`; and
`delta_star_curve_comparison_t8.png` compares the frozen full-data curve, the
80% default-anchor curve, and the 80% wide-anchor curve.

The exact split and digests are stored in `system_split_t8.npz` and
`system_split_t8.json`. If an existing output directory contains a different
system split or incompatible solar prior, the runner stops and requires a new
output directory instead of mixing results.

Each training subdirectory also contains `mlr_diagnostics_t8.json`. Formal
multi-chain runs record the maximum R-hat, minimum effective sample size,
per-chain BFMI, divergences, acceptance probability, and maximum NUTS steps.
