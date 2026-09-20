# T8 server run guide

The post-7bc core T8 model uses a mass-independent raw-u outlier likelihood.
The [EM-style workflow](EM_MLR_WORKFLOW.md) is retained as a sensitivity and
optimization experiment. This guide supplies the calibration and lookup inputs
and documents the explicit fixed-shape MCMC workflow.

This guide runs the approved T8a/T8c/T8b pipeline in the `dyn` Conda
environment.  The runner writes only T8-named products and refuses a legacy
posterior or dynamics lookup whose metadata does not match the current model.
Do not start a full real-data fit until the mock smoke test has completed.

## Environment and inputs

```bash
conda activate dyn
python -m pip install -e .
python -c 'import numpy, jax, numpyro; print("T8 environment OK")'
```

Set the FITS path and an output directory on the server.  The input table must
contain the uncorrected columns `feh_jcaps_1`, `feh_jcaps_2`,
`jc_sigma_m_h_1`, and `jc_sigma_m_h_2`; calibrated/binary-corrected columns
are never substituted.

```bash
export DATA=/path/to/input.fits
export OUTPUT_DIR=/path/to/results/hierarchical_metallicity_t8
```

Run the static and focused unit checks before submitting a fit:

```bash
python -m py_compile \
  src/binary_masses/hierarchical_metallicity.py \
  src/examples/run_hierarchical_metallicity_test.py
python test/test_hierarchical_metallicity.py
```

## Mock smoke test

The smoke test is intentionally small and does not use the survey table:

```bash
python -u src/examples/run_hierarchical_metallicity_test.py \
  --stage all --mock --mock-systems 2 --quick \
  --output-dir /path/to/results/t8_mock
```

This two-system run is a minimal pipeline check, not an MCMC convergence test.
For a stronger check, increase `--mock-systems` to 32.  A larger mock can
legitimately trigger the Rice precision gate: inspect
`lookup_convergence_t8.json`, increase `--lookup-sigma-extent` or
`--lookup-velocity-nodes`, and rerun the lookup rather than bypassing the gate.

Check that `latent_metallicity_weights_t8.npz`,
`dynamics_likelihood_lookup_t8.npz`, and `mlr_mcmc_t8.npz` can be loaded and
that `mlr_monotonicity.json` reports zero dense-grid violations.  A failed
metadata or lookup-domain check is a stop condition; do not copy a T7 product
into the output directory.

## Formal stages

Stages are restartable.  T8a calibrates JCAPS plus the 4000 K full-grid CMD
mask and saves the averaged 81-node latent-metallicity weights.  T8c builds
the Rice lookup using 1024 geometric mass-scale nodes, `K=10`, and `R=64`.
Before the full table is built, T8c deterministically selects 32 systems across
the fractional-error range and 16 mass-scale nodes.  It compares the adopted
`K=10, R=64` quadrature with `K=14, R=128` and adaptive integrals in both
integration coordinates.  The run stops if any non-floor component differs by
more than `1e-3` in log likelihood.  Floor locations are classified explicitly:
a table floor is accepted only when the float64 adaptive reference is also at
or below that floor.  The quadrature interval is split exactly
at the transformed `v=s*80` support boundary.  T8b then runs the hard-monotone
tensor-spline MLR using the saved weights and lookup only.

```bash
python -u src/examples/run_hierarchical_metallicity_test.py \
  --stage calibration --data "$DATA" --output-dir "$OUTPUT_DIR"

python -u src/examples/run_hierarchical_metallicity_test.py \
  --stage lookup --data "$DATA" --output-dir "$OUTPUT_DIR"

python -u src/examples/run_hierarchical_metallicity_test.py \
  --stage mlr --data "$DATA" --output-dir "$OUTPUT_DIR"
```

Use `--quick` only for diagnostics; it changes the system cap, MCMC size, and
lookup mass-node count to 512.  `--mlr-knot-x`, `--mlr-knot-z`, spline degrees,
prior scales, and solar-anchor scales are available as explicit overrides.
The default T8 knot vectors are the 8 x 4 design in the handoff document.

## SLURM

The generic launcher does not hard-code a partition or account:

```bash
sbatch --partition=PARTITION --account=ACCOUNT --gres=gpu:1 \
  --export=ALL,STAGE=calibration,DATA="$DATA",OUTPUT_DIR="$OUTPUT_DIR",CONDA_ENV=dyn \
  scripts/run_t8_hierarchical_metallicity.slurm
```

Repeat with `STAGE=lookup` and `STAGE=mlr`, or submit `STAGE=all` after the
mock run.  Optional runner arguments can be passed as a whitespace-separated
`EXTRA_ARGS` environment variable.  If a job stops, inspect the stage's JSON
metadata and restart that stage; lookup reuse is allowed only when row order,
data digest, Rice normalization, quadrature, and interpolation metadata all
match.  The generic script leaves GPU resource syntax to `sbatch`, because it
varies by cluster, and verifies that JAX can see a GPU by default.  For a
deliberate CPU smoke test, omit `--gres=gpu:1`, select a CPU partition if the
cluster requires one, and export `REQUIRE_GPU=0`.
