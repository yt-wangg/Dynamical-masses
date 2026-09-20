# Default EM-style MLR workflow

The default method alternates conditional MAP updates of the MLR and the
normal velocity shape plus outlier fraction. It uses the observed-data
likelihood marginalized over the saved metallicity posterior and Rice
measurement errors. It is EM-style optimization, not a strict EM algorithm
with an explicit expectation step, and does not produce posterior intervals.

## Run

From the repository root, using an environment with the repository dependencies:

```bash
bash scripts/run_mlr.sh \
  --data data/jd_msms_single_bic_1kpc_filtered_cmdcut_cutb_jcaps_err_mhcal_good.fits \
  --baseline results/hierarchical_metallicity_t8_1_formal \
  --shape-stack results/hierarchical_metallicity_t8_2_formal/dynamics_likelihood_shapestack_t8.npz \
  --output results/em_mlr_default
```

Set `PYTHON` to select an interpreter. The implementation is
`src/examples/run_em_mlr_pilot.py`; its CLI options pass through unchanged.
Required input products are described in the [T8 server guide](T8_SERVER_RUN.md).
Use a new output directory for each experiment.

Defaults are a fixed random subset of 2000 systems, seed 20260919, at most 10
full iterations, 80 optimizer iterations per block, and tolerance 0.0002.
Both default-shape and S2-shape starts are fitted to this same subset. Both
use the same initial MLR from the baseline posterior; `--initial-mlr` can
select another saved posterior. This is not a fresh PARSEC MLR initialization.
`--max-systems` explicitly changes sample size.

Each system retains its complete saved metallicity posterior. The normal
component uses a 27-node shape lookup with smoothstep weights; the outlier
is a Rice-convolved TN(40,13,[0,80]) in raw u, independent of mass and
metallicity. The MLR retains monotonicity, PARSEC-relative regularization,
and the solar anchor.

The runner saves the selected rows, objective traces, final MAP states,
shape parameters, mass comparisons, and `summary.json`. These are optimizer
results. Use the separate [joint MCMC pipeline](T82_JOINT_SHAPE_PIPELINE.md)
for posterior intervals and independent metallicity-bin fitting. The default
EM runner fits the selected sample together; it does not split [M/H] bins.

## Limitations

The current convergence flag uses the shape-half-step improvement rather
than the full-cycle improvement. Smoothstep interpolation has zero slope at
shape grid nodes, so apparent stationarity needs checking. The outlier
quadrature check does not validate normal-shape interpolation. The shared
formal metallicity calibration has three divergences. Agreement between
starts alone does not validate the physical mass scale or remove PARSEC
prior dependence.
