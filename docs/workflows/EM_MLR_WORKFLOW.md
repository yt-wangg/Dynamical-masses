# Physics-informed alternating-MAP / EM-style workflow

This workflow alternates conditional MAP updates of two parameter blocks:

1. the hard-monotone MLR parameters;
2. the raw-u outlier fraction and the good-velocity-shape parameters
   `(log B, log uc, log C)`.

Both blocks maximize the same joint log posterior. This is therefore a block
coordinate / alternating-MAP calculation. It is called EM-style because it
iterates between two coupled model components, but it is **not** a strict
expectation-maximization algorithm with an explicit E-step.

The scientific motivation is that the default good shape is derived from the
orbital-population calculation and is therefore retained as the physical
reference. The data are allowed to recalibrate that shape only within the
predefined physical box and under a prior centered on the physical reference.

## Run

From the repository root:

```bash
bash scripts/run_mlr.sh \
  --data data/jd_msms_single_bic_1kpc_filtered_cmdcut_cutb_jcaps_err_mhcal_good.fits \
  --baseline results/hierarchical_metallicity_t8_1_formal \
  --shape-stack results/hierarchical_metallicity_t8_2_formal/dynamics_likelihood_shapestack_t8.npz \
  --output results/em_mlr_default
```

The implementation is
`src/examples/run_em_mlr_pilot.py`. `scripts/run_mlr.sh` simply forwards
its command-line arguments.

Both the `default` physical shape and the `S2` shape are run on the same
selected systems and from the same initial MLR state. The two starts are a
stability diagnostic: agreement increases confidence that the optimizer is
finding the same basin, but it is not a substitute for posterior uncertainty.

### Resume a saved pair of trajectories

Continuation reads both vectors from the checkpoint and writes a new result
directory. For the full solar-only result, the formal continuation command is:

```bash
MPLCONFIGDIR=/tmp/dyn-em-mpl-cache \
  bash scripts/run_mlr.sh \
  --data results/em_solar_only_full_rebuild_20260925/input.fits \
  --baseline results/em_solar_only_full_rebuild_20260925 \
  --dynamics-lookup results/em_solar_only_full_rebuild_20260925/dynamics_likelihood_lookup_t8.npz \
  --initial-map-states results/em_solar_only_full_rebuild_20260925/em_like/map_states.npz \
  --output results/em_solar_only_full_rebuild_20260926_em_continuation \
  --max-systems 14876 \
  --max-iterations 100 --block-maxiter 200 \
  --direct-nodes 12 --direct-chunk 128
```

`--direct-chunk 128` changes only CPU batching and has been validated against
the source objective; the quadrature order remains 12 nodes per interval.
The resume check requires the adjacent `selected_subset.npz` to have exactly
the same `positions` and `row_indices` as the current selection, validates
both finite 38-component vectors, and rejects writing back into the source
directory. `summary.json` records the source checkpoint and prior iteration
metadata. A resumed trace may restart its local cycle counter at zero, while
`cumulative_iteration` preserves the provenance.

## Good-shape likelihood evaluation

The default is now:

```text
--shape-evaluation direct
```

The good-component Rice likelihood is evaluated continuously at the current
`(B, uc, C)` using piecewise Gauss-Legendre quadrature. This avoids using
the 27-node shape stack as the differentiable optimization surface and removes
the artificial zero-gradient stationarity caused by the previous smoothstep
interpolation.

The 27-node stack is still required to define/record the validated shape box
and to provide the raw-u outlier table for the selected rows.

A faster diagnostic mode remains available:

```bash
--shape-evaluation linear_stack
```

This uses linear trilinear interpolation in log-shape coordinates. It is
intended for diagnostics and comparisons rather than the preferred formal
alternating-MAP result.

The shared shape-stack implementation also uses linear, rather than smoothstep,
cell fractions. Linear interpolation is only piecewise differentiable at the
grid nodes, but it does not force the first derivative to zero there.

## Physics-centered shape prior

The shape update is not an unconstrained empirical fit. Inside the existing
hard shape box, the MAP target includes Gaussian calibration priors in
log-parameter space centered on the orbital-model reference:

```text
B  = 0.002544
uc = 35.67
C  = 3.1
```

The default prior widths are:

```text
sigma(log B)  = ln(2)
sigma(log uc) = ln(1.3)
sigma(log C)  = ln(2)
```

Thus a factor-of-two change in `B` or `C`, and a factor-1.3 change in
`uc`, corresponds to approximately one prior standard deviation. These
priors are deliberately broad: they anchor the calibration to the
physics-derived shape without preventing a substantial data-driven shift.

The widths are explicit CLI options:

```bash
--shape-prior-sigma-log-b ...
--shape-prior-sigma-log-uc ...
--shape-prior-sigma-log-c ...
```

Changing them is a scientific model choice and should be recorded as a
separate sensitivity experiment.

## Convergence criterion

Convergence is evaluated after a **complete MLR + shape cycle**, rather than
from the shape half-step alone.

A cycle is stable only when all of the following hold:

1. every cycle metric is finite, and the absolute full-cycle log-posterior
   change divided by `1 + |log posterior at cycle start|` is below
   `--objective-rtol` (default `1e-7`);
2. both optimizer blocks report success and their candidate states are
   accepted;
3. the maximum change in `log B`, `log uc`, and `log C` is below
   `--shape-tol` (default `1e-3`), the maximum relative mass change on the
   fixed grid `M_G = 3.5..13.5` and `[M/H] = -1..0.6` is below
   `--mass-rtol` (default `1e-3`), and the absolute outlier-fraction change is
   below `--f-tol` (default `1e-3`);
4. the scaled projected-gradient infinity norms of both blocks are below
   `--projected-grad-tol` (default `1e-6`). The optimizer coordinates use
   displacement scales 0.02 for `c0` and 0.10 for every other coordinate.
   At a lower bound, a bounded component contributes zero only when the
   minimization gradient is non-negative; at an upper bound it contributes
   zero only when that gradient is non-positive. Boundary flags are saved in
   each trace row.

By default, the criteria must hold for two consecutive full cycles:

```text
--stable-cycles 2
```

The trace records the full-cycle objective improvement, relative improvement,
shape change, MLR mass change, outlier-fraction change, projected-gradient
norms, boundary flags, and consecutive-stability count. All conditions must
hold for two consecutive complete cycles.

## Default optimization settings

Unless overridden:

- fixed deterministic subset: 2000 systems;
- seed: 20260919;
- maximum full cycles: 10;
- maximum L-BFGS-B iterations per block: 80;
- direct quadrature nodes per interval: 12;
- direct system chunk: 16;
- raw-u outlier: Rice-convolved TN(40,13,[0,80]), independent of mass and
  metallicity.

Each system retains its complete saved latent-metallicity posterior.

## Output and interpretation

The runner saves:

- `selected_subset.npz`;
- `trace_default.jsonl`;
- `trace_S2.jsonl`;
- `map_states.npz`;
- `summary.json`;
- diagnostic plots.

The outputs are MAP optimizer results, not posterior samples and not credible
intervals. Use the
[joint-shape MCMC workflow](T82_JOINT_SHAPE_PIPELINE.md)
when posterior covariance or credible intervals are required.

The preferred interpretation is:

- fixed physical shape: baseline result under the orbital-population model;
- alternating MAP: physics-informed self-consistent recalibration of the shape
  and MLR;
- joint MCMC: posterior robustness/covariance cross-check.

Agreement of the `default` and `S2` starts is evidence of optimization
stability only. It does not by itself prove the physical mass scale, remove
PARSEC dependence, or quantify the width of a shape--MLR degeneracy.

## Remaining limitations

The direct likelihood is computationally more expensive than stack
interpolation. The metallicity posterior and MLR regularization are still
shared with the T8 model, including its PARSEC-relative structure and solar
anchor. Any formal scientific result should retain the fixed-shape baseline
and compare it explicitly with the alternating-MAP result rather than
presenting the recalibrated shape in isolation.
