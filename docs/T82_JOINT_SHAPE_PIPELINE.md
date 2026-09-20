# T8.2 independent metallicity-bin pipeline

The default fitting method is the [EM-style workflow](EM_MLR_WORKFLOW.md).
This joint MCMC pipeline is selected explicitly for posterior intervals and
independent metallicity-bin comparisons.

`src/examples/run_t82_joint_shape_mcmc.py` jointly samples the monotone MLR,
normal-component velocity shape, and outlier fraction within each metallicity
bin. The bins have independent parameters and no pooling. Both start labels use
the same PARSEC-projected initial MLR:

- `parsec`: B=0.002544, uc=35.67, C=3.1.
- `s2`: B=0.00103, uc=40.98, C=11.53.

Systems are assigned using their saved latent [M/H] posterior median, with
boundaries [-1, -0.5, 0, 0.3, 0.6]. Each assigned system retains its complete
81-node metallicity posterior in the likelihood. The formal sample contains
186, 9618, 4687, and 385 systems in these bins.

The outlier density is a Rice-convolved TN(40,13,[0,80]) in raw u, independent
of mass and metallicity. The normal density depends on u/sqrt(M1+M2).
The 8 by 4 monotone MLR surface retains PARSEC-relative regularization and a
solar anchor at M_G=4.67, [M/H]=0. The model is not independent of PARSEC.

## Inputs and execution

Run from the repository root in an environment containing the repository
requirements. Garching uses `/home/jdli/nexus/collab/Dynamical-masses` and
`/home/jdli/nexus/miniforge3/envs/dyn/bin/python`. The observed FITS, calibrated
metallicity posterior, and shape tables remain external data products.
See [T8 server instructions](T8_SERVER_RUN.md) for their construction.

For memory-mapped table access, unpack the saved stack once:

```python
from pathlib import Path
import numpy as np

source = Path('results/hierarchical_metallicity_t8_2_formal')
target = source / 'shapestack_unpacked_20260919'
target.mkdir(exist_ok=True)
with np.load(source / 'dynamics_likelihood_shapestack_t8.npz') as stack:
    for name in ['row_indices', 'sqrt_mtot_grid', 'log_good_stack', 'log_bad_stack']:
        np.save(target / (name + '.npy'), stack[name])
```

One full-bin run, with the same sampling lengths as the current Garching jobs:

```bash
export JAX_PLATFORMS=cpu
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
python src/examples/run_t82_joint_shape_mcmc.py \
  --data data/jd_msms_single_bic_1kpc_filtered_cmdcut_cutb_jcaps_err_mhcal_good.fits \
  --posterior results/hierarchical_metallicity_t8_1_formal/latent_metallicity_weights_t8.npz \
  --nodes-json results/hierarchical_metallicity_t8_2_formal/shapestack_nodes_t8.json \
  --shape-stack-unpacked results/hierarchical_metallicity_t8_2_formal/shapestack_unpacked_20260919 \
  --output results/t82_joint_shape_metal_bins_corrected_start/bin0_parsec_jitter \
  --bin-index 0 --start parsec --warmup 1500 --samples 2000 --chains 4 \
  --initial-jitter 0.03 --seed 20260921 --skip-validation
```

Use bin indices 0 through 3 and labels `parsec` and `s2`, with separate output
directories. The S2 jobs use seed 20260923. No subsampling flag is supplied.
The command uses a probability mixture of the 27 shape-node likelihoods with
linear weights in log B, log uc, log C; it is not a direct evaluation of the
analytic density at the interpolated parameters. Omitting
`--shape-stack-unpacked` selects direct quadrature instead. The optional
adaptive-reference validation tests that direct quadrature only.

Each run saves `run_config.json`, `mlr_mcmc_t8.npz`, `mlr_summary_t8.csv`,
`mlr_diagnostics_t8.json`, and `run_timing.json`. Initial jitter is applied to
physical parameter values before converting them to NumPyro's unconstrained
sampler coordinates. The grouped posterior preserves chain and draw axes.

## Plotting and related experiments

`PYTHONPATH=src python src/examples/plot_t82_completed_bins.py` plots bins 0
and 3 from `results/t82_joint_shape_metal_bins_formal_20260919/`, using its
`bin{index}_{start}_jitter` subdirectories. It evaluates both posteriors and
PARSEC at bin midpoints -0.75 and +0.45, showing pointwise posterior medians,
16–84% intervals, and relative mass residuals. All 8000 saved draws per run are
used. The plot script also saves a numerical summary and PDF.

The fixed-shape outlier comparison uses `run_mass_independent_outlier.py`,
`scripts/run_t8_mio_chains.sh`, and `merge_mass_independent_outlier_chains.py`.
`run_em_mlr_pilot.py` is an alternating conditional MAP experiment, not a strict
EM algorithm or posterior sampler. `plot_em_parsec.py` plots its saved endpoints;
`plot_mlr_residuals_parsec.py` plots the fixed-shape comparisons.

## Limitations of the saved September 19 runs

The existing `t82_joint_shape_metal_bins_formal_20260919` runs pass jittered
physical starts directly to the unconstrained sampler interface. Their labels
do not establish that the actual starts were the requested S2 and default
shapes. Their posterior curves and within-run diagnostics can be inspected,
but a controlled comparison of those specific starts requires new runs with
the initialization conversion in the current pipeline.

Shape-table interpolation accuracy and float32 precision sensitivity remain
unverified for the full-bin runs. `--skip-validation` is recorded in the run
configuration and does not establish numerical accuracy. The shared formal
metallicity calibration has three divergences. The alternating-MAP pilot uses
smoothstep interpolation and a shape-half-step stopping criterion; its reported
convergence is not a full-cycle convergence test. These limitations must be
resolved before interpreting mass offsets as a physical failure of PARSEC.
