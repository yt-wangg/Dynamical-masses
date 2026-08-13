# FeH-global real-data workflow

## Input and metallicity convention

The runner reads
`data/jd_msms_single_bic_1kpc_filtered_cmdcut_cutb_jcaps_err_mhcal_good.fits`.
This table is already quality-selected.  One system metallicity is used for both
components:

- observed value: `jc_m_h_fit_cal_1`;
- 1-sigma error: `jc_sigma_m_h_cal_1`.

This is the primary's measurement, not an average of the two component fits.
The mass model nevertheless treats it as the shared metallicity of the coeval
binary.

## Three nested held-out runs

Run all commands with exactly the same held-out fraction and seed.  The three
stages are nested:

1. `core`: `a0 + a1*x + b1*z`;
2. `quadratic`: core plus `a2*x^2 + b2*z^2`;
3. `cross`: quadratic plus the free interaction `c_xy*x*z`.

```bash
conda run -n dyn python src/examples/run_on_data_feh_global.py \
  --stage core --heldout-fraction 0.2 --heldout-seed 20260812

conda run -n dyn python src/examples/run_on_data_feh_global.py \
  --stage quadratic --heldout-fraction 0.2 --heldout-seed 20260812

conda run -n dyn python src/examples/run_on_data_feh_global.py \
  --stage cross --heldout-fraction 0.2 --heldout-seed 20260812

conda run -n dyn python src/examples/run_on_data_feh_global.py --compare-only
```

The default is to save the full chain-aware MCMC record.  Use
`--no-save-full-mcmc` only for a disposable test.  The grouped NPZ contains
posterior arrays plus `diverging`, `energy`, `potential_energy`, `num_steps`,
`accept_prob`, and derived `tree_depth`.

After fitting only the training systems, the runner evaluates the same Rice,
outlier, FeH-quadrature, and inferred FeH-population likelihood on each held-out
system for a subset of posterior draws.  For held-out system `i`, it records

```text
lppd_i = log(mean_s(exp(log p(y_i | theta_s))))
```

and compares models using the paired sum of `lppd_i` differences.  The reported
standard error is

```text
sqrt(N_test * variance(lppd_i(model B) - lppd_i(model A))).
```

Prefer the larger held-out lppd.  A small positive difference relative to its
paired standard error is not convincing evidence for the larger model.  In
particular, retain `c_xy` only if `cross - quadratic` is reproducibly positive.
Repeat the comparison with a second split seed, or ultimately use several
fixed folds, before treating a marginal difference as scientific evidence.

The default `--heldout-posterior-samples 256` controls evaluation cost, not the
MCMC run.  Repeat the comparison with 512 draws to check Monte Carlo stability
if two stages are close.

## Metallicities outside [-1, 0.6]

The code does not hard-cut systems whose observed metallicity lies outside the
isochrone interval.  With FeH uncertainty enabled, their latent true
metallicity is integrated only over `[-1, 0.6]`.  This keeps more information
than deleting rows: an observation just below -1 with a broad error can still
support latent values above -1.

There is nevertheless a real support limitation.  Values well below -1 with
small errors all collapse against the same latent boundary.  They cannot teach
the model how the mass relation behaves below -1, and a large number of them
can create a boundary pile-up in the inferred population density.  The omitted
normalization of the truncated-Gaussian quadrature proposal depends only on
the observed data and fixed bounds, so it does not alter parameter inference
or comparisons that use the identical test set and bounds.

Recommended order of preference:

1. Extend the isochrone grid to cover the scientifically required metallicity
   range, then extend `feh_min` consistently.  Changing `feh_min` without an
   isochrone grid is not sufficient because the baseline would still clip.
2. If the scientific target is explicitly the current isochrone domain, retain
   the error marginalization but report the posterior probability that each
   system lies inside the domain.  Perform a sensitivity run excluding systems
   with negligible in-domain probability.
3. Do not silently clip observed metallicities to -1.  That turns different
   measurements into identical exact values and understates uncertainty.

The new quality-selected table still has a non-negligible observed tail below
-1, so boundary sensitivity should be reported alongside the main result.

## Why a soft derivative penalty is not a hard monotonic model

The current surface is

```text
log10 M(x,z) = log10 M_iso(x,z) + delta(x,z).
```

A grid penalty subtracts a finite log-probability when a derivative has the
wrong sign.  It therefore discourages violations but can never guarantee their
absence.  The numerical effect of a quoted penalty strength also depends on
the units, grid spacing, and the fact that the code averages squared
violations.  A value such as 10 can consequently be negligible.

Simply forcing the derivative of `delta` to be negative is also unnecessarily
strong.  The physical condition concerns the total mass surface:

```text
d log10(M_iso) / d M_G + d delta / d M_G <= 0.
```

The isochrone baseline is already decreasing, so a mildly increasing
correction can still produce a valid total relation.

## Recommended smooth hard-monotone parameterization

For monotonicity in absolute magnitude, parameterize the derivative of the
*total* log mass rather than assigning unconstrained polynomial coefficients.
One useful construction is

```text
s_k(z) = -softplus(eta_k(z))
d log10 M(M_G,z) / d M_G = sum_k B_k(M_G) * s_k(z)
```

where `B_k` are non-negative B-spline basis functions.  Every `s_k` is negative
and every `B_k` is non-negative, so the derivative is non-positive everywhere,
not only at a checking grid.  Integrate the derivative basis from a reference
magnitude and add a reference-level function:

```text
log10 M(M_G,z) = h(z) + sum_k I_k(M_G) * s_k(z),
```

where `I_k` are integrated B-splines.  Choose the integration origin at the
solar anchor; the solar constraint then acts mainly on `h(0)` instead of on a
strongly correlated combination of all coefficients.

Let `eta_k(z)` be a low-order spline in metallicity with hierarchical shrinkage
across neighbouring magnitude knots.  The `softplus` transform is smooth, so
NUTS sees no discontinuous `-inf` boundary.  Eight to twelve magnitude basis
functions and roughly four to six metallicity basis functions are a reasonable
starting point, followed by held-out tuning of the smoothing scales.

An equivalent discrete construction is a monotone lattice:

```text
g_0(z) = free reference log mass
g_{j+1}(z) = g_j(z) - softplus(rho_j(z))
```

at ordered magnitude knots, with monotone interpolation between knots.  This
is often easier to implement and debug than an I-spline and has the same key
property: all adjacent mass differences have the correct sign by construction.

Metallicity monotonicity is a separate scientific assumption.  It should not
be inherited automatically from magnitude monotonicity.  If it is justified,
use positive/negative cumulative increments along the metallicity axis as
well.  Enforcing both axes simultaneously is most transparent with a
two-dimensional monotone lattice whose adjacent increments are transformed by
`softplus`.  If the expected sign with metallicity changes with magnitude, do
not impose global FeH monotonicity; regularize the interaction instead and let
held-out prediction decide whether it is supported.

Before replacing the current polynomial, validate the constrained model with:

1. prior-predictive curves and derivative maps;
2. posterior probability of any derivative violation on a much finer grid
   (it should be exactly zero up to numerical precision);
3. recovery on mock data, including boundary metallicities;
4. the same fixed held-out split used for the three polynomial stages;
5. divergence, BFMI, tree-depth, and posterior predictive checks.

