# Post-`7bc861a` model improvements and sensitivity-study history

This note documents why the dynamical-mass pipeline changed after
`7bc861a2f004db9caec0527410c1ef732bc113a2`, which changes are now treated as
core model/correctness improvements, and which are retained as sensitivity
experiments.

## 1. Baseline and motivating discrepancy

The reference baseline is commit `7bc861a`. In that version, both the normal
and outlier velocity components were expressed in the scaled coordinate

[
\tilde u = u / \sqrt{M_{\rm tot}/M_\odot}.
]

The initial motivation for revisiting the model came from comparing the inferred
MLR with PARSEC and other published MLRs. In the magnitude range driving the
discrepancy, the dynamical MLR could be tens of percent high, reaching roughly
40% in the motivating comparison. That was too large to treat as a small
calibration offset.

The post-`7bc861a` work therefore proceeded as a sequence of mechanism tests:

1. localize which systems and mixture branches favor larger masses;
2. audit component normalization, Jacobians, interpolation, and posterior
   bookkeeping;
3. remove artificial mass dependence from the outlier branch;
4. test sensitivity to the normal-component velocity-shape parameters;
5. compare fixed-shape, jointly sampled-shape, and alternating-MAP/EM-style
   updates from different starting shapes.

## 2. Score diagnostic: locating upward mass pressure

A local (M_G)-window score diagnostic perturbs masses through

[
\ln m_k \rightarrow \ln m_k + \epsilon h(M_{G,k})
]

and evaluates the per-system likelihood response

[
S_j = \left.\frac{\partial \ln L_j}{\partial \epsilon}\right|_{\epsilon=0}.
]

The diagnostic was hardened to use all posterior chains, distinguish the local
linear score from an exact finite (+10\%) mass step, enforce strict lookup
domains, compare chain-rule derivatives against JAX autodiff, compare lookup
slopes against direct adaptive Rice integration, and require finite/conservative
summary outputs.

The important scientific result was that upward mass pressure was concentrated
in the fast-velocity tail and systems with substantial outlier responsibility.
Both normal and outlier branches contributed, focusing the next tests on the
mixture construction and velocity-shape assumptions.

## 3. Exact finite-support normalization

The raw good-component basis over its finite support had integral

[
C_{\rm good}=0.9978600946
]

rather than exactly one. T8.1 divides by this constant so that each mixture
component is normalized. For a scaled-coordinate component the observed-space
density carries the (1/s) Jacobian, with
(s=\sqrt{M_{\rm tot}/M_\odot}).

Tests now cover component normalization, mixture normalization, and the failure
mode in which an outlier branch omits its Jacobian. In that broken case the
observed-space integral becomes mass dependent and can directly reward larger
inferred masses.

The 0.2% good-basis normalization correction itself is far too small and too
mass independent to explain the motivating tens-of-percent MLR discrepancy.
It is retained as a correctness fix, not as the physical explanation.

## 4. Core model change: mass-independent raw-(u) outlier

The key post-baseline model change concerns the outlier coordinate.

### Legacy `7bc861a` outlier

The outlier was defined in (\tilde u=u/s), so its likelihood changed when the
MLR changed the inferred mass:

[
p_{\rm out}^{\rm legacy}(u\mid M)=\frac{1}{s}q(u/s).
]

This lets systems with high outlier responsibility improve their outlier
likelihood partly by moving the inferred mass scale.

### Revised outlier

The revised outlier is defined directly in observed raw (u):

[
L_{\rm out,j}
=
\int_0^{80}
p_{\rm Rice}(u_j\mid v,\sigma_{u,j})
\,\mathrm{TN}(v;40,13,[0,80])\,dv .
]

It is independent of mass and metallicity. The normal component still carries
the intended dynamical mass information through (u/\sqrt{M_{\rm tot}}).

The core library now exposes this raw-(u) Rice-convolved outlier likelihood,
and the main T8 runner uses it by default. The option

`--outlier-coordinate scaled_tilde_u`

retains the legacy `7bc861a` formulation explicitly for reproduction and
sensitivity comparisons.

The mass-independent-outlier experiments show that removing the outlier/mass
coupling reduces the inferred mass scale, confirming that the legacy outlier
model contributed to the upward pressure. The discrepancy nevertheless
remained, so this was not the full explanation.

## 5. Remaining discrepancy: sensitivity to the normal velocity shape

After removing outlier mass coupling, the inferred MLR remained sensitive to
parameters controlling the normal (\tilde u) distribution: (B), (u_c),
and (C).

This motivated three complementary experiments.

### 5.1 Fixed-shape sensitivity

The normal-shape constants can be overridden while refitting the MLR. These runs
directly answer how much the inferred MLR moves under plausible changes in the
normal velocity kernel.

### 5.2 Joint shape + MLR MCMC

A 27-node grid in ((\log B,\log u_c,\log C)) allows the good-shape
parameters to be sampled jointly with the MLR. This promotes the
shape--mass degeneracy into posterior covariance.

The associated correctness fixes retained in core include:

- correct stack indexing,
  (k=9i_B+3i_{u_c}+i_C), matching the builder loop order;
- conversion of physical NumPyro starting values to unconstrained sampler
  coordinates;
- strict lookup-domain handling;
- direct continuous Rice quadrature as an independent reference;
- tests for node reproduction, interpolation weights, save/load behavior, and
  initialization transforms;
- extension of the lower (u_c) prior support after exploratory runs repeatedly
  contacted the previous lower boundary.

Any numerical quick-run result produced before the stack-index correction, or
while the posterior was pressing against a prior boundary, should not be treated
as a final scientific measurement. The robust conclusion is qualitative:
allowing the normal velocity shape to vary changes the inferred MLR
substantially, demonstrating a real shape--mass degeneracy.

### 5.3 Alternating conditional-MAP / EM-style experiment

A separate experiment alternates:

1. optimize the MLR with the good-shape parameters held fixed;
2. optimize (B,u_c,C) and the outlier fraction with the MLR held fixed;
3. repeat until the objective is approximately stationary.

This is EM-style alternating optimization, not a strict expectation-maximization
algorithm and not a full posterior calculation.

Two shape starts are retained: the default/T8 start and an S2 start. The purpose
is to test sensitivity to initialization and local optima, not to claim that
agreement between endpoints proves the physical model is correct.

The alternating-MAP path is therefore retained as a sensitivity/optimization
workflow rather than the sole default scientific inference.

## 6. Core changes retained relative to `7bc861a`

The following are treated as core model or correctness improvements:

1. exact finite-support component normalization;
2. consistent scaled-coordinate Jacobian accounting;
3. mass-independent raw-(u) outlier as the default contamination model;
4. explicit legacy mass-coupled outlier option for controlled reproduction;
5. strict lookup-domain behavior rather than silent clamping;
6. all-chain posterior handling and draw-count consistency checks;
7. corrected shape-stack axis indexing;
8. correct NumPyro initialization transforms;
9. direct Rice-quadrature reference evaluation;
10. NaN-safe responsibilities and finite-output validation;
11. metadata/provenance for normalization, shape constants, outlier semantics,
    and experiment configuration.

## 7. Sensitivity experiments intentionally retained

The repository keeps the following because they explain why the model changed
and provide a reproducible robustness trail:

- (M_G)-window score diagnostic;
- legacy mass-coupled versus raw-(u) outlier comparison;
- fixed (B,u_c,C) good-shape sensitivity;
- 27-node joint shape + MLR inference;
- direct continuous shape quadrature checks;
- independent metallicity-bin MCMC;
- alternating conditional-MAP/EM-style fits from default and S2 starts;
- comparisons against PARSEC and external MLRs.

These are controlled comparisons and should not be silently conflated with the
definition of the core model.

## 8. Recommended presentation sequence

For explaining the model evolution:

**A. Baseline `7bc861a`.** Show the original MLR and the PARSEC/external
comparison, including the region where inferred masses were up to roughly 40%
higher.

**B. Likelihood-pressure diagnostic.** Show that the upward pressure is
concentrated in fast/high-outlier-responsibility systems.

**C. Outlier intervention.** Compare legacy mass-coupled and raw-(u)
mass-independent outliers with the good shape held fixed.

**D. Good-shape sensitivity.** With the revised outlier fixed, vary
(B,u_c,C) and refit the MLR.

**E. Shape freedom without alternating MAP.** Use joint shape + MLR MCMC to
propagate velocity-shape uncertainty into the inferred MLR.

**F. Alternating-MAP/EM-style comparison.** Run from both default and S2 shape
starts and compare with the joint-MCMC and fixed-shape results.

This sequence separates the two main mechanisms: contamination-model mass
coupling and normal-kernel/MLR degeneracy.

## 9. What the current experiments establish

The experiments support the following conclusions:

- the original mass-coupled outlier branch can push the inferred mass scale
  upward;
- removing that coupling reduces but does not eliminate the discrepancy;
- the remaining MLR is materially sensitive to the normal (\tilde u)
  distribution;
- velocity-shape uncertainty therefore belongs in the model/sensitivity budget;
- a fixed-shape result understates this source of model dependence.

They do not establish that any single revised MLR is the uniquely correct
absolute mass scale. The result still depends on PARSEC-relative regularization,
the solar anchor, metallicity calibration, selection, and the assumed velocity
families.

## 10. Reproducibility map

Reference baseline:

`7bc861a2f004db9caec0527410c1ef732bc113a2`

Important post-baseline development commits:

- `7550d88`: initial (M_G)-window score diagnostic;
- `b0017cf`: all-chain/domain/finite-step diagnostic hardening;
- `9803200`: exact T8.1 normalization and fixed-shape sensitivity;
- `e493728`: stricter score validation and shape-stack infrastructure;
- `c7513da`: joint good-shape + MLR sampling;
- `299b0db`, `3297c48`: lower (u_c) prior support extended after boundary
  contact;
- `84204da`: mass-independent-outlier experiments, metallicity-bin pipeline,
  direct shape evaluation, stack-index correction, and sampler-init correction;
- `cda7fc3`: alternating-MAP workflow and entry point.

The current integration keeps the scientifically useful experiment trail while
promoting only the justified likelihood/correctness changes into the core model.
