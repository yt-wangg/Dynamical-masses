# Plan: mass-independent outlier model with unified \(\tilde u\) integration

## Goal

Refactor the likelihood so that:

1. The **good binary component** retains the physically motivated mass scaling

   $$
   u_{\rm true}=s_j\tilde u,
   \qquad
   s_j\equiv\sqrt{\frac{m_{{\rm tot},j}}{M_\odot}}.
   $$

2. The **outlier component is defined in physical velocity \(u\)-space**, with fixed physical parameters \((\mu_o,\sigma_o)\), and therefore does **not** broaden or shift when the inferred binary mass changes.

3. For numerical convenience, **both components are evaluated using the same fixed \(\tilde u\) integration grid**.

4. Do not introduce a second production integration over \(u\). The distinction between \(u\)-space and \(\tilde u\)-space concerns the **generative model**, not the numerical integration coordinate.

5. Use the direct \(u\)-space integration only as an independent validation/reference implementation. Once equivalence is established, production likelihood evaluation should use only the simplified \(\tilde u\)-space implementation.

---

# 1. Reference generative model in physical velocity space

For binary \(j\),

$$
s_j =
\sqrt{
\frac{m_{{\rm tot},j}(\theta_{\rm MLR})}{M_\odot}
}.
$$

The observational model is

$$
p(u_{{\rm obs},j}\mid u_{\rm true})
=
{\rm Rice}
\left(
u_{{\rm obs},j}
\mid
u_{\rm true},
\sigma_{u,j}
\right).
$$

The intrinsic mixture is

$$
p(u_{\rm true}\mid s_j)
=
(1-f_o)\,
p_{\rm good}(u_{\rm true}\mid s_j)
+
f_o\,
p_{\rm out}(u_{\rm true}).
$$

The good component is

$$
p_{\rm good}(u_{\rm true}\mid s_j)
=
\frac{1}{s_j}
p_{\rm good}
\left(
\frac{u_{\rm true}}{s_j}
\mid B,u_c,C
\right).
$$

The outlier component is defined directly in physical velocity:

$$
p_{\rm out}(u_{\rm true})
=
\frac{
G_o(u_{\rm true}\mid\mu_o,\sigma_o^2)
}{
C_o
}.
$$

Here \(\mu_o,\sigma_o\) are physical-velocity parameters and must not depend on \(m_{\rm tot}\), \(s_j\), or the MLR.

Therefore the reference likelihood is

$$
\begin{aligned}
\mathcal L_j
=
\int
\Bigg[
&(1-f_o)
\frac{1}{s_j}
p_{\rm good}
\left(
\frac{u_{\rm true}}{s_j}
\mid B,u_c,C
\right)
\\
&+
f_o
\frac{
G_o(u_{\rm true}\mid\mu_o,\sigma_o^2)
}{
C_o
}
\Bigg]
\\
&\qquad\times
{\rm Rice}
\left(
u_{{\rm obs},j}
\mid
u_{\rm true},
\sigma_{u,j}
\right)
\,du_{\rm true}.
\end{aligned}
$$

Treat this equation as the **scientific definition of the model**.

The production implementation may transform variables for numerical efficiency, but it must remain mathematically equivalent to this expression.

---

# 2. Production calculation: use one \(\tilde u\) grid

Use

$$
u_{\rm true}=s_j\tilde u,
\qquad
du_{\rm true}=s_j\,d\tilde u.
$$

The Rice scale identity is

$$
{\rm Rice}
(u_{\rm obs}\mid s_j\tilde u,\sigma_u)
=
\frac{1}{s_j}
{\rm Rice}
\left(
\frac{u_{\rm obs}}{s_j}
\middle|
\tilde u,
\frac{\sigma_u}{s_j}
\right).
$$

After the transformation, evaluate both mixture components using the same fixed \(\tilde u\) grid.

The production likelihood should be

$$
\boxed{
\begin{aligned}
\mathcal L_j^{\rm mix}
=
\int
\Bigg[
&(1-f_o)
\frac{
p_{\rm good}(\tilde u\mid B,u_c,C)
}{s_j}
\\
&+
f_o
\frac{
G_o(s_j\tilde u\mid\mu_o,\sigma_o^2)
}{
C_o
}
\Bigg]
\\
&\qquad\times
{\rm Rice}
\left(
\frac{u_{{\rm obs},j}}{s_j}
\middle|
\tilde u,
\frac{\sigma_{u,j}}{s_j}
\right)
d\tilde u .
\end{aligned}
}
$$

Important:

$$
\boxed{
\text{good term has an explicit }1/s_j
}
$$

while, when written as

$$
G_o(s_j\tilde u\mid\mu_o,\sigma_o^2),
$$

the outlier term has **no additional \(1/s_j\)**.

This is not a missing Jacobian. In the outlier contribution, the \(s_j\) from

$$
du_{\rm true}=s_jd\tilde u
$$

cancels the \(1/s_j\) introduced by transforming the Rice density.

Only if the outlier density itself is rewritten as a Gaussian in normalized coordinates,

$$
G_o(s_j\tilde u\mid\mu_o,\sigma_o^2)
=
\frac1{s_j}
G_o
\left(
\tilde u
\middle|
\frac{\mu_o}{s_j},
\frac{\sigma_o^2}{s_j^2}
\right),
$$

does an explicit \(1/s_j\) reappear.

Prefer the physical-space evaluation

$$
G_o(s_j\tilde u\mid\mu_o,\sigma_o^2)
$$

in production code because it makes the scientific meaning harder to accidentally change.

---

# 3. Simplified internal calculation

For every likelihood evaluation:

1. Compute

   $$
   s_j=\sqrt{m_{{\rm tot},j}/M_\odot}.
   $$

2. Keep one global integration grid

   $$
   \tilde u_k.
   $$

3. Evaluate the good shape

   $$
   p_{{\rm good},k}
   =
   p_{\rm good}(\tilde u_k\mid B,u_c,C).
   $$

4. Construct physical true velocities only as a broadcasted derived array:

   $$
   u_{{\rm true},jk}
   =
   s_j\tilde u_k.
   $$

5. Evaluate

   $$
   p_{{\rm out},jk}
   =
   \frac{
   G_o(u_{{\rm true},jk}\mid\mu_o,\sigma_o^2)
   }{C_o}.
   $$

6. Evaluate one Rice kernel

   $$
   R_{jk}
   =
   {\rm Rice}
   \left(
   \frac{u_{{\rm obs},j}}{s_j}
   \middle|
   \tilde u_k,
   \frac{\sigma_{u,j}}{s_j}
   \right).
   $$

7. Form

   $$
   I_{jk}
   =
   \left[
   (1-f_o)\frac{p_{{\rm good},k}}{s_j}
   +
   f_o p_{{\rm out},jk}
   \right]R_{jk}.
   $$

8. Perform one numerical integration:

   $$
   \mathcal L_j
   =
   \int I_j(\tilde u)d\tilde u.
   $$

Conceptually, the production path should reduce to

```text
s = sqrt(mtot / Msun)

u_true_grid = s[:, None] * tilde_u_grid[None, :]

good_pdf = good_shape(tilde_u_grid, shape_params)
outlier_pdf = outlier_pdf_u(u_true_grid, mu_out, sigma_out)

rice = Rice(
    u_obs[:, None] / s[:, None],
    tilde_u_grid[None, :],
    sigma_u[:, None] / s[:, None]
)

integrand = (
    (1 - f_out) * good_pdf[None, :] / s[:, None]
    + f_out * outlier_pdf / C_out
) * rice

likelihood = integrate_over_tilde_u(integrand)
```

Adapt broadcasting to the existing codebase, but preserve this mathematical structure.

---

# 4. HARD ACCEPTANCE CRITERION: direct \(u\) integration vs. \(\tilde u\) integration

This is a **mandatory acceptance criterion for the refactor**, not an optional diagnostic.

Implement an independent reference likelihood that evaluates the original physical-space expression directly:

$$
\mathcal L_j^{(u)}
=
\int
p(u_{\rm true}\mid s_j)
{\rm Rice}
(u_{{\rm obs},j}\mid u_{\rm true},\sigma_{u,j})
\,du_{\rm true}.
$$

Separately evaluate the production implementation:

$$
\mathcal L_j^{(\tilde u)}
=
\int
I_j(\tilde u)\,d\tilde u.
$$

Require

$$
\boxed{
\mathcal L_j^{(u)}
\simeq
\mathcal L_j^{(\tilde u)}
}
$$

to within an explicitly chosen numerical integration tolerance.

Test this over a representative grid of:

* low, typical, and high \(m_{\rm tot}\);
* several \(u_{\rm obs}\);
* several \(\sigma_u\);
* different \(f_o\);
* several outlier parameters;
* several good-shape parameters;
* pure-good \(f_o=0\);
* pure-outlier \(f_o=1\);
* intermediate mixtures.

Also compare log likelihoods where appropriate.

### This test is a merge/blocking condition

Do **not** consider the likelihood refactor complete merely because optimization runs, plots look reasonable, or recovered MLR values appear plausible.

The modification is accepted only after the independent direct-\(u\) implementation and the production \(\tilde u\) implementation agree within numerical tolerance.

This test is specifically intended to catch:

* missing Jacobians;
* duplicated Jacobians;
* incorrect Rice rescaling;
* accidental mass scaling of the outlier model;
* incorrect Gaussian parameter transformations;
* normalization errors;
* integration-grid truncation errors.

Once this equivalence is established, retain the direct-\(u\) calculation as a **test/reference implementation only**, not as part of the production inference path.

---

# 5. Additional required validation

Verify explicitly that the physical outlier distribution is invariant under changes in \(s\):

$$
p_{\rm out}(u\mid s_1)
=
p_{\rm out}(u\mid s_2).
$$

Verify

$$
G_o(s\tilde u\mid\mu_o,\sigma_o^2)
=
\frac1s
G_o
\left(
\tilde u
\middle|
\frac{\mu_o}{s},
\frac{\sigma_o^2}{s^2}
\right).
$$

Verify pure-component limits \(f_o=0\) and \(f_o=1\).

Verify convergence with respect to the \(\tilde u\) grid extent and resolution.

If \(G_o/C_o\) is truncated, verify that \(C_o\) corresponds to the intended physical-\(u\) definition and is not accidentally made binary-dependent by the numerical coordinate transformation.

---

# 6. Remove obsolete and redundant numerical machinery

After the new likelihood passes the hard \(u\)-vs-\(\tilde u\) equivalence test, simplify the production code aggressively.

The goal is **not** to preserve old numerical machinery merely because it already exists.

Inspect the old likelihood path and remove calculations that are no longer scientifically or numerically necessary.

In particular, remove or consolidate, where applicable:

* duplicate evaluation of the same Rice kernel;
* separate good/outlier quadratures when both can use the same \(\tilde u\) grid;
* redundant conversions back and forth between \(u\) and \(\tilde u\);
* repeated computation of \(s_j\), \(1/s_j\), \(u_{\rm obs}/s_j\), or \(\sigma_u/s_j\);
* unnecessary construction of multiple equivalent velocity grids;
* old fixed-\(\tilde u\) outlier PDFs that encode the obsolete mass-scaling assumption;
* legacy normalization paths that are no longer part of the new generative model;
* duplicated Gaussian evaluations in both physical and normalized coordinates;
* obsolete interpolation or integration branches retained only for the previous likelihood formulation;
* intermediate arrays that can be eliminated through broadcasting without harming readability or numerical stability;
* compatibility code for the old outlier model unless it is explicitly needed for a regression test;
* repeated normalization or numerical integration that can be computed analytically or cached safely;
* old diagnostic calculations that are no longer consumed by tests, outputs, or scientific diagnostics.

Do not remove a computation merely because it looks redundant. First determine whether it contributes to:

1. the actual likelihood;
2. numerical stability;
3. required normalization;
4. a scientifically meaningful diagnostic;
5. a validation test.

If it serves none of these purposes, remove it.

---

# 7. Keep reference calculations out of the production likelihood

There should be a clear distinction between:

### Production implementation

Use only the efficient unified \(\tilde u\)-grid likelihood required during optimization/MCMC/alternating-MAP.

### Validation implementation

Keep the direct \(u\)-space integral and any mathematically equivalent alternative forms only in tests or dedicated validation utilities.

Do not calculate both forms during normal inference.

Do not keep two parallel likelihood implementations active in the fitting loop “for safety”.

Correctness should be established by tests, not by repeatedly performing redundant calculations during every likelihood evaluation.

---

# 8. Prefer analytic simplification before numerical computation

Whenever an expression can be simplified algebraically without changing the model, simplify it before numerical evaluation.

In particular, do not numerically implement Jacobian factors that analytically cancel only to multiply/divide them again later.

The transformed likelihood should be coded in its already-simplified form:

$$
\left[
(1-f_o)\frac{p_{\rm good}(\tilde u)}{s_j}
+
f_o\frac{G_o(s_j\tilde u)}{C_o}
\right]
R_j(\tilde u).
$$

Avoid a computational sequence such as

```text
physical_pdf
-> transform_pdf
-> apply_jacobian
-> transform_rice
-> apply_inverse_jacobian
-> multiply
```

when the analytically reduced expression is already known.

The code should mirror the final mathematical likelihood as directly as possible.

---

# 9. Performance and clarity objective

After correctness is established, profile the likelihood implementation.

Optimize only meaningful bottlenecks.

Prefer:

* vectorization across binaries and/or integration nodes;
* precomputation of quantities independent of current fit parameters;
* caching only when cache validity is unambiguous;
* one common integration grid;
* one common integration routine;
* minimal temporary arrays;
* numerically stable log-likelihood accumulation.

Do not introduce complicated numerical optimization that obscures the generative model for negligible performance gain.

The final implementation should make it easy for a future reader to see the correspondence

$$
\text{equation}
\longleftrightarrow
\text{code}.
$$

---

# 10. Keep the outlier refactor separate from good-shape calibration

There are two distinct scientific changes:

### A. Correct the outlier generative model

$$
p_{\rm out}(u)
$$

is fixed in physical velocity space and independent of inferred mass.

### B. Allow constrained good-shape calibration

$$
p_{\rm good}(\tilde u)
\rightarrow
p_{\rm good}(\tilde u\mid B,u_c,C).
$$

Do not mix these two changes unnecessarily in the implementation.

The corrected likelihood should work for both:

* fixed physical good shape;
* alternating-MAP calibrated good shape.

This separation is important for both testing and scientific interpretation.

---

# 11. Final acceptance checklist

The refactor is complete only when all of the following are true:

* [ ] The outlier distribution is defined by fixed \((\mu_o,\sigma_o)\) in physical \(u\)-space.
* [ ] Changing \(m_{\rm tot}\) does not shift or broaden the physical outlier distribution.
* [ ] Production inference uses one common \(\tilde u\) integration grid.
* [ ] Good and outlier components share the same Rice evaluation where possible.
* [ ] There is no unnecessary second \(u\)-space quadrature in production.
* [ ] Direct \(u\)-space and transformed \(\tilde u\)-space likelihoods agree within numerical tolerance.
* [ ] **The \(u\)-vs-\(\tilde u\) equivalence test is treated as a hard acceptance/merge criterion.**
* [ ] Pure-good and pure-outlier limits pass.
* [ ] Grid convergence has been checked.
* [ ] Outlier normalization has been verified in the intended physical domain.
* [ ] Obsolete fixed-\(\tilde u\) outlier code has been removed from production.
* [ ] Redundant transformations, integrations, normalization operations, and intermediate arrays have been removed where safe.
* [ ] Direct-\(u\) integration remains only as a validation/reference implementation.
* [ ] The production code closely mirrors the simplified mathematical likelihood.
* [ ] Fixed-good-shape and alternating-MAP modes both use the same corrected underlying likelihood.

## Guiding principle

> Define each distribution in the physically intended variable space, but choose the numerical integration variable independently for computational convenience.

For this model,

$$
\boxed{
p_{\rm good}(u\mid m_{\rm tot})
\leftrightarrow
p_{\rm good}(\tilde u\mid B,u_c,C)
}
$$

while

$$
\boxed{
p_{\rm out}(u)
=
\text{mass-independent physical outlier distribution}.
}
$$

Both are nevertheless evaluated efficiently using

$$
\boxed{
\int d\tilde u.
}
$$

The direct \(u\)-space formulation is the scientific reference; the unified \(\tilde u\)-space formulation is the production implementation; agreement between them is the mandatory proof that the transformation has been implemented correctly.
