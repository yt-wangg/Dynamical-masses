# Explaining the post-`7bc861a` improvements: scientific logic and plotting plan

## 0. Goal of this document

This document explains the scientific logic of the improvements made relative to the `7bc861a` baseline, and proposes a figure plan for presenting them clearly to my advisor.

The main goal is **not** to say “I changed some code”, but to tell a coherent scientific story:

> The `7bc861a` baseline produced an MLR that is about 40% higher than other mass--luminosity relations.  
> So the key question became: **why is it high, and is that high result physically reasonable?**

The update therefore focuses on **diagnosing which modeling assumptions systematically push the inferred masses upward**, and then showing how the inferred MLR changes after targeted improvements.

---

## 1. Core scientific narrative

### 1.1 Starting point: the baseline discrepancy

The starting point is the `7bc861a` baseline result:

- compared with PARSEC / empirical mass--luminosity relations,
- my inferred MLR is systematically higher,
- by roughly **40%** in mass normalization.

This is the motivation for the entire update.

The scientific question is:

> Is this 40% offset a real astrophysical signal, or is it driven by specific modeling choices in the inference pipeline?

So the update is framed as a **diagnostic study of the mass normalization**.

---

### 1.2 Overall logic of the update

The update follows the following logic:

1. **Identify the symptom**  
   The baseline MLR is high by about 40%.

2. **Run sensitivity tests**  
   Check which parts of the model tend to bias the inferred masses upward.

3. **Make targeted improvements**  
   Two key improvements emerged:
   - **(A) Outlier model change:** outlier modeling should be built on **raw \(u\)** rather than **\(\tilde u\)**.
   - **(B) Good-shape sensitivity:** the inferred MLR is highly sensitive to the assumed good-binary velocity shape, motivating a **variable good-shape** version.

4. **Show the effect of each step separately**  
   I want to present the change in results after:
   - Step A only
   - Step B only
   - and optionally both together

This makes the scientific reasoning transparent.

---

## 2. Scientific interpretation of the baseline

### 2.1 Why the baseline was physically motivated

In the original approach, the good-binary velocity shape was not an arbitrary empirical function.  
It was motivated by orbital physics: starting from Keplerian binaries and marginalizing over nuisance variables such as eccentricity distribution, orbital orientation, and orbital phase, one obtains a theoretical distribution for

\[
\tilde u \equiv \frac{u}{\sqrt{M_{\rm tot}}}.
\]

So the original philosophy was:

- the **good shape** is externally motivated by orbital-population physics,
- while the **MLR** is the quantity to be inferred from the data.

Therefore, fixing the good shape in the baseline was scientifically natural. :contentReference[oaicite:2]{index=2}

### 2.2 Why the baseline needed to be questioned

However, the baseline result revealed two concerns:

1. the inferred MLR is much higher than PARSEC / other empirical relations;
2. the inferred MLR is highly sensitive to the assumed shape of \(p(\tilde u)\).

This means the baseline discrepancy cannot simply be ignored.  
Instead, it motivates a structured investigation into **which assumptions are controlling the mass normalization**. :contentReference[oaicite:3]{index=3}

---

## 3. Key scientific question behind the update

The central diagnostic question is:

> Which parts of the model tend to increase the inferred masses?

This is more precise than just asking whether the final MLR is “good” or “bad”.

The update specifically investigates whether high inferred masses come from:

- the **choice of variable used in the outlier model**;
- the **assumed good-binary velocity shape**;
- or both.

This is why the update is naturally presented as a **sensitivity analysis followed by targeted corrections**.

---

## 4. Key improvement A: outlier model on raw \(u\), not on \(\tilde u\)

### 4.1 Motivation

One important modeling change is:

- **Old approach:** outlier model built on \(\tilde u\)
- **New approach:** outlier model built on raw \(u\)

This change is scientifically important because \(\tilde u\) contains the mass scaling:

\[
\tilde u = \frac{u}{\sqrt{M_{\rm tot}}}.
\]

If the outlier model is defined directly in \(\tilde u\), then the classification between “good” and “outlier” components can become entangled with the inferred mass scale itself.

By moving the outlier model to **raw \(u\)**:

- the outlier treatment is anchored more directly to the observed velocity-like quantity,
- and it is less entangled with the mass normalization that the MLR is trying to infer.

### 4.2 Scientific message

The scientific message of this improvement is:

> Before changing the physical good-shape model, first check whether the baseline mass inflation is partly caused by how the outlier component is parameterized.

### 4.3 What result change I want to demonstrate

I want to show explicitly:

- baseline result (`7bc861a`)
- result after **only** changing the outlier model to raw \(u\)

This isolates the effect of Improvement A.

---

## 5. Key improvement B: allow the good shape to vary

### 5.1 Motivation from sensitivity analysis

After the sensitivity tests, I found that the inferred MLR is highly sensitive to the assumed good-binary shape.

This led to the second key improvement:

- move from a **fixed** good shape
- to a **variable / calibrated** good shape version.

The scientific reason is **not** “the baseline MLR looks bad, so I changed the shape until it looked better”.

The correct interpretation is:

> The good shape has a physical basis, but it still depends on population assumptions (e.g. eccentricity distribution, selection function, and whether the true binary population matches those assumptions).  
> Since the inferred MLR is highly sensitive to those shape parameters, I need to account for this source of model uncertainty. :contentReference[oaicite:4]{index=4}

### 5.2 Preferred interpretation of the variable-shape model

The variable-shape version should be described as:

- a **self-consistent calibration**
- or an **EM-style / alternating optimization**
- rather than “letting the data freely rewrite the shape”.

The idea is:

1. start from a physics-motivated shape,
2. infer an MLR,
3. update the shape within a physically reasonable range,
4. iterate toward a self-consistent solution. :contentReference[oaicite:5]{index=5} :contentReference[oaicite:6]{index=6}

### 5.3 Scientific message

The message of Improvement B is:

> Once I confirmed that the MLR is strongly sensitive to the assumed good shape, I introduced a controlled way to calibrate that shape rather than treating the original fixed shape as exact.

### 5.4 What result change I want to demonstrate

I want to show explicitly:

- result after Improvement A only
- result after Improvement B (good shape variable)

This isolates the effect of Improvement B on top of the previous correction.

---

## 6. Recommended storyline for advisor presentation

The recommended scientific storyline is:

### Stage 1: Physics baseline
Start with the original fixed-shape baseline and show the 40% discrepancy.

This answers:

> If I fully trust the original orbital-population shape model, what MLR do I infer? :contentReference[oaicite:7]{index=7}

### Stage 2: Sensitivity diagnosis
Run sensitivity tests to identify which model ingredients tend to push the masses upward.

This answers:

> Where does the high mass normalization come from?

### Stage 3: Targeted corrections
Present the two key targeted improvements:

1. outlier model on raw \(u\);
2. variable good shape.

### Stage 4: Compare results step by step
Show how the inferred MLR changes after each improvement.

This is the most important presentation principle:

> Each step should be shown separately, so that the advisor can see **which change causes which shift** in the inferred MLR.

---

## 7. Plotting plan

I want the figures to make the logic visually obvious.  
The figures should emphasize **stepwise causal diagnosis**, not just “before vs after”.

---

### Figure 1. Baseline motivation: the 40% discrepancy

**Purpose:** establish the problem.

**Plot content:**
- `7bc861a` baseline MLR
- PARSEC / empirical reference MLRs
- optional ratio panel:
  \[
  \frac{M_{\rm baseline}(L)}{M_{\rm reference}(L)}
  \]

**Main visual message:**
- the baseline is systematically high by about 40%.

**Suggested caption message:**
> The `7bc861a` baseline yields a mass--luminosity relation systematically above standard reference relations, motivating a diagnostic analysis of which modeling assumptions control the mass normalization.

---

### Figure 2. Sensitivity test: which model ingredients raise the inferred masses?

**Purpose:** identify the source of upward mass bias.

**Possible content options:**
- a bar chart or summary table of sensitivity experiments;
- each row = one model ingredient perturbed;
- metric on x-axis = change in MLR normalization or change in inferred mass at a representative luminosity.

**Key ingredients to test / summarize:**
- outlier model variable choice (\(\tilde u\) vs raw \(u\));
- good-shape parameters / functional form;
- optionally other relevant components if tested.

**Main visual message:**
- not all model ingredients matter equally;
- the strongest effects come from:
  1. the outlier model definition,
  2. the good-shape assumption.

**Suggested caption message:**
> Sensitivity tests show that the inferred mass scale is particularly affected by the outlier model parameterization and the assumed good-binary velocity shape.

---

### Figure 3. Improvement A only: outlier model on raw \(u\)

**Purpose:** isolate the effect of the outlier-model change.

**Plot content:**
- overlay MLR curves for:
  - baseline (`7bc861a`)
  - updated model with outlier component defined on raw \(u\)
- optionally add a lower panel showing fractional shift relative to baseline.

**Main visual message:**
- demonstrate how much of the high normalization is removed (or changed) by this step alone.

**Suggested caption message:**
> Replacing the outlier model on \(\tilde u\) with one defined on raw \(u\) changes the inferred MLR by [describe direction and magnitude], indicating that part of the baseline offset was tied to the outlier-model parameterization.

---

### Figure 4. Good-shape sensitivity

**Purpose:** justify introducing a variable-shape model.

**Plot content:**
Two-panel layout recommended.

**Upper panel:**
- several plausible good-shape distributions \(p(\tilde u)\)

**Lower panel:**
- the corresponding inferred MLRs

or alternatively:
- x-axis = a shape summary parameter (e.g. \(u_c\) or median \(\tilde u\))
- y-axis = inferred mass at fixed luminosity, or MLR normalization

**Main visual message:**
- modest, plausible changes in the good shape can cause large shifts in the inferred MLR.

**Suggested caption message:**
> The inferred MLR is highly sensitive to the assumed good-binary velocity shape, showing that shape uncertainty must be treated as a major source of model uncertainty. :contentReference[oaicite:8]{index=8}

---

### Figure 5. Improvement B: variable good shape

**Purpose:** isolate the effect of allowing good shape calibration.

**Plot content:**
- overlay MLR curves for:
  - result after Improvement A only
  - result after Improvement A + variable good shape
- optionally show the updated good-shape distribution in a side panel.

**Main visual message:**
- once good-shape uncertainty is allowed to enter the inference, the MLR shifts again in a physically interpretable way.

**Suggested caption message:**
> Because the inferred MLR is strongly shape-sensitive, we allow the physics-motivated good shape to be recalibrated within a reasonable range, leading to a more self-consistent joint solution for the velocity distribution and the MLR. :contentReference[oaicite:9]{index=9}

---

### Figure 6. Final step-by-step comparison summary

**Purpose:** present the cumulative logic in one plot.

**Plot content:**
Overlay the following curves:
1. `7bc861a` baseline
2. + outlier-on-raw-\(u\) result
3. + variable-good-shape result
4. PARSEC / empirical reference relations

**Main visual message:**
- this is the cleanest “story summary” figure;
- it shows how the result changes step by step;
- each shift is scientifically interpretable.

**Suggested caption message:**
> Stepwise comparison of the baseline and successive model improvements. The figure separates the effect of changing the outlier model from the effect of accounting for good-shape uncertainty, clarifying how each modification alters the inferred mass scale.

---

## 8. Recommended talk track for each figure

### Figure 1 talk track
> Here is the starting point: the `7bc861a` baseline is about 40% high relative to standard MLRs.  
> So the goal of this update is not “to make the answer look better”, but to understand why the inferred masses are high and whether that is physically justified.

### Figure 2 talk track
> I then ran sensitivity tests to identify which model ingredients tend to raise the inferred masses.  
> Two ingredients stood out: the outlier model definition, and the assumed good-binary velocity shape.

### Figure 3 talk track
> The first targeted change was to redefine the outlier model on raw \(u\) instead of \(\tilde u\).  
> This helps separate outlier treatment from the mass scaling embedded in \(\tilde u\).

### Figure 4 talk track
> The sensitivity tests also showed that the inferred MLR depends strongly on the assumed good shape.  
> This means the good-shape assumption is not a minor detail; it is one of the dominant controls on the mass normalization.

### Figure 5 talk track
> Because of that strong sensitivity, I introduced a variable-good-shape version.  
> The goal is not to discard the original physics-based shape, but to allow a constrained self-consistent calibration around it.

### Figure 6 talk track
> This final comparison shows the effect of each improvement step separately.  
> It makes clear which changes are due to the outlier-model correction and which are due to accounting for good-shape uncertainty.

---

## 9. Important wording choices

### 9.1 Preferred wording
Use wording like:

- “diagnosing the origin of the 40% offset”
- “testing which assumptions control the mass normalization”
- “sensitivity analysis”
- “targeted model improvements”
- “physics-motivated baseline”
- “self-consistent calibration”
- “variable good-shape version”
- “EM-style / alternating calibration” (if needed)

### 9.2 Avoid wording like
Avoid phrasing such as:

- “the baseline looked wrong, so I changed the model”
- “I tuned the shape until the MLR matched literature”
- “the variable-shape model fixes the problem”

These phrasings sound too outcome-driven.

### 9.3 Better scientific framing
Use instead:

> I began from a physically motivated fixed-shape baseline.  
> The baseline MLR showed a large offset, which motivated sensitivity tests.  
> These tests revealed that the outlier model and the assumed good shape have strong leverage on the inferred mass scale.  
> I therefore introduced two targeted changes and evaluated their effects separately.

---

## 10. Optional figure-order summary (short version)

If I need a compact 4-figure version, use:

1. **Baseline vs references**  
   show the ~40% discrepancy

2. **Sensitivity summary**  
   show which ingredients push masses upward

3. **Baseline vs raw-\(u\) outlier model**  
   isolate Improvement A

4. **Stepwise final comparison**  
   baseline → raw-\(u\) outlier model → variable good shape → references

If I have space for more detail, add:

5. **Good-shape sensitivity figure**
6. **Variable-good-shape result figure**

---

## 11. One-paragraph summary

Relative to the `7bc861a` baseline, the new update is best understood as a structured investigation into the origin of the baseline’s \(\sim 40\%\) high mass normalization. I first asked which modeling assumptions tend to increase the inferred masses, and sensitivity tests identified two particularly important ingredients: the variable used in the outlier model and the assumed good-binary velocity shape. This led to two key improvements: (1) redefining the outlier model on raw \(u\) rather than \(\tilde u\), and (2) introducing a variable-good-shape version after finding strong sensitivity of the MLR to the assumed good shape. In the presentation, these two improvements should be shown step by step, so that their effects on the inferred MLR can be interpreted separately.

---

## 12. AI instruction block (for future use)

If another AI is asked to help make figures or slides based on this document, it should follow these rules:

1. Preserve the scientific storyline:
   - baseline discrepancy
   - sensitivity diagnosis
   - targeted improvements
   - stepwise comparison of results

2. Do not collapse all updates into a single “old vs new” comparison.

3. Always show the effect of the two key improvements separately:
   - outlier model on raw \(u\)
   - variable good shape

4. Figures should emphasize causal interpretability:
   - what changed,
   - why it changed,
   - how much the MLR moved after each step.

5. Use clean, presentation-friendly visuals suitable for advisor discussion.
