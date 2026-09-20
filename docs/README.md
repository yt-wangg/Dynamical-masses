# Documentation index

The repository root intentionally keeps only the main [README](../README.md).
All other Markdown documentation is organized here by purpose.

## Current workflows

- [T8 server run guide](workflows/T8_SERVER_RUN.md) — standard metallicity calibration, Rice lookup, and MLR execution.
- [T8.2 joint-shape pipeline](workflows/T82_JOINT_SHAPE_PIPELINE.md) — joint good-shape + MLR MCMC and metallicity-bin fits.
- [Alternating-MAP / EM-style workflow](workflows/EM_MLR_WORKFLOW.md) — optimization/sensitivity workflow, not the primary posterior inference.

## Model and methodology

- [Post-`7bc861a` model improvements](model/POST_7BC_MODEL_IMPROVEMENTS.md) — scientific motivation, correctness fixes, and interpretation of the current model.
- [Hierarchical metallicity lookup model](model/HIERARCHICAL_METALLICITY_LOOKUP_MODEL.md) — model design and lookup architecture.

## Validation and sensitivity studies

- [T8 holdout validation](validation/T8_HOLDOUT_VALIDATION.md)
- [T8 outlier sensitivity server workflow](validation/T8_OUTLIER_SENSITIVITY_SERVER.md)
- [Mass-independent outlier Garching plan](validation/T8_MASS_INDEPENDENT_OUTLIER_GARCHING_PLAN.md)

## Baselines

- [T8 baseline 2026-09-13](baselines/T8_BASELINE_20260913.md)

## Project and repository maintenance

- [Contributing](project/CONTRIBUTING.md)
- [GitHub setup guide](project/GitHub_Setup_Guide.md)
- [Repository summary](project/REPOSITORY_SUMMARY.md)
- [Claude project notes](project/CLAUDE.md)

## Archive

Historical handoffs, superseded workflows, and T8 drafts are kept for provenance.
They should not be treated as the current execution instructions.

### Handoffs

- [Dynamical masses handoff](archive/handoffs/DYNAMICAL_MASSES_HANDOFF.md)
- [Discussion handoff — 2026-08-19](archive/handoffs/DYNAMICAL_MASSES_DISCUSSION_HANDOFF_2026-08-19.md)

### Earlier workflows

- [Final FeH JCAPS Student-t workflow — 2026-09-10](archive/workflows/FINAL_FEH_JCAPS_STUDENT_T_WORKFLOW_2026-09-10.md)
- [T7 JCAPS Student-t PARSEC CMD MLR workflow — 2026-09-11](archive/workflows/T7.%20JCAPS_STUDENT_T_PARSEC_CMD_MLR_WORKFLOW_2026-09-11.md)
- [FeH-global workflow](archive/workflows/FEH_GLOBAL_WORKFLOW.md)

### T8 drafts

- [T8 AI handoff](archive/t8_drafts/T8.%20AI_HANDOFF_JCAPS_CMD_MONOTONE_MLR_LOOKUP_2026-09-13.md)
- [T8a calibration draft](archive/t8_drafts/T8a.%20JCAPS_STUDENT_T_PARSEC_CMD_CALIBRATION_4000K_2026-09-12.md)
- [T8b monotone MLR draft](archive/t8_drafts/T8b.%20HARD_MONOTONE_PARSEC_RELATIVE_TENSOR_SPLINE_MLR_2026-09-12.md)
- [T8c Rice lookup draft](archive/t8_drafts/T8c.%20RICE_DYNAMICS_LIKELIHOOD_LOOKUP_2026-09-12.md)
