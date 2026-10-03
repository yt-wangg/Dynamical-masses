# Deprecated: variable-shape p(u~) workflows

These scripts fit or sample the shape of the mass-scaled velocity distribution
p(u~) (alternating-MAP / EM-style optimization, joint good-shape + MLR MCMC,
metallicity-bin joint fits) and the diagnostics built on them. With the
`u_obs/sigma_u > 3` cut modelled in the likelihood, the shape is held fixed and
this code is no longer part of the main model.

The scripts depend on library code that was removed from the main branch
(`DynamicsLikelihoodShapeStack`, `--sample-dynamics-shape`). To run one, check
out git tag `pre-selection-truncation-main` and use
`PYTHONPATH=src/examples:src` with the script path as it was before the move
(`src/examples/<name>.py`).

`diagnose_pu_selection.py` and `diagnose_pu_selection_refit.py` are the
diagnostics that showed the fitted shape absorbing the selection cut.
