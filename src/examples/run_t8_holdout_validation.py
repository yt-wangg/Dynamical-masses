#!/usr/bin/env python3
"""Run the T8 80/20 held-out and solar-anchor sensitivity validation.

The frozen full-sample T8 products are read-only inputs.  This runner makes a
deterministic system-level split, refits the MLR on the 80% training systems,
and evaluates the untouched 20% systems.  A second fit changes only the solar
anchor width, so its effect is directly comparable with the default fit.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import logsumexp
from scipy.optimize import minimize_scalar


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from binary_masses.hierarchical_metallicity import (  # noqa: E402
    DynamicsLikelihoodLookup,
    MetallicityPosteriorGrid,
    MonotoneTensorSplineMLR,
    array_digest,
    marginalize_dynamics,
)
import run_hierarchical_metallicity_test as t8  # noqa: E402


DEFAULT_BASELINE_DIR = REPO_ROOT / "results" / "hierarchical_metallicity_t8_20260913"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "hierarchical_metallicity_t8_20260913_holdout"
DEFAULT_SOLAR_SIGMA = 0.01 / np.log(10.0)
FIT_SAMPLE_NAMES = (
    "c0", "a", "b", "r", "log_lambda_x", "log_lambda_z", "f_outlier"
)


def require_baseline_posterior(posterior, *, arrays):
    """Validate T8 model metadata without requiring file checksums.

    The historical input digest includes a derived color uncertainty whose
    last bit can differ across CPU math libraries.  The hold-out workflow does
    not use that color uncertainty, so only this final digest mismatch is
    ignored.  All preceding T8 schema, column, grid, and draw-record checks
    remain active.
    """
    try:
        t8._require_t8_posterior_metadata(posterior, mock=False, arrays=arrays)
        return
    except ValueError as exc:
        expected_message = "T8 posterior input-data digest does not match"
        if expected_message not in str(exc):
            raise
    print(
        "Ignoring the legacy full-input checksum mismatch. T8 schema, row order, "
        "and dynamics lookup alignment remain validated."
    )


def split_system_positions(
    row_indices,
    *,
    test_fraction: float = 0.20,
    seed: int = 20260914,
    convergence_rows=(),
):
    """Return reproducible sorted train/test positions with lookup-check coverage."""
    rows = np.asarray(row_indices, dtype=np.int64)
    if rows.ndim != 1 or rows.size < 4 or np.unique(rows).size != rows.size:
        raise ValueError("At least four unique system row indices are required.")
    if not 0.0 < float(test_fraction) < 0.5:
        raise ValueError("test_fraction must lie strictly between 0 and 0.5.")
    n_test = int(round(rows.size * float(test_fraction)))
    n_test = min(max(n_test, 1), rows.size - 1)
    rng = np.random.default_rng(int(seed))
    test = np.sort(rng.choice(rows.size, size=n_test, replace=False))
    is_test = np.zeros(rows.size, dtype=bool)
    is_test[test] = True

    checked = np.flatnonzero(np.isin(rows, np.asarray(convergence_rows, dtype=np.int64)))
    if checked.size:
        checked_train = checked[~is_test[checked]]
        checked_test = checked[is_test[checked]]
        if checked_test.size == 0:
            entering = int(rng.choice(checked_train))
            leaving_candidates = test[~np.isin(test, checked)]
            if leaving_candidates.size == 0:
                raise ValueError("Cannot preserve a Rice convergence row in both split groups.")
            leaving = int(rng.choice(leaving_candidates))
            is_test[entering], is_test[leaving] = True, False
        elif checked_train.size == 0:
            leaving = int(rng.choice(checked_test))
            entering_candidates = np.flatnonzero(~is_test & ~np.isin(np.arange(rows.size), checked))
            if entering_candidates.size == 0:
                raise ValueError("Cannot preserve a Rice convergence row in both split groups.")
            entering = int(rng.choice(entering_candidates))
            is_test[leaving], is_test[entering] = False, True

    train = np.flatnonzero(~is_test)
    test = np.flatnonzero(is_test)
    if train.size + test.size != rows.size or np.intersect1d(train, test).size:
        raise AssertionError("Internal train/test split inconsistency.")
    return train, test


def select_validation_cohort(row_indices, *, max_systems, seed, convergence_rows):
    """Select a small reproducible smoke cohort while retaining checked rows."""
    rows = np.asarray(row_indices, dtype=np.int64)
    if max_systems is None or int(max_systems) >= rows.size:
        return np.arange(rows.size, dtype=np.int64)
    size = int(max_systems)
    if size < 4:
        raise ValueError("max_systems must be at least four.")
    rng = np.random.default_rng(int(seed))
    checked = np.flatnonzero(np.isin(rows, np.asarray(convergence_rows, dtype=np.int64)))
    required_count = min(2, checked.size, size)
    required = (
        rng.choice(checked, size=required_count, replace=False)
        if required_count else np.empty(0, dtype=np.int64)
    )
    pool = np.setdiff1d(np.arange(rows.size), required, assume_unique=True)
    extra = rng.choice(pool, size=size - required.size, replace=False)
    return np.sort(np.concatenate([required, extra]))


def subset_inputs(
    arrays: Mapping[str, np.ndarray],
    posterior: MetallicityPosteriorGrid,
    lookup: DynamicsLikelihoodLookup,
    positions,
    *,
    label: str,
):
    """Slice aligned T8 arrays and retain valid inherited lookup provenance."""
    pos = np.asarray(positions, dtype=np.int64)
    selected_arrays = {key: np.asarray(value)[pos] for key, value in arrays.items()}
    rows = selected_arrays["row_indices"]

    posterior_metadata = dict(posterior.metadata)
    posterior_metadata.update(
        {
            "subset": label,
            "parent_row_indices_digest": array_digest(posterior.row_indices),
            "row_indices_digest": array_digest(rows),
            "n_systems": int(rows.size),
        }
    )
    selected_posterior = MetallicityPosteriorGrid(
        row_indices=np.asarray(posterior.row_indices)[pos],
        z_grid=np.asarray(posterior.z_grid).copy(),
        probabilities=np.asarray(posterior.probabilities)[pos],
        z_quantiles=np.asarray(posterior.z_quantiles)[pos],
        bad_probabilities=np.asarray(posterior.bad_probabilities)[pos],
        metadata=posterior_metadata,
    )
    selected_posterior.validate()

    lookup_metadata = json.loads(json.dumps(dict(lookup.metadata)))
    parent_digest = str(lookup_metadata["data_digest"])
    selected_digest = array_digest(
        rows, selected_arrays["u"], selected_arrays["u_sigma"]
    )
    convergence = dict(lookup_metadata["convergence_check"])
    tested_rows = np.asarray(convergence["selected_row_indices"], dtype=np.int64)
    surviving = tested_rows[np.isin(tested_rows, rows)]
    if surviving.size == 0:
        raise ValueError(
            f"The {label} subset contains no row from the saved Rice convergence check."
        )
    convergence.update(
        {
            "source_data_digest": selected_digest,
            "sample_system_count": int(surviving.size),
            "selected_row_indices": surviving.tolist(),
            "selected_row_indices_digest": array_digest(surviving),
            "source_subset": "system_split_of_converged_parent_lookup",
            "parent_source_data_digest": parent_digest,
        }
    )
    lookup_metadata.update(
        {
            "data_digest": selected_digest,
            "parent_data_digest": parent_digest,
            "convergence_check": convergence,
            "subset": label,
            "n_systems": int(rows.size),
        }
    )
    selected_lookup = DynamicsLikelihoodLookup(
        row_indices=np.asarray(lookup.row_indices)[pos],
        sqrt_mtot_grid=np.asarray(lookup.sqrt_mtot_grid).copy(),
        log_good=np.asarray(lookup.log_good)[pos],
        log_bad=np.asarray(lookup.log_bad)[pos],
        metadata=lookup_metadata,
    )
    selected_lookup.validate_for(
        row_indices=rows,
        u=selected_arrays["u"],
        u_sigma=selected_arrays["u_sigma"],
    )
    return selected_arrays, selected_posterior, selected_lookup


def load_grouped_samples(path: Path):
    samples = {}
    with np.load(path, allow_pickle=False) as saved:
        for key in saved.files:
            if key.startswith("posterior__"):
                values = np.asarray(saved[key])
                samples[key[len("posterior__"):]] = values.reshape(
                    (-1,) + values.shape[2:]
                )
    missing = sorted(set(FIT_SAMPLE_NAMES).difference(samples))
    if missing:
        raise ValueError(f"MLR trace is missing posterior variables: {missing}.")
    return samples


def make_mlr(mass_surface, *, solar_sigma: float):
    return MonotoneTensorSplineMLR(
        mass_surface,
        knots_x=t8.T8_MLR_DEFAULT_KNOT_X,
        knots_z=t8.T8_MLR_DEFAULT_KNOT_Z,
        degree_x=3,
        degree_z=3,
        tau_D=0.10,
        tau_x=0.05,
        tau_Z=0.05,
        tau_xZ=0.05,
        solar_anchor_mean=0.0,
        solar_anchor_sigma=float(solar_sigma),
    )


def fit_training_model(
    *,
    label,
    solar_sigma,
    arrays,
    posterior,
    lookup,
    mass_surface,
    output_dir,
    warmup,
    samples,
    chains,
    seed,
    target_accept,
    grid_draws,
):
    fit_dir = output_dir / label
    fit_dir.mkdir(parents=True, exist_ok=True)
    mlr = make_mlr(mass_surface, solar_sigma=solar_sigma)
    mlr.set_data(
        row_indices=arrays["row_indices"],
        u=arrays["u"],
        u_sigma=arrays["u_sigma"],
        absg=arrays["absg"],
        metallicity_grid=posterior,
        dynamics_lookup=lookup,
    )
    print(
        f"Fitting {label}: {len(arrays['row_indices'])} training systems, "
        f"solar sigma={solar_sigma:.8g} dex."
    )
    sampler = mlr.run_mcmc(
        num_warmup=warmup,
        num_samples=samples,
        num_chains=chains,
        seed=seed,
        progress_bar=True,
        target_accept_prob=target_accept,
    )
    grid = mlr.correction_grid(max_draws=grid_draws)
    t8.save_grouped_mcmc(sampler, fit_dir / t8.T8_MLR_NAME)
    t8.save_posterior_summary(mlr.posterior_samples, fit_dir / "mlr_summary_t8.csv")
    diagnostic_path = fit_dir / "mlr_diagnostics_t8.json"
    t8.save_sampler_diagnostics(sampler, diagnostic_path)
    diagnostics = json.loads(diagnostic_path.read_text(encoding="utf-8"))
    fields = sampler.get_extra_fields(group_by_chain=True)
    energy = np.asarray(fields.get("energy"), dtype=np.float64)
    if energy.ndim == 2 and energy.shape[1] > 1:
        variance = np.var(energy, axis=1)
        if np.all(variance > 0.0):
            bfmi = np.mean(np.diff(energy, axis=1) ** 2, axis=1) / variance
            diagnostics["bfmi_by_chain"] = bfmi.tolist()
            diagnostics["min_bfmi"] = float(np.min(bfmi))
    if int(chains) >= 2:
        from numpyro.diagnostics import summary as numpyro_summary

        convergence = numpyro_summary(
            sampler.get_samples(group_by_chain=True), group_by_chain=True
        )
        r_hat = np.concatenate(
            [np.ravel(np.asarray(values["r_hat"], dtype=float)) for values in convergence.values()]
        )
        n_eff = np.concatenate(
            [np.ravel(np.asarray(values["n_eff"], dtype=float)) for values in convergence.values()]
        )
        diagnostics["max_r_hat"] = float(np.nanmax(r_hat))
        diagnostics["min_effective_sample_size"] = float(np.nanmin(n_eff))
    diagnostic_path.write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
    np.savez_compressed(fit_dir / "mlr_derived_grid_t8.npz", **grid)
    metadata = {
        "schema": "t8-holdout-training-fit-v1",
        "label": label,
        "n_training_systems": int(len(arrays["row_indices"])),
        "training_row_indices_digest": array_digest(arrays["row_indices"]),
        "sampled_scalar_count": int(mlr.n_params),
        "solar_anchor_mean_log10_mass": 0.0,
        "solar_anchor_sigma_log10_mass": float(solar_sigma),
        "penalty_scales_dex": {
            "tau_D": 0.10,
            "tau_x": 0.05,
            "tau_Z": 0.05,
            "tau_xZ": 0.05,
        },
        "mcmc": {
            "warmup": int(warmup),
            "samples_per_chain": int(samples),
            "chains": int(chains),
            "seed": int(seed),
            "target_accept": float(target_accept),
        },
        "initialization": dict(mlr.initialization_metadata),
    }
    (fit_dir / "mlr_model.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return mlr, grid


def attach_saved_fit(
    *, mass_surface, solar_sigma, arrays, posterior, lookup, trace_path
):
    mlr = make_mlr(mass_surface, solar_sigma=solar_sigma)
    mlr.set_data(
        row_indices=arrays["row_indices"],
        u=arrays["u"],
        u_sigma=arrays["u_sigma"],
        absg=arrays["absg"],
        metallicity_grid=posterior,
        dynamics_lookup=lookup,
    )
    mlr.posterior_samples = load_grouped_samples(trace_path)
    return mlr


def require_saved_fit_metadata(fit_dir, *, solar_sigma, training_rows):
    path = Path(fit_dir) / "mlr_model.json"
    if not path.exists():
        raise FileNotFoundError(f"Saved fit metadata is missing: {path}")
    metadata = json.loads(path.read_text(encoding="utf-8"))
    if metadata.get("schema") != "t8-holdout-training-fit-v1":
        raise ValueError(f"Saved fit has an unexpected schema: {path}")
    if metadata.get("training_row_indices_digest") != array_digest(training_rows):
        raise ValueError(f"Saved fit uses a different training split: {path}")
    if not np.isclose(
        float(metadata.get("solar_anchor_sigma_log10_mass", np.nan)),
        float(solar_sigma),
        rtol=0.0,
        atol=1e-15,
    ):
        raise ValueError(
            f"Saved fit uses a different solar-anchor prior width: {path}"
        )
    return metadata


def evaluate_heldout(model, samples, arrays, posterior, lookup, *, max_draws):
    """Return paired per-system PARSEC and corrected held-out log likelihoods."""
    n_available = int(np.asarray(samples["c0"]).shape[0])
    draw_indices = np.linspace(
        0, n_available - 1, min(int(max_draws), n_available), dtype=np.int64
    )
    z = posterior.z_grid[None, :]
    baseline_sqrt = np.sqrt(
        model.mass_surface.mass_from_absg_mh(arrays["absg"][:, 0, None], z)
        + model.mass_surface.mass_from_absg_mh(arrays["absg"][:, 1, None], z)
    )
    baseline_good = lookup.interpolate_numpy(baseline_sqrt, component="good")
    baseline_bad = lookup.interpolate_numpy(baseline_sqrt, component="bad")
    parsec_ll = np.empty((draw_indices.size, len(arrays["row_indices"])), dtype=np.float64)
    corrected_ll = np.empty_like(parsec_ll)
    for out_index, draw in enumerate(draw_indices):
        params = {name: samples[name][draw] for name in FIT_SAMPLE_NAMES[:-1]}
        g1 = model.g_from_raw(arrays["absg"][:, 0, None], z, params)
        g2 = model.g_from_raw(arrays["absg"][:, 1, None], z, params)
        corrected_sqrt = np.sqrt(10.0**g1 + 10.0**g2)
        corrected_good = lookup.interpolate_numpy(corrected_sqrt, component="good")
        corrected_bad = lookup.interpolate_numpy(corrected_sqrt, component="bad")
        f_outlier = float(samples["f_outlier"][draw])
        log_f = np.log(f_outlier)
        log_good_weight = np.log1p(-f_outlier)
        parsec_ll[out_index] = marginalize_dynamics(
            np.logaddexp(
                log_good_weight + baseline_good, log_f + baseline_bad
            ),
            posterior.probabilities,
        )
        corrected_ll[out_index] = marginalize_dynamics(
            np.logaddexp(
                log_good_weight + corrected_good, log_f + corrected_bad
            ),
            posterior.probabilities,
        )
    return {
        "draw_indices": draw_indices,
        "parsec_log_likelihood": parsec_ll,
        "mlr_log_likelihood": corrected_ll,
        "delta_log_likelihood": corrected_ll - parsec_ll,
    }


def parsec_likelihood_basis(mass_surface, arrays, posterior, lookup):
    z = posterior.z_grid[None, :]
    sqrt_mtot = np.sqrt(
        mass_surface.mass_from_absg_mh(arrays["absg"][:, 0, None], z)
        + mass_surface.mass_from_absg_mh(arrays["absg"][:, 1, None], z)
    )
    good = lookup.interpolate_numpy(sqrt_mtot, component="good")
    bad = lookup.interpolate_numpy(sqrt_mtot, component="bad")
    return good, bad, np.asarray(posterior.probabilities, dtype=np.float64)


def parsec_log_likelihood_from_basis(good, bad, probabilities, f_outlier_values):
    values = np.asarray(f_outlier_values, dtype=np.float64)
    output = np.empty((values.size, good.shape[0]), dtype=np.float64)
    for index, f_outlier in enumerate(values):
        output[index] = marginalize_dynamics(
            np.logaddexp(
                np.log1p(-f_outlier) + good,
                np.log(f_outlier) + bad,
            ),
            probabilities,
        )
    return output


def parsec_log_likelihood_at_f(
    mass_surface, arrays, posterior, lookup, f_outlier_values
):
    """Evaluate the uncorrected PARSEC likelihood on a one-dimensional f grid."""
    good, bad, probabilities = parsec_likelihood_basis(
        mass_surface, arrays, posterior, lookup
    )
    return parsec_log_likelihood_from_basis(
        good, bad, probabilities, f_outlier_values
    )


def fit_parsec_outlier_posterior(
    mass_surface, arrays, posterior, lookup, *, quadrature_nodes
):
    """Fit PARSEC's Beta(3,12) basis weight on training systems by quadrature."""
    good, bad, probabilities = parsec_likelihood_basis(
        mass_surface, arrays, posterior, lookup
    )

    def log_posterior_at(f_outlier):
        if not 0.0 < f_outlier < 1.0:
            return -np.inf
        log_likelihood = parsec_log_likelihood_from_basis(
            good, bad, probabilities, [f_outlier]
        )[0]
        return float(
            np.sum(log_likelihood)
            + 2.0 * np.log(f_outlier)
            + 11.0 * np.log1p(-f_outlier)
        )

    optimum = minimize_scalar(
        lambda value: -log_posterior_at(value),
        bounds=(1e-6, 1.0 - 1e-6),
        method="bounded",
        options={"xatol": 1e-8},
    )
    if not optimum.success:
        raise ValueError("Could not locate the PARSEC f_outlier posterior mode.")
    mode = float(optimum.x)
    mode_log_posterior = log_posterior_at(mode)
    finite_distance = min(mode, 1.0 - mode)
    h = min(1e-3, max(1e-5, 0.01 * finite_distance))
    curvature = (
        log_posterior_at(mode + h)
        - 2.0 * mode_log_posterior
        + log_posterior_at(mode - h)
    ) / h**2
    if not np.isfinite(curvature) or curvature >= 0.0:
        raise ValueError("PARSEC f_outlier posterior curvature is not negative at its mode.")
    local_sigma = float(np.sqrt(-1.0 / curvature))
    half_width = max(10.0 * local_sigma, 0.02)
    for _ in range(8):
        lower = max(1e-8, mode - half_width)
        upper = min(1.0 - 1e-8, mode + half_width)
        boundary_drop = np.array(
            [
                mode_log_posterior - log_posterior_at(lower),
                mode_log_posterior - log_posterior_at(upper),
            ]
        )
        if np.all(boundary_drop >= 30.0) or (lower <= 1e-8 and upper >= 1.0 - 1e-8):
            break
        half_width *= 1.8
    else:
        raise ValueError("Could not bound the PARSEC f_outlier posterior tails.")

    legendre_x, legendre_w = np.polynomial.legendre.leggauss(int(quadrature_nodes))
    f_grid = 0.5 * (upper - lower) * legendre_x + 0.5 * (upper + lower)
    quadrature_weights = 0.5 * (upper - lower) * legendre_w
    train_log_likelihood = parsec_log_likelihood_from_basis(
        good, bad, probabilities, f_grid
    )
    log_prior = 2.0 * np.log(f_grid) + 11.0 * np.log1p(-f_grid)
    log_unnormalized = (
        np.log(quadrature_weights) + log_prior + np.sum(train_log_likelihood, axis=1)
    )
    log_weights = log_unnormalized - logsumexp(log_unnormalized)
    return {
        "f_outlier_grid": f_grid,
        "posterior_weights": np.exp(log_weights),
        "posterior_log_weights": log_weights,
        "quadrature_nodes": int(quadrature_nodes),
        "posterior_mode": mode,
        "local_laplace_sigma": local_sigma,
        "integration_bounds": np.array([lower, upper]),
        "boundary_log_posterior_drop": boundary_drop,
    }


def parsec_posterior_predictive(
    mass_surface, arrays, posterior, lookup, parsec_posterior
):
    log_likelihood = parsec_log_likelihood_at_f(
        mass_surface,
        arrays,
        posterior,
        lookup,
        parsec_posterior["f_outlier_grid"],
    )
    return logsumexp(
        parsec_posterior["posterior_log_weights"][:, None] + log_likelihood,
        axis=0,
    )


def posterior_predictive_log_likelihood(log_likelihood_draws):
    values = np.asarray(log_likelihood_draws, dtype=np.float64)
    return logsumexp(values, axis=0) - np.log(values.shape[0])


def percentile_summary(values):
    q = np.percentile(np.asarray(values), [2.5, 16.0, 50.0, 84.0, 97.5])
    return {name: float(value) for name, value in zip(
        ("q02_5", "q16", "q50", "q84", "q97_5"), q
    )}


def summarize_heldout(result, parsec_reference_lppd):
    delta = np.asarray(result["delta_log_likelihood"])
    pointwise_mean = np.mean(delta, axis=0)
    total_delta = np.sum(delta, axis=1)
    mlr_lppd = posterior_predictive_log_likelihood(result["mlr_log_likelihood"])
    predictive_delta = mlr_lppd - np.asarray(parsec_reference_lppd)
    return {
        "n_evaluation_draws": int(delta.shape[0]),
        "n_heldout_systems": int(delta.shape[1]),
        "total_delta_log_likelihood": percentile_summary(total_delta),
        "paired_same_f_draw_comparison": {
            "mean_delta_log_likelihood_per_system": float(np.mean(pointwise_mean)),
            "median_delta_log_likelihood_per_system": float(np.median(pointwise_mean)),
            "pointwise_mean_delta_quantiles": percentile_summary(pointwise_mean),
        },
        "posterior_predictive_comparison_to_refit_parsec": {
            "total_delta_log_likelihood": float(np.sum(predictive_delta)),
            "mean_delta_log_likelihood_per_system": float(np.mean(predictive_delta)),
            "median_delta_log_likelihood_per_system": float(np.median(predictive_delta)),
            "pointwise_delta_quantiles": percentile_summary(predictive_delta),
            "fraction_systems_improved": float(np.mean(predictive_delta > 0.0)),
            "fraction_systems_worse": float(np.mean(predictive_delta < 0.0)),
        },
    }


def solar_summary(samples, sigma):
    values = np.asarray(samples["solar_anchor_g"], dtype=np.float64)
    return {
        "prior_mean_log10_mass": 0.0,
        "prior_sigma_log10_mass": float(sigma),
        "posterior_mean_log10_mass": float(np.mean(values)),
        "posterior_sd_log10_mass": float(np.std(values, ddof=1)),
        "posterior_quantiles": percentile_summary(values),
        "posterior_mean_in_prior_sigma": float(np.mean(values) / sigma),
    }


def curve_percentiles(grid):
    return np.percentile(np.asarray(grid["delta_star"]), [16.0, 50.0, 84.0], axis=0)


def save_curve_comparison(path, baseline_grid, default_grid, wide_grid):
    baseline_q = curve_percentiles(baseline_grid)
    default_q = curve_percentiles(default_grid)
    wide_q = curve_percentiles(wide_grid)
    np.savez_compressed(
        path,
        absg_grid=default_grid["absg_grid"],
        z_grid=default_grid["z_grid"],
        percentiles=np.array([16.0, 50.0, 84.0]),
        full_baseline_delta_star_percentiles=baseline_q,
        train_default_delta_star_percentiles=default_q,
        train_wide_delta_star_percentiles=wide_q,
    )
    return baseline_q, default_q, wide_q


def save_curve_csv(path, x, z, baseline_q, default_q, wide_q):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "absg", "mh",
                "full_p16_dex", "full_p50_dex", "full_p84_dex",
                "train_default_p16_dex", "train_default_p50_dex", "train_default_p84_dex",
                "train_wide_p16_dex", "train_wide_p50_dex", "train_wide_p84_dex",
            ]
        )
        for iz, mh in enumerate(z):
            for ix, absg in enumerate(x):
                writer.writerow(
                    [absg, mh, *baseline_q[:, iz, ix], *default_q[:, iz, ix], *wide_q[:, iz, ix]]
                )


def plot_curve_comparison(path, x, z, baseline_q, default_q, wide_q):
    figure, axes = plt.subplots(2, 3, figsize=(13, 7.5), sharex=True, sharey=True)
    axes = axes.ravel()
    for index, mh in enumerate(z):
        ax = axes[index]
        ax.plot(x, baseline_q[1, index], color="0.25", linestyle="--", label="Full-data baseline")
        ax.plot(x, default_q[1, index], color="tab:blue", label="80% fit: default anchor")
        ax.fill_between(x, default_q[0, index], default_q[2, index], color="tab:blue", alpha=0.16)
        ax.plot(x, wide_q[1, index], color="tab:orange", label="80% fit: wide anchor")
        ax.fill_between(x, wide_q[0, index], wide_q[2, index], color="tab:orange", alpha=0.14)
        ax.axhline(0.0, color="0.65", linewidth=0.8)
        ax.set_title(f"[M/H] = {mh:+.1f}")
        ax.invert_xaxis()
    axes[-1].axis("off")
    for ax in axes[:5]:
        ax.set_xlabel(r"$M_G$")
        ax.set_ylabel(r"$\Delta_*$ (dex)")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower right", bbox_to_anchor=(0.96, 0.08), frameon=False)
    figure.suptitle("T8 MLR hold-out and solar-anchor sensitivity")
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_heldout(path, parsec_lppd, default_result, wide_result):
    default_delta = (
        posterior_predictive_log_likelihood(default_result["mlr_log_likelihood"])
        - parsec_lppd
    )
    wide_delta = (
        posterior_predictive_log_likelihood(wide_result["mlr_log_likelihood"])
        - parsec_lppd
    )
    lo, hi = np.percentile(np.concatenate([default_delta, wide_delta]), [0.5, 99.5])
    bins = np.linspace(lo, hi, 60)
    figure, ax = plt.subplots(figsize=(8, 5))
    ax.hist(default_delta, bins=bins, histtype="step", density=True, linewidth=1.8, label="Default anchor")
    ax.hist(wide_delta, bins=bins, histtype="step", density=True, linewidth=1.8, label="Wide anchor")
    ax.axvline(0.0, color="0.35", linestyle="--", linewidth=1.0)
    ax.set_xlabel(r"Held-out $\Delta\log \mathcal{L}$ per system")
    ax.set_ylabel("Density")
    ax.set_title("Held-out MLR improvement relative to PARSEC")
    ax.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def save_pointwise_csv(path, rows, parsec_lppd, default_result, wide_result):
    default_lppd = posterior_predictive_log_likelihood(default_result["mlr_log_likelihood"])
    wide_lppd = posterior_predictive_log_likelihood(wide_result["mlr_log_likelihood"])
    values = [
        parsec_lppd,
        default_lppd,
        default_lppd - parsec_lppd,
        wide_lppd,
        wide_lppd - parsec_lppd,
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "row_index",
                "parsec_posterior_predictive_loglike",
                "default_mlr_posterior_predictive_loglike", "default_delta_predictive_loglike",
                "wide_mlr_posterior_predictive_loglike", "wide_delta_predictive_loglike",
            ]
        )
        for index, row in enumerate(rows):
            writer.writerow([int(row), *(float(value[index]) for value in values)])


def save_or_validate_split(output_dir, cohort_rows, train_rows, test_rows, metadata):
    npz_path = output_dir / "system_split_t8.npz"
    json_path = output_dir / "system_split_t8.json"
    if npz_path.exists():
        with np.load(npz_path, allow_pickle=False) as saved:
            expected = {
                "cohort_row_indices": cohort_rows,
                "train_row_indices": train_rows,
                "test_row_indices": test_rows,
            }
            for name, values in expected.items():
                if not np.array_equal(np.asarray(saved[name]), np.asarray(values)):
                    raise ValueError(
                        "Existing validation split does not match the requested settings; "
                        "use a new output directory."
                    )
        return
    np.savez_compressed(
        npz_path,
        cohort_row_indices=np.asarray(cohort_rows, dtype=np.int64),
        train_row_indices=np.asarray(train_rows, dtype=np.int64),
        test_row_indices=np.asarray(test_rows, dtype=np.int64),
    )
    json_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage", choices=("fit-default", "fit-wide", "evaluate", "all"), default="all"
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=REPO_ROOT / "data" / "jd_msms_single_bic_1kpc_filtered_cmdcut_cutb_jcaps_err_mhcal_good.fits",
    )
    parser.add_argument("--baseline-dir", type=Path, default=DEFAULT_BASELINE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--test-fraction", type=float, default=0.20)
    parser.add_argument("--split-seed", type=int, default=20260914)
    parser.add_argument("--seed", type=int, default=20260914)
    parser.add_argument("--wide-solar-factor", type=float, default=3.0)
    parser.add_argument("--warmup", type=int, default=None)
    parser.add_argument("--samples", type=int, default=None)
    parser.add_argument("--chains", type=int, default=None)
    parser.add_argument("--target-accept", type=float, default=0.9)
    parser.add_argument("--evaluation-draws", type=int, default=None)
    parser.add_argument("--grid-draws", type=int, default=None)
    parser.add_argument("--parsec-f-nodes", type=int, default=None)
    parser.add_argument("--max-systems", type=int, default=None)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    if not 0.0 < args.test_fraction < 0.5:
        parser.error("--test-fraction must lie strictly between 0 and 0.5.")
    if args.wide_solar_factor <= 1.0:
        parser.error("--wide-solar-factor must be greater than one.")
    if args.quick:
        args.max_systems = 64 if args.max_systems is None else args.max_systems
        args.warmup = 50 if args.warmup is None else args.warmup
        args.samples = 50 if args.samples is None else args.samples
        args.chains = 1 if args.chains is None else args.chains
        args.evaluation_draws = 16 if args.evaluation_draws is None else args.evaluation_draws
        args.grid_draws = 16 if args.grid_draws is None else args.grid_draws
        args.parsec_f_nodes = 32 if args.parsec_f_nodes is None else args.parsec_f_nodes
    else:
        args.warmup = 1000 if args.warmup is None else args.warmup
        args.samples = 1000 if args.samples is None else args.samples
        args.chains = 4 if args.chains is None else args.chains
        args.evaluation_draws = 512 if args.evaluation_draws is None else args.evaluation_draws
        args.grid_draws = 512 if args.grid_draws is None else args.grid_draws
        args.parsec_f_nodes = 128 if args.parsec_f_nodes is None else args.parsec_f_nodes
    if min(
        args.warmup, args.samples, args.chains, args.evaluation_draws,
        args.grid_draws, args.parsec_f_nodes,
    ) < 1:
        parser.error("MCMC and posterior evaluation sizes must be positive.")
    if args.max_systems is not None and args.max_systems < 4:
        parser.error("--max-systems must be at least four.")
    return args


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    posterior_path = args.baseline_dir / t8.T8_POSTERIOR_NAME
    lookup_path = args.baseline_dir / t8.T8_LOOKUP_NAME
    baseline_trace_path = args.baseline_dir / t8.T8_MLR_NAME
    for path in (posterior_path, lookup_path, baseline_trace_path):
        if not path.exists():
            raise FileNotFoundError(f"Required frozen T8 baseline artifact is missing: {path}")

    posterior = MetallicityPosteriorGrid.load(posterior_path)
    lookup = DynamicsLikelihoodLookup.load(lookup_path)
    arrays = t8.filter_data(
        t8.load_real_data(args.data),
        max_systems=None,
        seed=args.split_seed,
        fixed_rows=posterior.row_indices,
    )
    require_baseline_posterior(
        posterior,
        arrays=arrays,
    )
    lookup.validate_for(
        row_indices=arrays["row_indices"], u=arrays["u"], u_sigma=arrays["u_sigma"]
    )
    if not np.array_equal(posterior.row_indices, lookup.row_indices):
        raise ValueError("Baseline posterior and lookup row orders differ.")

    checked_rows = np.asarray(
        lookup.metadata["convergence_check"]["selected_row_indices"], dtype=np.int64
    )
    cohort_positions = select_validation_cohort(
        arrays["row_indices"],
        max_systems=args.max_systems,
        seed=args.split_seed,
        convergence_rows=checked_rows,
    )
    if cohort_positions.size != len(arrays["row_indices"]):
        arrays, posterior, lookup = subset_inputs(
            arrays, posterior, lookup, cohort_positions, label="holdout_validation_cohort"
        )
        checked_rows = np.asarray(
            lookup.metadata["convergence_check"]["selected_row_indices"], dtype=np.int64
        )

    train_positions, test_positions = split_system_positions(
        arrays["row_indices"],
        test_fraction=args.test_fraction,
        seed=args.split_seed,
        convergence_rows=checked_rows,
    )
    train_arrays, train_posterior, train_lookup = subset_inputs(
        arrays, posterior, lookup, train_positions, label="t8_holdout_training_80_percent"
    )
    test_arrays, test_posterior, test_lookup = subset_inputs(
        arrays, posterior, lookup, test_positions, label="t8_holdout_test_20_percent"
    )
    split_metadata = {
        "schema": "t8-system-holdout-split-v1",
        "split_unit": "binary system row",
        "split_constraint": (
            "at least one parent Rice-convergence row is retained in each group"
        ),
        "validation_scope": (
            "MLR dynamics only: held-out u/u_sigma never enter either MLR fit; "
            "the full-sample T8a metallicity posterior is reused as a fixed covariate posterior"
        ),
        "split_seed": int(args.split_seed),
        "requested_test_fraction": float(args.test_fraction),
        "n_cohort_systems": int(len(arrays["row_indices"])),
        "n_training_systems": int(len(train_positions)),
        "n_test_systems": int(len(test_positions)),
        "cohort_row_indices_digest": array_digest(arrays["row_indices"]),
        "training_row_indices_digest": array_digest(train_arrays["row_indices"]),
        "test_row_indices_digest": array_digest(test_arrays["row_indices"]),
        "baseline_dir": str(args.baseline_dir.resolve()),
        "input_data": str(args.data.resolve()),
        "baseline_posterior_digest": posterior.metadata.get("input_data_digest"),
        "baseline_lookup_data_digest": lookup.metadata.get("parent_data_digest", lookup.metadata.get("data_digest")),
    }
    save_or_validate_split(
        args.output_dir,
        arrays["row_indices"],
        train_arrays["row_indices"],
        test_arrays["row_indices"],
        split_metadata,
    )
    print(
        f"System split: {len(train_positions)} train / {len(test_positions)} held out "
        f"(seed={args.split_seed})."
    )

    mass_surface, _ = t8.build_surfaces()
    wide_sigma = DEFAULT_SOLAR_SIGMA * float(args.wide_solar_factor)
    if args.stage in {"fit-default", "all"}:
        fit_training_model(
            label="default_anchor",
            solar_sigma=DEFAULT_SOLAR_SIGMA,
            arrays=train_arrays,
            posterior=train_posterior,
            lookup=train_lookup,
            mass_surface=mass_surface,
            output_dir=args.output_dir,
            warmup=args.warmup,
            samples=args.samples,
            chains=args.chains,
            seed=args.seed,
            target_accept=args.target_accept,
            grid_draws=args.grid_draws,
        )
    if args.stage in {"fit-wide", "all"}:
        fit_training_model(
            label="wide_anchor",
            solar_sigma=wide_sigma,
            arrays=train_arrays,
            posterior=train_posterior,
            lookup=train_lookup,
            mass_surface=mass_surface,
            output_dir=args.output_dir,
            warmup=args.warmup,
            samples=args.samples,
            chains=args.chains,
            seed=args.seed + 1000,
            target_accept=args.target_accept,
            grid_draws=args.grid_draws,
        )
    if args.stage in {"fit-default", "fit-wide"}:
        print(f"Fit stage complete. Outputs: {args.output_dir}")
        return

    default_trace = args.output_dir / "default_anchor" / t8.T8_MLR_NAME
    wide_trace = args.output_dir / "wide_anchor" / t8.T8_MLR_NAME
    for path in (default_trace, wide_trace):
        if not path.exists():
            raise FileNotFoundError(f"Required training fit is missing: {path}")
    require_saved_fit_metadata(
        args.output_dir / "default_anchor",
        solar_sigma=DEFAULT_SOLAR_SIGMA,
        training_rows=train_arrays["row_indices"],
    )
    require_saved_fit_metadata(
        args.output_dir / "wide_anchor",
        solar_sigma=wide_sigma,
        training_rows=train_arrays["row_indices"],
    )
    default_samples = load_grouped_samples(default_trace)
    wide_samples = load_grouped_samples(wide_trace)
    default_model = attach_saved_fit(
        mass_surface=mass_surface,
        solar_sigma=DEFAULT_SOLAR_SIGMA,
        arrays=train_arrays,
        posterior=train_posterior,
        lookup=train_lookup,
        trace_path=default_trace,
    )
    wide_model = attach_saved_fit(
        mass_surface=mass_surface,
        solar_sigma=wide_sigma,
        arrays=train_arrays,
        posterior=train_posterior,
        lookup=train_lookup,
        trace_path=wide_trace,
    )
    default_result = evaluate_heldout(
        default_model, default_samples, test_arrays, test_posterior, test_lookup,
        max_draws=args.evaluation_draws,
    )
    wide_result = evaluate_heldout(
        wide_model, wide_samples, test_arrays, test_posterior, test_lookup,
        max_draws=args.evaluation_draws,
    )
    print("Fitting the PARSEC-only f_outlier baseline on training systems...")
    parsec_posterior = fit_parsec_outlier_posterior(
        mass_surface,
        train_arrays,
        train_posterior,
        train_lookup,
        quadrature_nodes=args.parsec_f_nodes,
    )
    parsec_lppd = parsec_posterior_predictive(
        mass_surface, test_arrays, test_posterior, test_lookup, parsec_posterior
    )
    np.savez_compressed(
        args.output_dir / "parsec_f_outlier_posterior_t8.npz",
        f_outlier_grid=parsec_posterior["f_outlier_grid"],
        posterior_weights=parsec_posterior["posterior_weights"],
        quadrature_nodes=np.asarray(parsec_posterior["quadrature_nodes"]),
        posterior_mode=np.asarray(parsec_posterior["posterior_mode"]),
        local_laplace_sigma=np.asarray(parsec_posterior["local_laplace_sigma"]),
        integration_bounds=parsec_posterior["integration_bounds"],
        boundary_log_posterior_drop=parsec_posterior["boundary_log_posterior_drop"],
    )
    np.savez_compressed(
        args.output_dir / "heldout_loglikelihood_t8.npz",
        row_indices=test_arrays["row_indices"],
        default_draw_indices=default_result["draw_indices"],
        default_parsec_log_likelihood=default_result["parsec_log_likelihood"].astype(np.float32),
        default_mlr_log_likelihood=default_result["mlr_log_likelihood"].astype(np.float32),
        default_delta_log_likelihood=default_result["delta_log_likelihood"].astype(np.float32),
        wide_draw_indices=wide_result["draw_indices"],
        wide_parsec_log_likelihood=wide_result["parsec_log_likelihood"].astype(np.float32),
        wide_mlr_log_likelihood=wide_result["mlr_log_likelihood"].astype(np.float32),
        wide_delta_log_likelihood=wide_result["delta_log_likelihood"].astype(np.float32),
        parsec_refit_posterior_predictive_log_likelihood=parsec_lppd.astype(np.float64),
    )
    save_pointwise_csv(
        args.output_dir / "heldout_pointwise_t8.csv",
        test_arrays["row_indices"], parsec_lppd, default_result, wide_result,
    )

    default_grid = default_model.correction_grid(max_draws=args.grid_draws)
    wide_grid = wide_model.correction_grid(max_draws=args.grid_draws)
    baseline_model = make_mlr(mass_surface, solar_sigma=DEFAULT_SOLAR_SIGMA)
    baseline_model.posterior_samples = load_grouped_samples(baseline_trace_path)
    baseline_grid = baseline_model.correction_grid(max_draws=args.grid_draws)
    baseline_q, default_q, wide_q = save_curve_comparison(
        args.output_dir / "delta_star_curve_comparison_t8.npz",
        baseline_grid, default_grid, wide_grid,
    )
    x = np.asarray(default_grid["absg_grid"])
    z = np.asarray(default_grid["z_grid"])
    save_curve_csv(
        args.output_dir / "delta_star_curve_comparison_t8.csv",
        x, z, baseline_q, default_q, wide_q,
    )
    plot_curve_comparison(
        args.output_dir / "delta_star_curve_comparison_t8.png",
        x, z, baseline_q, default_q, wide_q,
    )
    plot_heldout(
        args.output_dir / "heldout_loglikelihood_comparison_t8.png",
        parsec_lppd, default_result, wide_result,
    )

    curve_summary = {
        "train_default_vs_full_median_curve": {
            "rms_difference_dex": float(np.sqrt(np.mean((default_q[1] - baseline_q[1]) ** 2))),
            "max_abs_difference_dex": float(np.max(np.abs(default_q[1] - baseline_q[1]))),
        },
        "wide_vs_default_train_median_curve": {
            "rms_difference_dex": float(np.sqrt(np.mean((wide_q[1] - default_q[1]) ** 2))),
            "max_abs_difference_dex": float(np.max(np.abs(wide_q[1] - default_q[1]))),
        },
    }
    report = {
        "schema": "t8-heldout-validation-report-v1",
        "interpretation": (
            "Only the 80% training systems enter either MLR fit. The 20% test "
            "systems are used only for the reported likelihood comparison."
        ),
        "validation_scope": (
            "This tests the T8b dynamical MLR conditional on the frozen T8a "
            "metallicity posterior. T8a used no dynamical u/u_sigma observations."
        ),
        "split": split_metadata,
        "default_anchor": {
            "heldout": summarize_heldout(default_result, parsec_lppd),
            "solar_anchor": solar_summary(default_samples, DEFAULT_SOLAR_SIGMA),
        },
        "wide_anchor": {
            "factor_relative_to_default": float(args.wide_solar_factor),
            "heldout": summarize_heldout(wide_result, parsec_lppd),
            "solar_anchor": solar_summary(wide_samples, wide_sigma),
        },
        "curve_comparison": curve_summary,
        "parsec_reference": {
            "model": "original PARSEC MLR with f_outlier refit on training systems",
            "f_outlier_prior": "Beta(3,12)",
            "quadrature_nodes": int(args.parsec_f_nodes),
            "posterior_mode": float(parsec_posterior["posterior_mode"]),
            "integration_bounds": parsec_posterior["integration_bounds"].tolist(),
            "boundary_log_posterior_drop": parsec_posterior[
                "boundary_log_posterior_drop"
            ].tolist(),
            "f_outlier_posterior_mean": float(
                np.sum(
                    parsec_posterior["f_outlier_grid"]
                    * parsec_posterior["posterior_weights"]
                )
            ),
        },
        "f_outlier_semantics": "basis weight only; not a physical outlier fraction",
    }
    (args.output_dir / "holdout_validation_summary_t8.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    print(f"Held-out validation complete. Outputs: {args.output_dir}")


if __name__ == "__main__":
    main()
