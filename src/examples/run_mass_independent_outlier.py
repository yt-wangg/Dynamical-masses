#!/usr/bin/env python3
"""Compare fixed-shape T8 MLRs with an outlier density defined in raw u.

Modes (the baseline and output directories are always explicit):

  validation  Verify data/posterior/lookup consistency, compute the per-system
              mass-independent outlier convolution, check it against adaptive
              reference integrals for every system, and write the root-level
              model settings, outlier likelihood, and validation report.
  timing      Full-sample single-chain timing trial; requires a completed
              validation in the output directory. The trial informs resource
              planning only and never enters the formal posterior.
  chain       One formal MCMC chain; the initial point is a saved draw from the
              baseline (A-group) posterior, the NUTS metric is the fixed dense
              baseline covariance, and results go to OUTPUT/chain_XX.
  full        Self-contained validation plus one median-initialized fit with
              the comparison plot; the exploratory quick workflow.
"""

import argparse
import csv
from dataclasses import replace
import json
from multiprocessing import Pool
from pathlib import Path
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import i0e, logit, ndtr

import run_hierarchical_metallicity_test as workflow
from binary_masses import hierarchical_metallicity as hm


ROOT = Path(__file__).resolve().parents[2]
OUTLIER_FLOOR = 1e-30
EXPECTED_SHAPE = {
    "good_basis_quadratic": 0.002544,
    "good_basis_turnover": 35.67,
    "good_basis_scale": 3.1,
    "outlier_u0": 40.0,
    "outlier_sigma": 13.0,
    "tilde_u_max": 80.0,
}


def parse_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--baseline", type=Path, required=True,
                        help="Directory holding the A-group posterior, lookup, and MLR.")
    parser.add_argument("--output", type=Path, required=True,
                        help="Output directory; chain mode writes to OUTPUT/chain_XX.")
    parser.add_argument("--mode", choices=("validation", "timing", "chain", "full"),
                        default="full")
    parser.add_argument("--chain-index", type=int, default=None,
                        help="Chain mode: baseline posterior chain used for the initial point.")
    parser.add_argument("--start-draw", type=int, default=500,
                        help="Chain mode: draw index within the baseline chain.")
    parser.add_argument("--reference-workers", type=int, default=8,
                        help="Processes for the adaptive reference integrals.")
    parser.add_argument("--validation-root", type=Path, default=None,
                        help="Directory holding the validated root artifacts "
                             "(default: --output; set for --mode timing when the "
                             "trial writes to its own subdirectory).")
    parser.add_argument("--progress-bar", action="store_true",
                        help="Show the NumPyro progress bar (off by default for nohup logs).")
    runner_args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining
    return runner_args, workflow.parse_args()


def outlier_log_likelihood(u, sigma, nodes=128):
    """Rice convolution of N(40,13), truncated to raw u in [0,80]."""
    x, weights = np.polynomial.legendre.leggauss(nodes)
    lo = np.minimum(np.maximum(u - 14 * sigma, 0), 80)
    hi = np.minimum(u + 14 * sigma, 80)
    half = (hi - lo) / 2
    v = (hi + lo)[:, None] / 2 + half[:, None] * x
    s = sigma[:, None]
    obs = u[:, None]
    log_rice = np.log(obs / s**2) - (obs - v)**2 / (2 * s**2)
    log_rice += np.log(i0e(obs * v / s**2))
    norm = ndtr(40 / 13) - ndtr(-40 / 13)
    log_bad = -0.5 * ((v - 40) / 13)**2 - np.log(13 * np.sqrt(2 * np.pi) * norm)
    integral = half * np.sum(weights * np.exp(log_rice + log_bad), axis=1)
    return np.log(np.maximum(integral, OUTLIER_FLOOR))


def _reference_component(pair):
    u, sigma = pair
    return hm.rice_component_reference_integral(
        float(u), float(sigma), 1.0, component="bad")


def reference_integrals(u, sigma, workers):
    with Pool(processes=int(workers)) as pool:
        return np.array(pool.map(_reference_component, zip(u.tolist(), sigma.tolist()),
                                 chunksize=8))


def validate_outlier_integral(log_bad, u, sigma, workers):
    """Gate the fit: non-floor systems within 1e-3 in log, floors confirmed."""
    started = time.monotonic()
    reference = reference_integrals(u, sigma, workers)
    floor = (u - 14.0 * sigma) >= 80.0
    nonfloor_error = float(np.max(np.abs(
        log_bad[~floor] - np.log(np.maximum(reference[~floor], OUTLIER_FLOOR)))))
    floor_reference_max = float(np.max(reference[floor])) if np.any(floor) else 0.0
    report = {
        "n_systems": int(u.size),
        "n_floor": int(np.count_nonzero(floor)),
        "floor_rule": "u - 14 sigma >= 80 collapses the local quadrature window",
        "nonfloor_max_log_error": nonfloor_error,
        "nonfloor_tolerance": 1e-3,
        "floor_reference_max": floor_reference_max,
        "floor_reference_threshold": OUTLIER_FLOOR,
        "reference_workers": int(workers),
        "elapsed_seconds": time.monotonic() - started,
        "pass": bool(nonfloor_error < 1e-3 and floor_reference_max <= OUTLIER_FLOOR),
    }
    return report, reference


def load_systems(runner_args, args):
    posterior = hm.MetallicityPosteriorGrid.load(
        runner_args.baseline / workflow.T8_POSTERIOR_NAME)
    arrays = workflow.filter_data(workflow.load_real_data(args.data), max_systems=None,
                                  seed=args.seed, fixed_rows=posterior.row_indices)
    workflow._require_t8_posterior_metadata(posterior, mock=args.mock, arrays=arrays)
    lookup = hm.DynamicsLikelihoodLookup.load(runner_args.baseline / workflow.T8_LOOKUP_NAME)
    lookup.validate_for(row_indices=arrays["row_indices"], u=arrays["u"],
                        u_sigma=arrays["u_sigma"])
    checks = {}
    for key, expected in EXPECTED_SHAPE.items():
        actual = float(lookup.metadata[key])
        if not np.isclose(actual, expected, rtol=1e-6, atol=0.0):
            raise ValueError(f"Lookup {key}={actual!r} does not match the fixed {expected!r}.")
        checks[key] = {"expected": expected, "lookup": actual}
    if not (lookup.metadata.get("good_basis_normalized")
            and float(lookup.metadata.get("good_basis_support_integral", 0.0)) == 1.0):
        raise ValueError("The lookup good basis is not the unit-integral T8.1 normalization.")
    if lookup.metadata.get("jacobian") != "1/s":
        raise ValueError("The lookup does not carry the 1/s Jacobian of the A-group model.")
    return posterior, arrays, lookup, checks


def modified_lookup(lookup, log_bad, error):
    new_bad = np.broadcast_to(log_bad[:, None], lookup.log_bad.shape).astype(np.float32).copy()
    assert np.all(new_bad == new_bad[:, :1])
    return replace(lookup, log_bad=new_bad,
                   metadata={**lookup.metadata, "outlier_coordinate": "raw_u",
                             "outlier_mass_dependent": False,
                             "outlier_integration": "128-point local Gauss-Legendre, 14 sigma",
                             "outlier_reference_max_log_error": error})


def build_model(runner_args, args, posterior, arrays, lookup, log_bad, error):
    modified = modified_lookup(lookup, log_bad, error)
    mass_surface, _ = workflow.build_surfaces()
    mlr = workflow._make_t8_mlr(mass_surface, args)
    mlr.set_data(row_indices=arrays["row_indices"], u=arrays["u"],
                 u_sigma=arrays["u_sigma"], absg=arrays["absg"],
                 metallicity_grid=posterior, dynamics_lookup=modified)
    return mlr


def load_saved_outlier(runner_args, posterior):
    """Chain/timing mode: reuse the validated root integral instead of refitting it."""
    validation_root = runner_args.validation_root or runner_args.output
    saved = np.load(validation_root / "outlier_likelihood.npz")
    if not np.array_equal(saved["row_indices"], posterior.row_indices):
        raise ValueError("Saved outlier likelihood rows do not match the posterior.")
    report = json.loads((validation_root / "validation.json").read_text())
    if not report.get("pass", False):
        raise ValueError("The root validation report did not pass; refusing to fit.")
    return np.asarray(saved["log_likelihood"], dtype=np.float64), report


def baseline_posterior_draws(runner_args):
    saved = np.load(runner_args.baseline / workflow.T8_MLR_NAME)
    if runner_args.mode == "chain":
        chains, draws = saved["posterior__f_outlier"].shape[:2]
        if not 0 <= runner_args.chain_index < chains:
            raise ValueError(f"--chain-index must be in [0, {chains}).")
        if not 0 <= runner_args.start_draw < draws:
            raise ValueError(f"--start-draw must be in [0, {draws}).")
    return saved


def initial_values_and_metric(runner_args, mlr, saved):
    parameter_names = sorted(mlr.sample_parameter_names)
    initial_values, unconstrained = {}, []
    for name in parameter_names:
        values = saved["posterior__" + name]
        if runner_args.mode == "chain":
            initial_values[name] = values[runner_args.chain_index, runner_args.start_draw]
        else:
            initial_values[name] = np.median(
                values.reshape((-1,) + values.shape[2:]), axis=0)
        flat = values.reshape((-1,) + values.shape[2:])
        transformed = logit(flat) if name == "f_outlier" else flat
        unconstrained.append(transformed.reshape(len(transformed), -1))
    covariance = np.cov(np.concatenate(unconstrained, axis=1), rowvar=False)
    covariance = 0.95 * covariance + 0.05 * np.diag(np.diag(covariance))
    return initial_values, covariance


def fit_and_save(runner_args, args, mlr, error, output_dir):
    from numpyro.infer.initialization import init_to_value

    output_dir.mkdir(parents=True, exist_ok=True)
    saved = baseline_posterior_draws(runner_args)
    initial_values, covariance = initial_values_and_metric(runner_args, mlr, saved)
    started = time.monotonic()
    sampler = mlr.run_mcmc(num_warmup=args.warmup, num_samples=args.samples,
                           num_chains=args.chains, seed=args.seed,
                           target_accept_prob=args.target_accept,
                           init_strategy=init_to_value(values=initial_values),
                           dense_mass=True, inverse_mass_matrix=covariance,
                           adapt_mass_matrix=False,
                           progress_bar=bool(runner_args.progress_bar))
    elapsed = time.monotonic() - started
    workflow.save_grouped_mcmc(sampler, output_dir / workflow.T8_MLR_NAME)
    workflow.save_sampler_diagnostics(sampler, output_dir / "mlr_diagnostics_t8.json")
    workflow.save_posterior_summary(mlr.posterior_samples, output_dir / "mlr_summary_t8.csv")
    config = {
        "mode": runner_args.mode,
        "baseline_directory": str(runner_args.baseline),
        "chain_index": runner_args.chain_index,
        "start_draw": runner_args.start_draw,
        "seed": int(args.seed),
        "warmup": int(args.warmup), "samples": int(args.samples),
        "chains": int(args.chains), "target_accept": float(args.target_accept),
        "dense_metric": "0.95 Cov + 0.05 diag(Cov) of the baseline posterior "
                        "in unconstrained space, f_outlier logit-transformed; "
                        "step size adapted only",
        "initialization": (
            f"baseline chain {runner_args.chain_index} draw {runner_args.start_draw}"
            if runner_args.mode == "chain"
            else "baseline posterior medians"),
        "outlier_reference_max_log_error": error,
        "elapsed_seconds": elapsed,
        "elapsed_includes_jit_compile": True,
        "seconds_per_mcmc_iteration": float(elapsed / (args.warmup + args.samples)),
        "mean_nuts_steps_per_iteration": float(np.mean(
            np.asarray(sampler.get_extra_fields()["num_steps"]))),
    }
    (output_dir / "chain_config.json").write_text(json.dumps(config, indent=2))
    return config


def compare_plot(grid, runner_args):
    old = np.load(runner_args.baseline / "mlr_derived_grid_t8.npz")
    x, z = grid["absg_grid"], grid["z_grid"]
    selected = [int(np.argmin(abs(z - value))) for value in [-0.8, 0, 0.4]]
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.7), sharex=True,
                             gridspec_kw={"height_ratios": [2, 1]})
    rows = []
    for col, iz in enumerate(selected):
        ax, ratio_ax = axes[:, col]
        for result, label, color in [(old, "Mass-scaled outliers", "#7b6ba8"),
                                     (grid, "Mass-independent outliers", "#da7030")]:
            p = result["mass_percentiles"][:, iz]
            ax.plot(x, p[1], color=color, label=label)
            ax.fill_between(x, p[0], p[2], color=color, alpha=0.18)
        ax.plot(x, grid["parsec_mass"][iz], "--", color="0.4", label="PARSEC")
        ax.set_title(f"[M/H] = {z[iz]:+.1f}")
        ratio = 100 * (grid["mass_percentiles"][1, iz] / old["mass_percentiles"][1, iz] - 1)
        ratio_ax.plot(x, ratio, color="#da7030")
        ratio_ax.axhline(0, color="0.5", lw=0.8)
        ratio_ax.set_xlabel(r"$M_G$ [mag]")
        for mg in [5, 7, 9, 11, 13]:
            ix = int(np.argmin(abs(x - mg)))
            rows.append({"mh": float(z[iz]), "M_G": float(x[ix]),
                         "old_mass": float(old["mass_percentiles"][1, iz, ix]),
                         "new_mass_p16": float(grid["mass_percentiles"][0, iz, ix]),
                         "new_mass_p50": float(grid["mass_percentiles"][1, iz, ix]),
                         "new_mass_p84": float(grid["mass_percentiles"][2, iz, ix]),
                         "median_change_percent": float(ratio[ix])})
    axes[0, 0].set_ylabel(r"Mass [$M_\odot$]")
    axes[1, 0].set_ylabel("Median change [%]")
    axes[0, 0].legend(fontsize=9)
    fig.suptitle("Fixed default dynamics shape · exploratory sample\n"
                 "Shading: 16–84% posterior interval", fontsize=12)
    fig.tight_layout()
    fig.savefig(runner_args.output / "mlr_comparison.png", dpi=180)
    fig.savefig(runner_args.output / "mlr_comparison.pdf")
    plt.close(fig)
    with (runner_args.output / "mlr_comparison.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main():
    runner_args, args = parse_args()
    posterior, arrays, lookup, checks = load_systems(runner_args, args)

    if runner_args.mode == "validation":
        log_bad = outlier_log_likelihood(arrays["u"], arrays["u_sigma"])
        validation, reference = validate_outlier_integral(
            log_bad, arrays["u"], arrays["u_sigma"], runner_args.reference_workers)
        runner_args.output.mkdir(parents=True, exist_ok=True)
        constant_on_mass = bool(np.all(
            np.broadcast_to(log_bad[:, None], lookup.log_bad.shape) == log_bad[:, None]))
        if not constant_on_mass:
            raise AssertionError("The mass-independent outlier is not constant on the mass grid.")
        np.savez_compressed(runner_args.output / "outlier_likelihood.npz",
                            row_indices=posterior.row_indices, log_likelihood=log_bad)
        np.savez_compressed(runner_args.output / "validation_reference_integrals.npz",
                            row_indices=posterior.row_indices, reference_integral=reference)
        metadata = json.loads((runner_args.baseline / "mlr_model.json").read_text())
        metadata.update(mode=runner_args.mode, outlier_coordinate="raw_u",
                        outlier_mass_dependent=False,
                        baseline_directory=str(runner_args.baseline),
                        n_systems=int(arrays["u"].size),
                        fixed_shape_checks=checks, validation=validation,
                        metallicity_posterior_source=str(
                            runner_args.baseline / workflow.T8_POSTERIOR_NAME),
                        dynamics_lookup_source=str(
                            runner_args.baseline / workflow.T8_LOOKUP_NAME))
        (runner_args.output / "model.json").write_text(json.dumps(metadata, indent=2))
        (runner_args.output / "validation.json").write_text(json.dumps(
            {**validation, "fixed_shape_checks": checks,
             "outlier_constant_on_mass_grid": constant_on_mass}, indent=2))
        print(json.dumps(validation, indent=2), flush=True)
        if not validation["pass"]:
            raise SystemExit("Outlier integral validation failed; do not fit.")
        return

    if runner_args.mode in ("timing", "chain"):
        saved_log_bad, validation = load_saved_outlier(runner_args, posterior)
        log_bad = outlier_log_likelihood(arrays["u"], arrays["u_sigma"])
        if not np.array_equal(log_bad, saved_log_bad):
            raise ValueError("Recomputed outlier likelihood differs from the validated save.")
        error = float(validation["nonfloor_max_log_error"])
        mlr = build_model(runner_args, args, posterior, arrays, lookup, log_bad, error)
        output_dir = (runner_args.output / f"chain_{runner_args.chain_index:02d}"
                      if runner_args.mode == "chain" else runner_args.output)
        config = fit_and_save(runner_args, args, mlr, error, output_dir)
        print(json.dumps(config, indent=2), flush=True)
        return

    # full mode: exploratory validation + median-initialized fit + comparison plot
    log_bad = outlier_log_likelihood(arrays["u"], arrays["u_sigma"])
    validation, reference = validate_outlier_integral(
        log_bad, arrays["u"], arrays["u_sigma"], runner_args.reference_workers)
    print(json.dumps(validation, indent=2), flush=True)
    if not validation["pass"]:
        raise SystemExit("Outlier integral validation failed; do not fit.")
    error = float(validation["nonfloor_max_log_error"])
    mlr = build_model(runner_args, args, posterior, arrays, lookup, log_bad, error)
    runner_args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(runner_args.output / "outlier_likelihood.npz",
                        row_indices=posterior.row_indices, log_likelihood=log_bad)
    metadata = json.loads((runner_args.baseline / "mlr_model.json").read_text())
    metadata.update(outlier_coordinate="raw_u", outlier_mass_dependent=False,
                    baseline_directory=str(runner_args.baseline),
                    warmup=args.warmup, samples=args.samples, chains=args.chains,
                    seed=args.seed,
                    initialization="baseline posterior medians; fixed dense baseline covariance for NUTS",
                    metallicity_posterior_source=str(
                        runner_args.baseline / workflow.T8_POSTERIOR_NAME))
    (runner_args.output / "model.json").write_text(json.dumps(metadata, indent=2))
    started = time.monotonic()
    from numpyro.infer.initialization import init_to_value
    saved = np.load(runner_args.baseline / workflow.T8_MLR_NAME)
    initial_values, covariance = initial_values_and_metric(runner_args, mlr, saved)
    sampler = mlr.run_mcmc(num_warmup=args.warmup, num_samples=args.samples,
                           num_chains=args.chains, seed=args.seed,
                           target_accept_prob=args.target_accept,
                           init_strategy=init_to_value(values=initial_values),
                           dense_mass=True, inverse_mass_matrix=covariance,
                           adapt_mass_matrix=False,
                           progress_bar=bool(runner_args.progress_bar))
    workflow.save_grouped_mcmc(sampler, runner_args.output / workflow.T8_MLR_NAME)
    workflow.save_sampler_diagnostics(sampler, runner_args.output / "mlr_diagnostics_t8.json")
    workflow.save_posterior_summary(mlr.posterior_samples,
                                    runner_args.output / "mlr_summary_t8.csv")
    grid = mlr.correction_grid()
    np.savez_compressed(runner_args.output / "mlr_derived_grid_t8.npz", **grid)
    rows = compare_plot(grid, runner_args)
    result = {"elapsed_seconds": time.monotonic() - started,
              "f_outlier_percentiles": np.percentile(
                  mlr.posterior_samples["f_outlier"], [16, 50, 84]).tolist(),
              "comparison": rows}
    (runner_args.output / "summary.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
