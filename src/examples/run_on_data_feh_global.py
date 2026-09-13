#!/usr/bin/env python3
"""
FeH-Global Difference-Polynomial Fitting on Real Data

This script mirrors the mock-data test in:
    Validation/V16. FeH-global model test.ipynb
but runs on real Gaia binary data. It uses the updated
DifferencePolyFehMLR interface (feh_model, cross_mode, quad_mode/quad_mask).

The isochrone baseline uses MH from the interpolated grid.
    If your system metallicity is [Fe/H], you are implicitly assuming MH approx [Fe/H].

Set ``use_feh_uncertainty=True`` and provide ``feh_sigma_column`` to marginalize each likelihood over latent true [Fe/H]. The intrinsic FeH distribution is inferred jointly as a smooth fixed-basis Gaussian mixture with a Dirichlet prior on its weights.

"""

import argparse
import json
import os
import sys

import numpy as np
from astropy.table import Table
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.special import logsumexp

# Add the src-layout package to Python path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from binary_masses.differencepoly_feh import (  # noqa: E402
    DifferencePolyFehMLR,
    DifferencePolyFehMassAbsgModel,
    IsochroneMassSurfaceModel,
)


def quadratic_stage_settings(stage):
    """Return the nested quadratic model settings used for comparison runs."""
    stage = str(stage).strip().lower()
    if stage == "core":
        return {"cross_mode": "free", "quad_mode": "core", "quad_mask": None}
    if stage == "quadratic":
        return {
            "cross_mode": "free",
            "quad_mode": None,
            "quad_mask": {
                "a0": True,
                "b1": True,
                "a1": True,
                "b2": True,
                "a2": True,
                "c_xy": False,
            },
        }
    if stage == "cross":
        return {"cross_mode": "free", "quad_mode": "full", "quad_mask": None}
    raise ValueError("stage must be 'core', 'quadratic', or 'cross'.")


def _save_grouped_mcmc_with_diagnostics(fitter, path):
    """Save chain-aware posterior arrays and NUTS diagnostics in one NPZ."""
    posterior = fitter.sampler.get_samples(group_by_chain=True)
    sample_stats = fitter.sampler.get_extra_fields(group_by_chain=True)

    payload = {
        f"posterior__{name}": np.asarray(values)
        for name, values in posterior.items()
    }
    for name, values in sample_stats.items():
        payload[f"sample_stats__{name}"] = np.asarray(values)

    if "num_steps" in sample_stats:
        num_steps = np.asarray(sample_stats["num_steps"])
        tree_depth = np.ceil(np.log2(num_steps + 1)).astype(np.int16)
        payload["sample_stats__tree_depth"] = tree_depth

    np.savez_compressed(path, **payload)
    return path


def _make_heldout_split(n_systems, fraction, seed):
    fraction = float(fraction)
    if not 0.0 < fraction < 1.0:
        raise ValueError("heldout_fraction must lie strictly between 0 and 1.")
    n_heldout = max(1, int(round(n_systems * fraction)))
    if n_heldout >= n_systems:
        raise ValueError("heldout_fraction leaves no training systems.")
    heldout = np.random.default_rng(seed).choice(
        n_systems, size=n_heldout, replace=False
    )
    heldout.sort()
    is_heldout = np.zeros(n_systems, dtype=bool)
    is_heldout[heldout] = True
    return np.flatnonzero(~is_heldout), heldout


def _save_heldout_predictive_result(
    *,
    fitter,
    heldout_data,
    heldout_indices,
    output_dir,
    suffix,
    model_label,
    feh_column,
    feh_sigma_column,
    use_feh_uncertainty,
    outlier_kappa,
    outlier_kappa_scale,
    max_posterior_samples,
    seed,
):
    """Evaluate and save a pointwise posterior predictive log score."""
    fitter.set_data(
        u_values=heldout_data["u"],
        u_sigma_values=(
            heldout_data["u_sigma"] if "u_sigma" in heldout_data.colnames else None
        ),
        absg1_values=heldout_data["absg1"],
        absg2_values=heldout_data["absg2"],
        feh_values=heldout_data[feh_column],
        feh_sigma_values=(
            heldout_data[feh_sigma_column] if use_feh_uncertainty else None
        ),
        outlier_kappa=outlier_kappa,
        outlier_kappa_scale=outlier_kappa_scale,
    )
    log_likelihood = fitter.heldout_log_likelihood_draws(
        max_posterior_samples=max_posterior_samples,
        seed=seed,
    )
    pointwise_lppd = logsumexp(log_likelihood, axis=0) - np.log(
        log_likelihood.shape[0]
    )
    total_lppd = float(np.sum(pointwise_lppd))
    total_se = float(
        np.sqrt(pointwise_lppd.size * np.var(pointwise_lppd, ddof=1))
    )

    npz_path = os.path.join(output_dir, f"heldout_predictive_{suffix}.npz")
    np.savez_compressed(
        npz_path,
        heldout_indices=np.asarray(heldout_indices),
        log_likelihood=log_likelihood,
        pointwise_lppd=pointwise_lppd,
    )
    summary = {
        "model_label": model_label,
        "n_heldout": int(pointwise_lppd.size),
        "n_posterior_draws": int(log_likelihood.shape[0]),
        "heldout_lppd": total_lppd,
        "heldout_lppd_per_system": float(np.mean(pointwise_lppd)),
        "heldout_lppd_se": total_se,
        "npz_path": os.path.basename(npz_path),
    }
    json_path = os.path.join(output_dir, f"heldout_predictive_{suffix}.json")
    with open(json_path, "w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2)
    print(
        f"Held-out log predictive density ({model_label}): "
        f"{total_lppd:.2f} over {pointwise_lppd.size} systems "
        f"({np.mean(pointwise_lppd):.4f} per system)."
    )
    return summary


def compare_heldout_predictive_results(output_dir):
    """Print paired held-out score differences for saved model stages."""
    import glob

    paths = sorted(glob.glob(os.path.join(output_dir, "heldout_predictive_*.json")))
    if not paths:
        raise FileNotFoundError(f"No heldout_predictive_*.json files in {output_dir}")

    results = []
    for path in paths:
        with open(path, encoding="utf-8") as stream:
            summary = json.load(stream)
        npz_path = os.path.join(output_dir, summary["npz_path"])
        with np.load(npz_path, allow_pickle=False) as saved:
            results.append(
                {
                    **summary,
                    "indices": np.asarray(saved["heldout_indices"]),
                    "pointwise": np.asarray(saved["pointwise_lppd"]),
                }
            )

    order = {"core": 0, "quadratic": 1, "cross": 2}
    labels = [item["model_label"] for item in results]
    duplicates = sorted({label for label in labels if labels.count(label) > 1})
    if duplicates:
        raise ValueError(
            "Multiple held-out result files exist for stage(s) "
            f"{duplicates}. Move obsolete runs out of {output_dir} before comparing."
        )
    missing = [label for label in order if label not in labels]
    if missing:
        raise ValueError(
            f"Missing held-out results for {missing}; run all three stages first."
        )
    results.sort(key=lambda item: order.get(item["model_label"], 99))
    print("\nHeld-out predictive comparison (larger is better):")
    for item in results:
        print(
            f"  {item['model_label']:10s}  "
            f"lppd={item['heldout_lppd']:12.2f}  "
            f"per-system={item['heldout_lppd_per_system']:.5f}"
        )

    print("\nPaired differences:")
    for simpler, larger in zip(results[:-1], results[1:]):
        if not np.array_equal(simpler["indices"], larger["indices"]):
            raise ValueError(
                f"Held-out indices differ for {simpler['model_label']} and "
                f"{larger['model_label']}; paired comparison is invalid."
            )
        delta_i = larger["pointwise"] - simpler["pointwise"]
        delta = float(np.sum(delta_i))
        se = float(np.sqrt(delta_i.size * np.var(delta_i, ddof=1)))
        print(
            f"  {larger['model_label']} - {simpler['model_label']}: "
            f"Delta lppd={delta:.2f} +/- {se:.2f}"
        )
    return results


def test_differencepoly_feh_model(
    data,
    *,
    output_dir,
    uncertainty_model="rice",
    feh_column="feh",
    use_feh_uncertainty=False,
    feh_sigma_column=None,
    feh_quadrature_nodes=11,
    feh_population_components=10,
    feh_population_concentration=1.0,
    feh_population_kernel_scale=1.0,
    feh_min=None,
    feh_max=None,
    order=2,
    absg_min=None,
    absg_max=None,
    mass_min=0.05,
    mass_max=2.0,
    absg_monotone_strength=10.0,
    feh_monotone_strength=0.0,
    monotone_n_feh=5,
    coeff_prior_scale=5.0,
    feh_coeff_prior_scale=1.0,
    f_outlier_init=0.1,
    outlier_u0_init=35.0,
    outlier_sigma_init=15.0,
    outlier_kappa=None,
    outlier_kappa_scale=0.05,
    fit_outlier_params=False,
    param_truths=None,
    truth_model=None,
    plot_corner_truths=True,
    num_warmup=800,
    num_samples=2000,
    num_chains=1,
    use_dense_mass=False,
    seed=42,
    iso_data_dir="data/interpolated_mass_data",
    feh_plot_values=None,
    feh_model="quadratic2d",
    cross_mode="free",
    quad_mode=None,
    quad_mask=None,
    anchor_enabled=False,
    anchor_absg=4.67,
    anchor_feh=0.0,
    anchor_mass=1.0,
    anchor_sigma=0.05,
    model_label=None,
    save_full_mcmc=False,
    heldout_fraction=0.0,
    heldout_seed=20260812,
    heldout_max_posterior_samples=256,
):
    """
    Fit the DifferencePolyFehMLR model on all data (no metallicity binning).

    Notes
    -----
    - The isochrone baseline uses MH from the interpolated grid.
      If your system metallicity is [Fe/H], you are implicitly assuming MH approx [Fe/H].
    - Set ``use_feh_uncertainty=True`` and provide ``feh_sigma_column`` to
      marginalize each likelihood over latent true [Fe/H]. The intrinsic FeH
      distribution is inferred jointly as a smooth fixed-basis Gaussian mixture
      with a Dirichlet prior on its weights.
    """
    print("\n" + "=" * 60)
    print("TESTING DIFFERENCE-POLY WITH CONTINUOUS METALLICITY")
    print("=" * 60)

    full_data = data

    if feh_column not in data.colnames:
        raise ValueError(f"Metallicity column {feh_column!r} not found in data.")
    if use_feh_uncertainty:
        if feh_sigma_column is None:
            raise ValueError(
                "feh_sigma_column is required when use_feh_uncertainty=True."
            )
        if feh_sigma_column not in data.colnames:
            raise ValueError(
                f"Metallicity uncertainty column {feh_sigma_column!r} not found in data."
            )

    def _robust_range(values, *, lo=1.0, hi=99.0):
        values = np.asarray(values, dtype=float)
        values = values[np.isfinite(values)]
        if values.size == 0:
            raise ValueError("No finite values available to infer plotting range.")
        return np.nanpercentile(values, [lo, hi])

    absg_all = np.concatenate(
        [np.asarray(full_data["absg1"]), np.asarray(full_data["absg2"])]
    )
    absg_lo, absg_hi = _robust_range(absg_all)
    plot_absg_min = absg_lo if absg_min is None else absg_min
    plot_absg_max = absg_hi if absg_max is None else absg_max
    if plot_absg_max <= plot_absg_min:
        raise ValueError(f"Invalid plotting absg range: [{plot_absg_min}, {plot_absg_max}].")

    feh_lo, feh_hi = _robust_range(full_data[feh_column])
    plot_feh_min = feh_lo if feh_min is None else feh_min
    plot_feh_max = feh_hi if feh_max is None else feh_max
    if plot_feh_max <= plot_feh_min:
        raise ValueError(f"Invalid plotting feh range: [{plot_feh_min}, {plot_feh_max}].")

    if feh_min is not None and feh_max is not None:
        observed_feh = np.asarray(full_data[feh_column], dtype=float)
        below = np.mean(observed_feh < feh_min)
        above = np.mean(observed_feh > feh_max)
        print(
            f"Observed metallicities outside latent/isochrone support "
            f"[{feh_min:+.2f}, {feh_max:+.2f}]: "
            f"{below:.2%} below, {above:.2%} above."
        )
        if use_feh_uncertainty and (below > 0 or above > 0):
            print(
                "These rows are retained: their latent true metallicity is "
                "marginalized inside the supported interval, not hard-cut."
            )

    os.makedirs(output_dir, exist_ok=True)

    heldout_data = None
    heldout_indices = None
    if float(heldout_fraction) > 0:
        train_indices, heldout_indices = _make_heldout_split(
            len(full_data), heldout_fraction, heldout_seed
        )
        data = full_data[train_indices]
        heldout_data = full_data[heldout_indices]
        split_path = os.path.join(
            output_dir,
            f"heldout_split_seed{heldout_seed}_frac{heldout_fraction:.3f}.npz",
        )
        np.savez_compressed(
            split_path,
            n_systems=np.asarray(len(full_data)),
            train_indices=train_indices,
            heldout_indices=heldout_indices,
        )
        print(
            f"Held-out split: {len(data)} training + {len(heldout_data)} test "
            f"systems (seed={heldout_seed})."
        )

    if model_label is None:
        if str(quad_mode).lower() in {"core", "a0a1b1", "linear"}:
            model_label = "core"
        elif isinstance(quad_mask, dict) and not quad_mask.get("c_xy", True):
            model_label = "quadratic"
        elif str(quad_mode).lower() in {"full", "all"}:
            model_label = "cross"
        else:
            model_label = "custom"

    print("1. Loading isochrone mass surface...")
    iso_surface = IsochroneMassSurfaceModel.from_interpolated_mass_data(
        iso_data_dir,
        mass_min=mass_min,
    )

    print("Setting up truth model...")
    truth_model_auto = truth_model
    if truth_model_auto is None and param_truths is not None:
        truth_absg_min = plot_absg_min if absg_min is None else absg_min
        truth_absg_max = plot_absg_max if absg_max is None else absg_max
        truth_feh_min = plot_feh_min if feh_min is None else feh_min
        truth_feh_max = plot_feh_max if feh_max is None else feh_max
        truth_model_auto = DifferencePolyFehMassAbsgModel(
            order=order,
            absg_min=truth_absg_min,
            absg_max=truth_absg_max,
            mass_min=mass_min,
            pivot=(truth_absg_min + truth_absg_max) / 2.0,
            deriv_penalty_strength=0.0,
            feh_monotone_strength=0.0,
            isochrone_surface_model=iso_surface,
            feh_min=truth_feh_min,
            feh_max=truth_feh_max,
            feh_model=feh_model,
            cross_mode=cross_mode,
        )

    print("2. Initializing DifferencePolyFehMLR fitter...")
    fitter = DifferencePolyFehMLR(
        order=order,
        isochrone_surface_model=iso_surface,
        feh_min=feh_min,
        feh_max=feh_max,
        mass_min=mass_min,
        mass_max=mass_max,
        absg_min=absg_min,
        absg_max=absg_max,
        uncertainty_model=uncertainty_model,
        f_outlier=f_outlier_init,
        outlier_u0=outlier_u0_init,
        outlier_sigma=outlier_sigma_init,
        outlier_kappa=outlier_kappa,
        outlier_kappa_scale=outlier_kappa_scale,
        fit_outlier_params=fit_outlier_params,
        param_truths=param_truths,
        absg_monotone_strength=absg_monotone_strength,
        feh_monotone_strength=feh_monotone_strength,
        coeff_prior_scale=coeff_prior_scale,
        feh_coeff_prior_scale=feh_coeff_prior_scale,
        monotone_n_feh=monotone_n_feh,
        feh_model=feh_model,
        cross_mode=cross_mode,
        anchor_enabled=anchor_enabled,
        anchor_absg=anchor_absg,
        anchor_feh=anchor_feh,
        anchor_mass=anchor_mass,
        anchor_sigma=anchor_sigma,
        use_feh_uncertainty=use_feh_uncertainty,
        feh_quadrature_nodes=feh_quadrature_nodes,
        feh_population_components=feh_population_components,
        feh_population_concentration=feh_population_concentration,
        feh_population_kernel_scale=feh_population_kernel_scale,
    )

    if plot_corner_truths and param_truths is not None:
        expected = fitter.poly_model.n_params + (3 if fit_outlier_params else 0)
        if len(param_truths) == expected:
            fitter.param_truths = list(param_truths)
        else:
            print(
                f"Skipping corner truths (expected {expected} values, got {len(param_truths)})."
            )

    print("3. Setting data...")
    fitter.set_data(
        u_values=data["u"],
        u_sigma_values=data["u_sigma"] if "u_sigma" in data.colnames else None,
        absg1_values=data["absg1"],
        absg2_values=data["absg2"],
        feh_values=data[feh_column],
        feh_sigma_values=(data[feh_sigma_column] if use_feh_uncertainty else None),
        outlier_kappa=outlier_kappa,
        outlier_kappa_scale=outlier_kappa_scale,
    )

    print("4. Running MCMC...")
    outlier_tag = "outlierfit" if fit_outlier_params else "outlierfixed"
    feh_error_tag = (
        f"feherr-{feh_quadrature_nodes}node-pop{feh_population_components}comp"
        f"-ks{feh_population_kernel_scale:.2f}"
        f"-alpha{feh_population_concentration:.2f}"
        if use_feh_uncertainty
        else "feherr-off"
    )
    suffix = (
        f"_model-diffpoly2dfeh_unc-{uncertainty_model}"
        f"_stage-{model_label}"
        f"_{outlier_tag}_order{order}"
        f"_monoz{feh_monotone_strength:.1e}"
        f"_feh[{plot_feh_min:+.2f},{plot_feh_max:+.2f}]"
        f"_fehmodel-{feh_model}_cross-{cross_mode}"
        f"_{feh_error_tag}"
        f"_skappa{outlier_kappa_scale:.2e}"
        + (
            f"_holdout{float(heldout_fraction):.3f}-seed{heldout_seed}"
            if heldout_data is not None
            else ""
        )
    )

    fitter.run_numpyro(
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        use_dense_mass=use_dense_mass,
        seed=seed,
        quad_mode=quad_mode,
        quad_mask=quad_mask,
        int_du=0.1,
        collect_diagnostics=save_full_mcmc,
    )

    print("5. Saving samples and plots...")
    np.savetxt(os.path.join(output_dir, f"mcmc_{suffix}.txt"), fitter.samples)
    if fitter.feh_population_samples is not None:
        np.savetxt(
            os.path.join(output_dir, f"feh_population_weights_{suffix}.txt"),
            fitter.feh_population_samples,
        )
    if save_full_mcmc:
        grouped_path = os.path.join(
            output_dir, f"mcmc_grouped_with_diagnostics_{suffix}.npz"
        )
        _save_grouped_mcmc_with_diagnostics(fitter, grouped_path)
        print(f"   Saved full chain-aware MCMC record: {grouped_path}")
    fitter.plot_results(output_dir=output_dir, output_suffix=suffix)

    feh_ref = float(np.nanmedian(np.array(data[feh_column], dtype=float)))
    absg_curve = np.linspace(plot_absg_min, plot_absg_max, 400)
    iso_mass_curve = iso_surface.mass_from_absg_mh(absg_curve, feh_ref)

    # fitter.plot_fitting_results(
    #     data=data,
    #     output_dir=output_dir,
    #     output_suffix=suffix,
    #     isochrone_curve_data_feh={
    #         "absg": absg_curve,
    #         "mass": iso_mass_curve,
    #     },
    #     iso_colname_dict={"absg": "absg", "mass": "mass", "feh": "MH"},
    # )

    feh_plot_values = (
        np.linspace(plot_feh_min, plot_feh_max, 5)
        if feh_plot_values is None
        else np.array(feh_plot_values, dtype=float)
    )
    feh_plot_values = feh_plot_values[np.isfinite(feh_plot_values)]
    if feh_plot_values.size == 0:
        raise ValueError("feh_plot_values must contain at least one finite value.")

    plot_mlr_multi_feh(
        fitter=fitter,
        iso_surface=iso_surface,
        feh_values=feh_plot_values,
        absg_min=plot_absg_min,
        absg_max=plot_absg_max,
        output_dir=output_dir,
        output_suffix=suffix,
        truths=param_truths,
        truth_model=truth_model_auto,
    )

    if heldout_data is not None:
        _save_heldout_predictive_result(
            fitter=fitter,
            heldout_data=heldout_data,
            heldout_indices=heldout_indices,
            output_dir=output_dir,
            suffix=suffix,
            model_label=model_label,
            feh_column=feh_column,
            feh_sigma_column=feh_sigma_column,
            use_feh_uncertainty=use_feh_uncertainty,
            outlier_kappa=outlier_kappa,
            outlier_kappa_scale=outlier_kappa_scale,
            max_posterior_samples=heldout_max_posterior_samples,
            seed=heldout_seed,
        )

    print("6. Done.")
    print(f"   Reference metallicity for isochrone curve: [Fe/H] approx {feh_ref:+.2f}")
    print(f"   Saved: {os.path.join(output_dir, f'mcmc_{suffix}.txt')}")
    return fitter


def plot_mlr_multi_feh(
    *,
    fitter,
    iso_surface,
    feh_values,
    absg_min,
    absg_max,
    output_dir,
    output_suffix,
    truths=None,
    truth_model=None,
):
    if fitter.samples is None or fitter.samples.size == 0:
        print("No samples available for multi-metallicity plot.")
        return

    if fitter.fit_outlier_params:
        coeff_samples = fitter.samples[:, :-3]
    else:
        coeff_samples = fitter.samples

    
    n_coeff = coeff_samples.shape[1]
    truth_coeffs = None
    truth_mass_fn = None

    # TRUTH HANDLING LOGIC:
    # - Same MLRs might correspond to different coefficient values, when order, absg range, feh range, and isochrone baseline differ. So we allow some flexibility in the truth input:
    #   - If the user provides a truth_model with mass_from_absg_feh(), we use that directly (this allows complete freedom in truth format, as long as it can be called with (absg, feh)).
    #   - Else, if the user provides truth coefficients that match the number of fitted coefficients, we assume they correspond directly to the fitted model's coefficients.
    if truths is not None:
        truth_arr = np.asarray(truths, dtype=float).reshape(-1)

        if fitter.fit_outlier_params and truth_arr.size == n_coeff + 3:
            truth_arr = truth_arr[:n_coeff]

        if truth_model is not None:
            if not hasattr(truth_model, "mass_from_absg_feh"):
                raise ValueError("truth_model must implement mass_from_absg_feh().")
            truth_coeffs = truth_arr
            truth_mass_fn = truth_model.mass_from_absg_feh
        elif truth_arr.size == n_coeff:
            truth_coeffs = truth_arr
            truth_mass_fn = fitter.mass_from_absg_feh
        elif truth_arr.size < n_coeff:
            truth_coeffs = np.pad(truth_arr, (0, n_coeff - truth_arr.size), constant_values=0.0)
            truth_mass_fn = fitter.mass_from_absg_feh
        elif fitter.poly_model.feh_model == "quadratic2d" and truth_arr.size in (5, 6):
            inferred_cross_mode = "free" if truth_arr.size == 6 else "product"
            truth_model_auto = DifferencePolyFehMassAbsgModel(
                order=2,
                absg_min=fitter.poly_model.absg_min,
                absg_max=fitter.poly_model.absg_max,
                mass_min=fitter.poly_model.mass_min,
                pivot=fitter.poly_model.pivot,
                deriv_penalty_strength=fitter.absg_monotone_strength,
                feh_monotone_strength=fitter.poly_model.feh_monotone_strength,
                isochrone_surface_model=fitter.iso_surface,
                feh_min=fitter.poly_model.feh_min,
                feh_max=fitter.poly_model.feh_max,
                feh_model="quadratic2d",
                cross_mode=inferred_cross_mode,
            )
            truth_coeffs = truth_arr
            truth_mass_fn = truth_model_auto.mass_from_absg_feh
        else:
            print(
                "Truth coefficients length does not match fitted model; skipping truth curves."
            )

    num_samples = min(2000, len(coeff_samples))
    if num_samples < len(coeff_samples):
        rng = np.random.default_rng(42)
        indices = rng.choice(len(coeff_samples), size=num_samples, replace=False)
        coeff_samples = coeff_samples[indices]

    absg_range = np.linspace(absg_min, absg_max, 600)
    feh_values = np.asarray(feh_values, dtype=float)
    feh_values = np.sort(feh_values[np.isfinite(feh_values)])
    if feh_values.size == 0:
        raise ValueError("feh_values must contain at least one finite value.")

    cmap = plt.cm.viridis
    norm = plt.Normalize(feh_values.min(), feh_values.max())

    fig = plt.figure(figsize=(10, 8))
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.07)
    ax_main = fig.add_subplot(gs[0])
    ax_resid = fig.add_subplot(gs[1], sharex=ax_main)
    ax_main.tick_params(axis="x", which="both", bottom=False, labelbottom=False)

    for feh in feh_values:
        color = cmap(norm(feh))
        masses = fitter.mass_from_absg_feh(absg_range, feh, coeff_samples)
        lower, median, upper = np.percentile(masses, [16, 50, 84], axis=0)

        ax_main.fill_between(absg_range, lower, upper, color=color, alpha=0.18)
        ax_main.plot(absg_range, median, color=color, linewidth=2)

        iso_mass = None
        if iso_surface is not None:
            iso_mass = iso_surface.mass_from_absg_mh(absg_range, feh)
            ax_main.plot(absg_range, iso_mass, color=color, linewidth=1.5, linestyle="--", alpha=1)

        if truth_coeffs is not None and truth_mass_fn is not None:
            true_masses = truth_mass_fn(absg_range, feh, truth_coeffs)
            ax_main.plot(absg_range, true_masses, color=color, linewidth=3, linestyle=":", alpha=0.8)

        if iso_mass is not None:
            safe_iso_mass = np.maximum(iso_mass, fitter.mass_min)
            residual = 100.0 * (median - iso_mass) / safe_iso_mass
            resid_lower = 100.0 * (lower - iso_mass) / safe_iso_mass
            resid_upper = 100.0 * (upper - iso_mass) / safe_iso_mass
            ax_resid.fill_between(absg_range, resid_lower, resid_upper, color=color, alpha=0.15)
            ax_resid.plot(absg_range, residual, color=color, linewidth=1.5)
        
        if feh==0:
            # Sun's position
            MG_sun = 4.67
            ax_main.scatter(MG_sun, 1.0,
                        marker='*',
                        s=180,
                        facecolor=color,
                        edgecolor='black',
                        linewidth=1.2,
                        zorder=5)
            
            ax_main.scatter(MG_sun, 1.0,
                    marker='o',
                    s=260,
                    facecolor='none',
                    edgecolor='black',
                    linewidth=1.2,
                    zorder=4)

            ax_main.annotate('Sun',
                    (MG_sun, 1.0),
                    xytext=(5,5),
                    textcoords='offset points',
                    fontsize=14)


    legend_lines = [
        Line2D([0], [0], color="black", linewidth=2, linestyle="-"),
        Line2D([0], [0], color="black", linewidth=1.5, linestyle="--"),
    ]
    legend_labels = [r"Fit median (band = 1$\sigma$)", "Isochrone"]
    if truth_coeffs is not None and truth_mass_fn is not None:
        legend_lines.append(Line2D([0], [0], color="black", linewidth=3, linestyle=":"))
        legend_labels.append("Truth")
    ax_main.legend(legend_lines, legend_labels, fontsize=10, loc="best")

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=[ax_main, ax_resid], pad=0.02, location="right")
    cbar.set_label("[Fe/H]", rotation=270, labelpad=12)

    ax_main.set_ylabel("Mass [$M_{\\odot}$]", fontsize=12)
    ax_main.set_xlim(absg_min, absg_max)
    ax_main.set_ylim(fitter.mass_min * 0.8, fitter.mass_max * 1.2)
    ax_main.set_yscale("log")
    ax_main.invert_xaxis()
    ax_resid.set_xlim(ax_main.get_xlim())
    fig.align_xlabels([ax_main, ax_resid])

    ax_resid.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax_resid.set_ylabel(
        r"$(M_{\rm fit}-M_{\rm iso})/M_{\rm iso}$ [\%]", fontsize=11
    )
    ax_resid.set_xlim(absg_min, absg_max)
    ax_resid.set_xlabel("$M_{\\mathrm{G}}$ [mag]", fontsize=12)

    if truth_coeffs is None and iso_surface is None:
        ax_resid.text(
            0.5,
            0.5,
            "Truth coefficients not provided",
            transform=ax_resid.transAxes,
            ha="center",
            va="center",
            color="gray",
            fontsize=10,
        )

    fig.tight_layout()

    os.makedirs(output_dir, exist_ok=True) if output_dir else None
    filename = f"fit_multifeh_{output_suffix}.png"
    fig.savefig(os.path.join(output_dir, filename), dpi=300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Fit and compare nested FeH-global DifferencePoly models."
    )
    parser.add_argument(
        "--stage",
        choices=("core", "quadratic", "cross"),
        default="core",
        help="core: a0/a1/b1; quadratic: add a2/b2; cross: add c_xy.",
    )
    parser.add_argument(
        "--compare-only",
        action="store_true",
        help="Compare already-saved held-out scores without running MCMC.",
    )
    parser.add_argument("--heldout-fraction", type=float, default=0.2)
    parser.add_argument("--heldout-seed", type=int, default=20260812)
    parser.add_argument("--heldout-posterior-samples", type=int, default=256)
    parser.add_argument(
        "--save-full-mcmc",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save grouped chains plus divergence/energy/step diagnostics.",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("FeH-Global DifferencePoly MLR on Real Data")
    print("=" * 70)

    output_dir = os.path.join(
        REPO_ROOT, "results", "data_diffpoly2d_anchor_cute_tfeherr_good"
    )
    if args.compare_only:
        compare_heldout_predictive_results(output_dir)
        return

    print("1. Importing data with metallicity...")
    data_path = os.path.join(
        REPO_ROOT,
        "data",
        "jd_msms_single_bic_1kpc_filtered_cmdcut_cutb_jcaps_err_mhcal_good.fits",
    )
    data = Table.read(data_path)

    # This input is already quality-selected; this assertion-like cut is
    # intentionally redundant and should not remove any rows.
    data = data[data["a_g_edhf_1"] <= 0.1]

    # random sub-sample for testing
    # rng = np.random.default_rng(27)
    # if len(data) > 5000:
    #     indices = rng.choice(len(data), size=5000, replace=False)
    #     data = data[indices]

    print("2. Computing u and u_sigma...")
    data["v"] = (
        4.74
        * np.sqrt((data["pmra2"] - data["pmra1"]) ** 2 + (data["pmdec2"] - data["pmdec1"]) ** 2)
        / data["parallax1"]
    )
    data["u"] = data["v"] * np.sqrt(data["sep_AU"])
    data["u_sigma"] = data["u"] / data["dpm_over_error"]

    print(f"   Using {len(data)} systems")

    # Use the uncorrected primary metallicity measurement and its uncorrected
    # error.  Corrected columns use the same binary-equality information as
    # this model and would duplicate that likelihood.
    feh_column = "feh_jcaps_1"
    feh_sigma_column = "jc_sigma_m_h_1"
    stage_settings = quadratic_stage_settings(args.stage)

    test_differencepoly_feh_model(
        data,
        output_dir=output_dir,
        uncertainty_model="rice",
        feh_column=feh_column,
        # To enable the FeH-error likelihood, set this to True and set
        # feh_sigma_column to the table column containing 1-sigma errors (dex).
        use_feh_uncertainty=True,
        feh_sigma_column=feh_sigma_column,
        feh_quadrature_nodes=11,
        feh_population_components=10,
        feh_population_concentration=1.0,
        feh_population_kernel_scale=1.0,
        feh_min=-1,
        feh_max=0.6,
        # order=1,
        absg_min=3.5,
        absg_max=13.5,
        mass_min=0.05,
        mass_max=2.0,
        absg_monotone_strength=0,
        feh_monotone_strength=0,
        monotone_n_feh=5,
        coeff_prior_scale=3.0,
        feh_coeff_prior_scale=3.0,
        f_outlier_init=0.2,
        outlier_u0_init=40,
        outlier_sigma_init=13,
        outlier_kappa=None,
        outlier_kappa_scale=1,
        fit_outlier_params=True,
        param_truths=None,
        num_warmup=1200,
        num_samples=2000,
        num_chains=4,
        use_dense_mass=True,
        seed=2,
        iso_data_dir=os.path.join(REPO_ROOT, "data", "interpolated_mass_data"),
        feh_plot_values=[-1.0, -0.5, 0.0, 0.3, 0.6],
        feh_model="quadratic2d",
        anchor_enabled=True,
        anchor_absg=4.67,
        anchor_feh=0.0,
        anchor_mass=1.0,
        anchor_sigma=0.01,
        cross_mode=stage_settings["cross_mode"],
        quad_mode=stage_settings["quad_mode"],
        quad_mask=stage_settings["quad_mask"],
        model_label=args.stage,
        save_full_mcmc=args.save_full_mcmc,
        heldout_fraction=args.heldout_fraction,
        heldout_seed=args.heldout_seed,
        heldout_max_posterior_samples=args.heldout_posterior_samples,
    )

    print("\nGenerated files (prefix):")
    print(f"  - {output_dir}/mcmc_model-differencepolyfeh_*.txt")
    print(f"  - {output_dir}/corner_model-differencepolyfeh_*.png")
    print(f"  - {output_dir}/fit_model-differencepolyfeh_*.png")
    print(f"  - {output_dir}/heldout_predictive_*.json/.npz")
    print("After all three stages, run this script with --compare-only.")
    print("\nFeH-global example completed successfully!")


if __name__ == "__main__":
    main()
