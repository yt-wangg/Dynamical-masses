#!/usr/bin/env python3
"""
FeH-Global Difference-Polynomial Fitting on Real Data

This script mirrors the mock-data test in:
    Validation/V16. FeH-global model test.ipynb
but runs on real Gaia binary data. It uses the updated
DifferencePolyFehMLR interface (feh_model, cross_mode, quad_mask).

Author: Yutong Wang
"""

import os
import sys

import numpy as np
from astropy.table import Table
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Add the package to Python path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO_ROOT)

from binary_masses.differencepoly_feh import (  # noqa: E402
    DifferencePolyFehMLR,
    IsochroneMassSurfaceModel,
)


def test_differencepoly_feh_model(
    data,
    *,
    output_dir,
    uncertainty_model="rice",
    feh_column="feh",
    feh_min=-1.5,
    feh_max=0.6,
    order=2,
    absg_min=3.5,
    absg_max=14.0,
    mass_min=0.05,
    mass_max=2.0,
    deriv_penalty_strength=10.0,
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
    num_warmup=800,
    num_samples=2000,
    num_chains=1,
    seed=42,
    iso_data_dir="data/interpolated_mass_data",
    feh_plot_values=None,
    feh_model="quadratic2d",
    cross_mode="free",
    quad_mask=None,
    feh_b_mask=None,
    feh_b0_positive=False,
    feh_b0_only=False,
    save_predictive_metrics=False,
    save_predictive_metrics_kwargs=None,
):
    """
    Fit the DifferencePolyFehMLR model on all data (no metallicity binning).

    Notes
    -----
    - The isochrone baseline uses MH from the interpolated grid.
      If your system metallicity is [Fe/H], you are implicitly assuming MH approx [Fe/H].
    """
    print("\n" + "=" * 60)
    print("TESTING DIFFERENCE-POLY WITH CONTINUOUS METALLICITY")
    print("=" * 60)

    if feh_column not in data.colnames:
        raise ValueError(f"Metallicity column {feh_column!r} not found in data.")

    os.makedirs(output_dir, exist_ok=True)

    print("1. Loading isochrone mass surface...")
    iso_surface = IsochroneMassSurfaceModel.from_interpolated_mass_data(
        iso_data_dir,
        mass_min=mass_min,
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
        deriv_penalty_strength=deriv_penalty_strength,
        feh_monotone_strength=feh_monotone_strength,
        coeff_prior_scale=coeff_prior_scale,
        feh_coeff_prior_scale=feh_coeff_prior_scale,
        monotone_n_feh=monotone_n_feh,
        feh_model=feh_model,
        cross_mode=cross_mode,
    )

    print("3. Setting data...")
    fitter.set_data(
        u_values=data["u"],
        u_sigma_values=data["u_sigma"] if "u_sigma" in data.colnames else None,
        absg1_values=data["absg1"],
        absg2_values=data["absg2"],
        feh_values=data[feh_column],
        outlier_kappa=outlier_kappa,
        outlier_kappa_scale=outlier_kappa_scale,
    )

    print("4. Running MCMC...")
    outlier_tag = "outlierfit" if fit_outlier_params else "outlierfixed"
    suffix = (
        f"_model-differencepolyfeh_unc-{uncertainty_model}"
        f"_fehcont_{outlier_tag}_order{order}"
        f"_feh[{feh_min:+.2f},{feh_max:+.2f}]"
        f"_fehmodel-{feh_model}_cross-{cross_mode}"
    )

    feh_b_mask_run = feh_b_mask
    feh_b0_positive_run = feh_b0_positive
    if feh_model == "linear" and feh_b0_only:
        feh_b_mask_run = np.zeros(order + 1, dtype=bool)
        feh_b_mask_run[0] = True
        feh_b0_positive_run = True

    predictive_path = None
    predictive_kwargs = save_predictive_metrics_kwargs
    if save_predictive_metrics:
        predictive_path = os.path.join(output_dir, f"predictive_metrics{suffix}.npz")
        if predictive_kwargs is None:
            predictive_kwargs = {
                "sample_limit": 500,
                "data_chunk": 2048,
                "int_du": 0.02,
                "num_workers": 2,
                "sample_chunk": 32,
                "show_progress": True,
            }

    fitter.run_numpyro(
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        seed=seed,
        feh_b_mask=feh_b_mask_run,
        feh_b0_positive=feh_b0_positive_run,
        quad_mask=quad_mask,
        save_predictive_metrics_path=predictive_path,
        save_predictive_metrics_kwargs=predictive_kwargs,
    )

    print("5. Saving samples and plots...")
    np.savetxt(os.path.join(output_dir, f"mcmc{suffix}.txt"), fitter.samples)
    fitter.plot_results(output_dir=output_dir, output_suffix=suffix)

    feh_ref = float(np.nanmedian(np.array(data[feh_column], dtype=float)))
    absg_curve = np.linspace(absg_min, absg_max, 400)
    iso_mass_curve = iso_surface.mass_from_absg_mh(absg_curve, feh_ref)

    fitter.plot_fitting_results(
        data=data,
        output_dir=output_dir,
        output_suffix=suffix,
        isochrone_curve_data_feh={
            "absg": absg_curve,
            "mass": iso_mass_curve,
        },
        iso_colname_dict={"absg": "absg", "mass": "mass", "feh": "MH"},
    )

    feh_plot_values = (
        np.linspace(feh_min, feh_max, 5) if feh_plot_values is None else np.array(feh_plot_values, dtype=float)
    )
    feh_plot_values = feh_plot_values[np.isfinite(feh_plot_values)]
    if feh_plot_values.size == 0:
        raise ValueError("feh_plot_values must contain at least one finite value.")

    plot_mlr_multi_feh(
        fitter=fitter,
        iso_surface=iso_surface,
        feh_values=feh_plot_values,
        absg_min=absg_min,
        absg_max=absg_max,
        output_dir=output_dir,
        output_suffix=suffix,
        truths=param_truths,
    )

    print("6. Done.")
    print(f"   Reference metallicity for isochrone curve: [Fe/H] approx {feh_ref:+.2f}")
    print(f"   Saved: {os.path.join(output_dir, f'mcmc{suffix}.txt')}")
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
):
    if fitter.samples is None or fitter.samples.size == 0:
        print("No samples available for multi-metallicity plot.")
        return

    if fitter.fit_outlier_params:
        coeff_samples = fitter.samples[:, :-3]
        truths = truths[:-3] if truths is not None else None
    else:
        coeff_samples = fitter.samples

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

    for feh in feh_values:
        color = cmap(norm(feh))
        masses = fitter.mass_from_absg_feh(absg_range, feh, coeff_samples)
        lower, median, upper = np.percentile(masses, [16, 50, 84], axis=0)

        ax_main.fill_between(absg_range, lower, upper, color=color, alpha=0.18)
        ax_main.plot(absg_range, median, color=color, linewidth=2)

        if iso_surface is not None:
            iso_mass = iso_surface.mass_from_absg_mh(absg_range, feh)
            ax_main.plot(absg_range, iso_mass, color=color, linewidth=1, linestyle="--", alpha=0.8)

        if truths is not None:
            true_masses = fitter.mass_from_absg_feh(absg_range, feh, truths)
            ax_main.plot(absg_range, true_masses, color=color, linewidth=3, linestyle=":", alpha=0.8)
            residual = median - true_masses
            resid_lower = lower - true_masses
            resid_upper = upper - true_masses
            ax_resid.fill_between(absg_range, resid_lower, resid_upper, color=color, alpha=0.15)
            ax_resid.plot(absg_range, residual, color=color, linewidth=1.5)

    legend_lines = [
        Line2D([0], [0], color="black", linewidth=2, linestyle="-"),
        Line2D([0], [0], color="black", linewidth=1.5, linestyle="--"),
        Line2D([0], [0], color="black", linewidth=3, linestyle=":"),
    ]
    ax_main.legend(legend_lines, [r"Fit median (band = 1$\sigma$)", "Isochrone", "Truth"], fontsize=10, loc="best")

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
    ax_resid.set_ylabel("Derived - Truth [$M_{\\odot}$]", fontsize=11)
    ax_resid.set_xlim(absg_min, absg_max)
    ax_resid.set_xlabel("$M_{\\mathrm{G}}$ [mag]", fontsize=12)

    if truths is None:
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
    filename = f"fit_multifeh{output_suffix}.png"
    fig.savefig(os.path.join(output_dir, filename), dpi=300)
    plt.close(fig)


def main():
    print("=" * 70)
    print("FeH-Global DifferencePoly MLR on Real Data")
    print("=" * 70)

    print("1. Importing data with metallicity...")
    data_path = os.path.join(REPO_ROOT, "data", "jd_single_1kpc_filtered.fits")
    data = Table.read(data_path)

    print("2. Computing u and u_sigma...")
    data["v"] = (
        4.74
        * np.sqrt((data["pmra2"] - data["pmra1"]) ** 2 + (data["pmdec2"] - data["pmdec1"]) ** 2)
        / data["parallax1"]
    )
    data["u"] = data["v"] * np.sqrt(data["sep_AU"])
    data["u_sigma"] = data["u"] / data["dpm_over_error"]

    print(f"   Using {len(data)} systems")

    output_dir = os.path.join(REPO_ROOT, "results", "data_diffpoly_feh_global")

    # Pick the metallicity column that exists in your table.
    # Common choices in this repo: "feh" or "feh_jcaps_1".
    feh_column = "feh"
    if feh_column not in data.colnames and "feh_jcaps_1" in data.colnames:
        feh_column = "feh_jcaps_1"

    test_differencepoly_feh_model(
        data,
        output_dir=output_dir,
        uncertainty_model="rice",
        feh_column=feh_column,
        feh_min=-1.0,
        feh_max=0.6,
        order=2,
        absg_min=3.5,
        absg_max=14.0,
        mass_min=0.05,
        mass_max=2.0,
        deriv_penalty_strength=10.0,
        feh_monotone_strength=0,
        monotone_n_feh=5,
        coeff_prior_scale=5.0,
        feh_coeff_prior_scale=1.0,
        f_outlier_init=0.1,
        outlier_u0_init=35,
        outlier_sigma_init=15,
        outlier_kappa=None,
        outlier_kappa_scale=1,
        fit_outlier_params=True,
        param_truths=None,
        num_warmup=500,
        num_samples=2500,
        num_chains=1,
        seed=11,
        iso_data_dir=os.path.join(REPO_ROOT, "data", "interpolated_mass_data"),
        feh_plot_values=[-1.0, -0.5, 0.0, 0.3, 0.6],
        feh_model="quadratic2d",
        cross_mode="free",
        quad_mask=None,
        feh_b0_only=False,
        save_predictive_metrics=False,
    )

    print("\nGenerated files (prefix):")
    print(f"  - {output_dir}/mcmc_model-differencepolyfeh_*.txt")
    print(f"  - {output_dir}/corner_model-differencepolyfeh_*.png")
    print(f"  - {output_dir}/fit_model-differencepolyfeh_*.png")
    print("\nFeH-global example completed successfully!")


if __name__ == "__main__":
    main()
