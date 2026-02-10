#!/usr/bin/env python3
"""
Continuous-Metallicity Difference-Polynomial Fitting Example

This example demonstrates how to fit a mass-luminosity relation (MLR) while
treating metallicity as a continuous covariate (one metallicity per system),
instead of binning by [Fe/H].

Model:
    log10 m_dyn(M_G, MH) = log10 m_iso(M_G, MH) + Σ_i (a_i + b_i * z(MH)) x(M_G)^i

where the baseline m_iso is taken from the interpolated isochrone grid.

Author: Yutong Wang
"""

import os
import sys
import jax

import numpy as np
from astropy.table import Table
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Add the package to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

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
    order=3,
    absg_min=3.5,
    absg_max=14.0,
    mass_min=0.05,
    mass_max=2.0,
    deriv_penalty_strength=10.0,
    feh_monotone_strength=0.0,
    monotone_n_feh=5,
    coeff_prior_scale=5.0,
    feh_coeff_prior_scale=1.0,
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
    feh_b0_only=False, # feh_b0_only=True (forces b_{i>0}=0), and automatically feh_b0_positive=True
    feh_b0_positive=False,
    feh_b_mask=None, # default: None -> all False
):
    """
    Fit the DifferencePolyFehMLR model on all data (no metallicity binning).

    Notes
    -----
    - The isochrone baseline uses MH from the interpolated grid.
      If your system metallicity is [Fe/H], you are implicitly assuming MH≈[Fe/H].
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
        f_outlier=0.1,
        outlier_u0=30,
        outlier_sigma=15,
        outlier_kappa=outlier_kappa,
        outlier_kappa_scale=outlier_kappa_scale,
        fit_outlier_params=fit_outlier_params,
        deriv_penalty_strength=deriv_penalty_strength,
        feh_monotone_strength=feh_monotone_strength,
        coeff_prior_scale=coeff_prior_scale,
        feh_coeff_prior_scale=feh_coeff_prior_scale,
        monotone_n_feh=monotone_n_feh,
        param_truths=param_truths,
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
    feh_b_mask_run = feh_b_mask
    feh_b0_positive_run = feh_b0_positive
    if feh_b0_only:
        feh_b_mask_run = np.zeros(order + 1, dtype=bool)
        feh_b_mask_run[0] = True
        feh_b0_positive_run = True

    outlier_tag = "outlierfit" if fit_outlier_params else "outlierfixed"
    suffix = (
        f"_model-differencepolyfeh_unc-{uncertainty_model}"
        f"_fehcont_{outlier_tag}_order{order}"
        f"_feh[{feh_min:+.2f},{feh_max:+.2f}]"
    )

    fitter.run_numpyro(
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        seed=seed,
        feh_b_mask=feh_b_mask_run,
        feh_b0_positive=feh_b0_positive_run,
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
    )

    print("6. Done.")
    print(f"   Reference metallicity for isochrone curve: [Fe/H]≈{feh_ref:+.2f}")
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
):
    if fitter.samples is None or fitter.samples.size == 0:
        print("No samples available for multi-metallicity plot.")
        return

    if fitter.fit_outlier_params:
        coeff_samples = fitter.samples[:, :-3]
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

    fig, ax = plt.subplots(figsize=(10, 6))

    for feh in feh_values:
        color = cmap(norm(feh))
        masses = fitter.mass_from_absg_feh(absg_range, feh, coeff_samples)
        lower, median, upper = np.percentile(masses, [16, 50, 84], axis=0)

        ax.fill_between(absg_range, lower, upper, color=color, alpha=0.18)
        ax.plot(absg_range, median, color=color, linewidth=2)

        if iso_surface is not None:
            iso_mass = iso_surface.mass_from_absg_mh(absg_range, feh)
            ax.plot(absg_range, iso_mass, color=color, linewidth=1.5, linestyle="--", alpha=0.8)

    legend_lines = [
        Line2D([0], [0], color="black", linewidth=2, linestyle="-"),
        Line2D([0], [0], color="black", linewidth=1.5, linestyle="--"),
    ]
    ax.legend(legend_lines, ["Fit median (band = 1σ)", "Isochrone"], fontsize=10, loc="best")

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label("[Fe/H]", rotation=270, labelpad=12)

    ax.set_xlabel("$M_{\\mathrm{G}}$ [mag]", fontsize=12)
    ax.set_ylabel("Mass [$M_{\\odot}$]", fontsize=12)
    ax.set_xlim(absg_min, absg_max)
    ax.set_ylim(fitter.mass_min * 0.8, fitter.mass_max * 1.2)
    ax.set_yscale("log")
    ax.invert_xaxis()
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True) if output_dir else None
    filename = f"fit_multifeh{output_suffix}.png"
    plt.savefig(os.path.join(output_dir, filename), dpi=300)
    plt.close(fig)


def main():
    print("=" * 70)
    print("Continuous-Metallicity DifferencePoly MLR Example")
    print("=" * 70)

    print("\n1. Importing data with metallicity...")
    data_path = "data/jd_single_1kpc_filtered.fits"
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

    output_dir = "results/data_diffpoly_feh_pred_kappa0p5"

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
        feh_min=-1.,
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
        outlier_kappa=None,
        outlier_kappa_scale=1,
        fit_outlier_params=True,
        param_truths=None,
        num_warmup=500,
        num_samples=2500,
        num_chains=1,
        seed=11,
        iso_data_dir="data/interpolated_mass_data",
        feh_plot_values=[-1.0, -0.5, 0.0, 0.3, 0.6],
        feh_b0_only=True,
    )

    print("\nGenerated files (prefix):")
    print(f"  - {output_dir}/mcmc_model-differencepolyfeh_*.txt")
    print(f"  - {output_dir}/corner_model-differencepolyfeh_*.png")
    print(f"  - {output_dir}/fit_model-differencepolyfeh_*.png")
    print("\nContinuous-metallicity example completed successfully!")


if __name__ == "__main__":
    main()
