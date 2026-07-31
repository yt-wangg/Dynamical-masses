#!/usr/bin/env python3
"""
FeH-Global Difference-Polynomial Fitting on Real Data

This script mirrors the mock-data test in:
    Validation/V16. FeH-global model test.ipynb
but runs on real Gaia binary data. It uses the updated
DifferencePolyFehMLR interface (feh_model, cross_mode, quad_mode/quad_mask).
"""

import os
import sys

import numpy as np
from astropy.table import Table
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Add the src-layout package to Python path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from binary_masses.differencepoly_feh import (  # noqa: E402
    DifferencePolyFehMLR,
    DifferencePolyFehMassAbsgModel,
    IsochroneMassSurfaceModel,
)


def test_differencepoly_feh_model(
    data,
    *,
    output_dir,
    uncertainty_model="rice",
    feh_column="feh",
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

    def _robust_range(values, *, lo=1.0, hi=99.0):
        values = np.asarray(values, dtype=float)
        values = values[np.isfinite(values)]
        if values.size == 0:
            raise ValueError("No finite values available to infer plotting range.")
        return np.nanpercentile(values, [lo, hi])

    absg_all = np.concatenate([np.asarray(data["absg1"]), np.asarray(data["absg2"])])
    absg_lo, absg_hi = _robust_range(absg_all)
    plot_absg_min = absg_lo if absg_min is None else absg_min
    plot_absg_max = absg_hi if absg_max is None else absg_max
    if plot_absg_max <= plot_absg_min:
        raise ValueError(f"Invalid plotting absg range: [{plot_absg_min}, {plot_absg_max}].")

    feh_lo, feh_hi = _robust_range(data[feh_column])
    plot_feh_min = feh_lo if feh_min is None else feh_min
    plot_feh_max = feh_hi if feh_max is None else feh_max
    if plot_feh_max <= plot_feh_min:
        raise ValueError(f"Invalid plotting feh range: [{plot_feh_min}, {plot_feh_max}].")

    os.makedirs(output_dir, exist_ok=True)

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
        outlier_kappa=outlier_kappa,
        outlier_kappa_scale=outlier_kappa_scale,
    )

    print("4. Running MCMC...")
    outlier_tag = "outlierfit" if fit_outlier_params else "outlierfixed"
    suffix = (
        f"_model-diffpoly2dfeh_unc-{uncertainty_model}"
        f"_{outlier_tag}_order{order}"
        f"_monoz{feh_monotone_strength:.1e}"
        f"_feh[{plot_feh_min:+.2f},{plot_feh_max:+.2f}]"
        f"_fehmodel-{feh_model}_cross-{cross_mode}"
        f"_skappa{outlier_kappa_scale:.2e}"
    )

    fitter.run_numpyro(
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        use_dense_mass=use_dense_mass,
        seed=seed,
        quad_mode=quad_mode,
        quad_mask=quad_mask,
    )

    print("5. Saving samples and plots...")
    np.savetxt(os.path.join(output_dir, f"mcmc_{suffix}.txt"), fitter.samples)
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

        if iso_surface is not None:
            iso_mass = iso_surface.mass_from_absg_mh(absg_range, feh)
            ax_main.plot(absg_range, iso_mass, color=color, linewidth=1.5, linestyle="--", alpha=1)

        if truth_coeffs is not None and truth_mass_fn is not None:
            true_masses = truth_mass_fn(absg_range, feh, truth_coeffs)
            ax_main.plot(absg_range, true_masses, color=color, linewidth=3, linestyle=":", alpha=0.8)
            residual = median - true_masses
            resid_lower = lower - true_masses
            resid_upper = upper - true_masses
            ax_resid.fill_between(absg_range, resid_lower, resid_upper, color=color, alpha=0.15)
            ax_resid.plot(absg_range, residual, color=color, linewidth=1.5)
        elif iso_mass is not None:
            residual = median - iso_mass
            resid_lower = lower - iso_mass
            resid_upper = upper - iso_mass
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
    ax_resid.set_yscale('log')
    resid_label = (
        r"Derived $-$ Truth [$M_{\odot}$]"
        if truth_coeffs is not None
        else r"Derived $-$ Isochrone [$M_{\odot}$]"
    )
    ax_resid.set_ylabel(resid_label, fontsize=11)
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
    print("=" * 70)
    print("FeH-Global DifferencePoly MLR on Real Data")
    print("=" * 70)

    print("1. Importing data with metallicity...")
    data_path = os.path.join(REPO_ROOT, "data", "jd_msms_single_bic_1kpc_filtered_cutb_fehloss.fits")
    data = Table.read(data_path)

    # cut on a_g_edhf_1
    data = data[data["a_g_edhf_1"] <= 0.1]

    # random sub-sample for testing
    rng = np.random.default_rng(27)
    if len(data) > 5000:
        indices = rng.choice(len(data), size=5000, replace=False)
        data = data[indices]

    print("2. Computing u and u_sigma...")
    data["v"] = (
        4.74
        * np.sqrt((data["pmra2"] - data["pmra1"]) ** 2 + (data["pmdec2"] - data["pmdec1"]) ** 2)
        / data["parallax1"]
    )
    data["u"] = data["v"] * np.sqrt(data["sep_AU"])
    data["u_sigma"] = data["u"] / data["dpm_over_error"]

    print(f"   Using {len(data)} systems")

    output_dir = os.path.join(REPO_ROOT, "results", "data_diffpoly2d_anchor_cute")

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
        feh_min=-1,
        feh_max=0.6,
        order=1,
        absg_min=3.5,
        absg_max=13.5,
        mass_min=0.05,
        mass_max=2.0,
        absg_monotone_strength=10.0,
        feh_monotone_strength=20000,
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
        num_warmup=800,
        num_samples=6000,
        num_chains=1,
        use_dense_mass=True,
        seed=33,
        iso_data_dir=os.path.join(REPO_ROOT, "data", "interpolated_mass_data"),
        feh_plot_values=[-1.0, -0.5, 0.0, 0.3, 0.6],
        feh_model="quadratic2d",
        anchor_enabled=True,
        anchor_absg=4.67,
        anchor_feh=0.0,
        anchor_mass=1.0,
        anchor_sigma=0.01,
        # cross_mode="free",
        # quad_mode=None,
        # quad_mask=None,
    )

    print("\nGenerated files (prefix):")
    print(f"  - {output_dir}/mcmc_model-differencepolyfeh_*.txt")
    print(f"  - {output_dir}/corner_model-differencepolyfeh_*.png")
    print(f"  - {output_dir}/fit_model-differencepolyfeh_*.png")
    print("\nFeH-global example completed successfully!")


if __name__ == "__main__":
    main()
