#!/usr/bin/env python3
"""Run the independent minimal hierarchical metallicity/MLR experiment."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys
from typing import Dict, Mapping, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from astropy.table import Table


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from binary_masses import hierarchical_metallicity as _hm  # noqa: E402
from binary_masses.hierarchical_metallicity import (  # noqa: E402
    DynamicsLikelihoodLookup,
    HierarchicalMetallicityCalibrator,
    IsochroneColorSurfaceModel,
    IsochroneMassSurfaceModel,
    MetallicityPosteriorGrid,
    MonotoneTensorSplineMLR,
    assess_rice_lookup_convergence,
    array_digest,
    color_uncertainty_from_flux_snr,
    grouped_mcmc_payload,
    simulate_mock_dataset,
)


INPUT_COLUMNS = {
    # Use the uncorrected XP measurements and their uncorrected errors only.
    # The *_cal columns are produced using the same binary-equality information
    # used by this hierarchical model and must not be fed back here.
    "feh_observed": ["feh_jcaps_1", "feh_jcaps_2"],
    "feh_sigma": ["jc_sigma_m_h_1", "jc_sigma_m_h_2"],
    "color_observed": ["bp_rp0_1", "bp_rp0_2"],
    "absg": ["absg1", "absg2"],
    "bp_snr": ["phot_bp_mean_flux_over_error1", "phot_bp_mean_flux_over_error2"],
    "rp_snr": ["phot_rp_mean_flux_over_error1", "phot_rp_mean_flux_over_error2"],
}

CMD_MIN_PARSEC_TEFF_K = 4000.0
T8_MODEL_ID = "t8_1_jcaps_student_t_independent_members"
T8_WORKFLOW_ID = "hierarchical_metallicity_t8_1_jcaps_student_t_teff4000_v1"
CALIBRATION_MODEL_ID = T8_MODEL_ID
T8_POSTERIOR_NAME = "latent_metallicity_weights_t8.npz"
T8_LOOKUP_NAME = "dynamics_likelihood_lookup_t8.npz"
T8_SHAPESTACK_NAME = "dynamics_likelihood_shapestack_t8.npz"
# T8.2 shape prior box: uniform over (log B, log uc, log C), covering the T8
# frozen constants, the published Hwang et al. (2019) constants, and the
# sample-calibrated fit, each with margin.
T82_SHAPE_AXES = {
    "log_b": (np.log(0.7e-3), np.log(1.52e-3), np.log(3.3e-3)),
    "log_uc": (np.log(33.0), np.log(38.1), np.log(44.0)),
    "log_c": (np.log(2.5), np.log(5.81), np.log(13.5)),
}
T8_MLR_NAME = "mlr_mcmc_t8.npz"
T8_MLR_DEFAULT_KNOT_X = np.array(
    [3.5, 3.5, 3.5, 3.5, 5.5, 7.5, 9.5, 11.5, 13.5, 13.5, 13.5, 13.5],
    dtype=float,
)
T8_MLR_DEFAULT_KNOT_Z = np.array(
    [-1.0, -1.0, -1.0, -1.0, 0.6, 0.6, 0.6, 0.6], dtype=float
)


def _stack_columns(table: Table, names) -> np.ndarray:
    missing = [name for name in names if name not in table.colnames]
    if missing:
        raise ValueError(f"Input table is missing columns: {missing}.")
    return np.column_stack([np.asarray(table[name], dtype=np.float64) for name in names])


def _selected_draw_indices(sample_count: int, max_draws: int, seed: int) -> np.ndarray:
    count = min(int(max_draws), int(sample_count))
    if count < 1:
        raise ValueError("The posterior contains no draws.")
    if count == sample_count:
        return np.arange(sample_count, dtype=np.int64)
    return np.sort(np.random.default_rng(int(seed)).choice(sample_count, size=count, replace=False)).astype(np.int64)


def load_real_data(data_path: Path) -> Dict[str, np.ndarray]:
    table = Table.read(data_path)
    arrays = {key: _stack_columns(table, names) for key, names in INPUT_COLUMNS.items()}
    arrays["color_sigma"] = color_uncertainty_from_flux_snr(
        arrays.pop("bp_snr"), arrays.pop("rp_snr")
    )
    if "u" in table.colnames and "u_sigma" in table.colnames:
        arrays["u"] = np.asarray(table["u"], dtype=np.float64)
        arrays["u_sigma"] = np.asarray(table["u_sigma"], dtype=np.float64)
    else:
        required = ["pmra1", "pmra2", "pmdec1", "pmdec2", "parallax1", "sep_AU", "dpm_over_error"]
        missing = [name for name in required if name not in table.colnames]
        if missing:
            raise ValueError(f"Cannot construct u; missing columns: {missing}.")
        velocity = 4.74 * np.sqrt(
            (np.asarray(table["pmra2"]) - np.asarray(table["pmra1"])) ** 2
            + (np.asarray(table["pmdec2"]) - np.asarray(table["pmdec1"])) ** 2
        ) / np.asarray(table["parallax1"])
        arrays["u"] = velocity * np.sqrt(np.asarray(table["sep_AU"]))
        arrays["u_sigma"] = arrays["u"] / np.asarray(table["dpm_over_error"])
    arrays["row_indices"] = np.arange(len(table), dtype=np.int64)
    return arrays


def _require_t8_posterior_metadata(
    posterior: MetallicityPosteriorGrid,
    *,
    mock: bool,
    arrays: Mapping[str, np.ndarray],
) -> None:
    """Reject legacy/posterior files whose assumptions are not T8.

    A lookup or MLR run must never silently reuse a T7 posterior: the CMD mask,
    raw JCAPS columns, and averaging rule are part of the statistical model.
    """
    metadata = dict(posterior.metadata)
    if metadata.get("schema") != "t8a-metallicity-posterior-v1":
        raise ValueError("The metallicity posterior schema is not T8a v1; rebuild calibration.")
    if metadata.get("workflow") != T8_WORKFLOW_ID:
        raise ValueError("The metallicity posterior workflow identifier is not current T8.")
    if metadata.get("model") != T8_MODEL_ID:
        raise ValueError(
            "The metallicity posterior is not a T8 posterior (expected model "
            f"{T8_MODEL_ID!r}); rebuild stage calibration."
        )
    if bool(metadata.get("mock", False)) != bool(mock):
        raise ValueError("The saved T8 metallicity posterior mock/real-data mode does not match this run.")
    if metadata.get("feh_columns") != INPUT_COLUMNS["feh_observed"]:
        raise ValueError("T8 posterior does not use the required raw feh_jcaps_1/2 columns.")
    if metadata.get("feh_sigma_columns") != INPUT_COLUMNS["feh_sigma"]:
        raise ValueError("T8 posterior does not use the required raw jc_sigma_m_h_1/2 columns.")
    if not np.isclose(float(metadata.get("cmd_min_parsec_teff_k", np.nan)), CMD_MIN_PARSEC_TEFF_K):
        raise ValueError("T8 posterior does not use the 4000 K full-grid CMD mask.")
    expected_grid = np.linspace(-1.0, 0.6, 81, dtype=np.float64)
    expected_weights = np.empty_like(expected_grid)
    expected_weights[0] = 0.5 * (expected_grid[1] - expected_grid[0])
    expected_weights[-1] = 0.5 * (expected_grid[-1] - expected_grid[-2])
    expected_weights[1:-1] = 0.5 * (expected_grid[2:] - expected_grid[:-2])
    if metadata.get("posterior_grid_nodes") != 81:
        raise ValueError("T8 posterior has an unexpected metallicity-grid size.")
    if not np.allclose(np.asarray(posterior.z_grid, dtype=np.float64), expected_grid, rtol=0.0, atol=1e-14):
        raise ValueError("T8 posterior does not use the exact 81-node [-1,0.6] grid.")
    if not np.allclose(
        np.asarray(metadata.get("metallicity_grid", []), dtype=np.float64),
        expected_grid,
        rtol=0.0,
        atol=1e-14,
    ):
        raise ValueError("T8 posterior metallicity-grid metadata is missing or inconsistent.")
    if not np.allclose(
        np.asarray(metadata.get("metallicity_grid_weights", []), dtype=np.float64),
        expected_weights,
        rtol=0.0, atol=1e-14,
    ):
        raise ValueError("T8 posterior trapezoid-weight metadata is missing or inconsistent.")
    if metadata.get("posterior_average_draw_count") is None or metadata.get("selected_draw_count") is None:
        raise ValueError("T8 posterior is missing selected posterior-draw metadata; rebuild calibration.")
    draw_indices = np.asarray(metadata.get("posterior_draw_indices", []), dtype=np.int64)
    if (
        draw_indices.ndim != 1
        or draw_indices.size < 1
        or draw_indices.size > 128
        or np.any(draw_indices < 0)
        or np.unique(draw_indices).size != draw_indices.size
        or int(metadata["posterior_average_draw_count"]) != draw_indices.size
        or int(metadata["selected_draw_count"]) != draw_indices.size
        or metadata.get("posterior_draw_indices_digest") != array_digest(draw_indices)
    ):
        raise ValueError("T8 posterior draw-index record is missing or inconsistent.")
    # Corrected/binary-equality columns are explicitly forbidden as inputs.  A
    # metadata marker makes accidental future substitutions fail loudly.
    forbidden = set(metadata.get("forbidden_metallicity_columns", []))
    expected_forbidden = {
        "jc_m_h_fit_1", "jc_m_h_fit_2", "jc_m_h_fit_cal_1", "jc_m_h_fit_cal_2",
        "jc_sigma_m_h_cal_1", "jc_sigma_m_h_cal_2",
    }
    if not expected_forbidden.issubset(forbidden):
        raise ValueError("T8 posterior has incomplete forbidden-column metadata.")
    expected_digest = array_digest(
        arrays["row_indices"], arrays["feh_observed"], arrays["feh_sigma"],
        arrays["absg"], arrays["color_observed"], arrays["color_sigma"],
    )
    if metadata.get("input_data_digest") != expected_digest:
        raise ValueError(
            "T8 posterior input-data digest does not match the current selected data; "
            "rebuild calibration."
        )


def _require_t8_lookup_metadata(lookup: DynamicsLikelihoodLookup, *, args) -> None:
    lookup.validate_t8_schema()
    metadata = dict(lookup.metadata)
    convergence = dict(metadata["convergence_check"])
    expected_sample_systems = 4 if args.quick else 32
    expected_scale_points = 4 if args.quick else 16
    expected = {
        "sqrt_mtot_points": int(args.lookup_mass_points),
        "velocity_quadrature_nodes": int(args.lookup_velocity_nodes),
        "sample_systems_requested": expected_sample_systems,
        "scale_points_requested": expected_scale_points,
        "outlier_u0": float(getattr(args, "outlier_u0", 40.0)),
        "outlier_sigma": float(getattr(args, "outlier_sigma", 13.0)),
    }
    mismatches = []
    for name in ("sqrt_mtot_points", "velocity_quadrature_nodes"):
        if int(metadata.get(name, -1)) != expected[name]:
            mismatches.append(f"{name}={metadata.get(name)!r}, expected {expected[name]}")
    for name in ("outlier_u0", "outlier_sigma"):
        if not np.isclose(float(metadata.get(name, np.nan)), expected[name], rtol=0.0, atol=0.0):
            mismatches.append(f"{name}={metadata.get(name)!r}, expected {expected[name]}")
    for name in ("sample_systems_requested", "scale_points_requested"):
        if int(convergence.get(name, -1)) != expected[name]:
            mismatches.append(f"convergence.{name}={convergence.get(name)!r}, expected {expected[name]}")
    for name, requested in (
        ("sqrt_mtot_min", args.lookup_sqrt_mass_min),
        ("sqrt_mtot_max", args.lookup_sqrt_mass_max),
        ("velocity_sigma_extent", args.lookup_sigma_extent),
    ):
        if not np.isclose(float(metadata.get(name, np.nan)), float(requested), rtol=0.0, atol=0.0):
            mismatches.append(f"{name}={metadata.get(name)!r}, expected {requested}")
    if mismatches:
        profile = "quick" if args.quick else "formal"
        raise ValueError(
            f"The saved lookup does not match the requested T8 {profile} execution profile: "
            + "; ".join(mismatches)
            + ". Rebuild lookup or pass the matching explicit CLI settings."
        )


def filter_data(
    arrays: Mapping[str, np.ndarray],
    *,
    max_systems: Optional[int],
    seed: int,
    fixed_rows: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    n_total = len(arrays["row_indices"])
    finite = np.ones(n_total, dtype=bool)
    for key in ("absg", "feh_observed", "feh_sigma", "color_observed", "color_sigma"):
        finite &= np.all(np.isfinite(arrays[key]), axis=1)
    finite &= np.isfinite(arrays["u"]) & np.isfinite(arrays["u_sigma"])
    finite &= np.all(arrays["feh_sigma"] > 0, axis=1)
    finite &= np.all(arrays["color_sigma"] > 0, axis=1)
    finite &= (arrays["u"] > 0) & (arrays["u_sigma"] > 0)
    in_magnitude_domain = np.all((arrays["absg"] >= 3.5) & (arrays["absg"] <= 13.5), axis=1)
    valid = finite & in_magnitude_domain
    valid_positions = np.flatnonzero(valid)
    print(
        f"Data selection: {n_total} input, {np.sum(~finite)} invalid, "
        f"{np.sum(finite & ~in_magnitude_domain)} outside M_G=[3.5,13.5], "
        f"{valid_positions.size} retained."
    )

    if fixed_rows is not None:
        fixed_rows = np.asarray(fixed_rows, dtype=np.int64)
        row_to_position = {int(row): int(pos) for pos, row in enumerate(arrays["row_indices"])}
        try:
            positions = np.array([row_to_position[int(row)] for row in fixed_rows], dtype=int)
        except KeyError as exc:
            raise ValueError(f"Saved row index {exc.args[0]} is absent from the input data.") from exc
        if np.any(~valid[positions]):
            bad_rows = fixed_rows[~valid[positions]][:10]
            raise ValueError(f"Saved rows no longer pass input validation: {bad_rows.tolist()}.")
    else:
        positions = valid_positions
        if max_systems is not None and positions.size > int(max_systems):
            positions = np.sort(
                np.random.default_rng(seed).choice(positions, size=int(max_systems), replace=False)
            )
            print(f"Subsampled {positions.size} systems with seed={seed}.")
    return {key: np.asarray(value)[positions] for key, value in arrays.items()}


def select_cmd_anchored_systems(
    arrays: Mapping[str, np.ndarray],
    posterior: MetallicityPosteriorGrid,
    dynamics_lookup: DynamicsLikelihoodLookup,
    color_surface: IsochroneColorSurfaceModel,
    *,
    require_both: bool = False,
):
    """Keep systems with one or both components in the fixed warm-CMD mask."""
    parsec_teff = color_surface.teff_from_absg_mh(
        arrays["absg"][:, :, None], posterior.z_grid[None, None, :]
    )
    component_mask = np.min(parsec_teff, axis=2) >= CMD_MIN_PARSEC_TEFF_K
    keep = np.all(component_mask, axis=1) if require_both else np.any(component_mask, axis=1)
    if not np.any(keep):
        raise ValueError("No system has a CMD-anchored warm component.")

    selection_text = (
        "both components have" if require_both else "at least one component has"
    )

    selected_arrays = {
        key: np.asarray(value)[keep] for key, value in arrays.items()
    }
    posterior_metadata = dict(posterior.metadata)
    posterior_metadata.update(
        {
            "subset": (
                f"{selection_text} minimum PARSEC Teff >= "
                f"{CMD_MIN_PARSEC_TEFF_K:g} K over the full latent-Z grid"
            ),
            "n_systems": int(np.sum(keep)),
            "n_cmd_components": int(np.sum(component_mask[keep])),
            "n_cmd_systems": int(np.sum(keep)),
        }
    )
    selected_posterior = MetallicityPosteriorGrid(
        row_indices=posterior.row_indices[keep],
        z_grid=posterior.z_grid.copy(),
        probabilities=posterior.probabilities[keep],
        z_quantiles=posterior.z_quantiles[keep],
        bad_probabilities=posterior.bad_probabilities[keep],
        metadata=posterior_metadata,
    )
    selected_posterior.validate()

    lookup_metadata = dict(dynamics_lookup.metadata)
    selected_lookup_digest = array_digest(
        selected_arrays["row_indices"], selected_arrays["u"], selected_arrays["u_sigma"]
    )
    selected_convergence = dict(lookup_metadata["convergence_check"])
    tested_rows = np.asarray(
        selected_convergence.get("selected_row_indices", []), dtype=np.int64
    )
    surviving_tested_rows = tested_rows[
        np.isin(tested_rows, selected_arrays["row_indices"])
    ]
    if surviving_tested_rows.size == 0:
        raise ValueError(
            "CMD-anchored selection removed every row used by the saved Rice "
            "convergence check; rebuild lookup for this subset."
        )
    selected_convergence.update(
        {
            "source_data_digest": selected_lookup_digest,
            "sample_system_count": int(surviving_tested_rows.size),
            "selected_row_indices": surviving_tested_rows.tolist(),
            "selected_row_indices_digest": array_digest(surviving_tested_rows),
            "source_subset": "cmd_anchored_selection_of_converged_lookup",
        }
    )
    lookup_metadata.update(
        {
            "data_digest": selected_lookup_digest,
            "convergence_check": selected_convergence,
            "subset": f"{selection_text} a CMD-anchored warm classification",
            "n_systems": int(np.sum(keep)),
        }
    )
    selected_lookup = DynamicsLikelihoodLookup(
        row_indices=dynamics_lookup.row_indices[keep],
        sqrt_mtot_grid=dynamics_lookup.sqrt_mtot_grid,
        log_good=dynamics_lookup.log_good[keep],
        log_bad=dynamics_lookup.log_bad[keep],
        metadata=lookup_metadata,
    )
    selected_lookup.validate_for(
        row_indices=selected_arrays["row_indices"],
        u=selected_arrays["u"],
        u_sigma=selected_arrays["u_sigma"],
    )
    print(
        f"CMD-anchored-only selection ({selection_text}): "
        f"{np.sum(keep)}/{keep.size} systems and "
        f"{np.sum(component_mask[keep])} warm components retained."
    )
    return selected_arrays, selected_posterior, selected_lookup


def build_surfaces():
    full_mass_surface = IsochroneMassSurfaceModel.from_interpolated_mass_data(
        REPO_ROOT / "data" / "interpolated_mass_data", mass_min=0.05
    )
    # T8b's approved clamped knot vector ends at M_G=13.5.  The packaged
    # interpolation grid has one additional 14-mag column, so restrict the
    # runtime surface to the declared T8 domain rather than extrapolating it.
    x_mask = full_mass_surface.absg_grid_np < 13.5 - 1e-12
    x_grid = np.concatenate([full_mass_surface.absg_grid_np[x_mask], [13.5]])
    mass_grid = np.column_stack(
        [full_mass_surface.mass_grid_np[:, x_mask],
         np.array([full_mass_surface.mass_from_absg_mh(13.5, mh) for mh in full_mass_surface.mh_grid_np])]
    )
    mass_surface = IsochroneMassSurfaceModel(
        x_grid,
        full_mass_surface.mh_grid_np,
        mass_grid,
        mass_min=0.05,
    )
    color_surface = IsochroneColorSurfaceModel.from_parsec_csv(
        REPO_ROOT / "data" / "PARSEC_logAge_6to10_0p5_MH_n1to0p6_0p2.csv",
        absg_grid=np.linspace(3.5, 13.5, 500),
    )
    return mass_surface, color_surface


def _ensure_mock_warm_cmd_coverage(
    arrays: Dict[str, np.ndarray], color_surface: IsochroneColorSurfaceModel
) -> bool:
    """Keep tiny synthetic smoke sets inside the 4000 K CMD support.

    The general recovery generator spans the full magnitude range, so a very
    small random draw can contain no warm member at all.  For mock-only runs,
    clip that rare case to the bright end so the T8 mask and downstream stages
    remain exercised.  Real-data arrays are never changed here.
    """
    teff = color_surface.teff_from_absg_mh(
        arrays["absg"][:, :, None], np.linspace(-1.0, 0.6, 81)[None, None, :]
    )
    if np.any(np.min(teff, axis=2) >= CMD_MIN_PARSEC_TEFF_K):
        return False
    arrays["absg"] = np.clip(np.asarray(arrays["absg"], dtype=float), 3.5, 5.8)
    return True


def save_grouped_mcmc(sampler, path: Path) -> None:
    np.savez_compressed(path, **grouped_mcmc_payload(sampler))


def _flatten_samples(samples: Mapping[str, np.ndarray]):
    for name, values in samples.items():
        values = np.asarray(values)
        if values.ndim == 1:
            yield name, values
        else:
            for index in np.ndindex(values.shape[1:]):
                label = name + "[" + ",".join(str(value) for value in index) + "]"
                yield label, values[(slice(None),) + index]


def save_posterior_summary(samples: Mapping[str, np.ndarray], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["parameter", "mean", "sd", "q05", "q16", "q50", "q84", "q95"])
        for name, values in _flatten_samples(samples):
            q05, q16, q50, q84, q95 = np.percentile(values, [5, 16, 50, 84, 95])
            writer.writerow(
                [name, np.mean(values), np.std(values, ddof=1), q05, q16, q50, q84, q95]
            )


def save_sampler_diagnostics(sampler, path: Path) -> None:
    fields = sampler.get_extra_fields(group_by_chain=True)
    diagnostics = {
        "num_divergences": int(np.sum(np.asarray(fields.get("diverging", 0)))),
        "mean_accept_prob": float(np.mean(np.asarray(fields.get("accept_prob", np.nan)))),
        "max_num_steps": int(np.max(np.asarray(fields.get("num_steps", 0)))),
        "num_chains": int(sampler.num_chains),
        "num_samples_per_chain": int(sampler.num_samples),
    }
    if sampler.num_samples >= 4:
        from numpyro.diagnostics import summary
        chain_summary = summary(sampler.get_samples(group_by_chain=True), group_by_chain=True)
        rhat = np.concatenate([np.asarray(item["r_hat"]).reshape(-1) for item in chain_summary.values()])
        ess = np.concatenate([np.asarray(item["n_eff"]).reshape(-1) for item in chain_summary.values()])
        diagnostics.update({
            "max_r_hat": float(np.max(rhat)) if np.all(np.isfinite(rhat)) else None,
            "min_n_eff": float(np.min(ess)) if np.all(np.isfinite(ess)) else None,
            "nonfinite_r_hat_count": int(np.sum(~np.isfinite(rhat))),
            "nonfinite_n_eff_count": int(np.sum(~np.isfinite(ess))),
        })
    path.write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")


def save_latent_summary(posterior: MetallicityPosteriorGrid, path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["row_index", "z_q16", "z_q50", "z_q84", "bad_probability_1", "bad_probability_2"])
        for row, quantile, bad in zip(
            posterior.row_indices, posterior.z_quantiles, posterior.bad_probabilities
        ):
            writer.writerow([int(row), *quantile.tolist(), *bad.tolist()])


def plot_calibration(
    calibrator: HierarchicalMetallicityCalibrator,
    posterior: MetallicityPosteriorGrid,
    output_dir: Path,
) -> None:
    samples = calibrator.posterior_samples
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    mg = np.linspace(3.5, 13.5, 300)
    fixed_bias = calibrator.bias_from_absg(mg)
    offset_name = "delta_z" if "delta_z" in samples else "z_offset"
    correction = samples[offset_name][:, None] + fixed_bias[None, :]
    lo, med, hi = np.percentile(correction, [16, 50, 84], axis=0)
    axes[0].fill_between(mg, lo, hi, color="tab:blue", alpha=0.20, label="68% interval")
    axes[0].plot(mg, med, color="tab:blue", label="T8 JCAPS bias + absolute offset")
    axes[0].plot(mg, fixed_bias, color="0.35", linestyle=":", label="Fixed JCAPS relative bias")
    axes[0].axhline(0, color="0.4", linestyle="--", linewidth=1)
    axes[0].axvline(8.5, color="0.75", linestyle="--", linewidth=1)
    axes[0].set(
        xlabel="$M_G$",
        ylabel="Observed JCAPS minus latent PARSEC [M/H] (dex)",
    )
    axes[0].legend(frameon=False)

    axes[1].hist(posterior.z_quantiles[:, 1], bins=50, histtype="step", density=True, label="posterior median Z")
    corrected_feh = calibrator.feh_observed - calibrator.fixed_bias
    axes[1].hist(
        corrected_feh.ravel(),
        bins=70,
        histtype="step",
        density=True,
        label="JCAPS after fixed relative correction",
    )
    axes[1].set(xlabel="[M/H]", ylabel="density")
    axes[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "metallicity_systematics.png", dpi=180)
    plt.close(fig)

    median_z = posterior.z_quantiles[:, 1]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for component, color in enumerate(("tab:blue", "tab:orange")):
        used = calibrator.cmd_mask[:, component]
        if not np.any(used):
            continue
        predicted = calibrator.color_surface.color_from_absg_mh(
            calibrator.absg[used, component], median_z[used]
        )
        residual = calibrator.color_observed[used, component] - predicted
        axes[0].scatter(
            calibrator.absg[used, component], residual, s=3, alpha=0.12, color=color,
            label=f"component {component + 1}",
        )
        axes[1].hist(residual, bins=70, histtype="step", density=True, color=color)
    axes[0].axhline(0, color="0.4", linestyle="--", linewidth=1)
    axes[0].set(xlabel="$M_G$", ylabel="BP-RP residual (mag)")
    axes[0].set_title(
        f"CMD terms used: {np.sum(calibrator.cmd_mask):,}/{calibrator.cmd_mask.size:,} "
        f"(PARSEC $T_{{\\rm eff}}\\geq{calibrator.cmd_min_teff:g}$ K)"
    )
    axes[0].legend(frameon=False, markerscale=3)
    axes[1].set(xlabel="BP-RP residual (mag)", ylabel="density")
    fig.tight_layout()
    fig.savefig(output_dir / "cmd_residuals.png", dpi=180)
    plt.close(fig)


def plot_mlr_grid(grid: Mapping[str, np.ndarray], output_dir: Path) -> None:
    absg, z_values = grid["absg_grid"], grid["z_grid"]
    parsec_mass = grid["parsec_mass"]
    raw_mass = np.asarray(grid["mass"])
    mass = np.asarray(grid.get("mass_percentiles", raw_mass))
    if mass.ndim == 3 and mass.shape[0] != 3:
        mass = np.percentile(mass, [16.0, 50.0, 84.0], axis=0)
    correction = np.asarray(grid.get("correction_percentiles", grid["correction_percent"]))
    raw_correction = np.asarray(grid["correction_percent"])
    if correction.ndim == 3 and correction.shape[0] != 3:
        correction = np.percentile(raw_correction, [16.0, 50.0, 84.0], axis=0)
    cmap = plt.get_cmap("viridis")
    norm = plt.Normalize(z_values.min(), z_values.max())
    fig, (ax_mass, ax_corr) = plt.subplots(2, 1, figsize=(8, 9), sharex=True)
    for index, z in enumerate(z_values):
        color = cmap(norm(z))
        ax_mass.plot(
            absg,
            parsec_mass[index],
            color=color,
            linestyle="--",
            linewidth=1.4,
            alpha=0.9,
        )
        ax_mass.fill_between(absg, mass[0, index], mass[2, index], color=color, alpha=0.15)
        ax_mass.plot(absg, mass[1, index], color=color)
        ax_corr.fill_between(absg, correction[0, index], correction[2, index], color=color, alpha=0.15)
        ax_corr.plot(absg, correction[1, index], color=color, label=f"{z:+.1f}")
    solar_color = cmap(norm(0.0))
    ax_mass.scatter(
        [4.67],
        [1.0],
        marker="*",
        s=130,
        facecolor=solar_color,
        edgecolor="black",
        linewidth=0.7,
        zorder=10,
    )
    ax_mass.legend(
        handles=[
            Line2D([0], [0], color="0.25", linewidth=1.8, label="Corrected MLR"),
            Line2D(
                [0], [0], color="0.25", linestyle="--", linewidth=1.4,
                label="Original PARSEC MLR",
            ),
            Line2D(
                [0], [0], marker="*", linestyle="none", markersize=11,
                markerfacecolor=solar_color, markeredgecolor="black", label="Sun ([M/H] = 0)",
            ),
        ],
        frameon=False,
        loc="upper left",
    )
    ax_mass.set_yscale("log")
    ax_mass.set_ylabel(r"Mass ($M_\odot$)")
    ax_mass.invert_xaxis()
    ax_corr.axhline(0, color="0.4", linestyle="--", linewidth=1)
    ax_corr.set(xlabel="$M_G$", ylabel="Correction relative to PARSEC (%)")
    ax_corr.legend(title="[M/H]", frameon=False, ncol=3)
    fig.tight_layout()
    fig.savefig(output_dir / "mlr_correction.png", dpi=180)
    plt.close(fig)


def plot_mlr_corner(
    samples: Mapping[str, np.ndarray], output_dir: Path, mg_knots: np.ndarray
) -> None:
    """Save a compact corner plot of representative T8 MLR parameters."""
    import corner

    if "c0" in samples:
        names = ["c0", "log_lambda_x", "log_lambda_z", "f_outlier"]
        labels = ["c0", "log lambda_x", "log lambda_Z", "f_outlier"]
    else:
        names = ["f0_knots", "fz_knots", "f_outlier"]
        labels = ["f0 knots", "fZ knots", "f_outlier"]
    columns = []
    expanded_labels = []
    for name, label in zip(names, labels):
        if name not in samples:
            continue
        values = np.asarray(samples[name])
        if values.ndim == 1:
            columns.append(values)
            expanded_labels.append(label)
        else:
            for idx in np.ndindex(values.shape[1:]):
                columns.append(values[(slice(None),) + idx])
                expanded_labels.append(f"{label}[{','.join(map(str, idx))}]")
    if not columns:
        return
    plot_samples = np.column_stack(columns)
    if plot_samples.shape[0] < max(4, plot_samples.shape[1]):
        print("Skipping corner plot for a smoke run with too few posterior draws.")
        return
    figure = corner.corner(
        plot_samples,
        labels=expanded_labels,
        quantiles=(0.16, 0.50, 0.84),
        show_titles=True,
        title_fmt=".3f",
        color="tab:blue",
        plot_datapoints=False,
        fill_contours=True,
        levels=(0.393, 0.865),
        label_kwargs={"fontsize": 11},
        title_kwargs={"fontsize": 9},
    )
    figure.savefig(output_dir / "mlr_corner.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def _make_t8_mlr(mass_surface, args):
    """Construct the hard-monotone T8 model from explicit CLI settings."""
    return MonotoneTensorSplineMLR(
        mass_surface,
        knots_x=np.asarray(args.mlr_knot_x, dtype=float),
        knots_z=np.asarray(args.mlr_knot_z, dtype=float),
        degree_x=int(args.mlr_degree_x),
        degree_z=int(args.mlr_degree_z),
        tau_D=float(args.mlr_tau_d),
        tau_x=float(args.mlr_tau_x),
        tau_Z=float(args.mlr_tau_z),
        tau_xZ=float(args.mlr_tau_xz),
        solar_anchor_mean=float(args.mlr_solar_mean),
        solar_anchor_sigma=float(args.mlr_solar_sigma),
    )


def save_mlr_grid_csv(grid: Mapping[str, np.ndarray], path: Path) -> None:
    mass_values = np.asarray(grid.get("mass_percentiles", grid["mass"]))
    correction_values = np.asarray(
        grid.get("correction_percentiles", grid["correction_percent"])
    )
    if mass_values.ndim == 3 and mass_values.shape[0] != 3:
        mass_values = np.percentile(mass_values, [16.0, 50.0, 84.0], axis=0)
    if correction_values.ndim == 3 and correction_values.shape[0] != 3:
        correction_values = np.percentile(correction_values, [16.0, 50.0, 84.0], axis=0)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["absg", "mh", "delta_star_p50_dex", "epsilon_P_dex", "delta_P_p50_dex",
             "correction_p16_percent", "correction_p50_percent", "correction_p84_percent",
             "mass_p16", "mass_p50", "mass_p84"]
        )
        for iz, z in enumerate(grid["z_grid"]):
            for ix, mg in enumerate(grid["absg_grid"]):
                delta_star = np.asarray(grid.get("delta_star", np.zeros((1, len(grid["z_grid"]), len(grid["absg_grid"])))))[..., iz, ix]
                epsilon = np.asarray(grid.get("epsilon_P", np.zeros((len(grid["z_grid"]), len(grid["absg_grid"])))))[iz, ix]
                delta_p = np.asarray(grid.get("delta_P", np.zeros((1, len(grid["z_grid"]), len(grid["absg_grid"])))))[..., iz, ix]
                delta_star50 = float(np.percentile(delta_star, 50.0)) if np.ndim(delta_star) else float(delta_star)
                delta_p50 = float(np.percentile(delta_p, 50.0)) if np.ndim(delta_p) else float(delta_p)
                writer.writerow(
                    [mg, z, delta_star50, float(epsilon), delta_p50,
                     *correction_values[:, iz, ix], *mass_values[:, iz, ix]]
                )


def recovery_report(samples: Mapping[str, np.ndarray], truth: Mapping[str, object], names) -> Dict[str, object]:
    result = {}
    for name in names:
        if name not in samples or name not in truth:
            continue
        draw = np.asarray(samples[name])
        expected = np.asarray(truth[name])
        lo, median, hi = np.percentile(draw, [5, 50, 95], axis=0)
        result[name] = {
            "truth": expected.tolist(),
            "median": np.asarray(median).tolist(),
            "inside_90_percent_interval": np.asarray((expected >= lo) & (expected <= hi)).tolist(),
            "sign_recovered": np.asarray(np.sign(median) == np.sign(expected)).tolist(),
        }
    return result


def run_calibration(
    arrays: Mapping[str, np.ndarray],
    color_surface: IsochroneColorSurfaceModel,
    output_dir: Path,
    args,
    truth: Optional[Mapping[str, object]],
) -> MetallicityPosteriorGrid:
    calibrator = HierarchicalMetallicityCalibrator(
        color_surface, cmd_min_teff=CMD_MIN_PARSEC_TEFF_K
    )
    calibrator.set_data(
        row_indices=arrays["row_indices"],
        absg=arrays["absg"],
        feh_observed=arrays["feh_observed"],
        feh_sigma=arrays["feh_sigma"],
        color_observed=arrays["color_observed"],
        color_sigma=arrays["color_sigma"],
    )
    n_cmd_components = int(np.sum(calibrator.cmd_mask))
    n_cmd_systems = int(np.sum(np.any(calibrator.cmd_mask, axis=1)))
    print(
        f"CMD selection: {n_cmd_components}/{calibrator.cmd_mask.size} components in "
        f"{n_cmd_systems}/{calibrator.row_indices.size} systems have PARSEC "
        f"Teff >= {CMD_MIN_PARSEC_TEFF_K:g} K over the full metallicity grid."
    )
    print("Running stage-one metallicity calibration...")
    sampler = calibrator.run_mcmc(
        num_warmup=args.warmup,
        num_samples=args.samples,
        num_chains=args.chains,
        seed=args.seed,
        target_accept_prob=args.target_accept,
    )
    try:
        sampler.print_summary()
    except AssertionError:
        # NumPyro's split-R-hat summary requires >=4 draws per chain; tiny
        # smoke tests intentionally use fewer draws and still produce valid
        # posterior arrays/diagnostics.
        print("Skipping NumPyro summary for a smoke run with fewer than four draws per chain.")
    sample_count = int(next(iter(calibrator.posterior_samples.values())).shape[0])
    selected_draw_indices = _selected_draw_indices(sample_count, args.posterior_grid_draws, args.seed)
    metadata = {
        "model": CALIBRATION_MODEL_ID,
        "schema": "t8a-metallicity-posterior-v1",
        "workflow": T8_WORKFLOW_ID,
        "feh_columns": INPUT_COLUMNS["feh_observed"],
        "feh_sigma_columns": INPUT_COLUMNS["feh_sigma"],
        "color_columns": INPUT_COLUMNS["color_observed"],
        "absg_columns": INPUT_COLUMNS["absg"],
        "input_path": str(args.data),
        "seed": int(args.seed),
        "mock": bool(args.mock),
        "n_systems": int(len(arrays["row_indices"])),
        "cmd_min_parsec_teff_k": CMD_MIN_PARSEC_TEFF_K,
        "cmd_temperature_selection": "minimum PARSEC Teff over the full latent metallicity grid",
        "n_cmd_components": n_cmd_components,
        "n_cmd_systems": n_cmd_systems,
        "metallicity_observation_model": calibrator.observation_model_metadata(),
        "bad_probability_semantics": "fixed zero; T8 has no good/bad metallicity mixture state",
        "posterior_grid_nodes": int(calibrator.z_grid.size),
        "posterior_grid_domain": [float(calibrator.z_grid[0]), float(calibrator.z_grid[-1])],
        "posterior_grid_quadrature": "81 equally spaced nodes with trapezoid weights",
        "posterior_average_draw_count": int(selected_draw_indices.size),
        "posterior_draw_selection_seed": int(args.seed),
        "posterior_draw_selection": "without replacement; selected indices are deterministically reproducible from seed",
        "posterior_draw_indices": selected_draw_indices.tolist(),
        "posterior_draw_indices_digest": array_digest(selected_draw_indices),
        "population_prior": "six truncated Gaussian bases, Dirichlet(1,...,1)",
        "cmd_surface_construction": {
            "source_age_selection": "9.3 < log10(age/yr) < 10.0",
            "source_gmag_selection": "3 < G_mag < 15",
            "evaluation_grid": "500 equally spaced M_G points on [3.5,13.5]",
            "duplicate_rule": "mean color and mean logTe at fixed source-table G_mag and MH",
            "color_curve": "up-to-cubic UnivariateSpline with s=0.001",
            "temperature_curve": "linear interpolation in logTe, exponentiated when tabulated",
            "runtime_lookup": "bilinear interpolation in tabulated color and linear-Teff grids",
        },
        "forbidden_metallicity_columns": sorted(
            ["jc_m_h_fit_1", "jc_m_h_fit_2", "jc_m_h_fit_cal_1", "jc_m_h_fit_cal_2", "jc_sigma_m_h_cal_1", "jc_sigma_m_h_cal_2"]
        ),
        "input_data_digest": array_digest(
            arrays["row_indices"], arrays["feh_observed"], arrays["feh_sigma"],
            arrays["absg"], arrays["color_observed"], arrays["color_sigma"]
        ),
    }
    posterior = calibrator.posterior_grid(
        max_draws=args.posterior_grid_draws, seed=args.seed, metadata=metadata
    )
    posterior.save(output_dir / T8_POSTERIOR_NAME)
    save_grouped_mcmc(sampler, output_dir / "calibration_mcmc_t8.npz")
    save_posterior_summary(calibrator.posterior_samples, output_dir / "calibration_summary_t8.csv")
    save_sampler_diagnostics(sampler, output_dir / "calibration_diagnostics_t8.json")
    (output_dir / "calibration_model.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    save_latent_summary(posterior, output_dir / "latent_metallicity_summary_t8.csv")
    plot_calibration(calibrator, posterior, output_dir)
    if truth is not None:
        report = recovery_report(
            calibrator.posterior_samples,
            truth,
            ["delta_z", "z_offset", "s_C", "cmd_scatter"],
        )
        (output_dir / "calibration_recovery.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
    return posterior


def run_mlr(
    arrays: Mapping[str, np.ndarray],
    posterior: MetallicityPosteriorGrid,
    dynamics_lookup,
    mass_surface: IsochroneMassSurfaceModel,
    output_dir: Path,
    args,
    truth: Optional[Mapping[str, object]],
    dynamics_shape_stack=None,
) -> None:
    metadata = posterior.metadata
    baseline_m1 = mass_surface.mass_from_absg_mh(
        arrays["absg"][:, 0, None], posterior.z_grid[None, :]
    )
    baseline_m2 = mass_surface.mass_from_absg_mh(
        arrays["absg"][:, 1, None], posterior.z_grid[None, :]
    )
    baseline_sqrt_mass = np.sqrt(baseline_m1 + baseline_m2)
    dynamics_source = dynamics_lookup if dynamics_lookup is not None else dynamics_shape_stack
    lookup_min, lookup_max = dynamics_source.sqrt_mtot_grid[[0, -1]]
    print(
        "Baseline PARSEC sqrt(total mass) range: "
        f"[{np.min(baseline_sqrt_mass):.4f}, {np.max(baseline_sqrt_mass):.4f}]; "
        f"lookup range: [{lookup_min:.4f}, {lookup_max:.4f}]."
    )
    if np.min(baseline_sqrt_mass) < lookup_min or np.max(baseline_sqrt_mass) > lookup_max:
        raise ValueError(
            "The uncorrected PARSEC mass surface lies outside the dynamics lookup domain; "
            "rebuild the lookup with wider --lookup-sqrt-mass-min/max values."
        )
    (output_dir / "mlr_lookup_metadata_t8.json").write_text(
        json.dumps(dict(dynamics_source.metadata), indent=2), encoding="utf-8"
    )
    mlr = _make_t8_mlr(mass_surface, args)
    observed_mg_min = float(np.min(arrays["absg"]))
    observed_mg_max = float(np.max(arrays["absg"]))
    domain_x = np.asarray(getattr(mlr, "x_domain", args.mlr_knot_x[[0, -1]]), dtype=float)
    if observed_mg_min < domain_x[0] or observed_mg_max > domain_x[-1]:
        raise ValueError(
            "Observed component magnitudes lie outside the requested MLR knots: "
            f"data=[{observed_mg_min:.6g}, {observed_mg_max:.6g}], "
            f"domain=[{domain_x[0]:.6g}, {domain_x[-1]:.6g}]."
        )
    mlr_model_metadata = {
        "n_systems": int(len(arrays["row_indices"])),
        "experimental_outlier_sensitivity": bool(args.outlier_sensitivity),
        "outlier_shape": {
            "mu": float(args.outlier_u0), "sigma": float(args.outlier_sigma),
            "support": [0.0, 80.0], "units": "km s^-1 sqrt(AU)",
        },
        "metallicity_posterior_source": str(args.metallicity_posterior.resolve()) if args.metallicity_posterior else str(output_dir / T8_POSTERIOR_NAME),
        "schema": "t8b-monotone-tensor-spline-mlr-v1",
        "experimental_shape_sensitivity": bool(args.shape_sensitivity),
        "sampled_dynamics_shape": bool(args.sample_dynamics_shape),
        "good_shape_constants": {
            "A": _hm.RICE_GOOD_A, "B": _hm.RICE_GOOD_B,
            "uc": _hm.RICE_GOOD_UC, "C": _hm.RICE_GOOD_C,
            "sampled": bool(args.sample_dynamics_shape),
            "prior_axes": (
                {k: list(map(float, v)) for k, v in T82_SHAPE_AXES.items()}
                if args.sample_dynamics_shape else None
            ),
        },
        "knot_x": np.asarray(args.mlr_knot_x, dtype=float).tolist(),
        "knot_z": np.asarray(args.mlr_knot_z, dtype=float).tolist(),
        "degree_x": int(args.mlr_degree_x),
        "degree_z": int(args.mlr_degree_z),
        "observed_component_mg_min": observed_mg_min,
        "observed_component_mg_max": observed_mg_max,
        "metallicity_posterior_subset": metadata.get("subset"),
        "sampled_parameters": list(mlr.sample_parameter_names),
        "sampled_scalar_count": int(mlr.n_params),
        "parsec_projection_coefficients": mlr.parsec_projection.tolist(),
        "parsec_projection": dict(mlr.projection_metadata),
        "lambda_reference_x_z": list(mlr.lambda_references()),
        "priors": {
            "c0": "Normal(Pi[0,0], 0.10)",
            "a_b_r": "Normal(0,1)",
            "log_lambda_x_z": "Normal(log(lambda_reference_x_z),0.7)",
            "f_outlier": "Beta(3,12)",
        },
        "f_outlier_semantics": (
            "normalized mixture probability; physical interpretation depends on "
            "the contamination and selection model. T8.1 normalizes the good basis "
            "to unit support integral so both mixture components integrate to one."
        ),
        "derived_output_definitions": {
            "mass": "final inferred mass in solar masses",
            "delta_star": "g - g_P_star in dex",
            "epsilon_P": "g_P_star - g_P in dex",
            "delta_P": "g - g_P = delta_star + epsilon_P in dex",
        },
        "penalty_scales_dex": {
            "tau_D": float(mlr.tau_D),
            "tau_x": float(mlr.tau_x),
            "tau_Z": float(mlr.tau_Z),
            "tau_xZ": float(mlr.tau_xZ),
        },
        "solar_anchor": {
            "M_G": 4.67,
            "Z": 0.0,
            "mean_log10_mass": float(mlr.solar_anchor_mean),
            "sigma_log10_mass": float(mlr.solar_anchor_sigma),
        },
    }
    (output_dir / "mlr_model.json").write_text(
        json.dumps(mlr_model_metadata, indent=2), encoding="utf-8"
    )
    mlr.set_data(
        row_indices=arrays["row_indices"],
        u=arrays["u"],
        u_sigma=arrays["u_sigma"],
        absg=arrays["absg"],
        metallicity_grid=posterior,
        dynamics_lookup=dynamics_lookup,
        dynamics_shape_stack=dynamics_shape_stack,
    )
    print("Running stage-two dynamical MLR correction...")
    sampler = mlr.run_mcmc(
        num_warmup=args.warmup,
        num_samples=args.samples,
        num_chains=args.chains,
        seed=args.seed + 1,
        target_accept_prob=args.target_accept,
    )
    mlr_model_metadata["initialization"] = dict(mlr.initialization_metadata)
    (output_dir / "mlr_model.json").write_text(
        json.dumps(mlr_model_metadata, indent=2), encoding="utf-8"
    )
    try:
        sampler.print_summary()
    except AssertionError:
        print("Skipping NumPyro summary for a smoke run with fewer than four draws per chain.")
    grid = mlr.correction_grid()
    np.savez_compressed(output_dir / "mlr_derived_grid_t8.npz", **grid)
    save_grouped_mcmc(sampler, output_dir / T8_MLR_NAME)
    save_posterior_summary(mlr.posterior_samples, output_dir / "mlr_summary_t8.csv")
    save_sampler_diagnostics(sampler, output_dir / "mlr_diagnostics_t8.json")
    save_mlr_grid_csv(grid, output_dir / "mlr_derived_grid_t8.csv")
    plot_mlr_grid(grid, output_dir)
    plot_mlr_corner(
        mlr.posterior_samples,
        output_dir,
        np.asarray(getattr(mlr, "mg_knots", args.mlr_knot_x), dtype=float),
    )
    monotonicity = {
        "cell_violation_fraction": float(np.asarray(grid.get("monotone_cell_violation_fraction", 0.0))),
        "draw_violation_fraction": float(np.asarray(grid.get("monotone_draw_violation_fraction", 0.0))),
        "magnitude_direction_violation_fraction": float(
            np.asarray(grid.get("monotone_x_cell_violation_fraction", 0.0))
        ),
        "metallicity_direction_violation_fraction": float(
            np.asarray(grid.get("monotone_z_cell_violation_fraction", 0.0))
        ),
        "definition": (
            "final mass must not increase with M_G and must not decrease with [M/H]"
        ),
    }
    (output_dir / "mlr_monotonicity.json").write_text(
        json.dumps(monotonicity, indent=2), encoding="utf-8"
    )
    if truth is not None:
        # The legacy mock generator uses a three-knot T7 truth, which is not
        # parameter-by-parameter comparable with T8's monotone tensor surface.
        (output_dir / "mlr_recovery.json").write_text(
            json.dumps(
                {
                    "status": "not_parameter_comparable",
                    "reason": "mock truth uses the legacy T7 f0/fZ parameterization",
                    "comparable_outputs": ["final_mass_surface", "dynamics_likelihood"],
                },
                indent=2,
            ),
            encoding="utf-8",
        )


def run_lookup(
    arrays: Mapping[str, np.ndarray], output_dir: Path, args
) -> DynamicsLikelihoodLookup:
    """Precompute the expensive Rice convolution independently of the MCMC."""
    print("Checking Rice lookup convergence on a deterministic subset...")
    convergence = assess_rice_lookup_convergence(
        row_indices=arrays["row_indices"],
        u=arrays["u"],
        u_sigma=arrays["u_sigma"],
        sqrt_mtot_min=args.lookup_sqrt_mass_min,
        sqrt_mtot_max=args.lookup_sqrt_mass_max,
        sample_systems=4 if args.quick else 32,
        scale_points=4 if args.quick else 16,
        velocity_sigma_extent=args.lookup_sigma_extent,
        velocity_quadrature_nodes=args.lookup_velocity_nodes,
        tolerance_log_likelihood=args.lookup_convergence_tolerance,
        outlier_u0=args.outlier_u0,
        outlier_sigma=args.outlier_sigma,
    )
    (output_dir / "lookup_convergence_t8.json").write_text(
        json.dumps(convergence, indent=2), encoding="utf-8"
    )
    if not convergence["passed"]:
        raise ValueError(
            "Rice lookup convergence failed the maximum absolute log-likelihood "
            f"tolerance {args.lookup_convergence_tolerance:g}; increase the local "
            "window/node settings before building the formal table."
        )
    print("Building per-system dynamics likelihood lookup...")
    lookup = DynamicsLikelihoodLookup.precompute(
        row_indices=arrays["row_indices"],
        u=arrays["u"],
        u_sigma=arrays["u_sigma"],
        sqrt_mtot_min=args.lookup_sqrt_mass_min,
        sqrt_mtot_max=args.lookup_sqrt_mass_max,
        sqrt_mtot_points=args.lookup_mass_points,
        velocity_quadrature_nodes=args.lookup_velocity_nodes,
        velocity_sigma_extent=args.lookup_sigma_extent,
        outlier_u0=args.outlier_u0,
        outlier_sigma=args.outlier_sigma,
        outlier_sensitivity=args.outlier_sensitivity,
        system_chunk=args.lookup_system_chunk,
    )
    lookup = DynamicsLikelihoodLookup(
        row_indices=lookup.row_indices,
        sqrt_mtot_grid=lookup.sqrt_mtot_grid,
        log_good=lookup.log_good,
        log_bad=lookup.log_bad,
        metadata={**dict(lookup.metadata), "convergence_check": convergence},
    )
    _require_t8_lookup_metadata(lookup, args=args)
    lookup_path = lookup.save(output_dir / T8_LOOKUP_NAME)
    diagnostics = dict(lookup.metadata)
    diagnostics.update(
        {
            "n_systems": int(lookup.row_indices.size),
            "file": str(lookup_path),
            "file_size_bytes": int(lookup_path.stat().st_size),
            "log_good_min": float(np.min(lookup.log_good)),
            "log_good_max": float(np.max(lookup.log_good)),
            "log_bad_min": float(np.min(lookup.log_bad)),
            "log_bad_max": float(np.max(lookup.log_bad)),
        }
    )
    (output_dir / "dynamics_likelihood_lookup_diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2), encoding="utf-8"
    )
    print(f"Saved dynamics lookup: {lookup_path}")
    return lookup


def run_shape_stack(arrays, output_dir, args):
    """Build the 27-node good-shape lookup stack (T8.2).

    The quadrature convergence gate runs once at the box center: quadrature
    accuracy depends on the (u, u_sigma, s) sampling, not on which member of
    the smooth shape family is integrated.
    """
    import binary_masses.hierarchical_metallicity as _hm
    axes = {name: np.asarray(values, dtype=np.float64) for name, values in T82_SHAPE_AXES.items()}
    center = {name: float(np.mean(values)) for name, values in axes.items()}
    _hm.set_good_shape_constants(
        B=float(np.exp(center["log_b"])),
        C=float(np.exp(center["log_c"])),
        uc=float(np.exp(center["log_uc"])),
    )
    print("Checking Rice lookup convergence at the shape-box center node...")
    convergence = assess_rice_lookup_convergence(
        row_indices=arrays["row_indices"],
        u=arrays["u"],
        u_sigma=arrays["u_sigma"],
        sqrt_mtot_min=args.lookup_sqrt_mass_min,
        sqrt_mtot_max=args.lookup_sqrt_mass_max,
        sample_systems=4 if args.quick else 32,
        scale_points=4 if args.quick else 16,
        velocity_sigma_extent=args.lookup_sigma_extent,
        velocity_quadrature_nodes=args.lookup_velocity_nodes,
        tolerance_log_likelihood=args.lookup_convergence_tolerance,
        outlier_u0=args.outlier_u0,
        outlier_sigma=args.outlier_sigma,
    )
    if not convergence["passed"]:
        raise ValueError("Rice convergence failed at the shape-box center; widen the quadrature settings.")
    print("Building the 27-node good-shape lookup stack...")
    stack = _hm.DynamicsLikelihoodShapeStack.build(
        row_indices=arrays["row_indices"],
        u=arrays["u"],
        u_sigma=arrays["u_sigma"],
        axes=axes,
        sqrt_mtot_min=args.lookup_sqrt_mass_min,
        sqrt_mtot_max=args.lookup_sqrt_mass_max,
        sqrt_mtot_points=args.lookup_mass_points,
        velocity_quadrature_nodes=args.lookup_velocity_nodes,
        velocity_sigma_extent=args.lookup_sigma_extent,
        outlier_u0=args.outlier_u0,
        outlier_sigma=args.outlier_sigma,
        system_chunk=args.lookup_system_chunk,
    )
    metadata = dict(stack.metadata)
    metadata["convergence_check"] = convergence
    stack.metadata = metadata
    (output_dir / "shapestack_convergence_t8.json").write_text(
        json.dumps(convergence, indent=2), encoding="utf-8"
    )
    (output_dir / "shapestack_nodes_t8.json").write_text(
        json.dumps({"axes": {k: list(map(float, v)) for k, v in axes.items()},
                    "nodes": stack.node_constants}, indent=2),
        encoding="utf-8",
    )
    stack_path = stack.save(output_dir / T8_SHAPESTACK_NAME)
    print(f"Saved dynamics shape stack: {stack_path}")
    return stack


def regenerate_mlr_plots(
    mass_surface: IsochroneMassSurfaceModel, output_dir: Path
) -> None:
    """Regenerate MLR grid and figures from an existing grouped MCMC file."""
    mcmc_path = output_dir / T8_MLR_NAME
    with np.load(mcmc_path, allow_pickle=False) as saved:
        samples = {}
        for key in saved.files:
            if not key.startswith("posterior__"):
                continue
            name = key[len("posterior__"):]
            values = np.asarray(saved[key])
            samples[name] = values.reshape((-1,) + values.shape[2:])
    model_path = output_dir / "mlr_model.json"
    if model_path.exists():
        model_metadata = json.loads(model_path.read_text(encoding="utf-8"))
        class _PlotArgs:
            mlr_knot_x = np.asarray(model_metadata.get("knot_x", T8_MLR_DEFAULT_KNOT_X), dtype=float)
            mlr_knot_z = np.asarray(model_metadata.get("knot_z", T8_MLR_DEFAULT_KNOT_Z), dtype=float)
            mlr_degree_x = int(model_metadata.get("degree_x", 3))
            mlr_degree_z = int(model_metadata.get("degree_z", 3))
            mlr_tau_d = 0.10
            mlr_tau_x = mlr_tau_z = mlr_tau_xz = 0.05
            mlr_solar_mean = 0.0
            mlr_solar_sigma = 0.01 / np.log(10.0)
        mlr = _make_t8_mlr(mass_surface, _PlotArgs())
    else:
        mlr = MonotoneTensorSplineMLR(mass_surface)
    mlr.posterior_samples = samples
    grid = mlr.correction_grid()
    np.savez_compressed(output_dir / "mlr_derived_grid_t8.npz", **grid)
    save_mlr_grid_csv(grid, output_dir / "mlr_derived_grid_t8.csv")
    plot_mlr_grid(grid, output_dir)
    plot_mlr_corner(samples, output_dir, np.asarray(getattr(mlr, "mg_knots", T8_MLR_DEFAULT_KNOT_X)))
    print(f"Regenerated MLR figures from {mcmc_path}.")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("calibration", "lookup", "mlr", "plot", "all"),
        default="all",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=REPO_ROOT / "data" / "jd_msms_single_bic_1kpc_filtered_cmdcut_cutb_jcaps_err_mhcal_good.fits",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "results" / "hierarchical_metallicity_t8_1",
    )
    parser.add_argument(
        "--metallicity-posterior",
        type=Path,
        default=None,
        help=(
            "Existing T8 latent_metallicity_weights_t8.npz to reuse for lookup/MLR; "
            "allows a new output directory without copying or overwriting old results."
        ),
    )
    parser.add_argument(
        "--dynamics-lookup",
        type=Path,
        default=None,
        help=(
            "Existing T8 dynamics_likelihood_lookup_t8.npz to reuse for MLR. "
            "Legacy or incomplete lookup metadata is rejected."
        ),
    )
    parser.add_argument("--quick", action="store_true", help="Use 2,000 systems, 200 warmup, 300 draws, and one chain.")
    parser.add_argument("--mock", action="store_true", help="Use a generated recovery data set instead of the real FITS table.")
    parser.add_argument("--mock-systems", type=int, default=300)
    parser.add_argument("--max-systems", type=int, default=None)
    parser.add_argument(
        "--cmd-anchored-only",
        action="store_true",
        help=(
            "In the MLR stage, retain only systems with at least one component "
            f"in the fixed PARSEC Teff >= {CMD_MIN_PARSEC_TEFF_K:g} K CMD mask. "
            "The supplied full-sample dynamics lookup is subset in memory."
        ),
    )
    parser.add_argument(
        "--both-cmd-anchored-only",
        action="store_true",
        help=(
            "In the MLR stage, retain only systems whose two components are both "
            f"in the fixed PARSEC Teff >= {CMD_MIN_PARSEC_TEFF_K:g} K CMD mask."
        ),
    )
    parser.add_argument("--mlr-knot-x", type=float, nargs="+", default=T8_MLR_DEFAULT_KNOT_X.tolist(),
                        help="Full clamped T8 M_G knot vector (default: 8x4 design).")
    parser.add_argument("--mlr-knot-z", type=float, nargs="+", default=T8_MLR_DEFAULT_KNOT_Z.tolist(),
                        help="Full clamped T8 [M/H] knot vector (default: 8x4 design).")
    parser.add_argument("--mlr-degree-x", type=int, default=3)
    parser.add_argument("--mlr-degree-z", type=int, default=3)
    parser.add_argument("--mlr-tau-d", type=float, default=0.10)
    parser.add_argument("--mlr-tau-x", type=float, default=0.05)
    parser.add_argument("--mlr-tau-z", type=float, default=0.05)
    parser.add_argument("--mlr-tau-xz", type=float, default=0.05)
    parser.add_argument("--mlr-solar-mean", type=float, default=0.0)
    parser.add_argument("--mlr-solar-sigma", type=float, default=0.01 / np.log(10.0))
    parser.add_argument("--warmup", type=int, default=None)
    parser.add_argument("--samples", type=int, default=None)
    parser.add_argument("--chains", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20260819)
    parser.add_argument("--posterior-grid-draws", type=int, default=128)
    parser.add_argument("--lookup-mass-points", type=int, default=None)
    parser.add_argument("--lookup-sqrt-mass-min", type=float, default=0.25)
    parser.add_argument("--lookup-sqrt-mass-max", type=float, default=2.5)
    parser.add_argument("--lookup-velocity-nodes", type=int, default=64)
    parser.add_argument("--lookup-sigma-extent", type=float, default=10.0)
    parser.add_argument("--lookup-convergence-tolerance", type=float, default=1e-3)
    parser.add_argument("--lookup-system-chunk", type=int, default=128)
    parser.add_argument("--outlier-u0", type=float, default=40.0)
    parser.add_argument("--outlier-sigma", type=float, default=13.0)
    parser.add_argument("--outlier-sensitivity", action="store_true",
                        help="Opt in to an experimental outlier-shape sensitivity run.")
    parser.add_argument("--good-shape-b", type=float, default=None,
                        help="Override the fixed good-basis B constant (shape sensitivity).")
    parser.add_argument("--good-shape-uc", type=float, default=None,
                        help="Override the fixed good-basis uc constant (shape sensitivity).")
    parser.add_argument("--good-shape-c", type=float, default=None,
                        help="Override the fixed good-basis C constant (shape sensitivity).")
    parser.add_argument("--shape-sensitivity", action="store_true",
                        help="Opt in to an experimental good-shape sensitivity run.")
    parser.add_argument("--sample-dynamics-shape", action="store_true",
                        help="T8.2: build a 27-node good-shape lookup stack and sample "
                             "(log B, log uc, log C) jointly with the MLR surface.")
    parser.add_argument("--target-accept", type=float, default=0.9)
    args = parser.parse_args()
    if args.cmd_anchored_only and args.both_cmd_anchored_only:
        parser.error(
            "--cmd-anchored-only and --both-cmd-anchored-only are mutually exclusive."
        )
    args.mlr_knot_x = np.asarray(args.mlr_knot_x, dtype=float)
    args.mlr_knot_z = np.asarray(args.mlr_knot_z, dtype=float)
    expected_x_knots = 8 + int(args.mlr_degree_x) + 1
    expected_z_knots = 4 + int(args.mlr_degree_z) + 1
    if args.mlr_knot_x.ndim != 1 or args.mlr_knot_x.size != expected_x_knots or np.any(~np.isfinite(args.mlr_knot_x)) or np.any(np.diff(args.mlr_knot_x) < 0):
        parser.error(
            f"--mlr-knot-x must contain {expected_x_knots} finite nondecreasing values "
            "for eight x-direction basis functions."
        )
    if args.mlr_knot_z.ndim != 1 or args.mlr_knot_z.size != expected_z_knots or np.any(~np.isfinite(args.mlr_knot_z)) or np.any(np.diff(args.mlr_knot_z) < 0):
        parser.error(
            f"--mlr-knot-z must contain {expected_z_knots} finite nondecreasing values "
            "for four Z-direction basis functions."
        )
    if not args.mlr_knot_x[0] <= 4.67 <= args.mlr_knot_x[-1]:
        parser.error("--mlr-knot-x must contain the solar anchor M_G=4.67.")
    if args.mlr_degree_x < 1 or args.mlr_degree_z < 1:
        parser.error("B-spline degrees must be positive.")
    if min(args.mlr_tau_d, args.mlr_tau_x, args.mlr_tau_z, args.mlr_tau_xz, args.mlr_solar_sigma) <= 0:
        parser.error("MLR prior and solar-anchor scales must be positive.")
    if args.quick:
        if args.max_systems is None:
            args.max_systems = 2000
        args.warmup = 200 if args.warmup is None else args.warmup
        args.samples = 300 if args.samples is None else args.samples
        args.chains = 1 if args.chains is None else args.chains
        args.lookup_mass_points = (
            512 if args.lookup_mass_points is None else args.lookup_mass_points
        )
    else:
        args.warmup = 1000 if args.warmup is None else args.warmup
        args.samples = 1000 if args.samples is None else args.samples
        args.chains = 4 if args.chains is None else args.chains
        args.lookup_mass_points = (
            1024 if args.lookup_mass_points is None else args.lookup_mass_points
        )
    if min(
        args.warmup,
        args.samples,
        args.chains,
        args.posterior_grid_draws,
        args.lookup_mass_points,
        args.lookup_velocity_nodes,
        args.lookup_system_chunk,
    ) < 1:
        parser.error("Sampling and lookup sizes must be positive.")
    if not (0 < args.lookup_sqrt_mass_min < args.lookup_sqrt_mass_max):
        parser.error("Require 0 < --lookup-sqrt-mass-min < --lookup-sqrt-mass-max.")
    if not args.outlier_sensitivity and (
        args.outlier_u0 != 40.0 or args.outlier_sigma != 13.0
    ):
        parser.error("T8 baseline fixes --outlier-u0=40 and --outlier-sigma=13; add --outlier-sensitivity for experiments.")
    if (
        args.lookup_sigma_extent <= 0
        or args.lookup_convergence_tolerance <= 0
        or args.outlier_sigma <= 0
        or not np.isfinite(args.outlier_u0)
        or not np.isfinite(args.outlier_sigma)
    ):
        parser.error("Lookup sigma extent and outlier sigma must be positive.")
    if args.outlier_sensitivity:
        if args.stage not in {"lookup", "mlr", "plot"}:
            parser.error("Outlier sensitivity reuses T8 calibration: use --stage lookup or mlr (or plot).")
        if args.stage != "plot" and args.metallicity_posterior is None:
            parser.error("Outlier sensitivity requires --metallicity-posterior from the frozen T8 baseline.")
        output = args.output_dir.resolve()
        frozen = REPO_ROOT / "results" / "hierarchical_metallicity_t8_20260913"
        protected = [frozen.resolve()]
        if args.metallicity_posterior is not None:
            protected.append(args.metallicity_posterior.resolve().parent)
        if any(output == parent or parent in output.parents for parent in protected) or (output / "calibration_model.json").exists():
            parser.error("Experimental output must be separate from the frozen calibration/baseline directory.")
        saved_model = output / "mlr_model.json"
        if saved_model.exists():
            previous = json.loads(saved_model.read_text())
            if not previous.get("experimental_outlier_sensitivity") or previous.get("outlier_shape", {}).get("mu") != args.outlier_u0 or previous.get("outlier_shape", {}).get("sigma") != args.outlier_sigma:
                parser.error("Output directory already contains a different MLR model; choose a new directory.")
    shape_overrides = {
        "B": args.good_shape_b, "uc": args.good_shape_uc, "C": args.good_shape_c,
    }
    if args.sample_dynamics_shape and any(v is not None for v in shape_overrides.values()):
        parser.error("--sample-dynamics-shape samples the shape; drop --good-shape-b/uc/c.")
    if args.sample_dynamics_shape and (args.cmd_anchored_only or args.both_cmd_anchored_only):
        parser.error("Shape sampling does not support the cmd-anchored subset yet.")
    if args.sample_dynamics_shape and args.stage == "plot":
        parser.error("--sample-dynamics-shape requires lookup and MLR stages (use --stage all, lookup+mlr).")
    if any(v is not None for v in shape_overrides.values()):
        if not args.shape_sensitivity:
            parser.error("Non-default good-shape constants require --shape-sensitivity.")
        if args.stage == "plot":
            parser.error("Shape sensitivity rebuilds the lookup: --stage lookup or mlr (or all).")
        output = args.output_dir.resolve()
        frozen = REPO_ROOT / "results" / "hierarchical_metallicity_t8_20260913"
        if output == frozen.resolve() or frozen.resolve() in output.parents or (output / "calibration_model.json").exists():
            parser.error("Shape-sensitivity output must be a fresh directory separate from the frozen baseline.")
        import binary_masses.hierarchical_metallicity as _hm
        overrides = {k: v for k, v in shape_overrides.items() if v is not None}
        _hm.set_good_shape_constants(**overrides)
        print(
            "Good-shape sensitivity override: "
            f"A={_hm.RICE_GOOD_A:.4g} B={_hm.RICE_GOOD_B:.4g} "
            f"uc={_hm.RICE_GOOD_UC:.4g} C={_hm.RICE_GOOD_C:.4g} "
            f"(raw support integral {_hm.RICE_GOOD_BASIS_RAW_INTEGRAL:.10f})"
        )
    return args


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    mass_surface, color_surface = build_surfaces()
    if args.stage == "plot":
        regenerate_mlr_plots(mass_surface, args.output_dir)
        print(f"Done. Outputs: {args.output_dir}")
        return
    truth = None
    mock_absg_clipped = False
    if args.mock:
        arrays, truth = simulate_mock_dataset(
            color_surface, mass_surface, n_systems=args.mock_systems, seed=args.seed
        )
        mock_absg_clipped = _ensure_mock_warm_cmd_coverage(arrays, color_surface)
        if mock_absg_clipped:
            print("Mock set had no 4000 K member; clipped mock magnitudes to the warm CMD support.")
        (args.output_dir / "mock_truth.json").write_text(
            json.dumps({**truth, "absg_clipped_for_t8_warm_cmd": mock_absg_clipped}, indent=2), encoding="utf-8"
        )
    else:
        arrays = load_real_data(args.data)

    posterior_path = (
        args.metallicity_posterior
        if args.metallicity_posterior is not None
        else args.output_dir / T8_POSTERIOR_NAME
    )
    if args.stage in {"calibration", "all"}:
        arrays = filter_data(
            arrays, max_systems=args.max_systems, seed=args.seed
        )
        posterior = run_calibration(
            arrays, color_surface, args.output_dir, args, truth
        )
    elif args.stage == "lookup":
        arrays = filter_data(
            arrays, max_systems=args.max_systems, seed=args.seed
        )
        posterior = None
    else:
        posterior = MetallicityPosteriorGrid.load(posterior_path)
        arrays = filter_data(
            arrays,
            max_systems=None,
            seed=args.seed,
            fixed_rows=posterior.row_indices,
        )

    if args.stage in {"mlr", "all"}:
        _require_t8_posterior_metadata(
            posterior, mock=bool(args.mock), arrays=arrays
        )

    lookup_path = (
        args.dynamics_lookup
        if args.dynamics_lookup is not None
        else args.output_dir / T8_LOOKUP_NAME
    )
    stack_path = args.output_dir / T8_SHAPESTACK_NAME
    dynamics_shape_stack = None
    dynamics_lookup = None
    if args.stage in {"lookup", "all"}:
        if args.sample_dynamics_shape:
            dynamics_shape_stack = run_shape_stack(arrays, args.output_dir, args)
        else:
            dynamics_lookup = run_lookup(arrays, args.output_dir, args)
    elif args.stage == "mlr":
        if args.sample_dynamics_shape:
            dynamics_shape_stack = _hm.DynamicsLikelihoodShapeStack.load(stack_path)
            dynamics_shape_stack.validate_for(
                row_indices=arrays["row_indices"], u=arrays["u"], u_sigma=arrays["u_sigma"]
            )
        else:
            dynamics_lookup = DynamicsLikelihoodLookup.load(lookup_path)
            _require_t8_lookup_metadata(dynamics_lookup, args=args)
            dynamics_lookup.validate_for(
                row_indices=arrays["row_indices"], u=arrays["u"], u_sigma=arrays["u_sigma"]
            )

    if args.stage in {"mlr", "all"}:
        if args.cmd_anchored_only or args.both_cmd_anchored_only:
            arrays, posterior, dynamics_lookup = select_cmd_anchored_systems(
                arrays,
                posterior,
                dynamics_lookup,
                color_surface,
                require_both=args.both_cmd_anchored_only,
            )
        run_mlr(
            arrays,
            posterior,
            dynamics_lookup,
            mass_surface,
            args.output_dir,
            args,
            truth,
            dynamics_shape_stack=dynamics_shape_stack,
        )
    print(f"Done. Outputs: {args.output_dir}")


if __name__ == "__main__":
    main()
