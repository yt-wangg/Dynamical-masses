#!/usr/bin/env python3
"""Compare the frozen T8 MLR with Mann et al. (2019) in 2MASS Ks.

The conversion follows the *same PARSEC evolutionary points* as the local
T8 Gaia table. CMD 3.8 supplies Ks at those points; its Gaia magnitudes are
never substituted for the Gaia magnitudes used to build T8.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.io import fits

ROOT = Path(__file__).resolve().parents[2]
WORKSPACE = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))
from binary_masses.hierarchical_metallicity import (  # noqa: E402
    IsochroneMassSurfaceModel,
    MonotoneTensorSplineMLR,
)

Z_SLICES = (-0.6, 0.0, 0.4)
T8_DIR = ROOT / "results" / "hierarchical_metallicity_t8_20260913"
OUT_DIR = ROOT / "results" / "t8_mann2019_ks_20260916"
GAIA_CSV = ROOT / "data" / "PARSEC_logAge_6to10_0p5_MH_n1to0p6_0p2.csv"
CMD_TABLE = OUT_DIR / "parsec_cmd38_gaia_dr2_2mass_targets.dat"
MANN_POST = WORKSPACE / "M_-M_K-" / "resources" / "Mk-M_7_trim.fits"
MANN_TABLE = WORKSPACE / "Model" / "mann2019" / "Table1.mrt"
MANN_COMPONENTS = WORKSPACE / "Model" / "mann2019" / "mann2019_component_masses.csv"
HWANG_CSV = ROOT / "src" / "examples" / "hwang_mlr" / "hwang2023_mlr.csv"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_t8_model():
    full = IsochroneMassSurfaceModel.from_interpolated_mass_data(
        ROOT / "data" / "interpolated_mass_data", mass_min=0.05
    )
    keep = full.absg_grid_np < 13.5 - 1e-12
    x = np.concatenate([full.absg_grid_np[keep], [13.5]])
    mass = np.column_stack([
        full.mass_grid_np[:, keep],
        [full.mass_from_absg_mh(13.5, z) for z in full.mh_grid_np],
    ])
    surface = IsochroneMassSurfaceModel(x, full.mh_grid_np, mass, mass_min=0.05)
    meta = json.loads((T8_DIR / "mlr_model.json").read_text())
    mlr = MonotoneTensorSplineMLR(
        surface,
        knots_x=np.asarray(meta["knot_x"]),
        knots_z=np.asarray(meta["knot_z"]),
        degree_x=int(meta["degree_x"]),
        degree_z=int(meta["degree_z"]),
        tau_D=float(meta["penalty_scales_dex"]["tau_D"]),
        tau_x=float(meta["penalty_scales_dex"]["tau_x"]),
        tau_Z=float(meta["penalty_scales_dex"]["tau_Z"]),
        tau_xZ=float(meta["penalty_scales_dex"]["tau_xZ"]),
        solar_anchor_mean=float(meta["solar_anchor"]["mean_log10_mass"]),
        solar_anchor_sigma=float(meta["solar_anchor"]["sigma_log10_mass"]),
    )
    with np.load(T8_DIR / "mlr_mcmc_t8.npz") as saved:
        names = ("c0", "a", "b", "r", "log_lambda_x", "log_lambda_z")
        samples = {
            name: saved[f"posterior__{name}"].reshape(
                (-1,) + saved[f"posterior__{name}"].shape[2:]
            )
            for name in names
        }
    return mlr, samples


def t8_draws(mlr, samples, mg, z, indices):
    result = np.empty((len(indices), len(mg)))
    names = ("c0", "a", "b", "r", "log_lambda_x", "log_lambda_z")
    for j, index in enumerate(indices):
        params = {name: samples[name][index] for name in names}
        result[j] = 10 ** np.asarray(mlr.g_from_raw(mg, z, params))
    if not np.all(np.isfinite(result)) or np.any(np.diff(result, axis=1) > 2e-10):
        raise ValueError("T8 draws are nonfinite or nonmonotone in M_G")
    return result


def band_map(gaia: pd.DataFrame, cmd: pd.DataFrame, z: float):
    # The T8 source contains one old-age grid point satisfying 9.3<logAge<10.
    # Stay on the monotone main sequence through the solar-mass region; the
    # subsequent turnoff/evolved points are rejected by the monotonicity checks.
    old = gaia[np.isclose(gaia.MH, z) & np.isclose(gaia.logAge, 9.6)
               & (gaia.Mini > 0.08) & (gaia.Mini < 1.10)
               & (gaia.label == 1)].sort_values("Mini")
    new = cmd[np.isclose(cmd.MH, z) & np.isclose(cmd.logAge, 9.6)
              & (cmd.Mini > 0.08) & (cmd.Mini < 1.10)
              & (cmd.label == 1)].sort_values("Mini")
    old = old[(old.Mini >= new.Mini.min()) & (old.Mini <= new.Mini.max())].copy()
    if len(old) < 10:
        raise ValueError(f"Too few paired PARSEC points at [M/H]={z}")
    # Interpolate only across initial mass, a common physical coordinate.
    old["Ksmag"] = np.interp(old.Mini, new.Mini, new.Ksmag)
    logl = np.interp(old.Mini, new.Mini, new.logL)
    logte = np.interp(old.Mini, new.Mini, new.logTe)
    max_logl = float(np.max(np.abs(old.logL - logl)))
    max_logte = float(np.max(np.abs(old.logTe - logte)))
    if max_logl > 1e-3 or max_logte > 1e-3:
        raise ValueError(f"Physical PARSEC points fail to match at [M/H]={z}")
    if np.any(np.diff(old.Gmag) >= 0) or np.any(np.diff(old.Ksmag) >= 0):
        raise ValueError(f"Gaia-to-Ks map is not one-to-one at [M/H]={z}")
    return old, {"n_points": len(old), "max_abs_logL_difference": max_logl,
                 "max_abs_logTe_difference": max_logte,
                 "mg_range": [float(old.Gmag.min()), float(old.Gmag.max())],
                 "mks_range": [float(old.Ksmag.min()), float(old.Ksmag.max())]}


def mann_draws(coefficients, mks, scatter_normal=None):
    offset = mks - 7.5
    mass = 10 ** np.polynomial.polynomial.polyval(offset, coefficients[:, :6].T)
    if scatter_normal is not None:
        mass = mass * (1 + np.exp(coefficients[:, 6, None]) * scatter_normal[:, None])
    return mass


def parsec_mass_curve(saved, mg, z):
    """Interpolate the raw PARSEC mass surface saved with the T8 baseline."""
    mass_at_mg = np.array([
        np.interp(mg, saved["absg_grid"], row)
        for row in saved["parsec_mass"]
    ])
    return np.asarray([
        np.interp(z, saved["z_grid"], column)
        for column in mass_at_mg.T
    ])


def mann_component_scatter():
    """Reconstruct Figure 9-style component points from the published systems.

    The component masses are allocated from each system's dynamical Mtot using
    the local Mann-relation mass ratio (mass_med).  They are therefore not
    independent dynamical component-mass measurements.
    """
    components = pd.read_csv(MANN_COMPONENTS)
    if len(components) != 124 or components.groupby("row_index").ngroups != 62:
        raise ValueError("Mann component table must contain 124 rows in 62 systems")
    if components.groupby("row_index").size().ne(2).any():
        raise ValueError("Each Mann system must have exactly two components")
    rows = []
    for line in MANN_TABLE.read_text().splitlines()[64:]:
        if not line.strip() or line.startswith("-"):
            continue
        name = line[2:12].strip()
        tail = line[15:].split()
        cursor = 8
        if tail[cursor] in {"b", "e"}:
            cursor += 1
        cursor += 2
        mtot = float(tail[cursor])
        e_mtot = float(tail[cursor + 1])
        rows.append((name, mtot, e_mtot))
    systems = pd.DataFrame(rows, columns=["system_id", "mtot_msun", "mtot_error_msun"])
    if len(systems) != 62:
        raise ValueError(f"Expected 62 Mann Table1 systems, found {len(systems)}")
    components = components.merge(systems, on="system_id", how="left", validate="many_to_one")
    if components.mtot_msun.isna().any():
        missing = components.loc[components.mtot_msun.isna(), "system_id"].unique()
        raise ValueError(f"Unmatched Mann systems: {missing.tolist()}")
    weights = components.groupby("row_index")["mass_med"].transform("sum")
    components["mass_allocated_msun"] = components.mtot_msun * components.mass_med / weights
    components["mtot_fractional_error"] = components.mtot_error_msun / components.mtot_msun
    components["mass_allocated_error_msun"] = (
        components.mass_allocated_msun * components.mtot_fractional_error
    )
    components["mks_mag"] = components.Ks - 5 * np.log10(components.dist_pc / 10)
    sums = components.groupby("row_index").mass_allocated_msun.sum()
    if not np.allclose(sums.to_numpy(), components.drop_duplicates("row_index").mtot_msun.to_numpy(), rtol=0, atol=2e-12):
        raise ValueError("Allocated Mann component masses do not sum to Mtot")
    error_columns = ["mtot_error_msun", "mtot_fractional_error", "mass_allocated_error_msun"]
    if not np.all(np.isfinite(components[["mks_mag", "mass_allocated_msun", *error_columns]])):
        raise ValueError("Mann component scatter contains nonfinite values")
    if np.any(components[error_columns].to_numpy() < 0):
        raise ValueError("Mann component scatter contains negative mass errors")
    expected_error = components.mass_allocated_msun * components.mtot_error_msun / components.mtot_msun
    if not np.allclose(components.mass_allocated_error_msun, expected_error, rtol=0, atol=1e-15):
        raise ValueError("Allocated component mass errors do not match the requested propagation")
    return components


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    gaia = pd.read_csv(GAIA_CSV)
    columns = ("Zini", "MH", "logAge", "Mini", "int_IMF", "Mass", "logL",
               "logTe", "logg", "label", "mbolmag", "Gmag", "G_BPmag",
               "G_RPmag", "B_Tmag", "V_Tmag", "Jmag", "Hmag", "Ksmag")
    cmd = pd.read_csv(CMD_TABLE, comment="#", sep=r"\s+", header=None, names=columns)
    mlr, samples = build_t8_model()
    saved = np.load(T8_DIR / "mlr_derived_grid_t8.npz")
    indices = np.linspace(0, len(samples["c0"]) - 1, 512, dtype=int)
    zero_check = t8_draws(mlr, samples, saved["absg_grid"], 0.0, indices)
    zero_error = float(np.max(np.abs(zero_check - saved["mass"][:, 2, :])))
    if zero_error > 2e-10:
        raise ValueError(f"Frozen T8 draw validation failed: {zero_error}")
    z0_parsec_error = float(np.max(np.abs(
        parsec_mass_curve(saved, saved["absg_grid"], 0.0)
        - saved["parsec_mass"][np.flatnonzero(np.isclose(saved["z_grid"], 0.0))[0]]
    )))
    if z0_parsec_error > 2e-12:
        raise ValueError(f"Frozen PARSEC mass validation failed: {z0_parsec_error}")

    mann_all = np.asarray(fits.getdata(MANN_POST), dtype=float)
    mann_indices = np.linspace(0, len(mann_all) - 1, 4096, dtype=int)
    mann = mann_all[mann_indices]
    mann_scatter_normal = np.random.default_rng(20260916).normal(size=len(mann))
    records = []
    summary = {}
    colors = {-.6: "#315da8", 0.0: "#d06a2d", .4: "#278368"}
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(8.5, 8), sharex=False,
                                      gridspec_kw={"height_ratios": [2.1, 1]},
                                      constrained_layout=True)
    plot_lo, plot_hi = 3.0, 11.0
    mann_lo, mann_hi = 4.0, 11.0
    mann_x = np.linspace(mann_lo, mann_hi, 161)
    mann_m = mann_draws(mann, mann_x, mann_scatter_normal)
    mann_q = np.percentile(mann_m, [16, 50, 84], axis=0)
    mann_components = mann_component_scatter()
    sun_artist, = top.plot(3.27, 1.0, marker="*", ms=13, color="#e0a400",
                           markeredgecolor="black", markeredgewidth=.7,
                           linestyle="none", label=r"Sun ($1\,M_\odot$)", zorder=8)
    top.errorbar(mann_components.mks_mag, mann_components.mass_allocated_msun,
                 yerr=mann_components.mass_allocated_error_msun, fmt="none",
                 ecolor="0.25", elinewidth=.55, capsize=1.4, alpha=.55, zorder=3)
    mann_artist = top.scatter(mann_components.mks_mag, mann_components.mass_allocated_msun,
                              s=19, color="black", alpha=.70, edgecolors="none",
                              label="Mann et al. (2019) stars", zorder=4)
    mann_curve, = top.plot(mann_x, mann_q[1], color="black", lw=1.5,
                           label="Mann et al. (2019) polynomial", zorder=3)
    this_handles, parsec_handles = [], []
    for z in Z_SLICES:
        mapping, info = band_map(gaia, cmd, z)
        # Strict common support for the frozen T8 M_G domain and band map.
        lo = max(plot_lo, float(mapping.Ksmag.min()))
        hi = min(plot_hi, float(mapping.Ksmag.max()))
        ks_grid = np.linspace(lo, hi, 151)
        mg_grid = np.interp(ks_grid, mapping.Ksmag.iloc[::-1], mapping.Gmag.iloc[::-1])
        if mg_grid.min() < 3.5 or mg_grid.max() > 13.5:
            valid = (mg_grid >= 3.5) & (mg_grid <= 13.5)
            ks_grid, mg_grid = ks_grid[valid], mg_grid[valid]
        t8 = t8_draws(mlr, samples, mg_grid, z, indices)
        mm = mann_draws(mann, ks_grid, mann_scatter_normal)
        tq = np.percentile(t8, [16, 50, 84], axis=0)
        mq = np.percentile(mm, [16, 50, 84], axis=0)
        parsec_mass = np.interp(ks_grid, mapping.Ksmag.iloc[::-1], mapping.Mass.iloc[::-1])
        if not np.all(np.isfinite(parsec_mass)):
            raise ValueError(f"Nonfinite raw PARSEC mass at [M/H]={z}")
        relation_difference = np.percentile(
            100 * (t8 / parsec_mass - 1), [16, 50, 84], axis=0
        )
        color = colors[z]
        this_line, = top.plot(ks_grid, tq[1], color=color, lw=1.9,
                              label=rf"This work $[M/H]={z:+.1f}$")
        parsec_ks = np.linspace(max(plot_lo, float(mapping.Ksmag.min())),
                                min(plot_hi, float(mapping.Ksmag.max())), 201)
        parsec_full = np.interp(parsec_ks, mapping.Ksmag.iloc[::-1],
                                mapping.Mass.iloc[::-1])
        parsec_line, = top.plot(parsec_ks, parsec_full, color=color, lw=1.4, ls="--",
                                label=rf"PARSEC $[M/H]={z:+.1f}$")
        this_handles.append(this_line)
        parsec_handles.append(parsec_line)
        top.fill_between(ks_grid, tq[0], tq[2], color=color, alpha=.16)
        bottom.plot(ks_grid, relation_difference[1], color=color, lw=1.9)
        for j in range(len(ks_grid)):
            records.append({"mh_dex": z, "mks_mag": ks_grid[j], "mg_mag": mg_grid[j],
                            "t8_mass_p16_msun": tq[0, j], "t8_mass_p50_msun": tq[1, j],
                            "t8_mass_p84_msun": tq[2, j], "mann_mass_p16_msun": mq[0, j],
                            "mann_mass_p50_msun": mq[1, j], "mann_mass_p84_msun": mq[2, j],
                            "parsec_mass_msun": parsec_mass[j],
                            "t8_relative_to_parsec_p16_percent": relation_difference[0, j],
                            "t8_relative_to_parsec_p50_percent": relation_difference[1, j],
                            "t8_relative_to_parsec_p84_percent": relation_difference[2, j]})
        info.update({"compared_mks_range": [float(ks_grid[0]), float(ks_grid[-1])],
                     "compared_t8_mass_range": [float(tq[1, -1]), float(tq[1, 0])],
                     "t8_relative_to_parsec_percent_range": [float(relation_difference[1].min()), float(relation_difference[1].max())],
                     "t8_relative_to_parsec_percent_at_mks_6": float(np.interp(6, ks_grid, relation_difference[1]))
                     if ks_grid[0] <= 6 <= ks_grid[-1] else None})
        summary[f"{z:+.1f}"] = info
    bottom.axhline(0, color="0.4", lw=.9)
    # Literature relations use the solar-metallicity PARSEC reference. Hwang
    # and Xiong are clipped to their own support; Mann remains valid only for
    # its published 4--11 mag interval.
    hwang = pd.read_csv(HWANG_CSV)
    solar_map, solar_info = band_map(gaia, cmd, 0.0)
    mann_valid = (mann_x >= solar_map.Ksmag.min()) & (mann_x <= solar_map.Ksmag.max())
    mann_ref_x = mann_x[mann_valid]
    solar_parsec_mann = np.interp(mann_ref_x, solar_map.Ksmag.iloc[::-1],
                                  solar_map.Mass.iloc[::-1])
    mann_difference = 100 * (mann_m[:, mann_valid] / solar_parsec_mann - 1)
    mann_diff_q = np.percentile(mann_difference, [16, 50, 84], axis=0)
    bottom.plot(mann_ref_x, mann_diff_q[1], color="black", lw=1.5)
    literature_records = [
        {"relation": "Mann2019", "mks_mag": float(mann_ref_x[j]),
         "mass_msun": float(mann_q[1, mann_valid][j]), "mass_error_msun": None,
         "parsec_mass_msun": float(solar_parsec_mann[j]),
         "relative_to_parsec_percent": float(mann_diff_q[1, j])}
        for j in range(len(mann_ref_x))
    ]
    summary["mann2019"] = {
        "mks_range": [float(mann_ref_x.min()), float(mann_ref_x.max())],
        "difference_percent_range": [float(mann_diff_q[1].min()),
                                     float(mann_diff_q[1].max())],
    }
    hwang_map_valid = (
        (hwang.M_G_ridge >= solar_map.Gmag.min())
        & (hwang.M_G_ridge <= solar_map.Gmag.max())
    )
    hwang_mks = np.full(len(hwang), np.nan)
    hwang_mks[hwang_map_valid] = np.interp(
        hwang.loc[hwang_map_valid, "M_G_ridge"],
        solar_map.Gmag.iloc[::-1],
        solar_map.Ksmag.iloc[::-1],
    )
    hwang_parsec = np.interp(hwang_mks, solar_map.Ksmag.iloc[::-1],
                             solar_map.Mass.iloc[::-1])
    hwang_valid = (
        (hwang.mass_nn > 0)
        & (hwang.n_stars > 0)
        & hwang_map_valid
        & (hwang_mks >= plot_lo) & (hwang_mks <= plot_hi)
    )
    hwang_artist, = top.plot(
        hwang_mks[hwang_valid], hwang.mass_nn[hwang_valid],
        color="#7d3c98", lw=1.8, ls="-.", label="Hwang et al. (2023) MS ridge",
    )
    top.fill_between(
        hwang_mks[hwang_valid],
        (hwang.mass_nn - hwang.mass_nn_err)[hwang_valid],
        (hwang.mass_nn + hwang.mass_nn_err)[hwang_valid],
        color="#7d3c98", alpha=.14,
    )
    hwang_difference = np.asarray(100 * (hwang.mass_nn / hwang_parsec - 1))
    bottom.plot(hwang_mks[hwang_valid], hwang_difference[hwang_valid],
                color="#7d3c98", lw=1.8, ls="-.")
    hwang_order = np.argsort(hwang_mks[hwang_valid])
    hwang_x = hwang_mks[hwang_valid][hwang_order]
    hwang_y = hwang_difference[hwang_valid][hwang_order]
    summary["hwang2023"] = {
        "mks_range": [float(hwang_x.min()), float(hwang_x.max())],
        "difference_percent_range": [float(hwang_y.min()), float(hwang_y.max())],
        "difference_percent_at_mks_6": float(np.interp(6, hwang_x, hwang_y))
        if hwang_x[0] <= 6 <= hwang_x[-1] else None,
        "difference_percent_at_mks_7": float(np.interp(7, hwang_x, hwang_y))
        if hwang_x[0] <= 7 <= hwang_x[-1] else None,
    }
    for j in np.flatnonzero(hwang_valid):
        literature_records.append({
            "relation": "Hwang2023", "mks_mag": float(hwang_mks[j]),
            "mass_msun": float(hwang.mass_nn.iloc[j]),
            "mass_error_msun": float(hwang.mass_nn_err.iloc[j]),
            "parsec_mass_msun": float(hwang_parsec[j]),
            "relative_to_parsec_percent": float(hwang_difference[j]),
        })
    def xiong_log_l(mass, z):
        x = np.log10(mass)
        return 0.066 + 4.141 * x + 0.314 * x**2 - 0.755 * x**3 + 0.189 * x**4 - 0.245 * z

    xiong_mass_grid = np.linspace(0.075, 1.20, 4000)
    xiong_logl_grid = xiong_log_l(xiong_mass_grid, 0.0)
    xiong_logl_valid = (
        (solar_map.logL >= xiong_logl_grid[0])
        & (solar_map.logL <= xiong_logl_grid[-1])
    )
    xiong_mass = np.full(len(solar_map), np.nan)
    xiong_mass[xiong_logl_valid] = np.interp(
        solar_map.loc[xiong_logl_valid, "logL"], xiong_logl_grid, xiong_mass_grid
    )
    xiong_valid = (
        xiong_logl_valid
        & (solar_map.Ksmag >= plot_lo) & (solar_map.Ksmag <= plot_hi)
    )
    xiong_artist, = top.plot(
        solar_map.Ksmag[xiong_valid], xiong_mass[xiong_valid],
        color="#c0392b", lw=1.8, ls=":", label="Xiong et al. (2023) Eq. (5), [M/H]=0",
    )
    xiong_ks = solar_map.Ksmag[xiong_valid].to_numpy()
    xiong_parsec = np.asarray(solar_map.Mass[xiong_valid])
    xiong_difference = 100 * (xiong_mass[xiong_valid] / xiong_parsec - 1)
    bottom.plot(xiong_ks, xiong_difference, color="#c0392b", lw=1.8, ls=":")
    xiong_order = np.argsort(xiong_ks)
    xiong_x = xiong_ks[xiong_order]
    xiong_y = xiong_difference[xiong_order]
    summary["xiong2023_eq5"] = {
        "mks_range": [float(xiong_x.min()), float(xiong_x.max())],
        "difference_percent_range": [float(xiong_y.min()), float(xiong_y.max())],
        "difference_percent_at_mks_6": float(np.interp(6, xiong_x, xiong_y))
        if xiong_x[0] <= 6 <= xiong_x[-1] else None,
        "difference_percent_at_mks_7": float(np.interp(7, xiong_x, xiong_y))
        if xiong_x[0] <= 7 <= xiong_x[-1] else None,
    }
    for j in range(len(xiong_x)):
        original = xiong_order[j]
        literature_records.append({
            "relation": "Xiong2023_Eq5", "mks_mag": float(xiong_x[j]),
            "mass_msun": float(xiong_mass[xiong_valid][original]),
            "mass_error_msun": None,
            "parsec_mass_msun": float(xiong_parsec[original]),
            "relative_to_parsec_percent": float(xiong_y[j]),
        })
    # Each comparison curve remains restricted to its own valid support.
    top.set(ylabel=r"Mass ($M_\odot$)", title="This work and literature relations in 2MASS $K_s$")
    bottom.set(xlabel=r"$M_{K_s}$ (mag)", ylabel="Relative to PARSEC (%)")
    reference_handles = [sun_artist, mann_artist, mann_curve, hwang_artist, xiong_artist]
    reference_legend = top.legend(reference_handles,
                                  [h.get_label() for h in reference_handles],
                                  frameon=False, ncol=2, loc="upper right",
                                  bbox_to_anchor=(1.0, 1.0))
    top.add_artist(reference_legend)
    # Matplotlib fills legend columns from top to bottom.  Supplying all
    # This-work handles followed by all PARSEC handles therefore makes two
    # explicit columns, with matching metallicities aligned by row.
    relation_handles = [*this_handles, *parsec_handles]
    top.legend(relation_handles, [h.get_label() for h in relation_handles],
               frameon=False, ncol=2, loc="upper right",
               bbox_to_anchor=(1.0, .72))
    for axis in (top, bottom): axis.grid(alpha=.2)
    top.text(0.01, 0.02,
             r"Model-dependent solar conversion: PARSEC [M/H]=0, logAge=9.6",
             transform=top.transAxes, fontsize=8.5, color="0.25")
    top.set_xlim(plot_lo, plot_hi)
    top.set_ylim(0.0, 1.25)
    bottom.set_xlim(plot_lo, plot_hi)
    fig.savefig(OUT_DIR / "t8_vs_mann2019_ks_with_literature_v3.png", dpi=200)
    fig.savefig(OUT_DIR / "t8_vs_mann2019_ks_with_literature_v3.pdf")
    plt.close(fig)
    mann_components.to_csv(OUT_DIR / "mann_component_scatter_with_errorbars.csv", index=False)
    pd.DataFrame(literature_records).to_csv(
        OUT_DIR / "literature_comparison_v3.csv", index=False
    )
    with (OUT_DIR / "comparison_with_parsec_and_literature_v3.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    metadata = {
        "t8_baseline": "T8_BASELINE_20260913.md", "t8_zero_slice_max_abs_mass_error": zero_error,
        "parsec_zero_slice_max_abs_mass_error": z0_parsec_error,
        "t8_draws": len(indices), "mann_posterior_draws": len(mann),
        "conversion": "At logAge=9.6 and fixed [M/H], pair PARSEC label=1 main-sequence CMD 3.8 Ks with the original T8 Gaia rows through Mini, then invert their one-to-one M_G to M_Ks curve. The paired Mini cutoff is 1.10 to avoid the turnoff.",
        "conversion_note": "The T8 Gaia table and CMD 3.8 Gaia passband differ; only their identical physical evolutionary coordinates are paired. CMD 3.8 Gaia magnitudes are never used in the conversion.",
        "cmd_source": "https://stev.oapd.inaf.it/cgi-bin/cmd_3.8",
        "cmd_release": "PARSEC v1.2S; Gaia DR2 + Tycho2 + 2MASS; OBC bolometric corrections; CMD 3.8 header",
        "mann_source": "https://github.com/awmann/M_-M_K-; Mk-M_7_trim.fits; metallicity-independent fifth-order relation",
        "mann_component_scatter": "124 Figure 9-style component points; M_Ks=Ks-5log10(dist_pc/10). Each system Mtot from Model/mann2019/Table1.mrt is allocated by the pair's mass_med ratio from mann2019_component_masses.csv.",
        "mann_component_scatter_limitation": "These are allocated component masses from system dynamical Mtot, not 124 independent dynamical measurements.",
        "mann_component_errorbars": "Vertical error bars propagate only each system's dynamical total-mass uncertainty e_Mtot to both allocated component masses, using sigma_component = mass_allocated * (e_Mtot / Mtot) at fixed mass_med allocation ratio. Mass-ratio allocation uncertainty is excluded.",
        "mann_formal_mks_range": [4.0, 11.0],
        "mann_recommended_mks_range": [4.5, 10.5],
        "mann_uncertainty": "Coefficient posterior plus the fitted per-star fractional intrinsic scatter; one seeded normal deviate per posterior draw is shared across M_Ks for a smooth pointwise interval.",
        "t8_uncertainty": "Saved T8 posterior draws; PARSEC band-conversion uncertainty is not included.",
        "parsec_overlay": "Dashed curves and lower-panel references use Mass from the paired label=1 PARSEC logAge=9.6 rows, interpolated in M_Ks. This direct isochrone interpolation extends farther in M_Ks than the frozen baseline's saved M_G mass grid. The lower panel shows 100(M_relation/M_PARSEC-1) percent. This work uses matching [M/H] PARSEC; Mann, Hwang, and Xiong use solar [M/H]=0 PARSEC.",
        "literature_overlays": {
            "hwang2023": "Hwang et al. (2023) official NN evaluated on a locally reconstructed main-sequence ridge in hwang2023_mlr.csv. The ridge was extracted by src/examples/hwang_mlr/hwang2023_mlr.py from the released per-star mass table; M_G ridge mapped to M_Ks using the model-dependent PARSEC [M/H]=0, logAge=9.6 band map.",
            "hwang_uncertainty": "mass_nn_err is the dropout-sampling standard deviation from the official NN evaluation, not a total empirical uncertainty.",
            "xiong2023_eq5": "Xiong et al. (2023; arXiv preprint 2022) Eq. (5) empirical bolometric logL(M,[M/H]) fit, evaluated at [M/H]=0 and inverted to M_Ks using the model-dependent PARSEC [M/H]=0, logAge=9.6 (logL, Ksmag) curve.",
            "common_support": "Panels show 3 <= M_Ks <= 11. Mann is plotted only over its formal 4--11 mag support; Hwang and Xiong are clipped to their own mapped/model support.",
        },
        "solar_reference": {"mass_msun": 1.0, "mks_mag": 3.27,
                            "note": "Solar marker is shown at M_Ks=3.27; Mann's formal relation begins at M_Ks=4 and is not extrapolated to the Sun."},
        "input_sha256": {str(path.relative_to(WORKSPACE)): sha256(path) for path in
                         (GAIA_CSV, CMD_TABLE, MANN_POST, MANN_TABLE, MANN_COMPONENTS, HWANG_CSV,
                          T8_DIR / "mlr_mcmc_t8.npz",
                          T8_DIR / "mlr_derived_grid_t8.npz")},
        "slices": summary,
    }
    (OUT_DIR / "metadata_with_literature_v3.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({"output": str(OUT_DIR), "slices": summary,
                      "t8_zero_slice_max_abs_mass_error": zero_error}, indent=2))


if __name__ == "__main__":
    main()
