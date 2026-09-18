#!/usr/bin/env python3
"""Compare T8, PARSEC, Mann+2019, Hwang+2023, and Xiong+2022 in Gaia BP-RP.

BP-RP is the native axis of Hwang+2023 (zero conversion).  T8 and PARSEC are
converted from M_G with the T8 project's own IsochroneColorSurfaceModel (the
same PARSEC color surface used by the calibration stage).  Xiong's logL(M,
[M/H]) is inverted on each slice's PARSEC (logL, Gmag) curve and then given a
color by the same surface; Mann's Ks relation is mapped through the solar
PARSEC band map and the same surface.  The bottom panel shows every relation
relative to PARSEC at the same [M/H] (solar for Mann and Hwang).  Xiong
slices -0.6 and +0.4 are outside its fitted range (-0.58 < [M/H] < +0.07).
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / "src"))
from binary_masses.hierarchical_metallicity import IsochroneColorSurfaceModel  # noqa: E402
from compare_t8_mann2019_ks_local import (  # noqa: E402
    GAIA_CSV,
    T8_DIR,
    band_map,
    build_t8_model,
    mann_draws as mann_mass_draws,
    parsec_mass_curve,
    t8_draws,
)

OUT_DIR = ROOT / "results" / "t8_relations_bprp_quick"
MANN_POST = ROOT.parent / "M_-M_K-" / "resources" / "Mk-M_7_trim.fits"
HWANG_CSV = ROOT / "src" / "examples" / "hwang_mlr" / "hwang2023_mlr.csv"

Z_SLICES = (-0.6, 0.0, 0.4)
XIONG_FITTED_RANGE = (-0.58, 0.07)
MASS_WINDOW = (0.075, 0.70)
MANN_CALIBRATION_MASS_MAX = 0.70  # Mann+2019 is calibrated only up to ~0.7 Msun


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mass-max", type=float, default=MASS_WINDOW[1],
        help="Upper mass of the common comparison window.",
    )
    parser.add_argument(
        "--suffix", type=str, default="",
        help="Filename suffix for the outputs (e.g. '_extended').",
    )
    return parser.parse_args()


def xiong_log_l(mass, z):
    x = np.log10(mass)
    return 0.066 + 4.141 * x + 0.314 * x**2 - 0.755 * x**3 + 0.189 * x**4 - 0.245 * z


def main():
    from astropy.io import fits

    args = parse_args()
    mass_window = (MASS_WINDOW[0], float(args.mass_max))
    suffix = str(args.suffix)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    color_surface = IsochroneColorSurfaceModel.from_parsec_csv(
        GAIA_CSV, absg_grid=np.linspace(3.5, 13.5, 500)
    )
    mlr, samples = build_t8_model()
    saved = np.load(T8_DIR / "mlr_derived_grid_t8.npz")
    indices = np.linspace(0, len(samples["c0"]) - 1, 512, dtype=int)
    mann = np.asarray(fits.getdata(MANN_POST), dtype=float)
    mann = mann[np.linspace(0, len(mann) - 1, 4096, dtype=int)]
    mann_scatter_normal = np.random.default_rng(20260916).normal(size=len(mann))
    hwang = pd.read_csv(HWANG_CSV)

    colors = {-.6: "#315da8", 0.0: "#d06a2d", .4: "#278368"}
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(8.5, 8.5), sharex=True,
                                      gridspec_kw={"height_ratios": [2.1, 1]},
                                      constrained_layout=True)

    def color_of(mg, z):
        mg = np.asarray(mg, dtype=float)
        return np.asarray(
            color_surface.color_from_absg_mh(mg, np.full_like(mg, float(z))),
            dtype=float,
        )

    records = []
    summary = {}
    # Reference M_G grid per slice; all curves are built on (M_G, Z) first and
    # then mapped to BP-RP, so every relation shares one color authority.
    mg_ref = np.linspace(3.5, 13.4, 500)

    # ---- solar-slice PARSEC (reference for Mann and Hwang) -----------------
    solar_map, _ = band_map(*_load_tables(), 0.0)
    parsec_solar = parsec_mass_curve(saved, mg_ref, 0.0)
    color_solar = color_of(mg_ref, 0.0)

    # ---- Mann+2019: Ks -> M_G -> color through solar PARSEC + surface ------
    mann_ks = np.interp(mg_ref, solar_map.Gmag.iloc[::-1], solar_map.Ksmag.iloc[::-1])
    mann_ok = (mann_ks >= 5.5) & (mann_ks <= 9.6) & np.isfinite(color_solar)
    mann_mass = mann_mass_draws(mann, mann_ks[mann_ok], mann_scatter_normal)
    mann_q = np.percentile(mann_mass, [16, 50, 84], axis=0)
    subset_ok = ((mann_q[1] > mass_window[0]) & (mann_q[1] <= MANN_CALIBRATION_MASS_MAX))
    mann_color = color_solar[mann_ok][subset_ok]
    mann_artist, = top.plot(mann_color, mann_q[1][subset_ok], color="black", lw=1.6,
                            label="Mann et al. (2019) $[M/H]\\approx0$")
    top.fill_between(mann_color, mann_q[0][subset_ok], mann_q[2][subset_ok],
                     color="black", alpha=.10)
    mann_diff_q = np.percentile(
        100 * (mann_mass / parsec_solar[mann_ok][None, :] - 1), [16, 50, 84], axis=0
    )
    bottom.plot(mann_color, mann_diff_q[1][subset_ok], color="black", lw=1.6)
    bottom.fill_between(mann_color, mann_diff_q[0][subset_ok],
                        mann_diff_q[2][subset_ok], color="black", alpha=.10)

    # ---- Hwang+2023: native BP-RP ------------------------------------------
    hw_valid = ((hwang.mass_nn > mass_window[0]) & (hwang.mass_nn < mass_window[1])
                & (hwang.n_stars > 0))
    hwang_artist, = top.plot(
        hwang.bp_rp[hw_valid], hwang.mass_nn[hw_valid],
        color="#7d3c98", lw=1.8, ls="-.", label="Hwang et al. (2023) $[M/H]\\approx0$",
    )
    top.fill_between(hwang.bp_rp[hw_valid],
                     (hwang.mass_nn - hwang.mass_nn_err)[hw_valid],
                     (hwang.mass_nn + hwang.mass_nn_err)[hw_valid],
                     color="#7d3c98", alpha=.14)
    parsec_at_hwang = parsec_mass_curve(saved, hwang.M_G_ridge.to_numpy()[hw_valid.to_numpy()], 0.0)
    hwang_diff = 100 * (hwang.mass_nn.to_numpy()[hw_valid.to_numpy()] / parsec_at_hwang - 1)
    bottom.plot(hwang.bp_rp[hw_valid], hwang_diff, color="#7d3c98", lw=1.8, ls="-.")

    top.plot([0.82], [1.0], marker="*", ms=13, color="#e0a400", markeredgecolor="black",
             markeredgewidth=.7, linestyle="none", label=r"Sun ($1\,M_\odot$)", zorder=8)

    gaia_csv, cmd_table = _load_tables()
    this_handles, xiong_handles, parsec_handles = [], [], []
    xiong_grid = np.linspace(mass_window[0], mass_window[1], 4000)
    for z in Z_SLICES:
        color = colors[z]
        slice_map, info = band_map(gaia_csv, cmd_table, z)
        # band_map's Mini cap only limits Xiong placement; T8's own domain is M_G >= 3.5
        mg_grid = np.linspace(3.5, min(13.4, slice_map.Gmag.max()), 400)
        color_grid = color_of(mg_grid, z)
        t8 = t8_draws(mlr, samples, mg_grid, z, indices)
        tq = np.percentile(t8, [16, 50, 84], axis=0)
        parsec = parsec_mass_curve(saved, mg_grid, z)
        valid = (tq[1] > mass_window[0]) & (tq[1] < mass_window[1]) & np.isfinite(parsec) & np.isfinite(color_grid)
        c_grid, t8, tq, parsec = color_grid[valid], t8[:, valid], tq[:, valid], parsec[valid]

        logl_grid = xiong_log_l(xiong_grid, z)
        xiong_mg = slice_map.Gmag.to_numpy()
        xm_points = np.interp(slice_map.logL.to_numpy(), logl_grid, xiong_grid)
        xiong_ok = (slice_map.logL.to_numpy() >= logl_grid[0]) & (slice_map.logL.to_numpy() <= logl_grid[-1])
        xiong_color = color_of(xiong_mg[xiong_ok], z)
        order = np.argsort(xiong_color)
        xiong_color = xiong_color[order]
        xiong_mass_at_points = xm_points[xiong_ok][order]
        in_window_x = (xiong_mass_at_points > mass_window[0]) & (xiong_mass_at_points < mass_window[1])
        extrapolated = not (XIONG_FITTED_RANGE[0] <= z <= XIONG_FITTED_RANGE[1])

        this_line, = top.plot(c_grid, tq[1], color=color, lw=1.9,
                              label=rf"This work (quick) $[M/H]={z:+.1f}$")
        parsec_line, = top.plot(c_grid, parsec, color=color, lw=1.4, ls="--",
                                label=rf"PARSEC $[M/H]={z:+.1f}$")
        note_suffix = " (extrap.)" if extrapolated else ""
        xiong_line, = top.plot(
            xiong_color[in_window_x], xiong_mass_at_points[in_window_x],
            color="#c0392b", lw=1.9, ls=":",
            alpha=.55 if extrapolated else 1.0,
            label=rf"Xiong et al. (2022) $[M/H]={z:+.1f}{note_suffix}",
        )
        this_handles.append(this_line)
        xiong_handles.append(xiong_line)
        parsec_handles.append(parsec_line)
        top.fill_between(c_grid, tq[0], tq[2], color=color, alpha=.16)

        t8_diff = np.percentile(100 * (t8 / parsec[None, :] - 1), [16, 50, 84], axis=0)
        bottom.plot(c_grid, t8_diff[1], color=color, lw=1.9)
        bottom.fill_between(c_grid, t8_diff[0], t8_diff[2], color=color, alpha=.16)
        xiong_parsec = np.interp(xiong_color[in_window_x], c_grid, parsec)
        xiong_diff = 100 * (xiong_mass_at_points[in_window_x] / xiong_parsec - 1)
        bottom.plot(xiong_color[in_window_x], xiong_diff, color="#c0392b", lw=1.9, ls=":",
                    alpha=.55 if extrapolated else 1.0)

        for j in range(len(c_grid)):
            records.append({"relation": "t8_quick", "mh_dex": z, "bp_rp": c_grid[j],
                            "mass_p16": tq[0, j], "mass_p50": tq[1, j], "mass_p84": tq[2, j],
                            "parsec_mass": parsec[j],
                            "difference_p50_percent": t8_diff[1, j]})
        for j in np.flatnonzero(in_window_x):
            records.append({"relation": "xiong2022", "mh_dex": z, "bp_rp": xiong_color[j],
                            "mass_p16": np.nan, "mass_p50": xiong_mass_at_points[j], "mass_p84": np.nan,
                            "parsec_mass": float(np.interp(xiong_color[j], c_grid, parsec)),
                            "difference_p50_percent": float(xiong_diff[j])})
        anchors = {}
        for c_a in (1.5, 2.0, 2.5, 3.0):
            if c_grid.min() <= c_a <= c_grid.max():
                anchors[f"t8_difference_percent_at_bp_rp_{c_a:.1f}"] = float(
                    np.interp(c_a, c_grid, t8_diff[1]))
            if xiong_color[in_window_x].min() <= c_a <= xiong_color[in_window_x].max():
                anchors[f"xiong_difference_percent_at_bp_rp_{c_a:.1f}"] = float(
                    np.interp(c_a, xiong_color[in_window_x], xiong_diff))
        summary[f"{z:+.1f}"] = {
            "bp_rp_range": [float(c_grid[0]), float(c_grid[-1])],
            "t8_vs_parsec_percent_range": [float(t8_diff[1].min()), float(t8_diff[1].max())],
            "xiong_extrapolated": extrapolated,
            "xiong_vs_parsec_percent_range": [float(xiong_diff.min()), float(xiong_diff.max())],
            **anchors,
        }

    bottom.axhline(0, color="0.4", lw=.9)
    top.set(ylabel=r"Mass ($M_\odot$)",
            title="Local quick T8, PARSEC, Mann 2019, Hwang 2023, Xiong 2022 in Gaia $BP-RP$")
    bottom.set(xlabel=r"$BP-RP$ (mag)", ylabel="Relative to PARSEC at same [M/H] (%)")
    relation_handles = [*this_handles, *xiong_handles,
                        mann_artist, hwang_artist,
                        *parsec_handles]
    top.legend(relation_handles, [h.get_label() for h in relation_handles],
               frameon=False, ncol=3, loc="upper right", bbox_to_anchor=(1.0, 1.0),
               fontsize=8.5, columnspacing=.9, handlelength=1.9)
    for axis in (top, bottom): axis.grid(alpha=.2)
    top.set_xlim(0.6, 4.0)
    top.set_ylim(0.0, mass_window[1] * 1.15 + 0.05)
    fig.savefig(OUT_DIR / f"t8_relations_bprp{suffix}.png", dpi=200)
    fig.savefig(OUT_DIR / f"t8_relations_bprp{suffix}.pdf")
    plt.close(fig)
    pd.DataFrame(records).to_csv(OUT_DIR / f"comparison_bprp{suffix}.csv", index=False)
    metadata = {
        "t8_source": str(T8_DIR) + " (LOCAL quick profile: 2000 systems, 1 chain, 300 draws)",
        "axis": "Gaia BP-RP; native for Hwang+2023",
        "color_authority": "T8 IsochroneColorSurfaceModel built from the same PARSEC table used by calibration; T8, PARSEC, Xiong, and Mann all pass through it.",
        "mann_conversion": "M_Ks -> M_G via the PARSEC [M/H]=0, logAge=9.6 band map, then color via the surface.",
        "xiong_source": "Xiong et al. 2022 (arXiv:2211.08647) eq. 5; logL(M,[M/H]) inverted on each slice's PARSEC (logL, Gmag) curve, then color via the surface.",
        "xiong_fitted_range": list(XIONG_FITTED_RANGE),
        "xiong_extrapolation_note": "Slices [M/H]=-0.6 and +0.4 are outside the fitted range and plotted at 55% alpha with an (extrap.) label.",
        "passband_note": "PARSEC surface colors use Gaia DR2 passbands; Hwang's ridge colors are EDR3 (small ~0.01-0.03 mag systematic).",
        "bottom_panel": "Each relation relative to PARSEC at the same [M/H] (solar for Mann and Hwang).",
        "mann_clip_note": "Mann+2019 is drawn only up to its 0.70 Msun calibration limit even when the window extends higher.",
        "mass_window_max": mass_window[1],
        "slices": summary,
    }
    (OUT_DIR / f"metadata_bprp{suffix}.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({"output": str(OUT_DIR), "slices": summary}, indent=2))


_TABLES = None


def _load_tables():
    global _TABLES
    if _TABLES is None:
        from compare_t8_mann2019_ks_local import CMD_COLUMNS, CMD_TABLE
        _TABLES = (pd.read_csv(GAIA_CSV),
                   pd.read_csv(CMD_TABLE, comment="#", sep=r"\s+", header=None,
                               names=CMD_COLUMNS))
    return _TABLES


if __name__ == "__main__":
    main()
