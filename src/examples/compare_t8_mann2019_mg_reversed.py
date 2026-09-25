#!/usr/bin/env python3
"""Compare the frozen MLR and PARSEC relations in absolute Gaia G magnitude.

The upper panel uses the complete frozen M_G grid. Mann component points are
converted from M_Ks with the paired PARSEC physical-point mapping, interpolated
in the system metallicity from Mann et al. (2019) Table 1. No extrapolation is
used for either metallicity or M_Ks.
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

ROOT = Path(__file__).resolve().parents[2]
WORKSPACE = ROOT.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
import compare_t8_mann2019_ks as base  # noqa: E402

OUT_DIR = base.OUT_DIR
OUT_STEM = "t8_vs_mann2019_mg_reversed"
MG_SUN = 4.67
CMD_MH_NODES = (-0.6, -0.4, -0.2, 0.0, 0.2, 0.4)


def parse_mann_systems():
    rows = []
    for line in base.MANN_TABLE.read_text().splitlines()[64:]:
        if not line.strip() or line.startswith("-"):
            continue
        rows.append({
            "system_id": line[2:12].strip(),
            "mtot_msun": float(line[64:71]),
            "mtot_error_msun": float(line[72:79]),
            "feh": float(line[80:85]),
        })
    systems = pd.DataFrame(rows)
    if len(systems) != 62 or systems.system_id.duplicated().any():
        raise ValueError("Expected 62 unique Mann Table 1 systems")
    return systems


def build_mappings(gaia, cmd):
    mappings = {}
    for z in CMD_MH_NODES:
        mapping, info = base.band_map(gaia, cmd, z)
        mappings[float(z)] = mapping
    return mappings


def inverse_map(mapping, mks):
    lo, hi = float(mapping.Ksmag.min()), float(mapping.Ksmag.max())
    if not (lo <= mks <= hi):
        return None
    return float(np.interp(mks, mapping.Ksmag.to_numpy()[::-1], mapping.Gmag.to_numpy()[::-1]))


def convert_mann_components(mann_components, mappings):
    systems = parse_mann_systems()
    comp = mann_components.merge(systems[["system_id", "feh"]], on="system_id",
                                how="left", validate="many_to_one")
    rows = []
    z_nodes = np.asarray(CMD_MH_NODES, dtype=float)
    for row in comp.itertuples(index=False):
        out = row._asdict()
        out["mg_mag"] = np.nan
        out["conversion_status"] = "invalid"
        out["conversion_reason"] = ""
        feh = float(row.feh)
        mks = float(row.mks_mag)
        if not np.isfinite(feh) or feh < z_nodes[0] or feh > z_nodes[-1]:
            out["conversion_status"] = "excluded"
            out["conversion_reason"] = "metallicity_outside_CMD_range"
            rows.append(out)
            continue
        exact = np.flatnonzero(np.isclose(z_nodes, feh, rtol=0, atol=1e-12))
        if exact.size:
            lo = hi = int(exact[0])
        else:
            hi = int(np.searchsorted(z_nodes, feh, side="right"))
            lo = hi - 1
        g0 = inverse_map(mappings[float(z_nodes[lo])], mks)
        g1 = inverse_map(mappings[float(z_nodes[hi])], mks)
        if g0 is None or g1 is None:
            out["conversion_status"] = "excluded"
            out["conversion_reason"] = "mks_outside_bracketing_CMD_support"
            rows.append(out)
            continue
        if lo == hi:
            mg = g0
        else:
            mg = g0 + (g1 - g0) * (feh - z_nodes[lo]) / (z_nodes[hi] - z_nodes[lo])
        if not (3.5 <= mg <= 13.5):
            out["conversion_status"] = "excluded"
            out["conversion_reason"] = "converted_mg_outside_requested_range"
            out["mg_mag"] = mg
            rows.append(out)
            continue
        out["mg_mag"] = mg
        out["conversion_status"] = "kept"
        out["conversion_reason"] = "paired_PARSEC_interpolation_no_extrapolation"
        rows.append(out)
    result = pd.DataFrame(rows)
    kept = result[result.conversion_status == "kept"]
    if len(result) != 124 or len(kept) == 0:
        raise ValueError("Mann component conversion produced an invalid row count")
    if not np.all(np.isfinite(kept[["mg_mag", "mass_allocated_msun",
                                    "mass_allocated_error_msun"]])):
        raise ValueError("Kept Mann points contain nonfinite values")
    return result


def inverse_grid(mapping, mg_grid):
    lo, hi = float(mapping.Gmag.min()), float(mapping.Gmag.max())
    keep = (mg_grid >= lo) & (mg_grid <= hi)
    mg = mg_grid[keep]
    mks = np.interp(mg, mapping.Gmag.to_numpy()[::-1], mapping.Ksmag.to_numpy()[::-1])
    return mg, mks


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    gaia = pd.read_csv(base.GAIA_CSV)
    columns = ("Zini", "MH", "logAge", "Mini", "int_IMF", "Mass", "logL",
               "logTe", "logg", "label", "mbolmag", "Gmag", "G_BPmag",
               "G_RPmag", "B_Tmag", "V_Tmag", "Jmag", "Hmag", "Ksmag")
    cmd = pd.read_csv(base.CMD_TABLE, comment="#", sep=r"\s+", header=None,
                      names=columns)
    mlr, samples = base.build_t8_model()
    saved = np.load(base.T8_DIR / "mlr_derived_grid_t8.npz")
    mg_grid = np.asarray(saved["absg_grid"], dtype=float)
    if not np.isclose(mg_grid[0], 3.5) or not np.isclose(mg_grid[-1], 13.5):
        raise ValueError("Frozen M_G grid does not span exactly 3.5 to 13.5")
    if np.any(np.diff(mg_grid) <= 0):
        raise ValueError("Frozen M_G grid is not strictly increasing")
    indices = np.linspace(0, len(samples["c0"]) - 1, 512, dtype=int)
    zero_check = base.t8_draws(mlr, samples, mg_grid, 0.0, indices)
    zero_error = float(np.max(np.abs(zero_check - saved["mass"][:, 2, :])))
    parsec_zero_error = float(np.max(np.abs(
        base.parsec_mass_curve(saved, mg_grid, 0.0)
        - saved["parsec_mass"][np.flatnonzero(np.isclose(saved["z_grid"], 0.0))[0]]
    )))
    if zero_error > 2e-10 or parsec_zero_error > 2e-12:
        raise ValueError("Frozen grid validation failed")

    mann_all = np.asarray(__import__("astropy.io.fits", fromlist=["fits"]).getdata(base.MANN_POST), dtype=float)
    mann_indices = np.linspace(0, len(mann_all) - 1, 4096, dtype=int)
    mann = mann_all[mann_indices]
    scatter_normal = np.random.default_rng(20260916).normal(size=len(mann))
    mappings = build_mappings(gaia, cmd)
    mann_components = base.mann_component_scatter()
    converted = convert_mann_components(mann_components, mappings)
    kept = converted[converted.conversion_status == "kept"]

    colors = {-.6: "#315da8", 0.0: "#d06a2d", .4: "#278368"}
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(8.5, 8), sharex=False,
                                      gridspec_kw={"height_ratios": [2.1, 1]},
                                      constrained_layout=True)
    sun_artist, = top.plot(MG_SUN, 1.0, marker="*", ms=13, color="#e0a400",
                           markeredgecolor="black", markeredgewidth=.7,
                           linestyle="none", label=r"Sun ($1\,M_\odot$)", zorder=8)
    top.errorbar(kept.mg_mag, kept.mass_allocated_msun,
                 yerr=kept.mass_allocated_error_msun, fmt="none", ecolor="0.25",
                 elinewidth=.55, capsize=1.4, alpha=.55, zorder=3)
    mann_artist = top.scatter(kept.mg_mag, kept.mass_allocated_msun, s=19,
                              color="black", alpha=.70, edgecolors="none",
                              label="Mann et al. (2019) stars", zorder=4)
    this_handles, parsec_handles = [], []
    summary = {}
    records = []
    top_records = []
    top_mass_max = 1.0
    for z in base.Z_SLICES:
        t8 = base.t8_draws(mlr, samples, mg_grid, z, indices)
        parsec = base.parsec_mass_curve(saved, mg_grid, z)
        tq = np.percentile(t8, [16, 50, 84], axis=0)
        top_mass_max = max(top_mass_max, float(np.max(tq[2])), float(np.max(parsec)))
        this_line, = top.plot(mg_grid, tq[1], color=colors[z], lw=1.9,
                              label=rf"This work $[M/H]={z:+.1f}$")
        parsec_line, = top.plot(mg_grid, parsec, color=colors[z], lw=1.4, ls="--",
                                label=rf"PARSEC $[M/H]={z:+.1f}$")
        this_handles.append(this_line)
        parsec_handles.append(parsec_line)
        top.fill_between(mg_grid, tq[0], tq[2], color=colors[z], alpha=.16)
        for j in range(len(mg_grid)):
            top_records.append({"mh_dex": z, "mg_mag": mg_grid[j],
                                "this_work_mass_p16_msun": tq[0, j],
                                "this_work_mass_p50_msun": tq[1, j],
                                "this_work_mass_p84_msun": tq[2, j],
                                "parsec_mass_msun": parsec[j]})

        mapping = mappings[float(z)]
        mg_common, mks_common = inverse_grid(mapping, mg_grid)
        mm = base.mann_draws(mann, mks_common, scatter_normal)
        mq = np.percentile(mm, [16, 50, 84], axis=0)
        valid = (tq[1][np.searchsorted(mg_grid, mg_common)] > .075) & (tq[1][np.searchsorted(mg_grid, mg_common)] < .70)
        valid &= (mq[1] > .075) & (mq[1] < .70)
        if valid.sum() < 5:
            raise ValueError(f"No adequate lower-panel support at [M/H]={z}")
        mg_common, mks_common, mm, mq = mg_common[valid], mks_common[valid], mm[:, valid], mq[:, valid]
        common_idx = np.searchsorted(mg_grid, mg_common)
        tq_common = tq[:, common_idx]
        t8_common = t8[:, common_idx]
        pair = mm[np.linspace(0, len(mm) - 1, len(t8), dtype=int)]
        rq = np.percentile(100 * (t8_common / pair - 1), [16, 50, 84], axis=0)
        parsec_diff = 100 * (base.parsec_mass_curve(saved, mg_common, z) / mq[1] - 1)
        bottom.plot(mg_common, rq[1], color=colors[z], lw=1.9)
        bottom.plot(mg_common, parsec_diff, color=colors[z], lw=1.4, ls="--")
        bottom.fill_between(mg_common, rq[0], rq[2], color=colors[z], alpha=.16)
        for j in range(len(mg_common)):
            records.append({"mh_dex": z, "mg_mag": mg_common[j], "mks_mag": mks_common[j],
                            "t8_mass_p16_msun": tq_common[0, j], "t8_mass_p50_msun": tq_common[1, j],
                            "t8_mass_p84_msun": tq_common[2, j], "mann_mass_p16_msun": mq[0, j],
                            "mann_mass_p50_msun": mq[1, j], "mann_mass_p84_msun": mq[2, j],
                            "parsec_mass_msun": base.parsec_mass_curve(saved, mg_common, z)[j],
                            "parsec_difference_percent": parsec_diff[j],
                            "difference_p16_percent": rq[0, j], "difference_p50_percent": rq[1, j],
                            "difference_p84_percent": rq[2, j]})
        summary[f"{z:+.1f}"] = {"mapping_mg_range": [float(mapping.Gmag.min()), float(mapping.Gmag.max())],
                                 "lower_panel_mg_range": [float(mg_common[0]), float(mg_common[-1])],
                                 "lower_panel_points": int(len(mg_common))}

    bottom.axhline(0, color="0.4", lw=.9)
    top.set(ylabel=r"Mass ($M_\odot$)", title="This work, PARSEC, and Mann et al. (2019) in Gaia $M_G$")
    bottom.set(xlabel=r"$M_G$ (mag)", ylabel="Relative to Mann et al. (2019) (%)")
    reference_handles = [sun_artist, mann_artist]
    reference_legend = top.legend(reference_handles, [h.get_label() for h in reference_handles],
                                  frameon=False, ncol=2, loc="upper left",
                                  bbox_to_anchor=(.01, 1.0))
    top.add_artist(reference_legend)
    top.legend([*this_handles, *parsec_handles], [h.get_label() for h in [*this_handles, *parsec_handles]],
               frameon=False, ncol=2, loc="upper left", bbox_to_anchor=(.01, .91))
    for axis in (top, bottom): axis.grid(alpha=.2)
    top.set_xlim(13.5, 3.5); top.set_ylim(0.05, top_mass_max * 1.04)
    bottom.set_xlim(13.5, 3.5)
    fig.savefig(OUT_DIR / f"{OUT_STEM}.png", dpi=200)
    fig.savefig(OUT_DIR / f"{OUT_STEM}.pdf")
    plt.close(fig)

    converted.to_csv(OUT_DIR / f"mann_component_scatter_{OUT_STEM}.csv", index=False)
    pd.DataFrame(top_records).to_csv(OUT_DIR / f"curves_{OUT_STEM}.csv", index=False)
    with (OUT_DIR / f"comparison_{OUT_STEM}.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0])); writer.writeheader(); writer.writerows(records)
    metadata = {
        "axis": "Absolute Gaia G magnitude M_G; x-axis reversed, 13.5 on left and 3.5 on right.",
        "top_curve_support": [3.5, 13.5],
        "solar_reference": {"mass_msun": 1.0, "mg_mag": MG_SUN,
                            "note": "Adopted standard reference value; marker only."},
        "conversion": "At logAge=9.6 and fixed [M/H], interpolate CMD 3.8 K_s in Mini onto original T8 PARSEC Gaia rows. For each Mann component, invert its M_Ks to M_G at the bracketing metallicity nodes, then interpolate M_G in [Fe/H].",
        "no_extrapolation": "Mann components are kept only when [Fe/H] is within the CMD node range and M_Ks lies within both bracketing mappings; all 124 rows are recorded with status and reason.",
        "mann_points_total": 124, "mann_points_kept": int(len(kept)),
        "mann_points_excluded": int(len(converted) - len(kept)),
        "conversion_reason_counts": converted.conversion_reason.value_counts().to_dict(),
        "t8_zero_slice_max_abs_mass_error": zero_error,
        "parsec_zero_slice_max_abs_mass_error": parsec_zero_error,
        "top_curve_rows": int(len(top_records)),
        "input_sha256": {str(path.relative_to(WORKSPACE)): base.sha256(path) for path in
                         (base.GAIA_CSV, base.CMD_TABLE, base.MANN_POST,
                          base.MANN_TABLE, base.MANN_COMPONENTS,
                          base.T8_DIR / "mlr_mcmc_t8.npz",
                          base.T8_DIR / "mlr_derived_grid_t8.npz")},
        "slices": summary,
        "lower_panel": "Common MG-to-MKs mapping and Mann validity/mass support only; it is shorter than the full top-panel curves.",
    }
    (OUT_DIR / f"metadata_{OUT_STEM}.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
