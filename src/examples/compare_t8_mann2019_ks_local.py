#!/usr/bin/env python3
"""Compare the LOCAL quick-run T8 MLR with Mann et al. (2019) in 2MASS Ks.

Same M_G -> M_Ks conversion as compare_t8_mann2019_ks.py: pair the T8 PARSEC
Gaia rows with a fresh CMD 3.8 (PARSEC v1.2S, Gaia DR2+Tycho2+2MASS) table at
identical physical points (Mini, logAge=9.6, [M/H]) and invert the one-to-one
Gmag-Ksmag curve.  T8 draws come from the local quick run in
results/hierarchical_metallicity_t8 (NOT the frozen formal baseline); the Mann
relation posterior is read from the cloned awmann/M_-M_K- repository.  The
Mann-star component scatter of the full script is omitted here because its
derived input tables are not published.
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
from astropy.io import fits

ROOT = Path(__file__).resolve().parents[2]
WORKSPACE = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))
from binary_masses.hierarchical_metallicity import (  # noqa: E402
    IsochroneMassSurfaceModel,
    MonotoneTensorSplineMLR,
)

Z_SLICES = (-0.6, 0.0, 0.4)
T8_DIR = ROOT / "results" / "hierarchical_metallicity_t8"
OUT_DIR = ROOT / "results" / "t8_mann2019_ks_quick"
GAIA_CSV = ROOT / "data" / "PARSEC_logAge_6to10_0p5_MH_n1to0p6_0p2.csv"
CMD_TABLE = ROOT / "results" / "t8_mann2019_ks_20260916" / "parsec_cmd38_gaia_dr2_2mass_targets.dat"
MANN_POST = WORKSPACE / "M_-M_K-" / "resources" / "Mk-M_7_trim.fits"
HWANG_CSV = ROOT / "src" / "examples" / "hwang_mlr" / "hwang2023_mlr.csv"

CMD_COLUMNS = [
    "Zini", "MH", "logAge", "Mini", "int_IMF", "Mass", "logL", "logTe",
    "logg", "label", "McoreTP", "C_O", "period0", "period1", "pmode",
    "Mloss", "tau1m", "X", "Y", "Xc", "Xn", "Xo", "Cexcess", "Z",
    "mbolmag", "Gmag", "G_BPmag", "G_RPmag", "B_Tmag", "V_Tmag",
    "Jmag", "Hmag", "Ksmag",
]


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
    old = gaia[np.isclose(gaia.MH, z) & np.isclose(gaia.logAge, 9.6)
               & (gaia.Mini > 0.08) & (gaia.Mini < 0.85)].sort_values("Mini")
    new = cmd[np.isclose(cmd.MH, z) & np.isclose(cmd.logAge, 9.6)
              & (cmd.Mini > 0.08) & (cmd.Mini < 0.85)].sort_values("Mini")
    old = old[(old.Mini >= new.Mini.min()) & (old.Mini <= new.Mini.max())].copy()
    if len(old) < 10:
        raise ValueError(f"Too few paired PARSEC points at [M/H]={z}")
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
    mass_at_mg = np.array([
        np.interp(mg, saved["absg_grid"], row)
        for row in saved["parsec_mass"]
    ])
    return np.asarray([
        np.interp(z, saved["z_grid"], column)
        for column in mass_at_mg.T
    ])


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    gaia = pd.read_csv(GAIA_CSV)
    cmd = pd.read_csv(CMD_TABLE, comment="#", sep=r"\s+", header=None, names=CMD_COLUMNS)
    mlr, samples = build_t8_model()
    saved = np.load(T8_DIR / "mlr_derived_grid_t8.npz")
    indices = np.linspace(0, len(samples["c0"]) - 1, 512, dtype=int)

    mann_all = np.asarray(fits.getdata(MANN_POST), dtype=float)
    mann = mann_all[np.linspace(0, len(mann_all) - 1, 4096, dtype=int)]
    mann_scatter_normal = np.random.default_rng(20260916).normal(size=len(mann))

    records = []
    summary = {}
    colors = {-.6: "#315da8", 0.0: "#d06a2d", .4: "#278368"}
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(8.5, 8), sharex=False,
                                      gridspec_kw={"height_ratios": [2.1, 1]},
                                      constrained_layout=True)
    mann_x = np.linspace(5.5, 9.6, 141)
    mann_m = mann_draws(mann, mann_x, mann_scatter_normal)
    mann_artist, = top.plot(mann_x, np.percentile(mann_m, 50, axis=0), color="black",
                            lw=1.6, label="Mann et al. (2019)")
    top.fill_between(mann_x, *np.percentile(mann_m, [16, 84], axis=0),
                     color="black", alpha=.12)
    this_handles, parsec_handles = [], []
    for z in Z_SLICES:
        mapping, info = band_map(gaia, cmd, z)
        lo = max(4.0, float(mapping.Ksmag.min()))
        hi = min(11.0, float(mapping.Ksmag.max()))
        ks_grid = np.linspace(lo, hi, 151)
        mg_grid = np.interp(ks_grid, mapping.Ksmag.iloc[::-1], mapping.Gmag.iloc[::-1])
        if mg_grid.min() < 3.5 or mg_grid.max() > 13.5:
            valid = (mg_grid >= 3.5) & (mg_grid <= 13.5)
            ks_grid, mg_grid = ks_grid[valid], mg_grid[valid]
        t8 = t8_draws(mlr, samples, mg_grid, z, indices)
        mm = mann_draws(mann, ks_grid, mann_scatter_normal)
        tq = np.percentile(t8, [16, 50, 84], axis=0)
        mq = np.percentile(mm, [16, 50, 84], axis=0)
        valid = (tq[1] > .075) & (tq[1] < .70) & (mq[1] > .075) & (mq[1] < .70)
        if valid.sum() < 5:
            raise ValueError(f"No adequate common mass range at [M/H]={z}")
        ks_grid, mg_grid, t8, mm = ks_grid[valid], mg_grid[valid], t8[:, valid], mm[:, valid]
        tq, mq = tq[:, valid], mq[:, valid]
        parsec_mass = parsec_mass_curve(saved, mg_grid, z)
        parsec_difference = 100 * (parsec_mass / mq[1] - 1)
        pair = mm[np.linspace(0, len(mm) - 1, len(t8), dtype=int)]
        rq = np.percentile(100 * (t8 / pair - 1), [16, 50, 84], axis=0)
        color = colors[z]
        this_line, = top.plot(ks_grid, tq[1], color=color, lw=1.9,
                              label=rf"This work (quick) $[M/H]={z:+.1f}$")
        parsec_line, = top.plot(ks_grid, parsec_mass, color=color, lw=1.4, ls="--",
                                label=rf"PARSEC $[M/H]={z:+.1f}$")
        this_handles.append(this_line)
        parsec_handles.append(parsec_line)
        top.fill_between(ks_grid, tq[0], tq[2], color=color, alpha=.16)
        bottom.plot(ks_grid, rq[1], color=color, lw=1.9)
        bottom.plot(ks_grid, parsec_difference, color=color, lw=1.4, ls="--")
        bottom.fill_between(ks_grid, rq[0], rq[2], color=color, alpha=.16)
        for j in range(len(ks_grid)):
            records.append({"mh_dex": z, "mks_mag": ks_grid[j], "mg_mag": mg_grid[j],
                            "t8_mass_p16_msun": tq[0, j], "t8_mass_p50_msun": tq[1, j],
                            "t8_mass_p84_msun": tq[2, j], "mann_mass_p16_msun": mq[0, j],
                            "mann_mass_p50_msun": mq[1, j], "mann_mass_p84_msun": mq[2, j],
                            "parsec_mass_msun": parsec_mass[j],
                            "parsec_difference_percent": parsec_difference[j],
                            "difference_p16_percent": rq[0, j],
                            "difference_p50_percent": rq[1, j],
                            "difference_p84_percent": rq[2, j]})
        info.update({"compared_mks_range": [float(ks_grid[0]), float(ks_grid[-1])],
                     "median_difference_percent_range": [float(rq[1].min()), float(rq[1].max())],
                     "median_difference_percent_at_mks_6":
                         float(np.interp(6, ks_grid, rq[1]))
                         if ks_grid[0] <= 6 <= ks_grid[-1] else None,
                     "median_difference_percent_at_mks_7":
                         float(np.interp(7, ks_grid, rq[1]))
                         if ks_grid[0] <= 7 <= ks_grid[-1] else None})
        summary[f"{z:+.1f}"] = info
    bottom.axhline(0, color="0.4", lw=.9)
    # Hwang+2023 NN relation: empirical (BP-RP, M_G) -> mass with no metallicity
    # dimension (near-solar local sample).  Its ridge M_G is placed on the M_Ks
    # axis through the same PARSEC [M/H]=0, logAge=9.6 band map used for T8.
    hwang = pd.read_csv(HWANG_CSV)
    solar_map, _ = band_map(gaia, cmd, 0.0)
    hwang_mks = np.interp(
        hwang.M_G_ridge,
        solar_map.Gmag.iloc[::-1],
        solar_map.Ksmag.iloc[::-1],
    )
    mann_at_hwang = np.percentile(
        mann_draws(mann, hwang_mks, mann_scatter_normal), 50, axis=0
    )
    hwang_valid = (
        (hwang.mass_nn > .075) & (hwang.mass_nn < .70)
        & (hwang.n_stars > 0)
        & (hwang_mks >= 5.5) & (hwang_mks <= 9.6)
    )
    hwang_artist, = top.plot(
        hwang_mks[hwang_valid], hwang.mass_nn[hwang_valid],
        color="#7d3c98", lw=1.8, ls="-.",
        label="Hwang et al. (2023)",
    )
    top.fill_between(
        hwang_mks[hwang_valid],
        (hwang.mass_nn - hwang.mass_nn_err)[hwang_valid],
        (hwang.mass_nn + hwang.mass_nn_err)[hwang_valid],
        color="#7d3c98", alpha=.14,
    )
    hwang_difference = 100 * (hwang.mass_nn / mann_at_hwang - 1)
    bottom.plot(hwang_mks[hwang_valid], hwang_difference[hwang_valid],
                color="#7d3c98", lw=1.8, ls="-.")
    summary["hwang2023"] = {
        "mks_range": [float(hwang_mks[hwang_valid].min()),
                      float(hwang_mks[hwang_valid].max())],
        "difference_percent_range": [float(hwang_difference[hwang_valid].min()),
                                     float(hwang_difference[hwang_valid].max())],
        "difference_percent_at_mks_6":
            float(np.interp(6, hwang_mks[hwang_valid], hwang_difference[hwang_valid]))
            if hwang_mks[hwang_valid].min() <= 6 <= hwang_mks[hwang_valid].max() else None,
        "difference_percent_at_mks_7":
            float(np.interp(7, hwang_mks[hwang_valid], hwang_difference[hwang_valid]))
            if hwang_mks[hwang_valid].min() <= 7 <= hwang_mks[hwang_valid].max() else None,
    }
    # Xiong et al. 2022 (LAMOST EBs, their eq. 5): logL(M, [M/H]); fitted
    # validity is -0.58 < [M/H] < +0.07, so only the solar slice is plotted.
    # Mass is obtained by inverting their (monotonic) logL on the PARSEC
    # [M/H]=0, logAge=9.6 (logL, Ksmag) curve.
    def xiong_log_l(mass, z):
        x = np.log10(mass)
        return 0.066 + 4.141 * x + 0.314 * x**2 - 0.755 * x**3 + 0.189 * x**4 - 0.245 * z
    xiong_mass_grid = np.linspace(0.075, 0.70, 4000)
    xiong_logl_grid = xiong_log_l(xiong_mass_grid, 0.0)
    xiong_mass = np.interp(solar_map.logL, xiong_logl_grid, xiong_mass_grid)
    xiong_valid = (
        (solar_map.logL >= xiong_logl_grid[0]) & (solar_map.logL <= xiong_logl_grid[-1])
        & (solar_map.Ksmag >= 5.5) & (solar_map.Ksmag <= 9.6)
    )
    xiong_artist, = top.plot(
        solar_map.Ksmag[xiong_valid], xiong_mass[xiong_valid],
        color="#c0392b", lw=1.8, ls=":",
        label="Xiong et al. (2022) $[M/H]=0$",
    )
    mann_at_xiong = np.percentile(
        mann_draws(mann, solar_map.Ksmag[xiong_valid].to_numpy(), mann_scatter_normal),
        50, axis=0,
    )
    xiong_difference = 100 * (xiong_mass[xiong_valid] / mann_at_xiong - 1)
    bottom.plot(solar_map.Ksmag[xiong_valid], xiong_difference,
                color="#c0392b", lw=1.8, ls=":")
    ks_xiong = solar_map.Ksmag[xiong_valid].to_numpy()
    xiong_difference = xiong_difference[np.argsort(ks_xiong)]
    ks_xiong = np.sort(ks_xiong)
    summary["xiong2022"] = {
        "mks_range": [float(ks_xiong.min()), float(ks_xiong.max())],
        "difference_percent_range": [float(xiong_difference.min()),
                                     float(xiong_difference.max())],
        "difference_percent_at_mks_6":
            float(np.interp(6, ks_xiong, xiong_difference))
            if ks_xiong.min() <= 6 <= ks_xiong.max() else None,
        "difference_percent_at_mks_7":
            float(np.interp(7, ks_xiong, xiong_difference))
            if ks_xiong.min() <= 7 <= ks_xiong.max() else None,
    }
    top.set(ylabel=r"Mass ($M_\odot$)",
            title="Local quick T8, PARSEC, Mann 2019, Hwang 2023, Xiong 2022 in 2MASS $K_s$")
    bottom.set(xlabel=r"$M_{K_s}$ (mag)", ylabel="Relative to Mann et al. (2019) (%)")
    relation_handles = [*this_handles, mann_artist, hwang_artist, xiong_artist, *parsec_handles]
    top.legend(relation_handles, [h.get_label() for h in relation_handles],
               frameon=False, ncol=2, loc="upper right", bbox_to_anchor=(1.0, .99))
    for axis in (top, bottom): axis.grid(alpha=.2)
    top.set_xlim(4.5, 10.5)
    top.set_ylim(0.0, 0.8)
    bottom.set_xlim(5.5, 9.6)
    fig.savefig(OUT_DIR / "t8_quick_vs_mann2019_ks.png", dpi=200)
    fig.savefig(OUT_DIR / "t8_quick_vs_mann2019_ks.pdf")
    plt.close(fig)
    pd.DataFrame(records).to_csv(OUT_DIR / "comparison_with_parsec.csv", index=False)
    metadata = {
        "t8_source": str(T8_DIR) + " (LOCAL quick profile: 2000 systems, 1 chain, 300 draws)",
        "mann_source": "https://github.com/awmann/M_-M_K-; Mk-M_7_trim.fits",
        "cmd_source": "https://stev.oapd.inaf.it/cgi-bin/cmd_3.8; PARSEC v1.2S; Gaia DR2+Tycho2+2MASS; OBC; logAge=9.6; MH=-1.0..0.6 step 0.2",
        "conversion": "At logAge=9.6 and fixed [M/H], interpolate CMD 3.8 Ks in Mini onto the original T8 PARSEC Gaia rows, then invert their one-to-one M_G to M_Ks curve.",
        "hwang_source": "Hwang et al. (2023) NN relation (github.com/HC-Hwang/HR_mass), extracted to src/examples/hwang_mlr/hwang2023_mlr.csv; near-solar metallicity, no Z dimension.",
        "hwang_conversion": "Ridge M_G mapped to M_Ks through the same PARSEC [M/H]=0, logAge=9.6 band map; empirical ridge ages are mixed, so this is an approximation.",
        "hwang_conversion_systematic": "Mapping via M_G (plotted) vs via BP-RP color agree to <=0.05 mag at BP-RP<2.2 where the ridge sits on solar PARSEC; at BP-RP 2.3-3.2 the empirical ridge is 0.15-0.46 mag brighter than PARSEC at fixed color, so the two routes differ by up to ~0.3 mag there. The color route places Hwang at fainter M_Ks (lower Mann mass), i.e. the plotted difference is the conservative one.",
        "hwang_validation": "Solar point 0.96 vs published 0.961; Jao-gap anchor 0.47+/-0.03 at (BP-RP,M_G)=(2.5,10.0) vs published 0.45+/-0.03 at (2.5,10.1).",
        "xiong_source": "Xiong et al. 2022 (arXiv:2211.08647) eq. 5, LAMOST eclipsing-binary empirical MLR; coefficients a1=0.066 a2=4.141 a3=0.314 a4=-0.245 with cubic/quartic terms fixed to PARSEC.",
        "xiong_conversion": "logL(M, Z=0) inverted on the PARSEC [M/H]=0, logAge=9.6 (logL, Ksmag) curve; fitted validity -0.58<[M/H]<+0.07 so only the solar slice is shown. No coefficient covariance published, so no uncertainty band.",
        "omitted": "Mann-star component scatter (requires unpublished derived tables).",
        "slices": summary,
    }
    (OUT_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({"output": str(OUT_DIR), "slices": summary}, indent=2))


if __name__ == "__main__":
    main()
