#!/usr/bin/env python3
"""Main-result figure: MLR in Gaia M_G and the ratio to PARSEC.

Panel (a): mass vs M_G for this work at [M/H] = -0.6, -0.3, 0, +0.3 (median and
68% band; PARSEC at the same [M/H] dashed), with Hwang+2024 (main-sequence
ridge) and the dynamical / independent mass measurements Mann+2019 (visual
orbits), Xiong+2023 (detached eclipsing-binary components), Chevalier+2023
(Gaia DR3 astrometric orbits + SB2) and Serenelli+2017 (asteroseismic dwarfs).
Panel (b): this work at [M/H]=0 and the literature relative to PARSEC [M/H]=0,
with the p(u~) shape systematic as a grey band (no statistical band in this panel): the eccentricity power law p(e) ~ e^alpha is varied over
alpha in [1.0, 1.5] around the adopted alpha = 1.233 (scripts/run_ecc_shape_sensitivity.sh).
A separate figure shows d log10 M / d[M/H] at fixed M_G (this work vs PARSEC).

Native M_G: this work, PARSEC, Hwang ridge, Chevalier (their M_G), Serenelli
(Gaia DR3 G and parallax, no extinction correction).  Model-dependent: Mann
(Ks -> M_G through the PARSEC [M/H]=0, logAge 9.6 band map) and Xiong (measured
logL -> M_G on PARSEC logAge 9.6 isochrones at the system [Fe/H]).

Literature inputs in data/literature/ were retrieved from VizieR:
  chevalier2023_components.csv  J/A+A/678/A19 tables 1-3 (M1, M2, MG1, MG2)
  serenelli2017_gaia.csv        J/ApJS/233/23 tables 3+4, CDS-XMatch to Gaia DR3 (2")
  xiong2023_table{2,3}.tsv      J/AJ/165/30 tables 2-3 (component mass, logL, [Fe/H])
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
import compare_t8_mann2019_ks_local as L  # noqa: E402

ROOT = L.ROOT
RUNS = ROOT / "results" / "ecc_shape_sensitivity_20261009"
CENTRAL = "alpha_1.233"            # adopted p(u~) shape
OUT = ROOT / "results" / "mlr_summary_figure"
PAPER_FIG = ROOT / "docs" / "paper" / "t9_aa" / "figs"
LIT = ROOT / "data" / "literature"
CMD_TABLE = ROOT / "results" / "ks_compare_default" / "parsec_cmd38_gaia_dr2_2mass_targets.dat"
CMD_COLUMNS = ("Zini", "MH", "logAge", "Mini", "int_IMF", "Mass", "logL", "logTe", "logg",
               "label", "mbolmag", "Gmag", "G_BPmag", "G_RPmag", "B_Tmag", "V_Tmag",
               "Jmag", "Hmag", "Ksmag")
MANN_POINTS = ROOT / "results" / "ks_compare_default" / "mann_component_scatter_with_errorbars.csv"
VARIANTS = ("alpha_1", "alpha_1.5")  # edges of the eccentricity systematic range
MG = np.linspace(3.5, 13.5, 301)
Z0, ZLO, ZHI = 0.0, -0.6, 0.4
Z_SLICES = (-0.6, -0.3, 0.0, 0.3)
# viridis samples (perceptually uniform, ordered): metal-poor purple -> metal-rich green
Z_COLORS = {-0.6: "#440154", -0.3: "#31678E", 0.0: "#22A884", 0.3: "#9BD93C"}
N_DRAW = 512
XLIM = (3.5, 13.5)

# Okabe-Ito categorical palette (validated: adjacent CVD dE >= 7.6, used with
# distinct line styles / marker shapes as secondary encoding).
C_THIS = "#0072B2"
C_HWANG = "#CC79A7"
C_XIONG = "#D55E00"
C_NSS = "#009E73"
C_SEIS = "#E69F00"
C_MANN = "black"
C_PARSEC = "0.45"

plt.rcParams.update({
    "font.size": 11, "axes.labelsize": 12, "legend.fontsize": 8.8,
    "xtick.labelsize": 10.5, "ytick.labelsize": 10.5,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.top": True, "ytick.right": True,
    "xtick.minor.visible": True, "ytick.minor.visible": True,
    "axes.linewidth": 0.9, "axes.grid": False,
    "font.family": "serif", "mathtext.fontset": "dejavuserif",
})


def draws(run, z, mg=MG, n=N_DRAW):
    L.T8_DIR = RUNS / run
    mlr, samples = L.build_t8_model()
    idx = np.linspace(0, len(samples["c0"]) - 1, n, dtype=int)
    return L.t8_draws(mlr, samples, mg, z, idx)


def parsec_curve(z, mg=MG):
    saved = np.load(RUNS / CENTRAL / "mlr_derived_grid_t8.npz")
    return L.parsec_mass_curve(saved, mg, z)


def load_xiong_points(gaia):
    """Xiong+23 detached-EB components (J/AJ/165/30 tables 2+3) on the M_G axis.

    Only unevolved components (logg >= 4.0, M < 1.4 Msun).  The measured logL is
    placed on M_G with PARSEC logAge 9.6 isochrones interpolated linearly in [M/H]
    at the system [Fe/H] (model-dependent bolometric correction).
    """
    import io
    rows = []
    for t in ("table2", "table3"):
        lines = [l for l in open(LIT / f"xiong2023_{t}.tsv") if not l.startswith("#") and l.strip()]
        d = pd.read_csv(io.StringIO("".join([lines[0]] + lines[3:])), sep="\t")
        for k in (1, 2):
            rows.append(pd.DataFrame({
                "mass": pd.to_numeric(d[f"Mass{k}"], errors="coerce"),
                "e_mass": pd.to_numeric(d[f"e_Mass{k}"], errors="coerce"),
                "logL": pd.to_numeric(d[f"Lum{k}"], errors="coerce"),
                "logg": pd.to_numeric(d[f"logg{k}"], errors="coerce"),
                "feh": pd.to_numeric(d["[Fe/H]"], errors="coerce")}))
    x = pd.concat(rows).dropna(subset=["mass", "logL", "feh", "logg"])
    x = x[(x.logg >= 4.0) & (x.mass < 1.4)].copy()
    iso_z = np.array(sorted(gaia.MH.unique()))
    isos = [gaia[np.isclose(gaia.MH, z) & np.isclose(gaia.logAge, 9.6) & (gaia.Mini > 0.08)
                 & (gaia.Mini < 1.2)].sort_values("logL") for z in iso_z]

    def mg_of(logl, f):
        f = float(np.clip(f, iso_z[0], iso_z[-1]))
        j = int(np.clip(np.searchsorted(iso_z, f) - 1, 0, len(iso_z) - 2))
        w = (f - iso_z[j]) / (iso_z[j + 1] - iso_z[j])
        g = [np.interp(logl, i.logL, i.Gmag, left=np.nan, right=np.nan) for i in isos[j:j + 2]]
        return (1 - w) * g[0] + w * g[1]

    x["MG"] = [mg_of(l, f) for l, f in zip(x.logL, x.feh)]
    return x[np.isfinite(x.MG) & (x.MG >= XLIM[0]) & (x.MG <= XLIM[1])]


def mann_feh():
    """System [Fe/H] from the Mann+19 Table 1 (fixed-width MRT)."""
    table = ROOT.parent / "Model" / "mann2019" / "Table1.mrt"
    lines = table.read_text().splitlines()
    start = max(i for i, l in enumerate(lines) if l.startswith("----")) + 1
    return pd.DataFrame([{"system_id": l[2:12].strip(), "feh": float(l[80:85])}
                         for l in lines[start:] if l.strip() and l[80:85].strip()])


def xiong_log_l(mass, z):
    x = np.log10(mass)
    return 0.066 + 4.141 * x + 0.314 * x**2 - 0.755 * x**3 + 0.189 * x**4 - 0.245 * z


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    PAPER_FIG.mkdir(parents=True, exist_ok=True)
    gaia = pd.read_csv(L.GAIA_CSV)
    cmd = pd.read_csv(CMD_TABLE, comment="#", sep=r"\s+", header=None, names=CMD_COLUMNS)
    solar_map, _ = L.band_map(gaia, cmd, 0.0)

    # ---- this work -------------------------------------------------------
    d0 = draws(CENTRAL, Z0)
    q = np.percentile(d0, [2.5, 16, 50, 84, 97.5], axis=0)
    p0 = parsec_curve(Z0)
    rq = np.percentile(100 * (d0 / p0 - 1), [2.5, 16, 50, 84, 97.5], axis=0)
    var_med = np.array([np.median(draws(v, Z0, n=128), axis=0) for v in VARIANTS])
    var_rel = 100 * (var_med / p0 - 1)
    sys_lo = np.minimum(var_rel.min(axis=0), rq[2])
    sys_hi = np.maximum(var_rel.max(axis=0), rq[2])

    def rel_to_parsec(mg, mass):
        return 100 * (np.asarray(mass) / np.interp(mg, MG, p0, left=np.nan, right=np.nan) - 1)

    # ---- metallicity dependence -----------------------------------------
    mg_c = np.linspace(4.5, 13.0, 200)
    slope = np.log10(draws(CENTRAL, ZHI, mg_c)) - np.log10(draws(CENTRAL, ZLO, mg_c))
    sq = np.percentile(slope, [16, 50, 84], axis=0)
    ps = np.log10(parsec_curve(ZHI, mg_c)) - np.log10(parsec_curve(ZLO, mg_c))

    # ---- Mann+2019 -------------------------------------------------------
    mann = np.asarray(fits.getdata(L.MANN_POST), dtype=float)
    mann = mann[np.linspace(0, len(mann) - 1, 4096, dtype=int)]
    eps = np.random.default_rng(20260916).normal(size=len(mann))
    ks_of_mg = np.interp(MG, solar_map.Gmag.iloc[::-1], solar_map.Ksmag.iloc[::-1])
    ok = (ks_of_mg >= 5.5) & (ks_of_mg <= 9.6)
    mm = L.mann_draws(mann, ks_of_mg[ok], eps)
    good = np.median(mm, axis=0) <= 0.70
    mg_m, mann_med = MG[ok][good], np.median(mm[:, good], axis=0)
    pts = pd.read_csv(MANN_POINTS)
    pts = pts[(pts.mks_mag >= 4.5) & (pts.mks_mag <= 10.0)]
    mg_pts = np.interp(pts.mks_mag, solar_map.Ksmag.iloc[::-1], solar_map.Gmag.iloc[::-1])
    pts_b = pts.merge(mann_feh(), on="system_id", how="left")
    pts_b = pts_b[pts_b.feh.abs() <= 0.15]
    mg_pts_b = np.interp(pts_b.mks_mag, solar_map.Ksmag.iloc[::-1], solar_map.Gmag.iloc[::-1])

    # ---- Hwang+2024 (native M_G ridge) -----------------------------------
    hw = pd.read_csv(L.HWANG_CSV)
    hw = hw[(hw.mass_nn > 0.075) & (hw.n_stars > 0) & (hw.M_G_ridge >= 3.5)
            & (hw.M_G_ridge <= 13.5)]

    # ---- Xiong+2023 eclipsing-binary components ---------------------------
    xi = load_xiong_points(gaia)

    # ---- Gaia DR3 orbits (Chevalier+2023) --------------------------------
    nss = pd.read_csv(LIT / "chevalier2023_components.csv")
    nss = nss[(nss.MG >= XLIM[0]) & (nss.MG <= XLIM[1]) & (nss.e_mass / nss.mass < 0.2)]

    # ---- asteroseismic dwarfs (Serenelli+2017) ---------------------------
    seis = pd.read_csv(LIT / "serenelli2017_gaia.csv")
    seis = seis[(seis.logg >= 4.1) & (seis.RUWE < 1.4) & (seis.MG >= XLIM[0])
                & (seis.Plx / seis.e_Plx > 20)]

    # ---- this work at several metallicities ------------------------------
    slices = {}
    for z in Z_SLICES:
        dz = d0 if z == Z0 else draws(CENTRAL, z)
        pz = parsec_curve(z)
        slices[z] = (np.percentile(dz, [16, 50, 84], axis=0), pz,
                     np.percentile(100 * (dz / pz - 1), [16, 50, 84], axis=0))

    # ---- figure 1: mass and ratio to PARSEC ------------------------------
    fig, (a, b) = plt.subplots(
        2, 1, figsize=(6.0, 6.6), sharex=True,
        gridspec_kw={"height_ratios": [1.9, 1.15], "hspace": 0.0})

    a.plot(hw.M_G_ridge, hw.mass_nn, color=C_HWANG, lw=2.0, ls="-.", zorder=6,
           label="Hwang+24 (MS ridge)")
    for z in Z_SLICES:
        (lo, med, hi), pz, _ = slices[z]
        col = Z_COLORS[z]
        a.plot(MG, pz, color=col, lw=1.0, ls="--", zorder=2.2)
        a.fill_between(MG, lo, hi, color=col, alpha=.35, lw=0, zorder=2)
        a.plot(MG, med, color=col, lw=1.25, zorder=2.5,
               label=rf"This work, $Z={z:+.1f}$")
    a.plot([], [], color="0.3", lw=1.2, ls="--", label=r"PARSEC, same $Z$")
    # literature markers (data are shown in panel b): legend keys only
    ek = dict(ls="none", mec="black", mew=0.6)
    a.plot([], [], "o", color="0.15", ms=4.2, label="Mann+19 (visual orbits)", **ek)
    a.plot([], [], "s", color="white", ms=5.5, label="Chevalier+23 (Gaia orbits + SB2)", **ek)
    a.plot([], [], "^", color=C_XIONG, ms=5.5, label="Xiong+23 (eclipsing binaries)", **ek)
    a.plot([], [], "D", color=C_SEIS, ms=5.0, label="Serenelli+17 (asteroseismology)", **ek)
    a.set_yscale("log")
    a.set(ylabel=r"Mass $[M_{\mathrm{\odot}}]$", ylim=(0.068, 1.6))
    a.set_yticks([0.1, 0.2, 0.5, 1.0])
    a.set_yticklabels(["0.1", "0.2", "0.5", "1"])
    handles, labels = a.get_legend_handles_labels()
    first = [i for i, l in enumerate(labels) if l.startswith("This work")]
    rest = [labels.index(k) for k in (r"PARSEC, same $Z$", "Hwang+24 (MS ridge)", "Mann+19 (visual orbits)", "Xiong+23 (eclipsing binaries)", "Chevalier+23 (Gaia orbits + SB2)", "Serenelli+17 (asteroseismology)")]
    order = first + rest
    a.legend([handles[i] for i in order], [labels[i] for i in order], frameon=False,
             loc="lower right", ncol=2, handlelength=2.0, columnspacing=0.8,
             borderaxespad=0.3, fontsize=8.3)
    a.text(0.03, 0.95, "(a)", transform=a.transAxes, ha="left", va="top", fontweight="bold")

    # panel (b): near-solar literature points only (|[M/H]| <= 0.15; Chevalier+23 has no metallicity and is shown unfiltered)
    xi_b = xi[xi.feh.abs() <= 0.15]
    seis_b = seis[seis["[M/H]"].abs() <= 0.15]
    b.axhline(0, color="0.5", lw=1.0, ls="-", zorder=1.5)
    # p(u~) systematic: envelope of the alpha_e = 1.0 and 1.5 runs around the adopted fit (no statistical band)
    sys_band = b.fill_between(MG, sys_lo, sys_hi, color="0.65", lw=0, zorder=2)
    sp = dict(ls="none", mec="black", mew=0.5, zorder=4)
    b.plot(mg_pts_b, rel_to_parsec(mg_pts_b, pts_b.mass_allocated_msun), "o", color="0.15",
           ms=3.6, **{**sp, "zorder": 4.2})
    b.plot(nss.MG, rel_to_parsec(nss.MG, nss.mass), "s", color="white", ms=4.6, **sp)
    b.plot(seis_b.MG, rel_to_parsec(seis_b.MG, seis_b.Mass), "D", color=C_SEIS, ms=4.2, **{**sp, "zorder": 4.6})
    b.plot(hw.M_G_ridge, rel_to_parsec(hw.M_G_ridge, hw.mass_nn), color=C_HWANG, lw=2.0,
           ls="-.", zorder=6)
    b.plot(xi_b.MG, rel_to_parsec(xi_b.MG, xi_b.mass), "^", color=C_XIONG, ms=4.6, **{**sp, "zorder": 4.4})
    for z in (Z0,):
        _, _, (_, rmed, _) = slices[z]
        (med_line,) = b.plot(MG, rmed, color=Z_COLORS[z], lw=1.25, zorder=2.5)
    b.set(xlabel=r"$M_{\mathrm{G}}$ [mag]", ylabel="Relative to PARSEC mass\n$Z=0$ (%)", ylim=(-38, 62))
    # legend key: green median line drawn on the grey systematic patch
    b.legend([(sys_band, med_line)], [r"$p(\tilde u)$ shape systematic"], frameon=False,
             loc="upper right", borderaxespad=0.3, fontsize=8.5)
    b.set_yticks([-20, 0, 20, 40])
    b.yaxis.label.set_size(9.5)
    b.text(0.03, 0.95, "(b)", transform=b.transAxes, ha="left", va="top", fontweight="bold")
    a.set_xlim(XLIM[1], XLIM[0])  # inverted M_G axis
    fig.align_ylabels((a, b))
    fig.subplots_adjust(left=0.13, right=0.97, top=0.985, bottom=0.085)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"mlr_summary.{ext}", dpi=250)
        fig.savefig(PAPER_FIG / f"mlr_summary.{ext}", dpi=250)
    plt.close(fig)

    # ---- figure 2: metallicity dependence --------------------------------
    fig, c = plt.subplots(figsize=(6.0, 3.3))
    c.axhline(0, color="0.6", lw=0.8)
    c.fill_between(mg_c, sq[0], sq[2], color=C_THIS, alpha=.35, lw=0)
    c.plot(mg_c, sq[1], color=C_THIS, lw=1.25, label="This work")
    c.plot(mg_c, ps, color="0.3", lw=1.6, ls="--", label="PARSEC")
    c.set(xlabel=r"$M_{\mathrm{G}}$ [mag]",
          ylabel=r"$\Delta\log_{10}M\,/\,\Delta Z$", xlim=XLIM[::-1], ylim=(-0.02, 0.34))
    c.legend(frameon=False, loc="upper right")
    c.text(0.97, 0.06, r"$Z=-0.6\rightarrow+0.4$, fixed $M_{\mathrm{G}}$",
           transform=c.transAxes, ha="right", va="bottom", fontsize=9.5)
    fig.subplots_adjust(left=0.13, right=0.97, top=0.97, bottom=0.17)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"mlr_metallicity_slope.{ext}", dpi=250)
        fig.savefig(PAPER_FIG / f"mlr_metallicity_slope.{ext}", dpi=250)
    plt.close(fig)

    pd.DataFrame({"mg": MG, "mass_p2p5": q[0], "mass_p16": q[1], "mass_p50": q[2],
                  "mass_p84": q[3], "mass_p97p5": q[4], "parsec": p0,
                  "rel_p50_percent": rq[2], "shape_sys_lo_percent": sys_lo,
                  "shape_sys_hi_percent": sys_hi}).to_csv(OUT / "mlr_summary_solar.csv", index=False)
    summary = {
        "n_points": {"mann": int(len(pts)), "chevalier": int(len(nss)), "serenelli": int(len(seis))},
        "relative_to_parsec_percent": {
            f"{m:g}": {"this_work": float(np.interp(m, MG, rq[2])),
                       "sys_lo": float(np.interp(m, MG, sys_lo)),
                       "sys_hi": float(np.interp(m, MG, sys_hi))} for m in (4, 5, 6, 8, 10, 12)},
        "median_rel_to_parsec_percent": {
            "chevalier_MG_6_10": float(np.nanmedian(rel_to_parsec(nss.MG, nss.mass)[
                (nss.MG >= 6) & (nss.MG <= 10)])),
            "serenelli": float(np.nanmedian(rel_to_parsec(seis.MG, seis.Mass))),
        },
        "dlogM_per_dex": {f"{m:g}": {"this_work": float(np.interp(m, mg_c, sq[1])),
                                      "parsec": float(np.interp(m, mg_c, ps))}
                          for m in (5, 7, 9, 11)},
    }
    # binned median offsets of the near-solar literature points (relative to PARSEC Z=0 and to this work)
    lit = {"mann": (mg_pts_b.to_numpy() if hasattr(mg_pts_b, "to_numpy") else np.asarray(mg_pts_b), pts_b.mass_allocated_msun.to_numpy()),
           "chevalier": (nss.MG.to_numpy(), nss.mass.to_numpy()),
           "xiong": (xi_b.MG.to_numpy(), xi_b.mass.to_numpy()),
           "serenelli": (seis_b.MG.to_numpy(), seis_b.Mass.to_numpy())}
    summary["literature_binned"] = {}
    for lo_, hi_ in ((3.5, 5.5), (5.5, 8.0), (8.0, 10.5), (10.5, 13.5)):
        row = {"this_work_vs_parsec_percent": float(np.mean(np.interp(np.linspace(lo_, hi_, 50), MG, rq[2])))}
        for name, (mg_l, m_l) in lit.items():
            sel = (mg_l >= lo_) & (mg_l < hi_) & np.isfinite(m_l)
            if sel.sum() == 0:
                continue
            vs_p = 100 * (m_l[sel] / np.interp(mg_l[sel], MG, p0) - 1)
            vs_t = 100 * (m_l[sel] / np.interp(mg_l[sel], MG, q[2]) - 1)
            row[name] = {"n": int(sel.sum()), "median_vs_parsec_percent": float(np.median(vs_p)),
                         "median_vs_this_work_percent": float(np.median(vs_t)),
                         "scatter_vs_this_work_percent": float(1.4826 * np.median(np.abs(vs_t - np.median(vs_t))))}
        summary["literature_binned"][f"{lo_:g}-{hi_:g}"] = row
    (OUT / "anchors.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
