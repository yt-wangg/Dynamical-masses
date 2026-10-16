#!/usr/bin/env python3
"""MLR comparison in metallicity bins.

Four bins in [Fe/H]/[M/H]; each bin has a mass panel and a ratio-to-PARSEC panel.
Per bin: this work at the bin centre (median + 68% band; the lighter band spans
the medians at the two bin edges), PARSEC at the bin centre (same colour, dashed), and the Mann+19 component points and Serenelli+17 asteroseismic
dwarfs and Xiong+23 detached-EB components whose own metallicity falls in the bin.  In the ratio panels each point
is divided by PARSEC at its own metallicity; curves by PARSEC at the bin centre.

Model-dependent conversions: Mann Ks -> M_G through PARSEC (logAge 9.6) band
maps interpolated in metallicity; Xiong logL -> M_G on PARSEC isochrones
(logAge 9.6) interpolated in metallicity (see make_mlr_summary_figure.load_xiong_points).
"""
from __future__ import annotations

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import compare_t8_mann2019_ks_local as L  # noqa: E402
import make_mlr_summary_figure as S  # noqa: E402  (style, paths, helpers)

ROOT = L.ROOT
MANN_TABLE = ROOT.parent / "Model" / "mann2019" / "Table1.mrt"
BINS = [(-0.7, -0.4), (-0.4, -0.15), (-0.15, 0.1), (0.1, 0.5)]
BAND_Z = np.array([-0.6, -0.4, -0.2, 0.0, 0.2, 0.4])  # metallicities with a Ks band map
N_DRAW = 256


def mann_feh():
    lines = MANN_TABLE.read_text().splitlines()
    start = max(i for i, l in enumerate(lines) if l.startswith("----")) + 1
    rows = []
    for l in lines[start:]:
        if l.strip() and l[80:85].strip():
            rows.append({"system_id": l[2:12].strip(), "feh": float(l[80:85]),
                         "feh_flag": l[86:87].strip()})
    return pd.DataFrame(rows)


def interp_in_z(z, grid, values):
    """Linear interpolation of per-grid arrays in metallicity (clipped)."""
    z = float(np.clip(z, grid[0], grid[-1]))
    j = int(np.clip(np.searchsorted(grid, z) - 1, 0, len(grid) - 2))
    w = (z - grid[j]) / (grid[j + 1] - grid[j])
    return (1 - w) * values[j] + w * values[j + 1]


def main():
    gaia = pd.read_csv(L.GAIA_CSV)
    cmd = pd.read_csv(S.CMD_TABLE, comment="#", sep=r"\s+", header=None, names=S.CMD_COLUMNS)
    maps = {z: L.band_map(gaia, cmd, z)[0] for z in BAND_Z}
    saved = np.load(S.RUNS / S.CENTRAL / "mlr_derived_grid_t8.npz")

    def parsec_at(mg, z):
        return float(L.parsec_mass_curve(saved, np.atleast_1d(mg), z)[0])

    # Mann components with system [Fe/H]; Ks -> M_G at the system metallicity
    pts = pd.read_csv(S.MANN_POINTS).merge(mann_feh(), on="system_id", how="left")
    pts = pts[(pts.mks_mag >= 4.5) & (pts.mks_mag <= 10.0) & pts.feh.notna()].copy()
    pts["MG"] = [interp_in_z(f, BAND_Z, [np.interp(k, maps[z].Ksmag.iloc[::-1],
                                                   maps[z].Gmag.iloc[::-1]) for z in BAND_Z])
                 for k, f in zip(pts.mks_mag, pts.feh)]
    pts["rel"] = [100 * (m / parsec_at(g, f) - 1)
                  for m, g, f in zip(pts.mass_allocated_msun, pts.MG, pts.feh)]

    seis = pd.read_csv(S.LIT / "serenelli2017_gaia.csv")
    seis = seis[(seis.logg >= 4.1) & (seis.RUWE < 1.4) & (seis.MG >= S.XLIM[0])
                & (seis.Plx / seis.e_Plx > 20)].copy()
    seis["feh"] = seis["[M/H]"]
    seis["rel"] = [100 * (m / parsec_at(g, f) - 1) if g >= 3.5 else np.nan
                   for m, g, f in zip(seis.Mass, seis.MG, seis.feh)]

    xi = S.load_xiong_points(gaia)
    xi["rel"] = [100 * (m / parsec_at(g, f) - 1) if g >= 3.5 else np.nan
                 for m, g, f in zip(xi.mass, xi.MG, xi.feh)]

    fig = plt.figure(figsize=(7.2, 7.0))
    outer = GridSpec(2, 2, figure=fig, hspace=0.035, wspace=0.035,
                     left=0.10, right=0.985, top=0.985, bottom=0.075)
    axes = []
    for k, (lo, hi) in enumerate(BINS):
        inner = GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[k // 2, k % 2],
                                        height_ratios=[1.8, 1.0], hspace=0.0)
        a = fig.add_subplot(inner[0])
        b = fig.add_subplot(inner[1], sharex=a)
        axes.append((a, b))
        zc = 0.5 * (lo + hi)
        col = S.Z_COLORS[S.Z_SLICES[k]]  # same colours as the summary figure, in bin order

        dc = S.draws(S.CENTRAL, zc, n=N_DRAW)
        pc = S.parsec_curve(zc)
        q = np.percentile(dc, [16, 50, 84], axis=0)
        rq = np.percentile(100 * (dc / pc - 1), [16, 50, 84], axis=0)
        e_lo = np.median(S.draws(S.CENTRAL, lo, n=128), axis=0)
        e_hi = np.median(S.draws(S.CENTRAL, hi, n=128), axis=0)

        a.fill_between(S.MG, e_lo, e_hi, color=col, alpha=.18, lw=0, zorder=1)
        a.fill_between(S.MG, q[0], q[2], color=col, alpha=.45, lw=0, zorder=2)
        a.plot(S.MG, q[1], color=col, lw=1.25, zorder=2.5, label="This work")
        a.plot(S.MG, pc, color=col, lw=1.0, ls="--", zorder=2.2, label="PARSEC")
        b.fill_between(S.MG, 100 * (e_lo / pc - 1), 100 * (e_hi / pc - 1), color=col,
                       alpha=.18, lw=0, zorder=1)
        b.fill_between(S.MG, rq[0], rq[2], color=col, alpha=.45, lw=0, zorder=2)
        b.plot(S.MG, rq[1], color=col, lw=1.25, zorder=2.5)
        b.axhline(0, color="0.5", lw=1.0, zorder=1.5)

        pm = pts[(pts.feh >= lo) & (pts.feh < hi)]
        sm = seis[(seis.feh >= lo) & (seis.feh < hi)]
        a.errorbar(pm.MG, pm.mass_allocated_msun, yerr=pm.mass_allocated_error_msun, marker="o",
                   ms=4.0, ls="none", color="0.15", ecolor="0.15", mec="black", mew=0.5,
                   elinewidth=0.8, zorder=4.2, label=f"Mann+19 (N={len(pm)})")
        a.errorbar(sm.MG, sm.Mass, yerr=[sm.e_Mass, sm.E_Mass], marker="D", ms=4.6, ls="none",
                   color=S.C_SEIS, ecolor=S.C_SEIS, mec="black", mew=0.5, elinewidth=0.8,
                   zorder=4.6, label=f"Serenelli+17 (N={len(sm)})")
        b.plot(pm.MG, pm.rel, "o", ms=3.4, color="0.15", mec="black", mew=0.5, zorder=4.2)
        b.plot(sm.MG, sm.rel, "D", ms=3.8, color=S.C_SEIS, mec="black", mew=0.5, zorder=4.6)
        xm = xi[(xi.feh >= lo) & (xi.feh < hi)]
        a.errorbar(xm.MG, xm.mass, yerr=xm.e_mass, marker="^", ms=5.0, ls="none",
                   color=S.C_XIONG, ecolor=S.C_XIONG, mec="black", mew=0.5, elinewidth=0.8,
                   zorder=4.4, label=f"Xiong+23 EBs (N={len(xm)})")
        b.plot(xm.MG, xm.rel, "^", ms=4.2, color=S.C_XIONG, mec="black", mew=0.5, zorder=4.4)

        a.set_yscale("log")
        a.set(ylim=(0.085, 2.0), xlim=S.XLIM[::-1])  # inverted M_G axis
        a.set_yticks([0.1, 0.2, 0.5, 1.0])
        a.set_yticklabels(["0.1", "0.2", "0.5", "1"])
        b.set(ylim=(-38, 38))
        a.text(0.03, 0.95, rf"${lo:+.2g}\leq Z<{hi:+.2g}$",
               transform=a.transAxes, ha="left", va="top", fontsize=10)
        a.legend(frameon=False, loc="lower right", fontsize=7.6, handlelength=1.8,
                 borderaxespad=0.2, labelspacing=0.25)
        plt.setp(a.get_xticklabels(), visible=False)
        if k % 2:
            plt.setp(a.get_yticklabels(), visible=False)
            plt.setp(b.get_yticklabels(), visible=False)
        else:
            a.set_ylabel(r"Mass [$M_\odot$]")
            b.set_ylabel("vs PARSEC (%)", fontsize=9.5)
        if k // 2:
            b.set_xlabel(r"$M_{\mathrm{G}}$ [mag]")
        else:
            plt.setp(b.get_xticklabels(), visible=False)

    fig.align_ylabels([ax for pair in axes[::2] for ax in pair])
    for ext in ("pdf", "png"):
        fig.savefig(S.OUT / f"mlr_feh_bins.{ext}", dpi=250)
        fig.savefig(S.PAPER_FIG / f"mlr_feh_bins.{ext}", dpi=250)
    plt.close(fig)
    counts = {f"{lo:+.2f}..{hi:+.2f}": {"mann": int(((pts.feh >= lo) & (pts.feh < hi)).sum()),
                                         "serenelli": int(((seis.feh >= lo) & (seis.feh < hi)).sum())}
              for lo, hi in BINS}
    print(counts)


if __name__ == "__main__":
    main()
