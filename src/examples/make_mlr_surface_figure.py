#!/usr/bin/env python3
"""Paper figure: the fitted MLR surface in the (M_G, [M/H]) plane.

(a) posterior median mass, (b) fit uncertainty (half-width of the 16-84% interval of the mass,
relative to the median), (c) correction relative to PARSEC at the same [M/H] (posterior median of
M / M_PARSEC - 1).  The adopted run (alpha_e = 1.233 p(u~) shape) is read from
results/ecc_shape_sensitivity_20261009; the model is rebuilt with plot_t8_mlr_surface_2d.  The dashed
contour encloses 95% of the sample components, with M_G of each star and the median of the shared
system metallicity posterior.  The M_G axis is inverted (bright stars on the right).
"""
from __future__ import annotations

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, TwoSlopeNorm
import numpy as np
import pandas as pd
from astropy.table import Table
from scipy.ndimage import gaussian_filter

sys.path.insert(0, str(Path(__file__).resolve().parent))
import plot_t8_mlr_surface_2d as P  # noqa: E402

ROOT = P.REPO_ROOT
RUN = ROOT / "results" / "ecc_shape_sensitivity_20261009" / "alpha_1.233"
METALLICITY = ROOT / "results" / "t8_selcut_20261001" / "latent_metallicity_summary_t8.csv"
DATA = ROOT / "data" / "jd_msms_single_bic_1kpc_filtered_cmdcut_cutb_jcaps_err_mhcal_good.fits"
OUT = ROOT / "results" / "mlr_surface_figure"
PAPER_FIG = ROOT / "docs" / "paper" / "t9_aa" / "figs"
N_DRAW, NX, NZ = 512, 201, 161
COVER_FRAC = 0.95
MASS_LEVELS = (0.15, 0.2, 0.3, 0.5, 0.7, 1.0)

plt.rcParams.update({
    "font.size": 10, "axes.labelsize": 11, "xtick.labelsize": 9.5, "ytick.labelsize": 9.5,
    "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True,
    "xtick.minor.visible": True, "ytick.minor.visible": True, "axes.linewidth": 0.9,
    "font.family": "serif", "mathtext.fontset": "dejavuserif",
})


def coverage(x, z):
    """Smoothed density of the sample components and the level enclosing COVER_FRAC of them."""
    tab = Table.read(DATA)
    met = pd.read_csv(METALLICITY)
    rows = met.row_index.to_numpy()
    mg = np.concatenate([np.asarray(tab["absg1"])[rows], np.asarray(tab["absg2"])[rows]])
    zz = np.concatenate([met.z_q50.to_numpy()] * 2)
    xe = np.linspace(x[0], x[-1], 81)
    ze = np.linspace(z[0], z[-1], 65)
    h, _, _ = np.histogram2d(mg, zz, [xe, ze])
    h = gaussian_filter(h, 1.2)
    v = np.sort(h.ravel())[::-1]
    level = v[np.searchsorted(np.cumsum(v), COVER_FRAC * v.sum())]
    return 0.5 * (xe[1:] + xe[:-1]), 0.5 * (ze[1:] + ze[:-1]), h.T, level


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    mlr, samples = P.load_model(RUN)
    x = np.linspace(float(mlr.knots_x[0]), float(mlr.knots_x[-1]), NX)
    z = np.linspace(float(mlr.knots_z[0]), float(mlr.knots_z[-1]), NZ)
    _, g, delta, _ = P.evaluate_draws(mlr, samples, x, z, N_DRAW)
    mass = 10.0**g
    q16, q50, q84 = np.percentile(mass, [16, 50, 84], axis=0)
    err = 100 * 0.5 * (q84 - q16) / q50
    corr = np.median(100 * (10.0**delta - 1), axis=0)
    cx, cz, dens, level = coverage(x, z)

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.0), sharey=True,
                             gridspec_kw={"wspace": 0.06})
    X, Z = np.meshgrid(x, z)
    im = [
        axes[0].pcolormesh(X, Z, q50, shading="auto", cmap="viridis",
                           norm=LogNorm(vmin=q50.min(), vmax=q50.max()), rasterized=True),
        axes[1].pcolormesh(X, Z, err, shading="auto", cmap="magma_r",
                           vmin=0, vmax=np.percentile(err, 99), rasterized=True),
    ]
    lim = np.ceil(np.max(np.abs(corr)) / 5) * 5
    im.append(axes[2].pcolormesh(X, Z, corr, shading="auto", cmap="RdBu_r",
                                 norm=TwoSlopeNorm(vmin=-lim, vcenter=0, vmax=lim), rasterized=True))
    cs = axes[0].contour(X, Z, q50, levels=MASS_LEVELS, colors="white", linewidths=0.6)
    axes[0].clabel(cs, fmt=lambda m: f"{m:g}", fontsize=7, inline_spacing=2)
    cs = axes[2].contour(X, Z, corr, levels=np.arange(-20, 21, 5), colors="0.2",
                         linewidths=0.5, linestyles="solid")
    axes[2].clabel(cs, fmt=lambda c: f"{c:+.0f}" if c else "0", fontsize=7, inline_spacing=2)

    labels = (r"Mass $[M_\odot]$", r"Fit uncertainty (%)", r"Relative to PARSEC (%)")
    for k, (ax, m, lab) in enumerate(zip(axes, im, labels)):
        ax.contour(cx, cz, dens, levels=[level], colors="k" if k else "white", linewidths=0.9,
                   linestyles="--")
        cb = fig.colorbar(m, ax=ax, location="top", pad=0.03, fraction=0.06, aspect=22)
        cb.set_label(lab, fontsize=9.5)
        cb.ax.tick_params(labelsize=8.5, direction="in")
        if k == 0:
            cb.set_ticks([0.1, 0.2, 0.5, 1.0])
            cb.set_ticklabels(["0.1", "0.2", "0.5", "1"])
            cb.minorticks_off()
        ax.set_xlim(x[-1], x[0])  # inverted M_G axis
        ax.set_ylim(z[0], z[-1])
        ax.set_xlabel(r"$M_{\mathrm{G}}$ [mag]")
        ax.text(0.04, 0.95, f"({'abc'[k]})", transform=ax.transAxes, ha="left", va="top",
                fontweight="bold", bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", pad=1.5))
    axes[0].set_ylabel(r"$[\mathrm{M/H}]$")
    fig.subplots_adjust(left=0.095, right=0.985, bottom=0.15, top=0.86)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"mlr_surface.{ext}", dpi=250)
        fig.savefig(PAPER_FIG / f"mlr_surface.{ext}", dpi=250)
    plt.close(fig)
    print(f"mass {q50.min():.3f}-{q50.max():.3f} Msun; fit uncertainty {err.min():.1f}-{err.max():.1f}%;"
          f" correction {corr.min():+.1f} to {corr.max():+.1f}%; coverage level {level:.2f}")


if __name__ == "__main__":
    main()
