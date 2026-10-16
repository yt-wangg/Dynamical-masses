#!/usr/bin/env python3
"""Paper figure: CMD of the analysed binary components in bins of the system
metallicity (posterior median of Z_j), with PARSEC main sequences at the bin edges.

Derived from plot_cmd_by_zj_bin.py, restyled to match the other paper figures
(make_mlr_summary_figure.py) and laid out for a two-column (figure*) float.
Z_j is the median of the Stage-1b posterior probability mass over the metallicity grid.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
from astropy.table import Table

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_mlr_summary_figure as S  # noqa: E402  (shared rcParams, paths)

ROOT = S.ROOT
sys.path.insert(0, str(ROOT / "src"))
from binary_masses.hierarchical_metallicity import IsochroneColorSurfaceModel  # noqa: E402

DATA_PATH = ROOT / "data" / "jd_msms_single_bic_1kpc_filtered_cmdcut_cutb_jcaps_err_mhcal_good.fits"
POSTERIOR_PATH = ROOT / "results" / "solar_bg466_zero_20260923" / "latent_metallicity_weights_t8.npz"
BIN_EDGES = np.array([-1.0, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6])
XLIM = (0.4, 3.5)
YLIM = (13.5, 3.5)
C_PARSEC = "black"

plt.rcParams.update({"font.size": 8, "axes.labelsize": 9, "legend.fontsize": 8,
                     "xtick.labelsize": 7.5, "ytick.labelsize": 7.5})


def posterior_median(probabilities: np.ndarray, z_grid: np.ndarray) -> np.ndarray:
    """Median of the posterior probability mass on a grid (linear interpolation of the CDF)."""
    cdf = np.cumsum(probabilities, axis=1)
    cdf /= cdf[:, -1:]
    idx = np.clip((cdf < 0.5).sum(axis=1), 1, z_grid.size - 1)
    rows = np.arange(len(idx))
    lo, hi = cdf[rows, idx - 1], cdf[rows, idx]
    w = np.where(hi > lo, (0.5 - lo) / (hi - lo), 0.0)
    return z_grid[idx - 1] + w * (z_grid[idx] - z_grid[idx - 1])


def main() -> None:
    table = Table.read(DATA_PATH)
    color = np.column_stack([np.asarray(table["bp_rp0_1"], float), np.asarray(table["bp_rp0_2"], float)])
    absg = np.column_stack([np.asarray(table["absg1"], float), np.asarray(table["absg2"], float)])
    with np.load(POSTERIOR_PATH, allow_pickle=False) as saved:
        rows = np.asarray(saved["row_indices"], dtype=np.int64)
        z_grid = np.asarray(saved["z_grid"], dtype=np.float64)
        probabilities = np.asarray(saved["probabilities"], dtype=np.float64)
        json.loads(str(saved["metadata_json"]))
    if probabilities.shape != (rows.size, z_grid.size) or rows.max() >= len(table):
        raise ValueError("Posterior weights do not match the input table.")
    zj = np.full(len(table), np.nan)
    zj[rows] = posterior_median(probabilities, z_grid)

    surface = IsochroneColorSurfaceModel.from_parsec_csv(
        ROOT / "data" / "PARSEC_logAge_6to10_0p5_MH_n1to0p6_0p2.csv",
        absg_grid=np.linspace(3.5, 13.5, 500))
    mg = surface.absg_grid_np

    fig, axes = plt.subplots(2, 4, figsize=(7.2, 4.5), sharex=True, sharey=True)
    norm = LogNorm(vmin=1)
    arts = []
    for k, (ax, lo, hi) in enumerate(zip(axes.ravel(), BIN_EDGES[:-1], BIN_EDGES[1:])):
        last = k == len(axes.ravel()) - 1
        sel = np.isfinite(zj) & (zj >= lo) & ((zj <= hi) if last else (zj < hi))
        c, g = color[sel].ravel(), absg[sel].ravel()
        ok = np.isfinite(c) & np.isfinite(g)
        arts.append(ax.hexbin(c[ok], g[ok], gridsize=40, extent=(*XLIM, 3.5, 13.5), mincnt=1,
                              cmap="viridis", norm=norm, linewidths=0, rasterized=True))
        for z in BIN_EDGES:  # all bin edges in every panel, translucent grey
            ax.plot(surface.color_from_absg_mh(mg, np.full_like(mg, z)), mg, color="0.5",
                    lw=0.6, ls="-", alpha=0.45, zorder=4, label="all bin edges" if z == BIN_EDGES[0] else None)
        for z, ls, lab in ((lo, "-", "lower edge"), (hi, "--", "upper edge")):
            ax.plot(surface.color_from_absg_mh(mg, np.full_like(mg, z)), mg, color=C_PARSEC,
                    lw=1.5, ls=ls, label=lab, zorder=5)
        op = r"\leq" if last else "<"
        ax.text(0.97, 0.97, rf"${lo:+.1f}\leq Z_j{op}{{{hi:+.1f}}}$" + f"\n$N$ = {sel.sum():,}",
                transform=ax.transAxes, ha="right", va="top", fontsize=7, zorder=6)
        ax.set(xlim=XLIM, ylim=YLIM)
    norm.vmax = max(float(np.max(a.get_array())) for a in arts if a.get_array().size)
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$M_{\mathrm{G}}$ [mag]")
    for ax in axes[-1]:
        ax.set_xlabel(r"$(G_{\rm BP}-G_{\rm RP})_0$ [mag]")
    axes[0, 0].legend(frameon=False, loc="lower left", fontsize=6.5, handlelength=1.8,
                      borderaxespad=0.3, labelspacing=0.2)
    fig.subplots_adjust(left=0.065, right=0.915, top=0.985, bottom=0.095, wspace=0.04, hspace=0.04)
    cax = fig.add_axes((0.93, 0.095, 0.012, 0.89))
    fig.colorbar(arts[0], cax=cax).set_label("Stars per hexagon")
    for ext in ("pdf", "png"):
        fig.savefig(S.OUT / f"cmd_by_zj_bin.{ext}", dpi=250)
        fig.savefig(S.PAPER_FIG / f"cmd_by_zj_bin.{ext}", dpi=250)
    plt.close(fig)
    print("saved", S.PAPER_FIG / "cmd_by_zj_bin.pdf")


if __name__ == "__main__":
    main()
