#!/usr/bin/env python3
"""Plot observed binary CMDs in latent-metallicity bins with PARSEC boundaries."""

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
from astropy.table import Table


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from binary_masses.hierarchical_metallicity import IsochroneColorSurfaceModel


DATA_PATH = (
    REPO_ROOT
    / "data"
    / "jd_msms_single_bic_1kpc_filtered_cmdcut_cutb_jcaps_err_mhcal_good.fits"
)
OUTPUT_PATH = (
    REPO_ROOT
    / "results"
    / "hierarchical_metallicity_t7_teff4730"
    / "cmd_by_zj_bin.png"
)
BIN_EDGES = np.array([-1.0, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6])


def main() -> None:
    table = Table.read(DATA_PATH)
    required = ("Zj", "bp_rp0_1", "bp_rp0_2", "absg1", "absg2")
    missing = [name for name in required if name not in table.colnames]
    if missing:
        raise ValueError(f"Input table is missing columns: {missing}.")

    zj = np.asarray(table["Zj"], dtype=float)
    color = np.column_stack(
        [np.asarray(table["bp_rp0_1"], dtype=float), np.asarray(table["bp_rp0_2"], dtype=float)]
    )
    absg = np.column_stack(
        [np.asarray(table["absg1"], dtype=float), np.asarray(table["absg2"], dtype=float)]
    )

    surface = IsochroneColorSurfaceModel.from_parsec_csv(
        REPO_ROOT / "data" / "PARSEC_logAge_6to10_0p5_MH_n1to0p6_0p2.csv",
        absg_grid=np.linspace(3.5, 13.5, 500),
    )
    mg_grid = surface.absg_grid_np

    fig, axes = plt.subplots(2, 4, figsize=(15, 8.5), sharex=True, sharey=True)
    axes = axes.ravel()
    density_norm = LogNorm(vmin=1)
    density_artists = []

    for index, (lower, upper) in enumerate(zip(BIN_EDGES[:-1], BIN_EDGES[1:])):
        ax = axes[index]
        if index == len(axes) - 1:
            selected = np.isfinite(zj) & (zj >= lower) & (zj <= upper)
            interval = rf"${lower:+.1f}\leq Z_j\leq {upper:+.1f}$"
        else:
            selected = np.isfinite(zj) & (zj >= lower) & (zj < upper)
            interval = rf"${lower:+.1f}\leq Z_j<{upper:+.1f}$"

        selected_color = color[selected].ravel()
        selected_absg = absg[selected].ravel()
        usable = np.isfinite(selected_color) & np.isfinite(selected_absg)
        density_artists.append(
            ax.hexbin(
                selected_color[usable],
                selected_absg[usable],
                gridsize=48,
                extent=(0.4, 3.5, 3.5, 13.5),
                mincnt=1,
                cmap="viridis",
                norm=density_norm,
                linewidths=0,
                rasterized=True,
            )
        )

        for metallicity, linestyle, label in (
            (lower, "-", "PARSEC lower edge"),
            (upper, "--", "PARSEC upper edge"),
        ):
            parsec_color = surface.color_from_absg_mh(
                mg_grid, np.full_like(mg_grid, metallicity)
            )
            ax.plot(
                parsec_color,
                mg_grid,
                color="black",
                linestyle=linestyle,
                linewidth=1.5,
                label=label,
                zorder=5,
            )

        ax.set_title(f"{interval}\nN = {np.sum(selected):,}", fontsize=10)
        ax.set_xlabel(r"$(G_{\rm BP}-G_{\rm RP})_0$ (mag)")
        ax.set_ylabel(r"$M_G$ (mag)")
        ax.grid(alpha=0.16, linewidth=0.6)
        ax.set_xlim(0.4, 3.5)
        ax.set_ylim(13.5, 3.5)

    density_norm.vmax = max(
        float(np.max(artist.get_array())) for artist in density_artists if artist.get_array().size
    )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.955),
        ncol=2,
        frameon=False,
    )
    fig.subplots_adjust(left=0.06, right=0.88, bottom=0.08, top=0.82, wspace=0.14, hspace=0.34)
    colorbar_ax = fig.add_axes((0.91, 0.18, 0.014, 0.58))
    colorbar = fig.colorbar(density_artists[0], cax=colorbar_ax)
    colorbar.set_label("Stars per hexagon")
    fig.suptitle("Observed CMDs by latent metallicity bin", y=0.995, fontsize=14)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
