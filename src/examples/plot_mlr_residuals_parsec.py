"""Plot formal A/B mass relations and fractional residuals relative to PARSEC."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "results/t8_mass_independent_outlier_formal_20260919"
BASELINE = ROOT / "results/hierarchical_metallicity_t8_1_formal"


def main():
    with np.load(BASELINE / "mlr_derived_grid_t8.npz") as saved:
        old = dict(saved)
    with np.load(OUTPUT / "mlr_derived_grid_t8.npz") as saved:
        new = dict(saved)
    for key in ["absg_grid", "z_grid", "parsec_mass", "percentiles"]:
        np.testing.assert_allclose(old[key], new[key], rtol=1e-10, atol=1e-12)
    x, z = new["absg_grid"], new["z_grid"]
    plt.rcParams.update({"font.size": 12, "axes.labelsize": 13})
    fig, axes = plt.subplots(2, len(z), figsize=(17, 7), sharex=True,
                             sharey="row", layout="constrained")
    for iz, mh in enumerate(z):
        reference = new["parsec_mass"][iz]
        for grid, label, color in [
            (old, "A: mass-scaled outliers", "#7b6ba8"),
            (new, "B: mass-independent outliers", "#da7030"),
        ]:
            mass = grid["mass_percentiles"][:, iz]
            residual = 100 * (mass / reference[None, :] - 1)
            for ax, values in [(axes[0, iz], mass), (axes[1, iz], residual)]:
                ax.plot(x, values[1], color=color, lw=2, label=label)
                ax.fill_between(x, values[0], values[2], color=color, alpha=0.2)
        axes[0, iz].plot(x, reference, "--", color="0.35", lw=2, label="PARSEC")
        axes[0, iz].set_title(f"[M/H] = {mh:+.1f}")
        axes[1, iz].axhline(0, ls="--", color="0.35", lw=1.5)
        axes[1, iz].set_xlabel(r"$M_G$ [mag]")
        axes[1, iz].set_xticks([4, 7, 10, 13])
    axes[0, 0].set_ylabel(r"Mass [$M_\odot$]")
    axes[1, 0].set_ylabel(r"$100\,(M/M_{\rm PARSEC}-1)$ [%]")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=3, frameon=False)
    for extension in ["png", "pdf"]:
        fig.savefig(OUTPUT / f"mlr_residuals_parsec.{extension}", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
