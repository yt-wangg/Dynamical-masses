"""Compare the saved alternating-MAP pilot with PARSEC at solar metallicity."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import run_hierarchical_metallicity_test as workflow
from run_em_mlr_pilot import vector_to_params


def main():
    root = Path(__file__).resolve().parents[2]
    output = root / "results/em_mlr_pilot_20260919_v7"
    surface, _ = workflow.build_surfaces()
    mlr = workflow._make_t8_mlr(surface, workflow.parse_args())
    x = np.linspace(3.5, 13.5, 401)
    reference = surface.mass_from_absg_mh(x, 0.0)
    fig, axes = plt.subplots(2, 1, figsize=(8.2, 7.5), sharex=True,
                             gridspec_kw={"height_ratios": [1.5, 1]},
                             layout="constrained")
    with np.load(output / "map_states.npz") as states:
        for name, color, label in [
            ("default", "#2479ad", "Default-shape start"),
            ("S2", "#e37727", "S2-shape start"),
        ]:
            mass = mlr.mass_from_absg_mh(x, 0.0, vector_to_params(states[name + "_vector"]))
            axes[0].plot(x, mass, color=color, lw=2.3, label=label)
            axes[1].plot(x, 100 * (mass / reference - 1), color=color, lw=2.3)
    axes[0].plot(x, reference, "--", color="0.35", lw=2, label="PARSEC")
    axes[1].axhline(0, ls="--", color="0.35", lw=1.5)
    axes[0].set_ylabel(r"Mass [$M_\odot$]")
    axes[1].set_ylabel(r"$100\,(M/M_{\rm PARSEC}-1)$ [%]")
    axes[1].set_xlabel(r"$M_G$ [mag]")
    axes[0].set_title("[M/H] = 0.0 · 2,000-system alternating-MAP pilot\nPreliminary endpoints; no posterior intervals", fontsize=12)
    axes[0].legend(frameon=False)
    for ax in axes:
        ax.grid(alpha=0.2)
        ax.tick_params(labelsize=11)
    for extension in ["png", "pdf"]:
        fig.savefig(output / f"em_parsec_solar.{extension}", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
