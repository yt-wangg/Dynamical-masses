#!/usr/bin/env python3
"""Paper figure (Appendix A): prior of the MLR surface projected to the mass--M_G plane at
three metallicities, compared with the posterior and PARSEC.

The prior is the 'raw-factor' prior on the sampled coordinates of the monotone surface (c0, a, b, r,
log lambda_x, log lambda_Z); the shrinkage, smoothness and solar-anchor factors act on the derived
surface and are not included (they are not normalised densities to draw from).  Restyled from
depracated/plot_em_raw_factor_prior_predictive.py, with the posterior of the adopted fit overlaid.
"""
from __future__ import annotations

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_mlr_summary_figure as S  # noqa: E402  (shared style, paths, helpers)
import compare_t8_mann2019_ks_local as L  # noqa: E402

Z_SLICES = (-0.5, 0.0, 0.5)
COLORS = {-0.5: "#414487", 0.0: "#22A884", 0.5: "#7AD151"}
N_PRIOR = 4000
SEED = 20261015

plt.rcParams.update({"font.size": 8, "axes.labelsize": 9, "legend.fontsize": 7.5,
                     "xtick.labelsize": 7.5, "ytick.labelsize": 7.5})


def main() -> None:
    L.T8_DIR = S.RUNS / S.CENTRAL
    mlr, _ = L.build_t8_model()
    rng = np.random.default_rng(SEED)
    mg = S.MG
    zs = np.asarray(Z_SLICES)
    c_centre = float(mlr.parsec_projection[0, 0])
    lam_x, lam_z = mlr.lambda_references()
    prior = np.empty((N_PRIOR, zs.size, mg.size))
    for i in range(N_PRIOR):
        params = {
            "c0": rng.normal(c_centre, 0.10),
            "a": rng.normal(0.0, 1.0, size=3),
            "b": rng.normal(0.0, 1.0, size=7),
            "r": rng.normal(0.0, 1.0, size=(7, 3)),
            "log_lambda_x": rng.normal(np.log(lam_x), 0.7),
            "log_lambda_z": rng.normal(np.log(lam_z), 0.7),
        }
        prior[i] = np.asarray(mlr.g_from_raw(mg[None, :], zs[:, None], params), dtype=float)
    if not np.all(np.isfinite(prior)):
        raise ValueError("Non-finite prior curves.")
    pq = np.percentile(10.0 ** prior, [2.5, 16, 50, 84, 97.5], axis=0)

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.9), sharex=True, sharey=True)
    for k, (ax, z) in enumerate(zip(axes, Z_SLICES)):
        col = COLORS[z]
        for d in range(60):
            ax.plot(mg, 10.0 ** prior[d, k], color="0.7", lw=0.3, alpha=0.35, zorder=1)
        ax.fill_between(mg, pq[0, k], pq[4, k], color=col, alpha=0.15, lw=0, zorder=2,
                        label="prior 95%")
        ax.fill_between(mg, pq[1, k], pq[3, k], color=col, alpha=0.30, lw=0, zorder=2.1,
                        label="prior 68%")
        ax.plot(mg, pq[2, k], color=col, lw=1.0, ls="-", zorder=2.5, label="prior median")
        post = np.percentile(S.draws(S.CENTRAL, z), [16, 50, 84], axis=0)
        ax.fill_between(mg, post[0], post[2], color="black", alpha=0.25, lw=0, zorder=3)
        ax.plot(mg, post[1], color="black", lw=1.2, zorder=3.5, label="posterior")
        ax.plot(mg, S.parsec_curve(z), color="0.35", lw=1.0, ls="--", zorder=3.2, label="PARSEC")
        ax.set_yscale("log")
        ax.set(xlim=S.XLIM, ylim=(2e-3, 8.0))
        ax.text(0.97, 0.95, rf"$Z={z:+.1f}$", transform=ax.transAxes,
                ha="right", va="top", fontsize=8)
        ax.set_xlabel(r"$M_{\mathrm{G}}$ [mag]")
    axes[0].set_ylabel(r"Mass $[M_{\mathrm{\odot}}]$")
    axes[0].legend(frameon=False, loc="lower left", fontsize=7, handlelength=1.8,
                   borderaxespad=0.3, labelspacing=0.2)
    fig.subplots_adjust(left=0.075, right=0.995, top=0.985, bottom=0.17, wspace=0.04)
    for ext in ("pdf", "png"):
        fig.savefig(S.OUT / f"prior_predictive.{ext}", dpi=250)
        fig.savefig(S.PAPER_FIG / f"prior_predictive.{ext}", dpi=250)
    plt.close(fig)
    print("saved", S.PAPER_FIG / "prior_predictive.pdf")
    print("prior median at M_G=5,9,13 for Z=0 (Msun):",
          [float(np.interp(m, mg, pq[2, 1])) for m in (5, 9, 13)])


if __name__ == "__main__":
    main()
