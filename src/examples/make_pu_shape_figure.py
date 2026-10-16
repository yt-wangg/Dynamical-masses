#!/usr/bin/env python3
"""p(u~) for eccentricity indices alpha_e = 1.0, 1.23 (adopted) and 1.5.

Shapes are Eq. (good) with the constants fitted by fit_shape_eccentricity_grid.py
(results/shape_eccentricity_grid/shapes.json), each normalised on (0, 80].  Upper panel: the densities;
lower panel: ratio to the adopted alpha_e = 1.23 shape where that density exceeds
2% of its peak.  Short vertical lines mark the medians.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SHAPES = ROOT / "results" / "shape_eccentricity_grid" / "shapes.json"
OUT = ROOT / "results" / "pu_shape_figure"
CASES = (("alpha_1", r"$\alpha_e=1.0$ (thermal)", ":", "#31678E"),
         ("alpha_1.233", r"$\alpha_e=1.23$ (adopted)", "-", "k"),
         ("alpha_1.5", r"$\alpha_e=1.5$", "--", "#B5521B"))
U_EDGE = 42.12  # sqrt(2 G Msun / AU) in km/s sqrt(AU)

plt.rcParams.update({
    "font.size": 10, "axes.labelsize": 11, "xtick.labelsize": 9.5, "ytick.labelsize": 9.5,
    "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True,
    "xtick.minor.visible": True, "ytick.minor.visible": True, "axes.linewidth": 0.9,
    "font.family": "serif", "mathtext.fontset": "dejavuserif",
})


def pdf(u, c):
    x = np.linspace(1e-6, 80, 160001)
    f = lambda w: w * np.exp(-c["B"] * w**2 - np.exp((w - c["uc"]) / c["C"]))
    return f(u) / np.trapezoid(f(x), x)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    shapes = json.loads(SHAPES.read_text())
    u = np.linspace(0, 48, 2401)
    ref = pdf(u, shapes["alpha_1.233"])
    fig, (a, b) = plt.subplots(2, 1, figsize=(3.6, 4.0), sharex=True,
                               gridspec_kw={"height_ratios": [2.0, 1.0], "hspace": 0.0})
    for key, label, ls, col in CASES:
        p = pdf(u, shapes[key])
        a.plot(u, p, ls=ls, color=col, lw=1.4, label=label)
        a.axvline(shapes[key]["median"], ymax=0.06, color=col, ls=ls, lw=1.2)
        ok = ref > 2e-2 * ref.max()  # ratio only where the densities are not negligible
        b.plot(u[ok], p[ok] / ref[ok], ls=ls, color=col, lw=1.4)
    for ax in (a, b):
        ax.axvline(U_EDGE, color="0.6", lw=0.8, zorder=0)
    a.set(ylabel=r"$p(\tilde u)$", ylim=(0, 0.066))
    a.text(U_EDGE - 0.6, 0.97 * a.get_ylim()[1], "energy limit", rotation=90, ha="right", va="top",
           fontsize=8, color="0.45")
    a.set_yticks([0.01, 0.02, 0.03, 0.04, 0.05])
    a.legend(frameon=False, loc="upper left", fontsize=8.5, borderaxespad=0.3)
    b.axhline(1, color="0.5", lw=0.8)
    b.set(xlabel=r"$\tilde u$ [km s$^{-1}$ AU$^{1/2}$]", ylabel=r"Ratio to $\alpha_e=1.23$",
          xlim=(0, 48), ylim=(0.8, 1.2))
    b.set_yticks([0.9, 1.0, 1.1])
    b.yaxis.label.set_size(9.5)
    fig.align_ylabels((a, b))
    fig.subplots_adjust(left=0.18, right=0.96, top=0.98, bottom=0.12)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"pu_shape_alpha.{ext}", dpi=250)
    plt.close(fig)
    for key, *_ in CASES:
        print(f"{key}: median u~ {shapes[key]['median']:.2f}")


if __name__ == "__main__":
    main()
