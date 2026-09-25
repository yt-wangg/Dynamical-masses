"""Mechanism diagnostic for sensitivity to the normal-component velocity shape.

This is a branch-only calculation.  It compares the normalized baseline and S2
good-component densities and evaluates their Rice-convolved likelihood for a
few observed binaries while varying only the trial total mass.  It does not
build a lookup or refit the MLR.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from binary_masses import hierarchical_metallicity as hm


SHAPES = {
    "Baseline": (0.002544, 35.67, 3.10),
    "S2": (0.00103, 40.98, 11.53),
}


def normalized_good_density(x: np.ndarray, shape: tuple[float, float, float]) -> tuple[np.ndarray, float]:
    """Return p_g(x) on [0, 80] and the raw basis normalization."""
    B, uc, C = shape
    A = hm.RICE_GOOD_A
    hm.set_good_shape_constants(B=B, uc=uc, C=C)
    # set_good_shape_constants updates the module-level normalization.
    raw = float(hm.RICE_GOOD_BASIS_RAW_INTEGRAL)
    x = np.asarray(x, dtype=float)
    density = (A / raw) * x * np.exp(-(B * x**2 + np.exp((x - uc) / C)))
    density = np.where((x > 0) & (x <= hm.RICE_GOOD_SUPPORT), density, 0.0)
    return density, raw


def choose_examples(u: np.ndarray, u_sigma: np.ndarray) -> list[int]:
    """Choose ordinary-error systems near representative good-component speeds."""
    chosen: list[int] = []
    for target in (10.0, 20.0, 30.0):
        candidates = np.flatnonzero(np.abs(u - target) <= 2.0)
        if candidates.size == 0:
            raise ValueError(f"No observed system near u={target}.")
        # Prefer small but non-pathological errors so the shape response is visible.
        score = np.abs(u[candidates] - target) + 0.12 * np.abs(u_sigma[candidates] - 2.0)
        chosen.append(int(candidates[np.argmin(score)]))
    return chosen


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=ROOT / "data/jd_msms_single_bic_1kpc_filtered_cmdcut_cutb_jcaps_err_mhcal_good.fits")
    parser.add_argument("--lookup", type=Path, default=ROOT / "results/t8_raw_u_solar_only_fixed_good_20260924/dynamics_likelihood_lookup_t8.npz")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/t8_good_shape_mass_sensitivity_20260924")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with fits.open(args.data, memmap=True) as hdul:
        table = hdul[1].data
        all_u = np.asarray(table["u"], dtype=float)
        all_u_sigma = np.asarray(table["u_sigma"], dtype=float)
    data_rows = np.arange(all_u.size, dtype=int)
    if args.lookup.exists():
        with np.load(args.lookup) as saved:
            data_rows = np.asarray(saved["row_indices"], dtype=int)
    u = all_u[data_rows]
    u_sigma = all_u_sigma[data_rows]
    valid = np.isfinite(u) & np.isfinite(u_sigma) & (u > 0) & (u_sigma > 0)
    data_rows, u, u_sigma = data_rows[valid], u[valid], u_sigma[valid]
    chosen = choose_examples(u, u_sigma)

    x = np.linspace(0.0, hm.RICE_GOOD_SUPPORT, 1200)
    density: dict[str, np.ndarray] = {}
    raw_norm: dict[str, float] = {}
    for name, shape in SHAPES.items():
        density[name], raw_norm[name] = normalized_good_density(x, shape)

    mass = np.unique(np.concatenate([np.linspace(0.70, 1.30, 19), [1.00, 1.10]]))
    log_likelihood: dict[str, np.ndarray] = {}
    for name, shape in SHAPES.items():
        hm.set_good_shape_constants(B=shape[0], uc=shape[1], C=shape[2])
        values = np.empty((len(chosen), mass.size), dtype=float)
        for i, row in enumerate(chosen):
            values[i] = [
                np.log(hm.rice_component_reference_integral(float(u[row]), float(u_sigma[row]), np.sqrt(m), component="good"))
                for m in mass
            ]
        log_likelihood[name] = values

    ref_index = int(np.argmin(np.abs(mass - 1.0)))
    delta = {name: values - values[:, [ref_index]] for name, values in log_likelihood.items()}

    plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.22})
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(9.2, 7.2), gridspec_kw={"height_ratios": [1.0, 1.35]})
    colors = ["#3568a8", "#4c9f70", "#c44e52", "#8a63a8"]
    ax0.plot(x, density["Baseline"], color="#3568a8", lw=2.2, label=r"Baseline $p_g(\tilde u)$: $B=0.002544$, $u_c=35.67$, $C=3.10$")
    ax0.plot(x, density["S2"], color="#d27a28", lw=2.2, ls="--", label=r"S2 $p_g(\tilde u)$: $B=0.00103$, $u_c=40.98$, $C=11.53$")
    ax0.set_ylabel(r"Normalized $p_g(\tilde u)$")
    ax0.set_xlabel(r"Scaled velocity $\tilde u$ (km s$^{-1}$ au$^{1/2}$)")
    ax0.set_title(r"Two normalized good-component shapes on $0 \leq \tilde u \leq 80$")
    ax0.legend(loc="upper right", fontsize=8, frameon=True)
    ax0.set_xlim(0, 80)

    for i, (row, color) in enumerate(zip(chosen, colors)):
        ax1.plot(mass, delta["Baseline"][i], color=color, lw=2.0, label=f"$u={u[row]:.1f}$, $\\sigma_u={u_sigma[row]:.1f}$")
        ax1.plot(mass, delta["S2"][i], color=color, lw=1.8, ls="--")
    ax1.axhline(0.0, color="#777777", lw=1.0)
    ax1.axvline(1.0, color="#555555", lw=1.2, ls=":", label=r"Reference mass $M_{\rm tot}=1.00\,M_\odot$")
    ax1.set_xlabel(r"Trial total mass $M_{\rm tot}$ ($M_\odot$)")
    ax1.set_ylabel(r"Good-branch $\Delta\log L$ from $1.00\,M_\odot$")
    ax1.set_title(r"Fixed observed $(u,\,\sigma_u)$: mass response of the good branch")
    ax1.legend(loc="lower left", fontsize=8, ncol=2, frameon=True)
    ax1.text(0.99, 0.03, "Solid = baseline; dashed = S2", transform=ax1.transAxes, ha="right", va="bottom", fontsize=8.5,
             bbox={"facecolor": "white", "alpha": 0.88, "edgecolor": "0.75"})
    fig.suptitle(r"Good-branch mass sensitivity to $p_g(\tilde u)$ shape", fontsize=13, y=0.985)
    fig.text(
        0.5, 0.932,
        r"$L_g(u\mid M_{\rm tot})=\int {\rm Rice}(u\mid v,\sigma_u)\,p_g(v/s)\,dv/s$, "
        r"$s=\sqrt{M_{\rm tot}/M_\odot}$; each lower-panel pair fixes one observed binary.",
        ha="center", fontsize=8.5,
    )
    fig.subplots_adjust(left=0.11, right=0.98, bottom=0.09, top=0.82, hspace=0.48)
    fig.savefig(args.output_dir / "good_shape_mass_sensitivity.png", dpi=240, bbox_inches="tight")
    fig.savefig(args.output_dir / "good_shape_mass_sensitivity.pdf", bbox_inches="tight")
    plt.close(fig)

    records = []
    for row in chosen:
        records.append({"row": int(data_rows[row]), "u": float(u[row]), "u_sigma": float(u_sigma[row])})
    summary = {
        "data": str(args.data.resolve()),
        "n_valid_systems": int(u.size),
        "support_max": 80.0,
        "reference_mass_msun": 1.0,
        "shapes": {name: {"B": s[0], "uc": s[1], "C": s[2], "raw_basis_integral": raw_norm[name]} for name, s in SHAPES.items()},
        "representative_systems": records,
        "delta_logL_at_1p10": {name: [float(values[i, np.argmin(np.abs(mass - 1.10))]) for i in range(len(chosen))] for name, values in delta.items()},
        "interpretation": "Good-component branch diagnostic only; no mixture fraction or MLR refit.",
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (args.output_dir / "representative_systems.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["row", "u_km_s_au_half", "u_sigma_km_s_au_half"])
        for item in records:
            writer.writerow([item["row"], item["u"], item["u_sigma"]])
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
