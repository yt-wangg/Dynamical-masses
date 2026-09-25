"""Visualize the mass coupling of the legacy scaled-velocity outlier term.

This is a branch-only diagnostic.  It reuses the accepted T8 lookup and does
not refit the MLR.  The legacy branch is the Rice convolution of the
truncated N(40, 13^2) density in ``tilde_u = u/sqrt(M_tot)`` with the
Jacobian ``1/sqrt(M_tot)``.  The current raw-u branch is shown as a flat
reference because its outlier likelihood is independent of mass.
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
from binary_masses.hierarchical_metallicity import rice_component_reference_integral


def interp_rows(grid: np.ndarray, values: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Linear interpolation in the lookup's raw ``s=sqrt(M_tot)`` coordinate."""
    return np.asarray([np.interp(x, grid, row) for row in values])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lookup",
        type=Path,
        default=ROOT / "results/hierarchical_metallicity_t8_20260913/dynamics_likelihood_lookup_t8.npz",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=ROOT / "data/jd_msms_single_bic_1kpc_filtered_cmdcut_cutb_jcaps_err_mhcal_good.fits",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results/t8_legacy_outlier_mass_coupling_20260923",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with np.load(args.lookup) as saved:
        row_indices = np.asarray(saved["row_indices"], dtype=int)
        sqrt_mass = np.asarray(saved["sqrt_mtot_grid"], dtype=float)
        log_bad = np.asarray(saved["log_bad"], dtype=float)
        metadata = json.loads(str(saved["metadata_json"]))
    for key, expected in (("outlier_u0", 40.0), ("outlier_sigma", 13.0), ("tilde_u_max", 80.0)):
        if not np.isclose(float(metadata[key]), expected):
            raise ValueError(f"Lookup {key}={metadata[key]} differs from plotted model {expected}.")
    with fits.open(args.data, memmap=True) as hdul:
        table = hdul[1].data
        u = np.asarray(table["u"][row_indices], dtype=float)
        u_sigma = np.asarray(table["u_sigma"][row_indices], dtype=float)

    # The reference mass is 1 Msun and the comparison is a 10% mass increase.
    # Both points are evaluated by interpolation in the saved lookup coordinate.
    reference_mass = 1.0
    comparison_mass = 1.10
    s_ref = np.sqrt(reference_mass)
    s_cmp = np.sqrt(comparison_mass)
    log_ref = interp_rows(sqrt_mass, log_bad, np.array([s_ref]))
    log_cmp = interp_rows(sqrt_mass, log_bad, np.array([s_cmp]))
    delta10 = (log_cmp - log_ref).ravel()

    positive_fraction = float(np.mean(delta10 > 0.0))
    quantiles = np.percentile(delta10, [1, 10, 25, 50, 75, 90, 99])
    summary = {
        "n_systems": int(delta10.size),
        "reference_mass_msun": reference_mass,
        "comparison_mass_msun": comparison_mass,
        "positive_fraction": positive_fraction,
        "delta_log_likelihood_quantiles": dict(zip(["p01", "p10", "p25", "p50", "p75", "p90", "p99"], quantiles)),
        "lookup_coordinate": "s=sqrt(M_tot/M_sun)",
        "legacy_density": "TN(40,13,[0,80]) in tilde_u=u/sqrt(M_tot), with Jacobian 1/s",
        "normalization": "Phi((80-40)/13)-Phi(-40/13)",
        "interpretation": "conditional outlier-branch contribution; not a net MLR mass shift",
    }

    # Choose one actual system near each illustrative observed-speed value.
    # Within a narrow u window, prefer an ordinary reported uncertainty.
    targets = [(20.0, "Slow"), (45.0, "Near crossover"), (80.0, "Fast")]
    chosen = []
    for target_u, _ in targets:
        candidates = np.flatnonzero(np.abs(u - target_u) <= 2.0)
        if candidates.size == 0:
            raise ValueError(f"No observed system near u={target_u}.")
        typical_sigma = float(np.median(u_sigma[candidates]))
        distance = np.abs(u[candidates] - target_u) / 2.0
        distance += np.abs(u_sigma[candidates] - typical_sigma) / max(typical_sigma, 1.0)
        chosen.append(int(candidates[np.argmin(distance)]))
    mass_curve = np.linspace(0.50, 2.25, 240)
    curve_delta = np.asarray([
        np.interp(np.sqrt(mass_curve), sqrt_mass, log_bad[idx]) - log_ref[idx, 0]
        for idx in chosen
    ])

    plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.22})
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11.4, 4.4), gridspec_kw={"width_ratios": [1.35, 1]})
    colors = ["#3568a8", "#666666", "#c44e52"]
    labels = [label for _, label in targets]
    for i, (idx, color, label) in enumerate(zip(chosen, colors, labels)):
        ax0.plot(mass_curve, curve_delta[i], color=color, lw=2.2,
                 label=f"{label} binary: $u={u[idx]:.1f}$")
        ax0.scatter([reference_mass, comparison_mass], [0.0, delta10[idx]], color=color, s=22, zorder=3)
    ax0.axhline(0, color="#178358", lw=2.5, ls=(0, (6, 3)),
                label=r"New raw-$u$ outlier: $\Delta\log L=0$", zorder=4)
    ax0.axvline(comparison_mass, color="#888888", lw=1, ls="--")
    ax0.set_xlabel(r"Total mass $M_{\rm tot}$ ($M_\odot$)")
    ax0.set_ylabel(r"Change in outlier $\log L$ from $1\,M_\odot$")
    ax0.set_title("One curve per observed binary")
    ax0.legend(frameon=True, fontsize=8, loc="lower right")

    ax1.scatter(u, delta10, s=7, alpha=0.16, color="#7b6ba8", linewidths=0)
    for idx, color in zip(chosen, colors):
        ax1.scatter(u[idx], delta10[idx], s=75, color=color, edgecolor="black", linewidth=0.7, zorder=5)
    ax1.axhline(0, color="#178358", lw=2.5, ls=(0, (6, 3)),
                label=r"New raw-$u$ outlier: $\Delta\log L=0$", zorder=4)
    ax1.set_xlim(0, min(110, float(np.max(u))))
    ax1.set_xlabel(r"Observed $u$ (km s$^{-1}$ au$^{1/2}$)")
    ax1.set_ylabel(r"Legacy $\Delta\log L$ for $M_{\rm tot}: 1.00\to1.10\,M_\odot$")
    ax1.set_title("All binaries; colored points match left")
    ax1.legend(frameon=True, fontsize=8, loc="upper left")
    ax1.text(0.97, 0.04, f"N = {delta10.size:,}; {100*positive_fraction:.1f}% positive overall",
             transform=ax1.transAxes, ha="right", va="bottom", fontsize=9,
             bbox={"facecolor": "white", "alpha": 0.88, "edgecolor": "0.75"})

    fig.suptitle(
        r"Legacy outlier: $L_{\rm old}(u\mid M_{\rm tot})=\int {\rm Rice}(u\mid v,\sigma_u)\,G_{\rm o}(v/s)\,dv/s$"
        "\n"
        r"New outlier: $L_{\rm new}(u)=\int {\rm Rice}(u\mid v,\sigma_u)\,G_{\rm o}(v)\,dv$",
        fontsize=11.5, y=0.98,
    )
    fig.text(0.5, 0.835,
             r"$s=\sqrt{M_{\rm tot}/M_\odot}$; $G_{\rm o}$ is Normal(40, 13$^2$) truncated to [0, 80]. Both panels show outlier-branch $\Delta\log L$ from $1\,M_\odot$.",
             ha="center", fontsize=8.2)
    fig.tight_layout()
    fig.savefig(args.output_dir / "legacy_outlier_mass_coupling.png", dpi=220, bbox_inches="tight")
    fig.savefig(args.output_dir / "legacy_outlier_mass_coupling.pdf", bbox_inches="tight")
    plt.close(fig)

    with (args.output_dir / "representative_systems.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["example", "lookup_row_index", "u_km_s_au_half", "u_sigma_km_s_au_half", "u_over_sigma", "delta_logL_1_to_1p1"])
        for (_, label), idx in zip(targets, chosen):
            writer.writerow([label, int(row_indices[idx]), float(u[idx]), float(u_sigma[idx]), float(u[idx] / u_sigma[idx]), float(delta10[idx])])

    # One direct adaptive integral checks the saved lookup interpolation and
    # makes the coordinate/Jacobian convention explicit in the output log.
    check_idx = chosen[-1]
    direct = rice_component_reference_integral(u[check_idx], u_sigma[check_idx], s_cmp, component="bad")
    lookup_value = float(interp_rows(sqrt_mass, log_bad, np.array([s_cmp]))[check_idx, 0])
    direct_log = float(np.log(direct))
    summary["reference_check"] = {
        "lookup_logL": lookup_value,
        "adaptive_logL": direct_log,
        "abs_difference": abs(lookup_value - direct_log),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
