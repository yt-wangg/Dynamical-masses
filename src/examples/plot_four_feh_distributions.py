"""Plot raw and b_G-corrected XP metallicity distributions on one axis."""
from pathlib import Path
import json

import numpy as np
from astropy.table import Table
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
FIT_DIR = ROOT / "results/xp_simple_student_t_feh_jcaps_20260910"
SOURCE = ROOT / "data/jd_msms_single_bic_1kpc_filtered_cmdcut_cutb_jcaps_err_mhcal_good.fits"
OUT = ROOT / "results/feh_raw_corrected_four_distributions_20260917"


def summary(values):
    q16, med, q84 = np.quantile(values, [0.16, 0.50, 0.84])
    return {
        "n": int(values.size),
        "min": float(values.min()),
        "max": float(values.max()),
        "q16": float(q16),
        "median": float(med),
        "q84": float(q84),
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fit = np.load(FIT_DIR / "fit_arrays.npz")
    model_summary = json.loads((FIT_DIR / "summary.json").read_text())
    source_rows = np.asarray(fit["source_row_index"], dtype=int)
    table = Table.read(SOURCE)

    knots = np.asarray(model_summary["bias_knots"], dtype=float)
    bias_values = np.asarray(model_summary["bias_values"], dtype=float)
    raw = {}
    corrected = {}
    for j in (1, 2):
        z = np.asarray(table[f"feh_jcaps_{j}"], dtype=float)[source_rows]
        g = np.asarray(table[f"absg{j}"], dtype=float)[source_rows]
        bias = np.interp(g, knots, bias_values)
        raw[j] = z
        corrected[j] = z - bias

    groups = {
        "primary_raw": raw[1],
        "primary_corrected": corrected[1],
        "secondary_raw": raw[2],
        "secondary_corrected": corrected[2],
    }
    if any(not np.all(np.isfinite(v)) for v in groups.values()):
        raise ValueError("Non-finite values found in one or more plotted groups")
    if any(v.size != source_rows.size for v in groups.values()):
        raise ValueError("Group size does not match the saved source-row selection")
    if source_rows.size != 13721:
        raise ValueError(f"Expected 13721 selected rows, found {source_rows.size}")

    all_values = np.concatenate(list(groups.values()))
    # One common set of edges makes the four density curves directly comparable.
    lo, hi = np.floor(all_values.min() * 20) / 20, np.ceil(all_values.max() * 20) / 20
    bins = np.linspace(lo, hi, 101)

    # Color encodes the metallicity state, while line style encodes the
    # component, so the same visual mapping applies to all four curves.
    colors = {"raw": "black", "corrected": "#e08214"}
    styles = {"primary": "-", "secondary": "-."}
    labels = {
        "primary_raw": "Primary raw",
        "primary_corrected": "Primary after subtracting $b_G$",
        "secondary_raw": "Secondary raw",
        "secondary_corrected": "Secondary after subtracting $b_G$",
    }
    max_density = max(
        float(np.histogram(values, bins=bins, density=True)[0].max())
        for values in groups.values()
    )
    fig, axes = plt.subplots(2, 1, figsize=(9.2, 8.6), sharex=True, sharey=True)
    for ax, component in zip(axes, ("primary", "secondary")):
        for y, state in zip((0.94, 0.84), ("raw", "corrected")):
            key = f"{component}_{state}"
            values = groups[key]
            stats = summary(values)
            label = (f"{labels[key]}: median {stats['median']:+.3f} dex "
                     f"[{stats['q16']:+.3f}, {stats['q84']:+.3f}]")
            ax.hist(values, bins=bins, density=True, histtype="step",
                    color=colors[state], linestyle=styles[component],
                    linewidth=1.2 if state == "raw" else 1.8,
                    label=label)
            ax.errorbar(stats["median"], y * max_density,
                        xerr=[[stats["median"] - stats["q16"]],
                              [stats["q84"] - stats["median"]]],
                        fmt="o", markersize=4, capsize=3, color=colors[state],
                        linestyle="none", alpha=0.9,
                        markerfacecolor="white" if state == "raw" else colors[state],
                        markeredgewidth=1.1)
        ax.set_title(component.capitalize())
        ax.set_ylabel("Density")
        ax.set_xlim(bins[0], bins[-1])
        ax.set_ylim(0, max_density * 1.08)
        ax.grid(axis="y", alpha=0.2)
        ax.legend(loc="upper right", fontsize=8.5, frameon=True)
    axes[0].text(0.01, 0.98, "Markers: median and 16–84% interval",
                 transform=axes[0].transAxes, ha="left", va="top", fontsize=8.5,
                 color="0.25")
    axes[-1].set_xlabel("XP metallicity $z$ [dex]")
    fig.suptitle("XP metallicity distributions before and after $b_G$ correction")
    fig.tight_layout()
    figure_path = OUT / "four_feh_distributions.png"
    fig.savefig(figure_path, dpi=220)
    plt.close(fig)

    stats = {
        "source": str(SOURCE),
        "fit_directory": str(FIT_DIR),
        "n_selected": int(source_rows.size),
        "source_rows_unique": int(np.unique(source_rows).size),
        "common_bin_count": int(len(bins) - 1),
        "common_bin_edges": [float(x) for x in bins],
        "bias_knots": knots.tolist(),
        "bias_values": bias_values.tolist(),
        "groups": {key: summary(values) for key, values in groups.items()},
        "figure": str(figure_path),
    }
    (OUT / "stats.json").write_text(json.dumps(stats, indent=2))
    print(figure_path)
    print(OUT / "stats.json")


if __name__ == "__main__":
    main()
