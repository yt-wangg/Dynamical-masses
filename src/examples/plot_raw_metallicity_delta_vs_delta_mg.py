"""Plot raw XP metallicity differences against absolute-G differences.

The sample selection intentionally mirrors ``SimpleStudentT`` in
``fit_xp_simple_student_t.py`` but does not load or rerun the fitted model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = ROOT / "data/jd_msms_single_bic_1kpc_filtered_cmdcut_cutb_jcaps_err_mhcal_good.fits"
DEFAULT_OUTPUT = ROOT / "results/raw_metallicity_delta_vs_delta_mg_20260917"


REQUIRED_COLUMNS = [
    "feh_jcaps_1",
    "feh_jcaps_2",
    "jc_sigma_m_h_1",
    "jc_sigma_m_h_2",
    "absg1",
    "absg2",
    "jc_at_bound_bits_1",
    "jc_at_bound_bits_2",
]


def load_selected(source: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return raw z2-z1, absg2-absg1, and source row indices after final cuts."""
    table = Table.read(source)
    missing = [name for name in REQUIRED_COLUMNS if name not in table.colnames]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    z = np.column_stack([
        np.asarray(table["feh_jcaps_1"], dtype=float),
        np.asarray(table["feh_jcaps_2"], dtype=float),
    ])
    err = np.column_stack([
        np.asarray(table["jc_sigma_m_h_1"], dtype=float),
        np.asarray(table["jc_sigma_m_h_2"], dtype=float),
    ])
    mg = np.column_stack([
        np.asarray(table["absg1"], dtype=float),
        np.asarray(table["absg2"], dtype=float),
    ])
    finite = np.all(np.isfinite(np.column_stack([z, err, mg])), axis=1)
    in_range = np.all((mg >= 3.5) & (mg <= 13.5), axis=1)
    positive_errors = np.all(err > 0, axis=1)
    unbounded = (
        np.asarray(table["jc_at_bound_bits_1"]) == 0
    ) & (np.asarray(table["jc_at_bound_bits_2"]) == 0)
    keep = finite & in_range & positive_errors & unbounded
    rows = np.flatnonzero(keep)
    return z[rows, 1] - z[rows, 0], mg[rows, 1] - mg[rows, 0], rows


def equal_count_bins(x: np.ndarray, n_bins: int) -> tuple[np.ndarray, list[np.ndarray]]:
    """Make contiguous equal-count bins, retaining all points exactly once."""
    order = np.argsort(x, kind="mergesort")
    groups = [part for part in np.array_split(order, n_bins) if len(part)]
    edges = np.empty(len(groups) + 1, dtype=float)
    edges[0] = float(np.min(x))
    edges[-1] = float(np.max(x))
    for i, group in enumerate(groups[:-1], start=1):
        edges[i] = 0.5 * (x[group[-1]] + x[groups[i]][0])
    return edges, groups


def plot(source: Path, output: Path, n_bins: int = 10) -> dict:
    y, x, rows = load_selected(source)
    expected = 13721
    if len(x) != expected:
        raise RuntimeError(f"Final SimpleStudentT selection has {len(x)} pairs; expected {expected}")

    edges, groups = equal_count_bins(x, n_bins)
    stats = []
    violin_data = []
    centers = []
    for i, group in enumerate(groups):
        values = y[group]
        q16, median, q84 = np.quantile(values, [0.16, 0.5, 0.84])
        lo = float(edges[i])
        hi = float(edges[i + 1])
        stats.append({
            "bin": i + 1,
            "x_low": lo,
            "x_high": hi,
            "x_median": float(np.median(x[group])),
            "n": int(len(group)),
            "y_median": float(median),
            "y_q16": float(q16),
            "y_q84": float(q84),
        })
        violin_data.append(values)
        centers.append(float(np.median(x[group])))

    output.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11.5, 7.6), constrained_layout=True)
    hb = ax.hexbin(
        x,
        y,
        gridsize=(90, 40),
        mincnt=1,
        bins="log",
        cmap="viridis",
        linewidths=0.15,
        alpha=0.88,
        zorder=1,
    )
    colorbar = fig.colorbar(hb, ax=ax, pad=0.012)
    colorbar.set_label("Pairs per hexagon (log scale)")

    # Equal-count bins can have very different widths; explicitly scale each
    # violin to its local x-bin width so narrow bins remain readable.
    # Keep each violin inside its own quantile interval.  This matters for
    # the broad, sparse right-tail bin whose median is far from its midpoint.
    centers_array = np.asarray(centers)
    widths = 1.6 * np.minimum(centers_array - edges[:-1], edges[1:] - centers_array)
    violins = ax.violinplot(
        violin_data,
        positions=centers,
        widths=widths,
        showmeans=False,
        showmedians=False,
        showextrema=False,
        points=120,
    )
    for body in violins["bodies"]:
        body.set_facecolor("#f28e2b")
        body.set_edgecolor("#8c4c13")
        body.set_linewidth(0.7)
        body.set_alpha(0.28)

    medians = np.array([item["y_median"] for item in stats])
    q16 = np.array([item["y_q16"] for item in stats])
    q84 = np.array([item["y_q84"] for item in stats])
    ax.vlines(centers_array, q16, q84, color="#d62728", lw=2.0, zorder=5, label="16th–84th percentile")
    ax.plot(centers_array, medians, "o-", color="#d62728", ms=4.5, lw=1.6, zorder=6, label="Median")

    ax.axhline(0.0, color="0.25", lw=0.9, ls="--", alpha=0.75, zorder=3)
    ax.set_xlim(float(np.min(x)), float(np.max(x)))
    y_pad = 0.04 * (float(np.max(y)) - float(np.min(y)))
    ax.set_ylim(float(np.min(y)) - y_pad, float(np.max(y)) + y_pad)
    ax.set_xlabel(r"$\Delta M_G = M_{G,2} - M_{G,1}$ (mag)")
    ax.set_ylabel(r"Raw $z_2-z_1 = [Fe/H]_2 - [Fe/H]_1$ (dex)")
    ax.set_title("Raw XP metallicity difference versus absolute-G difference")
    ax.legend(loc="upper right", frameon=True, framealpha=0.9)
    ax.grid(color="0.85", lw=0.6, alpha=0.6)
    fig.savefig(output / "raw_metallicity_delta_vs_delta_mg.png", dpi=220)
    plt.close(fig)

    result = {
        "source": str(source),
        "selection": "Finite raw z/errors/M_G, positive raw errors, 3.5 <= both M_G <= 13.5, and both jc_at_bound_bits == 0.",
        "metallicity_columns": ["feh_jcaps_1", "feh_jcaps_2"],
        "error_columns": ["jc_sigma_m_h_1", "jc_sigma_m_h_2"],
        "x_definition": "absg2 - absg1",
        "y_definition": "feh_jcaps_2 - feh_jcaps_1",
        "n_pairs": int(len(x)),
        "n_bins": int(len(groups)),
        "bin_count_total": int(sum(item["n"] for item in stats)),
        "x_min": float(np.min(x)),
        "x_max": float(np.max(x)),
        "y_min": float(np.min(y)),
        "y_max": float(np.max(y)),
        "bins": stats,
        "source_rows_sha_note": "source_row_indices are saved only for reproducibility of this plotting run",
    }
    (output / "bin_stats.json").write_text(json.dumps(result, indent=2) + "\n")
    np.save(output / "source_row_indices.npy", rows)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bins", type=int, default=10)
    args = parser.parse_args()
    result = plot(args.source, args.output, args.bins)
    print(json.dumps({key: result[key] for key in ("n_pairs", "n_bins", "bin_count_total", "x_min", "x_max", "y_min", "y_max")}, indent=2))


if __name__ == "__main__":
    main()
