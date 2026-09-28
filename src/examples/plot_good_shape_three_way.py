"""Compare the current EM-like, fixed, and Hwang good-shape distributions.

The comparison is intentionally intrinsic: it evaluates the analytic good-shape
basis on 0 <= \tilde{u} <= 80 and normalizes each curve on that same interval.
It does not refit the EM-like model or include the Rice measurement convolution.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EM = DEFAULT_ROOT / "results/em_solar_only_full_rebuild_20260928_polish_server"
DEFAULT_OUT = DEFAULT_ROOT / "results/em_good_shape_three_way_20260928"
SUPPORT_MAX = 80.0


def density_basis(u: np.ndarray, shape: tuple[float, float, float]) -> np.ndarray:
    """Return the unnormalized p(tilde-u) basis for (B, uc, C)."""
    B, uc, C = shape
    u = np.asarray(u, dtype=float)
    exponent = -B * u**2 - np.exp((u - uc) / C)
    return np.where(u >= 0.0, u * np.exp(exponent), 0.0)


def normalized_curve(
    u: np.ndarray, shape: tuple[float, float, float]
) -> tuple[np.ndarray, float]:
    """Normalize a shape on the requested finite support."""
    basis = density_basis(u, shape)
    norm = float(np.trapezoid(basis, u))
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError(f"Invalid normalization for shape {shape}: {norm}")
    return basis / norm, norm


def cumulative_density(u: np.ndarray, density: np.ndarray) -> np.ndarray:
    """Return a cumulative trapezoid integral, normalized to one."""
    du = np.diff(u)
    cdf = np.empty_like(density)
    cdf[0] = 0.0
    cdf[1:] = np.cumsum(0.5 * (density[1:] + density[:-1]) * du)
    return cdf / cdf[-1]


def load_em_shapes(path: Path) -> tuple[dict[str, tuple[float, float, float]], dict]:
    """Read both EM-like endpoints and convergence metadata from summary.json."""
    summary_path = path / "summary.json"
    payload = json.loads(summary_path.read_text())
    shapes: dict[str, tuple[float, float, float]] = {}
    for run_name in ("default", "S2"):
        result = payload["results"][run_name]
        final = result["final_shape"]
        shapes[f"EM-like {run_name}"] = tuple(float(value) for value in final)
    return shapes, payload


def pairwise_metrics(
    names: list[str], densities: dict[str, np.ndarray], cdfs: dict[str, np.ndarray], u: np.ndarray
) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            delta = densities[left] - densities[right]
            tv = 0.5 * float(np.trapezoid(np.abs(delta), u))
            ks = float(np.max(np.abs(cdfs[left] - cdfs[right])))
            rows.append({"left": left, "right": right, "total_variation": tv, "ks_distance": ks})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--em-output", type=Path, default=DEFAULT_EM)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--support-max", type=float, default=SUPPORT_MAX)
    parser.add_argument("--n-grid", type=int, default=200001)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.support_max <= 0.0 or args.n_grid < 1001:
        raise ValueError("support-max must be positive and n-grid must be at least 1001")

    em_shapes, em_summary = load_em_shapes(args.em_output)
    fixed_shape = tuple(float(v) for v in em_summary["formal_fixed_shape_B_initial"])
    shapes: dict[str, tuple[float, float, float]] = {
        "Current EM-like (default)": em_shapes["EM-like default"],
        "Current EM-like (S2)": em_shapes["EM-like S2"],
        "Your fixed 1.5 arcsec": fixed_shape,
        "Hwang": (0.00224, 36.09, 3.85),
    }

    u = np.linspace(0.0, args.support_max, args.n_grid)
    densities: dict[str, np.ndarray] = {}
    cdfs: dict[str, np.ndarray] = {}
    raw_norms: dict[str, float] = {}
    for name, shape in shapes.items():
        densities[name], raw_norms[name] = normalized_curve(u, shape)
        cdfs[name] = cumulative_density(u, densities[name])

    fixed_name = "Your fixed 1.5 arcsec"
    metric_rows: list[dict[str, float | str | bool]] = []
    for name, shape in shapes.items():
        density = densities[name]
        cdf = cdfs[name]
        mode_index = int(np.argmax(density))
        median = float(np.interp(0.5, cdf, u))
        tail_30 = float(1.0 - np.interp(30.0, u, cdf))
        tail_40 = float(1.0 - np.interp(40.0, u, cdf))
        metric_rows.append(
            {
                "name": name,
                "B": shape[0],
                "uc": shape[1],
                "C": shape[2],
                "raw_basis_integral_0_80": raw_norms[name],
                "normalized_integral_0_80": float(np.trapezoid(density, u)),
                "mode": float(u[mode_index]),
                "median": median,
                "tail_probability_gt_30": tail_30,
                "tail_probability_gt_40": tail_40,
                "is_current_em_like": name.startswith("Current EM-like"),
            }
        )

    names = list(shapes)
    pairs = pairwise_metrics(names, densities, cdfs, u)
    default_name = "Current EM-like (default)"
    s2_name = "Current EM-like (S2)"
    default_s2 = next(
        row for row in pairs if {row["left"], row["right"]} == {default_name, s2_name}
    )

    colors = {
        "Current EM-like (default)": "#c44e52",
        "Current EM-like (S2)": "#c44e52",
        "Your fixed 1.5 arcsec": "#3568a8",
        "Hwang": "#d27a28",
    }
    styles = {
        "Current EM-like (default)": {"lw": 2.6, "ls": "-"},
        "Current EM-like (S2)": {"lw": 1.6, "ls": ":"},
        "Your fixed 1.5 arcsec": {"lw": 2.3, "ls": "--"},
        "Hwang": {"lw": 2.3, "ls": "-."},
    }
    plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.22})
    fig, (ax_pdf, ax_cdf) = plt.subplots(
        2, 1, figsize=(9.2, 7.2), gridspec_kw={"height_ratios": [1.2, 1.0]}, sharex=True
    )
    for name in names:
        ax_pdf.plot(u, densities[name], color=colors[name], label=name, **styles[name])
    ax_pdf.set_ylabel(r"Normalized $p(\tilde{u})$")
    ax_pdf.set_title(r"Intrinsic good-shape distributions on $0 \leq \tilde{u} \leq 80$")
    ax_pdf.legend(loc="upper right", fontsize=8, frameon=True)
    ax_pdf.text(
        0.66,
        0.65,
        "Provisional EM-like endpoints\nB at lower bound: 0.0007",
        transform=ax_pdf.transAxes,
        ha="left",
        va="top",
        fontsize=8.5,
        bbox={"facecolor": "white", "alpha": 0.88, "edgecolor": "0.75"},
    )
    for name, color, linestyle in (
        (default_name, colors[default_name], "-"),
        (s2_name, colors[s2_name], ":"),
        ("Hwang", colors["Hwang"], "-."),
    ):
        ax_cdf.plot(
            u,
            cdfs[name] - cdfs[fixed_name],
            color=color,
            ls=linestyle,
            lw=2.0 if name == default_name else 1.5,
            label=f"{name} minus fixed",
        )
    ax_cdf.axhline(0.0, color="#777777", lw=1.0)
    ax_cdf.set_xlabel(r"Scaled velocity $\tilde{u}$ (km s$^{-1}$ AU$^{1/2}$ $M_\odot^{-1/2}$)")
    ax_cdf.set_ylabel(r"$F(\tilde{u}) - F_{\rm fixed}(\tilde{u})$")
    ax_cdf.set_title("Cumulative difference relative to your fixed 1.5 arcsec shape")
    ax_cdf.legend(loc="lower right", fontsize=8, frameon=True)
    ax_cdf.set_xlim(0.0, args.support_max)
    ax_cdf.set_ylim(*np.array(ax_cdf.get_ylim()) * 1.12)
    fig.suptitle(r"Comparison of $p(\tilde{u})$ good-shape assumptions", fontsize=13, y=0.985)
    fig.subplots_adjust(left=0.11, right=0.98, bottom=0.09, top=0.88, hspace=0.42)
    fig.savefig(args.output_dir / "p_utilde_three_shape_comparison.png", dpi=240, bbox_inches="tight")
    fig.savefig(args.output_dir / "p_utilde_three_shape_comparison.pdf", bbox_inches="tight")
    plt.close(fig)

    with (args.output_dir / "shape_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metric_rows[0]))
        writer.writeheader()
        writer.writerows(metric_rows)
    with (args.output_dir / "pairwise_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(pairs[0]))
        writer.writeheader()
        writer.writerows(pairs)
    report = {
        "support_max": args.support_max,
        "n_grid": args.n_grid,
        "normalization": "Each analytic basis is normalized independently by trapezoid integration on [0, 80].",
        "em_output": str(args.em_output.resolve()),
        "em_converged": {name: bool(em_summary["results"][name]["converged"]) for name in ("default", "S2")},
        "shape_box_B_lower": math.exp(em_summary["settings"]["shape_box"]["log_b"][0]),
        "shapes": {
            name: {"B": shape[0], "uc": shape[1], "C": shape[2]} for name, shape in shapes.items()
        },
        "shape_metrics": metric_rows,
        "pairwise_metrics": pairs,
        "default_vs_S2": default_s2,
        "interpretation": "Intrinsic analytic good-shape comparison only; no Rice convolution, outlier mixture, or refit.",
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
