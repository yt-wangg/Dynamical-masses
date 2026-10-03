"""Plot intrinsic normalized good-shape curves for the EM-like and T8 runs.

The figure compares analytic good-shape bases on a common finite support.  It
does not refit any model or apply the Rice measurement convolution.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_EM = ROOT / "results/em_solar_only_full_rebuild_20260928_b_relaxed_cont40"
DEFAULT_T8 = ROOT / "results/t8_raw_u_solar_only_recalibrated_baseline_20260924"
DEFAULT_OUT = ROOT / "results/em_t8_hwang_good_shape_comparison_20260929"


def density_basis(u: np.ndarray, shape: tuple[float, float, float]) -> np.ndarray:
    """Return the unnormalized good-shape basis for (B, uc, C)."""
    B, uc, C = shape
    u = np.asarray(u, dtype=float)
    exponent = -B * u**2 - np.exp((u - uc) / C)
    return np.where(u >= 0.0, u * np.exp(exponent), 0.0)


def normalized_curve(
    u: np.ndarray, shape: tuple[float, float, float]
) -> tuple[np.ndarray, float]:
    basis = density_basis(u, shape)
    raw_integral = float(np.trapezoid(basis, u))
    if not np.isfinite(raw_integral) or raw_integral <= 0.0:
        raise ValueError(f"Invalid raw integral for shape {shape}: {raw_integral}")
    density = basis / raw_integral
    return density, raw_integral


def read_shapes(em_dir: Path, t8_dir: Path) -> tuple[dict[str, tuple[float, float, float]], dict]:
    summary = json.loads((em_dir / "summary.json").read_text())
    states = np.load(em_dir / "map_states.npz")
    shapes: dict[str, tuple[float, float, float]] = {}
    cross_checks: dict[str, dict[str, object]] = {}
    for run_name in ("default", "S2"):
        summary_shape = tuple(float(v) for v in summary["results"][run_name]["final_shape"])
        vector_shape = tuple(float(v) for v in np.exp(states[f"{run_name}_vector"][35:38]))
        difference = np.asarray(summary_shape) - np.asarray(vector_shape)
        if not np.allclose(summary_shape, vector_shape, rtol=2e-8, atol=2e-10):
            raise ValueError(f"{run_name} summary/vector shape mismatch: {summary_shape} vs {vector_shape}")
        shapes[f"EM-like {run_name}"] = summary_shape
        cross_checks[run_name] = {
            "summary_shape": list(summary_shape),
            "map_vector_indices": [35, 36, 37],
            "map_vector_log_shape": states[f"{run_name}_vector"][35:38].tolist(),
            "map_shape_after_exp": list(vector_shape),
            "max_abs_difference": float(np.max(np.abs(difference))),
            "converged": bool(summary["results"][run_name]["converged"]),
        }

    model = json.loads((t8_dir / "mlr_model.json").read_text())
    constants = model["good_shape_constants"]
    shapes["T8 recalibrated baseline"] = tuple(float(constants[key]) for key in ("B", "uc", "C"))
    shapes["Hwang et al. (2023)"] = (0.00224, 36.09, 3.85)
    metadata = {
        "em_summary": str((em_dir / "summary.json").resolve()),
        "em_map_states": str((em_dir / "map_states.npz").resolve()),
        "t8_model": str((t8_dir / "mlr_model.json").resolve()),
        "cross_checks": cross_checks,
        "t8_good_shape_constants": constants,
    }
    return shapes, metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--em-output", type=Path, default=DEFAULT_EM)
    parser.add_argument("--t8-output", type=Path, default=DEFAULT_T8)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--support-max", type=float, default=80.0)
    parser.add_argument("--n-grid", type=int, default=200001)
    args = parser.parse_args()
    if args.support_max <= 0 or args.n_grid < 1001:
        raise ValueError("support-max must be positive and n-grid must be at least 1001")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    shapes, metadata = read_shapes(args.em_output, args.t8_output)
    u = np.linspace(0.0, args.support_max, args.n_grid)
    curves: dict[str, np.ndarray] = {}
    raw_integrals: dict[str, float] = {}
    normalized_integrals: dict[str, float] = {}
    metrics: dict[str, dict[str, float]] = {}
    for name, shape in shapes.items():
        curve, raw_integral = normalized_curve(u, shape)
        curves[name] = curve
        raw_integrals[name] = raw_integral
        normalized_integrals[name] = float(np.trapezoid(curve, u))
        cdf = np.empty_like(curve)
        cdf[0] = 0.0
        cdf[1:] = np.cumsum(0.5 * (curve[1:] + curve[:-1]) * np.diff(u))
        metrics[name] = {
            "B": shape[0],
            "uc": shape[1],
            "C": shape[2],
            "raw_basis_integral_0_80": raw_integral,
            "normalized_integral_0_80": normalized_integrals[name],
            "mode": float(u[np.argmax(curve)]),
            "median": float(np.interp(0.5, cdf / cdf[-1], u)),
        }

    plotted = {
        "EM-like default": (r"variable $p(\tilde{u})$", "#c44e52", {"lw": 2.8, "ls": "-"}),
        "T8 recalibrated baseline": (r"our fixed $p(\tilde{u})$", "#3568a8", {"lw": 2.3, "ls": ":"}),
        "Hwang et al. (2023)": (r"Hwang's fixed $p(\tilde{u})$", "#d27a28", {"lw": 2.3, "ls": "-."}),
    }
    plt.rcParams.update({"font.size": 12, "axes.grid": True, "grid.alpha": 0.22})
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    for name, (label, color, style) in plotted.items():
        ax.plot(u, curves[name], color=color, label=label, **style)
    ax.set_xlim(0.0, args.support_max)
    ax.set_xlabel(r"Scaled velocity $\tilde{u}$ (km s$^{-1}$ AU$^{1/2}$ $M_\odot^{-1/2}$)", fontsize=13)
    ax.set_ylabel(r"Normalized good-shape density $p_g(\tilde{u})$", fontsize=13)
    ax.set_title(r"Intrinsic good-shape comparison on $0 \leq \tilde{u} \leq 80$", fontsize=14)
    ax.tick_params(axis="both", labelsize=11)
    ax.legend(loc="upper right", fontsize=11, frameon=True)
    fig.tight_layout()
    fig.savefig(args.output_dir / "em_t8_hwang_good_shape_comparison.png", dpi=240, bbox_inches="tight")
    fig.savefig(args.output_dir / "em_t8_hwang_good_shape_comparison.pdf", bbox_inches="tight")
    plt.close(fig)

    report = {
        "support_max": args.support_max,
        "n_grid": args.n_grid,
        "normalization": "Each analytic basis is normalized independently by trapezoid integration on [0, 80].",
        "shapes": {name: {"B": shape[0], "uc": shape[1], "C": shape[2]} for name, shape in shapes.items()},
        "raw_basis_integrals": raw_integrals,
        "normalized_integrals": normalized_integrals,
        "metrics": metrics,
        "sources": metadata,
        "interpretation": "Intrinsic analytic good-shape comparison only; no Rice convolution, outlier mixture, or refit.",
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
