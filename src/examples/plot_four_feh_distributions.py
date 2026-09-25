"""Plot primary/secondary XP metallicity distributions before and after ``b_G``."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

DEFAULT_RESULT_DIR = REPO_ROOT / "results" / "solar_bg466_zero_20260923"
DEFAULT_OUTPUT_NAME = "primary_secondary_feh_correction.png"
DEFAULT_STATS_NAME = "primary_secondary_feh_correction_stats.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT_DIR)
    parser.add_argument("--output-name", default=DEFAULT_OUTPUT_NAME)
    parser.add_argument("--stats-name", default=DEFAULT_STATS_NAME)
    return parser.parse_args()


def load_model_metadata(result_dir: Path) -> dict:
    """Load calibration metadata, falling back to posterior metadata."""
    model_path = result_dir / "calibration_model.json"
    if model_path.exists():
        return json.loads(model_path.read_text())
    posterior_path = result_dir / "latent_metallicity_weights_t8.npz"
    with np.load(posterior_path, allow_pickle=False) as posterior:
        return json.loads(str(posterior["metadata_json"]))


def resolve_input_path(metadata: dict) -> Path:
    raw_path = Path(metadata["input_path"])
    candidates = [raw_path] if raw_path.is_absolute() else [REPO_ROOT / raw_path]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Could not resolve input_path={metadata['input_path']!r}; tried "
        + ", ".join(str(path) for path in candidates)
    )


def summary(values: np.ndarray) -> dict:
    q16, median, q84 = np.quantile(values, [0.16, 0.50, 0.84])
    return {
        "n": int(values.size),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "q16": float(q16),
        "median": float(median),
        "q84": float(q84),
    }


def bias_from_metadata(absg: np.ndarray, metadata: dict) -> np.ndarray:
    """Evaluate the saved, solar-recentered b_G curve at absolute magnitude."""
    observation = metadata.get("metallicity_observation_model", metadata)
    knots = np.asarray(observation["bias_mg_knots"], dtype=float)
    values = np.asarray(observation["bias_values_dex"], dtype=float)
    bias = np.interp(np.asarray(absg, dtype=float), knots, values)
    solar_zero = float(np.interp(4.66, knots, values))
    if not np.isclose(solar_zero, 0.0, atol=1e-10):
        raise ValueError(f"Saved b_G curve is not solar-recentered: b_G(4.66)={solar_zero}")
    return bias


def format_label(name: str, values: np.ndarray) -> str:
    stats = summary(values)
    return (
        f"{name}: median {stats['median']:+.3f} dex "
        f"[{stats['q16']:+.3f}, {stats['q84']:+.3f}]"
    )


def main() -> None:
    args = parse_args()
    result_dir = args.result_dir.expanduser().resolve()
    result_dir.mkdir(parents=True, exist_ok=True)

    metadata = load_model_metadata(result_dir)
    posterior_path = result_dir / "latent_metallicity_weights_t8.npz"
    with np.load(posterior_path, allow_pickle=False) as posterior:
        row_indices = np.asarray(posterior["row_indices"], dtype=np.int64)

    metadata_n_systems = metadata.get("n_systems")
    if metadata_n_systems is not None and row_indices.size != int(metadata_n_systems):
        raise ValueError(
            f"Posterior row count {row_indices.size} does not match metadata n_systems={metadata_n_systems}"
        )
    if np.unique(row_indices).size != row_indices.size:
        raise ValueError("Saved row_indices are not unique")

    source_path = resolve_input_path(metadata)
    table = Table.read(source_path)
    if np.any(row_indices < 0) or np.any(row_indices >= len(table)):
        raise ValueError("Saved row_indices are outside the input FITS table")

    raw: dict[int, np.ndarray] = {}
    corrected: dict[int, np.ndarray] = {}
    bias: dict[int, np.ndarray] = {}
    for component in (1, 2):
        raw_values = np.asarray(table[f"feh_jcaps_{component}"], dtype=float)[row_indices]
        mg_values = np.asarray(table[f"absg{component}"], dtype=float)[row_indices]
        component_bias = bias_from_metadata(mg_values, metadata)
        corrected_values = raw_values - component_bias
        if not (
            np.all(np.isfinite(raw_values))
            and np.all(np.isfinite(mg_values))
            and np.all(np.isfinite(component_bias))
            and np.all(np.isfinite(corrected_values))
        ):
            raise ValueError(f"Non-finite values found for component {component}")
        raw[component] = raw_values
        bias[component] = component_bias
        corrected[component] = corrected_values

    all_values = np.concatenate([raw[1], corrected[1], raw[2], corrected[2]])
    low = np.floor(float(np.min(all_values)) * 20.0) / 20.0
    high = np.ceil(float(np.max(all_values)) * 20.0) / 20.0
    if not high > low:
        raise ValueError("Metallicity values do not span a usable histogram range")
    bins = np.linspace(low, high, 101)

    colors = {"raw": "#202020", "corrected": "#A65D3A"}
    line_styles = {1: "-", 2: "-."}
    labels = {
        (1, "raw"): "Primary raw",
        (1, "corrected"): r"Primary calibrated ($b_G(4.66)=0$)",
        (2, "raw"): "Secondary raw",
        (2, "corrected"): r"Secondary calibrated ($b_G(4.66)=0$)",
    }
    groups = {
        (1, "raw"): raw[1],
        (1, "corrected"): corrected[1],
        (2, "raw"): raw[2],
        (2, "corrected"): corrected[2],
    }
    max_density = max(
        float(np.histogram(values, bins=bins, density=True)[0].max())
        for values in groups.values()
    )

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(10.6, 9.0),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    for ax, component in zip(axes, (1, 2)):
        for state in ("raw", "corrected"):
            values = groups[(component, state)]
            ax.hist(
                values,
                bins=bins,
                density=True,
                histtype="step",
                color=colors[state],
                linestyle=line_styles[component],
                linewidth=1.35 if state == "raw" else 1.9,
                label=format_label(labels[(component, state)], values),
            )
            stats = summary(values)
            marker_fraction = {"raw": 0.94, "corrected": 0.84}[state]
            marker_y = marker_fraction * max_density
            ax.errorbar(
                stats["median"],
                marker_y,
                xerr=[[stats["median"] - stats["q16"]], [stats["q84"] - stats["median"]]],
                fmt="o",
                markersize=4.8,
                capsize=3.2,
                color=colors[state],
                linestyle="none",
                markerfacecolor="white" if state == "raw" else colors[state],
                markeredgewidth=1.1,
                zorder=4,
            )
        ax.set_title("Primary" if component == 1 else "Secondary", fontsize=15)
        ax.set_ylabel("Density", fontsize=13)
        ax.set_xlim(bins[0], bins[-1])
        ax.set_ylim(0.0, max_density * 1.08)
        ax.tick_params(labelsize=11)
        ax.grid(axis="y", alpha=0.20)
        ax.legend(loc="upper right", fontsize=10, frameon=True, framealpha=0.9)

    axes[0].text(
        0.01,
        0.98,
        "Markers: median and 16–84% interval",
        transform=axes[0].transAxes,
        ha="left",
        va="top",
        fontsize=10,
        color="#555555",
    )
    axes[-1].set_xlabel("XP metallicity $z$ [dex]", fontsize=13)
    fig.suptitle(
        r"XP metallicity distributions before and after $b_G$ correction",
        fontsize=17,
    )

    figure_path = result_dir / args.output_name
    fig.savefig(figure_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    observation = metadata.get("metallicity_observation_model", metadata)
    stats_payload = {
        "source": str(source_path),
        "result_directory": str(result_dir),
        "posterior": str(posterior_path),
        "row_alignment": "input FITS rows indexed by latent_metallicity_weights_t8.npz row_indices",
        "n_selected": int(row_indices.size),
        "source_rows_unique": int(np.unique(row_indices).size),
        "common_bin_count": int(len(bins) - 1),
        "common_bin_edges": [float(value) for value in bins],
        "correction_definition": "feh_jcaps - b_G(absg)",
        "bias_anchor": "b_G(M_G=4.66)=0",
        "bias_mg_knots": np.asarray(observation["bias_mg_knots"], dtype=float).tolist(),
        "bias_values_dex": np.asarray(observation["bias_values_dex"], dtype=float).tolist(),
        "groups": {
            "primary_raw": summary(raw[1]),
            "primary_corrected": summary(corrected[1]),
            "secondary_raw": summary(raw[2]),
            "secondary_corrected": summary(corrected[2]),
        },
        "max_abs_correction_identity_error": float(
            max(
                np.max(np.abs((raw[component] - corrected[component]) - bias[component]))
                for component in (1, 2)
            )
        ),
        "figure": str(figure_path),
    }
    stats_path = result_dir / args.stats_name
    stats_path.write_text(json.dumps(stats_payload, indent=2) + "\n")
    print(figure_path)
    print(stats_path)


if __name__ == "__main__":
    main()
