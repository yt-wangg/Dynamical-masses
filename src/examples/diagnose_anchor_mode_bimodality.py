#!/usr/bin/env python3
"""Diagnose the two posterior modes in the saved hard-solar T8 fit.

This script is deliberately read-only with respect to the inference result.  It
reconstructs the pre-positive-slope hard-solar colour model saved in
``anchor_mode_solar_hard_20260922``; its four ``color_d`` values are polynomial
coefficients, not the newer positive knot-slope parameters.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from binary_masses.hierarchical_metallicity import (  # noqa: E402
    JCAPS_BIAS_MG_KNOTS,
    JCAPS_BIAS_VALUES,
    array_digest,
)
from run_hierarchical_metallicity_test import filter_data, load_real_data  # noqa: E402


SOLAR_MG = 4.66
SOLAR_COLOR = 0.818
Z_GRID_MIN = -1.0
Z_GRID_MAX = 0.6
COLOR_SURFACE_CENTER = 8.5
COLOR_SURFACE_HALF_RANGE = 5.0


def old_polynomial_basis(mg):
    x = (np.asarray(mg, dtype=float) - COLOR_SURFACE_CENTER) / COLOR_SURFACE_HALF_RANGE
    return np.stack([np.ones_like(x), x, x**2, x**3], axis=-1)


def old_solar_color_slope(color_d):
    return float(old_polynomial_basis(SOLAR_MG) @ np.asarray(color_d, dtype=float))


def old_color_at_mg_z(mg, z, color_a, color_d):
    basis = old_polynomial_basis(mg)
    solar_basis = old_polynomial_basis(SOLAR_MG)
    return (
        SOLAR_COLOR
        + np.sum((basis[..., 1:] - solar_basis[1:]) * np.asarray(color_a), axis=-1)
        + np.asarray(z) * np.sum(basis * np.asarray(color_d), axis=-1)
    )


def quantiles(values):
    values = np.asarray(values, dtype=float)
    return {
        "q16": float(np.percentile(values, 16)),
        "q50": float(np.percentile(values, 50)),
        "q84": float(np.percentile(values, 84)),
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=REPO_ROOT / "results" / "anchor_mode_solar_hard_20260922",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=REPO_ROOT / "data" / "jd_msms_single_bic_1kpc_filtered_cmdcut_cutb_jcaps_err_mhcal_good.fits",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "results" / "anchor_mode_solar_hard_20260922" / "diagnostic_bimodality_20260922",
    )
    return parser.parse_args()


def load_inputs(result_dir: Path, data_path: Path):
    posterior_path = result_dir / "latent_metallicity_weights_t8.npz"
    mcmc_path = result_dir / "calibration_mcmc_t8.npz"
    if not posterior_path.exists() or not mcmc_path.exists():
        raise FileNotFoundError("Saved posterior and MCMC files are required in result-dir.")
    with np.load(posterior_path, allow_pickle=False) as posterior:
        row_indices = np.asarray(posterior["row_indices"], dtype=np.int64)
        metadata = json.loads(str(posterior["metadata_json"].item()))
    with np.load(mcmc_path, allow_pickle=False) as saved:
        samples = {name: np.asarray(saved[name]) for name in saved.files}
    raw_arrays = load_real_data(data_path)
    arrays = filter_data(raw_arrays, max_systems=None, seed=20260819, fixed_rows=row_indices)
    input_digest = array_digest(
        row_indices,
        arrays["feh_observed"], arrays["feh_sigma"], arrays["absg"],
        arrays["color_observed"], arrays["color_sigma"],
    )
    return row_indices, metadata, samples, arrays, input_digest


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    row_indices, metadata, samples, arrays, input_digest = load_inputs(args.result_dir, args.data)
    delta = samples["posterior__delta_z"]
    color_a = samples["posterior__color_a"]
    color_d = samples["posterior__color_d"]
    weights = samples["posterior__population_weights"]
    energy = samples["sample_stats__potential_energy"]
    diverging = samples["sample_stats__diverging"]
    n_chains, n_draws = delta.shape
    chain_colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    chain_labels = [f"chain {index + 1}" for index in range(n_chains)]
    centers = np.linspace(Z_GRID_MIN, Z_GRID_MAX, weights.shape[-1])
    slopes = np.asarray([
        [old_solar_color_slope(draw) for draw in chain_draws]
        for chain_draws in color_d
    ])

    near_solar = (
        np.isfinite(arrays["absg"][:, 0]) & np.isfinite(arrays["absg"][:, 1])
        & ((arrays["absg"][:, 0] >= 4.4) & (arrays["absg"][:, 0] <= 4.9)
           | (arrays["absg"][:, 1] >= 4.4) & (arrays["absg"][:, 1] <= 4.9))
    )
    near_count = int(np.sum(near_solar))
    mg_values = arrays["absg"][near_solar]
    feh_values = arrays["feh_observed"][near_solar]
    color_values = arrays["color_observed"][near_solar]
    mg_mask = (mg_values >= 4.4) & (mg_values <= 4.9)
    z_corr = feh_values - np.interp(mg_values, JCAPS_BIAS_MG_KNOTS, JCAPS_BIAS_VALUES)

    summary = {
        "result_dir": str(args.result_dir),
        "data_path": str(args.data),
        "model_semantics": "pre-positive-slope hard-solar polynomial color_d; color_d is four cubic coefficients",
        "row_count": int(row_indices.size),
        "row_indices_min": int(row_indices.min()),
        "row_indices_max": int(row_indices.max()),
        "row_indices_digest": array_digest(row_indices),
        "input_subset_digest": input_digest,
        "saved_metadata_input_digest": metadata.get("input_data_digest"),
        "near_solar_mg_range": [4.4, 4.9],
        "near_solar_system_count": near_count,
        "near_solar_component_count": int(np.sum(mg_mask)),
        "chains": [],
    }
    for chain in range(n_chains):
        dominant = int(np.argmax(np.median(weights[chain], axis=0)))
        summary["chains"].append({
            "chain": chain + 1,
            "delta_z": quantiles(delta[chain]),
            "solar_dcolor_dz_mag_per_dex": quantiles(slopes[chain]),
            "dominant_population_component": dominant,
            "dominant_population_center_dex": float(centers[dominant]),
            "population_weight_medians": np.median(weights[chain], axis=0).tolist(),
            "population_weight_q16": np.percentile(weights[chain], 16, axis=0).tolist(),
            "population_weight_q84": np.percentile(weights[chain], 84, axis=0).tolist(),
            "potential_energy": quantiles(energy[chain]),
            "divergences": int(np.sum(diverging[chain])),
        })

    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
    ax = axes[0, 0]
    for chain in range(n_chains):
        ax.scatter(delta[chain], slopes[chain], s=5, alpha=0.18, color=chain_colors[chain], label=chain_labels[chain])
        ax.scatter(np.median(delta[chain]), np.median(slopes[chain]), s=55, color=chain_colors[chain], edgecolor="black", zorder=5)
    ax.set(xlabel="$\\delta_z$ [dex]", ylabel="$\\partial C_\\theta/\\partial Z$ at $M_G=4.66$ [mag dex$^{-1}$]")
    ax.set_title("Posterior modes: saved polynomial-slope fit")
    ax.legend(frameon=False)

    ax = axes[0, 1]
    for component, color in enumerate(("tab:blue", "tab:orange")):
        valid = mg_mask[:, component] & np.isfinite(z_corr[:, component]) & np.isfinite(color_values[:, component])
        ax.scatter(z_corr[valid, component], color_values[valid, component], s=7, alpha=0.28, color=color, label=f"component {component + 1}")
    xline = np.linspace(float(np.nanmin(z_corr[mg_mask])), float(np.nanmax(z_corr[mg_mask])), 100)
    for chain in range(n_chains):
        zoff = np.median(delta[chain])
        slope = np.median(slopes[chain])
        ax.plot(xline, SOLAR_COLOR + slope * (xline - zoff), color=chain_colors[chain], linewidth=2, label=f"{chain_labels[chain]} median map")
    ax.scatter([0.0], [SOLAR_COLOR], color="black", marker="*", s=80, zorder=6, label="solar reference")
    ax.set(xlabel="$b_G$-corrected observed XP [M/H] [dex]", ylabel="dereddened BP-RP [mag]")
    ax.set_title(f"Near-solar systems: {near_count} systems, {int(np.sum(mg_mask))} components")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1, 0]
    x = np.arange(weights.shape[-1])
    width = 0.8 / n_chains
    for chain in range(n_chains):
        med = np.median(weights[chain], axis=0)
        lo = med - np.percentile(weights[chain], 16, axis=0)
        hi = np.percentile(weights[chain], 84, axis=0) - med
        ax.errorbar(x + (chain - (n_chains - 1) / 2) * width, med, yerr=[lo, hi], fmt="o", color=chain_colors[chain], label=chain_labels[chain])
    ax.set_xticks(x, [f"{center:+.2f}" for center in centers])
    ax.set(xlabel="Fixed population center $Z$ [dex]", ylabel="population weight")
    ax.set_title("Population components by chain")
    ax.legend(frameon=False)

    ax = axes[1, 1]
    for chain in range(n_chains):
        ax.plot(delta[chain], color=chain_colors[chain], alpha=0.75, label=f"{chain_labels[chain]} $\\delta_z$")
    ax.set(xlabel="draw", ylabel="$\\delta_z$ [dex]")
    ax2 = ax.twinx()
    for chain in range(n_chains):
        ax2.plot(energy[chain], color=chain_colors[chain], alpha=0.18, linestyle="--")
    ax2.set_ylabel("potential energy [dashed]")
    ax.set_title("Chain traces; dashed lines show potential energy")
    ax.legend(frameon=False, fontsize=8, loc="upper right")

    fig.suptitle("Hard-solar T8 posterior bimodality diagnostic\n(saved pre-positive-slope polynomial semantics)", fontsize=13)
    fig.tight_layout()
    figure_path = args.output_dir / "anchor_mode_bimodality_diagnostic.png"
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)

    summary["figure"] = str(figure_path)
    (args.output_dir / "anchor_mode_bimodality_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    markdown = f"""# Hard-solar posterior bimodality diagnostic

本诊断只读取已有结果 `{args.result_dir}`，没有重新拟合。该结果是在新的正斜率 knot 参数化之前生成的，因此这里严格恢复旧的四参数多项式 `color_d` 语义；图和数值不能直接当作新版正斜率模型的结果。

使用原始 FITS 的 `{len(row_indices)}` 个系统，行号来自保存的 `latent_metallicity_weights_t8.npz`。row-index digest 为 `{summary['row_indices_digest']}`，输入子集 digest 为 `{input_digest}`。使用的 XP 金属丰度列是 `feh_jcaps_1/2`，并用固定的相对 `b_G(M_G)` 做横轴修正。

![bimodality diagnostic](anchor_mode_bimodality_diagnostic.png)

近太阳样本采用 `4.4 < M_G < 4.9`，共 `{near_count}` 个系统、`{int(np.sum(mg_mask))}` 个成员。第二幅图的曲线是每条链中值的
\[
C=0.818+s_{{\odot}}(z_{{\rm corr}}-Z_{{\rm off}}),
\]
其中 `s_sun` 来自旧多项式 `color_d` 在 `M_G=4.66` 的导数。它用于显示两条后验模式对观测的不同解释，不是新版正斜率模型的最终预测。

每条链的数值摘要保存在 `anchor_mode_bimodality_summary.json`。直接可见的诊断是：比较两条链的 `delta_z`、太阳处颜色斜率、总体权重、势能轨迹和 divergence；即使 divergence 为零，只要链中值和总体权重集中在不同区域，仍应把结果视为未收敛或多峰。

局限：该图只诊断已保存的两条链，没有增加采样、重跑数据或替换旧结果。新版硬太阳正斜率模型必须重新拟合后，才能判断这种模式结构是否保留。
"""
    (args.output_dir / "README.md").write_text(markdown, encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
