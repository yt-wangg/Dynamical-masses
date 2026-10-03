"""Plot one-dimensional prior factors with markers for all 38 EM MAP coordinates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import betaln, ndtr


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "examples"))

import run_hierarchical_metallicity_test as workflow  # noqa: E402


DEFAULT_EM = ROOT / "results/em_solar_only_full_rebuild_20260928_b_relaxed_cont40"
DEFAULT_OUT = ROOT / "results/em_map_prior_factors_20260929"


def build_mlr():
    mass_surface, _ = workflow.build_surfaces()
    return workflow._make_t8_mlr(
        mass_surface,
        argparse.Namespace(
            mlr_knot_x=workflow.T8_MLR_DEFAULT_KNOT_X,
            mlr_knot_z=workflow.T8_MLR_DEFAULT_KNOT_Z,
            mlr_degree_x=3,
            mlr_degree_z=3,
            mlr_tau_d=0.10,
            mlr_tau_x=0.05,
            mlr_tau_z=0.05,
            mlr_tau_xz=0.05,
            mlr_solar_mean=0.0,
            mlr_solar_sigma=0.01 / np.log(10.0),
        ),
    )


def normal_density(x, mean, sigma):
    return np.exp(-0.5 * ((x - mean) / sigma) ** 2) / (sigma * np.sqrt(2.0 * np.pi))


def truncated_normal_density(x, mean, sigma, lo, hi):
    normalization = ndtr((hi - mean) / sigma) - ndtr((lo - mean) / sigma)
    return normal_density(x, mean, sigma) / normalization


def style_axis(ax, title, map_value, hard_bounds=None):
    ax.set_title(title, fontsize=10)
    ax.axvline(map_value, color="#c44e52", lw=1.8, label="MAP")
    if hard_bounds is not None:
        ax.axvline(hard_bounds[0], color="0.35", ls="--", lw=0.9)
        ax.axvline(hard_bounds[1], color="0.35", ls="--", lw=0.9)
    ax.tick_params(labelsize=8)
    ax.grid(alpha=0.2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--em-output", type=Path, default=DEFAULT_EM)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary = json.loads((args.em_output / "summary.json").read_text())
    with np.load(args.em_output / "map_states.npz", allow_pickle=False) as states:
        vector = np.asarray(states["default_vector"], dtype=float)
    mlr = build_mlr()
    lambda_ref_x, lambda_ref_z = mlr.lambda_references()
    shape_box = summary["settings"]["shape_box"]
    shape_centers = np.log([0.002544, 35.67, 3.1])
    shape_sigmas = np.log([2.0, 1.3, 2.0])
    shape_bounds = np.asarray([
        [shape_box["log_b"][0], shape_box["log_b"][2]],
        [shape_box["log_uc"][0], shape_box["log_uc"][2]],
        [shape_box["log_c"][0], shape_box["log_c"][2]],
    ])

    coordinate_rows = []
    def add_normal(name, value, mean, sigma, group):
        coordinate_rows.append({"name": name, "value": float(value), "prior": f"Normal({mean:.8g},{sigma:.8g})", "z_score": float((value - mean) / sigma), "group": group})

    add_normal("c0", vector[0], mlr.parsec_projection[0, 0], 0.10, "mlr")
    for i, value in enumerate(vector[1:4]): add_normal(f"a[{i}]", value, 0.0, 1.0, "mlr")
    for i, value in enumerate(vector[4:11]): add_normal(f"b[{i}]", value, 0.0, 1.0, "mlr")
    for i, value in enumerate(vector[11:32]): add_normal(f"r[{i // 3},{i % 3}]", value, 0.0, 1.0, "mlr")
    add_normal("log_lambda_x", vector[32], np.log(lambda_ref_x), 0.7, "mlr")
    add_normal("log_lambda_z", vector[33], np.log(lambda_ref_z), 0.7, "mlr")
    f_value = float(vector[34])
    coordinate_rows.append({"name": "f_outlier", "value": f_value, "prior": "Beta(3,12)", "z_score": None, "group": "outlier"})
    shape_names = ["log B", "log uc", "log C"]
    for i, name in enumerate(shape_names):
        coordinate_rows.append({"name": name, "value": float(vector[35 + i]), "prior": f"truncated Normal({shape_centers[i]:.8g},{shape_sigmas[i]:.8g})", "z_score": float((vector[35 + i] - shape_centers[i]) / shape_sigmas[i]), "group": "shape"})

    mlr_rows = [row for row in coordinate_rows if row["group"] == "mlr"]
    fig, axes = plt.subplots(7, 5, figsize=(12.0, 14.5), constrained_layout=True)
    for ax, row in zip(axes.flat, mlr_rows):
        mean = 0.0 if row["name"] not in ("c0", "log lambda_x", "log lambda_z") else row["value"] - row["z_score"] * (0.1 if row["name"] == "c0" else 0.7)
        sigma = 1.0 if row["name"] not in ("c0", "log lambda_x", "log lambda_z") else (0.1 if row["name"] == "c0" else 0.7)
        x = np.linspace(min(mean - 3.2 * sigma, row["value"] - 0.2 * sigma),
                        max(mean + 3.2 * sigma, row["value"] + 0.2 * sigma), 250)
        ax.plot(x, normal_density(x, mean, sigma), color="#4c78a8", lw=1.5)
        style_axis(ax, row["name"], row["value"])
        ax.set_xlabel("value", fontsize=8)
    for ax in axes.flat[len(mlr_rows):]: ax.axis("off")
    axes[0, 0].legend(handles=[plt.Line2D([], [], color="#4c78a8", lw=1.5, label="Univariate prior factor"), plt.Line2D([], [], color="#c44e52", lw=1.5, label="MAP")], fontsize=8, frameon=True)
    fig.suptitle("MLR parameters (34): prior factors and MAP\n"
                 "lambda_x/z set edge-slope scales; shrinkage, smoothness, and the solar anchor also constrain the MLR",
                 fontsize=12)
    fig.savefig(args.output_dir / "em_map_prior_mlr.png", dpi=210, bbox_inches="tight")
    fig.savefig(args.output_dir / "em_map_prior_mlr.pdf", dpi=210, bbox_inches="tight")
    plt.close(fig)

    shape_specs = list(zip(shape_names, vector[35:38], shape_centers, shape_sigmas, shape_bounds))
    fig, axes = plt.subplots(1, 3, figsize=(10.0, 4.2))
    for ax, (name, value, mean, sigma, bounds) in zip(axes, shape_specs):
        x = np.linspace(*bounds, 300)
        norm = ndtr((bounds[1] - mean) / sigma) - ndtr((bounds[0] - mean) / sigma)
        ax.plot(x, normal_density(x, mean, sigma) / norm, color="#4c78a8", lw=1.8)
        style_axis(ax, name, value, bounds)
        ax.set_xlabel("log-coordinate", fontsize=9)
        ax.set_ylabel("Truncated prior density", fontsize=9)
    fig.suptitle("Good-shape parameters (3): fixed centers and fitted MAP values", fontsize=14)
    fig.text(0.5, 0.035, r"$p_{\rm good}(\tilde u)=\tilde u\exp[-B\tilde u^2-\exp((\tilde u-u_c)/C)]/Z$ for $0<\tilde u<80$; $B,u_c,C$ are fitted here.", ha="center", fontsize=9)
    fig.subplots_adjust(bottom=0.28, top=0.82, wspace=0.25)
    fig.savefig(args.output_dir / "em_map_prior_shape.png", dpi=210, bbox_inches="tight")
    fig.savefig(args.output_dir / "em_map_prior_shape.pdf", dpi=210, bbox_inches="tight")
    plt.close(fig)

    x = np.linspace(1e-5, 0.7, 600)
    beta_density = np.exp(2.0 * np.log(x) + 11.0 * np.log1p(-x) - betaln(3.0, 12.0))
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    ax.plot(x, beta_density, color="#7b5ea7", lw=2.0, label="Beta(3,12) prior factor")
    ax.axvline(f_value, color="#c44e52", lw=1.8, label=f"MAP f = {f_value:.3f}")
    ax.set_xlabel("Mixture weight f", fontsize=10)
    ax.set_ylabel("Prior density", fontsize=10)
    ax.set_title("Outlier parameter (1): fixed prior and fitted MAP", fontsize=13)
    ax.legend(fontsize=9, frameon=True)
    fig.text(0.5, 0.18, r"$p_{\rm out}(v)=\phi((v-40)/13)/(13[\Phi((80-40)/13)-\Phi(-40/13)]),\quad 0<v<80$", ha="center", fontsize=10)
    fig.text(0.5, 0.12, r"$p_{\rm out}(u\mid\sigma_u)=\int_0^{80}{\rm Rice}(u\mid v,\sigma_u)\,p_{\rm out}(v)\,dv$", ha="center", fontsize=10)
    fig.text(0.5, 0.06, r"$p_{\rm mix}(u\mid M,\sigma_u)=(1-f)\,p_{\rm good,obs}(u\mid M,\sigma_u)+f\,p_{\rm out}(u\mid\sigma_u)$; only $f$ is fitted here.", ha="center", fontsize=9)
    fig.subplots_adjust(bottom=0.36, top=0.88)
    fig.savefig(args.output_dir / "em_map_prior_outlier.png", dpi=210, bbox_inches="tight")
    fig.savefig(args.output_dir / "em_map_prior_outlier.pdf", dpi=210, bbox_inches="tight")
    plt.close(fig)

    report = {
        "source_em_output": str(args.em_output.resolve()),
        "endpoint": "default MAP",
        "converged": bool(summary["results"]["default"]["converged"]),
        "coordinate_order": ["c0", "a[3]", "b[7]", "r[7,3]", "log_lambda_x", "log_lambda_z", "f_outlier", "log B", "log uc", "log C"],
        "coordinate_groups": {"MLR": 34, "good_shape": 3, "outlier": 1},
        "coordinates": coordinate_rows,
        "prior_definitions": {
            "c0": "Normal(PARSEC projection[0,0], 0.10 dex)",
            "a_b_r": "independent standard Normal factors; coupled by D shrinkage/smoothness/solar-anchor terms in the joint MLR prior",
            "log_lambda_x_z": "Normal(log(lambda_ref_x,z), 0.7)",
            "f_outlier": "Beta(3,12)",
            "shape_logs": "Gaussian factors centered at log(0.002544,35.67,3.1) with sigmas (ln2,ln1.3,ln2), truncated to the saved hard shape box",
            "good_density": "p_good(tilde u) = tilde u exp[-B tilde u^2 - exp((tilde u-u_c)/C)] / Z for 0 < tilde u < 80",
            "outlier_intrinsic": "p_out(v)=phi((v-40)/13)/(13*(Phi((80-40)/13)-Phi(-40/13))) for 0<v<80; p_out(u|sigma_u)=integral_0^80 Rice(u|v,sigma_u)*p_out(v) dv; f is the mixture weight",
            "mlr_lambda_role": "log_lambda_x and log_lambda_z set the positive edge-slope scales in the monotone MLR parameterization; they are not the smoothness penalty strengths",
            "fixed_vs_fitted": "The good-shape prior centers and scales, outlier Normal parameters, and prior hyperparameters are fixed; B, uc, C, and f are fitted coordinates.",
            "additional_constraints": "Hard monotonicity is structural; MLR D penalties and solar anchor couple raw parameters beyond these plotted one-dimensional factors.",
        },
        "shape_box": shape_box,
        "shape_prior_truncation_normalizations": {
            name: float(ndtr((bounds[1] - shape_centers[i]) / shape_sigmas[i]) - ndtr((bounds[0] - shape_centers[i]) / shape_sigmas[i]))
            for i, (name, bounds) in enumerate(zip(shape_names, shape_bounds))
        },
        "limitation": "These are univariate prior factors with MAP markers, not posterior distributions or full joint MLR marginals. Prior leverage requires changed-prior refits.",
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
