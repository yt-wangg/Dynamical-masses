"""Diagnose standardized prior displacement at the EM-like default endpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import betaln


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "examples"))

import run_em_mlr_pilot as em  # noqa: E402
import run_hierarchical_metallicity_test as workflow  # noqa: E402


DEFAULT_EM = ROOT / "results/em_solar_only_full_rebuild_20260928_b_relaxed_cont40"
DEFAULT_OUT = ROOT / "results/em_prior_constraint_diagnostic_20260929"


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--em-output", type=Path, default=DEFAULT_EM)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary = json.loads((args.em_output / "summary.json").read_text())
    with np.load(args.em_output / "map_states.npz", allow_pickle=False) as states:
        vector = np.asarray(states["default_vector"], dtype=float)
    params = em.vector_to_params(vector)
    mlr = build_mlr()

    theta = np.asarray(mlr.theta_from_raw(params), dtype=float)
    D = theta - np.asarray(mlr.parsec_projection, dtype=float)
    d2x, d2z, dxz = mlr.spacing_aware_penalties(D)
    prior_terms = mlr.log_prior_terms(params)
    rms_groups = {
        "D shrinkage": float(np.sqrt(np.mean((D / mlr.tau_D) ** 2))),
        "d2x smoothness": float(np.sqrt(np.mean((d2x / mlr.tau_x) ** 2))),
        "d2z smoothness": float(np.sqrt(np.mean((d2z / mlr.tau_Z) ** 2))),
        "dxz smoothness": float(np.sqrt(np.mean((dxz / mlr.tau_xZ) ** 2))),
        "a standard Normal": float(np.sqrt(np.mean(np.asarray(params["a"]) ** 2))),
        "b standard Normal": float(np.sqrt(np.mean(np.asarray(params["b"]) ** 2))),
        "r standard Normal": float(np.sqrt(np.mean(np.asarray(params["r"]) ** 2))),
    }
    lambda_ref_x, lambda_ref_z = mlr.lambda_references()
    solar_g = float(mlr.g_from_raw(4.67, 0.0, params))
    scalar_z = {
        "c0": float((params["c0"] - mlr.parsec_projection[0, 0]) / 0.10),
        "log lambda_x": float((params["log_lambda_x"] - np.log(lambda_ref_x)) / 0.7),
        "log lambda_z": float((params["log_lambda_z"] - np.log(lambda_ref_z)) / 0.7),
        "solar anchor": float((solar_g - mlr.solar_anchor_mean) / mlr.solar_anchor_sigma),
        "log B": float((vector[35] - np.log(0.002544)) / np.log(2.0)),
        "log uc": float((vector[36] - np.log(35.67)) / np.log(1.3)),
        "log C": float((vector[37] - np.log(3.1)) / np.log(2.0)),
    }
    f_outlier = float(vector[em.F_INDEX])
    f_grid = np.linspace(1e-5, 0.7, 1000)
    beta_density = np.exp(2.0 * np.log(f_grid) + 11.0 * np.log1p(-f_grid) - betaln(3.0, 12.0))
    beta_mode = 2.0 / 13.0
    shape_box = summary["settings"]["shape_box"]

    fig, axes = plt.subplots(1, 3, figsize=(11.2, 6.0), gridspec_kw={"width_ratios": [1.35, 1.25, 1.0]})
    names = list(rms_groups)
    axes[0].barh(np.arange(len(names)), [rms_groups[name] for name in names], color="#4c78a8")
    axes[0].axvline(1.0, color="0.3", ls="--", lw=1.2, label="Unit RMS")
    axes[0].set_yticks(np.arange(len(names)), names, fontsize=8)
    axes[0].set_xlabel("RMS standardized displacement", fontsize=10)
    axes[0].set_title("Grouped Gaussian penalties", fontsize=12)
    axes[0].invert_yaxis()
    axes[0].legend(fontsize=8, frameon=True)

    scalar_names = list(scalar_z)
    axes[1].axvline(0.0, color="0.4", lw=1.0)
    axes[1].axvline(1.0, color="0.5", ls="--", lw=1.0)
    axes[1].axvline(-1.0, color="0.5", ls="--", lw=1.0)
    scalar_colors = ["#d27a28"] * 4 + ["#7b5ea7"] * 3
    axes[1].barh(np.arange(len(scalar_names)), [scalar_z[name] for name in scalar_names], color=scalar_colors)
    axes[1].set_yticks(np.arange(len(scalar_names)), scalar_names, fontsize=8)
    axes[1].set_xlabel("Scalar standardized displacement", fontsize=10)
    axes[1].set_title("Scalar prior displacements", fontsize=12)
    axes[1].invert_yaxis()

    axes[2].plot(f_grid, beta_density, color="#7b5ea7", lw=2.2, label=r"Beta(3,12) prior")
    axes[2].axvline(f_outlier, color="#c44e52", lw=2.0, label=f"MAP f = {f_outlier:.3f}")
    axes[2].axvline(beta_mode, color="0.3", ls="--", lw=1.1, label=f"Prior mode = {beta_mode:.3f}")
    axes[2].set_xlabel(r"Outlier mixture weight $f$", fontsize=10)
    axes[2].set_ylabel("Prior density", fontsize=10)
    axes[2].set_title("Outlier prior", fontsize=12)
    axes[2].legend(fontsize=8, frameon=True)
    fig.suptitle("EM-like default endpoint: prior-constraint diagnostic", fontsize=14)
    fig.text(0.03, 0.175,
             r"Direct MLR: $D=\theta-\Pi_{\rm PARSEC}$; $\sigma_D=0.10$ dex, "
             r"$\sigma_{\rm smooth}=0.05$ dex; $c_0\sim N(\Pi_{00},0.10\,{\rm dex})$, "
             r"$a,b,r\sim N(0,1)$.", fontsize=8.8)
    fig.text(0.03, 0.125,
             r"Scales/anchor: $\log\lambda_{x,z}\sim N(\log\lambda_{\rm ref},0.7)$; "
             r"$\log_{10} M(4.67,0)\sim N(0,0.01/\ln 10\,{\rm dex})$.", fontsize=8.8)
    fig.text(0.03, 0.075,
             r"Indirect dynamics: $\log(B,u_c,C)\sim N(\log(0.002544,35.67,3.1),"
             r"[\ln2,\ln1.3,\ln2])$ within a hard box; $f\sim {\rm Beta}(3,12)$.", fontsize=8.8)
    fig.text(0.03, 0.025,
             "Hard MLR monotonicity is structural, not a prior distribution; endpoint displacement does not measure causal prior leverage.",
             fontsize=8.5)
    fig.tight_layout(rect=(0, 0.22, 1, 0.94))
    fig.savefig(args.output_dir / "em_prior_constraint_diagnostic.png", dpi=220, bbox_inches="tight")
    fig.savefig(args.output_dir / "em_prior_constraint_diagnostic.pdf", bbox_inches="tight")
    plt.close(fig)

    report = {
        "source_em_output": str(args.em_output.resolve()),
        "endpoint": "default",
        "converged": bool(summary["results"]["default"]["converged"]),
        "prior_formulas": {
            "D_shrinkage": "-0.5 sum((D/tau_D)^2), tau_D=0.10 dex",
            "D_smoothness_x": "-0.5 sum((d2x/tau_x)^2), tau_x=0.05 dex",
            "D_smoothness_Z": "-0.5 sum((d2z/tau_Z)^2), tau_Z=0.05 dex",
            "D_smoothness_xZ": "-0.5 sum((dxz/tau_xZ)^2), tau_xZ=0.05 dex",
            "c0": "Normal(parsec_projection[0,0], 0.10 dex)",
            "a_b_r": "independent standard Normal",
            "log_lambda_x_z": "Normal(log(lambda_ref_x,z), 0.7)",
            "solar_anchor": "Normal(0, 0.01/ln(10)) in log10 mass dex",
            "shape": "log(B,uc,C) ~ Normal(log(0.002544,35.67,3.1), [ln(2),ln(1.3),ln(2)]) within a hard shape box",
            "f_outlier": "Beta(3,12)",
            "hard_monotonicity": "structural constraint enforced by the MLR parameterization; not a prior distribution",
        },
        "grouped_rms_standardized_displacement": rms_groups,
        "scalar_z_scores": scalar_z,
        "log_prior_terms_from_mlr": {key: float(value) for key, value in prior_terms.items()},
        "shape_box": shape_box,
        "shape_B_uc_C": np.exp(vector[35:38]).tolist(),
        "f_outlier": f_outlier,
        "beta_prior_mode": beta_mode,
        "solar_anchor": {"g_log10_mass": solar_g, "z_score": scalar_z["solar anchor"]},
        "limitation": "Standardized displacement describes the endpoint relative to the specified prior scales; it does not establish causal prior leverage. That requires changed-prior refits.",
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
