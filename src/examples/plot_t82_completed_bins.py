"""Compare independent metallicity-bin MLR posteriors at bin midpoints."""
from pathlib import Path
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import run_hierarchical_metallicity_test as workflow
from binary_masses.hierarchical_metallicity import _bspline_basis_numpy


def main():
    root = Path(__file__).resolve().parents[2]
    output = root / "results/t82_joint_shape_metal_bins_formal_20260919"
    surface, _ = workflow.build_surfaces()
    mlr = workflow._make_t8_mlr(surface, workflow.parse_args())
    x = np.linspace(3.5, 13.5, 251)
    bx = _bspline_basis_numpy(x, mlr.knots_x, mlr.degree_x)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True,
                             gridspec_kw={"height_ratios": [1.3, 1]}, layout="constrained")
    summary = {}
    for col, (bin_id, z, label) in enumerate([
        (0, -0.75, r"$-1.0 \leq [M/H] < -0.5$"),
        (3, 0.45, r"$0.3 \leq [M/H] \leq 0.6$"),
    ]):
        reference = surface.mass_from_absg_mh(x, z)
        bz = _bspline_basis_numpy(z, mlr.knots_z, mlr.degree_z)
        medians = {}
        for start, color, name in [("parsec", "#2479ad", "PARSEC start"),
                                    ("s2", "#e37727", "S2 start")]:
            folder = output / f"bin{bin_id}_{start}_jitter"
            config = json.loads((folder / "run_config.json").read_text())
            with np.load(folder / "mlr_mcmc_t8.npz") as saved:
                keys = ["a", "b", "c0", "r", "log_lambda_x", "log_lambda_z"]
                p = {k: saved["posterior__" + k].reshape((-1,) + saved["posterior__" + k].shape[2:]) for k in keys}
                theta = np.array([mlr.theta_from_raw({k: v[i] for k, v in p.items()}) for i in range(len(p["c0"]))])
            mass = 10 ** np.einsum("xi,z,niz->nx", bx, bz, theta)
            q = np.percentile(mass, [16, 50, 84], axis=0)
            residual = 100 * (q / reference - 1)
            for row, values in enumerate([q, residual]):
                axes[row, col].fill_between(x, values[0], values[2], color=color, alpha=0.18)
                axes[row, col].plot(x, values[1], color=color, lw=2, label=name,
                                    ls="-" if start == "parsec" else "--")
            medians[start] = q[1]
            summary[f"bin{bin_id}_{start}"] = {
                "evaluation_mh": z, "n_systems": config["n_systems"],
                "Mg": [5, 7, 9, 11, 13],
                "mass_p16_p50_p84": [np.interp([5, 7, 9, 11, 13], x, a).tolist() for a in q],
                "parsec_mass": np.interp([5, 7, 9, 11, 13], x, reference).tolist(),
            }
        summary[f"bin{bin_id}_max_start_median_difference_percent"] = float(np.max(np.abs(100*(medians["s2"]/medians["parsec"]-1))))
        axes[0, col].plot(x, reference, color="0.25", lw=1.7, ls=":", label="PARSEC model")
        axes[1, col].axhline(0, color="0.3", lw=1, ls=":")
        axes[0, col].set_title(f"{label}  (N = {config['n_systems']})\nCurves evaluated at [M/H] = {z:+.2f}", fontsize=12)
        axes[0, col].legend(frameon=False, fontsize=10)
        axes[1, col].set_xlabel(r"$M_G$ [mag]")
    axes[0, 0].set_ylabel(r"Mass [$M_\odot$]")
    axes[1, 0].set_ylabel(r"$100\,(M/M_{\rm PARSEC}-1)$ [%]")
    for ax in axes.flat:
        ax.grid(alpha=0.18)
        ax.set_xlim(3.5, 13.5)
    fig.suptitle("Independent-bin joint MCMC: posterior medians and 16–84% intervals", fontsize=13)
    for ext in ["png", "pdf"]:
        fig.savefig(output / f"completed_bins_mlr_parsec.{ext}", dpi=180)
    plt.close(fig)
    (output / "completed_bins_mlr_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
