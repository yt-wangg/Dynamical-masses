"""Per-system score diagnostic for the M_G-window mass pressure (T8.1).

On the saved dynamics lookup, latent-metallicity weights and MLR posterior,
perturb the component masses inside an M_G window,

    ln m_k -> ln m_k + eps * h(M_G_k),   h = 1 inside [MG_MIN, MG_MAX],

and report, per system j, the likelihood score at eps = 0,

    S_j = d ln L_j / d eps,

averaged over the MLR posterior draws.  S_j > 0 means system j asks for larger
masses inside the window.  The chain rule splits each score into good- and
outlier-branch contributions via the lookup slopes and the mixture
responsibilities, so the same evaluation also answers who is pushing: the
velocity-normal body, the fast tail, or the outlier branch.  No refit and no
lookup rebuild is involved; the score is reported per +10% window mass,
S_10 = S * ln(1.1), i.e. the log-likelihood gain a 10% mass increase would buy.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from astropy.table import Table

import binary_masses.hierarchical_metallicity as hm

DRAWS = ("c0", "a", "b", "r", "log_lambda_x", "log_lambda_z", "f_outlier")


def load_stage_two(run_dir: Path, data_path: Path) -> dict:
    lookup = hm.DynamicsLikelihoodLookup.load(run_dir / "dynamics_likelihood_lookup_t8.npz")
    table = Table.read(data_path)
    rows = np.asarray(lookup.row_indices, dtype=np.int64)
    u = np.asarray(table["u"], dtype=np.float64)[rows]
    u_sigma = np.asarray(table["u_sigma"], dtype=np.float64)[rows]
    absg = np.column_stack(
        [np.asarray(table["absg1"], dtype=np.float64), np.asarray(table["absg2"], dtype=np.float64)]
    )[rows]
    lookup.validate_for(row_indices=rows, u=u, u_sigma=u_sigma)

    with np.load(run_dir / "latent_metallicity_weights_t8.npz", allow_pickle=False) as latent:
        if not np.array_equal(latent["row_indices"], rows):
            raise ValueError("Latent-metallicity rows do not match the dynamics lookup rows.")
        z_grid = np.asarray(latent["z_grid"], dtype=np.float64)
        z_probabilities = np.asarray(latent["probabilities"], dtype=np.float64)

    with np.load(run_dir / "mlr_mcmc_t8.npz", allow_pickle=False) as mcmc:
        draws = {
            name.removeprefix("posterior__"): np.asarray(value[0], dtype=np.float64)
            for name, value in mcmc.items()
            if name.startswith("posterior__") and name.removeprefix("posterior__") in DRAWS
        }
    missing = [name for name in DRAWS if name not in draws]
    if missing:
        raise ValueError(f"MLR posterior is missing sampled parameters: {missing}")

    mlr_model = json.loads((run_dir / "mlr_model.json").read_text(encoding="utf-8"))
    return {
        "rows": rows,
        "u": u,
        "u_sigma": u_sigma,
        "absg": absg,
        "sep_AU": np.asarray(table["sep_AU"], dtype=np.float64)[rows],
        "z_grid": z_grid,
        "z_probabilities": z_probabilities,
        "draws": draws,
        "knots_x": np.asarray(mlr_model["knot_x"], dtype=np.float64),
        "knots_z": np.asarray(mlr_model["knot_z"], dtype=np.float64),
        "degree_x": int(mlr_model["degree_x"]),
        "degree_z": int(mlr_model["degree_z"]),
        "parsec_projection": np.asarray(mlr_model["parsec_projection_coefficients"], dtype=np.float64),
        "lookup": lookup,
    }


def evaluate_scores(data: dict, *, mg_min: float, mg_max: float, draw_chunk: int = 60) -> dict:
    """Per-system scores at the MLR posterior and at the PARSEC projection."""
    jax = hm._configure_jax_gpu_fallback()
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    lookup = data["lookup"]
    grid = np.asarray(lookup.sqrt_mtot_grid, dtype=np.float64)
    absg = data["absg"]
    n = absg.shape[0]

    h = ((absg >= mg_min) & (absg <= mg_max)).astype(np.float64)
    bx1 = hm._bspline_basis_numpy(absg[:, 0], data["knots_x"], data["degree_x"])
    bx2 = hm._bspline_basis_numpy(absg[:, 1], data["knots_x"], data["degree_x"])
    bz = hm._bspline_basis_numpy(data["z_grid"], data["knots_z"], data["degree_z"])
    log_w = np.log(np.maximum(data["z_probabilities"], 1e-30))

    grid_j = jnp.asarray(grid)
    good_j = jnp.asarray(np.asarray(lookup.log_good, dtype=np.float64))
    bad_j = jnp.asarray(np.asarray(lookup.log_bad, dtype=np.float64))
    raw_to_theta = hm.MonotoneTensorSplineMLR._theta_from_raw_jax
    stub = SimpleNamespace(K_x=data["knots_x"].size - data["degree_x"] - 1,
                           K_Z=data["knots_z"].size - data["degree_z"] - 1)

    def make_scorer(bx1s, bx2s, log_ws, h1s, h2s, row_index):
        """Chain-rule score decomposition and eps-likelihood for one system set."""
        bx1s_j, bx2s_j, log_ws_j, h1s_j, h2s_j = map(
            jnp.asarray, (bx1s, bx2s, log_ws, h1s, h2s)
        )
        good_rows, bad_rows = jnp.asarray(good_j[row_index]), jnp.asarray(bad_j[row_index])

        def interp(table, s):
            upper = jnp.clip(jnp.searchsorted(grid_j, s, side="right"), 1, grid_j.size - 1)
            lower = upper - 1
            yl = jnp.take_along_axis(table, lower, axis=1)
            yu = jnp.take_along_axis(table, upper, axis=1)
            xl, xu = grid_j[lower], grid_j[upper]
            return yl + (s - xl) / (xu - xl) * (yu - yl)

        def slope(table, s):
            upper = jnp.clip(jnp.searchsorted(grid_j, s, side="right"), 1, grid_j.size - 1)
            lower = upper - 1
            return (jnp.take_along_axis(table, upper, axis=1) - jnp.take_along_axis(table, lower, axis=1)) / (
                grid_j[upper] - grid_j[lower]
            )

        def scale_from_theta(theta, eps):
            m1 = jnp.power(10.0, jnp.einsum("ni,qh,ih->nq", bx1s_j, bz, theta))
            m2 = jnp.power(10.0, jnp.einsum("ni,qh,ih->nq", bx2s_j, bz, theta))
            return m1, m2, jnp.sqrt(m1 * jnp.exp(h1s_j * eps) + m2 * jnp.exp(h2s_j * eps))

        def per_draw_stats(theta, f_outlier):
            m1, m2, s = scale_from_theta(theta, jnp.asarray(0.0))
            log_a = jnp.log1p(-f_outlier) + interp(good_rows, s)
            log_b = jnp.log(f_outlier) + interp(bad_rows, s)
            log_cond = jnp.logaddexp(log_a, log_b)
            log_joint = log_ws_j + log_cond
            per_system = jax.nn.logsumexp(log_joint, axis=1)
            p_q = jax.nn.softmax(log_joint, axis=1)
            r_good, r_bad = jnp.exp(log_a - log_cond), jnp.exp(log_b - log_cond)
            ds_deps = (m1 * h1s_j + m2 * h2s_j) / (2.0 * s)
            push_good = r_good * slope(good_rows, s) * ds_deps
            push_bad = r_bad * slope(bad_rows, s) * ds_deps
            return (
                per_system,
                jnp.sum(p_q * (push_good + push_bad), axis=1),
                jnp.sum(p_q * push_good, axis=1),
                jnp.sum(p_q * push_bad, axis=1),
                jnp.sum(p_q * r_bad, axis=1),
                jnp.sum(p_q * s, axis=1),
                jnp.sum(p_q * ds_deps, axis=1),
            )

        def mixture_loglik(eps_value, thetas, f_values):
            def one(theta, f):
                _, _, s = scale_from_theta(theta, eps_value)
                log_cond = jnp.logaddexp(
                    jnp.log1p(-f) + interp(good_rows, s), jnp.log(f) + interp(bad_rows, s)
                )
                return jax.nn.logsumexp(log_ws_j + log_cond, axis=1)

            total = 0.0
            for start in range(0, f_values.shape[0], draw_chunk):
                stop = min(start + draw_chunk, f_values.shape[0])
                total = total + jnp.sum(jax.vmap(one)(thetas[start:stop], f_values[start:stop]), axis=0)
            return total / f_values.shape[0]

        return per_draw_stats, mixture_loglik

    draws = data["draws"]
    n_draws = draws["c0"].shape[0]
    f_all = jnp.asarray(draws["f_outlier"])
    posterior_thetas = jnp.concatenate(
        [
            jax.vmap(lambda *a: raw_to_theta(stub, *a))(
                *[jnp.asarray(draws[name][start:min(start + draw_chunk, n_draws)]) for name in DRAWS[:-1]]
            )
            for start in range(0, n_draws, draw_chunk)
        ],
        axis=0,
    )
    parsec_theta = jnp.asarray(data["parsec_projection"])

    per_draw_stats, _ = make_scorer(bx1, bx2, log_w, h[:, 0:1], h[:, 1:2], np.arange(n))

    def run_chain_rule(theta_stack):
        theta_is_batched = jnp.ndim(theta_stack) == 3
        out = {key: np.zeros(n) for key in ("loglik", "S", "S_good", "S_bad", "p_outlier", "s_mean", "ds_deps_mean")}
        for start in range(0, n_draws, draw_chunk):
            stop = min(start + draw_chunk, n_draws)
            theta_block = theta_stack[start:stop] if theta_is_batched else theta_stack
            f_block = f_all[start:stop]
            in_axes = (0 if theta_is_batched else None, 0)
            batch = jax.vmap(per_draw_stats, in_axes=in_axes)(theta_block, f_block)
            weight = (stop - start) / n_draws
            for key, values in zip(out, batch):
                out[key] += weight * np.asarray(values, dtype=np.float64).mean(axis=0)
        return out

    out = run_chain_rule(posterior_thetas)
    out["S_parsec"] = run_chain_rule(parsec_theta)["S"]

    # Cross-check the chain-rule score against autodiff on a small subset; the
    # full-data jacobian would need tens of GB, this needs none.
    check = np.random.default_rng(0).choice(n, size=min(64, n), replace=False)
    check_draws = np.linspace(0, n_draws, num=min(32, n_draws), dtype=int)
    check_scorer, check_mixture = make_scorer(
        bx1[check], bx2[check], log_w[check], h[check, 0:1], h[check, 1:2], check
    )
    check_thetas = posterior_thetas[check_draws]
    check_f = f_all[check_draws]
    subset = check_scorer(check_thetas[0], check_f[0])
    for start in range(1, len(check_draws), 1):
        one = check_scorer(check_thetas[start], check_f[start])
        subset = tuple(a + b for a, b in zip(subset, one))
    subset = tuple(np.asarray(value, dtype=np.float64) / len(check_draws) for value in subset)
    auto = jax.jacobian(lambda e: check_mixture(e, check_thetas, check_f))(jnp.asarray(0.0))
    relative = np.abs(np.asarray(auto) - subset[1]) / np.maximum(np.abs(subset[1]), 1e-6)
    if not np.all(relative < 1e-4):
        raise RuntimeError(f"Chain-rule score disagrees with autodiff (max rel {np.max(relative):.2e}).")
    return out, h


def summarize(data: dict, result: dict, h: np.ndarray, *, mg_min: float, mg_max: float) -> dict:
    """Aggregate the per-system scores into the breakdown table and plot."""
    ln10p = float(np.log(1.1))
    absg = data["absg"]
    brighter = np.argmin(absg, axis=1)
    fainter = 1 - brighter
    mg_brighter = absg[np.arange(len(absg)), brighter]
    mg_fainter = absg[np.arange(len(absg)), fainter]
    delta_g = mg_fainter - mg_brighter
    in_brighter = h[np.arange(len(absg)), brighter] > 0
    in_fainter = h[np.arange(len(absg)), fainter] > 0
    category = np.where(
        in_brighter & in_fainter, "both",
        np.where(in_brighter, "brighter_only",
                 np.where(in_fainter, "fainter_only", "none")),
    )

    # Velocity residual against the good kernel at the posterior-mean scale.
    tilde = np.linspace(1e-3, hm.RICE_GOOD_SUPPORT, 4001)
    density = hm.rice_good_raw(tilde)
    mean_tilde_u = float(np.trapezoid(tilde * density, tilde) / np.trapezoid(density, tilde))
    u_predicted = result["s_mean"] * mean_tilde_u
    u_ratio = data["u"] / u_predicted

    s10 = result["S"] * ln10p
    s10_good = result["S_good"] * ln10p
    s10_bad = result["S_bad"] * ln10p

    def block_sum(mask):
        return {
            "n": int(np.sum(mask)),
            "S10_total": float(np.sum(s10[mask])),
            "S10_good": float(np.sum(s10_good[mask])),
            "S10_bad": float(np.sum(s10_bad[mask])),
            "S10_median": float(np.median(s10[mask])) if np.any(mask) else None,
        }

    categories = ["both", "brighter_only", "fainter_only", "none"]
    u_quartile = np.quantile(u_ratio, [0.25, 0.5, 0.75])
    u_bins = np.digitize(u_ratio, u_quartile)
    quartile_labels = [
        f"Q1 (u/u_pred <= {u_quartile[0]:.2f})",
        f"Q2 ({u_quartile[0]:.2f}-{u_quartile[1]:.2f})",
        f"Q3 ({u_quartile[1]:.2f}-{u_quartile[2]:.2f})",
        f"Q4 (> {u_quartile[2]:.2f})",
    ]
    summary = {
        "mg_window": [mg_min, mg_max],
        "n_systems": int(len(s10)),
        "n_draws": int(len(data["draws"]["c0"])),
        "units": "d ln L for a +10% mass increase of window components (S * ln 1.1)",
        "total": block_sum(np.ones(len(s10), dtype=bool)),
        "total_at_parsec_S10": float(np.sum(result["S_parsec"] * ln10p)),
        "by_category": {name: block_sum(category == name) for name in categories},
        "by_u_ratio_quartile": {
            quartile_labels[index]: block_sum(u_bins == index)
            for index in range(4)
        },
        "by_delta_g": {
            "dG<1": block_sum(delta_g < 1.0),
            "dG>=1": block_sum(delta_g >= 1.0),
        },
        "by_outlier_responsibility": {
            "p_out<0.5": block_sum(result["p_outlier"] < 0.5),
            "p_out>=0.5": block_sum(result["p_outlier"] >= 0.5),
        },
        "mean_tilde_u_good_kernel": mean_tilde_u,
    }
    per_system = {
        "row_index": data["rows"],
        "u": data["u"],
        "u_sigma": data["u_sigma"],
        "absg_brighter": mg_brighter,
        "absg_fainter": mg_fainter,
        "delta_g": delta_g,
        "sep_AU": data["sep_AU"],
        "category": category,
        "u_ratio": u_ratio,
        "p_outlier": result["p_outlier"],
        "loglik": result["loglik"],
        "sqrt_mtot": result["s_mean"],
        "S": result["S"],
        "S_good": result["S_good"],
        "S_bad": result["S_bad"],
        "S_parsec": result["S_parsec"],
    }
    return summary, per_system


def plot_breakdown(per_system: dict, summary: dict, path: Path, *, mg_min: float, mg_max: float) -> None:
    ln10p = float(np.log(1.1))
    s10 = per_system["S"] * ln10p
    s10_good = per_system["S_good"] * ln10p
    s10_bad = per_system["S_bad"] * ln10p
    categories = ["both", "brighter_only", "fainter_only", "none"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    ax = axes[0, 0]
    sums_good = [summary["by_category"][name]["S10_good"] for name in categories]
    sums_bad = [summary["by_category"][name]["S10_bad"] for name in categories]
    ax.bar(categories, sums_good, color="tab:blue", label="good kernel")
    ax.bar(categories, sums_bad, bottom=sums_good, color="tab:red", label="outlier branch")
    for index, total in enumerate(np.asarray(sums_good) + np.asarray(sums_bad)):
        ax.text(index, total, f"{total:+.1f}", ha="center", va="bottom" if total >= 0 else "top", fontsize=9)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_ylabel(r"$\sum_j \Delta\ln L_j$ per +10% window mass")
    ax.set_title(f"Likelihood pressure by window membership (M_G in [{mg_min:g}, {mg_max:g}])")
    ax.legend(fontsize=9)

    ax = axes[0, 1]
    fainter = per_system["absg_fainter"]
    fainter_in_window = (fainter >= mg_min) & (fainter <= mg_max)
    near = per_system["delta_g"] < 1.0
    edges = np.linspace(mg_min, mg_max, 7)
    centers = 0.5 * (edges[:-1] + edges[1:])
    width = 0.38 * (edges[1] - edges[0])
    for offset, (mask, label, color) in enumerate((
        (near, "dG < 1 (matched-brightness pair)", "tab:purple"),
        (~near, "dG >= 1 (bright primary + faint companion)", "tab:green"),
    )):
        sums = [
            np.sum(s10[fainter_in_window & mask & (fainter >= edges[b]) & (fainter < edges[b + 1])])
            for b in range(len(centers))
        ]
        ax.bar(centers + (offset - 0.5) * width, sums, width=width, color=color, label=label)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xlabel("M_G of the fainter component (restricted to the window)")
    ax.set_ylabel(r"$\sum_j \Delta\ln L_j$ per +10% window mass")
    ax.set_title("Pressure vs companion magnitude and brightness ratio")
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    u_quartile = np.quantile(per_system["u_ratio"], [0.25, 0.5, 0.75])
    u_bins = np.digitize(per_system["u_ratio"], u_quartile)
    good_sums, bad_sums = [], []
    for index in range(4):
        mask = u_bins == index
        good_sums.append(np.sum(s10_good[mask]))
        bad_sums.append(np.sum(s10_bad[mask]))
    bounds = [u_quartile[0], u_quartile[1], u_quartile[2]]
    labels = [
        f"Q1\n<= {bounds[0]:.2f}",
        f"Q2\n{bounds[0]:.2f}-{bounds[1]:.2f}",
        f"Q3\n{bounds[1]:.2f}-{bounds[2]:.2f}",
        f"Q4\n> {bounds[2]:.2f}",
    ]
    x = np.arange(4)
    ax.bar(x, good_sums, color="tab:blue", label="good kernel")
    ax.bar(x, bad_sums, bottom=good_sums, color="tab:red", label="outlier branch")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(x, labels)
    ax.set_xlabel("u_obs / (s * <tilde_u>_good)")
    ax.set_ylabel(r"$\sum_j \Delta\ln L_j$ per +10% window mass")
    ax.set_title("Pressure by velocity residual quartile")
    ax.legend(fontsize=9)

    ax = axes[1, 1]
    sc = ax.scatter(per_system["u_ratio"], s10, c=per_system["p_outlier"], s=8, cmap="viridis", alpha=0.8)
    fig.colorbar(sc, ax=ax, label="outlier responsibility")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xlim(0, np.quantile(per_system["u_ratio"], 0.995))
    ax.set_ylim(-0.12, 0.5)
    ax.set_xlabel("u_obs / (s * <tilde_u>_good)")
    ax.set_ylabel("per-system score per +10% window mass")
    ax.set_title("Per-system scores vs velocity residual")

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=Path("results/hierarchical_metallicity_t8_1"))
    parser.add_argument(
        "--data", type=Path,
        default=Path("data/jd_msms_single_bic_1kpc_filtered_cmdcut_cutb_jcaps_err_mhcal_good.fits"),
    )
    parser.add_argument("--mg-min", type=float, default=7.0)
    parser.add_argument("--mg-max", type=float, default=10.0)
    parser.add_argument("--output", type=Path, default=Path("results/mg_window_score_t8_1"))
    args = parser.parse_args()

    data = load_stage_two(args.run_dir, args.data)
    result, h = evaluate_scores(data, mg_min=args.mg_min, mg_max=args.mg_max)
    summary, per_system = summarize(data, result, h, mg_min=args.mg_min, mg_max=args.mg_max)

    args.output.mkdir(parents=True, exist_ok=True)
    table = Table()
    for key, value in per_system.items():
        table[key] = value
    table.write(args.output / "mg_window_scores.csv", format="csv", overwrite=True)
    (args.output / "mg_window_score_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    plot_breakdown(per_system, summary, args.output / "mg_window_scores.png",
                   mg_min=args.mg_min, mg_max=args.mg_max)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

