"""Per-system score diagnostic for the M_G-window mass pressure (T8.1).

On the saved dynamics lookup, latent-metallicity posterior and MLR posterior,
perturb the component masses inside an M_G window,

    ln m_k -> ln m_k + eps * h(M_G_k),   h = 1 inside [MG_MIN, MG_MAX],

and report, per system j, the local score at eps = 0,

    S_j = d ln L_j / d eps,

averaged over all posterior draws (all chains).  S_j > 0 means system j asks
for larger masses inside the window.  The chain rule splits each score into
good- and outlier-branch contributions via the lookup slopes and the mixture
responsibilities.  The reported quantity

    S10_linear = ln(1.1) * S_j

is a LOCAL LINEAR estimate of the log-likelihood gain of a +10% window-mass
increase, not the exact finite-step difference; the exact

    dlnL10_exact = ln L_j(eps = ln 1.1) - ln L_j(0)

is computed separately over the full mixture and metallicity marginalization.

Validation built in:
- chain-rule scores vs. autodiff jacobian on a random subset;
- lookup slopes vs. direct adaptive Rice integration (central differences at
  two step sizes) on selected extreme systems;
- domain and floor accounting: no evaluation may leave the saved lookup
  domain with substantial metallicity probability weight.

Outputs: per-system CSV, summary JSON, per-draw totals npz, and two figures.
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
POSTERIOR_NAME = "latent_metallicity_weights_t8.npz"


def flatten_chains(value):
    """Flatten [chain, draw, ...] posterior samples; return (flat, n_chains)."""
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim < 2:
        raise ValueError("Expected posterior samples with shape [chain, draw, ...].")
    n_chains = int(arr.shape[0])
    return arr.reshape((arr.shape[0] * arr.shape[1],) + arr.shape[2:]), n_chains


def load_stage_two(run_dir: Path, data_path: Path, metallicity_posterior: Path | None) -> dict:
    lookup = hm.DynamicsLikelihoodLookup.load(run_dir / "dynamics_likelihood_lookup_t8.npz")
    table = Table.read(data_path)
    rows = np.asarray(lookup.row_indices, dtype=np.int64)
    u = np.asarray(table["u"], dtype=np.float64)[rows]
    u_sigma = np.asarray(table["u_sigma"], dtype=np.float64)[rows]
    absg = np.column_stack(
        [np.asarray(table["absg1"], dtype=np.float64), np.asarray(table["absg2"], dtype=np.float64)]
    )[rows]
    lookup.validate_for(row_indices=rows, u=u, u_sigma=u_sigma)

    mlr_model = json.loads((run_dir / "mlr_model.json").read_text(encoding="utf-8"))
    if metallicity_posterior is not None:
        posterior_path = Path(metallicity_posterior)
        if not posterior_path.exists():
            raise FileNotFoundError(f"Metallicity posterior not found: {posterior_path}")
    else:
        recorded = mlr_model.get("metallicity_posterior_source")
        posterior_path = Path(recorded) if recorded else run_dir / POSTERIOR_NAME
        if not posterior_path.exists():
            posterior_path = run_dir / POSTERIOR_NAME
    posterior = hm.MetallicityPosteriorGrid.load(posterior_path)
    posterior.validate()
    posterior_model = str(posterior.metadata.get("model", ""))
    if not (posterior_model.startswith("t8") or "_t8_" in posterior_model):
        raise ValueError(f"Metallicity posterior is not a T8 product: {posterior_model}")
    if not np.array_equal(posterior.row_indices, rows):
        raise ValueError("Metallicity posterior rows do not match the dynamics lookup rows.")

    with np.load(run_dir / "mlr_mcmc_t8.npz", allow_pickle=False) as mcmc:
        draws = {}
        chain_counts = {}
        for name, value in mcmc.items():
            if not name.startswith("posterior__"):
                continue
            short = name.removeprefix("posterior__")
            if short in DRAWS:
                draws[short], chain_counts[short] = flatten_chains(value)
    missing = [name for name in DRAWS if name not in draws]
    if missing:
        raise ValueError(f"MLR posterior is missing sampled parameters: {missing}")
    if len(set(chain_counts.values())) != 1:
        raise ValueError("Posterior parameters disagree on the number of chains.")

    mg_min_obs, mg_max_obs = float(np.min(absg)), float(np.max(absg))
    if not (
        np.isclose(mg_min_obs, float(mlr_model["observed_component_mg_min"]), atol=1e-9)
        and np.isclose(mg_max_obs, float(mlr_model["observed_component_mg_max"]), atol=1e-9)
    ):
        raise ValueError("Observed M_G range does not match the MLR run metadata.")

    return {
        "rows": rows,
        "u": u,
        "u_sigma": u_sigma,
        "absg": absg,
        "sep_AU": np.asarray(table["sep_AU"], dtype=np.float64)[rows],
        "z_probabilities": np.asarray(posterior.probabilities, dtype=np.float64),
        "z_grid": np.asarray(posterior.z_grid, dtype=np.float64),
        "draws": draws,
        "n_chains": chain_counts[DRAWS[0]],
        "chain_ids": np.repeat(np.arange(chain_counts[DRAWS[0]]), draws[DRAWS[0]].shape[0] // chain_counts[DRAWS[0]]),
        "knots_x": np.asarray(mlr_model["knot_x"], dtype=np.float64),
        "knots_z": np.asarray(mlr_model["knot_z"], dtype=np.float64),
        "degree_x": int(mlr_model["degree_x"]),
        "degree_z": int(mlr_model["degree_z"]),
        "parsec_projection": np.asarray(mlr_model["parsec_projection_coefficients"], dtype=np.float64),
        "lookup": lookup,
        "provenance": {
            "run_dir": str(run_dir),
            "data_path": str(data_path),
            "metallicity_posterior_path": str(posterior_path),
            "metallicity_posterior_model": posterior_model,
            "n_chains": chain_counts[DRAWS[0]],
            "n_draws_total": int(draws[DRAWS[0]].shape[0]),
            "u_u_sigma_digest": lookup.metadata.get("data_digest"),
            "absg_digest": hm.array_digest(absg),
            "z_probabilities_digest": hm.array_digest(rows, np.asarray(posterior.probabilities, dtype=np.float64)),
        },
    }


def evaluate_scores(data: dict, *, mg_min: float, mg_max: float, draw_chunk: int = 40):
    """Per-system scores, exact finite-step differences, and cross-checks."""
    jax = hm._configure_jax_gpu_fallback()
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    lookup = data["lookup"]
    grid = np.asarray(lookup.sqrt_mtot_grid, dtype=np.float64)
    floor = float(lookup.metadata.get("floor", 1e-30))
    absg = data["absg"]
    n = absg.shape[0]

    h = ((absg >= mg_min) & (absg <= mg_max)).astype(np.float64)
    bx1 = hm._bspline_basis_numpy(absg[:, 0], data["knots_x"], data["degree_x"])
    bx2 = hm._bspline_basis_numpy(absg[:, 1], data["knots_x"], data["degree_x"])
    bz = hm._bspline_basis_numpy(data["z_grid"], data["knots_z"], data["degree_z"])
    z_prob = data["z_probabilities"]
    log_w = np.log(np.maximum(z_prob, 1e-30))

    grid_j = jnp.asarray(grid)
    good_j = jnp.asarray(np.asarray(lookup.log_good, dtype=np.float64))
    bad_j = jnp.asarray(np.asarray(lookup.log_bad, dtype=np.float64))
    raw_to_theta = hm.MonotoneTensorSplineMLR._theta_from_raw_jax
    stub = SimpleNamespace(K_x=data["knots_x"].size - data["degree_x"] - 1,
                           K_Z=data["knots_z"].size - data["degree_z"] - 1)
    floor_log = float(np.log(floor)) + 1e-3

    def make_scorer(bx1s, bx2s, log_ws, z_ws, h1s, h2s, row_index):
        """Score decomposition, eps-likelihood, and domain stats for one system set."""
        bx1s_j, bx2s_j, log_ws_j, z_ws_j, h1s_j, h2s_j = map(
            jnp.asarray, (bx1s, bx2s, log_ws, z_ws, h1s, h2s)
        )
        good_rows, bad_rows = jnp.asarray(good_j[row_index]), jnp.asarray(bad_j[row_index])

        def interp(table, s):
            return hm.lookup_interpolate(s, grid_j, table)

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

        def domain_stats(eps_value, thetas):
            """Out-of-domain and floor-hit accounting, per draw and system."""
            def one(theta):
                _, _, s = scale_from_theta(theta, eps_value)
                outside = (s < grid_j[0]) | (s > grid_j[-1])
                p_out = jnp.sum(z_ws_j * outside, axis=1)
                good_ll = interp(good_rows, s)
                bad_ll = interp(bad_rows, s)
                floor_hits = jnp.sum((good_ll <= floor_log) | (bad_ll <= floor_log))
                return jnp.sum(outside), p_out, floor_hits, jnp.min(s), jnp.max(s)

            blocks = [
                jax.vmap(one)(thetas[start:min(start + draw_chunk, thetas.shape[0])])
                for start in range(0, thetas.shape[0], draw_chunk)
            ]
            stacked = [
                np.concatenate([np.asarray(block[k], dtype=np.float64) for block in blocks], axis=0)
                for k in range(len(blocks[0]))
            ]
            n_out_draws, p_out, floor_hits, s_min, s_max = stacked
            return {
                "eps": float(eps_value),
                "n_outside_nodes": float(np.sum(n_out_draws)),
                "max_p_outside_system": float(np.max(p_out)),
                "n_floor_hits": float(np.sum(floor_hits)),
                "s_min": float(np.min(s_min)),
                "s_max": float(np.max(s_max)),
            }

        return per_draw_stats, mixture_loglik, domain_stats

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

    per_draw_stats, mixture_loglik, domain_stats = make_scorer(
        bx1, bx2, log_w, z_prob, h[:, 0:1], h[:, 1:2], np.arange(n)
    )

    def run_chain_rule(theta_stack):
        theta_is_batched = jnp.ndim(theta_stack) == 3
        out = {key: np.zeros(n) for key in ("loglik", "S", "S_good", "S_bad", "p_outlier", "s_mean", "ds_deps_mean")}
        s_draw = np.zeros(n_draws)
        for start in range(0, n_draws, draw_chunk):
            stop = min(start + draw_chunk, n_draws)
            theta_block = theta_stack[start:stop] if theta_is_batched else theta_stack
            f_block = f_all[start:stop]
            in_axes = (0 if theta_is_batched else None, 0)
            batch = jax.vmap(per_draw_stats, in_axes=in_axes)(theta_block, f_block)
            values_np = [np.asarray(values, dtype=np.float64) for values in batch]
            weight = (stop - start) / n_draws
            for key, values in zip(out, values_np):
                out[key] += weight * values.mean(axis=0)
            s_draw[start:stop] = values_np[1].sum(axis=1)
        return out, s_draw

    out, s_draw = run_chain_rule(posterior_thetas)
    parsec_out, s_draw_parsec = run_chain_rule(parsec_theta)
    out["S_parsec"] = parsec_out["S"]

    # Exact finite-step difference for eps = ln(1.1), with a hard domain guard.
    eps10 = float(np.log(1.1))
    stats = {"eps0": domain_stats(jnp.asarray(0.0), posterior_thetas),
             "eps_ln1p1": domain_stats(jnp.asarray(eps10), posterior_thetas)}
    if stats["eps0"]["max_p_outside_system"] > 1e-6 or stats["eps_ln1p1"]["max_p_outside_system"] > 1e-6:
        raise RuntimeError(
            "Perturbed mass scale leaves the lookup domain with substantial "
            f"metallicity weight: {json.dumps(stats)}"
        )
    ll0 = mixture_loglik(jnp.asarray(0.0), posterior_thetas, f_all)
    ll10 = mixture_loglik(jnp.asarray(eps10), posterior_thetas, f_all)
    if not np.all(np.isfinite(np.asarray(ll10, dtype=np.float64))):
        raise RuntimeError("eps = ln(1.1) evaluation produced non-finite log-likelihoods.")
    out["dlnL10_exact"] = np.asarray(ll10, dtype=np.float64) - np.asarray(ll0, dtype=np.float64)

    # Chain-rule vs autodiff on a small random subset (validates the algebra).
    check = np.random.default_rng(0).choice(n, size=min(64, n), replace=False)
    check_draws = np.linspace(0, n_draws - 1, num=min(32, n_draws), dtype=np.int64)
    if np.any(check_draws < 0) or np.any(check_draws >= n_draws):
        raise RuntimeError("Autodiff-check draw indices out of range.")
    check_scorer, check_mixture, _ = make_scorer(
        bx1[check], bx2[check], log_w[check], z_prob[check], h[check, 0:1], h[check, 1:2], check
    )
    check_thetas = posterior_thetas[check_draws]
    check_f = f_all[check_draws]
    subset = None
    for d in range(len(check_draws)):
        one = check_scorer(check_thetas[d], check_f[d])
        subset = one if subset is None else tuple(a + b for a, b in zip(subset, one))
    subset = tuple(np.asarray(value, dtype=np.float64) / len(check_draws) for value in subset)
    auto = jax.jacobian(lambda e: check_mixture(e, check_thetas, check_f))(jnp.asarray(0.0))
    relative = np.abs(np.asarray(auto) - subset[1]) / np.maximum(np.abs(subset[1]), 1e-6)
    if not np.all(relative < 1e-4):
        raise RuntimeError(f"Chain-rule score disagrees with autodiff (max rel {np.max(relative):.2e}).")

    extras = {
        "S_draw": s_draw,
        "S_parsec_draw": s_draw_parsec,
        "chain_ids": data["chain_ids"],
        "n_chains": data["n_chains"],
        "domain_stats": stats,
        "reference_check": reference_derivative_check(
            data, out, np.asarray(posterior_thetas, dtype=np.float64),
            bx1, bx2, bz, h, mg_max=mg_max,
        ),
    }
    return out, h, extras


def reference_derivative_check(
    data, result, thetas, bx1, bx2, bz, h, *, mg_max, n_systems=8, n_draws=3, node_cover=0.9, max_nodes=15,
) -> dict:
    """Lookup slopes vs. direct adaptive Rice integration on extreme systems.

    Central differences of the FULL mixture + metallicity marginalization are
    evaluated twice for the same (draws, nodes, eps): once through the saved
    lookup, once through hm.rice_component_reference_integral.  Agreement
    isolates the accuracy of the tabulated slopes themselves.
    """
    from scipy.special import logsumexp

    rng = np.random.default_rng(1)
    z_prob = data["z_probabilities"]
    n = len(result["S"])
    picks = [
        int(np.argmax(result["S"])),
        int(np.argmin(result["S"])),
        int(np.argmax(result["p_outlier"])),
        int(np.argsort(result["p_outlier"])[-2]),
        int(np.argmax(data["u"] / result["s_mean"])),
    ]
    margin = np.minimum(result["s_mean"] - 0.25, 2.5 - result["s_mean"])
    picks += list(np.argsort(margin)[:2])
    picks += list(rng.choice(np.flatnonzero(result["S"] != 0), size=1, replace=False))
    picks = list(dict.fromkeys(picks))[:n_systems]
    draw_idx = np.linspace(0, thetas.shape[0] - 1, num=n_draws, dtype=np.int64)
    hs = (0.03, 0.1)

    records = []
    for sys_index in picks:
        weight = z_prob[sys_index]
        order = np.argsort(weight)[::-1][:max_nodes]
        cut = int(np.searchsorted(np.cumsum(weight[order]), node_cover * weight.sum()) + 1)
        kept = np.sort(order[:min(max_nodes, cut, weight.size)])
        w_sel = weight[kept] / weight[kept].sum()
        bx1_sys, bx2_sys = bx1[sys_index], bx2[sys_index]
        bz_sel = bz[kept]
        h1, h2 = h[sys_index, 0], h[sys_index, 1]
        u_sys, us_sys = float(data["u"][sys_index]), float(data["u_sigma"][sys_index])
        record = {"system": int(sys_index), "n_nodes": int(kept.size),
                  "S_chainrule": float(result["S"][sys_index])}
        grid_lut = np.asarray(data["lookup"].sqrt_mtot_grid, dtype=np.float64)
        row_good = np.asarray(data["lookup"].log_good[sys_index], dtype=np.float64)
        row_bad = np.asarray(data["lookup"].log_bad[sys_index], dtype=np.float64)
        for step in hs:
            ref_diffs, lut_diffs = [], []
            for d in draw_idx:
                theta = thetas[d]
                f = float(data["draws"]["f_outlier"][d])
                g1 = (bx1_sys @ theta) @ bz_sel.T
                g2 = (bx2_sys @ theta) @ bz_sel.T
                m1, m2 = np.power(10.0, g1), np.power(10.0, g2)
                ll_ref, ll_lut = [], []
                for sign in (1.0, -1.0):
                    s = np.sqrt(m1 * np.exp(h1 * sign * step) + m2 * np.exp(h2 * sign * step))
                    good_ref = np.array([
                        hm.rice_component_reference_integral(u_sys, us_sys, float(s_q), component="good") for s_q in s
                    ])
                    bad_ref = np.array([
                        hm.rice_component_reference_integral(u_sys, us_sys, float(s_q), component="bad") for s_q in s
                    ])
                    ll_ref.append(logsumexp(np.log(w_sel) + np.log((1.0 - f) * good_ref + f * bad_ref)))
                    good_lut = np.exp(np.interp(s, grid_lut, row_good))
                    bad_lut = np.exp(np.interp(s, grid_lut, row_bad))
                    ll_lut.append(logsumexp(np.log(w_sel) + np.log((1.0 - f) * good_lut + f * bad_lut)))
                ref_diffs.append((ll_ref[0] - ll_ref[1]) / (2.0 * step))
                lut_diffs.append((ll_lut[0] - ll_lut[1]) / (2.0 * step))
            record[f"S_reference_h{step}"] = float(np.mean(ref_diffs))
            record[f"S_lookup_fd_h{step}"] = float(np.mean(lut_diffs))
        record["reference_over_lookup"] = float(
            record[f"S_reference_h{hs[0]}"] / record[f"S_lookup_fd_h{hs[0]}"]
        ) if record[f"S_lookup_fd_h{hs[0]}"] != 0 else None
        records.append(record)
    return {
        "description": (
            "Central-difference scores of the full marginalized likelihood via "
            "direct adaptive Rice integration vs. the saved lookup (same draws, "
            "nodes and steps). Agreement validates the tabulated slopes."
        ),
        "systems": records,
        "max_abs_reference_minus_lookup_h0p03": float(np.max(np.abs([
            r["S_reference_h0.03"] - r["S_lookup_fd_h0.03"] for r in records
        ]))),
    }


def summarize(data: dict, result: dict, extras: dict, h: np.ndarray, *, mg_min: float, mg_max: float) -> dict:
    """Aggregate per-system scores into breakdown tables, plots and files."""
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

    tilde = np.linspace(1e-3, hm.RICE_GOOD_SUPPORT, 4001)
    density = hm.rice_good_raw(tilde)
    mean_tilde_u = float(np.trapezoid(tilde * density, tilde) / np.trapezoid(density, tilde))
    u_predicted = result["s_mean"] * mean_tilde_u
    u_ratio = data["u"] / u_predicted

    s10 = result["S"] * ln10p
    s10_good = result["S_good"] * ln10p
    s10_bad = result["S_bad"] * ln10p
    exact = result["dlnL10_exact"]

    def block_sum(mask):
        def split(values):
            inside = values[mask]
            return {"pos": float(inside[inside > 0].sum()), "neg": float(inside[inside < 0].sum())}

        return {
            "n": int(np.sum(mask)),
            "S10_linear_total": float(np.sum(s10[mask])),
            "S10_linear_good": float(np.sum(s10_good[mask])),
            "S10_linear_bad": float(np.sum(s10_bad[mask])),
            "S10_linear_good_pos_neg": split(s10_good),
            "S10_linear_bad_pos_neg": split(s10_bad),
            "S10_linear_median": float(np.median(s10[mask])) if np.any(mask) else None,
            "dlnL10_exact_total": float(np.sum(exact[mask])),
        }

    categories = ["both", "brighter_only", "fainter_only", "none"]
    u_quartile = np.quantile(u_ratio, [0.25, 0.5, 0.75])
    u_bins = np.digitize(u_ratio, u_quartile)
    quartile_labels = [
        f"Q1 (u_ratio <= {u_quartile[0]:.2f})",
        f"Q2 ({u_quartile[0]:.2f}-{u_quartile[1]:.2f})",
        f"Q3 ({u_quartile[1]:.2f}-{u_quartile[2]:.2f})",
        f"Q4 (> {u_quartile[2]:.2f})",
    ]

    s10_draw = extras["S_draw"] * ln10p
    chain_ids = extras["chain_ids"]
    per_chain = [
        {
            "chain": int(chain),
            "n_draws": int(np.sum(chain_ids == chain)),
            "S10_linear_total_mean": float(np.mean(s10_draw[chain_ids == chain])),
        }
        for chain in np.unique(chain_ids)
    ]

    summary = {
        "mg_window": [mg_min, mg_max],
        "n_systems": int(len(s10)),
        "n_chains": int(extras["n_chains"]),
        "n_draws": int(s10_draw.size),
        "units": {
            "S10_linear": "ln(1.1) * dlnL/deps at eps=0, a LOCAL LINEAR estimate, not the exact finite-step gain",
            "dlnL10_exact": "lnL(eps=ln 1.1) - lnL(0) over the full mixture and metallicity marginalization",
        },
        "total": block_sum(np.ones(len(s10), dtype=bool)),
        "S10_linear_at_parsec_projection_with_fitted_f": float(np.sum(result["S_parsec"] * ln10p)),
        "parsec_comparison_note": (
            "Holds the f_outlier draws from the fitted surface fixed and evaluates "
            "the score at the PARSEC spline projection; a derivative comparison, "
            "not a refit and not a fraction of solved bias."
        ),
        "by_category": {name: block_sum(category == name) for name in categories},
        "by_u_ratio_quartile": {
            quartile_labels[index]: block_sum(u_bins == index) for index in range(4)
        },
        "by_outlier_responsibility": {
            "p_out<0.5": block_sum(result["p_outlier"] < 0.5),
            "p_out>=0.5": block_sum(result["p_outlier"] >= 0.5),
        },
        "total_S10_linear_per_draw": {
            "mean": float(np.mean(s10_draw)),
            "std": float(np.std(s10_draw)),
            "q16": float(np.quantile(s10_draw, 0.16)),
            "q84": float(np.quantile(s10_draw, 0.84)),
            "min": float(np.min(s10_draw)),
            "max": float(np.max(s10_draw)),
        },
        "per_chain_total_S10_linear_mean": per_chain,
        "exact_vs_linear": {
            "total_exact": float(np.sum(exact)),
            "median_ratio_exact_over_linear": float(
                np.median((exact / s10)[np.abs(s10) > 1e-8])
            ),
            "n_sign_flips_exact_vs_linear": int(np.sum(np.sign(exact) != np.sign(s10))),
        },
        "domain_and_floor": extras["domain_stats"],
        "reference_derivative_check": extras["reference_check"],
        "mean_tilde_u_good_kernel": mean_tilde_u,
        "interpretation_notes": {
            "quartile_pattern": (
                "A correctly specified kernel with correct masses also concentrates "
                "the positive derivative in the fast quartile (the quartile sums of "
                "the mean score vanish overall). Whether the observed pattern "
                "exceeds model expectation must be judged against simulated ranges."
            ),
            "u_ratio": "u_obs / (s_posterior_mean * <tilde_u>_good); s depends on the fitted MLR, so this is model-conditioned.",
        },
        "provenance": data["provenance"],
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
        "S10_linear": s10,
        "dlnL10_exact": exact,
        "S_parsec": result["S_parsec"],
    }
    return summary, per_system


def plot_breakdown(per_system: dict, summary: dict, path: Path, *, mg_min: float, mg_max: float) -> dict:
    s10 = per_system["S10_linear"]
    s10_good = per_system["S_good"] * float(np.log(1.1))
    s10_bad = per_system["S_bad"] * float(np.log(1.1))
    categories = ["both", "brighter_only", "fainter_only", "none"]
    hidden_report = {}

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    ax = axes[0, 0]
    sums_good = [summary["by_category"][name]["S10_linear_good"] for name in categories]
    sums_bad = [summary["by_category"][name]["S10_linear_bad"] for name in categories]
    ax.bar(categories, sums_good, color="tab:blue", label="good kernel")
    ax.bar(categories, sums_bad, bottom=sums_good, color="tab:red", label="outlier branch")
    for index, total in enumerate(np.asarray(sums_good) + np.asarray(sums_bad)):
        ax.text(index, total, f"{total:+.1f}", ha="center", va="bottom" if total >= 0 else "top", fontsize=9)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_ylabel(r"$\sum_j S_j \times \ln 1.1$ (local linear)")
    ax.set_title(f"Local score by window membership (M_G in [{mg_min:g}, {mg_max:g}])")
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
        sums = []
        for b in range(len(centers)):
            upper_ok = fainter <= edges[b + 1] if b == len(centers) - 1 else fainter < edges[b + 1]
            sums.append(np.sum(s10[fainter_in_window & mask & (fainter >= edges[b]) & upper_ok]))
        ax.bar(centers + (offset - 0.5) * width, sums, width=width, color=color, label=label)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xlabel("M_G of the fainter component (restricted to the window, last bin closed)")
    ax.set_ylabel(r"$\sum_j S_j \times \ln 1.1$ (local linear)")
    ax.set_title("Local score vs companion magnitude and brightness ratio")
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    u_quartile = np.quantile(per_system["u_ratio"], [0.25, 0.5, 0.75])
    u_bins = np.digitize(per_system["u_ratio"], u_quartile)
    good_sums = [np.sum(s10_good[u_bins == index]) for index in range(4)]
    bad_sums = [np.sum(s10_bad[u_bins == index]) for index in range(4)]
    labels = [
        f"Q1\n<= {u_quartile[0]:.2f}",
        f"Q2\n{u_quartile[0]:.2f}-{u_quartile[1]:.2f}",
        f"Q3\n{u_quartile[1]:.2f}-{u_quartile[2]:.2f}",
        f"Q4\n> {u_quartile[2]:.2f}",
    ]
    x = np.arange(4)
    ax.bar(x, good_sums, color="tab:blue", label="good kernel")
    ax.bar(x, bad_sums, bottom=good_sums, color="tab:red", label="outlier branch")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(x, labels)
    ax.set_xlabel("u_obs / (s * <tilde_u>_good), model-conditioned")
    ax.set_ylabel(r"$\sum_j S_j \times \ln 1.1$ (local linear)")
    ax.set_title("Local score by velocity-residual quartile")
    ax.legend(fontsize=9)

    ax = axes[1, 1]
    sc = ax.scatter(per_system["u_ratio"], s10, c=per_system["p_outlier"], s=8, cmap="viridis", alpha=0.8)
    fig.colorbar(sc, ax=ax, label="outlier responsibility")
    ax.axhline(0.0, color="black", linewidth=0.8)
    x_zoom = float(np.quantile(per_system["u_ratio"], 0.995))
    ax.set_xlim(0, x_zoom)
    ax.set_ylim(-0.12, 0.5)
    ax.set_xlabel("u_obs / (s * <tilde_u>_good), model-conditioned")
    ax.set_ylabel(r"$S_j \times \ln 1.1$ (local linear)")
    ax.set_title("Per-system local scores vs velocity residual")

    hidden = (per_system["u_ratio"] > x_zoom) | (s10 < -0.12) | (s10 > 0.5)
    hidden_report = {
        "n_hidden": int(np.sum(hidden)),
        "S10_linear_sum_hidden": float(np.sum(s10[hidden])),
        "S10_linear_max_hidden": float(np.max(s10[hidden])) if np.any(hidden) else None,
        "S10_linear_min_hidden": float(np.min(s10[hidden])) if np.any(hidden) else None,
        "note": "Main panel limits: u_ratio <= 99.5th pct, S10 in [-0.12, 0.5]; full-range figure saved separately.",
    }
    if np.any(hidden):
        ax.annotate(
            f"{hidden_report['n_hidden']} pts outside range\n"
            f"sum {hidden_report['S10_linear_sum_hidden']:+.2f}\n"
            f"max {hidden_report['S10_linear_max_hidden']:+.2f}",
            xy=(0.97, 0.97), xycoords="axes fraction", ha="right", va="top", fontsize=8,
            bbox=dict(boxstyle="round", fc="white", alpha=0.8),
        )

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return hidden_report


def plot_full_range(per_system: dict, path: Path) -> None:
    s10 = per_system["S10_linear"]
    fig, ax = plt.subplots(figsize=(7, 5.5))
    sc = ax.scatter(per_system["u_ratio"], s10, c=per_system["p_outlier"], s=10, cmap="viridis", alpha=0.8)
    fig.colorbar(sc, ax=ax, label="outlier responsibility")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xlabel("u_obs / (s * <tilde_u>_good), model-conditioned")
    ax.set_ylabel(r"$S_j \times \ln 1.1$ (local linear)")
    ax.set_title("Per-system local scores, full range")
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
    parser.add_argument("--metallicity-posterior", type=Path, default=None,
                        help="Metallicity posterior npz; defaults to the source recorded in mlr_model.json.")
    parser.add_argument("--mg-min", type=float, default=7.0)
    parser.add_argument("--mg-max", type=float, default=10.0)
    parser.add_argument("--output", type=Path, default=Path("results/mg_window_score_t8_1"))
    args = parser.parse_args()

    data = load_stage_two(args.run_dir, args.data, args.metallicity_posterior)
    result, h, extras = evaluate_scores(data, mg_min=args.mg_min, mg_max=args.mg_max)
    summary, per_system = summarize(data, result, extras, h, mg_min=args.mg_min, mg_max=args.mg_max)

    args.output.mkdir(parents=True, exist_ok=True)
    table = Table()
    for key, value in per_system.items():
        table[key] = value
    table.write(args.output / "mg_window_scores.csv", format="csv", overwrite=True)
    (args.output / "mg_window_score_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    np.savez(
        args.output / "mg_window_score_draws.npz",
        S10_linear_per_draw=extras["S_draw"] * float(np.log(1.1)),
        S10_parsec_per_draw=extras["S_parsec_draw"] * float(np.log(1.1)),
        chain_ids=extras["chain_ids"],
    )
    hidden_report = plot_breakdown(per_system, summary, args.output / "mg_window_scores.png",
                                   mg_min=args.mg_min, mg_max=args.mg_max)
    plot_full_range(per_system, args.output / "mg_window_scores_full.png")
    summary["hidden_points_main_panel"] = hidden_report
    (args.output / "mg_window_score_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print(json.dumps({k: summary[k] for k in (
        "total", "S10_linear_at_parsec_projection_with_fitted_f",
        "by_outlier_responsibility", "total_S10_linear_per_draw",
        "per_chain_total_S10_linear_mean", "exact_vs_linear",
        "domain_and_floor", "hidden_points_main_panel",
    )}, indent=2))
    print("reference_derivative_check max|ref-lut| h=0.03:",
          summary["reference_derivative_check"]["max_abs_reference_minus_lookup_h0p03"])


if __name__ == "__main__":
    main()
