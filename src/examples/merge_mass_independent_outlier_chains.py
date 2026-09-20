#!/usr/bin/env python3
"""Merge the per-chain mass-independent-outlier runs and produce formal outputs.

Combines OUTPUT/chain_XX/mlr_mcmc_t8.npz into a chain-aware merged posterior,
checks rank-normalized split R-hat, bulk/tail ESS, divergences, and
lookup-boundary contact, and writes the A/B/PARSEC comparison figures, the
representative-mass table, per-source outlier probabilities, and the
observed-u posterior predictive check.
"""

import argparse
import csv
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import ndtr, ndtri
from scipy.stats import rankdata

import run_hierarchical_metallicity_test as workflow
import run_mass_independent_outlier as runner
from binary_masses import hierarchical_metallicity as hm


REPRESENTATIVE_MG = (5.0, 7.0, 9.0, 11.0, 13.0)
OVERLAY_MH = (-1.0, -0.5, 0.0, 0.3, 0.6)
TRACE_POINTS = ((9.0, 0.0), (5.0, -1.0), (13.0, 0.6))
TRACE_STRIDE = 4
RHAT_LIMIT = 1.01
ESS_LIMIT = 400.0
SPARSE_SYSTEMS_PER_BIN = 25
COLOR_A = "#7b6ba8"
COLOR_B = "#da7030"
SURFACE_PARAMS = ("c0", "a", "b", "r", "log_lambda_x", "log_lambda_z")


def parse_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--baseline", type=Path, required=True,
                        help="A-group directory (posterior, lookup, MLR, derived grid).")
    parser.add_argument("--run-root", type=Path, required=True,
                        help="Mass-independent-outlier output root holding chain_XX.")
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--grid-draws", type=int, default=512,
                        help="Posterior draws for the derived MLR grid and figures.")
    parser.add_argument("--p-out-draws", type=int, default=256,
                        help="Posterior draws for per-source outlier probabilities.")
    parser.add_argument("--predictive-draws", type=int, default=200,
                        help="Posterior draws for the observed-u predictive check.")
    parser.add_argument("--boundary-fraction", type=float, default=0.005,
                        help="Fraction of the lookup domain treated as boundary contact.")
    merge_args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining
    return merge_args, workflow.parse_args()


def _rank_normalize(x):
    # Ranks are taken over all chains and draws jointly (ties averaged), then
    # mapped through the normal quantile function (Blom offset).
    ranks = rankdata(x, method="average").reshape(x.shape)
    return ndtri((ranks - 0.375) / (x.size + 0.25))


def diagnostics_for(x):
    """Rank-normalized split R-hat and bulk/tail ESS for one scalar parameter."""
    from numpyro.diagnostics import effective_sample_size, split_gelman_rubin

    bulk = float(effective_sample_size(_rank_normalize(x)))
    q05, q95 = np.quantile(x, [0.05, 0.95])
    tail = float(min(effective_sample_size((x <= q05).astype(np.float64)),
                     effective_sample_size((x <= q95).astype(np.float64))))
    folded = _rank_normalize(np.abs(x - np.median(x)))
    rhat = float(max(split_gelman_rubin(_rank_normalize(x)),
                     split_gelman_rubin(folded)))
    return {"r_hat_rank_normalized": rhat, "ess_bulk": bulk, "ess_tail": tail}


def scalar_views(posterior):
    for key in sorted(posterior):
        if not key.startswith("posterior__"):
            continue
        name = key[len("posterior__"):]
        values = np.asarray(posterior[key])
        for index in np.ndindex(values.shape[2:]):
            label = name if not index else f"{name}[{','.join(map(str, index))}]"
            yield label, values[(slice(None), slice(None)) + index]


def interp_rows(grid, table, x):
    """Row-wise linear interpolation; grid (K,), table (n, K), x (n, M)."""
    n, m = x.shape
    upper = np.clip(np.searchsorted(grid, x), 1, len(grid) - 1)
    lower = upper - 1
    weight = (x - grid[lower]) / (grid[upper] - grid[lower])
    rows = np.repeat(np.arange(n), m)
    low = table[rows, lower.ravel()].reshape(n, m)
    high = table[rows, upper.ravel()].reshape(n, m)
    return low * (1.0 - weight) + high * weight


def flat_samples(merged):
    return {key[len("posterior__"):]: np.asarray(value).reshape(
                (-1,) + np.asarray(value).shape[2:])
            for key, value in merged.items() if key.startswith("posterior__")}


def surface_draw(mlr, merged, chain, draw_index):
    params = {name: np.asarray(merged["posterior__" + name])[chain, draw_index]
              for name in SURFACE_PARAMS}
    return params


def sparse_counts(absg, bins):
    counts, _ = np.histogram(np.concatenate([absg[:, 0], absg[:, 1]]), bins=bins)
    return counts


def shade_sparse(axes, bins, sparse):
    for ax in np.atleast_1d(axes):
        for start, width in zip(bins[:-1][sparse], np.diff(bins)[sparse]):
            ax.axvspan(start, start + width, color="0.88", zorder=0)


def merge_chains(merge_args, run_root):
    chain_dirs = [run_root / f"chain_{i:02d}" for i in range(merge_args.chains)]
    payloads = []
    for directory in chain_dirs:
        path = directory / workflow.T8_MLR_NAME
        if not path.exists():
            raise SystemExit(f"Missing chain result: {path}")
        payloads.append(np.load(path))
    keys = sorted(payloads[0].files)
    merged = {}
    for key in keys:
        arrays = [np.asarray(payload[key]) for payload in payloads]
        shapes = {array.shape[1:] for array in arrays}
        if len(shapes) != 1:
            raise SystemExit(f"Inconsistent shapes for {key}: {shapes}")
        merged[key] = np.concatenate(arrays, axis=0)
    np.savez_compressed(run_root / workflow.T8_MLR_NAME, **merged)
    per_chain_divergences = [
        int(np.sum(np.asarray(payload["sample_stats__diverging"]))) for payload in payloads
    ]
    return merged, per_chain_divergences


def write_diagnostics(merge_args, merged, per_chain_divergences, mlr, run_root):
    parameters = {label: diagnostics_for(values) for label, values in scalar_views(merged)}
    worst = {
        "max_r_hat_rank_normalized": max(item["r_hat_rank_normalized"]
                                        for item in parameters.values()),
        "min_ess_bulk": min(item["ess_bulk"] for item in parameters.values()),
        "min_ess_tail": min(item["ess_tail"] for item in parameters.values()),
    }
    divergences = int(np.sum(np.asarray(merged["sample_stats__diverging"])))
    gates = {
        "r_hat_below_1p01": worst["max_r_hat_rank_normalized"] < RHAT_LIMIT,
        "ess_bulk_at_least_400": worst["min_ess_bulk"] >= ESS_LIMIT,
        "ess_tail_at_least_400": worst["min_ess_tail"] >= ESS_LIMIT,
        "no_divergences": divergences == 0,
    }
    report = {"n_chains": merge_args.chains,
              "draws_per_chain": int(np.asarray(merged["posterior__f_outlier"]).shape[1]),
              "worst": worst, "limits": {"r_hat": RHAT_LIMIT, "ess": ESS_LIMIT},
              "divergences_total": divergences,
              "divergences_per_chain": per_chain_divergences,
              "gates": gates, "parameters": parameters}
    (run_root / "mlr_diagnostics_t8.json").write_text(json.dumps(report, indent=2))
    workflow.save_posterior_summary(flat_samples(merged), run_root / "mlr_summary_t8.csv")

    chains, draws = np.asarray(merged["posterior__f_outlier"]).shape[:2]
    figure, axes = plt.subplots(2, 2, figsize=(11, 6.4))
    ax = axes[0, 0]
    for chain in range(chains):
        ax.plot(np.arange(draws), merged["posterior__f_outlier"][chain], lw=0.4,
                label=f"chain {chain}")
    ax.set_ylabel(r"$f_{\rm outlier}$")
    ax.set_xlabel("draw")
    ax.legend(fontsize=8, ncol=2)
    stride_indices = np.arange(0, draws, TRACE_STRIDE)
    for ax, (mg, mh) in zip(axes.flat[1:], TRACE_POINTS):
        for chain in range(chains):
            values = [10.0 ** mlr.g_from_raw(np.asarray(mg), np.asarray(mh),
                                            surface_draw(mlr, merged, chain, index))
                      for index in stride_indices]
            ax.plot(stride_indices, values, lw=0.4)
        ax.set_ylabel(rf"$M(M_G={mg:g}, [M/H]={mh:+g})$ [$M_\odot$]")
        ax.set_xlabel("draw")
    figure.suptitle(f"Merged B-group traces (every {TRACE_STRIDE}th draw)")
    figure.tight_layout()
    figure.savefig(run_root / "traces.png", dpi=160)
    plt.close(figure)
    return report


def main():
    merge_args, workflow_args = parse_args()
    started = time.monotonic()
    run_root = merge_args.run_root
    rng = np.random.default_rng(20260919)

    merged, per_chain_divergences = merge_chains(merge_args, run_root)

    runner_state = SimpleNamespace(baseline=merge_args.baseline, output=run_root,
                                   mode="merge", chain_index=None, start_draw=0,
                                   reference_workers=1, progress_bar=False)
    posterior, arrays, lookup, checks = runner.load_systems(runner_state, workflow_args)
    saved = np.load(run_root / "outlier_likelihood.npz")
    log_bad = np.asarray(saved["log_likelihood"], dtype=np.float64)
    validation = json.loads((run_root / "validation.json").read_text())
    mlr = runner.build_model(runner_state, workflow_args, posterior, arrays, lookup,
                             log_bad, float(validation["nonfloor_max_log_error"]))
    mlr.posterior_samples = flat_samples(merged)

    report = write_diagnostics(merge_args, merged, per_chain_divergences, mlr, run_root)

    grid = mlr.correction_grid(max_draws=merge_args.grid_draws)
    np.savez_compressed(run_root / "mlr_derived_grid_t8.npz", **grid)

    old = np.load(merge_args.baseline / "mlr_derived_grid_t8.npz")
    if not (np.allclose(old["absg_grid"], grid["absg_grid"])
            and np.allclose(old["z_grid"], grid["z_grid"])):
        raise SystemExit("Baseline and merged derived grids use different M_G or [M/H] grids.")
    x, z = grid["absg_grid"], grid["z_grid"]
    bins = np.arange(3.5, 13.75, 0.25)
    counts = sparse_counts(arrays["absg"], bins)
    sparse = counts < SPARSE_SYSTEMS_PER_BIN

    figure, axes = plt.subplots(2, len(OVERLAY_MH), figsize=(3.1 * len(OVERLAY_MH), 6.6),
                                sharex=True, gridspec_kw={"height_ratios": [2.4, 1]})
    table_rows = []
    for column, mh in enumerate(OVERLAY_MH):
        iz = int(np.argmin(abs(z - mh)))
        ax, ratio_ax = axes[0, column], axes[1, column]
        shade_sparse([ax, ratio_ax], bins, sparse)
        for result, label, color in [(old, "A: mass-scaled outliers", COLOR_A),
                                     (grid, "B: mass-independent outliers", COLOR_B)]:
            p = result["mass_percentiles"][:, iz]
            ax.plot(x, p[1], color=color, label=label)
            ax.fill_between(x, p[0], p[2], color=color, alpha=0.18)
        ax.plot(x, grid["parsec_mass"][iz], "--", color="0.35", label="PARSEC")
        ax.set_title(f"[M/H] = {z[iz]:+.1f}")
        change = 100.0 * (grid["mass_percentiles"][1, iz] / old["mass_percentiles"][1, iz] - 1.0)
        ratio_ax.plot(x, change, color=COLOR_B)
        ratio_ax.axhline(0.0, color="0.5", lw=0.8)
        if column == 0:
            ax.set_ylabel(r"Mass [$M_\odot$]")
            ratio_ax.set_ylabel("B/A median\nchange [%]")
        ratio_ax.set_xlabel(r"$M_G$ [mag]")
        for mg in REPRESENTATIVE_MG:
            ix = int(np.argmin(abs(x - mg)))
            k = int(np.clip(np.searchsorted(bins, x[ix]) - 1, 0, counts.size - 1))
            table_rows.append({
                "mh": float(z[iz]), "M_G": float(x[ix]),
                "A_p16": float(old["mass_percentiles"][0, iz, ix]),
                "A_p50": float(old["mass_percentiles"][1, iz, ix]),
                "A_p84": float(old["mass_percentiles"][2, iz, ix]),
                "B_p16": float(grid["mass_percentiles"][0, iz, ix]),
                "B_p50": float(grid["mass_percentiles"][1, iz, ix]),
                "B_p84": float(grid["mass_percentiles"][2, iz, ix]),
                "PARSEC": float(grid["parsec_mass"][iz, ix]),
                "B_over_A_median_change_percent": float(change[ix]),
                "systems_in_neighbourhood": int(np.sum(
                    counts[max(k - 1, 0):min(k + 2, counts.size)])),
            })
    axes[0, 0].legend(fontsize=8)
    figure.suptitle("Fixed default dynamics shape · full 14,876-system sample · "
                    f"grey bands: fewer than {SPARSE_SYSTEMS_PER_BIN} systems per 0.25 mag",
                    fontsize=11)
    figure.tight_layout()
    figure.savefig(run_root / "mlr_ab_parsec_overlay.png", dpi=170)
    plt.close(figure)
    with (run_root / "mass_table.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(table_rows[0]))
        writer.writeheader()
        writer.writerows(table_rows)

    f_a = np.asarray(np.load(merge_args.baseline / workflow.T8_MLR_NAME)["posterior__f_outlier"])
    f_b = np.asarray(merged["posterior__f_outlier"])
    figure, ax = plt.subplots(figsize=(6.4, 4.4))
    for values, label, color in [(f_a.ravel(), "A: mass-scaled outliers", COLOR_A),
                                 (f_b.ravel(), "B: mass-independent outliers", COLOR_B)]:
        ax.hist(values, bins=60, density=True, histtype="step", lw=1.8, color=color, label=label)
        q16, q50, q84 = np.percentile(values, [16, 50, 84])
        ax.axvline(q50, color=color, lw=1.0, ls=":")
        print(f"f_outlier {label}: 16/50/84 = {q16:.4f}/{q50:.4f}/{q84:.4f}", flush=True)
    ax.set_xlabel(r"$f_{\rm outlier}$")
    ax.set_ylabel("posterior density")
    ax.legend()
    figure.tight_layout()
    figure.savefig(run_root / "f_outlier_comparison.png", dpi=170)
    plt.close(figure)

    f_draws = mlr.posterior_samples["f_outlier"]
    indices = rng.choice(f_draws.size, size=min(merge_args.p_out_draws, f_draws.size),
                         replace=False)
    sqrt_grid = np.asarray(lookup.sqrt_mtot_grid, dtype=np.float64)
    log_good = np.asarray(lookup.log_good, dtype=np.float64)
    good_mixture = np.empty((arrays["absg"].shape[0], indices.size))
    domain = (float(sqrt_grid[0]), float(sqrt_grid[-1]))
    lower_margin, upper_margin, outside = np.inf, np.inf, 0
    for column, index in enumerate(indices):
        params = {name: mlr.posterior_samples[name][index] for name in SURFACE_PARAMS}
        s = np.sqrt(
            np.power(10.0, mlr.g_from_raw(arrays["absg"][:, 0, None], mlr.z_grid[None, :], params))
            + np.power(10.0, mlr.g_from_raw(arrays["absg"][:, 1, None], mlr.z_grid[None, :], params)))
        good_mixture[:, column] = np.sum(
            posterior.probabilities * np.exp(interp_rows(sqrt_grid, log_good, s)), axis=1)
        lower_margin = min(lower_margin, float(np.min(s)) - domain[0])
        upper_margin = min(upper_margin, domain[1] - float(np.max(s)))
        if np.min(s) < domain[0] or np.max(s) > domain[1]:
            outside += 1
    fraction = f_draws[indices][None, :]
    likelihood = (1.0 - fraction) * good_mixture + fraction * np.exp(log_bad)[:, None]
    p_out = fraction * np.exp(log_bad)[:, None] / likelihood
    p_out_mean = np.mean(p_out, axis=1)
    np.savez_compressed(run_root / "per_source_outlier_probability.npz",
                        row_indices=posterior.row_indices, u=arrays["u"],
                        u_sigma=arrays["u_sigma"], absg1=arrays["absg"][:, 0],
                        p_out_mean=p_out_mean, p_out_std=np.std(p_out, axis=1),
                        draw_indices=indices)
    figure, axes = plt.subplots(1, 2, figsize=(10.4, 4.2))
    for ax, coordinate, label in [
            (axes[0], arrays["u"], r"raw $u$ [km s$^{-1}\sqrt{\rm AU}$]"),
            (axes[1], arrays["absg"][:, 0], r"$M_G$ of primary [mag]")]:
        edges = np.quantile(coordinate, np.linspace(0, 1, 21))
        centers = 0.5 * (edges[:-1] + edges[1:])
        which = np.digitize(coordinate, edges[1:-1])
        medians = np.array([np.median(p_out_mean[which == k]) if np.any(which == k)
                            else np.nan for k in range(20)])
        ax.hexbin(coordinate, p_out_mean, gridsize=60, bins="log", cmap="viridis", mincnt=1)
        ax.plot(centers, medians, color=COLOR_B, lw=2.0, label="binned median")
        ax.set_xlabel(label)
        ax.set_ylabel(r"mean $P({\rm outlier}\mid j)$")
        ax.set_ylim(0.0, 1.0)
        ax.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(run_root / "per_source_outlier_probability.png", dpi=170)
    plt.close(figure)

    predictive_indices = rng.choice(f_draws.size,
                                    size=min(merge_args.predictive_draws, f_draws.size),
                                    replace=False)
    w_grid = np.linspace(0.0, 80.0, 2001)
    good_pdf = hm.rice_good_raw(w_grid)
    good_cdf = np.concatenate([[0.0], np.cumsum(0.5 * np.diff(w_grid)
                                                * (good_pdf[:-1] + good_pdf[1:]))])
    good_cdf /= good_cdf[-1]
    outlier_norm = ndtr(40.0 / 13.0) - ndtr(-40.0 / 13.0)
    cumulative_z = np.cumsum(posterior.probabilities, axis=1)
    z_max = cumulative_z.shape[1] - 1
    simulated = np.empty((predictive_indices.size, arrays["u"].size))
    for row, index in enumerate(predictive_indices):
        params = {name: mlr.posterior_samples[name][index] for name in SURFACE_PARAMS}
        z_choice = (rng.random(arrays["absg"].shape[0])[:, None]
                    > cumulative_z).sum(axis=1).clip(0, z_max)
        z_selected = mlr.z_grid[z_choice]
        s = np.sqrt(np.power(10.0, mlr.g_from_raw(arrays["absg"][:, 0], z_selected, params))
                    + np.power(10.0, mlr.g_from_raw(arrays["absg"][:, 1], z_selected, params)))
        good_component = s * np.interp(rng.random(s.size), good_cdf, w_grid)
        bad_component = 40.0 + 13.0 * ndtri(ndtr(-40.0 / 13.0) + rng.random(s.size) * outlier_norm)
        velocity = np.where(rng.random(s.size) < f_draws[index], bad_component, good_component)
        sigma = arrays["u_sigma"]
        simulated[row] = np.hypot(velocity + sigma * rng.standard_normal(s.size),
                                  sigma * rng.standard_normal(s.size))
    figure, axes = plt.subplots(1, 2, figsize=(10.4, 4.2))
    upper = np.quantile(np.concatenate([arrays["u"], simulated.ravel()]), 0.999)
    grid_u = np.linspace(1e-3, upper, 160)
    axes[0].hist(arrays["u"], bins=grid_u, density=True, histtype="step", lw=1.8,
                 color="0.2", label="observed $u$")
    axes[0].hist(simulated.ravel(), bins=grid_u, density=True, histtype="step", lw=1.8,
                 color=COLOR_B, label="posterior predictive")
    axes[0].set_xlabel(r"raw $u$ [km s$^{-1}\sqrt{\rm AU}$]")
    axes[0].set_ylabel("density")
    axes[0].legend(fontsize=8)
    for values, color in [(arrays["u"], "0.2"), (simulated.ravel(), COLOR_B)]:
        ordered = np.sort(values)
        axes[1].plot(ordered, np.arange(1, ordered.size + 1) / ordered.size, color=color, lw=1.4)
    axes[1].set_xlabel(r"raw $u$ [km s$^{-1}\sqrt{\rm AU}$]")
    axes[1].set_ylabel("empirical CDF")
    figure.suptitle("Posterior predictive of observed $u$ (same errors, same systems)",
                    fontsize=11)
    figure.tight_layout()
    figure.savefig(run_root / "u_posterior_predictive.png", dpi=170)
    plt.close(figure)

    boundary = {"domain": list(domain), "min_lower_margin": lower_margin,
                "min_upper_margin": upper_margin,
                "draws_outside_domain": outside,
                "contact_threshold_fraction": merge_args.boundary_fraction,
                "contact": bool(min(lower_margin, upper_margin)
                                < merge_args.boundary_fraction * (domain[1] - domain[0])
                                or outside > 0)}
    summary = {
        "elapsed_seconds": time.monotonic() - started,
        "diagnostics_gates": report["gates"],
        "diagnostics_worst": report["worst"],
        "divergences_total": report["divergences_total"],
        "f_outlier_A_percentiles": np.percentile(f_a, [16, 50, 84]).tolist(),
        "f_outlier_B_percentiles": np.percentile(f_b, [16, 50, 84]).tolist(),
        "lookup_boundary": boundary,
        "pass": bool(all(report["gates"].values()) and not boundary["contact"]),
    }
    (run_root / "merge_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    if not summary["pass"]:
        raise SystemExit("Merged-run gates failed; inspect mlr_diagnostics_t8.json "
                         "and merge_summary.json before using the results.")


if __name__ == "__main__":
    main()
