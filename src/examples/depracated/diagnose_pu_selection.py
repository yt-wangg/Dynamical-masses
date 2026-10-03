"""Selection-effect diagnostics for p(tilde-u) (handoff MLR_p_utilde_discussion.md, sec. 4.5-4.6).

Level 1: forward-simulate the fixed-T8 and EM models, apply the sample's hard cut
         u_obs/sigma_u > 3, and compare predicted vs observed CDFs.
Level 2: profile the global mass scale alpha (m -> alpha*m) with the shape and f
         held fixed, with and without the truncation normalization
         Pi_j(m) = P(u_obs > 3 sigma_j | m, mixture).

Point-estimate diagnostics only; nothing is refit except a 1-D profile in alpha.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import ive
from scipy.stats import ncx2

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "examples"))

import plot_observed_u_em_fixed_predictive as base  # noqa: E402
import run_hierarchical_metallicity_test as workflow  # noqa: E402

CUT = 3.0
DEFAULT_OUT = ROOT / "results/pu_selection_diagnostic_20261001"


def total_mass(mlr, params, absg, z):
    return np.asarray(mlr.mass_from_absg_mh(absg[:, 0], z, params), float) + np.asarray(
        mlr.mass_from_absg_mh(absg[:, 1], z, params), float
    )


def parsec_mass(mlr, absg, z):
    return sum(10.0 ** np.asarray(mlr.g_parsec_projection(absg[:, k], z), float) for k in range(2))


def good_density_grid(shape, grid):
    B, uc, C = shape
    basis = grid * np.exp(-B * grid**2 - np.exp((grid - uc) / C))
    return basis / np.trapezoid(basis, grid)


def outlier_density_grid(grid, mu=base.OUTLIER_MU, sig=base.OUTLIER_SIGMA, vmax=base.OUTLIER_MAX):
    from scipy.stats import norm

    dens = norm.pdf(grid, mu, sig)
    return dens / np.trapezoid(np.where(grid <= vmax, dens, 0.0), grid)


_Q_X = np.linspace(0.0, 400.0, 80001)
_Q_TAB = ncx2.sf(CUT**2, 2, _Q_X**2)


def marcum_q1_cut(x):
    """Q1(x, CUT) = P(Rice > CUT*sigma | v = x*sigma); tabulated since only x enters."""
    return np.interp(x, _Q_X, _Q_TAB)


def rice_pdf(u, v, sigma):
    x = u * v / sigma**2
    return u / sigma**2 * np.exp(-0.5 * ((u - v) / sigma) ** 2) * ive(0, x)


def mixture_terms(u, sigma, mass, shape, f, grid):
    """Return (L_j, Pi_j): mixture likelihood p(u_j|m) and pass probability P(u>3 sigma|m)."""
    wgrid = np.trapezoid  # alias for readability
    pg = good_density_grid(shape, grid)
    po = outlier_density_grid(grid)
    sq = np.sqrt(mass)[:, None]
    v_good = sq * grid[None, :]
    rice_g = rice_pdf(u[:, None], v_good, sigma[:, None])
    L_good = wgrid(pg[None, :] * rice_g, grid, axis=1)
    surv_g = marcum_q1_cut(v_good / sigma[:, None])
    Pi_good = wgrid(pg[None, :] * surv_g, grid, axis=1)
    rice_o = rice_pdf(u[:, None], grid[None, :], sigma[:, None])
    L_out = wgrid(po[None, :] * rice_o, grid, axis=1)
    surv_o = marcum_q1_cut(grid[None, :] / sigma[:, None])
    Pi_out = wgrid(po[None, :] * surv_o, grid, axis=1)
    return (1 - f) * L_good + f * L_out, (1 - f) * Pi_good + f * Pi_out


def ks_like(a, b):
    """Max CDF distance between two samples (a observed, b weighted-equal simulated)."""
    x = np.sort(np.concatenate([a, b]))
    ca = np.searchsorted(np.sort(a), x, side="right") / a.size
    cb = np.searchsorted(np.sort(b), x, side="right") / b.size
    return float(np.max(np.abs(ca - cb)))


def simulate(mlr, params, shape, f, arrays, probabilities, z_grid, reps, rng):
    n = len(arrays["u"])
    sig = arrays["u_sigma"]
    z_cdf = np.cumsum(probabilities / probabilities.sum(1, keepdims=True), axis=1)
    u_all, sig_all, m_all, pass_all = [], [], [], []
    for _ in range(reps):
        z = z_grid[np.sum(z_cdf < rng.random(n)[:, None], axis=1).clip(0, z_grid.size - 1)]
        mass = total_mass(mlr, params, arrays["absg"], z) if params is not None else parsec_mass(mlr, arrays["absg"], z)
        isout = rng.random(n) < f
        v = np.where(isout, base.sample_truncated_normal(rng, n), np.sqrt(mass) * base.sample_good(rng.random(n), shape))
        u = base.rice_draw(v, sig, rng.normal(size=n), rng.normal(size=n))
        u_all.append(u); sig_all.append(sig); m_all.append(mass); pass_all.append(u / sig > CUT)
    return (np.concatenate(x) for x in (u_all, sig_all, m_all, pass_all))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=base.DEFAULT_DATA)
    ap.add_argument("--posterior", type=Path, default=base.DEFAULT_POSTERIOR)
    ap.add_argument("--em-output", type=Path, default=base.DEFAULT_EM)
    ap.add_argument("--fixed-output", type=Path, default=base.DEFAULT_FIXED)
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--reps", type=int, default=32)
    ap.add_argument("--n-grid", type=int, default=401)
    ap.add_argument("--seed", type=int, default=20261001)
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    arrays = workflow.filter_data(
        workflow.load_real_data(args.data), max_systems=None, seed=0,
        fixed_rows=np.load(args.em_output / "selected_subset.npz")["row_indices"],
    )
    with np.load(args.posterior, allow_pickle=False) as p:
        probabilities, z_grid = np.asarray(p["probabilities"], float), np.asarray(p["z_grid"], float)
    mlr, fixed_meta = base.build_mlr(args.fixed_output)
    fixed_params, fixed_f, _ = base.posterior_medians(args.fixed_output)
    em_params, em_shape, em_f, _ = base.em_params(args.em_output)
    fixed_shape = tuple(float(fixed_meta["good_shape_constants"][k]) for k in ("B", "uc", "C"))
    u, sig, absg = arrays["u"], arrays["u_sigma"], arrays["absg"]
    r_obs = u / sig
    z_mean = probabilities @ z_grid / probabilities.sum(1)
    report = {"n": int(u.size), "min_r": float(r_obs.min()), "frac_r_3_6": float(np.mean(r_obs < 6))}

    # ---------------- Level 1: forward simulation with the cut ----------------
    models = {
        "fixed T8": (fixed_params, fixed_shape, fixed_f),
        "EM-like": (em_params, em_shape, em_f),
        "PARSEC mass + fixed shape": (None, fixed_shape, fixed_f),
    }
    rng = np.random.default_rng(args.seed)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    level1 = {}
    colors = {"fixed T8": "#3568a8", "EM-like": "#c44e52", "PARSEC mass + fixed shape": "#4c9a4c"}
    for name, (params, shape, f) in models.items():
        us, ss, ms, ps = simulate(mlr, params, shape, f, arrays, probabilities, z_grid, args.reps, rng)
        rs = us / ss
        pass_frac = float(ps.mean())
        up, rp = us[ps], rs[ps]
        level1[name] = {
            "predicted_pass_fraction": pass_frac,
            "KS_r_cut_applied": ks_like(r_obs, rp),
            "KS_r_no_cut": ks_like(r_obs, rs),
            "KS_u_cut_applied": ks_like(u, up),
            "KS_u_no_cut": ks_like(u, us),
            "median_u_obs": float(np.median(u)), "median_u_pred_cut": float(np.median(up)),
            "median_u_pred_nocut": float(np.median(us)),
            "median_r_obs": float(np.median(r_obs)), "median_r_pred_cut": float(np.median(rp)),
        }
        c = colors[name]
        for ax, obs, a, b, lim in (
            (axes[0], r_obs, rp, rs, (3, 60)), (axes[1], u, up, us, (0, 150)),
        ):
            xs = np.linspace(*lim, 600)
            ax.plot(xs, np.searchsorted(np.sort(a), xs) / a.size, color=c, label=f"{name} (cut)")
            ax.plot(xs, np.searchsorted(np.sort(b), xs) / b.size, color=c, ls=":", lw=1, label=f"{name} (no cut)")
        # mass-scale hint: ratio of observed to predicted-with-cut median u
    for ax, obs, lim, lab in ((axes[0], r_obs, (3, 60), r"$u/\sigma_u$"), (axes[1], u, (0, 150), r"$u$")):
        xs = np.linspace(*lim, 600)
        ax.plot(xs, np.searchsorted(np.sort(obs), xs) / obs.size, color="k", lw=2, label="observed")
        ax.set_xlabel(lab); ax.set_ylabel("CDF"); ax.grid(alpha=0.25)
    axes[0].legend(fontsize=7); axes[0].set_xscale("log"); axes[0].set_xlim(3, 60)
    report["level1"] = level1

    # ---------------- Level 2: profile of global mass scale alpha ----------------
    grid = np.linspace(1e-3, base.OUTLIER_MAX, args.n_grid)
    alphas = np.exp(np.linspace(np.log(0.6), np.log(1.8), 25))
    level2 = {}
    for name, (params, shape, f) in models.items():
        m0 = total_mass(mlr, params, absg, z_mean) if params is not None else parsec_mass(mlr, absg, z_mean)
        ll_plain, ll_trunc = [], []
        for a in alphas:
            L, Pi = mixture_terms(u, sig, a * m0, shape, f, grid)
            ll_plain.append(np.sum(np.log(L)))
            ll_trunc.append(np.sum(np.log(L / Pi)))
        ll_plain, ll_trunc = np.array(ll_plain), np.array(ll_trunc)

        def peak(ll):
            i = int(np.argmax(ll)); lo, hi = max(i - 2, 0), min(i + 3, len(ll))
            c = np.polyfit(np.log(alphas[lo:hi]), ll[lo:hi], 2)
            return float(np.exp(-c[1] / (2 * c[0])))

        level2[name] = {
            "alpha_hat_plain": peak(ll_plain), "alpha_hat_truncated": peak(ll_trunc),
            "mean_pass_prob_at_alpha1": float(np.mean(mixture_terms(u, sig, m0, shape, f, grid)[1])),
        }
        axes[2].plot(alphas, ll_plain - ll_plain.max(), color=colors[name], label=f"{name}: plain")
        axes[2].plot(alphas, ll_trunc - ll_trunc.max(), color=colors[name], ls="--", label=f"{name}: truncated")
    axes[2].set_xlabel(r"mass scale $\alpha$"); axes[2].set_ylabel(r"$\Delta\ln L$"); axes[2].set_ylim(-60, 2)
    axes[2].set_xscale("log"); axes[2].grid(alpha=0.25); axes[2].legend(fontsize=6)
    report["level2"] = level2
    fig.tight_layout()
    fig.savefig(args.output_dir / "pu_selection_diagnostic.png", dpi=200)
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
