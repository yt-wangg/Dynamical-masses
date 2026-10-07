"""Mass-scale stability and posterior-predictive diagnostics for the selection-truncated MLR.

Step 1 of the p(u~) shape programme (handoff 3.3-3.5).  The shape of p(u~) is held fixed.
True masses cannot depend on projected separation, distance, etc., whereas an incorrect
eccentricity distribution, separation distribution or selection model does.  So for groups
of systems binned in such a variable we fit a free mass-scale factor alpha (m_tot -> alpha*m_tot)
on top of the posterior MLR and look for trends.

Part A: profile log-likelihood in alpha per bin (selection-corrected, metallicity-marginalized,
        averaged over thinned posterior draws), alpha_hat and a profile 1-sigma interval.
Part B: posterior-predictive check: simulate u_obs from the fitted model, apply u/sigma_u>cut,
        compare quantiles of u, u/sigma_u and the tail fraction with the data in each bin.  Every observed system
        carries equal weight in the prediction (each passing replicate of system j is weighted 1/n_pass_j), i.e. the
        prediction is p(u | pass, system j) averaged over the observed systems.  Pooling all passing replicates
        instead would weight systems by their pass probability and overstate the high-u tail (see
        diagnose_u_predictive_compare.py, which also compares two fitted runs).
Also reports the mean selection-pass probability A_j per bin (handoff 3.4).
"""
import argparse, json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "src" / "examples"))
import run_hierarchical_metallicity_test as workflow  # noqa: E402
import binary_masses.hierarchical_metallicity as hm  # noqa: E402
from binary_masses.hierarchical_metallicity import (  # noqa: E402
    DynamicsLikelihoodLookup, MetallicityPosteriorGrid, raw_u_outlier_log_likelihood, lookup_interpolate,
)
from astropy.table import Table  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--data", type=Path, required=True)
ap.add_argument("--run-dir", type=Path, default=ROOT / "results/t8_selcut_20261001",
                help="directory with latent_metallicity_weights_t8.npz, dynamics_likelihood_lookup_t8.npz, mlr_mcmc_t8.npz")
ap.add_argument("--posterior", type=Path, default=None,
                help="latent_metallicity_weights_t8.npz; default: <run-dir>/latent_metallicity_weights_t8.npz "
                     "(runs that reused another metallicity posterior, e.g. selcut_shape_sensitivity_*/default, need this)")
ap.add_argument("--output-dir", type=Path, default=ROOT / "results/alpha_stability_20261005")
ap.add_argument("--cut", type=float, default=3.0)
ap.add_argument("--n-draws", type=int, default=16)
ap.add_argument("--n-bins", type=int, default=5)
ap.add_argument("--alpha-min", type=float, default=0.6)
ap.add_argument("--alpha-max", type=float, default=1.6)
ap.add_argument("--n-alpha", type=int, default=41)
ap.add_argument("--n-rep", type=int, default=200, help="predictive replicates per system (Part B)")
ap.add_argument("--max-systems", type=int, default=None, help="smoke test: use the first N systems")
ap.add_argument("--skip-ppc", action="store_true")
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()
OUT = args.output_dir; OUT.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(args.seed)

# ---------------------------------------------------------------- data and fitted model
posterior = MetallicityPosteriorGrid.load(args.posterior or args.run_dir / "latent_metallicity_weights_t8.npz")
lookup = DynamicsLikelihoodLookup.load(args.run_dir / "dynamics_likelihood_lookup_t8.npz")
arrays = workflow.filter_data(workflow.load_real_data(args.data), max_systems=None, seed=0,
                              fixed_rows=posterior.row_indices)
table = Table.read(args.data)
rows = arrays["row_indices"]
raw_bad = raw_u_outlier_log_likelihood(arrays["u"], arrays["u_sigma"], support_max=80.0, mu=40.0, sigma=13.0,
                                       quadrature_nodes=256)
mass_surface, _ = workflow.build_surfaces()
_argv, sys.argv = sys.argv, [sys.argv[0], "--stage", "mlr"]
T8 = workflow.parse_args()
sys.argv = _argv
mlr = workflow._make_t8_mlr(mass_surface, T8)
mlr.set_data(row_indices=rows, u=arrays["u"], u_sigma=arrays["u_sigma"], absg=arrays["absg"],
             metallicity_grid=posterior, dynamics_lookup=lookup,
             raw_u_outlier_log_likelihood=raw_bad, selection_cut=args.cut)

N = len(rows) if args.max_systems is None else min(args.max_systems, len(rows))
sl = slice(0, N)
u, sig = arrays["u"][sl], arrays["u_sigma"][sl]
absg = arrays["absg"][sl]
zprob = mlr.z_probabilities[sl]
z_grid = mlr.z_grid
logz = np.log(np.maximum(zprob, 1e-30))
sqrt_grid = mlr.sqrt_mtot_grid

# per-system binning variables (columns of the input catalog)
def col(name):
    return np.asarray(table[name], dtype=float)[rows][sl]
d_pc = 1000.0 / col("parallax1")  # parallax in mas; `pairdistance` is an angular separation, not a distance
variables = {
    "sep_AU": col("sep_AU"),
    "sep_arcsec": col("sep_arcsec"),
    "distance_pc": d_pc,
    "M_G_primary": np.min(absg, axis=1),
    "M_G_secondary": np.max(absg, axis=1),
    "FeH_posterior_mean": zprob @ z_grid,
    "u_over_sigma": u / sig,
    "sigma_u": sig,
    "ruwe_max": np.maximum(col("ruwe1"), col("ruwe2")),
}

# ---------------------------------------------------------------- posterior draws
mc = np.load(args.run_dir / "mlr_mcmc_t8.npz")
flat = {k.split("__", 1)[1]: mc[k].reshape((-1,) + mc[k].shape[2:]) for k in mc.files if k.startswith("posterior__")}
n_tot = flat["c0"].shape[0]
draw_idx = np.linspace(0, n_tot - 1, args.n_draws).astype(int)
bx1 = hm._bspline_basis_numpy(absg[:, 0], mlr.knots_x, mlr.degree_x)
bx2 = hm._bspline_basis_numpy(absg[:, 1], mlr.knots_x, mlr.degree_x)
bz = hm._bspline_basis_numpy(z_grid, mlr.knots_z, mlr.degree_z)

def draw_masses(i):
    p = {k: flat[k][i] for k in ("c0", "a", "b", "r", "log_lambda_x", "log_lambda_z")}
    th = mlr.theta_from_raw(p)
    m1 = 10.0 ** np.einsum("ni,qh,ih->nq", bx1, bz, th)
    m2 = 10.0 ** np.einsum("ni,qh,ih->nq", bx2, bz, th)
    return m1 + m2, float(flat["f_outlier"][i])  # (n, q) total mass, f

# ---------------------------------------------------------------- Part A: alpha profile
import jax.numpy as jnp
from scipy.special import logsumexp

log_good_t = jnp.asarray(mlr.log_good_lookup[sl], dtype=jnp.float32)
log_pass_t = jnp.asarray(mlr.log_pass_good[sl], dtype=jnp.float32)
log_bad_raw = mlr.raw_u_outlier_log_likelihood[sl]
log_pass_bad = mlr.log_pass_bad[sl]
alphas = np.exp(np.linspace(np.log(args.alpha_min), np.log(args.alpha_max), args.n_alpha))
ell = np.zeros((N, alphas.size))  # mean over draws of per-system log-likelihood
for k, i in enumerate(draw_idx):
    mtot, f = draw_masses(i)
    for ia, al in enumerate(alphas):
        sq = np.clip(np.sqrt(al * mtot), sqrt_grid[0], sqrt_grid[-1])
        sqj = jnp.asarray(sq, dtype=jnp.float32)
        lg = np.asarray(lookup_interpolate(sqj, sqrt_grid, log_good_t), dtype=np.float64)
        lp = np.asarray(lookup_interpolate(sqj, sqrt_grid, log_pass_t), dtype=np.float64)
        lc = np.logaddexp(np.log1p(-f) + lg, np.log(f) + log_bad_raw[:, None])
        lpass = np.logaddexp(np.log1p(-f) + lp, np.log(f) + log_pass_bad[:, None])
        ell[:, ia] += logsumexp(logz + lc - lpass, axis=1) / len(draw_idx)
    print(f"draw {k + 1}/{len(draw_idx)} done", flush=True)

# mean selection-pass probability A_j at alpha~1 for the median draw
mtot, f = draw_masses(draw_idx[len(draw_idx) // 2])
sqj = jnp.asarray(np.clip(np.sqrt(mtot), sqrt_grid[0], sqrt_grid[-1]), dtype=jnp.float32)
lp = np.asarray(lookup_interpolate(sqj, sqrt_grid, log_pass_t), dtype=np.float64)
lpass = np.logaddexp(np.log1p(-f) + lp, np.log(f) + log_pass_bad[:, None])
A_j = np.exp(logsumexp(logz + lpass, axis=1))

def alpha_fit(sel):
    prof = ell[sel].sum(axis=0)
    la = np.log(alphas)
    j = int(np.argmax(prof))
    lo, hi = max(j - 3, 0), min(j + 4, la.size)
    c = np.polyfit(la[lo:hi], prof[lo:hi], 2)
    if c[0] >= 0:
        return np.nan, np.nan, prof
    x0 = -c[1] / (2 * c[0])
    return float(np.exp(x0)), float(np.sqrt(-1.0 / (2 * c[0]))), prof  # alpha_hat, sigma(ln alpha)

rows_out = []
glob = alpha_fit(np.ones(N, bool))
print(f"global alpha_hat={glob[0]:.4f}  sigma_ln_alpha={glob[1]:.4f}")
for name, x in variables.items():
    edges = np.unique(np.quantile(x, np.linspace(0, 1, args.n_bins + 1)))
    for b in range(len(edges) - 1):
        sel = (x >= edges[b]) & ((x < edges[b + 1]) if b < len(edges) - 2 else (x <= edges[b + 1]))
        if sel.sum() < 30:
            continue
        a, s, _ = alpha_fit(sel)
        rows_out.append(dict(variable=name, bin=b, lo=float(edges[b]), hi=float(edges[b + 1]), n=int(sel.sum()),
                             alpha_hat=a, sigma_ln_alpha=s, mean_A=float(A_j[sel].mean()),
                             median_u_over_sigma=float(np.median((u / sig)[sel]))))
import csv
with open(OUT / "alpha_by_bin.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows_out[0].keys())); w.writeheader(); w.writerows(rows_out)
for r in rows_out:
    print(f"{r['variable']:>18s} [{r['lo']:9.3g},{r['hi']:9.3g}] n={r['n']:5d} alpha={r['alpha_hat']:.3f}"
          f" +-{r['sigma_ln_alpha']:.3f} meanA={r['mean_A']:.3f}")

# trend significance: weighted linear regression of ln(alpha_hat) on bin index
trend = {}
for name in variables:
    rr = [r for r in rows_out if r["variable"] == name and np.isfinite(r["alpha_hat"])]
    if len(rr) >= 3:
        xb = np.arange(len(rr), dtype=float); yb = np.log([r["alpha_hat"] for r in rr])
        wt = 1.0 / np.array([r["sigma_ln_alpha"] for r in rr]) ** 2
        X = np.vstack([np.ones_like(xb), xb - xb.mean()]).T
        cov = np.linalg.inv(X.T @ (X * wt[:, None])); beta = cov @ (X.T @ (wt * yb))
        trend[name] = dict(slope_per_bin=float(beta[1]), sigma=float(np.sqrt(cov[1, 1])),
                           significance=float(beta[1] / np.sqrt(cov[1, 1])),
                           chi2_const=float(np.sum(wt * (yb - np.average(yb, weights=wt)) ** 2)), dof=len(rr) - 1)
json.dump(dict(global_alpha=glob[0], global_sigma_ln_alpha=glob[1], trends=trend,
               n_draws=int(len(draw_idx)), alphas=alphas.tolist()), open(OUT / "alpha_summary.json", "w"), indent=2)
np.savez(OUT / "alpha_profiles.npz", alphas=alphas, ell=ell, **{k: v for k, v in variables.items()})

# ---------------------------------------------------------------- Part B: posterior predictive
if not args.skip_ppc:
    w = np.linspace(0.0, 80.0, 4001)
    shape = w * np.exp(-hm.RICE_GOOD_B * w ** 2 - np.exp((w - hm.RICE_GOOD_UC) / hm.RICE_GOOD_C))
    cdf_good = np.cumsum(shape); cdf_good /= cdf_good[-1]
    vo = np.linspace(0.0, 80.0, 4001)
    out_pdf = np.exp(-0.5 * ((vo - hm.RICE_OUTLIER_MU) / hm.RICE_OUTLIER_SIGMA) ** 2)
    cdf_out = np.cumsum(out_pdf); cdf_out /= cdf_out[-1]
    mtot, f = draw_masses(draw_idx[len(draw_idx) // 2])
    R = args.n_rep
    cum_z = np.cumsum(zprob, axis=1)
    iz = np.minimum((rng.random((N, R))[:, :, None] > cum_z[:, None, :]).sum(axis=2), z_grid.size - 1)
    m_rep = np.take_along_axis(mtot, iz, axis=1)                       # (N, R)
    is_out = rng.random((N, R)) < f
    ut = rng.random((N, R))
    v_true = np.where(is_out, np.interp(ut, cdf_out, vo), np.interp(ut, cdf_good, w) * np.sqrt(m_rep))
    u_rep = np.hypot(v_true + sig[:, None] * rng.standard_normal((N, R)), sig[:, None] * rng.standard_normal((N, R)))
    keep = u_rep / sig[:, None] > args.cut
    npass = keep.sum(axis=1)
    wt_rep = np.where(keep, 1.0 / np.maximum(npass, 1)[:, None], 0.0)    # unit weight per observed system
    s_rep = u_rep / sig[:, None]
    def wq(x, wt, q):
        o = np.argsort(x); cw = np.cumsum(wt[o]); return float(np.interp(np.asarray(q) * cw[-1], cw, x[o]))
    ppc_rows = []
    for name in ("sep_AU", "distance_pc", "M_G_primary", "FeH_posterior_mean", "sigma_u"):
        x = variables[name]
        edges = np.unique(np.quantile(x, np.linspace(0, 1, args.n_bins + 1)))
        for b in range(len(edges) - 1):
            sel = (x >= edges[b]) & ((x < edges[b + 1]) if b < len(edges) - 2 else (x <= edges[b + 1]))
            uo, so = u[sel], (u / sig)[sel]
            m = keep[sel]; up = u_rep[sel][m]; sp = s_rep[sel][m]; wt = wt_rep[sel][m]
            row = dict(variable=name, bin=b, lo=float(edges[b]), hi=float(edges[b + 1]), n=int(sel.sum()),
                       obs_u_med=float(np.median(uo)), pred_u_med=wq(up, wt, 0.5),
                       obs_u_p90=float(np.quantile(uo, 0.9)), pred_u_p90=wq(up, wt, 0.9),
                       obs_frac_u_gt_60=float(np.mean(uo > 60)), pred_frac_u_gt_60=float(np.sum(wt[up > 60]) / np.sum(wt)),
                       obs_s_med=float(np.median(so)), pred_s_med=wq(sp, wt, 0.5),
                       pred_pass_frac=float(keep[sel].mean()))
            ppc_rows.append(row)
    with open(OUT / "ppc_by_bin.csv", "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(ppc_rows[0].keys())); wr.writeheader(); wr.writerows(ppc_rows)
    for r in ppc_rows:
        print(f"PPC {r['variable']:>18s} b{r['bin']} n={r['n']:5d} u_med {r['obs_u_med']:.1f}/{r['pred_u_med']:.1f}"
              f"  p90 {r['obs_u_p90']:.1f}/{r['pred_u_p90']:.1f}  f(u>60) {r['obs_frac_u_gt_60']:.3f}/{r['pred_frac_u_gt_60']:.3f}"
              f"  u/s med {r['obs_s_med']:.2f}/{r['pred_s_med']:.2f}")
    # overall quantile table (equal weight per system)
    allq = np.linspace(0.05, 0.95, 19)
    mk = keep
    np.savez(OUT / "ppc_overall_quantiles.npz", q=allq, obs_u=np.quantile(u, allq),
             pred_u=np.array([wq(u_rep[mk], wt_rep[mk], x) for x in allq]), obs_s=np.quantile(u / sig, allq),
             pred_s=np.array([wq(s_rep[mk], wt_rep[mk], x) for x in allq]))

# ---------------------------------------------------------------- figure
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
names = list(variables)
fig, axs = plt.subplots(3, 3, figsize=(13, 10))
for ax, name in zip(axs.ravel(), names):
    rr = [r for r in rows_out if r["variable"] == name]
    xm = [0.5 * (r["lo"] + r["hi"]) for r in rr]
    ax.errorbar(xm, [r["alpha_hat"] for r in rr], yerr=[r["alpha_hat"] * r["sigma_ln_alpha"] for r in rr], fmt="o-")
    ax.axhline(1, color="gray", ls=":"); ax.axhline(glob[0], color="r", ls="--", lw=0.8)
    ax.set_xlabel(name); ax.set_ylabel("alpha (mass scale)")
    if name in ("sep_AU", "distance_pc", "sep_arcsec", "sigma_u"):
        ax.set_xscale("log")
fig.tight_layout(); fig.savefig(OUT / "alpha_by_bin.png", dpi=130)
print("wrote", OUT)
