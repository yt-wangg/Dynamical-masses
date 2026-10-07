"""Posterior-predictive comparison of the observed u (and u/sigma_u) distributions for two fitted MLR runs.

For each run (fixed p(u~) shape stored in its mlr_model.json, MLR posterior draws, f_outlier) simulate replicate
observations of every system: draw [M/H] from the system's metallicity posterior, mass from the MLR, a good or
outlier component (probability f), the latent speed, and Rice noise with the system's sigma_u; keep replicates with
u/sigma_u > cut.  Each system is given equal total weight.  Compares CDFs and plots CDF residuals (data - model),
overall and in quintiles of s_proj / distance / M_G, with a DKW 95% band for the data.
"""
import argparse, csv, json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "src" / "examples"))
import run_hierarchical_metallicity_test as workflow  # noqa: E402
import binary_masses.hierarchical_metallicity as hm  # noqa: E402
from binary_masses.hierarchical_metallicity import MetallicityPosteriorGrid  # noqa: E402
from astropy.table import Table  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--data", type=Path, default=ROOT / "data/jd_msms_single_bic_1kpc_filtered_cmdcut_cutb_jcaps_err_mhcal_good.fits")
ap.add_argument("--new-dir", type=Path, default=ROOT / "results/selcut_shape_sensitivity_20261005/default")
ap.add_argument("--old-dir", type=Path, default=ROOT / "results/t8_selcut_20261001")
ap.add_argument("--posterior", type=Path, default=ROOT / "results/t8_selcut_20261001/latent_metallicity_weights_t8.npz")
ap.add_argument("--output-dir", type=Path, default=ROOT / "results/u_predictive_compare_20261007")
ap.add_argument("--cut", type=float, default=3.0)
ap.add_argument("--n-draws", type=int, default=8)
ap.add_argument("--n-rep", type=int, default=50, help="replicates per system per posterior draw")
ap.add_argument("--n-bins", type=int, default=5)
ap.add_argument("--seed", type=int, default=7)
args = ap.parse_args()
OUT = args.output_dir; OUT.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(args.seed)

posterior = MetallicityPosteriorGrid.load(args.posterior)
arrays = workflow.filter_data(workflow.load_real_data(args.data), max_systems=None, seed=0, fixed_rows=posterior.row_indices)
table = Table.read(args.data); rows = arrays["row_indices"]
u, sig, absg = arrays["u"], arrays["u_sigma"], arrays["absg"]
N = len(u); zprob = posterior.probabilities; z_grid = posterior.z_grid
mass_surface, _ = workflow.build_surfaces()
_argv, sys.argv = sys.argv, [sys.argv[0], "--stage", "mlr"]
T8 = workflow.parse_args(); sys.argv = _argv
mlr = workflow._make_t8_mlr(mass_surface, T8)
bx1 = hm._bspline_basis_numpy(absg[:, 0], mlr.knots_x, mlr.degree_x)
bx2 = hm._bspline_basis_numpy(absg[:, 1], mlr.knots_x, mlr.degree_x)
bz = hm._bspline_basis_numpy(z_grid, mlr.knots_z, mlr.degree_z)
col = lambda n: np.asarray(table[n], float)[rows]
variables = {"s_proj [AU]": col("sep_AU"), "distance [pc]": 1000.0 / col("parallax1"),
             "M_G (primary)": np.min(absg, axis=1)}

def simulate_run(run_dir):
    shape = json.loads((run_dir / "mlr_model.json").read_text())["good_shape_constants"]
    B, uc, C = shape["B"], shape["uc"], shape["C"]
    w = np.linspace(0, 80, 4001)
    cdf_g = np.cumsum(w * np.exp(-B * w**2 - np.exp((w - uc) / C))); cdf_g /= cdf_g[-1]
    v = np.linspace(0, 80, 4001)
    cdf_o = np.cumsum(np.exp(-0.5 * ((v - hm.RICE_OUTLIER_MU) / hm.RICE_OUTLIER_SIGMA) ** 2)); cdf_o /= cdf_o[-1]
    mc = np.load(run_dir / "mlr_mcmc_t8.npz")
    flat = {k.split("__", 1)[1]: mc[k].reshape((-1,) + mc[k].shape[2:]) for k in mc.files if k.startswith("posterior__")}
    idx = np.linspace(0, flat["c0"].shape[0] - 1, args.n_draws).astype(int)
    cum_z = np.cumsum(zprob, axis=1)
    U, S = [], []
    for i in idx:
        p = {k: flat[k][i] for k in ("c0", "a", "b", "r", "log_lambda_x", "log_lambda_z")}
        th = mlr.theta_from_raw(p)
        mtot = 10.0 ** np.einsum("ni,qh,ih->nq", bx1, bz, th) + 10.0 ** np.einsum("ni,qh,ih->nq", bx2, bz, th)
        f = float(flat["f_outlier"][i]); R = args.n_rep
        iz = np.minimum((rng.random((N, R))[:, :, None] > cum_z[:, None, :]).sum(axis=2), z_grid.size - 1)
        m = np.take_along_axis(mtot, iz, axis=1)
        out = rng.random((N, R)) < f; q = rng.random((N, R))
        vt = np.where(out, np.interp(q, cdf_o, v), np.interp(q, cdf_g, w) * np.sqrt(m))
        ur = np.hypot(vt + sig[:, None] * rng.standard_normal((N, R)), sig[:, None] * rng.standard_normal((N, R)))
        U.append(ur); S.append(ur / sig[:, None])
    ur = np.concatenate(U, axis=1); sr = np.concatenate(S, axis=1)
    keep = sr > args.cut
    npass = keep.sum(axis=1)
    wt = np.where(keep, 1.0 / np.maximum(npass, 1)[:, None], 0.0)       # each system carries unit weight
    return dict(u=ur, s=sr, w=wt, keep=keep, shape=shape, f=float(np.median(flat["f_outlier"])),
                pass_frac=float(keep.mean()), n_systems_no_pass=int((npass == 0).sum()))

runs = {"new (weighted shape)": simulate_run(args.new_dir), "old (V1 shape)": simulate_run(args.old_dir)}
for k, r in runs.items():
    print(f"{k}: shape={r['shape']}  f_outlier={r['f']:.4f}  pass fraction={r['pass_frac']:.3f}  systems with no passing replicate={r['n_systems_no_pass']}")

def wcdf(x_eval, x, w):
    o = np.argsort(x); cw = np.cumsum(w[o]); cw /= cw[-1]
    return np.interp(x_eval, x[o], cw, left=0.0, right=1.0)

def ecdf(x_eval, x):
    return np.searchsorted(np.sort(x), x_eval, side="right") / len(x)

def model_cdf(run, key, sel, x_eval):
    m = np.zeros(N, bool); m[sel] = True
    mask = run["keep"] & m[:, None]
    return wcdf(x_eval, run[key][mask], run["w"][mask])

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
colors = {"new (weighted shape)": "tab:red", "old (V1 shape)": "tab:blue"}
quant = [0.5, 0.9, 0.99]
stat_rows = []

def stats(sel, label):
    for key, xd in (("u", u), ("s", u / sig)):
        grid = np.linspace(0, np.quantile(xd[sel], 0.995) * 1.05, 800)
        Fd = ecdf(grid, xd[sel])
        for name, r in runs.items():
            Fm = model_cdf(r, key, sel, grid)
            ks = np.max(np.abs(Fd - Fm)); n = sel.sum()
            mq = [grid[np.searchsorted(Fm, q)] for q in quant]; dq = [np.quantile(xd[sel], q) for q in quant]
            stat_rows.append(dict(group=label, variable="u" if key == "u" else "u/sigma_u", model=name, n=int(n), KS=ks,
                                  KS_95pct_threshold=1.36 / np.sqrt(n), **{f"data_q{int(100*q)}": d for q, d in zip(quant, dq)},
                                  **{f"model_q{int(100*q)}": m for q, m in zip(quant, mq)},
                                  frac_u_gt_60_data=float(np.mean(u[sel] > 60)) if key == "u" else np.nan,
                                  frac_u_gt_60_model=float(np.sum(r["w"][(r["keep"] & np.isin(np.arange(N), np.flatnonzero(sel))[:, None]) & (r["u"] > 60)]) /
                                                          np.sum(r["w"][r["keep"] & np.isin(np.arange(N), np.flatnonzero(sel))[:, None]])) if key == "u" else np.nan))

def panel(ax_cdf, ax_res, key, sel, xlabel, title=None):
    xd = (u if key == "u" else u / sig)
    grid = np.linspace(0, np.quantile(xd[sel], 0.995) * 1.05, 800)
    Fd = ecdf(grid, xd[sel]); n = sel.sum(); eps = np.sqrt(np.log(2 / 0.05) / (2 * n))
    ax_cdf.plot(grid, Fd, "k", lw=2, label="data")
    ax_res.fill_between(grid, -eps, eps, color="0.85", label="DKW 95% (data)")
    for name, r in runs.items():
        Fm = model_cdf(r, key, sel, grid)
        ax_cdf.plot(grid, Fm, color=colors[name], lw=1.3, label=name)
        ax_res.plot(grid, Fd - Fm, color=colors[name], lw=1.5, label=name)
    ax_res.axhline(0, color="k", lw=0.6); ax_res.set_xlabel(xlabel); ax_cdf.set_ylabel("CDF"); ax_res.set_ylabel("CDF residual (data − model)")
    if title: ax_cdf.set_title(title, fontsize=9)

# overall
allsel = np.ones(N, bool); stats(allsel, "all")
fig, ax = plt.subplots(2, 2, figsize=(11, 7.5), gridspec_kw={"height_ratios": [2, 1.2]}, sharex="col")
panel(ax[0, 0], ax[1, 0], "u", allsel, r"$u$ [km s$^{-1}$ AU$^{1/2}$]", "observed u")
panel(ax[0, 1], ax[1, 1], "s", allsel, r"$u/\sigma_u$", r"observed $u/\sigma_u$ (selection $>3$)")
ax[0, 0].legend(fontsize=8); ax[1, 0].legend(fontsize=8)
plt.tight_layout(); fig.savefig(OUT / "u_predictive_overall.png", dpi=130); plt.close(fig)

# by quantile bins of s_proj / distance / M_G
for vname, x in variables.items():
    edges = np.unique(np.quantile(x, np.linspace(0, 1, args.n_bins + 1)))
    nb = len(edges) - 1
    fig, ax = plt.subplots(4, nb, figsize=(3.4 * nb, 11), sharex="row")
    for b in range(nb):
        sel = (x >= edges[b]) & ((x < edges[b + 1]) if b < nb - 1 else (x <= edges[b + 1]))
        lab = f"{vname.split(' [')[0]} {edges[b]:.3g}–{edges[b+1]:.3g}  (n={sel.sum()})"
        stats(sel, f"{vname} bin{b}: {edges[b]:.3g}-{edges[b+1]:.3g}")
        panel(ax[0, b], ax[1, b], "u", sel, r"$u$", lab)
        panel(ax[2, b], ax[3, b], "s", sel, r"$u/\sigma_u$")
        for r_ in (1, 3): ax[r_, b].set_ylim(-0.12, 0.12)
        if b: [ax[r_, b].set_ylabel("") for r_ in range(4)]
    ax[0, 0].legend(fontsize=7); ax[1, 0].legend(fontsize=7)
    fig.suptitle(f"Observed vs predicted CDFs in quantile bins of {vname}", y=0.995)
    plt.tight_layout(); fig.savefig(OUT / f"u_predictive_by_{vname.split(' ')[0].lower().replace('_','')}.png", dpi=110); plt.close(fig)

with open(OUT / "u_predictive_stats.csv", "w", newline="") as fh:
    wr = csv.DictWriter(fh, fieldnames=list(stat_rows[0])); wr.writeheader(); wr.writerows(stat_rows)
print(f"{'group':34s} {'var':9s} {'model':22s} {'n':>6s} {'KS':>6s} {'thr':>6s}  q50 d/m   q90 d/m   q99 d/m   f(u>60) d/m")
for r in stat_rows:
    print(f"{r['group']:34s} {r['variable']:9s} {r['model']:22s} {r['n']:6d} {r['KS']:6.3f} {r['KS_95pct_threshold']:6.3f}  "
          f"{r['data_q50']:5.1f}/{r['model_q50']:5.1f} {r['data_q90']:5.1f}/{r['model_q90']:5.1f} {r['data_q99']:5.1f}/{r['model_q99']:5.1f}"
          + (f"  {r['frac_u_gt_60_data']:.3f}/{r['frac_u_gt_60_model']:.3f}" if r['variable'] == 'u' else ""))
print("wrote", OUT)
