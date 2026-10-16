"""Size of the 'universal p(u~)' approximation, and the weighting rebuilt on the systems used by the MLR.

Part A  The (s_proj, d) weights of the shape pipeline (fit_shape_eccentricity_grid.py) were built from the whole
        catalogue table (15036 rows).  Rebuild them from exactly the systems that enter the MLR likelihood (the rows of
        the metallicity posterior, 14876) and refit alpha = 1, 1.233, 1.5.
Part B  The likelihood uses one shape p(u~) for every system, standing in for p(u~ | s_j, d_j).  Assuming the
        Hwang+2022 separation-dependent e distribution, p(e) ~ e^alpha(s_proj,j), each system has its own
        p(u~ | alpha_j).  Compare the selection-corrected log-likelihood
            l_j(mu) = log{ [(1-f) G_j + f B_j] / [(1-f) A_good,j + f A_bad,j] },   m_j -> mu m_j,
        with (i) per-system shapes, (ii) their sample average, (iii) constant alpha = 1.233 and alpha = 1, on the masses of the
        fitted MLR, in bins of s_proj.  The best-fit mass-scale mu_hat per bin and for the whole sample shows how much the
        universal shape biases the mass scale, and how it would produce a trend with s_proj.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
from astropy.table import Table
from scipy.optimize import curve_fit
from scipy.special import ive
from scipy.stats import ncx2

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "src" / "examples"))
import run_hierarchical_metallicity_test as workflow  # noqa: E402
import binary_masses.hierarchical_metallicity as hm  # noqa: E402
from binary_masses.hierarchical_metallicity import MetallicityPosteriorGrid  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--data", type=Path, default=ROOT / "data/jd_msms_single_bic_1kpc_filtered_cmdcut_cutb_jcaps_err_mhcal_good.fits")
ap.add_argument("--run-dir", type=Path, default=ROOT / "results/ecc_shape_sensitivity_20261009/alpha_1.233",
                help="fitted MLR run supplying masses and f_outlier")
ap.add_argument("--posterior", type=Path, default=ROOT / "results/t8_selcut_20261001/latent_metallicity_weights_t8.npz")
ap.add_argument("--out", type=Path, default=ROOT / "results/universal_shape_approx_20261011")
ap.add_argument("--n", type=int, default=4_000_000, help="simulated binaries for Part A")
ap.add_argument("--n-shape", type=int, default=6_000_000, help="simulated binaries per alpha for Part B tables")
ap.add_argument("--n-bins", type=int, default=5)
ap.add_argument("--seed", type=int, default=20261011)
args = ap.parse_args()
args.out.mkdir(parents=True, exist_ok=True)

V0 = 29.785
tab = Table.read(args.data)
posterior = MetallicityPosteriorGrid.load(args.posterior)
rows = np.asarray(posterior.row_indices, int)
s_all = np.asarray(tab["sep_AU"], float); d_all = 1000.0 / np.asarray(tab["parallax1"], float)
s_mlr, d_mlr = s_all[rows], d_all[rows]
print(f"catalogue rows {len(tab)}, MLR systems {len(rows)}")

def hwang22_alpha(sep):
    return 1.25 * np.tanh((np.log10(sep) - 1.87) / 0.88) + 0.12

def pooled_alpha_mle(alpha_j):
    # p(e) ~ e^alpha on [0,1], one e per system; E[ln e] = -1/(alpha_j+1) -> MLE alpha = n / sum(1/(alpha_j+1)) - 1
    return len(alpha_j) / np.sum(1.0 / (alpha_j + 1.0)) - 1.0

def kepler_true_anomaly(e, M):
    E = M.copy()
    for _ in range(60):
        E = E - (E - e * np.sin(E) - M) / (1 - e * np.cos(E))
    return np.arctan2(np.sqrt(1 - e**2) * np.sin(E), np.cos(E) - e)

def simulate(r, a, alpha, mtot=2.0):
    e = np.minimum(r["e"] ** (1.0 / (np.asarray(alpha) + 1.0)), 0.999)
    f = kepler_true_anomaly(e, 2 * np.pi * r["M"])
    Om, om = 2 * np.pi * r["Om"], 2 * np.pi * r["om"]; ci = 2 * r["ci"] - 1
    p = a * (1 - e**2); r0 = p / (1 + e * np.cos(f)); w = om + f
    rx = r0 * (np.cos(Om) * np.cos(w) - ci * np.sin(Om) * np.sin(w)); ry = r0 * (np.sin(Om) * np.cos(w) + ci * np.cos(Om) * np.sin(w))
    v0 = V0 * np.sqrt(mtot / p)
    vx = -v0 * (np.cos(Om) * (np.sin(w) + e * np.sin(om)) + ci * np.sin(Om) * (np.cos(w) + e * np.cos(om)))
    vy = -v0 * (np.sin(Om) * (np.sin(w) + e * np.sin(om)) - ci * np.cos(Om) * (np.cos(w) + e * np.cos(om)))
    s = np.hypot(rx, ry)
    return np.hypot(vx, vy) * np.sqrt(s) / np.sqrt(mtot), s

form = lambda x, A, B, C, uc: A * x * np.exp(-B * x**2 - np.exp((x - uc) / C))
xx = np.linspace(1e-3, 90, 90001)
def pdf_of(p):
    y = form(xx, *p); return y / np.trapezoid(y, xx)
def median_of(p):
    c = np.cumsum(pdf_of(p)); return float(xx[np.searchsorted(c, c[-1] / 2)])

# ------------------------------------------------------------------ Part A
print("\n=== Part A: weights from the 15036-row table vs the 14876 MLR systems ===")
rng = np.random.default_rng(args.seed)
n = args.n
a = 10 ** rng.uniform(1.0, 5.0, n); d = 10 ** rng.uniform(np.log10(5.0), np.log10(1000.0), n)
shared = {k: rng.random(n) for k in ("e", "M", "Om", "om", "ci")}
N_S, N_D = 24, 18
se = np.linspace(np.log10(s_all.min()), np.log10(s_all.max()), N_S + 1)
de = np.linspace(np.log10(d_all.min()), np.log10(d_all.max()), N_D + 1)
H_full, _, _ = np.histogram2d(np.log10(s_all), np.log10(d_all), [se, de], density=True)
H_mlr, _, _ = np.histogram2d(np.log10(s_mlr), np.log10(d_mlr), [se, de], density=True)

def weights(s, d, H):
    m = (s >= s_all.min()) & (s <= s_all.max()) & (d >= d_all.min()) & (d <= d_all.max())
    Hs, _, _ = np.histogram2d(np.log10(s[m]), np.log10(d[m]), [se, de], density=True)
    i = np.clip(np.digitize(np.log10(s[m]), se) - 1, 0, N_S - 1); j = np.clip(np.digitize(np.log10(d[m]), de) - 1, 0, N_D - 1)
    return m, H[i, j] / np.maximum(Hs[i, j], 1e-12)

def fit_shape(ut, w):
    h, ed = np.histogram(ut, bins=100, range=(0, 40), density=True, weights=w); c = 0.5 * (ed[1:] + ed[:-1])
    return curve_fit(form, c, h, p0=[4.95e-3, 2.24e-3, 3.85, 36.09], maxfev=20000)[0]

def ml_mass_ratio(p_true, p_model):
    pt, pm = pdf_of(p_true), pdf_of(p_model); rs = np.linspace(0.8, 1.25, 901)
    L = [np.trapezoid(pt * np.log(np.maximum(np.interp(xx / np.sqrt(r), xx, pm, right=0) / np.sqrt(r), 1e-300)), xx) for r in rs]
    return rs[int(np.argmax(L))]

partA = {}
print(f"{'alpha':>6s} {'weights':>8s} {'B':>10s} {'uc':>8s} {'C':>7s} {'median':>7s}   mass change of the MLR-system shape vs the full-table shape")
for al in (1.0, 1.233, 1.5):
    ut, s = simulate(shared, a, np.full(n, al))
    sh = {}
    for name, H in (("table", H_full), ("MLR", H_mlr)):
        m, w = weights(s, d, H); sh[name] = fit_shape(ut[m], w)
    ml = ml_mass_ratio(sh["table"], sh["MLR"]) - 1.0
    for name in ("table", "MLR"):
        p = sh[name]
        print(f"{al:6.3f} {name:>8s} {p[1]:10.4e} {p[3]:8.4f} {p[2]:7.4f} {median_of(p):7.3f}" + (f"   {100*((median_of(sh['table'])/median_of(sh['MLR']))**2-1):+.2f}% (median), {100*ml:+.2f}% (ML)" if name == "MLR" else ""))
    partA[f"alpha_{al:g}"] = {k: dict(A=float(v[0]), B=float(v[1]), C=float(v[2]), uc=float(v[3]), median=median_of(v)) for k, v in sh.items()}
json.dump(partA, open(args.out / "partA_same_systems.json", "w"), indent=2)

# ------------------------------------------------------------------ Part B
print("\n=== Part B: universal vs per-system shape ===")
alpha_j = hwang22_alpha(s_mlr)
print(f"alpha_j (Hwang+22 at the observed s_proj): 5/50/95% = {np.percentile(alpha_j, [5, 50, 95]).round(3)}, pooled MLE alpha_eff = {pooled_alpha_mle(alpha_j):.3f}"
      f"  (all 15036 rows: {pooled_alpha_mle(hwang22_alpha(s_all)):.3f})")

# tables of p(u~ | alpha) on a node grid, common random numbers across alpha
DX, XMAX = 0.25, 50.0
nodes = np.arange(0.0, XMAX + DX / 2, DX); nn = len(nodes)
alphas = np.round(np.arange(-0.1, 1.601, 0.1), 3)
rng2 = np.random.default_rng(args.seed + 1)
ns = args.n_shape
sh2 = {k: rng2.random(ns) for k in ("e", "M", "Om", "om", "ci")}
P = np.zeros((len(alphas), nn))
for ia, al in enumerate(alphas):
    ut, _ = simulate(sh2, np.ones(ns), np.full(ns, al))
    h, ed = np.histogram(ut, bins=np.arange(0.0, XMAX + DX / 2, DX), density=True)     # nn-1 bins
    node = np.zeros(nn); node[1:-1] = 0.5 * (h[:-1] + h[1:]); node[-1] = 0.5 * h[-1]
    P[ia] = node / np.trapezoid(node, nodes)
print("tables built; median u~(alpha):", {float(al): round(float(nodes[np.searchsorted(np.cumsum(P[i]) / np.sum(P[i]), 0.5)]), 2) for i, al in enumerate(alphas) if round(al, 1) in (0.0, 1.0, 1.5)})

def interp_alpha(al):
    t = np.clip((np.asarray(al) - alphas[0]) / (alphas[1] - alphas[0]), 0, len(alphas) - 1 - 1e-9)
    k = np.floor(t).astype(int); return k, t - k
kj, tj = interp_alpha(alpha_j)
P_mix = np.mean((1 - tj)[:, None] * P[kj] + tj[:, None] * P[np.minimum(kj + 1, len(alphas) - 1)], axis=0)
def row_for(al):
    k, t = interp_alpha(np.array([al])); return ((1 - t[0]) * P[k[0]] + t[0] * P[k[0] + 1])
P_1233, P_1 = row_for(1.233), row_for(1.0)

def pdf_eval(U, variant):
    """p(u~=U) for every system (rows of U)."""
    pos = np.clip(U / DX, 0, nn - 1 - 1e-9); i0 = np.floor(pos).astype(int); fr = pos - i0
    if variant == "per-system":
        k = kj[:, None]; t = tj[:, None]; k1 = np.minimum(k + 1, len(alphas) - 1)
        a0 = (1 - fr) * P[k, i0] + fr * P[k, i0 + 1]; a1 = (1 - fr) * P[k1, i0] + fr * P[k1, i0 + 1]
        out = (1 - t) * a0 + t * a1
    else:
        row = {"mixture": P_mix, "alpha=1.233": P_1233, "alpha=1": P_1}[variant]
        out = (1 - fr) * row[i0] + fr * row[i0 + 1]
    return np.where(U <= XMAX, out, 0.0)

# data, masses of the fitted MLR, f_outlier
posterior = MetallicityPosteriorGrid.load(args.posterior)
arrays = workflow.filter_data(workflow.load_real_data(args.data), max_systems=None, seed=0, fixed_rows=posterior.row_indices)
u, sig, absg = arrays["u"], arrays["u_sigma"], arrays["absg"]; N = len(u)
assert np.array_equal(arrays["row_indices"], rows)
mass_surface, _ = workflow.build_surfaces()
_argv, sys.argv = sys.argv, [sys.argv[0], "--stage", "mlr"]; T8 = workflow.parse_args(); sys.argv = _argv
mlr = workflow._make_t8_mlr(mass_surface, T8)
zbar = posterior.probabilities @ posterior.z_grid
bx1 = hm._bspline_basis_numpy(absg[:, 0], mlr.knots_x, mlr.degree_x); bx2 = hm._bspline_basis_numpy(absg[:, 1], mlr.knots_x, mlr.degree_x)
bzj = hm._bspline_basis_numpy(zbar, mlr.knots_z, mlr.degree_z)
mc = np.load(args.run_dir / "mlr_mcmc_t8.npz")
flat = {k.split("__", 1)[1]: mc[k].reshape((-1,) + mc[k].shape[2:]) for k in mc.files if k.startswith("posterior__")}
idx = np.linspace(0, flat["c0"].shape[0] - 1, 64).astype(int)
theta = np.mean([mlr.theta_from_raw({k: flat[k][i] for k in ("c0", "a", "b", "r", "log_lambda_x", "log_lambda_z")}) for i in idx], axis=0)
mtot = 10 ** np.einsum("ni,nh,ih->n", bx1, bzj, theta) + 10 ** np.einsum("ni,nh,ih->n", bx2, bzj, theta)
f = float(np.median(flat["f_outlier"]))
print(f"MLR run {args.run_dir.name}: f_outlier = {f:.4f};  total mass 5/50/95% = {np.percentile(mtot, [5, 50, 95]).round(3)}")

log_bad = hm.raw_u_outlier_log_likelihood(u, sig, support_max=80.0, mu=40.0, sigma=13.0, quadrature_nodes=256)
log_pass_bad = hm.selection_pass_log_tables(sig, np.array([1.0]), cut=3.0)[1]
xg = np.arange(0, 80.0, 0.01); Q1 = ncx2.sf(9.0, 2, xg**2)
def Q1_of(x): return np.interp(x, xg, Q1, right=1.0)
tgrid = np.linspace(-14, 14, 129)

def ell(variant, mu):
    m = mu * mtot; sq = np.sqrt(m)
    v = u[:, None] + sig[:, None] * tgrid[None, :]; ok = v >= 0; vv = np.where(ok, v, 0.0)
    rice = (u[:, None] / sig[:, None] ** 2) * np.exp(-0.5 * ((u[:, None] - vv) / sig[:, None]) ** 2) * ive(0, u[:, None] * vv / sig[:, None] ** 2)
    integrand = np.where(ok, pdf_eval(vv / sq[:, None], variant) / sq[:, None] * rice, 0.0)
    G = np.trapezoid(integrand, vv, axis=1)
    ug = nodes[None, :]
    Ag = np.trapezoid(pdf_eval(np.broadcast_to(ug, (N, nn)), variant) * Q1_of(ug * sq[:, None] / sig[:, None]), nodes, axis=1)
    L = (1 - f) * G + f * np.exp(log_bad); A = (1 - f) * Ag + f * np.exp(log_pass_bad)
    return np.log(np.maximum(L, 1e-300)) - np.log(np.maximum(A, 1e-300)), Ag

variants = ["per-system", "mixture", "alpha=1.233", "alpha=1"]
lmus = np.linspace(-0.2, 0.2, 21); mus = np.exp(lmus)
E = {v: np.zeros((N, len(mus))) for v in variants}; Ag1 = {}
for v_ in variants:
    for i, mu in enumerate(mus):
        E[v_][:, i], ag = ell(v_, mu)
        if abs(lmus[i]) < 1e-9: Ag1[v_] = ag
    print(f"  {v_} done", flush=True)

def peak(l):
    j = int(np.argmax(l)); lo, hi = max(j - 3, 0), min(j + 4, len(lmus))
    c = np.polyfit(lmus[lo:hi], l[lo:hi], 2)
    return (np.exp(-c[1] / (2 * c[0])), np.sqrt(-1 / (2 * c[0]))) if c[0] < 0 else (np.nan, np.nan)

edges = np.unique(np.quantile(s_mlr, np.linspace(0, 1, args.n_bins + 1)))
groups = [("all", np.ones(N, bool))] + [(f"s_proj {edges[b]:.0f}-{edges[b+1]:.0f} AU", (s_mlr >= edges[b]) & ((s_mlr < edges[b + 1]) if b < len(edges) - 2 else (s_mlr <= edges[b + 1]))) for b in range(len(edges) - 1)]
rowsB = []
print(f"\nbest-fit mass scale mu_hat on the MLR masses (1 = no change) and the A_good(mu=1) of each variant")
print(f"{'group':28s} {'n':>6s} " + " ".join(f"{v:>13s}" for v in variants) + "   mean<alpha_j> | mean ln(A_sys/A_1.233)")
for name, sel in groups:
    mu_hat = {v: peak(E[v][sel].sum(axis=0)) for v in variants}
    dlnA = np.mean(np.log(Ag1["per-system"][sel] / Ag1["alpha=1.233"][sel]))
    rowsB.append(dict(group=name, n=int(sel.sum()), mean_alpha_j=float(alpha_j[sel].mean()), mean_dlnA_sys_vs_1233=float(dlnA),
                      **{f"mu_hat_{v}": float(mu_hat[v][0]) for v in variants}, **{f"sigma_ln_mu_{v}": float(mu_hat[v][1]) for v in variants}))
    print(f"{name:28s} {sel.sum():6d} " + " ".join(f"{mu_hat[v][0]:13.4f}" for v in variants) + f"   {alpha_j[sel].mean():.3f} | {dlnA:+.4f}")
json.dump(dict(partB=rowsB, alpha_eff_mlr=float(pooled_alpha_mle(alpha_j)), f_outlier=f, lmus=lmus.tolist()), open(args.out / "partB_universal_vs_per_system.json", "w"), indent=2)

print("\nrelative to per-system (the 'truth' under the Hwang+22 alpha(s) model):  mu_hat(variant)/mu_hat(per-system) - 1  [%]")
for r in rowsB:
    print(f"{r['group']:28s} " + " ".join(f"{v}: {100*(r['mu_hat_'+v]/r['mu_hat_per-system']-1):+6.2f}%" for v in variants[1:]))

import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 3, figsize=(17, 4.6))
for i, al in enumerate(alphas[::3]): ax[0].plot(nodes, P[::3][i], label=f"α={al:g}")
ax[0].plot(nodes, P_mix, "k--", lw=2, label="sample mixture"); ax[0].set(xlabel=r"$\tilde u$", ylabel=r"$p(\tilde u\,|\,\alpha)$", xlim=(0, 45)); ax[0].legend(fontsize=7)
xs = np.arange(len(rowsB) - 1)
for v_, c in zip(variants[1:], ("tab:blue", "tab:red", "tab:green")):
    ax[1].errorbar(xs + 0.12 * (variants.index(v_) - 2), [100 * (r["mu_hat_" + v_] / r["mu_hat_per-system"] - 1) for r in rowsB[1:]], fmt="o-", color=c, label=v_ + " vs per-system")
ax[1].axhline(0, color="k", lw=0.6); ax[1].set_xticks(xs); ax[1].set_xticklabels([r["group"].replace("s_proj ", "") for r in rowsB[1:]], fontsize=7, rotation=15)
ax[1].set_ylabel("mass-scale bias of the universal shape [%]"); ax[1].legend(fontsize=7)
ax[2].hist(np.log(Ag1["per-system"] / Ag1["alpha=1.233"]), bins=80); ax[2].set(xlabel=r"$\ln(A_{\rm good,sys}/A_{\rm good,\alpha=1.233})$", ylabel="systems")
fig.tight_layout(); fig.savefig(args.out / "universal_shape_approx.png", dpi=130); print("wrote", args.out)
