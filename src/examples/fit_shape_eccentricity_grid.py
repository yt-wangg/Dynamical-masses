"""p(u~) shape as a function of the eccentricity distribution (hyperparameter alpha in p(e) ~ e^alpha).

Same pipeline as Validation/V20 (which produced the library default shape): broad log-uniform (a, d)
population, random phase and orientation, weights w = H_data(log s, log d) / H_sim(log s, log d), and a
fit of p(u~) = A u~ exp(-B u~^2 - exp((u~-uc)/C)) to the weighted histogram of u~ on [0, 40].  Only the
eccentricity distribution changes between cases, and all cases share the same random numbers (orbital phase,
orientation, quantile of e), so differences between cases carry no simulation noise:
  * constant alpha = 0 (uniform), 0.5, 1 (thermal), 1.233, 1.3, 1.5, 2.  alpha = 1.233 is the adopted
    central value: Hwang+2022 alpha(s_proj) evaluated for every system of the sample, one e drawn per
    system, and the power law fitted to the pooled e (MLE); [1.0, 1.5] is the adopted systematic range;
  * Hwang+2022 separation-dependent alpha(s) = 1.25 tanh((log10 s - 1.87)/0.88) + 0.12
    (as in Model/V3; evaluated at the semi-major axis a, since e must be drawn before projection).

For each case prints (B, uc, C), the median u~ and two mass-scale proxies relative to the alpha = 1 case:
the median scaling (m ~ u~^-2 at fixed u) and the maximum-likelihood scale factor that maps the alpha = 1
shape onto the case shape.  Writes results/shape_eccentricity_grid/shapes.json and a figure.
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table
from scipy.optimize import curve_fit

ROOT = Path(__file__).resolve().parents[2]
ap = argparse.ArgumentParser()
ap.add_argument("--data", type=Path, default=ROOT / "data/jd_msms_single_bic_1kpc_filtered_cmdcut_cutb_jcaps_err_mhcal_good.fits")
ap.add_argument("--n", type=int, default=4_000_000)
ap.add_argument("--seed", type=int, default=20261009)
ap.add_argument("--out", type=Path, default=ROOT / "results/shape_eccentricity_grid")
args = ap.parse_args()
args.out.mkdir(parents=True, exist_ok=True)

V0 = 29.785  # km/s: circular speed at 1 AU around 1 Msun
N_S, N_D = 24, 18
ALPHAS = (0.0, 0.5, 1.0, 1.233, 1.3, 1.5, 2.0)
V20_THERMAL = dict(A=4.9901e-3, B=2.2897e-3, C=3.5855, uc=36.2925)  # library default before 2026-10-11

tab = Table.read(args.data)
s_data = np.asarray(tab["sep_AU"], float)
d_data = 1000.0 / np.asarray(tab["parallax1"], float)


def hwang22_alpha(sep):
    return 1.25 * np.tanh((np.log10(sep) - 1.87) / 0.88) + 0.12


def kepler_true_anomaly(e, M):
    E = M.copy()
    for _ in range(60):
        E = E - (E - e * np.sin(E) - M) / (1 - e * np.cos(E))
    return np.arctan2(np.sqrt(1 - e**2) * np.sin(E), np.cos(E) - e)


def simulate(r, a, alpha, mtot=2.0):
    """r: dict of shared uniform draws (keys e, M, Om, om, ci)."""
    e = r["e"] ** (1.0 / (np.asarray(alpha) + 1.0))  # p(e) ~ e^alpha
    e = np.minimum(e, 0.999)
    f = kepler_true_anomaly(e, 2 * np.pi * r["M"])
    Om, om = 2 * np.pi * r["Om"], 2 * np.pi * r["om"]
    ci = 2 * r["ci"] - 1
    p = a * (1 - e**2)
    r0 = p / (1 + e * np.cos(f))
    w = om + f
    rx = r0 * (np.cos(Om) * np.cos(w) - ci * np.sin(Om) * np.sin(w))
    ry = r0 * (np.sin(Om) * np.cos(w) + ci * np.cos(Om) * np.sin(w))
    v0 = V0 * np.sqrt(mtot / p)
    vx = -v0 * (np.cos(Om) * (np.sin(w) + e * np.sin(om)) + ci * np.sin(Om) * (np.cos(w) + e * np.cos(om)))
    vy = -v0 * (np.sin(Om) * (np.sin(w) + e * np.sin(om)) - ci * np.cos(Om) * (np.cos(w) + e * np.cos(om)))
    s = np.hypot(rx, ry)
    return np.hypot(vx, vy) * np.sqrt(s) / np.sqrt(mtot), s


se = np.linspace(np.log10(s_data.min()), np.log10(s_data.max()), N_S + 1)
de = np.linspace(np.log10(d_data.min()), np.log10(d_data.max()), N_D + 1)
H_data, _, _ = np.histogram2d(np.log10(s_data), np.log10(d_data), [se, de], density=True)


def weights(s, d):
    m = (s >= s_data.min()) & (s <= s_data.max()) & (d >= d_data.min()) & (d <= d_data.max())
    H_sim, _, _ = np.histogram2d(np.log10(s[m]), np.log10(d[m]), [se, de], density=True)
    i = np.clip(np.digitize(np.log10(s[m]), se) - 1, 0, N_S - 1)
    j = np.clip(np.digitize(np.log10(d[m]), de) - 1, 0, N_D - 1)
    return m, H_data[i, j] / np.maximum(H_sim[i, j], 1e-12)


form = lambda x, A, B, C, uc: A * x * np.exp(-B * x**2 - np.exp((x - uc) / C))


def fit_shape(ut, w):
    h, ed = np.histogram(ut, bins=100, range=(0, 40), density=True, weights=w)
    c = 0.5 * (ed[1:] + ed[:-1])
    popt, _ = curve_fit(form, c, h, p0=[4.95e-3, 2.24e-3, 3.85, 36.09], maxfev=20000)
    return c, h, popt


xx = np.linspace(1e-3, 90, 90001)


def pdf(p):
    y = form(xx, *p)
    return y / np.trapezoid(y, xx)


def median_of(p):
    c = np.cumsum(pdf(p)); return xx[np.searchsorted(c, c[-1] / 2)]


def ml_mass_ratio(p_true, p_model):
    """Mass ratio m_model/m_true that maximizes E_true[log p_model] for noiseless u~ drawn from p_true."""
    pt, pm = pdf(p_true), pdf(p_model)
    rs = np.linspace(0.7, 1.4, 1401)
    L = [np.trapezoid(pt * np.log(np.maximum(np.interp(xx / np.sqrt(r), xx, pm, right=0) / np.sqrt(r), 1e-300)), xx)
         for r in rs]
    return rs[int(np.argmax(L))]


rng = np.random.default_rng(args.seed)
a = 10 ** rng.uniform(1.0, 5.0, args.n)
d = 10 ** rng.uniform(np.log10(5.0), np.log10(1000.0), args.n)
shared = {k: rng.random(args.n) for k in ("e", "M", "Om", "om", "ci")}
cases = {f"alpha_{al:g}": np.full(args.n, al) for al in ALPHAS}
cases["alpha_hwang22_s"] = hwang22_alpha(a)

res = {}
for name, alpha in cases.items():
    ut, s = simulate(shared, a, alpha)
    m, w = weights(s, d)
    c, h, popt = fit_shape(ut[m], w)
    res[name] = dict(A=popt[0], B=popt[1], C=popt[2], uc=popt[3], median=median_of(popt),
                     weighted_mean_alpha=float(np.average(alpha[m], weights=w)), centers=c, hist=h)

ref = [res["alpha_1"][k] for k in ("A", "B", "C", "uc")]
print(f"{'case':18s} {'<alpha>':>7s} {'B':>10s} {'uc':>7s} {'C':>6s} {'median':>7s} {'dm(median)':>10s} {'dm(ML)':>8s}")
for name, r in res.items():
    p = [r[k] for k in ("A", "B", "C", "uc")]
    r["mass_pct_median"] = 100 * ((res["alpha_1"]["median"] / r["median"]) ** 2 - 1)
    r["mass_pct_ml"] = 100 * (ml_mass_ratio(ref, p) - 1)
    print(f"{name:18s} {r['weighted_mean_alpha']:7.2f} {r['B']:10.3e} {r['uc']:7.3f} {r['C']:6.3f} {r['median']:7.2f}"
          f" {r['mass_pct_median']:+9.1f}% {r['mass_pct_ml']:+7.1f}%")
print(f"{'V20 thermal (old)':18s} {'':7s} {V20_THERMAL['B']:10.3e} {V20_THERMAL['uc']:7.3f} {V20_THERMAL['C']:6.3f}"
      f" {median_of([V20_THERMAL[k] for k in ('A', 'B', 'C', 'uc')]):7.2f}")

(args.out / "shapes.json").write_text(json.dumps(
    {n: {k: float(v) for k, v in r.items() if k not in ("centers", "hist")} for n, r in res.items()}, indent=2))

fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
cm = plt.get_cmap("viridis")
for i, (name, r) in enumerate(res.items()):
    col = "k" if name == "alpha_hwang22_s" else cm(i / (len(ALPHAS) - 1))
    ls = "--" if col == "k" else ("-" if name in ("alpha_1", "alpha_1.233", "alpha_1.5") else ":")
    p = [r[k] for k in ("A", "B", "C", "uc")]
    ax[0].step(r["centers"], r["hist"], where="mid", color=col, alpha=0.25, lw=0.8)
    ax[0].plot(xx[xx < 45], pdf(p)[xx < 45], color=col, lw=1.6,
               label=f"{name} (median {r['median']:.2f})", ls=ls)
    with np.errstate(divide="ignore", invalid="ignore"):
        ax[1].plot(xx[xx < 33], (pdf(p) / pdf(ref))[xx < 33], color=col, lw=1.6, ls=ls)
ax[0].set(xlabel=r"$\tilde u$", ylabel=r"$p(\tilde u)$", xlim=(0, 45))
ax[0].legend(fontsize=7.5, frameon=False)
ax[1].axhline(1, color="0.5", lw=0.8)
ax[1].set(xlabel=r"$\tilde u$", ylabel=r"$p(\tilde u)\,/\,p(\tilde u\,|\,\alpha=1)$", ylim=(0.5, 1.6))
fig.tight_layout()
fig.savefig(args.out / "shapes_vs_alpha.png", dpi=150)
print("wrote", args.out)
