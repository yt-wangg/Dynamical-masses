"""Fitted p(u~) shapes for several angular-resolution cuts theta_min and no cut, under two a-distributions.

Uses the vectorized re-implementation of Validation/V1 (thermal e, random orientation/phase).  Left: Raghavan a
(as in the notebook); middle: log-uniform a reweighted to the observed s_proj of the data; right: median u~ vs theta_min.
"""
import argparse
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.table import Table
from scipy.optimize import curve_fit

ROOT = Path(__file__).resolve().parents[2]
ap = argparse.ArgumentParser()
ap.add_argument("--data", type=Path, default=ROOT / "data/jd_msms_single_bic_1kpc_filtered_cmdcut_cutb_jcaps_err_mhcal_good.fits")
ap.add_argument("--n", type=int, default=4_000_000)
ap.add_argument("--cuts", type=float, nargs="+", default=[0.4, 1.5, 3.0])
ap.add_argument("--out", type=Path, default=ROOT / "results/alpha_stability_20261005/shape_vs_resolution_cut.png")
args = ap.parse_args()
rng = np.random.default_rng(1)
t = Table.read(args.data)
d_data = 1000.0 / np.asarray(t["parallax1"], float)
s_data = np.asarray(t["sep_AU"], float)
V0 = 29.785

def kepler_f(e, M):
    E = M.copy()
    for _ in range(60):
        E = E - (E - e * np.sin(E) - M) / (1 - e * np.cos(E))
    return np.arctan2(np.sqrt(1 - e**2) * np.sin(E), np.cos(E) - e)

def simulate(n, a, e_alpha=1.0):
    e = rng.random(n) ** (1.0 / (e_alpha + 1.0))
    f = kepler_f(e, 2 * np.pi * rng.random(n))
    Om, om = 2 * np.pi * rng.random(n), 2 * np.pi * rng.random(n)
    ci = 2 * rng.random(n) - 1
    p = a * (1 - e**2); r0 = p / (1 + e * np.cos(f)); w = om + f
    rx = r0 * (np.cos(Om) * np.cos(w) - ci * np.sin(Om) * np.sin(w))
    ry = r0 * (np.sin(Om) * np.cos(w) + ci * np.cos(Om) * np.sin(w))
    v0 = V0 * np.sqrt(2.0 / p)
    vx = -v0 * (np.cos(Om) * (np.sin(w) + e * np.sin(om)) + ci * np.sin(Om) * (np.cos(w) + e * np.cos(om)))
    vy = -v0 * (np.sin(Om) * (np.sin(w) + e * np.sin(om)) - ci * np.cos(Om) * (np.cos(w) + e * np.cos(om)))
    s = np.hypot(rx, ry)
    return np.hypot(vx, vy) * np.sqrt(s) / np.sqrt(2.0), s

form = lambda x, A, B, C, u0: A * x * np.exp(-B * x**2 - np.exp((x - u0) / C))
def fit(ut, wt=None):
    h, ed = np.histogram(ut, bins=100, range=(0, 40), density=True, weights=wt)
    c = 0.5 * (ed[1:] + ed[:-1])
    popt, _ = curve_fit(form, c, h, p0=[4.95e-3, 2.24e-3, 3.85, 36.09], maxfev=20000)
    return c, h, popt

def wmedian(x, w=None):
    o = np.argsort(x); cw = np.cumsum(np.ones_like(x) if w is None else w[o]); return x[o][np.searchsorted(cw, cw[-1] / 2)]

n = args.n
d = rng.choice(d_data, n)
a_rag = (10 ** rng.normal(5.03, 2.28, n) / 365.25) ** (2 / 3) * 2.0 ** (1 / 3)
a_lu = 10 ** rng.uniform(1.0, 5.0, n)
ut1, s1 = simulate(n, a_rag); th1 = s1 / d
ut2, s2 = simulate(n, a_lu); th2 = s2 / d
edges = np.linspace(np.log10(s_data.min()), np.log10(s_data.max()), 25)
hd, _ = np.histogram(np.log10(s_data), edges, density=True)

cuts = [None] + list(args.cuts)
labels = ["no cut"] + [f"θ > {c:g}″" for c in args.cuts]
colors = ["k", "tab:blue", "tab:orange", "tab:red", "tab:green"][: len(cuts)]
x = np.linspace(0, 45, 900)
proj = lambda x: form(x, 5.434e-3, 2.544e-3, 3.100, 35.67)
hw = lambda x: form(x, 4.95e-3, 2.24e-3, 3.85, 36.09)

fig, ax = plt.subplots(2, 3, figsize=(16, 9))
meds = {"Raghavan a": [], "s_proj-matched a": []}
table = []
for col, (name, ut, th, s_, matched) in enumerate([("Raghavan a (as in V1)", ut1, th1, s1, False),
                                                   ("log-uniform a reweighted to data s_proj", ut2, th2, s2, True)]):
    a = ax[0, col]
    for cut, lab, c in zip(cuts, labels, colors):
        m = np.ones_like(ut, bool) if cut is None else th > cut
        wt = None
        if matched:
            m &= (s_ >= s_data.min()) & (s_ <= s_data.max())
            hs, _ = np.histogram(np.log10(s_[m]), edges, density=True)
            ib = np.clip(np.digitize(np.log10(s_[m]), edges) - 1, 0, len(hd) - 1)
            wt = hd[ib] / np.maximum(hs[ib], 1e-12)
        cen, h, popt = fit(ut[m], wt)
        med = wmedian(ut[m], wt)
        meds["s_proj-matched a" if matched else "Raghavan a"].append(med)
        table.append((name, lab, m.sum(), *popt[1:], med))
        a.step(cen, h, where="mid", color=c, alpha=0.35, lw=1)
        a.plot(x, form(x, *popt), color=c, lw=2, label=f"{lab}  (median {med:.2f})")
    a.plot(x, proj(x), "m--", lw=1.5, label="project shape (median 15.74)")
    a.plot(x, hw(x), "c:", lw=1.5, label="Hwang+24")
    a.set_title(name); a.set_xlabel(r"$\tilde u$"); a.set_ylabel(r"$p(\tilde u)$"); a.set_xlim(0, 45); a.legend(fontsize=8)
    b = ax[1, col]
    ref = form(x, *fit(ut[np.ones_like(ut, bool) if not matched else (s_ >= s_data.min()) & (s_ <= s_data.max())],
                       None if not matched else None)[2]) if False else proj(x)
    for cut, lab, c in zip(cuts, labels, colors):
        m = np.ones_like(ut, bool) if cut is None else th > cut
        wt = None
        if matched:
            m &= (s_ >= s_data.min()) & (s_ <= s_data.max())
            hs, _ = np.histogram(np.log10(s_[m]), edges, density=True)
            ib = np.clip(np.digitize(np.log10(s_[m]), edges) - 1, 0, len(hd) - 1)
            wt = hd[ib] / np.maximum(hs[ib], 1e-12)
        _, _, popt = fit(ut[m], wt)
        pdf = form(x, *popt); pdf /= np.trapezoid(pdf, x)
        pr = proj(x) / np.trapezoid(proj(x), x)
        b.plot(x, pdf / np.maximum(pr, 1e-12), color=c, lw=2, label=lab)
    b.axhline(1, color="m", ls="--"); b.set_ylim(0.6, 1.6); b.set_xlim(0, 42)
    b.set_xlabel(r"$\tilde u$"); b.set_ylabel("fitted pdf / project-shape pdf"); b.legend(fontsize=8)

a = ax[0, 2]
for k, (lab, mm) in enumerate(meds.items()):
    a.plot(range(len(cuts)), mm, "o-", label=lab)
a.axhline(15.74, color="m", ls="--", label="project shape"); a.set_xticks(range(len(cuts))); a.set_xticklabels(labels)
a.set_ylabel(r"median $\tilde u$"); a.legend(fontsize=8); a.set_title(r"median $\tilde u$ vs resolution cut")
a = ax[1, 2]
for lab, mm in meds.items():
    a.plot(range(len(cuts)), [100 * ((15.74 / v) ** 2 - 1) for v in mm], "o-", label=lab)
a.axhline(0, color="m", ls="--"); a.set_xticks(range(len(cuts))); a.set_xticklabels(labels)
a.set_ylabel("implied mass change vs project shape [%]\n(m ∝ ũ⁻² at fixed observed u)"); a.legend(fontsize=8)
fig.tight_layout(); args.out.parent.mkdir(parents=True, exist_ok=True); fig.savefig(args.out, dpi=130)
print(f"{'a-distribution':42s} {'cut':10s} {'N':>9s} {'B':>10s} {'C':>6s} {'u0':>6s} {'median':>7s}")
for r in table:
    print(f"{r[0]:42s} {r[1]:10s} {r[2]:9d} {r[3]:10.3e} {r[4]:6.2f} {r[5]:6.2f} {r[6]:7.2f}")
