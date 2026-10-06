"""How the fitted p(u~) shape depends on the Gaia angular-resolution cut and on the simulated a-distribution.

Re-implements (vectorized) the simulation of Validation/V1. Phase sampling.ipynb: thermal e (p(e) ~ e),
random orientation and phase, theta = s_proj/d > theta_min, then fits
p(u~) = A u~ exp(-B u~^2 - exp((u~-u0)/C)) to the histogram of u~ on [0, 40] (unweighted curve_fit from the Hwang
starting point, as in the notebook).  u~ does not depend on a or m, so only the SELECTION changes with the
a- and distance distributions.

Cases: no cut / Raghavan a with theta>1.5 / theta>2.1 / a log-uniform reweighted to the observed s_proj of the
data sample with theta>1.5 and 2.1.  Prints fitted (B, u0, C), median u~ and the implied mass-scale change
(m ~ u~^-2 at fixed observed u) relative to the project shape.
"""
import argparse
from pathlib import Path
import numpy as np
from astropy.table import Table
from scipy.optimize import curve_fit

ROOT = Path(__file__).resolve().parents[2]
ap = argparse.ArgumentParser()
ap.add_argument("--data", type=Path, default=ROOT / "data/jd_msms_single_bic_1kpc_filtered_cmdcut_cutb_jcaps_err_mhcal_good.fits")
ap.add_argument("--n", type=int, default=3_000_000)
ap.add_argument("--seed", type=int, default=1)
args = ap.parse_args()
rng = np.random.default_rng(args.seed)
t = Table.read(args.data)
d_data = 1000.0 / np.asarray(t["parallax1"], float)
s_data = np.asarray(t["sep_AU"], float)

V0 = 29.785  # km/s at 1 AU for 1 Msun: u~ = V0 * sqrt(r_proj/p) * |v_proj|/v0 (a- and m-independent)

def kepler_f(e, M):
    E = M.copy()
    for _ in range(60):
        E = E - (E - e * np.sin(E) - M) / (1 - e * np.cos(E))
    return np.arctan2(np.sqrt(1 - e**2) * np.sin(E), np.cos(E) - e)

def simulate(n, a, d, e_alpha=1.0):
    e = rng.random(n) ** (1.0 / (e_alpha + 1.0))
    f = kepler_f(e, 2 * np.pi * rng.random(n))
    Om, om = 2 * np.pi * rng.random(n), 2 * np.pi * rng.random(n)
    ci = 2 * rng.random(n) - 1; si = np.sqrt(1 - ci**2)
    r0 = a * (1 - e**2) / (1 + e * np.cos(f))            # AU
    w = om + f
    rx = r0 * (np.cos(Om) * np.cos(w) - ci * np.sin(Om) * np.sin(w))
    ry = r0 * (np.sin(Om) * np.cos(w) + ci * np.cos(Om) * np.sin(w))
    p = a * (1 - e**2)
    v0 = V0 * np.sqrt(2.0 / p)                            # m=2 Msun
    vx = -v0 * (np.cos(Om) * (np.sin(w) + e * np.sin(om)) + ci * np.sin(Om) * (np.cos(w) + e * np.cos(om)))
    vy = -v0 * (np.sin(Om) * (np.sin(w) + e * np.sin(om)) - ci * np.cos(Om) * (np.cos(w) + e * np.cos(om)))
    s = np.hypot(rx, ry); v = np.hypot(vx, vy)
    return v * np.sqrt(s) / np.sqrt(2.0), s, e

def theta_arcsec(s_au, d_pc):
    return s_au / d_pc  # AU/pc = arcsec

def fit_shape(ut, wt=None, rng_=(0, 40), bins=100):
    h, edges = np.histogram(ut, bins=bins, range=rng_, density=True, weights=wt)
    c = 0.5 * (edges[1:] + edges[:-1])
    f = lambda x, A, B, C, u0: A * x * np.exp(-B * x**2 - np.exp((x - u0) / C))
    popt, _ = curve_fit(f, c, h, p0=[4.95e-3, 2.24e-3, 3.85, 36.09], maxfev=20000)
    return popt

def wmedian(x, w=None):
    o = np.argsort(x); cw = np.cumsum(np.ones_like(x) if w is None else w[o]); return x[o][np.searchsorted(cw, cw[-1] / 2)]

n = args.n
d = rng.choice(d_data, n)
# a-distribution A: Raghavan+2010 log-normal period (days), m_tot=2
P = 10 ** rng.normal(5.03, 2.28, n)
a_rag = (P / 365.25) ** (2 / 3) * 2.0 ** (1 / 3)
# a-distribution B: log-uniform, reweighted to the data's observed s_proj
a_lu = 10 ** rng.uniform(1.0, 5.0, n)

BASE = dict(B=2.544e-3, C=3.100, u0=35.67)
res = {}
def report(name, ut, wt=None):
    A, B, C, u0 = fit_shape(ut, wt)
    med = wmedian(ut, wt)
    res[name] = med
    print(f"{name:42s} N={len(ut):8d}  B={B:.3e} u0={u0:6.2f} C={C:5.2f}  median u~={med:5.2f}")

ut, s, e = simulate(n, a_rag, d)
th = theta_arcsec(s, d)
report("Raghavan a, no cut", ut)
for tm in (1.5, 2.1):
    m = th > tm
    report(f"Raghavan a, theta>{tm}", ut[m])
    print(f"      post-cut s_proj median {np.median(s[m]):8.0f} AU (data {np.median(s_data):.0f}); theta median {np.median(th[m]):.2f}")

ut2, s2, e2 = simulate(n, a_lu, d)
th2 = theta_arcsec(s2, d)
edges = np.linspace(np.log10(s_data.min()), np.log10(s_data.max()), 25)
hd, _ = np.histogram(np.log10(s_data), edges, density=True)
for tm in (0.0, 1.5, 2.1):
    m = (th2 > tm) & (s2 >= s_data.min()) & (s2 <= s_data.max())
    hs, _ = np.histogram(np.log10(s2[m]), edges, density=True)
    ib = np.clip(np.digitize(np.log10(s2[m]), edges) - 1, 0, len(hd) - 1)
    wt = hd[ib] / np.maximum(hs[ib], 1e-12)
    report(f"log-uniform a, s_proj-matched, theta>{tm}", ut2[m], wt)

# reference: median u~ of the project shape
x = np.linspace(0, 80, 8001)
pdf = x * np.exp(-BASE["B"] * x**2 - np.exp((x - BASE["u0"]) / BASE["C"])); cdf = np.cumsum(pdf); cdf /= cdf[-1]
mref = x[np.searchsorted(cdf, 0.5)]
print(f"\nproject shape (B=2.544e-3,u0=35.67,C=3.10): median u~ = {mref:.2f}")
print("implied mass change at fixed observed u, m ~ u~^-2, relative to the project shape:")
for k, v in res.items():
    print(f"  {k:42s} {100 * ((mref / v) ** 2 - 1):+6.1f} %")
