"""Level 3: refit the p(tilde-u) shape (B, uc, C), f_outlier and mass scale alpha with and
without the u/sigma_u>3 truncation normalization, masses fixed to the PARSEC projection.

Tests whether the EM-drifted shape is just the selection function absorbed into p(tilde-u).
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "src" / "examples"))
import diagnose_pu_selection as d  # noqa: E402
import plot_observed_u_em_fixed_predictive as base  # noqa: E402
import run_hierarchical_metallicity_test as workflow  # noqa: E402

FIXED_SHAPE = (2.544e-3, 35.67, 3.1)

ap = argparse.ArgumentParser()
ap.add_argument("--mode", required=True, choices=["plain_a1", "trunc_a1", "plain_afree", "trunc_afree", "trunc_afree_edge"])
ap.add_argument("--stride", type=int, default=2)
ap.add_argument("--n-grid", type=int, default=201)
args = ap.parse_args()

arrays = workflow.filter_data(workflow.load_real_data(base.DEFAULT_DATA), max_systems=None, seed=0,
                              fixed_rows=np.load(base.DEFAULT_EM / "selected_subset.npz")["row_indices"])
with np.load(base.DEFAULT_POSTERIOR, allow_pickle=False) as p:
    prob, zg = np.asarray(p["probabilities"], float), np.asarray(p["z_grid"], float)
mlr, _ = base.build_mlr(base.DEFAULT_FIXED)
sl = slice(None, None, args.stride)
u, sig, absg = arrays["u"][sl], arrays["u_sigma"][sl], arrays["absg"][sl]
zmean = (prob @ zg / prob.sum(1))[sl]
m0 = d.parsec_mass(mlr, absg, zmean)
grid = np.linspace(1e-3, base.OUTLIER_MAX, args.n_grid)
trunc = args.mode.startswith("trunc"); afree = "afree" in args.mode

def unpack(x):
    B, uc, C = np.exp(x[0]), np.exp(x[1]), np.exp(x[2])
    f = 1 / (1 + np.exp(-x[3])); a = np.exp(x[4]) if afree else 1.0
    return (B, uc, C), f, a

def nll(x):
    shape, f, a = unpack(x)
    if not (24 < shape[1] < 44 and 1.5 < shape[2] < 12 and 1e-5 < shape[0] < 2e-2): return 1e12
    L, Pi = d.mixture_terms(u, sig, a * m0, shape, f, grid)
    return -float(np.sum(np.log(L / Pi) if trunc else np.log(L)))

x0 = np.array([np.log(FIXED_SHAPE[0]), np.log(FIXED_SHAPE[1]), np.log(FIXED_SHAPE[2]), -1.4, 0.0])
res = minimize(nll, x0, method="Nelder-Mead", options=dict(maxiter=600, xatol=1e-3, fatol=1e-3, adaptive=True))
shape, f, a = unpack(res.x)
gg = np.linspace(1e-3, 80, 8001)
pdf = gg * np.exp(-shape[0] * gg**2 - np.exp((gg - shape[1]) / shape[2])); c = np.cumsum(pdf); c /= c[-1]
out = dict(mode=args.mode, n=int(u.size), nll=res.fun, shape=shape, f=f, alpha=a,
           q50_q90_q99=[float(np.interp(q, c, gg)) for q in (.5, .9, .99)], nfev=res.nfev, success=bool(res.success))
# reference: nll at the original physical shape with alpha=1 and fitted-f-free? report same objective
out["nll_at_fixed_shape_a1_f0.2"] = (lambda x: nll(x))(x0 if not afree else x0)
Path(ROOT / "results/pu_selection_diagnostic_20261001").mkdir(exist_ok=True, parents=True)
(ROOT / f"results/pu_selection_diagnostic_20261001/refit_{args.mode}.json").write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
