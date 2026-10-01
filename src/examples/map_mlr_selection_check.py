"""MAP check of the selection-truncated MLR likelihood (fixed physical p(tilde-u) shape).

Fits the T8 MLR by L-BFGS MAP with and without the u/sigma_u>3 truncation
normalization and reports the mass correction relative to the PARSEC projection.
Point estimate only; no MCMC.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "src" / "examples"))
import plot_observed_u_em_fixed_predictive as base  # noqa: E402
import run_hierarchical_metallicity_test as workflow  # noqa: E402
from binary_masses.hierarchical_metallicity import (  # noqa: E402
    DynamicsLikelihoodLookup, MetallicityPosteriorGrid, assess_rice_lookup_convergence,
    raw_u_outlier_log_likelihood,
)

OUT = ROOT / ("results/pu_selection_mlr_map_20261001" if "--smoke" not in sys.argv else "/tmp/pu_smoke")
ap = argparse.ArgumentParser()
ap.add_argument("--cut", type=float, default=3.0)
ap.add_argument("--mass-points", type=int, default=1024)
ap.add_argument("--smoke", action="store_true", help="tiny local run, output to /tmp")
ap.add_argument("--maxiter", type=int, default=400)
args = ap.parse_args()
OUT.mkdir(parents=True, exist_ok=True)

arrays = workflow.filter_data(workflow.load_real_data(base.DEFAULT_DATA), max_systems=None, seed=0,
                              fixed_rows=np.load(base.DEFAULT_EM / "selected_subset.npz")["row_indices"])
posterior = MetallicityPosteriorGrid.load(base.DEFAULT_POSTERIOR)
lookup_path = OUT / "lookup.npz"
if lookup_path.exists():
    lookup = DynamicsLikelihoodLookup.load(lookup_path)
else:
    convergence = assess_rice_lookup_convergence(
        row_indices=arrays["row_indices"], u=arrays["u"], u_sigma=arrays["u_sigma"],
        sqrt_mtot_min=0.25, sqrt_mtot_max=2.5, sample_systems=32, scale_points=16,
        velocity_sigma_extent=14.0, velocity_quadrature_nodes=128,
        tolerance_log_likelihood=1e-3, outlier_u0=40.0, outlier_sigma=13.0)
    if not convergence["passed"]:
        raise ValueError("Rice lookup convergence check failed.")
    built = DynamicsLikelihoodLookup.precompute(
        row_indices=arrays["row_indices"], u=arrays["u"], u_sigma=arrays["u_sigma"],
        sqrt_mtot_points=args.mass_points, velocity_quadrature_nodes=128,
        velocity_sigma_extent=14.0, system_chunk=128)
    lookup = DynamicsLikelihoodLookup(
        row_indices=built.row_indices, sqrt_mtot_grid=built.sqrt_mtot_grid,
        log_good=built.log_good, log_bad=built.log_bad,
        metadata={**dict(built.metadata), "convergence_check": convergence})
    lookup.save(lookup_path)
raw_bad = raw_u_outlier_log_likelihood(arrays["u"], arrays["u_sigma"], support_max=80.0, mu=40.0, sigma=13.0,
                                       quadrature_nodes=256)
mass_surface, _ = workflow.build_surfaces()

def fit(cut):
    import jax, jax.numpy as jnp
    from jax.flatten_util import ravel_pytree
    from scipy.optimize import minimize
    from numpyro.infer.util import initialize_model
    from numpyro.infer.initialization import init_to_value
    mlr, _ = base.build_mlr(base.DEFAULT_FIXED)
    mlr.set_data(row_indices=arrays["row_indices"], u=arrays["u"], u_sigma=arrays["u_sigma"], absg=arrays["absg"],
                 metallicity_grid=posterior, dynamics_lookup=lookup, raw_u_outlier_log_likelihood=raw_bad,
                 selection_cut=cut)
    init = {k: jnp.asarray(v) for k, v in mlr.initial_raw_parameters().items()}
    model = mlr._build_numpyro_model()
    margs = (jnp.asarray(mlr.absg), jnp.asarray(mlr.z_probabilities),
             jnp.asarray(mlr.log_good_lookup), jnp.asarray(mlr.log_bad_lookup))
    info = initialize_model(jax.random.PRNGKey(0), model, model_args=margs,
                            init_strategy=init_to_value(values=init), dynamic_args=False)
    z0, unravel = ravel_pytree(info.param_info.z)
    pot = jax.jit(jax.value_and_grad(lambda z: info.potential_fn(unravel(z))))
    def fun(z):
        v, g = pot(jnp.asarray(z))
        return float(v), np.asarray(g, dtype=np.float64)
    res = minimize(fun, np.asarray(z0, dtype=np.float64), jac=True, method="L-BFGS-B",
                   options=dict(maxiter=args.maxiter, maxfun=args.maxiter * 2))
    cons = info.postprocess_fn(unravel(jnp.asarray(res.x)))
    params = {k: np.asarray(v) for k, v in cons.items() if k in ("c0", "a", "b", "r", "log_lambda_x", "log_lambda_z")}
    f = float(cons["f_outlier"])
    xs = np.array([3.5, 4.0, 5.0, 6.0, 7.0, 8.6, 10.0])
    row = {}
    for z in (-0.5, 0.0, 0.3):
        row[f"Z={z}"] = {f"{x}": float(10 ** mlr.correction_log10(np.array([x]), np.array([z]), params)[0]) for x in xs}
    return dict(cut=cut, potential=float(res.fun), success=bool(res.success), nit=int(res.nit), f_outlier=f,
                mass_ratio_to_PARSEC_single_star=row)

out = {name: fit(c) for name, c in (("untruncated", None), ("truncated", args.cut))}
(OUT / "report.json").write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
