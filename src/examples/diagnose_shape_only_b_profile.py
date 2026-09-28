#!/usr/bin/env python3
"""Conditional shape refit after relaxing the direct-model B lower bound."""
from pathlib import Path
import argparse, csv, json
import numpy as np
from scipy.optimize import minimize

import run_em_mlr_pilot as pilot
from binary_masses import hierarchical_metallicity as hm
import run_hierarchical_metallicity_test as workflow


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--baseline', type=Path, required=True)
    p.add_argument('--lookup', type=Path, required=True)
    p.add_argument('--map-states', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--endpoint', choices=('default', 'S2'), default='default')
    p.add_argument('--maxiter', type=int, default=30)
    p.add_argument('--direct-chunk', type=int, default=128)
    p.add_argument('--min-b', type=float, default=1e-5)
    args = p.parse_args()
    if not 0 < args.min_b < 0.0007:
        p.error('--min-b must be positive and below the original 0.0007 bound')
    args.output.mkdir(parents=True, exist_ok=True)
    posterior = hm.MetallicityPosteriorGrid.load(args.baseline / pilot.T8_POSTERIOR_NAME)
    arrays = workflow.filter_data(workflow.load_real_data(args.data), max_systems=None,
                                  seed=20260919, fixed_rows=posterior.row_indices)
    positions = np.arange(len(posterior.row_indices), dtype=np.int64)
    subset = pilot.subset_posterior(posterior, positions)
    subset_arrays = {key: np.asarray(value)[positions] for key, value in arrays.items()}
    full_lookup = hm.DynamicsLikelihoodLookup.load(args.lookup)
    mass_surface, _ = workflow.build_surfaces()
    mlr = workflow._make_t8_mlr(mass_surface, argparse.Namespace(
        mlr_knot_x=workflow.T8_MLR_DEFAULT_KNOT_X, mlr_knot_z=workflow.T8_MLR_DEFAULT_KNOT_Z,
        mlr_degree_x=3, mlr_degree_z=3, mlr_tau_d=0.10, mlr_tau_x=0.05, mlr_tau_z=0.05,
        mlr_tau_xz=0.05, mlr_solar_mean=0.0, mlr_solar_sigma=0.01 / np.log(10.0)))
    mlr.set_data(row_indices=subset.row_indices, u=subset_arrays['u'],
                 u_sigma=subset_arrays['u_sigma'], absg=subset_arrays['absg'],
                 metallicity_grid=subset, dynamics_lookup=full_lookup,
                 dynamics_shape_stack=None,
                 raw_u_outlier_log_likelihood=pilot.outlier_log_likelihood(
                     subset_arrays['u'], subset_arrays['u_sigma']))
    ns = argparse.Namespace(shape_evaluation='direct', direct_nodes=12, direct_chunk=args.direct_chunk,
                            shape_prior_sigma_log_b=float(pilot.DEFAULT_SHAPE_PRIOR_SIGMA[0]),
                            shape_prior_sigma_log_uc=float(pilot.DEFAULT_SHAPE_PRIOR_SIGMA[1]),
                            shape_prior_sigma_log_c=float(pilot.DEFAULT_SHAPE_PRIOR_SIGMA[2]))
    objective, logpost, loglike = pilot.build_objective(mlr, ns)
    with np.load(args.map_states, allow_pickle=False) as saved:
        base = np.asarray(saved[f'{args.endpoint}_vector'], dtype=np.float64)
    baseline_logpost = float(logpost(base))
    v0 = base.copy()
    idx = np.arange(pilot.F_INDEX, pilot.FULL_DIM, dtype=np.int64)

    def fun(x):
        v = v0.copy(); v[idx] = x
        val, grad = objective(v)
        return float(val), np.asarray(grad[idx], dtype=np.float64)

    result = minimize(fun, v0[idx], jac=True, method='L-BFGS-B',
                      bounds=[(1e-5, 1-1e-5), (np.log(args.min_b), np.log(0.0033)),
                              (np.log(24), np.log(44)), (np.log(2.5), np.log(13.5))],
                      options={'maxiter': int(args.maxiter), 'ftol': 1e-9,
                               'gtol': 1e-5, 'maxls': 20})
    vf = v0.copy(); vf[idx] = result.x
    final_logpost = float(logpost(vf))
    final_grad = np.asarray(objective(vf)[1], dtype=np.float64)
    row = {'endpoint': args.endpoint, 'start_B': float(np.exp(base[pilot.SHAPE_OFFSET])),
           'min_B': args.min_b, 'final_B': float(np.exp(vf[pilot.SHAPE_OFFSET])),
           'final_uc': float(np.exp(vf[pilot.SHAPE_OFFSET+1])),
           'final_C': float(np.exp(vf[pilot.SHAPE_OFFSET+2])),
           'final_f': float(vf[pilot.F_INDEX]), 'log_posterior': final_logpost,
           'baseline_log_posterior': baseline_logpost,
           'objective_delta': final_logpost - baseline_logpost,
           'final_log_b_gradient': float(final_grad[pilot.SHAPE_OFFSET]),
           'final_log_b_at_lower_bound': bool(np.isclose(vf[pilot.SHAPE_OFFSET], np.log(args.min_b))),
           'success': bool(result.success), 'nit': int(result.nit), 'message': str(result.message)}
    csv_path = args.output / 'shape_only_profile.csv'
    (args.output / 'shape_only_profile.json').write_text(json.dumps([row], indent=2))
    with csv_path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        writer.writeheader(); writer.writerow(row)
    print(row, flush=True)


if __name__ == '__main__':
    main()
