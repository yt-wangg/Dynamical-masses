"""Fit the adopted feh_jcaps model: per-star Student-t + b_G + s."""
from pathlib import Path
import json
import time

import numpy as np
import torch
from astropy.table import Table
from numpy.polynomial.legendre import leggauss
from scipy.optimize import minimize


torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/xp_simple_student_t_feh_jcaps_20260910'
SOURCE = ROOT / 'data/jd_msms_single_bic_1kpc_filtered_cmdcut_cutb_jcaps_err_mhcal_good.fits'


class SimpleStudentT:
    """Independent per-star Student-t measurements with one shared floor."""

    def __init__(self, nodes=48):
        table = Table.read(SOURCE)
        required = ['feh_jcaps_1', 'feh_jcaps_2',
                    'jc_sigma_m_h_1', 'jc_sigma_m_h_2',
                    'absg1', 'absg2',
                    'jc_at_bound_bits_1', 'jc_at_bound_bits_2']
        missing = [name for name in required if name not in table.colnames]
        if missing:
            raise ValueError(f'Missing required columns: {missing}')
        self.z = np.column_stack([np.asarray(table[f'feh_jcaps_{j}'], float)
                                  for j in (1, 2)])
        self.e = np.column_stack([np.asarray(table[f'jc_sigma_m_h_{j}'], float)
                                  for j in (1, 2)])
        self.g = np.column_stack([np.asarray(table[f'absg{j}'], float)
                                  for j in (1, 2)])
        valid = (np.all(np.isfinite(np.column_stack([self.g, self.z, self.e])), axis=1)
                 & np.all((self.g >= 3.5) & (self.g <= 13.5) & (self.e > 0), axis=1))
        rows0 = np.flatnonzero(valid)
        train0 = np.random.default_rng(20260906).random(len(rows0)) < .75
        keep = ((np.asarray(table['jc_at_bound_bits_1'][rows0]) == 0)
                & (np.asarray(table['jc_at_bound_bits_2'][rows0]) == 0))
        self.rows = rows0[keep]
        self.train = train0[keep]
        self.test = ~self.train
        self.z, self.e, self.g = [value[self.rows] for value in (self.z, self.e, self.g)]
        self.knots = np.array([3.5, 6., 8.5, 11., 13.5])

        def basis(x):
            return np.column_stack([np.interp(x, self.knots, vector)
                                    for vector in np.eye(5)])

        self.X = (basis(self.g[:, 1])-basis(self.g[:, 0]))[:, [0, 1, 3, 4]]
        self.d = self.z[:, 1]-self.z[:, 0]
        masks = [('train', self.train), ('test', self.test),
                 ('all', np.ones(len(self.rows), bool))]
        self.data = {name:tuple(torch.as_tensor(value[mask])
                                for value in (self.d, self.X, self.e))
                     for name, mask in masks}
        self.set_nodes(nodes)

    def set_nodes(self, n):
        x, w = leggauss(n)
        self.nodes = n
        self.u = torch.as_tensor((x+1)/2)
        self.logw = torch.log(torch.as_tensor(w/2))

    @staticmethod
    def logpdf(x, scale, nu):
        return (torch.lgamma((nu+1)/2)-torch.lgamma(nu/2)
                -.5*torch.log(nu*torch.pi)-torch.log(scale)
                -.5*(nu+1)*torch.log1p((x/scale)**2/nu))

    def conv(self, r, sa, sb, nu):
        """Exact Student-t difference density by split tangent quadrature."""
        radius = abs(r)[:, None]
        sa, sb = sa[:, None], sb[:, None]

        def half(a, b):
            span = torch.atan(radius/(2*a))+np.pi/2
            theta = -np.pi/2+span*self.u[None, :]
            x = a*torch.tan(theta)
            terms = (self.logpdf(x, a, nu)+self.logpdf(radius-x, b, nu)
                     + torch.log(a)-2*torch.log(torch.cos(theta))
                     + torch.log(span)+self.logw[None, :])
            return torch.logsumexp(terms, dim=1)

        return torch.logaddexp(half(sa, sb), half(sb, sa))

    def evaluate(self, p, data, extra_r=None):
        d, X, e = data
        s = torch.exp(p[4])
        nu = torch.exp(p[5])
        r = d-X@p[:4] if extra_r is None else extra_r
        a = torch.sqrt(e**2+s**2)
        return self.conv(r, a[:, 0], a[:, 1], nu), r

    def lp(self, p, split='all'):
        with torch.no_grad():
            return self.evaluate(torch.as_tensor(p), self.data[split])[0].numpy()

    def objective(self, p):
        p = torch.tensor(p, requires_grad=True)
        loss = -self.evaluate(p, self.data['train'])[0].sum()
        loss.backward()
        return float(loss.detach()), p.grad.numpy().copy()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    model = SimpleStudentT(48)
    starts = [np.r_[np.zeros(4), np.log(s), np.log(nu)]
              for s, nu in [(.02, 2.), (.06, 5.), (.12, 20.), (.25, 3.)]]
    bounds = [(-3, 3)]*4+[(-8, 1), (np.log(.5), np.log(300))]
    opts = []
    for j, p0 in enumerate(starts):
        tic = time.time()
        opt = minimize(model.objective, p0, jac=True, method='L-BFGS-B', bounds=bounds,
                       options={'maxiter':700, 'ftol':2e-10, 'gtol':1e-5})
        opts.append(opt)
        print('FIT', j, opt.success, opt.fun, 's/nu', np.exp(opt.x[4]),
              np.exp(opt.x[5]), 'seconds', round(time.time()-tic), flush=True)
    good = [opt for opt in opts if opt.success]
    if not good:
        raise RuntimeError('No successful Student-t fit.')
    p = min(good, key=lambda opt:opt.fun).x
    model.set_nodes(96)
    refine = minimize(model.objective, p, jac=True, method='L-BFGS-B', bounds=bounds,
                      options={'maxiter':400, 'ftol':2e-11, 'gtol':1e-5})
    if not refine.success:
        raise RuntimeError(str(refine.message))
    p = refine.x
    lp96 = model.lp(p)
    model.set_nodes(160)
    lp = model.lp(p)
    max_change = float(np.max(abs(lp-lp96)))
    if max_change > .005:
        raise RuntimeError(f'Quadrature did not converge: {max_change}')
    summary = {
        'model':'Independent per-star Student-t core with b_G and shared extra scale s; exact pair convolution.',
        'source':str(SOURCE),
        'metallicity_columns':['feh_jcaps_1', 'feh_jcaps_2'],
        'error_columns':['jc_sigma_m_h_1', 'jc_sigma_m_h_2'],
        'selection':'Finite inputs, positive errors, 3.5 <= M_G <= 13.5, and both jc_at_bound_bits == 0.',
        'split':'Seed 20260906 assigned before the boundary cut.',
        'n_parameters':6,
        'n_train':int(model.train.sum()),
        'n_test':int(model.test.sum()),
        's':float(np.exp(p[4])),
        'nu':float(np.exp(p[5])),
        'bias_knots':model.knots.tolist(),
        'bias_values':np.insert(p[:4], 2, 0.).tolist(),
        'parameters':p.tolist(),
        'train_loglike':float(lp[model.train].sum()),
        'test_loglike':float(lp[model.test].sum()),
        'initial_start_nll':[float(opt.fun) for opt in opts],
        'max_logpdf_change_96_160':max_change,
    }
    (OUT/'summary.json').write_text(json.dumps(summary, indent=2))
    np.savez_compressed(OUT/'fit_arrays.npz', parameters=p, logpdf=lp,
                        source_row_index=model.rows, is_train=model.train)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
