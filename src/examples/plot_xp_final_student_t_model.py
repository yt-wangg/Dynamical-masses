"""Clean diagnostic figure for the adopted feh_jcaps Student-t observation model."""
import json

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from fit_xp_simple_student_t import SimpleStudentT, OUT


def averaged_density(model, p, grid, standardized=False):
    ids = np.flatnonzero(model.test)
    values = np.empty((len(ids), len(grid)))
    pair_scale = np.sqrt(np.sum(model.e**2, axis=1)+2*np.exp(p[4])**2)
    with torch.no_grad():
        for start in range(0, len(ids), 128):
            ii = ids[start:start+128]
            n = len(ii)
            flat = np.repeat(ii, len(grid))
            x = np.tile(grid, n)
            jac = 1.
            if standardized:
                scale = np.repeat(pair_scale[ii], len(grid))
                x = x*scale
                jac = scale
            data = tuple(torch.as_tensor(v[flat]) for v in model.data['all'])
            lp = model.evaluate(torch.as_tensor(p), data,
                                extra_r=torch.as_tensor(x))[0].numpy()
            values[start:start+n] = (np.exp(lp)*jac).reshape(n, -1)
    return values.mean(axis=0)


def main():
    model = SimpleStudentT(96)
    saved = np.load(OUT/'fit_arrays.npz')
    p = saved['parameters']
    summary = json.loads((OUT/'summary.json').read_text())
    residual = model.d-model.X@p[:4]
    pair_scale = np.sqrt(np.sum(model.e**2, axis=1)+2*np.exp(p[4])**2)
    q = residual/pair_scale
    test = model.test
    r_grid = np.linspace(-2, 2, 401)
    q_grid = np.linspace(-8, 8, 501)
    r_density = averaged_density(model, p, r_grid)
    q_density = averaged_density(model, p, q_grid, standardized=True)

    plt.rcParams.update({'font.size':10, 'axes.spines.top':False,
                         'axes.spines.right':False})
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.7))
    axes[0].hist(residual[test], bins=np.linspace(-2,2,101), density=True,
                 histtype='step', color='.35', lw=1.4, label='Held-out data')
    axes[0].plot(r_grid, r_density, color='#1776aa', lw=2,
                 label='Student-t pair model')
    axes[0].axvline(0, color='.3', ls=':', lw=.8)
    axes[0].set(xlim=(-2,2), xlabel='Metallicity residual r [dex]', ylabel='Density',
                title='Held-out residual distribution')
    axes[0].legend(fontsize=8)

    axes[1].hist(q[test], bins=np.linspace(-8,8,121), density=True,
                 histtype='step', color='.35', lw=1.4, label='Held-out data')
    axes[1].plot(q_grid, q_density, color='#1776aa', lw=2,
                 label='Student-t pair model')
    axes[1].axvline(0, color='.3', ls=':', lw=.8)
    axes[1].set(xlim=(-6,6), xlabel='q = r / pair scale', ylabel='Density',
                title='Error-normalized residual')
    axes[1].legend(fontsize=8)

    bias = np.asarray(summary['bias_values'])
    axes[2].plot(model.knots, bias, 'o-', color='#1776aa', lw=2)
    axes[2].axhline(0, color='.3', ls=':', lw=.8)
    axes[2].set(xlabel='Absolute G magnitude', ylabel='Relative bias bG [dex]',
                title='Fitted magnitude-dependent bias')
    fig.suptitle(f'Adopted feh_jcaps observation model | s={summary["s"]:.3f} dex per star, '
                 f'nu={summary["nu"]:.2f}', fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT/'03_final_model_diagnostics.png', dpi=190)
    plt.close(fig)
    print(OUT/'03_final_model_diagnostics.png')


if __name__ == '__main__':
    main()
