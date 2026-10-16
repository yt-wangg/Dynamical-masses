#!/usr/bin/env python3
"""Paper figure: observed distribution of u compared with the posterior predictive distribution of the
adopted fit (alpha_e = 1.233), with the good and outlier components shown separately.

Simulation as in diagnose_u_predictive_compare.py: for each posterior draw of the MLR and f_outlier, every
system receives replicate observations (metallicity from its Stage-1b posterior, mass from the MLR, good or
outlier component, Rice noise with its sigma_u), keeps those with u/sigma_u > 3, and carries unit weight.
Panel (a): histogram of u (data, predictive median and 68% spread over posterior draws).
Panel (b): CDF residual data - model with the DKW 95% band of the data.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_mlr_summary_figure as S  # noqa: E402
ROOT = S.ROOT
sys.path.insert(0, str(ROOT / "src"))
import run_hierarchical_metallicity_test as workflow  # noqa: E402
import binary_masses.hierarchical_metallicity as hm  # noqa: E402
from binary_masses.hierarchical_metallicity import MetallicityPosteriorGrid  # noqa: E402

DATA = ROOT / "data/jd_msms_single_bic_1kpc_filtered_cmdcut_cutb_jcaps_err_mhcal_good.fits"
RUN = S.RUNS / S.CENTRAL
POSTERIOR = ROOT / "results/t8_selcut_20261001/latent_metallicity_weights_t8.npz"
CUT, N_DRAWS, N_REP, SEED = 3.0, 12, 50, 20261016
BINS = np.arange(0, 92, 2.0)

plt.rcParams.update({"font.size": 8, "axes.labelsize": 9, "legend.fontsize": 7.5,
                     "xtick.labelsize": 7.5, "ytick.labelsize": 7.5})


def main() -> None:
    rng = np.random.default_rng(SEED)
    post = MetallicityPosteriorGrid.load(POSTERIOR)
    arrays = workflow.filter_data(workflow.load_real_data(DATA), max_systems=None, seed=0,
                                  fixed_rows=post.row_indices)
    u, sig, absg = arrays["u"], arrays["u_sigma"], arrays["absg"]
    n = len(u)
    zprob, z_grid = post.probabilities, post.z_grid
    mass_surface, _ = workflow.build_surfaces()
    argv, sys.argv = sys.argv, [sys.argv[0], "--stage", "mlr"]
    t8 = workflow.parse_args()
    sys.argv = argv
    mlr = workflow._make_t8_mlr(mass_surface, t8)
    bx1 = hm._bspline_basis_numpy(absg[:, 0], mlr.knots_x, mlr.degree_x)
    bx2 = hm._bspline_basis_numpy(absg[:, 1], mlr.knots_x, mlr.degree_x)
    bz = hm._bspline_basis_numpy(z_grid, mlr.knots_z, mlr.degree_z)

    shape = json.loads((RUN / "mlr_model.json").read_text())["good_shape_constants"]
    B, uc, C = shape["B"], shape["uc"], shape["C"]
    w = np.linspace(0, 80, 4001)
    cdf_g = np.cumsum(w * np.exp(-B * w**2 - np.exp((w - uc) / C))); cdf_g /= cdf_g[-1]
    cdf_o = np.cumsum(np.exp(-0.5 * ((w - hm.RICE_OUTLIER_MU) / hm.RICE_OUTLIER_SIGMA) ** 2)); cdf_o /= cdf_o[-1]
    mc = np.load(RUN / "mlr_mcmc_t8.npz")
    flat = {k.split("__", 1)[1]: mc[k].reshape((-1,) + mc[k].shape[2:])
            for k in mc.files if k.startswith("posterior__")}
    idx = np.linspace(0, flat["c0"].shape[0] - 1, N_DRAWS).astype(int)
    cum_z = np.cumsum(zprob, axis=1)

    centres = 0.5 * (BINS[1:] + BINS[:-1])
    width = np.diff(BINS)
    h_all, h_good, h_out, cdfs, f_sel = [], [], [], [], []
    grid = np.linspace(0, 90, 901)
    for i in idx:
        p = {k: flat[k][i] for k in ("c0", "a", "b", "r", "log_lambda_x", "log_lambda_z")}
        th = mlr.theta_from_raw(p)
        mtot = (10.0 ** np.einsum("ni,qh,ih->nq", bx1, bz, th)
                + 10.0 ** np.einsum("ni,qh,ih->nq", bx2, bz, th))
        f = float(flat["f_outlier"][i])
        iz = np.minimum((rng.random((n, N_REP))[:, :, None] > cum_z[:, None, :]).sum(axis=2), z_grid.size - 1)
        m = np.take_along_axis(mtot, iz, axis=1)
        out = rng.random((n, N_REP)) < f
        q = rng.random((n, N_REP))
        vt = np.where(out, np.interp(q, cdf_o, w), np.interp(q, cdf_g, w) * np.sqrt(m))
        ur = np.hypot(vt + sig[:, None] * rng.standard_normal((n, N_REP)), sig[:, None] * rng.standard_normal((n, N_REP)))
        keep = ur / sig[:, None] > CUT
        npass = keep.sum(axis=1)
        wt = np.where(keep, 1.0 / np.maximum(npass, 1)[:, None], 0.0)
        tot = wt.sum()
        for store, mask in ((h_all, keep), (h_good, keep & ~out), (h_out, keep & out)):
            hh, _ = np.histogram(ur[mask], bins=BINS, weights=wt[mask])
            store.append(hh / tot / width)
        o = np.argsort(ur[keep]); cw = np.cumsum(wt[keep][o]); cw /= cw[-1]
        cdfs.append(np.interp(grid, ur[keep][o], cw, left=0.0, right=1.0))
        f_sel.append(float(wt[keep & out].sum() / tot))
    h_all, h_good, h_out, cdfs = map(np.asarray, (h_all, h_good, h_out, cdfs))

    hd, _ = np.histogram(u, bins=BINS)
    hd_pdf = hd / n / width
    err = np.sqrt(hd) / n / width
    Fd = np.searchsorted(np.sort(u), grid, side="right") / n
    Fm = np.median(cdfs, axis=0)
    ks = float(np.max(np.abs(Fd - Fm)))
    eps = np.sqrt(np.log(2 / 0.05) / (2 * n))

    fig, a = plt.subplots(figsize=(3.5, 2.8))
    lo, hi = np.percentile(h_all, [16, 84], axis=0)
    a.fill_between(centres, lo, hi, step="mid", color=S.C_THIS, alpha=0.3, lw=0)
    a.step(centres, np.median(h_all, axis=0), where="mid", color=S.C_THIS, lw=1.2, label="total (model)")
    a.step(centres, np.median(h_good, axis=0), where="mid", color="0.35", lw=1.0, ls="--", label="good component")
    a.step(centres, np.median(h_out, axis=0), where="mid", color=S.C_XIONG, lw=1.0, ls=":", label="outlier component")
    a.errorbar(centres, hd_pdf, yerr=err, fmt="o", ms=2.6, color="black", elinewidth=0.7, label="data", zorder=5)
    a.set(xlabel=r"$u\ [\mathrm{km\,s^{-1}\sqrt{AU}}]$", ylabel="Probability density", xlim=(0, 90),
          ylim=(0, 1.08 * max(hd_pdf.max(), np.median(h_all, axis=0).max())))
    a.legend(frameon=False, loc="upper right", fontsize=7)
    fig.subplots_adjust(left=0.15, right=0.985, top=0.985, bottom=0.17)
    for ext in ("pdf", "png"):
        fig.savefig(S.OUT / f"u_predictive.{ext}", dpi=250)
        fig.savefig(S.PAPER_FIG / f"u_predictive.{ext}", dpi=250)
    plt.close(fig)
    fo = np.asarray(flat["f_outlier"])
    print(json.dumps({
        "n_systems": int(n),
        "f_outlier_posterior_p16_50_84": [float(x) for x in np.percentile(fo, [16, 50, 84])],
        "predicted_outlier_fraction_among_selected_p16_50_84": [float(x) for x in np.percentile(f_sel, [16, 50, 84])],
        "KS_distance_data_vs_median_model": ks, "DKW95_halfwidth": float(eps),
        "max_abs_residual_in_DKW_units": ks / float(eps),
        "data_frac_u_gt_60": float(np.mean(u > 60)),
        "model_frac_u_gt_60_median": float(np.median([
            (np.sum(hh[centres > 60] * width[centres > 60])) for hh in h_all])),
        "data_u_quantiles_50_90_99": [float(x) for x in np.quantile(u, [0.5, 0.9, 0.99])],
    }, indent=1))


if __name__ == "__main__":
    main()
