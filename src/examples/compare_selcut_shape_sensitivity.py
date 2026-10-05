"""Compare selection-truncated MLR fits that differ only in the fixed good-shape constants.

Post-processing only.  Prints mass ratios to the baseline (median, with the 16/84 band of the
per-draw ratio) at selected M_G and [M/H], plus f_outlier and the shape constants of each run.
"""
import argparse, json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ap = argparse.ArgumentParser()
ap.add_argument("--baseline", type=Path, default=ROOT / "results/t8_selcut_20261001")
ap.add_argument("--cases", type=Path, default=ROOT / "results/selcut_shape_sensitivity_20261005")
ap.add_argument("--output-dir", type=Path, default=None)
args = ap.parse_args()
out = args.output_dir or args.cases
MG = [3.5, 5.0, 6.0, 7.0, 8.5, 10.0, 13.0]

def load(d):
    g = np.load(d / "mlr_derived_grid_t8.npz")
    m = json.loads((d / "mlr_model.json").read_text())
    mc = np.load(d / "mlr_mcmc_t8.npz")
    diag = json.loads((d / "mlr_diagnostics_t8.json").read_text())
    return g, m, mc["posterior__f_outlier"].ravel(), diag

gb, mb, fb, db = load(args.baseline)
absg, zg = gb["absg_grid"], gb["z_grid"]
idx = [int(np.argmin(abs(absg - x))) for x in MG]
report = {"baseline": dict(shape=mb["good_shape_constants"], f_outlier=float(np.median(fb)),
                           r_hat=db["max_r_hat"], divergences=db["num_divergences"])}
lines = []
for d in sorted(p for p in args.cases.iterdir() if (p / "mlr_derived_grid_t8.npz").is_file()):
    g, m, f, diag = load(d)
    if not (np.array_equal(g["absg_grid"], absg) and np.array_equal(g["z_grid"], zg)):
        raise SystemExit(f"{d.name}: grids differ from baseline")
    # compare posterior medians (independent chains -> ratio of medians) and the median-of-draw band
    ratio = g["mass"] / np.median(gb["mass"], axis=0)[None]   # (draws, z, absg)
    pct = (np.percentile(ratio, [16, 50, 84], axis=0) - 1.0) * 100.0
    sc = m["good_shape_constants"]
    report[d.name] = dict(shape=sc, f_outlier=float(np.median(f)), r_hat=diag["max_r_hat"],
                          divergences=diag["num_divergences"], min_n_eff=diag["min_n_eff"],
                          mass_diff_percent={f"Z={z:g}": {f"MG={x:g}": [float(pct[k, iz, ix]) for k in range(3)]
                                                          for x, ix in zip(MG, idx)}
                                             for iz, z in enumerate(zg)})
    lines.append(f"\n{d.name}: B={sc['B']:.4g} uc={sc['uc']:.4g} C={sc['C']:.4g}  f_outlier={np.median(f):.4f} "
                 f"(baseline {np.median(fb):.4f})  Rhat={diag['max_r_hat']:.4f}  div={diag['num_divergences']}")
    lines.append("mass change vs baseline [%] (median [16,84] of draw ratio), columns M_G=" + ",".join(f"{x:g}" for x in MG))
    for iz, z in enumerate(zg):
        lines.append(f"  [M/H]={z:+.2f}: " + "  ".join(f"{pct[1, iz, ix]:+5.1f}" for ix in idx))
print("\n".join(lines))
(out / "shape_sensitivity_summary.json").write_text(json.dumps(report, indent=2))
