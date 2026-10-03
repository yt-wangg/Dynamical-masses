"""Plot the intrinsic good shape before and after conditional B-bound relaxation."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
OLD = ROOT / "results/em_solar_only_full_rebuild_20260928_polish_server/summary.json"
NEW = ROOT / "results/em_solar_only_full_rebuild_20260928_shape_only_b_profile/shape_only_profile.json"


def density(u, shape):
    b, uc, c = shape
    raw = u * np.exp(-b * u**2 - np.exp((u - uc) / c))
    return raw / np.trapezoid(raw, u)


def main():
    old = json.loads(OLD.read_text())["results"]["default"]
    new = json.loads(NEW.read_text())[0]
    old_shape = old["final_shape"]
    new_shape = [new["final_B"], new["final_uc"], new["final_C"]]
    u = np.linspace(0, 80, 100001)
    p_old = density(u, old_shape)
    p_new = density(u, new_shape)
    tv = 0.5 * np.trapezoid(np.abs(p_new - p_old), u)
    cdf_diff = np.zeros_like(u)
    cdf_diff[1:] = np.cumsum(0.5 * np.diff(u) * (
        p_new[1:] + p_new[:-1] - p_old[1:] - p_old[:-1]
    ))
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 6),
                                  gridspec_kw={"height_ratios": [2, 1]})
    ax1.plot(u, p_old, label=f"Old bound: B = {old_shape[0]:.4g}", lw=2)
    ax1.plot(u, p_new, label=f"Relaxed bound: B = {new_shape[0]:.4g}", lw=2)
    ax1.set_ylabel(r"Normalized $p(\tilde{u})$")
    ax1.set_title("Conditional shape fit; MLR held fixed")
    ax1.legend()
    ax2.plot(u, cdf_diff, color="black")
    ax2.axhline(0, color="gray", lw=0.8)
    ax2.set_xlabel(r"Scaled velocity $\tilde{u}$")
    ax2.set_ylabel("CDF(new) - CDF(old)")
    ax2.text(0.98, 0.1, f"TV distance = {tv:.4f}", transform=ax2.transAxes,
             ha="right", va="bottom")
    fig.tight_layout()
    output = NEW.parent / "conditional_shape_comparison.png"
    fig.savefig(output, dpi=180)
    plt.close(fig)
    print(json.dumps({"output": str(output), "tv_distance": float(tv),
                      "delta_log_posterior": float(new["log_posterior"] - old["final"]["log_posterior"])}))


if __name__ == "__main__":
    main()
