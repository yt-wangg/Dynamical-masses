"""Compare XP and APOGEE ASPCAP metallicity differences for matched binaries.

The two metallicities are evaluated on the identical set of binary pairs.  The
figure shows individual pair measurements at their true delta-M_G values, with
independent per-star measurement errors propagated to each delta-z.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = ROOT / "data/jd_msms_single_bic_1kpc_filtered_cmdcut_cutb_jcaps_err_mhcal_good.fits"
DEFAULT_APOGEE = Path("/Users/ytwang/Downloads/astraAllStarASPCAP-0.6.0.fits")
DEFAULT_OUTPUT = ROOT / "results/xp_apogee_delta_metallicity_vs_delta_mg_20260917"

XP_COLUMNS = (
    "source_id1", "source_id2", "feh_jcaps_1", "feh_jcaps_2",
    "jc_sigma_m_h_1", "jc_sigma_m_h_2", "absg1", "absg2",
    "jc_at_bound_bits_1", "jc_at_bound_bits_2",
)


def _finite_positive(values: np.ndarray) -> np.ndarray:
    return np.isfinite(values) & (values > 0)


def load_apogee(path: Path) -> tuple[dict[int, tuple[float, float]], dict[str, int]]:
    """Read ASPCAP HDU 2 and retain the highest-SNR valid row per Gaia ID."""
    with fits.open(path, memmap=True) as hdul:
        if len(hdul) <= 2 or hdul[2].data is None:
            raise ValueError(f"{path} has no populated ASPCAP HDU 2")
        data = hdul[2].data
        names = set(data.names)
        needed = {"gaia_dr3_source_id", "m_h_atm", "e_m_h_atm", "flag_bad", "snr"}
        missing = sorted(needed - names)
        if missing:
            raise ValueError(f"Missing ASPCAP columns: {missing}")
        gaia = np.asarray(data["gaia_dr3_source_id"], dtype=np.int64)
        mh = np.asarray(data["m_h_atm"], dtype=float)
        err = np.asarray(data["e_m_h_atm"], dtype=float)
        snr = np.asarray(data["snr"], dtype=float)
        flag_bad = np.asarray(data["flag_bad"], dtype=bool)

    valid = (gaia > 0) & np.isfinite(mh) & np.isfinite(err) & (err > 0) & (~flag_bad)
    valid_count = int(np.count_nonzero(valid))
    best: dict[int, tuple[float, float, float]] = {}
    total_by_id: dict[int, int] = {}
    for gid, value, uncertainty, signal in zip(gaia[valid], mh[valid], err[valid], snr[valid]):
        key = int(gid)
        total_by_id[key] = total_by_id.get(key, 0) + 1
        candidate = (float(signal) if np.isfinite(signal) else -np.inf, float(value), float(uncertainty))
        if key not in best or candidate[0] > best[key][0]:
            best[key] = candidate
    selected = {key: (value, uncertainty) for key, (_, value, uncertainty) in best.items()}
    duplicate_ids = sum(count > 1 for count in total_by_id.values())
    return selected, {
        "rows_total": int(len(data)),
        "rows_valid": valid_count,
        "unique_valid_gaia_ids": int(len(selected)),
        "duplicate_valid_gaia_ids": int(duplicate_ids),
        "duplicate_rows_valid": int(sum(count - 1 for count in total_by_id.values())),
    }


def load_pairs(source: Path, apogee: dict[int, tuple[float, float]]) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    with fits.open(source, memmap=True) as hdul:
        table = hdul[1].data
        missing = [name for name in XP_COLUMNS if name not in table.names]
        if missing:
            raise ValueError(f"Missing XP columns: {missing}")
        sid = np.column_stack([np.asarray(table["source_id1"], dtype=np.int64), np.asarray(table["source_id2"], dtype=np.int64)])
        xp = np.column_stack([np.asarray(table["feh_jcaps_1"], dtype=float), np.asarray(table["feh_jcaps_2"], dtype=float)])
        xp_err = np.column_stack([np.asarray(table["jc_sigma_m_h_1"], dtype=float), np.asarray(table["jc_sigma_m_h_2"], dtype=float)])
        mg = np.column_stack([np.asarray(table["absg1"], dtype=float), np.asarray(table["absg2"], dtype=float)])
        unbounded = (np.asarray(table["jc_at_bound_bits_1"]) == 0) & (np.asarray(table["jc_at_bound_bits_2"]) == 0)

    finite = np.all(np.isfinite(np.column_stack([xp, xp_err, mg])), axis=1)
    in_range = np.all((mg >= 3.5) & (mg <= 13.5), axis=1)
    xp_valid = finite & np.all(xp_err > 0, axis=1) & in_range & unbounded
    rows_xp = np.flatnonzero(xp_valid)

    ap = np.full((len(sid), 2), np.nan)
    ap_err = np.full((len(sid), 2), np.nan)
    for i, pair in enumerate(sid):
        for j, gid in enumerate(pair):
            value = apogee.get(int(gid))
            if value is not None:
                ap[i, j], ap_err[i, j] = value
    ap_valid = np.all(np.isfinite(np.column_stack([ap, ap_err])), axis=1) & np.all(ap_err > 0, axis=1)
    keep = xp_valid & ap_valid
    rows = np.flatnonzero(keep)
    return {
        "rows": rows,
        "x": mg[rows, 1] - mg[rows, 0],
        "xp_y": xp[rows, 1] - xp[rows, 0],
        "xp_err": np.hypot(xp_err[rows, 0], xp_err[rows, 1]),
        "apogee_y": ap[rows, 1] - ap[rows, 0],
        "apogee_err": np.hypot(ap_err[rows, 0], ap_err[rows, 1]),
    }, {
        "rows_total": int(len(sid)),
        "rows_xp_selected": int(len(rows_xp)),
        "rows_both_apogee_matched": int(np.count_nonzero(xp_valid & np.all(np.isfinite(ap), axis=1))),
        "rows_final_identical_pairs": int(len(rows)),
    }


def plot(source: Path, apogee_path: Path, output: Path) -> dict:
    apogee, apogee_counts = load_apogee(apogee_path)
    data, selection = load_pairs(source, apogee)

    output.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.4, 3.6), constrained_layout=False)
    fig.subplots_adjust(left=0.12, right=0.985, top=0.965, bottom=0.17)
    series = [
        (data["xp_y"], data["xp_err"], "#d62728", "XP $\\Delta$[Fe/H]", "o"),
        (data["apogee_y"], data["apogee_err"], "#1f77b4", "ASPCAP $\\Delta$[M/H]", "s"),
    ]
    for values, errors, color, label, marker in series:
        ax.scatter(
            data["x"], values, marker=marker, s=43, color=color, alpha=0.82,
            edgecolors="black", linewidths=0.35, label=label, zorder=3,
        )
        ax.errorbar(data["x"], values, yerr=errors, fmt="none", ecolor=color,
                    elinewidth=0.75, capsize=1.8, capthick=0.75, alpha=0.75, zorder=2)
    ax.axhline(0.0, color="0.3", lw=0.9, ls="--", alpha=0.75)
    ax.set_xlabel(r"$\Delta M_G = M_{G,2} - M_{G,1}$ (mag)")
    ax.set_ylabel(r"$\Delta z = z_2 - z_1$ (dex)")
    ax.legend(frameon=True, framealpha=0.9, ncol=2, fontsize=8.2,
              loc="best", handletextpad=0.35, columnspacing=0.8)
    ax.grid(color="0.85", lw=0.6, alpha=0.6)
    x_span = float(np.max(data["x"]) - np.min(data["x"]))
    x_pad = max(0.08, 0.04 * x_span)
    ax.set_xlim(float(np.min(data["x"])) - x_pad, float(np.max(data["x"])) + x_pad)
    fig.savefig(output / "xp_apogee_delta_metallicity_vs_delta_mg.png", dpi=180)
    plt.close(fig)

    result = {
        "source": str(source), "apogee_source": str(apogee_path),
        "selection": "Finite XP values/errors/M_G, positive XP errors, both M_G in [3.5, 13.5], both jc_at_bound_bits == 0, and valid APOGEE ASPCAP values/errors with flag_bad == 0.",
        "crossmatch_key": "Gaia DR3 source_id1/2 to APOGEE gaia_dr3_source_id",
        "xp_metallicity_columns": ["feh_jcaps_1", "feh_jcaps_2"],
        "xp_error_columns": ["jc_sigma_m_h_1", "jc_sigma_m_h_2"],
        "apogee_metallicity_column": "m_h_atm", "apogee_error_column": "e_m_h_atm",
        "x_definition": "absg2 - absg1", "y_definition": "z2 - z1",
        "error_definition": {
            "xp": "sqrt(jc_sigma_m_h_1^2 + jc_sigma_m_h_2^2)",
            "apogee": "sqrt(e_m_h_atm_1^2 + e_m_h_atm_2^2)",
            "assumption": "independent per-star measurement errors",
        },
        "n": int(len(data["x"])), "apogee_counts": apogee_counts,
        "selection_counts": selection,
        "ranges": {
            "delta_mg": [float(np.min(data["x"])), float(np.max(data["x"]))],
            "xp_delta_z": [float(np.min(data["xp_y"])), float(np.max(data["xp_y"]))],
            "apogee_delta_z": [float(np.min(data["apogee_y"])), float(np.max(data["apogee_y"]))],
        },
    }
    (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--apogee", type=Path, default=DEFAULT_APOGEE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = plot(args.source, args.apogee, args.output)
    print(json.dumps({"output": str(args.output), "n": result["n"]}, indent=2))


if __name__ == "__main__":
    main()
