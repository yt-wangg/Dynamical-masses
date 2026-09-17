#!/usr/bin/env bash
#SBATCH --job-name=dyn-t8-outlier
#SBATCH --output=t8-outlier-%j.out
#SBATCH --error=t8-outlier-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00

# Run the three approved T8 outlier-shape sensitivity cases.  Cluster-specific
# partition, account, and GPU options belong in the sbatch command.
set -euo pipefail

STAGE="${STAGE:-both}"              # both, lookup, or mlr
CASE="${CASE:-all}"                 # all, mu30_sigma13, mu40_sigma20, mu30_sigma20
CONDA_ENV="${CONDA_ENV:-dyn}"
REQUIRE_GPU="${REQUIRE_GPU:-1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
cd "${PROJECT_ROOT}"

DATA="${DATA:-${PROJECT_ROOT}/data/jd_msms_single_bic_1kpc_filtered_cmdcut_cutb_jcaps_err_mhcal_good.fits}"
BASELINE_DIR="${BASELINE_DIR:-${PROJECT_ROOT}/results/hierarchical_metallicity_t8_20260913}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/results/t8_outlier_sensitivity_20260916}"
RUNNER="${RUNNER:-${PROJECT_ROOT}/src/examples/run_hierarchical_metallicity_test.py}"
POSTERIOR="${POSTERIOR:-${BASELINE_DIR}/latent_metallicity_weights_t8.npz}"

case_names=(mu30_sigma13 mu40_sigma20 mu30_sigma20)
case_mu=(30 40 30)
case_sigma=(13 20 20)

case_index() {
    local requested="$1" index
    for index in "${!case_names[@]}"; do
        if [[ "${case_names[index]}" == "${requested}" ]]; then
            printf '%s\n' "${index}"
            return 0
        fi
    done
    echo "Unknown CASE=${requested}; choose all or one of: ${case_names[*]}" >&2
    exit 2
}

if [[ "${STAGE}" != both && "${STAGE}" != lookup && "${STAGE}" != mlr ]]; then
    echo "STAGE must be both, lookup, or mlr (got ${STAGE})." >&2
    exit 2
fi
if [[ ! -f "${RUNNER}" ]]; then
    echo "Cannot find runner: ${RUNNER}" >&2
    exit 2
fi
if [[ ! -f "${DATA}" ]]; then
    echo "Cannot find input data: ${DATA}" >&2
    exit 2
fi
if [[ ! -f "${POSTERIOR}" ]]; then
    echo "Cannot find baseline metallicity posterior: ${POSTERIOR}" >&2
    exit 2
fi
if ! command -v conda >/dev/null 2>&1; then
    echo "conda is not on PATH; load the site module before submitting this job." >&2
    exit 2
fi
CONDA_BASE="$(conda info --base)"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

python - "${BASELINE_DIR}" "${OUTPUT_ROOT}" <<'PY'
from pathlib import Path
import sys

baseline = Path(sys.argv[1]).expanduser().resolve()
output = Path(sys.argv[2]).expanduser().resolve()
if output == baseline or baseline in output.parents:
    raise SystemExit(
        f"OUTPUT_ROOT must be separate from BASELINE_DIR; got {output} under {baseline}"
    )
PY

python -c 'import jax; print("JAX devices:", jax.devices())'
if [[ "${REQUIRE_GPU}" == "1" ]]; then
    python -c 'import jax; assert any(d.platform == "gpu" for d in jax.devices()), "No JAX GPU device is visible"'
fi

verify_existing_case() {
    local name="$1" mu="$2" sigma="$3" output_dir="$4" verify_mlr="$5"
    python - "${name}" "${mu}" "${sigma}" "${output_dir}" "${verify_mlr}" <<'PY'
import json
from pathlib import Path
import sys
import numpy as np

name, mu, sigma, directory, verify_mlr = sys.argv[1:]
mu, sigma = float(mu), float(sigma)
directory = Path(directory)
lookup = directory / "dynamics_likelihood_lookup_t8.npz"
with np.load(lookup, allow_pickle=False) as saved:
    meta = json.loads(str(saved["metadata_json"].item()))
checks = {
    "experimental_outlier_sensitivity": meta.get("experimental_outlier_sensitivity") is True,
    "outlier_u0": meta.get("outlier_u0") == mu,
    "outlier_sigma": meta.get("outlier_sigma") == sigma,
    "velocity_sigma_extent": meta.get("velocity_sigma_extent") == 14,
    "velocity_quadrature_nodes": meta.get("velocity_quadrature_nodes") == 128,
    "sqrt_mtot_points": meta.get("sqrt_mtot_points") == 1024,
    "convergence_passed": meta.get("convergence_check", {}).get("passed") is True,
}
if verify_mlr == "1":
    mlr = json.loads((directory / "mlr_model.json").read_text())
    checks["mlr_experimental"] = mlr.get("experimental_outlier_sensitivity") is True
    checks["mlr_mu"] = mlr.get("outlier_shape", {}).get("mu") == mu
    checks["mlr_sigma"] = mlr.get("outlier_shape", {}).get("sigma") == sigma
failed = [key for key, passed in checks.items() if not passed]
if failed:
    raise SystemExit(
        f"Existing outputs for {name} have mismatched settings: {failed}. "
        "Move that case directory aside before rerunning."
    )
PY
}

run_case() {
    local name="$1" mu="$2" sigma="$3"
    local output_dir="${OUTPUT_ROOT}/${name}"
    local lookup="${output_dir}/dynamics_likelihood_lookup_t8.npz"
    local mlr="${output_dir}/mlr_mcmc_t8.npz"
    local common=(
        --data "${DATA}"
        --output-dir "${output_dir}"
        --metallicity-posterior "${POSTERIOR}"
        --outlier-sensitivity
        --outlier-u0 "${mu}"
        --outlier-sigma "${sigma}"
        --lookup-mass-points 1024
        --lookup-velocity-nodes 128
        --lookup-sigma-extent 14
        --warmup 1000 --samples 1000 --chains 4
        --seed 20260819 --target-accept 0.9
    )
    mkdir -p "${output_dir}"
    echo "T8 outlier case=${name} stage=${STAGE} output=${output_dir}"

    if [[ "${STAGE}" == both || "${STAGE}" == lookup ]]; then
        if [[ -f "${lookup}" && -f "${output_dir}/lookup_convergence_t8.json" ]]; then
            verify_existing_case "${name}" "${mu}" "${sigma}" "${output_dir}" 0
            echo "Lookup already present; reusing ${lookup}"
        else
            python -u "${RUNNER}" --stage lookup "${common[@]}"
        fi
    fi

    if [[ "${STAGE}" == both || "${STAGE}" == mlr ]]; then
        if [[ ! -f "${lookup}" ]]; then
            echo "Missing lookup for ${name}: run STAGE=lookup first." >&2
            exit 2
        fi
        if [[ -f "${mlr}" \
            && -f "${output_dir}/mlr_diagnostics_t8.json" \
            && -f "${output_dir}/mlr_monotonicity.json" \
            && -f "${output_dir}/mlr_summary_t8.csv" \
            && -f "${output_dir}/mlr_derived_grid_t8.npz" \
            && -f "${output_dir}/mlr_model.json" ]]; then
            verify_existing_case "${name}" "${mu}" "${sigma}" "${output_dir}" 1
            echo "MLR fit already present; reusing ${mlr}"
        else
            python -u "${RUNNER}" --stage mlr "${common[@]}" \
                --dynamics-lookup "${lookup}"
        fi
    fi
}

if [[ "${CASE}" == all ]]; then
    for index in "${!case_names[@]}"; do
        run_case "${case_names[index]}" "${case_mu[index]}" "${case_sigma[index]}"
    done
else
    index="$(case_index "${CASE}")"
    run_case "${case_names[index]}" "${case_mu[index]}" "${case_sigma[index]}"
fi

echo "T8 outlier sensitivity stages completed: CASE=${CASE}, STAGE=${STAGE}"
