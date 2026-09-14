#!/usr/bin/env bash
#SBATCH --job-name=dyn-t8-holdout
#SBATCH --output=t8-holdout-%j.out
#SBATCH --error=t8-holdout-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00

# Generic T8 held-out validation launcher. Pass cluster-specific partition,
# account, and GPU options to sbatch rather than hard-coding them here.
set -euo pipefail

STAGE="${STAGE:-all}"
CONDA_ENV="${CONDA_ENV:-dyn}"
DATA="${DATA:-}"
BASELINE_DIR="${BASELINE_DIR:-}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
REQUIRE_GPU="${REQUIRE_GPU:-1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
cd "${PROJECT_ROOT}"

BASELINE_DIR="${BASELINE_DIR:-${PROJECT_ROOT}/results/hierarchical_metallicity_t8_20260913}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/results/hierarchical_metallicity_t8_20260913_holdout}"
if [[ -z "${DATA}" ]]; then
    DATA="${PROJECT_ROOT}/data/jd_msms_single_bic_1kpc_filtered_cmdcut_cutb_jcaps_err_mhcal_good.fits"
fi

if ! command -v conda >/dev/null 2>&1; then
    echo "conda is not on PATH; load the site module before submitting this job." >&2
    exit 2
fi
CONDA_BASE="$(conda info --base)"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

extra_args=()
if [[ -n "${EXTRA_ARGS:-}" ]]; then
    read -r -a extra_args <<< "${EXTRA_ARGS}"
fi

mkdir -p "${OUTPUT_DIR}"
echo "T8 holdout stage=${STAGE} env=${CONDA_ENV} output=${OUTPUT_DIR}"
python -c 'import jax; print("JAX devices:", jax.devices())'
if [[ "${REQUIRE_GPU}" == "1" ]]; then
    python -c 'import jax; assert any(d.platform == "gpu" for d in jax.devices()), "No JAX GPU device is visible"'
fi
python -u src/examples/run_t8_holdout_validation.py \
    --stage "${STAGE}" \
    --data "${DATA}" \
    --baseline-dir "${BASELINE_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    "${extra_args[@]}"
