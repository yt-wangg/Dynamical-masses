#!/usr/bin/env bash
#SBATCH --job-name=dyn-t8
#SBATCH --output=t8-%j.out
#SBATCH --error=t8-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00

# Generic T8 launcher.  Cluster-specific --partition/--account and GPU resource
# options belong in the sbatch command or a site-specific wrapper.
set -euo pipefail

STAGE="${STAGE:-all}"
CONDA_ENV="${CONDA_ENV:-dyn}"
DATA="${DATA:-}"
REQUIRE_GPU="${REQUIRE_GPU:-1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
cd "${PROJECT_ROOT}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/results/hierarchical_metallicity_t8}"

if ! command -v conda >/dev/null 2>&1; then
    echo "conda is not on PATH; load the site module before submitting this job." >&2
    exit 2
fi
CONDA_BASE="$(conda info --base)"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
# Conda's activate.d toolchain scripts read unset variables; nounset must be
# suspended around activation (recommended by conda for strict-mode scripts).
set +u
conda activate "${CONDA_ENV}"
set -u

SCRIPT="${SCRIPT:-${PROJECT_ROOT}/src/examples/run_hierarchical_metallicity_test.py}"
if [[ ! -f "${SCRIPT}" ]]; then
    echo "Cannot find runner: ${SCRIPT}" >&2
    exit 2
fi

data_args=()
if [[ -n "${DATA}" ]]; then
    data_args=(--data "${DATA}")
fi

# EXTRA_ARGS is optional whitespace-separated runner arguments, e.g.
# EXTRA_ARGS='--max-systems 2000 --warmup 200'.
extra_args=()
if [[ -n "${EXTRA_ARGS:-}" ]]; then
    read -r -a extra_args <<< "${EXTRA_ARGS}"
fi

mkdir -p "${OUTPUT_DIR}"
echo "T8 stage=${STAGE} env=${CONDA_ENV} output=${OUTPUT_DIR}"
python -c 'import jax; print("JAX devices:", jax.devices())'
if [[ "${REQUIRE_GPU}" == "1" ]]; then
    python -c 'import jax; assert any(d.platform == "gpu" for d in jax.devices()), "No JAX GPU device is visible"'
fi
python -u "${SCRIPT}" \
    --stage "${STAGE}" \
    --output-dir "${OUTPUT_DIR}" \
    ${data_args[@]+"${data_args[@]}"} \
    ${extra_args[@]+"${extra_args[@]}"}
