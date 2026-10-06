#!/usr/bin/env bash
#SBATCH --job-name=dyn-selcut-shape
#SBATCH --output=selcut-shape-%j.out
#SBATCH --error=selcut-shape-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00

# Good-shape sensitivity of the selection-truncated (u/sigma_u>3) MLR.
# Each case rebuilds the dynamics lookup with different fixed (B, uc, C), reuses the
# baseline T8 metallicity posterior, and refits the MLR with the truncation normalization.
# Cluster-specific partition/account/GPU options belong in the sbatch command.
set -euo pipefail

STAGE="${STAGE:-both}"            # both, lookup, or mlr
CASE="${CASE:-all}"               # all, default, hwang, refit_trunc, refit_untrunc
CONDA_ENV="${CONDA_ENV:-dyn}"
REQUIRE_GPU="${REQUIRE_GPU:-1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
cd "${PROJECT_ROOT}"

DATA="${DATA:-${PROJECT_ROOT}/data/jd_msms_single_bic_1kpc_filtered_cmdcut_cutb_jcaps_err_mhcal_good.fits}"
BASELINE_DIR="${BASELINE_DIR:-${PROJECT_ROOT}/results/t8_selcut_20261001}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/results/selcut_shape_sensitivity_20261005}"
RUNNER="${RUNNER:-${PROJECT_ROOT}/src/examples/run_hierarchical_metallicity_test.py}"
POSTERIOR="${POSTERIOR:-${BASELINE_DIR}/latent_metallicity_weights_t8.npz}"

# name  B  uc  C.  "default" runs the library default shape (no override flags): B=2.2897e-3 uc=36.2925
# C=3.5855, the (s_proj,d)-weighted fit of Validation/V20.  The earlier baseline results/t8_selcut_20261001
# used the legacy V1 shape B=2.544e-3 uc=35.67 C=3.100 (not rerun; compare against it).
case_names=(default hwang refit_trunc refit_untrunc)
case_B=(default 2.24e-3 2.9e-3 1.0e-4)      # hwang: Hwang+2024; refit_trunc: handoff 3.2 Nelder-Mead point
case_uc=(default 36.09 36.9 26.4)           # under the cut; refit_untrunc: untruncated refit (B->0 floored at 1e-4)
case_C=(default 3.85 2.8 6.7)

case_index() {
    local requested="$1" index
    for index in "${!case_names[@]}"; do
        if [[ "${case_names[index]}" == "${requested}" ]]; then printf '%s\n' "${index}"; return 0; fi
    done
    echo "Unknown CASE=${requested}; choose all or one of: ${case_names[*]}" >&2; exit 2
}

[[ "${STAGE}" == both || "${STAGE}" == lookup || "${STAGE}" == mlr ]] || { echo "STAGE must be both, lookup or mlr." >&2; exit 2; }
[[ -f "${RUNNER}" ]] || { echo "Cannot find runner: ${RUNNER}" >&2; exit 2; }
[[ -f "${DATA}" ]] || { echo "Cannot find data: ${DATA}" >&2; exit 2; }
[[ -f "${POSTERIOR}" ]] || { echo "Cannot find metallicity posterior: ${POSTERIOR}" >&2; exit 2; }
command -v conda >/dev/null 2>&1 || { echo "conda is not on PATH; load the site module first." >&2; exit 2; }
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

python -c 'import jax; print("JAX devices:", jax.devices())'
if [[ "${REQUIRE_GPU}" == "1" ]]; then
    python -c 'import jax; assert any(d.platform == "gpu" for d in jax.devices()), "No JAX GPU device is visible"'
fi

run_case() {
    local name="$1" B="$2" uc="$3" C="$4"
    local out="${OUTPUT_ROOT}/${name}"
    local lookup="${out}/dynamics_likelihood_lookup_t8.npz"
    local shape_args=()
    if [[ "${B}" != default ]]; then
        shape_args=(--shape-sensitivity --good-shape-b "${B}" --good-shape-uc "${uc}" --good-shape-c "${C}")
    fi
    local common=(
        --data "${DATA}" --output-dir "${out}" --metallicity-posterior "${POSTERIOR}"
        ${shape_args[@]+"${shape_args[@]}"}
        --selection-cut 3
        --lookup-mass-points 1024 --lookup-velocity-nodes 128 --lookup-sigma-extent 14
        --warmup 1000 --samples 1000 --chains 4 --seed 20260819 --target-accept 0.9
    )
    mkdir -p "${out}"
    echo "case=${name} B=${B} uc=${uc} C=${C} stage=${STAGE} output=${out}"
    if [[ "${STAGE}" == both || "${STAGE}" == lookup ]]; then
        if [[ -f "${lookup}" && -f "${out}/lookup_convergence_t8.json" ]]; then
            echo "Lookup already present; reusing ${lookup}"
        else
            python -u "${RUNNER}" --stage lookup "${common[@]}"
        fi
    fi
    if [[ "${STAGE}" == both || "${STAGE}" == mlr ]]; then
        [[ -f "${lookup}" ]] || { echo "Missing lookup for ${name}: run STAGE=lookup first." >&2; exit 2; }
        if [[ -f "${out}/mlr_mcmc_t8.npz" && -f "${out}/mlr_derived_grid_t8.npz" && -f "${out}/mlr_model.json" ]]; then
            echo "MLR fit already present; reusing ${out}"
        else
            python -u "${RUNNER}" --stage mlr "${common[@]}" --dynamics-lookup "${lookup}"
        fi
    fi
}

if [[ "${CASE}" == all ]]; then
    for index in "${!case_names[@]}"; do
        run_case "${case_names[index]}" "${case_B[index]}" "${case_uc[index]}" "${case_C[index]}"
    done
else
    index="$(case_index "${CASE}")"
    run_case "${case_names[index]}" "${case_B[index]}" "${case_uc[index]}" "${case_C[index]}"
fi
echo "Done: CASE=${CASE}, STAGE=${STAGE}"
