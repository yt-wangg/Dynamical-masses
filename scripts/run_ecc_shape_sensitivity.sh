#!/usr/bin/env bash
#SBATCH --job-name=dyn-ecc-shape
#SBATCH --output=ecc-shape-%j.out
#SBATCH --error=ecc-shape-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00

# Eccentricity-distribution sensitivity of the selection-truncated (u/sigma_u>3) MLR.
# The p(u~) shape is refitted (src/examples/fit_shape_eccentricity_grid.py, same (s_proj,d)-weighted
# pipeline as Validation/V20) for p(e) ~ e^alpha; each case rebuilds the dynamics lookup with that fixed
# (B, uc, C), reuses the baseline T8 metallicity posterior, and refits the MLR.
# alpha=1 (thermal) reproduces the library default shape, i.e. results/selcut_shape_sensitivity_20261005/default.
# Cluster-specific partition/account/GPU options belong in the sbatch command.
set -euo pipefail

STAGE="${STAGE:-both}"            # both, lookup, or mlr
CASE="${CASE:-all}"               # all (= alpha_1.233 alpha_1 alpha_1.5) or one case name below
CONDA_ENV="${CONDA_ENV:-dyn}"
REQUIRE_GPU="${REQUIRE_GPU:-1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
cd "${PROJECT_ROOT}"

DATA="${DATA:-${PROJECT_ROOT}/data/jd_msms_single_bic_1kpc_filtered_cmdcut_cutb_jcaps_err_mhcal_good.fits}"
BASELINE_DIR="${BASELINE_DIR:-${PROJECT_ROOT}/results/t8_selcut_20261001}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/results/ecc_shape_sensitivity_20261009}"
RUNNER="${RUNNER:-${PROJECT_ROOT}/src/examples/run_hierarchical_metallicity_test.py}"
POSTERIOR="${POSTERIOR:-${BASELINE_DIR}/latent_metallicity_weights_t8.npz}"

# name  B  uc  C  from results/shape_eccentricity_grid/shapes.json (p(e) ~ e^alpha, all cases simulated with
# common random numbers).  Adopted: central alpha=1.233 (Hwang+2022 alpha(s_proj) per system of the sample, pooled e
# fitted by MLE) and systematic range alpha in [1.0, 1.5].  CASE=all runs these three; the remaining cases are
# stress tests (run with CASE=<name>).  alpha=0 is not run: its fit has B<0.
case_names=(alpha_1.233 alpha_1 alpha_1.5 alpha_0.5 alpha_1.3 alpha_2 alpha_hwang22_s)
case_B=(2.5050e-03 2.2943e-03 2.6522e-03 8.6055e-04 2.5478e-03 2.8479e-03 2.5034e-03)
case_uc=(37.2461 36.2851 37.9042 27.7589 37.4244 38.7937 37.3235)
case_C=(2.4500 3.5567 1.7151 7.9676 2.2195 1.0798 2.3890)
ADOPTED_CASES=(alpha_1.233 alpha_1 alpha_1.5)

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
    for name in "${ADOPTED_CASES[@]}"; do
        index="$(case_index "${name}")"
        run_case "${case_names[index]}" "${case_B[index]}" "${case_uc[index]}" "${case_C[index]}"
    done
else
    index="$(case_index "${CASE}")"
    run_case "${case_names[index]}" "${case_B[index]}" "${case_uc[index]}" "${case_C[index]}"
fi
echo "Done: CASE=${CASE}, STAGE=${STAGE}"
