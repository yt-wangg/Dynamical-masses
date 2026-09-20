#!/bin/bash
# T8 mass-independent-outlier, stage 1: full-sample validation + timing trial.
# Runs on astronode-garching-node03; launch with nohup and check back later.
#
#   nohup bash scripts/run_t8_mio_validation_timing.sh > /tmp/t8_mio_stage1.log 2>&1 &
set -u

source /nexus/posix0/MIA-astro-env/hxr/jdli/miniforge3/etc/profile.d/conda.sh
conda activate dyn

REPO=/home/jdli/nexus/collab/Dynamical-masses
BASELINE=$REPO/results/hierarchical_metallicity_t8_1_formal
ROOT=$REPO/results/t8_mass_independent_outlier_formal_20260919
PY=/home/jdli/nexus/miniforge3/envs/dyn/bin/python

mkdir -p "$ROOT"
cd "$REPO/src/examples"
export JAX_PLATFORMS=cpu
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

echo "[stage1] validation start: $(date -Is)"
"$PY" run_mass_independent_outlier.py \
  --baseline "$BASELINE" --output "$ROOT" --mode validation --reference-workers 8 \
  > "$ROOT/validation.log" 2>&1
status=$?
echo "[stage1] validation exit=$status: $(date -Is)"
if [ $status -ne 0 ]; then
  echo "[stage1] VALIDATION FAILED; see $ROOT/validation.log"
  exit 1
fi

echo "[stage1] timing trial start: $(date -Is)"
taskset -c 0-7 "$PY" run_mass_independent_outlier.py \
  --baseline "$BASELINE" --output "$ROOT/timing_trial" --validation-root "$ROOT" \
  --mode timing --warmup 100 --samples 100 --chains 1 --seed 20260919 \
  > "$ROOT/timing_trial.log" 2>&1
status=$?
echo "[stage1] timing trial exit=$status: $(date -Is)"
if [ $status -ne 0 ]; then
  echo "[stage1] TIMING TRIAL FAILED; see $ROOT/timing_trial.log"
  exit 1
fi

echo "[stage1] both stages completed: $(date -Is)"
