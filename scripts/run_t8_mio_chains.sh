#!/bin/bash
# T8 mass-independent-outlier, stage 2: four formal chains, one process each.
# Launch only after stage 1 (validation + timing trial) passed. Cores are
# disjoint 8-core sets; adjust to the node's current load before launching.
#
#   nohup bash scripts/run_t8_mio_chains.sh > /tmp/t8_mio_stage2.log 2>&1 &
set -u

source /nexus/posix0/MIA-astro-env/hxr/jdli/miniforge3/etc/profile.d/conda.sh
conda activate dyn

REPO=/home/jdli/nexus/collab/Dynamical-masses
BASELINE=$REPO/results/hierarchical_metallicity_t8_1_formal
ROOT=$REPO/results/t8_mass_independent_outlier_formal_20260919
PY=/home/jdli/nexus/miniforge3/envs/dyn/bin/python

cd "$REPO/src/examples"
export JAX_PLATFORMS=cpu
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

SEEDS=(20260920 20260921 20260922 20260923)
CORE_SETS=("0-7" "8-15" "16-23" "24-31")

for i in 0 1 2 3; do
  if [ -f "$ROOT/chain_0$i.pid" ] && kill -0 "$(cat "$ROOT/chain_0$i.pid")" 2>/dev/null; then
    echo "[stage2] chain $i already running (pid $(cat "$ROOT/chain_0$i.pid")); aborting."
    exit 1
  fi
done

for i in 0 1 2 3; do
  echo "[stage2] launching chain $i (seed ${SEEDS[$i]}, cores ${CORE_SETS[$i]}): $(date -Is)"
  taskset -c "${CORE_SETS[$i]}" nohup "$PY" run_mass_independent_outlier.py \
    --baseline "$BASELINE" --output "$ROOT" --mode chain --chain-index "$i" \
    --start-draw 500 --seed "${SEEDS[$i]}" \
    --warmup 1500 --samples 2000 --chains 1 --target-accept 0.95 \
    > "$ROOT/chain_0$i.launch.log" 2>&1 &
  echo $! > "$ROOT/chain_0$i.pid"
done

echo "[stage2] all four chains launched: $(date -Is)"
wait
echo "[stage2] all four chains finished: $(date -Is)"
