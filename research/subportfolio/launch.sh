#!/usr/bin/env bash
# Fan the draws across cores on a big machine, then merge.
#
#   ./launch.sh mid                       # the 400-name study, defaults
#   ./launch.sh index                     # the 5000-name study
#   WORKERS=64 DRAWS=256 ./launch.sh index   # explicit count
#   RESERVE=2 ./launch.sh index              # keep only 2 cores free
#   WORKERS=4 THREADS=5 ./launch.sh mid      # set the split by hand
#   SUBSET=sector ./launch.sh index           # sector sub-indices, not random names
#
# Two dimensions to budget, not one: WORKERS processes each running THREADS
# threads, with WORKERS * THREADS held under the core count.
#
# BLAS gets one thread per worker, as before. The race does NOT. With
# winning[fast] the expensive step runs in fastrace, which links rayon and
# reads only RAYON_NUM_THREADS -- none of the BLAS variables reach it. Left
# alone it opens a pool per worker: 32 OS threads each, so 25 workers put 800
# threads on 28 cores and drove the load average past 600.
#
# Pinning it to 1 is the other mistake. Measured on one race+factor call,
# k=3, n=400:
#
#     RAYON_NUM_THREADS unset (28 cores)      68.1s
#     RAYON_NUM_THREADS=1                   1350.4s
#
# a 19.8x speedup, so the kernel is close to perfectly parallel and the work
# per draw is a fixed number of core-seconds. Worker count does not change the
# total, only how much the machine is thrashed, so prefer few workers with
# several threads each.
set -euo pipefail
cd "$(dirname "$0")"

# The interpreter, so a venv that is not on PATH still works:
#   PYTHON=../../allocation-py312/bin/python ./launch.sh mid
PY="${PYTHON:-python}"

# Keep the machine awake for the whole run. An overnight study on a laptop
# that idles out at 2am is a wasted night, and the shards only checkpoint
# while they are running.
CAFF=""
command -v caffeinate >/dev/null 2>&1 && CAFF="caffeinate -ims"

SCALE="${1:-mid}"
SUBSET="${SUBSET:-random}"
# Leave headroom. Taking every core makes the machine unusable for whoever is
# sitting at it, and the last few workers buy very little: the draws are
# independent, so the run is already near-linear well short of saturation.
NCPU="$( (command -v nproc >/dev/null && nproc) || sysctl -n hw.ncpu )"
RESERVE="${RESERVE:-8}"
WORKERS="${WORKERS:-5}"

case "$SCALE" in
  mid)   N="${N:-400}";  M="${M:-60}";  K="${K:-3}"; SEED="${SEED:-12}"
         DRAWS="${DRAWS:-25}";  TS="${TS:-20 40 100}"
         TS_EST="${TS_EST:-15 20 30 40 60 80 100 150 250}" ;;
  index) N="${N:-5000}"; M="${M:-200}"; K="${K:-2}"; SEED="${SEED:-4}"
         DRAWS="${DRAWS:-32}";  TS="${TS:-52 104}"
         TS_EST="${TS_EST:-26 52 78 104 156 260}" ;;
  *) echo "usage: $0 [mid|index]" >&2; exit 2 ;;
esac

TAG="${SCALE}-${SUBSET}-n${N}-m${M}-k${K}-s${SEED}"
THREADS="${THREADS:-$(( WORKERS > 0 ? (NCPU - RESERVE) / WORKERS : 1 ))}"
[ "$THREADS" -lt 1 ] && THREADS=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
       VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
       RAYON_NUM_THREADS="$THREADS"
echo "$NCPU cores, reserving $RESERVE: $WORKERS workers x $THREADS rayon threads"

echo "checking the install before spending anything long"
"$PY" smoke.py

mkdir -p results logs
echo "$TAG: $DRAWS draws over $WORKERS workers"
for ((i = 0; i < WORKERS; i++)); do
  $CAFF "$PY" run.py --scale "$SCALE" --n "$N" --m "$M" --k "$K" --seed "$SEED" --subset "$SUBSET" \
    --draws "$DRAWS" --Ts $TS --Ts-est $TS_EST \
    --shard "$i" --shards "$WORKERS" --tag "$TAG" \
    > "logs/${TAG}-shard${i}.log" 2>&1 &
done
wait

"$PY" merge.py --tag "$TAG" | tee "results/${TAG}.txt"
