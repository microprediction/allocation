#!/usr/bin/env bash
# Fan the draws across cores on a big machine, then merge.
#
#   ./launch.sh mid                       # the 400-name study, defaults
#   ./launch.sh index                     # the 5000-name study
#   WORKERS=64 DRAWS=256 ./launch.sh index
#
# One BLAS thread per worker on purpose. The work is already parallel across
# draws, so letting each worker start its own thread pool oversubscribes the
# box and makes the whole thing slower. That is also what ran this out of
# memory on a laptop.
set -euo pipefail
cd "$(dirname "$0")"

SCALE="${1:-mid}"
WORKERS="${WORKERS:-$( (command -v nproc >/dev/null && nproc) || sysctl -n hw.ncpu )}"

case "$SCALE" in
  mid)   N="${N:-400}";  M="${M:-60}";  K="${K:-3}"; SEED="${SEED:-12}"
         DRAWS="${DRAWS:-25}";  TS="${TS:-20 40 100}"; EXTRA="" ;;
  index) N="${N:-5000}"; M="${M:-200}"; K="${K:-2}"; SEED="${SEED:-4}"
         DRAWS="${DRAWS:-32}";  TS="${TS:-52 104}"
         RANK="${RANK:-5}"; EXTRA="--rank $RANK" ;;
  *) echo "usage: $0 [mid|index]" >&2; exit 2 ;;
esac

TAG="${SCALE}-n${N}-m${M}-k${K}-s${SEED}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
       VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1

echo "checking the install before spending anything long"
python smoke.py

mkdir -p results logs
echo "$TAG: $DRAWS draws over $WORKERS workers"
for ((i = 0; i < WORKERS; i++)); do
  python run.py --scale "$SCALE" --n "$N" --m "$M" --k "$K" --seed "$SEED" \
    --draws "$DRAWS" --Ts $TS --shard "$i" --shards "$WORKERS" --tag "$TAG" \
    $EXTRA \
    > "logs/${TAG}-shard${i}.log" 2>&1 &
done
wait

python merge.py --tag "$TAG" | tee "results/${TAG}.txt"
