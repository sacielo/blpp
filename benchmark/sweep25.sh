#!/bin/bash
# 25 repetitions x 17 sizes of `build/benchmark d n`; raw tables -> runs-<host>/rep_NN.tsv
# columns: rep n op ms flop_s
#
# Runs are machine-local and uncommitted; one runs-<hostname>/ folder per
# system so results from different machines can coexist. Plot with:
#   python3 analyze25.py          (needs numpy + matplotlib)
#
# The stock build uses a 0.5 s / 3-iter / 10 s timing budget (full sweep is
# slow on weak cores). For a faster sweep rebuild with a tighter budget:
#   cmake -S . -B build -DCMAKE_CXX_FLAGS="-DK_MIN_TIME=0.25 -DK_MIN_ITERS=2 -DK_MAX_TIME=1.5"
set -u
cd "$(dirname "$0")"
NREP=${NREP:-25}
HOST=$(hostname | cut -d. -f1)
RUNS=runs-$HOST
SIZES="16 32 64 128 256 512 1024 2048 4096 8192 16384 32768 65536 131072 262144 524288 1048576"
[ -x build/benchmark ] || { echo "build/benchmark missing; see CMakeLists.txt"; exit 1; }
mkdir -p "$RUNS"
for r in $(seq 1 "$NREP"); do
    out=$RUNS/rep_$(printf %02d "$r").tsv
    : > "$out"
    for n in $SIZES; do
        ./build/benchmark d "$n" \
          | awk -v r="$r" -v n="$n" '$8 == "OK" && $3 + 0 > 0 { print r, n, $1, $2, $3 }' >> "$out"
    done
    echo "rep $r done $(date +%T)"
done
echo SWEEP-DONE
