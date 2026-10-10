#!/bin/bash
# 25 repetitions x 17 sizes of `bench_sweep d n`; raw tables -> runs/rep_NN.tsv
# columns: rep n op ms flop_s
set -u
cd "$(dirname "$0")"
SIZES="16 32 64 128 256 512 1024 2048 4096 8192 16384 32768 65536 131072 262144 524288 1048576"
mkdir -p runs
for r in $(seq 1 25); do
    out=runs/rep_$(printf %02d "$r").tsv
    : > "$out"
    for n in $SIZES; do
        ./bench25/bench_sweep d "$n" \
          | awk -v r="$r" -v n="$n" '$8 == "OK" && $3 + 0 > 0 { print r, n, $1, $2, $3 }' >> "$out"
    done
    echo "rep $r done $(date +%T)"
done
echo SWEEP-DONE
