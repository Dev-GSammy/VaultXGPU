#!/usr/bin/env bash
# join_runs.sh -- join a run_bench.sh CSV with its .runs.csv sidecar.
#
# run_bench.sh writes two files: the plotter's own CSV (all timings and counters)
# and a sidecar holding what the plotter does not know about -- the sweep tag, the
# drive, the repeat index and the measured energy. Both carry a unique `run` id.
# This merges them into one wide CSV for plotting.
#
# Usage:
#   ./join_runs.sh experiments/<host>/bench_sweep.csv            # to stdout
#   ./join_runs.sh experiments/<host>/bench_sweep.csv joined.csv
set -euo pipefail

MAIN="${1:?usage: join_runs.sh <bench_*.csv> [output.csv]}"
SIDE="${MAIN%.csv}.runs.csv"
OUT="${2:-}"

[[ -f "$MAIN" ]] || { echo "no such file: $MAIN" >&2; exit 1; }
[[ -f "$SIDE" ]] || { echo "no sidecar next to it: $SIDE" >&2; exit 1; }

join_it() {
    awk -F, -v OFS=, '
        NR == FNR {
            if (FNR == 1) { for (i = 2; i <= NF; i++) sh[i] = $i; sn = NF; next }
            for (i = 2; i <= NF; i++) side[$1, i] = $i
            next
        }
        FNR == 1 {
            line = $0
            for (i = 2; i <= sn; i++) line = line OFS sh[i]
            print line
            next
        }
        {
            line = $0
            for (i = 2; i <= sn; i++) line = line OFS side[$8, i]
            print line
        }
    ' "$SIDE" "$MAIN"
}

if [[ -n "$OUT" ]]; then
    mkdir -p "$(dirname "$OUT")"
    join_it > "$OUT"
    echo "wrote $OUT"
else
    join_it
fi
