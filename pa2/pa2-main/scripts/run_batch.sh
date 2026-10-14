#!/bin/bash
# Run every given binary tag on all traces, JOBS simulations in parallel.
#   JOBS=16 scripts/run_batch.sh <trace_dir> <results_dir> <tag> [<tag> ...]
tdir=$1; rdir=$2; shift 2
cd "$(dirname "$0")/.."
for tag in "$@"; do for t in trace1 trace2 trace3; do echo "$tag $t"; done; done |
    xargs -P "${JOBS:-16}" -n 2 sh -c 'scripts/run_one.sh "$0" "$1" "'"$tdir"'" "'"$rdir"'"'
