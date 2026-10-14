#!/bin/bash
# Run one binary on one trace with 25M warm-up and 25M simulated instructions.
#
#   scripts/run_one.sh <tag> <trace_name> <trace_dir> <results_dir>
#
# Output: <results_dir>/<tag>/<trace_name>.txt
tag=$1; trace=$2; tdir=$3; rdir=$4
cd "$(dirname "$0")/.."
mkdir -p "$rdir/$tag"
out="$rdir/$tag/$trace.txt"
[ -s "$out" ] && exit 0                 # already done
[ -e "$out.tmp" ] && exit 0             # being run by another runner
./bin/"$tag" -warmup_instructions 25000000 -simulation_instructions 25000000 \
    -traces "$tdir/$trace.champsimtrace.gz" > "$out.tmp" 2>&1 && mv "$out.tmp" "$out"
