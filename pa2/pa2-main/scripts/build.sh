#!/bin/bash
# Build one ChampSim configuration and store it as bin/<tag>.
# scripts/build.sh <tag> <l1d_prefetcher> <l2c_replacement> [MACRO=VALUE ...]

set -e
tag=$1; pref=$2; repl=$3; shift 3
cd "$(dirname "$0")/.."
mkdir -p bin build_logs
if [ $# -gt 0 ]; then
    f=replacement/_${tag}.l2c_repl
    { for d in "$@"; do echo "#define ${d%%=*} ${d#*=}"; done
      echo "#include \"${repl}.l2c_repl\""; } > "$f"
    repl=_${tag}
fi
TERM=${TERM:-dumb} MAKEFLAGS=-j16 ./build_champsim.sh "$pref" "$repl" "$tag" > "build_logs/$tag.log" 2>&1 || { echo "BUILD FAILED: $tag (see build_logs/$tag.log)"; rm -f "replacement/_${tag}.l2c_repl"; exit 1; }
rm -f "replacement/_${tag}.l2c_repl"
mv "bin/hashed_perceptron-no-${pref}-no-no-no-no-no-lru-lru-lru-${repl}-lru-lru-lru-lru-1core-${tag}" "bin/$tag"
echo "built bin/$tag"
