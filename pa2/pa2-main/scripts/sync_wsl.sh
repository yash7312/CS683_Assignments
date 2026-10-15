#!/bin/bash
# strip CRLF line endings
# (the Windows checkout uses CRLF; build_champsim.sh needs LF).
#   scripts/sync_wsl.sh <src_repo_dir> <dst_dir>
set -e
src=$1; dst=$2
mkdir -p "$dst"
rsync -a --delete --exclude .git --exclude bin --exclude obj --exclude build_logs --exclude results --exclude report "$src/" "$dst/"
cd "$dst"
find . -type f \( -name "*.cc" -o -name "*.h" -o -name "*_repl" -o -name "*_pref" -o -name "*.bpred" -o -name "*.sh" -o -name "*.py" -o -name Makefile \) -exec sed -i 's/\r$//' {} +
chmod +x build_champsim.sh scripts/*.sh
