#!/bin/bash

cd "$(dirname "$0")/.."

CONFIGS=(
# ---- references ----------------------------------------------------------
"base_lru     no  lru"
"base_rrip    no  rrip"
"spp_lru      spp lru"
# Task 2A final I-RRIP and Task 2B final dynamic I-RRIP use the file defaults
"irrip_def    no  irrip"

# ---- Task 2A: I-RRIP ablation ---------------------------------------------
# I = RRPV of inserted instruction line, D = RRPV of inserted data line,
# S = data promotion step (0 = to RRPV 0), W = writeback hit promotes,
# E = data-first eviction
"ia_I0D2      no  irrip IRRIP_INS_INSTR=0 IRRIP_INS_DATA=2 IRRIP_DATA_PROMO_STEP=0 IRRIP_WB_PROMOTE=1 IRRIP_DATA_FIRST_EVICT=0"
"ia_I1D2      no  irrip IRRIP_INS_INSTR=1 IRRIP_INS_DATA=2 IRRIP_DATA_PROMO_STEP=0 IRRIP_WB_PROMOTE=1 IRRIP_DATA_FIRST_EVICT=0"
"ia_I0D3      no  irrip IRRIP_INS_INSTR=0 IRRIP_INS_DATA=3 IRRIP_DATA_PROMO_STEP=0 IRRIP_WB_PROMOTE=1 IRRIP_DATA_FIRST_EVICT=0"
"ia_I1D3      no  irrip IRRIP_INS_INSTR=1 IRRIP_INS_DATA=3 IRRIP_DATA_PROMO_STEP=0 IRRIP_WB_PROMOTE=1 IRRIP_DATA_FIRST_EVICT=0"
"ia_I0D2_S1   no  irrip IRRIP_INS_INSTR=0 IRRIP_INS_DATA=2 IRRIP_DATA_PROMO_STEP=1 IRRIP_WB_PROMOTE=1 IRRIP_DATA_FIRST_EVICT=0"
"ia_I0D3_S1   no  irrip IRRIP_INS_INSTR=0 IRRIP_INS_DATA=3 IRRIP_DATA_PROMO_STEP=1 IRRIP_WB_PROMOTE=1 IRRIP_DATA_FIRST_EVICT=0"
"ia_I0D2_W0   no  irrip IRRIP_INS_INSTR=0 IRRIP_INS_DATA=2 IRRIP_DATA_PROMO_STEP=0 IRRIP_WB_PROMOTE=0 IRRIP_DATA_FIRST_EVICT=0"
"ia_I0D3_W0   no  irrip IRRIP_INS_INSTR=0 IRRIP_INS_DATA=3 IRRIP_DATA_PROMO_STEP=0 IRRIP_WB_PROMOTE=0 IRRIP_DATA_FIRST_EVICT=0"
"ia_I0D2_E1   no  irrip IRRIP_INS_INSTR=0 IRRIP_INS_DATA=2 IRRIP_DATA_PROMO_STEP=0 IRRIP_WB_PROMOTE=1 IRRIP_DATA_FIRST_EVICT=1"
"ia_I0D3_E1   no  irrip IRRIP_INS_INSTR=0 IRRIP_INS_DATA=3 IRRIP_DATA_PROMO_STEP=0 IRRIP_WB_PROMOTE=1 IRRIP_DATA_FIRST_EVICT=1"
"ia_I0D2_W0E1 no  irrip IRRIP_INS_INSTR=0 IRRIP_INS_DATA=2 IRRIP_DATA_PROMO_STEP=0 IRRIP_WB_PROMOTE=0 IRRIP_DATA_FIRST_EVICT=1"
"ia_I0D3_W0E1 no  irrip IRRIP_INS_INSTR=0 IRRIP_INS_DATA=3 IRRIP_DATA_PROMO_STEP=0 IRRIP_WB_PROMOTE=0 IRRIP_DATA_FIRST_EVICT=1"
"ia_I0D3_S1W0   no  irrip IRRIP_INS_INSTR=0 IRRIP_INS_DATA=3 IRRIP_DATA_PROMO_STEP=1 IRRIP_WB_PROMOTE=0 IRRIP_DATA_FIRST_EVICT=0"
"ia_I0D3_S1W0E1 no  irrip IRRIP_INS_INSTR=0 IRRIP_INS_DATA=3 IRRIP_DATA_PROMO_STEP=1 IRRIP_WB_PROMOTE=0 IRRIP_DATA_FIRST_EVICT=1"

# ---- Task 2B: dynamic I-RRIP --------------------------------------------
# DYN_MODE 0: hint formula, instruction share in SRRIP leaders vs threshold T
"d0_T05       no  dyn_irrip DYN_MODE=0 DYN_THRESHOLD_PCT=5"
"d0_T10       no  dyn_irrip DYN_MODE=0 DYN_THRESHOLD_PCT=10"
"d0_T15       no  dyn_irrip DYN_MODE=0 DYN_THRESHOLD_PCT=15"
"d0_T20       no  dyn_irrip DYN_MODE=0 DYN_THRESHOLD_PCT=20"
"d0_T25       no  dyn_irrip DYN_MODE=0 DYN_THRESHOLD_PCT=25"
"d0_T30       no  dyn_irrip DYN_MODE=0 DYN_THRESHOLD_PCT=30"
"d0_T40       no  dyn_irrip DYN_MODE=0 DYN_THRESHOLD_PCT=40"
"d0_T50       no  dyn_irrip DYN_MODE=0 DYN_THRESHOLD_PCT=50"
"d0_T60       no  dyn_irrip DYN_MODE=0 DYN_THRESHOLD_PCT=60"
"d0_T70       no  dyn_irrip DYN_MODE=0 DYN_THRESHOLD_PCT=70"
"d0_T25_E256  no  dyn_irrip DYN_MODE=0 DYN_THRESHOLD_PCT=25 DYN_EPOCH=256"
"d0_T25_E4096 no  dyn_irrip DYN_MODE=0 DYN_THRESHOLD_PCT=25 DYN_EPOCH=4096"
# DYN_MODE 1: SRRIP leaders vs I-RRIP leaders, PSEL on (weighted) misses
"d1_W1_L32    no  dyn_irrip DYN_MODE=1 DYN_INSTR_MISS_WEIGHT=1 DYN_LEADERS=32"
"d1_W2_L32    no  dyn_irrip DYN_MODE=1 DYN_INSTR_MISS_WEIGHT=2 DYN_LEADERS=32"
"d1_W4_L32    no  dyn_irrip DYN_MODE=1 DYN_INSTR_MISS_WEIGHT=4 DYN_LEADERS=32"
"d1_W1_L16    no  dyn_irrip DYN_MODE=1 DYN_INSTR_MISS_WEIGHT=1 DYN_LEADERS=16"
"d1_W1_L64    no  dyn_irrip DYN_MODE=1 DYN_INSTR_MISS_WEIGHT=1 DYN_LEADERS=64"
"d1_W1_L32_E256  no  dyn_irrip DYN_MODE=1 DYN_INSTR_MISS_WEIGHT=1 DYN_LEADERS=32 DYN_EPOCH=256"
"d1_W1_L32_E4096 no  dyn_irrip DYN_MODE=1 DYN_INSTR_MISS_WEIGHT=1 DYN_LEADERS=32 DYN_EPOCH=4096"
"d1_W3_L32    no  dyn_irrip DYN_MODE=1 DYN_INSTR_MISS_WEIGHT=3 DYN_LEADERS=32"
"d1_W8_L32    no  dyn_irrip DYN_MODE=1 DYN_INSTR_MISS_WEIGHT=8 DYN_LEADERS=32"
"d1_W2_L16    no  dyn_irrip DYN_MODE=1 DYN_INSTR_MISS_WEIGHT=2 DYN_LEADERS=16"
"d1_W2_L64    no  dyn_irrip DYN_MODE=1 DYN_INSTR_MISS_WEIGHT=2 DYN_LEADERS=64"
"d1_W2_L32_E256  no  dyn_irrip DYN_MODE=1 DYN_INSTR_MISS_WEIGHT=2 DYN_LEADERS=32 DYN_EPOCH=256"
"d1_W2_L32_E4096 no  dyn_irrip DYN_MODE=1 DYN_INSTR_MISS_WEIGHT=2 DYN_LEADERS=32 DYN_EPOCH=4096"
# instruction : data miss weight 3 : 2 (= 1.5)
"d1_W3D2_L16  no  dyn_irrip DYN_MODE=1 DYN_INSTR_MISS_WEIGHT=3 DYN_DATA_MISS_WEIGHT=2 DYN_LEADERS=16"
"d1_W3D2_L32  no  dyn_irrip DYN_MODE=1 DYN_INSTR_MISS_WEIGHT=3 DYN_DATA_MISS_WEIGHT=2 DYN_LEADERS=32"
"d1_W3D2_L64  no  dyn_irrip DYN_MODE=1 DYN_INSTR_MISS_WEIGHT=3 DYN_DATA_MISS_WEIGHT=2 DYN_LEADERS=64"

# ---- Task 3: SPP at the L1D + L2C policy ------------------------------------
# the final dynamic I-RRIP is the file default (DYN_MODE=1, weight 2, 32 leaders)
"spp_rrip     spp rrip"
"spp_irrip    spp irrip"
"spp_dyn_nopf spp dyn_irrip DYN_PF_AWARE=0"
"spp_dyn      spp dyn_irrip"
"spp_dyn_ins2 spp dyn_irrip DYN_PF_INS=2"
"spp_dyn_use1 spp dyn_irrip DYN_PF_USE_RRPV=1"
"spp_dyn_use2 spp dyn_irrip DYN_PF_USE_RRPV=2"
"spp_dyn_noev spp dyn_irrip DYN_PF_EVICT_FIRST=0"
"spp_dyn_stat spp dyn_irrip DYN_PF_IN_STATS=1"
)

cmd=$1; shift
case "$cmd" in
build)
    for c in "${CONFIGS[@]}"; do
        set -- $c
        [ -x "bin/$1" ] || scripts/build.sh "$@" || exit 1
    done ;;
run)
    tdir=$1; rdir=$2; shift 2
    if [ $# -eq 0 ]; then
        set -- $(for c in "${CONFIGS[@]}"; do echo "${c%% *}"; done)
    fi
    scripts/run_batch.sh "$tdir" "$rdir" "$@" ;;
*)
    echo "usage: $0 build | run <trace_dir> <results_dir> [tag ...]"; exit 1 ;;
esac
