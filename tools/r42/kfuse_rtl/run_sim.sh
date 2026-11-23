#!/bin/bash
# run_sim.sh TAG G W N [extra -P overrides and +args...]  -- on GB10, from the tree root (~/r42/<NAME>):
#   compiles tb_kfuse_lane against rtl/core/tnca_kfuse_lane.v + the generated R41 back end with iverilog,
#   generates the vectors (the model, scalar_step), runs vvp, writes <out>/<TAG>.{hex.sha256,log}
# Environment: OUT (default ~/r42/kfuse-rtl_out), NICE (default 19), PROCS for the generator (default 2),
# NOGEN=1 reuses an existing vector file, SEED, PARAMS ("-Ptb_kfuse_lane.CUTS=8'b..." etc.), PLUSARGS.
set -u
TAG=$1; G=$2; W=$3; N=$4
OUT=${OUT:-$HOME/r42/kfuse-rtl_out}
mkdir -p $OUT
source ~/tc_env/env.sh
ROOT=$(cd $(dirname $0)/../../.. && pwd)
VEC=$OUT/vec_G${G}_W${W}_N${N}_s${SEED:-1}.hex
if [ -z "${NOGEN:-}" ] || [ ! -f $VEC ]; then
    nice -n 19 python3 $ROOT/tools/r42/kfuse_rtl_vectors.py --G $G --W $W --N $N --seed ${SEED:-1} --procs ${PROCS:-2} --out $VEC ${GENARGS:-} > $OUT/${TAG}.gen.log 2>&1
    sha256sum $VEC > $VEC.sha256
fi
SRC="$ROOT/rtl/core/tnca_kfuse_lane.v $ROOT/rtl/gen/tnca_mac_b0_xls.v $ROOT/rtl/gen/tnca_fmad_norm_xls.v $ROOT/rtl/gen/tnca_fma_round_xls.v"
nice -n 19 iverilog -g2005 -o $OUT/$TAG.vvp -Ptb_kfuse_lane.G=$G -Ptb_kfuse_lane.W=$W -Ptb_kfuse_lane.NMAX=$((N + 200000)) ${PARAMS:-} \
    $ROOT/tools/r42/kfuse_rtl/tb_kfuse_lane.v $SRC > $OUT/$TAG.compile.log 2>&1 || { echo COMPILE FAILED; cat $OUT/$TAG.compile.log; exit 2; }
NV=$(wc -l < $VEC)
( time nice -n 19 vvp -n $OUT/$TAG.vvp +VEC=$VEC +N=$NV ${PLUSARGS:-} ) > $OUT/$TAG.log 2>&1
tail -3 $OUT/$TAG.log
