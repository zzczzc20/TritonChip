#!/bin/bash
# build.sh OUTDIR [GENERICS] -- Verilator build of the FMA stage miters on GB10 from the worktree mirror.
set -e
OUT=${1:?outdir}; HERE=$(cd $(dirname $0) && pwd); ROOT=$(cd $HERE/../../.. && pwd)
export VERILATOR_ROOT=$HOME/tc_env/root/usr/share/verilator VERILATOR_BIN=../../bin/verilator_bin
VL=$HOME/tc_env/root/usr/bin/verilator
mkdir -p $OUT && cd $OUT
R=$ROOT/rtl
SRC="$R/core/tnca_fmax_align.v $R/core/tnca_fmax_alignp.v $R/core/tnca_fmax_mul.v $R/core/tnca_lzc81.v $R/core/tnca_fmax_norm.v $R/core/tnca_fmax_round.v $R/gen/tnca_fma_align_xls.v $R/gen/tnca_fmad_norm_xls.v $R/gen/tnca_fma_round_xls.v"
cp $HERE/fast_miter.cpp .
nice -n 19 $VL --cc --exe --build -O3 -Wno-fatal -Wno-lint -Wno-style --top-module miter_all -Mdir obj_all -GADD=${ADD:-0} -GSHIFT=${SHIFT:-0} $HERE/miter_all.v $SRC $HERE/fast_miter.cpp -CFLAGS -O2 -j 4 > build_fma.log 2>&1 || { tail -20 build_fma.log; exit 2; }
echo built $OUT/obj_all/Vmiter_all
