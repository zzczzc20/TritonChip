#!/bin/bash
# build_mac.sh OUTDIR BE -- Verilator build of the MAC lane miter at back-end encoding BE (GB10, worktree mirror).
set -e
OUT=${1:?outdir}; BE=${2:?BE}; HERE=$(cd $(dirname $0) && pwd); ROOT=$(cd $HERE/../../.. && pwd)
export VERILATOR_ROOT=$HOME/tc_env/root/usr/share/verilator VERILATOR_BIN=../../bin/verilator_bin
VL=$HOME/tc_env/root/usr/bin/verilator
mkdir -p $OUT && cd $OUT
R=$ROOT/rtl; G=$R/gen
SRC="$R/core/tnca_mac_lane.v $R/core/tnca_lzc81.v $R/core/tnca_fmax_norm.v $R/core/tnca_fmax_round.v $R/core/tnca_fmax_align.v $R/core/tnca_fmax_alignp.v $R/core/tnca_fmax_mul.v
     $G/tnca_widen_xls.v $G/tnca_mac_s1_xls.v $G/tnca_mac_s2_xls.v $G/tnca_mac_f2_xls.v $G/tnca_mac_b0_xls.v $G/tnca_fmad_norm_xls.v $G/tnca_fma_round_xls.v"
cp $HERE/mac_miter.cpp .
nice -n 19 $VL --cc --exe --build -O3 -Wno-fatal -Wno-lint -Wno-style --top-module mac_miter -Mdir obj_mac -GBE=$BE \
    $HERE/mac_miter.v $SRC $HERE/mac_miter.cpp -CFLAGS -O2 -j 4 > build_mac.log 2>&1 || { tail -20 build_mac.log; exit 2; }
echo built $OUT/obj_mac/Vmac_miter
