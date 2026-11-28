#!/bin/bash
# build_unitp.sh OUTDIR -- Verilator build of the tnca_fmax_unitp miter (GB10); run OUTDIR/obj/Vunitp_miter (exhaustive, a few minutes)
set -e
OUT=${1:?outdir}; HERE=$(cd $(dirname $0) && pwd); ROOT=$(cd $HERE/../.. && pwd)
export VERILATOR_ROOT=$HOME/tc_env/root/usr/share/verilator VERILATOR_BIN=../../bin/verilator_bin
VL=$HOME/tc_env/root/usr/bin/verilator
mkdir -p $OUT && cd $OUT
cp $HERE/unitp_miter.cpp .
nice -n 19 $VL --cc --exe --build -O3 -Wno-fatal -Wno-lint -Wno-style --top-module unitp_miter -Mdir obj \
    $HERE/unitp_miter.v $ROOT/rtl/core/tnca_fmax_mul.v $ROOT/rtl/core/tnca_fmax_unitp.v unitp_miter.cpp -CFLAGS -O2 -j 4 > build.log 2>&1 || { tail -20 build.log; exit 2; }
echo built $OUT/obj/Vunitp_miter
