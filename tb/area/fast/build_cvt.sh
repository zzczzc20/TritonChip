#!/bin/bash
# build_cvt.sh OUTDIR -- Verilator build of the int<->float conversion miter (GB10, worktree mirror)
set -e
OUT=${1:?outdir}; HERE=$(cd $(dirname $0) && pwd); ROOT=$(cd $HERE/../../.. && pwd)
export VERILATOR_ROOT=$HOME/tc_env/root/usr/share/verilator VERILATOR_BIN=../../bin/verilator_bin
VL=$HOME/tc_env/root/usr/bin/verilator
mkdir -p $OUT && cd $OUT
cp $HERE/cvt_int_miter.cpp .
nice -n 19 $VL --cc --exe --build -O3 -Wno-fatal -Wno-lint -Wno-style --top-module cvt_int_miter -Mdir obj_cvt -I$ROOT/rtl/common \
    $HERE/cvt_int_miter.v $ROOT/tools/r42/area/exp/cvt_int_ref.v $ROOT/rtl/core/tnca_cvt_int.v cvt_int_miter.cpp -CFLAGS -O2 -j 4 > build_cvt.log 2>&1 || { tail -20 build_cvt.log; exit 2; }
echo built $OUT/obj_cvt/Vcvt_int_miter
