#!/bin/bash
# build_cvt_p.sh OUTDIR -- Verilator build of the tnca_cvt_int_p miter (GB10, worktree mirror); run: OUTDIR/obj_cvt/Vcvt_int_miter fn rm | random N SEED
set -e
OUT=${1:?outdir}; HERE=$(cd $(dirname $0) && pwd); ROOT=$(cd $HERE/../.. && pwd)
export VERILATOR_ROOT=$HOME/tc_env/root/usr/share/verilator VERILATOR_BIN=../../bin/verilator_bin
VL=$HOME/tc_env/root/usr/bin/verilator
mkdir -p $OUT && cd $OUT
cp $ROOT/tb/area/fast/cvt_int_miter.cpp .
nice -n 19 $VL --cc --exe --build -O3 -Wno-fatal -Wno-lint -Wno-style --top-module cvt_int_miter -Mdir obj_cvt -I$ROOT/rtl/common \
    $HERE/cvt_int_p_miter.v $ROOT/rtl/core/tnca_cvt_int.v $ROOT/rtl/core/tnca_cvt_int_p.v cvt_int_miter.cpp -CFLAGS -O2 -j 4 > build_cvt.log 2>&1 || { tail -20 build_cvt.log; exit 2; }
echo built $OUT/obj_cvt/Vcvt_int_miter
