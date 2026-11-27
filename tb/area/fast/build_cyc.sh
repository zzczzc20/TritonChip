#!/bin/bash
# build_cyc.sh OUTDIR CUTS SHIFT XLS_ALIGN LAT -- Verilator build of the cycle-level FMA miter (GB10, worktree mirror).
set -e
OUT=${1:?outdir}; CUTS=${2:?cuts like 1010}; SHIFT=${3:-1}; XA=${4:-0}; LAT=${5:-2}
HERE=$(cd $(dirname $0) && pwd); ROOT=$(cd $HERE/../../.. && pwd)
export VERILATOR_ROOT=$HOME/tc_env/root/usr/share/verilator VERILATOR_BIN=../../bin/verilator_bin
VL=$HOME/tc_env/root/usr/bin/verilator
mkdir -p $OUT && cd $OUT
R=$ROOT/rtl; G=$R/gen
SRC="$R/core/tnca_fp32_fma.v $R/core/tnca_fmax_fma.v $R/core/tnca_fmax_add.v $R/core/tnca_fmax_sel.v $R/core/tnca_fmax_round.v
     $R/core/tnca_fmax_alignp.v $R/core/tnca_fmax_mul.v $R/core/tnca_fmax_align.v $R/core/tnca_lzc81.v $R/core/tnca_fmax_norm.v
     $G/tnca_fma_align_xls.v $G/tnca_fma_norm_xls.v $G/tnca_fma_round_xls.v $G/tnca_fmad_norm_xls.v $G/tnca_fma_l3_xls.v $G/tnca_fmad_l3_xls.v"
# Verilator resolves the pins of the refusing branches (tnca_fp32_fma g_bad): drop those instance lines in a copy
mkdir -p src; grep -v 'LAT_must_be_2_or_3' $R/core/tnca_fp32_fma.v > src/tnca_fp32_fma.v
SRC2=$(echo $SRC | sed "s#$R/core/tnca_fp32_fma.v#src/tnca_fp32_fma.v#")
cp $HERE/fma_cyc_miter.cpp $HERE/fma_vectors.h .
nice -n 19 $VL --cc --exe --build -O3 -Wno-fatal -Wno-lint -Wno-style --top-module fma_cyc_miter -Mdir obj_cyc \
    -GCUTS=4\'b$CUTS -GSHIFT=$SHIFT -GXLS_ALIGN=$XA -GLAT=$LAT \
    $HERE/fma_cyc_miter.v $SRC2 fma_cyc_miter.cpp -CFLAGS "-O2 -I." -j 4 > build_cyc.log 2>&1 || { tail -20 build_cyc.log; exit 2; }
echo built $OUT/obj_cyc/Vfma_cyc_miter
