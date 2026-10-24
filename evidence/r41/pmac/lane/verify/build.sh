#!/bin/bash
# build.sh CUTS OUTDIR -- Verilate lane_tb (every variant at CUTS, beside the
# current lane) into OUTDIR. Run from anywhere; sources from this tree.
set -e
CUTS=${1:?cuts, e.g. 1100}; OUT=${2:?outdir}
here=$(cd $(dirname $0) && pwd)
root=$(cd $here/../../../../.. && pwd)
mkdir -p $OUT; cd $OUT
G=$root/rtl/gen
verilator --cc --exe --build -O3 -Wno-fatal -Wno-lint -Wno-style --top-module lane_tb \
    -GCUTS=4\'b$CUTS \
    $here/lane_tb.v $root/rtl/core/tnca_pmac_lane.v $G/tnca_widen_xls.v \
    $G/tnca_fdp_v1_f1_xls.v $G/tnca_fdp_v2l_f1_xls.v $G/tnca_fdp_v2d_f1_xls.v $G/tnca_fdp_f2_xls.v \
    $G/tnca_fdp_v3_f1_xls.v $G/tnca_fdp_v3_f2_xls.v $G/tnca_fma_alignp_xls.v \
    $G/tnca_fma_norm_xls.v $G/tnca_fma_round_xls.v \
    $here/ref/*.v $here/lane_check.cpp -j 4 > build.log 2>&1 || { tail -30 build.log; exit 2; }
sha256sum $here/lane_tb.v $root/rtl/core/tnca_pmac_lane.v $G/tnca_fdp_*_xls.v $G/tnca_fma_alignp_xls.v \
    $G/tnca_fma_norm_xls.v $G/tnca_fma_round_xls.v $here/ref/*.v > sources.sha256
echo "built CUTS=$CUTS latency $((1 + $(echo $CUTS | tr -cd 1 | wc -c)))"
