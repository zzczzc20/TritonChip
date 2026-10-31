#!/bin/bash
# build.sh CUTS OUTDIR -- Verilate mac_tb (tnca_mac_lane at CUTS beside the
# current lane) into OUTDIR. Sources from this tree.
set -e
CUTS=${1:?cuts, e.g. 1110}; OUT=${2:?outdir}
here=$(cd $(dirname $0) && pwd)
root=$(cd $here/../../../../.. && pwd)
mkdir -p $OUT; cd $OUT
G=$root/rtl/gen
SRC="$root/rtl/core/tnca_mac_lane.v $G/tnca_widen_xls.v $G/tnca_mac_s1_xls.v $G/tnca_mac_s2_xls.v $G/tnca_mac_f2_xls.v
     $G/tnca_mac_b0_xls.v $G/tnca_fmad_norm_xls.v $G/tnca_fma_round_xls.v"
verilator --cc --exe --build -O3 -Wno-fatal -Wno-lint -Wno-style --top-module mac_tb \
    -GCUTS=5\'b$CUTS $here/mac_tb.v $SRC $here/ref/*.v $here/mac_check.cpp -j 4 \
    > build.log 2>&1 || { tail -30 build.log; exit 2; }
sha256sum $here/mac_tb.v $here/mac_check.cpp $SRC $here/ref/*.v > sources.sha256
echo "built CUTS=$CUTS latency $((1 + $(echo $CUTS | tr -cd 1 | wc -c)))"
