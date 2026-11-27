#!/bin/bash
# run_ooc.sh NAME TOP GENERICS...   (x86 host, from the tree root ~/r42_area)
# One out-of-context synthesis (SYNTH_ONLY=1) or synthesis + place + route of a screen top at PERIOD (default 10 ns),
# 6 threads, nice 5, results in $OUTROOT/NAME (default ~/r42_area_out). Refuses to start when two Vivado jobs run.
# FILES (env) overrides the source list; default = the whole area file set below.
set -u
NAME=$1; TOP=$2; shift 2
source ~/x86_link_env.sh
ROOT=$(cd $(dirname $0)/../../.. && pwd)
OUTROOT=${OUTROOT:-$HOME/r42_area_out}
PERIOD=${PERIOD:-10}
OUT=$OUTROOT/$NAME
mkdir -p $OUTROOT
n=$(pgrep -fc 'unwrapped/lnx64.o/vivado -mode batch' || true)
if [ "${n:-0}" -ge 2 ]; then echo "two Vivado jobs already running, not starting $NAME"; exit 3; fi
rm -rf $OUT; mkdir -p $OUT
R=$ROOT/rtl
DEFAULT_FILES="$ROOT/tools/r42/area/ooc_tops.v $R/core/tnca_fp32_fma.v $R/core/tnca_mac_lane.v $R/core/tnca_fpu_multi.v \
 $R/gen/tnca_fma_align_xls.v $R/gen/tnca_fma_norm_xls.v $R/gen/tnca_fma_round_xls.v $R/gen/tnca_fmad_norm_xls.v \
 $R/gen/tnca_fma_l3_xls.v $R/gen/tnca_fmad_l3_xls.v $R/gen/tnca_mac_b0_xls.v $R/gen/tnca_widen_xls.v $R/gen/tnca_mac_s1_xls.v \
 $R/gen/tnca_mac_s2_xls.v $R/gen/tnca_mac_f2_xls.v $R/gen/tnca_mac_f1_xls.v $R/gen/tnca_fma_alignp_xls.v $R/core/tnca_lzc81.v $R/core/tnca_fmax_norm.v $R/core/tnca_fmax_round.v $R/core/tnca_fmax_mul.v $R/core/tnca_fmax_alignp.v $R/core/tnca_fmax_align.v $R/core/tnca_sfu.v $R/gen/tnca_sfu_exp2_xls.v $R/gen/tnca_sfu_log2_xls.v $R/gen/tnca_sfu_rcp_xls.v $R/gen/tnca_sfu_rsqrt_xls.v $R/gen/tnca_cvt_comb_xls.v"
FILES=${FILES:-$DEFAULT_FILES}
echo "$(date -u +%FT%TZ) start $NAME top=$TOP generics=$* period=$PERIOD synth_only=${SYNTH_ONLY:-0}" >> $OUTROOT/run.log
( cd $OUT && OOC_GENERICS="$*" INCDIRS=$ROOT/rtl/common SYNTH_ONLY=${SYNTH_ONLY:-0} FLATTEN=${FLATTEN:-} nice -n 5 vivado -mode batch -nojournal -nolog \
    -source $ROOT/tools/r42/area/ooc.tcl -tclargs $TOP $PERIOD xcu280-fsvh2892-2L-e $OUT 6 $FILES > $OUT/vivado.log 2>&1 )
echo "$(date -u +%FT%TZ) end $NAME exit $?" >> $OUTROOT/run.log
