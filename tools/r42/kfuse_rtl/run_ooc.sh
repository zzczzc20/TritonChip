#!/bin/bash
# run_ooc.sh NAME TOP GENERICS...   (on the x86 host, from the tree root ~/r42_kfuse_rtl)
# One out-of-context synthesis (SYNTH_ONLY=1) or synthesis + place + route of a screen top at PERIOD (default 10 ns),
# 6 threads, nice 5, results in $OUTROOT/NAME (default ~/r42_kfuse_out). Refuses to start when two Vivado jobs run.
set -u
NAME=$1; TOP=$2; shift 2
source ~/x86_link_env.sh
ROOT=$(cd $(dirname $0)/../../.. && pwd)
OUTROOT=${OUTROOT:-$HOME/r42_kfuse_out}
PERIOD=${PERIOD:-10}
OUT=$OUTROOT/$NAME
n=$(pgrep -fc 'unwrapped/lnx64.o/vivado -mode batch' || true)
if [ "${n:-0}" -ge 2 ]; then echo "two Vivado jobs already running, not starting $NAME"; exit 3; fi
rm -rf $OUT; mkdir -p $OUT
R=$ROOT/rtl
FILES="$R/core/tnca_kfuse_lane.v $ROOT/tools/r42/kfuse_rtl/ooc_tops.v $R/gen/tnca_mac_b0_xls.v $R/gen/tnca_fmad_norm_xls.v $R/gen/tnca_fma_round_xls.v"
case "$TOP" in kfuse_ooc_mac*)
    FILES="$FILES $R/core/tnca_mac_lane.v $R/gen/tnca_widen_xls.v $R/gen/tnca_mac_s1_xls.v $R/gen/tnca_mac_s2_xls.v $R/gen/tnca_mac_f2_xls.v";;
esac
echo "$(date -u +%FT%TZ) start $NAME top=$TOP generics=$* period=$PERIOD synth_only=${SYNTH_ONLY:-0}" >> $OUTROOT/run.log
( cd $OUT && OOC_GENERICS="$*" SYNTH_ONLY=${SYNTH_ONLY:-0} FLATTEN=${FLATTEN:-} nice -n 5 vivado -mode batch -nojournal -nolog \
    -source $ROOT/tools/r42/kfuse_rtl/ooc.tcl -tclargs $TOP $PERIOD xcu280-fsvh2892-2L-e $OUT 6 $FILES > $OUT/vivado.log 2>&1 )
echo "$(date -u +%FT%TZ) end $NAME exit $?" >> $OUTROOT/run.log
