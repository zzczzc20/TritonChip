#!/bin/bash
# usage: run_xp_tag.sh TREE OUTFILE  (on GB10)  the transfer engine alone behind a fixed-latency memory (rtl-eng): word-walk tags 16 against 32
# (TENSOR_TAG_W 4 and 5; the image's 64 line slots), both with half-line rows, one transposed weight tile and an A tile per step.
source $HOME/tc_env/env.sh; cd $1 || exit 2
OUT=$2; : > $OUT; T=$(mktemp -d)
for lat in 12 24 40 60; do
 for cfg in "16 0 0" "32 0 0" "16 0 1" "32 0 1" "16 2 1" "32 1 0"; do
  set -- $cfg; KS=$1; XB=$2; XM=$3
  for TG in 4 5; do
   for XN in 4 8; do
    nice -n 19 iverilog -g2012 -Irtl/common -s tb_xfer_perf -Ptb_xfer_perf.XP=1 -Ptb_xfer_perf.KS=$KS -Ptb_xfer_perf.XB=$XB -Ptb_xfer_perf.XM=$XM -Ptb_xfer_perf.XN=$XN \
      -Ptb_xfer_perf.FIXLAT=$lat -Ptb_xfer_perf.VW_STALL=0 -Ptb_xfer_perf.REQ_STALL=0 -Ptb_xfer_perf.LHALF=1 -Ptb_xfer_perf.LSW=6 -Ptb_xfer_perf.TGW=$TG -o $T/xp.vvp tb/core/tb_xfer_perf.v rtl/core/tnca_tensor.v 2>&1 | tail -2
    nice -n 19 vvp $T/xp.vvp | grep -E "XFER|FATAL|error" | sed "s/^XFER/XFER TGW=$TG/" >> $OUT
   done
  done
 done
done
rm -rf $T; echo DONE >> $OUT
