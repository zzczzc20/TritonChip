#!/bin/bash
# usage: run_xp.sh OUTFILE   runs the micro-benchmark over patterns and latencies (RTL simulation, Icarus)
source $HOME/tc_env/env.sh; cd $HOME/r42/xfer || exit 2
T=$HOME/r42/xfer_out/tbx; OUT=$1; : > $OUT
for lat in 12 24 40 60; do
 for cfg in "16 0 1" "32 0 1" "16 1 1" "32 1 1" "16 0 2" "32 0 2" "16 1 2" "32 1 2" "16 0 4" "32 0 4" "16 1 4" "32 1 4"; do
  set -- $cfg; KS=$1; XB=$2; XM=$3
  for XN in 4 8; do
   nice -n 19 iverilog -g2012 -Irtl/common -s tb_xfer_perf -Ptb_xfer_perf.XP=1 -Ptb_xfer_perf.KS=$KS -Ptb_xfer_perf.XB=$XB -Ptb_xfer_perf.XM=$XM -Ptb_xfer_perf.XN=$XN \
     -Ptb_xfer_perf.FIXLAT=$lat -Ptb_xfer_perf.VW_STALL=0 -Ptb_xfer_perf.REQ_STALL=0 -o $T/xp_$$.vvp $T/tb_xfer_perf.v rtl/core/tnca_tensor.v 2>&1 | tail -2
   nice -n 19 vvp $T/xp_$$.vvp | grep -E "XFER|FATAL|error" >> $OUT
  done
 done
done
echo DONE >> $OUT
