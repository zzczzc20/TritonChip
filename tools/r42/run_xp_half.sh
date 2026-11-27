#!/bin/bash
# usage: run_xp_half.sh TREE OUTFILE  (on GB10)  the transfer engine alone behind a fixed-latency memory (rtl-eng): cycles per K step
# of one operand, word engine, line engine (LINE_HALF 0) and line engine with half-line rows (LINE_HALF 1). RTL simulation, Icarus.
source $HOME/tc_env/env.sh; cd $1 || exit 2
OUT=$2; : > $OUT; T=$(mktemp -d)
for lat in 12 24 40 60; do
 for cfg in "16 2 1" "32 2 1" "16 1 0" "32 1 0" "16 0 0" "32 0 0" "16 0 1" "32 0 1"; do
  set -- $cfg; KS=$1; XB=$2; XM=$3
  for LH in 0 1; do
   for XN in 4 8; do
    nice -n 19 iverilog -g2012 -Irtl/common -s tb_xfer_perf -Ptb_xfer_perf.XP=1 -Ptb_xfer_perf.KS=$KS -Ptb_xfer_perf.XB=$XB -Ptb_xfer_perf.XM=$XM -Ptb_xfer_perf.XN=$XN \
      -Ptb_xfer_perf.FIXLAT=$lat -Ptb_xfer_perf.VW_STALL=0 -Ptb_xfer_perf.REQ_STALL=0 -Ptb_xfer_perf.LHALF=$LH -o $T/xp.vvp tb/core/tb_xfer_perf.v rtl/core/tnca_tensor.v 2>&1 | tail -2
    nice -n 19 vvp $T/xp.vvp | grep -E "XFER|FATAL|error" | sed "s/^XFER/XFER LHALF=$LH/" >> $OUT
   done
  done
 done
done
rm -rf $T; echo DONE >> $OUT
