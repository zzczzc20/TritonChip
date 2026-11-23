#!/bin/bash
# usage: eval_point.sh TREE OUTDIR POINTID...   per point: kernel dump (compile-only) and a single-core RTL run with counters and
# the issue probe, in parallel. Results: OUTDIR/<tag>/ (dump/, run.log), OUTDIR/rtl/<tag>.jsonl, OUTDIR/rtl/<tag>.iprobe.
# Domain: single-core RTL simulation (a screen), profile core-vrf-sync-v5-x8-g2c4-dp, default knobs unless the environment sets some.
TREE=$1; OUT=$2; shift 2
R=$TREE/tools/r42/fp8sw
PROBE=${PROBE_DEVICE:-$HOME/r42/fp8sw_out/probe/device.vvp}
mkdir -p $OUT/rtl
for ID in "$@"; do
  TAG=$(echo "$ID" | tr '/' '_')
  bash $R/dump_point.sh $TREE $OUT $ID &
  TNCA_DEVICE=$PROBE TNCA_DEVICE_PLUSARGS="+iprobe=$OUT/rtl/$TAG.iprobe" bash $R/rtl_point.sh $TREE $OUT/rtl $ID --counters &
done
wait
