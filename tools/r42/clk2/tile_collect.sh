#!/bin/bash
# tile_collect.sh S RUN [RUN...]   S = scratchpad holding the vx helper. Pulls the small results of ~/r42_clk2_out/RUN (a tile screen) into
# $S/clk2/RUN/ (endpoints.csv, top_paths_routed.rpt, timing_routed.rpt, utilization_routed.rpt) and prints the per-unit census.
S=$1; shift
HERE=$(cd $(dirname $0) && pwd)
for R in "$@"; do
  D=$S/clk2/$R; mkdir -p $D
  for f in endpoints.csv top_paths_routed.rpt timing_routed.rpt utilization_routed.rpt utilization_synth.rpt; do
    $S/vx "cat ~/r42_clk2_out/$R/$f" < /dev/null > $D/$f 2>/dev/null
  done
  grep -A4 "Design Timing Summary" $D/timing_routed.rpt | tail -2
  python3 -I $HERE/tile_census.py $D/endpoints.csv ${PERIOD:-5.6} 2
done
