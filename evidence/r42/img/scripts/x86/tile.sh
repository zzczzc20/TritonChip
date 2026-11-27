#!/bin/bash
# tile.sh DIR... : routed tile screen summary of r42img runs (tile0)
for d in "$@"; do
  t=$HOME/r42img/$d/tile0; [ -f $t/utilization_routed.rpt ] || { echo "$d: no routed report yet"; continue; }
  clb=$(grep -m1 -E "^\| CLB  " $t/utilization_routed.rpt | awk -F'|' '{gsub(/ /,"",$3); print $3}')
  lutl=$(grep -m1 -E "^\| LUT as Logic" $t/utilization_routed.rpt | awk -F'|' '{gsub(/ /,"",$3); print $3}')
  lutm=$(grep -m1 -E "^\| LUT as Memory" $t/utilization_routed.rpt | awk -F'|' '{gsub(/ /,"",$3); print $3}')
  ff=$(grep -m1 -E "^\| CLB Registers" $t/utilization_routed.rpt | awk -F'|' '{gsub(/ /,"",$3); print $3}')
  dsp=$(grep -m1 -E "^\| DSPs" $t/utilization_routed.rpt | awk -F'|' '{gsub(/ /,"",$3); print $3}')
  bram=$(grep -m1 -E "^\| Block RAM Tile" $t/utilization_routed.rpt | awk -F'|' '{gsub(/ /,"",$3); print $3}')
  wns=$(grep -m1 -A2 "WNS(ns)" $t/timing_routed.rpt | tail -1 | awk '{print $1}')
  whs=$(grep -m1 -A2 "WNS(ns)" $t/timing_routed.rpt | tail -1 | awk '{print $5}')
  fail=$(tail -n 1 $t/../tile0.log | head -1); nf=$(wc -l < $t/failing_endpoints.tsv)
  echo "$d CLB=$clb LUTlogic=$lutl LUTmem=$lutm FF=$ff DSP=$dsp BRAM=$bram WNS=$wns WHS=$whs failing_endpoints_lines=$nf"
done
