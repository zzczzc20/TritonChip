#!/bin/bash
# units.sh DIR... : tile-0 core, matrix unit, transfer engine rows of the kernel synthesis (Total LUTs, FFs, LUTRAM, DSP)
for d in "$@"; do
  f=$HOME/r42img/$d/out/utilization.rpt
  echo "== $d"
  awk -F'|' '/^\|             u_core  /{c++} c==1 && /^\|               u_mtx  /{print "u_mtx", $4, $5, $6, $8, $12} c==1 && /^\|               u_txf  /{print "u_txf", $4, $5, $6, $8, $12} c==1 && /^\|             u_core  /{print "u_core", $4, $5, $6, $8, $12} ' $f | head -3
  grep -E "^\| tnca_multicore_hbm_xo " $f | head -1 | awk -F'|' '{print "kernel", $4, $5, $6, $8, $12}'
done
