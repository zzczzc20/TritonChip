#!/bin/bash
# util.sh DIR... : kernel synthesis summary (hierarchical utilization) of r42img runs
for d in "$@"; do
  f=$HOME/r42img/$d/out/utilization.rpt; [ -f $f ] || { echo "$d: no report"; continue; }
  echo "== $d"
  grep -E '^\| +(tnca_multicore_hbm_xo|u_system|g_tile\[[0-9]\]\.u_tile|u_core|g_core|u_mtx|u_txf) ' $f | head -3 | cut -c1-210 | sed -n '1p;3,4p'
  grep -E 'g_tile\[[0-9]\]\.u_tile  ' $f | cut -c1-210
done
