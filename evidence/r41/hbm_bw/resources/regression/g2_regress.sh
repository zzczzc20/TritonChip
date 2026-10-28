#!/bin/bash
# R41 G2 regression for NUCA_SHARED_DATA_READ on GB10. One job, <= 6 cores.
cd "$(dirname "$0")"
source ~/tc_env/env.sh >/dev/null 2>&1
out=g2_regress; mkdir -p $out
echo "commit $(cat COMMIT) start $(date -u +%FT%TZ) host $(hostname)" > $out/summary.txt
run() { name=$1; shift; echo "== $name: $*" >> $out/summary.txt
  ( time "$@" ) > $out/$name.log 2>&1; rc=$?; echo "$name exit $rc" >> $out/summary.txt; }
run nuca make nuca
run nuca_dco make nuca-dco
run nuca_shared_read make nuca-shared-read
run cache_maintenance make r40-cache-maintenance
run image_geometry_base python3 tools/r40/image_geometry_tests.py --image-tcl images/image.tcl
run image_geometry_shared_read python3 tools/r40/image_geometry_tests.py --image-tcl images/image_shared_read.tcl
echo "end $(date -u +%FT%TZ)" >> $out/summary.txt
