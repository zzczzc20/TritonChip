#!/bin/bash
# R41 G2 regression for NUCA_SHARED_DATA_READ=2 (and with AXI_LINE_LUTRAM) on GB10.
cd "$(dirname "$0")"
source ~/tc_env/env.sh >/dev/null 2>&1
out=g2_regress4; mkdir -p $out
echo "commit $(cat COMMIT) start $(date -u +%FT%TZ) host $(hostname)" > $out/summary.txt
run() { name=$1; shift; echo "== $name: $*" >> $out/summary.txt
  ( time "$@" ) > $out/$name.log 2>&1; rc=$?; echo "$name exit $rc" >> $out/summary.txt; }
run nuca make nuca
run nuca_dco make nuca-dco
run nuca_shared_read make nuca-shared-read
run image_geometry_shared_read2 python3 tools/r40/image_geometry_tests.py --image-tcl images/image_shared_read2.tcl
run image_geometry_c1b_c2 python3 tools/r40/image_geometry_tests.py --image-tcl images/image_c1b_c2.tcl
echo "end $(date -u +%FT%TZ)" >> $out/summary.txt
