#!/bin/bash
# R41 G2 regression for AXI_LINE_LUTRAM (and C1+C2 together) on GB10.
cd "$(dirname "$0")"
source ~/tc_env/env.sh >/dev/null 2>&1
out=g2_regress2; mkdir -p $out
echo "commit $(cat COMMIT) start $(date -u +%FT%TZ) host $(hostname)" > $out/summary.txt
run() { name=$1; shift; echo "== $name: $*" >> $out/summary.txt
  ( time "$@" ) > $out/$name.log 2>&1; rc=$?; echo "$name exit $rc" >> $out/summary.txt; }
run line_axi_default python3 tools/r40/line_axi_tests.py
run line_axi_lutram python3 tools/r40/line_axi_tests.py --line-lutram
run multicore_axi python3 tools/r40/multicore_axi_tests.py
run multicore_axi_3core python3 tools/r40/multicore_axi_tests.py --cores 3 --write-delay 300
run image_geometry_c1_c2 python3 tools/r40/image_geometry_tests.py --image-tcl images/image_c1_c2.tcl
echo "end $(date -u +%FT%TZ)" >> $out/summary.txt
