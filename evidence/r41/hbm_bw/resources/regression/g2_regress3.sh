#!/bin/bash
# R41 G2: the board top at the exact image parameters, base and candidates.
cd "$(dirname "$0")"
source ~/tc_env/env.sh >/dev/null 2>&1
out=g2_regress3; mkdir -p $out
echo "commit $(cat COMMIT) start $(date -u +%FT%TZ) host $(hostname)" > $out/summary.txt
run() { name=$1; shift; echo "== $name: $*" >> $out/summary.txt
  ( time "$@" ) > $out/$name.log 2>&1; rc=$?; echo "$name exit $rc" >> $out/summary.txt; }
run image_geometry_base python3 tools/r40/image_geometry_tests.py --image-tcl images/image.tcl
run image_geometry_shared_read python3 tools/r40/image_geometry_tests.py --image-tcl images/image_shared_read.tcl
run image_geometry_line_lutram python3 tools/r40/image_geometry_tests.py --image-tcl images/image_line_lutram.tcl
run image_geometry_c1_c2 python3 tools/r40/image_geometry_tests.py --image-tcl images/image_c1_c2.tcl
echo "end $(date -u +%FT%TZ)" >> $out/summary.txt
