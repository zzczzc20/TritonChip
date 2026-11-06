#!/bin/bash
# R41 G2: the two-context board top with its profile, the default board top
# after the runtime change, and make compiler with tnca-opt and FileCheck.
cd "$(dirname "$0")"
source ~/tc_env/env.sh >/dev/null 2>&1
out=g2_regress7; mkdir -p $out
echo "commit $(cat COMMIT) start $(date -u +%FT%TZ) host $(hostname)" > $out/summary.txt
run() { name=$1; shift; echo "== $name: $*" >> $out/summary.txt
  ( time "$@" ) > $out/$name.log 2>&1; rc=$?; echo "$name exit $rc" >> $out/summary.txt; }
run geometry_image python3 tools/r40/image_geometry_tests.py --image-tcl images/image.tcl
for i in image_g2_ctx2 image_g2_all; do
  run geometry_$i python3 tools/r40/image_geometry_tests.py --image-tcl images/$i.tcl --image-profile core-vrf-sync-v5-x8-g2
done
run mmio_checks make r40-multicore-mmio-checks
run tnca_opt_cmake cmake -S compiler -B compiler/build -G Ninja -DMLIR_DIR=$HOME/tc_env/root/usr/lib/llvm-20/lib/cmake/mlir -DLLVM_DIR=$HOME/tc_env/root/usr/lib/llvm-20/lib/cmake/llvm -DCMAKE_BUILD_TYPE=Release
run tnca_opt_build ninja -j6 -C compiler/build
TNCA_LLVM=$HOME/tc_env/root/usr/lib/llvm-20 R1_STRICT=1 run compiler make compiler
echo "end $(date -u +%FT%TZ)" >> $out/summary.txt
