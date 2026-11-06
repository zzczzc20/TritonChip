#!/bin/bash
# R41 G2 approved options: regressions on GB10 (one job, <= 6 cores).
cd "$(dirname "$0")"
source ~/tc_env/env.sh >/dev/null 2>&1
out=g2_regress6; mkdir -p $out
echo "commit $(cat COMMIT) start $(date -u +%FT%TZ) host $(hostname)" > $out/summary.txt
run() { name=$1; shift; echo "== $name: $*" >> $out/summary.txt
  ( time "$@" ) > $out/$name.log 2>&1; rc=$?; echo "$name exit $rc" >> $out/summary.txt; }
run nuca_shared_read make nuca-shared-read
run nuca_addr_fold make nuca-addr-fold
run nuca make nuca
run nuca_dco make nuca-dco
run core_g2 make core-g2
run profile_tests make r1-profile-tests
run top make top
run mesh make mesh
run line_axi make r40-line-axi
run cache_maintenance make r40-cache-maintenance
for i in image image_g2_ctx2 image_g2_addr35 image_g2_bar8_epoch38 image_g2_all; do
  run geometry_$i python3 tools/r40/image_geometry_tests.py --image-tcl images/$i.tcl
done
run tnca_opt_cmake cmake -S compiler -B compiler/build -G Ninja -DMLIR_DIR=$HOME/tc_env/root/usr/lib/llvm-20/lib/cmake/mlir -DLLVM_DIR=$HOME/tc_env/root/usr/lib/llvm-20/lib/cmake/llvm -DCMAKE_BUILD_TYPE=Release
run tnca_opt_build ninja -j6 -C compiler/build
run compiler make compiler
echo "end $(date -u +%FT%TZ)" >> $out/summary.txt
