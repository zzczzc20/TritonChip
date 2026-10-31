#!/bin/bash
# R41 G2: make compiler (tnca-opt + FileCheck, torch environment) and the MMIO
# checks, on this tree and on the unchanged base tree, same environment.
cd "$(dirname "$0")"
source ~/tc_env/env.sh >/dev/null 2>&1
out=$PWD/g2_regress9; mkdir -p $out
echo "commit $(cat COMMIT) start $(date -u +%FT%TZ) host $(hostname)" > $out/summary.txt
run() { name=$1; shift; echo "== $name: $*" >> $out/summary.txt
  ( time "$@" ) > $out/$name.log 2>&1; rc=$?; echo "$name exit $rc" >> $out/summary.txt; }
P=$HOME/tc_env/torchenv/bin/python
run mmio_checks make r40-multicore-mmio-checks PYTHON=$P TRITON_PYTHON=$P
run compiler env TNCA_LLVM=$HOME/tc_env/root/usr/lib/llvm-20 make compiler PYTHON=$P
B=$HOME/r41bw_aud/src_92b2c85b1518
mkdir -p $B/compiler/build; cp -r compiler/build/. $B/compiler/build/ 2>/dev/null
( cd $B && run base_compiler env TNCA_LLVM=$HOME/tc_env/root/usr/lib/llvm-20 make compiler PYTHON=$P )
( cd $B && run base_mmio_checks make r40-multicore-mmio-checks PYTHON=$P TRITON_PYTHON=$P )
echo "base tree: $B (commit 92b2c85, before the G2 work)" >> $out/summary.txt
echo "end $(date -u +%FT%TZ)" >> $out/summary.txt
