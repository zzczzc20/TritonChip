#!/bin/bash
# R41 G2: re-run of g2_regress7's two environment failures.
cd "$(dirname "$0")"
source ~/tc_env/env.sh >/dev/null 2>&1
out=g2_regress8; mkdir -p $out
echo "commit $(cat COMMIT) start $(date -u +%FT%TZ) host $(hostname)" > $out/summary.txt
run() { name=$1; shift; echo "== $name: $*" >> $out/summary.txt
  ( time "$@" ) > $out/$name.log 2>&1; rc=$?; echo "$name exit $rc" >> $out/summary.txt; }
run mmio_checks make r40-multicore-mmio-checks
run compiler env TNCA_LLVM=$HOME/tc_env/root/usr/lib/llvm-20 make compiler PYTHON=$HOME/tc_env/torchenv/bin/python
echo "end $(date -u +%FT%TZ)" >> $out/summary.txt
