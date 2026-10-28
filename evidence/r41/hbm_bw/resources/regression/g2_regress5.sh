#!/bin/bash
# R41 G2: default-build regression (all new parameters at 0) on GB10.
cd "$(dirname "$0")"
source ~/tc_env/env.sh >/dev/null 2>&1
out=g2_regress5; mkdir -p $out
echo "commit $(cat COMMIT) start $(date -u +%FT%TZ) host $(hostname)" > $out/summary.txt
run() { name=$1; shift; echo "== $name: $*" >> $out/summary.txt
  ( time "$@" ) > $out/$name.log 2>&1; rc=$?; echo "$name exit $rc" >> $out/summary.txt; }
run lint make lint
run top make top
run mesh make mesh
run grid_dispatch make r40-grid-dispatch
run coherent_host make r40-coherent-host
run multicore_shell make r40-multicore-shell
run line_axi make r40-line-axi
run compiler make compiler
echo "end $(date -u +%FT%TZ)" >> $out/summary.txt
