#!/bin/bash
# core-run2 at FMA_IMPL 465 for the zero-cycle bits and the arm-Z mask (GB10)
R=~/r42/clk2/tools/r42/clk2/run_one.sh; T=~/r42/clk2; O=~/r42/clk2_out/gZ; mkdir -p $O
for M in 967 256 511; do bash $R $T sim/build_z_$M $O/z_m$M.log R41_RUN2_FMA_IMPL=465 R42_CLKFIX2=$M; done
