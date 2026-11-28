#!/bin/bash
R=~/r42/clk2/tools/r42/clk2/run_one.sh; T=~/r42/clk2; O=~/r42/clk2_out/g3
mkdir -p $O
bash $R $T sim/build_n465_0 $O/n465_m0.log R41_RUN2_FMA_IMPL=465
for M in 63 127; do bash $R $T sim/build_n465_$M $O/n465_m$M.log R41_RUN2_FMA_IMPL=465 R42_CLKFIX2=$M; done
