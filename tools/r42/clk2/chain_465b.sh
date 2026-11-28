#!/bin/bash
R=~/r42/clk2/tools/r42/clk2/run_one.sh; T=~/r42/clk2; O=~/r42/clk2_out/g2
for M in 63 56 24; do bash $R $T sim/build_c2_465_$M $O/c2_465_m$M.log R41_RUN2_FMA_IMPL=465 R42_CLKFIX2=$M; done
