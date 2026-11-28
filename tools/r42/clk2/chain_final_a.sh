#!/bin/bash
# final gates at the commit under test, FMA_IMPL 465: default identity against base465, then the combinations
R=~/r42/clk2/tools/r42/clk2/run_one.sh; T=~/r42/clk2; O=~/r42/clk2_out/gF; mkdir -p $O
bash $R $T sim/build_f465_0 $O/f465_m0.log R41_RUN2_FMA_IMPL=465
for M in 127 56 7; do bash $R $T sim/build_f465_$M $O/f465_m$M.log R41_RUN2_FMA_IMPL=465 R42_CLKFIX2=$M; done
