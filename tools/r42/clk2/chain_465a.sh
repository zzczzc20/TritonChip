#!/bin/bash
# GB10 chain: FMA_IMPL=465 reference on the base tree, identity at defaults and single/combined bits on the clk2 tree
R=~/r42/clk2/tools/r42/clk2/run_one.sh; B=~/r42/clk2base; T=~/r42/clk2; O=~/r42/clk2_out/g2
bash $R $B sim/build_base465 $O/base465.log R41_RUN2_FMA_IMPL=465
for M in 8 16 32; do bash $R $T sim/build_c2_465_$M $O/c2_465_m$M.log R41_RUN2_FMA_IMPL=465 R42_CLKFIX2=$M; done
