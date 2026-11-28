#!/bin/bash
R=~/r42/clk2/tools/r42/clk2/run_one.sh; T=~/r42/clk2; O=~/r42/clk2_out/g3
mkdir -p $O
bash $R $T sim/build_n1_0 $O/n1_m0.log
for M in 71 32; do bash $R $T sim/build_n1_$M $O/n1_m$M.log R42_CLKFIX2=$M; done
