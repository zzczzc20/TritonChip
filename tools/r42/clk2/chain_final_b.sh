#!/bin/bash
# final gates at the commit under test, FMA_IMPL 1 (the default machine): default identity against the base tree, then the options that do not need the area FMA
R=~/r42/clk2/tools/r42/clk2/run_one.sh; T=~/r42/clk2; O=~/r42/clk2_out/gF; mkdir -p $O
bash $R $T sim/build_f1_0 $O/f1_m0.log
for M in 71 64 1; do bash $R $T sim/build_f1_$M $O/f1_m$M.log R42_CLKFIX2=$M; done
