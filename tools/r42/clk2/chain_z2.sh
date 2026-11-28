#!/bin/bash
# after chain_z: the transfer-engine bit at the default machine (cycle identity expected) and the arm-Z mask again on the final RTL
R=~/r42/clk2/tools/r42/clk2/run_one.sh; T=~/r42/clk2; O=~/r42/clk2_out/gZ; mkdir -p $O
until grep -q EXIT $O/z_m511.log 2>/dev/null; do sleep 20; done
bash $R $T sim/build_z_64 $O/z_m64_fma1.log R42_CLKFIX2=64
bash $R $T sim/build_z_967b $O/z_m967b.log R41_RUN2_FMA_IMPL=465 R42_CLKFIX2=967
