#!/bin/bash
# union arms A and Z at core level (GB10): core-run2 at FMA_IMPL 465 with CLKFIX 383 and CLKFIX2 255 (A) / 967 (Z), against the base tree's FMA_IMPL 465 run
T=~/r42/union; O=~/r42/clk2_out/gUA; mkdir -p $O
R=~/r42/clk2/tools/r42/clk2/run_one.sh
bash $R $T sim/build_ua_A $O/ua_A.log R41_RUN2_FMA_IMPL=465 R42_CLKFIX=383 R42_CLKFIX2=255
bash $R $T sim/build_ua_Z $O/ua_Z.log R41_RUN2_FMA_IMPL=465 R42_CLKFIX=383 R42_CLKFIX2=967
