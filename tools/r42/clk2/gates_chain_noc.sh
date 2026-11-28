#!/bin/bash
R=$1; O=$2
bash $R/tools/r42/clk2/gates_noc.sh $R $O/noc > $O/noc.log 2>&1
bash $R/tools/r42/clk2/gates_top.sh $R $O/top > $O/top.log 2>&1
echo CHAIN_DONE >> $O/top.log
