#!/bin/bash
TREE=$HOME/r42/dse; BIN=$HOME/r42/dse_out/tnca-chip-v5; IDF=$HOME/r42/dse_out/ids/tiny27.txt
R=$HOME/r42/clk2/tools/r42/clk2/tlm_run.sh
for pair in "base:$HOME/r42/shrink_out/cfg/dse_base.cfg" "c2:$HOME/r42/clk2_out/tlmcfg/dse_c2.cfg" "c2fp:$HOME/r42/clk2_out/tlmcfg/dse_c2fp.cfg" "c2mx:$HOME/r42/clk2_out/tlmcfg/dse_c2mx.cfg"; do
  n=${pair%%:*}; c=${pair#*:}
  JOBS=3 bash $R $TREE $c tiny27_$n $BIN $IDF
  echo "done $n" >> $HOME/r42/clk2_out/tlm_chain.log
done
