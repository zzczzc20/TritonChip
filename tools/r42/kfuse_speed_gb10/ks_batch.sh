#!/bin/bash
# usage: ks_batch.sh IDFILE TAG VARIANT...   runs the variants one after the other with the patched binary
IDF=$1; TAG=$2; shift 2
for V in "$@"; do
  $HOME/r42/kfuse-speed_out/ks_tlm_list_b2.sh $HOME/r42/kfuse-speed_out/cfg/$V.cfg ${TAG}_$V $IDF
  echo "VARIANT $V done" 
done
echo BATCH_DONE
