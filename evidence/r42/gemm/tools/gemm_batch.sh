#!/bin/bash
# usage: batch.sh TREEDIR OUTNAME "ENV..." POINT...   sequential points; waits while load is high
TREE=$1; OUT=$2; ENVS=$3; shift 3
for p in "$@"; do
  while [ "$(cut -d' ' -f1 /proc/loadavg | cut -d. -f1)" -ge 13 ]; do sleep 20; done
  env $ENVS ~/r42/gemm_out/pt.sh $TREE $OUT $p
done
echo BATCHDONE
