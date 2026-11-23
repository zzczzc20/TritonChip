#!/bin/bash
# usage: ks_dump2.sh TREE OUTNAME POINTID...   compile-only dump with private compile caches (one dir per point)
TREE=$1; OUT=$2; shift 2
cd $TREE || exit 2
rm -rf python/tnca/data
source $HOME/tc_env/env.sh; export PYTHONPATH=$PWD:$PWD/python:$PWD/compiler/python:$PWD/tools/asm:$HOME/r42/kfuse-speed_out
PY=$HOME/tc_env/torchenv/bin/python
D=$HOME/r42/kfuse-speed_out/dump2/$OUT
if [[ "$TREE" == *xltree* ]]; then RUNNER=$HOME/r42/kfuse-speed_out/ks_runpts.py; else RUNNER="-m archbench.run"; fi
mkdir -p $D
for ID in "$@"; do
  TAG=$(echo "$ID" | tr '/' '_')
  rm -rf $D/$TAG; mkdir -p $D/$TAG/triton $D/$TAG/inductor
  TNCA_COMPILE_CACHE_DIR=$D/$TAG/tnca_cc TRITON_CACHE_DIR=$D/$TAG/triton TORCHINDUCTOR_CACHE_DIR=$D/$TAG/inductor TNCA_KERNEL_DUMP=1 TNCA_DUMP_DIR=$D/$TAG/dump \
    nice -n 19 $PY $RUNNER --platform tnca --profile core-vrf-sync-v5-x8-g2c4-dp --point "$ID" --compile-only --out $D/$TAG.jsonl > $D/$TAG.log 2>&1
  echo "$ID rc=$?"
done
