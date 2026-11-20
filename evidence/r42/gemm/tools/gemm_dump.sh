#!/bin/bash
# usage: dump.sh TREEDIR OUTNAME POINT [VAR=VAL ...]   compile-only kernel dump
TREE=$1; OUT=$2; POINT=$3; shift 3
for kv in "$@"; do export "$kv"; done
cd $TREE || exit 2
source ~/tc_env/env.sh
export PYTHONPATH=$PWD/python:$PWD/compiler/python:$PWD/tools/asm PYTHONDONTWRITEBYTECODE=1
PY=$HOME/tc_env/torchenv/bin/python
PROFILE=${PROFILE:-core-vrf-sync-v5-x8-g2c4-dp}
tag=$(echo $POINT | tr '/' '_')
O=$HOME/r42/gemm_out/$OUT/dump_$tag; rm -rf $O; mkdir -p $O
TNCA_KERNEL_DUMP=1 TNCA_DUMP_DIR=$O nice -n 19 $PY -m archbench.run --platform tnca --profile $PROFILE --point $POINT --compile-only --out $O/cov.jsonl > $O/log 2>&1
tail -2 $O/log; ls $O
