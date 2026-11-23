#!/bin/bash
# usage: xf_alldump.sh TREE OUTNAME IDFILE [VAR=VAL ...]   compile-only kernel dumps of every point in IDFILE (one process at a time)
TREE=$1; OUT=$2; IDF=$3; shift 3
for kv in "$@"; do export "$kv"; done
cd $TREE || exit 2
rm -rf python/tnca/data
source ~/tc_env/env.sh
export PYTHONPATH=$PWD/python:$PWD/compiler/python:$PWD/tools/asm PYTHONDONTWRITEBYTECODE=1
PY=$HOME/tc_env/torchenv/bin/python
PROFILE=${PROFILE:-core-vrf-sync-v5-x8-g2c4-dp}
O=$HOME/r42/xfer_out/$OUT; mkdir -p $O
for POINT in $(cat $IDF); do
  tag=$(echo $POINT | tr '/' '_')
  d=$O/$tag; rm -rf $d; mkdir -p $d
  TNCA_COMPILE_CACHE_DIR=$d/cache TNCA_KERNEL_DUMP=1 TNCA_DUMP_DIR=$d nice -n 19 $PY -m archbench.run --platform tnca --profile $PROFILE --point $POINT --compile-only --out $d/cov.jsonl > $d/log 2>&1
  rm -rf $d/cache
  echo "$POINT $(grep -c COMPILE_FAILED $d/log)" >> $O/summary.txt
done
echo ALLDUMP_DONE >> $O/summary.txt
