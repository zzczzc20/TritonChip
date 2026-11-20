#!/bin/bash
# usage: pt.sh TREEDIR OUTNAME POINT [VAR=VAL ...]   single ArchBench point, single-core RTL sim, counters + dump
TREE=$1; OUT=$2; POINT=$3; shift 3
for kv in "$@"; do export "$kv"; done
cd $TREE || exit 2
rm -rf python/tnca/data
source ~/tc_env/env.sh
export PYTHONPATH=$PWD/python:$PWD/compiler/python:$PWD/tools/asm
export PYTHONDONTWRITEBYTECODE=1
PY=$HOME/tc_env/torchenv/bin/python
PROFILE=${PROFILE:-core-vrf-sync-v5-x8-g2c4-dp}
O=$HOME/r42/gemm_out/$OUT; mkdir -p $O
tag=$(echo $POINT | tr '/' '_')
rm -f $O/$tag.jsonl
if [ -n "$DUMP" ]; then
  rm -rf $O/dump_$tag; mkdir -p $O/dump_$tag
  TNCA_KERNEL_DUMP=1 TNCA_DUMP_DIR=$O/dump_$tag nice -n 19 $PY -m archbench.run --platform tnca --profile $PROFILE --point $POINT --compile-only --out $O/dump_$tag/cov.jsonl > $O/dump_$tag/log 2>&1
fi
if [ -n "$PROBE" ]; then
  export TNCA_DEVICE=$HOME/r42/gemm_out/probe/device.vvp
  export TNCA_DEVICE_PLUSARGS="+gprobe=$O/$tag.probe"
  [ "$PROBE" = 2 ] && export TNCA_DEVICE_PLUSARGS="$TNCA_DEVICE_PLUSARGS +gprobe_ev"
fi
nice -n 19 $PY -m archbench.run --platform tnca --profile $PROFILE --point $POINT --counters --out $O/$tag.jsonl > $O/$tag.log 2>&1
echo DONE $? >> $O/$tag.log
