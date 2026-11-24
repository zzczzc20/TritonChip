#!/bin/bash
# usage: xf_mc3.sh TREE OUTNAME "TNCA_KEY=VAL ..." POINT...   three-core board-top RTL stand-in at the R41 run-2 image geometry (RTL simulation, never board)
# link_budget_rtl.py deletes every TNCA_* environment variable, so settings go through its --set option.
TREE=$1; OUT=$2; ENVS=$3; shift 3
cd $TREE || exit 2
rm -rf python/tnca/data
source ~/tc_env/env.sh; export PYTHONPATH=$PWD/python:$PWD/compiler/python:$PWD/tools/asm PYTHONDONTWRITEBYTECODE=1
SETS=""; for kv in $ENVS; do SETS="$SETS --set $kv"; done
PY=$HOME/tc_env/torchenv/bin/python
D=~/r42/xfer_out/mc/$OUT; mkdir -p $D
for p in "$@"; do f=$(echo $p | tr '/' '_')
  nice -n 19 $PY tools/r41/link_budget_rtl.py --point $p --out $D/$f.jsonl --image /home/zhongchun/r42/images/r41_run2_sim.tcl --profile core-vrf-sync-v5-x8-g2c4-dp --memory-bytes $((8<<20)) $SETS > $D/$f.log 2>&1; echo "$p exit $?"
done
echo MCDONE
