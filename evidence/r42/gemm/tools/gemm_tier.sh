#!/bin/bash
# usage: tier.sh TREEDIR OUTNAME TIER JOBS   whole ArchBench tier, single-core RTL sim, default knobs
TREE=$1; OUT=$2; TIER=$3; JOBS=$4
cd $TREE || exit 2
rm -rf python/tnca/data
source ~/tc_env/env.sh
export PYTHONPATH=$PWD/python:$PWD/compiler/python:$PWD/tools/asm PYTHONDONTWRITEBYTECODE=1
PY=$HOME/tc_env/torchenv/bin/python
rm -f ~/r42/gemm_out/$OUT.jsonl
nice -n 19 $PY -m archbench.run --platform tnca --profile core-vrf-sync-v5-x8-g2c4-dp --tier $TIER --jobs $JOBS --out ~/r42/gemm_out/$OUT.jsonl > ~/r42/gemm_out/$OUT.log 2>&1
echo TIERDONE $? >> ~/r42/gemm_out/$OUT.log
