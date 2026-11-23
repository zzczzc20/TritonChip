#!/bin/bash
# usage: arch_pt.sh OUTDIR POINT   one ArchBench point, single-core RTL sim (probe device), counters + issue log
OUTDIR=$1; POINT=$2
cd ~/r42/arch || exit 2
rm -rf python/tnca/data
source ~/tc_env/env.sh
export PYTHONPATH=$PWD/python:$PWD/compiler/python:$PWD/tools/asm PYTHONDONTWRITEBYTECODE=1
PY=$HOME/tc_env/torchenv/bin/python
PROFILE=core-vrf-sync-v5-x8-g2c4-dp
O=$HOME/r42/arch_out/$OUTDIR; mkdir -p $O
tag=$(echo $POINT | tr '/' '_')
rm -f $O/$tag.jsonl $O/$tag.probe
export TNCA_DEVICE=$HOME/r42/arch_out/probe/device.vvp
export TNCA_DEVICE_PLUSARGS="+aprobe=$O/$tag.probe"
nice -n 19 $PY -m archbench.run --platform tnca --profile $PROFILE --point $POINT --counters --out $O/$tag.jsonl > $O/$tag.log 2>&1
echo "$POINT DONE $?" >> $HOME/r42/arch_out/$OUTDIR.done
