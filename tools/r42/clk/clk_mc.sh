#!/bin/bash
# clk_mc.sh TREE IMAGE PROFILE OUT POINTS_FILE JOBS   (GB10) -- the points of POINTS_FILE (one ArchBench point id per line) on the three-core board-top RTL stand-in at an image's geometry
# (tools/r41/link_budget_rtl.py), like ~/r42/r42_mc.sh but for a point list. Results ~/r42/mc/OUT/<point>.jsonl ; nice 19.
TREE=$1; IMAGE=$2; PROFILE=$3; OUT=$4; PTS=$5; JOBS=$6; shift 6
cd ~/r42/$TREE || exit 2
rm -rf python/tnca/data
source ~/tc_env/env.sh; export PYTHONPATH=$PWD/python:$PWD/compiler/python:$PWD/tools/asm
PY=$HOME/tc_env/torchenv/bin/python
D=~/r42/mc/$OUT; mkdir -p $D
cp $PTS $D/points.txt
run1() { p=$1; f=$(echo $p | tr '/' '_'); [ -s $D/$f.jsonl ] && exit 0
  nice -n 19 $PY tools/r41/link_budget_rtl.py --point $p --out $D/$f.jsonl --image $IMAGE --profile $PROFILE --memory-bytes $((8<<20)) > $D/$f.log 2>&1; echo "$p exit $?"; }
export -f run1; export D PY IMAGE PROFILE
cat $D/points.txt | xargs -P $JOBS -I{} bash -c 'run1 {}'
echo DONE
