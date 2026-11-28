#!/bin/bash
# iss_mc.sh TREE IMAGE PROFILE OUTNAME POINTS_FILE JOBS   (GB10; TREE relative to ~/r42, IMAGE relative to TREE)
# ArchBench points of POINTS_FILE (one id per line) on the three-core board-top RTL stand-in at an image's geometry
# (tools/r42/img_mc.py = tools/r41/link_budget_rtl.py + output digests + the FP8 Hub proxies). Results ~/r42/issue_out/mc/OUTNAME/.
# Run it from an immutable snapshot of the tree: changing the tree under a running job changes the device fingerprint and fails its later points.
TREE=$1; IMAGE=$2; PROFILE=$3; OUT=$4; PTS=$5; JOBS=$6
cd ~/r42/$TREE || exit 2
rm -rf python/tnca/data
source ~/tc_env/env.sh; export PYTHONPATH=$PWD/python:$PWD/compiler/python:$PWD/tools/asm
export TNCA_CACHE=$HOME/r42/issue_out/cache_mc
PY=$HOME/tc_env/torchenv/bin/python
D=$HOME/r42/issue_out/mc/$OUT; mkdir -p $D
run1() { p=$1; f=$(echo $p | tr '/' '_'); [ -s $D/$f.jsonl ] && exit 0
  LE_DIGEST=$D/digest.jsonl nice -n 19 $PY tools/r42/img_mc.py --point $p --out $D/$f.jsonl --image $IMAGE --profile $PROFILE --memory-bytes $((8<<20)) > $D/$f.log 2>&1; echo "$p exit $?"; }
export -f run1; export D PY IMAGE PROFILE
grep -v '^#' $PTS | grep / | xargs -P $JOBS -I{} bash -c 'run1 {}'
echo DONE
