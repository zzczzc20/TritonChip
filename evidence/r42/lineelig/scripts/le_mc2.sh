#!/bin/bash
# le_mc.sh TREE OUTDIR IMAGE PROFILE POINT [POINT...] : the three-core board-top RTL stand-in at an image's geometry (rtl-3c), default knobs
TREE=$1; D=$2; IMAGE=$3; PROFILE=$4; shift 4
cd ~/r42/$TREE || exit 2
rm -rf python/tnca/data
source ~/tc_env/env.sh; export PYTHONPATH=$PWD/python:$PWD/compiler/python:$PWD/tools/asm PYTHONDONTWRITEBYTECODE=1
PY=$HOME/tc_env/torchenv/bin/python
mkdir -p $D; export LE_DIGEST=$D/digest.jsonl
for p in "$@"; do f=$(echo $p | tr '/' '_')
  s=$(date +%s)
  nice -n 19 $PY tools/r42/lineelig_mc.py --point $p --out $D/$f.jsonl --image $IMAGE --profile $PROFILE --memory-bytes $((8<<20)) > $D/$f.log 2>&1
  echo "$p exit $? wall $(( $(date +%s) - s )) s"
done
echo MCDONE
