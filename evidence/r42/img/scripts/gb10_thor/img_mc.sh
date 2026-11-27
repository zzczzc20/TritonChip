#!/bin/bash
# img_mc.sh TREE OUTDIR IMAGE PROFILE "SET1 SET2..." POINT... : rtl-3c (three-core board-top RTL stand-in at an image's geometry), default knobs
# digest: sha256 of every compared output (tools/r42/lineelig_mc.py). SETs are --set K=V pairs (use - for none).
TREE=$1; D=$2; IMAGE=$3; PROFILE=$4; SETS=$5; shift 5
cd ~/r42/$TREE || exit 2
rm -rf python/tnca/data
source ~/tc_env/env.sh; export PYTHONPATH=$PWD/python:$PWD/compiler/python:$PWD/tools/asm PYTHONDONTWRITEBYTECODE=1
PY=$HOME/tc_env/torchenv/bin/python
mkdir -p $D; export LE_DIGEST=$D/digest.jsonl
SETARGS=""; [ "$SETS" != "-" ] && for kv in $SETS; do SETARGS="$SETARGS --set $kv"; done
for p in "$@"; do f=$(echo $p | tr '/' '_')
  [ -s $D/$f.jsonl ] && { echo "$p skipped (record exists)"; continue; }
  s=$(date +%s)
  nice -n 19 $PY ${IMG_MC_PY:-tools/r42/lineelig_mc.py} --point $p --out $D/$f.jsonl --image $IMAGE --profile $PROFILE --memory-bytes $((8<<20)) $SETARGS > $D/$f.log 2>&1
  echo "$p exit $? wall $(( $(date +%s) - s )) s"
done
echo MCDONE
