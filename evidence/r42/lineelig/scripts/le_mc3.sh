#!/bin/bash
# le_mc3.sh TREE OUTDIR IMAGE PROFILE "TNCA_KEY=VAL ..." POINT [POINT...] : as le_mc2.sh with compiler settings through --set (rtl-3c; a record with --set counts for no target)
TREE=$1; D=$2; IMAGE=$3; PROFILE=$4; ENVS=$5; shift 5
cd ~/r42/$TREE || exit 2
rm -rf python/tnca/data
source ~/tc_env/env.sh; export PYTHONPATH=$PWD/python:$PWD/compiler/python:$PWD/tools/asm PYTHONDONTWRITEBYTECODE=1
PY=$HOME/tc_env/torchenv/bin/python
mkdir -p $D; export LE_DIGEST=$D/digest.jsonl
SETS=""; [ "$ENVS" = "-" ] && ENVS=""; for kv in $ENVS; do SETS="$SETS --set $kv"; done
for p in "$@"; do f=$(echo $p | tr '/' '_')
  s=$(date +%s)
  nice -n 19 $PY tools/r42/lineelig_mc.py --point $p --out $D/$f.jsonl --image $IMAGE --profile $PROFILE --memory-bytes $((8<<20)) $SETS > $D/$f.log 2>&1
  echo "$p exit $? wall $(( $(date +%s) - s )) s"
done
echo MCDONE
