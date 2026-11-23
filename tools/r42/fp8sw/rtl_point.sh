#!/bin/bash
# usage: rtl_point.sh TREE OUTDIR POINTID [extra archbench.run args]   one point on the single-core RTL simulator, default knobs
# (profile core-vrf-sync-v5-x8-g2c4-dp), record to OUTDIR/<tag>.jsonl
TREE=$1; OUT=$2; ID=$3; shift 3
cd $TREE || exit 2
. tools/r42/fp8sw/env.sh
TAG=$(echo "$ID" | tr '/' '_')
mkdir -p $OUT
rm -f $OUT/$TAG.jsonl
nice -n 19 $PY tools/r42/fp8sw/run_with_proxies.py --platform tnca --profile ${PROFILE:-core-vrf-sync-v5-x8-g2c4-dp} --point "$ID" "$@" \
  --out $OUT/$TAG.jsonl > $OUT/$TAG.log 2>&1
echo "$ID rc=$?"
