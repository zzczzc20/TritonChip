#!/bin/bash
# usage: dump_point.sh TREE OUTDIR POINTID [PROFILE]   compile-only kernel dump (.ttir/.tnir/.s) of one point, private caches
TREE=$1; OUT=$2; ID=$3; PROFILE=${4:-core-vrf-sync-v5-x8-g2c4-dp}
cd $TREE || exit 2
. tools/r42/fp8sw/env.sh
TAG=$(echo "$ID" | tr '/' '_')
D=$OUT/$TAG
rm -rf $D; mkdir -p $D/triton $D/inductor
TNCA_COMPILE_CACHE_DIR=$D/tnca_cc TRITON_CACHE_DIR=$D/triton TORCHINDUCTOR_CACHE_DIR=$D/inductor \
  TNCA_KERNEL_DUMP=1 TNCA_DUMP_DIR=$D/dump \
  nice -n 19 $PY tools/r42/fp8sw/run_with_proxies.py --platform tnca --profile $PROFILE --point "$ID" --compile-only --out $D/rec.jsonl > $D/run.log 2>&1
echo "$ID rc=$?"
