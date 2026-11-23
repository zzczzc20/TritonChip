#!/bin/bash
# usage: arch_dump.sh OUTDIR POINT...   compile-only kernel dumps, tree ~/r42/arch, one dump dir per point
OUTDIR=$1; shift
cd ~/r42/arch || exit 2
source ~/tc_env/env.sh
export PYTHONPATH=$PWD/python:$PWD/compiler/python:$PWD/tools/asm PYTHONDONTWRITEBYTECODE=1
PY=$HOME/tc_env/torchenv/bin/python
PROFILE=core-vrf-sync-v5-x8-g2c4-dp
for POINT in "$@"; do
  tag=$(echo $POINT | tr '/' '_')
  O=$HOME/r42/arch_out/$OUTDIR/dump_$tag; rm -rf $O; mkdir -p $O
  TNCA_KERNEL_DUMP=1 TNCA_DUMP_DIR=$O nice -n 19 $PY -m archbench.run --platform tnca --profile $PROFILE --point $POINT --compile-only --out $O/cov.jsonl > $O/log 2>&1
  echo "$POINT $?" >> $HOME/r42/arch_out/$OUTDIR.done
done
echo ALLDONE >> $HOME/r42/arch_out/$OUTDIR.done
