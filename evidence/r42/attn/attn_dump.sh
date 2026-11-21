#!/bin/bash
# usage: attn_dump.sh TREE TAG POINT...   compile-only kernel dump per point into ~/r42/attn_out/dump/TAG/<n>
TREE=$1; TAG=$2; shift 2
cd ~/r42/$TREE || exit 2
rm -rf python/tnca/data
source ~/tc_env/env.sh; export PYTHONPATH=$PWD/python:$PWD/compiler/python:$PWD/tools/asm
PY=$HOME/tc_env/torchenv/bin/python
$PY -c "import tnca,tnca_cc,tnca_asm; print(tnca.__file__, tnca_cc.__file__, tnca_asm.__file__)"
PROFILE=${PROFILE:-core-vrf-sync-v5-x8-g2c4-dp}
for P in "$@"; do
  N=$(echo $P | tr '/.' '__')
  D=$HOME/r42/attn_out/dump/$TAG/$N
  rm -rf $D; mkdir -p $D
  TNCA_KERNEL_DUMP=1 TNCA_DUMP_DIR=$D nice -n 19 $PY -m archbench.run --platform tnca --profile $PROFILE --point $P --compile-only --out $D/cov.jsonl > $D/log.txt 2>&1
  echo "$P rc=$?"
done
echo DONE
