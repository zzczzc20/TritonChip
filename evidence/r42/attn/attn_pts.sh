#!/bin/bash
# usage: attn_pts.sh TREE OUTNAME JOBS POINT...   run points on single-core RTL sim, records to ~/r42/attn_out/OUTNAME.jsonl
TREE=$1; OUT=$2; JOBS=$3; shift 3
cd ~/r42/$TREE || exit 2
rm -rf python/tnca/data
source ~/tc_env/env.sh; export PYTHONPATH=$PWD/python:$PWD/compiler/python:$PWD/tools/asm
PY=$HOME/tc_env/torchenv/bin/python
$PY -c "import tnca,tnca_cc,tnca_asm; print(tnca.__file__, tnca_cc.__file__, tnca_asm.__file__)"
PROFILE=${PROFILE:-core-vrf-sync-v5-x8-g2c4-dp}
mkdir -p ~/r42/attn_out/parts/$OUT
printf '%s\n' "$@" | nice -n 19 xargs -P $JOBS -I{} sh -c 'N=$(echo {} | tr "/." "__"); '$PY' -m archbench.run --platform tnca --profile '$PROFILE' --point {} --out ~/r42/attn_out/parts/'$OUT'/$N.jsonl > ~/r42/attn_out/parts/'$OUT'/$N.log 2>&1; echo {} rc=$?'
cat ~/r42/attn_out/parts/$OUT/*.jsonl > ~/r42/attn_out/$OUT.jsonl
echo DONE
