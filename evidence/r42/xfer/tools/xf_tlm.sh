#!/bin/bash
# usage: xf_tlm.sh TREE CFG OUTNAME BINARY ID [ID...]   one or more ArchBench points through the TLM (3 cores, run-3 compiler profile)
# domain: SystemC/TLM model. Output: ~/r42/xfer_out/tlm/OUTNAME
TREE=$1; CFG=$2; OUTN=$3; BIN=$4; shift 4
cd $TREE || exit 2
rm -rf python/tnca/data
source $HOME/tc_env/env.sh
export PYTHONPATH=$PWD/python:$PWD/compiler/python:$PWD/tools/asm
PY=$HOME/tc_env/torchenv/bin/python
OUT=$HOME/r42/xfer_out/tlm/$OUTN
rm -rf $OUT; mkdir -p $(dirname $OUT)
OUTSTAND=$(grep '^outstanding=' $CFG | cut -d= -f2)
ARGS=()
for ID in "$@"; do ARGS+=(--point "$ID"); done
nice -n 19 $PY tools/r42/kfuse_speed_tlm_run.py --source-root $PWD --binary $BIN --config $CFG --out $OUT \
  "${ARGS[@]}" --active-cores 3 --profile core-vrf-sync-v5-x8-g2c4-dp \
  --env TNCA_RTL_MEM_PORTS=3 --env TNCA_RTL_OUTSTANDING=$OUTSTAND --env TNCA_TRANSFER_POLICY=resident \
  $XF_ENV --jobs 1 --timeout 7200
echo RUN_EXIT=$?
