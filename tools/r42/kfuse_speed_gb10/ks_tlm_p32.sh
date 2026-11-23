#!/bin/bash
# usage: ks_tlm.sh CFG OUTNAME BINARY ID [ID...]    one ArchBench point per call through the TLM (3 cores, run-3 compiler profile)
CFG=$1; OUTN=$2; BIN=$3; shift 3
cd $HOME/r42/kfuse-speed || exit 2
rm -rf python/tnca/data
source $HOME/tc_env/env.sh
export PYTHONPATH=$PWD/python:$PWD/compiler/python:$PWD/tools/asm
PY=$HOME/tc_env/torchenv/bin/python
OUT=$HOME/r42/kfuse-speed_out/tlm/$OUTN
mkdir -p $(dirname $OUT)
OUTSTAND=$(grep '^outstanding=' $CFG | cut -d= -f2)
ARGS=()
for ID in "$@"; do ARGS+=(--point "$ID"); done
nice -n 19 $PY tools/r42/kfuse_speed_tlm_run.py --source-root $PWD --binary $BIN --config $CFG --out $OUT \
  "${ARGS[@]}" --active-cores 3 --profile core-vrf-sync-v5-x8-g2c4-dp \
  --env TNCA_RTL_MEM_PORTS=32 --env TNCA_RTL_OUTSTANDING=$OUTSTAND --env TNCA_TRANSFER_POLICY=resident \
  --jobs 1 --timeout 7200
echo RUN_EXIT=$?
