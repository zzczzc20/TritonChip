#!/bin/bash
# usage: ks_tlm_1c.sh PROFILE CFG OUTNAME BINARY ID...   one-core TLM run (TNCA_ACTIVE_CORES=1) for the single-core RTL comparison
PROFILE=$1; CFG=$2; OUTN=$3; BIN=$4; shift 4
cd $HOME/r42/kfuse-speed || exit 2
rm -rf python/tnca/data
source $HOME/tc_env/env.sh
export PYTHONPATH=$PWD/python:$PWD/compiler/python:$PWD/tools/asm
PY=$HOME/tc_env/torchenv/bin/python
OUT=$HOME/r42/kfuse-speed_out/tlm/$OUTN
mkdir -p $(dirname $OUT)
ARGS=()
for ID in "$@"; do ARGS+=(--point "$ID"); done
nice -n 19 $PY tools/r42/kfuse_speed_tlm_run.py --source-root $PWD --binary $BIN --config $CFG --out $OUT \
  "${ARGS[@]}" --active-cores 1 --profile $PROFILE \
  --env TNCA_RTL_MEM_PORTS=1 --env TNCA_RTL_OUTSTANDING=8 --env TNCA_TRANSFER_POLICY=per-kernel \
  --jobs 1 --timeout 7200
echo RUN_EXIT=$?
