#!/bin/bash
# DSE worker (R42-dse): one TLM run (domain: SystemC/TLM model) of a list of ArchBench points on GB10.
# usage: dse_run.sh TREE CFG OUTNAME BINARY IDFILE [POINTS_MODULE_RELPATH]     env: JOBS (default 3), CORES (default: nx*ny of CFG), DSE_ENV (extra --env args)
TREE=$1; CFG=$2; OUTN=$3; BIN=$4; IDF=$5; PM=$6
cd $TREE || exit 2
rm -rf python/tnca/data
source $HOME/tc_env/env.sh
export PYTHONPATH=$PWD/python:$PWD/compiler/python:$PWD/tools/asm
PY=$HOME/tc_env/torchenv/bin/python
OUT=$HOME/r42/dse_out/tlm/$OUTN
rm -rf $OUT; mkdir -p $(dirname $OUT)
# SEED=<output dir of an earlier run with the SAME software tree>: reuse its Inductor/Triton compile caches (kernels do not depend on the model configuration)
OUTSTAND=$(grep '^outstanding=' $CFG | cut -d= -f2)
NX=$(grep '^nx=' $CFG | cut -d= -f2); NY=$(grep '^ny=' $CFG | cut -d= -f2)
OUTL=$(grep '^outlets=' $CFG | cut -d= -f2)
CORES=${CORES:-$((NX*NY))}
ARGS=()
while read -r ID; do [ -n "$ID" ] && ARGS+=(--point "$ID"); done < $IDF
PMA=""; [ -n "$PM" ] && PMA="--points-module $PM"
nice -n 19 $PY $HOME/r42/dse/tools/r42/kfuse_speed_tlm_run.py --source-root $PWD $PMA --binary $BIN --config $CFG --out $OUT \
  "${ARGS[@]}" --active-cores $CORES --profile core-vrf-sync-v5-x8-g2c4-dp \
  --env TNCA_RTL_MEM_PORTS=$OUTL --env TNCA_RTL_OUTSTANDING=$OUTSTAND --env TNCA_TRANSFER_POLICY=resident \
  ${SEED:+--cache-seed $SEED} ${MEMB:+--memory-bytes $MEMB} $DSE_ENV --jobs ${JOBS:-3} --timeout 7200 > $OUT.log 2>&1
echo RUN_EXIT=$? >> $OUT.log
python3 $HOME/r42/dse/tools/r42/kfuse_speed_collect.py $OUT $HOME/r42/dse_out/col/$OUTN.json >> $OUT.log 2>&1
