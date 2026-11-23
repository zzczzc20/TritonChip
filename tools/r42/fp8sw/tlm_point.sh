#!/bin/bash
# usage: tlm_point.sh TREE OUTDIR CFGNAME POINTID...    ArchBench points (incl. the kfuse-speed FP8 proxies) through the SystemC/TLM
# model, three cores, run-3 compiler profile, the compiler's default knobs. Domain: model prediction (not RTL, not board).
# Needs OUTDIR/cfg/CFGNAME.cfg (python3 tools/r42/kfuse_speed_tlm_cfg.py OUTDIR/cfg) and the model binary OUTDIR/tlm-build/tnca-chip.
TREE=$1; OUT=$2; CFGN=$3; shift 3
cd $TREE || exit 2
. tools/r42/fp8sw/env.sh
CFG=$OUT/cfg/$CFGN.cfg
OUTSTAND=$(grep '^outstanding=' $CFG | cut -d= -f2)
ARGS=()
for ID in "$@"; do ARGS+=(--point "$ID"); done
nice -n 19 $PY tools/r42/kfuse_speed_tlm_run.py --source-root $PWD --points-module $PWD/tools/r42/fp8sw/points.py \
  --binary $OUT/tlm-build/tnca-chip --config $CFG --out $OUT/tlm/${TLMTAG:-run} \
  "${ARGS[@]}" --active-cores 3 --profile core-vrf-sync-v5-x8-g2c4-dp \
  --env TNCA_RTL_MEM_PORTS=3 --env TNCA_RTL_OUTSTANDING=$OUTSTAND --env TNCA_TRANSFER_POLICY=resident \
  --jobs 1 --timeout 7200
echo RUN_EXIT=$?
