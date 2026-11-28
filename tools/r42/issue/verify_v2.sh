#!/bin/bash
# R42 issue: verification of the final VRF_FWD RTL from an immutable snapshot (GB10, one process at a time).
#   verify_v2.sh SNAPSHOT OUTDIR
# 1. build the probe benches, 2. lockstep (random programs, 1/2/4 contexts, reference = VRF_FWD 0), 3. core gates of tools/r42/issue/gates_core.sh
SNAP=$1; OUT=$2
mkdir -p $OUT
cd $SNAP; source ~/tc_env/env.sh
bash tools/r42/issue/probe/build_probe.sh $OUT/bench > $OUT/bench_build.log 2>&1
python3 tools/r42/issue/lockstep_gen.py $OUT/ls_progs 100 201 > $OUT/ls_gen.log 2>&1
python3 tools/r42/issue/lockstep_run.py $OUT/ls_progs $OUT/bench/tb_run2.vvp $OUT/bench/tb_fwd.vvp $OUT/lockstep.tsv 1,2,4 > $OUT/lockstep.log 2>&1
bash tools/r42/issue/gates_core.sh ~/r42/issue_out/base $SNAP $OUT/gates > $OUT/gates.log 2>&1
echo VERIFY_DONE >> $OUT/done.txt
