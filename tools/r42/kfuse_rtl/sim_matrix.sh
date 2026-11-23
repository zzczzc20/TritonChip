#!/bin/bash
# sim_matrix.sh LISTFILE [PARALLEL]  -- on GB10 from the tree root: runs run_sim.sh for every line of LISTFILE
#   line: TAG G W N SEED "PARAMS" "PLUSARGS"       (PARAMS e.g. -Ptb_kfuse_lane.TREE=1 ; PLUSARGS e.g. +BUBBLE=30)
# at most PARALLEL (default 3) jobs at once, nice 19; waits while the 1-minute load is above 16.
# Vector files are shared by (G, W, N, SEED); the first job of a kind generates it, so lines with the same vectors
# must not be started together: the list is ordered, generation happens in a pre-pass.
set -u
LIST=$1; PAR=${2:-3}
export OUT=${OUT:-$HOME/r42/kfuse-rtl_out}
mkdir -p $OUT
HERE=$(cd $(dirname $0) && pwd)
# pre-pass: generate each distinct vector file once
source ~/tc_env/env.sh
ROOT=$(cd $HERE/../../.. && pwd)
cut -d' ' -f2-5 $LIST | sort -u | while read G W N SEED; do
    V=$OUT/vec_G${G}_W${W}_N${N}_s${SEED}.hex
    if [ ! -f $V ]; then
        nice -n 19 python3 $ROOT/tools/r42/kfuse_rtl_vectors.py --G $G --W $W --N $N --seed $SEED --procs 4 --out $V > $V.gen.log 2>&1
        sha256sum $V > $V.sha256
    fi
done
n=0
while IFS= read -r line; do
    [ -z "$line" ] && continue
    eval "set -- $line"
    TAG=$1; G=$2; W=$3; N=$4; SEED=$5; PARAMS=${6:-}; PLUSARGS=${7:-}
    while [ $(jobs -rp | wc -l) -ge $PAR ]; do sleep 5; done
    while [ $(cut -d. -f1 /proc/loadavg) -ge 16 ]; do sleep 20; done
    ( SEED=$SEED NOGEN=1 PARAMS="$PARAMS" PLUSARGS="$PLUSARGS" bash $HERE/run_sim.sh $TAG $G $W $N > $OUT/$TAG.run.out 2>&1 ) &
done < $LIST
wait
echo "matrix done: $(date -u +%FT%TZ)" >> $OUT/matrix.log
