#!/bin/bash
# run_all.sh RUNDIR CUTS... -- on GB10: build mac_tb per CUTS, run the FP32
# miter and the single-product checks; the multi-product dumps and the oracle
# check run for the first CUTS only (the arithmetic is the same at every cut
# placement; the others check the wrapper's plumbing).
# Env: MITER_N (per class), S16_N (per code pair and slot), DUMP_N (per class).
set -u
source ~/tc_env/env.sh; export PATH=$PATH:$HOME/tc_env/root/usr/bin; export VERILATOR_ROOT=$HOME/tc_env/vlroot
here=$(cd $(dirname $0) && pwd)
RUN=${1:?rundir}; shift
mkdir -p $RUN
MITER_N=${MITER_N:-2000000}; S16_N=${S16_N:-500000}; DUMP_N=${DUMP_N:-500000}
first=1
for CUTS in "$@"; do
    lat=$((1 + $(echo $CUTS | tr -cd 1 | wc -c)))
    B=$RUN/build_$CUTS
    $here/build.sh $CUTS $B > $RUN/build_$CUTS.out 2>&1 || { echo "BUILD FAIL $CUTS"; cat $RUN/build_$CUTS.out; continue; }
    export LAT_NEW=$lat
    $B/obj_dir/Vmac_tb miter $MITER_N 11 > $RUN/miter_$CUTS.log 2>&1; echo "miter $CUTS exit $?"
    $B/obj_dir/Vmac_tb single8 12 > $RUN/single8_$CUTS.log 2>&1; echo "single8 $CUTS exit $?"
    $B/obj_dir/Vmac_tb single16 $S16_N 13 > $RUN/single16_$CUTS.log 2>&1; echo "single16 $CUTS exit $?"
    if [ $first = 1 ]; then
        first=0
        for cls in 0 1 2 3 4 5 10 11 12 13 14 15 16; do
            $B/obj_dir/Vmac_tb dump $cls $DUMP_N 21 > $RUN/dump_$cls.txt 2>&1 &
            while [ $(jobs -r | wc -l) -ge 5 ]; do sleep 1; done
        done
        wait
        python3 $here/check_mac.py -j 6 $RUN/dump_*.txt > $RUN/check_mac.log 2>&1; echo "check_mac exit $?"
    fi
done
echo ALLDONE
