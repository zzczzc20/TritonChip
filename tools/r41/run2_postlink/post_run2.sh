#!/bin/bash
# R41 run 2: after the link ends, the read-only post-route analysis of the routed checkpoint
# (analyze_run2.tcl, 2 threads, a copy of run 1's) and the threshold check (check_run2.py).
# Never touches the card and never writes into the build. Start it after the link has ended,
# or detached now: it waits for build/attempt.exit before it starts Vivado.
#   bash tools/r41/run2_postlink/post_run2.sh BUILD_ROOT SOURCE_ROOT
set -u
B=$1; SRC=$2; A=$B/analysis_g4
while [ ! -f $B/build/attempt.exit ]; do sleep 120; done
[ -e $A ] && { echo "exists: $A"; exit 1; }
mkdir -p $A.d; log=$A.d/post.log
echo "$(date -u +%FT%TZ) attempt exit $(cat $B/build/attempt.exit)" >> $log
I=$B/xo/tmp/link/vivado/vpl/prj/prj.runs/impl_1
DCP=$(ls $I/level0_wrapper_postroute_physopt.dcp 2>/dev/null || ls $I/level0_wrapper_routed.dcp 2>/dev/null)
[ -n "$DCP" ] || { echo "no routed checkpoint" >> $log; ls $I >> $log 2>&1; echo FAIL > $A.d/post.exit; exit 1; }
echo "dcp $DCP $(sha256sum $DCP | cut -c1-64)" >> $log
source ~/x86_link_env.sh >/dev/null 2>&1
vivado -mode batch -nojournal -log $A.d/vivado.log -source $SRC/tools/r41/run2_postlink/analyze_run2.tcl -tclargs $DCP $A >> $log 2>&1
echo "vivado exit $?" >> $log
python3 $SRC/tools/r41/run2_postlink/check_run2.py $A --json $A.d/check.json >> $log 2>&1
echo $? > $A.d/post.exit
