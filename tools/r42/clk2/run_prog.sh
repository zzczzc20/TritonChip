#!/bin/bash
# run_prog.sh TREE MASK PROG [vvp args]   (GB10) -- one core-run2 program at FMA_IMPL 465 and CLKFIX2=MASK (builds only the bench and that program)
TREE=$1; M=$2; P=$3; shift 3
source ~/tc_env/env.sh
cd $TREE
B=sim/build_p_$M
nice -n 19 make BUILD=$B $B/tb_core_run2.vvp $B/$P.hex R41_RUN2_FMA_IMPL=465 R42_CLKFIX2=$M > $B.build.log 2>&1 || { tail -5 $B.build.log; exit 2; }
vvp -M ~/tc_env/root/usr/lib/aarch64-linux-gnu/ivl $B/tb_core_run2.vvp +hex=$B/$P.hex +maxcycles=3000000 "$@" 2>&1 | grep -E "FAIL|cycles=|PASS|trap" | head -5
