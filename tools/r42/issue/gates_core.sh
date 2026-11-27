#!/bin/bash
# R42 issue core gates (GB10, one process at a time, nice 19). Usage: gates_core.sh BASE_TREE ISSUE_TREE OUTDIR
#   1. core-run2 on the base tree (origin/R42 head sources) and on the issue tree at defaults: every program PASS and the cycle counts identical
#   2. core-run2 on the issue tree with R42_VRF_FWD=1: every program PASS (results identical, cycles are reported)
#   3. core-g2c4, core-mmadp and unit on the issue tree at defaults
BASE=$1; TREE=$2; OUT=$3
mkdir -p $OUT
source ~/tc_env/env.sh
( cd $BASE && nice -n 19 make BUILD=sim/build_base core-run2 > $OUT/core_run2_base.log 2>&1; echo "EXIT=$?" >> $OUT/core_run2_base.log )
( cd $TREE && nice -n 19 make BUILD=sim/build_issue core-run2 > $OUT/core_run2_default.log 2>&1; echo "EXIT=$?" >> $OUT/core_run2_default.log )
( cd $TREE && nice -n 19 make BUILD=sim/build_fwd core-run2 R42_VRF_FWD=1 > $OUT/core_run2_fwd.log 2>&1; echo "EXIT=$?" >> $OUT/core_run2_fwd.log )
( cd $TREE && nice -n 19 make BUILD=sim/build_issue core-g2c4 > $OUT/core_g2c4.log 2>&1; echo "EXIT=$?" >> $OUT/core_g2c4.log )
( cd $TREE && nice -n 19 make BUILD=sim/build_issue core-mmadp > $OUT/core_mmadp.log 2>&1; echo "EXIT=$?" >> $OUT/core_mmadp.log )
( cd $TREE && nice -n 19 make BUILD=sim/build_issue unit > $OUT/unit.log 2>&1; echo "EXIT=$?" >> $OUT/unit.log )
echo GATES_DONE >> $OUT/done.txt
