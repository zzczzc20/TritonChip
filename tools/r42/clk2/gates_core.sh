#!/bin/bash
# R42 clk2 core gates (GB10, one process at a time, nice 19). Usage: gates_core.sh BASE_TREE TREE OUTDIR [MASK...]
#   1. core-run2 on the base tree (origin/R42 head sources) and on the clk2 tree at defaults (R42_CLKFIX2=0): all PASS, cycles identical
#   2. core-run2 on the clk2 tree with R42_CLKFIX2=<mask> for each mask given: every program PASS (cycles are reported)
#   3. core-g2c4, core-mmadp and unit on the clk2 tree at defaults
BASE=$1; TREE=$2; OUT=$3; shift 3
mkdir -p $OUT
source ~/tc_env/env.sh
( cd $BASE && [ -f $OUT/core_run2_base.log ] || { nice -n 19 make BUILD=sim/build_base core-run2 > $OUT/core_run2_base.log 2>&1; echo "EXIT=$?" >> $OUT/core_run2_base.log; } )
( cd $TREE && nice -n 19 make BUILD=sim/build_c2 core-run2 > $OUT/core_run2_default.log 2>&1; echo "EXIT=$?" >> $OUT/core_run2_default.log )
python3 $TREE/tools/r42/clk2/gate_cmp.py $OUT/core_run2_base.log $OUT/core_run2_default.log --identical > $OUT/identity_default.txt 2>&1; echo "identity exit=$?" >> $OUT/identity_default.txt
for M in "$@"; do
  ( cd $TREE && nice -n 19 make BUILD=sim/build_c2_$M core-run2 R42_CLKFIX2=$M > $OUT/core_run2_m$M.log 2>&1; echo "EXIT=$?" >> $OUT/core_run2_m$M.log )
  python3 $TREE/tools/r42/clk2/gate_cmp.py $OUT/core_run2_base.log $OUT/core_run2_m$M.log > $OUT/cmp_m$M.txt 2>&1
done
( cd $TREE && nice -n 19 make BUILD=sim/build_c2 core-g2c4 > $OUT/core_g2c4.log 2>&1; echo "EXIT=$?" >> $OUT/core_g2c4.log )
( cd $TREE && nice -n 19 make BUILD=sim/build_c2 core-mmadp > $OUT/core_mmadp.log 2>&1; echo "EXIT=$?" >> $OUT/core_mmadp.log )
( cd $TREE && nice -n 19 make BUILD=sim/build_c2 unit > $OUT/unit.log 2>&1; echo "EXIT=$?" >> $OUT/unit.log )
echo GATES_DONE >> $OUT/done.txt
