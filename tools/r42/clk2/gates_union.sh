#!/bin/bash
# R42 union gates (GB10, one process at a time, nice 19). Usage: gates_union.sh BASE_TREE UNION_TREE OUTDIR
#   default identity against the base tree (origin/R42): core-run2 and core-r42; every program PASS with CLKFIX (all nine bits, 511) and with CLKFIX 511 +
#   CLKFIX2 127 (FMA_IMPL 465); then core-g2c4, core-mmadp, unit, matrix-ovl at defaults.
BASE=$1; U=$2; OUT=$3
mkdir -p $OUT
source ~/tc_env/env.sh
run() { # TREE TAG TARGET [make args]
  t=$1; tag=$2; tgt=$3; shift 3
  ( cd $t && nice -n 19 make BUILD=sim/build_u_$tag $tgt "$@" > $OUT/${tgt}_$tag.log 2>&1; echo "EXIT=$?" >> $OUT/${tgt}_$tag.log )
}
cmp() { python3 $U/tools/r42/clk2/gate_cmp.py "$@"; }
for tgt in core-run2 core-r42; do
  [ -f $OUT/${tgt}_base.log ] || run $BASE base $tgt
  run $U def $tgt
  cmp $OUT/${tgt}_base.log $OUT/${tgt}_def.log --identical > $OUT/identity_${tgt}.txt 2>&1; echo "identity exit=$?" >> $OUT/identity_${tgt}.txt
  run $U c511 $tgt R42_CLKFIX=511
  cmp $OUT/${tgt}_base.log $OUT/${tgt}_c511.log > $OUT/cmp_${tgt}_c511.txt 2>&1
  run $U c511x127 $tgt R41_RUN2_FMA_IMPL=465 R42_CLKFIX=511 R42_CLKFIX2=127
  cmp $OUT/${tgt}_base.log $OUT/${tgt}_c511x127.log > $OUT/cmp_${tgt}_c511x127.txt 2>&1
done
for tgt in core-g2c4 core-mmadp unit matrix-ovl; do run $U def $tgt; done
echo UNION_GATES_DONE >> $OUT/done.txt
