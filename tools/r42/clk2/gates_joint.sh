#!/bin/bash
# R42 clk2 joint gates (GB10, one process at a time, nice 19). Usage: gates_joint.sh BASE_TREE TREE OUTDIR
#   default identity (cycle lines identical to the base tree) of core-r42 (MTX_PIPE 1), core-r42img (MTX_PIPE 1 + MAC_F32 1), core-r42fp32 (MAC_F32 1),
#   the same three with FMA_IMPL 465 and CLKFIX2 = 127 (every program PASS), and the remaining default gates of the joint list
BASE=$1; TREE=$2; OUT=$3
mkdir -p $OUT
source ~/tc_env/env.sh
run() { # TREE TAG TARGET [make args]
  t=$1; tag=$2; tgt=$3; shift 3
  ( cd $t && nice -n 19 make BUILD=sim/build_j_$tag $tgt "$@" > $OUT/${tgt}_$tag.log 2>&1; echo "EXIT=$?" >> $OUT/${tgt}_$tag.log )
}
for tgt in core-r42 core-r42img core-mmaf32; do
  run $BASE base $tgt
  run $TREE def $tgt
  python3 $TREE/tools/r42/clk2/gate_cmp.py $OUT/${tgt}_base.log $OUT/${tgt}_def.log --identical > $OUT/identity_${tgt}.txt 2>&1; echo "identity exit=$?" >> $OUT/identity_${tgt}.txt
  run $TREE m127 $tgt R41_RUN2_FMA_IMPL=465 R42_CLKFIX2=127
  python3 $TREE/tools/r42/clk2/gate_cmp.py $OUT/${tgt}_base.log $OUT/${tgt}_m127.log > $OUT/cmp_${tgt}_m127.txt 2>&1
done
for tgt in core-abuf16 matrix-ovl tensor-line32 r41-xpose8-rtl r38-xpose-rtl; do run $TREE def $tgt; done
echo JOINT_DONE >> $OUT/done.txt
