#!/bin/bash
# after the orphaned LMF attention run: attribution of the pointwise differences (the same image with the A profile name), then matmul small LMF, LM and gemv small A, L, LMF
while kill -0 1518293 2>/dev/null; do sleep 15; done
IMGD=$HOME/r42/img_mc/tools/r42/images
export R42_VVP_CACHE=$HOME/r42/img_out/vvpcache
for arm in l m; do
  ~/r42/img_out/img_mc.sh img_mc $HOME/r42/img_out/mc/attr_${arm}_aprof $IMGD/r42img_${arm}_sim.tcl core-vrf-sync-v5-x8-g2c4-dp - op.add/tiny/fp32-n256 op.softmax/tiny/fp16-r4-c64
done
for arm in LMF LM; do ~/r42/img_out/img_arm.sh $arm p3 op.matmul/small/fp16-m64-k128-n128; done
for arm in A L LMF; do ~/r42/img_out/img_arm.sh $arm gemvs op.gemv/small/fp16-k256-n512; done
