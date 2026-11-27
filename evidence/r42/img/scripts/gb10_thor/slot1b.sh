#!/bin/bash
# after slot1: pointwise points on arm M, then conv2d tiny (fp32) on A, LMF, F
while pgrep -f "[o]ut/slot1.sh" > /dev/null; do sleep 20; done
~/r42/img_out/img_arm.sh M p2 op.add/tiny/fp32-n256 op.silu/tiny/fp16-n256 op.softmax/tiny/fp16-r4-c64 op.rmsnorm/tiny/fp16-r4-d64
for arm in A LMF F; do ~/r42/img_out/img_arm.sh $arm conv op.conv2d/tiny/fp32-n1-cin4-h8-w8-cout8; done
