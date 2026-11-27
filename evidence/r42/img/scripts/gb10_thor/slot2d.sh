#!/bin/bash
# after the orphaned attribution run (L image, A profile): matmul small LMF, binary32 effect on attention (F), pointwise on LF, gemv small A and LMF
while kill -0 1531941 2>/dev/null; do sleep 15; done
~/r42/img_out/img_arm.sh LMF p3 op.matmul/small/fp16-m64-k128-n128
~/r42/img_out/img_arm.sh F attn op.attention_prefill/tiny/fp16-b1-h2-s16-d16 op.attention_decode/tiny/fp16-b1-h2-l16-d16
~/r42/img_out/img_arm.sh LF p2 op.add/tiny/fp32-n256 op.silu/tiny/fp16-n256 op.softmax/tiny/fp16-r4-c64 op.rmsnorm/tiny/fp16-r4-d64
for arm in A LMF; do ~/r42/img_out/img_arm.sh $arm gemvs op.gemv/small/fp16-k256-n512; done
