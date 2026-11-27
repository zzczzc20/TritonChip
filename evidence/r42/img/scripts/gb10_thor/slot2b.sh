#!/bin/bash
# after slot2: attention tiny (A, LMF), matmul small on LMF and LM, gemv small on A, L, LMF
while pgrep -f "[o]ut/slot2.sh" > /dev/null; do sleep 20; done
for arm in A LMF; do ~/r42/img_out/img_arm.sh $arm attn op.attention_prefill/tiny/fp16-b1-h2-s16-d16 op.attention_decode/tiny/fp16-b1-h2-l16-d16; done
for arm in LMF LM; do ~/r42/img_out/img_arm.sh $arm p3 op.matmul/small/fp16-m64-k128-n128; done
for arm in A L LMF; do ~/r42/img_out/img_arm.sh $arm gemvs op.gemv/small/fp16-k256-n512; done
