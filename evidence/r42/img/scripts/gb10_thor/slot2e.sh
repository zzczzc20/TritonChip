#!/bin/bash
# after the orphaned attribution run: the name-matched arms (the same image under the profile name of the arm compared with), matmul small LMF, binary32 effect on attention
while kill -0 1531941 2>/dev/null; do sleep 15; done
for arm in LN LMN MN; do ~/r42/img_out/img_arm.sh $arm p4 op.gemv/tiny/fp16-k64-n64 op.linear/tiny/fp16-m16-k64-n64 op.matmul/tiny/fp16-m16-k32-n16 op.bmm/tiny/fp16-b2-m16-k16-n16; done
~/r42/img_out/img_arm.sh LMFN p5 op.gemv/tiny/fp16-k64-n64 op.linear/tiny/fp16-m16-k64-n64 op.matmul/tiny/fp16-m16-k32-n16 op.bmm/tiny/fp16-b2-m16-k16-n16 op.matmul/tiny/fp32-m16-k32-n16
~/r42/img_out/img_arm.sh LMF p3 op.matmul/small/fp16-m64-k128-n128
~/r42/img_out/img_arm.sh F attn op.attention_prefill/tiny/fp16-b1-h2-s16-d16 op.attention_decode/tiny/fp16-b1-h2-l16-d16
