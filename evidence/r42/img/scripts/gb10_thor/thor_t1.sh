#!/bin/bash
P1="op.matmul/tiny/fp16-m16-k32-n16 op.linear/tiny/fp16-m16-k64-n64 op.gemv/tiny/fp16-k64-n64 op.bmm/tiny/fp16-b2-m16-k16-n16 op.matmul/tiny/fp32-m16-k32-n16"
for arm in L M; do ~/r42/img_out/img_arm.sh $arm p1 $P1; done
