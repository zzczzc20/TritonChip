#!/bin/bash
# fp32 matmul at the small size: A, F, LMF (one process)
for arm in A F LMF; do ~/r42/img_out/img_arm.sh $arm fp32s op.matmul/small/fp32-m64-k128-n128; done
