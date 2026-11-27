#!/bin/bash
for arm in A L M; do ~/r42/img_out/img_arm.sh $arm p3 op.matmul/small/fp16-m64-k128-n128; done
