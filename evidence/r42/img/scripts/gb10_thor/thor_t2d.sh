#!/bin/bash
while pgrep -f "[o]ut/t2.sh" > /dev/null; do sleep 20; done
~/r42/img_out/img_arm.sh LM p2 op.add/tiny/fp32-n256 op.silu/tiny/fp16-n256 op.softmax/tiny/fp16-r4-c64 op.rmsnorm/tiny/fp16-r4-d64
IMG_MC_PY=tools/r42/img_mc.py ~/r42/img_out/img_arm.sh L fp8 op.fp8_linear/xl/fp8-m16-k256-n128
