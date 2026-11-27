#!/bin/bash
while pgrep -f "[o]ut/t3.sh" > /dev/null; do sleep 20; done
for arm in L M; do ~/r42/img_out/img_arm.sh $arm attn op.attention_prefill/tiny/fp16-b1-h2-s16-d16 op.attention_decode/tiny/fp16-b1-h2-l16-d16; done; IMG_MC_PY=tools/r42/img_mc.py ~/r42/img_out/img_arm.sh LMF fp8 op.fp8_linear/xl/fp8-m16-k256-n128
