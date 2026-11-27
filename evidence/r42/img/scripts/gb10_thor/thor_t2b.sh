#!/bin/bash
while pgrep -f "[o]ut/t2.sh" > /dev/null; do sleep 20; done
IMG_MC_PY=tools/r42/img_mc.py ~/r42/img_out/img_arm.sh L fp8 op.fp8_linear/xl/fp8-m16-k256-n128
