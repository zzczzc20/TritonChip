#!/bin/bash
# rtl-1c T17 for the single items, to attribute the points that moved: MTX_PIPE alone (core-r42) and MAC_F32 alone (core-r42-fp32)
while pgrep -f "[o]ut/slot3b.sh" > /dev/null; do sleep 15; done
T17="op.add/tiny/fp16-n1000 op.add/tiny/fp32-n256 op.silu/tiny/fp16-n256 op.gelu_tanh/tiny/fp16-n256 op.swiglu/tiny/fp16-n256 op.softmax/tiny/fp16-r4-c64 op.softmax/tiny/fp32-r3-c37 op.rmsnorm/tiny/fp16-r4-d64 op.layernorm/tiny/fp16-r4-d64 op.argmax/tiny/fp16-r2-v256 op.rope/tiny/fp16-b1-h2-s16-d16 op.cumsum/tiny/fp32-r4-c64 op.embedding/tiny/fp16-v256-d64-t16 op.transpose/tiny/fp32-r32-c32 op.matmul/tiny/fp16-m16-k32-n16 op.matmul/tiny/fp32-m16-k32-n16 op.linear/tiny/fp16-m16-k64-n64 op.gemv/tiny/fp16-k64-n64 op.bmm/tiny/fp16-b2-m16-k16-n16 op.attention_decode/tiny/fp16-b1-h2-l16-d16 op.attention_prefill/tiny/fp16-b1-h2-s16-d16"
~/r42/img_out/img_1c.sh img_mc core-r42 - 1 t17_m --point $T17
~/r42/img_out/img_1c.sh img_mc core-r42-fp32 - 1 t17_f --point $T17
