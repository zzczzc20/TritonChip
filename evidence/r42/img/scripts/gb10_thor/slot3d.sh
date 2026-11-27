#!/bin/bash
# one process at a time: the rest of the rtl-1c tiny sweeps, the final RTL gates, T17 for the single items, the small tier
while kill -0 1477823 2>/dev/null; do sleep 15; done
~/r42/img_out/img_1c.sh img_mc core-vrf-sync-v5-x8-g2c4-dp tiny 1 tiny_merged
~/r42/img_out/img_1c.sh img_mc core-r42-img tiny 1 tiny_lmf
T17="op.add/tiny/fp16-n1000 op.add/tiny/fp32-n256 op.silu/tiny/fp16-n256 op.gelu_tanh/tiny/fp16-n256 op.swiglu/tiny/fp16-n256 op.softmax/tiny/fp16-r4-c64 op.softmax/tiny/fp32-r3-c37 op.rmsnorm/tiny/fp16-r4-d64 op.layernorm/tiny/fp16-r4-d64 op.argmax/tiny/fp16-r2-v256 op.rope/tiny/fp16-b1-h2-s16-d16 op.cumsum/tiny/fp32-r4-c64 op.embedding/tiny/fp16-v256-d64-t16 op.transpose/tiny/fp32-r32-c32 op.matmul/tiny/fp16-m16-k32-n16 op.matmul/tiny/fp32-m16-k32-n16 op.linear/tiny/fp16-m16-k64-n64 op.gemv/tiny/fp16-k64-n64 op.bmm/tiny/fp16-b2-m16-k16-n16 op.attention_decode/tiny/fp16-b1-h2-l16-d16 op.attention_prefill/tiny/fp16-b1-h2-s16-d16"
~/r42/img_out/img_1c.sh img_mc core-r42 - 1 t17_m --point $T17
~/r42/img_out/img_1c.sh img_mc core-r42-fp32 - 1 t17_f --point $T17
~/r42/img_out/img_gates.sh img_t ~/r42/img_out/gates_final core-g2c4 core-run2 core-mmadp core-r42 core-r42fp32 core-mmaf32 matrix-ovl core-abuf16 core-r42img
cd ~/r42/img_t && nice -n 19 python3 tools/r42/img_tests.py matlock --ref ~/r42/img_out/tnca_matrix_ref.v --seeds 1 --cmds 100 > ~/r42/img_out/matlock_final.log 2>&1
~/r42/img_out/img_1c.sh img_mc core-vrf-sync-v5-x8-g2c4-dp small 1 small_merged
~/r42/img_out/img_1c.sh img_mc core-r42-img small 1 small_lmf
