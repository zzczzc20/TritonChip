| point (small tier) | hot innermost loop | instr | slot A | inc | cmp | br | const | mov | addr | pred | vector/ldst/mtx/txf/bar |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| model.gpt2_block_small_fp16-d128-h4-b1-s64 | `__rctx_program` (fused__scaled_dot_product_fl) | 670 | 53 % | 0 | 1 | 1 | 74 | 4 | 225 | 49 | vx 157, ld/st 149, mtx 0, txf 4, bar 2 |
| model.llama_decode_small_fp16-b1-p128 | `for_5` (fused__to_copy__unsafe_view_) | 738 | 48 % | 1 | 35 | 6 | 30 | 0 | 279 | 3 | vx 379, ld/st 5, mtx 0, txf 0, bar 0 |
| model.llama_prefill_small_fp16-b1-s64 | `__rctx_program` (fused__scaled_dot_product_fl) | 3400 | 30 % | 0 | 1 | 1 | 94 | 6 | 785 | 128 | vx 1493, ld/st 760, mtx 128, txf 0, bar 0 |
| model.mlp_small_fp16-t64-d128 | `gemm_pipe` (fused_addmm_gelu_t_0) | 38 | 53 % | 2 | 2 | 3 | 3 | 8 | 2 | 0 | vx 0, ld/st 0, mtx 2, txf 8, bar 8 |
| model.resnet_block_small_fp32-n1-c16-hw16 | `for_1` (fused__native_batch_norm_leg) | 1748 | 20 % | 0 | 4 | 2 | 120 | 2 | 134 | 96 | vx 1116, ld/st 130, mtx 144, txf 0, bar 0 |
| op.add_small_fp16-n16384 | `tiles_1` (fused_add_0) | 126 | 11 % | 1 | 1 | 1 | 0 | 0 | 11 | 0 | vx 64, ld/st 48, mtx 0, txf 0, bar 0 |
| op.argmax_small_fp16-r8-v2048 | `for_1` (fused_argmax_0) | 22 | 50 % | 2 | 1 | 1 | 0 | 0 | 0 | 7 | vx 10, ld/st 1, mtx 0, txf 0, bar 0 |
| op.attention_decode_small_fp16-b1-h4-l256-d32 | `for_1` (fused__scaled_dot_product_fl) | 724 | 39 % | 2 | 65 | 1 | 0 | 0 | 192 | 24 | vx 436, ld/st 4, mtx 0, txf 0, bar 0 |
| op.attention_prefill_small_fp16-b1-h4-s64-d32 | `__rctx_program` (fused__scaled_dot_product_fl) | 859 | 22 % | 0 | 1 | 1 | 9 | 3 | 135 | 32 | vx 573, ld/st 101, mtx 0, txf 0, bar 0 |
| op.bmm_small_fp16-b8-m64-k32-n64 | `gemm_pipe` (fused_bmm_0) | 46 | 61 % | 6 | 2 | 3 | 3 | 8 | 6 | 0 | vx 0, ld/st 0, mtx 2, txf 8, bar 8 |
| op.conv2d_small_fp32-n1-cin16-h16-w16-cout16 | `for_1` (fused_convolution_2) | 1748 | 20 % | 0 | 4 | 2 | 120 | 2 | 134 | 96 | vx 1116, ld/st 130, mtx 144, txf 0, bar 0 |
| op.cumsum_small_fp32-r64-c256 | `for_1` (fused_cumsum_0) | 98 | 5 % | 3 | 1 | 1 | 0 | 0 | 0 | 0 | vx 91, ld/st 2, mtx 0, txf 0, bar 0 |
| op.embedding_small_fp16-v1024-d128-t64 | `__rctx_program` (fused_embedding_0) | 681 | 47 % | 0 | 35 | 5 | 3 | 1 | 276 | 1 | vx 356, ld/st 4, mtx 0, txf 0, bar 0 |
| op.gelu_tanh_small_fp16-n16384 | `tiles_1` (fused_gelu_0) | 540 | 2 % | 1 | 1 | 1 | 0 | 0 | 9 | 0 | vx 496, ld/st 32, mtx 0, txf 0, bar 0 |
| op.gemv_small_fp16-k256-n512 | `gemm_pipe` (fused_mm_t_0) | 38 | 53 % | 2 | 2 | 3 | 3 | 8 | 2 | 0 | vx 0, ld/st 0, mtx 2, txf 8, bar 8 |
| op.layernorm_small_fp16-r64-d256 | `for_5` (fused_native_layer_norm_0) | 55 | 13 % | 5 | 1 | 1 | 0 | 0 | 0 | 0 | vx 32, ld/st 16, mtx 0, txf 0, bar 0 |
| op.linear_small_fp16-m64-k256-n256 | `gemm_pipe` (fused_addmm_t_0) | 38 | 53 % | 2 | 2 | 3 | 3 | 8 | 2 | 0 | vx 0, ld/st 0, mtx 2, txf 8, bar 8 |
| op.matmul_small_fp16-m64-k128-n128 | `gemm_pipe` (fused_mm_0) | 38 | 53 % | 2 | 2 | 3 | 3 | 8 | 2 | 0 | vx 0, ld/st 0, mtx 2, txf 8, bar 8 |
| op.matmul_small_fp32-m64-k128-n128 | `for_5` (fused_mm_0) | 610 | 4 % | 1 | 1 | 2 | 16 | 4 | 1 | 0 | vx 545, ld/st 0, mtx 32, txf 4, bar 4 |
| op.rmsnorm_small_fp16-r64-d256 | `for_3` (fused__to_copy_add_mean_mul_) | 38 | 16 % | 4 | 1 | 1 | 0 | 0 | 0 | 0 | vx 20, ld/st 12, mtx 0, txf 0, bar 0 |
| op.rope_small_fp16-b1-h4-s64-d32 | `tiles_1` (fused_add_mul_2) | 336 | 4 % | 1 | 1 | 1 | 0 | 0 | 11 | 0 | vx 242, ld/st 80, mtx 0, txf 0, bar 0 |
| op.silu_small_fp16-n16384 | `tiles_1` (fused_silu_0) | 332 | 4 % | 1 | 1 | 1 | 0 | 0 | 9 | 0 | vx 288, ld/st 32, mtx 0, txf 0, bar 0 |
| op.softmax_small_fp16-r64-c512 | `for_3` (fused__softmax_0) | 53 | 9 % | 3 | 1 | 1 | 0 | 0 | 0 | 0 | vx 40, ld/st 8, mtx 0, txf 0, bar 0 |
| op.swiglu_small_fp16-n16384 | `tiles_1` (fused_mul_silu_0) | 382 | 4 % | 1 | 1 | 1 | 0 | 0 | 11 | 0 | vx 320, ld/st 48, mtx 0, txf 0, bar 0 |
| op.transpose_small_fp32-r128-c256 | `__rctx_program` (fused_clone_t_0) | 325 | 43 % | 0 | 1 | 1 | 33 | 1 | 71 | 32 | vx 118, ld/st 34, mtx 32, txf 0, bar 0 |
