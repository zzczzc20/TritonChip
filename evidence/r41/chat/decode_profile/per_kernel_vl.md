#### Per kernel: steady decode step 3, 1 profiled layer(s) (domain: TLM)

Footprint = bytes of the tensor arguments (weights / activations / KV cache or linear-attention state), each distinct view once. NUCA bytes = core word requests into the NUCA homes x 8 B (TLM-measured, includes re-reads); TTE bytes = the tensor-transfer engine's share of them; DRAM = 64-byte lines at the outlets (reads + writes).

| # | Part | Class | Kernel | Grid | Cycles | Footprint W / A / KV B | NUCA B | NUCA B/cyc | TTE B | DRAM B | DRAM B/cyc | Code load / evicted |
|---:|---|---|---|---|---:|---|---:|---:|---:|---:|---:|---|
| 99 | fixed | other | `triton_poi_fused__to_copy_expand_mul_select_slice_unsqueeze_0` | 12x1x1 | 14,191 | 256 / 1,568 / 0 | 3,552 | 0.250 | 0 | 2,176 | 0.153 | load / 4 |
| 100 | layer L0 | norm | `triton_red_fused__to_copy_add_embedding_mean_mul_pow_rsqrt_1` | 1x1x1 | 257,143 | 622,333,952 / 12,296 / 0 | 24,640 | 0.096 | 0 | 21,184 | 0.082 | load / 0 |
| 101 | layer L0 | fp8_gemv | `w8a8_block_dynamic_fp8_matmul_kernel` | 1x16x1 | 9,396,737 | 4,195,328 / 8,192 / 0 | 21,154,944 | 2.251 | 4,325,376 | 30,191,168 | 3.213 | load / 4 |
| 102 | layer L0 | fp8_gemv | `w8a8_block_dynamic_fp8_matmul_kernel` | 1x8x1 | 4,900,293 | 2,097,664 / 6,144 / 0 | 10,577,472 | 2.159 | 2,162,688 | 13,549,504 | 2.765 | - / 0 |
| 103 | layer L0 | fp8_gemv | `w8a8_block_dynamic_fp8_matmul_kernel` | 1x8x1 | 4,837,147 | 2,097,664 / 6,144 / 0 | 10,577,472 | 2.187 | 2,162,688 | 12,171,008 | 2.516 | - / 0 |
| 104 | layer | norm | `triton_red_fused__to_copy_mean_pow_view_2` | 16x1x1 | 5,778 | 0 / 4,160 / 0 | 9,600 | 1.661 | 4,096 | 7,424 | 1.285 | load / 0 |
| 105 | layer L0 | norm_rotary | `triton_poi_fused__to_copy_add_cat_mean_mul_neg_pow_rsqrt_slice_transpose_view_3` | 64x1x1 | 4,730 | 256 / 6,208 / 0 | 15,360 | 3.247 | 0 | 5,696 | 1.204 | load / 0 |
| 106 | layer | norm | `triton_red_fused__to_copy_mean_pow_view_4` | 8x1x1 | 3,247 | 0 / 2,080 / 0 | 4,800 | 1.478 | 2,048 | 3,776 | 1.163 | load / 0 |
| 107 | layer L0 | norm_rotary | `triton_poi_fused__to_copy_add_cat_mean_mul_neg_pow_rsqrt_slice_transpose_view_5` | 32x1x1 | 2,010 | 256 / 3,104 / 0 | 7,680 | 3.821 | 0 | 640 | 0.318 | load / 0 |
| 108 | layer L0 | attention | `triton_poi_fused__to_copy_add_arange_cat_index_copy_mean_mul_pow_rsqrt_select_slice_transpose_unsqueeze_view_6` | 64x1x1 | 87,177 | 256 / 5,672 / 131,072 | 26,624 | 0.305 | 0 | 6,144 | 0.070 | load / 0 |
| 109 | layer L0 | attention | `triton_red_fused__scaled_dot_product_flash_attention_for_cpu__to_copy__unsafe_view_add_cat_clone_expand_mean_mul_pow_rsqrt_select_slice_transpose_unsqueeze_view_7` | 1024x1x1 | 6,042,470 | 256 / 13,888 / 131,072 | 3,481,600 | 0.576 | 524,288 | 198,016 | 0.033 | load / 1 |
| 110 | layer | attention | `triton_red_fused__scaled_dot_product_flash_attention_for_cpu_add_arange_expand_le_scalar_tensor_unsqueeze_where_8` | 16x1x1 | 24,703 | 0 / 8,280 / 0 | 18,560 | 0.751 | 4,096 | 5,056 | 0.205 | load / 1 |
| 111 | layer L0 | attention | `triton_poi_fused_add_arange_index_copy_transpose_view_9` | 64x1x1 | 38,635 | 0 / 2,056 / 131,072 | 6,144 | 0.159 | 0 | 4,352 | 0.113 | load / 0 |
| 112 | layer L0 | attention | `triton_red_fused__scaled_dot_product_flash_attention_for_cpu__unsafe_view_clone_expand_transpose_unsqueeze_view_10` | 2048x1x1 | 712,158 | 0 / 8,272 / 131,072 | 2,359,296 | 3.313 | 524,288 | 154,880 | 0.217 | load / 0 |
| 113 | layer L0 | fp8_gemv | `w8a8_block_dynamic_fp8_matmul_kernel` | 1x16x1 | 9,393,048 | 4,195,328 / 8,192 / 0 | 21,154,944 | 2.252 | 4,325,376 | 30,130,944 | 3.208 | load / 5 |
| 114 | layer L0 | norm | `triton_red_fused__to_copy_add_embedding_mean_mul_pow_rsqrt_11` | 1x1x1 | 262,766 | 622,333,952 / 8,200 / 0 | 24,936 | 0.095 | 256 | 21,056 | 0.080 | load / 2 |
| 115 | layer L0 | fp8_gemv | `w8a8_block_dynamic_fp8_matmul_kernel` | 1x48x1 | 27,878,350 | 12,585,984 / 16,384 / 0 | 63,464,832 | 2.276 | 12,976,128 | 93,156,480 | 3.342 | - / 0 |
| 116 | layer L0 | fp8_gemv | `w8a8_block_dynamic_fp8_matmul_kernel` | 1x48x1 | 27,765,311 | 12,585,984 / 16,384 / 0 | 63,464,832 | 2.286 | 12,976,128 | 89,677,120 | 3.230 | - / 0 |
| 117 | layer | act_gate | `triton_poi_fused_mul_silu_12` | 384x1x1 | 21,615 | 0 / 36,864 / 0 | 46,080 | 2.132 | 0 | 44,416 | 2.055 | load / 0 |
| 118 | layer L0 | fp8_gemv | `w8a8_block_dynamic_fp8_matmul_kernel` | 1x16x1 | 28,041,496 | 12,585,984 / 16,384 / 0 | 63,454,336 | 2.263 | 12,976,128 | 77,675,008 | 2.770 | - / 0 |
| 119 | fixed | norm | `triton_red_fused__to_copy_add_embedding_mean_mul_pow_rsqrt_13` | 1x1x1 | 274,389 | 622,333,952 / 12,296 / 0 | 33,432 | 0.122 | 512 | 23,936 | 0.087 | load / 1 |
| 120 | fixed | bf16_gemv | `triton_tem_fused__to_copy_add_embedding_mean_mm_mul_pow_rsqrt_slice_t_view_14` | 9496x1x1 | 420,623,160 | 622,329,856 / 307,968 / 0 | 2,298,411,840 | 5.464 | 1,558,255,616 | 1,480,819,328 | 3.521 | load / 0 |
| 121 | fixed | copy_cast | `triton_poi_fused_add_copy__15` | 1x1x1 | 173 | 0 / 8 / 0 | 40 | 0.231 | 0 | 128 | 0.740 | load / 0 |

