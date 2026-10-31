#### Per kernel: steady decode step 3, 4 profiled layer(s) (domain: TLM)

Footprint = bytes of the tensor arguments (weights / activations / KV cache or linear-attention state), each distinct view once. NUCA bytes = core word requests into the NUCA homes x 8 B (TLM-measured, includes re-reads); TTE bytes = the tensor-transfer engine's share of them; DRAM = 64-byte lines at the outlets (reads + writes).

| # | Part | Class | Kernel | Grid | Cycles | Footprint W / A / KV B | NUCA B | NUCA B/cyc | TTE B | DRAM B | DRAM B/cyc | Code load / evicted |
|---:|---|---|---|---|---:|---|---:|---:|---:|---:|---:|---|
| 447 | fixed | other | `triton_poi_fused__to_copy_expand_mul_select_slice_unsqueeze_0` | 6x1x1 | 13,628 | 128 / 800 / 0 | 1,800 | 0.132 | 0 | 1,408 | 0.103 | load / 0 |
| 448 | layer L0 | norm | `triton_red_fused__to_copy_add_embedding_mean_mul_pow_rsqrt_1` | 1x1x1 | 330,133 | 1,271,403,520 / 15,368 / 0 | 30,784 | 0.093 | 0 | 37,440 | 0.113 | load / 0 |
| 449 | layer L0 | fp8_gemv | `w8a8_block_dynamic_fp8_matmul_kernel` | 1x64x1 | 43,654,656 | 20,976,640 / 21,504 / 0 | 105,769,472 | 2.423 | 21,626,880 | 115,401,344 | 2.644 | - / 0 |
| 450 | layer L0 | fp8_gemv | `w8a8_block_dynamic_fp8_matmul_kernel` | 1x32x1 | 22,408,329 | 10,488,320 / 13,312 / 0 | 52,884,736 | 2.360 | 10,813,440 | 57,012,288 | 2.544 | - / 0 |
| 451 | layer L0 | linear_attn | `triton_poi_fused_cat_transpose_2` | 2048x1x1 | 82,820 | 0 / 65,536 / 65,536 | 180,224 | 2.176 | 0 | 227,712 | 2.749 | load / 0 |
| 452 | layer | copy_cast | `triton_poi_fused_cat_transpose_3` | 512x1x1 | 32,254 | 0 / 32,768 / 0 | 90,112 | 2.794 | 0 | 165,888 | 5.143 | load / 0 |
| 453 | layer L0 | linear_attn | `triton_poi_fused_convolution_silu_slice_squeeze_unsqueeze_4` | 512x1x1 | 93,196 | 65,536 / 98,304 / 0 | 552,960 | 5.933 | 0 | 217,344 | 2.332 | load / 0 |
| 454 | layer | linear_attn | `triton_red_fused__to_copy_clone_expand_mul_silu_split_with_sizes_sum_transpose_unsqueeze_view_5` | 32x1x1 | 10,761 | 0 / 16,512 / 0 | 19,200 | 1.784 | 8,192 | 6,528 | 0.607 | load / 0 |
| 455 | layer L0 | linear_attn | `triton_tem_fused__to_copy__unsafe_view_add_embedding_exp_mean_mm_mul_neg_pow_rsqrt_select_softplus_t_transpose_view_6` | 2x1x1 | 347,692 | 163,968 / 5,248 / 0 | 605,136 | 1.740 | 410,112 | 198,336 | 0.570 | load / 7 |
| 456 | layer L0 | linear_attn | `triton_red_fused__to_copy__unsafe_view_add_clone_exp_expand_mul_neg_rsqrt_select_silu_softplus_split_with_sizes_sum_transpose_unsqueeze_view_7` | 4096x1x1 | 2,303,338 | 0 / 33,024 / 2,097,152 | 6,815,744 | 2.959 | 1,048,576 | 7,412,288 | 3.218 | load / 0 |
| 457 | layer L0 | bf16_gemv | `triton_tem_fused__to_copy__unsafe_view_mm_sigmoid_t_transpose_view_8` | 2x1x1 | 329,084 | 163,840 / 5,248 / 0 | 604,976 | 1.838 | 410,112 | 185,472 | 0.564 | load / 1 |
| 458 | layer L0 | linear_attn | `triton_poi_fused__to_copy__unsafe_view_add_clone_copy__exp_expand_mul_neg_rsqrt_select_sigmoid_silu_softplus_split_with_sizes_sub_transpose_unsqueeze_view_9` | 32768x1x1 | 4,113,351 | 0 / 2,130,304 / 2,097,152 | 12,582,912 | 3.059 | 0 | 8,458,688 | 2.056 | load / 0 |
| 459 | layer | linear_attn | `triton_red_fused__to_copy_clone_expand_mul_silu_split_with_sizes_sum_transpose_unsqueeze_view_10` | 32x1x1 | 11,434 | 0 / 16,512 / 0 | 19,200 | 1.679 | 8,192 | 11,456 | 1.002 | load / 0 |
| 460 | layer | linear_attn | `triton_red_fused__to_copy_add_clone_div_expand_mul_rsqrt_select_silu_split_with_sizes_sum_transpose_unsqueeze_view_11` | 4096x1x1 | 2,345,136 | 0 / 2,130,048 / 0 | 6,750,208 | 2.878 | 1,048,576 | 7,921,280 | 3.378 | load / 0 |
| 461 | layer | norm | `triton_red_fused__to_copy_mean_pow_transpose_view_zeros_like_12` | 32x1x1 | 10,754 | 0 / 16,512 / 0 | 27,392 | 2.547 | 8,192 | 17,920 | 1.666 | load / 0 |
| 462 | layer L0 | norm | `triton_poi_fused__to_copy_add_mean_mul_pow_rsqrt_silu_transpose_view_zeros_like_13` | 256x1x1 | 16,590 | 256 / 32,896 / 0 | 53,248 | 3.210 | 0 | 15,232 | 0.918 | load / 0 |
| 463 | layer L0 | fp8_gemv | `w8a8_block_dynamic_fp8_matmul_kernel` | 1x20x1 | 23,625,272 | 10,488,320 / 13,312 / 0 | 52,880,800 | 2.238 | 10,813,440 | 73,156,416 | 3.097 | load / 1 |
| 464 | layer L0 | norm | `triton_red_fused__to_copy_add_embedding_mean_mul_pow_rsqrt_14` | 1x1x1 | 329,133 | 1,271,403,520 / 10,248 / 0 | 31,080 | 0.094 | 256 | 24,768 | 0.075 | load / 5 |
| 465 | layer L0 | fp8_gemv | `w8a8_block_dynamic_fp8_matmul_kernel` | 1x72x1 | 49,500,355 | 23,598,720 / 23,552 / 0 | 118,990,656 | 2.404 | 24,330,240 | 128,944,256 | 2.605 | load / 8 |
| 466 | layer L0 | fp8_gemv | `w8a8_block_dynamic_fp8_matmul_kernel` | 1x72x1 | 49,432,732 | 23,598,720 / 23,552 / 0 | 118,990,656 | 2.407 | 24,330,240 | 129,124,096 | 2.612 | - / 0 |
| 467 | layer | act_gate | `triton_poi_fused_mul_silu_15` | 576x1x1 | 37,718 | 0 / 55,296 / 0 | 69,120 | 1.833 | 0 | 96,960 | 2.571 | load / 0 |
| 468 | layer L0 | fp8_gemv | `w8a8_block_dynamic_fp8_matmul_kernel` | 1x20x1 | 51,388,041 | 23,598,720 / 23,552 / 0 | 118,973,600 | 2.315 | 24,330,240 | 127,614,528 | 2.483 | - / 0 |
| 469 | layer L1 | norm | `triton_red_fused__to_copy_add_embedding_mean_mul_pow_rsqrt_16` | 1x1x1 | 345,987 | 1,271,403,520 / 15,368 / 0 | 41,624 | 0.120 | 512 | 32,064 | 0.093 | load / 0 |
| 470 | layer L1 | fp8_gemv | `w8a8_block_dynamic_fp8_matmul_kernel` | 1x64x1 | 43,642,860 | 20,976,640 / 21,504 / 0 | 105,769,472 | 2.424 | 21,626,880 | 115,051,136 | 2.636 | - / 0 |
| 471 | layer L1 | fp8_gemv | `w8a8_block_dynamic_fp8_matmul_kernel` | 1x32x1 | 22,422,232 | 10,488,320 / 13,312 / 0 | 52,884,736 | 2.359 | 10,813,440 | 56,916,032 | 2.538 | - / 0 |
| 472 | layer L1 | linear_attn | `triton_poi_fused_cat_transpose_2` | 2048x1x1 | 73,242 | 0 / 65,536 / 65,536 | 180,224 | 2.461 | 0 | 179,584 | 2.452 | load / 0 |
| 473 | layer | copy_cast | `triton_poi_fused_cat_transpose_3` | 512x1x1 | 31,936 | 0 / 32,768 / 0 | 90,112 | 2.822 | 0 | 166,848 | 5.224 | load / 0 |
| 474 | layer L1 | linear_attn | `triton_poi_fused_convolution_silu_slice_squeeze_unsqueeze_4` | 512x1x1 | 93,214 | 65,536 / 98,304 / 0 | 552,960 | 5.932 | 0 | 217,280 | 2.331 | load / 0 |
| 475 | layer | linear_attn | `triton_red_fused__to_copy_clone_expand_mul_silu_split_with_sizes_sum_transpose_unsqueeze_view_5` | 32x1x1 | 10,695 | 0 / 16,512 / 0 | 19,200 | 1.795 | 8,192 | 6,336 | 0.592 | load / 0 |
| 476 | layer L1 | linear_attn | `triton_tem_fused__to_copy__unsafe_view_add_exp_mm_mul_neg_select_softplus_t_transpose_view_17` | 2x1x1 | 318,677 | 163,968 / 5,248 / 0 | 605,136 | 1.899 | 410,112 | 217,984 | 0.684 | load / 3 |
| 477 | layer L1 | linear_attn | `triton_red_fused__to_copy__unsafe_view_add_clone_exp_expand_mul_neg_rsqrt_select_silu_softplus_split_with_sizes_sum_transpose_unsqueeze_view_7` | 4096x1x1 | 2,336,461 | 0 / 33,024 / 2,097,152 | 6,815,744 | 2.917 | 1,048,576 | 7,354,496 | 3.148 | load / 0 |
| 478 | layer L1 | bf16_gemv | `triton_tem_fused__to_copy__unsafe_view_mm_sigmoid_t_transpose_view_8` | 2x1x1 | 317,267 | 163,840 / 5,248 / 0 | 604,976 | 1.907 | 410,112 | 212,672 | 0.670 | load / 0 |
| 479 | layer L1 | linear_attn | `triton_poi_fused__to_copy__unsafe_view_add_clone_copy__exp_expand_mul_neg_rsqrt_select_sigmoid_silu_softplus_split_with_sizes_sub_transpose_unsqueeze_view_9` | 32768x1x1 | 4,011,071 | 0 / 2,130,304 / 2,097,152 | 12,582,912 | 3.137 | 0 | 8,457,984 | 2.109 | load / 0 |
| 480 | layer | linear_attn | `triton_red_fused__to_copy_clone_expand_mul_silu_split_with_sizes_sum_transpose_unsqueeze_view_10` | 32x1x1 | 10,723 | 0 / 16,512 / 0 | 19,200 | 1.791 | 8,192 | 11,520 | 1.074 | load / 0 |
| 481 | layer | linear_attn | `triton_red_fused__to_copy_add_clone_div_expand_mul_rsqrt_select_silu_split_with_sizes_sum_transpose_unsqueeze_view_11` | 4096x1x1 | 2,345,651 | 0 / 2,130,048 / 0 | 6,750,208 | 2.878 | 1,048,576 | 7,421,440 | 3.164 | load / 0 |
| 482 | layer | norm | `triton_red_fused__to_copy_mean_pow_transpose_view_zeros_like_12` | 32x1x1 | 11,101 | 0 / 16,512 / 0 | 27,392 | 2.468 | 8,192 | 17,792 | 1.603 | load / 0 |
| 483 | layer L1 | norm | `triton_poi_fused__to_copy_add_mean_mul_pow_rsqrt_silu_transpose_view_zeros_like_13` | 256x1x1 | 16,449 | 256 / 32,896 / 0 | 53,248 | 3.237 | 0 | 17,024 | 1.035 | load / 0 |
| 484 | layer L1 | fp8_gemv | `w8a8_block_dynamic_fp8_matmul_kernel` | 1x20x1 | 23,610,760 | 10,488,320 / 13,312 / 0 | 52,880,800 | 2.240 | 10,813,440 | 73,128,128 | 3.097 | load / 1 |
| 485 | layer L1 | norm | `triton_red_fused__to_copy_add_embedding_mean_mul_pow_rsqrt_18` | 1x1x1 | 230,622 | 1,271,403,520 / 30,728 / 0 | 47,352 | 0.205 | 1,024 | 48,448 | 0.210 | load / 0 |
| 486 | layer L1 | fp8_gemv | `w8a8_block_dynamic_fp8_matmul_kernel` | 1x72x1 | 49,412,287 | 23,598,720 / 23,552 / 0 | 118,990,656 | 2.408 | 24,330,240 | 128,928,960 | 2.609 | load / 13 |
| 487 | layer L1 | fp8_gemv | `w8a8_block_dynamic_fp8_matmul_kernel` | 1x72x1 | 49,411,648 | 23,598,720 / 23,552 / 0 | 118,990,656 | 2.408 | 24,330,240 | 129,049,664 | 2.612 | - / 0 |
| 488 | layer | act_gate | `triton_poi_fused_mul_silu_15` | 576x1x1 | 38,528 | 0 / 55,296 / 0 | 69,120 | 1.794 | 0 | 101,440 | 2.633 | load / 0 |
| 489 | layer L1 | fp8_gemv | `w8a8_block_dynamic_fp8_matmul_kernel` | 1x20x1 | 51,337,236 | 23,598,720 / 23,552 / 0 | 118,973,600 | 2.317 | 24,330,240 | 127,813,184 | 2.490 | - / 0 |
| 490 | layer L2 | norm | `triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_19` | 1x1x1 | 89,243 | 5,120 / 25,600 / 0 | 41,608 | 0.466 | 512 | 41,088 | 0.460 | load / 0 |
| 491 | layer L2 | fp8_gemv | `w8a8_block_dynamic_fp8_matmul_kernel` | 1x64x1 | 43,657,455 | 20,976,640 / 21,504 / 0 | 105,769,472 | 2.423 | 21,626,880 | 115,104,448 | 2.637 | - / 0 |
| 492 | layer L2 | fp8_gemv | `w8a8_block_dynamic_fp8_matmul_kernel` | 1x32x1 | 22,401,244 | 10,488,320 / 13,312 / 0 | 52,884,736 | 2.361 | 10,813,440 | 56,883,200 | 2.539 | - / 0 |
| 493 | layer L2 | linear_attn | `triton_poi_fused_cat_transpose_2` | 2048x1x1 | 73,145 | 0 / 65,536 / 65,536 | 180,224 | 2.464 | 0 | 181,632 | 2.483 | load / 0 |
| 494 | layer | copy_cast | `triton_poi_fused_cat_transpose_3` | 512x1x1 | 31,986 | 0 / 32,768 / 0 | 90,112 | 2.817 | 0 | 166,656 | 5.210 | load / 0 |
| 495 | layer L2 | linear_attn | `triton_poi_fused_convolution_silu_slice_squeeze_unsqueeze_4` | 512x1x1 | 91,586 | 65,536 / 98,304 / 0 | 552,960 | 6.038 | 0 | 216,832 | 2.368 | load / 0 |
| 496 | layer | linear_attn | `triton_red_fused__to_copy_clone_expand_mul_silu_split_with_sizes_sum_transpose_unsqueeze_view_5` | 32x1x1 | 10,706 | 0 / 16,512 / 0 | 19,200 | 1.793 | 8,192 | 6,400 | 0.598 | load / 0 |
| 497 | layer L2 | linear_attn | `triton_tem_fused__to_copy__unsafe_view_add_embedding_exp_mean_mm_mul_neg_pow_rsqrt_select_softplus_t_transpose_view_6` | 2x1x1 | 328,398 | 163,968 / 5,248 / 0 | 605,136 | 1.843 | 410,112 | 192,320 | 0.586 | load / 0 |
| 498 | layer L2 | linear_attn | `triton_red_fused__to_copy__unsafe_view_add_clone_exp_expand_mul_neg_rsqrt_select_silu_softplus_split_with_sizes_sum_transpose_unsqueeze_view_7` | 4096x1x1 | 2,353,961 | 0 / 33,024 / 2,097,152 | 6,815,744 | 2.895 | 1,048,576 | 7,328,960 | 3.113 | load / 0 |
| 499 | layer L2 | bf16_gemv | `triton_tem_fused__to_copy__unsafe_view_mm_sigmoid_t_transpose_view_8` | 2x1x1 | 334,986 | 163,840 / 5,248 / 0 | 604,976 | 1.806 | 410,112 | 197,248 | 0.589 | load / 0 |
| 500 | layer L2 | linear_attn | `triton_poi_fused__to_copy__unsafe_view_add_clone_copy__exp_expand_mul_neg_rsqrt_select_sigmoid_silu_softplus_split_with_sizes_sub_transpose_unsqueeze_view_9` | 32768x1x1 | 4,017,055 | 0 / 2,130,304 / 2,097,152 | 12,582,912 | 3.132 | 0 | 8,460,224 | 2.106 | load / 0 |
| 501 | layer | linear_attn | `triton_red_fused__to_copy_clone_expand_mul_silu_split_with_sizes_sum_transpose_unsqueeze_view_10` | 32x1x1 | 10,958 | 0 / 16,512 / 0 | 19,200 | 1.752 | 8,192 | 11,072 | 1.010 | load / 0 |
| 502 | layer | linear_attn | `triton_red_fused__to_copy_add_clone_div_expand_mul_rsqrt_select_silu_split_with_sizes_sum_transpose_unsqueeze_view_11` | 4096x1x1 | 2,364,487 | 0 / 2,130,048 / 0 | 6,750,208 | 2.855 | 1,048,576 | 7,388,480 | 3.125 | load / 0 |
| 503 | layer | norm | `triton_red_fused__to_copy_mean_pow_transpose_view_zeros_like_12` | 32x1x1 | 10,670 | 0 / 16,512 / 0 | 27,392 | 2.567 | 8,192 | 16,576 | 1.554 | load / 1 |
| 504 | layer L2 | norm | `triton_poi_fused__to_copy_add_mean_mul_pow_rsqrt_silu_transpose_view_zeros_like_13` | 256x1x1 | 17,205 | 256 / 32,896 / 0 | 53,248 | 3.095 | 0 | 16,960 | 0.986 | load / 0 |
| 505 | layer L2 | fp8_gemv | `w8a8_block_dynamic_fp8_matmul_kernel` | 1x20x1 | 23,492,180 | 10,488,320 / 13,312 / 0 | 52,880,800 | 2.251 | 10,813,440 | 70,218,624 | 2.989 | load / 3 |
| 506 | layer L2 | norm | `triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_20` | 1x1x1 | 93,224 | 5,120 / 20,480 / 0 | 41,896 | 0.449 | 768 | 33,664 | 0.361 | load / 0 |
| 507 | layer L2 | fp8_gemv | `w8a8_block_dynamic_fp8_matmul_kernel` | 1x72x1 | 49,450,523 | 23,598,720 / 23,552 / 0 | 118,990,656 | 2.406 | 24,330,240 | 129,048,640 | 2.610 | load / 13 |
| 508 | layer L2 | fp8_gemv | `w8a8_block_dynamic_fp8_matmul_kernel` | 1x72x1 | 49,456,269 | 23,598,720 / 23,552 / 0 | 118,990,656 | 2.406 | 24,330,240 | 129,012,416 | 2.609 | - / 0 |
| 509 | layer | act_gate | `triton_poi_fused_mul_silu_15` | 576x1x1 | 37,989 | 0 / 55,296 / 0 | 69,120 | 1.819 | 0 | 100,416 | 2.643 | load / 0 |
| 510 | layer L2 | fp8_gemv | `w8a8_block_dynamic_fp8_matmul_kernel` | 1x20x1 | 51,343,874 | 23,598,720 / 23,552 / 0 | 118,973,600 | 2.317 | 24,330,240 | 127,609,792 | 2.485 | - / 0 |
| 511 | layer L3 | norm | `triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_21` | 1x1x1 | 109,773 | 5,120 / 25,600 / 0 | 52,440 | 0.478 | 1,024 | 42,048 | 0.383 | load / 0 |
| 512 | layer L3 | fp8_gemv | `w8a8_block_dynamic_fp8_matmul_kernel` | 1x64x1 | 43,654,485 | 20,976,640 / 21,504 / 0 | 105,769,472 | 2.423 | 21,626,880 | 115,133,056 | 2.637 | - / 0 |
| 513 | layer L3 | fp8_gemv | `w8a8_block_dynamic_fp8_matmul_kernel` | 1x8x1 | 5,993,762 | 2,622,080 / 7,168 / 0 | 13,221,184 | 2.206 | 2,703,360 | 13,995,392 | 2.335 | - / 0 |
| 514 | layer L3 | fp8_gemv | `w8a8_block_dynamic_fp8_matmul_kernel` | 1x8x1 | 5,994,692 | 2,622,080 / 7,168 / 0 | 13,221,184 | 2.205 | 2,703,360 | 13,959,104 | 2.329 | - / 0 |
| 515 | layer | norm | `triton_red_fused__to_copy_mean_pow_split_view_22` | 16x1x1 | 8,604 | 0 / 16,448 / 0 | 13,696 | 1.592 | 4,096 | 13,184 | 1.532 | load / 0 |
| 516 | layer L3 | norm_rotary | `triton_poi_fused__to_copy_add_cat_mean_mul_neg_pow_rsqrt_slice_split_transpose_view_23` | 32x1x1 | 2,310 | 512 / 17,472 / 0 | 7,680 | 3.325 | 0 | 1,344 | 0.582 | load / 0 |
| 517 | layer L3 | norm_rotary | `triton_poi_fused__to_copy_add_cat_mean_mul_pow_rsqrt_select_slice_split_transpose_unsqueeze_view_24` | 64x1x1 | 49,947 | 512 / 21,312 / 0 | 25,600 | 0.513 | 0 | 2,816 | 0.056 | load / 0 |
| 518 | layer L3 | norm_rotary | `triton_poi_fused__to_copy_add_cat_mean_mul_pow_rsqrt_select_slice_split_transpose_unsqueeze_view_25` | 192x1x1 | 92,720 | 512 / 22,592 / 0 | 26,112 | 0.282 | 0 | 9,536 | 0.103 | load / 0 |
| 519 | layer | norm | `triton_red_fused__to_copy_mean_pow_view_26` | 4x1x1 | 3,222 | 0 / 2,064 / 0 | 3,424 | 1.063 | 1,024 | 4,864 | 1.510 | load / 0 |
| 520 | layer L3 | norm_rotary | `triton_poi_fused__to_copy_add_cat_mean_mul_neg_pow_rsqrt_slice_transpose_view_27` | 8x1x1 | 1,007 | 512 / 2,320 / 0 | 1,920 | 1.907 | 0 | 1,024 | 1.017 | load / 0 |
| 521 | layer L3 | norm_rotary | `triton_poi_fused__to_copy_add_cat_mean_mul_pow_rsqrt_select_slice_transpose_unsqueeze_view_28` | 16x1x1 | 16,501 | 512 / 3,856 / 0 | 6,400 | 0.388 | 0 | 0 | 0.000 | load / 0 |
| 522 | layer L3 | norm_rotary | `triton_poi_fused__to_copy_add_cat_mean_mul_pow_rsqrt_select_slice_transpose_unsqueeze_view_29` | 48x1x1 | 23,135 | 512 / 3,600 / 0 | 6,528 | 0.282 | 0 | 576 | 0.025 | load / 0 |
| 523 | layer L3 | attention | `triton_poi_fused_add_arange_index_copy_30` | 64x1x1 | 38,926 | 0 / 2,056 / 131,072 | 6,144 | 0.158 | 0 | 3,264 | 0.084 | load / 0 |
| 524 | layer L3 | attention | `triton_red_fused__scaled_dot_product_flash_attention_for_cpu__unsafe_view_clone_expand_unsqueeze_31` | 1024x1x1 | 736,336 | 0 / 12,288 / 131,072 | 1,409,024 | 1.914 | 262,144 | 190,592 | 0.259 | load / 0 |
| 525 | layer | attention | `triton_red_fused__scaled_dot_product_flash_attention_for_cpu_add_arange_expand_le_scalar_tensor_unsqueeze_where_32` | 16x1x1 | 25,285 | 0 / 8,280 / 0 | 18,560 | 0.734 | 4,096 | 7,744 | 0.306 | load / 4 |
| 526 | layer L3 | attention | `triton_poi_fused_add_arange_index_copy_transpose_view_33` | 64x1x1 | 38,724 | 0 / 2,056 / 131,072 | 6,144 | 0.159 | 0 | 4,544 | 0.117 | load / 0 |
| 527 | layer L3 | attention | `triton_red_fused__scaled_dot_product_flash_attention_for_cpu__unsafe_view_clone_expand_mul_sigmoid_split_transpose_unsqueeze_view_34` | 4096x1x1 | 1,587,601 | 0 / 28,752 / 131,072 | 4,784,128 | 3.013 | 1,048,576 | 347,840 | 0.219 | load / 0 |
| 528 | layer L3 | fp8_gemv | `w8a8_block_dynamic_fp8_matmul_kernel` | 1x20x1 | 23,554,369 | 10,488,320 / 13,312 / 0 | 52,880,800 | 2.245 | 10,813,440 | 71,997,952 | 3.057 | load / 8 |
| 529 | layer L3 | norm | `triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_35` | 1x1x1 | 117,226 | 5,120 / 35,840 / 0 | 52,752 | 0.450 | 1,280 | 59,456 | 0.507 | load / 0 |
| 530 | layer L3 | fp8_gemv | `w8a8_block_dynamic_fp8_matmul_kernel` | 1x72x1 | 49,467,606 | 23,598,720 / 23,552 / 0 | 118,990,656 | 2.405 | 24,330,240 | 128,970,304 | 2.607 | load / 6 |
| 531 | layer L3 | fp8_gemv | `w8a8_block_dynamic_fp8_matmul_kernel` | 1x72x1 | 49,476,994 | 23,598,720 / 23,552 / 0 | 118,990,656 | 2.405 | 24,330,240 | 129,044,416 | 2.608 | - / 0 |
| 532 | layer | act_gate | `triton_poi_fused_mul_silu_15` | 576x1x1 | 39,091 | 0 / 55,296 / 0 | 69,120 | 1.768 | 0 | 100,608 | 2.574 | load / 0 |
| 533 | layer L3 | fp8_gemv | `w8a8_block_dynamic_fp8_matmul_kernel` | 1x20x1 | 51,364,484 | 23,598,720 / 23,552 / 0 | 118,973,600 | 2.316 | 24,330,240 | 127,419,136 | 2.481 | - / 0 |
| 534 | fixed | norm | `triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_36` | 1x1x1 | 68,082 | 5,120 / 10,240 / 0 | 31,344 | 0.460 | 512 | 16,576 | 0.243 | load / 0 |
| 535 | fixed | bf16_gemv | `triton_tem_fused__to_copy_add_mean_mm_mul_pow_rsqrt_slice_t_view_37` | 15520x1x1 | 873,745,606 | 1,271,398,400 / 501,760 / 0 | 4,694,117,120 | 5.372 | 3,182,469,120 | 1,971,030,848 | 2.256 | load / 0 |
| 536 | fixed | copy_cast | `triton_poi_fused_add_copy__38` | 1x1x1 | 194 | 0 / 8 / 0 | 40 | 0.206 | 0 | 128 | 0.660 | load / 0 |
| 537 | layer L0 | linear_attn | `triton_poi_fused_copy_copy__slice_39` | 2048x1x1 | 79,348 | 0 / 81,920 / 65,536 | 180,224 | 2.271 | 0 | 193,088 | 2.433 | load / 0 |
| 538 | layer L1 | linear_attn | `triton_poi_fused_copy_copy__slice_39` | 2048x1x1 | 83,471 | 0 / 81,920 / 65,536 | 180,224 | 2.159 | 0 | 213,632 | 2.559 | - / 0 |
| 539 | layer L2 | linear_attn | `triton_poi_fused_copy_copy__slice_39` | 2048x1x1 | 82,705 | 0 / 81,920 / 65,536 | 180,224 | 2.179 | 0 | 213,312 | 2.579 | - / 0 |

