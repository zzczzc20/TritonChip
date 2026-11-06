## `run-usage/report.json`

label `g2-usage`, software tree `files:9458e56b1fb26cab97edebc42db8b33cbeb290a92b7dcdb72839ad5b07e82f67`, profile `core-vrf-sync-v5`; points 113, status {'pass': 113}

497 launched kernels.

| Resource | Distribution over kernels (value: kernels) | Maximum | Where |
|---|---|---:|---|
| accumulators | 0: 335, 1: 103, 2: 10, 3: 9, 5: 36, 6: 4 | 6 | `model.resnet_block/medium/fp32-n1-c24-hw16 :: triton_tem_fused__native_batch_norm_legit_no_training_add_convolution_relu_3` |
| barriers | 0: 373, 1: 4, 2: 7, 3: 113 | 3 | `model.gpt2_block/tiny/fp16-d64-h4-b1-s16 :: triton_tem_fused_addmm_native_layer_norm_t_view_1` |
| descriptors | 0: 265, 1: 33, 2: 165, 3: 18, 4: 8, 5: 8 | 5 | `model.llama_decode/medium/fp16-b1-p256 :: triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_28` |

role-SPMD contexts per kernel (decision `rctx_spmd`): none: 46, 2: 247, 4: 204

| Operator | max accumulators | max barriers | max descriptors | max shared windows | max roles |
|---|---:|---:|---:|---:|---:|
| model.gpt2_block | 5 | 3 | 2 | 0 | 4 |
| model.llama_decode | 2 | 3 | 5 | 0 | 4 |
| model.llama_prefill | 5 | 3 | 5 | 0 | 4 |
| model.mlp | 5 | 3 | 2 | 0 | 2 |
| model.resnet_block | 6 | 0 | 1 | 0 | 2 |
| op.add | 0 | 0 | 0 | 0 | 4 |
| op.argmax | 0 | 0 | 0 | 0 | 4 |
| op.attention_decode | 0 | 0 | 1 | 0 | 4 |
| op.attention_prefill | 5 | 2 | 2 | 0 | 4 |
| op.bmm | 1 | 3 | 2 | 0 | 2 |
| op.conv2d | 5 | 0 | 1 | 0 | 2 |
| op.copy | 0 | 0 | 0 | 0 | 4 |
| op.cumsum | 0 | 0 | 0 | 0 | 4 |
| op.embedding | 0 | 0 | 0 | 0 | 4 |
| op.gelu_tanh | 0 | 0 | 0 | 0 | 4 |
| op.gemv | 1 | 3 | 2 | 0 | 2 |
| op.layernorm | 0 | 0 | 0 | 0 | 4 |
| op.linear | 1 | 3 | 2 | 0 | 2 |
| op.matmul | 3 | 3 | 2 | 0 | 2 |
| op.rmsnorm | 0 | 0 | 0 | 0 | 4 |
| op.rope | 0 | 0 | 0 | 0 | 4 |
| op.silu | 0 | 0 | 0 | 0 | 4 |
| op.softmax | 0 | 0 | 1 | 0 | 4 |
| op.swiglu | 0 | 0 | 0 | 0 | 4 |
| op.transpose | 1 | 0 | 0 | 0 | 2 |

