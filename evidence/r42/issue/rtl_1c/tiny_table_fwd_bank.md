| point | base cycles | VRF_FWD | bank alloc | VRF_FWD + bank alloc | x_vrf base -> fwd+bank |
|---|--:|--:|--:|--:|--:|
| model.gpt2_block/tiny/fp16-d64-h4-b1-s16 | 173772 | 1.045 | 1.020 | 1.068 | 14779 -> 4840 |
| model.llama_decode/tiny/fp16-b1-p16 | 177091 | 1.013 | 1.020 | 1.026 | 7218 -> 1707 |
| model.llama_prefill/tiny/fp16-b1-s16 | 473768 | 1.017 | 1.021 | 1.027 | 27845 -> 7675 |
| model.mlp/tiny/fp16-t16-d64 | 79981 | 1.070 | 1.014 | 1.092 | 7238 -> 2144 |
| model.resnet_block/tiny/fp32-n1-c8-hw8 | 246856 | 1.024 | 1.042 | 1.068 | 25055 -> 4454 |
| op.add/tiny/fp16-n1000 | 2373 | 1.003 | 1.012 | 1.015 | 67 -> 8 |
| op.add/tiny/fp32-n256 | 828 | 1.006 | 1.000 | 1.006 | 1 -> 1 |
| op.argmax/tiny/fp16-r2-v256 | 1999 | 1.003 | 1.023 | 1.017 | 213 -> 150 |
| op.attention_decode/tiny/fp16-b1-h2-l16-d16 | 9188 | 1.058 | 1.030 | 1.096 | 711 -> 154 |
| op.attention_prefill/tiny/fp16-b1-h2-s16-d16 | 23340 | 1.035 | 1.026 | 1.060 | 2574 -> 967 |
| op.bmm/tiny/fp16-b2-m16-k16-n16 | 1858 | 1.018 | 1.004 | 1.016 | 46 -> 10 |
| op.conv2d/tiny/fp32-n1-cin4-h8-w8-cout8 | 110375 | 1.022 | 1.055 | 1.102 | 11700 -> 2118 |
| op.cumsum/tiny/fp32-r4-c64 | 1078 | 1.000 | 1.043 | 1.043 | 44 -> 0 |
| op.embedding/tiny/fp16-v256-d64-t16 | 4251 | 1.000 | 1.021 | 1.021 | 96 -> 0 |
| op.gelu_tanh/tiny/fp16-n256 | 1971 | 1.002 | 1.034 | 1.024 | 164 -> 98 |
| op.gemv/tiny/fp16-k64-n64 | 4763 | 1.012 | 1.000 | 1.012 | 33 -> 3 |
| op.layernorm/tiny/fp16-r4-d64 | 2046 | 1.036 | 1.041 | 1.083 | 116 -> 53 |
| op.linear/tiny/fp16-m16-k64-n64 | 5577 | 1.001 | 0.997 | 1.003 | 22 -> 4 |
| op.matmul/tiny/fp16-m16-k32-n16 | 1009 | 1.000 | 1.000 | 1.000 | 0 -> 0 |
| op.matmul/tiny/fp32-m16-k32-n16 | 3182 | 1.018 | 1.015 | 1.033 | 470 -> 350 |
| op.rmsnorm/tiny/fp16-r4-d64 | 1409 | 1.150 | 1.022 | 1.158 | 67 -> 30 |
| op.rope/tiny/fp16-b1-h2-s16-d16 | 3872 | 1.003 | 1.043 | 1.046 | 182 -> 0 |
| op.silu/tiny/fp16-n256 | 1460 | 1.025 | 1.008 | 1.030 | 124 -> 79 |
| op.softmax/tiny/fp16-r4-c64 | 2467 | 1.004 | 0.952 | 0.995 | 223 -> 73 |
| op.softmax/tiny/fp32-r3-c37 | 1343 | 1.000 | 1.037 | 0.989 | 123 -> 44 |
| op.swiglu/tiny/fp16-n256 | 1659 | 0.999 | 1.018 | 1.014 | 131 -> 85 |
| op.transpose/tiny/fp32-r32-c32 | 2274 | 1.012 | 1.034 | 1.039 | 118 -> 16 |
| **geometric mean (27)** | | **1.021** | **1.019** | **1.039** | |
| cycle-weighted | 1339790 | 1.025 | 1.027 | 1.050 | |
