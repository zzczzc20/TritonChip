Model (uncalibrated core; home calibrated, `cal`/`cal_line`), registered medium points, profile core-vrf-sync-v5-x8-g2c4.

| Arm | pass | max B/cycle/core | geometric mean B/cycle/core |
|---|---:|---:|---:|
| default compiler, no line engine | 171/171 | 3.9 | 0.64 |
| default compiler + line engine | 171/171 | 3.9 | 0.64 |
| mode 2, no line engine | 171/171 | 21.7 | 1.25 |
| mode 3 + line engine | 171/171 | 33.5 | 1.27 |

| Comparison (speedup = cycles of first / cycles of second) | points | geometric mean | min | max | slower by > 2 % |
|---|---:|---:|---:|---:|---:|
| default compiler + line engine vs default compiler, no line engine | 171 | 1.001 | 1.000 | 1.044 | 0 |
| mode 2, no line engine vs default compiler, no line engine | 171 | 1.943 | 1.000 | 6.952 | 0 |
| mode 3 + line engine vs mode 2, no line engine | 171 | 1.019 | 0.362 | 2.664 | 15 |
| mode 3 + line engine vs default compiler, no line engine | 171 | 1.980 | 0.994 | 8.901 | 0 |

mode 3 + line engine, best point per family/dtype (B/cycle/core):
add bf16 4.1, add fp16 4.1, add fp32 33.5, cast bf16 8.9, cast fp16 8.9, cast fp32 7.4, clone bf16 8.5, clone fp16 8.5, clone fp32 27.8, copy bf16 8.5, copy fp16 8.5, copy fp32 27.8, div bf16 3.2, div fp16 3.2, div fp32 7.1, exp bf16 2.7, exp fp16 2.7, exp fp32 4.6, gelu bf16 0.6, gelu fp16 0.6, gelu fp32 1.1, layernorm bf16 0.4, layernorm fp16 0.4, layernorm fp32 0.9, max bf16 1.0, max fp16 1.0, max fp32 1.9, mean bf16 1.0, mean fp16 1.0, mean fp32 1.9, mul bf16 4.1, mul fp16 4.1, mul fp32 33.5, relu bf16 5.0, relu fp16 5.0, relu fp32 22.0, rmsnorm bf16 0.5, rmsnorm fp16 0.5, rmsnorm fp32 1.1, sigmoid bf16 1.3, sigmoid fp16 1.3, sigmoid fp32 2.2, silu bf16 1.2, silu fp16 1.2, silu fp32 2.2, softmax bf16 0.3, softmax fp16 0.3, softmax fp32 0.6, sub bf16 4.1, sub fp16 4.1, sub fp32 33.5, sum bf16 1.0, sum fp16 1.0, sum fp32 1.9

slowest mode 3 + engine against mode 2: [(0.362, 'op.gelu/membound/fp32-medium-contig-r1024-c1024'), (0.454, 'op.sigmoid/membound/fp32-medium-contig-r1024-c1024'), (0.534, 'op.silu/membound/fp32-medium-contig-r1024-c1024'), (0.737, 'op.exp/membound/fp32-medium-contig-r1024-c1024'), (0.877, 'op.div/membound/fp32-medium-contig-r1024-c1024'), (0.977, 'op.cast/membound/bf16-to-fp32-medium-broadcast-r2048-c1024')]
slowest default + engine against default: [(1.0, 'op.add/membound/bf16-medium-broadcast-r2048-c1024'), (1.0, 'op.add/membound/bf16-medium-contig-r2048-c1024'), (1.0, 'op.add/membound/bf16-medium-strided-r2048-c1024'), (1.0, 'op.add/membound/fp16-medium-broadcast-r2048-c1024')]
