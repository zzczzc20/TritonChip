Model, registered medium points, profile core-vrf-sync-v5-x8-g2c4, memory cal/cal_line, scheduler model banked2 (fixed).

| Arm | pass | max B/cycle/core | geometric mean | vs the same arm scheduled with the flip-flop model |
|---|---:|---:|---:|---:|
| default compiler | 171/171 | 3.8 | 0.65 | 1.007 (min 0.891, max 1.136) |
| default compiler + line engine | 171/171 | 3.8 | 0.65 | 1.007 (min 0.891, max 1.136) |
| mode 2 | 171/171 | 21.7 | 1.27 | 1.019 (min 0.946, max 1.362) |
| mode 3 + line engine | 171/171 | 33.8 | 1.31 | 1.013 (min 0.941, max 1.261) |

| Comparison | points | geometric mean | min | max | slower by > 2 % |
|---|---:|---:|---:|---:|---:|
| default compiler + line engine vs default compiler | 171 | 1.001 | 1.000 | 1.028 | 0 |
| mode 2 vs default compiler | 171 | 1.967 | 1.000 | 6.965 | 0 |
| mode 3 + line engine vs mode 2 | 171 | 1.030 | 0.976 | 2.004 | 3 |
| mode 3 + line engine vs default compiler | 171 | 2.027 | 1.000 | 8.942 | 0 |
