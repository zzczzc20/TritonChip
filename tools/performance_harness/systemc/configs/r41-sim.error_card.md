# R41 SystemC model: error card

Model `configs/r41-sim.cfg` against RTL simulation cycles of the same programs (three active cores, the evaluator's measurement path). Ratio = model / RTL; below 1 the model is fast.

Measured 2026-11-08 on Myserver: base, LSU_INFLIGHT 2 and SETS 64 on visible and holdout items; MTX_VRF_REG 0, 0-8 and the compiler variant on visible items only (their holdout RTL runs were cut for time). Items: 17 visible ArchBench points (calibration) and 23 agent-visible test cases (15 designed shapes + the skill's 8 examples, the holdout); no validation or held-out point.

## Absolute (author's rule: holdout median <= 10 %, p95 <= 20 %)

| items | compared | median abs error | p95 | geomean model/RTL |
|---|---|---|---|---|
| holdout | 23/23 | 4.5% | 15.3% | 0.958 |
| visible | 17/17 | 3.4% | 11.4% | 0.976 |

Verdict on the holdout: **pass**.

## Families

| family | holdout | visible | worst |
|---|---|---|---|
| add | 1.017 | 1.021 | +2.1% |
| argmax | - | 0.891 | -10.9% |
| attention_decode | 0.913 | 0.969 | -8.7% |
| attention_prefill | 0.939 | 0.873 | -12.7% |
| embedding | - | 0.977 | -2.3% |
| example:batched | 1.003 | - | +0.3% |
| example:cached_context | 0.883 | - | -11.7% |
| example:long_stream_row | 0.957 | - | -4.3% |
| example:one_row | 1.023 | - | +2.3% |
| example:product_fp16_tail | 0.971 | - | -2.9% |
| example:product_fp32 | 0.843 | - | -15.7% |
| example:projection_bias | 1.045 | - | +4.5% |
| example:two_launches | 0.920 | - | -8.0% |
| gemv | 0.957, 1.003 | 0.978 | -4.3% |
| layernorm | - | 0.905 | -9.5% |
| linear | 1.017, 1.026 | 0.994 | +2.6% |
| matmul | 0.952 | 1.014 | -4.8% |
| rmsnorm | 0.911 | 0.891 | -10.9% |
| rope | 0.832, 0.924 | 1.034 | -16.8% |
| silu | 1.062 | 1.034 | +6.2% |
| softmax | 0.928 | 0.949, 0.994 | -7.2% |
| swiglu | - | 1.111 | +11.1% |
| transpose | 0.958 | 1.055 | +5.5% |
| wfp8_linear | 0.983 | 0.928 | -7.2% |

## Knobs (change acceptance)

Rules per knob, against the base: direction right on >= 90 % of items the RTL moves by >= 3 %; captured gain in [0.75, 1.25]; no model change > 3 % where the RTL moves < 1 %.

| knob | judged on | items moved >= 3 % | direction | captured gain | false gains | verdict |
|---|---|---|---|---|---|---|
| LSU_INFLIGHT 1 -> 2 (word units behind the hybrid LSU) | holdout | 20/23 | 20/20 | 0.938 | 0 | pass |
| SETS 32 -> 64 (home cache sets) | holdout | 6/23 | 6/6 | 1.157 | 0 | pass |
| MTX_VRF_REG 1 -> 0 (matrix VRF read register) | visible | 0/17 | - | - | 0 | no RTL effect here (model also flat); holdout not measured |
| 0-8: all three above together | visible | 13/17 | 13/13 | 0.991 | 0 | pass; holdout not measured |
| compiler: TNCA_TILED_POINTWISE 0 -> 3 and TNCA_BANK_ALLOC 0 -> 1 | visible | 8/17 | 7/8 | 1.146 | 0 | **model blind: RTL only**; holdout not measured |

- compiler: TNCA_TILED_POINTWISE 0 -> 3 and TNCA_BANK_ALLOC 0 -> 1: Fails on op.linear small only (RTL 1.038, model 1.000): TNCA_BANK_ALLOC renumbers vector registers and the RTL gains from fewer VRF bank conflicts; the model has no VRF bank ports beyond one instruction's sources, so it cannot see register placement. The tiled pointwise lowering is captured. Read: bank-aware register allocation is model blind, RTL only.
