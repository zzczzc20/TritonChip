# R42 speed prediction for a wider lane and lower instruction overhead (TLM, calibrated and checked against the board)

> **Binding:** [`doc/engineering_discipline.md`](../../../doc/engineering_discipline.md). Domain: SystemC/TLM model prediction applied
> as a ratio to the `ab-int2` board cycles; nothing here is a board result. Tables: `tables.txt`, `predict_113.json`,
> `predict_xl.json`; validation: `validation_board.txt`, `rtl_g_check.txt`, `board_engine_utilisation.txt`; commands: `commands.txt`.
> Tools `tools/r42/kfuse_speed_*.py`; three default-off TLM keys (`mtx_dp_g_override`, `mtx_dp_b_regs`, `issue_free_mode`) leave
> the unpatched cycles bit-identical on 30 points.

## Result

Neither a wider lane nor lower instruction overhead moves the 113-point geometric mean: best case G=16 with the scalar/branch
instructions of the K loop free is **+0.70 %** (model). On the 37 matrix-engine GEMM points +2.2 %. 45 of 113 points modelled,
68 held at board cycles (attention kernels have no matrix command, 1.000).

| scenario | 113-point geomean | 37 GEMM points |
|---|---|---|
| today G, S1 | +0.17 % | +0.53 % |
| G=8 S0 / S1 / S2 | +0.42 / +0.62 / -0.03 % | +1.3 / +1.9 / -0.1 % |
| G=16 S0 / S1 / S2 | +0.52 / +0.70 / +0.03 % | +1.6 / +2.2 / +0.1 % |

S1 = scalar and branch instructions of the matrix K loop free (10 of 19 per K step); S2 = every K-loop instruction free of the issue slot.

Why: on the 10 single-GEMM board points the matrix engine is busy 28-37 % of the time (board cycles x 168-cycle command); one K step needs the
tensor engine 75 + 99 = 174 cycles for the A tile and the transposed B tile against a 168-cycle FP16 K16 command, so the transfer
path, not the lane, sets the step time. Even with ideal memory G=8 gives only +2.7 % on the 30 tiny/small/medium GEMM points.

## xl tier (32 points, 18 with FP8 projections; model, no FP8 board record exists)

| scenario | 32 points | 18 FP8 points |
|---|---|---|
| E4M3 G=8 / G=16 | +0.68 / +0.99 % | +1.1 / +1.6 % |
| S1 / S2 | +2.4 / +3.4 % | +4.1 / +6.3 % |
| S3 (what-if: weight and activation preparation before the last matrix command free; the 2,404-instruction scale/accumulate epilogue stays) | +27.8 % | +54.6 % |
| G=16 + S3 | +29.5 % | +58.3 % |
| existing `_scaled_mm` stream + one activation quantisation (a different existing kernel) | +34.0 % | +68.2 % |

The Hub kernel as compiled is neither engine- nor issue-bound (M>=16: engine 13 % busy, issue 23 %); one K block costs 52.7k core-cycles at
8,635 instructions. S3 is a projection (BALANCE.md "2,652 instructions"), not existing code.

## Model quality

77 tiny/small/medium board points: 33 GEMM-shaped, geomean model/board 0.984, median error 9.8 %, 26 of 33 within 20 %. With large/hbm via
reduced-grid proxies: 37 matrix-engine points, geomean 0.947, median 7.9 %, 31 of 37 within 20 %. Outside 20 %: llama_decode (0.69-0.78),
llama_prefill large/hbm (0.77, 0.66), gemv large/hbm, attention_decode, FP32 matmul/conv/resnet. The model understates the lane's
benefit by 3-4 points when memory does not bind (single-core RTL G=2/G=1 ratio: model 0.806/0.782/0.647 vs RTL 0.780/0.747/0.611).

## What it means

No image for lane width or instruction overhead (plausible range 0 to +1.5 % on the 113 points). Software first for xl FP8 (Hub kernel
instruction stream). The next hardware lever is the tensor transfer path: tensor-engine plain-tile load slots 8 -> 32 gives +9.1 % on the 30
GEMM points (+2.3 % on the 113 geomean if applied to those only); ideal memory 1.47x.

## Failures and deviations

`dma_outstanding=64` left every point unchanged (key does not bind in the timeline model); the valid sensitivity is `dma_unit_slots`. Large and
hbm tiers use proxies; G=16 with S1/S2 on nine proxy points is extrapolated from G=8 (flagged `extrapolated`). xl non-FP8 parts keep the tier's
planning estimate. No FP8 board record exists, so the xl numbers have no board validation.
