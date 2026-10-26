# R40-03.gemm-tile: why the 16,32,16 tile is slower (diagnosis, not a gate record)

On the board the fe1b campaign gives the 16,32,16 candidate an S_geo of 0.303 against the
16,16,16 control. The 95 % CI is [0.202, 0.444], over 9 points. The candidate value was set in
advance by the compile-coverage rule: 16,64,16 overflowed the code store on `model.mlp/medium`.

## What the model says

These are the board's two configurations of two points, run through the SystemC/TLM model:
- binary: M2 `tnca-chip`, sha256 `2402d46d…`;
- config: `configs/r40-board-m2.cfg`, sha256 `dc2c1763…`, uncalibrated;
- knobs: exactly as the campaign recorded them;
- source: the delivery tree fe1b.

| Point | Board 16,16,16 | Board 16,32,16 | Board ratio | Model 16,16,16 | Model 16,32,16 | Model ratio |
|---|---:|---:|---:|---:|---:|---:|
| `op.linear/medium/fp16-m128-k320-n256` | 1,949,824 | 11,875,930 | 6.09x | 1,826,404 | 10,785,379 | 5.91x |
| `op.matmul/medium/fp16-m128-k192-n128` | — | — | 3.76x (1/0.266) | 546,693 | 1,922,041 | 3.52x |

The model reproduces the board's slowdown to within 7 %, so its counters can be used for
attribution. They are in `counters_diff.txt` (core 0; per-run summaries are in `runs.tgz`).

| Counter (linear/medium, core 0) | 16,16,16 | 16,32,16 | Ratio |
|---|---:|---:|---:|
| instructions | 244,300 | 2,941,516 | 12.0x |
| matrix-engine commands | 5,376 | 256,256 | 47.7x |
| `a.vread` (accumulator read-back) | 4,096 | 331,776 | 81x |
| `a.row` (row accumulator ops) | 0 | 245,760 | new |
| LSU instructions | 13,780 | 616,532 | 44.7x |
| matrix-engine busy cycles | 795,776 | 1,085,056 | 1.36x |
| dependency-wait cycles | 1,168,260 | 20,614,526 | 17.6x |

**Attribution.** The matrix work itself grows by only 36 %. What grows is everything around it:
- a 16x32 accumulator is not lowered as two independent 16x16 MMA fragments;
- the generated kernel moves it row by row instead (`a.row`, `a.vread`, per-row LSU traffic);
- each row move waits on the matrix engine (dependency and park cycles).

This is a code-generation limit of the TNCA GEMM template for a non-native N tile, not a hardware
limit.

**R41 target:** lower an N=32 (or N=64) logical tile as independent 16x16 fragments, with no
row-wise accumulator traffic, then re-run the gate.
