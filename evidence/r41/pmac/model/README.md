# Packed MMA (m.mma.dp) in the SystemC/TLM model: a WHAT-IF

Status: **what-if, 2026-10-22.** Every number here is an uncalibrated SystemC/TLM model prediction of an opcode that is not built. The model's R40 holdout median error is 21.5 %, so read only ratios between configurations of the same point, never absolute cycles. Nothing here is a board measurement.

## Result in one paragraph

On the B40 three-core configuration, the primary what-if (lane V3: E4M3×E4M3 at G=4, lane latency +1) cuts the matrix engine's busy time per FP8 MMA from 302 to 87 cycles (81 with B prefetch), a 3.5–3.7× cut. The FP8 W8A8 linears speed up by only **0.99×–1.04×**. The 16-bit G=2 data point (V2) gives BF16 linears **1.00×–1.04×**. In this model the MMA engine is almost never on the critical path. The fraction of core time spent stalled on the engine (`matrix_engine_wait`) is 0.4–3.4 % for FP8 and 0.1–4.1 % for BF16, and that fraction bounds the gain. FP8 linears are bound by the LSU B transpose (37–58 LSU instructions per MMA). BF16 linears are bound by the transfer engine and the home slices. Packing pays off only after those are fixed.

## What was run

- **Model change** (`tools/performance_harness/systemc`, commit 2411609, reviewed here):
  - three optional config keys, all absent from shipped configurations: `mtx_dp_group_16` (1|2), `mtx_dp_group_8` (1|2|4) and `mtx_dp_b_prefetch` (0|1);
  - when any key is active, the MMA sweep becomes `A + (K/G)·(t_rf + 16·mac)`, or `A + t_rf + (K/G)·16·mac` with prefetch. A is the unchanged ABUF fragment read; `busy = sweep + mtx_pipe_cycles + 1 + mtx_complete_cycles`;
  - timing only: the numerics stay the per-K `m.mma.async` path, and the counter `matrix_dp_whatif_commands` counts affected MMAs;
  - `run_graphs.py --points-module` registers extra points outside the frozen suites, and the cold-call counter snapshot lets the steady call be isolated;
  - `mtx_dp_group_8` applies to both FP8 formats. Lane V3 packs only E4M3×E4M3, but every FP8 point here is E4M3×E4M3, so the two agree on these points.
- **Points** (`pmac_points.py`, plain Torch):
  - `op.w8a8_linear`: static W8A8 via `torch._scaled_mm`, e4m3×e4m3, fp32 accumulation, bf16 out, with the (N, K) weight passed as `.t()`;
  - `op.bf16_linear`: `nn.Linear` without bias, as a control;
  - M=1 and M=128, K×N = 2048×6144 and 2560×9216. Every model run passes numerically (116 runs, max rel. err ≤ 3.2e-3).
- **Configuration:**
  - base `r40-board-m2.cfg` (= `configs/g1-pf0.cfg`), on the fast DRAM path with the B40 DRAM map;
  - knobs `TNCA_ACTIVE_CORES=3`, `TNCA_TRANSFER_POLICY=resident`, `TNCA_RTL_MEM_PORTS=3`, `TNCA_RTL_OUTSTANDING=8` (B40 image: 3 cores, 3 ports);
  - one untimed cold call, then one measured steady call. The counters are those of the steady call alone.
- **Configurations** (`configs/`):
  - `g1-pf0`: today;
  - `g1-pf1`: prefetch only;
  - **`g4-pf0-pipe5` / `g4-pf1-pipe5`: the primary V3 what-if** (8-bit G=4, 16-bit G=1, `mtx_pipe_cycles` 4→5);
  - `g2g4-pf*`: G=2/4 at pipe+0, as an optimistic bound;
  - `g2g4-pf*-pipe6`: the secondary V2 what-if, 16-bit G=2 at pipe+2. Lane V2 needs at least 5 registers and does not fit SLR0 (`../README.md` §7).

### Default identity (the keys change nothing when absent or at their defaults)

| Check | Result |
|---|---|
| `validate_matrix.py` + `validate.py`, unmodified binary against what-if binary, same paths | **identical**: 389 files. 318 are byte-identical; 33 are identical after substituting the binary SHA256; 36 differ only in host wall time or that hash; 2 are RTL testbench `.vvp` images. 0 failures. Both runs `passed` |
| Graph level: `bf16_linear` and `w8a8_linear` m1-k2048-n6144 under the unmodified binary, and under the what-if binary with the keys written at their defaults, against the reference run | **identical**: every counter (cold and final), every launch's start/end cycle, the status and the error |
| `whatif_matrix_check.py`: single MMA, every dtype × K16/32/64, under default, keys written at defaults, G=1 pf, G=2/2, G=2/4 and G=2/4 pf, on `r40-board-m2.cfg` and `r40-three-core.cfg` | **passed**, 72/72 each. The reference bytes match, and Δ`matrix_engine_busy_cycles` equals Δ(formula) exactly |

## Overview (speedup of device cycles against G=1 today)

"MMA busy share" is the engine busy cycles divided by (3 cores × device cycles). "MMA wait share" is the share of core time that a context stalled waiting for the engine. That second share is the part of MMA time that is on the critical path.

| point | G=1 cycles | MMA busy share | MMA wait share | V3 G=4 | V3 G=4 + pf | G=1 + pf | 16-bit G=2 pipe+2 | 16-bit G=2 + pf pipe+2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `w8a8_linear m1-k2048-n6144` | 53,965,314 | 9.2 % | 0.4 % | 0.990x | 0.994x | 0.995x | n/a | n/a |
| `w8a8_linear m1-k2560-n9216` | 51,977,068 | 17.9 % | 1.6 % | 1.041x | 1.043x | 1.008x | n/a | n/a |
| `w8a8_linear m128-k2048-n6144` | 445,561,279 | 8.9 % | 2.5 % | 0.997x | 0.998x | 0.983x | n/a | n/a |
| `w8a8_linear m128-k2560-n9216` | 369,937,961 | 20.1 % | 3.4 % | 1.009x | 1.023x | 0.986x | n/a | n/a |
| `bf16_linear m1-k2048-n6144` | 22,724,270 | 22.4 % | 4.1 % | n/a | n/a | 1.003x | 1.012x | 1.014x |
| `bf16_linear m1-k2560-n9216` | 42,560,789 | 22.4 % | 3.5 % | n/a | n/a | 1.008x | 1.022x | 1.023x |
| `bf16_linear m128-k2048-n6144` | 124,848,897 | 32.6 % | 0.1 % | n/a | n/a | 0.999x | 1.008x | 1.037x |
| `bf16_linear m128-k2560-n9216` | 212,894,950 | 35.8 % | 0.1 % | n/a | n/a | 1.001x | 1.002x | 1.002x |

Model noise floor: configurations that change only scheduling order move these points by up to ±2 %. "G=1 + B prefetch" cuts the busy time by 10 % yet gives 0.983×–1.008×. "V3 image, BF16 stays G=1" changes only the pipe (and, with pf, the B read) for BF16 and gives 0.994×–1.009×. Treat any |speedup − 1| below about 2 % as zero.

## Per point

"Busy ratio" is the reduction in engine busy cycles. "Amdahl ceiling" is the speedup if busy time were serial: 1/((1−s)+s/ratio). It is not reached, because asynchronous MMAs overlap other work (see below).

#### `op.w8a8_linear/hbm/bf16-m1-k2048-n6144`

| configuration | device cycles | speedup | MMA busy (3 cores) | busy / MMA | busy share | busy ratio | MMA wait share | Amdahl ceiling |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| G=1 (today) | 53,965,314 | 1.000x | 14,850,432 | 302 | 9.2 % | 1.00x | 0.4 % | 1.00x |
| G=1, B prefetch | 54,210,347 | 0.995x | 13,375,872 | 272 | 8.2 % | 1.11x | 0.3 % | 1.01x |
| **V3: G=4, pipe+1** | 54,488,853 | 0.990x | 4,282,752 | 87 | 2.6 % | 3.47x | 0.0 % | 1.07x |
| **V3: G=4, B prefetch, pipe+1** | 54,303,773 | 0.994x | 3,987,840 | 81 | 2.4 % | 3.72x | 0.0 % | 1.07x |
| G=4, pipe+0 (lower bound on pipe cost) | 54,469,657 | 0.991x | 4,233,600 | 86 | 2.6 % | 3.51x | 0.0 % | 1.07x |
| G=4, B prefetch, pipe+0 | 54,632,404 | 0.988x | 3,938,688 | 80 | 2.4 % | 3.77x | 0.0 % | 1.07x |

#### `op.w8a8_linear/hbm/bf16-m1-k2560-n9216`

| configuration | device cycles | speedup | MMA busy (3 cores) | busy / MMA | busy share | busy ratio | MMA wait share | Amdahl ceiling |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| G=1 (today) | 51,977,068 | 1.000x | 27,842,112 | 302 | 17.9 % | 1.00x | 1.6 % | 1.00x |
| G=1, B prefetch | 51,586,196 | 1.008x | 25,077,312 | 272 | 16.2 % | 1.11x | 1.3 % | 1.02x |
| **V3: G=4, pipe+1** | 49,931,589 | 1.041x | 8,027,712 | 87 | 5.4 % | 3.47x | 0.1 % | 1.15x |
| **V3: G=4, B prefetch, pipe+1** | 49,829,612 | 1.043x | 7,474,752 | 81 | 5.0 % | 3.72x | 0.1 % | 1.15x |
| G=4, pipe+0 (lower bound on pipe cost) | 49,954,138 | 1.040x | 7,935,552 | 86 | 5.3 % | 3.51x | 0.1 % | 1.15x |
| G=4, B prefetch, pipe+0 | 50,057,019 | 1.038x | 7,382,592 | 80 | 4.9 % | 3.77x | 0.0 % | 1.15x |

#### `op.w8a8_linear/hbm/bf16-m128-k2048-n6144`

| configuration | device cycles | speedup | MMA busy (3 cores) | busy / MMA | busy share | busy ratio | MMA wait share | Amdahl ceiling |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| G=1 (today) | 445,561,279 | 1.000x | 118,803,456 | 302 | 8.9 % | 1.00x | 2.5 % | 1.00x |
| G=1, B prefetch | 453,446,560 | 0.983x | 107,006,976 | 272 | 7.9 % | 1.11x | 2.2 % | 1.01x |
| **V3: G=4, pipe+1** | 446,983,865 | 0.997x | 34,262,016 | 87 | 2.6 % | 3.47x | 0.5 % | 1.07x |
| **V3: G=4, B prefetch, pipe+1** | 446,579,465 | 0.998x | 31,902,720 | 81 | 2.4 % | 3.72x | 0.4 % | 1.07x |
| G=4, pipe+0 (lower bound on pipe cost) | 446,884,964 | 0.997x | 33,868,800 | 86 | 2.5 % | 3.51x | 0.5 % | 1.07x |
| G=4, B prefetch, pipe+0 | 446,545,488 | 0.998x | 31,509,504 | 80 | 2.4 % | 3.77x | 0.4 % | 1.07x |

#### `op.w8a8_linear/hbm/bf16-m128-k2560-n9216`

| configuration | device cycles | speedup | MMA busy (3 cores) | busy / MMA | busy share | busy ratio | MMA wait share | Amdahl ceiling |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| G=1 (today) | 369,937,961 | 1.000x | 222,736,896 | 302 | 20.1 % | 1.00x | 3.4 % | 1.00x |
| G=1, B prefetch | 375,047,125 | 0.986x | 200,618,496 | 272 | 17.8 % | 1.11x | 3.2 % | 1.02x |
| **V3: G=4, pipe+1** | 366,658,288 | 1.009x | 64,221,696 | 87 | 5.8 % | 3.47x | 0.8 % | 1.17x |
| **V3: G=4, B prefetch, pipe+1** | 361,690,256 | 1.023x | 59,798,016 | 81 | 5.5 % | 3.72x | 0.7 % | 1.17x |
| G=4, pipe+0 (lower bound on pipe cost) | 363,033,436 | 1.019x | 63,484,416 | 86 | 5.8 % | 3.51x | 0.8 % | 1.17x |
| G=4, B prefetch, pipe+0 | 364,024,421 | 1.016x | 59,060,736 | 80 | 5.4 % | 3.77x | 0.7 % | 1.17x |

#### `op.bf16_linear/hbm/bf16-m1-k2048-n6144`

| configuration | device cycles | speedup | MMA busy (3 cores) | busy / MMA | busy share | busy ratio | MMA wait share | Amdahl ceiling |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| G=1 (today) | 22,724,270 | 1.000x | 15,243,648 | 310 | 22.4 % | 1.00x | 4.1 % | 1.00x |
| G=1, B prefetch | 22,656,879 | 1.003x | 13,769,088 | 280 | 20.3 % | 1.11x | 2.9 % | 1.02x |
| V3 image, BF16 stays G=1 (pipe+1 only) | 22,720,473 | 1.000x | 15,292,800 | 311 | 22.4 % | 1.00x | 4.1 % | 1.00x |
| V3 image + B prefetch, BF16 stays G=1 | 22,693,014 | 1.001x | 13,818,240 | 281 | 20.3 % | 1.10x | 3.0 % | 1.02x |
| G=2, pipe+0 (optimistic) | 22,461,297 | 1.012x | 8,165,760 | 166 | 12.1 % | 1.87x | 0.2 % | 1.12x |
| G=2, B prefetch, pipe+0 (optimistic) | 22,391,783 | 1.015x | 7,477,632 | 152 | 11.1 % | 2.04x | 0.1 % | 1.13x |
| secondary V2: G=2, pipe+2 | 22,456,955 | 1.012x | 8,264,064 | 168 | 12.3 % | 1.84x | 0.2 % | 1.11x |
| secondary V2: G=2, B prefetch, pipe+2 | 22,406,205 | 1.014x | 7,575,936 | 154 | 11.3 % | 2.01x | 0.1 % | 1.13x |

#### `op.bf16_linear/hbm/bf16-m1-k2560-n9216`

| configuration | device cycles | speedup | MMA busy (3 cores) | busy / MMA | busy share | busy ratio | MMA wait share | Amdahl ceiling |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| G=1 (today) | 42,560,789 | 1.000x | 28,579,392 | 310 | 22.4 % | 1.00x | 3.5 % | 1.00x |
| G=1, B prefetch | 42,213,126 | 1.008x | 25,814,592 | 280 | 20.4 % | 1.11x | 2.5 % | 1.02x |
| V3 image, BF16 stays G=1 (pipe+1 only) | 42,431,971 | 1.003x | 28,671,552 | 311 | 22.5 % | 1.00x | 3.6 % | 1.00x |
| V3 image + B prefetch, BF16 stays G=1 | 42,187,643 | 1.009x | 25,906,752 | 281 | 20.5 % | 1.10x | 2.5 % | 1.02x |
| G=2, pipe+0 (optimistic) | 41,653,158 | 1.022x | 15,308,352 | 166 | 12.3 % | 1.87x | 0.1 % | 1.12x |
| G=2, B prefetch, pipe+0 (optimistic) | 41,597,052 | 1.023x | 14,018,112 | 152 | 11.2 % | 2.04x | 0.0 % | 1.13x |
| secondary V2: G=2, pipe+2 | 41,655,383 | 1.022x | 15,492,672 | 168 | 12.4 % | 1.84x | 0.1 % | 1.11x |
| secondary V2: G=2, B prefetch, pipe+2 | 41,587,479 | 1.023x | 14,202,432 | 154 | 11.4 % | 2.01x | 0.0 % | 1.13x |

#### `op.bf16_linear/hbm/bf16-m128-k2048-n6144`

| configuration | device cycles | speedup | MMA busy (3 cores) | busy / MMA | busy share | busy ratio | MMA wait share | Amdahl ceiling |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| G=1 (today) | 124,848,897 | 1.000x | 121,949,184 | 310 | 32.6 % | 1.00x | 0.1 % | 1.00x |
| G=1, B prefetch | 124,932,776 | 0.999x | 110,152,704 | 280 | 29.4 % | 1.11x | 0.1 % | 1.03x |
| V3 image, BF16 stays G=1 (pipe+1 only) | 124,741,176 | 1.001x | 122,342,400 | 311 | 32.7 % | 1.00x | 0.1 % | 1.00x |
| V3 image + B prefetch, BF16 stays G=1 | 125,093,536 | 0.998x | 110,545,920 | 281 | 29.5 % | 1.10x | 0.1 % | 1.03x |
| G=2, pipe+0 (optimistic) | 123,972,584 | 1.007x | 65,326,080 | 166 | 17.6 % | 1.87x | 0.0 % | 1.18x |
| G=2, B prefetch, pipe+0 (optimistic) | 121,510,725 | 1.027x | 59,821,056 | 152 | 16.4 % | 2.04x | 0.0 % | 1.20x |
| secondary V2: G=2, pipe+2 | 123,918,526 | 1.008x | 66,112,512 | 168 | 17.8 % | 1.84x | 0.0 % | 1.18x |
| secondary V2: G=2, B prefetch, pipe+2 | 120,426,590 | 1.037x | 60,607,488 | 154 | 16.8 % | 2.01x | 0.0 % | 1.20x |

#### `op.bf16_linear/hbm/bf16-m128-k2560-n9216`

| configuration | device cycles | speedup | MMA busy (3 cores) | busy / MMA | busy share | busy ratio | MMA wait share | Amdahl ceiling |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| G=1 (today) | 212,894,950 | 1.000x | 228,635,136 | 310 | 35.8 % | 1.00x | 0.1 % | 1.00x |
| G=1, B prefetch | 212,742,566 | 1.001x | 206,516,736 | 280 | 32.4 % | 1.11x | 0.1 % | 1.04x |
| V3 image, BF16 stays G=1 (pipe+1 only) | 212,698,069 | 1.001x | 229,372,416 | 311 | 35.9 % | 1.00x | 0.1 % | 1.00x |
| V3 image + B prefetch, BF16 stays G=1 | 214,170,608 | 0.994x | 207,254,016 | 281 | 32.3 % | 1.10x | 0.1 % | 1.03x |
| G=2, pipe+0 (optimistic) | 217,698,897 | 0.978x | 122,466,816 | 166 | 18.8 % | 1.87x | 0.0 % | 1.20x |
| G=2, B prefetch, pipe+0 (optimistic) | 212,188,124 | 1.003x | 112,144,896 | 152 | 17.6 % | 2.04x | 0.0 % | 1.22x |
| secondary V2: G=2, pipe+2 | 212,512,361 | 1.002x | 123,941,376 | 168 | 19.4 % | 1.84x | 0.0 % | 1.20x |
| secondary V2: G=2, B prefetch, pipe+2 | 212,560,196 | 1.002x | 113,619,456 | 154 | 17.8 % | 2.01x | 0.0 % | 1.22x |

## Why the speedups are sub-linear (from the counters)

Core-time fractions of the steady call, G=1 against the primary what-if (FP8: V3 G=4 + prefetch; BF16: G=2 + prefetch, pipe+2). The fractions overlap and sum over the contexts of a core, so they can exceed 1.

| point | config | MMA busy | MMA wait (exposed) | dependency wait | LSU wait | pipeline freeze | barrier sleep | DMA admission wait | home queue | LSU instr / MMA | core-cycles / MMA |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `w8a8_linear m1-k2048-n6144` | G=1 | 0.092 | 0.004 | 1.160 | 0.624 | 0.660 | 0.091 | 0.000 | 2.59 | 58.2 | 3294 |
| `w8a8_linear m1-k2048-n6144` | V3 G=4 pf | 0.024 | 0.000 | 1.184 | 0.666 | 0.703 | 0.024 | 0.000 | 2.67 | 58.2 | 3314 |
| `w8a8_linear m1-k2560-n9216` | G=1 | 0.179 | 0.016 | 1.149 | 0.452 | 0.499 | 0.178 | 0.000 | 2.68 | 58.2 | 1692 |
| `w8a8_linear m1-k2560-n9216` | V3 G=4 pf | 0.050 | 0.001 | 1.212 | 0.504 | 0.557 | 0.049 | 0.000 | 2.97 | 58.2 | 1622 |
| `w8a8_linear m128-k2048-n6144` | G=1 | 0.089 | 0.025 | 1.065 | 0.693 | 0.714 | 0.138 | 0.003 | 2.48 | 37.5 | 3399 |
| `w8a8_linear m128-k2048-n6144` | V3 G=4 pf | 0.024 | 0.004 | 1.091 | 0.751 | 0.773 | 0.068 | 0.013 | 2.52 | 37.5 | 3407 |
| `w8a8_linear m128-k2560-n9216` | G=1 | 0.201 | 0.034 | 1.035 | 0.492 | 0.522 | 0.286 | 0.004 | 3.33 | 37.4 | 1505 |
| `w8a8_linear m128-k2560-n9216` | V3 G=4 pf | 0.055 | 0.007 | 1.111 | 0.568 | 0.604 | 0.140 | 0.022 | 3.61 | 37.4 | 1472 |
| `bf16_linear m1-k2048-n6144` | G=1 | 0.224 | 0.041 | 0.062 | 0.371 | 0.393 | 0.579 | 0.388 | 2.62 | 25.2 | 1387 |
| `bf16_linear m1-k2048-n6144` | G=2 pf pipe+2 | 0.113 | 0.001 | 0.054 | 0.326 | 0.352 | 0.446 | 0.520 | 2.08 | 25.2 | 1368 |
| `bf16_linear m1-k2560-n9216` | G=1 | 0.224 | 0.035 | 0.065 | 0.390 | 0.414 | 0.598 | 0.367 | 2.87 | 25.2 | 1385 |
| `bf16_linear m1-k2560-n9216` | G=2 pf pipe+2 | 0.114 | 0.000 | 0.060 | 0.365 | 0.391 | 0.474 | 0.487 | 2.55 | 25.2 | 1354 |
| `bf16_linear m128-k2048-n6144` | G=1 | 0.326 | 0.001 | 0.003 | 0.007 | 0.008 | 1.288 | 0.634 | 2.29 | 0.3 | 953 |
| `bf16_linear m128-k2048-n6144` | G=2 pf pipe+2 | 0.168 | 0.000 | 0.003 | 0.007 | 0.008 | 1.131 | 0.791 | 2.54 | 0.3 | 919 |
| `bf16_linear m128-k2560-n9216` | G=1 | 0.358 | 0.001 | 0.003 | 0.006 | 0.007 | 1.321 | 0.603 | 3.28 | 0.2 | 866 |
| `bf16_linear m128-k2560-n9216` | G=2 pf pipe+2 | 0.178 | 0.000 | 0.003 | 0.007 | 0.007 | 1.143 | 0.781 | 3.28 | 0.2 | 865 |

1. **The engine is mostly hidden.** `m.mma.async` returns at issue, and the engine overlaps the context's own loads and the other contexts. Only `matrix_engine_wait` (0.1–4.1 % of core time at G=1) is exposed. Packing removes almost all of that wait, which goes to 0.0–0.7 %, so the gain is bounded by about that share. It does not approach the busy share (9–36 %) or the Amdahl ceiling (1.07–1.22×). Matching examples:
   - `w8a8 m128-k2560-n9216`: wait 3.4 % → 0.7 %, speedup 1.023×;
   - `bf16 m128`: wait 0.1 %, speedup 1.00×.
2. **FP8 W8A8 is bound by the B transpose in the LSU.** With no 8-bit transposed write in the transfer engine (`../README.md` §6.2), the `.t()` weight becomes gathers plus byte stores: 37.5 LSU instructions per MMA at M=128 and 58 at M=1, against 0.2–0.3 for BF16 at M=128. Each core spends 1.04–1.16 of its time in dependency wait and 0.45–0.69 in LSU wait, and the home-slice queue sits at 2.5–3.3 (fractions as in the table).
   - So FP8 at G=1 is 1.2–3.6× slower than BF16 at the same shape: 445.6 M against 124.8 M cycles at M=128, K=2048, and 1.2× at M=1, K=2560.
   - Even the best V3 point (361.7 M, m128-k2560) stays 1.7× slower than BF16 at G=1.
   - Under packing, the freed engine time reappears as more dependency and LSU wait: `w8a8 m128-k2560` goes from 1.035 to 1.111 and from 0.492 to 0.568. Contexts reach the next gather sooner and queue longer at the home slices (home queue 3.33 → 3.61).
3. **BF16 linears are bound by the transfer engine.** At M=128 the time goes to barrier sleep (1.29–1.32) and DMA admission wait (0.60–0.63). Contexts wait for tiles, and the MMA overlaps the transfers.
   - With G=2 the barrier sleep falls by 0.16–0.18 and the DMA admission wait rises by the same amount (0.60 → 0.78): the time moves from waiting on the engine's tiles to waiting at the transfer queue.
   - At M=1 the engine is exposed a little (wait 3.5–4.1 %), and G=2 gives the expected 1.2–2.3 %.
4. **B prefetch and pipe+1 are second order.** Prefetch takes about 6 more cycles per K16 FP8 MMA (87 → 81) and pipe+1 adds 1. Both are within the noise floor at graph level. Pipe+1 costs V3 nothing measurable compared with pipe+0: 0.990–1.041× against 0.991–1.040×.

## What else must change before packing pays

In order, from the counters above:

1. **The FP8 B operand path.** Either the n-major B layout (`mod[7]` plus BBUF, `../README.md` §6.2) or an 8-bit transposed write in `tnca_tensor.v`, so that a W8A8 linear uses the same transfer-engine tile path as BF16. That alone removes 37–58 LSU instructions per MMA and the 1.2–3.6× FP8-over-BF16 penalty. It is worth far more than packing, and it is the precondition for packing to show at all.
2. **Transfer-engine admission and home-slice throughput.** These are the limits for BF16, and for FP8 once (1) is done: deeper transfer queues, more outstanding requests, faster home slices, and more tile reuse (for example `TNCA_GEMM_TILE=16,64,16` or a producer/consumer ring). They must raise the MMA wait share well above the current 0–4 % before G=4 can return more than a few percent.
   - Not measured here. The planned sensitivity runs (`s-dma4`, `s-home2`, tile and ring levers; configs kept in `configs/s-*.cfg`) never ran: a script-permission error, visible in `sens.log` in the tgz. That is the next experiment.
3. **The model's FP8 numerics.** `m.mma.dp` must take the `tools/model/fdp.py` semantics (RO_48 group sum) before a functional `grouped` run, and `mtx_dp_group_8` should be restricted to E4M3 if V3 is the implemented lane.

## Not done (reduced scope, 2026-10-22)

- 9216×2560 shapes, matmul 2048³ and SDPA in the reduced tables.
  - The G=1 / G=2/4 runs exist in the tgz (`summary-main.json`): matmul 2048³ gives 1.001–1.003×, SDPA has no MMA (1.000×), and `w8a8 m128-k9216-n2560` gives 0.946–0.959×. That last case is a scheduling-order effect of the same kind as the noise floor, larger here, and it was not studied further.
- The sensitivity levers in §"What else must change" (2).

## Files

| File | Content |
|---|---|
| `pmac_points.py` | the W8A8 and BF16 linear points (`--points-module`) |
| `configs/*.cfg` | every configuration, each with a header line saying what differs from `r40-board-m2.cfg` |
| `summarize.py`, `report_tables.py` | summary.json and the tables above |
| `whatif_matrix_check.py`, `identity_compare.py`, `graph_identity.py` | the timing-formula check and the two identity checks |
| `pmac-model-evidence.tgz` | Raw evidence: per-run `record.json`, `env.json`, counters (final and cold), `launches.json`, `child.log`; every `results.json` and run log (including the failed `sens.log` and `trace.log`); `summary.json`, `summary-main.json`, `tables.md`; the identity and what-if check outputs; the run scripts and the B40 DRAM map |
