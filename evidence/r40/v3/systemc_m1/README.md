# R40 v3: native SystemC model, milestone M1 (execute the real compiler output)

The native SystemC/TLM-2.0 model (`tools/performance_harness/systemc`) now executes the compiler's
assembled kernels for every point of the frozen native77 suite end to end through the standard path
(torch.compile -> Inductor -> Triton -> TNCA artifact -> Session -> model backend), with results
matching the PyTorch reference at each point's own tolerance and zero host fallback. No RTL simulator
is in the model's execution path; the RTL was used only offline as the numeric oracle and for the speed
comparison. Nothing under `archbench/`, `python/`, `compiler/`, `tools/asm/`, `spec/`, `rtl/` or
`tb/device/` was modified. **All modeled cycle counts are uncalibrated predictions** (evidence level
`model-prediction-uncalibrated`); they are not board or RTL measurements and establish no acceptance.

Host: GB10 (`gx10-6646`, aarch64), SystemC 3.0.2. Product tree `files:fe1b5199db515440d2d3407aed61ef1eed5edb627e5baf9cda10be5a7198cf75`. Model binaries (`bin_m1c`):
`tnca-chip` `28d7110cd57b7678a3998308df12391cc75803746e00e15aefcc8fce4fba372b`;
`tnca-numeric-probe` `ae1bab06888192ac42bd2a331b601adee38f9bcaa7157e9d8a1c20affce518c9`;
`tnca-tlm` `66d75bc672d0bc70db175ee263f6d6b7601e5d8d2611dcfc349235c2c5688a89`;
the final harness source rebuilds them byte-identically.

## 1. Baseline (unchanged pre-M1 model)

| validation | result | wall s |
|---|---|---:|
| `validate_numeric` | passed (rc 0) | 11.027 |
| `validate_chip` | passed (rc 0) | 0.345 |
| `validate_roles` | passed (rc 0) | 0.356 |
| `validate_dma` | passed (rc 0) | 0.494 |
| `validate` | passed (rc 0) | 13.117 |
| `validate_monitor` | passed (rc 0) | 15.565 |

## 2. Opcode coverage of the compiled suites, before and after

Scanned 228 (point, configuration) compile-preflight runs (A = delivered default over native77+HBM14+large24, B = A/B experiment configurations), 1235 kernel images, 836008 instruction words. Verdicts after M1 come from the model's own decoder (`tnca-chip --classify`); "before" is the pre-M1 executor's acceptance logic.

| source | runs | every word executable before M1 | ...and RTL-exact before M1 | every word implemented after M1 |
|---|---:|---:|---:|---:|
| A | 113 | 40 | 18 | 113 |
| B | 115 | 29 | 14 | 115 |

Opcodes whose status changed (all other opcodes in the scan were already executed and remain so):

| opcode | major.func | static count | kernels | points | before M1 | after M1 | timing class |
|---|---|---:|---:|---:|---|---|---|
| `a.vread` | 0x41.0x00 | 35568 | 342 | 58 | refused | implemented | issue+matrix_engine_free+acc_read_latency |
| `a.vwrite` | 0x41.0x01 | 9520 | 146 | 46 | refused | implemented | issue+matrix_engine_occupancy |
| `fence` | 0x21.0x20 | 2241 | 500 | 63 | refused | implemented | issue+own_scoreboard_drain |
| `v.rcp.approx` | 0x16.0x01 | 1026 | 295 | 54 | implemented-inexact | implemented | issue+sfu_occupancy+sfu_latency |
| `v.exp2.approx` | 0x16.0x00 | 770 | 103 | 45 | implemented-inexact | implemented | issue+sfu_occupancy+sfu_latency |
| `v.rsqrt.approx` | 0x16.0x02 | 469 | 208 | 27 | implemented-inexact | implemented | issue+sfu_occupancy+sfu_latency |
| `m.zero` | 0x40.0x00 | 419 | 320 | 52 | refused | implemented | issue+matrix_engine_occupancy |
| `trap` | 0x02.0x18 | 318 | 82 | 15 | refused | implemented | issue+fault |
| `m.mma.async` | 0x40.0x01 | 296 | 249 | 37 | refused | implemented | issue+matrix_engine_occupancy+barrier_completion |
| `vf.reduce.max` | 0x15.0x09 | 62 | 53 | 28 | implemented-inexact | implemented | issue+fp_reduction_latency |

The full per-opcode and per-variant table is `opcode_inventory.md`; no opcode in the scan is refused
after M1. Static counts only. Before M1 there was also no runtime adapter, so no ArchBench point could run
in the model end to end regardless of opcode coverage.

## 3. Implemented in M1

- Decode-once executor with an explicit verdict per word (implemented / illegal / refused) and a timing class; illegal encodings are ILLEGAL_INSTRUCTION faults, unimplemented ones (device atomics) a model refusal.
- Matrix engine: accumulators A0..A15, `m.zero`, `m.mma.async` FP16/BF16/E5M2/E4M3 at K16/K32/K64 with result and source-free barrier completions, `a.vread`, `a.vwrite`, `a.mul.row`, `a.fma.row`, `a.zero` (RTL zero sweep); one command at a time, core-wide; busy-time formula from `tnca_matrix.v` structure.
- RSLICE (role-mask bit 31, log2 slices in 30:29) remapping of accumulator, barrier, descriptor and window indices and of barrier tokens; RPAIR vector-base aliasing from bit 28; per-context precise `mb.arrive` drain (own register writes, own transfer and matrix commands) as `tnca_core.v` does with RPAIR.
- `fence` (own-scoreboard drain), `trap` (TRAP_USER launch fault, other contexts woken through barrier invalidation), complete special-register table, `.ftz` FP modifiers, scalar LSU lane-0 masking.
- RTL-exact numerics: vector FP through the `tnca_fp32_fma` contract, min/max/compare/reduction folds as `tnca_vector_fpu.v`, float-to-integer conversion as the RTL computes it, and a bit-exact port of `xls/tnca_sfu.x` with its generated tables.
- Physical-grid dispatcher (`tnca_grid_dispatch.v` order, 1/2/4-wide program groups, per-core argument stride, active-core mask), per-program launches, device deadline (status TIMEOUT), faults (status FAULT).
- Host server mode (`tnca-chip --serve`) and runtime backends `systemc-tlm` / `systemc-tlm-grid` registered from outside the product tree (`tlm_backend.py`), one long-lived model process per Session; `run_graphs.py` runs ArchBench points through `archbench.platforms.tnca.run` unchanged.
- Every timing constant moved to `configs/r40-three-core.cfg` and marked uncalibrated; nothing was fitted.
- Host speed: counters bound once, cached configuration, integer time conversion, persistent Role Context, DMA-controller, MMA-completion and memory-beat processes (no SystemC process per program, command or beat; an 80k-program grid went from 5.7 s and 57 MB to 0.24 s and 7 MB with identical cycles), paged backing store for multi-GiB windows.

## 4. Validation (final binaries)

| validation | result | wall s |
|---|---|---:|
| `validate_numeric` | passed (rc 0) | 14.014 |
| `validate_chip` | passed (rc 0) | 0.501 |
| `validate_roles` | passed (rc 0) | 0.461 |
| `validate_dma` | passed (rc 0) | 0.886 |
| `validate_matrix` | passed (rc 0) | 6.569 |
| `validate` | passed (rc 0) | 15.979 |
| `validate_monitor` | passed (rc 0) | 19.988 |

Offline RTL oracles (iverilog over the product RTL, captured once; the model never runs RTL): `tnca_vector_fpu` 4260 vectors over 71 (func, modifier) combinations (vf.*, every conversion, FP reductions, SFU) -> 4260 checked, **0 mismatches**; `tnca_matrix` 502 commands (MMA in every dtype and K, row operations) -> 12288 accumulator words, **0 mismatches**. The matrix capture also records the engine's busy cycles with a behavioural vector file (calibration input, not fitted): mma.k16.dt0 [287], mma.k16.dt1 [287], mma.k16.dt2 [283], mma.k16.dt3 [283], mma.k32.dt0 [551], mma.k32.dt1 [551], mma.k32.dt2 [559], mma.k32.dt3 [559], mma.k64.dt0 [1095], mma.k64.dt1 [1095], mma.k64.dt2 [1095], mma.k64.dt3 [1095], row0 [3], row1 [8], row2 [8], zero [18].

Mutation check: a separately built mutant (`-DTNCA_MUTATION_AVREAD_ROW`, a.vread reads the next row) fails the end-to-end reference comparison on every MMA point and passes the controls:

| point | status | max rel err |
|---|---|---:|
| `op.matmul/tiny/fp16-m16-k32-n16` | not-measured | 1.37 |
| `op.linear/tiny/fp16-m16-k64-n64` | not-measured | 1.25 |
| `op.bmm/tiny/fp16-b2-m16-k16-n16` | not-measured | 1.31 |
| `model.mlp/tiny/fp16-t16-d64` | not-measured | 1.5 |
| `op.add/tiny/fp16-n1000` | pass | 0.000301 |
| `op.softmax/tiny/fp16-r4-c64` | pass | 7.34e-05 |

## 5. End-to-end ArchBench graphs through the model

Configuration: the delivered default (B40-N compile knobs per point from the compile preflight; native tensors; physical grid; `TNCA_ACTIVE_CORES=1`; `TNCA_RTL_MEM_PORTS=1`, `TNCA_RTL_OUTSTANDING=1` mapped onto the model). Device cycles are the steady call's (after ArchBench's cold call), uncalibrated.

**native77: 77 / 77 pass** (medium 25/25, small 25/25, tiny 27/27).

| point | status | launches | device_cycles | max_rel_err | tol | process_wall_s | model_launch_host_s |
|---|---|---:|---:|---:|---:|---:|---:|
| `model.gpt2_block/tiny/fp16-d64-h4-b1-s16` | pass | 13 | 392336 | 0.000289 | 0.05 | 16.64 | 0.46 |
| `model.llama_decode/tiny/fp16-b1-p16` | pass | 42 | 679510 | 0.00025 | 0.05 | 16.87 | 0.73 |
| `model.llama_prefill/tiny/fp16-b1-s16` | pass | 40 | 1038967 | 0.000226 | 0.05 | 25.17 | 0.88 |
| `model.mlp/tiny/fp16-t16-d64` | pass | 2 | 104874 | 0.000426 | 0.05 | 10.50 | 0.12 |
| `model.resnet_block/tiny/fp32-n1-c8-hw8` | pass | 5 | 329102 | 1.28e-07 | 0.001 | 20.50 | 0.48 |
| `op.add/tiny/fp16-n1000` | pass | 1 | 4165 | 0.000301 | 0.02 | 7.44 | 0.02 |
| `op.add/tiny/fp32-n256` | pass | 1 | 1350 | 0 | 0.0002 | 8.46 | 0.01 |
| `op.argmax/tiny/fp16-r2-v256` | pass | 1 | 2403 | 0 | 0 | 9.76 | 0.01 |
| `op.attention_decode/tiny/fp16-b1-h2-l16-d16` | pass | 3 | 23740 | 0.000216 | 0.02 | 8.08 | 0.03 |
| `op.attention_prefill/tiny/fp16-b1-h2-s16-d16` | pass | 7 | 67737 | 0.000236 | 0.02 | 12.82 | 0.06 |
| `op.bmm/tiny/fp16-b2-m16-k16-n16` | pass | 1 | 4189 | 0.00041 | 0.02 | 9.95 | 0.01 |
| `op.conv2d/tiny/fp32-n1-cin4-h8-w8-cout8` | pass | 3 | 143024 | 2.26e-07 | 0.0002 | 12.52 | 0.37 |
| `op.cumsum/tiny/fp32-r4-c64` | pass | 1 | 2079 | 7.22e-08 | 0.001 | 8.40 | 0.01 |
| `op.embedding/tiny/fp16-v256-d64-t16` | pass | 1 | 48626 | 0 | 0 | 9.12 | 0.04 |
| `op.gelu_tanh/tiny/fp16-n256` | pass | 1 | 1937 | 0.000277 | 0.02 | 7.58 | 0.01 |
| `op.gemv/tiny/fp16-k64-n64` | pass | 1 | 18406 | 0.000361 | 0.02 | 9.62 | 0.02 |
| `op.layernorm/tiny/fp16-r4-d64` | pass | 1 | 2462 | 0.000339 | 0.02 | 9.54 | 0.01 |
| `op.linear/tiny/fp16-m16-k64-n64` | pass | 1 | 11781 | 0.000251 | 0.02 | 11.69 | 0.04 |
| `op.matmul/tiny/fp16-m16-k32-n16` | pass | 1 | 2878 | 0.000409 | 0.02 | 8.71 | 0.01 |
| `op.matmul/tiny/fp32-m16-k32-n16` | pass | 1 | 6108 | 1.5e-07 | 0.0002 | 11.39 | 0.01 |
| `op.rmsnorm/tiny/fp16-r4-d64` | pass | 1 | 1926 | 0.000349 | 0.02 | 11.39 | 0.01 |
| `op.rope/tiny/fp16-b1-h2-s16-d16` | pass | 3 | 5052 | 0.000234 | 0.02 | 7.96 | 0.03 |
| `op.silu/tiny/fp16-n256` | pass | 1 | 1467 | 0.00016 | 0.02 | 10.28 | 0.01 |
| `op.softmax/tiny/fp16-r4-c64` | pass | 1 | 4733 | 7.34e-05 | 0.02 | 8.74 | 0.01 |
| `op.softmax/tiny/fp32-r3-c37` | pass | 1 | 1415 | 5.96e-08 | 0.0002 | 10.20 | 0.02 |
| `op.swiglu/tiny/fp16-n256` | pass | 1 | 1704 | 0.000167 | 0.02 | 10.39 | 0.01 |
| `op.transpose/tiny/fp32-r32-c32` | pass | 1 | 5365 | 0 | 0 | 8.72 | 0.02 |
| `model.gpt2_block/small/fp16-d128-h4-b1-s64` | pass | 13 | 6785223 | 0.000272 | 0.05 | 27.62 | 5.08 |
| `model.llama_decode/small/fp16-b1-p128` | pass | 84 | 8246202 | 0.000388 | 0.05 | 34.35 | 9.97 |
| `model.llama_prefill/small/fp16-b1-s64` | pass | 78 | 29785120 | 0.000385 | 0.05 | 57.85 | 32.24 |
| `model.mlp/small/fp16-t64-d128` | pass | 2 | 2275337 | 0.00046 | 0.05 | 9.57 | 1.43 |
| `model.resnet_block/small/fp32-n1-c16-hw16` | pass | 5 | 1544778 | 2.06e-07 | 0.001 | 22.22 | 2.24 |
| `op.add/small/fp16-n16384` | pass | 1 | 147742 | 0.000303 | 0.02 | 7.89 | 0.24 |
| `op.argmax/small/fp16-r8-v2048` | pass | 1 | 48252 | 0 | 0 | 6.09 | 0.11 |
| `op.attention_decode/small/fp16-b1-h4-l256-d32` | pass | 3 | 556532 | 9.64e-05 | 0.02 | 10.68 | 0.77 |
| `op.attention_prefill/small/fp16-b1-h4-s64-d32` | pass | 7 | 3197449 | 0.000309 | 0.02 | 19.70 | 4.32 |
| `op.bmm/small/fp16-b8-m64-k32-n64` | pass | 1 | 1842789 | 0.000265 | 0.02 | 12.83 | 1.82 |
| `op.conv2d/small/fp32-n1-cin16-h16-w16-cout16` | pass | 3 | 758535 | 4.39e-07 | 0.0002 | 12.74 | 1.09 |
| `op.cumsum/small/fp32-r64-c256` | pass | 1 | 160675 | 7.09e-08 | 0.001 | 10.01 | 0.10 |
| `op.embedding/small/fp16-v1024-d128-t64` | pass | 1 | 385165 | 0 | 0 | 10.97 | 0.53 |
| `op.gelu_tanh/small/fp16-n16384` | pass | 1 | 169993 | 0.000237 | 0.02 | 12.23 | 0.29 |
| `op.gemv/small/fp16-k256-n512` | pass | 1 | 669215 | 0.000315 | 0.02 | 11.15 | 0.64 |
| `op.layernorm/small/fp16-r64-d256` | pass | 1 | 159716 | 0.00035 | 0.02 | 8.77 | 0.33 |
| `op.linear/small/fp16-m64-k256-n256` | pass | 1 | 901567 | 0.000225 | 0.02 | 13.71 | 0.94 |
| `op.matmul/small/fp16-m64-k128-n128` | pass | 1 | 162469 | 0.00033 | 0.02 | 9.19 | 0.21 |
| `op.matmul/small/fp32-m64-k128-n128` | pass | 1 | 506132 | 4.04e-07 | 0.0002 | 10.86 | 0.61 |
| `op.rmsnorm/small/fp16-r64-d256` | pass | 1 | 126111 | 0.000251 | 0.02 | 13.08 | 0.32 |
| `op.rope/small/fp16-b1-h4-s64-d32` | pass | 3 | 138085 | 0.00041 | 0.02 | 9.84 | 0.19 |
| `op.silu/small/fp16-n16384` | pass | 1 | 135317 | 0.000241 | 0.02 | 10.37 | 0.11 |
| `op.softmax/small/fp16-r64-c512` | pass | 1 | 591106 | 0.000121 | 0.02 | 9.29 | 0.42 |
| `op.swiglu/small/fp16-n16384` | pass | 1 | 180123 | 0.000285 | 0.02 | 10.19 | 0.17 |
| `op.transpose/small/fp32-r128-c256` | pass | 1 | 338513 | 0 | 0 | 12.77 | 0.36 |
| `model.gpt2_block/medium/fp16-d160-h5-b1-s64` | pass | 13 | 10006458 | 0.000269 | 0.05 | 30.68 | 8.69 |
| `model.llama_decode/medium/fp16-b1-p256` | pass | 84 | 37341525 | 0.000662 | 0.05 | 56.63 | 37.07 |
| `model.llama_prefill/medium/fp16-b1-s64` | pass | 78 | 59680242 | 0.000605 | 0.05 | 89.84 | 61.63 |
| `model.mlp/medium/fp16-t128-d128` | pass | 2 | 4522039 | 0.000406 | 0.05 | 14.07 | 4.24 |
| `model.resnet_block/medium/fp32-n1-c24-hw16` | pass | 5 | 7288758 | 2.72e-07 | 0.001 | 33.88 | 8.91 |
| `op.add/medium/fp16-n49152` | pass | 1 | 443226 | 0.000326 | 0.02 | 10.13 | 0.40 |
| `op.argmax/medium/fp16-r32-v4096` | pass | 1 | 375336 | 0 | 0 | 13.71 | 0.66 |
| `op.attention_decode/medium/fp16-b1-h8-l256-d32` | pass | 3 | 1109281 | 9.64e-05 | 0.02 | 10.34 | 1.10 |
| `op.attention_prefill/medium/fp16-b1-h4-s96-d32` | pass | 7 | 7050000 | 0.000217 | 0.02 | 20.64 | 6.11 |
| `op.bmm/medium/fp16-b16-m64-k32-n64` | pass | 1 | 3684561 | 0.000265 | 0.02 | 17.20 | 4.85 |
| `op.conv2d/medium/fp32-n1-cin24-h16-w16-cout24` | pass | 3 | 3529237 | 5.97e-07 | 0.0002 | 20.24 | 6.63 |
| `op.cumsum/medium/fp32-r128-c256` | pass | 1 | 319560 | 1.42e-07 | 0.001 | 10.89 | 0.49 |
| `op.embedding/medium/fp16-v1024-d128-t256` | pass | 1 | 1656003 | 0 | 0 | 10.76 | 1.61 |
| `op.gelu_tanh/medium/fp16-n65536` | pass | 1 | 671367 | 0.000213 | 0.02 | 10.01 | 0.75 |
| `op.gemv/medium/fp16-k320-n448` | pass | 1 | 717210 | 0.000271 | 0.02 | 10.77 | 0.72 |
| `op.layernorm/medium/fp16-r128-d512` | pass | 1 | 625368 | 0.000377 | 0.02 | 11.05 | 0.52 |
| `op.linear/medium/fp16-m128-k320-n256` | pass | 1 | 2216045 | 0.000227 | 0.02 | 13.25 | 2.63 |
| `op.matmul/medium/fp16-m128-k192-n128` | pass | 1 | 599501 | 0.000306 | 0.02 | 11.48 | 0.48 |
| `op.matmul/medium/fp32-m128-k192-n128` | pass | 1 | 1439167 | 5.99e-07 | 0.0002 | 12.60 | 2.27 |
| `op.rmsnorm/medium/fp16-r128-d512` | pass | 1 | 489502 | 0.00038 | 0.02 | 11.21 | 0.64 |
| `op.rope/medium/fp16-b1-h8-s128-d32` | pass | 3 | 748901 | 0.000371 | 0.02 | 9.56 | 0.40 |
| `op.silu/medium/fp16-n65536` | pass | 1 | 532763 | 0.000259 | 0.02 | 11.17 | 0.67 |
| `op.softmax/medium/fp16-r80-c512` | pass | 1 | 736273 | 0.00021 | 0.02 | 13.00 | 0.37 |
| `op.swiglu/medium/fp16-n49152` | pass | 1 | 540951 | 0.000392 | 0.02 | 10.49 | 0.56 |
| `op.transpose/medium/fp32-r160-c256` | pass | 1 | 425614 | 0 | 0 | 10.53 | 0.42 |

Same suite, tiny and small tiers with all three model cores active (`TNCA_ACTIVE_CORES=3`): **52 / 52 pass** (`native77_tiny_small_three_cores.json`).

A/B experiment configurations of the compile preflight (B_fix2) at tiny/small/medium tiers: **64 / 64 compiled configurations pass**; 1 configuration failed compilation in the preflight itself (code store overflow) and fails identically here. All runs use the v3 campaign's watchdog (`TNCA_DEADLINE=2e11`); under the runtime default of 5e6 cycles five gemm_tile launches would be aborted (`bconfig_tiny_small_medium.json`).

HBM-core suite (14 points; 8 GiB model window, campaign watchdog, 4 h per point): 13 pass of 13 completed; `model.llama_prefill/hbm/fp16-b1-s512` was still running when this bundle was collected (no result is claimed for it) (`hbm_core.json`):

| point | status | launches | device_cycles | max_rel_err | tol | process_wall_s | model_launch_host_s |
|---|---|---:|---:|---:|---:|---:|---:|
| `op.add/hbm/fp16-n67108864` | pass | 1 | 605147405 | 0.000247 | 0.02 | 416.85 | 387.80 |
| `op.copy/hbm/fp16-n67108864` | pass | 1 | 417411520 | 0 | 0.02 | 245.09 | 223.62 |
| `op.silu/hbm/fp16-n67108864` | pass | 1 | 553956953 | 0.000355 | 0.02 | 413.57 | 391.35 |
| `op.rmsnorm/hbm/fp16-r8192-d4096` | pass | 1 | 258980152 | 0.000307 | 0.02 | 316.05 | 301.42 |
| `op.softmax/hbm/fp16-r4096-c8192` | pass | 1 | 636675865 | 0.000149 | 0.02 | 325.27 | 309.80 |
| `op.transpose/hbm/fp32-r8192-c8192` | pass | 1 | 1120288256 | 0 | 0 | 317.38 | 298.32 |
| `op.embedding/hbm/fp16-v65536-d2048-t8192` | pass | 1 | 900905779 | 0 | 0 | 314.09 | 298.04 |
| `op.gemv/hbm/fp16-k4096-n11008` | pass | 1 | 297708515 | 0.000474 | 0.02 | 164.52 | 154.94 |
| `op.attention_decode/large/fp16-b1-h32-l2048-d64` | pass | 3 | 145412722 | 5.71e-05 | 0.02 | 59.02 | 51.22 |
| `op.linear/hbm/fp16-m512-k4096-n11008` | pass | 1 | 8936749190 | 0.000369 | 0.02 | 2194.08 | 2183.73 |
| `op.attention_prefill/large/fp16-b1-h32-s512-d64` | pass | 7 | 2255871064 | 0.000263 | 0.02 | 1619.76 | 1606.21 |
| `model.llama_decode/hbm/fp16-b1-p512` | pass | 84 | 566228643 | 0.00149 | 0.05 | 383.83 | 356.80 |
| `op.matmul/hbm/fp16-m4096-k4096-n4096` | pass | 1 | 12602408317 | 0.000332 | 0.02 | 5784.42 | 5774.65 |

## 6. Speed: model vs RTL simulator on the same graphs

Same tiny graphs, per-program launches in both (`systemc-tlm` vs `sim-file` core-vrf-sync-v5). Host seconds of the steady call's launches (RTL: one call; model: median of five, run alone on the host); the two devices differ (single-core local-SRAM RTL device vs three-core NUCA/HBM model with uncalibrated timing), so only wall time is compared, not cycles.

| point | RTL cycles | RTL s | RTL cycles/s | model cycles | model s | model cycles/s | steady-call speedup | process wall RTL / model s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `op.add/tiny/fp32-n256` | 777 | 2.25 | 346 | 845 | 0.0022 | 376057 | 1001x | 79 / 6.2 |
| `op.softmax/tiny/fp32-r3-c37` | 1703 | 8.40 | 203 | 1404 | 0.0025 | 551020 | 3296x | 94 / 6.3 |
| `op.matmul/tiny/fp16-m16-k32-n16` | 1492 | 2.67 | 558 | 2866 | 0.0024 | 1199665 | 1118x | 82 / 4.9 |
| `op.layernorm/tiny/fp16-r4-d64` | 3418 | 13.58 | 252 | 2455 | 0.0029 | 844223 | 4670x | 99 / 4.3 |
| `op.attention_decode/tiny/fp16-b1-h2-l16-d16` | 9920 | 30.99 | 320 | 9227 | 0.0095 | 974855 | 3274x | 136 / 5.0 |

For these tiny graphs the model time is dominated by fixed host-side launch overhead (about 2 ms per launch on this path, versus well under a millisecond of simulation). On the 20 native77 graphs with more than 1 s of model time it sustains 1.1-3.2 million modeled cycles per host second (median 1.9 M; cold plus steady call cycles over `model_launch_host_s`), and 1.7-8.2 M (median 3.8 M) on the HBM-core points, i.e. roughly 3-4 orders of magnitude above the RTL simulator's 200-560 cycles per second.

## 7. RTL finding

Following the RTL bit-exactly exposed 17015 of 172912 float-to-integer test vectors (`validate_numeric.py`) where the RTL result is not the ISA-text result: `rtl/core/tnca_vector_fpu.v` differs from ISA sec 8.4 in four places: (1) |x| in [2^24, 2^31) converts to 0/+-1 in every mode (8-bit `csh` wraps), (2) odd |x| in [2^23, 2^24) round up under rne (`chalf` is 0 when csh = 0), (3) unsigned x in [2^31, 2^32) saturates to 0xffffffff, (4) unsigned negative x in (-1, 0) gives 0xffffffff under rne (x < -0.5) and rmi. The model follows the RTL; none of these conversions occurs in the frozen suites (opcode inventory). These look like RTL defects worth an RTL-side check; the product tree was not touched.

## 8. Remaining gaps (priority order)

- Timing calibration: every latency/occupancy key is an uncalibrated hypothesis (next milestone, board counters; the RTL-isolated matrix busy cycles above are available as inputs). Modeled cycles must not be ranked or compared with board time yet.
- Core timing fidelity: blocking LSU per context (no LSU_INFLIGHT/non-blocking stores), no VRF bank/port contention, no instruction fetch, issue arbitration is first-come per cycle rather than the RTL round robin with RCTX_SWITCH; DMA and NUCA transactions are 8-byte words.
- Model speed for HBM/large-tier graphs: one SystemC wait per instruction and per memory word, so `op.matmul/hbm` (12.6 G modeled cycles) took 1.6 h of host time and the 24-point large tier was not attempted; a bounded per-core time quantum (opt-in, accuracy-checked) and line-granular transfer transactions are the next levers.
- Host transfers and cache-ownership handoff are coherent but take zero modeled time.
- Device atomics are refused (unused by the suites; old-value width/extension not yet verified against `tnca_lsu.v`); `s.wrspecial` is illegal as in the RTL.
- MMA fragments are read at issue and the model waits for the issuing context's own in-flight vector writes to them; the RTL streams them without an interlock (differs only for programs that race their fragments).
- Decoder legality is checked for the compiler-used subset only; a full differential against `tools/model/gen_decoder_vectors.py` / `tnca_decoder.v` is not done (e.g. the RTL rejects a predicated `v.extract` that the ISA table marks maskable).
- Per-program mode models the three-core NUCA chip, not the single-core local-SRAM `sim-file` device, so its cycles are not comparable with RTL `sim-file` cycles.

## 9. Files

- `identity.json`: binaries, harness source and configuration hashes
- `baseline.json`: pre-M1 validations
- `validation_m1.json`: final validations, float-to-integer RTL/ISA divergences, matrix directed tests
- `rtl_oracles.json`: RTL oracle captures and the model check
- `opcode_inventory.json, opcode_inventory.md`: opcode inventory with before/after verdicts
- `native77.json`: all native77 points, delivered configuration
- `native77_tiny_small_three_cores.json`: tiny+small with three active cores
- `bconfig_tiny_small_medium.json`: A/B experiment configurations
- `hbm_core.json`: HBM-core points
- `rtl_vs_model_speed.json`: model vs RTL wall time on the same graphs
- `mutation_avread_row.json`: mutant sensitivity check
- Raw records, per-launch logs, counters, kernel assembly and oracle captures: GB10 `/home/zhongchun/tlm_m1/m1_raw_evidence.tar.gz` (46361311 bytes, sha256 `768297aa6b0f205fe88c6a339e4fba663ff0d203c9ff4f44d479412ce125fb9c`).

Reproduce on the GB10 (paths as used here):

```sh
cmake -S tools/performance_harness/systemc -B BUILD -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH=$HOME/tc_env/systemc-3.0.2/install && cmake --build BUILD --parallel 4
python3 tools/performance_harness/systemc/run_graphs.py --source-root SRC --binary BUILD/tnca-chip \
  --out OUT --suite SRC/spec/r40/suites/existing.json --knobs-from ~/r40v3_fix2/A_fix2/runs \
  --torch-extensions-dir EXT --jobs 4
python3 tools/performance_harness/systemc/opcode_inventory.py --runs A=~/r40v3_fix2/A_fix2/runs \
  --runs B=~/r40v3_fix2/B_fix2/runs --classifier BUILD/tnca-chip --out inv.json --markdown inv.md
python3 tools/performance_harness/systemc/rtl_oracle/rtl_oracle.py --root SRC --out ORACLE_VFPU
python3 tools/performance_harness/systemc/rtl_oracle/matrix_oracle.py --root SRC --out ORACLE_MTX
BUILD/tnca-numeric-probe vfpu ORACLE_VFPU/oracle.txt; BUILD/tnca-numeric-probe matrix ORACLE_MTX
```
