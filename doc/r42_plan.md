# R42: plan, registration and log

> **Binding:** [`engineering_discipline.md`](engineering_discipline.md). Every number below names its domain
> (RTL simulation, SystemC/TLM model, board, analytic). Nothing here is a board result unless it says so.

Branch `R42`, opened from `main` at `4b61abd8` on 2026-11-20. Goal given by the author: implement the reasonable
parts of the scaling blueprint (wider tensor core, line-wide memory path, compiler and runtime work) and obtain a
significant improvement of the ArchBench geometric mean.

## 1. Registration

The first registration (tiny and small tiers, single-core simulator, 2026-11-20) was withdrawn the same day on the
author's instruction, before any R42 change existed: the yardstick is the whole suite and the baseline is the R41
three-core instance. Source: the author's chat message of 2026-11-20, "你为什么只以tiny small为基准？以及你的baseline应该是R41
3核实例". The single-core runs started under the first registration are kept as diagnostics only.

**Benchmark set.** All 113 frozen original ArchBench points (tiers tiny, small, medium, large, hbm), every point,
through the standard path with default knobs.

**Baseline (B41).** The R41 three-core U280 instance: three cores, a NUCA home slice per core, three HBM outlets,
100 MHz. Two R41 images have complete 113-point board records, and both are reported:
- B41-run1, image `30e79769` (profile `core-vrf-sync-v5-x8`, 256-bit outlets), the image of `doc/r41_report.md`;
- B41-run3, image `791b8af2` (profile `core-vrf-sync-v5-x8-g2c4-dp`, line-wide path, `m.mma.dp`), the latest R41
  three-core image (`evidence/r41/hbm_bw/g4/run3/board/`). It is the stronger baseline and the primary one: work
  that R41 already built is not counted as an R42 gain.

**Candidate.** The `R42` head as a three-core instance of the same shell, default knobs.

**Domain.** Board measurements are the only basis of a result (author, chat, 2026-11-20: "以上板测试数据为唯一可靠基准").
The R42 result is the 113 points measured on the U280 with an R42 three-core image, against the board records of
the R41 images above, same harness and boundary. That needs a full link and a card load; the load is requested from
the author when a candidate image has passed its gates.

RTL simulation of the three-core image (`fpga-multicore-sim`, board top, image geometry), the SystemC/TLM model and
single-core RTL simulation are screens: they decide what goes into the image, they are labelled as such, and they
never stand in for a board number.

**Metric.** `S_geo = exp(mean_i log(T_B41,i / T_R42,i))` over the points of a domain, `T` the serial-grid device
time at equal clock (cycles); the per-tier and per-family geometric means and the per-point table are reported
beside it. A point failing numerically on either side is reported failed and the result is "not certified".
Every point slower than the baseline by more than 2 % is listed with its cause. A clock-frequency gain is a
separate claim that needs routed timing.

**Numerics.** Each point's frozen tolerance. No tolerance changes.

**Hardware cost.** Every RTL change records per-module resources and out-of-context timing before it is counted as
deliverable on the U280; SLR packing of the R41 image is the budget (`evidence/r41/hbm_bw/RESOURCES.md`).

## 2. Triage of the blueprint

Ordering rule: the result is a board geometric mean over 113 points, so a family at eta 0.02 counts as much as one
at 0.6. The board baseline (`evidence/r42/baseline/b41_board.json`, run-3 image) puts attention_prefill (0.021),
conv2d (0.032), resnet_block (0.035), attention_decode (0.057), rope, softmax, bmm and the norms at the bottom.
The R41 image is about 96 % packed in SLR0, so hardware that goes into the R42 image has to be nearly free or paid
for by a removal.

**Stage A: software on the R41 run-3 image** (measurable on the board without a link)

| Item | Blueprint row | State |
|---|---|---|
| Tiled element-wise lowering as the default | Full Inductor fusion and numerics | in work (`R42-red`) |
| Reductions, norms, softmax, argmax, rope lowering | same | in work (`R42-red`) |
| Attention (SDPA) and bmm lowering | TNCA_SDPA / fusion | in work (`R42-attn`) |
| GEMM register blocking and overlap of transfers with matrix commands | 64x64 blocking, async pipelining | next |

**Stage B: RTL for the R42 image** (one link, then the board)

| Item | Blueprint row | State |
|---|---|---|
| Matrix engine without bubbles: B prefetch, command overlap | Zero-bubble matrix pipeline | in work (`R42-mtx`) |
| FP32 element format on the matrix engine (conv2d, resnet, fp32 matmul) | native formats | in work (`R42-fp32`) |
| h4 set index, allocate-on-fill in the home slice | Production entry prefetch + h4 | next, small |
| Line-wide path (LINE_ENGINE, hybrid LSU) | Native 512-bit datapath | already in the R41 run-3 image; kept |
| `m.mma.dp` grouped dot product | K-fused tensor core | already in the R41 run-3 image; kept |

**Not in this round, with the reason**

| Item | Reason |
|---|---|
| 128-512 MAC/cycle core, DSP48E2 mapping, 6 cores at 250-300 MHz | A new physical design: every lane mapped to DSPs, the VRF read rate raised to feed it, a new floorplan and timing closure. It cannot be screened by one link on a 96 %-packed image. The matrix engine is busy about 44 % of a GEMM point's cycles (R40 model attribution), so the binding limits today are transfers and issue; those are attacked first. |
| Dual-issue front end | The front-end-bound share measured in R39 is 4.3 %. |
| URAM NUCA, 16-24 HBM ports, 256/512-bit NoC | The home engine and the core port bind before the outlets (R41 link budget: outlets busy 2-5 %). Kept for the image after this one. |
| C++ MLIR middle end | It changes compile time, not device cycles, which is what ArchBench scores. |
| Command ring, device-side decode loop | They change host time, not the serial-grid device time ArchBench scores. |
| Hardware multicast, offline weight pre-packing, graph-level layout pass | No ArchBench point is bound by them today; revisit with the model points. |

## 3. Log

**2026-11-20.**
- Branch opened; registration rewritten twice on the author's instructions (113 points; R41 three-core baseline;
  board only).
- Board baseline table from the R41 records: `tools/r42/baseline_table.py` -> `evidence/r42/baseline/b41_board.json`.
  Run 3 over run 1: geometric mean 1.347x over the 113 points (board).
- Screening environment: GB10 and thor. A first set of single-core screening runs on GB10 imported the compiler
  package (`tnca_cc`) from an unrelated old mirror because only `python/` was on `PYTHONPATH`; those records were
  discarded unused (moved to `~/r42/invalid_stale_compiler` on GB10) and the runs restarted with
  `PYTHONPATH=python:compiler/python:tools/asm`.
- Work packages started on their own branches: `R42-red` (tiled element-wise default, reductions and norms),
  `R42-attn` (attention, bmm), `R42-gemm` (register blocking, transfer overlap), `R42-mtx` (matrix engine B
  prefetch and command overlap, RTL), `R42-fp32` (FP32 matrix format, ISA + RTL + compiler); a board worker
  re-measures the 113-point baseline on the run-3 image with the software of `main` through the guarded queue.

**2026-11-21. First board A/B (`ab-int1`).** Board, U280, run-3 image `791b8af2` on both sides, 100 MHz,
serial-grid device cycles, default knobs, 113/113 points pass on both sides
(`evidence/r42/board/ab-int1/comparison.json`, computed by `tools/r42/board_compare.py`).
- Baseline `b41-run3-base`: software of `main` `4b61abd8`. It reproduces the R41 run-3 records at a geometric mean
  of 0.9995.
- Candidate: `R42` at `da6be2b8` (the merge of `R42-red` and `R42-attn`): tiled element-wise lowering by default,
  the loop pass, 64-lane reduction blocks, the attention/bmm/GEMM-pointer fixes.
- Result: geometric mean speed-up **1.613x** over the 113 points; sum of cycles 1.305x. By tier: hbm 1.72x,
  large 1.70x, medium 1.84x, small 1.68x, tiny 1.28x.
- Slower by more than 2 %: `op.matmul/small/fp16-m64-k128-n128` (1.049), `op.gemv/hbm` (1.024),
  `op.gemv/large` (1.022). Causes not yet established.
This is a software-only result on the R41 image. The RTL work (`R42-mtx`, `R42-fp32`) and `R42-gemm` are not in it.

## 4. XL tier registration (2026-11-23)

> **Binding:** [`engineering_discipline.md`](engineering_discipline.md) 2.3. This section is the registration of a new
> ArchBench tier. It is written, frozen (`archbench/suites/xl.json`) and pushed before any xl result exists: no xl point has
> run on a board, an RTL simulator or the SystemC/TLM model. Every number below is labelled with its domain.

**Request.** The author asked on 2026-11-23, as relayed in the brief of worker `xl`: "we need a suite bigger than `large`,
with a higher share of FP8 matrix multiplication and of really AI-practical kernels, but make sure the suite does not blow
the device memory." The 113 original points and their five frozen suites are unchanged (`python -m archbench.suite --check`;
their `suites/*.json` are byte-identical to `origin/R42`).

**What it is.** A tier `xl` of 32 points (`archbench/xl.py`), 18 of them (56%) built on block-wise FP8 matrix
multiplication through the Hub kernels `kernels-community/finegrained-fp8` (v4, kernel commit `fe2bb66`), loaded unchanged
and pinned by file sha256 (`archbench/fp8_hub.py`, `archbench/third_party/finegrained_fp8`: `matmul_2d` with E4M3 weights
and 128 x 128 fp32 scales, activations quantised per row and 128-wide K block inside the kernel, fp16 in and out;
`fp8_act_quant`). Everything else is fp16. All points go through the standard path (`torch.compile`, Inductor, Triton, the
tnca backend, default knobs); nothing in the compiler, runtime or harness recognises a name or a shape.
The shapes are those of the decoder the R41 FP8 work validated, the text decoder of Qwen3-VL-2B-Instruct-FP8: hidden 2048,
MLP 6144, 16 query heads and 8 KV heads of 128, 28 layers, vocabulary 151936, tied embedding and head
(`tools/r41/balance_model.py` `MODELS["vl"]`, checked there against the archived `plan.json` FP8 FLOPs; its KV state of
29 360 128 B is 28 x 2 x 8 x 256 x 128 x 2 B).

### 4.1 The points

| # | point | live MiB | bound Mcycles (binding) | est. device time (eta table) | pessimistic | tolerance | compile-only |
|--:|---|--:|--:|--:|--:|--:|---|
| 1 | `model.fp8_decoder_decode/xl/fp8-b1-p2048-layers28` | 2187 | 283.44 (mem_bytes) | 14.9 s | 40.6 s | 0.2 | compiles, 86 s, 618 launches, 50 kernels |
| 2 | `model.fp8_decoder_prefill/xl/fp8-b1-s2048-layers4` | 1743 | 5,642.52 (fp_lane) | 11.7 min | 24.3 min | 0.02 | compiles, 92 s, 82 launches, 22 kernels |
| 3 | `model.fp8_decoder_prefill/xl/fp8-b1-s512-layers28` | 2034 | 9,298.96 (fp_lane) | 10.9 min | 33.0 min | 0.02 | compiles, 130 s, 562 launches, 22 kernels |
| 4 | `model.fp8_layer_decode/xl/fp8-b1-p2048-layers1` | 80 | 7.34 (mem_bytes) | 0.5 s | 1.4 s | 0.02 | compiles, 37 s, 22 launches, 15 kernels |
| 5 | `model.fp8_layer_prefill/xl/fp8-b1-s512-layers1` | 147 | 331.97 (fp_lane) | 23.2 s | 70.6 s | 0.02 | compiles, 82 s, 23 launches, 17 kernels |
| 6 | `op.argmax/xl/fp16-r16-v151936` | 5 | 0.61 (mem_bytes) | 0.0 s | 0.0 s | 0 | compiles, 3 s, 1 launches, 1 kernels |
| 7 | `op.attention_decode/xl/fp16-b1-h16-kv8-l2048-d128` | 24 | 1.05 (mem_bytes) | 0.1 s | 0.1 s | 0.02 | compiles, 3 s, 3 launches, 3 kernels |
| 8 | `op.attention_decode/xl/fp16-b1-h16-kv8-l4096-d128` | 49 | 2.10 (mem_bytes) | 0.3 s | 0.3 s | 0.02 | compiles, 3 s, 3 launches, 3 kernels |
| 9 | `op.attention_decode/xl/fp16-b1-h16-kv8-l8192-d128` | 98 | 4.20 (mem_bytes) | 0.6 s | 0.6 s | 0.02 | compiles, 3 s, 3 launches, 3 kernels |
| 10 | `op.attention_prefill/xl/fp16-b1-h16-kv8-s1024-d128` | 212 | 68.22 (fp_lane) | 27.5 s | 27.5 s | 0.02 | compiles, 8 s, 7 launches, 7 kernels |
| 11 | `op.attention_prefill/xl/fp16-b1-h16-kv8-s2048-d128` | 808 | 272.76 (fp_lane) | 110.0 s | 110.0 s | 0.02 | compiles, 8 s, 7 launches, 7 kernels |
| 12 | `op.embedding/xl/fp16-v151936-d2048-t2048` | 602 | 2.10 (mem_bytes) | 0.7 s | 0.7 s | 0 | compiles, 3 s, 1 launches, 1 kernels |
| 13 | `op.fp8_act_quant/xl/fp8-m2048-k2048` | 12 | 1.59 (mem_bytes) | 0.1 s | 0.3 s | 0.08 | compiles, 2 s, 1 launches, 1 kernels |
| 14 | `op.fp8_act_quant/xl/fp8-m2048-k6144` | 36 | 4.77 (mem_bytes) | 0.2 s | 0.9 s | 0.08 | compiles, 2 s, 1 launches, 1 kernels |
| 15 | `op.fp8_linear/xl/fp8-m1-k2048-n12288` | 24 | 3.15 (mem_bytes) | 0.2 s | 0.6 s | 0.004 | compiles, 32 s, 1 launches, 1 kernels |
| 16 | `op.fp8_linear/xl/fp8-m1-k2048-n151936` | 297 | 38.94 (mem_bytes) | 1.9 s | 7.6 s | 0.004 | compiles, 32 s, 1 launches, 1 kernels |
| 17 | `op.fp8_linear/xl/fp8-m1-k2048-n4096` | 8 | 1.05 (mem_bytes) | 0.1 s | 0.2 s | 0.004 | compiles, 33 s, 1 launches, 1 kernels |
| 18 | `op.fp8_linear/xl/fp8-m1-k6144-n2048` | 12 | 1.58 (mem_bytes) | 0.1 s | 0.3 s | 0.004 | compiles, 33 s, 1 launches, 1 kernels |
| 19 | `op.fp8_linear/xl/fp8-m16-k2048-n12288` | 24 | 5.07 (fp_lane) | 0.3 s | 1.0 s | 0.004 | compiles, 63 s, 1 launches, 1 kernels |
| 20 | `op.fp8_linear/xl/fp8-m16-k2048-n151936` | 302 | 62.72 (fp_lane) | 3.1 s | 12.3 s | 0.004 | compiles, 65 s, 1 launches, 1 kernels |
| 21 | `op.fp8_linear/xl/fp8-m16-k6144-n2048` | 12 | 2.54 (fp_lane) | 0.1 s | 0.5 s | 0.004 | compiles, 62 s, 1 launches, 1 kernels |
| 22 | `op.fp8_linear/xl/fp8-m2048-k2048-n12288` | 80 | 649.42 (fp_lane) | 32.5 s | 2.1 min | 0.004 | compiles, 60 s, 1 launches, 1 kernels |
| 23 | `op.fp8_linear/xl/fp8-m2048-k2048-n2048` | 20 | 108.32 (fp_lane) | 5.4 s | 21.2 s | 0.004 | compiles, 61 s, 1 launches, 1 kernels |
| 24 | `op.fp8_linear/xl/fp8-m2048-k2048-n4096` | 32 | 216.54 (fp_lane) | 10.8 s | 42.4 s | 0.004 | compiles, 60 s, 1 launches, 1 kernels |
| 25 | `op.fp8_linear/xl/fp8-m2048-k6144-n2048` | 44 | 324.97 (fp_lane) | 16.2 s | 63.6 s | 0.004 | compiles, 61 s, 1 launches, 1 kernels |
| 26 | `op.fp8_linear/xl/fp8-m512-k2048-n12288` | 38 | 162.36 (fp_lane) | 8.1 s | 31.8 s | 0.004 | compiles, 60 s, 1 launches, 1 kernels |
| 27 | `op.fp8_linear/xl/fp8-m512-k6144-n2048` | 20 | 81.24 (fp_lane) | 4.1 s | 15.9 s | 0.004 | compiles, 59 s, 1 launches, 1 kernels |
| 28 | `op.gemv/xl/fp16-k2048-n151936` | 594 | 77.83 (mem_bytes) | 2.2 s | 2.2 s | 0.02 | compiles, 3 s, 1 launches, 1 kernels |
| 29 | `op.linear/xl/fp16-m2048-k2048-n6144-bias0` | 56 | 805.31 (fp_lane) | 12.6 s | 12.6 s | 0.02 | compiles, 3 s, 1 launches, 1 kernels |
| 30 | `op.rmsnorm_residual/xl/fp16-r2048-d2048` | 40 | 4.19 (mem_bytes) | 0.3 s | 0.3 s | 0.02 | compiles, 2 s, 1 launches, 1 kernels |
| 31 | `op.rope/xl/fp16-b1-h16-s2048-d128` | 33 | 2.23 (mem_bytes) | 0.4 s | 0.4 s | 0.02 | compiles, 3 s, 3 launches, 3 kernels |
| 32 | `op.swiglu/xl/fp16-r2048-i6144` | 96 | 9.44 (mem_bytes) | 0.5 s | 0.5 s | 0.02 | compiles, 3 s, 1 launches, 1 kernels |

Total estimated device time 27.2 min (pessimistic 67.2 min); largest point 11.7 min (pessimistic 33.0 min); FP8 matrix points 18/32 = 56%; largest live set 2187 MiB of 3072 MiB.

"Live MiB" is arguments + outputs + scratch as declared by the point (4.3). "Bound" is the strict TNCA lower bound on the
cores of the R41 run-3 image (profile `core-vrf-sync-v5-x8-g2c4-dp`) from the semantic demand. "Est." and "pessimistic" are
planning figures (4.4), not results.

### 4.2 What was shrunk, and what was left out

Shrunk against the real model, stated as the reduction:
- **Prefill at 2048 tokens covers 4 of 28 layers** (1/7): `fp8_decoder_prefill/.../s2048-layers4`. The whole decoder at 2048
  tokens would be 7 x that point, about 82.0 min at the planning eta, over the ~12 minutes a point may take. The whole
  28-layer decoder is registered at 512 tokens.
- **Decode with 16 tokens is registered for the projections and the head only**, not for attention or a KV cache: sixteen
  sequences of 2048 positions across 28 layers are 3.7 GB of cache, past the 3 GiB cap. Decode attention is `b = 1`, with
  caches of 2048, 4096 and 8192 entries.
- The text decoder only (no vision tower, no multimodal RoPE), RoPE with the LLaMA default base, no per-head q/k norm;
  separate q, k, v, gate and up projections as in the checkpoint. Weights and activations come from the point's seed, as
  for every other point (`archbench/README.md`): ArchBench measures time and agreement with the reference, which do not
  need trained values.
- Activations are fp16 (the dtype of every ArchBench point), not the checkpoint's bf16. bf16 compiles with the same kernels
  (checked once on the CPU path) and is not registered.
- The lm_head of the FP8 checkpoint is a bf16 tied matrix; the real head is the fp16 `gemv` point (a 594 MiB weight). The
  FP8 head points (`fp8_linear` with N = 151936) are what a fully quantised deployment would run.

Not defined: the Hub kernels' grouped/batched mixture-of-experts entry points (no model here uses them), FP8 attention or an
FP8 KV cache (no Hub kernel exists for them), MXFP8/MXFP4 weights (the Hub `mxfp_dynamic_matmul` path; the TNCA matrix
engine has E4M3 and E5M2, and block scales that are not UE8M0 are what the checkpoints use).

### 4.3 Memory: at most 3 GiB live, declared and asserted

The board exposes 4 GiB of device memory to a launch. Every point's definition carries `extra.memory` (hashed into the
suite): `args_bytes` (inputs, weights, scales, caches, RoPE tables, embedding table), `out_bytes`, `scratch_bytes` and their
sum `live_bytes`; `xl.memory()` raises at registration above `CAP_BYTES` = 3 GiB, and `python -m archbench.selftest`
recomputes arguments and outputs from the point's own tensors (built on the meta device, so the check costs no memory) and
fails on any difference. Scratch is an upper bound by construction: every intermediate of the largest layer alive at once,
no buffer reuse and no fusion saving, attention scores kept in fp16 and the softmax written out in binary32 (12 bytes per
score). The largest live set is 2187 MiB (`fp8_decoder_decode`, 28 layers: 1.4 GiB of E4M3 weights and scales, the
594 MiB tied table, a 235 MiB KV cache). Not observed: no backend that can hold 3 GiB exists off the board, so these are
analytic bytes verified against tensors, not a `heap_peak_bytes` reading; the first board run records it.

### 4.4 Runtime estimate (planning, domain: analytic from the R41 board baseline; no xl point has run)

Per point, `T = sum over its parts of bound_part / eta_part`, 100 MHz (`archbench/xl_estimate.py`, recorded per point in
`xl.json` with the parts):
- the bound as in 4.5;
- eta of an fp16 part: that of the same family's `large` point on the board in `evidence/r42/baseline/b41_board.json`
  (run-3 image), recomputed by `tools/r42/xl_eta.py` into `archbench/suites/xl_eta.json`: attention prefill 0.025, attention decode 0.074, rmsnorm 0.151, swiglu 0.205, rope 0.057, gemv 0.361, linear 0.638, embedding 0.028, argmax 0.174;
- eta of an FP8 matrix part: **0.2, the figure of the brief**; no FP8 point has run on a board;
- composites are split into FP8 projections, attention, element-wise kernels and the head, each at its own eta, because
  they differ by an order of magnitude (the attention of a layer at 512 tokens is 2 % of its MACs and 30% of its estimated time).

Result: 27.2 min of device time per pass, the largest point 11.7 min (`model.fp8_decoder_prefill/xl/fp8-b1-s2048-layers4`); no point exceeds the ~12 minutes.
Host work is not in it: cold compile of the tier on thor (two points in parallel) totals 23.1 min of process time
(for scale, the 113-point suite spent 6 077 s of process time for 8.2 min of device time, about 49 s of host work per point),
the CPU references take up to 75 s for the 28-layer prefill (thor), and the board-host figures are unknown.
**Sensitivity.** A pessimistic column replaces the 0.2 by the efficiency of the SystemC/TLM model (uncalibrated, domain
`model`) on the four real-layer shapes of `evidence/r41/fp8/kernels/README.md` (eta 0.051, profile without `m.mma.dp`):
67.2 min per pass, the largest point 33.0 min (`model.fp8_decoder_prefill/xl/fp8-b1-s512-layers28`), i.e. past the 12-minute mark for `model.fp8_decoder_prefill/xl/fp8-b1-s2048-layers4`, `model.fp8_decoder_prefill/xl/fp8-b1-s512-layers28`.
If the first board run shows the pessimistic figures, the tier takes that long; nothing in the registration depends on the
estimate.

### 4.5 Bound assumptions

Written in `archbench/PROTOCOL.md` 4.1 before any result: the E4M3 matrix rate is 64 MAC/cycle on a profile with `matrix_dp` and
`matrix_fp8` (16 lanes x `m.mma.dp.async` DP4, ISA 9C.4, `rtl/core/tnca_matrix.v`, `rtl/core/tnca_mac_lane.v`), 16 otherwise;
the vector FPU adds 16 FMA lanes; other work runs on the vector FPU only; the memory term is the core's one 8-byte
port. A point with 16-bit MACs inside an FP8 composite is credited the E4M3 rate for all (a lower, still valid, bound). fp16
points keep the convention of the 113 points (`mma_mac` 16), which on the dp image is loose by up to 1.5x for a compute-bound
fp16 point. The GPU tensor rate for E4M3 is 2 x the fp16 one (`specs/gpu_gb10.json`, additive). Demand per operation: PROTOCOL.md 4.1.

### 4.6 Tolerances (fixed now, from CPU rounding floors and the FP8 rounding floor)

| Points | Tolerance | Derivation |
|---|---|---|
| 13 `fp8_linear` | 4e-3 | four fp16 ulps of the top binade, 4 x 2^-10; the binary32-accumulating CPU twin differs from the binary64 reference by at most 7.7e-4 over 13 points x 3 seeds |
| 2 `fp8_act_quant` | 8e-2 | one E4M3 code of the top binade, 32/448; the faithful (non-IEEE) quotient of `fp8_act_quant` flips about 0.5 % of codes by one ulp |
| 4 `fp8_layer_*` and `fp8_decoder_prefill` | 2e-2 | the registry fp16 value; measured floors 6.6e-4 to 4.0e-3 |
| `fp8_decoder_decode` | 0.2 | 3 x the largest floor, 4.4e-2 in the new K/V of the deepest layers (logits 5e-3), rounded up; detects gross failure only |
| 12 fp16 points | 2e-2, 0 for `embedding` and `argmax` | registry; the argmax data of seed 0 has no tie at its maximum (selftest) |

The reference of an FP8 point is the same module in binary64 with exact block-wise arithmetic (PROTOCOL.md 4.2). The floors
are CPU measurements: `tools/r42/xl_tolerance.py`, `evidence/r42/xl/tolerance/`. The decoder initialisation (depth-scaled
output projections, unit embedding, final norm H^-1/2) was chosen by those measurements, before any device output, because with
unit-variance branches and a 0.02 embedding two correct implementations disagree by 26 % of the largest logit after 28 layers
(`v1_unit_init.log`), with only the output projections depth-scaled by 25 % (`v2_resid_only_scaled_embedding_0.02.log`), and with the
unit embedding as well by 0.5 % (`floors_final.log`, the points as registered); no tolerance is changed after a device output exists.

### 4.7 Verification done on thor (domain: CPU and compile-only; no board, no RTL simulation, no TLM)

All at the tree of the commit that adds this section, on thor (aarch64, 14 cores shared), nice 19, at most 3 processes; PYTHONPATH with `python`, `compiler/python`
and `tools/asm` of the tree, checked to import from it; `evidence/r42/xl/` holds the raw outputs.
- `python -m archbench.selftest`: PASS (tiny, small and xl points: MACs never above the FLOP counter, exact for the dense
  families, bytes, live memory <= 3 GiB, bounds positive on `core-sim`, `core-vrf-sync-v5/hbm`, the dp profile and the GPU spec,
  scoring refusals, suites).
- `python -m archbench.suite --freeze` then `--check`: the six existing suite files (`tiny`, `small`, `medium`, `large`, `hbm`, `r41_membound`) byte-identical, `xl.json` 32 points,
  `sha256` field 2fd48cb3da4817c1129e4fef95e6ea6ef177c6d791da68bc5f4dd0c0cd8779d2 (file sha256 42dd79b5939c72a8eb26deb4ff97d65dee84373deb7f23b97d10dbd060b4bd16).
- `--compile-only` of every point on `core-vrf-sync-v5-x8-g2c4-dp` (`tools/r42/xl_compile_only.sh`, `xl_compile_summary.py`,
  cold caches per point): 32 of 32 compile (none fail). The table above lists compile seconds, launches (equal to the generated wrapper's
  count), distinct compiled kernels and the verdict. The record status of 32 of 32 points is `capacity-exceeded`
  by construction of the CPU-tensor path (a launch's arguments exceed its 384 KiB staging heap); that is not a compile
  failure, and `verdict` says `compiles` only when nothing else is in the record, no host fallback appears and the launch
  count matches. The FP8 projections compile to static 32 x `m.mma.dp.async.k32.e4m3` per 128-deep K block at M = 2048 (the kernel body is a loop over 16-row chunks); the 28-layer decoders compile to 618 launches of 50 distinct kernels (decode) and 562 launches of 22 (prefill at 512 tokens).
- CPU numerical check of the smallest point of each family (`tools/r42/xl_cpu_check.sh`; fp16 families: the module in fp16
  eager against the registry reference; FP8 families: the CPU twin against the binary64 reference): 15 of 15 pass; largest peak RSS 3.53 GiB; largest error / tolerance 0.20 (`model.fp8_decoder_decode/xl/fp8-b1-p2048-layers28`), the two exact points (`embedding`, `argmax`) 0 mismatches.
- The unchanged Hub kernels in Triton's CPU interpreter against the references (`tools/r42/xl_interpreter_check.py`): with
  inputs on which every float8 conversion is exact (the interpreter's own conversion is not faithful), `matmul_2d` is
  bit-equal to the CPU twin and within one fp16 ulp of the binary64 reference on five shapes (M = 1 to 130), and
  `fp8_act_quant` codes and scales are equal. This checks the reference's structure against the real kernel source.

### 4.8 Not done

- No xl point ran on a board, the RTL simulator or the SystemC/TLM model. FP8 efficiency, correctness on TNCA and the device
  memory high-water mark are unknown until the first board pass; the first pass is the measurement, and every point is
  reported.
- The compile-only sweep is on the CPU-tensor path: the native path needs a device with the whole 3 GiB heap, which no host
  except the board offers.
- The `archbench.run` launch-count check counted only `triton_*` kernels; it now also counts user-defined Triton kernels
  (`<name>_<n>.run(`), otherwise every FP8 point would be reported `not-measured` (`1 launches, but the wrapper makes 0`).
  The 113 original points are unaffected in the way that matters: `--compile-only` of the 52 tiny and small points with the changed count gives 52 `not-measured` and 0 others, none `runtime-fault` for a launch-count mismatch (`evidence/r42/xl/nonreg_tiny_small.json`).

### 4.9 Running it

```sh
# on the FPGA host, as the 113 points (evidence/r42/board/RECIPE.md): one process per point, ids from suites/xl.json
python -m archbench.run --platform tnca --backend fpga-u280-multicore --profile core-vrf-sync-v5-x8-g2c4-dp \
    --backend-args '<the run-3 backend arguments>' --point <id> --timeout 28800 --out results/xl.jsonl \
    --set TNCA_ACTIVE_CORES=3 --set TNCA_TORCH_DEVICE=native --set TNCA_TRANSFER_POLICY=resident
```
The Hub kernels need no network: `archbench/third_party/finegrained_fp8` is used when neither `ARCHBENCH_FINEGRAINED_FP8`
nor the `kernels` package provides the pinned build. Report every point: a point that fails stays failed.
