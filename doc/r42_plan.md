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

**2026-11-23. Numerics decision for the wide core (author, chat: "尽管数值精度有别，我们还是要比较FP8之前版本和用了DSP后的新版本。最终关注的是软件层面精度是正确的").**
A DSP-based K-fused dot-product lane (G products aligned to the group's maximum exponent, a wide fixed-point adder
tree, one rounding per group) may differ in bits from the sequential one-rounding-per-step fp32 FMA of ISA sec 9C.1.
The author accepts that, and wants the previous FP8 version and the new DSP version compared. What is judged is that
accuracy at the software level is correct. Consequences, registered here before any emulation result exists:
1. The new lane is a NEW mode/opcode beside the existing one; the sequential contract stays available and
   reproducible, so the two versions can be compared on the same image and the same points.
2. Accuracy gate (no tolerance is changed after results are seen): every point of the 113 set and of the `xl` tier
   stays inside its frozen tolerance with the new lane, AND the relative L2 error of each GEMM-based point against
   its binary64 reference is at most 1.5x the error of the sequential path on the same inputs, AND the end-to-end
   checks the R41 contract names (rel L2 <= max(0.05, 2x CPU rounding floor) with the chosen-token match, and the
   registered chat-quality check) still pass. A failure is a result and is kept.
3. Order of work: a bit-accurate Python model of the lane and an accuracy study on CPU first (no hardware, no board);
   RTL and resources only after the gate is met in the model; speed is then measured on the board with the same
   A/B recipe as the software rounds.

**2026-11-23. Second board A/B (`ab-int2`).** Board, U280, run-3 image `791b8af2` on both sides, 100 MHz, serial-grid
device cycles, default knobs, 113/113 points pass on both sides (`evidence/r42/board/ab-int2/comparison.json`).
- Candidate: `R42` at `5327118c` (`ab-int1` plus `R42-gemm`: broadcast-row fragment and the GEMM software pipeline).
  Queue ran 14:07-16:10 UTC, 5,432 s of board time, no failed configuration, no missing formal record.
- Result: geometric mean speed-up **1.638x** over the 113 points (`ab-int1`: 1.613x); sum of cycles 1.312x
  (`ab-int1`: 1.305x). By tier (new/old): hbm 0.567, large 0.581, medium 0.537, small 0.584, tiny 0.774.
- No point is slower than the baseline by more than 2 % (largest ratio 1.019). The three `ab-int1` regressions are
  gone: `op.gemv/hbm` and `op.gemv/large` now 1.19x and 1.17x faster, `op.matmul/small/fp16-m64-k128-n128` within noise.
  Family gemv 0.99 -> 1.17, llama_decode 1.73 -> 1.91.
- What the GEMM pipeline did not do: fp16 GEMM points are unchanged to within 1-4 % (matmul fp16 large 1.04x, hbm
  1.04x, linear 1.02, mlp 1.03). Those points are bound by the matrix engine and the transfer rate, not by the issue of
  software around them; the software headroom there is exhausted on this image.
- Remaining gap to the target: the families at 1.0x (embedding, transpose, linear, mlp, fp16 matmul) are the bulk of
  what the run-3 image cannot give more of in software; each needs hardware (matrix engine rate, line path), which
  is the case for the next image. The K-fused DSP lane study (`R42-kfuse`) and the `xl` tier board A/B are open.

**2026-11-23. K-fused lane numerics study (`R42-kfuse`, model, CPU).** Report: `evidence/r42/kfuse/REPORT.md`.
The registered gate (frozen tolerance for every point studied, rel-L2 vs binary64 at most 1.5x the sequential path, whole-layer
xl FP8 checks) passes for E4M3, E5M2, FP16 and BF16 at every registered guard width W in {0,3,8,24} and G in {2,4,8,16}.
The `sum_fma` structure (the existing `m.mma.dp` back end) is never worse than the sequential contract; a G=16 lane with a
29-bit adder tree and W=0 is enough. The numerics are therefore not what blocks a wider DSP-mapped lane; area, issue rate and
the transfer path are. Not covered: end-to-end token match (needs the model), the two prefill decoders. Next, in the order
registered: RTL of a DSP48E2-mapped lane and its resources (screen), then a speed prediction, then the board.

**2026-11-23. K-fused E4M3 lane on DSP48E2 (`R42-kfuse-rtl`, RTL sim + OOC screen).** Report: `evidence/r42/kfuse_rtl/REPORT.md`.
- Bit-exact to the numerics model (`sum_fma`, `trunc`, W=0): 10,175,920 vectors over 20 configurations, 0 mismatches; netlist
  simulation clean; 16 of 16 non-equivalent mutants caught (RTL sim, iverilog).
- Mapping: 3G-1 DSP48E2 per lane (two per product: one forms the 4x4 mantissa product, one aligns/truncates by a one-hot
  multiply; a DSP adder tree sums), because the window anchor needs the exact leading one of each product, so a PCIN
  cascade cannot align first. The unchanged R41 back end (1,437 LUT/lane) is 53 % of the G=16 lane.
- OOC place and route at 10 ns, one lane: G=4 1,734 LUT / 11 DSP, G=8 1,979 / 23, G=16 2,538 / 47 (434, 247, 159 LUT per
  MAC/cycle); R41 DP4 path in the same flow 1,729 LUT / 4 DSP (432 per MAC/cycle). G=16 closes at 4 ns (+0.224 ns);
  latency 12/13/14 cycles for G=4/8/16.
- Per core, derived: a 16-lane E4M3 array is 28.0k / 31.9k / 40.7k LUT (176 / 368 / 752 DSP) for G=4/8/16 against 44.7k LUT
  for the shared-lane array, at 64 / 128 / 256 MAC/cycle. It is E4M3 only: as a replacement it would drop the FP32, FP16/BF16
  and E5M2 modes; beside the shared lanes it fits no SLR. Not an image, not a board number.
- Consequence: the lane screens well for FP8 but the 113 points are FP16/FP32; a FP16 K-fused lane (64-bit exact field today)
  has not been built. The speed prediction (`R42-kfuse-speed`) decides whether an FP8-only wide lane is worth an image.

**2026-11-23. Speed prediction for the wide lane (`R42-kfuse-speed`, TLM calibrated on RTL and checked against the ab-int2 board records).**
Report: `evidence/r42/kfuse_speed/REPORT.md`. Model prediction, not a board result.
- A wider lane (E4M3 G=8/16, FP16 G=4/8 timing-only) and lower K-loop instruction overhead (scalar/branch free, or all issue-free)
  move the 113-point geometric mean by -0.5 to +0.7 %, the 37 matrix-engine GEMM points by -1.4 to +2.2 %. The matrix engine is busy
  28-37 % of the time on the single-GEMM board points; the tensor transfer path (A tile 75 + transposed B tile 99 cycles per K step) sets
  the step time. Model error on the GEMM points: median 8-10 %, six families outside 20 % (held constant).
- Decision: no image for lane width or issue-rate work. The K-fused DSP lane (screened in RTL, bit-exact, OOC resources recorded) is kept
  as a screened option; it is E4M3-only and pays only after the preparation overhead of the FP8 Hub kernel is gone.
- xl tier (model): restructuring the Hub kernel's instruction stream (hoisting weight/activation preparation out of the K loop) is worth
  +28 % on the 32 xl points and needs no hardware; the wider lane adds under 2 % on top.
- Next levers, in order of expected gain: software for the FP8 Hub kernel (xl), the tensor transfer path (software first: tile sizes, B
  layout; hardware candidate: plain-tile load slots 8 -> 32, +9.1 % on 30 GEMM points in the model), and the 1.0x families
  (embedding, transpose, cumsum) that the board shows unchanged.

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

**2026-11-24. xl tier on the board (`xl-base` vs `xl-cand2`).** Board, U280, run-3 image `791b8af2` on both sides (one load, 20:24 UTC;
the card was held by another user's image for 3 h 04 min before and the guarded queue waited its 15 idle minutes), 100 MHz,
serial-grid device cycles, default knobs, 32/32 points pass on both sides (`evidence/r42/board/xl-cand2/comparison.json`,
`tools/r42/board_compare_xl.py`). Baseline = `main` 4b61abd8 plus the xl point definitions only (compiler/python/rtl diff empty);
candidate = R42 head 9c795da3 (ab-int2 software) plus the xl definitions, i.e. WITHOUT the later `R42-fp8sw` changes.
- Geometric mean new/old 0.7688 (1.30x); summed device cycles 149.8e9 -> 97.4e9. FP8 matmul kernels (`fp8_linear` 13, `fp8_act_quant` 2):
  0.9992 (unchanged, as expected: round 2 does not touch the Hub kernel); FP8 composites (layers, decoders; 5 points) 0.6925 (the attention,
  norm and pointwise parts); fp16 (12 points) 0.5814. Slower: only `op.linear/xl/fp16-m2048-k2048-n6144-bias0` 1.037 (one sample per side).
- xl-base board figures: 1,498 s of device time, FP8 share 1,358 s; eta geometric mean by family fp8_linear 0.230, fp8_act_quant 0.242,
  fp8_decoder_prefill 0.117, fp8_decoder_decode 0.114, fp8_layer_decode 0.093, fp8_layer_prefill 0.156; all 18 FP8 points inside their
  frozen tolerance (worst error/tolerance 0.893, `op.fp8_act_quant`); heap peak 2.27 GB of the 8 GiB window.
- Next: the `R42-fp8sw` Hub-kernel changes (single-core RTL 1.68x per K block for M >= 16, TLM projection 1.42x on the 18 FP8 points) go
  into the next candidate together with `R42-xfer` and `R42-copy`, measured against these same baselines.

**2026-11-27. Third board round registered (`ab-int3`, `xl-cand3`), before any measured run.** Candidate = R42 `44070eb0`: the merge of
`R42-fp8sw` (Hub FP8 kernel lowering), `R42-copy` (embedding, transpose, cumsum lowering) and `R42-xfer` (N-blocked 16x32x16 GEMM
tile by default) on top of the ab-int2 software; tree `files:aaab5ffc...`, staged on the FPGA host as `~/r42/src_int3`. Same run-3 image
`791b8af2`, same baselines (`b41-run3-base` for the 113 points, `xl-base` for the 32 xl points), default knobs, same guarded recipe (15
idle minutes before any load; the card currently holds another user's image, idle). Gate evidence before the registration (single-core
RTL screen, thor): `make compiler` passes; tiny tier 27/27 pass, S_geo 1.449 against main; the slower-than-2 % points of the three
workers are known: `op.add/tiny/fp32-n256` (R42-red tiled default, +6.5 %), `op.transpose/small` +2.3 % (screen memory port has no
latency, the copy worker isolated the code in `_transposed_rows_load`), `model.mlp/small` +5.2 % (N-blocked tile; the screen cannot show
K-loop gains, rtl-3c shows linear/small -18 %). The board decides; any point slower than 2 % on the board is listed with its cause.
Hardware screen finished meanwhile (`R42-lineelig`, not in this candidate): half-line rows on the tensor engine, +597 LUT/core,
three-core stand-in 0.74-0.81x cycles on GEMM-shaped tiny/small points; model prediction +4 % (2-7 %) on the 113 geometric mean.

**2026-11-27. Yardstick change (author, chat: "你既然已经加入xl了就不要报113点了，应该是113+xl点数，R41镜像板测一次，最新板测一次，算几何平均，我期待的性能提升至少是4x-8x").**
From now on the reported result is the geometric mean over the 145 points = the 113 original points plus the 32 xl points, board, both
sides measured once each: baseline = R41 software on the run-3 image (`b41-run3-base`, reproduces the R41 records at 0.9995, and `xl-base`),
candidate = the latest round (`ab-intN` + `xl-candN`), same image, same harness. Tool: `tools/r42/board_compare_all.py` (combines the two
comparison files, per-tier and per-family, failures listed). The 113-only number stays in the per-round entries as a breakdown, not as the
headline. A clock-frequency gain stays a separate, routed-timing claim and is reported next to the cycle number, not mixed into it.
The author's expectation is 4x-8x. Registered here before the next result: it is an expectation, not a promise; what the evidence says
about it is stated with each result.
- State after round 2 (`ab-int2` + `xl-cand2`, board): **1.556x** over 145 points (113 original 1.638x, xl 1.301x; by tier tiny 1.291,
  small 1.711, medium 1.862, large 1.722, hbm 1.764, xl 1.301); 1 point slower than 2 % (`op.linear/xl/fp16-m2048-k2048-n6144-bias0`, 1.037).
  `evidence/r42/board/combined_round2.json`.
- What 4x-8x means in efficiency terms (board, strict single-core bound of ArchBench): the geometric mean of eta over the 134 points that have a
  recorded bound is 0.141 for the baseline and 0.225 now; 4x needs about 0.9 and 8x about 1.8, i.e. essentially every point at or above the
  single-core roofline of the spec at the same time (today 4 of 134 points are at or above 1, 24 at or above 0.5, 22 below 0.1). Share of the
  remaining sum of ln(1/eta): fp8_linear 9.6 %, attention_decode 9.2 %, attention_prefill 8.1 %, conv2d 6.5 %, resnet_block 6.3 %, softmax 6.1 %,
  rope 5.6 %, argmax 4.1 %, matmul 4.1 %, llama_decode 3.9 %.

**2026-11-27. Iterative hardware design toward the 4x expectation (author, chat: "当然，你要达到4x目标需要迭代设计。剩余3次是完整链接预算，但你可以进行无数次ooc综合，我看不到有什么不够的").**
Budget as stated by the author: 3 full links remain (5 in total for Stage 2, 2 used); out-of-context (OOC) synthesis is not limited. A card load of a new
image still waits for the author's approval of that image (SERVERS.md rule), asked for when a link has passed its gates. Plan, model first:
1. `R42-img` (running): LINE_HALF + MAC_F32 + MTX_PIPE as the first image candidate (link 1 at the earliest after its gates).
2. Design-space exploration on the calibrated TLM (`R42-dse`): which physical levers (core count, clock with memory latency scaled in cycles,
   LSU/SFU/VRF width, outstanding requests, line path) move the 145-point wall-time geometric mean, checked against all board records first.
3. Area and timing iterations by OOC (`R42-area`): DSP-mapped fp32 FMA and matrix lanes (bit-exact to the XLS reference), what bounds the clock in
   the routed run-3 reports, per-core LUT/DSP, and the number of cores each SLR then holds.
4. Images 2 and 3 are built from what steps 2 and 3 support; each is judged by the same board yardstick (113 + xl, cycles and wall time reported
   separately).

**2026-11-27. Board result `ab-int3` (113 original points, run-3 image `791b8af2`, R42 `44070eb0` against `b41-run3-base`).**
Queue 08:41-10:44 UTC (identity 3/3, `original` step 212 runs, `suite_status complete`, no failed check, no missing formal record), card back to
idle on the same image afterwards. `board_compare.py`: 113/113 points pass on both sides, **geometric mean new/old 0.4709 = 2.123x** (median ratio
0.564, ab-int2 was 1.638x); total device cycles 48.18 G -> 31.15 G. **No point is slower than 2 %** (the four candidates listed at registration,
`op.add/tiny/fp32-n256`, `op.transpose/small`, `model.mlp/small`, and the xl linear point, are all faster on the board: 1.41x, 4.7x, 1.17x for the first three;
the xl point is measured in `xl-cand3`). By tier (speedup): tiny 1.529, small 2.179, medium 2.413, large 2.259, hbm 2.854. Largest family gains:
embedding 12.6x, copy 5.7x, transpose 5.1x, bmm 4.8x, rmsnorm 3.1x, attention_prefill 3.0x, add 3.0x, cumsum 2.7x; smallest: mlp 1.15x, linear 1.21x,
resnet_block 1.34x, conv2d 1.34x, matmul 1.35x, gemv 1.36x. Sample spread inside a formal record: median 0.47 %, max 15.8 %. Evidence:
`evidence/r42/board/ab-int3/{records.tgz,logs.tgz,comparison.json}`. The 145-point headline needs `xl-cand3` (launched 10:46 UTC on the already loaded image, same guarded recipe).

**2026-11-27 11:38 UTC. `xl-cand3` stopped by a card change (kept as evidence, not retried silently).** The queue started 10:46 on the loaded run-3 image
(identity 3/3 pass, then the xl step): `model.fp8_decoder_decode/xl/fp8-b1-p2048-layers28` pass, `model.fp8_decoder_prefill/xl/fp8-b1-s2048-layers4` pass
(255.09 s of board wall time), then `model.fp8_decoder_prefill/xl/fp8-b1-s512-layers28` -> runtime-fault `DEPENDENCY_FAILED ... tnca_u280.xclbin is not the image
the device is running`: between 11:35:29 and 11:38:08 UTC another user loaded image `31132a91...` (300 MHz) over ours while the queue was between runs. The registered guards
(check before every run, image identity in the run) refused to drive that image, as designed; no number from that run exists. The card now holds
`31132a91...` and is idle. Records of the stopped attempt: `evidence/r42/board/xl-cand3/stop1_{records,logs}.tgz`. Next step per RECIPE.md: `board_ab.sh resume xl-cand3`
(15 idle minutes, the author-approved run-3 image `791b8af2` is reloaded with `program.sh --check/--go`, completed runs reused, the failed one repeated).

**2026-11-27. `R42-area` milestone M1 (worker report, branch `R42-area` 706538c0; OOC and synthesis numbers, domain: OOC xcu280 routed with registered I/O and ideal clock, not board).**
fp32 FMA rebuilt (`rtl/core/tnca_fmax_fma`), bit-identical to the generated FMA: 13 M-vector stage miters, cycle-level miters at LAT 2 and 3 (9 classes x 1 M),
MAC-lane miters, exact Fraction oracle (516,240 vectors), int<->float converter exhaustive (4.3e10 words), non-equivalent mutants all caught; default parameters keep the
generated RTL. DSP48E2 adders (3 sum adders inferred on DSP): 1,396 LUT / 5 DSP against 1,351 LUT / 2 DSP for the existing XLS LAT-3 FMA, i.e. they save 50-90 LUT
per FMA against the old LAT-2 FMA but are no smaller than the existing LAT-3 instance; the gain is the restructure (-6 % routed, -9 % in kernel-synthesis accounting).
Tile 0 at the run-3 image parameters (synthesis): 253,898 -> 248,890 LUT (-2.0 %), converter lane 504 -> 418 LUT (-1.4k LUT per core). Arithmetic mapping is worth
about 2-2.5 % of a core and does **not** change the cores per SLR (two per SLR needs a core of about 138k LUT, derived). Run-3 routed census: kernel-clock endpoints with slack
< 1 ns: 18,548, < 2 ns: 89,603. Consequence for the plan: DSP mapping of the arithmetic is not the lever for 4x (neither area nor clock); the worker continues with tile/LSU
OOC at 6.67 and 5 ns, the core-run2 regression with the area back end, and the report.

**2026-11-27. `R42-area` merged into `R42` (RTL default-off: `FMA_IMPL`/`BE` default 0 keeps the generated RTL; 90/90 core-run2 PASS with identical cycles at BE 13 + converter). Final report: `evidence/r42/area/REPORT.md`.**
What it says about the 4x expectation (OOC, synthesis and derived figures, no board number): arithmetic restructure and DSP mapping are worth 2-2.5 % of a core (tile 0 253,898 -> about 248-249k LUT);
**cores per SLR stay at 1** (2 per SLR need a core of about 138-141k LUT, i.e. -44 % or more, a different core); the routed run-3 image has 18,548 kernel-clock endpoints below +1 ns and 89,603 below +2 ns in
every unit, so the clock plateau of this netlist is 100-110 MHz and about 120 MHz after fixing the nine path families listed in the report (each costs about one cycle). Together: area/clock iteration of the present
core gives at most about +20 % wall time through clock, which a new image (a full link) would have to pay for; the 4x-8x expectation has to come from cycles or from a different machine organisation (see `R42-dse`).
Not finished by the worker (x86 slots contested): hybrid-LSU OOC at 10/6.67/5 ns, tile-0 screen at 6.67 ns, cycle cost of the latency-adding fixes. A clarification about the worker's reported self-killed shells
(rule: no pattern kills) was requested before closing the item.

**2026-11-27. Rule slip by the `R42-area` worker, recorded.** Asked about its self-killed shells, the worker reported (from its own record) five pattern kills on the shared x86 user `vivado`
(`for p in $(pgrep -f "run_queue.sh"); do kill $p; done` and the same shape with `ps | grep "[r]un_queue.sh queue_..."`, `queue_dcp`, `queue_t2`, `run_ooc.sh lsu_p10`), against the standing rule
(no pattern kills, signal only recorded PIDs). Patterns were its own script names; every `ps` listing it printed showed only its own processes carrying them (the R42-img jobs run as `img_chain.sh` and
`synth_r40_multicore.tcl`); the output of four of the kills was lost with the shell, so it cannot prove nothing foreign matched. No other worker's or user's job is known to have died: the early R42-img synthesis pair
ended with exit 0 (1,212 s and 1,252 s) and the later ones were running when looked at; a check of the R42-img worker's logs for unexplained signals in that window is requested. Also five explicit-PID kills of its own jobs
and one on GB10 (iverilog mutation script and `vvp`, listed by path). Remedy: every job is started with a recorded PID, signals go only to recorded PIDs after `ps -o pid,user,args -p PID`, `ps`/`pgrep` output never feeds a kill loop.

**2026-11-27 13:20 UTC. `xl-cand3` resumed run ended `failed-preserved` with 31 of 32 xl points measured; one-point repeat registered (`xl-cand3r`) before it runs.**
After the guarded reload (idle rule met 12:14:56 UTC on another user's image `f9295d5d...`, our run-3 image loaded 12:15 UTC, 100 MHz) the resumed queue measured 45 further runs in 3,309 s on the same
image and software tree, no further card change; `experiment.py` reuses completed runs and keeps a recorded failure as final, so the step reports `configurations 32, failed [model.fp8_decoder_prefill/xl/fp8-b1-s512-layers28],
missing_formal [same]`. That point's only attempt is the 11:38 UTC card-change refusal (no workload ran, `to("tnca")` was refused by the image-identity guard): a missing measurement, not a result.
Records of this stop: `evidence/r42/board/xl-cand3/stop2_{records,logs}.tgz` (and `stop1_*`). To complete the suite without rewriting anything, `tools/r42/board_ab.py register` got `--only POINT --of NAME`
(xl suite, repeatable): a registration holding just the named point(s), same software commit `44070eb0` and tree, image, board block, knobs, sampling and identity manifest; `coverage` is omitted for the subset because
`board_batch.py` checks full-suite coverage only when it is given (suite file sha256 and the point definition sha256 stay registered and checked); the change is listed in the registration's
`changes_against_run3`. Registered as `xl-cand3r` (one point, identity first) and committed before launch. The xl comparison will combine `xl-cand3` (31 points) and `xl-cand3r` (1 point).

**2026-11-27 13:48 UTC. `xl-cand3r` stopped before any run: the card changed hands again during the guarded 15-minute idle wait.** Queue start 13:28:51 on our image (already loaded), then the queue's own idle wait
(`wait_idle`, 15 consecutive idle minutes before the identity manifest); at about 13:48 another user's image `f9295d5d...` (300 MHz) replaced ours and the queue stopped with
`loaded image f9295d5d... is not the declared image ea1fa1af...`. No identity point and no xl point ran, so no number and no failed record exists for the point. Records: `evidence/r42/board/xl-cand3r/stop1_{records,logs}.tgz`.
Observation: both stops of this series (11:38, 13:48) came while our image sat loaded and idle (between runs, or in the queue's idle wait); the other user's loads follow the same idle rule, so our own mandatory idle
waits (about 30 minutes per queue: before the identity manifest and again before the suite step) are exactly the windows in which the card can be taken. Resumed 13:50 UTC (`board_ab.sh resume xl-cand3r`: fresh 15 idle minutes,
guarded reload of the author-approved run-3 image, then identity and the one xl point).

**2026-11-27. Author's question (chat): why is the gain so small, "不让你开发使用DSP URAM的方法了吗" (was developing DSP/URAM methods not allowed?).** Answer recorded with the evidence:
- Allowed, and only partly done. DSP: the E4M3 K-fused DSP lane (`R42-kfuse-rtl`, merged) and DSP48E2 adders in the fp32 FMA (`R42-area`: no gain over the XLS latency-3 FMA). URAM/BRAM for arrays: **not tried**; the area worker skipped it by judgement (a synchronous read adds a cycle on the slice hit path), without a measurement, although OOC synthesis is unlimited. That was an omission.
- Why DSP alone cannot help: calibrated-TLM screening on the 77 tiny/small/medium points (model domain, `origin/R42-dse:evidence/r42/dse/screen.json`, effect on the 145 points): K-fusion G=8 (wider DSP MAC lane) 1.005x, SFU x2 1.014x, VRF without conflicts 1.002x, LSU x4 1.013x; i.e. arithmetic width is not what the cycles wait for. What moves it: cores 4 / 6: 1.17x / 1.40x, clock 200 MHz with ns-fixed memory latency 1.88x (1.60x with added latency; 250 MHz 2.27x), single-cycle vector issue 1.13x, ideal memory 1.23x, LINE_HALF 1.03x, sets128 1.046x.
- So DSP/URAM/BRAM are useful only as a means to AREA (a tile of about 127-141k LUT instead of 250k lets 2 tiles share an SLR: 4-6 cores) and to TIMING (pipelined, memory-based structures); they do not make the 16-lane arithmetic faster. Today's tile: matrix engine 61.6k, vector FPU 49.9k, NUCA slice 39.5k (9.9k LUTRAM), integer ALU 23.2k, banked VRF 21.3k (+13.8k crossbar), transfer 19.5k, LSU 19.3k.
- Action: worker `shrink` (branch `R42-shrink`, worktree `/home/ubuntu/r42-wt/shrink`): memory-like vs arithmetic census per unit, then OOC-measured URAM/BRAM and DSP candidates (slice array, accumulator file, VRF/crossbar, MSHR/queues, MAC lanes for fp16/fp32, ALU/FPU logic), cumulative tile size by kernel synthesis, cycle cost into the TLM, floorplan arithmetic for 2 tiles per SLR, and the nine clock families; reports predicted 145-point effect per step (model domain).
