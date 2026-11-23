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
