# R41 decode profile: where decode time goes for the FP8 LLMs on TNCA

> **Binding:** [`doc/engineering_discipline.md`](../../../doc/engineering_discipline.md). Every number below is
> labelled by domain: **board** (U280, run-1 image), **TLM** (SystemC/TLM model, uncalibrated, `core-vrf-sync-v5-x8`,
> 3 cores, 100 MHz, default TNCA settings), **static** (read from the compiled assembly or the compiler source) or
> **analytic** (arithmetic on recorded numbers). No card access was used.

Status: **Part 1**: how the FP8 weights reach the matrix engine (static, plus TLM counters). **Part 2**: per-kernel and
per-class breakdown of one steady decode step for both models, the bound analysis and the priorities.

## Part 1. How the Hub FP8 block-matmul gets weights and scales into the matrix engine

### What was read

| Item | File |
|---|---|
| Compiled kernel, Qwen3-VL run (TLM dump, `TNCA_KERNEL_DUMP=1`, profile `core-vrf-sync-v5-x8`, sha256 of the `.s` = `d8beb47438ce...5615df`) | `decode_profile/w8a8_block_dynamic_fp8_matmul_kernel.vl.s.gz` (11,072 lines) |
| Same kernel from the Qwen3.5 TLM run (`tlm-q35-default-v1`) | `decode_profile/w8a8_block_dynamic_fp8_matmul_kernel.q35.s.gz`; **byte-identical** to the VL one (`cmp`) |
| Instruction counts, computed by a script from those files | `decode_profile/static_count.py`, outputs `static_count_vl.txt`, `static_count_q35.txt` |
| Compiler lowering that emits the weight path | `compiler/python/tnca_cc/frontend.py` (the `x8dot_*` block, around lines 3040-3175: "tl.dot ... FP8, B as W's rows written transposed by the engine (ISA sec 9E, tensor_transpose_8bit)") |
| Hub kernel source | `kernels-community/finegrained-fp8`, snapshot `0e5764aa...`, `build/torch-cuda/matmul.py`, `w8a8_block_dynamic_fp8_matmul_kernel` |
| TTE model and ISA | `tools/performance_harness/systemc/include/tnca_tensor_engine.hpp`, `doc/isa/TNCA-ISA-v0.1.md` sec 9E, `rtl/core/tnca_tensor.v` (`NTAG = 16`) |
| TLM raw logs for the counters quoted | `decode_profile/vl-l1/` (`launches.json.gz`, `report.json`, `run.log.gz`, `identity.json`, `model.cfg`), extractor `gemv_counters.py`, output `gemv_counters_vl_step3.txt` |

One compiled kernel serves every FP8 linear of both models in decode: Qwen3-VL and Qwen3.5 launch the same
`w8a8_block_dynamic_fp8_matmul_kernel` (the Hub block-scaled matmul; there is no separate decode GEMV kernel on this
route), with N and K as run-time arguments and tile `BLOCK_M=16, BLOCK_N=128, BLOCK_K=128`. A decode token has M=1, so
15 of 16 rows of every tile are padding.

### The answer (static)

**FP8 weights travel by the X8 transposed tensor transfer, `t.ld.async.b.t` (TTE descriptor D0), straight into the
window registers (v56-v63) that `m.mma.async.k32.e4m3` reads. No vector load touches a weight byte, and the vector
unit never dequantises weights.** Evidence, in the K loop (`for_25`, asm lines 725-9466) of the kernel:

- 32 `t.ld.async.b.t` per K iteration (one per 16-column chunk x 4 K sub-steps of 32), each followed by one
  `m.mma.async.k32.e4m3` that consumes the window; `vf.cvt.*e4m3` appears 128 times, all on the *activation*
  quantisation path (f32 -> e4m3 of A), none on B.
- 32 `v.ld.gather.b` weight-side loads exist, but only inside the 32 **fallback arms** (21-22 instructions each) that
  the compiler emits for columns that wrap (`offs_bn % N`). A run-time check (`br.nz ok, x8dot_done_N`) skips them,
  and a paired predicated `t.ld.async.b` is issued but predicated off (it still completes, ISA sec 9A.3). In decode
  the check holds, so the gather arms do not execute.
- **Block scales** are not applied to weights: the matrix engine multiplies raw e4m3 values and the scales are applied
  in FP32 after each K block (`acc += dot * a_s * b_s`), via `a.vread` of the accumulators (256 per iteration), vector
  FP multiplies and a scratch spill/reload (`v.st.w` 384, `v.ld.w` 544 per iteration). The weight scale `Bs` itself is
  read by 8 `v.ld.gather.w` per K iteration (one 4 B fp32 per 128x128 block, all 128 lanes the same word); 1 KiB of
  scales per 4 MiB of weights, negligible.

### Bytes per request, requests in flight, instructions per weight line

| Quantity | Value | Basis |
|---|---|---|
| Weight tile per K iteration | 128 x 128 FP8 = 16,384 B = 256 x 64 B lines | static |
| Transfers per K iteration | 32 `t.ld.async.b.t`, each a 16 rows x 32 B box = 512 B (8 lines' worth of bytes) | static, `static_count_vl.txt` |
| Shape of one transfer | one 8-byte unit per source row per 8-column block: 16 rows x 4 blocks = **64 requests of 8 B**, each to a *different* row (stride K bytes); each row's 32 B is half of a 64 B line, the other half comes 8 transfers later (the next K sub-step) | static, ISA sec 9E ("reads one 8-byte unit down every source row"), `tnca_tensor_engine.hpp` |
| **Bytes per request** | **8 B** (one core word request into a NUCA home; `dma_read_words` x 8 B = 1.031 x the weight bytes in every GEMV launch below). No 64 B request exists in the default configuration. | static + TLM |
| **Requests in flight per core** | **16** (`dma_outstanding = 16`; RTL `NTAG = 16`), 8 DMA unit slots (`dma_unit_slots = 8`); the core port accepts one request per cycle (8 B/cycle/core, 24 B/cycle for 3 cores = 2.4 GB/s at 100 MHz, `doc/r40_tasks_and_acceptance.md`) | `model.cfg`, RTL parameter |
| Serialisation | per 512 B transfer the code does `mb.arm; t.ld.async; mb.wait; mb.arm; m.mma.async; mb.wait`: **no double buffering, one transfer outstanding at a time** (`mb.arm` 64, `mb.wait` 64 in the loop) | static |
| Instructions that deliver one 512 B weight transfer | 45 lines between `x8dot_done_27` and `x8dot_done_29`, of which 22 are the skipped fallback arm: **23 executed** (coordinate moves, 2 `t.ld.async`, `mb.arm/wait`, `m.mma`), i.e. **2.9 per 64 B of weights** | static |
| Whole K iteration, per 64 B of weights | 7,979 fast-path instructions / 256 lines = **31.2 per 64 B** (includes A staging and quantisation, accumulator epilogue, scale application) | static, `static_count_vl.txt` |

### Measured on the TLM model: the GEMV is not near any bandwidth limit

TLM, Qwen3-VL 1 layer, steady decode call 3 (`decode_profile/gemv_counters_vl_step3.txt`), three FP8 launches
of different size:

| Launch | Weights | Cycles | Weight B/cycle (chip) | NUCA bytes / cycle | DRAM reads / weights | Cycles per 512 B step (critical core) | Instr/cycle (core 1) |
|---|---:|---:|---:|---:|---:|---:|---:|
| #101 N=K=2048 | 4,195,328 | 9,396,737 | 0.446 | 2.251 | 5.61x | 3,059 | 0.087 |
| #102 N=1024, K=2048 | 2,097,664 | 4,900,293 | 0.428 | 2.159 | 5.00x | 3,190 | 0.084 |
| #115 N=6144, K=2048 | 12,585,984 | 27,878,350 | 0.451 | 2.276 | 5.79x | 3,203 | 0.083 |

- Weight delivery is **0.43-0.45 B/cycle for the whole 3-core chip** (about 44 MB/s at 100 MHz), against the 24 B/cycle
  core-port interface (1.9%). The board's 40.7 s per token for 2.65 GB of language weights is 65 MB/s, the same order
  (board: 4,065,511,739 and 4,065,552,515 device cycles in the two steady decode calls, `board-51aef19.tgz`).
- The cores issue 0.08-0.09 instructions per cycle: the kernel is **stall-bound, not issue-bound**. Per core, barrier
  sleep (waiting for the transfer) is 28% of the launch, matrix-engine busy 15-18%, dependency wait 6-8%
  (`core_*_barrier_sleep_cycles`, `matrix_engine_busy_cycles`, `dependency_wait_cycles`); the remainder is not attributed
  by these counters.
- The weights themselves are read once by the TTE (1.03x), but **vector-LSU scratch traffic is about 4x the weight
  bytes** (NUCA bytes minus TTE bytes = 16.8 MB for 4.2 MB of weights in #101: A staging, accumulator spill and reload),
  and the NUCA (3 homes x 32 sets x 8 ways x 64 B = 48 KiB, `model.cfg`) sends **5.0-5.8x the weight bytes as DRAM
  line reads** plus 1.6x as writebacks. That the scratch traffic is what evicts the weight lines is a hypothesis; this
  profile shows the correlation, not the cause.

### Which path would serve weights at line rate with prepacked 64 B lines?

- **Nothing in the present machine moves a 64 B line per request.** Every path (TTE, vector LSU) enters the NUCA as
  8 B core-word requests through an 8 B/cycle core port; the wide-request path (`tl_core_beat_bytes`) is an
  exploratory, non-default model option and is not a result for the target (discipline sec 2.2).
- Of the existing paths, the **non-transposed `t.ld.async`** is the one that would stream prepacked weights best: it
  expands a box into sequential 8 B units (8 per line, `tnca_tensor_engine.hpp` last branch of `fast_transfer`), needs
  no per-row gather, no corner turn and no vector registers, and goes straight into the window the MMA reads. The
  X8 transposed load issues the *same number* of 8 B requests (64 per 512 B) but each to a different line.
  A software prepack (W re-tiled offline into the MMA's k32 x n16 order, so one 512 B contiguous box = 8 whole lines)
  would cut the delivery instructions per 512 B transfer from 23 to roughly 8-10 (descriptor coordinates, `t.ld`,
  `mb.arm/wait`, `mma`) and make each line's 8 requests consecutive (analytic estimate from the instruction listing; not
  compiled or measured).
- Its ceiling, by Little's law with 16 requests in flight, is `min(8, 16 x 8 / L)` B/cycle/core for per-request latency
  L cycles: 8 B/cycle only if L <= 16 cycles, i.e. NUCA hits with a prefetch or double-buffered issue; at a miss
  latency near 100 cycles it is about 1.3 B/cycle/core (analytic; the per-request latency was not measured). Today's
  code waits on each transfer before issuing the next, so it sees the full latency of a 64-request transfer
  (~3,100 cycles per step measured above including all other work).
- Caveat on the repack itself: the Hub kernel takes W as `(N, K)`. Re-tiling the checkpoint weights changes the layout
  the standard path hands to the kernel, so it is a compiler/loader layout pass or it is a side path
  (discipline sec 2.2.2-2.2.4); it needs the author's decision before it is built and is not a result here.

### Reading for hardware and compiler priorities (from part 1 only)

1. The weight path is not limited by the 8 B/cycle interface: it uses under 2% of it. Latency exposure (one transfer
   at a time, no prefetch/double buffering) and the surrounding vector scratch traffic dominate. A compiler change
   that issues the next transfer before waiting for the current one (software pipelining across the 32 steps) is
   the first lever; it needs no new hardware.
2. M=1 decode pays the full 16-row tile (activation quantisation, accumulator spills) for one useful row; a GEMV-shaped
   lowering with M=1 would remove most of the 16 x scratch traffic.
3. Hardware: 8 B per request is the interface for every path. Wider requests would help only after (1) and (2).

## Part 2. Per-kernel and per-class breakdown of one steady decode step

### Method, domains and limits

- **Domain: TLM** (SystemC/TLM model, binary sha256 `276a8f12...`, `core-vrf-sync-v5-x8`, 3 cores, 100 MHz, config in
  each run's `model.cfg`, `TNCA_DRAM_PATH=fast`, default TNCA settings; launcher `decode_profile/tlmrun-profile.sh`, a
  copy of `p3/tlmrun-default.sh` with new run directories). Qwen3-VL-2B-FP8 with **1** decoder layer; Qwen3.5-4B-FP8
  with **4** layers (3 linear-attention + 1 full-attention, one period of its layer pattern). Prompt "Say hi.", bucket 64,
  4 tokens (1 prefill call + 3 decode calls), no teacher check (`--check-tokens` 0), so the generated tokens are not a
  quality claim. The tables use the **last decode call** (call 3), in which nothing is compiled.
- **The step is observed, not changed.** `tools/models/decode_profile.py` runs `chat.py` unchanged and wraps the
  launcher. For every launch it records the kernel, grid, device cycles, each tensor argument classified as weight
  (named model parameter), KV/state (named cache tensor) or activation, the code-store load/eviction, and the model's
  counters before and after the launch (the model's read-only `C` command, read while the device is idle). The device
  cycles of all four calls of the VL run equal those of the uninstrumented archived run `tlm-vl-default-v2`
  (582,976,965 / 540,405,261 / 539,710,624 / 540,586,727), so the instrumentation does not move the device.
  A first attempt read the counters on the wrong thread and attributed them to the wrong launch; its counters were
  discarded (its cycles are the same), its files are kept on GB10 (`~/r41-fp8prof/runs/*.v1-*`), and the first Qwen3.5
  run was stopped and restarted with the fix.
- **Scaling to full depth is analytic**: full = fixed part (embedding, final norm, lm_head, cache bookkeeping) +
  (full layers / profiled layers) x per-layer part (x28 for VL; x8 for Qwen3.5, i.e. 24 linear-attention + 8
  full-attention layers, exactly its layer pattern). No full-depth TLM run was made. A launch belongs to the per-layer
  part if it touches a decoder-layer parameter or cache/state, or follows one before the next fixed launch.
- **Bytes.** *NUCA bytes* = 8 B x (hits + misses + merges) at the NUCA homes: every core word request, from the vector
  LSU and from the TTE, as the model counted it. *TTE bytes* = 8 B x `dma_read_words`. *DRAM bytes* = 64 B x (outlet
  reads + writes). The per-kernel files also give the argument footprint (weights / activations / KV bytes of the
  tensor arguments; the 622 MB embedding table appears as the footprint of the embedding-lookup kernels and is not
  traffic).
- **Class** is assigned from the kernel name and its arguments by `classify()` in `decode_profile_report.py`. Fused
  Inductor kernels carry several operations and take the first matching class. Qwen3.5's delta-rule, conv1d, and
  qkv-split kernels are in "linear-attention / conv ops"; its attention class holds only the full-attention layers' KV
  update and SDPA.
- **Per-kernel tables** (every launch of the step: grid, cycles, footprints, NUCA/TTE/DRAM bytes, B/cycle, code
  load/eviction): `decode_profile/per_kernel_vl.md` (23 launches), `decode_profile/per_kernel_q35.md` (93 launches).
  Raw: `decode_profile/vl-l1/` and `decode_profile/q35-l4/` (`launches.json.gz`, `report.json`, `run.log.gz`,
  `identity.json`, `model.cfg`); derived: `analysis_vl.json.gz`, `analysis_q35.json.gz`. Reproduce with
  `python3 tools/models/decode_profile_report.py RUN --full-layers N [--board-report vl-full-chat/report.json]`.
- **Cross-check against the board.** The archived board run (`board-51aef19.tgz`) records per call only totals
  (launches, device cycles, wall time, host reads/writes). It has **no per-launch timing and no code-store counters**, so
  the class shares cannot be checked per class. The totals: board (Qwen3-VL, full depth, non-default `TNCA_IEEE_DIV=1`,
  steady calls 2 and 3) 4,065,511,739 and 4,065,552,515 device cycles and 563 launches; TLM scaled 3,771,806,705 cycles
  (0.928 of the board) and 536 launches. The 27 extra board launches are not explained by this data. Qwen3.5 has no
  board run.

### Qwen3-VL-2B-FP8 (TLM, 1 layer profiled, scaled to 28)

##### Per class, scaled to 28 layers (TLM per-layer part x28 + fixed part; the scaling is analytic)

| Class | Launches/token | Cycles/token | Share of cycles | NUCA bytes/token | Share of NUCA bytes | NUCA B/cycle | TTE bytes/token | DRAM bytes/token | Code loads / evictions per token |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| FP8 matmul/GEMV (weights; act-quant + block dequant fused) | 196 | 3,141,946,696 | 83.3% | 7,107,767,296 | 74.2% | 2.262 | 1,453,326,336 | 9,703,434,496 | 56 / 252 |
| BF16 matmul/GEMV (unquantized weights, incl. lm_head) | 1 | 420,623,160 | 11.2% | 2,298,411,840 | 24.0% | 5.464 | 1,558,255,616 | 1,480,819,328 | 1 / 0 |
| attention (QK, softmax, AV, KV write/read) | 140 | 193,344,004 | 5.1% | 164,982,272 | 1.7% | 0.853 | 29,474,816 | 10,316,544 | 140 / 56 |
| rmsnorm and other norms (+ residual add / embedding lookup) | 113 | 15,084,541 | 0.4% | 1,824,760 | 0.0% | 0.121 | 179,712 | 1,520,256 | 113 / 57 |
| rmsnorm fused with rotary | 56 | 188,720 | 0.0% | 645,120 | 0.0% | 3.418 | 0 | 177,408 | 56 / 0 |
| activation/gating (silu*mul etc.) | 28 | 605,220 | 0.0% | 1,290,240 | 0.0% | 2.132 | 0 | 1,243,648 | 28 / 0 |
| copies, casts, clones | 1 | 173 | 0.0% | 40 | 0.0% | 0.231 | 0 | 128 | 1 / 0 |
| other | 1 | 14,191 | 0.0% | 3,552 | 0.0% | 0.250 | 0 | 2,176 | 1 / 4 |
| **total** | 536 | 3,771,806,705 | 100% | 9,574,925,120 | 100% | 2.539 | 3,041,236,480 | 11,197,513,984 | 396 / 369 |

Predicted full-depth device time per token (analytic scaling of TLM): 37.718 s at 100 MHz (3,771,806,705 cycles, 536 launches).

Board steady decode step 2 (full depth; domain: board): 4,065,511,739 device cycles (40.655 s), 563 launches, 63.095 s wall: 22.440 s outside device time = 39.86 ms per launch. TLM full-depth prediction / board device cycles = 0.928.

Board steady decode step 3 (full depth; domain: board): 4,065,552,515 device cycles (40.656 s), 563 launches, 64.126 s wall: 23.470 s outside device time = 41.69 ms per launch. TLM full-depth prediction / board device cycles = 0.928.

Smallest launch in the step: 173 cycles (TLM), an upper bound on the fixed device cost of a launch; x 536 launches = 92,728 cycles/token.

##### Bound analysis (analytic, from the TLM-measured NUCA bytes per class)

Seconds per token if a class's NUCA bytes moved at chip-level rate R and nothing else cost time. The last rows add the fixed per-launch costs: device (smallest launch above) and host (board ms per launch, worst steady step).

| Class | 1 GB/s | 2 GB/s | 4 GB/s | 8 GB/s | 19 GB/s |
|---|---:|---:|---:|---:|---:|
| FP8 matmul/GEMV (weights; act-quant + block dequant fused) | 7.1078 | 3.5539 | 1.7769 | 0.8885 | 0.3741 |
| BF16 matmul/GEMV (unquantized weights, incl. lm_head) | 2.2984 | 1.1492 | 0.5746 | 0.2873 | 0.1210 |
| attention (QK, softmax, AV, KV write/read) | 0.1650 | 0.0825 | 0.0412 | 0.0206 | 0.0087 |
| rmsnorm and other norms (+ residual add / embedding lookup) | 0.0018 | 0.0009 | 0.0005 | 0.0002 | 0.0001 |
| rmsnorm fused with rotary | 0.0006 | 0.0003 | 0.0002 | 0.0001 | 0.0000 |
| activation/gating (silu*mul etc.) | 0.0013 | 0.0006 | 0.0003 | 0.0002 | 0.0001 |
| copies, casts, clones | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| other | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| **all classes, s/token** | 9.5749 | 4.7875 | 2.3937 | 1.1969 | 0.5039 |
| **tokens/s, bytes only** | 0.104 | 0.209 | 0.418 | 0.836 | 1.984 |
| **tokens/s, + device launch cost (0.0009 s)** | 0.104 | 0.209 | 0.418 | 0.835 | 1.981 |
| *lower bound: each matmul weight read once (2,031,960,064 B; embedding table excluded), tokens/s* | 0.492 | 0.984 | 1.969 | 3.937 | 9.351 |
| **tokens/s, + device and board host launch cost (22.34 s)** | 0.0313 | 0.0369 | 0.0404 | 0.0425 | 0.0438 |


### Qwen3.5-4B-FP8 (TLM, 4 layers profiled, scaled to 32)

##### Per class, scaled to 32 layers (TLM per-layer part x8 + fixed part; the scaling is analytic)

| Class | Launches/token | Cycles/token | Share of cycles | NUCA bytes/token | Share of NUCA bytes | NUCA B/cycle | TTE bytes/token | DRAM bytes/token | Code loads / evictions per token |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| FP8 matmul/GEMV (weights; act-quant + block dequant fused) | 200 | 7,593,234,760 | 86.9% | 17,980,138,496 | 76.8% | 2.368 | 3,676,569,600 | 19,924,292,096 | 64 / 424 |
| BF16 matmul/GEMV (unquantized weights, incl. lm_head) | 25 | 881,596,302 | 10.1% | 4,708,636,544 | 20.1% | 5.341 | 3,192,311,808 | 1,975,793,984 | 25 / 8 |
| attention (QK, softmax, AV, KV write/read) | 40 | 19,414,976 | 0.2% | 49,792,000 | 0.2% | 2.565 | 10,518,528 | 4,431,872 | 40 / 32 |
| rmsnorm and other norms (+ residual add / embedding lookup) | 129 | 13,987,570 | 0.2% | 4,819,952 | 0.0% | 0.345 | 281,088 | 3,524,800 | 129 / 48 |
| rmsnorm fused with rotary | 48 | 1,484,960 | 0.0% | 593,920 | 0.0% | 0.400 | 0 | 122,368 | 48 / 0 |
| activation/gating (silu*mul etc.) | 32 | 1,226,608 | 0.0% | 2,211,840 | 0.0% | 1.803 | 0 | 3,195,392 | 32 / 0 |
| linear-attention / conv ops (Qwen3.5) | 216 | 224,026,256 | 2.6% | 664,939,392 | 2.8% | 2.968 | 60,567,552 | 581,809,664 | 200 / 80 |
| copies, casts, clones | 25 | 769,602 | 0.0% | 2,162,728 | 0.0% | 2.810 | 0 | 3,995,264 | 25 / 0 |
| other | 1 | 13,628 | 0.0% | 1,800 | 0.0% | 0.132 | 0 | 1,408 | 1 / 0 |
| **total** | 716 | 8,735,754,662 | 100% | 23,413,296,672 | 100% | 2.680 | 6,940,248,576 | 22,497,166,848 | 564 / 592 |

Predicted full-depth device time per token (analytic scaling of TLM): 87.358 s at 100 MHz (8,735,754,662 cycles, 716 launches).

Smallest launch in the step: 194 cycles (TLM), an upper bound on the fixed device cost of a launch; x 716 launches = 138,904 cycles/token.

##### Bound analysis (analytic, from the TLM-measured NUCA bytes per class)

Seconds per token if a class's NUCA bytes moved at chip-level rate R and nothing else cost time. The last rows add the fixed per-launch costs: device (smallest launch above) and host (board ms per launch, worst steady step).

| Class | 1 GB/s | 2 GB/s | 4 GB/s | 8 GB/s | 19 GB/s |
|---|---:|---:|---:|---:|---:|
| FP8 matmul/GEMV (weights; act-quant + block dequant fused) | 17.9801 | 8.9901 | 4.4950 | 2.2475 | 0.9463 |
| BF16 matmul/GEMV (unquantized weights, incl. lm_head) | 4.7086 | 2.3543 | 1.1772 | 0.5886 | 0.2478 |
| attention (QK, softmax, AV, KV write/read) | 0.0498 | 0.0249 | 0.0124 | 0.0062 | 0.0026 |
| rmsnorm and other norms (+ residual add / embedding lookup) | 0.0048 | 0.0024 | 0.0012 | 0.0006 | 0.0003 |
| rmsnorm fused with rotary | 0.0006 | 0.0003 | 0.0001 | 0.0001 | 0.0000 |
| activation/gating (silu*mul etc.) | 0.0022 | 0.0011 | 0.0006 | 0.0003 | 0.0001 |
| linear-attention / conv ops (Qwen3.5) | 0.6649 | 0.3325 | 0.1662 | 0.0831 | 0.0350 |
| copies, casts, clones | 0.0022 | 0.0011 | 0.0005 | 0.0003 | 0.0001 |
| other | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| **all classes, s/token** | 23.4133 | 11.7066 | 5.8533 | 2.9267 | 1.2323 |
| **tokens/s, bytes only** | 0.043 | 0.085 | 0.171 | 0.342 | 0.812 |
| **tokens/s, + device launch cost (0.0014 s)** | 0.043 | 0.085 | 0.171 | 0.342 | 0.811 |
| *lower bound: each matmul weight read once (4,841,359,360 B; embedding table excluded), tokens/s* | 0.207 | 0.413 | 0.826 | 1.652 | 3.925 |


### What the tables show

| | Qwen3-VL-2B-FP8 | Qwen3.5-4B-FP8 |
|---|---|---|
| FP8 matmul/GEMV: share of device cycles / of NUCA bytes (TLM, scaled) | 83.3% / 74.2% | 86.9% / 76.8% |
| BF16 matmul (lm_head; Qwen3.5 also in_proj_a/b): cycles / bytes | 11.2% / 24.0% | 10.1% / 20.1% |
| attention: cycles / bytes | 5.1% / 1.7% | 0.2% / 0.2% |
| linear-attention / conv ops: cycles / bytes | n/a | 2.6% / 2.8% |
| norms, rotary, activation, copies, other: cycles | 0.4% | 0.2% |
| FP8 class weight bytes per cycle (chip, 3 cores) | 0.449 (1.41 GB / 3.14 G cycles) | 0.470 (3.57 GB / 7.59 G cycles) |
| TLM device time per token at 100 MHz, scaled | 37.7 s | 87.4 s |
| Launches per token (TLM scaled) | 536 (board: 563) | 716 |
| Code-store loads / evictions per token (TLM scaled; board not recorded) | 396 / 369 | 564 / 592 |

- **The FP8 matmul dominates both models (83-87% of device time) and runs at 0.45-0.47 weight bytes per cycle for the
  whole chip**, about 2% of the 24 B/cycle core-port interface. It takes the X8 transposed TTE path of part 1; its time
  is stalls (0.08-0.09 instructions per cycle) and its vector-LSU scratch traffic is about 4x the weight bytes.
- **The BF16 lm_head is the second class (10-11%)** at 1.48 weight B/cycle for VL, 3x the FP8 path's rate, through
  the Inductor matmul template over the tied bf16 embedding (622 MB for VL, 1.27 GB for Qwen3.5). Its NUCA traffic is 3.7x
  its weights.
- Attention, norms, rotary, activation and copies together are about 5.5% (VL) and 0.4% (Qwen3.5) of device time. This
  profile uses a 64-token bucket, so KV traffic (131,072 B per VL layer) is small; it understates attention for a long
  conversation.
- **Fixed per-launch cost.** Device side (TLM): the smallest launch is 173 cycles (VL) and 194 cycles (Qwen3.5), so the
  fixed device cost of all launches is at most 93 k and 139 k cycles per token, about 0.002% of the totals: negligible.
  Host side (board, VL): (wall - device time) / launches is 39.9 ms and 41.7 ms per launch in the two steady calls, i.e.
  22.4 s and 23.5 s of the 63-64 s per token (35-37%). That residual holds everything outside device time (launch round
  trips, code uploads after evictions, the logits download, host work), not launches alone; the board data cannot
  split it.
- **Bound analysis (analytic).** The tables give, per class, seconds per token if its measured NUCA bytes moved at 1,
  2, 4, 8 or 19 GB/s chip-level, and tokens/s over all classes. With the bytes this code moves today: VL
  0.10 / 0.21 / 0.42 / 0.84 / 1.98 tokens/s, Qwen3.5 0.043 / 0.085 / 0.17 / 0.34 / 0.81 tokens/s. With every matmul
  weight read exactly once (lower bound: 2.03 GB for VL, 4.84 GB for Qwen3.5): VL 0.49 / 0.98 / 1.97 / 3.94 / 9.35 and
  Qwen3.5 0.21 / 0.41 / 0.83 / 1.65 / 3.93 tokens/s. The FP8 class dominates every column. Adding the board's host
  cost per launch (about 22 s per VL token) caps VL at 0.031-0.044 tokens/s at any of these rates: **once bandwidth is
  above about 1 GB/s, the per-launch host cost, not bandwidth, is the limit** unless it is reduced. The 22 s is the
  board residual of an image run with non-default settings, carried over as measured, not modelled.

### Implications for hardware and compiler priorities

1. **Compiler, first: the FP8 matmul loop.** It waits for each 512 B weight transfer before issuing the next (one
   outstanding), and stages a 16-row activation tile for M=1. Software-pipelined transfers (several `t.ld.async` in
   flight across the 32 steps, a prefetch of the next K block) and an M=1 lowering that avoids the 16-row staging and the
   accumulator spill are general mechanisms and address 83-87% of decode time. The 5-6x DRAM read amplification and the 4x
   LSU scratch traffic in the GEMV are the first things to remove.
2. **Compiler, second: launches and code-store reloads.** 536-716 launches and 396-564 code loads per token (TLM,
   scaled) against a 128 KiB code store, and the board spends 35-37% of a token outside device time. Fewer, larger
   launches (fusing a layer's small kernels) and a kernel set that fits the code store attack that share, which more
   bandwidth does not touch.
3. **The BF16 lm_head** is 10-11% and moves 3.7x its weight bytes; it should use the same pipelined weight path.
4. **Hardware: not bandwidth yet.** The weight path uses 2% of the 8 B/cycle/core interface, so a wider interface or HBM
   path will not show until the items above remove the stalls. The counters point to, but do not prove, two candidates:
   more outstanding requests per core, and a NUCA large enough to keep a K block's weight lines and the scratch apart
   (48 KiB in this configuration).

## Part 3. LSU in-flight ablation (TLM; requested for the run-2 freeze)

Domain: **TLM**. Same setup as part 2 (Qwen3-VL 1 layer, `core-vrf-sync-v5-x8`, default settings), with one model-config
change: `lsu_slots=1` instead of 2 (`decode_profile/abl-lsuslots1-vl-l1/model.cfg`; the only diff against
`vl-l1/model.cfg`). `tl_lsu_units=1` was tried first and is **refused by the model** ("exploratory LSU units must be
2|4|8|16"; the refused run is kept in `decode_profile/abl-lsu1-refused/`); and `tl_lsu_units` is read only on the
`core_pipeline=1` path, while this profile uses `core_pipeline=0`, where the in-flight limit of the LSU is `lsu_slots`.
The model has no hybrid (line path / word path) LSU, so a cut applies to every access: an upper bound on the cost of
dropping the second legacy unit. Generated tokens are identical to the default run. Launcher: `tlmrun-profile.sh` with
`TLMCFG=<cfg>`; compare: `ablation_compare.py`; output `ablation_lsuslots1_vl.txt`.

| Call | FP8 launches (cycles, default -> lsu_slots=1) | Whole call (cycles) | Ratio |
|---|---|---|---|
| steady decode, M=1 (call 3) | #101 9,396,737 -> 9,401,950 (1.0006); #102 4,900,293 -> 4,929,564 (1.0060); #115 27,878,350 -> 27,849,825 (0.9990); all 7 FP8 launches 0.9988 | 540,586,727 -> 558,243,818 | **1.0327** |
| prefill (11-token prompt, padded to M=16; not a 64-row tile) | all 7 FP8 launches 120,472,575 -> 120,642,364 | 582,976,965 -> 596,291,207 | **1.0228** |

- **The FP8 matmul launches do not change** (ratio 0.995-1.007 per launch, 0.9988 for the seven decode launches, 1.0014
  for the seven prefill launches; run-to-run differences of that size come from the model's timing interactions).
  Their weight path is the TTE, and the vector LSU traffic they generate is not on the critical path.
- The whole-call increase comes from the other classes: the bf16 lm_head template (decode #120: 420,623,160 ->
  438,349,194, ratio 1.0421; prefill #51 1.0271), and the attention kernels (prefill 1.09-1.20, decode #112 1.040,
  #109 1.006).
- Not measured: a 64-row prefill tile (the prefill here has 11 tokens), and Qwen3.5.

