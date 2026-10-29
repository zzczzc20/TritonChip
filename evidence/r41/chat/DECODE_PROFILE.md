# R41 decode profile: where decode time goes for the FP8 LLMs on TNCA

> **Binding:** [`doc/engineering_discipline.md`](../../../doc/engineering_discipline.md). Every number below is
> labelled by domain: **board** (U280, run-1 image), **TLM** (SystemC/TLM model, uncalibrated, `core-vrf-sync-v5-x8`,
> 3 cores, 100 MHz, default TNCA settings), **static** (read from the compiled assembly or the compiler source) or
> **analytic** (arithmetic on recorded numbers). No card access was used.

Status: **Part 1** (this commit): how the FP8 weights reach the matrix engine. **Part 2**: per-kernel and per-class
breakdown (added by a later commit).

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
