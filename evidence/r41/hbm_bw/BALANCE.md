# R41 run-2: throughput-matching analysis for W8A8 FP8 prefill and decode

Status: 2026-10-29, branch `r41-balance` (from `origin/r41-hbm-bw` at `35513d2d`, the merged run-2 candidate image). **No card was used and no simulation was run for this document.** [`doc/engineering_discipline.md`](../../../doc/engineering_discipline.md) is binding. Every number below is one of:

- **rtl**: read from RTL source or from an RTL-simulation log in this repository;
- **spec**: an image parameter or ISA text;
- **tlm**: a recorded SystemC/TLM run (uncalibrated model);
- **board**: a recorded run-1 board result (100 MHz);
- **analytic**: arithmetic on the above, by `tools/r41/balance_model.py` (reads the raw files, constants carry their source in the script);
- **projection**: analytic arithmetic that rests on a stated assumption about a change that does not exist yet. A projection is never a result.

The tables between `BEGIN`/`END` markers are generated: `python3 tools/r41/balance_model.py --out evidence/r41/hbm_bw/balance/balance_model.json --update-md evidence/r41/hbm_bw/BALANCE.md`. Do not edit them by hand.

## 0. Verdict

**Go for run 2 as prepared (no hardware change is required for balance), with one condition and one caveat.** The datapaths are not balanced for W8A8, but the dominant mismatch is in software.

1. **Prefill is bound by vector instruction issue, not by the matrix engine and not by memory.** The compiled Hub kernel executes **11,192 vector-core instructions per 16-token x 128 x 128 K-block iteration (262,144 MACs)** (analytic on the recorded `core-vrf-sync-v5` assembly). At one instruction per cycle that is 23.4 MAC/cycle/core, against 45.0 for the DP4 engine as the TLM measures a K32 command and 60.2 for its structural peak. With the DP4 lane, prefill reaches about **37 % of the DP4 peak at best (23 % with no overlap)**; the 4x lane buys 1.7x over the legacy engine in the same kernel (1.68x with perfect overlap, 1.73x with none), not 4x. The TLM agrees in direction: the DP lane gave a geometric mean of 1.09x on 59 compiled FP8/FP16 GEMM cases, at most 1.30x (`mma_dp/README.md`).
2. **The issue stream must shrink by about 1.9x (to 5.8 k instructions, the engine's 5,824 cycles per iteration) to expose the DP4 engine; a cut to about 2.7 k makes prefill engine-bound at about 70 % of the DP4 peak** (projection). That is the matched state. The pieces (§5): weight fragments by tensor-engine descriptor instead of 16 row loads with 8 bookkeeping instructions each (bookkeeping and address arithmetic are 56 % of the stream), activation quantisation hoisted out of the per-N-block program (about 2.3 k), and a larger accumulator tile for reuse. All are generic compiler or kernel-route work; none needs RTL.
3. **Decode has a hardware mismatch that software cannot fix.** At M = 1 the engine consumes at most 4 weight bytes per cycle per core (DP4 peak; 2.8 as measured per command), because a 16-token tile holds one real token. The memory path of the candidate delivers 29 B/cycle/core (the brief's figure). The ratio is **7-10x**. The software-fixed decode is engine-bound at about 0.4-0.6 tokens/s for Qwen3-VL-2B, against a memory ceiling of 4.3 tokens/s. As compiled today it is issue-bound at about 0.2-0.3.
4. **Condition for run 2:** the image must keep `MTX_ABUF 1` and the compiler must keep emitting K32 for E4M3 (it does in the recorded assembly). K64 is outside the A buffer, and with `MTX_VRF_REG 1` it reads the VRF every row cycle: 566 cycles against 182 for K32 in the TLM, 45 % against 70 % of the 64 MAC/cycle peak.
5. **Caveat on the fragment shape:** the line tensor engine moves only whole-64-byte-line rows; the A-buffered K32 fragment has 32-byte rows. A K32 weight fragment therefore takes the 8-byte word walk (8 B/cycle/core at most, about 2-3 B/cycle/core by tags-over-round-trip), which is about the engine's need (2.8-4 B/cycle/core), not above it. §3.4 and §7 give the cheap hardware fix (A buffer to 16 registers) and a software alternative (prepacked fragments). It is optional for run 2.
6. **Not a mismatch:** memory bandwidth for prefill (5 B/cycle/core needed at the DP4 peak against 29; 2 with a 64x64 tile), the VRF read port (9-11 % busy), SFU rate (1 element/cycle against about 1-3 % of cycles), the accumulator file.

Confidence: the **structure** (which stage binds and by how much the issue stage exceeds the engine) rests on recorded RTL/assembly counts and is high. The **absolute tokens/s** rest on IPC = 1 (a bound, not a measurement), on a K-block iteration cost that the run-1 TLM puts at 126 k cycles, and on no recorded run-2 execution of any of this kernel. Treat the run-2 tokens/s as a bracket (§6.3), not a prediction. Every run-2 row is analytic or projection; nothing about run 2 is a model or board result.

## 1. Inputs and what they fix

| Item | Value | Domain | Source |
|---|---|---|---|
| Cores, clock | 3, 100 MHz | spec | `g4/run2/image.tcl` (`MEM_PORTS 3`), `configs/r40-three-core.cfg` |
| Lanes per core | 16 (`VLEN`), one `tnca_mac_lane` per lane | rtl | `rtl/core/tnca_matrix.v` `g_mac[VLEN]` |
| Run-2 matrix options | `MAC_DP 1`, `MAC_CUTS 29` (0b11101), `MTX_VRF_REG 1`, `MTX_ABUF 1`, `FMA_IMPL 1`, `NUM_RCTX 4`, `HAS_SFU 1` (one slice) | spec | `g4/run2/image.tcl` at `35513d2d` |
| Lane modes | FP32 1x; FP16/BF16 and mixed FP8 2x; E4M3 x E4M3 4x products per lane per cycle | rtl | `fma_dslx/README.md` §1, `tnca_matrix.v` `kstep` |
| Command shape | k outer, m inner: per k-step one B-register read (`S_LOADB`, 1 cycle) then 16 row cycles | rtl | `tnca_matrix.v` state machine |
| Registers | 256 physical x 64 B = 16 KiB per core, 64 (4 KiB) per context; 4 banks, 1R+1W each; 5 read requesters (matrix = port 3, tensor engine = port 4) | rtl | `tnca_core.v` `tnca_vrf_banked` instance |
| Accumulators | 16 slots x 16 rows x 16 lanes x 4 B = 16 KiB; one read and one write of a row (64 B) per MAC cycle | rtl | `tnca_matrix.v`, `tnca_acc_rf.v` |
| NUCA | 32 sets x 8 ways x 64 B = 16 KiB per home, 48 KiB for 3 homes | spec | image `SETS 32 WAYS 8` |
| Memory ceilings | 29 B/cycle/core read, 38 copy-like (**the brief's figures; I found no recorded source for either in the repository**); one 64 B line per cycle at the port, home (chained fill) and 512-bit outlet; 11.2 B/cycle vector-LSU path best measured; 8 B/cycle transposed X8 word walk | brief / rtl | `G4_DESIGN.md` §8.5, §9.6; `LINK_BUDGET.md` §4 |
| Layer shapes | Qwen3-VL-2B-FP8: hidden 2048, 28 layers, K x N in {2048x2048, 2048x1024, 2048x6144, 6144x2048}. Qwen3.5-4B-FP8: hidden 2560, 32 layers, K x N in {2560x8192, 2560x1024, 2560x4096, 4096x2560, 2560x9216, 9216x2560}. Block scale 128 x 128, bf16 activations quantised per 128-wide K group | recorded | `models-tnca-evidence.tgz` (K, N lists); `fp8/qwen/*/plan.json` |
| Layer counts check | the shapes above reproduce the recorded `fp8_gemm_flops` of `plan.json` decode step exactly (Qwen3-VL 2,818,572,288; Qwen3.5 7,130,316,800, relative difference 0.0) | analytic | `balance_model.json` `plans` |
| lm_head | tied bf16 embedding, vocabulary 151,936 (VL) and 248,320 (Qwen3.5); not FP8 | recorded | `r41_chat.md`, `logits_download_bytes` 303,872 / 496,640 |
| Kernel | Hub `w8a8_block_dynamic_fp8_matmul_kernel`, compiled for `core-vrf-sync-v5`; swapped dot (weights are the A operand), 16-token chunks, 8 accumulators, 32 `m.mma.async.k32.e4m3` per K block | recorded | `fp8/kernels/README.md`, `fp8-kernels-evidence.tgz` `asm/` |

**What the 16-token chunk means.** `row_split` runs the whole program body (the K loop included) once per 16-row chunk and recomputes everything that does not carry the row dimension, the weight tile among it (`compiler/python/tnca_cc/row_split.py`). So the weights are re-read per 16 tokens, and a program is one 128-wide N block over the whole K. One *iteration* below is one K block of that: 16 tokens x 128 outputs x 128 K = 262,144 MACs, 16 KiB of e4m3 weights, 4 KiB of bf16 activations.

## 2. Compute rate (question 1)

<!-- BEGIN peak -->
| Mode (per core) | G | MAC/cycle (raw) | after 1-cycle step bubble | chip GMAC/s (3 cores) | chip TFLOP/s-eq |
|---|---:|---:|---:|---:|---:|
| legacy m.mma.async (any format; FP32 vector FMA is 16/cycle too) | 1 | 16 | 15.1 | 4.8 | 0.0096 |
| DP2 (FP16/BF16/mixed FP8) | 2 | 32 | 30.1 | 9.6 | 0.0192 |
| DP4 (E4M3 x E4M3) | 4 | 64 | 60.2 | 19.2 | 0.0384 |
<!-- END peak -->

- The engine has 16 lanes. With the E4M3 x E4M3 pair a lane does 4 products per cycle: 64 MAC/cycle/core, 192 MAC/cycle for 3 cores, 19.2 GMAC/s, **38.4 GFLOP/s (0.0384 TFLOP/s)** for the chip at 100 MHz. FP16/BF16 and mixed FP8 are 19.2 GFLOP/s, the legacy `m.mma.async` of run 1 (any format) and the FP32 vector FMA are 9.6 GFLOP/s each. These are rtl-structural peaks.
- The one-cycle B-register bubble per k-step takes 6 % off every mode (16/17).
- A whole command is lower still. **tlm** busy cycles for one 16x16xK command, from `mma_dp/chip_validation.json`:

<!-- BEGIN eng -->
| K | legacy `m.mma.async` e4m3 (busy cycles) | `m.mma.dp.async` e4m3 x e4m3 | ratio | MAC/cycle legacy | MAC/cycle DP4 | DP4 % of 64 |
|---|---:|---:|---:|---:|---:|---:|
| 16 | 313 | 110 | 2.85 | 13.1 | 37.2 | 58 % |
| 32 | 589 | 182 | 3.24 | 13.9 | 45.0 | 70 % |
| 64 | 2,149 | 566 | 3.80 | 7.6 | 28.9 | 45 % |
<!-- END eng -->

  K32 E4M3 reaches 45.0 MAC/cycle (70 % of the 64 peak) and the DP lane is 3.24x the legacy engine on that command. The remaining 30 % is the A-buffer fill (8 reads), the drain (`PIPE` + 1 = 6) and model overheads; the RTL-structural sum is 150 cycles (8 + 8 x 17 + 6), the model says 182. `MTX_VRF_REG 1` adds one cycle to each of the 16 VRF reads of a K32 command (RTL lockstep log, `timing/README.md`: delta equals accepted reads), so the run-2 RTL figure is near 166. The sensitivity table (§6.3) spans 150-220.
- Qwen3-VL-2B needs 1.41 GMAC per token; the DP4 peak is therefore 13.6 tokens/s and the realised-command peak (45 MAC/cycle/core) 9.6 tokens/s. Qwen3.5-4B (3.57 GMAC per token) peaks at 5.4 and 3.8.

## 3. Operand supply and memory (questions 2 and 3)

### 3.1 VRF to engine

Per MAC cycle in DP4 mode the lanes consume one 32-bit A word (4 e4m3, broadcast to the 16 lanes) and, from `b_row`, 16 x 32 bit of B. `b_row` is loaded once per k-step and held for the 16 row cycles. So the steady operand demand is **4 B of A + 4 B of B = 8 B/cycle/core** (legacy: 2 B/cycle; DP2: 4). The A word comes from `MTX_ABUF` (the A fragment is read once, 8 registers for K32, then swept), so the VRF sees 8 reads (A fill) + 8 reads (B) = 16 words of 64 B per 150-182 cycles: **5.6-6.8 B/cycle, 9-11 % of the matrix port's 64 B/cycle (one 512-bit word per cycle)**. The accumulator file does one row read and one row write (64 B each) per MAC cycle in every mode, so DP4 does 4x the MACs per accumulator access.

**The 4x mode is compute-bound (command structure), not operand-bound, with `MTX_ABUF`.** Without the buffer (K64 E4M3: 16 registers, outside the buffer) the sweep reads the VRF once per row cycle, 100 % of port 3, and with `MTX_VRF_REG 1` each read costs one extra cycle: the TLM shows 566 cycles for K64 (2.08 cycles per row) against 182 for K32. Hence the condition in §0.

### 3.2 Accumulator tile and reuse

The compiled kernel uses 8 of the 16 accumulator slots (one 16-token x 128-output tile). The slots allow a 64 x 64 output tile (4 x 4 slots). Per K step the bytes brought in are N_tile weights + M_tile e4m3 activations, so the MAC per byte are M N/(M + N): 14.2 for 16 x 128 (with bf16 activations 12.7), 32 for 64 x 64. Register capacity allows it: an A fragment is 8 registers and a B fragment 8, of 64 per context. This is a tiling choice (software).

### 3.3 Arithmetic intensity and required bandwidth

Current tiling: weights re-read per 16-token chunk, bf16 activations re-read per 128-column program, bf16 output. Alternative: 64 x 64 tile, activations pre-quantised to e4m3. The last two columns are the B/cycle/core the stage must deliver for the engine to run at the DP4 peak (128 FLOP/cycle/core divided by FLOP/B).

<!-- BEGIN ai -->
| Shape (K x N) | T tokens | weights re-read (MB) | AI current (FLOP/B) | AI 64x64 pre-quant (FLOP/B) | B/cycle/core needed at DP4 peak, current | same, 64x64 |
|---|---:|---:|---:|---:|---:|---:|
| VL q/o 2048x2048 | 128 | 33.6 | 25.3 | 62.1 | 5.1 | 2.1 |
| VL q/o 2048x2048 | 512 | 134.2 | 25.3 | 62.1 | 5.1 | 2.1 |
| VL q/o 2048x2048 | 2048 | 536.9 | 25.3 | 62.1 | 5.1 | 2.1 |
| VL q/o 2048x2048 | 1 | 4.2 | 2.0 | 2.0 | 65.1 | 65.1 |
| VL gate/up 2048x6144 | 128 | 100.7 | 25.3 | 62.1 | 5.1 | 2.1 |
| VL gate/up 2048x6144 | 512 | 402.7 | 25.3 | 62.1 | 5.1 | 2.1 |
| VL gate/up 2048x6144 | 2048 | 1,610.6 | 25.3 | 62.1 | 5.1 | 2.1 |
| VL gate/up 2048x6144 | 1 | 12.6 | 2.0 | 2.0 | 65.1 | 65.1 |
| VL down 6144x2048 | 128 | 100.7 | 25.5 | 63.3 | 5.0 | 2.0 |
| VL down 6144x2048 | 512 | 402.7 | 25.5 | 63.3 | 5.0 | 2.0 |
| VL down 6144x2048 | 2048 | 1,610.6 | 25.5 | 63.3 | 5.0 | 2.0 |
| VL down 6144x2048 | 1 | 12.6 | 2.0 | 2.0 | 65.0 | 65.0 |
| Q35 gate/up 2560x9216 | 128 | 188.7 | 25.3 | 62.4 | 5.0 | 2.1 |
| Q35 gate/up 2560x9216 | 512 | 755.0 | 25.3 | 62.4 | 5.0 | 2.1 |
| Q35 gate/up 2560x9216 | 2048 | 3,019.9 | 25.3 | 62.4 | 5.0 | 2.1 |
| Q35 gate/up 2560x9216 | 1 | 23.6 | 2.0 | 2.0 | 65.0 | 65.0 |
| Q35 down 9216x2560 | 128 | 188.7 | 25.5 | 63.6 | 5.0 | 2.0 |
| Q35 down 9216x2560 | 512 | 755.0 | 25.5 | 63.6 | 5.0 | 2.0 |
| Q35 down 9216x2560 | 2048 | 3,019.9 | 25.5 | 63.6 | 5.0 | 2.0 |
| Q35 down 9216x2560 | 1 | 23.6 | 2.0 | 2.0 | 65.0 | 65.0 |
| Q35 lin-attn qkv 2560x8192 | 128 | 167.8 | 25.3 | 62.4 | 5.0 | 2.1 |
| Q35 lin-attn qkv 2560x8192 | 512 | 671.1 | 25.3 | 62.4 | 5.0 | 2.1 |
| Q35 lin-attn qkv 2560x8192 | 2048 | 2,684.4 | 25.3 | 62.4 | 5.0 | 2.1 |
| Q35 lin-attn qkv 2560x8192 | 1 | 21.0 | 2.0 | 2.0 | 65.0 | 65.0 |
<!-- END ai -->

- Prefill: **5 B/cycle/core at the DP4 peak with today's tiling, 2 with a 64 x 64 tile**, against 29 (read ceiling), 11.2 (vector-LSU best measured) or 8 (X8 word walk): the memory path has 1.6x-6x margin even on its weakest figure, and DRAM traffic at T = 2048 (about 180 GB of weight re-reads for Qwen3-VL-2B at one 16-token pass per weight byte, 3 cores x 29 B/cycle) is about 10 % of the compute time of the software-fixed case. **Prefill is not memory-bound at any prompt length that fits the registered chat flow.**
- Decode (T = 1): AI = 2 FLOP/B at every shape. This is the memory-bound regime; the question is what the engine can consume (§3.5).
- NUCA reuse is negligible: 48 KiB total against a weight strip of 128 x K bytes (256 KiB at K = 2048).

### 3.4 Weight fragment delivery

The weight tile reaches the matrix as 16 row loads of 32 bytes per K32 fragment in the recorded `core-vrf-sync-v5` assembly (each with about 8 scalar/vector bookkeeping instructions: `v.movi`, `v.ld.w`, `p.and`, `v.extract`, `s.add`, `s.addi`, `v.or`, `v.and`, `v.mov`). A tensor-engine descriptor load would replace that. Hardware limits on that path:

| Path | Per core | Domain | Note |
|---|---:|---|---|
| Word-walk tensor engine, 16 tags x 8 B over a 40-60 cycle round trip | 2.1-3.2 B/cycle | analytic | tags from `LINK_BUDGET.md` §4; the round trip is not measured on the board |
| Same, at the 8 B/cycle port limit | 8 B/cycle | rtl | `G4_DESIGN.md` §9.6 |
| Line tensor engine (`TENSOR_LINE 1`), whole-64 B-line rows | up to ~58 (0.9 line/cycle) | rtl | `G4_DESIGN.md` §9.2, §9.6 |
| Engine need, K32 DP4, 16 tokens per weight | 2.8 (measured command) - 4 (peak) B/cycle | analytic | |

The K32 fragment has 32-byte rows, so it is **not eligible for the line engine** (it needs whole 64-byte-line rows, line-aligned) and takes the word walk, whose tag-limited rate sits right at the engine's need. A K64 fragment is line-shaped but is outside `MTX_ABUF`. The two fixes are in §7 (M4).

### 3.5 Decode ladder

At M = 1 the 16-token chunk holds one real token, so each weight byte is used once, and the engine consumes at most 4 weight bytes per cycle per core (4 A bytes per MAC cycle of the sweep).

<!-- BEGIN ladder_decode -->
| Stage (decode, weights consumed) | B/cycle/core | chip GB/s (3 cores, 100 MHz) | domain |
|---|---:|---:|---:|
| Memory path, task-brief read ceiling | 29.00 | 8.70 | brief |
| Memory path, vector-LSU best measured | 11.23 | 3.37 | rtl |
| Memory path, X8 transposed word walk | 8.00 | 2.40 | rtl |
| Engine geometry at M=1 (DP4 peak; 4 weight bytes per cycle) | 4.00 | 1.20 | rtl (analytic) |
| Engine geometry at M=1 (32 K32 commands, 182 cycles each) | 2.81 | 0.84 | tlm + analytic |
| Vector issue as compiled (16,384 B per 11,192 instr) | 1.46 | 0.44 | analytic on asm |
| Vector issue, software projection (2,652 instr) | 6.18 | 1.85 | analytic projection |
| Run-1 TLM point (16 KiB per 126,080 cycles) | 0.13 | 0.04 | tlm |
<!-- END ladder_decode -->

<!-- BEGIN decode_bound -->
| Model | Ceiling | bytes per token (recorded plan.json) | tokens/s at that ceiling |
|---|---:|---:|---:|
| Qwen3-VL-2B-Instruct-FP8 | Memory path, task-brief read ceiling | 2.032 | 4.281 |
| Qwen3-VL-2B-Instruct-FP8 | Memory path, vector-LSU best measured | 2.032 | 1.658 |
| Qwen3-VL-2B-Instruct-FP8 | Memory path, X8 transposed word walk | 2.032 | 1.181 |
| Qwen3-VL-2B-Instruct-FP8 | Engine geometry at M=1 (DP4 peak; 4 weight bytes per cycle) | 2.032 | 0.590 |
| Qwen3-VL-2B-Instruct-FP8 | Engine geometry at M=1 (32 K32 commands, 182 cycles each) | 2.032 | 0.415 |
| Qwen3.5-4B-FP8 | Memory path, task-brief read ceiling | 4.847 | 1.795 |
| Qwen3.5-4B-FP8 | Memory path, vector-LSU best measured | 4.847 | 0.695 |
| Qwen3.5-4B-FP8 | Memory path, X8 transposed word walk | 4.847 | 0.495 |
| Qwen3.5-4B-FP8 | Engine geometry at M=1 (DP4 peak; 4 weight bytes per cycle) | 4.847 | 0.248 |
| Qwen3.5-4B-FP8 | Engine geometry at M=1 (32 K32 commands, 182 cycles each) | 4.847 | 0.174 |
<!-- END decode_bound -->

The bytes per token are the recorded `graph_weight_bytes_read` of the decode step (FP8 body plus the bf16 lm_head). The memory path would give 4.3 tokens/s (Qwen3-VL) if the engine could take the bytes; the engine geometry caps it at 0.4-0.6, the issue stage as compiled at about 0.2-0.3.

## 4. Stage ladder for prefill (question 3, "which stage binds")

Sustained MAC/cycle/core on one iteration, each stage alone (a stage is binding when it is the lowest that cannot overlap with the others):

<!-- BEGIN ladder_prefill -->
| Stage (prefill, per core) | MAC/cycle sustained | chip GMAC/s (3 cores, 100 MHz) | domain |
|---|---:|---:|---:|
| MAC lanes, DP4 peak (16 lanes x 4) | 64.0 | 19.20 | rtl |
| MAC lanes, DP4 after the 1-cycle step bubble | 60.2 | 18.07 | rtl (analytic) |
| Engine, whole K32 command as measured | 45.0 | 13.50 | tlm |
| VRF operand supply: 8 B/cycle needed vs 64 B/cycle port, ABUF on | - | - | rtl |
| Vector issue, kernel as compiled (11,192 instr, IPC 1) | 23.4 | 7.03 | analytic on asm |
|   of which pre-mma region alone | 95.8 | 28.75 | analytic on asm |
|   of which mma window (weight row loads) alone | 43.7 | 13.11 | analytic on asm |
|   of which post-mma scale/accumulate alone | 106.6 | 31.97 | analytic on asm |
| Vector issue, software projection (2,652 instr) | 98.8 | 29.65 | analytic projection |
| Memory path 29 B/cycle at AI 25.3 FLOP/B (12.7 MAC/B) | 366.9 | 110.06 | analytic |
| Memory path 29 B/cycle at 64x64 tile (31 MAC/B) | 901.9 | 270.57 | analytic |
| Memory path 8 B/cycle (X8 word walk) at 12.7 MAC/B | 101.2 | 30.36 | analytic |
<!-- END ladder_prefill -->

The lowest stage is the vector issue as compiled (23.4), then the engine as measured (45.0). **Prefill binds on vector issue.** Each region of the loop alone supports 44 (weight row loads), 96 (activation quantisation, B scatter) or 107 (block scale and accumulate) MAC/cycle, which is close to or above what the DP4 engine delivers; it is their sum that is too long. The memory path (29 B/cycle) is 8x above the engine and 16x above the issue stage; even the 8 B/cycle X8 word walk is 2.2x above the engine.

## 5. Epilogue costs that are not MACs (question 4)

Instruction counts of the loop body of the recorded `w8a8_block_dynamic_fp8_matmul_kernel` for `core-vrf-sync-v5` (the run-1 profile; no assembly of the run-2 profile is recorded, so this is the proxy). Counted from the archive by the script, non-comment lines between `for_26:` and `endfor_27:`; the three regions are cut at the first and last `m.mma.async`:

<!-- BEGIN asm -->
| Region of one K-block loop iteration | instructions |
|---|---:|
| pre_mma (act-quant, B scatter) | 2,735 |
| mma_window (weight row loads, mma) | 5,997 |
| post_mma (scale, accumulate, store) | 2,460 |
| total loop body | 11,192 |

| Class | instructions | share |
|---|---:|---:|
| vector int/move | 3,540 | 31.6 % |
| scalar/predicate/ctl | 3,339 | 29.8 % |
| vector fp | 2,185 | 19.5 % |
| vector load/store | 1,704 | 15.2 % |
| matrix/accumulator | 424 | 3.8 % |
<!-- END asm -->

The position-based regions overlap in function: the scheduler interleaved the activation quantisation of later K groups into the mma window (the window holds 96 `v.rcp.approx`, 96 `vf.cvt.f32.e4m3`, 96 `v.st.scatter.b` and 384 `vf.fms`). By mnemonic, the whole loop is:

<!-- BEGIN mnem -->
| Role (my reading of the mnemonics) | mnemonics | instructions in the loop body |
|---|---:|---:|
| activation quantisation, distinctive | `v.ld.h` 128, `vf.cvt.bf16.f32` 128, `vf.abs` 128, `vf.max.ieee` 240, `vf.reduce.max.b` 16, `v.rcp.approx` 129, `vf.cvt.f32.e4m3` 128, `v.st.scatter.b` 128 | 1,025 |
| IEEE division glue (amax/448, x/scale) | `vf.fms` 516, `vf.neg` 258, `vf.cmp.o` 258, `v.sel` 262 | 1,294 |
| block scale and accumulate, distinctive | `a.vread` 256, `vf.add` 128, `a.vwrite` 128, `v.st.scatter.w` 128 | 640 |
| weight and scale loads | `v.ld.w` 1056 | 1,056 |
| lane extract / splat / bit-merge bookkeeping | `v.extract` 912, `v.splat` 288, `v.or` 384, `v.and` 385, `v.mov` 288, `v.movi` 681, `v.lshr` 130 | 3,068 |
| scalar address arithmetic and predicates | `s.addi` 1465, `s.add` 817, `s.movz` 401, `s.mov` 4, `p.and` 512 | 3,199 |
| matrix command and wait | `m.mma.async.k32.e4m3` 32, `mb.arm` 32, `mb.wait` 32 | 96 |
| vf.mul | `vf.mul` 385 | 385 |
<!-- END mnem -->

- **Activation quantisation, about 2.3 k instructions** (1,025 distinctive plus the IEEE-division glue: the `amax / 448` and `x / scale` of `div_ieee`, one faithful quotient plus one correction). It is repeated by every N-block program of the grid (16 for N = 2048, 72 for N = 9216), because the Hub kernel fuses the quantisation: the same activations are quantised N/128 times.
- **Weight-fragment assembly.** 512 predicated 32-byte row loads per iteration (16 per K32 fragment, 32 fragments) and the lane-extract/constant/scalar-address bookkeeping around them: the `v.extract`, `v.movi`, `s.add`, `s.addi`, `p.and` groups repeat about 8 times per row load in the recorded listing. **Bookkeeping plus scalar address arithmetic is 6.3 k of the 11.2 k instructions (56 %)**, which is where a descriptor-based load would remove the most.
- **Block scale and accumulate, 2,460 instructions after the last mma** (`a.vread` of the accumulator rows, `vf.mul` by the row's activation scale and by the 16-lane weight-scale chunk, `vf.add` into the carried tile, `a.vwrite`, stores). There is no `a.mul.row`/`a.fma.row`. Fusing the two scales and the add into one FMA row operation would change the rounding the reference fixes ("TNCA does not contract to an FMA", `fp8/kernels/README.md`); that is not an option without the author.
- Whole-program prologue and epilogue (about 1.6 k instructions per program, 16-72 iterations per program) are under 1 % per iteration and are left out.

**bf16 conversion and quantisation to E4M3** are inside these regions (the vector FPU, 16 lanes, one `vf.*` per cycle). **rmsnorm** runs as its own kernel before the matmul; per element it is a square, a reduction, one SFU `rsqrt` per token and two multiplies: by estimate about 0.4-0.5 instruction cycles per element, 1-2 Mcycles per norm at T = 2048 and hidden 2048, under 0.1 % of the layer.

**Non-GEMM work as a share of prefill** (analytic estimates, not measurements: attention matmul on the DP2 path at 50 % of the DP2 peak, softmax exp and SwiGLU `silu` (two SFU operations) at one SFU element per cycle per core; the delta-rule recurrence of Qwen3.5's 24 linear-attention layers is not modelled):

<!-- BEGIN other -->
| Model | Scenario | T | GEMM Mcycles | attention matmul Mcycles | softmax exp Mcycles | SwiGLU SFU Mcycles | non-GEMM share |
|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen3-VL-2B-Instruct-FP8 | run-2, kernel as compiled, DP4 engine | 512 | 10,269 | 313.8 | 19.6 | 58.7 | 3.7 % |
| Qwen3-VL-2B-Instruct-FP8 | run-2, kernel as compiled, DP4 engine | 2048 | 41,075 | 5,013.2 | 313.3 | 234.9 | 11.9 % |
| Qwen3-VL-2B-Instruct-FP8 | run-2, DP4, issue cut by software (tile loads, quant hoisted) | 512 | 5,344 | 313.8 | 19.6 | 58.7 | 6.8 % |
| Qwen3-VL-2B-Instruct-FP8 | run-2, DP4, issue cut by software (tile loads, quant hoisted) | 2048 | 21,374 | 5,013.2 | 313.3 | 234.9 | 20.6 % |
| Qwen3.5-4B-FP8 | run-2, kernel as compiled, DP4 engine | 512 | 25,977 | 179.3 | 5.6 | 100.7 | 1.1 % |
| Qwen3.5-4B-FP8 | run-2, kernel as compiled, DP4 engine | 2048 | 103,910 | 2,864.7 | 89.5 | 402.7 | 3.1 % |
| Qwen3.5-4B-FP8 | run-2, DP4, issue cut by software (tile loads, quant hoisted) | 512 | 13,518 | 179.3 | 5.6 | 100.7 | 2.1 % |
| Qwen3.5-4B-FP8 | run-2, DP4, issue cut by software (tile loads, quant hoisted) | 2048 | 54,072 | 2,864.7 | 89.5 | 402.7 | 5.8 % |
<!-- END other -->

The SFU (1 element/cycle) is not a bottleneck: the exps of a causal attention layer at T = 2048 are 0.8 % of the layer's GEMM cycles at the DP4-as-compiled rate. The **attention matmul grows from 5-10 % to about 20 % of prefill once the GEMM is fast**, which is the largest non-GEMM item and the one with the least evidence: how the registered attention lowers (matmul on the engine or on the vector FPU) is not recorded here. If it ran on the vector FPU at 16 MAC/cycle it would be 3x worse and a first-order cost at T = 2048.

## 6. End-to-end (question 5)

### 6.1 The algebra

For a linear of shape (K, N) and T tokens: iterations = ceil(T/16) x ceil(N/128) x ceil(K/128). Whole model, per 16-token chunk: **86,016 iterations (Qwen3-VL), 217.6 k (Qwen3.5)**. Cycles of the matmuls = iterations x T_iter / 3 cores. Add the non-GEMM cycles of §5. Tokens/s = T x 10^8 / cycles. Decode is one chunk (M = 1 padded to 16) plus the lm_head: bf16, 622 MB (VL) or 1.27 GB (Qwen3.5), taken at the path ceiling (29 B/cycle/core x 3) and, in brackets, at the body's own bytes-per-cycle rate. T_iter by scenario, cycles per core per iteration:

<!-- BEGIN scen -->
| Scenario (cycles per core per 16-token x 128 x 128 iteration) | domain | issue | engine (32 cmds) | memory | T opt (overlap) | T pess (serial) | binding stage (opt) | % of DP4 peak |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| run-1 image, TLM point 16x9216x2560 (as compiled, legacy engine) | tlm | - | - | - | 126,080 | 126,080 | memory latency / LSU MLP (run-1) | 3.2-3.2 % |
| run-2, kernel as compiled, legacy engine (no DP) | analytic | 11,192 | 18,848 | 706 | 18,848 | 30,746 | mma | 21.7-13.3 % |
| run-2, kernel as compiled, DP4 engine | analytic | 11,192 | 5,824 | 706 | 11,192 | 17,722 | issue | 36.6-23.1 % |
| run-2, DP4, issue cut by software (tile loads, quant hoisted) | analytic projection | 2,652 | 5,824 | 565 | 5,824 | 9,041 | mma | 70.3-45.3 % |
<!-- END scen -->

The last column is relative to the DP4 peak (64 MAC/cycle/core) in every row; the legacy-engine row is at 87 % of its own 16 MAC/cycle peak.

- *Run-1 row (tlm):* 60,518,611 cycles for the 16 x 9216 x 2560 case (1 x 72 x 1 programs, 480 iterations per core) = 126,080 cycles per iteration, 0.13 B/cycle/core of weights. It is consistent with a memory-latency bound: 126,080 / 1,056 `v.ld.w` is 119 cycles per load, on an image with two LSU commands in flight and a 73-cycle mean round trip (`LINK_BUDGET.md` §4); the TLM run does not report which stage stalls.
- *Legacy engine, as compiled:* T_iter = max(issue 11,192; 32 x 589 = 18,848; memory 706) = engine-bound (the legacy engine runs at 87 % of its own 16 MAC/cycle peak).
- *DP4 as compiled:* issue 11,192 > engine 5,824 > memory 706. *opt* assumes the issue stream, the engine and the memory overlap perfectly (4 contexts per core make this possible, not certain); *pess* assumes no overlap.
- *Software projection:* issue 2,652 = 32 x 6 (one descriptor load, arm and wait per fragment) + 2,460 (the scale epilogue kept as compiled). The activation quantisation is assumed amortised away (hoisted out of the N loop) and the weight rows delivered by descriptor. This is an assumption about code that does not exist; it is not a measurement and the 6-instruction figure is an estimate.

### 6.2 Tokens/s

*opt* (perfect overlap):

<!-- BEGIN e2e_opt -->
| Model | Scenario (T opt) | prefill tok/s T=128 | T=512 | T=2048 | decode tok/s, lm_head at path ceiling (at body rate) |
|---|---:|---:|---:|---:|---:|
| Qwen3-VL-2B-Instruct-FP8 | run-1 image, TLM point 16x9216x2560 (as compiled, legacy engine) | 0.44 | 0.44 | 0.44 | 0.025 (0.025) |
| Qwen3-VL-2B-Instruct-FP8 | run-2, kernel as compiled, legacy engine (no DP) | 2.94 | 2.90 | 2.74 | 0.183 (0.128) |
| Qwen3-VL-2B-Instruct-FP8 | run-2, kernel as compiled, DP4 engine | 4.92 | 4.80 | 4.39 | 0.305 (0.216) |
| Qwen3-VL-2B-Instruct-FP8 | run-2, DP4, issue cut by software (tile loads, quant hoisted) | 9.33 | 8.93 | 7.60 | 0.574 (0.415) |
| Qwen3.5-4B-FP8 | run-1 image, TLM point 16x9216x2560 (as compiled, legacy engine) | 0.17 | 0.17 | 0.17 | 0.010 (0.010) |
| Qwen3.5-4B-FP8 | run-2, kernel as compiled, legacy engine (no DP) | 1.17 | 1.16 | 1.15 | 0.072 (0.054) |
| Qwen3.5-4B-FP8 | run-2, kernel as compiled, DP4 engine | 1.96 | 1.95 | 1.91 | 0.121 (0.091) |
| Qwen3.5-4B-FP8 | run-2, DP4, issue cut by software (tile loads, quant hoisted) | 3.75 | 3.71 | 3.57 | 0.229 (0.174) |
<!-- END e2e_opt -->

*pess* (no overlap):

<!-- BEGIN e2e_pess -->
| Model | Scenario (T pess) | prefill tok/s T=128 | T=512 | T=2048 | decode tok/s, lm_head at path ceiling (at body rate) |
|---|---:|---:|---:|---:|---:|
| Qwen3-VL-2B-Instruct-FP8 | run-1 image, TLM point 16x9216x2560 (as compiled, legacy engine) | 0.44 | 0.44 | 0.44 | 0.025 (0.025) |
| Qwen3-VL-2B-Instruct-FP8 | run-2, kernel as compiled, legacy engine (no DP) | 1.81 | 1.79 | 1.73 | 0.113 (0.079) |
| Qwen3-VL-2B-Instruct-FP8 | run-2, kernel as compiled, DP4 engine | 3.12 | 3.07 | 2.90 | 0.194 (0.137) |
| Qwen3-VL-2B-Instruct-FP8 | run-2, DP4, issue cut by software (tile loads, quant hoisted) | 6.07 | 5.89 | 5.29 | 0.375 (0.268) |
| Qwen3.5-4B-FP8 | run-1 image, TLM point 16x9216x2560 (as compiled, legacy engine) | 0.17 | 0.17 | 0.17 | 0.010 (0.010) |
| Qwen3.5-4B-FP8 | run-2, kernel as compiled, legacy engine (no DP) | 0.72 | 0.71 | 0.71 | 0.045 (0.033) |
| Qwen3.5-4B-FP8 | run-2, kernel as compiled, DP4 engine | 1.24 | 1.24 | 1.22 | 0.077 (0.057) |
| Qwen3.5-4B-FP8 | run-2, DP4, issue cut by software (tile loads, quant hoisted) | 2.42 | 2.41 | 2.35 | 0.149 (0.112) |
<!-- END e2e_pess -->

The prefill rows include the §5 non-GEMM estimates. The run-1 decode row is the board's decode (0.0246 tokens/s of device time; 0.0157 with host time, 563 launches) for Qwen3-VL by construction (non-body cycles fitted), and an extrapolation for Qwen3.5 (no Qwen3.5 board run exists). The run-1 prefill rows are the TLM iteration cost at every T; the TLM point at M = 129 is 1.86x more expensive per iteration (below), so those rows are optimistic.

### 6.3 Cross-check against the board and the TLM, and sensitivity

<!-- BEGIN xcheck -->
| Quantity (Qwen3-VL-2B, run-1 image) | value | domain |
|---|---:|---:|
| FP8 K-block iterations per 16-token chunk, whole model (from shapes) | 86,016 | analytic |
| TLM point 16x9216x2560: cycles / grid | 60,518,611 / [1, 72, 1] | tlm |
| cycles per iteration per core from the TLM point | 126,080 | tlm |
| board steady decode device cycles (calls 2, 3) | 4,065,511,739 / 4,065,552,515 | board |
| board prefill, 11 tokens, device cycles | 4,238,154,739 | board |
| cycles per iteration implied by the board decode (upper bound, all work attributed) | 141,794 | board + analytic |
| decode cycles predicted from the TLM iteration cost (body only) | 3,614,978,364 | tlm + analytic |
| board minus predicted body (lm_head, attention, norms, launch gaps) | 450,533,375 | board - model |
| prediction / board | 0.889 | - |

| TLM point (run-1 profile, block-dynamic kernel) | grid | cycles | iterations per core | cycles per iteration |
|---|---:|---:|---:|---:|
| 7x6144x2048 | 1x48x1 | 32,523,722 | 256 | 127,046 |
| 7x2048x6144 | 1x16x1 | 34,001,676 | 256 | 132,819 |
| 16x9216x2560 | 1x72x1 | 60,518,611 | 480 | 126,080 |
| 129x2048x2048 | 2x16x1 | 180,272,870 | 768 | 234,730 |
<!-- END xcheck -->

- The run-1 calibration predicts the board's steady decode device cycles to **0.889** (body only; the remaining 11 % is lm_head, attention, norms and launch gaps). The board's 11-token prefill (4.238 x 10^9 cycles) is within 4 % of its decode (4.066 x 10^9), as the model says: both are one chunk through the same iterations. **This validates the iteration-cost structure for run 1 only. No run-2 number has any board or TLM check.**
- The TLM gives 126-133 k cycles per iteration for M <= 16 and 235 k for M = 129 (grid 2 x 16): per-iteration cost rises 1.86x when the kernel loops chunks inside a program. The cause is not diagnosed here; it affects every prefill estimate at T >= 128, in all scenarios, by an unknown factor up to that.
- **TLM check of the DP lane (tlm, `mma_dp/cases_summary.json`):** the 59 compiled FP8/FP16/BF16 GEMM graphs run 1.0025-1.30x faster with `m.mma.dp.async`, geometric mean 1.09; the largest, 128 x 192 x 384 e4m3, runs 316 k cycles base and 277 k DP (1.14x), against a 3.24x ratio of the isolated command (table above). This is consistent with an engine that does not bind, but it is not a test of this document's issue-bound claim: those kernels differ from the Hub kernel and the model's run-1 memory path may dominate them. It does not test the run-2 numbers.

Sensitivity of the Qwen3-VL-2B numbers to the largest assumptions (GEMM only; decode with lm_head at the path ceiling):

<!-- BEGIN sens -->
| Case (Qwen3-VL-2B) | overlap | T iter (cycles) | prefill tok/s at T=512 (GEMM only) | decode tok/s |
|---|---:|---:|---:|---:|
| as compiled, IPC 1.0, cmd 182 | opt | 11,192 | 4.99 | 0.305 |
| as compiled, IPC 1.0, cmd 182 | pess | 17,581 | 3.17 | 0.196 |
| as compiled, IPC 0.7 | opt | 15,989 | 3.49 | 0.215 |
| as compiled, IPC 0.7 | pess | 22,378 | 2.49 | 0.154 |
| as compiled, IPC 0.5 | opt | 22,384 | 2.49 | 0.154 |
| as compiled, IPC 0.5 | pess | 28,773 | 1.94 | 0.120 |
| software projection, IPC 1.0, cmd 182 | opt | 5,824 | 9.58 | 0.574 |
| software projection, IPC 1.0, cmd 182 | pess | 9,041 | 6.17 | 0.375 |
| software projection, IPC 0.5 | opt | 5,824 | 9.58 | 0.574 |
| software projection, IPC 0.5 | pess | 11,693 | 4.77 | 0.292 |
| software projection, cmd 150 (RTL structural) | opt | 4,800 | 11.63 | 0.691 |
| software projection, cmd 150 (RTL structural) | pess | 8,017 | 6.96 | 0.422 |
| software projection, cmd 220 (+20 %) | opt | 7,040 | 7.93 | 0.478 |
| software projection, cmd 220 (+20 %) | pess | 10,257 | 5.44 | 0.332 |
| software projection, issue 5,000 instr | opt | 5,824 | 9.58 | 0.574 |
| software projection, issue 5,000 instr | pess | 11,389 | 4.90 | 0.300 |
<!-- END sens -->

The biggest unknowns, in order: (1) the achievable IPC of the as-compiled loop (1.0 is an upper bound; the G4 RTL probes show IPC 0.15-0.5 on streaming code, though those are memory-bound), a factor 2 at IPC 0.5; (2) the per-chunk cost at M > 16 (up to 1.86x in the run-1 TLM); (3) overlap between issue and engine (1.5-1.6x between *opt* and *pess*); (4) the engine command cycles (150-220: +-20 %); (5) the issue count of the projection.

## 7. Mismatch report (question 6)

"Mismatch" = adjacent stages differing by more than 2x in sustained rate on the same work.

| # | Stages (rates) | Ratio | Cause | Fixable by | Rough cost | Run 2 or later |
|---|---|---:|---|---|---|---|
| M1 | Vector issue as compiled (23.4 MAC/cycle) below the DP4 engine peak (60.2) | 2.6x (1.9x against the measured command) | activation quantisation repeated per N block (about 2.3 k), per-row weight loads with address/lane bookkeeping (6.3 k bookkeeping and address arithmetic in all), scale epilogue (2.5 k) | **software**: descriptor fragment loads, quantisation hoisted into one pass over the activations (the Hub's separate-quantisation kernels take pre-quantised activations: the v1 `w8a8_block_fp8_matmul` ran 24.8 M against 32.5 M TLM cycles for 7 x 6144 x 2048, quantisation excluded, run-1 profile), 64 x 64 tile | compiler/kernel-route work; no RTL | **run 2 can ship**; the gain from DP4 depends on this |
| M2 | Memory path (29 B/cycle/core) above prefill demand (2-5 B/cycle/core) | 6-14x, in the harmless direction | the memory path is sized for streaming kernels | none needed | none | no action |
| M3 | Decode: memory path (29) over engine geometry at M = 1 (2.8-4) | 7-10x | a 16-token tile with one real token; the engine's A broadcast + 16 lanes cannot use weights faster than 4 B/cycle | **hardware (later)**, or **software: batching 16 positions** (speculative decoding or multi-sequence decode uses the engine at M = 16 at the same cost) | see below | later; not a run-2 blocker |
| M3b | Decode: issue as compiled (1.46 B/cycle/core) under the memory path | 20x | M1's instruction stream at M = 1 | software (M1) | - | as M1 |
| M4 | K32 fragment (32-byte rows, `MTX_ABUF`-eligible) against the line tensor engine (whole-line rows) | line engine unusable for the weight fragment: 58 vs 2-3 B/cycle available | `MTX_ABUF` holds 8 registers, so FP8 K <= 32 | **hardware, optional**: widen the A buffer to 16 registers so K64 (64-byte rows) runs buffered; or **software**: store the weights pre-packed as contiguous 512-byte fragments at load time (a layout transform) so the box is line-shaped | A buffer: +512 B of registers per core (about 4 k FF, estimate, no screen); layout: loader change | optional for run 2, recommended for the next image if M1's fix shows the word walk binding |
| M5 | `MTX_VRF_REG 1` with a buffer-less K64 command (2 cycles per row) | 2.2x below K32 | registered VRF response; no A buffer at K64 | compiler (keep K32) or M4's wider buffer | - | run 2: keep K32 (condition in §0) |
| M6 | DP4 command structure (45 of 64 MAC/cycle) | 1.4x | B-load bubble (6 %), A fill, drain, serial commands | not > 2x; a pipelined next-command start would recover about 15 % | RTL control, small | later |
| M7 | SFU 1 element/cycle against exp and silu needs | not a mismatch (1-3 % of cycles) | - | - | - | no action |
| M8 | Attention matmul (unknown lowering) | up to 20 % of prefill at T = 2048 when GEMM is fast | not examined | software (check the lowering) | - | check before judging |

**Hardware options for the decode mismatch (M3), none recommended for run 2:**
1. A short-M sweep in `tnca_matrix` (the 16-row loop bound as a field), with the weights as the B operand (`W^T`, 16 lanes of output) and the activation vector as A. One B register (64 B = 4 k x 16 outputs) per two cycles (B read + one row) would consume 32 B/cycle/core, at the memory path. It needs the accumulate hazard solved (the pipeline is 5 cycles deep against one row per two cycles, so partial sums would rotate over several accumulator rows), an ISA modifier, author approval and compiler support. A design idea, not evaluated; rough cost under 1 k LUT (estimate).
2. A DP4 form in the vector FPU (16 lanes x 4 products against a packed weight vector): the lane delta measured by G2 (+633 LUT, +6 DSP per lane at CUTS 1110, `fma_dslx/README.md` §4) times 16 lanes is about +10 k LUT and +96 DSP per core (estimate, not screened), on an image whose SLR0 is 96.7 % packed. A packed-weight layout is also required.
3. The no-hardware route is M3's batching in software, which is not a decode-latency fix for a single chat stream.

## 8. Recommendation

For run 2: **no hardware change for balance.** Link the candidate as prepared. Confirm the two image/compiler facts of §0 (`MTX_ABUF 1`, K32 emission) and do not read the DP4 lane's benefit off the isolated command (3.24x): until M1 is fixed the realised gain is below 1.7x, and the decode gain is the same.

The effort that matches the datapath is the software list under M1. For the image after run 2, in order of value per cost: (a) the A buffer to 16 registers (M4) if the word walk binds weights supply after M1; (b) a short-M / GEMV mode (M3, option 1) if single-stream decode matters; (c) nothing for the SFU, the VRF ports or the memory path.

## 9. Not measured, and limits

- No run-2 board or TLM execution of the Hub FP8 kernel exists in the repository. The DP lane's TLM profile (`core-vrf-sync-v5-x8-g2c4-dp`) was exercised on small compiled cases only.
- The assembly is the `core-vrf-sync-v5` compile; the X8 profile (`t.ld.async.b.t` loads of the weights, run 1: 53.0 M against 60.5 M TLM cycles for 16 x 9216 x 2560) has a different instruction mix that is not recorded here.
- IPC 1.0, overlap, and the software projection are assumptions; §6.3 brackets them.
- The 29 and 38 B/cycle/core ceilings are from the task brief and I found no recorded source for them; the alternative ceilings (11.2, 8) are from `G4_DESIGN.md`. None of them changes a conclusion: prefill is not memory-bound under any of them.
- Qwen3.5's gated-delta-net recurrence, vision tower, Q/K norms and RoPE are not costed.
- The model does not include host time (563 launches per decode token on run 1: 23 s of 64 s wall).

## Reproduction

```sh
python3 tools/r41/balance_model.py --print                       # tables and checks on stdout
python3 tools/r41/balance_model.py --out evidence/r41/hbm_bw/balance/balance_model.json \
    --update-md evidence/r41/hbm_bw/BALANCE.md                    # regenerate the tables above
```

Inputs read: `evidence/r41/fp8/kernels/fp8-kernels-evidence.tgz` (assembly, TLM report), `evidence/r41/hbm_bw/mma_dp/{chip_validation,cases_summary}.json`, `evidence/r41/chat/board-51aef19-summary.json`, `evidence/r41/fp8/qwen/{vl,q35}/plan.json`. The shape and layer-count lists in the script are checked against `plan.json` (relative difference 0.0).
