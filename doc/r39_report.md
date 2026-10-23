# R39 Report: Standard-Path Performance

Version: 2026-10-10. Sections 1–7 correspond to R39-11 (v4 final, image u280_r39e) in [`r39_tasks_and_acceptance.md`](r39_tasks_and_acceptance.md). Section 8 covers R39-12/13 (Mode A, v5, medium tier), with final image u280_r39h; the final status of R39 is given in §8.5.

All figures in this round are derived from the standard path: `torch.compile` → Inductor (tensor descriptors enabled) → Triton → tnca → U280, or device-level RTL simulation matching the board cycle-for-cycle point by point. Figures from reference paths (descriptors disabled, `compile_aot`) serve only for comparison and do not enter acceptance.

## 1. Conclusion

| Acceptance Item (R39-11) | Requirement | Result |
|---|---|---|
| tiny + small numerics | All pass | 52/52 |
| S_geo vs. R39-00 baseline | ≥ 3.0× | **4.315×** |
| Device time E vs. baseline | ≥ 3× | **3.55×** (0.00725 → 0.02576) |
| No points slower than baseline | — | Slowest is embedding tiny, 1.974× |
| 100 MHz timing | Closure | u280_r39d: post-routing ap_clk WNS +0.087 ns, measured 100.017 / 99.992 MHz; final u280_r39e: WNS +0.053 ns, measured 100.003 / 99.990 MHz |
| B4 cycle-exact point by point | 27/27 | u280_r39e: full sequential run 26/27; attention_decode board 10252, sim 10251. Reruns twice in isolation or immediately following argmax both yield 10251 on board (§4) |
| Leave-one-out ≥ 3 % each | — | 6 items qualified; 4 contexts, RCTX_SWITCH, FE_NOBUBBLE, VRF_BYPASS, FPU_OVERLAP < 3 %, downgraded to "no significant contribution" (§3) |

## 2. Cumulative Table (On-Board, tiny + small, 52 Points, Relative to Frozen Baseline)

| Step | Image / Software | S_geo | E | Total Cycles |
|---|---|---|---|---|
| R39-00 baseline | u280_r39b, all switches off | 1.000 | 0.00725 | 1.000 |
| + R39-01 instruction scheduling | Measured alone on nd100c | 1.158 (vs. same image without schedule) | — | — |
| + R39-02a code shape | u280_r39b, software fully enabled | 2.448 | 2.13× (supported-only) | 0.381× |
| + R39-06/07/09 hardware | u280_r39c (v2) | 2.840 | 2.57× | 0.334× |
| + R39-05/08/10 & sparse TensorMap | u280_r39d (v4) | **4.315** | **3.55×** | **0.224×** |

The single step from v2 to v4 on-board yields S_geo 1.519 and total cycles 0.669×.

Stall breakdown (tiny, baseline → final):

| | Baseline | Final |
|---|---|---|
| Issue (IPC) | 0.277 | 0.425 |
| Hazard | 32.1 % | 16.4 % |
| VRF | 11.2 % | 4.0 % |
| Frontend | 13.9 % | 4.3 % |
| LSU | 12.6 % | 22.0 % (absolute cycles 0.44×) |

## 3. Leave-One-Out (RTL Simulation, tiny, Relative to Final)

Simulation matches the board cycle-for-cycle point by point (v2: 27/27; v4 see §4), so the figures below represent on-board performance. Each row disables only one item while keeping all others enabled.

| Disabled Item | S_geo Slowdown | Suite Cycles | Worst Single Point when Enabled |
|---|---|---|---|
| R39-02a code shape | 2.378× | +148.4 % | No slowdown |
| R39-05 role-SPMD | 1.314× | +26.3 % | 0.995 |
| sparse TensorMap | 1.037× | +5.6 % | No slowdown |
| R39-01 instruction scheduling | 1.052× | +3.0 % | 0.977 |
| R39-05 4 contexts (reverting to v3's 2) | 1.076× | +0.5 % | No slowdown |
| R39-06/07 VRF_PREFETCH | 1.147× | +12.2 % | 1.004 |
| R39-08 LSU_INFLIGHT | 1.084× | +10.7 % | 1.008 |
| R39-05 RCTX_SWITCH | 1.056× | +1.3 % | 0.982 |
| R39-09 FE_NOBUBBLE | 1.051× | +2.1 % | 0.883 |
| R39-10 FPU_OVERLAP | 1.035× | +0.8 % | 0.933 |
| R39-06 VRF_BYPASS | 1.032× | +2.4 % | 0.957 |

(Last column: cycle ratio of the worst point when enabling this item versus disabling it; values < 1 indicate that enabling the item slowed down that point.)

- **Work packages with suite cycles ≥ 3 %**: R39-02a, R39-05 role-SPMD, VRF_PREFETCH, LSU_INFLIGHT, sparse TensorMap, R39-01 (exactly 3.0 %).
- **Recorded as "no significant contribution" (suite cycles < 3 %)**: 4 contexts, RCTX_SWITCH, FE_NOBUBBLE, VRF_BYPASS, FPU_OVERLAP.
  - Their geometric mean speedups are all between 3.2% and 7.6%, but high-cycle model points are dominated by GEMM and LSU, which these items do not affect.
  - None of these items have zero impact: the step from v1 to v2 (VRF_BYPASS, VRF_PREFETCH, FE_NOBUBBLE combined) provided S_geo 1.158 on-board. When disabling one of them in the final configuration, the remaining items and multiple contexts already hide most stalls.
- **Timing-sensitive point**: softmax fp16 tiny. Disabling any of FE_NOBUBBLE, FPU_OVERLAP, VRF_BYPASS, or RCTX_SWITCH makes it 2–12% faster (4489 cycles fully enabled vs. 3965 cycles with FE_NOBUBBLE disabled). The cause remains under investigation and is listed as follow-up work (see [`spec/r40/baseline.json`](../spec/r40/baseline.json) B39-OI-3). Even so, it remains 4.36× faster than baseline.

The 4-context item accounts for only +0.5 % suite cycles, qualifying as "no significant contribution" under R39-11 rules: model points are dominated by GEMM templates, and resource-holding kernels use at most 2 contexts (§5). By geometric mean of points, 4 contexts is still 7.6 % faster, entirely from non-resource-holding kernels.

## 4. On-Board Confirmation and B4

- u280_r39d: tiny 27/27, small 25/25; B1 conformance passed.
- B4 (board vs. device-level simulation of identical RTL and software): 24/27 cycle-exact point by point. Differences: add fp16 2391/2395, attention_decode 10252/10251, softmax fp32 1702/1710.
- Cause: The instruction-fetch round-robin pointer and VRF read-arbitration round-robin pointer were not reset on launch. On board, the device ran continuously across points (and ran B1 first), while simulation started each point from reset. With 1 context these pointers do not affect cycle counts; with multiple contexts they alter interleaving order.
- Fix: Reset both pointers to 0 on launch. Verified in simulation after fix: within the same session, regardless of prior runs, the same kernel always takes 369 cycles.
- Final image u280_r39e includes this fix, SHA-256 `6a3e76a4…4b18cf`.
  - On-board: B1 conformance passed; tiny 27/27, small 25/25 numeric pass.
  - B4: add fp16 and softmax fp32 match simulation cycle for cycle.
  - attention_decode takes 10252 cycles in a full sequential run, vs. 10251 in simulation.
- Retesting attention_decode: Run twice in isolation on the same image, or run immediately following the preceding point argmax, both yield 10251 on-board, matching simulation.
  - Thus, this 1-cycle delta is not residual state from the preceding kernel, nor is it stably reproducible.
  - Root cause remains unisolated and is tracked as an open item (see [`spec/r40/baseline.json`](../spec/r40/baseline.json) B39-OI-2).
  - Simulation does not model host-side asynchronous behavior.

## 5. What Was Done (by Work Package)

See [`r39_tasks_and_acceptance.md`](r39_tasks_and_acceptance.md) for details on progress across sections. Key points:

- **R39-01**: List scheduling per basic block using empirical machine model (`tnca_cc.schedule`).
- **R39-02a**: `no_x_dim`; lane-uniform indices kept in scalar registers; build TensorMap once at entry; `acc + dot` in-place accumulation.
- **R39-05**:
  - Mode B: each context loops across programs in a grid row.
  - `RCTX_SWITCH`: stalled context yields to other contexts.
  - `RSLICE`: per-core resources sliced across contexts based on launch slice count.
  - 4 contexts (v4).
  - Compiler selects context count per kernel: resource-holding kernels capped at 2, others use all contexts.
- **R39-06/07/09**: `VRF_BYPASS`, `VRF_PREFETCH`, `FE_NOBUBBLE`.
- **R39-08**: `LSU_INFLIGHT=2`, loads and stores can overlap; sparse TensorMap.
- **R39-10**: `FPU_OVERLAP`.
- **Defects / Bugs**:
  - Bug B4 (distinct from acceptance item B4): writeback freeze drops FP/MUL results.
  - B5: VRF read arbitration starves matrix port.
  - Launch did not reset round-robin pointers.

## 6. What Was Not Done

- R39-02b wide blocks (multi-register 1-D tiles): importer explicitly rejects 64-element blocks.
- R39-03 software pipelining, R39-04 bank-aware allocation.
- R39-09 branch prediction: taken branch penalty accounts for only ~0.4%, discarded based on data.
- R39-10 matrix unit widening: MAC array remains 1 row per cycle.
  - On v4, FP16 K16 MMA measured 551 cycles, not the earlier estimate of ~256 cycles: each MAC waited for a VRF read of an A element.
  - v5's `MTX_ABUF` reduced it to 311 cycles (§8).
- Hardware export of per-context issued counters: simulation uses `+ctxtrace`.

## 7. Overhead

| | v2 (u280_r39c) | v4 (u280_r39d) | v5 (u280_r39f) | v5 Final (u280_r39h) |
|---|---|---|---|---|
| CLB LUT (incl. platform shell) | 314 851 | 333 236 (+5.8 %) | 336 181 (+0.9 %) | 336 334 (+0.05 %) |
| Registers | 216 965 | 226 423 (+4.4 %) | 227 541 (+0.5 %) | 227 496 |
| BRAM tiles | 351 | 367 | 367 | 367 |
| Routed WNS (100 MHz) | +0.073 ns | +0.087 ns | +0.101 ns | +0.088 ns |

v5's RPAIR and transposed transfers consume negligible resources; so do `MTX_ABUF` (8 A-element buffer) and `RCTX_SWITCH=2`.

OOC synthesis (core only `tnca_fpga_top`, 10 ns): v4 vs. v2 adds +10 % LUTs, +19 % registers, worst path identical (FPU FMA operand select).

## 8. R39-12 and R39-13: Mode A, v5, and Medium Tier

### 8.1 What Was Done

- **Mode A** (`rctx_roles`, off by default): runs the three upstream warp-specialization passes on importer IR (`tnca_cc.partition.specialize_loop`):
  1. PartitionScheduling: partitioning.
  2. aref: 2-stage window ring, each stage with a full/empty barrier pair.
  3. PartitionLoops: separate loop per role.

  A program is split into producer (transfer) and consumer (MMA and epilogue), running on a Role Context pair (`RPAIR`); pairs then execute grid rows via Mode B.
- **v5 Hardware** (profile core-vrf-sync-v5):
  - `RPAIR`.
  - Precise `mb.arrive` under RPAIR.
  - Transposed transfer `HAS_TENSOR_XPOSE`.
  - `MTX_ABUF`: MMA buffers A slices beforehand, lowering FP16 K16 from 551 to 311 cycles.
  - `RCTX_SWITCH=2`: contexts waiting for matrix unit or transfer engine also yield.
- **Compiler**:
  - Linear B (W transpose) uses single transposed transfer instead of 16 gathers plus scratch reordering.
  - Masked contiguous rows load/store one predicated memory access per row.
  - Single-user tiles deferred to row-by-row compute at use site; GEMM epilogue no longer routes through scratch and two accumulators.
  - Zero-initialized accumulators use single `m.zero`.
- **Defects**:
  - B6: software-constructed barrier tokens in `mb.wait`/`mb.test` not rewritten by slice (remnant from R39-05).
  - Mode A bug: bmm template carries pointer tile through scratch across loop iterations; pre-loop write was initially placed only in consumer, leading to incorrect results.
    - Pre-loop writes are now duplicated in both roles.
    - Loops where one role might read memory written by another role are not partitioned.
- **Medium tier** (R39-13): 25 points, each kernel's parameters fit within 384 KiB heap, scale 2–8× small.

### 8.2 On-Board Results (Image u280_r39f: v5 without MTX_ABUF and RCTX_SWITCH=2)

- Timing and clocks: 100 MHz, routed ap_clk WNS +0.101 ns; measured 100.002 / 99.947 MHz; xclbin sha256 `7631515f9cde…`.
- B1 passed; tiny 27/27, small 25/25, medium 25/25.
- **B4: 27/27 cycle-exact point by point.**

| Comparison (On-Board, Mode A off) | tiny | small | medium | Combined 77 Points |
|---|---|---|---|---|
| v4 final / frozen baseline | 4.304 | 4.325 | 4.309 | 4.313 |
| v5 (r39f) / v4 final | 1.193 | 1.195 | 1.173 | **1.187** (total cycles 0.745×) |
| v5 (r39f) / frozen baseline | 5.135 | 5.166 | 5.056 | **5.119** |

No point slower than v4 final.

### 8.3 Mode A Ablation

- **On-board** (r39f, `rctx_roles` on vs. off): 77 points all correct, S_geo **0.997**.
  - Single-program matmul tiny is 1.084× faster.
  - GEMMs already hidden by Mode B are slower, up to 5 %: mlp 0.956, matmul small 0.953, linear tiny 0.955.
- **Simulation** (tiny, v5 final software and hardware): S_geo **1.006**.
  - Faster: gemv 1.112, matmul 1.072, llama_decode 1.047.
  - Slower: mlp 0.966, linear 0.980, gpt2_block 0.988.
- **Conclusion**: Under R39-11 rules, Mode A is classified as "no significant contribution", **off by default**.
  - Cause 1: These GEMMs are matrix-unit bound. Mode B already allows transfers and MMA of two programs to hide each other.
  - Cause 2: Mode A causes both roles to compute addresses redundantly, consuming issue slots on a single-issue core.
  - Mode A provides gains only when program count is small (1 per grid row) or producer is heavy (gemv rearranging A rows).
  - Retained as a fully functional, tested option.

### 8.4 Final Image u280_r39h (v5 with MTX_ABUF and RCTX_SWITCH=2)

- Timing and clocks: 100 MHz, routed ap_clk WNS +0.088 ns; measured 100.022 / 99.989 MHz; xclbin sha256 `c56d492f6a33…44ea7d`.
- Software: `be87779` (source tree on board matches file for file).
- B1 conformance passed. Run with Mode A both off and on: tiny 27/27, small 25/25, medium 25/25 numeric pass.

| Comparison (On-Board, Mode A off) | tiny | small | medium | Combined 77 Points |
|---|---|---|---|---|
| v5 final / frozen baseline | 5.538 | 5.617 | 5.491 | **5.548** (total cycles 0.161×) |
| v5 final / v4 final | 1.287 | 1.299 | 1.274 | **1.287** (total cycles 0.679×) |
| v5 final / v5 (r39f) | 1.078 | 1.087 | 1.086 | 1.084 (total cycles 0.911×) |

- No point slower than v4 final; lowest is non-matrix kernels (add, argmax, etc.) at 1.000.
- GEMM-family 27 points (linear, gemv, matmul, bmm, mlp, gpt2_block, llama_*) vs. v4 final achieve S_geo **1.880**; remaining 50 points achieve 1.048.
  - linear 3.01–3.64, mlp 3.41–3.48, gemv 2.71–2.99, gpt2_block 1.84–2.23, llama_prefill 1.62–1.67, llama_decode 1.17–1.70, fp16 matmul 1.39–1.70.
- Compared to r39f, `MTX_ABUF` and `RCTX_SWITCH=2` accelerate fp16 matmul and linear by an additional 1.51–1.70×.
  - fp32 matmul small and medium are 1.3% and 1.4% slower (0.987, 0.986), conv2d fp32 is 0.7% slower.
  - All are under 2%, specific cause not isolated (see [`spec/r40/baseline.json`](../spec/r40/baseline.json) B39-OI-4).

**Mode A (with `ring_estimate` gating) ablation**, on-board, `rctx_roles` on vs. off, 77 points:

- All numerically correct; S_geo **1.0045** (tiny 1.006, small 1.004, medium 1.003), 0 regressions; suite cycles −0.37 %.
- Gating specializes only loops where producer still has loads/stores. All 9 altered points speed up:
  - gemv 1.064–1.112 (A row rearrangement).
  - llama_decode 1.008–1.047.
  - llama_prefill 1.000–1.009.
- Remaining 68 points unspecialized, cycle-exact with Mode A off; including mlp, matmul, linear which previously regressed on r39f without gating.
- GEMM-family 27 points S_geo 1.013.
- **Conclusion**: Suite cycle delta is well below R39-11's 3 % threshold; retained as "no significant contribution", **off by default**.
  - Selection rule: `rctx_roles=1` splits only when `ring_estimate` predicts a gain and neither role writes memory read by the other; `2` skips estimation (for testing).
  - If considered for default on in R40, it must be re-ablated under R40 protocols.

**B4** (board vs. device-level simulation of identical RTL and software, tiny, Mode A off):

- Full sequential run: 26/27 cycle-exact. llama_decode board 545164 vs. simulation 545162.
- Retesting on same image:
  - Alone twice: 545162, matching simulation.
  - Immediately following preceding suite point gpt2_block: 545164, reproducible.
- Thus, the 2-cycle delta stems from unreset state left behind by gpt2_block. Further isolation:
  - This state persists across processes and across kernels. After gpt2_block, running llama_decode, then add, then llama_decode in isolation still yields 545164.
  - Mismatch isolated to a single kernel: `triton_poi_fused__unsafe_view_neg_split_transpose_view_3` (grid 2, once per layer, twice total), each taking 161 → 162 cycles. Remaining 40 launches are cycle-exact.
  - Not memory adjacent to buffers: adding 4 KiB guard margins filled with 0xFF and 0x00 on both sides still yields 545164, with bit-identical results.
  - Simulation cannot reproduce: running gpt2_block followed by llama_decode twice within the same simulation session yields 545162 every time.
    - However, buffer addresses in this simulation session differ from those of a fresh process on board.
    - On board, each point opens a new Session as well.
- Conclusion: An unreset hardware state at launch that affects only multi-context interleaving, analogous to attention_decode in §4. Numerics are unaffected; cycle difference is 0.0004%.
- Tracked as open item, handed off to R40-00: frozen into B39 as open item (`B39-OI-1`), to be investigated before R40 HBM images.

### 8.5 Final Status of R39

- Signed-off software and image: `be87779` + u280_r39h (v5). Default switches: `TNCA_INDEX_OPT=1`, `TNCA_RCTX_SPMD=1`, `TNCA_SPARSE_TENSORMAPS=1`, `TNCA_RCTX_ROLES=0`.
- Relative to R39-00 frozen baseline: 77 points S_geo **5.548**, total cycles 0.161×; 0 regressions.
- Open items:
  - B4: llama_decode (above, B39-OI-1) and r39e attention_decode (§4, B39-OI-2): 1–2 cycles caused by cross-launch residual state.
  - softmax fp16 tiny sensitive to timing switches (§3, B39-OI-3).
  - fp32 matmul small/medium and conv2d fp32 0.7–1.4 % slower on u280_r39h than u280_r39f (B39-OI-4).
- These are frozen into B39 open items for R40-00 ([`spec/r40/baseline.json`](../spec/r40/baseline.json)).
