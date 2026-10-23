# R39: Standard-Path Performance — Removing R38 and Earlier Compromises

Version: 1.2, 2026-10-10. Status: **R39 CLOSED**. Sign-off: software `be87779` + image `u280_r39h` (v5). R39-01, R39-02a, R39-05/06/07/08/09/10, R39-12, R39-13 are all in the final release; Mode A is recorded as "no significant contribution", off by default. Results and open issues: see [`r39_report.md`](r39_report.md) §8.4–8.5.
Baseline: `r39` branch, `74f841c` (r38-dco: U280 100 MHz undivided on-board, ArchBench B4 cycle-exact point by point).
Sign-off name: `R39_STANDARD_PATH_PERF`.

## 0. Scope

### 0.1 Standard path, and reference-only paths

From R39 onwards, **all performance conclusions come only from the standard path**:

```text
torch.compile(model) → Inductor (tensor descriptors enabled) → Triton → tnca Triton backend
  → .tnca → tnca.runtime.Session → device (U280 board; or cycle-exact RTL simulation matching the board point by point)
```

Handwritten Triton kernels through the same frontend (`kernel[grid](...)` or Triton source via `aot.compile_aot`) also belong to the standard path.

The following paths are **reference-only**: they must continue to pass regressions, but their performance figures do not constitute any R39 conclusion, nor do they serve as evidence of speedup for any R39 change:

- TNCA IR builder, e.g. GEMM planner (`plan_gemm`/`build_gemm`);
- Handwritten assembly ([`tb/core/progs/`](../tb/core/progs/));
- TileBench via AOT (R37 447);
- Inductor path with descriptors disabled.

### 0.2 Baseline (measured)

Data source: `74f841c`, U280 100 MHz image ([`evidence/r38/u280_100mhz/`](../evidence/r38/u280_100mhz/)), ArchBench tiny 27 points, `--counters`, profile `core-vrf-sync-v1`. Table below is the sum over all 27 points:

| Metric | Value |
|---|---|
| Device cycles | 15 460 084 |
| IPC (issued / cycles) | **0.279** |
| Register hazard stall (`perf_x_hazard`) | **32.6 %** |
| LSU stall (`perf_x_lsu`) | 11.6 % |
| VRF bank/port stall (`perf_x_vrf`) | 11.3 % |
| Transfer engine / FP / matrix unit stall | 1.3 % / 0.2 % / 0.0 % |
| Unattributed (neither issued nor in any `perf_x_*`, i.e., frontend bubbles and squashes) | **14.0 %** |
| Data port utilization (read / write beats per cycle) | 0.136 / 0.097 |
| Single-point η = L/T | 0.001–0.216, mostly ~0.01 |
| Inductor configuration | `max_block = 16`, **tensor descriptors disabled** |

Two conclusions:

- **Memory is not the bottleneck.** The data port is utilized only ~1/4.
- **The core is starved.** Over 70% of cycles are spent waiting for registers, LSU, VRF ports, or the frontend.

ArchBench itself did not yet run on the standard path: descriptors were disabled by default, and record keys did not distinguish whether they were enabled or disabled.

## 1. Compromise List

Evidence is from codebase review on 2026-10-09, line numbers referenced to `74f841c`. C denotes compiler and runtime, M denotes microarchitecture, B denotes suspected bugs.

### 1.1 Compiler and runtime

| # | Compromise | Evidence | Cost |
|---|---|---|---|
| C1 | **No instruction scheduling**: Issued in program order; target latencies (`fp_result_cycles`, `mul_result_cycles`) are not read by any code | `codegen.py:5-7` "scheduling is program order"; `asm.py:9-11`; `compiler_overview.md:107` "largest known performance gap" | Dependent instructions are back-to-back, primary cause of 32.6 % hazard stalls |
| C2 | **Block width cap 16**: Inductor `STAGE1_MAX_BLOCK = 16`, compiler refuses blocks spanning multiple vector registers | `inductor.py:53`; `frontend.py:1488-1514` | Large fixed overhead amortized per element; matrix unit busy ≤ 0.5 % (`compiler_overview.md:411`) |
| C3 | Reduction one row per program (`XBLOCK = 1`), persistent reduction disabled | `inductor.py:99-112, 186` | Program count and overhead multiplied for reduction-like operators |
| C4 | GEMM/conv has only one 16×16×16 configuration; MMA constantly uses K16, K32/K64 modes never used | `inductor.py:119-148`; `frontend.py:2787` | GEMM cannot tile for reuse, matrix unit idle |
| C5 | **No asynchronous overlap**: Each `t.ld.async`/`t.st.async`/MMA waits immediately after issue; `num_stages`/`num_warps` fixed to 1, no TTGIR passes run | `frontend.py:2569-2587, 2785-2792`; `triton_backend/compiler.py:43-45` | DMA, MMA, and compute serialized |
| C6 | K-loop rebuilds TensorMap on every iteration (16 `sstore`s plus `td.load`, ~96 cycles), accumulator copied ~3 times before and after MMA | `frontend.py:2389-2399, 2776-2783, 595-608` | GEMM inner loop overhead far exceeds MMA itself |
| C7 | Descriptor loads mostly bypass transfer engine: host descriptor never used, single-row box routes to LSU, box ≤ 16 elements routes to gather | `descriptors.py:296-378` | Standard-path data movement falls back to slowest channels |
| C8 | FP32 `tl.dot` does not enter matrix unit (per-k `vextract` + `splat` + `vfma`); computed f16 operands repackaged via memory (`v.pack` reserved unimplemented) | `frontend.py:2807-2834, 2271-2315`; `r37_isa_gaps.md:123-130` | FP32 GEMM and fused GEMM take slow paths |
| C9 | Register allocation is linear scan, bank-unaware, cannot spill; windows, accumulators, barriers statically allocated at use points and never freed for entire kernel; tiles spill prematurely to scratch with `fence` | `asm.py:49-58`; `codegen.py:121-147`; `frontend.py:3063-3070` | 11.3 % VRF stalls; register pressure forces smaller blocks |
| C10 | No loop unrolling, LICM, or FMA contraction | `frontend.py:569-625`; `upstream/passes.py:11-13`; `compiler.py:58-61` | Loop overhead and dependency chains cannot be amortized |
| C11 | **Only 1 role context used**; grid executed program by program serially, each program requiring one host LAUNCH | `ir.py:692`; `aot.py:828-829`; `runtime.py:736-750` (`role_mask=1`) | Hardware second context idle; 4-context latency hiding from original design draft completely absent |
| C12 | Parameters uploaded and downloaded in bulk on every launch, no device-resident tensors | `triton_backend/driver.py:9-16`; `architecture_overview.md:388` | Wall-clock dominated by host round-trips (does not affect device cycles, affects end-to-end) |

### 1.2 Microarchitecture

| # | Compromise | Evidence | Cost |
|---|---|---|---|
| M1 | **Only 2 role contexts and no independent instruction queues**. Design intended 4: R0 address, prefetch and transfer-in; R1 matrix; R2 reduction and epilogue; R3 transfer-out; each context with its own queue and ready scheduling. Current implementation freezes the entire frontend on stall of one context | `tnca_core.v:55, 320-332, 1689`; [`design/tnca_v1_architecture_draft.md`](design/tnca_v1_architecture_draft.md):91-122, 385, 445-466 | Cannot fill stalls with instructions from other contexts |
| M2 | **Banked VRF has no bypass** (`AVAIL_V` constantly false): single-cycle vector producer to consumer has ~3 bubbles, FP ~5; even unrelated instructions stall 2 cycles in R stage | `tnca_core.v:866-872, 617-672`; `r3_status.md:55` (+16 % / +13 %) | Hardware half of hazard stalls |
| M3 | VRF has only 4 banks, each 1 read 1 write, 5 requestors; already written operand collector (4 entries) not wired into core; matrix and transfer engine read ports contest banks with operand fetch | `tnca_core.v:781-786`; `tnca_operand_collector.v` only instantiated in tb; [`spec/r3v2_contract.json:46-50`](../spec/r3v2_contract.json) claims 4×3 | 11.3 % VRF stalls |
| M4 | **LSU permits only one instruction in flight at any time** (`req_ready = (state == S_IDLE)`). Fixed overhead S_SETUP, S_DRAIN plus one registration handshake, back-to-back memory access ~9 cycles; store also waits for response | `tnca_lsu.v:30-33, 151, 422-466`; [`conformance.md`](conformance.md#d2--one-memory-instruction-in-flight) D2 still open | 11.6 % LSU stalls |
| M5 | Data port 8 B/cycle, shared by LSU and transfer engine, engine strictly prioritized (`pick_b = b_req_valid`) | `tnca_mem_mux.v:69`; `tnca_core.v:81-82` | LSU starved during transfers |
| M6 | Single-issue, writeback port one result per cycle, 8-deep reservation; `wb_steal`, `stall_rsv` freeze frontend | `tnca_core.v:1517-1523, 1676` | Mutual blocking when results cluster |
| M7 | Frontend: synchronous fetch, one-entry fetch buffer, predict-not-taken, resolved at X stage, taken branch ~4 bubbles | `tnca_core.v:1722-1735`; [`conformance.md`](conformance.md#d1--instruction-fetch-was-a-single-cycle-asynchronous-read) D1 (closed, ~+8 %) | Major part of 14 % unattributed cycles; currently lacks dedicated counter |
| M8 | Matrix unit k is outer loop, each k first loads B then computes 16 cycles, K16 ~272 cycles; banked VRF port 3 read adds 1 cycle per access (measured +42–46 %); only one command at any time | `tnca_matrix.v:391-427`; `r3_status.md:55` | Even if compiler feeds it, unit throughput is halved |
| M9 | Transfer engine accepts only one command at any time, `td.load`, `w.def` also stall; `t.prefetch` discarded; descriptor reads serialized (32 words) | `tnca_tensor.v:288, 605-606`; `tnca_core.v:1335-1340, 1638-1640` | Descriptor pipelining cannot overlap |
| M10 | FP reduction serialized (4 stages × (FP_LAT+1)), regular FP instructions stall during reduction; SFU one lane per cycle | `tnca_vector_fpu.v:441-475`; `tnca_sfu.v:11-18`; `tnca_core.v:1601, 1662` | Softmax, norm, activation operators slow |

### 1.3 Suspected bugs (to be verified prior to any performance work)

| # | Description | Evidence |
|---|---|---|
| B1 | On the cycle when banked VRF rejects a write (`vrf_wb_freeze`), the scoreboard may still be cleared, causing subsequent reads to fetch stale data | `tnca_core.v:2084-2088` lacks freeze condition (inferred from code review, unverified) |
| B2 | VRF arbitration `rr_ptr` is only 2 bits wide for 5 requestors; port 4 (transfer engine store read) is never prioritized first | `tnca_vrf_banked.v:204`; `tnca_core.v:786` |
| B3 | Documentation outdated: `architecture_overview.md:406` still claims on-board image has only rank 1–2; `r3_counters.json` priority list lacks vrf, vrfwb | Same as left |

## 2. Common Rules

- **R1 Standard path.** Performance figures come only from the §0.1 standard path, generated by ArchBench; reference-path numbers in reports must appear in separate tables marked `reference-path`.
- **R2 Every change has a switch; when switched off, cycle-exact to previous.**
  - Compiler changes: compile option or Inductor option, e.g. `schedule=0/1`, `max_block`, `num_stages`.
  - Hardware changes: RTL parameter plus profile capability bit, e.g. `VRF_BYPASS`, `NUM_RCTX`, `LSU_INFLIGHT`.
  - Acceptance requirement: when switch is off, ArchBench tiny device cycles per point must be **identical point-by-point** with the previous frozen baseline; R2 frozen baseline 149/149 identical.
  - Every cycle delta when switch is on must be explained by a `design_change` record ([`spec/r38/design_change.schema.json`](../spec/r38/design_change.schema.json)).
- **R3 Ablation protocol.** Speedup of each work package given only by an A/B ablation: [`tools/r39/ablate.py`](../tools/r39/ablate.py), suite is ArchBench tiny + small on standard path. Report contents:
  - Per-point Δcycles, η;
  - Suite E (balanced weights);
  - Geometric mean speedup S_geo;
  - Shift in stall attributions (each `perf_x_*` and frontend bubble share);
  - Correctness.

  Passing conditions:
  - Meet the stated threshold of the work package;
  - Any point slower than baseline by > 2 % must have per-point explanation, otherwise rejected;
  - 0 numeric failures.
- **R4 Measurement vehicle.**
  - Compiler and runtime changes measured on U280 (tiny ~6 min, small ~8 min).
  - Hardware changes first measured on RTL simulation for tiny (`fpga-u50-sim`, deterministic, cycle-exact with board point by point), then confirmed on board with final image.
  - Every board image must pass B4 (board vs identical RTL simulation cycle-exact point by point) before its numbers can be used.
- **R5 Clock and resources.**
  - Hardware changes must not break timing closure at default board frequency (100 MHz undivided); estimate first with OOC synthesis, finalized by link result.
  - Each hardware change reports delta in CLB LUTs, FFs, BRAMs, DSPs.
  - If a change requires lowering frequency, report both "iso-frequency speedup" and "seconds-level speedup adjusted for actual frequency".
- **R6 Correctness gate (on every merge).**
  - Reference path: `make r37` 447/447; core, core-vrf, core-slowmem, core-ooo, top, nuca; `r38-baseline-check`.
  - Standard path: torch 24/24 (both descriptor on and off modes); tdesc 10/10; ArchBench numeric all passing; on-board B1 conformance.
- **R7 Frozen baselines.** R39-00 freezes standard path baseline. Each subsequent accepted work package updates frozen baseline once with amendment citing which package it comes from.
- **R8 Inherited from R38.** null is not 0; failures remain in reports; generated artifacts written to `sim/build/r39/`, not checked into Git.

## 3. Work Packages and Acceptance

Numbered `R39-xx`. Acceptance tests are executable commands returning non-zero on failure. Thresholds are **lower bounds**; basis for estimates in parentheses.

### R39-00: Standard-path default, ablation tool, bug verification, baseline freeze

Deliverables:

1. **Standard path default**:
   - ArchBench defaults to tensor descriptors enabled: `configure(tensor_descriptors=True)`, disabled via `--no-tensor-descriptors`.
   - `platform.inductor` records Inductor config completely, with explicit `true`/`false`.
   - Record keys (`report.load`) include Inductor configuration hash and RTL/profile configuration hash, preventing different configurations from overwriting each other.
2. **Ablation tool**: [`tools/r39/ablate.py`](../tools/r39/ablate.py): takes two ArchBench JSONL files (or runs each once under given switches), outputs R3-specified per-point table, E, S_geo, stall attribution differences, and regression list; exits non-zero under `--gate` on threshold violation.
3. **Frontend bubble counter**: New counter `perf_x_frontend`, attributing cycles that neither issued nor stalled to fetch-empty, squash, and branch redirect, ensuring `issued + Σ perf_x_* = cycles` holds exactly.
4. **Suspected bugs**: B1 reproduced with targeted test (or disproven) before fixing; B2 pointer width fixed and 5-port fairness tested; B3 documentation updated.
5. **Frozen baseline**: Run standard-path ArchBench tiny + small on U280 100 MHz image (rebuilt with B1/B2 fixes and new counter), frozen as [`spec/r39/baseline_archbench.json`](../spec/r39/baseline_archbench.json) (per-point cycles, η, counters, configuration hash).

Acceptance tests:

1. [`tools/r39/ablate_tests.py`](../tools/r39/ablate_tests.py):
   - Artificially adding 5 % cycles to one point must flag regression and fail under `--gate`;
   - Two runs of same profile with different Inductor configurations do not merge;
   - Two identical inputs yield S_geo exactly 1.0.
2. Conservation: Extended `profiler_tests.py` verifies `perf_issued + perf_stall_total + perf_x_frontend == perf_cycles` on each program, run once on core-sim and once on core-vrf-sync-v1.
3. B1: Targeted test `t38_wbfreeze` fails before fix (if reproduced) and passes after; if un-reproduced, test retained and conclusion documented. B2: `t29_vrfbank` extended 5-port test proves every requestor is served.
4. Baseline: tiny + small pass numerics on board; tiny B4 cycle-exact point by point; baseline file records commit, image sha256, configuration hash.
5. Standard-path baseline comparison table against §0.2 (without descriptors) documented in report. Measurement step only, no threshold.

Progress (2026-10-09):

- Deliverables 1–4 completed (`7121da0`, `db59522`).
- **B4 (new, found during RTL review, fixed): banked VRF writeback freeze drops results.** When transfer engine grabs bank, writeback stage freezes and retries, but FP and MUL pipelines are unthrottled; results arriving during freeze cycle are dropped: register retains stale value, scoreboard bit never clears, subsequent reader hangs forever.
  - Reproduction: `t39_freeze_drop`, banked core hangs after 3 `wb_retry`s.
  - Fix: Writeback given priority for bank (`tnca_vrf_banked` comments intended this originally); transfer engine holds write and retries next cycle on rejection (`tnca_tensor vw_ready`), does not report transfer complete while writes are held.
  - Verification: t39 passes; core, core-vrf, core-slowmem, core-ooo 35/35 each, xpose 3/3, count and fairness tests pass. B1 reproduced and fixed by `t38_wbfreeze`; B2 proved by [`tools/r39/vrf_fairness_test.py`](../tools/r39/vrf_fairness_test.py) that 4-port and 5-port configurations serve every requestor; B3 documentation updated.
- **First standard-path full run failed 8 points** (nd100c image, descriptors enabled). Three root causes, all resolved:
  1. **Scalar pointer lowering**: Inductor builds descriptors on parameter slices, written as `tl.make_tensor_descriptor(p + k, ...)`. Importer only handled `p + k` in pointer chains for loads, not as values. Affected gpt2_block, llama_prefill (tiny, small). Fix: compute scalar `tt.addptr` address only when used as value (`frontend._scalar_ptr_value`); pointers used solely by loads emit no extra instructions; artifact identity 447/447 preserved.
  2. **Range tree split across two dimensions**: Inductor matched `y // 4`, `y % 4` to two descriptor dimensions with block shape `[(YBLOCK+3)//4, XBLOCK, Min(4, YBLOCK)]`; kernel required permuting tile to 4D, which importer lacked lowering for, and rows were only 1/4 or 1/2 vector width. Affected conv2d, resnet_block. Fix: `TNCATMACompatibilityChecker` rejects this access pattern, falling back to pointer access in Inductor.
  3. **Numeric errors (llama_decode, llama_prefill)**: Rank-1 descriptor reading weights relabeled via `[16] → [16, 1] → [1, 16]`. `tt.trans` treated it as renaming while tile representation remained "one value per row", causing subsequent operations to read lane 0 across all lanes. Opposite direction `[1, 16] → [16, 1] → [16]` before descriptor store had same issue. Fix: `frontend._relabel` converts representation between "one value per row" and "one row"; columns extracted from the same register restored directly without penalty. New targeted tests `rank1_relabel_in_loop`, `rank1_store_relabel`, tdesc 12/12.
- **Diagnostic knobs**, retained for debugging, off by default:
  - `TNCA_TDESC_BISECT=lo:hi`: Numbers descriptor accesses in Inductor query order, applies descriptors only within range, bisecting to offending access (isolated to access 69).
  - `TNCA_GUARD_BYTES` / `TNCA_GUARD_CHECK`: Adds poisoned margins to buffer boundaries; catches out-of-bounds writes (former) and reads via dual poison values (latter). Ruled out OOB read/write.
- **Counter regression**: After adding `perf_x_frontend`, older images returned refused for counter 20. `Session._snapshot` dropped all counters, leaving R39 branch ArchBench records without counters. Fix: skip only missing counters.
- **Results (nd100c image; official baseline awaited r39b image):**
  - Standard-path tiny + small **52/52 numeric pass**, all with counters.
  - tiny: IPC 0.277. Stall shares: hazard 32.1 %, LSU 12.6 %, VRF 11.2 %, frontend 13.9 %.
  - small: IPC 0.264. Stall shares: hazard 35.3 %, LSU 12.4 %, VRF 10.0 %, frontend 13.3 %.
  - Comparing point by point against §0.2 (descriptors disabled), tiny S_geo is 0.998 (0.925–1.068). At `max_block = 16`, most kernels span only one vector width; descriptors differ little from plain loads until wide blocks in R39-02.
- **Baseline freeze (u280_r39b image, 2026-10-09):**
  - Image: RTL `db59522`, B1/B2/B3 fixed, all R39 switches off; xclbin sha256 `807d4014…48e2`; measured clocks 100.092 / 99.919 MHz.
  - Software: `0b7ef29` tree, `TNCA_SCHEDULE=0 TNCA_INDEX_OPT=0 TNCA_NO_X_DIM=0`, matching `d4ee950` standard path.
  - tiny + small **52/52 numeric pass**. tiny IPC 0.277, hazard 32.1 %, LSU 12.6 %, VRF 11.2 %, frontend 13.9 %; small IPC 0.264, hazard 35.2 %, LSU 12.4 %, VRF 10.0 %, frontend 13.3 %.
  - **B4 cycle-exact point by point: 27/27.** Board cycles match board shell RTL simulation (`fpga-u50-sim` backend, identical software and switches) point by point.
  - Frozen as [`spec/r39/baseline_archbench.json`](../spec/r39/baseline_archbench.json) (image, software, config hash, per-point cycles, η, counters, B4 outcome) and [`spec/r39/baseline_archbench.jsonl`](../spec/r39/baseline_archbench.jsonl) (52 raw records, serving as A-side for `tools/r39/ablate.py`).
- **R39-00 deliverables 1–5 all satisfied; R39-00 closed.**

### R39-01: Instruction scheduling (compiler)

Deliverables:

- Basic-block list scheduler for all standard-path kernels. Latency and resource models loaded from target, shared with cost model:
  - FP, MUL latencies;
  - load-to-use latency;
  - banked VRF read latency and bank conflict costs (`bankcost.py`);
  - Asynchronous MMA, DMA;
  - Writeback port reservation.
- Scheduling constraints: respect effect, barrier, and memory dependencies; operate within register budget (jointly with R39-04).
- Switch: compile option `schedule=0/1`, default 1.

Acceptance:

1. With switch off, ArchBench tiny cycles match R39-00 baseline point by point; R2 frozen baseline 149/149 identical.
2. R6 correctness gates pass fully.
3. Ablation (board, tiny + small):
   - Suite `perf_x_hazard` cycles drop ≥ 30 %;
   - S_geo ≥ **1.15×**;
   - No unexplained regressions.

   (Estimation basis: hazard stalls account for 32.6 %, software reordering expected to resolve over half; hardware bypass belongs to R39-06.)

Progress (2026-10-09):

- **Implementation**:
  - [`compiler/python/tnca_cc/schedule.py`](../compiler/python/tnca_cc/schedule.py): basic-block list scheduler. Dependence rules:
    - Pure operations constrained only by data dependencies;
    - Memory reads can pass pure operations and each other, but never effectful operations;
    - Effects preserve original program order;
    - `vinsert`, `assign` treated as in-place writes;
    - Readers of window aliases do not pass subsequent effect;
    - Labels, branches, exits demarcate block boundaries.
  - Selection heuristic: earliest issue slot first, then critical path, then original program order.
  - Register pressure tracked across vector, scalar, and pred register files based on allocator live intervals (including cross-loop extensions), capped at original peak plus margin; candidate window limited to 48 instructions from program order. Blocks exceeding budget revert to program order.
  - `aot._emit` tries in order: default margin and window → zero margin, window 16 → program order, logging fallback in provenance.
  - Unused parameter block loads dropped during scheduling: written by runtime for current launch, reading does not fault.
- **Machine model measured on hardware**, documented in [`spec/r39_sched_model.json`](../spec/r39_sched_model.json):
  - Core latencies from [`tools/r39/latency_probe.py`](../tools/r39/latency_probe.py) (banked VRF). Integer vector dependency distance 4 cycles, FP 6, SFU 29, scalar ALU 2, scalar MUL 5.
  - **On banked VRF, even independent vector instructions issue at most every 2 cycles** (`vrf` stall 1 cycle; same bank adds another cycle). Hardware issue belonging to R39-07, scheduler cannot eliminate.
  - Memory latencies from [`tools/r39/load_latency.py`](../tools/r39/load_latency.py) on board: scalar load 11 cycles, 64-byte vector load 16–20 cycles; LSU holds single in-flight access.
- **Legality**: [`tools/r39/schedule_tests.py`](../tools/r39/schedule_tests.py) checks reordered blocks via independent rules: 894 compilations, ~10,000 blocks, 0 violations, 0 model estimations degraded.
- **Ablation** (nd100c image, tiny + small, `TNCA_SCHEDULE=0/1`), 52/52 numeric pass:
  - S_geo **1.158** (tiny 1.164, small 1.151) ≥ 1.15 ✓;
  - Suite `perf_x_hazard` cycles drop to **0.633× (−37 %)** ≥ 30 % ✓;
  - No regressions (slowest op.silu/small 1.000, fastest op.rope/tiny 1.520) ✓;
  - Suite total cycles 0.931×. LSU stall cycles rise to 1.27×: compute compressed, single-issue LSU becomes bottleneck (belongs to R39-08).
  - First iteration tracked only vector pressure, GEMM kernels reverted to program order due to scalar register shortage, S_geo was 1.138; passed after fixing scalar tracking.
- **Latent bugs exposed and fixed**: Mask operations received non-predicate operands; codegen printed them with same register number as predicate, reading arbitrary state. Succeeded accidentally under program order, failed under alternative allocation:
  - Inductor `xmask ^ true`: `true` materialized as vector `v.movi -1`; `index_assert_traps` stopped trapping after scheduling.
  - argmax passed scalar 0/1 to `p.xor`.
  - Attention kernel loop-carried mask tile passed vector as condition to `v.sel`.

  Fix: `frontend._as_pred` converts vectors/scalars to predicates; loop-carried masks placed in predicate registers (`p.mov`). Added checks: `Builder.vselect` and codegen for `p.*`, `v.sel`, and execution masks reject non-predicate operands, catching errors at compile time. Reference-path artifacts unaffected (447/447 identical).
- **Defaults**: Standard path (Triton/Inductor plugin) defaults to `schedule=1`; `compile_aot` (reference path) defaults to 0, preserving R2 baseline. `plugin_identity` passes plugin choice explicitly to AOT, ensuring both entry points emit identical code.
- **Findings (handed off to subsequent packages)**: Inductor `[1, R]` reduction kernels contain over 10× more instructions than the computation itself:
  - Index tiles materialized repeatedly in loops via `iota*1 + splat`;
  - Contiguous loads degrade to gather, routed via scratch and fence;
  - Single-row tiles routed through accumulators.

  See R39-02 progress.

### R39-02: Wide blocks and multi-register tiles (compiler + Inductor); R39-02a: Code shape

Deliverables:

- Compiler supports blocks spanning multiple vector registers:
  - 1D blocks ≤ 1024 elements;
  - 2D tiles R×C, C multiple of 16;
  - Reductions with `XBLOCK > 1` and persistent reduction;
  - Masks, tails, and broadcasts on wide tiles.
- Inductor `max_block` relaxed from 16 to configurable (defaulted by register budget, e.g. 128), offering candidate configurations per kernel class.
- GEMM/conv templates provide 32×32, 64×64, composed of multiple MMA fragments.
- Switch: `configure(max_block=16)` restores old behavior.

Acceptance:

1. `max_block=16` matches baseline point by point.
2. torch 24/24 (both modes). New targeted wide block tests: 1D 1000 elements, 2D 37×100, row-width 256 reduction covering tails and masks, numerics match eager.
3. Ablation:
   - pointwise and reduction points S_geo ≥ **1.5×**;
   - Full suite S_geo ≥ **1.3×**.

   (Basis: fixed overhead per program amortized over more elements.)

Progress (2026-10-09): **R39-02a** done first, wide blocks deferred.

Profiling post-R39-01 code revealed greater compromise in code shape itself:
- Static instructions dominated by index and address materialization;
- Inductor `[1, R]` reductions repeatedly materialize index tiles and route via gather/scratch/fence;
- GEMM templates take ~2700 instructions per K step.

R39-02 partitioned into:
- **02a**: Code shape (completed, ablated below);
- **02b**: Wide blocks and program merging (deliverables/acceptance as above).

R39-02a changes gated by two switches; when disabled, artifacts byte-identical (447/447):

- **`TNCA_NO_X_DIM` (Inductor)**: Loop reductions strip unit x-dimension, generating 1D `[R0_BLOCK]` blocks (`tnca.inductor._install_no_x_dim`). Importer added support for:
  - `tensor<1>` to `tensor<N>` broadcast;
  - Loop-carried masks in predicate registers.
- **compile option `index_opt` (`TNCA_INDEX_OPT`)**:
  - Lane-uniform grids on scalars: `64*xindex + r0_index` remains contiguous access.
  - Non-negative and multiple properties derived from `assume(x >= 0)` and constant multiplication: `(u + arange) % M` remains contiguous when u is multiple of block length; matrix template `rm % M` avoids gather.
  - `expand_dims` on contiguous indices treated as affine terms.
  - Scalar division performed on scalar registers: reciprocal multiplication for constant divisors, fast path for divisor 1, replacing 32-round vector restoring division.
  - TensorMap for pointer tiles built once at kernel entry, tile origin passed as coordinates, avoiding 256-byte rebuild every K step.
  - `acc + tl.dot(a, b, 0)` accumulated directly in accumulator; yield does not copy when loop-carried value already in place.
  - `acc.to(f16)` preceding store converted row-by-row on exit, avoiding scratch round-trip.
  - Affine row components evaluated once; identical comparisons deduplicated.
- **Effects (per kernel on simulator)**:
  - RMSNorm 4×64: 14 332 → 4 668 cycles (3.1×).
  - 16×32×16 matmul: 20 539 → 2 602 cycles (7.9×), K loop retains only two transfers and one MMA.
  - Static instructions across 52 points: pointwise −50 %, reduction −38 %, templates −56 %.
- **Ablation** (nd100c image, tiny + small; A is R39-01 default, B adds `TNCA_NO_X_DIM=1 TNCA_INDEX_OPT=1`), 52/52 numeric pass:
  - S_geo **2.114** (tiny 2.129, small 2.098), suite total cycles 0.409×;
  - E: tiny 0.0070 → 0.0126, small 0.0100 → 0.0199;
  - 0 regressions.
- **Pending (prioritized by benefit)**:
  - f32 `tl.dot` (P@V in attention, fp32 conv) uses vector FMA rather than matrix unit;
  - Conv templates retain gather;
  - Wide blocks and program merging (02b).

### R39-03: Asynchronous overlap and software pipelining (compiler)

Deliverables:

- Software pipelining for K loops and attention loops:
  - `num_stages` effective (≥ 2): next block's `t.ld.async` issued before current block computation, wait deferred to use point;
  - MMA issued asynchronously, waiting only when result needed.
- TensorMap constructed once outside loop (LICM), coordinates updated inside.
- Accumulators preserved in-place across iterations without copying.
- Descriptor loading channel selection prioritizes transfer engine:
  - Host descriptors can use engine;
  - Single-row and small boxes selected by cost model among LSU, gather, engine.
- FP32 `tl.dot`: propose path to utilize matrix unit (e.g. TF32 / bf16 decomposition approximations optional, preserving IEEE results by default; document reason if infeasible).
- Switch: `num_stages=1` restores old behavior.

Acceptance:

1. `num_stages=1` matches point by point; tdesc 10/10 and torch 24/24 pass.
2. New counters demonstrate true overlap: cycles where transfer engine busy and issue active coincide, and cycles where matrix unit busy and issue active coincide (non-exclusive, proof only).
3. Ablation:
   - GEMM, attention, linear, model points S_geo ≥ **1.5×**;
   - Matrix unit busy duty cycle ≥ 10 % (baseline ≤ 0.5 %).

### R39-04: Register allocation (compiler)

Deliverables:

- Live-interval reuse of windows, accumulators, barriers, descriptor slots.
- Bank-aware allocation: instruction sources placed in distinct banks where possible (`bankcost.py` connected to allocator).
- Spill to scratch on register exhaustion rather than compile failure.
- Tiles spilled to scratch only when necessary, eliminating redundant `fence`s.
- Switch: `alloc=linear`.

Acceptance:

1. When disabled, matches point by point; correctness gates pass; cases previously rejected for register exhaustion compile and evaluate correctly (targeted tests).
2. Ablation: suite `perf_x_vrf` drops ≥ 40 %, S_geo ≥ **1.05×**.

### R39-05: 4 role contexts and program concurrency (hardware + runtime + compiler)

Deliverables:

- **Hardware**: On-board profile updated to `NUM_RCTX = 4`, each context with 64 vector registers, VRF 16 KiB total implemented in BRAM. Each context has private instruction slot and ready logic: on stall of one context, frontend issues from another ready context, preventing stalls from freezing frontend.
- **Runtime and compiler (Mode B)**:
  - Single LAUNCH starts 4 contexts concurrently with `role_mask = 0xF`;
  - Compiler wraps kernel body in persistent loop: `pid = rctx; pid < grid_size; pid += 4`;
  - Grid executes concurrently on device, eliminating per-program host round-trips;
  - Timing protocol changes from `serial-grid` to `grid-concurrent`: launch accept to all contexts idle.
- **Original Mode A** (program partitioned by roles into producer/consumer) categorized as optional follow-up, omitted from this acceptance round.
- Switch: `NUM_RCTX` parameter 2 or 4; runtime `concurrency=1/2/4`. `concurrency=1` matches old behavior point by point.

Acceptance:

1. New targeted tests:
   - 4 contexts concurrently run distinct programs correctly;
   - Extended stall in one context (e.g. long LSU access) does not block issue in other contexts, verified by per-context issued counters.
2. core, core-vrf, and all regressions pass on `NUM_RCTX = 4`; conformance passes.
3. Ablation (RTL simulation tiny, confirmed on board with final image):
   - Grid points with ≥ 4 programs achieve S_geo ≥ **1.6×**;
   - Full suite S_geo ≥ **1.3×**;
   - Suite IPC ≥ 0.45.
4. R5: 100 MHz timing closure, resource deltas tabulated.

Progress (2026-10-09):

- **Mode B (compiler + runtime)**: `tnca_cc.rctx_spmd`, compile option `rctx_spmd` (`TNCA_RCTX_SPMD`). Enabled by default on standard path after ablation; `compile_aot` defaults off.
  - Kernel body wrapped in loop: context r executes programs r, r + R, ... (R = target context count) up to `num_programs(0)`. `program_id(0)` is loop variable; loop tail is 4 instructions.
  - Parameter loads, constants, other grid dimensions evaluated once outside loop. Hoists scalars only; values mutated via `assign` not hoisted (hoisting initial loop counter caused second program to resume from end of first — caught by ArchBench cumsum, norm, softmax; fixed and tested in `ragged_trip_counts`).
  - Manifest: `execution.role_mask = 2^R − 1`, `program_to_role` one entry per context, `entry_points` one per context. Runtime issues launch per (y, z), starting min(R, gx) contexts; scratch scaled by context count.
- **RSLICE (hardware capability `rctx_slice`)**: Launch role-mask bit 31 requests context-slicing of per-core resources: instruction immediates and barrier token resource IDs ORed with context ID at D→R and unit inputs; context r accesses r-th slice of accumulators, barriers, descriptors, windows.
  - Compiler guarantees IDs within slice size, making OR idempotent: `mb.arm` returned tokens already sliced.
  - Kernels utilizing these resources or scratch wrapped in Mode B only on targets with `rctx_slice`; scratch loads offset by `ROLE_ID * scratch_bytes`.
  - Older images take only low bits of role-mask, ignoring bit 31; compiler avoids emitting slice-dependent code for them.
  - **Slice count determined by launch**: role-mask bits 30:29 specify log2(slices); 0 means one slice per context. On 4-context cores, kernels can use 2 contexts, each receiving half resources.
    - Reason: 4-way slicing on v4 provides 4 accumulators per context; GEMM templates in dominant model points require 6–8, thus refused SPMD. Also, accepted GEMMs ran slower on 4 contexts than on v3's 2 contexts: gpt2_block 785k vs 667k cycles, mlp 468k vs 390k, attention_decode 13.0k vs 11.4k, sharing single matrix unit and transfer engine.
    - Compiler heuristic: select maximum power-of-two context count fitting kernel resources; kernels holding per-core resources (accumulators, descriptors, windows, barriers, scratch) capped at 2 contexts (`TNCA_RCTX_SPMD_SHARED`, default 2).
    - Capping matrix kernels initially left attention reduction kernels slower on v4 than v3 (7141 vs 6155 cycles across three 32-program kernels) due to TensorMap contention over single memory path. Kernels without resource contention utilize all 4 contexts (e.g. add achieves 1.14×).
    - `t39_slice` passes on 4-context bench with both `+slicek=1` (2 contexts, half slice each) and default 4-way slice.
  - TensorMaps built from invariants in entry block constructed once: 32 stores into scratch and snapshot hoisted outside loop. Slicing stores preserves observable ordering since only snapshot reads TensorMap. Measured `tdesc_rows` (5 programs): 0.99× rebuilding TensorMap per program vs 1.62× hoisted; `dot_grid` 1.17× (limited by matrix unit, R39-10).
  - Rejects tokens counting barrier epochs from kernel entry (`barrier_token`); standard path does not emit them.
- **RCTX_SWITCH (hardware)**: When R stage instruction stalls on register hazard and D stage holds instruction from another context, R stage instruction is dropped (having reserved no resources), allowing frontend to advance. Stalled context refetches from `r_pc` and parks until writeback arrives or accumulator completes.
  - Applied only to hazards, not LSU busy: yielding on LSU busy degraded performance 8 % in `t39_switch` because dropped memory accesses take 3 cycles to re-enter R after LSU frees.
  - Accounting: yield cycle counted in `perf_x_hazard`, maintaining `issued + stall + frontend = cycles`; `perf_rctx_switch` logs yield events (simulation readable).
- **4 contexts**: Profile `core-vrf-sync-v4`: `NUM_RCTX = 4`, 64 vector registers per context, 256 total (16 KiB).
- **Profiles**: `core-vrf-sync-v3` = v2 + `RCTX_SWITCH` + `LSU_INFLIGHT=2` + `FPU_OVERLAP` + `RSLICE` (2 contexts); `core-vrf-sync-v4` = v3 + 4 contexts. Simulation validated first, board images linked subsequently. Individual timing parameters toggled via `TNCA_RTL_PARAMS="RCTX_SWITCH=0,..."`.
- **Tests**:
  - `t39_switch` (new): All contexts concurrently execute pointer chasing, MUL chains, SFU chains, and regular FP/reductions during SFU passes; per-role assertions pass across core, core-vrf, r39hw, r39sw, r39sw4.
  - `t39_isolate` (new, acceptance 1 item 2): context 0 executes 24 dependent SFU passes while remaining contexts run compute loops touching neither SFU nor memory; `+ctxtrace` logs exit cycles and retired instructions. With 2 contexts, compute context exits cycle 1392 with RCTX_SWITCH vs cycle 1794 without.
  - `t39_slice` (new, `gen_t39_slice.py`): Contexts execute TensorMap → transfer engine → window → barrier → MMA → `a.vread` using identical names (`d1`, `w1`, `b0`, `b1`, `a1`); results asserted per context; passes on r39sw (2 contexts) and r39sw4 (2/3/4 contexts).
  - [`tools/r39/rctx_spmd_tests.py`](../tools/r39/rctx_spmd_tests.py) (new): odd grids, single program, 2D grids, inner loops, varying trip counts, cumsum, masked softmax; 3 resource-holding kernels accepted on targets with `rctx_slice` and rejected elsewhere.
  - Full R3.7 regression (447 cases, `TNCA_RCTX_SPMD=1` with R39-01/02a switches): **447/447** passing.
  - Tensor descriptor tests (`TNCA_RCTX_SPMD=1`, v3): 12/12 passing.
  - `core-r39sw` (v3 switches): 37 programs × 2 memory models pass.
- **Ablation (v2 RTL simulation tiny, 13 non-resource-holding points, A is v2 SPMD off):**

  | B | S_geo | Hazard Share | FPU Share |
  |---|---|---|---|
  | v2 + SPMD | 1.223 | 16.9 % → 4.2 % | 2.1 % → 5.6 % |
  | v3 (RCTX_SWITCH only) + SPMD | 1.226 | → 6.4 % | → 4.5 % |
  | v3 (RCTX_SWITCH + LSU_INFLIGHT + FPU_OVERLAP) + SPMD | 1.278 | → 8.3 % | → 1.8 % |

  v3 with SPMD off matches v2 cycle-exact point by point (S_geo exactly 1.0000). SFU-dense points (gelu, silu, swiglu) slowed down with RCTX_SWITCH alone because regular FP stalled during SFU passes, causing dropped instructions to incur refetch penalties; `FPU_OVERLAP` eliminates this stall, restoring speedup to 1.29–1.45×.
- **Timing and resources (post Vivado OOC synthesis, full device `tnca_fpga_top`, 10 ns):**

  | Configuration | WNS | Worst Path | CLB LUT | Registers | BRAM | DSP |
  |---|---|---|---|---|---|---|
  | v2 (2 contexts) | 3.177 ns | FPU FMA operand select, 6.805 ns | 195 579 | 53 759 | 167.5 | 171 |
  | v4 (4 contexts + RCTX_SWITCH + LSU_INFLIGHT=2 + FPU_OVERLAP + RSLICE) | 3.139 ns | Same path, 6.843 ns | 215 202 (+10 %) | 64 146 (+19 %) | 167.5 | 171 |

  New logic does not enter critical path. 256 vector registers fit in identical BRAM count. Board image `u280_r39d` (v4, 100 MHz) entered linking.
- **Pending**: v4 full-suite ablation with resource-holding kernels; board confirmation (B4 cycle-exact); hardware export of per-context issued counters (logged in simulation via `+ctxtrace`).

### R39-06: Banked VRF bypass network (hardware)

Deliverables:

- X→R, W→R vector result bypass, allowing dependent instructions to consume results without waiting for register file write.
- FP, MUL latencies forwarded on availability cycle, releasing scoreboard early.
- Eliminate 2-cycle R-stage wait for independent instructions via D-stage read issue or operand prefetch.
- Switch: RTL parameter `VRF_BYPASS = 0/1`.

Acceptance:

1. Targeted test `t39_deplat` on banked file measures bubble cycles between dependent pairs:
   - Single-cycle vector producer to consumer = **0** (baseline ~3);
   - FP producer to consumer = **FP_LAT − 1** (baseline ~5);
   - Test asserts old values when bypass disabled.
2. Full regressions pass; B1 test passes with bypass enabled.
3. Ablation: suite `perf_x_hazard` drops ≥ 40 % (measured post R39-01), S_geo ≥ **1.15×**.
4. R5: 100 MHz closure.

Progress (2026-10-09):

- RTL switches `VRF_BYPASS` (write-first read bypass during writeback) and `VRF_PREFETCH` (D-stage prefetch of vector source when R read port idle) integrated into profile `core-vrf-sync-v2`; board image `u280_r39c` linked.
- Core bench measurements ([`tools/r39/latency_probe.py`](../tools/r39/latency_probe.py)): independent vector instructions 1 cycle/inst (was 2), dependent `v.add` 3 cycles (was 4), `vf` 5 cycles (was 6). Scheduler machine model adds `banked2`.
- `core-r39hw` (v2) 37 programs × 2 memory models pass.
- **Suite ablation (v1 → v2, all three switches, RTL simulation tiny, software R39-01 + R39-02a default):** 25/27 points (2 llama points running), S_geo **1.158**, slowest point 1.016×, 0 regressions.
  - Stall breakdown: VRF 10.1 % → 5.1 % (cycles −56 %, meeting R39-07 metric via `VRF_PREFETCH`); frontend 17.0 % → 10.7 % (cycles −45 %, meeting R39-09 metric via `FE_NOBUBBLE`).
  - Hazard cycles dropped only 4 %, **missing R39-06 −40 % target**: bypass eliminates only writeback-cycle hazard; remaining hazards stem from load, SFU, and reduction latencies, addressed via multi-context in R39-05.
  - Largest gains: cumsum 1.68×, embedding 1.49×, matmul fp32 1.35×, argmax 1.27×.
- **Board confirmation (u280_r39c, v2, 100 MHz, routed WNS +0.073 ns; xclbin sha256 `8d38ca27…5cd4`).** Software on current default (R39-01 schedule, R39-02a code shape enabled; SPMD and sparse TensorMap disabled):
  - B1 conformance passes; tiny 27/27 numeric pass.
  - Relative to frozen baseline ([`spec/r39/baseline_archbench.jsonl`](../spec/r39/baseline_archbench.jsonl), r39b image, all switches off): tiny **S_geo 2.872**, E 0.00614 → 0.01522 (2.48×), total cycles 0.371×, slowest point 1.148×, 0 regressions.
  - small 25/25 pass, S_geo 2.807. tiny + small total 52/52: **S_geo 2.840**, E 0.00725 → 0.01866 (**2.57×**), total cycles 0.334×; smallest gain silu small, 1.086×. Measured clocks 100.038 / 99.967 MHz.
  - Board cycles match v2 device-level RTL simulation point by point (e.g. gpt2_block 899716, swiglu 3392).
  - Built prior to B5 fix, contains that defect; not triggered by 1–2 context executions.

### R39-07: VRF port and bank conflicts (hardware)

Deliverables:

- Wire existing operand collector (4 entries) into core, or expand banks to 8 with added read ports (determined by synthesis cost; report both paths).
- Matrix and transfer engine read ports arbitrated with operand fetch using starvation protection.
- Switches: `VRF_BANKS`, `OPERAND_COLLECTOR`.

Acceptance:

1. Targeted test: 3-operand instruction targeting same bank drops from 3 cycles to ≤ 2; matrix reads no longer halve MMA rate (K16 MMA cycles within 10 % of behavioral file).
2. Regressions pass.
3. Ablation: `perf_x_vrf` drops ≥ 50 %, S_geo ≥ **1.08×**.
4. R5: 100 MHz closure.

Progress (2026-10-10):

- **B5 (new, discovered in R39-05 4-context testing, fixed): banked VRF read arbitration starves matrix port.** On v4 with 3-program dot tile grid, matrix unit hung in MAC state requesting reads each cycle without grant; context slept on barrier, other contexts backed up in R/D stages.
  - Cause: round-robin pointer moved past "last port served in current cycle". D-stage prefetch (port 1, sharing bank with matrix port 3) and port 0 (another bank) requested each cycle; serving 1 and 0 in order 1 2 3 4 0 wrapped pointer back to 1, leaving port 3 permanently behind port 1.
  - Fix: pointer advances past first port served in current cycle, guaranteeing each persistent requestor is served within NUM_PORTS cycles.
  - Test: [`tools/r39/vrf_fairness_test.py`](../tools/r39/vrf_fairness_test.py) adds case 8 (two banks served in same cycle); old arbiter fails (port 3 served 1/24 cycles), new arbiter passes (max wait 3 cycles).
  - Latent since v2 (`VRF_PREFETCH`), unmasked by 4 contexts. u280_r39c (v2) built with old RTL; v4 image u280_r39d rebuilt with fixed RTL; simulation ablations rerun with fix.

### R39-08: Non-blocking LSU and memory ports (hardware)

Deliverables:

- LSU supports multiple in-flight memory instructions (depth ≥ 4), completed via scoreboard writeback; redundant cycles in S_SETUP, S_DRAIN eliminated.
- Posted writes for stores: preserve order and fence semantics; fence waits for all in-flight stores.
- Fair arbitration between LSU and transfer engine.
- Evaluate cost/benefit of widening data port (128-bit) or splitting ports.
- Switches: `LSU_INFLIGHT = 1/4`, `MEM_ARB`.

Acceptance:

1. Targeted tests:
   - Back-to-back independent load interval ≤ 2 cycles (baseline ~9);
   - load-to-use latency ≤ 4 cycles;
   - Fence following store operates correctly;
   - Atomics, faults, `t23_amofault` unchanged;
   - 3 memory models pass.
2. Ablation: `perf_x_lsu` drops ≥ 50 %, S_geo ≥ **1.1×**.
3. R5: 100 MHz closure.

Progress (2026-10-09):

- RTL switch `LSU_INFLIGHT = 2` ([`rtl/core/tnca_lsu_pair.v`](../rtl/core/tnca_lsu_pair.v)): two `tnca_lsu` units share memory port. Second unit accepts next load/store while first drains.
  - Second unit accepted only after first issues its final beat, ensuring earlier requests reach port first. Memory executes requests in arrival order and reorders only responses (board `tnca_fpga_mem` single port; `tnca_mem_model` executes on accept), ensuring overlap matches program order.
  - Atomics require both units idle; `fence` semantic preserved (waits for `req_ready`, i.e. both units idle).
  - Extended from loads to stores; TensorMap store sequences avoid per-store wait for write response (`tdesc_rows` 4565 → 3355 cycles, `dot_grid` 10428 → 8712 cycles in single context).
  - Out-of-order execution across disparate slices (multi-slice NUCA) would require address checks; none present in profiles carrying switch.
  - Top bit of beat tag identifies unit; earliest completion exits first.
- `t39_switch` on v3 bench (1-cycle memory) drops 2266 → 2106 cycles (jointly with FPU_OVERLAP); single-program vector add 79 → 67 cycles.
- `core-r39sw` passes completely. Suite ablation reported with v3 (see R39-05).
- **Software side: Sparse TensorMaps.** compile option `sparse_tensormaps` (`TNCA_SPARSE_TENSORMAPS`). Standard path defaults on after ablation. Runtime uploads zeroed scratch with parameter block, eliminating stores for 25 zero words in rank-2 TensorMaps.
  - Results: tensor descriptor tests 12/12 pass; `tdesc_rows` 4565 → 2510 cycles (SPMD off), 1958 cycles combined with SPMD (2.33× over baseline); `dot_grid` 10428 → 7848 cycles, combined 7201 cycles (1.45×).
  - Suite ablation reported with v4 (see R39-05).
- Deferred: depth 4, posted writes, fair arbitration with transfer engine.

### R39-09: Frontend (hardware)

Deliverables:

- Decompose 14 % frontend cycles by cause using R39-00 `perf_x_frontend`.
- Address findings via:
  - Deeper fetch buffer;
  - Simple branch prediction (backward taken, or BTB);
  - Shortened redirect path for taken branches.
- Switches: `FETCH_DEPTH`, `BPRED`.

Acceptance:

1. Targeted test: tight loop branch penalty ≤ 1 cycle/iteration (baseline ~4).
2. Ablation: `perf_x_frontend` drops ≥ 40 %, S_geo ≥ **1.05×**.
3. R5: 100 MHz closure.

Progress (2026-10-09):

- RTL switch `FE_NOBUBBLE`: decoder consuming buffered word no longer blocks fetch, preventing stall from losing fetch slot. Part of `core-vrf-sync-v2`.
- **Branch prediction abandoned based on data**: in v2 RTL simulation of tiny suite, taken branch redirects totaled only 3570 across 2.49M cycles (~3 cycles each, ~0.4 % of execution time) because hot kernels unroll into straight-line code. Remaining 10.7 % frontend cycles stem from `mb.wait` sleep and post-stall fetch bubbles, mitigated by multi-context (R39-05).

### R39-10: Matrix and reduction unit throughput (hardware, data-driven)

Deliverables:

- Matrix unit: eliminate extra cycle on port 3 of banked register file (joint with R39-07); overlap B loading with MAC; allow subsequent MMA to initiate during drain.
- FP reduction: pipeline tree reduction without stalling regular FP.
- Implemented only if counters post-R39-02/03 show unit is bottleneck; provide ablation if implemented or data if deferred.

Acceptance: implemented sub-items achieve S_geo ≥ 1.2× on target kernels (GEMM or softmax/norm); un-implemented items justified by data.

Progress (2026-10-09):

- RTL switch `FPU_OVERLAP`: SFU receives independent datapath and latches sources on start; only reduction borrows FMA array. Regular vector FP stalls only during reduction folding; SFU pass no longer waits for in-flight FP instructions (previously stalled all contexts for ~22 cycles).
- `t39_switch` adds segment E: regular FP and reduction issued during SFU pass, verified per lane.
- Combined with SPMD, SFU-dense points drop FPU stall share from 4.5 % to 1.8 % (see R39-05).

### R39-11: Overall ablation and sign-off

Deliverables: [`r39_report.md`](r39_report.md), containing:

- **Cumulative table**: incremental S_geo and E as packages enabled sequentially.
- **Leave-one-out table**: marginal contribution of each package disabled individually with all others enabled.
- **Board confirmation**: ArchBench tiny + small on final image, B4 cycle-exact point by point, measured clocks.
- **Overhead**: resource and clock costs, reference path figures (labeled separately).

Acceptance:

1. Final image on U280 satisfies:
   - ArchBench tiny + small numeric pass;
   - S_geo relative to R39-00 baseline ≥ **3.0×**;
   - Device time E ≥ 3× baseline;
   - 0 regressions against baseline.
2. Leave-one-out: disabling any implemented package increases suite cycles ≥ 3 % (or downgraded in report to "no significant contribution").
3. Board 100 MHz timing closure, B4 cycle-exact point by point, measured clock 100 ± 0.5 MHz.
4. All R6 gates pass; every cycle change covered by a `design_change` record.

Progress (2026-10-10, board final):

- **Image u280_r39d**: core-vrf-sync-v4, 4 contexts, switches RCTX_SWITCH, RSLICE (variable slices), LSU_INFLIGHT=2 (load/store overlap), FPU_OVERLAP, v2 switches, and B5 fix. 100 MHz, routed ap_clk WNS +0.087 ns; platform CLB LUT 25.6 %, BRAM 18.2 %. xclbin sha256 `335e48a2…24ce8`, measured clocks 100.017 / 99.992 MHz.
- **Software**: standard-path defaults: R39-01 schedule, R39-02a code shape, R39-05 SPMD, sparse TensorMaps.
- B1 conformance passes; ArchBench tiny 27/27, small 25/25 numeric pass.
- **Relative to frozen baseline (tiny + small, 52 points)**:

  | Suite | S_geo | E | Total Cycles | Slowest Point |
  |---|---|---|---|---|
  | tiny | 4.305 | 0.00614 → 0.02087 (3.40×) | 0.251× | — |
  | small | 4.325 | 0.00884 → 0.03356 (3.80×) | 0.222× | — |
  | **Combined** | **4.315** | **0.00725 → 0.02576 (3.55×)** | **0.224×** | embedding tiny 1.974× |

  `ablate.py --gate --min-speedup 3.0 --max-regress 1.0` passes: S_geo ≥ 3.0, 0 regressions. IPC 0.265 → 0.411.
- Relative to v2 image (`u280_r39c`) (board gains of R39-05/08/10 + sparse TensorMaps): S_geo 1.519, total cycles 0.669×; smallest gain matmul fp32 tiny, 1.095×.
- Stall breakdown (tiny, baseline → final): hazard 32.1 % → 16.4 %, VRF 11.2 % → 4.0 %, frontend 13.9 % → 4.3 %. LSU increases from 12.6 % to 22.0 %, becoming dominant stall (absolute cycles 0.44×).
- **Final image u280_r39e**:
  - Adds round-robin reset on launch.
  - Timing and clocks: 100 MHz, routed ap_clk WNS +0.053 ns; measured clocks 100.003 / 99.990 MHz; xclbin sha256 `6a3e76a4…4b18cf`.
  - On board: B1 passes; tiny 27/27, small 25/25 numeric pass.
  - B4: sequential run 26/27 cycle-exact point by point. Single mismatch is attention_decode: board 10252 vs simulation 10251.
  - Isolated reruns on same image (twice alone, or immediately following preceding argmax) yield 10251 on board. Unstable 1-cycle delta noted as open issue.

### R39-12: Mode A — overlapping memory and compute across roles with partition.py (hardware + compiler + runtime)

Background: Mode B in R39-05 provides program-level concurrency, but intra-program execution remains serialized. Standard-path GEMM inner loop sequentially executes load A, gather B to scratch and permute, construct TensorMap, load B, and MMA. Memory and compute do not overlap.

Deliverables:

- **Compiler**: Automatic context specialization on importer IR (`tnca_cc.partition`: classify → graph → partition → map_to_rctx), splitting loops with MMA into two roles:
  - **producer**: TRANSFER partition: load A and B, gather and permute, TensorMap, `dma_load` into window.
  - **consumer**: MATRIX and EPILOGUE: MMA, accumulator read, writeback.
  - Inter-role communication via 2-stage window ring with full/empty barrier pairs; roles compute epochs locally (sec 7.3) without token passing.
  - Producer prepares step k+1 while consumer computes step k.
  - Loop control duplicated per role with identical trip counts; address math recomputed locally; accumulator initialization and epilogue assigned to consumer.
- **Registers**: Paired roles share 64 logical vector registers: window at top, producer and consumer private registers in disjoint ranges. Codegen allocator supplies ranges per role.
- **Hardware (launch pair mode)**:
  - role-mask bit 28 pairs contexts 2k and 2k+1: identical vbase (from 2k), RSLICE sliced by pair (`rctx >> 1`).
  - Roles within a pair share windows, barriers, descriptors, accumulators, while preserving inter-program isolation.
  - Default behavior preserved when bit cleared.
- **Composition with Mode B**: 2 programs × 2 roles on v4. Pairs execute grid rows persistently.
- **Switches**: compile option `rctx_roles` (`TNCA_RCTX_ROLES`), hardware parameter `RPAIR`.
- **Transposed transfer for B operand (included in package)**:
  - Linear B is transpose of W, column-continuous. Originally required 16 gathers plus scratch copy per K step: 32 stores, fence, 16 loads, TensorMap construction, and DMA transfer.
  - Worth overlapping only when producer is reduced to two transfers per step. R38 S1-07 added transpose write in transfer engine (ISA sec 9E mod[5], `HAS_TENSOR_XPOSE`), but v4 left it disabled.
  - Importer reads tile as transpose; engine transposes on write into window. Restricted to 16-bit tiles used exclusively by `tl.dot`.
- **Precise `mb.arrive` (`RPAIR`)**:
  - `arrive` previously waited for all engines idle; consumer releasing a stage thus waited for producer's next transfers, serializing the roles.
  - Under `RPAIR`, `arrive` waits only for local context matrix and transfer commands (permitted by ISA sec 9A.5).
- **Profile core-vrf-sync-v5** = v4 + `HAS_TENSOR_XPOSE` + `RPAIR`, target `tnca-sim-vrf-v5`, capabilities `tensor_transpose`, `rctx_pair`.

Acceptance:

1. Targeted tests:
   - Core level: 2 roles exchange K steps across window ring correctly;
   - 4 contexts: 2 pairs execute distinct programs concurrently with correct results;
   - Triton GEMM (`dot_grid`, K ≥ 2 steps) numerically correct, with `+ctxtrace` verifying execution overlap.
2. Full core regressions, R3.7, torch tests, `rctx_spmd_tests` pass; `compile_aot` default produces identical artifacts (plugin identity).
3. Ablation (RTL simulation tiny + medium tier): GEMM points (linear, gemv, matmul, bmm, mlp, gpt2_block, llama_*) achieve S_geo ≥ **1.2×** (relative to R39-05 final), full suite no regressions.
4. R5: 100 MHz timing closure, board confirmation, B4 cycle-exact point by point.

Progress (2026-10-10, RTL simulation; board image u280_r39f linking):

- **Hardware** (`f9297a8`):
  - `RPAIR`, precise `mb.arrive` under RPAIR.
  - Bug B6: software-constructed barrier tokens in `mb.wait`/`mb.test` not rewritten by slice.
  - Profile core-vrf-sync-v5.
- **Compiler** (`644482e`):
  - `tnca_cc.partition.specialize_loop` (PartitionScheduling → aref → PartitionLoops on importer IR).
  - `pair` mode in `rctx_spmd`.
  - `codegen.emit_roles`.
  - Transposed transfer for B operand.
  - Masked continuous row epilogue.
  - Deferred single-user tile computation.
- **Targeted tests**:
  - `t39_pair` passes on fast and slow memories.
  - Exposed B6 on out-of-order memory: pair 1 consumer woke prematurely on pair 0 barrier.
  - [`tools/r39/rctx_roles_tests.py`](../tools/r39/rctx_roles_tests.py):
    - v5: 6 multi-step GEMM loops specialized, numerically correct, 1.008–1.132× faster than Mode B alone.
    - Single-step GEMM folded by Triton rejected as expected.
    - v4: all rejected, numerically correct.
- **Regressions**: core-r39sw5 84/84, R3.7 447/447, torch 24/24, plugin identity O0/O1/O2 149/149, rctx_spmd_tests v4/v5 10/10.
- **Ablation (tiny, RTL simulation, relative to v4 final)**:
  - v5 software/hardware package (transposed transfers + epilogue): GEMM S_geo 1.53 (linear 1.99, gemv 2.22, mlp 2.53, gpt2_block 1.89, llama_decode 1.57, llama_prefill 1.44), full suite 1.17, 0 regressions.
  - Adding Mode A on tiny yields no gain: linear 0.955, gemv 0.948, mlp 0.969, llama_decode 0.974, matmul 1.084.
  - Reason: tiny GEMMs have 2–4 K steps; Mode B already overlaps transfer and MMA across programs, whereas Mode A duplicates address calculation across roles on a single-issue core.
  - Mode A value to be measured on medium tier with deeper K loops and on board.
- **Evaluated and rejected**: `ACC_RD2` (second accumulator read port to avoid waiting for other context's MMA).
  - linear 1.065, mlp 1.02, gemv 0.97, geometric mean ~1.01.
  - Reverted to avoid rebuilding image.

Progress (2026-10-10 afternoon, on board):

- **Image u280_r39f** (v5 without MTX_ABUF and RCTX_SWITCH=2):
  - 100 MHz, WNS +0.101 ns, measured clocks 100.002 / 99.947 MHz.
  - B1 passes; tiny 27/27, small 25/25, medium 25/25 correct with Mode A enabled and disabled.
  - **B4 27/27 cycle-exact point by point.**
  - v5 relative to v4 final: 77 points S_geo 1.187 (total cycles 0.745×), 0 regressions. Relative to frozen baseline: 5.119.
- **Mode A ablation**:
  - On-board r39f: 77 points S_geo 0.997 (only matmul tiny faster at 1.084; mlp, matmul small/medium, linear tiny 2–5 % slower).
  - Simulation (tiny, v5 final): 1.006.
  - Classified per R39-11 as "no significant contribution", **disabled by default**.
  - Reason: GEMMs are matrix-unit bound; Mode B already overlaps transfer and MMA across programs; Mode A duplicates address calculation across roles on single-issue core.
  - Mode A retains speedup when program count is small or producer is heavy (gemv); kept as tested option.
- **Acceptance 3 (GEMM S_geo ≥ 1.2× relative to R39-05 final) achieved by full v5 package**:
  - Simulation tiny GEMMs achieve 1.91 (9 points), full suite 1.287, 0 regressions.
  - Gains driven by transposed transfers, epilogue optimizations, `m.zero`, `MTX_ABUF` (MMA 551 → 311 cycles), and `RCTX_SWITCH=2`, **not Mode A**.
- **Mode A correctness fix**:
  - bmm template loop-carried pointer tile via scratch failed on medium tier.
  - Pre-loop writes now duplicated in both roles; loops where one role reads memory written by another role are not partitioned.
  - Covered in `rctx_roles_tests`.
- **Final image u280_r39h** (v5 with `MTX_ABUF` and `RCTX_SWITCH=2`) linking; board results to follow.

Final Status (2026-10-10 evening, on-board final u280_r39h):

- 100 MHz, WNS +0.088 ns, measured clocks 100.022 / 99.989 MHz; LUT 336 334.
- B1 passes; Mode A on/off: tiny 27/27, small 25/25, medium 25/25 pass.
- **Acceptance 3**: v5 final relative to v4 final (R39-05 final):
  - GEMM 27 points (tiny + small + medium) S_geo **1.880**;
  - Full suite 77 points 1.287 (total cycles 0.679×), 0 regressions;
  - Relative to frozen baseline: 5.548.
- **Mode A** (with `ring_estimate` gating): 77 points S_geo 1.0045, 0 regressions, suite cycles −0.37 %.
  - 9 modified points all speed up: gemv 1.06–1.11, llama_decode 1.01–1.05, llama_prefill 1.000–1.009.
  - Remaining 68 points cycle-exact.
  - Retained as "no significant contribution", **disabled by default**.
- **Acceptance 4 (B4)**: Full sequential run 26/27 cycle-exact. llama_decode board 545164 vs simulation 545162.
  - Isolated reruns match simulation at 545162. Re-running immediately after gpt2_block reproduces 545164, tracing to unreset inter-launch state from gpt2_block.
  - Isolated to kernel `..._neg_split_transpose_view_3` (grid 2): 161 → 162 cycles. Not memory boundary related; single-session simulation does not reproduce. Evidence in [`r39_report.md`](r39_report.md) §8.4.
  - Recorded as open issue, frozen into B39 (R40-00).

### R39-13: Evaluation load expansion (ArchBench medium tier)

Deliverables:

- Add `medium` tier between small and large, covering all families (op and model).
  - Constraint: kernel arguments (input, output, weights) fit within 384 KiB heap out of 512 KiB device memory. Runtime uploads arguments per launch.
  - Scale: 2–8× small. gemv small already uses 256 KiB weights, scaled marginally.
- Configurations recorded in `archbench` (`ops.py`, `models.py`, `registry.TIERS`), frozen as `suites/medium.json`; `ablate.py` supports per-tier breakdowns.
- Board executes full medium tier; simulation reserved for debugging.

Acceptance:

1. Medium tier passes numerics completely on final image.
2. Freeze R39-00 baseline medium figures on r39b image with baseline software.
3. R39-11 board conclusions verified across tiny + small + medium: S_geo, E, 0 regressions.

Progress (2026-10-10, on board):

- **Medium tier** (`5a7eab4`, `2d51e3c`): 25 points, 24 families, maximum launch parameters ≤ 352 KiB.
  - Runtime launch overhead ~13 KB (per-context scratch, TensorMap, alignment).
  - gpt2_block and transpose initially exceeded heap; resized and refrozen.
- **Frozen baseline medium figures** (Acceptance 2):
  - Setup: u280_r39b image, `0b7ef29` software, switches off, profile core-vrf-sync-v1.
  - Result: 25/25 numeric pass, recorded in [`spec/r39/baseline_archbench_medium.jsonl`](../spec/r39/baseline_archbench_medium.jsonl).
- **v4 final on medium tier** (Acceptance 3 component): u280_r39e, R39-05 final software, 25/25 pass. Relative to baseline:
  - S_geo **4.309**; E 0.00751 → 0.02669 (3.55×).
  - 0 regressions, smallest gain embedding at 1.983×.
  - Issue share 0.277 → 0.515.
  - Consistent with tiny + small (4.315, 3.55×).

Final Status (2026-10-10 evening):

- **Acceptance 1**: medium 25/25 pass numerics on final image u280_r39h with Mode A enabled and disabled.
- **Acceptance 2**: frozen baseline medium figures in [`spec/r39/baseline_archbench_medium.jsonl`](../spec/r39/baseline_archbench_medium.jsonl).
- **Acceptance 3**: v5 final relative to frozen baseline: tiny 5.538, small 5.617, medium 5.491, 0 regressions; R39-11 conclusions confirmed across all three tiers.

## 4. Execution Order

1. **R39-00 before all else.** Standard-path baseline and ablation tools must precede performance claims.
2. **Compiler packages R39-01 → 04.** Measured on board in minutes, prioritized first.
3. **Hardware packages R39-06, 08, 07, 05, 09.** Evaluated in RTL simulation and OOC synthesis, consolidated into board image links (~4 hours each). R39-05 (4 contexts) scheduled after bypass and LSU to measure latency hiding on residual stalls.
4. **R39-10 data-driven.**
5. **R39-11 wrap-up.**

After completing each work package: update frozen baseline, `design_change` records, status line in [`r39_tasks_and_acceptance.md`](r39_tasks_and_acceptance.md), and push.
