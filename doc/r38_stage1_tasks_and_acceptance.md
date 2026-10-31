# R38 Stage 1: Cross-Layer Closed Loop, Profiler, and ArchBench v0 without FPGA

Version: 1.0, 2026-09-22. State: **IN PROGRESS / NOT ACCEPTED**.
Upstream plan: [R38 ISCA roadmap](r38_isca_roadmap.md). Current progress: [r38_stage1_status.md](r38_stage1_status.md). Starting point: `r38` branch, `3b5b28f` (above R37 447/447).
Sign-off identifier: `R38_STAGE1_RTL_LOOP_READY`. It does **not** signify that any FPGA, board-level, energy, NVIDIA comparison, or LLM conclusions hold.

## 0. Mapping the Five Goals Across Two Stages

R38's mainline progresses in the following order, where each item is a prerequisite for the next:

| # | Goal | Stage 1 (Small operators + RTL simulation) | Stage 2 (FPGA large-scale simulation / physical measurement) |
|---|---|---|---|
| 1 | **ArchBench standard**: 45 operators, several thousand points, distance to roofline ceiling under different configurations, weighted by operator runtime | Protocol, lower bounds, $\eta$/E (harmonic mean weighted by reference runtime), AQ formula, all anti-gaming self-tests; calculate $E_c$ on existing RTL configurations (`core-sim`, `core-vrf-sync-v1`, `core-4r`, and elaboration parameter variants) | Full suite of thousands of points × small/medium/large FPGA budget tiers; cross-scale and cross-substrate AQ validation |
| 2 | **cost model + profiler** driving compiler + ISA + hardware co-optimization, improving ArchBench scores | profiler v0, single-core cost model v0, one real three-layer feature and cross-layer case ablation | Multi-core / full-grid model, FPGA calibration, physical metrics (Fmax and resources) incorporated |
| 3 | **Agent (Codex) automated co-tuning**: optimization quality and token cost with/without cost model and profiler | 2×2 protocol, isolated workspaces, token/tool accounting, pilot | Formal runs (≥5 per group), statistical conclusions |
| 4 | **ArchBench comparison against NVIDIA** | Specify GPU scoring protocol under the same framework (spec_peak, lower bounds, weight sources); no comparative conclusions produced | GPU physical measurements + FPGA physical measurements, layered conclusions per roadmap §11 |
| 5 | **PyTorch support, running LLMs on FPGA** (screenshots in Appendix) | Small graphs on simulator (P1+ on RTL; supersedes initial P0/P1 scope) | Decoder → small models → FPGA demo and screenshots |

Quantitative definition of Goal 1: For configuration $c$ and point $i$, $\eta_{i,c} = L_{i,c} / T_{i,c}$ (where $L$ is the strict multi-resource roofline lower bound, and $T$ is measured device time). The harmonic mean $E_c = (\sum_i w_i / \eta_{i,c})^{-1}$ weighted by frozen reference runtime weights $w_i$ defines the "distance to the ceiling". To remove the effect of scaling the same design up or down, AQ is computed against a common budget lower bound (roadmap §4.10). Both $E$ and AQ are reported, with $S$ (absolute speedup) as an accompanying metric to prevent artificially raising $E$ by deflating peak capacities.

### 0.1 Why Divide into Two Stages

All work in the R38 roadmap depends on credible timing and correctness feedback. The baseline environment provides only Icarus Verilog RTL simulation (`core-sim`, `core-vrf-sync-v1`, `core-4r`, `system-2x2`), Yosys structural proxies, and small TileBench cases. Without an FPGA or board-level clocks, external memory, or power instrumentation, RTL simulation throughput is approximately 1–2 wall-clock seconds per small case.

Tasks are partitioned by **evidence source** rather than work package names: if the required evidence for a task can be fully provided by "small operators + RTL cycles + structural proxies", it belongs to Stage 1. If its conclusions depend on board-level frequencies, physical resources, power, throughput across thousands of points and multiple configurations, or large-capacity external memory, it belongs to Stage 2. Stage 1 prepares protocols and tools for Stage 2, but **must not draw Stage 2 conclusions from Stage 1 data**.

### 0.2 Overall Task Classification

| Roadmap Work Package / Section | Stage 1 (Current RTL Environment) | Stage 2 (Requires FPGA Large-Scale Simulation / Board Measurement) |
|---|---|---|
| R38-00 Baseline and evidence audit (§1, §14.4) | Complete: R37 reproduction, evidence registry, authentic AOT/MLIR path | — (Stage 2 appends `synthesized` / `board-measured` evidence) |
| R38-01 Minimal three-layer closed loop (§3) | Complete: one real compiler+ISA+RTL feature, compatibility refusal, rollback, manifest | Replicate same closed loop on FPGA, measure L3/L4 feedback latency |
| R38-02 ArchBench (§4, §4.10) | Protocol, schemas, single-core `spec_peak`, strict lower bounds $L$, $E$/$A$/$S$/$U$, all Appendix A self-tests, AQ calculator + `AQ-RTL-canonical` (1× budget, `NOT CERTIFIED`), §4.10.9 items 3/5/7 | Small/medium/large budget tiers (LUT/DSP/BRAM/Fmax), 1×–8× multi-budget AQ, cross-substrate consistency (§4.10.9 items 1/2/8), `AQ-certified-domain` |
| §5 2,880-point workload | 45-operator manifest and split rules frozen; RTL-feasible dev/validation subset (157 points) | Full 2,880 points × configurations × O0/O1/O2 measurements, complete held-out evaluation, 100–300 FPGA anchors |
| R38-03 Profiler (§7) | Complete v0: device-side counters readable through public runtime, ROI, conservation, IR→PC mapping, RF/compute/memory bottleneck attribution | Board DRAM controller / NoC physical counters, impact of trace muxes on P&R |
| R38-04 Cost model (§6) | v0: single-core single-program analytical model + calibration, error/ranking/interval on unseen shape families | Multi-core / full grid, unseen hardware configurations, FPGA calibration, physical layer (Fmax/resources) |
| R38-05 Cross-layer case studies (§8) | Case A (RF / operand collector), Case B (async descriptor pipeline) fully ablated on single-core RTL | Case C (16-tile grid / NUCA / admission), Fmax/resource costs for all cases |
| R38-06 Controlled studies and Agent (§9, §10) | Protocol, task suites, isolation and accounting tools; 2×2 pilot (1 run per group, 2 tasks) | B0–B6 formal controls, ≥5 runs per group, held-out evaluation, statistical conclusions |
| R38-07 Measured platform (§11) | — | FPGA P&R / bitstreams, NVIDIA strong baseline, energy consumption |
| R38-08 PyTorch / LLM (§12) | P1+ small graphs on simulator (supersedes P0/P1) | P2–P5: decoder block, full small models, FPGA demo |
| R38-09 Paper evidence (§15) | Related work matrix (HASCO / ACT / Chipyard / Gemmini / ArchEval) documentation | Held-out freeze, main table reconstruction, artifact release |

### 0.3 Explicit Non-Goals for Stage 1

- Do not report MHz/GHz, wall-clock seconds, Watts, mm², or physical LUT/DSP counts; Yosys numbers must be labeled `structural-proxy`.
- Do not report `Session` serial-grid accumulated cycles as full-chip throughput; all grid results must be labeled `serial-grid`.
- Do not draw multi-core scalability conclusions (`system-2x2` is used only for functional and counter correctness).
- Do not freeze formal held-out benchmark results in Stage 1; only freeze the **rules** (split, weights, lower-bound versions).

## 1. Stage 1 Final Deliverables

1. A single-command runnable L1 closed loop: `make r38-smoke` executing unmodified TileBench → public AOT → `.tnca` → `Session` → RTL, yielding correctness + device cycles + counters + bound efficiency + one structured measurement record.
2. Device profiler v0 readable through the public runtime, with counters satisfying conservation identities and proven non-perturbing.
3. ArchBench v0 scoring library: auditable strict lower bounds, $E$/$A$/$S$/$U$/AQ calculators, and all ten Appendix A self-tests fully executable.
4. One real three-layer feature (compiler + ISA contract + RTL), with capability bits, compatibility refusal, rollback verification, and a complete ablation of at least one cross-layer case (including resource proxies and negative results).
5. Single-core analytical cost model v0 and its evaluation report on unseen shape families.
6. Evidence registry: every capability labeled as `documented / implemented / functionally-validated / cycle-validated / synthesized / board-measured`; Stage 1 must not contain `board-measured`.

## 2. Common Rules (Mandatory Across All Work Packages)

- **R1 Public path**: All performance and correctness measurements must pass strictly through `tnca.aot.compile_aot -> .tnca -> tnca.runtime.Session`; bypassing through private lowering/codegen queries is forbidden.
- **R2 No regression**: Every commit must maintain R37 `O0/O1/O2 = 447/447`. When features are disabled, device cycles must match the S1-01 baseline **case-for-case** (fast memory model determinism). Any cycle change must be justified by an explicit `design_change` record.
- **R3 null is not 0**: Unknown fields must be set to `null`; schema validation rejects using 0 to represent unmeasured data.
- **R4 Evidence tags**: Every measurement record must carry `evidence_level ∈ {predicted, measured-rtl}`; Stage 1 produces no other values.
- **R5 Failure preservation**: Failed cases, failed candidate implementations, and rolled-back features must be preserved in records, never purged from reports.
- **R6 Artifacts**: Generated reports are written to `sim/build/r38/` (not tracked in Git); Git tracks only schemas, scripts, frozen suite descriptions, and hashes.

## 3. Work Packages and Acceptance Criteria

Numbered `S1-xx`. Acceptance tests for each work package must be executable commands returning zero on pass; manual inspection items are listed separately.

### S1-00 — Phase Partitioning and Evidence Registry

- Deliverables: this document; `spec/r38/evidence_registry.json`, listing R37/R38 capabilities (operator coverage, public AOT, counters, MLIR passes, async execution, INT8/FP8 paths, grid, synthesis, board, etc.) with evidence levels and linked artifacts.
- Acceptance tests (`python3 tools/r38/evidence_tests.py`):
  1. Every record's `level` belongs to the six-level enum (`documented`, `implemented`, `functionally-validated`, `cycle-validated`, `synthesized`, `board-measured`), and linked files exist in the repository or are declared under `sim/build/`.
  2. Entries with `level >= functionally-validated` must reference an existing test command (Makefile target or script).
  3. Any entry labeled `board-measured` causes test failure (forbidden in Stage 1).
  4. `synthesized` entries must declare `structural_proxy: true`.
  5. Mutation testing: artificially changing an entry to `board-measured`, removing a linked file, removing a required command, specifying a nonexistent command, omitting `structural_proxy`, or introducing an unknown level must be caught and fail the test (6/6 built-in mutations caught).

### S1-01 — Baseline Reproduction and Cycle Baseline

- Deliverables: re-run R37 three optimization levels on the starting commit; `sim/build/r38/baseline_O{0,1,2}.json` recording per-case `serial-grid` cycles, per-program cycles, artifact hashes, source hashes, profile config hashes, and tool versions (iverilog, triton, python). Freeze `spec/r38/baseline_cycles.json` (case → O2 cycles) as the baseline for rule R2.
- Acceptance tests:
  1. `make r37` (or `make r38-baseline`) achieves 447/447.
  2. `tools/r38/baseline.py --check` (or `make r38-baseline-check`): runs all 149 O2 cases twice; device cycles are identical across runs (determinism) and match `spec/r38/baseline_cycles.json` exactly.
  3. Baseline JSON records have non-empty `artifact_sha256`, `source_sha256`, and `config_hash`.

### S1-02 — Profiler v0: Device Counters Readable via Public Runtime

- Deliverables:
  - RTL: `tnca_shell` adds counter read port `perf_sel/perf_rdata` (combinational read mux generated from `spec/r38/counters.json`, zero execution disturbance), covering `cycles`, `stall_exclusive` (including `perf_x_vrf`/`vrfwb`), `stall_overlapping`, `control`, and `memory` beat groups from `spec/r3_counters.json`.
  - Transport: `spec/r2_transport_v1.json` protocol extended (Transport 1.1) with op 7 `PERF_READ` (a0 = counter index → r0 = low 32 bits, r1 = high 32 bits); legacy devices return `REFUSED` for op 7, runtime reports `counters: {'available': False}` rather than 0.
  - Runtime: `Session(..., counters=True)` takes snapshots before and after launch for each program and computes deltas; `event.counters` provides per-program and whole-grid aggregates; disabled by default.
  - `spec/r38/counters.json`: index ↔ name ↔ group ↔ source module, driving both RTL (`rtl/tnca_perf_sel.vh` via `tools/r38/gen_counters.py`) and runtime.
- Acceptance tests (`tools/r38/profiler_tests.py` / `make r38-profiler`, executed on `core-sim` and `core-vrf-sync-v1`):
  1. **Conservation**: for each program, $\sum \text{perf\_x\_*} == \text{perf\_stall\_total}$ (exact equality); additionally $\text{perf\_issued} + \text{perf\_stall\_total} + \text{perf\_x\_frontend} == \text{perf\_cycles}$ (cycle partition).
  2. **Identity**: $\Delta \text{perf\_issued} == \text{done\_retired}$ (instructions retired per launch); $0 \le \text{done\_cycles} + \text{run\_window\_slack} - \Delta \text{perf\_cycles}$, where $\text{run\_window\_slack} = 3$ is derived from RTL structure and recorded in `counters.json`.
  3. **Containment**: $\text{perf\_stall\_total} \le \Delta \text{perf\_cycles}$; each overlapping stall count $\le \Delta \text{perf\_cycles}$.
  4. **Monotonicity**: snapshots never decrease within a session.
  5. **No perturbation**: all 149 R37 O2 cases pass numerically under `counters=True`, with cycles identical case-for-case to the S1-01 baseline.
  6. **Compatibility**: running on a legacy device image (without op 7) causes runtime to report `available: False`, without raising exceptions or returning zeros.
  7. **Mutations**: two RTL mutations (swapping counter indices, grounding an exclusive counter to constant 0) must be caught by tests 1–3.
  8. R36 export/port checks (`make r36-rtl-tests r36-build-tests`) pass after updating the port inventory. (Also fixed during S1-02: `done_retired` reporting per-launch rather than cumulative instructions).

### S1-03 — Measurement Records and Candidate Manifests

- Deliverables: `spec/r38/measurement.schema.json` (roadmap §13.2), `spec/r38/candidate.schema.json` / candidate hashing $V = (C, I, H, R, W, M, P, E)$ in `tools/r38/manifest.py`, and `spec/r38/design_change.schema.json` (roadmap §3.2); generation and validation in `tools/r38/manifest.py`.
- Acceptance tests (`python3 tools/r38/manifest_tests.py`):
  1. Every record generated by S1-04 smoke suite passes schema validation.
  2. Missing fields, missing units, illegal `evidence_level`, or substituting `null` with `0` for unmeasured fields (per `positive_or_null` and `stage1_null` rules) are rejected.
  3. Modifying any file in RTL / compiler / ISA / runtime changes the corresponding H / C / I / R hash component, while leaving others unchanged.
  4. Identical inputs in two clean working directories produce identical candidate hashes (reproducibility).
  5. Scorer entry rejects records with mismatched `bounds.version` or timing protocol.

### S1-04 — Performance Harness and Smoke / Dev Suite

- Deliverables: `tools/r38/run.py` (strictly using public AOT/runtime); `spec/r38/suites/smoke.json` (8 points covering different bottlenecks: FP16 GEMM, INT8 GEMM, softmax, layernorm, histogram [2 stages], flash_attention, 2d_conv, matrix_transpose; 9 cases total); `spec/r38/suites/dev.json` (13 operators, 157 parameterized points across 46 shape families, partitioned into dev / validation / heldout at 60/20/20 with entire families held out); `make r38-smoke`, `tools/r38/run.py --suite dev`.
- Acceptance tests:
  1. `make r38-smoke` passes all cases numerically (9/9 cases across 8 points), each producing a valid measurement record with cycles, counters, lower bounds (S1-05), and efficiency.
  2. Smoke suite repeated 3 times on fast memory model yields identical cycles; on slow memory model reports median and range.
  3. `make r38-smoke` wall clock ≤ 15 minutes (900 s L1 budget), actual duration logged in report.
  4. Dev suite numeric failures = 0; any failure recorded with explicit status (`compile-failure`, `runtime-fault`, `numeric-error`, `capacity-exceeded`), never conflated as "slow".
  5. Split check: points in the same family do not cross splits; held-out points in Stage 1 may only be evaluated once by S1-06 (access count logged in `sim/build/r38/dev/heldout_access.log` and checked by tests).

### S1-05 — ArchBench v0 Scoring Library

- Deliverables: `tools/r38/archbench/`:
  - `peaks.py`: derives `spec_peak` per cycle from `compiler/targets/*.json` and profile elaboration (FP lanes, MMA MACs, SIMD lane ops, SFU ops, LSU/shared memory bytes, issue width), **never** from measurements.
  - `bounds.py`: strict lower bounds $L = \max(\max_r W_r/C_r, \max_\ell Q_\ell/B_\ell)$, covering elementwise, SFU, reduction/norm, GEMM/batched GEMM, conv, copy/transpose; sort/gather/attention in v0 raise `BoundUnavailable` / report `unavailable` (never fabricated).
  - `score.py`: calculates $\eta$, $E$ (harmonic mean), $A$ (arithmetic mean), $S$ (speedup), $U$, AQ (roadmap §4.10.6); handles missing/failed points per roadmap §4.8.
  - `canonical.py`: canonical primitive library v0 (cost coefficients version θ0: `THETA0`), defining 1× reference budget caps across logic, storage, and communication.
- Acceptance tests (`python3 tools/r38/archbench_tests.py`):
  1. Roadmap **Appendix A items 1–10** implemented as individual tests (`a1_uniform_2x_faster` through `a10_energy_without_sensor_stays_empty`).
  2. §4.7 worked example: design A and B $\eta$, $S$, arithmetic and harmonic means match document.
  3. §4.10.9 item 3 (larger designs cannot score in smaller budget), item 5 (gaming does not raise AQ), item 7 (degraded configs score lower: extra barrier, reduced RF ports score lower with counter attribution; pending degraded RTL builds).
  4. For all bounded points in baseline, smoke, and dev suites, measured $\eta \le 1$; any $\eta > 1$ fails test and outputs audit info (never clipped).
  5. GEMM hand-calculated bound (16×16×16 FP16) matches script, stored as test constant.
  6. AQ output label fixed to `AQ-RTL-canonical`, and outputs `NOT CERTIFIED` if any registered point is missing or unmeasured.

### S1-06 — Cost Model v0 (Single-Core, Single-Program)

- Deliverables: `tools/r38/costmodel/`: extracts static instruction mix, memory bytes, and dependency chains from artifact machine code and manifest; analytical resource scheduling + lookup-table calibration of residuals; outputs predicted cycles and 90% confidence interval.
- Acceptance (one-time evaluation on dev suite held-out family via `tools/r38/costmodel/evaluate.py --final`):
  1. Median relative error ≤ 20%, P90 ≤ 40%, Spearman rank correlation ≥ 0.7; 90% interval empirical coverage ∈ [80%, 98%].
  2. Separately reports error on unseen shape families and unseen optimization levels, without pooling.
  3. Reports modeling and calibration wall clock and measurement sample counts.
  4. If criteria in item 1 are not met, record S1-06 as `NOT ACCEPTED (model-limited)` with mismatch categories identified; do not tweak thresholds or repartition held-out splits. (Status noted in [r38_stage1_status.md](r38_stage1_status.md): accepted with held-out n=26, median 0.0%, P90 7.3%, Spearman 0.999, coverage 0.88, with noted absence of GEMM/transpose/softmax/rmsnorm families in the held-out split).

### S1-07 — Minimal Three-Layer Closed Loop (One Real Feature)

- Deliverables: based on S1-02/S1-04 counter data, implemented transposed tensor-DMA write (`HAS_TENSOR_XPOSE`, profile `core-xpose-v1`, capability `tensor_transpose`):
  - Microarchitecture / RTL: `rtl/core/tnca_tensor.v`, `tnca_shell.v`, etc., 16×8-byte corner-turn buffer, elaboration parameter `HAS_TENSOR_XPOSE`. `mod[5]` without feature and non-zero `mod[4:3]` raise `ILLEGAL_INSTRUCTION`.
  - ISA contract: ISA §9E transposed layout geometry, capability `tensor_transpose` in `compiler/targets/tnca-sim-xpose-v1.json`, profile `core-xpose-v1`.
  - Compiler: `compiler/python/tnca_cc/frontend.py` `_plan_transposes`: 16-bit descriptor load used by `tt.trans` emitted as `t.ld.async.h.t`.
  - Runtime: `python/tnca/profiles.py`, `runtime.py`: loader rule `check_executes`: rejects `core-xpose-v1` artifacts on `core-sim` with typed error `CAPABILITY_MISSING(tensor_transpose)`; legacy artifacts run on `core-xpose-v1`.
  - Documented as `spec/r38/design_changes/xpose-dma-001.json`.
- Acceptance tests (`tools/r38/loop_tests.py`, `make r38-xpose-rtl`):
  1. Feature disabled (`core-sim`): 447/447, cycles identical to baseline case-for-case.
  2. Feature enabled (`core-xpose-v1`): 447/447, smoke FP16 GEMM 16×16×32 speedup 1.46× (13549 → 9249 cycles), explained by counters (memory beats 3156 → 1616, LSU stalls 2954 → 1185).
  3. Compatibility: new artifacts on old hardware rejected before launch (`CAPABILITY_MISSING`); old artifacts on new hardware run with baseline cycles; illegal descriptor strides trap `DESCRIPTOR_FAULT`.
  4. Rollback: with feature disabled (`core-sim`), all 149 O2 compiled artifacts are byte-identical to baseline; RTL structural proxy matches parent commit engine (gates/flops delta ≤ 0.5%).
  5. Resources: Yosys structural proxy (+4734 gates / +11.2%, +1089 flops / +12.5%, +3 depth) recorded in design change record.

### S1-08 — Cross-Layer Case A/B Ablation

- Deliverables: four-way ablation for S1-07 feature (compiler-only, hardware-only, both, baseline parent) on smoke and dev suites; report S, E, counter attribution, structural proxy; write research card `doc/r38_case_<id>.md` per roadmap Appendix B, including at least one zero-gain or negative-gain exploration.
- Acceptance:
  1. All four configurations pass 447/447.
  2. Interaction effects reported via log-speedup differences, never additive.
  3. Every number in research card reproducible from records under `sim/build/r38/` via verification script (`tools/r38/verify_case.py`).

### S1-09 — L0/L1/L2 Promotion Gates and DevOps Metrics

- Deliverables: `tools/r38/gate.py` with commands `l0`, `l1`, `l2`, `selftest`; Makefile targets `make r38-l0` (schema / encoding / counter generator check / evidence tests / archbench tests), `make r38-smoke` (L1), `make r38-l2` (`r38-l0 r37 r38-baseline-check r38-profiler r38-smoke`). Each gate run appends one line to `sim/build/r38/devops_log.jsonl` (candidate_hash, level, verdict, feedback_latency, error details). Pointer `sim/build/r38/last_known_good.json` updates ONLY when L2 passes.
- Acceptance:
  1. Selftest (`tools/r38/gate.py selftest`): injects numerically wrong candidate; L1 fails, `last_known_good` remains unchanged, failure evidence retained in log.
  2. L0 ≤ 5 minutes, L1 ≤ 15 minutes, measured latency recorded in devops log.

### S1-10 — Agent (Codex) Pilot Protocol & PyTorch P1+

- PyTorch: P1+ plan ([r38_pytorch_p1plus.md](r38_pytorch_p1plus.md) superseding initial P0/P1 scope): ArchBench(torch) tiny + small 52/52 compile without host fallback; regression tests; tiny tier on RTL 27/27 pass.
- Agent: Codex evaluation protocol; fixed model ID and snapshot date; 2 frozen tasks, 2×2 (cost model × profiler) isolated workspace groups; token/tool/wall-clock accounting scripts; 1 pilot run per group. (Protocol validation only, no final conclusions).
- NVIDIA: scoring protocol for GPU on ArchBench written ([archbench/PROTOCOL.md](../archbench/PROTOCOL.md)): spec_peak derivation, identical lower bound formula, identical weights; exercised on GB10.
- Non-blocking for `R38_STAGE1_RTL_LOOP_READY`, but completion status must be explicitly reported in [r38_stage1_status.md](r38_stage1_status.md).

### S1-11 — Related Work Matrix

- Deliverables: `doc/r38_related_work_matrix.md`, addressing the seven questions in roadmap §2.1 across HASCO, ACT, Chipyard, Gemmini, ArchEval, DCO/LLaMCAT, citing specific sections. Fact-based review only.

## 4. Execution Order and Gates

```text
S1-00 ── S1-01 ──┬── S1-05 ArchBench v0 (Goal 1, metrics established first)
                 ├── S1-02 profiler ── S1-03 ── S1-04 ──┬── S1-06 cost model
                 │                                       └── S1-07 ── S1-08 cross-layer cases
                 ├── S1-09 gates (runs in parallel starting from S1-04)
                 └── S1-11 related work      S1-10 Agent / PyTorch (post-gated)
```

S1-05 formulas, lower bounds, and self-tests do not depend on counters and can be completed using S1-01 baseline cycles; after S1-04 is completed, $\eta \le 1$ is re-verified across smoke and dev points.

| Gate | Condition | If Not Met |
|---|---|---|
| G1a | S1-00, S1-01, S1-05 (metrics first) accepted | Do not begin any performance conclusions or feature selection |
| G1b | S1-02, S1-03, S1-04 accepted; all smoke $\eta \le 1$ | Pause cost model and features; fix timing/lower bounds first |
| G1c | S1-07 accepted | Do not proceed to Agent / methodology comparisons |
| Stage 1 Sign-off | S1-00 … S1-09 accepted (S1-06 may be accepted as model-limited with recorded boundaries); [r38_stage1_status.md](r38_stage1_status.md) links all evidence | Continue Stage 1; do not draw Stage 2 conclusions |

## 5. Hand-off Artifacts Entering Stage 2

- Frozen protocols: workload / measurement / candidate schemas, split rules, lower bound versions, canonical θ0.
- Hardware harness directly reusable on FPGA backend (transport op 7 MMIO equivalent connects directly to counters).
- Single-core cost model and its documented failure envelope (Stage 2 recalibrates for multi-core / board).
- Cross-layer feature and case study cards (Stage 2 measures physical Fmax/resources on FPGA and large-scale held-out evaluations).
