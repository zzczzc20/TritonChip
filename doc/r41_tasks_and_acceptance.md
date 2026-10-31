# R41: Standard Torch Inference and Compiler-Hardware-Performance Model Co-Iteration

**Version:** 0.2  
**Date:** 2026-10-20  
**Status:** PROPOSED / NOT FROZEN / NOT ACCEPTED  

**Stage Sign-off Names:**

- `R41_S1_COMPILER_AND_PERFORMANCE_HARNESS`
- `R41_S2_100MHZ_AND_FULL_MODEL`

The new files, tools, and Make targets introduced in this document represent proposed deliverables and do not imply that they have already been implemented.

---

## 0. Contract Principles

### 0.1 Two Stages, Two Explicit Delivery Boundaries

| Stage | Primary Execution Platform | Must Deliver | Not a Sign-off Prerequisite for This Stage |
|---|---|---|---|
| **Stage 1** | Loaded R40 3-core, 90 MHz, 4 GiB image | Critical compiler fixes; budgeted selection mechanism; complete performance_harness; 100 MHz candidate design and model feasibility preparation | New bitstream, 100 MHz, full Qwen on board |
| **Stage 2** | New 3-core image, target and sign-off frequency at 100 MHz | Real timing closure; capacity and long lifecycle; full Qwen text inference; original suite and bounded XL; on-board validation of selected optimizations | All research mechanisms yielding gains, all target multiples achieved, physical 8/16/32-core delivery |

Stage 1 can be released and signed off independently.

When Stage 2 is not completed, the overall R41 cannot be marked as complete; however, an established Stage 1 delivery cannot be revoked.

### 0.2 Distinguishing Three Types of Requirements

| Type | Meaning | Closure Method |
|---|---|---|
| **Required** | Mandatory functionality, correctness, safety, or evidence | Must pass |
| **Study** | Bounded research or candidate evaluation that must be completed | Can be a positive result, negative result, or inconclusive within budget; must submit evidence and decision |
| **Target** | Aspirational target for performance, efficiency, or coverage | Report whether achieved; does not independently block stage sign-off |

Special provisions:

- `closed-negative / default-off` can close a research task; **it is not a performance pass**.
- Unimplemented, unexecuted, or unevidenced items cannot be recorded as capability-pass.
- A Required item cannot be substituted by writing a negative result report.
- Unfinished Required items stopped due to budget exhaustion remain `blocked` or `fail`; they must not be converted into Targets during closeout.

### 0.3 Staged Freezing Instead of Repeatedly Modifying the Same Contract

Common principles and the Stage 1 scope are frozen first.

Stage 2 specific new workloads, numerical tolerances, resource budgets, and measurement plans are frozen after the Stage 1 feasibility checks are completed. At that time, the following are permitted:

- Official reference implementations;
- Existing R40 measurements;
- Compilation scale and capacity preflights;
- Functional trial runs that do not involve candidate speedup selection.

Favorable acceptance points must not be cherry-picked based on the gains of Stage 2 candidate optimizations.

**The two freezes govern different stages; they do not represent altering a failing contract from v1 to v2 and then to v3.**

---

## 1. R40 Baseline and Carryover Items

### 1.1 Historical Status Remains Unaltered

R40 retains the following status:

> `closed-with-failures`, `R40-10.acceptance` not passed.

R41 does not require retroactively passing all historical R40 contracts, nor does it retroactively recognize R40 as passed.

The R40 baseline is designated **B40-OBS**, avoiding `B40-F` which implies formal sign-off:

| Item | B40-OBS |
|---|---|
| Software | `files:fe1b5199db515440d2d3407aed61ef1eed5edb627e5baf9cda10be5a7198cf75` |
| Image | `a88ebdf59cfbbb2001ebd5b1b2c739c5d8cd33931f1e9202f18cac07c4c83083` |
| Frequency | Measured 90 MHz |
| Physical Configuration | 3 cores, 3 HBM outlets, 4 GiB exposed window |
| Evidence | R40 v3 final campaign, gate records, report, and identity amendment |

These identities originate from the R40 final report; an immutable reference is preserved at R41 freeze. ([github.com](https://github.com/zzczzc20/TritonChip/blob/r40/doc/r40_report.md))

### 1.2 Carryover Ledger

Tracked in [`spec/r41/carryover.json`](../spec/r41/carryover.json), containing at least:

| R40 Item | Stage 1 | Stage 2 | Status | R41 Handling Principle |
|---|---|---|---|---|
| `R40-01.device-time` | Fix unnecessary synchronization; diagnose address placement sensitivity | Re-verify relevant points on final image | open | Do not retroactively alter R40 verdicts |
| `R40-03.gemm-tile` | 16x32 resident fragments, layout and resource legality; fix non-native tile fragment lowering | Joint validation with selected MMA | open | Compiler fixes take priority over array expansion |
| `R40-03.wide-blocks` | Select by legality and budget; distinguish device time and host gap | Final strategy regression | open | Do not enforce globally wide blocks |
| `R40-02.batched` | Targeted diagnosis of finite launch chains | Re-verify only with explicit fixes | open | Do not pursue 1.2× indefinitely |
| `R40-04.bandwidth/R40-06.ports` | Establish home-slice and core-side service ledger | Study service pipelining candidates | open | Do not treat outlet count as speedup |
| `R40-06.locality` | Design standard-path consumer and safety boundaries | Real integration and board test; default off if insufficient gain | open | Functional integration Required, speedup Target |
| `R40-07.long-k` | Conservative selection among Mode B, single role, Mode A | Retest with transfer engine candidates | open | Mode A not forced default-on |
| `R40-07.matrix` | Candidate modeling, resource and semantic preflight | Evaluate at least one RTL candidate; validate selected implementation | open | Do not force specific array dimensions |
| `R40-08.prefill` | Register, transfer, and compute attribution; D256 feasibility | General D256 and hybrid model implementation | open | Do not force the use of slower fused paths |
| `R40-09.autotune/holdout` | Budgeted selection, cache, and new holdout | Frozen holdout validation | open | Do not require tuning all points |
| `M1-M4/frequency` | Merge, related regressions, census, OOC, and candidate list | 100 MHz new image and board bring-up | open | Not a hardware delivery until brought up on board |
| `SystemC M2/M3` | Complete harness, dual memory paths, calibration workflow | Update new RTL profiles and domain of validity | open | Model results do not replace board measurements |

The locality, autotune, holdout, MMA, and conditional wide-ingress ablations explicitly carried over by the R40 contract retain their original provenance and status. ([raw.githubusercontent.com](https://raw.githubusercontent.com/zzczzc20/TritonChip/r40/doc/r40_tasks_and_acceptance_v3.md))

---

## 2. Immutable Requirements Common to Both Stages

### 2.1 Sole Official Execution Path

```text
Official PyTorch model / model-agnostic Torch graph
  → torch.Tensor(device="tnca")
  → torch.compile(backend="inductor", fullgraph=True)
  → Inductor generic decomposition, scheduling, fusion, Triton templates
  → Triton frontend / TNCA backend
  → Common lowering / ISA / artifact
  → Same Session, allocator, event, grid dispatcher
  → TNCA / NUCA / NoC / HBM
```

Requirements:

- Tensor descriptors remain enabled.
- Pointer lowering with explicit rationale is permitted; not all accesses are required to become descriptors.
- Special-casing by model name, layer name, checkpoint hash, or benchmark ID is prohibited.
- CPU/CUDA numerical fallbacks and implicit tensor staging are prohibited.
- Direct invocation of dedicated TNCA kernels bypassing Inductor via model adapters is prohibited.
- Generic Triton templates implemented by engineers are permitted for use by the common Inductor selection mechanism.
- Legitimate fallbacks must remain within the standard TNCA path (e.g., shrinking tiles, using generic decompositions, or conservative scheduling).

Model prefill and fixed-bucket decode are separately fullgraph; the Python generation loop may serve as host orchestration.

### 2.2 Numerics and Protocols

- Shapes, dtypes, tolerances, and point hashes of original frozen points remain unchanged.
- Existing MMA instructions preserve per-K FP32 FMA rounding semantics.
- Old instructions must not be stealthily altered via split-K, tree reduction, or reduced accumulation precision.
- `source-free` and `result-ready` must not be conflated.
- Aliasing, asynchronous lease, masked store, ownership, fault drain, and completion visibility semantics must be maintained.
- Insufficient compilation resources must be rejected early or resolved by selecting a legal configuration, rather than encountering code store or register overflow mid-execution.

The current matrix unit provides 16 MAC/cycle; the four existing MMA input formats share a common FMA via widening. Five-format FMA support does not equate to five-format MMA support. ([raw.githubusercontent.com](https://raw.githubusercontent.com/zzczzc20/TritonChip/r40/rtl/core/tnca_matrix.v))

### 2.3 Preservation of the Original Suite

Final software identities in both stages must preserve:

- Native tiny/small/medium: 77 points;
- Large: 24 points;
- Frozen HBM inventory;
- Original R40 4-layer model workloads;
- ISA/compiler/runtime/protocol regressions relevant to modifications.

Full board testing is executed once on the final candidate of each stage. During development, only relevant subsets are run; rerunning all large workloads on every commit is not required.

---

## 3. Technical Backbone: Unify Execution Resource Plan First, Then Localize Implementation Edits

R41 will not construct a second compiler or a second runtime.

It is proposed to introduce a lightweight **execution resource plan**, produced as a sidecar by existing compilation phases to bridge existing layout, allocation, scheduling, models, and selectors:

```text
Logical tile / valid mask
    ↓
Physical fragment, pack, conversion
    ↓
VRF / window / accumulator lifetime
    ↓
DMA / MMA / LSU commands and dependencies
    ↓
Resource demands, legal candidates, and performance diagnosis
```

It must record at least:

- Physical fragment count, effective MACs, and padding MACs;
- Accumulator round-trips within the K loop;
- Peak VRF, window, ACC, barrier, and code bytes;
- Descriptor and pointer transfers;
- Transfer/MMA command counts, dependencies, and potential overlap;
- Rationales for selecting or rejecting configurations.

**This plan is a common resource description, not a common numerical oracle.**  
Numerical verification continues to use independent references; SystemC continues to execute real artifacts rather than merely executing compiler cost summaries.

`LinearLayout` is currently used for view composition and layout transformation classification, while physical bank allocation is implemented separately. R41 should connect these existing components rather than reproving that "LinearLayout was invoked." ([raw.githubusercontent.com](https://raw.githubusercontent.com/zzczzc20/TritonChip/r40/compiler/python/tnca_cc/layout.py))

---

# Stage 1: Software Fixes and Rapid Performance Iteration Loop on 90 MHz

## S1-00: Freeze Baseline, Scope, and Budgets

### Deliverables

- B40-OBS identity and read-only evidence index.
- Inventory and hash audit of the original suite.
- Stage 1 required/study/target checklist.
- Fixed performance diagnosis subset, selector development set, and holdout partitioning.
- Compilation, modeling, board testing, and worker budgets.
- Carryover ledger.

### Required

1. Do not depend on retroactively passing the historical R40 final acceptance.
2. Old PASS records cannot be copied directly as R41 PASS records.
3. Records with identical identity, point, configuration, and timing boundary may be reused to avoid rerunning a baseline across multiple gates.
4. Records with differing images, code, cache states, or timing boundaries must not be deduplicated erroneously.
5. In frozen files, required workloads, references, tolerances, and budgets must not be `null`.

---

## S1-01: Deliver performance_harness as a Complete Workflow

### Objective

Rather than merely adding a SystemC simulator, establish an integrated workflow:

```text
Question & hypothesis
  → Compiled artifact / resource ledger
  → Analytical lower bounds
  → SystemC rapid screening
  → Detailed memory model when needed
  → Small-scale RTL verification
  → FPGA verification
  → Prediction residual, attribution, and next decision
```

The existing harness already contains cost constraints, performance counters, Proton, RTL experiment interfaces, and physical reporting interfaces; R41 should reuse and integrate these rather than establishing a parallel toolchain. ([raw.githubusercontent.com](https://raw.githubusercontent.com/zzczzc20/TritonChip/r40/tools/performance_harness/README.md))

### A. Unified Experiment and Evidence Schema

An experiment definition must include:

- Hypothesis, competing explanations, expected counter shifts;
- Workload, inputs, reference, and configurations;
- Software, artifact, image, or model identities;
- Clock frequency, memory map, tool versions;
- Measurement boundaries and budgets;
- Results, residuals, conclusions, and next steps.

Unified support for:

- Compile-only;
- Analytical screening;
- SystemC native execution;
- RTL diagnostics;
- FPGA measurement;
- Compare/report.

Support checkpoint-resume, exact deduplication, and retention of failure records.

### B. SystemC-3/TLM-2 Native Execution

Required:

1. Continue executing real compiler outputs through the shared Session path.
2. Do not embed an RTL simulator inside the execution path.
3. Do not drive full-chip predictions using captured on-board stall traces.
4. Explicitly reject unimplemented opcodes, protocols, or capabilities.
5. Record functional results, timing predictions, and simulator wall time separately.
6. `off/aggregate/trace` modes must not alter architectural outcomes; traces must be bounded with drop events reported.
7. New RTL semantics must update corresponding model profiles rather than continuing to use legacy chip configurations.

M2 has already incorporated structural modeling for core pipelines, LSU, VRF banks, transfer engines, and home-slices; these should not be listed as "implemented from scratch." However, its current overall timing status remains uncalibrated: holdout median error 21.5%, p95 error 34.7%. ([raw.githubusercontent.com](https://raw.githubusercontent.com/zzczzc20/TritonChip/r40/evidence/r40/v3/systemc_m2/README.md))

### C. Fast Memory Path and Ramulator2 Path

Deliver two switchable paths:

| Path | Purpose |
|---|---|
| Fast, RTL-structured service model | Day-to-day compilation screening; issue/register/engine bottleneck analysis |
| Ramulator2 detailed DRAM path | Workloads sensitive to banks, rows, read/write turnarounds, refresh, etc. |

Requirements:

- Pin dependency revisions and configurations.
- Clearly delineate boundaries among AXI, NoC, home engine, controller, and DRAM.
- Do not double-count latencies in the detailed path that are already accounted for in the fast model.
- Accurately propagate backpressure, completion events, and write responses.
- Use verifiable PC/address maps.
- The detailed path must not serve as the default path for all experiments.

**Integrating Ramulator2 does not automatically imply higher model accuracy.** It is mandatory to evaluate which residuals it resolves and how much simulation cost it adds. In the R40 report, M3 remains ongoing work and cannot be recorded prematurely as calibrated. ([github.com](https://github.com/zzczzc20/TritonChip/blob/r40/doc/r40_report.md))

### D. Calibration and Domain of Validity

Separately record:

- Structural constants;
- Fitted parameters;
- Calibration set;
- Validation set;
- Sealed holdout;
- Applicable images, configurations, and workload families.

Rules:

1. Do not fit parameters on the holdout set.
2. Data repeatedly utilized for diagnostics can no longer be designated as unseen holdout data.
3. Separately report absolute cycle errors, A/B ratio errors, ranking inversions, and false gain assertions.
4. Return an explicit status for out-of-domain scenarios without outputting pseudo-precise conclusions.
5. When model uncertainty exceeds candidate variance, do not disqualify valid candidates based on the model.
6. When the overall model fails to meet the historical 10%/20% accuracy standard, it must remain marked as `uncalibrated`; validation results for a clearly defined domain may be reported separately.

**Delivery of the harness does not imply that the full-chip timing model has achieved comprehensive qualification.**

### E. Stage 1 Acceptance

Required:

- A clean environment can build and execute a smoke campaign.
- Complete at least three closed-loop case studies:
  1. GEMM fragment/codegen;
  2. Memory service / ports;
  3. Pipeline / engine admission.
- Include at least one negative result.
- For at least one candidate not used in fitting, preserve its prediction before board testing and record its residual after board testing.
- Simulator speed relative to RTL must be measured on identical small workloads; do not fabricate exact multiples from timed-out runs.
- Retain known model mismatch points rather than pruning them to artificially inflate accuracy metrics.

A gate requiring "all model points median ≤10%" is not set as a blocking gate for Stage 1.

---

## S1-02: Fix Wide GEMM Tiles and Connect Resource-Driven Layout

### Required Deliverables

First fix the `16×32` logical tile:

- Represented with two legal `16×16` accumulator fragments.
- Keep accumulators resident throughout the K loop.
- Cease repeatedly reading back, computing, and writing back accumulators row-by-row merely to sustain a non-native logical tile.
- Necessary readbacks in the epilogue are not considered violations.
- K order remains unchanged.
- Tail, mask, non-contiguous input, and out-of-resource paths must have test coverage.

The R40 issue was not that "wide tiles are inherently unsuitable for TNCA," but that the existing lowering generated approximately 12× instruction bloat and massive row-by-row accumulator traffic. ([raw.githubusercontent.com](https://raw.githubusercontent.com/zzczzc20/TritonChip/r40/evidence/r40/v3/R40-03.gemm-tile/model_diagnosis/README.md))

### Scope Boundaries

Stage 1 does not mandate supporting all wide tiles at once:

- `16×32` is the mandatory repair scope.
- Larger tiles are supported progressively according to resource legality.
- Unsuitable shapes retain their original legal implementations.
- Falling back perpetually to `16×16` while claiming multi-fragment path support is prohibited.

### Layout and Allocation

Connect the execution resource plan to existing:

- LinearLayout transform classification;
- Fragment/pack;
- Window lifetime;
- Bank-aware allocation;
- Retile/refusal.

Constructing a generic optimal scheduler or a full spill subsystem is not required in this stage.

### Acceptance

1. At least two standard Torch graphs genuinely execute the repaired multi-fragment path.
2. Numerics, asynchronous ownership, and resource accounting pass.
3. Generated code and dynamic instruction counts demonstrate that pathological transfers have been eliminated.
4. Re-measure the entire original GEMM-tile subset.
5. Separately report:
   - Repair gains relative to the buggy lowering;
   - Real gains relative to the original conservative `16×16` production path.

Do not repackage "fixing a pathological candidate" as an equivalent speedup across the entire product.

---

## S1-03: Synchronization, Wide Blocks, and Budgeted Autotuning

### A. Synchronization Fixes

Audit the mapping from `tl.debug_barrier()` to TNCA ordering.

- Weaken or eliminate synchronization only when required ordering can be formally proven.
- Globally turning barriers into no-ops is strictly prohibited.
- Provide negative test cases covering cross-role, store/reload, aliasing, and asynchronous read/write scenarios.

The R40 softmax regression contained redundant device fences; certain tiny points exhibited variations due to staged-buffer memory placement under identical machine code, which must not all be attributed to the compiler. ([raw.githubusercontent.com](https://raw.githubusercontent.com/zzczzc20/TritonChip/r40/evidence/r40/v3/R40-01.device-time/regression_diagnosis.md))

### B. Minimum Viable Implementation of TNCA_TUNE

Required:

- Integrated into the standard Inductor/TNCA selection path.
- Candidates depend on semantics, shapes, strides, dtypes, resources, and capabilities—not benchmark names.
- A conservative legal configuration is always present.
- Default to a small candidate pool (recommended ≤4); the hard limit remains at 16 measured candidates per kernel.
- Per-graph tuning incremental budget is capped at 10 minutes; baseline compilation costs are accounted for separately.
- Worker timeouts are actively terminated; device tasks properly drain and recover.
- Tuning samples are isolated from formal measurements.
- Cache keys cover code, configuration, hardware, and memory-mapping identities affecting legality and selection.
- Simulator wall time must not be used to select configurations.

Initial support may focus on finite candidates:

- Narrow / wide blocks;
- Original tile / repaired multi-fragment tile;
- Mode B / single-role pipelining / existing Mode A;
- Legal fused / decomposed paths;
- Single / batched submission.

Searching the Cartesian product of all configuration switches at once is not required.

### C. Policy Safety

- When measurement budget is exhausted, results are inconclusive, or models are out-of-domain, retain conservative configurations.
- Selection failures must not degrade into CPU fallbacks.
- Mechanisms not proven beneficial remain disabled by default.
- Stage 1 does not mandate that autotuning achieve an overall 1.1× speedup, but it must deliver genuine selection, caching, budgeting, and error handling.

### D. Locality/DCO Preparation

Stage 1 completes:

- Generic consumer design;
- Mapping between storage/alias/lifetime and policy;
- Which existing control signals or instructions will be executed;
- Rejection rules for unsafe conditions;
- Stage 2 board validation plan.

Merely setting an environment variable without a functional consumer does not constitute integration.

---

## S1-04: 100 MHz Candidate Design and Full Model Feasibility Preparation

This work package proceeds in parallel, but does not require Stage 1 to load a new FPGA image.

### A. Hardware Candidates

Starting from M1–M4 of `r40-final-image`:

- Register host grant;
- Register host_allow;
- Localize and register memory controls;
- Fix float-to-int per ISA §8.4.

These fixes have candidate validation, but have not been deployed on board in a new image. ([raw.githubusercontent.com](https://raw.githubusercontent.com/zzczzc20/TritonChip/r40-final-image/evidence/r40/v3/final_image_candidate/README.md))

Further address timing by **structural categories** rather than patching hundreds of paths individually:

- Global `stall_front` and high-fanout enables;
- Scalar read → barrier/scoreboard → issue;
- VRF bank/collector/writeback;
- FMA input selection and pipeline balancing;
- LSU/mesh interface;
- Slice RAM controls.

Prioritize:

- Local control signal duplication;
- Register boundaries with explicit valid/ready semantics;
- Local credit/queues;
- Reducing cross-unit combinational feedback.

**Simply delaying stall by one cycle without managing in-flight instructions, replay, flush, and ownership is prohibited.**

### B. Required Deliverables

- Known critical path categories, owning modules, candidate modifications, and validation plans.
- Related RTL regressions after merging M1–M4.
- Out-of-context (OOC) / unit synthesis and regional resource preflight.
- Fixed candidate list for the next full design implementation.

A uniform positive slack margin across OOC runs is not required, nor may OOC be equated to routed timing closure.

### C. Qwen Feasibility Preparation

Complete in Stage 1:

- Distinct storage statistics for official weights and language branch;
- Required operator and state dtype inventory;
- Compilation and functional checks for representative graphs: D256 attention, depthwise conv, FP32 recurrence;
- Code size, compilation time, IR, and memory peak estimations;
- Inventory of differences between public model interfaces and cache adapters.

Trial runs on full Qwen may occur early, but are not blocking items for Stage 1 sign-off.

---

## S1-05: Stage 1 Sign-off

`make r41-stage1-acceptance`

Required:

1. Required deliverables from S1-00 through S1-04 are completed.
2. Final software passes full original suite numerical regression on the existing 90 MHz image.
3. Multi-fragment fixes are actively executed; delivering dead or unused code is prohibited.
4. Tuning is officially integrated with functioning conservative fallbacks.
5. performance_harness completes three closed-loop case studies.
6. New default policies satisfy the non-regression rules in Section 6.
7. Rebuild software and models from a clean archive with fresh caches, and re-execute a bounded set of frozen board points.
8. Publish Stage 1 report and Stage 2 ready/blocker list.

**Stage 1 does not wait for a new bitstream.**

---

# Stage 2: 100 MHz New Image, Full Model, and Scale Validation

## S2-00: Freeze Stage 2 Minimum Delivery Set

Freeze prior to formal implementation:

- Final model revision, weights, and tokenizer;
- Newly added core model points;
- XL-core;
- Numerical tolerances;
- Mandatory hardware modifications;
- Candidate study list;
- Compilation, board time, tuning, and implementation budgets;
- Final performance baselines and comparisons.

By default, schedule at most:

1. One integrated implementation run;
2. One corrective implementation run based on routing results.

No unrelated features may be added following the first implementation round.

If 100 MHz is not achieved after exhausting the implementation budget, submit an explicit failure status and next steps rather than rewriting the contract by reducing frequency.

---

## S2-01: Three-Core 100 MHz Physical Delivery

### Required

1. Three physical cores continue to reside across three SLRs.
2. The new image incorporates M1–M4 and selected timing fixes.
3. Operates at an actual 100 MHz frequency.
4. Post-route setup, hold, pulse-width, and project CDC audits pass.
5. Does not strip existing SFU functionality, precision, or semantic breadth to trade for timing closure.
6. Binds final image hash, source code, synthesis/implementation parameters, tool versions, and actual clock frequency.
7. Contains no unexplained routing or physical design errors.
8. Completes at least a 2-hour continuous mixed run; may be merged with formal campaigns provided that session and resource growth logs are preserved.

An 8-hour endurance run is an extended target and not a prerequisite for initial delivery.

### Decoupling Frequency from Performance

In an idealized case where device cycles remain identical, moving from 90 to 100 MHz corresponds to approximately an 11.1% speedup; however, additional pipeline stages may alter cycle counts.

Therefore, report concurrently:

- Actual completion wall time;
- Device cycles;
- Overhead of additional pipeline stages and resources.

Frequency scaling must not be automatically claimed as compiler or memory optimization gains.

---

## S2-02: Memory Service and MMA — Bounded Candidates, Selective Integration

### A. Memory Service

Prioritize studying home-slice and core-side services:

- Resource footprint of lookup/hit/fill/wake/writeback;
- Decoupling request admission from completion;
- Small, bounded queues;
- Hit-under-miss and fairness;
- Command admission in the transfer engine;
- Outstanding request demands from actual consumers.

Do not prematurely turn the effort into "more memory ports, more MSHRs, wider buses."

In R40 port experiments, performance models indicated that a single outlet was not fully saturated, and adding outlets did not relieve home engine and core-side stalls; this should serve as the starting point for candidate screening, without presuming that all new workloads share the exact same bottleneck. ([raw.githubusercontent.com](https://raw.githubusercontent.com/zzczzc20/TritonChip/r40/evidence/r40/v3/model_attribution/README.md))

### B. Transfer Engine

Implementing a small command queue is permitted initially, but must distinguish:

- Ability to admit multiple commands;
- Ability to overlap setup overhead;
- Ability to overlap data transfer;
- Whether it genuinely shortens the graph's critical path.

**Merely shifting waiting from the issue stage into a queue does not qualify as throughput optimization.**

### C. MMA and XLS

Evaluate at least one RTL candidate targeted at an identified bottleneck, such as:

- Operand buffering / reuse;
- Scaling parallelism along independent M/N outputs;
- Accumulator banking / porting;
- FMA pipeline balancing;
- Effective utilization for skinny-M/GEMV.

The first round does not require jumping from 16 MAC/cycle to a large array scale.

Candidates must preserve:

- Existing four MMA fragment formats;
- Five-format FMA / conversions;
- Per-K rounding;
- `source-free` / `result-ready`;
- Independent numerical oracle.

XLS is an implementation technique, not a sign-off achievement. The generated microarchitecture must demonstrate what it improves, accompanied by reports of actual FPGA mapping; omitting integration is permitted if no superior trade-off is found.

### D. Study Closure Conditions

Each candidate must provide:

- Hypothesis and supporting evidence;
- SystemC sensitivity experiments;
- Relevant RTL semantic verification;
- Area and timing preflight;
- Rationales for integration or rejection.

Candidates included in the final image must incorporate standard Torch on-board comparisons.

**It is not required that memory, MMA, and pipelining each independently achieve 1.5×.**

---

## S2-03: Locality, Pipelining, and Private Resource Rebalancing

### A. Minimum Viable Locality/DCO Delivery

Required:

- At least two model-agnostic standard Torch graphs genuinely exercise supported locality/DCO operations.
- Policies are traceable to storage, aliasing, and lifetimes.
- Unsafe aliases and incomplete writers/readers must not use aggressive dead/bypass hints.
- Numerical correctness, ownership, and fault paths pass across off/on modes.
- On-board memory traffic and latency comparisons are provided.

If there is no performance gain, the functionality can still be delivered while keeping the policy default-off.

### B. Pipeline Comparison

Compare under fixed core, context, and total register budgets:

- Mode B;
- Single-role software pipelining;
- Mode A two-stage ring.

A 3-stage ring is added only if models or performance counters indicate necessity; it is not a mandatory baseline combination.

### C. Private Resource Rebalancing

Incorporate asymmetric producer/consumer resource demands into research, adhering to the following priority:

1. Improve existing legal window/lifetime allocation;
2. Evaluate static, per-kernel asymmetric resource allocation;
3. Investigate runtime register re-allocation only if strictly necessary.

NVIDIA's `setmaxnreg` relies on physical register pooling, execution consistency, and synchronization rules; it must not be misconstrued as an capability that the TNCA compiler can attain simply by altering a budget constant. ([docs.nvidia.com](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html))

Any new TNCA resource allocation capability affecting RTL or ABI must explicitly declare its capability flag. Existing RPAIR aliasing does not equate to adding register capacity.

This item is a Study; introducing a dynamic resource allocation ISA is not mandatory.

---

## S2-04: Capacity, Long Lifecycles, and Full Qwen Text Inference

### A. Model Scope

Required:

> The full language branch of the official Qwen3.5-2B checkpoint, with authentic weights, all 24 layers, full vocabulary, BF16 backbone, and reference-mandated FP32 states, completing prefill and continuous autoregressive decode.

Excludes:

- Vision encoders and multimodal inference;
- Multi-token prediction (MTP);
- Speculative decoding;
- Primary quantization sign-off.

The official configuration contains 18 linear attention layers and 6 full attention layers, with a head dimension of 256 for full attention and FP32 recurrent-state dtypes. It cannot be substituted by generic Llama + KV support. ([huggingface.co](https://huggingface.co/Qwen/Qwen3.5-2B/raw/main/config.json))

The official implementation provides text model classes containing standard Torch compute and external kernel replacement hooks; the formal path must audit the actual executed branches. ([raw.githubusercontent.com](https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/models/qwen3_5/modeling_qwen3_5.py))

### B. Core Model Workloads

To govern initial delivery scale, establish explicit IDs rather than reusing draft names Q0–Q3:

| ID | Batch | Prefill | Continuous Decode | Classification |
|---|---:|---:|---:|---|
| `Q41-core` | 1 | 128 | 16 | Required |
| `Q41-tail` | 1 | 129 | 4 | Required |
| `Q41-512` | 1 | 512 | 128 | Target |
| `Q41-2k` | 1 | 2048 | 256 | Target |
| `Q41-8k` | 1 | 8192 | 64 | Target |
| `Q41-b4` | 4 | 512 | 128 | Target |

This classifies the **new, not yet frozen model scope**, leaving all original R40 frozen points untouched.

Full layer count, weights, and vocabulary remain intact; initial delivery does not mandate concurrently completing long contexts, long decode runs, and batch scaling.

### C. Implementation Sequence

Establish a legitimate B41-C first, then optimize:

1. Primitives;
2. Single-layer and state updates;
3. Full `Q41-core`;
4. Full `Q41-tail`;
5. Optional extensions.

The correctness baseline may employ generic decompositions without forcing maximal fusion upfront.

Key requirements:

- FP32 recurrent states must not be downgraded in precision to utilize existing BF16 MMAs.
- Do not force the use of fused SDPA known to be slower in R40.
- When claiming tiled/linear memory attention, separate proof of allocation and traffic must be supplied.
- Do not retain historical per-token recurrent states indefinitely.
- Within a fixed decode bucket, position updates must not trigger per-token recompilation.

### D. Capacity

Plan implementation around 6–8 GiB reachable capacity, with hard requirements evaluated against actual peak usage:

\[
C_{\text{usable}} \ge 1.25\,H_{\text{peak}}
\]

Peak usage encompasses:

- Distinct weights and tied aliases;
- KV cache, recurrent states, and conv states;
- Activations, scratchpad, and pack cache;
- Code store, descriptors, reserved regions, and allocator overhead.

Do not mistake total checkpoint file size for language-branch memory footprint, nor treat it as per-step decode read traffic. The official index includes weights outside the language branch. ([huggingface.co](https://huggingface.co/Qwen/Qwen3.5-2B/raw/main/model.safetensors.index.json))

Required:

- Genuine coverage of large sizes/offsets and >4 GiB addresses;
- Reserved memory-map regions are never overwritten by tensors;
- Negative test cases for high-bit truncation, out-of-bounds, and illegal aliasing;
- Bounded grid admission;
- Execution of >65,535 programs via standard Torch;
- Verification across >65,536 barrier arm cycles;
- Code eviction, allocator reuse, and zero state corruption across consecutive calls;
- No implicit host retransmission of steady-state weights or states.

### E. Numerics

Before freezing Stage 2, determine new tolerances via cross-validation against independent references; tolerances must not be relaxed after observing TNCA outputs.

Verify:

- Logits;
- Various state tensors;
- Teacher-forced continuous evolution;
- Reset and re-invocation;
- Tails, masks, and extreme gating values.

In teacher-forced tests, TNCA and reference receive identical tokens while maintaining their respective internal states independently.

Free generation serves as a functional demonstration; matching output strings cannot substitute for numerical verification.

---

## S2-05: Bounded XL, Original Suite, and Final Selector

### A. XL-core

Do not build an open-ended large benchmark matrix.

Propose freezing 8 primary points:

| Category | Proposed Primary Point |
|---|---|
| Large storage streaming | Single input storage >2 GiB, real reads/writes reaching high addresses |
| Transpose | `16384×16384 BF16`, genuinely materialized output |
| GEMM | `M=512,K=2048,N=6144,BF16` |
| GEMV | `M=1,K=2048,N=6144,BF16` |
| Full output projection | `M=1,K=2048,N=248320,BF16` |
| Embedding / gather | Large tables, int64 indices, duplicate indices, cross-region access |
| Attention | D256, GQA, causal, ragged sequence lengths |
| Gated recurrence | FP32 states, chunk boundaries, continuous updates |

Exact inputs, shapes, strides, tolerances, and hashes are written into the Stage 2 manifest.

Cross-combinations of additional dtypes, sequence lengths, and batch sizes are extensions and do not automatically become required.

### B. Original Suite

All original native77, large24, HBM inventory, and 4-layer model workloads are preserved.

If any original large point fails, the large suite cannot be claimed as passed.

### C. Holdout

- Retain original R40 holdout inventory and historical states.
- Explicitly flag any points that were involved in strategy development.
- Freeze a new, compact, unseen R41 holdout set during Stage 1.
- Compare conservative configurations against actual selector outputs in final evaluation.
- Regret is measured solely against evaluated legal candidates, without pretending to represent global optimal regret.

Required is correct selection, budget adherence, and complete execution. `S_geo ≥ 1.1` and p90 regret ≤ 15% remain Targets.

### D. Multi-Core Scaling

- Validate active cores 1, 2, and 3 on the same final image.
- Perform primary scaling comparisons only on pre-registered representative graphs and one core model point.
- Sweeping all core counts across every XL point, model point, and candidate is not required.
- Maintain existing parameterized regressions; new 8/16-core functional campaigns and 32-core planning do not block 3-core delivery.

---

## S2-06: Final Sign-off and Independent Reproduction

`make r41-stage2-acceptance`

Required:

1. 100 MHz new image qualifies under real physical criteria.
2. All Required functional and numerical checks pass.
3. Full original suite and XL-core completed.
4. Full `Q41-core` and `Q41-tail` completed.
5. All enabled new performance mechanisms have corresponding board measurements and default policy justifications.
6. Update SystemC profiles for the final image, re-reporting valid domains and error margins; identical accuracy across all domains is not required.
7. Rebuild software and models from a clean archive with fresh caches.
8. Re-execute representative legacy points, one high-address point, and full `Q41-core` in an independent session.
9. Archive final hardware build toolchain and reports without mandating an additional redundant place-and-route run to reproduce an identical bitstream.
10. Report all unmet targets, negative results, and disabled mechanisms.

---

## 6. Baselines, Timing, and Performance Evaluation

### 6.1 Baselines

| Name | Definition |
|---|---|
| **B40-OBS** | Actually delivered R40 v3 software, image, and 90 MHz clock |
| **B41-S1** | Stage 1 final software on the legacy image |
| **B41-C** | Generic correctness/capacity baseline required to run new workloads legitimately |
| **B41-F** | Stage 2 final software, image, and policy |

B41-C must not intentionally disable existing effective optimizations to manufacture an artificially slow denominator.

For new models that cannot run on B40-OBS, computing an "infinite speedup" relative to it is prohibited.

### 6.2 Comparisons

- Compiler A/B: identical image, identical clock, identical resources.
- Hardware A/B: identical software strategy, reusing the same artifact whenever feasible.
- If an iso-frequency hardware comparison cannot be established, report "joint hardware and frequency migration" rather than falsely asserting an independent hardware contribution.
- Generating an additional expensive bitstream merely to complete an aesthetic ablation is not required.
- Construct small interaction matrices only for primary mechanisms that demonstrate genuine coupling.

### 6.3 Timing

Separately report:

- Cold start;
- Resident graph makespan;
- Resident call latency;
- Request latency / TTFT;
- Complete decode step.

Completion must represent genuine execution completion.

Account for actually incurred:

- Code reload;
- Repacking;
- State updates;
- Token selection;
- Mandatory synchronization;
- Data copies within formal boundaries.

Model wall time is used solely to evaluate harness throughput and cannot be used to rank FPGA candidates.

### 6.4 Sampling Budget

- In each stage, first verify device cycle stability using a small set of anchor points.
- Stable long device tasks permit a modest number of complete samples rather than mechanically executing dozens of iterations.
- Points near decision boundaries receive supplementary runs per pre-registered rules.
- Host latency uses interleaved repetitions and must not borrow determinism assumptions from device cycles.
- Full models require at least two independent sessions, one of which may serve as independent reproduction.
- When token sample counts are insufficient, report descriptive statistics only, without asserting a reliable p95.
- If gains cannot be confirmed after exhausting the budget, record `inconclusive` rather than continuing to sample until statistical significance is reached.

Clearly distinguish between bootstrap confidence intervals across points and noise intervals across repeated runs; do not conflate the two.

### 6.5 Non-Regression Rules for New Default Optimizations

This rule governs R41 only and does not alter historical R40 verdicts.

- Original suite numerical correctness is fully Required.
- Across the frozen matched performance set, an overall non-regression budget of at most 5% is permitted.
- Individual regressions >5% must be reported item by item.
- Any newly confirmed regression >10% blocks the corresponding performance optimization from being enabled by default; it must be fixed, routed to a legal conservative path, or reverted.
- Mandatory semantic fixes documented prior to freeze record their costs separately and must not be portrayed as performance optimizations.

Explaining the cause alone is not an acceptable justification for retaining severe default regressions.

### 6.6 Performance Targets Without Hard Package-by-Package Multiple Gates

It is recommended to retain the following Targets:

| Item | Target |
|---|---:|
| Repaired GEMM strategy vs. legal conservative path | Target subset ≥1.15× |
| Memory service improvements | Target subset ≥1.2× |
| Qwen optimizations vs. legal B41-C | Target prefill/decode each ≥1.2× |
| Final selector holdout | ≥1.1× |
| SystemC overall predictive qualification | Strive for median ≤10%, p95 ≤20% |

Speedup claims may only be asserted for portions supported by actual measurements.

Do not mandate:

- Every optimization must show ≥3% LOO;
- Every memory direction must achieve a fixed percentage of theoretical peak;
- Mode A must win;
- Fused attention must win;
- XLS candidates must be merged;
- New matrix units must reach a predetermined speedup multiple.

---

## 7. Minimum Ablation Set: Preserving Causality While Governing Experiment Count

Mandatory:

1. Matched A/B comparisons for every major default-enabled mechanism.
2. Leave-one-out (LOO) evaluations for the two to three primary mechanism groups in the final system.
3. Small matrices for mechanisms exhibiting actual interactions.

Prioritized interactions:

- Fragment/layout × allocation;
- Pipeline × matrix;
- Cores × memory service.

Add the following only when relevant claims are asserted:

- Batch × weight reuse;
- Locality × cache policy;
- Role resource allocation × pipeline.

Pre-registering necessary interaction groups is permitted, without requiring every auxiliary tweak within a group to independently clear a gain threshold.

Prohibited:

- Bundling all mechanisms into a single switch without attribution;
- Linearly summing speedups across disparate features;
- Rerunning the entire large test suite for every minor modification.

---

## 8. Execution Rules to Prevent Process Drift

### 8.1 Limiting Concurrent Workstreams

Stage 1 primary workstreams:

1. Fragment / codegen;
2. performance_harness;
3. Budgeted selection.

Timing closure and model feasibility proceed in parallel without blocking the release of the primary workstreams.

Stage 2 primary workstreams:

1. Timing closure and integrated image;
2. Full model and capacity;
3. Selected bounded service / MMA improvements.

### 8.2 Candidate Admission

New candidates must articulate:

- Which measured bottleneck is targeted;
- What result would refute the hypothesis;
- Projected modification and verification scope;
- Whether a new bitstream is required;
- Feasibility within the current stage budget.

"Major refactorings" lacking clear bottleneck evidence must not enter the critical path of the active stage.

### 8.3 Experiment Stop Rules

When research exhausts its pre-registered budget, it must select one of:

- Merge;
- Default off;
- Abandon;
- Carry forward unfinished items to the next stage.

Do not convert closeout into open-ended exploration by continuously adding candidates and measurements.

### 8.4 Post-Freeze Modifications

- Implementation bugfixes and diagnostic additions are permitted.
- Re-acquire affected evidence after modifying the product.
- Final claims must bind to final identities.
- Deleting slow points, truncating required workloads, relaxing tolerances, or swapping denominators is strictly prohibited.
- When Required items are incomplete, the stage may be closed as failed, but the contract must not be altered to mark them as passed.

**The purpose of a deliverable contract is to constrain mandatory commitments within a tractable scope, not to guarantee that every outcome can be stamped PASS.**

---

## 9. Files, Commands, and Evidence

Proposed minimal new layout:

```text
doc/r41_tasks_and_acceptance.md
doc/r41_stage1_report.md
doc/r41_report.md

spec/r41/carryover.json
spec/r41/stage1/plan.json
spec/r41/stage1/lock.json
spec/r41/stage2/plan.json
spec/r41/stage2/lock.json
spec/r41/stage2/model_lock.json
spec/r41/stage2/xl_core.json
spec/r41/holdout.json

evidence/r41/stage1/
evidence/r41/stage2/
```

Common implementations of `performance_harness` remain located in:

```text
tools/performance_harness/
tools/performance_harness/systemc/
```

Do not dump reusable models, runners, and comparators into `tools/r41/`.

Proposed entry points:

```sh
make r41-freeze-check STAGE=1
make r41-harness-check
make r41-stage1-acceptance

make r41-freeze-check STAGE=2
make r41-stage2-acceptance

make r41-acceptance
```

Acceptance checkers reject at least:

- Missing original large points;
- Truncating layers or replacing model weights;
- CPU numerical fallbacks;
- Descriptor enabled but standard path unexecuted without stated justification;
- 90 MHz records masquerading as 100 MHz;
- Model predictions masquerading as board measurements;
- Submission timestamps masquerading as completion timestamps;
- Tuning runs contaminating formal measurement samples;
- Fitting parameters on holdout sets;
- Mismatched images, invalid configurations, or copying historical PASS records.

---

## 10. Definition of Done

### Stage 1 Completion

> On the legacy 90 MHz 3-core image, critical compiler lowerings are repaired and original workloads remain correct; budgeted selection is genuinely integrated; and `performance_harness` bridges hypotheses, real compiled artifacts, models, RTL, and board measurements into a reproducible, rapid iteration loop.

### Stage 2 Completion

> The 3-core new image satisfies physical 100 MHz requirements; the full language branch of official Qwen3.5-2B executes frozen core workloads via the standard Torch path; the full original suite and XL-core pass; enabled optimizations are grounded in empirical evidence; and the domain of validity and limitations of the performance models are explicitly documented.

### R41 Completion

> All Required criteria in both stages are passed. Performance Targets may remain unachieved, but unmet portions must be reported explicitly without masquerading as realized speedups.

---

**The core trade-off of this contract is: never compromise authenticity or correctness, but eliminate the expectation that "all research must succeed" and "all dimensions must be swept comprehensively in one iteration."**

Architecturally, the highest-leverage interventions are three cohesive modifications rather than an omnibus switch:

1. **Mapping logical tiles directly to resident physical fragments**, establishing the foundation for GEMM efficiency, reduced register pressure, compact code size, and attention;
2. **Transforming long combinational feedback into localized, bounded, observable service interfaces**, serving timing closure, home-slice throughput, and transfer admission;
3. **Harmonizing execution resource descriptions across compiler, SystemC, and selector while preserving independent correctness oracles**, enabling every optimization to swiftly expose when a bottleneck has merely shifted.

This realistic scope is far more likely to produce a deliverable R41 than concurrently promising wider ingress, larger MMA arrays, 3-stage Mode A, dynamic register allocation, globally high-precision simulation models, and exhaustive long-context matrices.
