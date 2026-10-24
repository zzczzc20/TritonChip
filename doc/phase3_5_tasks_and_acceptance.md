# Phase 3.5: Reusing Upstream Infrastructure and Delivering Observable Compiler Optimizations

Version: 1.0, 2026-09-20. Status: **PLANNED / NOT ACCEPTED** (execution tracked in [r35_status.md](r35_status.md)).
Inspection baseline: `03c88fad3f9482a22b1ded1aaf183b84709b6abc`.
Prerequisite: Assumes [R3 v2](phase3_tasks_and_acceptance.md) is signed off upon entering this phase (`R3_V2_CORE_VRF_ACCEPTED`); interface constraints: [R2 interfaces](r2_interfaces.md); design basis: [compiler overview](design/tnca_v1_compiler_draft.md).

**Objective: Reuse Triton LinearLayout and appropriate Triton/MLIR optimization infrastructure without adding SMEM, modifying the ISA, or increasing hardware resources, enabling the same program to generate better TNCA machine code and measuring attributable cycle reductions on the current small-scale RTL.**

This is an added phase and does not alter the sign-off criteria of R1, R2, or R3. The completion sign-off identifier is fixed to `R35_COMPILER_ACCEPTED`, which is not equivalent to a full MLIR backend, generic Triton GEMM, or production release.

## 1. Current Baseline and Scope Boundaries

Initial source inspection reveals:

- In `python/tnca/aot.py`, `opt_level=0..2` was explicitly marked as reserved and did not alter emission.
- The product main path was Triton TTIR → Python TNCA IR → codegen; `LinearLayout` / `toLinearLayout` was not directly utilized.
- The C++ TNCA dialect, async interfaces, and ownership verifier existed, but this does not imply that MLIR optimization results have entered product machine code.
- Existing `partition.py`, `pipeline.py`, `gemm.py`, and `report.py` are reusable; TNCA-specific scheduling policies do not equate to having reused upstream partition/pipeline passes.
- `compiler/pin/toolchain.json` records pinned Triton source and companion LLVM, while R1 acceptance additionally has a Triton wheel / distribution MLIR 20.1.2 path. Adding C++ dependencies must first resolve version boundaries.
- At drafting time, R3 v2 synchronous VRF integration was in progress; per the explicit precondition, this delivery plan assumes R3 v2 is signed off when entering this phase. It adopts its signed-off `core-vrf-sync-v1` profile and counter/drain interfaces, without treating hardware completion as a task of this phase. Newly added R3 v2 contracts/inventories in the workspace are outside the scope of this modification.

### 1.1 Mandatory Deliverables

1. Directly reuse the locked version of upstream LinearLayout implementation as the actual computation component for TNCA layout derivation.
2. The product compilation path genuinely executes MLIR canonicalization/CSE and consumes the optimized results.
3. Reuse Triton AxisInfo on the supported TTIR subset to pass proven facts such as contiguity and divisibility to TNCA.
4. TNCA-specific layout selection, bank-constrained register allocation, and bounded scheduling; all preserving asynchronous resource safety.
5. Frozen baselines, optimization modes, ablation experiments, numerical verification, actual RTL cycle and resource counters; at least two real workloads meeting the Section 6 gain threshold, with at least one gain causally attributable to LL-driven optimization.
6. Public compilation APIs after installation, artifact/cache/report/dump, and reproduction entry points.

### 1.2 Non-Goals / What This Phase Does Not Add

Does not add software-addressable scratchpad, does not secretly define NUCA regions as SMEM, does not add banks/ports/MACs/roles, does not alter floating-point numerical contracts. Nor does it require porting the entire TritonGPU pipeline at once, GPU warp specialization, SMEM swizzle lowering, general LICM, whole-loop software pipelining, or system-level autotuning.

The compiler may alter legal data placement and instruction scheduling, but must account for the cost of actual copies, transfers, and waits. Merely altering reports, deleting necessary waits, modifying baselines, or adding hardware do not count as performance gains for this phase.

## 2. Reused vs Self-Implemented Components

| Component | Reuse Scope | TNCA Self-Implemented Scope | Level in This Phase |
|---|---|---|---|
| Triton LinearLayout | Upstream layout representation and composition/mapping operations, invoked via thin adapter | VRF/lane/fragment/ARF semantics, legal conversions, and physical allocation | Must enter product path |
| MLIR canonicalizer + CSE | Directly invoke upstream passes, operating on supported pure-value IR | Import/export, effect boundaries, numerical and resource checks | Must have positive cases altering machine code |
| Triton AxisInfo | Pinned TTIR analysis, conservatively exporting proven facts | TNCA address/vectorization choices, ABI assumption constraints | Must have actual lowering decisions and negative cases |
| MLIR pass manager, verifier, diagnostics/dump | Manage the controlled pass pipeline above | Stage dumps, provenance tracking, post-optimization ownership checks | Mandatory |
| TNCA allocation / local scheduling | Use LL mappings and verified dependency information | Bank-aware allocation, window contiguity, return/credit costs | Mandatory |
| LLVM/MLIR liveness, dominance, or other analyses | Adopted only if directly applicable to production IR | Union with async lease live ranges | Optional; unused libraries cannot count as deliverables |
| LICM, SCF software pipelining, Triton remove-layout-conversions/partition passes | Compatibility exploration permitted | Target policy, loop/effect/communication legality | Deferred, not critical dependencies for this phase |

Upstream LL maps hardware locations to tensor indices and supports duplicate value positions; it is neither general address arithmetic nor a register allocator. Specific interfaces follow the repository's locked version of [LinearLayout.h](https://github.com/triton-lang/triton/blob/c01b6774b1865984607d89d89d3a10833de92037/include/triton/Tools/LinearLayout.h).

Analysis results from AxisInfo must be translated by an adapter into explicit proven/unknown facts, and hints must not be treated as facts holding for arbitrary dynamic pointers; see the locked version of [AxisInfo.h](https://github.com/triton-lang/triton/blob/c01b6774b1865984607d89d89d3a10833de92037/include/triton/Analysis/AxisInfo.h).

Generic optimizations rely on correct operation semantics. Implementation must follow [MLIR canonicalization](https://mlir.llvm.org/docs/Canonicalization/), [CSE and pass documentation](https://mlir.llvm.org/docs/Passes/#-cse), and [side effects/speculation interfaces](https://mlir.llvm.org/docs/Rationale/SideEffectsAndSpeculation/). The specific choices, gain metrics, and test matrices in this document constitute TNCA's engineering contract, not upstream performance guarantees for TNCA.

## 3. LinearLayout Contract in an Architecture Without SMEM

### 3.1 Distinguish Three Layers

```text
logical tensor indices + valid mask
              ↑ LL mapping
virtual register / lane / packed element / MMA fragment position
              ↓ TNCA allocation + window constraints
physical VRF register → bank / row; ARF slot / row
```

- **Role Context is an independent instruction stream and resource owner, not a warp.** Do not pack the role count into distributed tensor layout dimensions, and do not interpret inter-role communication using warp shuffle semantics.
- VRF's 16 lanes and FP16 packed elements are distinct coordinates; their correspondence must be explicitly defined. MMA fragment and ARF output coordinates are also modeled separately.
- Physical bank is determined by register ID after actual allocation. One cannot claim bank conflicts vanish solely from looking at virtual LL; if allocation is not a linear mapping, explicit allocation tables are used for evaluation, rather than forcing a representation as a GF(2) linear function.
- General strides, dynamic bases, address carries, and non-power-of-two boundaries are still handled by normal index/address IR. LL can operate on bounded padded tiles; legal masks are stored separately, and padding positions must never be treated as accessible memory.
- Broadcast/replication may not be bijective; ordinary inverse matrices cannot be unconditionally computed. Equivalent sources must be explicitly selected, and coverage and consistency across all valid outputs must be verified.

### 3.2 Conversion Legality and Costs

Every layout conversion must be categorized into one of the following classes and have lowering records:

1. **Pure compile-time rename/view**: No data movement needed; permitted only when elements and packing order seen by all consumers match, with 0 extra data instructions.
2. **Conversions achievable by existing instructions**: Register copy, legal lane operations, or existing tensor transfers; list actual instructions, scratch VRF registers, bytes, and waits.
3. **Unsupported by current ISA**: Return diagnostics with locations or retain the legal original layout. Silently generating SMEM, inventing lane exchange instructions, or invoking host rearrangement of output is strictly prohibited.

This phase initially covers identity, legal reshape/transpose combinations, broadcast, masked tail, FP16 MMA A/B fragments, and ARF result correspondence. Permutations that LL can represent do not automatically enter supported scope. Unsupported cross-lane permutations must have rejection test cases.

### 3.3 Layout and Bank Optimization Strategy

Candidate count per kernel ≤ 8; only one formal O2 plan is selected for the same source input. Candidate selection does not trigger RTL tuning. Scoring must include at least: added conversion instructions/bytes, actual bank request conflicts, live register peak, window contiguity/alignment, barrier/credit occupancy, and critical dependency length.

Retain legal baseline candidates as fallbacks. Unknown score items must not be set to 0; when predicted improvement lacks evidence, conservatively select baseline and record rationale. Plans must never be selected by table lookup on benchmark names, output data, or measured case cycles.

Fixed contiguous register requirements for shared windows, MMA fragment order, inter-role aliasing, and async leases are hard constraints. Local bank allocation must not violate these conditions. When no candidate satisfies constraints, return the original plan without increasing resources to accommodate it.

## 4. Product Compilation Path and Dependency Organization

### 4.1 Connecting the Narrow Path First Without Waiting for Complete R2 Migration

Allows two-stage connection:

```text
Triton TTIR → upstream analysis/canonicalization → supported TNCA IR
TNCA IR pure regions → small MLIR bridge → canonicalize/CSE → TNCA IR
                      ↓
       LL layout planning → allocation/scheduling → safety checks → emission
```

TNCA IR inputs must also reach LL and applicable pure-value optimizations. Low-level control flow need not be migrated to MLIR in this phase, but bridged results must replace actual emission inputs; running passes on copies and then discarding them is prohibited.

Pure-value regions have explicit input/output and effect boundaries. Load/store, DMA, MMA, barrier, window ownership, trap, and potentially faulting operations cannot be crossed by default and must not be marked Pure to obtain CSE. Only constant folding, integer address expression simplification, redundant pure expression elimination, and dead pure value elimination that do not alter numerical semantics are initially permitted. Load CSE across memory effects, speculative accesses, floating-point reassociation, or new FMA contractions are not in default optimizations.

Check input legality before optimization, check semantics/effects after optimization, and re-execute existing ownership/footprint/resource checks after completing physical allocation. When lowering register pressure, liveness must include asynchronous leases still held by DMA/MMA after issue.

### 4.2 Upstream Dependencies and Versions

- Preferred: Standalone `tnca-layout-opt` compile-time tool or equivalent isolated adapter, directly building and calling pinned upstream LL/analyses; passing only versioned MLIR text or data protocols, with no MLIR objects passed across ABI boundaries.
- The tool itself uses a matched set of Triton/LLVM; do not link distribution MLIR 20 and another pinned LLVM library together in one process. If changing to in-process bindings, toolchain consistency must be proven and the support matrix frozen.
- G0 records exact upstream commits, minimal source/library dependency closures, patches, license/NOTICE, build commands, and hashes. Do not automatically upgrade R1/R2 toolchains, and do not depend on `main` or unexported private Python symbols.
- If TTIR frontend and optimization tools differ in version, explicit text dialect compatibility verification is required; when incompatible, lock to the same matching frontend/compiler environment or provide controlled diagnostics, rather than assuming wheel-internal MLIR objects are directly interoperable.
- Compilers may depend on upstream tools; runtime artifact loading must not import Triton/LLVM/LL, nor require their installation.
- Only when upstream original implementations genuinely participate in layout computation may one claim "reusing LL". An independent Python oracle is used only for verification and must not become a covert replacement implementation in production.

### 4.3 Optimization Levels, Caching, and Reporting

| Mode | Definition |
|---|---|
| B0 | Existing product compilation behavior frozen at G0, including its existing optimizations; serves as a fair baseline, existing optimizations are not deliberately removed |
| O0 | Compatibility path of the new driver; machine code is byte-for-byte identical to B0 on frozen cases, and must not be artificially degraded aside from versioned metadata |
| O1 | Upstream canonicalization/CSE, proven AxisInfo lowering; retains legacy layout and allocation policies |
| O2 | O1 + LL-driven layout elimination/selection + bank-aware allocation + conservative local scheduling |
| O2-no-LL | Ablation: retains O1 and standalone-capable scheduling, turns off LL-driven layout/allocation decisions, uses legacy legal plans; no computation may be deleted |

Modes represent the implementation contract; `opt_level` must implement these behaviors. Explicitly requesting O1/O2 when dependencies are missing must fail; when an optimization is not applicable, retaining original code within that mode is permitted with not-applicable reasons reported.

Cache keys must include optimization level, pass pipeline version, LL upstream revision, target/implementation fingerprint, and code-affecting assumptions; dynamic buffer addresses and runtime alpha do not enter the compilation key. Diagnostic options do not affect code hash.

Reports output stage IRs, layout bases/valid masks, conversion classifications, actual physical bank allocations, live peaks/async occupancies, copy/instruction counts, selected and rejected plans, pass remarks, and compile elapsed time. Analytical model predictions, static instruction counts, and RTL measured cycles are reported separately.

## 5. Interface with R3 and Implementation Sequence

- **Starting baseline**: R3 v2 signed-off `core-vrf-sync-v1`, port/queue contracts, counters, runtime, and drain semantics. G0 verifies sign-off evidence and freezes hardware assets, without re-undertaking R3 implementation work.
- **Implementation sequence**: LL adapter and toolchain isolation → pure-value optimization / AxisInfo → layout and bank-aware allocation → safety differentials → observable gain gates on frozen hardware. Components can proceed in parallel, but final acceptance has no "hardware not ready" waiver.
- All primary gains and LL ablations are measured on the signed-off banked profile. The legacy `core-sim` serves only for compatibility regression and cannot substitute for actual finite-port measurements.
- Performance pairings must use identical signed-off RTL binaries / hardware fingerprints; B0 and O2 change only compiler behavior. If hardware changes, corresponding R3 verification must be completed first, establishing a new baseline before re-running formal pairings; hardware gains cannot be attributed to the compiler.
- Does not wait for R2 four-tile runtime, full MLIR control flow, or dual-environment production releases. This phase uses verified single-core capabilities and preserves genuine `source_kind`.

## 6. Small-Scale Performance Experiments and Hard Acceptance Thresholds

### 6.1 Six Frozen Workloads

G0 constructs the following sources from existing programs / supportable operators, freezing source hash, input seed 42, memory mapping, mode, mask, source path, and oracle. Hand-crafting poor B0 assembly or inserting dead copies to manufacture wins is strictly prohibited; before optimizing, prove the source expresses actual computation requirements.

| ID | Fixed Workload | Input and Sizes | Primary Observation |
|---|---|---|---|
| L01 | masked AXPY | Real Triton; N=17, BLOCK=16, grid=2; fast | Dynamic scalar, tail block, address overhead |
| L02 | Two outputs sharing one intermediate | Real Triton; N=16, grid=1; both outputs verified; fast | CSE/address and mask reuse; neither output can be deleted |
| L03 | Legal tensor view/layout composition, then elementwise | TNCA IR; 4×4 FP32, grid=1; fast | LL conversion elimination and physical bank allocation; independently enumerate element mappings |
| L04 | masked reduction | Existing supported input path, frozen at G0; 16 stored elements, 15 valid; fast | Dead pure-value elimination, live range, reduction order unchanged |
| L05 | FP16 GEMM | TNCA IR; 16×16×16, single role; fast | TensorMap→VRF→MMA fragment layout and conversion costs |
| L06 | FP16 GEMM pipeline | TNCA IR; 16×16×32, two roles / 2-slot; fixed slow/reorder | Real bank conflicts, window lifetime, scheduling |

L02 formula is frozen at G0, e.g. `t=x*alpha; y=t+b; z=t-b`, sharing requirements derived from source without inserting duplicate assembly to artificially inflate gains. L03's legal view combinations must first align with current ISA; if pure rename is already zero-cost, it remains as a non-regression control and baseline is not forced to move data. L03/L06 are predetermined ablation subjects and must be reported regardless of whether they win.

### 6.2 Sole Execution Matrix

| Group | Runs | Description |
|---|---:|---|
| Primary comparison | 12 | Six workloads × B0/O2, all on the same banked candidate |
| Ablation | 4 | L03/L06 each run O1, O2-no-LL; O2 run shared with primary comparison |
| Legacy profile compatibility regression | 4 | L01/L02 × B0/O1, on the same `core-sim` binary; compatibility only, not a gain gate |
| Post-installation dynamic parameters | 3 | L01 O2, same artifact, same session, three sets of alpha / buffer; compiled once |
| Total | **23 kernel launches** | All programs in the grid count toward cycles; no hidden loop repetitions |

Additionally permits at most 24 component/negative RTL invocations, all logged. Other layout, IR, mutation, and random differentials execute on host without multiplying three seeds across the entire RTL. Total added RTL runs ≤ 47, cumulative elapsed time ≤ 2,400 s / environment, ≤ 1,500,000 cycles; single core invocation ≤ 180 s / 50,000 cycles, single component invocation ≤ 60 s / 10,000 cycles. Waveforms off by default; failure traces ≤ 256 cycles / 10 MiB. Inherited regressions are accounted for separately and cannot be dropped via budget manipulation.

RTL simulation is slow; this phase does not add GEMM K64/K128, large matrices, four tiles, or large autotuning sweeps. Software candidate search ≤ 8 per kernel, default total sweep ≤ 48, ≤ 60 s without launching RTL.

### 6.3 Performance Acceptance Criteria

Define `C` as device cycles from **launch accepted to all result stores committed and resource drain completed**; ROI reported separately for explanatory purposes only and cannot substitute for total C. Initial memory/reset/cache states, inputs, outputs, hardware, grid, and resource capacities are identical. Different plans may have different actual resource occupancies and request timings, which must be reported.

Primary comparison must simultaneously satisfy:

1. **All six workloads correct**: B0 and O2 match independent oracles, tail/canary/error/drain checks pass. GEMM and reduction preserve existing accumulation/reduction order; fast-math trading precision is prohibited.
2. **At least two distinct workloads show measured gains**: Each `C_B0 - C_O2 ≥ max(8 cycles, ceil(0.05 × C_B0))`; at least one is not the pure layout micro-benchmark L03.
3. **At least one workload shows causal LL attribution**: On predetermined L03 or L06, `C_noLL - C_O2 ≥ max(4 cycles, ceil(0.03 × C_noLL))`; accompanied by matching evidence of copy instruction reduction or accepted request / conflict stall changes. All other switches identical on both sides; O1 gains cannot be attributed to LL.
4. **No meaningful regressions**: For each workload, `C_O2 - C_B0 ≤ max(4 cycles, ceil(0.02 × C_B0))`; total cycles across six workloads do not increase. Slower cases must be fully reported; geometric mean alone is prohibited.
5. **Gain sources explainable**: Every winning case must correlate with at least one physical change—dynamic instructions, copy bytes, bank conflict stalls, dependency waits, or external requests; static instruction reduction without C reduction does not count as a Section 6.3 condition 2 gain.

The 5% and 3% thresholds are phase targets to be achieved, not measured speedups or guarantees. G0 evaluates existing code for opportunity; if unmet, measurements and failures are retained, marked `OPTIMIZATION_DELIVERED / PERFORMANCE_NOT_MET`, continuing compiler fixes or proposing justified scope adjustments for the next version. Workloads must not be altered, sizes inflated, thresholds lowered, or B0 intentionally degraded after the fact to turn failures green.

This phase does not demand 2× speedup. On short kernels, eliminating a single port stall or a few dynamic instructions is valuable, but must meet the verifiable thresholds above.

### 6.4 Compilation Overhead

Toolchain and simulator builds accounted for separately. For warm helper and cache miss, O2 compilation for each of the six workloads ≤ 30 s; median compilation time must not exceed `max(2 × O1 median, 1 s)`; first helper startup logged separately and not mixed into hardware cycles. Each workload measured across 3 repetitions for compilation only, without running RTL; medians taken and all records retained. Cache hits must prove via call counts that frontend/LL/optimizer were not re-executed.

## 7. Work Packages and Acceptance Standards

All IDs prefixed with `R35-`; responsible roles assigned to individuals during execution. Every sub-condition must be executed; no percentage-based acceptance.

### R35-01 — Dependencies, Baseline, Opportunity, and Inventory (Maintainer / Compiler)

| ID | Mandatory Condition |
|---|---|
| R35-BASE-01 | Lock B0 source/revision, six workloads, 23-run matrix, hardware hash, budget, and toolchain; candidate implementation source does not modify benchmark oracles; inventory verified complete by independent checker |
| R35-BASE-02 | Establish table of applicable IR, real upstream symbols/libraries, inputs/outputs, and effect boundaries for each pass; list expected gain mechanisms and current blockers; unknown throughput not counted as zero |
| R35-BASE-03 | Baseline executable on candidate hardware; O0 and B0 machine code identical; hardware changes, missing ablations, inflated sizes, or mixed profiles rejected prior to simulation launch |

### R35-02 — Upstream Component Adapters and Installable Build (Compiler / Build)

| ID | Mandatory Condition |
|---|---|
| R35-UP-01 | Directly execute pinned upstream LL/MLIR/AxisInfo, with traceable implementation provenance; minimal build and post-installation self-tests pass without relying on developer absolute paths |
| R35-UP-02 | Triton/LLVM version mismatches, bridge protocol mismatches, missing/abnormal helpers report stable errors; O2 does not silently fall back to no-LL; no cross-version MLIR ABI mixed linking |
| R35-UP-03 | Bridge output genuinely enters emission: controlled mutations of layout/pure-value results must be caught by designated checker or code diff; invoking passes but discarding output does not pass |

### R35-03 — LinearLayout Semantics and Legal Lowering (Layout / Verification)

| ID | Mandatory Condition |
|---|---|
| R35-LL-01 | Section 3 layouts match independent enumeration oracle element-by-element; fixed seeds {1, 7, 42}, ≤ 64 host cases per seed, covering power-of-two tiles, valid tails, broadcasts, packed FP16, and A/B/ARF mappings |
| R35-LL-02 | Composition/conversion results cover all valid elements without duplicate writes or missing outputs; non-bijective handling, dynamic stride/mask, and unsupported lane permutations have explicit positive and negative cases; role is not used as a warp/layout dimension |
| R35-LL-03 | Every conversion classified as rename / real instruction / refused; actual copy/bytes/live temporaries verifiable from emitted code; SMEM allocation, phantom shuffles, or host output fixups prohibited |

### R35-04 — Pure-Value Optimization and AxisInfo (MLIR / Frontend)

| ID | Mandatory Condition |
|---|---|
| R35-OPT-01 | Canonicalization/CSE genuinely executed on product O1/O2 paths; redundant pure values / address computations from at least one non-synthetic source eliminated, changing machine code; pre- and post-pass dumps match remarks |
| R35-OPT-02 | AxisInfo drives at least one legal address/vector access lowering choice; falls back to conservative plan when alignment/contiguity cannot be proven; dynamic unaligned, tail, and aliased inputs do not inherit erroneous assumptions |
| R35-OPT-03 | CSE/DCE does not delete or cross store/atomic/DMA/barrier/trap/potential fault; floating-point NaN/Inf/signed-zero/subnormal and existing FP contracts pass regression; no additional reassociation/contraction enabled |

### R35-05 — Bank-Aware Allocation and Bounded Scheduling (Backend)

| ID | Mandatory Condition |
|---|---|
| R35-ALLOC-01 | Cost model uses final physical register → bank mapping; same-bank and cross-bank directed cases match R3 interfaces; pure renames charge 0 moves, actual moves fully accounted for |
| R35-ALLOC-02 | Window contiguity/alignment, aliasing, ARF order, peak registers, and barrier/credit within original budget; async leases extend liveness; pressure exceeding budget yields diagnostics or legal baseline plan, with no hidden spills |
| R35-SCHED-01 | Scheduling preserves def-use, memory/effect, control, accumulator-order, and async-resource edges; reorders only provably independent local instructions; retains legal baseline when gain is unproven or unknown |

### R35-06 — Safety Differentials and Checker Counter-Examples (Verification)

| ID | Mandatory Condition |
|---|---|
| R35-SAFE-01 | All affected R1 OWN/FP, frontend rejection, R2 ABI/cache/artifact, and R3 single-core safety cases pass on new compiler; all optimization levels invoke final physical ownership verifier |
| R35-SAFE-02 | Host differentials verify positive and boundary negative cases for each pass; dynamic parameters not incorrectly consteval'd; renaming functions/registers/symbols does not alter optimization choices or results |
| R35-MUT-01 | All specified mutations caught by designated detectors: wrong LL basis → LL-01; dropped tail mask → LL-02; effectful op marked Pure → OPT-03; early release of async value → SAFE-01; bank ignored in cost model → ALLOC-01; opt level omitted from cache key → PKG-02; model cycle passed off as measured → PERF-01; forged PASS / hidden slow case → EV-01 |

Mutations use isolated copies; clean baseline must pass; unrelated build failures do not count as detection. Mutations requiring RTL count against component budget and do not duplicate the full performance matrix.

### R35-07 — Real Gains and Ablation (Performance / Verification)

| ID | Mandatory Condition |
|---|---|
| R35-PERF-01 | 23 launches executed per inventory; raw cycles, numerical values, canaries, drain, hardware/code hashes, and actual elapsed time complete; R3 banked precondition valid; host wall time or model predictions cannot substitute for device C |
| R35-PERF-02 | All five conditions of Section 6.3 satisfied simultaneously; complete tables of B0/O2 across six workloads and L03/L06 ablations published; at least two measured winners, at least one LL causal attribution holds; failed cases must not be deleted |
| R35-PERF-03 | Section 6.2 and 6.4 limits on sizes, run counts, cycles, wall time, candidate search, and compilation overhead passed; low-budget injections verify watchdog and process cleanup |

### R35-08 — Product API, Cache, Report, and Dump (Product Compiler)

| ID | Mandatory Condition |
|---|---|
| R35-PKG-01 | Non-editable install, arbitrary cwd, repo not in sys.path; public API O0/O1/O2 execute correctly; L01 O2 across three dynamic launches yields correct results, sharing 1 code hash, 1 device process, 0 extra compiler calls |
| R35-PKG-02 | Opt level, pass/LL revisions, hardware profile, and code assumptions change cache key; identical inputs recompile deterministically; corrupted cache rejected; runtime-only install loads artifact without importing compiler dependencies |
| R35-OBS-01 | Dumps include input/optimized IR, LL decisions, actual allocations, and final assembly; report matches real emission; debug/dump does not alter code; unapplied optimizations report reasons; GEMM `source_kind` not mislabeled as Triton |

### R35-09 — Regression, Reproduction, and Evidence Sign-off (CI / Maintainer)

| ID | Mandatory Condition |
|---|---|
| R35-REG-01 | Frozen impact scope regressions all PASS; shared compiler changes affecting legacy core-4r/system paths rerun relevant existing cases, without expanding new performance matrix into full-system sweeps; prior phase evidence does not substitute for candidate regression |
| R35-EV-01 | Verifier recalculates gains, regressions, budgets, and verdicts from raw values/counters/exit codes against frozen inventory; rejects mixed source/hardware/toolchain, modified cycles, missing ablations, tampered logs, or modifying summary PASS alone |
| R35-REL-01 | Locked primary Linux environment completes full gate; second independent Linux build reproduces LL/IR/safety tests and L03/L06 B0/O2/O2-no-LL with identical deterministic code/values/cycles; sign-off tables, raw evidence, known boundaries complete |

Second environment does not share build artifacts, but may share read-only download cache; does not require full replication of 23-run matrix, nor does this claim completion of original R1/R2 dual-environment production releases. Other primary environment winners do not add new matrices in the second environment, but complete replayable evidence must be retained.

## 8. Delivery Artifacts, Entry Points, and Acceptance Order

Planned landing locations:

- `compiler/.../layout_adapter`: Upstream LL isolation layer; `analysis_adapter`: AxisInfo / proven facts.
- `compiler/.../optimization`: Pure-value bridge, pass pipeline, TNCA layout and allocation policies.
- `spec/r35_{toolchain,workloads,required_cases}.json`: Versioned contracts and inventories.
- `tools/r35/`: Runners, differentials/mutations, performance accounting, independent verifier.
- `doc/r35_status.md`: Records status against the 27 standards of this document, initially NOT VERIFIED.

Entry points:

```sh
make r35-scout
make r35-check-plan
make r35-layout-tests
make r35-opt-tests
make r35-safety-tests
make r35-perf
make r35-package-tests
make r35-acceptance
make r35-verify-evidence EVIDENCE=<run-directory>
```

Execution registry provides unified deduplication; missing tools, missing prerequisites, unimplemented detectors, and fail/skip/xfail/timeout all prevent sign-off. R&D reruns retain past failures and generate new run IDs; overwriting records is prohibited.

Evidence layout must at least include:

```text
contracts/     Contract hash, B0 source, toolchain, hardware/layout/numerical/resource contracts
inventory/     Primary 23-run, components, ablations, inherited regressions, second-environment subset
build/         compiler/upstream/helper/RTL/wheel hashes and build commands
ir/            Per-stage input/output IR, LL bases, valid masks, allocations, remarks
code/          B0/O0/O1/O2/no-LL code, artifacts, static resource comparisons
runs/          Raw commands, numerical oracles, exit codes, counters, cycles, elapsed time
performance/   Per-case gains/regressions, LL ablations, counter attributions, compile overhead
mutations/     Clean vs mutated results, designated detectors, proof of unmodified candidate source
inherited/     Impact regressions for current candidate, R3 prerequisite verification references
acceptance.json  contract_id=r35-compiler; criterion-by-criterion verdicts, budgets, environment
```

| Gate | Completion Condition | When Not Satisfied |
|---|---|---|
| A: Dependencies and LL semantics | R35-01/02/03 | No automatic layout optimization wired in |
| B: Genuine optimization and safe | R35-04/05/06 | Formal gain gate not run; fix semantics/ownership first |
| C: Observable gains | R35-07, completed on accepted R3 hardware | Unrun marked NOT VERIFIED; gains unmet marked PERFORMANCE_NOT_MET, no sign-off |
| D: Product and evidence | R35-08/09 | Nothing claimed deliverable |

**Only when all 27 standards and their sub-conditions, two measured gains, the LL causal gain, non-regression constraints, budgets, and reproductions pass simultaneously is `R35_COMPILER_ACCEPTED` emitted.**

Subsequent engineering follow-up after acceptance connects to [Phase 3.6: Multi-backend Engineering Integration and Prototype Branch Handover Preparation](phase3_6_tasks_and_acceptance.md). R3.6 integrates common product chains and backend interfaces, without fixing an FPGA target or taking on physical board bring-up; concrete platform implementations proceed in independent branches.

Future expansion of compiler and hardware capabilities: after proving the narrow path effective, expand MLIR main path coverage, loop optimizations, and software pipelining; integrate with R3B synchronous ARF for a complete layout/cost model; and consider cross-tile execution once R2 system runtime is established. The current phase does not use these distant roadmap goals to delay delivering single-core compiler gains.
