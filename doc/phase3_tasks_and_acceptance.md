# Phase 3 (R3 v2): Single-Core Finite-Port VRF Integration Closed Loop

Version: 2.0, 2026-09-20. Status: **PLANNED / NOT ACCEPTED**.
Inspection baseline: `71317b35afb9d2602b0510da84188a0fbde15e29`.

Basis: [Current state review](r3_replan_review_2026-09-20.md), [Architecture design](design/tnca_v1_architecture_draft.md), [Compiler design](design/tnca_v1_compiler_draft.md).
This document supersedes the original Phase 3 execution plan; the [original v1 contract](phase3_tasks_and_acceptance_v1.md) is preserved. Original R1/R2 contracts remain unchanged.

**RTL simulation is slow. The new phase implements only a single single-core candidate, a small suite of small programs, and directed component tests. Scope is narrowed without lowering correctness, backpressure, ownership, or evidence standards for the retained features.**

## 1. Phase Objectives and Sign-off Boundaries

This phase answers only one question: **Can the existing synchronous banked VRF and operand collector be connected to a real core, correctly execute small programs under contention from all VRF clients and restricted return and writeback conditions, and be driven via the post-installation single-core runtime?**

Final deliverables:

1. An explicit opt-in single-core experimental implementation using a real synchronous, finite-port VRF; the legacy implementation is retained as default and differential reference.
2. SIMD, LSU, tensor/DMA, matrix, and all other actual VRF clients access storage exclusively through controlled ports, with no hidden direct read/write paths.
3. Finite collector, return reservation, writeback arbitration, reset, and fault draining; asynchronous ownership does not regress.
4. Numerical, resource conservation, cycle, and port evidence for six small workloads, as well as multiple launches with post-installation dynamic parameters.
5. Structural synthesis evidence for the genuinely integrated VRF and a stage gate capable of rejecting missing or falsified evidence.

The sign-off milestone identifier is fixed to **`R3_V2_CORE_VRF_ACCEPTED`**. It is not equivalent to the legacy `R3 ACCEPTED`, R2 sign-off, or production release.

### 1.1 Scope Boundaries

| Mandatory in This Phase | Deferred to Follow-up; Does Not Count as PASS in This Phase |
|---|---|
| 8 KiB VRF synchronous read, 4 banks, 1R1W per bank, 1-cycle read latency | ARF synchronous memory refactoring; retained and explicitly marked as behavioral in this phase |
| Single core, 2 Role Contexts; all VRF clients governed by the same resource contract | 4-role candidate, four-tile candidate, mesh/NUCA performance models |
| Timing adaptation and real writeback backpressure for SIMD/LSU/tensor/MMA | New MAC arrays, dual-issue, throughput tuning, and comprehensive unit peak performance commitments |
| Use existing TNCA IR / Triton adapter with post-installation runtime | Full MLIR main path, generic Triton GEMM lowering |
| Small GEMM, AXPY, vector, and reduction verification | softmax / RMSNorm / LayerNorm new composite operators and large shape sweeps |
| Exact comparison against port/collector micro-model, actual core cycle measurements | Whole-kernel model error thresholds, automatic layout optimization, system holdout set calibration |
| Single locked Linux full gate + directed reproduction on a second environment | Full dual-environment production release from original R1/R2/legacy R3 |

The choice of fixed 4 banks / 1 cycle reuses existing components and minimizes integration variables. Existing 2-cycle component tests are retained, but no second release candidate is added. Deferring ARF does not allow bypassing VRF to add an infinite read port for MMA.

### 1.2 Dependencies and Regression Rules

- Historical sign-off of R1 only proves `97c0cf90`. Every piece of evidence in this phase must bind to the new candidate commit; historical verdicts cannot be directly read to claim candidate pass.
- Does not wait for full completion of R2; depends on its existing single-core compile/artifact/ABI/session path. Missing capabilities necessary for this phase must be implemented; incomplete R2 cannot be used as an excuse to skip them.
- New experimental profile proposed ID `core-vrf-sync-v1`, inheriting resources from `core-sim`: 2 RCTXs, identical ISA/register capacity/SFU; implementation differences recorded in fingerprint. Default behavior of `core-sim`, `core-4r`, and `system-2x2` remains unchanged.
- The new profile must still support inherited ISA functionality. The six workloads serve only as directed integration evidence and must not be used to silently drop other instructions; existing single-core ISA regressions must execute on the candidate.
- Legacy profiles in the current commit must run existing core fast/slow/ooo regressions, compiler/ownership/footprint checks, artifact read-only checks, and profile verifications. The candidate must run equivalent single-core applicable regressions along with the cases in this document.
- G0 must freeze the specific ID, command, implementation, and rationale for each inherited case. When shared files change, relevant core-4r/system regressions must also run; only items modifying exclusively the new profile with no shared semantic changes may be omitted from this phase's inventory, accompanied by a source impact explanation. Cases must not be deleted due to elapsed time or failures.
- Inherited regressions and newly added budgets are accounted for separately and deduplicated uniformly. Phase sign-off does not claim that original R1/R2 are fully re-signed-off on the current candidate; formal product release still executes the original contracts.

## 2. Current Baseline and the Real Critical Path

Currently, banks, VRF, collector, micro-models, counters, and synthesis tooling exist; core still uses behavioral RF, and ARF still reads asynchronously. In `r3-flow-tests`, FLOW-03 explicitly fails due to unimplemented return/writeback tests.

Execution sequence is fixed to: **Acceptance inventory and interface freezing → Reset semantics and VRF wrapper → Integration of all clients → Return/writeback and ownership → Small-program differentials → Post-installation closed loop → Evidence sign-off.**

Do not prematurely add model candidates, operator varieties, or full-system scale. The earliest reviewable deliverable of this phase is "core genuinely uses finite ports," not additional standalone module PASSes.

## 3. Frozen Resource and Timing Contracts

G0 delivers the implementation specification, containing at least the following fields and verified via readback from elaborated instances:

| Item | Requirement for This Phase |
|---|---|
| VRF capacity | 128 × 16 × 32 bit = 65,536 bit = 8 KiB; 2 roles each with 64 logical registers; physical aliasing preserves existing contracts |
| bank | 4 banks; each bank 32 × 512 bit; lowest 2 bits of physical register select bank, remaining bits select row |
| ports | At most 1 accepted read and 1 accepted write per bank per cycle; replicating data arrays to add read ports is prohibited |
| synchronous read | Request accepted at clock edge t; data consumable at t+1 at earliest; fully define valid/ready and data holding under downstream stall |
| collector | Reuse existing 4 entries / at most 3 sources per instruction; instruction issues only after all sources are collected; each accepted read has tag and return destination |
| writeback | Enumerate all write clients; fixed depth of FIFO/holding, maximum in-flight transactions, arbitration policy, and credits declared item-by-item; unbounded queues prohibited |
| RDW/mask | Fixed write-first; bypass only for accepted writes, merged lane-by-lane; rejected writes must not take effect early |
| reset | Preserve zero-initialization behavior of existing architecture; valid metadata zeroing or bounded scrub permitted; resetting the entire data array is prohibited, as is converting legacy zero reads to X |
| ARF | Retain existing behavioral implementation and accumulation order, reported as behavioral; does not count toward "synchronous storage completed" |
| drain | halt/successful completion must not precede completion of all accepted reads/writes, asynchronous operations, and held leases; error paths must release or explicitly poison session |

Bits for data memory, valid, collector, bypass, and FIFOs must be itemized separately. Implementation of valid/mask must not invalidate lane initialization semantics under partial writes.

Fairness assumes bounded downstream ready: freeze contender count C, maximum grant burst duration B, arbitration pipeline latency A; continuously requesting contenders wait at most `(C-1)×B + A`. Fixed completion time cannot be promised without external forward progress; in that case, watchdog must trigger and output stall reasons. If the implementation uses beat-by-beat arbitration, B is computed per beat without treating an entire DMA transfer as permanent occupancy.

## 4. Small-Scale Workloads and Execution Costs

### 4.1 Sole Newly Added Core Inventory

Each item runs exactly two implementations, reference and candidate, for **12 executions total**. Seed fixed to 42; no Cartesian expansion across fast/slow/ooo; each run executes one kernel once. Reference is the legacy implementation within the same candidate tree and cannot reuse results from another commit.

| ID | Fixed Workload | Input Path / Memory | Verification Purpose |
|---|---|---|---|
| C01 | FP32 vecadd, N=16, grid=1 | Existing compiler; fast | Basic operand collection, SIMD writeback |
| C02 | FP32 masked AXPY, N=17, BLOCK=16, grid=2 sequential single-core execution | Real Triton adapter; fast | Tail block, dynamic alpha, LSU/writeback; does not claim dual-tile |
| C03 | FP16 GEMM, 16×16×16, single role | TNCA IR; fast | tensor→VRF→MMA→store numerical closed loop |
| C04 | FP16 GEMM, 16×16×32, serial schedule | TNCA IR; fixed slow/reorder | Two K16 steps, return contention baseline |
| C05 | FP16 GEMM, 16×16×32, two roles / 2-slot pipeline | TNCA IR; identical addresses, inputs, memory config, and seed as C04 | Window reuse, concurrent clients, resource release |
| C06 | FP32 reduction, 16 stored elements, 15 valid elements, 1 row | TNCA IR; fast | Mask, dependency chain, read/write destinations |

Both C04 and C05 freeze identical available register/window/barrier budgets. Different schedules may alter external request timings; save actual return traces without claiming that identical seed makes two traces identical cycle-by-cycle. C05 is permitted to be slower than C04; reasons must be disclosed without artificially inflating K in pursuit of cosmetic speedups.

**Newly added GEMM upper bounds: M=N=16, K≤32; no batching, no hidden repetitions.** Reduction at most one row of 16 elements; elementwise N≤32, grid≤2. Existing larger shapes in legacy regressions are neither deleted nor copied into the newly added performance matrix.

Numerical acceptance criteria: GEMM follows existing ISA increasing-k fused FP32 accumulation contract, compared against an independent exact oracle; vecadd/AXPY/reduction compare against existing instruction rounding and reduction orders. Mutual consistency between reference and candidate is insufficient; both must match independent oracles. Inspect full outputs, trailing sentinels, canaries on both sides, fault=0, and final resource drain.

### 4.2 Post-Installation Runtime Cases

Run C02 three additional times on candidate only: identical artifact, alpha values 2.5, -1.0, 0.25; separate output buffers; single session, single device process. Code hash unchanged, extra compiler invocations after initial compilation = 0. Counted toward newly added RTL, totaling **15 core kernel executions**; full R2 matrix is not rerun.

### 4.3 Budget Rules

| Category | Execution Cap | Single Wall-Clock / Cycle Limit |
|---|---:|---|
| Core comparison + post-installation execution | 12 + 3 = 15 kernel runs | 180 s / 50,000 cycles |
| Component and negative cases, including mutation RTL baseline/mutant runs | 40 RTL invocations | 60 s / 10,000 cycles; ≤256 accepted requests per scenario |
| Newly added RTL total | ≤55 runs; persistent processes also counted per kernel | Cumulative elapsed ≤1,800 s; cumulative ≤1,000,000 cycles / full environment |
| Micro-model tests | Does not launch RTL; ≤32 config combinations | 30 s |
| Structural synthesis | Selected VRF/collector/adapter and integrated structural checks | 1,200 s total |

The above are engineering budgets, not measured achievements for this run. G0 runs reference C01/C03 on locked Linux as scouts; records with matching commit/artifact/config can directly take two of the formal 12 slots. Candidate must be measured separately. Build times are recorded; build and simulation budgets are accounted for separately; compile/setup have independent global watchdogs, frozen by G0 before formal runs begin.

Subprocess elapsed times are summed; total wall-clock time after parallel execution must not conceal costs. Startup, upload, execution, and download of persistent processes are charged once; cycles of each kernel are registered independently. Retries preserve original failures without overwriting; debug runs use separate directories and bounded budgets, and cannot use their PASS to overwrite a formal run's FAIL.

Waveforms disabled by default; diagnostic captures at most one failing case's 256-cycle window, 10 MiB trace. Timeouts clean up entire process groups; reduced dummy budgets verify timeout mechanisms without waiting half an hour.

If G0 finds budgets insufficient, first identify initialization, logging, wasteful loops, and simulation build overhead; if still unsatisfied, submit a revised budget basis in a new version before formal runs, keeping current version NOT ACCEPTED. Post-hoc threshold modification to turn failures into passes is forbidden.

## 5. Work Packages and Explicit Acceptance Criteria

All new test cases use `R3V2-*` prefix to avoid confusion with legacy PLAN/RF/SYN IDs in older scripts. Each requirement must map to real commands, assertions, exit codes, and artifacts; initial status is universally NOT VERIFIED.

### G0 — Baseline, Interfaces, Manifests, and Gate Skeleton

Deliverables: Versioned profile, resource contracts, 15-kernel manifest, component inventory, inherited regression inventory, responsible roles, tool locks, and budgets. Assignees: Maintainer / Verification.

| ID | Non-waivable Acceptance Criteria |
|---|---|
| R3V2-BASE-01 | All cases have ID, implementation, source/artifact hash, profile, seed, dimensions, memory mode, and budget; mandatory set determined by independent frozen inventory, not verified post-hoc by whatever appears in result files |
| R3V2-BASE-02 | Checker rejects missing candidate, duplicate reference, unknown profile, memory mode masquerading as implementation, zero/negative/out-of-bounds dimensions, duplicate cases; all fail prior to launching RTL |
| R3V2-BASE-03 | Linux scout has real commands, exit codes, cycles, and elapsed times; candidate resource contracts contain no mandatory nulls; historical R1 serves solely as provenance, current commit inherited regression plan is complete |

### G1 — VRF Wrapper, Initialization, and Storage Semantics

Deliverables: Synchronous bank interface, valid/zero scheme, and standalone protocol model. Assignees: RTL / Verification.

| ID | Non-waivable Acceptance Criteria |
|---|---|
| R3V2-RF-01 | Capacity, 4 banks, reverse address mapping, role physical aliasing, and boundary checks consistent; accepted reads/writes per bank per cycle each ≤1, all responses arrive exactly once |
| R3V2-RF-02 | Bounded sequences with seeds {1, 7, 42} cover masks, RDW, conflicts, consecutive requests, read response stalls; initialized lanes match standalone model item-by-item; directed inputs guarantee each property is actually triggered |
| R3V2-RF-03 | Cold reset, read-after-partial-write, relaunch, in-flight transactions during reset, zero/write contention conform to frozen contract; stale epoch returns cannot pollute new state; architecturally unwritten contents still read zero, underlying X not mistaken for valid output |

### G2 — Integration of All Clients into Real Core

Deliverables: New profile top/parameters, timing adaptation for each read client, and port arbitration. Assignees: RTL.

| ID | Non-waivable Acceptance Criteria |
|---|---|
| R3V2-CORE-01 | Elaboration/structural reports prove new top active datapath instantiates banked VRF; itemize SIMD/LSU/tensor/MMA/other clients; no legacy RF shadow copies, hierarchical direct reads, or extra combinational datapaths |
| R3V2-CORE-02 | Three-source reads from same bank must grant serially, cross-bank reads may proceed in parallel; instructions do not issue until all sources return; paused response leaves payload/tag stable, value and destination entry correct upon resumption |
| R3V2-CORE-03 | Each read client has at least one directed program triggering real stall; all clients can complete independently and participate in contention; candidate cannot be passed by disabling MMA/tensor/second role |

### G3 — Writeback, Return, Ownership, and Bounded Progress

Deliverables: Arbitration across all write clients, finite return holding, credits/epochs, and draining assertions. Assignees: RTL / Compiler Verification.

| ID | Non-waivable Acceptance Criteria |
|---|---|
| R3V2-FLOW-01 | SIMD, DMA, and other clients concurrently writing same bank in same cycle: all accepted transactions eventually commit once; unaccepted retain payload and retry; full FIFO prevents issuing transactions lacking return capacity |
| R3V2-FLOW-02 | Under continuous requests and ready downstream, per-request wait conforms to frozen fairness bound; completely drains after fixed bounded backpressure; issued/accepted/returned/committed/cancelled conserved, credits eventually returned |
| R3V2-FLOW-03 | Reusing collector entries, flushes/faults interleaved with late returns do not corrupt data; leases released from real last-consumption/completion events; inherited OWN/FP positive/negative cases hold on candidate, cannot free early at issue |

### G4 — Six Small Programs and Post-Installation Single-Core Closed Loop

Deliverables: Controlled workload sources, standalone oracles, artifacts, execution logs, and post-installation scripts. Assignees: Compiler / Runtime / Verification.

| ID | Non-waivable Acceptance Criteria |
|---|---|
| R3V2-E2E-01 | C01–C06 × 2 implementations (12 runs total) all executed, passing numerical, mask, canary, fault, and drain checks in Section 4.1; GEMM source_kind must be genuine tnca-ir |
| R3V2-E2E-02 | Wheel installed non-editable, repository not in sys.path, arbitrary cwd; new profile prepare/compile/run/download succeed; 3 dynamic alpha launches share one device process, numerically correct, code hash and compiler invocation count conform to Section 4.2 |
| R3V2-E2E-03 | Two-way mismatch between reference artifact and candidate device rejected prior to launch 0; stale incorrect cache not used; pre-existing artifacts loadable with only runtime and run assets installed, without requiring source checkout |

### G5 — Verifiable Counters and Finite Micro-Model

Deliverables: Port/collector/writeback event definitions, program timelines, micro-model, and discrepancy reports. Assignees: Performance / Verification.

| ID | Non-waivable Acceptance Criteria |
|---|---|
| R3V2-COUNT-01 | Bank read/write accepted, conflicts, collector wait, writeback wait, in-flight/credit match independent short traces event-by-event; primary stall attributions mutually exclusive and sum to total stall; overlapping view listed separately |
| R3V2-COUNT-02 | Each core case reports cycle boundaries for launch, first issue, last store committed, drain; counters windowed to identical boundaries; outputs host/build/sim wall, total cycles, ROI cycles, without deriving GHz from simulation speed |
| R3V2-MODEL-01 | Serial read, cross-bank, same-bank, queue full, return backpressure, post-flush reuse: micro-model matches RTL cycles/counts exactly across these six directed patterns; predictions saved with hashes beforehand; backfilling model from measurements prohibited |

Candidate is not required to be faster than reference. All cycle increases disclose waiting causes; unexplained residuals marked unknown. Kernel-level predictions serve only as experimental records with error bounds; this phase does not sign off on whole-program/NUCA/NoC model accuracy.

### G6 — Structural Synthesis and Inherited Regressions

Deliverables: Memory/port inventory at real integration hierarchy, component synthesis reports, inherited regression evidence. Assignees: RTL / CI.

| ID | Non-waivable Acceptance Criteria |
|---|---|
| R3V2-SYN-01 | Candidate VRF data is exactly 65,536 bits, with ≥90% in synchronous memory objects; all memory ports/clocks/masks conform to profile; valid/collector/FIFO listed separately, without including legacy ARF bits in this ratio |
| R3V2-SYN-02 | New banks, collectors, adapters, and associated control modules have no unexplained latches, multiple drivers, combinational loops, or width issues; integration hierarchy/port connectivity checks pass; labeled strictly memory-inferred, physical area/Fmax/power marked NOT MEASURED |
| R3V2-REG-01 | Frozen candidate single-core regressions and legacy profile affected regressions from G0 all PASS; legacy default profile preserved; no skip/xfail from missing dependencies, no regression in ownership/numerical/error completion semantics |

Time-consuming full-system flattening and generic gate mapping are not mandated; local synthesis cannot substitute for integration connectivity checks, nor claim that ARF is synchronized.

### G7 — Negative Cases, Evidence Verification, and Phase Sign-off

Deliverables: A non-interactive gate, standalone verifier, evidence directory, and sign-off table. Assignees: Verification / Maintainer.

| ID | Non-waivable Acceptance Criteria |
|---|---|
| R3V2-MUT-01 | All mutations in table below detected by designated semantic checkers; clean baseline passes; unrelated syntax/compilation failures do not count as detection; injected into isolated copies, source hash unchanged |
| R3V2-EV-01 | Standalone verifier recomputes verdict from mandatory inventory and raw results; rejects missing cases, altered results/logs, mixed commit/profile/artifacts, omitted slow cases, fabricated cycle/wall summaries; cannot trust written PASS fields |
| R3V2-REL-01 | Primary Linux environment passes all mandatory items; second independent build environment completes specified smoke; budgets, tool locks, assignees, and artifacts complete; all G0–G7 conditions satisfied simultaneously before outputting R3_V2_CORE_VRF_ACCEPTED |

Mandatory mutations to detect:

| ID | Mutation | Designated Detector |
|---|---|---|
| M01 | Read advanced by one cycle / return from wrong bank | RF-02, CORE-02 |
| M02 | Drop writeback reservation or ignore full | FLOW-01/02 |
| M03 | Stale tag writes into new entry after flush | FLOW-03 |
| M04 | Drop one masked write or early source-free | RF-02 / FLOW-03; execute both sub-mutations |
| M05 | Double-count stalls / omit one accepted count | COUNT-01 |
| M06 | Model permits infinite reads to same bank | MODEL-01 |
| M07 | Missing candidate, increased K, fabricated elapsed time, missing evidence | BASE-02 / EV-01; execute all four sub-mutations |

Detector abbreviations in table refer to `R3V2-` prefixed IDs in this document. Both sub-mutations for M01/M05 are executed; RTL costs count toward the 40 component quota; host-only negative cases do not launch RTL.

## 6. Environment, Entry Points, and Evidence

Primary environment: Locked Linux x86_64, adhering to R1 tool versions and recording newly added dependencies for this phase. Second environment can be another independent Linux build directory/container or macOS development machine; may share only read-only download caches, not build artifacts. Second environment runs RF/collector directed tests, candidate for C02/C03/C05, and post-installation 3 dynamic launches; numerical results and deterministic counters must match, wall times recorded separately. Platform discrepancies must be diagnosed, not ignored.

The second environment is a cross-environment smoke test for this phase and **cannot masquerade as the full dual-Linux release acceptance of the legacy contract**. Its execution count and budget are audited independently against a sub-manifest, not merged with the primary environment to pad coverage.

The following entry points are interfaces to be implemented; this document update did not create these Make targets:

```sh
make r3v2-plan                 # Verify contracts/manifests/blast radius, without launching RTL
make r3v2-components           # Directed tests for bank/collector/return
make r3v2-core                 # Six workloads × reference/candidate
make r3v2-runtime              # Post-installation candidate profile, 3 launches
make r3v2-synth                # Memory-preserving + integrated connectivity checks
make r3v2-acceptance           # Uniformly deduplicate and execute all mandatory items in one run registry
make r3v2-verify-evidence EVIDENCE=<run-directory>
```

Sub-target passes do not imply overall sign-off; the master gate must detect unimplemented checkers and return non-zero. Missing dependencies, unrun tests, timeouts, and unexpected fail/skip/xfail/xpass cannot be signed off. Successful reruns must generate a new run ID, preserving failure history.

Evidence directory contains at least:

```text
contracts/     Contract version/hash, profiles, port/queue/reset/fairness contracts
inventory/     15 core runs, components, mutations, inherited regressions, second-env sub-manifest
build/         commit/clean-tree, source and runtime asset hashes, tool locks, hierarchy and instance parameters
runs/          Real command, exit code, budget, logs, oracle, counters, short traces per run
runtime/       Wheel source, installation location, artifacts, device/compiler invocation counts
model/         Micro-graph inputs, a priori prediction hashes, event-by-event comparisons
synthesis/     Memory ports/bits, control state bits, integration checks, unmeasured physical metrics
mutations/     Clean vs. mutant results, specific detectors, proof of unchanged sources
inherited/     Per-case results and blast-radius rationale; cannot simply cite historical ACCEPTED
acceptance.json  contract_id=r3-v2-core-vrf; environment, per-item results, budgets, derived verdict
```

All 24 `R3V2-*` criteria in the acceptance inventory must have evidence; a single test may prove multiple criteria, but every subcondition of each criterion must have independently readable inspection results. All work packages record actual assignees, completion commits, and evidence paths upon implementation; sign-off is never based on percentages or "code already written".

## 7. Execution Gates and Subsequent Roadmap

| Gate | Must Hold First | Cannot Claim Prematurely |
|---|---|---|
| Gate A: G0/G1 | Profile, initialization, ports, and budgets verifiable | Core is already using banked VRF |
| Gate B: G2/G3 | All-client integration, return/writeback bounded progress | Small programs / product entry points have passed |
| Gate C: G4/G5 | Numerical comparison, post-installation execution, port counters | Whole-kernel model is calibrated |
| Gate D: G6/G7 | Structure, regressions, negative cases, evidence, and environments complete | Project production readiness or legacy R2/R3 signed off |

Disposition of legacy contract requirements must remain traceable in subsequent plans:

| Legacy R3 Work Package | Retained in New Phase | Deferred Delivery |
|---|---|---|
| R3-01/11/12 | Accurate inventory, budget, mutation, partial sign-off in G0/G7 | Full dual-environment system release |
| R3-02/09 | Port/collector micro-model in G5 | Full compiler schedule mapping, kernel holdout accuracy, NUCA/NoC |
| R3-03/04/10 | VRF integration and structure in G1/G2/G3/G6 | Synchronous ARF and full storage system contracts |
| R3-05/06/07 | Client timing, local counters, implementation fingerprint | Full unit throughput, automatic bank-aware layout, and cost feedback |
| R3-08 | Six small workloads and single-core runtime | BF16/tail/Norm new performance matrix, four-tile product closed loop |

Following acceptance of this phase, [Phase 3.5](phase3_5_tasks_and_acceptance.md) is added: reusing LinearLayout and compiler optimization tooling on frozen accepted hardware to deliver observable single-core performance gains. It does not depend on completion of synchronous ARF or four-tile runtime.

Hardware follow-up remains R3B (ARF and single-core model closure), followed by R3C (system calibration and release) once the R2 four-tile runtime is genuinely complete. These directions require independent tasks & acceptance contracts; PASS in this phase or Phase 3.5 cannot be automatically backfilled into legacy contracts.

**Only when Gates A–D, all 24 criteria, all subconditions, budgets, and environment evidence pass simultaneously can R3 v2 be signed off. Current status remains NOT ACCEPTED.**
