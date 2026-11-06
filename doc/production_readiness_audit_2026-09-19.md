# TritonChip Current State Audit and Production Readiness Roadmap

Audit date: 2026-09-19. Baseline: `6c90063` (workspace audit, not an isolated clean checkout).

Design basis: [TNCA-v1 Architecture Draft](design/tnca_v1_architecture_draft.md), [TNCA-v1 Compiler Draft](design/tnca_v1_compiler_draft.md). The audit scope covers RTL, Python compilation flow, MLIR dialects and verifier, test entry points, generation toolchains, synthesis flows, and existing status documentation. This audit only adds audit materials and does not modify the implementation. At the start of the audit, `flow/synth/synth.py` had pre-existing uncommitted modifications, which were kept as-is; conclusions concerning this file apply to the current workspace. At the conclusion of the audit, `Readme.md` and `doc/synthesis.md` ([synthesis.md](synthesis.md)) were externally updated with synthesis data; differences were inspected and preserved, and no writes were made to these two files during this audit. Papers cited in the original designs, upstream version freshness, and remote machines in the XLS configuration were not re-verified or accessed.

## 1. Conclusions and Next-Stage Objectives

For the complete implementation scope and non-waivable acceptance criteria of Phase 1, see the [R1 Tasks and Acceptance Contract](phase1_tasks_and_acceptance.md). That document details the phased recommendations of this report; the completion of R1 is governed strictly by that contract.

**The project possesses a valuable functional prototype: a small subset of programs from the real Triton frontend can run on the 4-tile RTL via Python lowering, and GEMM, multi-role ready/free protocols, NUCA, and NoC have dedicated tests. However, the project does not currently meet the delivery criteria for a production compiler, production runtime, or signoff-ready silicon.**

The chain of trust must be addressed before expanding operator coverage:

1. Asynchronous ownership verification has reproduced detection omissions; the compiler can emit source-register overwrite sequences explicitly forbidden by the ISA.
2. E2E test entry points swallow failure exit codes; missing build dependencies allow stale simulation artifacts to participate in "verification".
3. Target files purporting to describe four-tile systems advertise SFU availability, whereas tile instances disable the SFU.
4. Current mesh parameterization, Triton plugin integration, runtime, and MLIR lowering have not formed a complete production loop.
5. RF ports, memory macros, timing, and physical implementations remain at the model and prototype stage; production PPA cannot be derived from functional tests or generic gate depths.

**Recommended next stage: R1 — "A reproducible, correctly failing, asynchronously resource-safe 2×2 AOT system baseline."** Before R1 concludes, 4×4 scaling, complex cache policies, Attention, or full LLVM targets should not be on the critical path. First resolve F01–F05 in this report, establish a concrete support matrix and release gate, and then connect existing functionality to a unified AOT entry point and manifest.

Production readiness is divided into three distinct acceptance tiers:

| Tier | Tangible Deliverables | Definition of Done |
|---|---|---|
| Software Deliverable | Installation package, AOT compiler, versioned manifest, simulation runtime, support matrix | Reproducible in a clean environment; unsupported inputs fail cleanly; supported programs execute correctly with diagnostics |
| Board Operable | FPGA/prototype board, physical memory and host interfaces, driver, recovery mechanisms | Decoupled from internal testbench state inspection; executes consecutive kernels and reclaims resources reliably |
| Silicon Manufacturable | RTL and physical signoff under targeted process/IP, DFT, test and reliability plans | Timing, area, power, testability, and system verification closed; cannot be substituted by software tests |

These three tiers cannot be conflated into a single "production-ready" label. Process nodes, packaging, memory interfaces, power budgets, target frequencies, and deployment modes do not yet form a product contract within the repository; the roadmap outlined below provides dependency-explicit milestones rather than speculative tapeout dates.

## 2. Completion Status Relative to Original Design

The evaluation below distinguishes between "existing constrained implementation" and "complete fulfillment of original design", rather than adopting the `done` tags in README.

| Original Design Requirement | Current Code Evidence | Audit Assessment |
|---|---|---|
| RF → NUCA → DRAM, no software-addressable SPM | `rtl/core`, `rtl/nuca`, `rtl/noc`, `rtl/tnca_top.v` | Implemented in prototype; retained and sound |
| Up to 4 RCTXs, roles decoupled from SIMD lanes | Core parameters, multi-role pipeline, `core4` profile | Verified on 2/4 RCTX single-core; does not prove all system profiles |
| DMA/MMA communicate via RF window, ready/free decoupled | `pipeline.py`, `partition.py`, barrier/tensor/matrix RTL | Core mechanism implemented; verification gaps exist in compiler (see F01/F06) |
| Thin TNCA execution layer preserving upstream arithmetic and control flow | `frontend.py` lowers to custom Python IR; C++ dialect exists independently | Reasonable bring-up path, but production path has not migrated to the hybrid MLIR of the original design |
| C0: Toolchain, target, per-stage IR dump | Pinned toolchain, generation scripts, `tnca-opt` exist; `STAGES` in backend is only declarative | Partially complete; target inconsistencies exist, real pass/dump pipeline not closed |
| C1: Real Triton → RTL | `triton_bridge.py`, `test_triton_e2e.py`, `test_e2e_system.py` | Constrained closed loop established; frontend only accepts blocks matching single vector register width, lacks general scf/dot support |
| C2: FP16/BF16 GEMM, tail, stride | `gemm.py` and GEMM RTL tests | Handcrafted TNCA IR GEMM path established; does not imply arbitrary Triton GEMMs are supported |
| C3: Explicit two-role asynchronous pipeline | `test_pipeline.py`, slow/ooo memory tests | Substantial test coverage; compiler safety cannot be inferred solely from positive test cases |
| C4: Automatic partitioning, loop cloning, P+M0+M1+E | `partition.py`, straight-line TTIR partitioning, specialized GEMM loops | Partially complete; README notes absence of literal P+M0+M1+E and lack of general loop driving from TTIR |
| C5: Attention and performance feedback | Report structures exist; no complete Attention compilation or system tests found | Incomplete |
| Stage 4: Cache hints, bypass, sharing group progress and admission | Current NUCA baseline; target specifies `cache_hints=false` | Independent mechanisms and ablation results not yet established; cannot claim paper mechanisms are realized |
| Stage 5: 4×4 mesh | Top-level NX/NY parameters exist, but NI home and source IDs fixed to 4 tiles | Incomplete parameterization (see F05) |
| Launch manifest, resource admission, drain on exit, job generation | Launch wires, testbench, simple launcher | Only partial hardware interfaces exist; lacks versioned host/runtime contracts |
| Manufacturable RF and datapath | Asynchronous reads, multi-port, full-array reset behavioral RF; Yosys gate-level proxy | Functional model/prototype; production storage macros and PPA closure incomplete |

The current default system differs from the original reference scale: 4 tiles, 2 RCTXs/tile, 128 physical V-registers (8 KiB/tile), 64 sets × 8 ways × 64 B (32 KiB/slice), and 8 MSHRs/slice; the original proposal's 16 tiles, 32 KiB VRF, 512 KiB/slice, and 96 MSHRs/slice served as design baselines, not existing implementation parameters. Furthermore, matrix RTL computation cycles cannot be assumed to be 256 MACs/cycle purely from the M16×N16 instruction shape.

## 3. Prioritized Findings and Remediation Plans

Priorities: P0 indicates correctness/verification defects blocking credible release; P1 represents prerequisites for scope expansion or system bring-up; P2 addresses scaling, performance, and volume engineering. Tags: `[Reproduced]` denotes verified by execution during this audit; `[Statically Confirmed]` denotes directly visible in code; `[Unverified Risk]` denotes identified risk requiring validation, not proven system failure.

### F01 — P0: Asynchronous Ownership Tracking Disconnected from Write Checking [Reproduced]

Evidence: In `compiler/python/tnca_cc/codegen.py:436`, `_op_window_put` directly emits `v.mov`; at `:588`, MMA lowering invokes `self.own.hold`, but this file never calls `self.own.check_write` or `check_write_range`. These checking methods already exist in `footprints.py`. Furthermore, DMA coordinate temporary registers are immediately returned to the allocator after emission; comments claim they are protected by an ownership hold, but hold alone does not prevent subsequent allocation or writes.

Under the default `check_protocol=True`, the [reproduction script](audit/2026-09-19/reproduce_ownership.py) generates and accepts:

```asm
m.mma.async.k16.f16 a0, v56, v48, s1
v.mov              v56, v0
mb.wait            s1
```

This confirms the compiler fails to reject overwriting source fragments while they are still being read; **this intentionally illegal program was not executed on the RTL, and no claim is made regarding observing erroneous numerical output**. The ISA assigns the responsibility of preventing such sequences to the compiler, making this an actual defect in the compiler's correctness boundary.

Another reproduced footprint calculation defect: the same `_op_mma_async` emits K32/K64 based on `op.attrs['k']`, yet computes footprints using the default K16 shape from `target.mma_shape`. In K64 FP16, A and B each require 16×64×2 B = 2048 B, corresponding to 32 64-byte vector registers; currently, a window allocated with only 8 registers is accepted. See the [K64 reproduction script](audit/2026-09-19/reproduce_k64_footprint.py) and [output](audit/2026-09-19/k64_footprint_reproduction.log). This is likewise a compiler rejection omission; its invalid RTL artifact was not simulated.

Remediation: Derive footprints dynamically from the instruction's actual K dimension, datatype, and layout rather than substituting default target shapes. Uniformly describe explicit and implicit read/write footprints and hold durations for every machine operation. Route all regular writes, window writes, DMA writes, and accumulator writes through this unified check. Ensure the allocator does not release occupied registers before sources are reusable and computation is complete. Avoid ad-hoc special cases in `window_put`. Clearly distinguish read leases, write leases, accumulator sequencing, and inter-role physical aliasing. Post-allocation verification must cover production entry points and must never be silently disabled by `check_protocol=False`.

Acceptance: Add negative test suites covering the above cases, DMA coordinate reuse, DMA and MMA aliasing in the same window, premature accumulator reads/writes, multi-consumer last-release semantics, and post-allocation overlapping aliases. Verify that valid programs with correct barriers still pass. Negative tests must report operation location, physical range, owner, and missing completion events during compilation.

### F02 — P0: End-to-End Failures Swallowed by Shell Pipeline [Reproduced]

Evidence: `Makefile:469` and `:480` execute `python ... 2>&1 | tail -4` without setting `pipefail`. Running the following command returns exit code **0**, even though `/usr/bin/false` fails unconditionally:

```sh
make triton-e2e e2e TRITON_PYTHON=/usr/bin/false
```

When Triton is missing, these targets also SKIP with a successful status. The test count reported by `compiler/test/run_tests.py` includes skipped tests; "PASS 76 tests, 7 skipped" cannot be construed as 76 valid tests executed. Furthermore, `rtl_harness.py` determines simulation success solely from stdout containing PASS/FAIL, without checking `proc.returncode` or enforcing wall-clock timeouts. The IR runner also utilizes shell pipelines, requiring unified exit code propagation auditing.

Remediation: Have Python test runners manage subprocesses directly, capturing complete stdout/stderr and exit codes; explicitly enable `pipefail` in shell scripts. While test skips may be permitted in development environments, release gates must mandate zero skips and zero failures across specified test suites, and fail if total test counts drop unexpectedly. Success must require valid exit codes, structured test summaries, and expected completion banners.

Acceptance: Intentionally failing interpreters, test cases, assemblers, simulators, missing dependencies, absent PASS banners, and timeouts must reliably trigger non-zero exit codes at the release gate. Persist full failure logs rather than truncating to the trailing four lines.

### F03 — P0: System Target Advertises SFU Unavailable in Hardware [Statically Confirmed]

Evidence: `compiler/targets/tnca-sim-v0.1.json` describes a four-tile system with `capabilities.sfu=true`; `tools/isa/gen_target.py:83` samples the core default `HAS_SFU`; however, `rtl/tile/tnca_tile.v:131` instantiates the core with explicit `.HAS_SFU(0)`. Test `t15_target` resides in the single-core regression suite and does not verify that multi-tile elaboration overrides match the manifest.

Impact: The compiler permits SFU instructions based on this target, which the full-system RTL will reject at runtime. Passing single-core SFU unit tests does not resolve this discrepancy.

Remediation: Define independent profiles for `core-sim`, `system-2x2`, and future FPGA platforms. Derive RTL parameters, target JSONs, capability readback expectations, and simulation commands from a single configuration source. Target files must describe elaborated instances rather than module defaults. Compilation artifacts must embed profile hashes, verified by the loader prior to execution.

Acceptance: Every published profile must read back all capabilities and resources from its top-level module. Provide positive and negative test cases for both SFU-enabled and SFU-disabled profiles. The system must explicitly reject incompatible binaries.

### F04 — P0: Incremental Builds Risk Validating Stale RTL [Reproduced Build Decision]

Evidence: In `Makefile:164`, prerequisites for `tb_core.vvp` do not include `RTL_GEN` or `.vh` header files. The following dry-run commands output only directory creation without re-invoking `iverilog`:

```sh
make -n -W rtl/gen/tnca_cvt_comb_xls.v sim/build/tb_core.vvp
make -n -W rtl/common/tnca_defs.vh sim/build/tb_core.vvp
```

This confirms missing dependency edges in the build graph; these source files were not modified during this audit. The `slow`, `ooo`, and other testbenches require similar inspection. Additionally, `compiler` schema/target prerequisites are automatically regenerated prior to running `--check`, contradicting comments stating that check targets verify without mutating stale files.

Remediation: Unify RTL source and include manifests or generate automatic dependency files; incorporate parameter configurations and tool versions into build fingerprints. Decouple `generate` from a read-only `check-generated` target, and enforce clean checkouts without workspace mutations in CI. Track Python generator modules and random seeds for generated code and test vectors.

Acceptance: Implement build-invalidation tests for every input category. Verify consistency between clean and incremental builds. Outdated schemas must fail verification rather than being silently overwritten.

### F05 — P1: 4×4 Topology Cannot Be Achieved Merely by Altering NX/NY [Statically Confirmed]

Evidence: `rtl/noc/tnca_core_ni.v:84` utilizes 2-bit home routing (1 bit each for X and Y); `rtl/tnca_top.v:177` hardcodes `.N_W(2)`, and external `mem_req_src`/`mem_rsp_src` signals are also 2 bits wide. Tile PID is hardwired to the tile ID. Increasing NX/NY will not automatically provide correct 16-slice home addressing or DRAM response routing.

Remediation: Unify node ID, X/Y coordinates, home slice, and source tag encodings; derive bus widths from supported topologies, accounting for N=1. Explicitly enumerate supported rectangular and power-of-two topologies; reject unsupported configurations during elaboration. Decouple grid program IDs from physical tile IDs, dispatching them in batches via the scheduler.

Acceptance: Validate 1×1 and 2×2 topologies before expanding to 4×4. Verify accessibility across all slices, correct out-of-order responses across all sources, and strict single-home mapping per physical address. Test grid sizes smaller than, equal to, and greater than physical tile counts. Do not claim 4×4 support until these verifications pass.

### F06 — P1: MLIR Verifier Does Not Constitute Complete Proof of Asynchronous Safety [Statically Confirmed]

Evidence: In `VerifyAsyncOwnership.cpp`, verification state is reset at each basic block boundary. `collectTouches` only inspects window operands for load/store and A/B operands for MMA, omitting MMA accumulators; it does not evaluate physical register ranges, aliasing, CFG fixed points, or cross-role channel lifecycles. Although `TNCAAsyncOps.cpp` defines footprint interfaces, the pass does not leverage them for unified verification. The existing IR test suite comprises only two files.

Remediation: In the short term, restrict accepted IR to the verified single-block subset, conservatively rejecting others. In the medium term, implement dataflow analysis for events and leases, establishing rigorous rules for CFG joins, loop back-edges, region captures, and cancellation paths. Resource analysis must verify logical ownership prior to register allocation and physical aliasing after allocation.

Acceptance: Ensure legitimate cross-block lifecycles do not produce false positives. Reject single-sided await branches, loop overwrites, nested regions, double releases, and accumulator hazards. Python and MLIR compilation paths must share semantic test cases rather than testing isolated implementations.

### F07 — P1: Production Backend, Parameter ABI, and Runtime Remain Skeletons [Statically Confirmed]

Evidence: In `compiler/backend/compiler.py`, `load_dialects` and `add_stages` raise `NotImplementedError`. Real TTIR is obtained by calling Triton internal APIs via `compiler/test/triton_bridge.py` and passed into the Python importer. `driver.py` merely constructs `vvp` command lines. Standalone installable compiler packages and production loaders are absent.

In `frontend.py:123`, function arguments are bound according to insertion order in the `arg_values` dictionary, hardcoding addresses and scalars into compile-time constants; `:102` traverses entire MLIR modules rather than specific target functions. Multi-function modules, altered parameter orders, or incomplete argument lists risk incorrect bindings or processing non-target functions. Type and floating-point constant parsing still partially relies on textual extraction.

In system tests (`test_e2e_system.py`), testing is limited to two elementwise kernels with BLOCK=16 over 4 blocks; `run_system` inspects a unified view of dirty cache and DRAM via testbench harnesses. While this demonstrates simulation correctness, **it does not demonstrate that host DMA can observe outputs on physical hardware**.

Remediation: Implement a constrained AOT CLI producing binary artifacts, manifests, and compilation reports. Formalize a parameter ABI specifying types, bitwidths, offsets, and alignments, binding by function signature. Introduce a testable runtime/loader layer managing admission, grid scheduling, fault reporting, and drain cycles before integrating the official Triton backend. Define explicit cache flush, coherence, and completion notification mechanisms for host visibility. Avoid concealing hardware integration logic within test helpers.

Acceptance: Re-execute identical binaries with varied arguments and buffer addresses. Reject invalid arguments or target version mismatches. Test grid sizes exceeding tile counts (grid > Ntiles), back-to-back launches, fault recovery, and delayed response packets. User space must retrieve correct results without reading RTL internal arrays.

### F08 — P1: Reproducible Build and Release Infrastructure Incomplete [Statically Confirmed]

Evidence: The repository lacks `.github` CI workflows, Python packaging, and dependency locking; default Triton paths point to virtual environments in developer home directories. Although LLVM and Triton versions are pinned, `toolchain.json.host_status` has drifted from reproducible local build states. The XLS driver relies on hardcoded personal remote addresses, private SSH keys, and local `cur` tool directories; recording tool version strings in file headers does not ensure a reproducible generation environment.

Remediation: Separate development and release environments; introduce lockfiles, environment diagnostic tools (`tool doctor`), standardized installation/build scripts, and CI workflows. Provide reproducible Linux local/container execution for XLS, treating remote execution as an optional transport and removing personal configurations from version control. Record source file sets, tool digests, parameters, and invocation commands in all generated artifacts. Retain existing LLVM download checksum mechanisms.

Acceptance: Rebuild and execute mandatory regressions from scratch on a clean Linux environment; maintain macOS as a tested development profile. Eliminate dependencies on maintainer SSH accounts. Archive tool inventories, execution logs, random seeds, and binary/manifest checksums with every release.

### F09 — P1/P2: Memory Progress Verification Does Not Support Broad "Deadlock-Free" Claims [Unverified Risk]

Evidence: Request/response separation, beat reservation, MSHR/target pools, hit-past-block, and random backpressure tests exist and are fundamentally sound. However, `tnca_mem_mux.v` enforces fixed priority for new tensor requests: `a_req_ready = !pick_b && m_req_ready`; under sustained tensor requests, no bounded waiting guarantee exists for the LSU. Prioritizing completion paths does not justify allowing arbitrary new DMA requests to permanently starve LSU traffic.

Remediation: Decouple completion/response traffic, resource-reclaiming control accesses, demand accesses, and prefetch requests; preserve dedicated resources for completion paths while applying provably fair arbitration to regular requests. Introduce assertions for ready/valid handshakes, request reservations, tag uniqueness, resource counters, and barrier generations. Explicitly document environmental assumptions (e.g., external memory eventual response) in liveness proofs.

Acceptance: Demonstrate sustained progress under heavy contention, simultaneous saturation of MSHR/target/WB/return queues, exception handling, and drain sequences. Provide bounded-wait guarantees or formal liveness proofs under stated fairness conditions. Ensure that disabling, dropping, or delaying hints does not alter functional results.

### F10 — P2: Register Files, Storage Macros, and Timing Not Ready for Physical Implementation [Statically Confirmed]

Evidence: `tnca_vector_rf.v` is a behavioral model featuring multi-port asynchronous reads, dual writes, and full-array resets; it cannot be directly synthesized into standard SRAM macros. `tnca_core.v` (~1811 lines) consolidates issue, hazard detection, writeback, asynchronous completion, and trap logic, meaning modifications to ports or latencies risk destabilizing protocols. Current gate-level synthesis metrics are proxies ([synthesis.md](synthesis.md)) rather than signoff STA against a specified PDK; comprehensive full-chip synthesis was not re-run during this audit.

Remediation: Define implementable SRAM/BRAM bank structures, port counts, read latencies, write conflict resolutions, and bypass semantics; introduce operand collectors, backpressure handling, and valid-state reset logic. Budget port requirements across DMA, SIMD, MMA, and ARF interfaces; balance compute parallelism against available memory bandwidth. Establish synthesis, SDC, STA, memory macro, and clock/reset methodologies under commercial standard-cell libraries prior to floorplanning and place-and-route.

Acceptance: Perform differential functional verification against the behavioral model. Inject random bank conflicts. Complete synthesis across target profiles. Provide WNS/TNS, area, and power metrics across specified clock frequencies, process nodes, and operating corners. Subsequent ASIC phases require CDC/RDC analysis, DFT/MBIST, formal equivalence, and physical signoff per product contracts.

### F11 — P2: Documentation, Performance Assertions, and Empirical Evidence Desynchronized [Statically Confirmed]

Evidence: `Readme.md` simultaneously lists completed phases alongside obsolete "Stage 2 next" roadmaps; `compiler/README.md` introduces milestone C0 while listing C0–C4 as done, followed by incomplete C1 tasks; audit scope and deviation metrics in `conformance.md` do not reflect the current tree. Although resource report structures exist, `KernelPlan`, `format_report`, and `try_plans` are only invoked by report unit tests outside definition and export; they are not integrated into production compilation. Comprehensive benchmarks, measured RF/NoC bandwidths, and model prediction error closure are absent.

Remediation: Maintain the original architecture and compiler design drafts as architectural intent ([Architecture Draft](design/tnca_v1_architecture_draft.md), [Compiler Draft](design/tnca_v1_compiler_draft.md)), while creating a current specification and capability matrix. Track profile targets, implementation paths, positive/negative tests, and recent verification commits for every feature. Bind performance reports to fixed tensor shapes, datatypes, seeds, tool versions, and compilation options; preserve legacy benchmark figures with explicit version tags rather than inheriting them implicitly.

Acceptance: Enable new contributors to distinguish among "instruction present in ISA", "implemented in target profile", "emitted by compiler", and "verified in full-system RTL". Benchmark reports must include raw hardware performance counters and baseline comparisons, avoiding the substitution of cache hit rates or gate depths for achieved throughput and frequency.

## 4. Engineering Refactoring Plan: Decoupling by Contract and Incremental Migration

Resolve credibility and correctness defects before undertaking architectural refactoring. Do not attempt a wholesale rewrite of Python compilation code into C++ upfront, nor reorganize directory hierarchies prior to defining interface contracts. Existing assemblers, bit-exact floating-point reference models, protocol simulators, and RTL microbenchmarks serve as regression baselines and must be preserved during migration.

The proposed boundary of responsibilities is as follows (directory names represent proposals and do not imply prior creation):

```text
spec/                      ISA, target/profile, ABI/manifest schema
compiler/
  frontend/                Pinned Triton adapter, entry point, and type checking
  lib/.../Transforms/      Layout, partition, channel, protocol, allocation
  python/tnca_cc/          CLI/API, debug and reference paths; gradual deprecation of production lowering
  test/                    IR/diagnostic/compiler negative tests
runtime/
  common/                  Manifest validation, resource admission, launch/fault lifecycle
  sim/                     Simulation transport; independent of test modules
  device/                  Subsequent real host/device transport
rtl/
  core/                    Fetch, issue/scoreboard, RF, writeback, completion
  mem/ or existing nuca/noc/ Retain existing module boundaries, unify interface and configuration source
verification/
  reference/               Independent reference models for FP/ISA/protocol
  unit/ integration/       Layered regression from core to full system
  formal/                  Handshake, ownership, resource conservation, and progress properties
benchmarks/                Workloads, baselines, counters, ablations, and reports
flow/                      Generation, simulation, FPGA, synthesis, and physical implementation
```

Immediate directory restructuring is not required. Incremental, independently reviewable modifications should proceed in the following order:

| Refactoring Unit | Explicit Contract | Migration and Regression Requirements |
|---|---|---|
| Configuration & ISA | Profiles govern elaboration, capabilities, and compiler targets simultaneously; ISA maintains a single authoring source | Add consistency tests before generating derived artifacts; preserve current ISA opcodes |
| Build & Test Runner | Every execution step reports returncode, timeout, required/optional status, logs, and tool fingerprints | Preserve Make entry points while replacing internal runners; verify failure injection |
| Compiler Driver | `source + signature + target + options → binary + manifest + report + dumps` | Extract production logic from test bridges/harnesses; redirect legacy tests to new API |
| Asynchronous Resource Layer | Explicit analysis results for events, leases, channels, and physical footprints | Replace ad-hoc hold/free logic across lowerings with shared semantic test suites |
| Pass Pipeline | Explicit input/output IR invariants; run verifier after each lowering pass | Connect narrow MLIR path first; replace Python lowering incrementally, comparing against baseline RTL |
| Codegen & Allocator | Separate allocation maps from IR to eliminate shared mutable state in `Value.reg` | Ensure deterministic compilation across repeated runs, varied role sequences, and distinct targets |
| Core RTL | Decouple issue/hazard, writeback arbitration, async completion/fault, and RF interfaces | Refactor one state owner at a time; ensure multiple simultaneous completions are preserved |
| Runtime & Benchmarking | Decouple production completion semantics from testbench inspection interfaces | Retain internal inspection for debugging; gate releases on realizable host output paths |

`partition.py` (~946 lines) should be decomposed into four distinct responsibilities: dependency graph analysis, cost modeling/mapping, channel insertion, and loop cloning. `codegen.py` (~630 lines) should be split into resource verification, register allocation, instruction selection, and code emission. Refactoring must ensure invariants reside in a single implementation location, rather than treating line count reduction as an acceptance criterion.

The production compilation flow will continue adhering to the original design: preserve general semantics in upstream `tt/arith/math/scf`, restricting TNCA-specific passes to resource management, layout transformations, and asynchronous lifecycles. Avoid expanding the bring-up Python IR into a redundant general-purpose optimizer. Developing full LLVM targets should occur only after machine IRs, ABIs, and scheduling interfaces stabilize, and is not a prerequisite for R1.

## 5. Phased Execution Plan and Acceptance Gates

Milestones are driven by exit criteria rather than merged code commits. Assigned roles represent functional responsibilities and do not assume existing organizational headcount.

### R1 — Credible and Reproducible 2×2 Baseline (Immediate Focus)

Responsibility: Jointly led by compiler and verification/build maintainers.

Deliverables: Remediate F01–F04; implement the supported subset for F06; add read-only generation checks and strict regression gates; formalize core/system profiles and current support matrices; integrate negative audit cases into official CI suites.

Exit Criteria:
- Rebuildable in a clean environment with zero test skips in mandatory suites; injected failures reliably fail CI.
- Asynchronous register overwrites, missing awaits, and target capability mismatches are rejected at the appropriate boundary.
- Every declared profile reads back its target contract from its top-level module.
- Clean and incremental builds produce identical outputs; all verification results record commit SHA, dirty status, tool versions, and arguments.
- Existing positive regressions experience zero regressions; all known failures possess tracked issues, owners, and release-blocking determinations.

Deliverables for R1 must be designated as a "Verified Experimental AOT Baseline" and not marketed as a production release.

### R2 — Unified AOT Compiler and Simulation Runtime

Prerequisite: R1. Responsibility: Compiler and runtime teams.

Deliverables: Installable package and CLI; pinned Triton adapter; signature/argument ABI, binary manifest, and resource admission; execution logging and IR dumps; standard grid scheduling and back-to-back kernel launches; incremental migration to MLIR pipeline while retaining Python for reference validation.

Exit Criteria: Re-execute identical binaries with modified base addresses and tensor shapes; validate grid sizes of 1, Ntiles, Ntiles+1, and multi-batch workloads; support tail blocks, strided layouts, conditional branches, and explicitly supported loops; observable fault, timeout, and drain handling; deterministic diagnostics for unsupported inputs. End-to-end execution must not rely on test helper scaffolding for production entry points.

### R3 — Compute and Memory Implementation with Performance Contracts

Prerequisite: R1 configuration and verification foundations; can proceed concurrently with portions of R2, but new RTL must pass identical validation gates. Responsibility: Microarchitecture, verification, and performance teams.

Deliverables: Implement banked RF/ARF interfaces, read latency models, and arbitration logic; measurable DMA/MMA/SIMD throughput; workloads for GEMM, reduction, softmax, and LayerNorm/RMSNorm; cross-validation between hardware counters and compiler resource reports.

Exit Criteria: Freeze minimum throughput, maximum latency, and resource budgets for every target workload prior to reporting reproducible measurements. Report RF bank stalls, MMA starvation cycles, barrier wait cycles, MSHR entry/target saturation, NoC link utilization, and DRAM traffic. Detail correctness and performance deltas relative to R1; do not report only peak performance samples.

Integrate Attention only after GEMM, reduction, SFU/software fallbacks, accumulator↔vector transfers, inter-role channels, and resource budgets are fully closed; start with a verifiable online-softmax variant before extending to causal masking, tail blocks, dynamic shapes, and QK/PV overlap.

### R4 — Controlled Scaling of 4×4 Mesh and Cache/Flow Mechanisms

Prerequisite: R2/R3 configuration, mesh, and measurement capabilities. Responsibility: NoC, NUCA, and compiler scheduling teams.

Deliverables: Remediate all topology encodings; scale mesh dimensions before introducing use-complete notifications, replacement/bypass policies, MSHR-aware admission control, and sharing-group regulation; generate cache metadata from finalized schedules.

Exit Criteria: Verify address routing, response delivery, atomicity, and resource progress across a 4×4 mesh; ensure each new mechanism can be toggled independently for baseline, single-feature, and combined ablations. Numerical outputs must remain bit-exact when hints are disabled or perturbed. Accounting must include actual data and control overheads per link, incorporating external memory bottlenecks into performance models.

### R5 — Physical Prototype and System Bring-Up

Prerequisite Product Decisions: Selection of FPGA platform or SoC/PCIe integration mode, memory interfaces, single-tenant vs. multi-tenant execution, reset granularity, and host coherence contracts. Responsibility: Systems, driver, and hardware teams.

Deliverables: Physical memory controllers and host transport; device driver/loader, buffer address validation, completion queues, and error recovery; transaction generation tracking or reliable drain isolation; performance and fault telemetry.

Exit Criteria: Direct host verification of output buffers; pass consecutive launches, error recovery, extended stress testing, backpressure scenarios, external memory faults, and reset boundaries; prevent stale task responses from corrupting subsequent task allocations. Verify rejection of out-of-bounds DMA and invalid binaries according to isolation goals. Initial versions may omit demand paging and preemption per original design, but must explicitly reject unsupported operations rather than executing partially.

### R6 — ASIC and Volume Signoff (If Custom Silicon Is Targeted)

Prerequisite: Pinned PDK, certified SRAM/IO/memory IPs, and fixed system power/area budgets; cannot be replaced by R5 software gates. Responsibility: RTL/DV, physical design, DFT, and system/manufacturing test teams.

Deliverables & Exit Criteria: Fully constrained synthesis and signoff STA, multi-corner physical implementation, CDC/RDC signoff, reset and power domain verification, formal equivalence checking, memory BIST/DFT implementation, reliability and error reporting infrastructure, power/IR-drop/electromigration analysis, physical verification (DRC/LVS), package/board interface validation, and manufacturing test programs. Coverage metrics, timing margins, and burn-in durations are established by product contracts and accompanied by formal waiver reviews.

All phases must maintain rollback mechanisms: experimental features disabled by default, backward compatibility with legacy benchmarks and binary profiles preserved, and ISA or resource semantic modifications strictly versioned with explicit rejection behaviors.

## 6. Required Enhancements to the Verification Methodology

| Hierarchy | Focus Areas | Execution Cadence |
|---|---|---|
| Per-Commit | Schema and profile consistency, compiler negative tests, IR verifier, fast RTL simulations, and linting | Mandatory; zero environment skips |
| Integration | Triton → AOT → single-core/system, fast/slow/ooo memory, 2/4 RCTXs, capability toggles | Merge gating; retain complete logs |
| Long Regression | Multi-seed runs, randomized memory/backpressure, FP corner cases, fault/reset/drain sequences, shared-address atomicity | Nightly; failed seeds independently replayable |
| Property Verification | Request/response conservation, credit/target leak checks, tag uniqueness, completion delivery, generation tracking | Protocol module modifications and periodic audits |
| Performance Regression | Shape/dtype/stride/tail matrix sweeps, hardware performance counters, relative delta against baseline | Performance-related changes and releases |
| Release Gate | Clean-environment reconstruction, mandatory suite zero-skip enforcement, installation and ABI compatibility, generated artifact consistency | Every release candidate |

Prioritize tests capable of falsifying safety assumptions rather than merely expanding happy-path coverage. RTL randomized testing must define coverage goals and report failure seeds. Formal verification must document environmental assumptions and bounding limits. Independent numerical oracles and software reference models must avoid sharing common failure modes with implementation code. For floating-point operations, distinguish between exact ISA contracts and approximate operator error budgets, avoiding the conflation of bit-exact matching with tolerance thresholds.

The existing `tools/run_test.sh` script rejects runs lacking PASS banners or exiting with non-zero status codes, which provides a reusable baseline; however, it cannot remediate upper-level Make targets swallowing exit codes, nor replace wall-clock timeouts. While randomized memory backpressure testing exists, it does not independently prove deadlock freedom across all future topologies, parameters, and policies.

## 7. Verification Records and Audit Boundaries

Detailed execution results are listed below and in accompanying log files. Testing was conducted in an existing development environment and build directory; **this does not constitute clean-environment reproducibility acceptance**. This audit did not install dependencies, rebuild pinned Triton/LLVM toolchains, invoke remote XLS instances, perform physical signoff, or exhaust the hardware state space.

| Command / Inspection | Audit Result | Demonstrated Capability |
|---|---|---|
| `make test` | Returned 0; 38 RTL files elaborated; 67 compiler RTL tests passed; core fast/slow/ooo, 3 NUCA variants, 20,000-packet mesh, 4 top-level runs, and E2E completed | Existing regression suite passed; 7 compiler-unit tests skipped; does not eliminate verification blind spots identified in F01–F04 |
| `make compiler` | Returned 0; 76 unit tests reported (7 skipped); 2 IR test files passed; 169 schema instructions and 218 shared constants verified | Constrained checks passed in current environment; does not prove capability consistency across system instances |
| `make triton-e2e e2e` | Logs recorded 5 tests OK and 2 tests OK respectively | Narrow Triton closed loop executed; Make return codes subject to F02 masking defect |
| Direct execution of two E2E Python files in Triton venv | Both processes returned 0, passing 5 and 2 tests respectively | Bypassed Make `tail` piping, confirming no hidden failures in this specific E2E run |
| `TRITON_PYTHON=/usr/bin/false` failure injection | Both Make targets still returned 0 | Reproduced defect F02 |
| `reproduce_ownership.py` | Illegal overwrite sequence accepted and emitted | Reproduced compiler rejection defect F01 |
| Two dry-runs of `make -n -W ... tb_core.vvp` | Did not re-invoke `iverilog` | Reproduced build dependency defect F04 |

Supplementary Materials:
- [Verification summary](audit/2026-09-19/verification_summary.json)
- [Environment, commit, and reproduction return codes](audit/2026-09-19/environment_and_reproductions.json)
- [Ownership reproducer script](audit/2026-09-19/reproduce_ownership.py) and [actual output](audit/2026-09-19/ownership_reproduction.log)
- [Full make test regression log](audit/2026-09-19/full_regression.log) (process exit code 0)
- [Compiler regression log](audit/2026-09-19/compiler.log), [Make E2E log](audit/2026-09-19/e2e.log)
- [Direct single-core E2E log](audit/2026-09-19/triton_e2e_direct.log), [Direct system E2E log](audit/2026-09-19/system_e2e_direct.log)

## 8. First Batch of Actionable Project Items

| Sequence | Change Item | Completion Criteria |
|---|---|---|
| 1 | Test return codes and strict mode | Commands fail on `/usr/bin/false`, missing dependencies, or test anomalies in release mode; preserve full logs |
| 2 | Fix build dependencies and read-only generation checks | Rebuild correctly upon `.vh`, generated RTL, or parameter changes; stale schemas fail verification |
| 3 | Unify core and system profiles | Resolve SFU capability discrepancies; verify top-level capability readbacks match manifests |
| 4 | Connect real codegen ownership verification | Reject negative cases from this report; pass valid asynchronous programs; eliminate unverified production bypasses |
| 5 | Formalize MLIR/TTIR support boundaries | Complete negative coverage for CFG, aliasing, and parameter boundaries; explicitly reject unsupported behavior |
| 6 | AOT CLI and minimal manifest/runtime loop | Independent of test modules; compile, load, repeatedly execute, and retrieve results post-installation |

The first five items constitute the critical path for R1, with the sixth initiating R2. Establishing this trusted delivery pipeline must precede prioritizing banked RFs, Attention, 4×4 scaling, and advanced cache/flow policies via benchmarks.
