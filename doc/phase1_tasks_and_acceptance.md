# Phase 1 (R1): Full Task Inventory and Non-Waivable Acceptance Contract

Version: 1.0, 2026-09-19. Status: **Specification established; implementation not yet accepted.**

Basis: [Current Code Audit](production_readiness_audit_2026-09-19.md), [Original Architecture Design](design/tnca_v1_architecture_draft.md), [Original Compiler Design](design/tnca_v1_compiler_draft.md). The audit baseline commit is `6c90063`; historical test records are archived in [audit/2026-09-19](audit/2026-09-19/verification_summary.json). Passing legacy regressions does not signify passing this contract.

This document operationalizes R1 from the audit into an executable contract. **The sole criterion for completion of this phase is: all mandatory deliverables, all mandatory test cases, and all evidentiary requirements in this document are fully satisfied on the identical candidate commit. There is no percentage-based completion acceptance, no conditional pass with known defects, and no temporary waivers.**

## 1. Phase Objectives, Boundaries, and Non-Waivable Rules

### 1.1 Sole Phase Objective

Deliver a **2×2 experimental AOT baseline** that can be reconstructed in a clean environment, reliably detects failures, enforces compiler prevention of asynchronous resource violations in supported programs, and aligns compilation targets with actual hardware instances.

Here, the AOT baseline refers to the existing flow of ahead-of-time compilation to assembly/binary and execution on RTL. R1 does not require completing R2's installable production CLI, dynamic parameter ABI, general grid scheduler, production loader, or physical host driver. R1 must provide unified **acceptance entry points and compilation safety checking entry points**, and the lack of a production entry point cannot be used as justification for disabling checks.

### 1.2 Scope Freeze

| Mandatory in R1 | Excluded from R1 (No New Capabilities) |
|---|---|
| Root-cause remediation of F01–F04; actual footprint verification for K16/K32/K64 and dtypes | New Attention/Conv operators or full Triton coverage |
| Closure of F05 boundaries: reject unsupported topologies prior to execution | 4×4 implementation, arbitrary grid scheduling |
| Safe subset implementation and explicit rejection for F06; preserve existing valid loops / multi-role paths | General cross-CFG, arbitrary nested region optimizing compiler |
| Function selection, parameter binding, and support boundary validation for current importer in F07 | Full passes of official Triton plugin, complete LLVM target, production runtime |
| Locked environment, CI, artifact regeneration, and evidence required for this baseline in F08 | FPGA/ASIC signoff, physical SRAMs, host coherence, production multi-tenant isolation |
| Progress and resource-conservation regressions for existing finite workloads in F09 | System-wide liveness proofs under unbounded traffic, new cache/flow policies |
| Synchronization of current status, capability matrix, and verification records in F11 | Full performance tuning, full-chip PPA commitments |

This boundary constitutes phased division of labor, not a waiver for failures within this phase. The phrase "first resolve F01–F05" in the audit summary means specifically for R1: complete fixes for F01–F04, and closing current error entry points for F05 by forbidding unsupported configurations; full 4×4 functional implementation is deferred to R4.

### 1.3 Acceptance Rules

1. **Zero failures, zero errors, zero skips, zero xfail, zero xpass, zero timeouts, and zero orphaned subprocesses.** Missing tools, absent hardware profiles, uninstalled FileCheck, or Triton import failures cause acceptance to fail.
2. Negative test cases must execute as formal tests: expected compilation rejection or hardware trap is the success condition. Negative tests must not be implemented as unittest skips, xfail, or ignored exit codes.
3. Missing evidence, unexecuted tests, unknown results, and corrupted result formats are all equivalent to failure; PASS must never be assumed by default.
4. Forbidding deletion or disabling of baseline tests, reduction of randomization scale, relaxation of numerical tolerances, inflation of resources to conceal original profile defects, removal of negative tests, or deletion of capability declarations to evade preserved capabilities frozen in this document.
5. "Retry until pass" is forbidden: retry=0 within release runs. Any single failure retains FAIL for that run; after remediation, full acceptance must be re-executed from scratch on a new candidate commit.
6. Results from another commit, another profile, single-core tests, or simulation models must not substitute for full-system results on the current candidate.
7. Modifying generated assembly, manually inserting barriers, pre-filling correct outputs, or bypassing the real frontend to satisfy corresponding E2E test cases is strictly prohibited.
8. All mandatory test cases must possess unique IDs; the discovered test case set must strictly equal the acceptance inventory set. Comparing total counts alone is insufficient; renaming or splitting must retain mappings to original semantic IDs.
9. Any modification to tracked source code, configurations, lockfiles, tests, or this contract invalidates previous signoffs for the candidate commit. Modifying inventories or thresholds to turn a known failure on the same candidate into a pass is prohibited.
10. No waiver fields, allowed-failure lists, or "development-environment-only" exceptions exist. If this contract is not fully satisfied, the phase status remains **NOT ACCEPTED**.

## 2. Mandatory Configurations and Capabilities to Preserve

### 2.1 Configuration Inventory

The following are R1 specification IDs, which must form machine-readable profiles upon implementation; legacy names may be retained as unambiguous aliases, but distinct instances must not be conflated into the same target.

| Profile ID | Hardware Scope | Mandatory Frozen Key Parameters | Mandatory Verification |
|---|---|---|---|
| `core-2r-sfu` | Standalone core | 2 RCTX, VLEN=16, ELEN=32, 128 physical VREGs, 64 logical VREGs/RCTX, SFU=1 | fast/slow/ooo, original core programs, compiler RTL, capability readback |
| `core-4r-sfu` | Standalone core | 4 RCTX, VLEN=16, ELEN=32, 128 physical VREGs, 32 logical VREGs/RCTX, SFU=1 | Capability readback, one producer / two consumers, multi-consumer last release |
| `system-2x2-nosfu` | 4-tile full system | NX=NY=2, 2 RCTX/tile, SFU=0; per slice: 64 sets, 8 ways, 64 B line, 8 MSHRs, 16 targets | Per-tile capability readback, 4 original top runs, real Triton full-system E2E |

Remaining resource quantities are governed by the frozen unified profile files and must correspond item-by-item to actual elaboration. `core-4r-sfu` must not be represented as an accepted 4-core / 4-RCTX full system. fast/slow/ooo are memory behavioral variants that must carry distinct build fingerprints without fabricating divergent ISA capabilities.

The R1 system profile does not support changing NX/NY to 1×1, 1×2, 2×1, 4×4, or non-positive values; such configurations must fail during configuration validation / elaboration rather than launching simulation into misrouting. Standalone core profiles do not equal a 1×1 mesh.

### 2.2 Capabilities Subject to Non-Regression

- All core programs, floating-point / conversion / SFU / decoder / assembler test cases from the audit baseline.
- Python TNCA IR masked elementwise, integer/FP32 reduction, gather; real Triton add/mul-add and existing role-partitioned closed loops.
- FP16/BF16 GEMM, existing stride and M/N/K tail support; original specialized GEMM planner and lowering.
- Two-role ready/free pipelines, 2/3-slot ring, 0/1/2/3/5/6 iterations, predicated-off async, error propagation, slow/ooo memory models.
- Existing unrolled and rolled specialized GEMM paths, including rolled K64/K128 and 3-slot K96; dual-consumer release on 4 RCTXs.
- NUCA hit-past-block, dirty writeback, MSHR/target tracking, mesh backpressure, device atomic operations, and existing exception isolation.

This inventory preserves capabilities by **concrete entry point**: hand-constructed TNCA IR GEMMs must not be labeled as real Triton GEMMs; single-core SFU must not be labeled as full-system SFU. Unimplemented general TTIR control flow may be explicitly rejected, but must not cause rejection of already functional specialized rolled pipelines.

## 3. All Work Packages and Independent Acceptance Gates

Each work package must deliver implementation, tests, documentation, and evidence; the assignee field must record actual individuals responsible. Roles in the tables represent functional responsibilities and do not substitute for final signoff attribution.

### R1-01: Freeze Specifications, Support Matrix, and Baseline Cases

Responsibility: Maintainers / Verification.

Mandatory deliverables:

- Export all baseline semantic test case IDs, entry points, profiles, inputs, expected outputs, and legacy test locations; establish `r1_required_cases.json` or equivalent version-controlled inventory.
- Divide support matrices into four explicit columns: "Defined in ISA", "Implemented in Profile", "Emittable by Compiler", and "Verified in Full-System RTL", alongside explicit dtype/K/shape/stride/control-flow constraints.
- Document read/write footprints, resource hold durations, release events, error completion behavior, barrier generation, and predicated-off semantics for all asynchronous operations.
- Map each new test case to work packages in this document and audit finding IDs; mark the two original design documents as architectural intent without overwriting them as descriptions of current state.

Acceptance: All baseline semantic IDs are traceable; no entries contain bare "supported" assertions without entry points, profiles, and tests; mandatory test inventory must not be empty; no unresolved boundaries, TBDs, TODOs, or unassigned owners. Items lacking implementation within the mandatory delivery scope must be marked FAIL/NOT RUN rather than N/A.

### R1-02: Unified Failure Propagation, Timeouts, and Result Formats (F02)

Responsibility: Build / Verification.

Mandatory deliverables:

- Remediate Make, Python runner, IR runner, assembler, and simulator invocation chains; any upstream non-zero exit code must not be swallowed by `tail`, shell pipes, or log formatters.
- `rtl_harness.run/run_system` must concurrently inspect process return codes, expected results, and complete output; determining success solely via `'PASS' in stdout` is strictly prohibited.
- All subprocesses must have wall-clock timeouts; RTL simulations must additionally enforce cycle timeouts. Subprocess timeouts must terminate the entire process group, reclaim descendant processes, and output TIMEOUT.
- Strict mode fails on any skip, xfail, or xpass; development mode may retain optional tests, but cannot generate R1 PASS evidence.
- Test builds, inputs, outputs, and dumps must utilize independent per-run and per-case directories; `cc_<name>` files must not be shared across runs, and residual artifacts from previous executions must never be read.

The following failure injections are mandatory and must be verified individually; test harnesses must observe expected failures, and the harness itself returning 0 indicates a passing negative test:

| ID | Injection | Mandatory Observed Result |
|---|---|---|
| RUN-01 | `TRITON_PYTHON=/usr/bin/false` invoking two E2E Make targets | Both targets exit non-zero |
| RUN-02 | Absent interpreter, Triton import exception, missing FileCheck | Explicit missing dependency diagnostic, strict non-zero; not SKIP |
| RUN-03 | Python assertion/error, non-zero assembler exit | Top-level gate non-zero, original traceback/stderr preserved |
| RUN-04 | Simulator outputs PASS then exits 7 | Judged FAIL, exit=7 recorded |
| RUN-05 | Simulator exits 0 with no result, concurrent PASS/FAIL, or corrupted format | All judged FAIL |
| RUN-06 | Subprocess hangs and spawns a hanging child process | Terminates entire process group at deadline; no surviving descendant processes within 5 seconds |
| RUN-07 | Empty suite, missing one mandatory case, duplicate case ID, skip, xfail, xpass | All judged FAIL |
| RUN-08 | Current dump deleted while retaining prior valid dump | Judged FAIL, does not consume stale output |
| RUN-09 | Shell pipeline fails at first stage and succeeds at final stage | Judged FAIL |
| RUN-10 | Two concurrent runs of identical case with distinct inputs/outputs | Both results correct, no artifact collision or cross-reading |

Fixed timeout limits: single metadata/unit/IR case: 120 seconds; single RTL simulation: 600 seconds; single native build step: 3600 seconds; entire acceptance run in pre-configured environment: 7200 seconds; standalone bootstrap: 14400 seconds. Test managers must terminate relevant subprocesses within 5 seconds of deadlines and report failure; test managers themselves must be supervised by an outer watchdog. Release runs must not increase or disable these limits via environment variables.

Failure injections use controlled 1-second mock process timeouts and need not wait the full 600 seconds. Baseline RTL cycle limits must not be inflated to mask progress regressions.

### R1-03: Build Dependencies, Invalidation, and Read-Only Artifact Verification (F04)

Responsibility: Build / RTL.

Mandatory deliverables: Unify source/include dependencies for every benchmark; incorporate `rtl/gen`, all `.vh` files, testbenches, memory models, imported files of Python generators, tool versions, profiles, seeds, and vector counts. Record build fingerprints in outputs; compilation parameter changes must invalidate caches.

Decouple `generate` from `check-generated`. The latter must not automatically generate or fix files before claiming consistency. The check path must cover ISA schemas, TableGen/C++ tables, opcode maps in documentation, all targets, XLS generated RTL, and associated tables.

| ID | Mandatory Experiment | Success Condition |
|---|---|---|
| BUILD-01 | Modify core RTL, common `.vh`, generated RTL, and memory model once each | All affected benchmarks re-invoke compiler; not just printing "up to date" |
| BUILD-02 | Switch 2/4 RCTX, SFU, slow/ooo, seed, vector count, and tool fingerprint | New output corresponds to new configuration; cannot reuse stale build |
| BUILD-03 | Modify generator or imported oracle/constants | Dependent outputs re-generated; necessary downstream compilations occur |
| BUILD-04 | Deliberately introduce stale schema/target/XLS RTL and run check only | Non-zero exit; tracked source tree matches pre-check file-by-file hashes |
| BUILD-05 | Clean build vs. legal incremental build on identical inputs | Binary, generated files, normalized reports, numerical outputs, deterministic cycle count identical |
| BUILD-06 | Repeated build without input changes | No redundant re-compilation/re-generation of dependencies; tests designated for every run may execute |
| BUILD-07 | Deliberately inject stale simulation/compilation artifacts from another profile | Fingerprint verification rejects prior to execution |

BUILD-01 through BUILD-04 must execute in isolated checkouts without dirtying the candidate tree. Relying solely on `make -n` is insufficient for final evidence; actual execution and artifact provenance must be archived.

### R1-04: Unified Profiles and Actual Instance Capabilities (F03, F05 Boundary)

Responsibility: RTL / Compiler.

Mandatory deliverables: Derive elaboration parameters, target JSONs, capability expectations, and test commands from a single configuration source; parameter overrides must propagate to physical instances. Correct system SFU=false and standalone core num_cores/mesh descriptions; tag compilation artifacts with profile ID, configuration hash, and ISA/schema version; existing simulation launchers must verify these before execution.

| ID | Test Case | Mandatory Result |
|---|---|---|
| CFG-01 | Top-level capability readback across three profiles | All readable resources/capabilities match item-by-item; system reads back all 4 tiles |
| CFG-02 | Configuration fields without hardware readback registers | Verified via elaboration-exported instance parameter list; not self-attestation by re-reading the same JSON |
| CFG-03 | Core SFU positive case vs. system SFU negative case | Core executes numerical computation; compiler rejects system up-front; direct instruction injection triggers specified hardware fault |
| CFG-04 | Deliberately mismatched target configuration hash, ISA version, or required capability | Launcher rejects before executing first instruction, no memory side effects |
| CFG-05 | Illegal topology, RCTX count/width mismatch, RF base out of bounds, invalid parameter combinations | Configuration/elaboration exits non-zero; no index truncation or silent correction |
| CFG-06 | `core-4r` profile check | Actual 4 RCTXs; cannot copy spurious 4-core information from `system-2x2` JSON |

Keeping system SFU disabled is part of the correct profile; SFU must not be enabled without verification merely to eliminate discrepancies. All parameter positive and negative tests must route through the identical configuration entry point.

### R1-05: Instruction-Accurate Resource Footprint Calculation (F01)

Responsibility: Compiler / ISA.

Mandatory deliverables: Establish unified explicit and implicit resource effect specifications for allocators, ownership checkers, lowerings, and diagnostics. MMA footprints must be derived dynamically from actual K, datatype, and layout; lifecycles of DMA coordinates, source/destination windows, accumulators, barriers, and descriptors must follow frozen contracts, not default target shapes.

With VLEN=16 and ELEN=32, one VREG equals 64 B. For M=N=16, each A/B fragment must satisfy:

| Input Format | K16 | K32 | K64 |
|---|---:|---:|---:|
| FP16/BF16 | 8 VREG | 16 VREG | 32 VREG |
| E5M2/E4M3 | 4 VREG | 8 VREG | 16 VREG |

| ID | Test Case | Mandatory Result |
|---|---|---|
| FP-01 | 12 combinations of the above 4 dtypes × 3 K values | Read footprint matches independent formula exactly; cannot underestimate or mask via inflated resources |
| FP-02 | Shorten A and B by 1 register for each combination | Compilation rejected; diagnostic reports actual K/dtype, required vs. available range |
| FP-03 | Exact length, length+1, beyond logical/physical RF end for each combination | Legal range accepted; out-of-bounds rejected; exceeding profile budget reports resource error |
| FP-04 | Audit negative case of K64 + 8-register FP16 windows | Rejected during compilation; must not defer to RTL fault |
| FP-05 | Rank=1/2 coordinate contiguous range, endpoint out of bounds, illegal dtype/K/rank/window offset | Legal values accurately modeled; illegal values yield uniform diagnostics without wrapping |
| FP-06 | Distinct role-local names mapping to overlapping physical RF | Detects physical overlap post-allocation; cannot rely solely on SSA identity |

FP-01 is a local resource formula test; it does not force a 32-logical-VREG profile to host two 32-register windows simultaneously. Insufficient capacity must be rejected during global resource allocation without skipping footprint test cases. Independent formula oracles must not invoke the footprint function under test.

### R1-06: Mandatory Asynchronous Safety Checks Across All Compilation Entry Points (F01, F06)

Responsibility: Compiler.

Mandatory deliverables:

- Verify logical lifecycles before allocation and physical ranges after allocation; all issued read/write operations must pass unified checking; modifying `_op_window_put` alone is prohibited.
- Support reader sets, exclusive writers, accumulator access ordering, event/barrier generations, and multi-role physical aliasing; each resource is reclaimed only upon corresponding completion/release.
- Verify and freeze real snapshot/read moments of DMA coordinates against ISA/RTL contracts: if snapshotted at issue, prove the source register is not read thereafter; if delayed, allocator must hold it until specified events. Premature deallocation based on unproven assumptions of "should be finished reading" is prohibited.
- Missing mandatory completion events, unknown resource effects, or unverified control flow must be rejected; operations must never be assumed side-effect-free by default.
- `check_protocol=False` must not disable issue safety. Debug model toggles may be retained, but all supported compilation paths must execute non-bypassable static and post-allocation checks; low-level raw assemblers do not constitute safe compiler entry points and must be explicitly distinguished in documentation.
- Compiling identical IR repeatedly or compiling roles in varying order must not be contaminated by shared mutable state such as `Value.reg`.

Every negative case below must pair with a valid positive case correcting only events, ordering, or ranges; positive cases cannot pass by eliminating computation:

| ID | Mandatory Defect or Constraint Tested |
|---|---|
| OWN-01 | Overwrite A/B while MMA is outstanding via window_put, regular vector write, or DMA refill |
| OWN-02 | Read/compute destination while DMA write is incomplete; two writers concurrently occupying overlapping ranges |
| OWN-03 | Asynchronous store source overwritten before source-reusable; semantics verified at least in MLIR/protocol layer containing this op |
| OWN-04 | DMA coordinates reused prior to contract-permitted moment; reuse allowed after completion |
| OWN-05 | Accumulator read, cleared, or reallocated while outstanding; legal hardware serialized updates accepted under explicit contract |
| OWN-06 | Missing await, waiting on wrong event / wrong generation, double release, expired event releasing new lease |
| OWN-07 | Two consumers where only one releases before window reuse; must await final consumer release |
| OWN-08 | Partial A/B overlap, multi-role physical aliasing, implicit read footprint exceeding explicit base register |
| OWN-09 | Predicated-off async still completes once; error completion publishes no valid data and wakes waiters |
| OWN-10 | Window, barrier, descriptor, accumulator, role count, or RF budget exceeded; partial issue forbidden |
| OWN-11 | Supported rolled/unrolled loops with cross-iteration overwrites, omitted releases, or exit without draining |
| OWN-12 | Varied role compilation orders, compiling identical IR twice, compiling after switching targets |

Acceptance: All violations rejected at compile time; error diagnostics are deterministic, specifying operation location (IR op/role ID if source location unavailable), resource type/range, current owner, conflicting event, and remediation guidance. Legal inputs for OWN-12 must produce deterministically identical assembly/binary and semantic reports.

For OWN-09 error completions, compiler tests verify protocol construction, while RTL tests verify runtime wakeups and faults; compilers are not expected to predict external dynamic memory faults. Tests that cannot be expressed in the current layer must be assigned to the layer where they can be expressed, rather than marked as skips.

### R1-07: MLIR and TTIR Support Boundaries Must Not Be Silently Crossed (F06, Partial F07)

Responsibility: Compiler.

Mandatory deliverables: MLIR verifier must use unified resource semantics and cover accumulators/aliasing; R1 MLIR path may be restricted to single blocks, but must explicitly reject unsupported multi-block constructs, nested regions, and cross-block leases. Existing Python specialized rolled pipelines must be proven via template protocol invariants or dataflow analysis, rather than simulating only the initial iteration before claiming all iterations are safe.

TTIR importer must select only the designated target function; function arguments must bind by signature name/index rather than dictionary insertion order; type, shape, operation, and constant parsing failures must return structured diagnostics rather than silently converting or crashing.

| ID | Mandatory Test Case | Success Criterion |
|---|---|---|
| IR-01 | MLIR single-block legal DMA/MMA/store/await; applicable OWN negative suite | Positives pass, negatives rejected per resource contract |
| IR-02 | Multi-block, branch with single-sided await, nested regions, event passed via block argument | Unsupported constructs explicitly report unsupported; must not accept unchecked |
| IR-03 | Python existing rolled K64/K128, 3-slot K96, and invalid cross-iteration reuse | Original positives pass; illegal lifecycles rejected; cannot disable loops wholesale |
| IR-04 | Multi-function module, target non-existent, varied dict insertion orders, missing/extra arguments, type mismatch | Only target function compiled; identical semantics yield identical result; invalid input cleanly rejected |
| IR-05 | BLOCK≠VLEN, unsupported scf/dot/layout/op/type, unknown resource effect | Explicitly rejected before or during lowering; no executable artifacts generated |
| IR-06 | Floating-point constants: ±0, finite, NaN/Inf, frontend literal encodings; corrupted parsing | Supported maintain bit semantics; unsupported/corrupted produce diagnostics without reading as another value |
| IR-07 | Same resource safety test in Python IR and MLIR | Consistent legality conclusions; discrepancies must be resolved, not masked |

In the absence of a general CFG implementation, explicit rejection in IR-02 defines the frozen support boundary; it does not constitute deleting existing capabilities. Diagnostics must not substitute KeyError, assertion traceback, or interpreter crashes for standard user error reporting.

### R1-08: Preserve and Expand RTL and E2E Regressions

Responsibility: Verification / RTL / Compiler.

Mandatory deliverables: Migrate all baseline tests to the unified runner; feed new compiler safety positive cases into RTL; add observable checks for simultaneous same-cycle completions, delayed/out-of-order responses, loop slot reuse, and multi-consumer releases; failing seeds must be independently replayable.

Fixed baseline regression matrix:

| ID | Mandatory Scope | Non-Waivable Scale / Result Requirements |
|---|---|---|
| REG-01 | All semantic cases from audit baseline | All preserved and passing; 7 skipped compiler-unit tests executed via actual entry points, deduplication permitted but cases cannot disappear |
| REG-02 | Baseline compiler RTL, IR, real Triton single-core/system | 67 compiler RTL test semantics, all cases across 2 IR files, 5 single-core and 2 system Triton test semantics executed |
| REG-03 | All baseline core programs | fast/slow/ooo all executed; expected fault programs verify exact fault and unaffected roles |
| REG-04 | Pipeline | iteration ∈ {0,1,2,3,5,6} × stages ∈ {2,3} × memory ∈ {fast,slow,ooo}, all 36 combinations pass |
| REG-05 | GEMM | Original FP16/BF16, strides, all tails, rolled/unrolled preserved; new K16/32/64 footprint/issue tests added |
| REG-06 | NUCA | fast/slow/ooo, ≥800 requests per seed per mode; hit-past-block, eviction/dirty WB, target/MSHR pressure counters genuinely triggered |
| REG-07 | Mesh | ≥20,000 packets per seed, dual VNs, all src-dst pairs, randomized ingress backpressure; zero packet loss, zero duplication, zero misrouting, zero in-stream reordering |
| REG-08 | Top | t10_mesh/t24_amo_mesh × single-role/all-roles: all 4 runs pass |
| REG-09 | Two consumers | core-4r, last-release semantics pass under slow/ooo; delaying different consumers verifies no premature reuse |
| REG-10 | Async errors and completion collision | DMA/MMA completions in same cycle both counted; error wakeup, suppressing invalid data, final drain, and fault match contract |
| REG-11 | Real Triton closure | TTIR generated by pinned Triton frontend; single-core masked tail, multi-role results, system 4-block output and remote traffic checks genuinely executed |

The 36 combinations of REG-04 each execute once on fixed stimulus; slow/ooo then execute the additional seeds below. REG-06/07, REG-09/10, and slow/ooo for REG-04 use **three seeds: {1, 7, 42}**; seeds must propagate to the PRNG controlling stimulus, stalls, and reordering, rather than merely being recorded in logs. Initial R1 acceptance must not truncate this matrix. Seed control additions must preserve baseline default vector/program semantics without swapping seeds to bypass failures.

All seed paths record input/scheduling trace hashes; the three seeds must produce distinct execution traces, while repeated runs with identical seeds must be deterministic and replayable. For combinatorial coverage, drivers must report "triggered" counters or assertion coverage; executing 20,000 packets without backpressure does not pass REG-07.

Floating-point: Exact ISA operations compare against existing exact oracles and specified NaN/zero behavior; SFU compares against error bounds defined in the current ISA. Exact comparisons must not be converted to tolerances, and SFU error bounds must not be relaxed. Preserve baseline default vector scales: FMA 41,240; decoder 773; conversions 866,495 inputs × 2 instances; SFU unit 16,740; core SFU 1,042 batches × 16 lanes; multi-format FPU 5,012. Restructured generators may increase but not decrease counts, while preserving original fixed-seed input semantics; repeating identical vectors to inflate counts is prohibited.

### R1-09: Reproducible Toolchain and Generation Pipeline (F08)

Responsibility: Build / Toolchain.

Mandatory deliverables:

- Pinned lockfiles fixing Triton/LLVM commits, Python and dependency versions, Icarus, CMake/Ninja, generation tools, and base OS/image digest; doctor checks versions, hashes, platform, and required capabilities.
- Establish mandatory **Linux x86_64** acceptance environment; reference runner requires at least 8 vCPUs, 32 GiB RAM, 100 GiB available disk; exact OS/image digest and tool locks must be pinned in the candidate commit. Do not reference maintainer-specific virtual environment paths.
- Provide local Linux or container execution for XLS with pinned revisions/dependencies; regenerate tracked RTL from all DSLX, tables, and parameters without relying on personal SSH accounts, private keys, or remote `cur` directories.
- Permit downloading public/licensed tool packages with fixed checksums; unverified local caches or floating tags must not serve as locked versions. Lockfiles must record source, version, and digest for every external dependency.
- Artifact provenance must cover transitive dependencies, not just top-level `.x` hashes. Tracked sources must not undergo automatic mutations during execution.

Acceptance: Two mutually isolated temporary environments A and B bootstrap and build from the same clean candidate checkout; B must not copy A's build, virtualenv, or generated artifacts, but may use verified read-only download caches. All R1 tests pass in both environments; ISA, profile, and XLS generated artifacts and target binaries must yield identical hashes. Intermediate host binaries containing paths/timestamps need not be byte-identical, but normalized exclusion fields must be pre-listed in the tracked inventory and must not strip semantic configuration, numerical results, or cycle counts.

macOS is retained as a development platform; R1 does not permit substituting macOS success for Linux dual-environment acceptance, nor claiming full macOS R1 certification without equivalent evidence.

### R1-10: CI Gating and Unmixable Acceptance Artifacts

Responsibility: CI / Verification.

The following stable entry points must be provided. **These commands are pending interface requirements; this document does not claim they currently exist:**

```sh
make r1-doctor
make r1-check-generated
make r1-gate-selftest
make r1-acceptance
make r1-verify-evidence EVIDENCE=<run-directory>
```

Failure of any entry point must exit non-zero; `r1-acceptance` fixes strict mode and retry=0, executing prerequisite checks and all mandatory tests without allowing users to disable any subset via flags. Fresh environment generation and source mutation experiments may run as managed CI jobs, but their results must aggregate into the single candidate acceptance rather than manually assembling a composite PASS.

Each run must archive at least:

```text
run.json                    commit, tree, clean status, environment/tool lock hashes, start/end time
profiles/                   configuration inputs, generated targets, elaboration parameters, capability readbacks
inventory.json              required/discovered/executed ID sets and mappings
results.json + junit.xml    per-case status, return code, duration, seed, profile, evidence paths
logs/                       full stdout/stderr; not truncated to tail
build/                      actual commands, dependency graph/fingerprints, generated artifacts/provenance hashes
cases/                      inputs, TTIR/TNCA IR, allocation/footprints, assembly/binary, outputs/counters
mutations/                  actual results of failure injection, build invalidation, checker mutations
checksums.sha256            integrity manifest across all artifacts
acceptance.json             final machine verdict and evidence corresponding to each work package
```

Acceptance: `r1-verify-evidence` exits non-zero upon missing files, a single modified output byte, mixed commits/profiles, deleted tests, fabricated discovered counts, or missing terminal states. The verifier must recompute hashes and set relations without blindly trusting `acceptance.json`'s PASS. CI success status must derive from this verifier; artifact upload failures also constitute run failure.

### R1-11: Proving Checkers Truly Detect Failures (Mutation)

Responsibility: Verification; modifications and executions take place in isolated clones.

Controlled mutations must be executed individually without committing mutations to the candidate tree:

| ID | Deliberate Fault Injection | Mandatory Detecting Test Case |
|---|---|---|
| MUT-01 | Remove ownership check for standard/window writes | OWN-01; cannot rely on fortuitous RTL numerical error |
| MUT-02 | Revert actual K footprint to target default K16 | FP-02/FP-04 |
| MUT-03 | Remove build dependencies for generated RTL or `.vh` | BUILD-01 |
| MUT-04 | Change system target SFU to true | CFG-01/CFG-03 |
| MUT-05 | Restore `python ... | tail` exit code swallowing | RUN-01 |
| MUT-06 | Permit unknown resource effects or unsupported CFG without check | IR-02/IR-05 |
| MUT-07 | Reclaim window prior to final consumer release | OWN-07/REG-09 |

Acceptance: Each of the seven mutations causes at least the specified test to fail, with failure diagnostics matching the injected fault; without mutations, baseline tests pass; candidate source hashes remain unchanged after mutation runs. Merely increasing test counts is not accepted as evidence for this work package.

### R1-12: Current Status Documentation, Handoff, and Final Signoff

Responsibility: Maintainers.

Must update `Readme.md`, `compiler/README.md`, `doc/conformance.md`, and toolchain status notes to reference the current support matrix, three profiles, strict test entry points, known limitations, and this contract. Historical audit records must be preserved without removing past defects from documentation.

Must attach the phase completion report: assignees, implementation commits, test case IDs, evidence paths, and final status for each work package. Any newly discovered correctness or verification blockers within the supported scope must receive regression tests and fixes; they cannot be excused merely because they were not listed in the initial F01–F04.

Acceptance: No unconditional completion claims (such as "C0/C1/C4 done") contradicting the actual support matrix; prototypes must not be represented as production-ready; unverified future phases must be clearly noted. All 12 items in the signoff table must be PASS, with all evidence validated by the automated verifier on the identical candidate commit.

## 4. Implementation Order and Dependencies

| Batch | Work Packages | Permitted Progression Post-Batch |
|---|---|---|
| 1 | R1-01, R1-02 | Frozen cases and trustworthy failure propagation available; root-cause fixes can begin |
| 2 | R1-03, R1-04, R1-09 | Build, profile, and toolchain trustworthy; comparable remediation evidence can be captured |
| 3 | R1-05, R1-06, R1-07 | Unified safety boundaries implemented across actual compilation entry points |
| 4 | R1-08, R1-11 | Full closure across positive tests, negative tests, stress regressions, and mutation detection |
| 5 | R1-10, R1-12 | Dual-environment full acceptance, evidence verification, and phase signoff |

CI frameworks and documentation may be developed concurrently, but cannot be signed off before dependencies conclude. Each fix commit runs relevant positive and negative tests; candidate release must execute the full suite from scratch without aggregating piecemeal per-commit runs into the final acceptance verdict.

## 5. Final Verdict Algorithm and Signoff Table

The final gate must execute the following decision algorithm, exiting non-zero if any condition fails:

```text
Candidate commit pinned AND source tree clean
AND configurations of the three profiles match top-level actual instances
AND tool locks / generation pipeline / dependency change experiments all PASS
AND required IDs == discovered IDs == executed IDs
AND all mandatory cases == PASS
AND skip / error / fail / xfail / xpass / timeout / retry / orphan_process all == 0
AND expected rejections of all negative cases and mutations verified by test runner
AND existing valid capabilities have no regression, newly discovered error entry points closed
AND independent environments A and B both fully PASS with deterministically identical specified artifacts
AND evidence files complete, hashes correct, commit/profile/run attribution consistent
AND R1-01 through R1-12 all PASS
=> ACCEPTED
OTHERWISE => NOT ACCEPTED
```

Comparison of required, discovered, and executed IDs is performed against unique execution IDs expanded per environment, profile, and seed; shared semantic case IDs are preserved for tracking. Executing the same test case twice cannot be counted as covering another missing test.

| Work Package | Initial Status | Assignee | Evidence & Final Verdict |
|---|---|---|---|
| R1-01 Specifications / Baseline | NOT VERIFIED | To be filled during execution | Undelivered |
| R1-02 Runner / Failure Propagation | NOT VERIFIED | To be filled during execution | Undelivered |
| R1-03 Build / Generated Artifacts | NOT VERIFIED | To be filled during execution | Undelivered |
| R1-04 Profile / Capabilities | NOT VERIFIED | To be filled during execution | Undelivered |
| R1-05 Footprint Calculation | NOT VERIFIED | To be filled during execution | Undelivered |
| R1-06 Ownership Verification | NOT VERIFIED | To be filled during execution | Undelivered |
| R1-07 IR / Frontend Boundaries | NOT VERIFIED | To be filled during execution | Undelivered |
| R1-08 RTL / E2E Regressions | NOT VERIFIED | To be filled during execution | Undelivered |
| R1-09 Reproducible Environments | NOT VERIFIED | To be filled during execution | Undelivered |
| R1-10 CI / Artifact Verification | NOT VERIFIED | To be filled during execution | Undelivered |
| R1-11 Mutation Verification | NOT VERIFIED | To be filled during execution | Undelivered |
| R1-12 Documentation / Handoff | NOT VERIFIED | To be filled during execution | Undelivered |

**The writing of this document does not alter the initial status above. Only actually completing all tasks and obtaining the requisite evidence permits declaring "R1 Complete" and initiating Phase 2 (R2) delivery dependent on R1 acceptance results.**
