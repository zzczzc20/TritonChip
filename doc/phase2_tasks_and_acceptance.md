# Phase 2 (R2): Unified AOT Compiler and Simulation Runtime Tasks and Acceptance Contract

Version: 1.0, 2026-09-19. Status: **Specification established; implementation not yet accepted; does not signify completion of R1.**

Prerequisite contract: [R1 tasks & acceptance](phase1_tasks_and_acceptance.md). Roadmap basis: [Production Readiness Audit](production_readiness_audit_2026-09-19.md), [Architecture Design](design/tnca_v1_architecture_draft.md) Section 18, [Compiler Design](design/tnca_v1_compiler_draft.md) Sections 2, 10, and 14–16.

This document freezes the delivery scope of R2 in advance during the execution of R1. Real-time progress of R1 is governed by its implementation evidence; this document does not duplicate its completion status or alter R1 thresholds. Prototyping of R2 designs, interfaces, and isolated components may proceed in advance; **final acceptance must be grounded on an already accepted R1 baseline and must re-pass the entire R1 suite on the R2 candidate commit.**

## 1. Phase Objectives and Irreplaceable Deliverables

In an installed environment without a source checkout, the user executes via the official CLI/Python API:

```text
Triton source program + signature + constexpr + profile
  → Inspectable, cacheable AOT artifact
  → Versioned manifest + fixed machine code + compilation report/IR dumps
  → Runtime allocates/uploads buffers, binds dynamic arguments, submits grid
  → Persistent RTL simulation device executes batches of programs
  → Completion/error events
  → Runtime downloads results, reclaims resources, proceeds to execute next kernel
```

**The identical artifact must be repeatedly runnable across different addresses, data, dynamic scalars, and grids without recompilation or modification of machine code.** When three programs run back-to-back as A→B→C, intermediate results must remain in the memory system of the same simulated device, without computation in Python or relaying through internal testbench arrays.

The delivery designation of R2 is "Installable Experimental AOT Compiler and RTL Simulation Runtime." This does not constitute FPGA/PCIe drivers, real-device host coherence, or production silicon qualification.

### 1.1 Scope Freeze

| Mandatory in R2 | Deferred to Subsequent Phases (Not Claimed Supported) |
|---|---|
| Non-editable installable package, AOT CLI/Python API, formal compiler driver | Full PyTorch device/JIT runtime integration |
| Pinned Triton adapter, real narrow MLIR compilation main path | Arbitrary Triton/Gluon, arbitrary TTGIR, or CUDA/PTX-specific semantics |
| manifest/ABI v1, dynamic arguments, deterministic artifact and cache | Complete LLVM target, dynamic linker, cross-profile binary compatibility |
| Arbitrary constrained grids and repeated launches on the same 2×2 RTL device | 4×4 mesh, hardware work stealing, concurrent multi-kernel execution, preemption |
| Buffer/event/resource admission, draining, defined fault recovery | IOMMU, multi-tenant isolation, demand paging, transparent migration |
| Simulated host memory access path implementable as a hardware protocol | Real PCIe/SoC transport or OS kernel driver |
| Consumable resource reports, deterministic diagnostics, and complete evidence | Attention, Conv, new cache/flow policies, PPA/throughput optimization commitments |

Support boundaries must not be used to delete R1 functionality. R1 dedicated GEMM, multi-role, and rolled pipeline paths must be preserved; capabilities not yet migrated to the real Triton frontend must be faithfully identified as TNCA IR/AOT inputs and must not be misrepresented as "already supporting Triton GEMM."

### 1.2 Preservation of R1 Strict Rules

- Zero unexpected failures, errors, skips, xfail, xpass, or timeouts across all mandatory test cases; retry=0, no waivers.
- In fault-injection tests, expected kernel FAULT/TIMEOUT must be verified by the test driver; a driver PASS does not equal a faulty kernel SUCCESS; the two statuses must be recorded separately.
- Missing dependencies, missing evidence, incomplete results, or unknown statuses constitute failure. Prior testing in R1 cannot substitute for R1 regression on the candidate commit.
- Achieving positive acceptance passes via recompilation, assembly modification, test-only barrier insertion, implicit downloading/uploading of intermediate values, simulator restarts, or pre-filling correct outputs is strictly prohibited.
- All numerical contracts, R1 profile resource budgets, and mandatory preserved tests must not be relaxed; there is no path of "removing checks first to get the runtime running."
- Only evidence generated from the identical clean candidate commit, its locked toolchain, and controlled profiles can be jointly signed off. Any change to implementation, contracts, or inventories requires a new full acceptance run.

## 2. R1→R2 Interface Contract and Configuration

### 2.1 Prerequisites from R1 Prior to Final Acceptance

| Input | How R2 Uses It | Verdict When Missing |
|---|---|---|
| R1 ACCEPTED evidence, commit/tree hash, tool locks, and normalization rules | Establish inheritance baseline and regression inventory | NOT ACCEPTED |
| Three real profiles, capability readback, elaboration fingerprint | Compilation, loading, and runtime consistency verification | Reject compilation/loading, do not guess capabilities |
| Mandatory footprint/ownership check entry points | Reused prior to emission and post-allocation across all AOT paths | Must not publish executable artifacts |
| Explicit support matrix and all semantic case IDs | Confirm no functional regression before and after migration | Incomplete mandatory set constitutes failure |
| Fail-safe runner, evidence validator, and generation toolchain | Extended by R2 rather than bypassed | Gate self-test failure halts acceptance |

R2 does not freeze internally evolving R1 function signatures in this document. During implementation, adapter boundaries, public data structures, and version contracts must be committed; locations using internal interfaces must be unified within adapters, not scattered across tests or runtime.

### 2.2 Profiles and Permitted RTL Changes

R1's `core-2r-sfu`, `core-4r-sfu`, and `system-2x2-nosfu` resource and SFU constraints are retained. All three profiles must support post-installation loading, argument binding, basic execution, and result download. General grid/multi-batch scheduling uses `system-2x2-nosfu` as the mandatory full-system acceptance target; core profiles only commit to grid=(1,1,1), and reject other grids up front.

Modifying RTL in R2 to support per-tile PID/launch, host memory access, job generation, and resource draining is part of this phase's scope. Implementation and interface fingerprints of profiles must be updated, and R1 must be re-run on the modified top level. **Resource names may remain unchanged, but implementation hashes must not masquerade as R1's legacy hashes.** Artifacts with R1 legacy hashes are explicitly rejected in R2 or regenerated via recompilation; they cannot be loaded silently.

Do not add RCTX, RF, NUCA, MSHR, or SFU resources to circumvent scheduling/allocation bugs. The system remains strictly 2×2; implementing "more programs" as "more physical tiles" is prohibited.

## 3. Product Contract: ABI, Artifact, Grid, and Completion Semantics

This section specifies public conventions that implementations must satisfy, rather than describing current code state.

### 3.1 Artifact and Manifest v1

An artifact contains at least `manifest.json`, immutable machine code, a read-only constant section (if used), resource/compilation reports, SHA-256 digests for each section, and provenance. Container formats and canonical serialization rules are codified in a controlled schema; loading must not execute bundled Python/pickle or shell commands. Debug information may be stored independently, but must not be a dependency for numerical execution.

Mandatory fields/information in the manifest:

| Category | Mandatory Fields / Information |
|---|---|
| Identity | Artifact format/ABI/ISA versions, kernel name, compiler/source/tool-lock hashes, constexpr and compilation options |
| Target | Profile ID, configuration/implementation fingerprint, required capabilities |
| Code | Code/constant section lengths, hashes, alignments, per-role entry points or explicit shared dispatch entry |
| Arguments | Argument names/order/types/offsets/sizes/alignments, read/write directions, buffer minimum extent expressions, permitted alias relationships |
| Execution | Active role mask, program→role mapping, supported grid dimensions, dynamic argument constraints |
| Resources | Physical mappings for per-role private RF and shared windows, ARF/barrier/descriptor/command upper bounds, spill/scratch bytes |
| Semantics | Completion scope, error contract, existence of cross-program dependencies, ownership verification version/result |

R2 artifacts support only an exact profile fingerprint; cross-profile portability is not in v1. Unknown versions, unknown required fields/capabilities, overlapping code sections, out-of-bounds entries, invalid masks, and conflicting resource mappings are all rejected. Ignorable extensions are permitted only in schema-defined `optional_metadata`, and must not affect addressing, resources, completion, or numerical semantics.

Runtime parameters, buffer addresses, grids, job IDs, and random seeds must not enter the compilation cache key; parameters explicitly specified as constexpr must enter the key. If compilation options, code, dependencies, profile, or tool locks change, the key must change. The manifest hash is used for integrity and compatibility, and does not claim to provide authentication against malicious producers.

### 3.2 Kernel Parameter ABI v1

- Device is little-endian; supports `i32/u32/i64/u64/f32/device_ptr64`. In this phase, FP16/BF16 are passed via buffer elements; other scalar ABI types are explicitly rejected.
- Arguments occupy 8-byte slots in frozen signature order, offset=8×index, argument block 8-byte aligned. 32-bit/f32 values occupy the lower 4 bytes, with upper 4 bytes zeroed; 64-bit arguments occupy all 8 bytes. A parameterless block has length 0; load header/ABI version is passed in the launch record, not mixed into the argument block.
- `f32` is encoded as a binary32 bit pattern, preserving ±0 and non-finite values permitted by schema/numerical contracts; bits must not be lost through decimal string round-trips. Out-of-range integers raise errors and are not truncated.
- Pointer arguments in the API receive a buffer handle + byte offset for the current session; the runtime verifies validity, bounds, alignment, read/write permissions, and alias contracts before writing device addresses. Arbitrary host pointers cannot be treated as device pointers.
- The argument block is snapshotted at submission and remains immutable until the task drains; modifying caller Python dicts/objects post-submission cannot alter already submitted tasks.
- `launch_arg_base` points to this parameter area; the kernel loads dynamic parameters via the ABI. **Embedding dynamic addresses/scalars into code or patching instructions at each launch is prohibited.** Arranging code/constant positions according to a controlled section table during loading is permitted, but must not masquerade as recompilation per argument.
- Buffer addresses supported in v1 may be simulated device physical addresses; v1 does not provide true IOMMU isolation guarantees. Non-contiguous/gather dynamic out-of-bounds accesses must trigger a controlled fault at simulated device memory bounds checks, without wrapping or corrupting other allocated buffers.

### 3.3 Grid and Scheduling Contract

`grid=(gx,gy,gz)`, where each axis is an integer in `0..2^32-1`; negative/non-integer values are rejected. The maximum total programs for a non-empty grid is **4096**, and multiplication must check for overflow. Any dimension being 0 indicates an empty task: it completes normally with a unique event, but launches no roles and issues no kernel data accesses.

Linear mapping is fixed to `linear = x + gx*(y + gy*z)`. `program_id(axis)` and `num_programs(axis)` must read logical grid values, rather than tile IDs or current batch size. All roles of the same program share the same PID; all roles of a program and their shared resources reside on the same tile as an indivisible unit.

R2 adopts deterministic batch scheduling: up to four programs are assigned to four tiles in ascending linear ID order; a tail batch with fewer than four programs activates only the required tiles. Each batch is permitted to fully drain before the next batch begins, without requiring inter-batch overlap; restarting the simulator or recompiling the kernel between batches is prohibited. Static validation and mapping for grids up to 4096 are exhaustively verified by host/unit tests, without requiring RTL to execute 4096 programs iteratively; mandatory RTL scale is defined in the E2E matrix.

R2 does not support cross-program barriers or spin-waiting protocols that depend on other programs executing first; compilers/APIs explicitly reject recognized semantics of this kind. Tests must not rely on uncontracted data races. Device atomic operations retain R1 semantics, but do not make such cross-program progress dependencies legal.

### 3.4 Session, Events, and Memory Visibility

Each session has at most one RUNNING kernel at any time; submissions are queued FIFO, supporting at least **8 non-terminal tasks** (including RUNNING). A 9th submission returns an explicit `QUEUE_FULL` when the queue is full, and must not block the API indefinitely or overwrite older tasks.

Event state machine:

```text
SUBMITTED → VALIDATED → QUEUED → RUNNING → DRAINING → SUCCEEDED
                 Any runtime fault → FAILING → DRAINING → FAILED
                 Unrecoverable drain failure → DEVICE_LOST (session POISONED)
```

Loading/argument/admission errors are returned before any role launches and do not emit a RUNNING event. Each accepted task reaches exactly one terminal state; job IDs are unique within a session epoch. `submit` supports an explicit `depends_on` event list; it accepts only previously submitted events in this session and forbids referencing future events, preventing cycles. FIFO guarantees execution ordering, while dependency events determine execution validity: if any predecessor fails, successors terminate with `FAILED/DEPENDENCY_FAILED` without launching any role. Back-to-back A→B→C pipelines must declare dependencies; buffers written by a FAILED task are marked invalid and must be re-established via explicit upload or a new full-overwrite task before consumption by normal read tasks.

`wait(event)` is idempotent; normal host wait timeouts signify only that a terminal state has not yet been observed, and must not mark tasks as succeeded or automatically cancel them. The device execution deadline is a separate configuration, with timeouts handled via the fault path.

Prerequisites for `SUCCEEDED`: all programs/roles have terminated; DMA/MMA/LSU/return/writeback and associated completion events have drained according to contract, ensuring no further writes to stale RF/windows; outputs are readable via the **formal runtime download interface**; resources and argument blocks can be safely reclaimed.

Host upload/download must pass exclusively through well-defined transport requests: normal arbitrated home-slice memory accesses, or synthesizable quiesce + clean/invalidate/maintenance port protocols. Concrete schemes are frozen in the implementation submission and must be tested under dirty cache and host overwrite scenarios. **Using testbench hierarchical references to read/write cache tag/data/RF arrays to implement product data interfaces is prohibited.** Internal testbench observation may still be used for verification assertions, but cannot serve as runtime result sources.

Host memory accesses occur when the device is quiescent; if a buffer has un-drained task leases, free/upload/download returns `BUFFER_BUSY` without implicit waiting or data corruption. A→B kernel dependencies are realized via FIFO/events and device memory without intermediate host downloading; host reads require an explicit wait followed by download. FAILED task outputs may contain partial results and must be marked without validity guarantees; transactional rollback is not promised. `close` returns `SESSION_BUSY` when un-terminal tasks exist, forbidding silent task dropping; when a device is lost, closing and process cleanup are permitted. A normal drain followed by close releases all resources; repeated close is idempotent.

## 4. All Work Packages and Mandatory Test Cases

All sub-conditions for each case ID must be executed; during implementation, generate a controlled inventory expanding profiles, inputs, seeds, and environments, without running only a subset of scenarios from a single row.

### R2-01: R1 Transition, Interface Freeze, and Support Matrix

Responsibility: Maintainers / Compiler / Runtime.

Deliverables: R1 input lock inventory; R2 schema/ABI/API documentation; dependency boundaries among compiler, runtime, transport, and test runner; semantic case mapping; assigned owners.

| ID | Acceptance |
|---|---|
| DEP-01 | Missing/forged/incomplete R1 acceptance evidence results in non-zero R2 gate; real-time PARTIAL status of R1 cannot be treated as passing prerequisites |
| DEP-02 | Full R1 suite runs on R2 candidate, all cases preserved and passing; implementation fingerprint changes have traceable mappings |
| DEP-03 | Every external capability has an entry point, profile, parameter constraints, and positive/negative cases; no unqualified claims of "supporting Triton/GEMM/async" |

Fixes established following R1 acceptance may continue to evolve on the R2 candidate, but must satisfy R1 anew; byte-for-byte identity between the R2 candidate and R1 sign-off commit is not required.

### R2-02: Installation Package, CLI, and Public APIs

Responsibility: Build / Compiler / Runtime.

Deliverables: Non-editable wheel/sdist with locked dependencies and entry points; decoupled build and runtime package boundaries; formal `tnca-aot` and `tnca-sim` commands alongside Python APIs. The stable entry point contract to implement is:

```text
tnca-aot compile / inspect / doctor
tnca-sim run / doctor
Python: compile_aot, inspect_artifact
Python Session: allocate, upload, load, submit, wait, download, free, recover, close
```

`compile` emits artifacts; `inspect` runs without launching the simulator; `run` executes through the same runtime API. Exact parameters, return types, JSON schemas, and error codes must be documented under version control. Kernel fault, loader error, or transport lost results in a non-zero CLI exit code. Importing packages must not run tests, connect to the network, or start the simulator.

| ID | Acceptance |
|---|---|
| PKG-01 | Non-editable installation in two clean environments; compile→run→download succeeds after removing repository from sys.path/PYTHONPATH and cwd |
| PKG-02 | Neither compilation nor execution imports `compiler/test` or reads `sim/build` within checkout; succeeds from arbitrary cwd and paths containing spaces |
| PKG-03 | Execution environment requires only artifact, runtime, locked simulator/runtime assets, running without Triton/LLVM/compiler; no automatic dependency pulling on launch |
| PKG-04 | Stable diagnostics for missing runtime assets, mismatched dependencies, read-only/permission-denied output paths; CLI and Python API consistent in success/failure and data |
| PKG-05 | In-package license, version, profile schema, and required runtime assets are complete; uninstall and reinstallation do not rely on developer cache |

Building the profile's simulator during the installation/explicit prepare phase is permitted, but performing builds during each kernel launch is prohibited; prepare artifacts must carry R1 fingerprints and be archived into evidence.

### R2-03: Unified Compiler Driver and Pinned Triton Adapter

Responsibility: Compiler.

Deliverables: Single product entry point for `source + signature + constexpr + profile + options → artifact`; remove necessary logic from test bridge; reject unknown options; unify source/function selection and argument binding. All entry points invoke R1 safety checks.

| ID | Acceptance |
|---|---|
| CMP-01 | Real Triton add, mul-add, and AXPY with dynamic f32 alpha compile via public entry point; results match independent reference |
| CMP-02 | Non-existent function, multi-function module, dict reordering, unknown option, and dtype/signature mismatches handled per contract, without silently catching broad exceptions to degrade |
| CMP-03 | Feed R1 audit negative cases and all applicable OWN/FP negative cases through product input path; must be rejected, and debug/offline/optimization options cannot bypass |
| CMP-04 | Compilation failure must not leave half-finished artifacts consumable by loader as successful; preexisting artifact at same path is not corrupted |
| CMP-05 | R1 dedicated TNCA IR GEMM/multi-role inputs generate same-format artifact via explicit second input adapter; entry point marked as `tnca-ir` rather than masquerading as Triton |

Direct launching of `@triton.jit` via PyTorch launch is not required; however, the compilation flow must actually invoke the pinned Triton frontend, and cannot substitute pre-saved TTIR for CMP-01.

### R2-04: Narrow MLIR Main Path and Control Flow Boundaries

Responsibility: Compiler.

Deliverables: Pass pipeline genuinely consuming TTIR; general arithmetic/control flow retains `arith/math/scf` semantics, while TNCA expresses resources/asynchrony; real pass boundary dumps, pre/post-allocation verifiers, and connection to final machine emission. The existing Python emitter may serve as the terminal machine encoding/assembler tool, but MLIR must not be reduced to an unreferenced side show.

General TTIR control flow newly added in R2 is limited to: **uniform scalar `scf.if` (with else) and single-layer `scf.for`, trip count dynamic i32 constrained to 0..8**; permits only synchronous elementwise/memory access and loop-carried values supported in this phase. Asynchronous leases crossing this general CFG, arbitrary nested loops, while loops, and recursion are explicitly rejected. R1's verified dedicated asynchronous rolled pipeline is preserved independently and must not regress due to these restrictions.

| ID | Acceptance |
|---|---|
| MLIR-01 | CMP-01 three classes of real Triton programs emit machine code via MLIR main path and execute on RTL; deleting/breaking actual lowering passes must cause compilation failure |
| MLIR-02 | scf.if true/false branches and scf.for trip counts {0,1,2,3,8} all execute; same artifact switches branches/iterations via dynamic parameters with correct results |
| MLIR-03 | trip<0 or >8 rejected at launch constraints; unsupported CFG/cross-CFG async rejected at compile time, without emitting unchecked instructions |
| MLIR-04 | Numerical differential testing between MLIR main path and retained Python reference on jointly supported inputs; includes masked tail, alias, different scalars, and role mappings |
| MLIR-05 | Dumps include real TTIR, target execution/protocol representations, allocation results, and machine/assembly; each has stage ID/hash; duplicate printing of identical IR masquerading as multiple passes is prohibited |

Rewriting all R1 dedicated pipelines into MLIR or full upstream JIT integration via `TNCABackend.add_stages` is not required in this phase; unmigrated entry points must remain governed by R1 checks and explicitly declare their actual compilation path in reports.

### R2-05: Artifact/Manifest Schema and Loader

Responsibility: Compiler / Runtime.

Deliverables: Format v1 per Section 3, schema validation, integrity checks, profile binding, atomic emission, and pre-execution loader/admission. Configuration is cross-checked against both manifest and device readbacks, rather than comparing two copies from the same JSON.

| ID | Acceptance |
|---|---|
| ART-01 | Normal artifact can be inspected, loaded, and executed, with all mandatory fields complete; total resources in manifest match actual allocations item by item |
| ART-02 | Single-byte modification to code/constant, truncation, section overlap, out-of-bounds entry, duplicate JSON keys, unknown version/mandatory fields are each rejected |
| ART-03 | Profile/hash/ISA/capability mismatch, invalid role mask, and conflicting resource mappings are rejected before first instruction, with zero kernel write side-effects |
| ART-04 | Missing files, mixing sections from another artifact, nested path traversals, and corruptions are rejected; loading does not execute artifact code as host scripts |
| ART-05 | Overflow checks on resource/parameter/code section sizes; rejects when exceeding device available space rather than launching after allocating partially |
| ART-06 | When compilation reports/debug info are removed: missing mandatory execution contracts causes rejection; missing optional debug specified by schema does not change execution results |

v1 rejects higher or unknown format/ABI major/minor versions by default; legacy experimental file compatibility is not promised. The current v1 golden artifact must load in an isolated installation environment, without requiring re-running source generation.

### R2-06: Dynamic Parameter ABI and Buffer Contract

Responsibility: Compiler / Runtime.

Deliverables: ABI encoding/decoding, parameter loading in code, signature constraints, and safe argument snapshotting. Loader uses typed buffer handles; resource leases and lifetimes are managed by runtime.

| ID | Acceptance |
|---|---|
| ABI-01 | Six ABI types validated against independent byte-level oracle for offsets/sizes/endianness; integer bounds, f32 ±0/finite values/permitted NaN/Inf all adhere to contract |
| ABI-02 | Same artifact executes across at least three sets of different buffer addresses, N, f32 alpha, and data; code hash unchanged, compiler invocation count remains 1 |
| ABI-03 | Missing/extra parameters, out-of-bounds integers, wrong types, bad alignment, undersized buffer, offset addition overflow, and expired/cross-session handles are all rejected |
| ABI-04 | Modifying caller dict/scalar containers or reusing temporary parameter objects post-submission; submitted task sees only the submission snapshot |
| ABI-05 | Legal in-place aliased elementwise is correct; undeclared overlapping input/output aliases rejected in advance; does not rely on scheduling happening to be safe |
| ABI-06 | Non-contiguous runtime gather/scatter out-of-bounds triggers precise device memory fault; cannot corrupt canaries of adjacent buffers |

Statically verifiable address ranges must be checked at launch; addresses depending on device computation are handled by explicit simulated memory bounds checks. These bounds checks must reside in the formal transport/device model, not merely in test oracles.

### R2-07: Compilation Cache, Determinism, and Immutable Code

Responsibility: Compiler / Build.

Deliverables: Content-addressed cache, concurrent safe publishing, hit/miss root causes and traces; runtime environment contains no dependency chain capable of triggering compilation silently.

| ID | Acceptance |
|---|---|
| CACHE-01 | Source, constexpr, options, profile, and tool lock changes each cause cache miss; dynamic address/value/grid changes reuse original artifact |
| CACHE-02 | Identical inputs built in two clean environments yield identical normalized hashes for code/constant/manifest |
| CACHE-03 | Two concurrent processes compiling same key produce no corrupt artifacts; injected interruption/corrupted cache causes explicit rejection or controlled rebuild, never consuming bad code |
| CACHE-04 | Continuous launches in runtime-only environment invoke compiler/assembler/iverilog 0 times; code pages and hash remain unchanged |

Rebuilding following cache corruption is permitted for **explicit compilation requests**; implicit repairs during runtime launch are prohibited. Comparison normalization may strip only pre-listed non-semantic paths and timestamps.

### R2-08: Runtime Session, Buffers, Queues, and Events

Responsibility: Runtime.

Deliverables: Section 3 APIs and state machines, persistent simulator session, buffer allocation registry, reference/lease counts, FIFO queue, and explicit error types.

| ID | Acceptance |
|---|---|
| RT-01 | Full lifecycle: allocate→upload→load→submit→wait→download→free; 0-byte allocation explicitly rejected; zero-parameter kernel still executes |
| RT-02 | 32 consecutive successful launches in same session with alternating addresses/inputs; no simulator restart, no leaked resources; includes intermediate A→B→C device memory chain |
| RT-03 | Concurrently accepts 8 non-terminal tasks, executing FIFO; 9th returns QUEUE_FULL when full; task completion frees slots for subsequent submissions |
| RT-04 | Repeated wait calls, out-of-order event queries, host wait timeout followed by wait: terminal state unique, tasks do not duplicate execution, not automatically cancelled |
| RT-05 | Active lease free/upload/download returns BUFFER_BUSY; available after termination and draining; double free and expired event/handle rejected under control |
| RT-06 | Two sessions using same kernel/buffer names with different data maintain unconfused files, address spaces, and events; busy close rejected; close post-drain/device-lost has no process leaks; repeated close idempotent |
| RT-07 | When A succeeds, B/C execute per depends_on; when A fails, B/C do not launch and report DEPENDENCY_FAILED; cross-session/unknown dependencies rejected; invalid output buffer cannot be consumed by normal read tasks |

Queue full conditions must be sustained via controllable device hold, rather than relying on fortuitous host scheduling timings. Buffer resource counts and execution event states must be recorded separately.

### R2-09: Multi-Batch Grid, PID, and Global Resource Admission

Responsibility: Runtime / RTL / Compiler.

Deliverables: Per-tile launch mask, PID/NPID/arg_base, logical→physical mapping, resource reservation, tail-batch deactivation, and inter-batch draining. Scheduling may be controlled by the host runtime, but computation for every scheduled program must be executed by the RTL.

| ID | Acceptance |
|---|---|
| GRID-01 | grids={(0,1,1),(1,1,1),(3,1,1),(4,1,1),(5,1,1),(9,1,1),(17,1,1),(2,3,1),(2,2,2)}, each logical PID executes exactly once |
| GRID-02 | Kernel outputs PID/NPID encoding and per-program write counts; matches independent mapping oracle, proving tile ID/batch size are not masqueraded as logical values |
| GRID-03 | Tail-batch inactive tiles do not fetch instructions or issue task data requests; empty grid launches zero roles and issues zero kernel memory writes |
| GRID-04 | Same multi-role artifact executes under grid=1/4/5; roles are not split across tiles; shared window/barrier is not occupied by another program |
| GRID-05 | Validate exact resources, resource deficit by 1, illegal shared mapping, and insufficient roles; insufficient resources rejected before any role launches |
| GRID-06 | Host mapping tests cover total programs 0..4096 and 3D boundaries; 4097, negative values, 32-bit axis overflow, and product overflow explicitly rejected |
| GRID-07 | Adjacent batches alternate different data; previous batch's RF/descriptor/barrier reused only after all async responses drain; verifiable via counters and generation traces |

The GLOBAL grid scheduler does not map intra-core `num_warps`; equating it to RCTX count is prohibited. New launch interfaces require busy/admission handshaking, and must not unconditionally clear previous state with `launch_valid` mid-execution.

### R2-10: Host Memory Transport and True Completion Boundary

Responsibility: Runtime / RTL / Verification.

Deliverables: Section 3 formal upload/download protocol, dirty-line visibility, host overwrite rules, and full-device quiescent determination. Must cover in-flight states of NI, mesh, slices, DRAM, returns, and writebacks beyond core idle.

| ID | Acceptance |
|---|---|
| MEM-01 | Kernel output resides in dirty cache without natural eviction; download after successful wait retrieves new value, not old DRAM value or testbench array bypass |
| MEM-02 | Host uploads new values to cached buffer during quiescence, then runs kernel; reads new value rather than stale cache |
| MEM-03 | A writes → B reads/transforms → C verifies, without intermediate host downloads; numerically correct across multiple batches and repeated launches in same session |
| MEM-04 | Inject latency into final store/writeback/return; cannot prematurely SUCCEEDED/free/reuse argument block; terminates exactly once after latency releases |
| MEM-05 | Runtime interface audit and dynamic hooks prove upload/download do not reference hierarchical cache/RF/tag/data arrays; product E2E passes after disabling legacy harness dumps |
| MEM-06 | Each host request has address/length/direction/response and completion records; failures do not masquerade as success; out-of-range upload/download rejected with canaries intact |

MEM-01 does not mandate that "all dirty lines must be written back to DRAM at kernel completion": coherent visibility guaranteed via home slices during formal host access is sufficient. If a flush approach is chosen, flush completion is a prerequisite for event success, and its writeback response must be awaited.

### R2-11: Faults, Deadlines, Draining, and Recovery

Responsibility: Runtime / RTL / Verification.

Deliverables: Error events containing job/epoch/role/PC/instruction or address details; known fault draining; timeout/transport lost poisoning and explicit recovery. Normal launches do not reset the device; recovery must not silently substitute stale results for new outputs.

| ID | Acceptance |
|---|---|
| ERR-01 | Illegal instruction, descriptor fault, out-of-bounds memory, barrier error individually injected; events FAILED, accurately attributed, cannot be SUCCEEDED |
| ERR-02 | DMA/MMA/LSU in flight during fault: new work forbidden, completion/release paths continue; session preserved when drainable, subsequent independent legal task succeeds |
| ERR-03 | Injected non-returning transaction triggers device deadline: session POISONED, unexecuted tasks terminate FAILED with DEVICE_LOST; queue cannot hang indefinitely or reuse compromised resources |
| ERR-04 | `recover` establishes new epoch, resets/rebuilds device and transport, invalidating all old buffer/module/event handles; re-uploading and loading allows legal tasks to succeed |
| ERR-05 | Late or duplicate responses/completions from old epoch/program: recognized and discarded/errored, without writing new RF/buffer or altering new events |
| ERR-06 | Simulator abnormal exit, transport message truncation/version mismatch: DEVICE_LOST, logs preserved, non-zero CLI, no orphan child processes |
| ERR-07 | 16 alternating fault/normal submissions covering post-drain reuse and post-recover rebuild; no event/resource leaks, unexecuted tasks never reported as executed |

Restarting the simulation process during recover when draining is impossible is permitted, but this represents explicit, observable recovery from device loss; it cannot be used for RT-02, multi-batch GRID runs, or normal success paths. FAILED output offers no rollback guarantees, and APIs must prohibit treating it as successful results; canaries isolating unrelated buffers must remain intact.

Job epochs may be implemented via explicit tags or "complete drain + transport epoch isolation"; this must be proven via ERR-05 rather than asserting late responses cannot occur.

### R2-12: Real Resource Reports, Diagnostics, and User-Reproducible Examples

Responsibility: Compiler / Runtime.

Deliverables: Compilation reports automatically connected to product driver; includes per-role RF / shared window, ARF/barrier/descriptor counts, actual K/dtype footprints, DMA unroll counts, channel/role schedules, rejection schemes and reasons; runtime reports grid/batch, job, execution/wait cycles, faults, and drain results.

| ID | Acceptance |
|---|---|
| OBS-01 | Resource reports cross-checked against manifest, allocation map, and emitted instructions; costs lacking measurement/model support marked unknown, not 0 to fake zero overhead |
| OBS-02 | Forced insufficient resources, invalid ABI, wrong profile, and bad IR yield source/IR/role/resource locations and stable error codes; normal user errors produce no raw tracebacks |
| OBS-03 | Save failed artifact, parameter metadata, inputs/seeds, trace, and replay command to reproduce identical semantic error in same locked environment; host PID/timestamp reproduction not required |
| OBS-04 | Provide at least three post-installation runnable examples: dynamic masked AXPY; 3-program chain with grid>4; multi-role GEMM + fault recovery example |

GEMM examples are permitted to use explicitly labeled TNCA IR inputs, but must pass through product compiler/loader/runtime without importing test modules. Reports make no commitment to accurately predict throughput in this phase.

### R2-13: End-to-End Matrix, Differential Testing, and Mutation

Responsibility: Verification.

All comparisons use independent reference calculations; the runtime must not compute reference outputs. Seed set is fixed to `{1,7,42}`; seeds for value generation and memory stall/reorder are recorded and propagated separately. All masks and tail output regions verify untouched extents using sentinels and adjacent canaries.

| ID | Mandatory Matrix |
|---|---|
| E2E-01 | Real Triton add/mul-add/dynamic AXPY, N={0,1,15,16,17,63,64,65}, BLOCK=16, grid.x=ceil(N/16), all three seeds executed |
| E2E-02 | GRID-01 nine grid shapes, run with PID/NPID output kernel and at least one real numerical kernel each; empty grid produces no compute output; two 3D grids must genuinely execute |
| E2E-03 | MLIR-02 two branches and five trip counts, dynamic argument reuse, masked tail, pass on both core and system profiles; profile capabilities verified against actual support |
| E2E-04 | R1 TNCA IR GEMM: FP16/BF16, K={16,32,64,128}, M/N={16/16,13/11}, two roles, fast/slow/ooo; tail K=19 tested separately; resources unchanged |
| E2E-05 | Persistent session sequence from RT-02/ERR-07 and consistency/drain scenarios from MEM-01..04 executed across three seeds |
| E2E-06 | core-4r dual-consumer program executed via installed compiler/runtime/API; no reuse before last consumer releases; preserves R1 numerical results |

E2E-04 runs the full numerical/memory matrix on `core-2r-sfu`; system profile additionally runs FP16/BF16 K32, M=N=16, grid=5 multi-role multi-batch GEMM, alongside an M=13/N=11/K=19 tail block set. K128 here represents multiple K-steps of specialized GEMM, without introducing a new K128 ISA encoding. Dynamic GEMM shapes are not a new requirement in this phase; runtime addresses and inputs must remain dynamic.

To prove the new gate is effective, each of the following controlled mutations must be caught by its specified test, while non-mutated positive cases pass:

| ID | Mutation | Catching Test |
|---|---|---|
| R2M-01 | Hardcode dynamic pointer/N to values from first launch | ABI-02, CACHE-04 |
| R2M-02 | Revert PID to tile ID or NPID to 4 | GRID-01/02 |
| R2M-03 | Emit success event before final store/return completes | MEM-04 |
| R2M-04 | Change download to read DRAM directly, bypassing dirty cache consistency | MEM-01 |
| R2M-05 | Remove artifact/profile hash or capability checks | ART-02/03 |
| R2M-06 | Permit stale epoch to complete new events or write resources | ERR-05 |
| R2M-07 | Make MLIR passes dump-only without participating in actual compilation | MLIR-01/05 |
| R2M-08 | Silently restart simulator or recompile between successful launches | RT-02, CACHE-04 |

All mutations run in isolated copies without entering candidate source code; detection must correspond to the failure root cause, and cannot rely on unrelated import/build failures to register as caught.

### R2-14: Dual-Environment Delivery, CI, and Final Evidence

Responsibility: Build / CI / Maintainers.

New stable entry points (the following are interfaces to be implemented, not claimed to already exist):

```sh
make r2-doctor
make r2-package
make r2-gate-selftest
make r2-acceptance
make r2-verify-evidence EVIDENCE=<run-directory>
```

Two independent Linux x86_64 environments A and B both build from the identical clean candidate and lockfiles, perform non-editable installation, and execute the entire R1+R2 mandatory matrix; B does not copy build/venv/installation artifacts from A. Verified read-only download caches are permitted; a runtime-only environment is separately used to prove PKG-03. macOS or local incremental testing cannot substitute.

Preserve R1 single-test, subprocess reclamation, native build, and bootstrap limits. R1 sub-acceptance remains subject to its original 7200-second limit; R2 newly added test portion has a 14400-second limit; combined acceptance (excluding bootstrap/build) has a 21600-second limit. Single simulator invocation limit remains 600 seconds, and long sequences must not be split into device re-initializations to mask this ceiling; segmenting timing by test case within a single persistent session is permitted, with all segments sharing session/epoch traces. Release retry=0; normal host wait deadlines and injected device deadlines are recorded separately.

Beyond R1 evidence, the following must be archived: packages/hashes, installation logs, isolation path verification, schema/ABI/golden artifacts, compilation invocations and cache traces, simulator build/start counts, all job/event state transitions, PID→tile/batch mappings, resource reservation/releases, transport memory accesses, generation/recovery traces, independent oracle inputs/outputs for each E2E test, mutation results, and complete logs.

| ID | Acceptance |
|---|---|
| REL-01 | Full suite passes on both A and B; code/constant/normalized manifest and deterministic results identical; missing dependencies in any environment is never SKIP |
| REL-02 | Installation package runs outside checkout; runtime-only golden artifact loads normally, unknown ABI versions rejected |
| REL-03 | Deleting a test case/output, tampering with a byte, mixing other commit/profile/epoch/session, missing logs, or falsifying counts fails evidence validator |
| REL-04 | required==discovered==executed (after expanding profile/seed/environment), all PASS; any uncaught mutation, orphan process, or missing terminal state yields non-zero gate |
| REL-05 | Documentation, API/schema, support matrix, examples, work package owners, and evidence paths complete; no TBDs/exceptions affecting mandatory delivery scope |

## 5. Refactoring Boundaries and Implementation Sequence

R2 does not accept based on "having moved directories". The following dependencies must genuinely hold:

```text
Public CLI/API → compiler driver → Triton adapter / MLIR passes / safe emission
Public CLI/API → runtime → artifact/ABI/admission → scheduler → transport → RTL
verification → invokes public interfaces above + independent oracles / fault injection
Product compiler/runtime do not depend on verification/test modules
```

Reusable product logic from the legacy test harness is migrated to official modules, with tests calling them inversely. The legacy Python IR/planner is preserved as an explicit second input / reference adapter; public runtime entry points must not call test helpers to assemble code or assertions.

| Batch | Work Packages | Dependable Results Upon Completion |
|---|---|---|
| 1 | R2-01, schema contracts of R2-05/06 | R1 transition, profile fingerprints, artifact/ABI no longer guessed independently across modules |
| 2 | R2-02, R2-03, R2-04 | Post-installation real compilation entry point and narrow MLIR main path |
| 3 | R2-05, R2-06, R2-07, R2-08 | Immutable code + dynamic parameters, persistent session, buffers, and events |
| 4 | R2-09, R2-10, R2-11 | Real grids, global resource admission, memory visibility, draining, and recovery |
| 5 | R2-12, R2-13, R2-14 | Usable reports/examples, differential and mutation testing, dual-environment final sign-off |

Prototyping interfaces across batches in advance is permitted, but stubbed/mocked success results cannot be used to accept cases depending on real RTL. Units/mocks serve to isolate issues, while E2E, memory, and recovery criteria must be grounded in actual implementation.

## 6. Final Judgment and Sign-off Table

```text
Valid R1 baseline evidence exists
AND Full R1 suite re-passes on the same R2 candidate commit
AND Installed package completes real compilation and execution outside checkout
AND Dynamic ABI, manifest, cache, and loader pass all positive and negative cases
AND MLIR main path genuinely participates in machine code generation for supported programs
AND Same artifact/session executes repeatedly across parameters, grids, and program batches
AND Formal transport guarantees host result visibility without stale cache corrupting new inputs
AND All tasks reach exactly one terminal state, reclaimed only post-drain, late responses never cross epochs
AND Known faults drainable; undrainable faults poison/recover; subsequent tasks correct
AND Complete R1+R2 case set without unexpected fail/skip/xfail/xpass/timeout/retry
AND All mutations caught; dual-environment artifacts and evidence verification pass
AND R2-01 ... R2-14 all PASS
=> R2 ACCEPTED
OTHERWISE => R2 NOT ACCEPTED
```

| Work Package | Initial Status | Owner | Final Evidence |
|---|---|---|---|
| R2-01 R1 Transition / Contract | NOT VERIFIED | To be filled at implementation | Not delivered |
| R2-02 Installation / API | NOT VERIFIED | To be filled at implementation | Not delivered |
| R2-03 Compiler Driver | NOT VERIFIED | To be filled at implementation | Not delivered |
| R2-04 MLIR Main Path | NOT VERIFIED | To be filled at implementation | Not delivered |
| R2-05 Artifact / Loader | NOT VERIFIED | To be filled at implementation | Not delivered |
| R2-06 Dynamic ABI | NOT VERIFIED | To be filled at implementation | Not delivered |
| R2-07 Cache / Determinism | NOT VERIFIED | To be filled at implementation | Not delivered |
| R2-08 Runtime Lifecycle | NOT VERIFIED | To be filled at implementation | Not delivered |
| R2-09 Grid / Admission | NOT VERIFIED | To be filled at implementation | Not delivered |
| R2-10 Host Transport / Completion | NOT VERIFIED | To be filled at implementation | Not delivered |
| R2-11 Faults / Recovery | NOT VERIFIED | To be filled at implementation | Not delivered |
| R2-12 Reports / Diagnostics / Examples | NOT VERIFIED | To be filled at implementation | Not delivered |
| R2-13 E2E / Mutation | NOT VERIFIED | To be filled at implementation | Not delivered |
| R2-14 Dual Environment / Release | NOT VERIFIED | To be filled at implementation | Not delivered |

**Drafting this contract does not alter the acceptance status of R1 or R2. If any item is missing, R2 must not be declared complete under claims that "the main flow runs", "only engineering polish remains", or "will be addressed in the next phase".**
