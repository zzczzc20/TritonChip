# Phase 3.6: Multi-Backend Engineering Integration and Prototype Branch Handover Preparation

Version: 1.0, 2026-09-20. Status: **PLANNED / NOT ACCEPTED**.
Prerequisite: **Assumes R3 v2 and R3.5 are accepted upon entering this phase.**
Associated: [R3 v2](phase3_tasks_and_acceptance.md), [R3.5](phase3_5_tasks_and_acceptance.md), [Architecture Design](design/tnca_v1_architecture_draft.md), [Compiler Design](design/tnca_v1_compiler_draft.md), [R2 Interfaces](r2_interfaces.md).

**This phase is responsible for mainline engineering preparation, not board bring-up.** Board selection, vendor IP, pin/board clock constraints, placement and routing, bitstreams, programming, and physical board debugging are executed by others in independent branches and are not included in R3.6 acceptance.

The objective is to unify previously fragmented compiler, runtime, RTL, verification, and build artifacts into an installable, exportable, reproducible, and extensible engineering codebase, enabling subsequent FPGA/ASIC adaptations to proceed along well-defined interfaces without duplicating the entire project or rewriting common semantics.

Final sign-off milestone identifier: `R36_ENGINEERING_READY`. It does not indicate that an FPGA is running, an ASIC has been implemented, or the project has completed production sign-off.

## 1. Mandatory Deliverables

1. A traceable integration baseline: R3.5 optimizing compiler → artifact → runtime → accepted RTL → independent result checks, fully reproducible from a clean environment.
2. Explicit separation among compilation targets, execution backends, hardware implementation technologies, and external platforms, decoupling current simulation tooling from generic product logic.
3. A complete portable hardware delivery package: common top/ports, timing and reset contracts, source file dependencies, generated artifacts, configuration, and memory adaptation boundaries; having operational semantics without depending on a testbench.
4. A set of device-agnostic runtime/protocol interfaces, alongside transport/platform adapters implementable by external branches.
5. Shared capability descriptions, compatibility checks, conformance suite, and engineering gates, preventing different backends from reinterpreting ISA/ABI/completion semantics independently.
6. Handover documentation, executable integration examples, known limitations, and upstreaming rules for FPGA and ASIC branches.

"Refactoring" must tangibly alter engineering usage: external developers must be able to attach a reference adapter in a clean environment and execute small programs through public product entry points; directory moves, architectural diagrams, or empty interfaces alone do not count as completion.

## 2. Scope and Responsibilities

| Mandatory for R3.6 Mainline | FPGA Prototype Branch Responsibilities | Subsequent ASIC Branch Responsibilities |
|---|---|---|
| Common ISA, ABI, target/profile, compiler, and runtime | Concrete device/board and toolchain selection | Concrete process technology, standard cell libraries, and physical tools |
| Hardware export, public interfaces, synthesizable boundary checks | Actual integration of vendor memory/clock/communication IP | Process adaptation for SRAM macros, clock/reset/test, etc. |
| Adaptation contracts for storage, clock/reset, host/device, and external memory | Pins, voltages, board-level constraints, external memory controllers | Synthesis mapping, place and route, STA, DFT, physical sign-off |
| Runnable reference adapter and shared conformance suite | Bitstream generation, programming, physical board bring-up, and actual measurement | Real PPA and tapeout delivery |
| Configuration, error propagation, diagnostics, evidence, and handover rules | Implementation of board drivers / connection media | Packaging, post-silicon bring-up, and volume production |

Does not specify FPGA models, vendors, host connection protocols, bus standards, or target operating frequencies. Nor does it require acquiring physical boards, vendor licenses, or PDKs to pass this phase.

This phase does not serve to complete all remaining R2 tasks. "Capabilities actually supported in this integration" and "undelivered grid/system runtime/other features" must be listed separately; the phase handover baseline must not depend on any unimplemented parts.

## 3. Multi-Backend Engineering Model

### 3.1 Four Concept Separations

| Concept | Describes | Prohibited Responsibilities |
|---|---|---|
| Compilation target/profile | ISA/ABI, capabilities, resources, layout, and compilation cost contracts | Cannot be equated to a specific simulator executable or a specific board |
| Execution backend | Device lifecycle, request/completion, memory access, and fault handling | Cannot duplicate compiler, buffer/event state machines, or silently mutate kernels |
| Hardware implementation | RTL implementation, memory ports/latency, resource configurations, and technology adaptation | Cannot alter capabilities or visible semantics while retaining legacy identity |
| Platform binding | Binding for simulator, board-level shell, process/IP, and concrete build environments | Cannot leak into generic ISA/ABI/algorithmic lowering |

Allows multiple future simulators, FPGA platforms, or ASIC implementations; this phase does not require implementing all of them. An ASIC build target is also not equivalent to a physical device that can be opened by runtime today.

### 3.2 Backend Delivery Depth in This Phase

| Track | Acceptance Scope in This Phase | Permitted Public Claim |
|---|---|---|
| Existing simulation execution | Product entry points, lifecycle, numerical correctness, faults, installation, and regressions fully operational | `operational`, with testing scope attached |
| External adapter integration | Reference adapter distinct from existing bindings, validated via actual RTL integration or protocol closed-loop, proving the replacement boundary holds | `integration-tested`; must not falsely claim physical board execution |
| FPGA | Shared hardware package, implementation interfaces, build extension points, integration tests, and handover instructions | `engineering-ready`; specific platform unverified |
| ASIC | Same shared hardware package, technology-dependent unit boundaries, storage/clock/reset contracts, and structural checks | `engineering-ready`; not mapped to real process |
| Optional functional model | If delivered, explicitly state its scope of numerical/protocol/performance applicability | Cannot substitute a functional model for RTL or physical performance |

Backend registration must distinguish between "interface defined", "reference implementation available", and "platform verified". Uninstalled or unimplemented backends must be explicitly rejected, rather than silently falling back to simulation and returning success.

## 4. Interfaces Requiring Genuine Integration

### 4.1 Compiler, Artifact, and Configuration

Retains R3.5's optimization path and safety checks. Platform selection enters compilation and loading via versioned target/implementation descriptions; simulator/board conditional logic must not be scattered throughout kernel sources or generic passes.

Distinguishes at minimum: semantic compatibility identity, implementation and resource identity, runtime asset identity, and platform build identity. Specific naming and field designs are determined during implementation, but semantics, provenance, hash coverage, and compatibility rules must be unambiguous.

- Different transport bindings sharing the same semantics and implementation must be able to use the same artifact; recompilation must not be required solely due to connection media changes.
- Changes in resources, layout, implementation latency assumptions, or instruction capabilities must affect the corresponding identity and cache decisions; stale artifacts must be explicitly rejected when recompilation is required.
- Does not promise that arbitrary FPGA/ASIC targets can run the same binary. Source-level re-compilability and binary compatibility must be recorded separately.
- If ABI/manifest must evolve, it must be versioned, providing an accept/reject matrix for legacy artifacts. Fields determining safety or compatibility must not be placed in ignorable metadata.
- Target definitions, runtime device info, RTL elaboration, export packages, and compilation reports must be cross-checked; multiple independent manually authored "ground truth configurations" are prohibited.

### 4.2 Runtime and Transport

The generic session/buffer/event state machine no longer directly bears the responsibility of launching specific simulators, exchanging temporary files, or operating platform resources. The existing simulation path is retained as one implementation; external adapters follow the same lifecycle and completion contracts.

Interfaces cover at minimum: discovery / capability query, connect / close, allocate visible range, upload / download, load / launch, wait / status, quiesce, error, and explicit recovery. Specific methods, connection mechanisms, and data encodings are not mandated in this document.

Must clearly define: request IDs and epochs, size/alignment/permission checks, concurrency capabilities, timeouts, whether cancellation is supported, output validity following errors, retry semantics, and post-recovery behavior of legacy handles. In particular, launch must never be automatically retried upon timeout, which could cause duplicate execution.

The reference adapter may execute within a simulation environment, but must traverse the same public interface used by external branches; merely using a mock to return correct values does not prove successful execution integration.

### 4.3 Portable Hardware Delivery Boundary

The exported shared hardware must possess an explicit top module, ports, and dependency closure; it must include control datapaths required for actual execution, rather than being an illustrative package containing only arithmetic cores.

Item-by-item definitions for timing, valid/ready conditions, backpressure capacity, and maximum inflight limits of instruction delivery, host control, data request/response, completion/error, clock/reset, and status/performance observation interfaces. If an interface adopts fixed latency, it must be frozen and tested rather than relying on fortuitous single-cycle simulation returns.

Hardware operational semantics must not depend on `$readmemh` dynamic loading, file polling, testbench hierarchical array writes, or DPI callbacks; while these may exist in simulation adapters, they must not serve as the sole control path for shared hardware. Assertions and diagnostics must be separated from synthesizable functional paths; necessary behaviors must not be stripped out by synthesis conditional compilation.

Backend shells adapt only platform interfaces without duplicating core, MMA, NUCA, or other common implementations. If future platforms genuinely require distinct microarchitectures, they must become explicit implementations rather than unlinked copies divorced from regressions.

### 4.4 Storage, Clock, and Reset Adaptation

Account item-by-item for VRF, ARF, instruction memory, cache data/metadata, and other physical memories: capacity, granularity, ports, synchronous/asynchronous read, latency, byte masks, read-during-write (RDW), initialization, clearing, and error behavior.

- Cannot claim "just swap in a RAM macro" without declaring current access semantics and implementation constraints.
- Validate replacement boundaries with at least one independent reference memory/interface implementation, covering legal latency and backpressure; replacements within the supported scope must not require altering ISA or operator sources.
- Memories that currently remain behavioral implementations must be explicitly listed as limitations. R3.6 does not automatically undertake the complete synchronous ARF refactoring, nor may it mark ARF as completed or guarantee its mapping to FPGA RAM.
- If a limitation implies that external branches must alter common completion/ownership semantics to integrate, it cannot be passed off as routine board-level work; interface gaps must be resolved in mainline, or the delivery profile must be narrowed and clearly specified.
- Contracts must govern reset, uninitialized state, inflight requests, clock enable, and clock-domain crossing boundaries. Common logic must not assume external IP is immediately available post-reset.

Maintain the architectural principle of **RF — NUCA — External Memory**; do not introduce software-addressable SMEM for prototype adaptation. Simulation external memory models, future on-chip prototype backing memory, caches, and scratchpads are distinct concepts and must be clearly distinguished; software semantics must not change simply because storage physically resides on-chip in an FPGA.

### 4.5 Host Visibility and Error Completion

Successful completion must guarantee that results reach the declared host-visible scope; "core idle" cannot be equated directly to all writes being visible. If passing through caches, write queues, or external controllers, the relationships between draining, write acknowledgment, and subsequent reads must be individually defined and validated.

Uploading overwriting prior cache contents, downloading dirty data, sequential tasks sharing device memory, partial transfers, late responses, device loss, and reset recovery must all have well-defined semantics. Supporting only serial host/device access at present is acceptable, but must be actively enforced by the runtime; unimplemented cache coherence or system atomics must not be claimed.

## 5. Completion Standards for Integration, Regression, and Handover

### 5.1 A Single Product Path

Executed from a clean environment via installed packages:

`source → R3.5 compiler → artifact → backend selection → upload/launch/wait/download → independent oracle`

Includes at minimum real Triton elementwise, masked tail, TNCA IR small GEMM, two-role pipeline, intra-device task dependency chains, and a fault recovery case. Preserves actual `source_kind` without misrepresenting TNCA IR examples as generic Triton GEMM.

Reuses existing small workloads; newly added GEMM must not exceed 16×16×32, elementwise must not exceed 32 elements, and reduction must not exceed a single 16-element row. Does not add large models, full system parameter sweeps, or expand RTL benchmarks under the guise of "board preparation".

### 5.2 Reproducible Baseline

Sign-off of R3/R3.5 only proves their frozen commits and implementations. This phase selects one accepted baseline, records the blast radius after interface refactoring, and reruns the corresponding candidate regressions. One cannot merely cite historical ACCEPTED verdicts to conclude refactoring has not regressed.

On the same hardware implementation, pure engineering refactoring must not alter the machine code (disregarding declared non-semantic metadata) or device cycles of R3.5's frozen workloads; necessary semantic or implementation changes must be separately identified, re-establishing corresponding regression and performance evidence rather than being conflated with directory/interface refactoring as "zero behavioral change".

RTL simulation is slow; a unified run registry must be used to reuse multiple checks from the same run, with budgets accounted by actual subprocess counts. The newly added run inventory, iteration counts, cycle/wall limits, and tool build budgets must be frozen at phase initiation and must not be left empty; budgets must be established based on measurements of existing small tests without speculating about new platform speeds. Over-budget runs retain FAIL verdicts; overwriting them with subsequent successful reruns is strictly prohibited.

### 5.3 Branch Handover Rehearsal

Establish an independent downstream example from formal export and installation packages without reading unexported files from the primary repository. Attach a reference adapter along declared extension points, compile or load artifacts, execute small programs on real RTL, and pass the shared conformance suite.

The integration process must not modify the generic compiler, runtime state machines, or compute cores; adding external configurations/adapters or utilizing public extension interfaces is permitted. Interface gaps identified during rehearsal must be resolved in mainline, rather than bypassed by privately patching common code within downstream examples.

Deliverables to physical board branches must include: baseline commit, package/asset hashes, interface versions, verified profiles, known limitations, integration guides, reproduction commands, error diagnostics, and upstreaming rules. External branches must not be required to guess completion semantics or duplicate test oracles.

## 6. Work Packages and Strict Acceptance Criteria

Total of 10 work packages and 30 criteria. Owners are listed by role; actual assignees, commits, and evidence are filled during execution. Initially all are **NOT VERIFIED**.

### R36-01: Integration Baseline and Capability Inventory (Maintainer / Verification)

| ID | Acceptance Criteria |
|---|---|
| R36-BASE-01 | R3/R3.5 prerequisite evidence is valid, with traceable commits, implementations, and profiles; freeze the delivery profile, complete feature inventory, inherited regressions, and execution budgets for this phase, with no undefined required items |
| R36-BASE-02 | Every existing capability is explicitly categorized as verified, restricted, or unimplemented alongside corresponding evidence; R2 remaining work and behavioral ARF limitations are clearly demarcated, with no contradictions between historical and current claims |
| R36-BASE-03 | Complete the Section 5.1 chain using product entry points in a clean environment; do not import test helpers as product logic, and do not rely on local build caches or developer paths |

### R36-02: Backend Abstraction and Dependency Direction (Architecture / Runtime)

| ID | Acceptance Criteria |
|---|---|
| R36-BE-01 | Target, implementation, backend, and platform responsibilities are independent and queryable; generic compiler/runtime contains no simulator- or board-specific launch or connection logic |
| R36-BE-02 | Existing simulation backend and independent reference adapter both operate through public interfaces; the latter executes numerical programs via real RTL or closed-loop protocols connected to real RTL, rather than mocking correct answers |
| R36-BE-03 | Unknown, missing, version-incompatible, or unimplemented backends produce deterministic diagnostics; no automatic fallback to simulation, and no marking engineering-ready as platform-verified |

### R36-03: Configuration, Artifact, and Compatibility (Compiler / Runtime)

| ID | Acceptance Criteria |
|---|---|
| R36-CFG-01 | Compilation target, device query, RTL instance, export descriptions, and reports are field-by-field consistent; changes in capability, resource, or latency assumptions alter corresponding identities or explicitly trigger recompilation |
| R36-CFG-02 | Two transport bindings for the same implementation execute correctly using the same artifact; insufficient capabilities, mismatched ABI/profiles, or incorrect runtime assets are rejected prior to launch with zero task side effects |
| R36-CFG-03 | Configuration and schema evolution has a versioning/migration matrix; cache does not hit across incompatible implementations; safety-critical fields are not placed in ignorable metadata; legacy artifact handling is explicit and tested |

### R36-04: Common RTL and Complete Export (RTL / Build)

| ID | Acceptance Criteria |
|---|---|
| R36-RTL-01 | Independent elaboration and execution of small programs directly from export packages; source lists, includes, generated RTL, parameters, and tool versions are complete; no main repository relative path leakage or undeclared blackboxes |
| R36-RTL-02 | Instruction delivery, control, data, and status/error/counter interfaces on the public top all possess timing contracts; validated under actual backpressure/latency/completion tests without relying on testbench hierarchical reads/writes for operational functionality |
| R36-RTL-03 | Synthesizable functional paths contain no runtime file services or simulator-only behaviors; structural checks reveal no unexplained latches, combinational loops, multi-drivers, or width mismatches; simulation bindings and future platform shells share the identical core implementation |

### R36-05: Storage and Clock/Reset Adaptation Boundaries (RTL)

| ID | Acceptance Criteria |
|---|---|
| R36-MEM-01 | Itemized accounting of capacity, ports, latency, masks, RDW, initialization, and replaceable scope for each physical memory array; unmapped technology is explicitly recorded, without passing off memory-inferred arrays as verified physical RAM |
| R36-MEM-02 | At least one independent reference implementation passes real RTL regressions through the common memory/instruction adaptation boundary; covers contract-permitted latency, backpressure, partial writes, and bank conflicts; numerical results, ownership, and draining are correct |
| R36-RST-01 | Directed tests cover startup, inflight requests during reset, restart, and external not-ready scenarios; stale epoch data does not pollute subsequent tasks; clock-domain crossing responsibilities and constraint lists are complete without implicit board assumptions |

### R36-06: Host Protocol, Completion, and Recovery (Runtime / RTL)

| ID | Acceptance Criteria |
|---|---|
| R36-IO-01 | Explicit semantics for upload, download, launch, query, quiesce, close, and unsupported operations; checks for size, address ranges, alignment, ordering, and duplicate requests, with invalid inputs producing zero device task side effects |
| R36-IO-02 | Under write latency, dirty cache lines, and sequential tasks sharing memory, downloads after SUCCEEDED return correct results; disallowed host/device concurrency is actively rejected rather than relying on user discipline |
| R36-IO-03 | Directed tests for late/duplicate responses, partial transfers, timeouts, device disconnection, and resets; no duplicate launches, no resurrected legacy handles; recoverable errors and session poisoning are handled separately |

### R36-07: Build, Installation, and Portable Delivery (Engineering / Build)

| ID | Acceptance Criteria |
|---|---|
| R36-BUILD-01 | A single public build entry point explicitly selects software packages, simulation runtime assets, common hardware exports, and external platform extensions; missing optional FPGA/ASIC tooling does not block mainline packages or simulation builds, nor falsely report platform build success |
| R36-BUILD-02 | Compiles and executes in non-editable installs, arbitrary current working directory, and with repository omitted from search paths; runtime-only installations load artifacts without dependencies on Triton, MLIR, compiler, or test tooling |
| R36-BUILD-03 | Common code, configuration, and export manifests are comparable across two independent clean builds with normalization rules declared in advance; dependencies, generated files, and third-party sources/licenses are complete; export packages pass read-only integrity verification |

### R36-08: Common Conformance and Regression Gates (Verification / CI)

| ID | Acceptance Criteria |
|---|---|
| R36-TEST-01 | A unified backend conformance suite covers lifecycle, data transfer, compatibility, completion, and faults; truly supported features are executed, while unsupported features verify controlled rejection; each test maps to protocol logs and oracles |
| R36-TEST-02 | Frozen R1/R2/R3/R3.5 blast-radius regressions pass completely on this phase's candidate; Section 5.2 machine code and cycle fidelity hold; newly added small workloads and budgets remain within limits without unexplained skips |
| R36-TEST-03 | Directed mutations corrupting capabilities, asset hashes, return epochs, premature completion, memory latency, and export dependencies are each caught by corresponding checkers; clean controls pass, and mutations remain isolated without polluting source files |

### R36-09: Downstream Branch Handover and Extension Verification (Maintainer / Platform Integration)

| ID | Acceptance Criteria |
|---|---|
| R36-HAND-01 | Section 5.3 clean downstream integration rehearsal passes; executes on real RTL using solely public extension points and delivered assets, with zero private patching of common cores or state machines |
| R36-HAND-02 | FPGA and ASIC each have an adaptation responsibility matrix and limitation inventory; does not mandate specific devices or processes, nor disguises memory/interface semantic gaps as board-level tasks; interface owners and change procedures are explicit |
| R36-HAND-03 | Handover package provides minimal correctness programs, fault diagnosis, event/counter definitions, and reproduction commands; specifies rules for upstreaming common semantic fixes, maintaining platform bindings in extension layers, and bumping interface versions |

### R36-10: Evidence, Status, and Final Sign-off (Maintainer / Verification)

| ID | Acceptance Criteria |
|---|---|
| R36-EV-01 | Independent verifier re-derives verdicts from raw exit codes, outputs/counters, hashes, and budgets against the frozen required suite; rejects omissions, tampering, mixed commits/implementations, or summary-only PASS alterations |
| R36-EV-02 | All delivery packages bind to a single integration candidate, with traceable source and generated asset states; failures and reruns retain independent run IDs; all 30 criteria and sub-clauses have documented evidence and owners |
| R36-REL-01 | All 10 work packages pass; simulation execution and downstream rehearsals are reproducible; sign-off identifier is strictly `R36_ENGINEERING_READY`; FPGA/ASIC physical board and physical silicon verification remain explicitly external follow-up items |

## 7. Evidence and Stage Gates

Does not mandate specific repository directory reorganization layouts. Implementations must provide the following traceable deliverables, with paths determined by the engineering plan:

- Prerequisite and current integration baselines, capability/limitation matrices, dependencies, and blast radius.
- Versioned interfaces, configuration/compatibility rules, hardware source dependencies, and export manifests.
- Installation packages, runtime assets, common hardware packages, and their respective hashes.
- Reference adapter, common test entry points, independent oracles, raw execution logs, and budgets.
- Clean downstream integration rehearsal records, handover guides, and platform responsibility tables.
- Machine-derived verdicts and status records corresponding item-by-item to the IDs in this document.

| Gate | Prerequisite Conditions | Permitted Claim Upon Completion |
|---|---|---|
| A: Baseline and Contracts | R36-01, R36-02, R36-03 | Engineering boundaries and compatibility rules frozen |
| B: Interfaces Genuinely Replaceable | R36-04, R36-05, R36-06 | Common hardware and host interfaces validated via reference adapters |
| C: Installable, Regressible, and Accessible | R36-07, R36-08, R36-09 | Mainline deliverable to independent platform branches |
| D: Evidence and Status | R36-10 and all preceding gates | `R36_ENGINEERING_READY` |

Absence of physical boards, vendor tools, or PDKs does not constitute a failure for R3.6, as they fall outside this contract. However, unimplemented public interfaces, non-functional adaptation examples, package dependencies on developer environments, or ambiguous error completion semantics represent mainline gaps that must be resolved, and cannot be signed off by deferring them as "left for board branches".

## 8. Project State After Phase Completion

The project maintains a stable common product pipeline, with simulation serving as an operational execution backend; FPGA and ASIC extend via tested engineering boundaries, while unverified platforms maintain explicit status. Subsequent branches can focus on platform implementation and physical verification without needing to re-create another compiler, ABI, runtime, testing, and evidence framework.

**This phase delivers "handoff-ready, integration-ready, and reproducible" as its acceptance outcome. It does not generate deliverables intended for immediate board programming, nor does it require physical board bring-up or physical silicon PPA as completion criteria.**
