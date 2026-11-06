# Phase 3 (R3): Compute and Memory Implementation, Fast Performance Model, and Small-Scale RTL Calibration Tasks and Acceptance Contract

Version: 1.0, 2026-09-20. Status: **Specification established; implementation not yet accepted; does not signify completion of R1 or R2.**

Prerequisite contracts: [R1](phase1_tasks_and_acceptance.md), [R2](phase2_tasks_and_acceptance.md). Design basis: [Architecture Design](design/tnca_v1_architecture_draft.md) Sections 5, 6, 16, and 20, and R3 in the [Production Readiness Audit](production_readiness_audit_2026-09-19.md).

**RTL simulation is slow. This phase uses only small-size, limited-combination RTL microbenchmarks to verify correctness, actual cycles, and resource contention; parameter exploration is performed by the fast performance model. Expanding R3 into large GEMM, Attention, or full-model inference RTL benchmarks is prohibited.**

Historical audits prove that small GEMMs and multi-role pipelines can execute on RTL, but this does not imply fast simulation speed on current machines. At the start of the phase, cycles/second and wall time must be empirically measured for each benchmark category to control execution costs. **Simulation wall time, hardware cycle count, and physical clock frequency are three distinct metrics and must be reported separately in all reports.**

## 1. Phase Objectives and Definition of Completion

R3 must deliver:

1. VRF/ARF with explicit bank, port, and synchronous read latency contracts, implementing finite port arbitration, operand collection, writeback/return, and correct backpressure.
2. Verifiable cycle and traffic counters for DMA, MMA, SIMD, SFU, RF, barrier, and NUCA/NoC that explain where small programs spend time.
3. A finite-resource performance model independent of RTL for exploring limited candidate configurations, calibrated and validated against pre-frozen RTL microbenchmarks.
4. A strictly bounded suite of GEMM, reduction, softmax, RMSNorm, and LayerNorm workloads executing real compilation artifacts, preserving R1/R2 correctness and runtime contracts.
5. Synthesizable storage structures where data genuinely maps to synchronous memory rather than full-array-reset flip-flop arrays, accompanied by structural synthesis reports honestly bounding physical library claims.

Completion of this phase signifies that "resource-constrained implementations are usable, critical throughput and latency contracts are established, and models have small-scale RTL empirical evidence." **It does not signify that the original 1 GHz target is achieved, does not signify that 16×16 MMA instructions execute 256 MACs per cycle, and does not signify that large-scale GEMM performance has been validated on RTL.**

### 1.1 Scope Boundaries and Prerequisites

| Mandatory in R3 | Excluded from This Phase (No Scope Expansion) |
|---|---|
| VRF/ARF synchronous memory, bank arbitration, collector, and correct issue/writeback under variable latency | Expanding RF/ARF capacity, adding RCTX, 4×4 mesh |
| RF port budgeting, measurements, and scheduling feedback for DMA/MMA/SIMD | Enlarging MAC arrays for benchmark chasing, dual-issue, general out-of-order execution |
| Small numerical workloads and two-role overlap calibration | Long-sequence Attention, full-model inference, GEMM 128×128 and larger |
| Model exploration, comparison between actual implementation and selected configuration | Generating RTL for every model candidate, full parameter sweeps for every kernel |
| Memory-preserving synthesis and structural checks | PDK / packaging / power integrity / DFT / tapeout signoff |
| Regression of R2 product entry points and event/drain semantics | New host driver, PCIe/FPGA board-level product |

Performance modeling and storage interface design may proceed in advance; final R3 acceptance must completely pass R1/R2 on the identical candidate commit. Existing R1/R2 test cases (such as K128, grid=17, etc.) continue to run under their original contracts; **the small-size upper bounds in this document constrain only newly added RTL workloads in R3, without deleting legacy regressions or adding new exhaustive combinations to legacy regressions.**

R2 profile names and architectural resource quantities are retained; changes to synchronous memory, ports, or latencies must update implementation fingerprints, compilation cost descriptions, and runtime assets. Compatibility of legacy artifacts is handled according to R2 exact fingerprint rules, and hardware changes must not be concealed.

## 2. Freeze Simulation Budget Before Writing Benchmarks

### 2.1 Three-Layer Verification Responsibilities

| Level | Question Answered | Mandatory Prevention |
|---|---|---|
| Standalone numerical/protocol models, component property verification | Are algorithms, floating-point, and ownership correct; is conservation maintained under port conflicts? | Using own outputs as oracle; covering only happy paths |
| Fast finite-resource performance model | Which candidate configurations are worth implementing; how do cycles, traffic, and bottlenecks change? | Calculating only roofline; backfilling predictions after reading total RTL cycles |
| Small-scale RTL microbenchmarks | How many cycles do instructions actually take; are resource contention, overlap, and model errors real? | Large matrix sweeps; simulating for hours on a sample that provides no new statistical information |

Component-level testing takes precedence; only properties that must traverse real cores/NUCA/mesh are run on the full system. Memory bank conflicts are triggered using dozens to hundreds of requests, rather than running large matrices to "wait for them to naturally appear". NUCA replacement pressure is induced using selected addresses within the same set, rather than streaming across large capacities.

### 2.2 Hard Upper Bounds for Newly Added RTL in R3

| Item | Permitted Range / Upper Bound |
|---|---|
| Single-core GEMM | M ≤ 16, N ≤ 16, K ≤ 64; tail block fixed to 13×11×19 |
| Full-system GEMM | Per program M ≤ 16, N ≤ 16, K ≤ 32; grid ≤ 4, at most 2×2 tiles |
| reduction/softmax/Norm | 1 row, at most 16 active elements; masked case 15 elements; no batch scaling |
| Newly added pure elementwise | N ≤ 64, grid ≤ 4 |
| GEMM performance measurement repetitions | 1 kernel run per simulation; no in-RTL looping of 100/1000 repetitions for averaging |
| Steady-state port microtests | ≤ 256 accepted requests; initialization/drain counted separately, cannot wrap large workloads |
| Matrix unit microtests | At most 2 consecutive MMA commands, single command K ≤ 32; K64 measured in designated core workload |
| Component randomized seeds | {1, 7, 42}; limited to designated components and fault injection, not multiplied across all workloads |
| Performance workload seed | Fixed 42, deterministic inputs; performance measurements do not require large numbers of random seeds |
| Memory scenarios | Specified item-by-item in the workload inventory; do not multiply by fast/slow/ooo by default |
| Default waveforms | Disabled; failure debugging captures bounded windows only, full-run VCD/FST is prohibited |

Newly added R3 performance inventory is fixed to **12 core + 4 system workloads** in Section 3. Each workload executes only two implementations, `reference` and `candidate`: **32 performance RTL executions per acceptance environment**.

In addition, newly added correctness/backpressure component RTL executions ≤ 48, and newly added unit throughput/counter calibration RTL executions ≤ 24. Thus, total newly added RTL runs per environment ≤ 104. Mutations are detected via controlled unit/host tests; if RTL runs are needed, they must consume the above component quota without replicating 16 workloads. The baseline is the frozen legacy implementation within the same candidate tree, and test results must not be borrowed from a different commit.

### 2.3 Time and Cycle Upper Bounds

| Run Category | Single-Run Wall-Clock Limit | Single-Run RTL Cycle Limit |
|---|---:|---:|
| Newly added component correctness/throughput | 60 s | 10,000 |
| Newly added core workload | 180 s | 50,000 |
| Newly added system workload | 300 s | 50,000 |
| Total cumulative newly added RTL execution | **3,600 s / environment** | **2,000,000 cycles / environment** |
| Fast model default sweep | **60 s / environment** | Does not launch RTL |

Cumulative wall-clock time refers to the sum of actual elapsed time of each subprocess; budget cannot be concealed through parallelism. Repeated requests, processes, or batches cannot bypass cycle and workload accounting. Both baseline and candidate are counted toward the budget. Each simulation process must also inherit R1 process group cleanup and outer watchdog timers.

The reference environment follows the locked R1/R2 Linux x86_64 runner, recording CPU, memory, tool versions, and concurrency. The initial timing scout runs only the reference implementation of P01, P04, S01, and S03, totaling 4 runs; all are counted toward the formal 32-run inventory as current results for those items, and must not be run as extra duplicate benchmarks. If budgets cannot be met, the phase remains uncompleted: resolve simulator build/execution efficiency, benchmark initialization, logging overhead, or wasteful implementation loops first; silently relaxing time limits, dropping required properties, or switching seeds after timeout to turn green is strictly forbidden.

Evaluating compiled simulators like Verilator as an implementation option is permitted, but unmeasured speedup multipliers must not be promised. Before switching, numerical and event consistency checks against selected existing small programs and components must be completed, preserving dedicated checks for reset, X, and uninitialized states; two-state simulation must not mask initialization errors. Passed R1/R2 standalone verification configurations cannot be replaced without evidence.

Existing R1/R2 execution budgets remain unchanged and are each executed once during final signoff; they are not doubled due to having two RF implementations. Both R3 acceptance environments must independently satisfy the budgets in this section; results from the two environments must not be spliced together into a single complete matrix.

## 3. Sole Default Workload Inventory

All workloads are small programs with no implicit batching, hidden multi-round repetitions, or Cartesian shape products. `cold` denotes normalized reset/memory initial state; software counters must not merely be reset while leaving unknown cache states behind. Compilation, loading, and host upload times are reported separately from device cycles.

### 3.1 Core: 12 Workloads

| ID | Workload | Fixed Input & Configuration | Purpose |
|---|---|---|---|
| P01 | GEMM FP16 | 16×16×16, single-role, fast, cold | Minimum baseline and timing scout |
| P02 | GEMM BF16 | 16×16×16, single-role, fast, cold | dtype and numerical path |
| P03 | GEMM FP16 | 16×16×32, single-role, fast, cold | K-scaling prediction, holdout validation |
| P04 | GEMM FP16 | 16×16×64, two-role, 2-slot, fixed slow/reorder scenario, cold | ready/free overlap and bank contention |
| P05 | GEMM FP16 tail | 13×11×19, single-role, fast, cold | Tail blocks and padding overhead, holdout validation |
| P06 | GEMM FP16 strided | 16×16×16, row stride=64 B for A and B, single-role, fast, cold | Layout and transfer, holdout validation |
| P07 | GEMM FP16 serial | Identical shape, memory/seed/input to P04; explicit serialized load→MMA, identical 2-slot resource budget | Paired ablation for P04; must not compare against a smaller task |
| P08 | FP32 reduction sum | 1×16, single-role, fast | Reduction throughput and addition order |
| P09 | Masked FP32 reduction | 16 elements stored, only first 15 valid, single-role, fast | Lane mask and zero side effects |
| P10 | FP32 stable softmax | 1×16, core-SFU enabled, fast | max/exp/reduce/normalize dependency chain |
| P11 | FP32 RMSNorm | 1×16, epsilon=1e-5, gamma=1, core-SFU enabled, fast | rsqrt/reduction, holdout validation |
| P12 | FP32 LayerNorm | 1×16, epsilon=1e-5, gamma=1, beta=0, core-SFU enabled, fast | Mean/variance and intermediate value lifetimes |

slow/reorder parameters for P04/P07 follow verified R1 memory model parameters and fix latency, stalls, and seeds in the manifest; overlap must not be compared using two different random processes. K64 in P04 can be executed as four K16 steps, maintaining ISA accumulation contracts.

P10–P12 can be compiled from formal TNCA IR input adapters without requiring newly added general Triton softmax/Norm lowering in this phase; they must be executed by R2 production compiler/loader/runtime, reporting real input paths. Calling Python to compute outputs and write back into buffers is forbidden.

### 3.2 System: 4 Workloads

| ID | Workload | Fixed Configuration | Purpose |
|---|---|---|---|
| S01 | Dynamic masked AXPY | N=17, BLOCK=16, grid=2, FP32, SFU disabled, cold | Small real Triton→system chain, tail tile |
| S02 | GEMM FP16 | Per-program 16×16×16, grid=1, single-role, cold | Single-program delta from core to NUCA/NoC |
| S03 | GEMM FP16 | Per-program 16×16×16, grid=4, single-role, independent outputs, cold | Four-tile finite contention; 4 small tiles total |
| S04 | GEMM FP16 | 16×16×32, grid=1, two-role, 2-slot, cold | System pipeline, return path, holdout validation |

The system profile remains `system-2x2-nosfu`; SFU must not be secretly enabled for Norm samples. The four programs in S03 sharing read-only inputs can be used as a fixed declared mapping, but must be declared prior to measurement; reference and candidate must use identical addresses and mappings. Other sharing and placement candidates are explored only in the model, without adding new full-system matrices.

### 3.3 Numerical and Memory Acceptance Criteria

- GEMM FP16/BF16 follows the original ISA increasing-k fused FP32 accumulation contract, compared against an independent exact oracle according to its NaN/zero rules; addition must not be reordered to switch banks or alter pipelines.
- P08/P09 are compared exactly according to existing reduction order; masked elements do not participate in computation and must not trigger out-of-bounds accesses.
- Bounded inputs for P10 are restricted to [-8, 8], including identical values, mixed positive/negative, and biased high/low values; P11/P12 are [-4, 4], including constant rows, near-constant rows, and alternating signs. Performance cases each fix one set of inputs; remaining numerical combinations are verified in standalone software/component tests, without expanding into extra full-system cases.
- P10 against high-precision mathematical oracle: per-element `abs_error ≤ 2e-5 + 2e-4*abs(ref)`, output non-negative and finite, `abs(sum(y)-1) ≤ 1e-4`.
- P11/P12 against independent oracle of at least 50 decimal digits of precision: per-element `abs_error ≤ 3e-5 + 3e-4*abs(ref)`, output finite; LayerNorm uses centered variance, and removing valid near-constant inputs to satisfy tests is not permitted.
- These error bounds are acceptance requirements for R3 newly added composite operators, and do not relax SFU instruction-level R1 error bounds. Supporting composite operators outside these input domains is not claimed in this phase; original SFU NaN/Inf/subnormal tests are retained.
- All RTL workloads inspect complete outputs, sentinels/canaries, expected fault=0, and resource draining; improved performance with even one incorrect value constitutes failure.

## 4. Storage and Performance Contracts

### 4.1 Relationship Between the Two Implementations

`reference`: The frozen R2 behavioral RF/ARF implementation within the same candidate tree, maintaining its original asynchronous read and port semantics, used for explaining numerical and cycle discrepancies.

`candidate`: The sole synchronous, finite bank/port implementation selected for R3; must pass the real top instance. Both implementations share identical capacity, ISA, operators, inputs, and software algorithms. Compilers may schedule according to their declared latency/port contracts, but reports must retain code, resources, and instruction counts for both sides; speedup must not be gained by dropping computation.

The legacy RF corresponds to an idealized multi-port implementation; candidate is not guaranteed to take fewer cycles. **The goal is not "GEMM speedup at all costs", but rather verified availability under realistic storage ports, fulfillment of throughput contracts, and itemized attribution of any degradations.** Cycle increases relative to baseline must be attributed and not concealed.

### 4.2 Mandatory Frozen Hardware Resource Parameters

- VRF capacity remains 128×16×32 bits = 8 KiB; ARF remains 16×16×16×32 bits = 16 KiB; replicating entire data arrays to fabricate free read ports is prohibited.
- candidate VRF selects either 4 or 8 banks; each bank accepts at most 1 read and 1 write per cycle, with synchronous read latency chosen as 1 or 2 cycles. Bank mapping, actual word width, and depth per bank are documented in the profile, with capacity summing exactly.
- ARF may be banked by row/lane, but real per-bank ports and return latency must be specified; port contention between MMA and vector row operations must not disappear into combinational logic.
- Byte/lane masks, same-address read/write, same-cycle DMA and SIMD writes, zero vs. normal write priority, invalid data reads, and payload stability under backpressure must be defined item-by-item.
- If memory primitives do not support specific write mask or RDW (read-during-write) semantics, explicit subbanks, bypasses, or read-modify-write (RMW) must be implemented, with all extra cycles, registers, and traffic fully accounted for. Adding ports non-existent in the implementation library to simulation RAMs is prohibited.
- FIFOs, collectors, and operand holding registers are finite resources and not arbitrarily software-addressable; capacity, full/empty states, and return reservations enter profiles and models.
- Reset must be implementable as clearing valid metadata or bounded scrubbing; architectural visible semantics of uninitialized data must preserve R1/R2 contracts. Deleting data reset and treating random or X values as permitted outputs is prohibited.

### 4.3 Computable Throughput and Progress Contracts

Prior to implementation, submit `unit_contracts`: request granularity, bank/port upper bounds, latency $L$, maximum acceptance rate $q$, pipeline/queue depth, setup cycles $S$, drain cycles $D$, number of arbitrating contenders $C$, and maximum grant burst duration $B$ for each unit. Parameters must be derived from hardware structures or existing unit measurements, and cannot be retrofitted into wider thresholds after test failures.

In targeted microtests with no external backpressure and no data dependencies, the completion upper bound for $N \le 256$ requests is defined by its explicit scheduling model as $S + \text{schedule}(N, \text{ports}, \text{banks}, L) + D$. Simple fixed-rate ports use $S + \lceil N/q \rceil + L + D$; failing to achieve the declared $q$ constitutes acceptance failure, and reporting merely "low utilization" is prohibited.

Arbitration employs a fixed verifiable fairness policy: if each grant non-preemptively holds at most $B$ cycles and downstream remains ready, any of $C$ continuously requesting contenders waits at most $(C - 1) \times B + \text{arbitration pipeline latency}$. Resources required for completion/return must be pre-reserved prior to issuance or have independent forward-progress paths; forward progress cannot rely on perpetually prioritizing new DMA requests.

For MMA, M16×N16 is the instruction shape; unit_contract must specify actual MAC lane count, reads/computations/waits per k/m step, ARF round-trips, and FMA dependencies; using 256 MAC/cycle directly as model peak is prohibited.

### 4.4 Fixed Measurement Metrics

Each workload concurrently reports:

- Host compilation/build/simulation wall time; RTL total cycles and ROI cycles; neither substitutes for the other.
- Timestamps for launch accepted, first valid instruction, first data request, first MMA issue, final result store completed, and full task drain completed.
- Useful MAC/FLOP/bytes and actual emitted padding operations; GEMM counts 1 MAC as 2 FLOPs, but padding is excluded from useful work.
- Unit busy/ready/stall, RF per-bank accepted reads/writes and conflicts, MMA operand starvation, ARF hazards, barrier/window waits, MSHR entry/target pressure, link/VN actual transfers, and DRAM bytes.

Stall causes may occur concurrently, but must be reported as both an overlapping bitset and a fixed-priority mutually exclusive primary cause count; summing overlapping stalls to exaggerate total time is prohibited. Short kernels include fixed launch overhead; steady-state throughput cannot be claimed by cherry-picking the single best cycle.

## 5. Work Packages and Mandatory Acceptance Test Cases

Test case IDs represent acceptance semantics and do not imply launching a new full-system simulation for each ID; sharing a single executed trace to verify multiple properties is permitted, provided the case→run mapping is recorded. Mandatory subconditions must not disappear as a result.

### R3-01: Freeze Baseline, Budget, and Performance Contracts

Responsibility: Microarchitecture / Verification.

Deliverables: Profiles for both implementations, 16 workload manifests, unit_contracts, component/microtest inventory, execution budgets and assignees; reference source/hash and all measurement definitions.

| ID | Mandatory Acceptance Requirement |
|---|---|
| PLAN-01 | R1/R2 prerequisite evidence valid, candidate regression inventory complete; no avoidance of errors by expanding resources in the new profile |
| PLAN-02 | Workloads automatically validate M/N/K, grid, row count, repetitions, seed, memory mode; reject out-of-bound configurations prior to launching RTL |
| PLAN-03 | Timing scout actually executed, recording cycles/second and wall time; expanded inventory totals exactly 32 performance runs; component/microtest and overall budgets not exceeded |
| PLAN-04 | unit_contracts complete; bank/port/L/q/S/D/queue sizes have explicit origins; unknown is not 0; all acceptance formulas frozen prior to candidate measurement |

### R3-02: Standalone Finite-Resource Performance Model

Responsibility: Performance Modeling / Compiler.

Deliverables: Discrete-event or equivalent cycle-accurate model explicitly modeling dependency DAGs, issue limits, RF bank/port constraints, ARF updates, DMA requests/returns, MMA/SFU/reduction occupancy, barriers/channels, finite queues, NUCA entries/targets, NoC/DRAM paths. Inputs are compiled schedules/operations and profiles, not reverse-reading RTL stall traces of the workload to "predict".

| ID | Mandatory Acceptance Requirement |
|---|---|
| MODEL-01 | Six small DAGs (pure serial chain, fully independent operations, same-bank conflicts, cross-bank concurrency, queue full, ready/free reuse) match hand-calculated cycles and resource counts exactly |
| MODEL-02 | Component models with reduced ports or increased latency modify results according to contract; no zero-cost layout transformations, infinite queues, or overlap of compute before dependent data arrives |
| MODEL-03 | Automated verification of resource conservation, cycle-free event progression, and outstanding leases; invalid configurations / deadlocks return controlled errors rather than predicting zero cycles |
| MODEL-04 | Deterministic reproduction on identical inputs; default sweep of ≤ 128 candidate-workload combinations completes within 60 s without launching RTL or large-matrix numerical calculations |
| MODEL-05 | Outputs predicted cycles, bottlenecks, traffic, rejected configuration reasons, and model validity scope; large-shape results marked strictly as model projection, not RTL measured |

May explore limited candidates such as 4/8 banks, 1/2-cycle read, 1/2 stages, 4/8 credits; total resources must not exceed profile budgets. Model candidates do not equal supported RTL profiles. Exactly one candidate is selected for implementation; generating 128 RTL configurations is prohibited.

### R3-03: Synchronous VRF/ARF and Storage Wrapper Layer

Responsibility: RTL.

Deliverables: Synchronous read request/response interfaces, finite-bank memory wrappers, mask/RDW semantics, valid/reset mechanisms, synthesizable implementation; differential validation against legacy behavioral model.

| ID | Mandatory Acceptance Requirement |
|---|---|
| RF-01 | Sum of bank capacities exactly matches 8 KiB VRF / 16 KiB ARF; address mapping is reversible; boundary, cross-bank, and role physical aliasing are correct |
| RF-02 | Selected synchronous read latency of 1/2 cycles, masks, same-address RDW, concurrent dual-client writes in the same cycle, zero vs. write contention verified per request against standalone model |
| RF-03 | Invalid contents do not leak after reset or relaunch; architectural semantics of zero-clearing / uninitialized reads do not regress, and are not masked by 2-state zero initialization |
| RF-04 | Accepted reads/writes per cycle do not exceed physical ports; every pending response has a reserved destination, with no loss, duplication, or silent overwrite |
| RF-05 | Memory configurations unsupported by write masks explicitly rejected or routed via accounted RMW; extra infinite ports must not be used in simulation wrappers |

Random sequences for RF-02/04 are ≤ 256 requests per seed across three seeds; identical address hotspots generated via directed inputs. Illegal concurrent writes to the same logical data prohibited by compiler contract must be explicitly asserted/rejected, rather than letting tests pick an accidental winner.

### R3-04: Operand Collector, Arbitration, and Return Liveness

Responsibility: RTL / Protocol Verification.

Deliverables: Finite operand collector, RF request arbitration, writeback/return reservations, fairness policies, backpressure, and corresponding core pipeline hazard updates.

| ID | Mandatory Acceptance Requirement |
|---|---|
| FLOW-01 | Collection order and data correctness verified for three reads from the same bank and cross-bank reads; instructions do not issue until operands are collected, replays do not re-execute |
| FLOW-02 | SIMD, MMA, and DMA concurrently contend for RF; each grant conforms to frozen fairness upper bound; counters reflect true waiting rather than dropped requests |
| FLOW-03 | When RF is busy, queues are full, or writeback conflicts on return: issued transactions complete without requiring re-allocation of non-existent resources |
| FLOW-04 | valid/payload remain stable under randomized stalls; request/response/tag conservation maintained; occupancy and leases return to zero after draining |
| FLOW-05 | Unissued operands and issued transactions handled correctly under branch squash, faults, and epoch changes; R2 legacy response isolation remains effective |

Bounded progress is verified only in environments where downstream is declared eventually ready and memory eventually responds; simultaneously test R2-mandated never-return → deadline/poison paths, and do not frame environment assumptions as if all errors are impossible.

### R3-05: Real Throughput of DMA/MMA/SIMD and ARF

Responsibility: Microarchitecture / RTL.

Deliverables: New unit timing adapted to synchronous RF/ARF; measurable accepted/issued/completed interfaces; necessary operand holding and pipeline adjustments. Default maintains existing compute lane counts without adding large arrays.

| ID | Mandatory Acceptance Requirement |
|---|---|
| UNIT-01 | Conflict-free RF reads/writes and ARF row accesses achieve unit_contract acceptance rates; complete setup/steady/drain cycles satisfy analytical formulas |
| UNIT-02 | ≤ 2 small MMA commands compared against standalone exact oracle, reporting true per-step MACs, reads/writes, and stalls; K variation does not use default K16 footprint |
| UNIT-03 | Sustained small transfers and finite credits from DMA to RF, with exact payload bytes, command counts, and completion counts; beat counts must not be conflated with barrier operation counts |
| UNIT-04 | Same-cycle DMA/MMA completions, ARF vector row updates, and MMA contention do not drop completions; source reusable and compute-done are not conflated |
| UNIT-05 | SIMD/SFU/reduction maintain numerical and mask consistency under new RF latencies; FMA accumulation order is not altered by repipelining |

Instruction correctness and port protocols must pass before tuning throughput. Any newly added data staging must be accounted for in register/memory structures, and cannot disappear from resource reports as an "implementation detail".

### R3-06: Hardware Counters, ROI, and Verifiable Traces

Responsibility: RTL / Performance Tooling.

Deliverables: Source, event definitions, reset/epoch/overflow rules, and read interfaces for all counters in Section 4.4; bounded traces for small ROIs.

| ID | Mandatory Acceptance Requirement |
|---|---|
| COUNT-01 | Reads/writes, grants, stalls, MACs, DMA bytes, and barrier completions in hand-crafted small traces match exactly; narrow counters have defined overflow behavior under fault injection |
| COUNT-02 | Sum of primary mutually exclusive stall counts equals blocked cycles; overlapping counts listed separately without double-counting total time |
| COUNT-03 | Per-link/VN counts on NoC match actual flit widths and control fields; dedicated DRAM paths listed separately without falsely claiming they traverse the mesh |
| COUNT-04 | ROI timestamps share a unified clock and boundaries; launch/output/drain overheads reconstruct total time; successful readings do not consume residues from preceding tasks |
| COUNT-05 | Disabling performance sampling does not alter numerical or protocol behavior; enabling default counters adds no software-visible storage hierarchy; trace sizes and windows are bounded |

Counters need not trigger every cache event that naturally does not occur in short workloads; unoccurred events display 0. Component tests verifying event definitions must directedly trigger corresponding branches, and cannot claim a branch is verified simply because full-system count is 0.

### R3-07: Compilation Layout, Resource Reporting, and Cost Feedback

Responsibility: Compiler.

Deliverables: Bank mapping, ports, and read latency integrated into TargetInfo; layout and pipeline planning account for real RF costs; R1 ownership preserved post-allocation; R2 manifest/report records new implementation fingerprint and actual resources.

| ID | Mandatory Acceptance Requirement |
|---|---|
| CC-01 | Predicts conflicts for same addresses/fragments under real bank mapping; displays all bytes and instruction counts for layout conversion, padding, and extra copies |
| CC-02 | 1/2-stage candidates exceeding RF/ARF/barrier/credit budgets are rejected; valid plans are not accepted by inflating resources |
| CC-03 | Asynchronous leases remain safe across RF read latency / collector latency; source free must not be advanced to instruction issue |
| CC-04 | Compiler plan selection/rejection is explainable; code, resources, and dynamic parameter ABIs between the two comparison implementations are not conflated; unknown costs explicitly marked unknown |
| CC-05 | P04 and P07 compared for overlap under identical numerical work and resource budgets; output timelines for preparation, waiting, transfer, and compute, without presuming speedup purely from "two roles" |

P04 is not required to be faster than P07 on all small tasks. If synchronization or short-task overhead offsets overlap, it must be reported truthfully and explained by the model; increasing K until only favorable results remain is forbidden.

### R3-08: Small Operator Numerical Paths and Product Closure

Responsibility: Compiler / Numerical Verification.

Deliverables: Controlled sources, standalone oracles, R2 artifacts, and post-installation replay commands for 16 workloads; stable algorithms and resource budgets for P10–P12 composite operators.

| ID | Mandatory Acceptance Requirement |
|---|---|
| NUM-01 | P01..P09 and S01..S04 verified against existing exact contracts and product outputs; tail padding must not expand active output regions |
| NUM-02 | P10..P12 satisfy error and domain requirements in Section 3.3; software and component checks for constant, near-constant, and alternating inputs all pass |
| NUM-03 | Softmax stably subtracts max; Norm epsilon, variance, and scaling semantics are explicit; errors are not masked by altering reference formulas |
| NUM-04 | Three non-SFU system negative cases rejected at compiler/loader capability boundaries; pseudo-results must not be produced on a full system with SFU disabled |
| NUM-05 | Workloads verified via formal post-installation APIs, with dynamic buffers, events, downloads, canaries, and draining checked per R2; no test helpers assembling correct answers |

The bounded numerical domains in this phase's samples do not represent a general math library; unverified domains must be explicitly identified in the support matrix. Numerical domains already committed by R1/R2 must not be contracted.

### R3-09: Model Calibration, Holdout Validation, and Finite Ablation

Responsibility: Performance Modeling / Verification.

Frozen calibration set: P01, P02, P04, P07, P08, P09, P10, P12, and S01, S02, S03. **Holdout set is P03, P05, P06, P11, and S04**; fitting model parameters using RTL cycles of these holdout cases prior to finalizing model parameters is prohibited. All cases undergo correctness checks; only performance model fitting usage is separated.

| ID | Mandatory Acceptance Requirement |
|---|---|
| CAL-01 | Model and RTL use identical artifacts/schedules, profiles, initial states, and external memory constraints; predictions saved with hashes prior to reading corresponding measurement results |
| CAL-02 | candidate core holdout cases individually satisfy: `abs(pred - measured) ≤ max(16 cycles, 0.10 * measured)`; S04 satisfies: `≤ max(64 cycles, 0.20 * measured)`; satisfied per item, not masked by averages |
| CAL-03 | Reads, writes, commands, and completions for deterministic components are exact; memory requests and valid bytes are exact per workload; explicit modeling boundaries when external queuing latency is indeterminate, without inserting actual RTL total cycles |
| CAL-04 | Resource and timeline ablations for P04 vs. P07 and S02 vs. S03 are consistent; performance deviations provide identifiable events, rather than merely stating "cache effects" |
| CAL-05 | reference vs. candidate report numerical results, total cycles / ROI, RF stalls, traffic, and causes of change per case; all degradations disclosed, without cherry-picking only the fastest cases |

If holdout error exceeds thresholds, the phase cannot be signed off. Remedying the model and conducting new formal acceptance under a new model version is permitted, but prior predictions and failure records must be preserved; adding case-name correction factors for P03/P05/P06/P11/S04 is prohibited. Reruns remain limited to the formal 32 performance inventory, and failed runs cannot be overwritten to green.

Mandatory ablations are limited to P04 ↔ P07, S02 ↔ S03, and reference ↔ candidate. Other bank/credit/stage candidates undergo model ablation only, without running extra full-chip RTL.

### R3-10: Storage Structure Synthesis and Physical Claim Boundaries

Responsibility: RTL / Synthesis.

Deliverables: Memory-preserving synthesis flow, memory instance/port/bitwidth reports, peripheral control logic area/FF counts, comparison against legacy data-reset structure; all top levels synthesizable/elaboratable under selected parameters, without hiding logic inside opaque blackboxes.

| ID | Mandatory Acceptance Requirement |
|---|---|
| SYN-01 | In pre-arithmetic-expansion memory structure checks, ≥ 90% of VRF/ARF data bits are retained as synchronous memory objects; remaining valid/bypass/collector bits accounted for separately; data capacity is correct |
| SYN-02 | Real read/write ports, clocks, masks, and latency of each memory match the profile; no asynchronous read substitutions, full data array resets, or implicit extra read replications |
| SYN-03 | Structural synthesis/lint of top level has no unexplained latches, combinational loops, width truncations, multiple drivers, or unconnected required ports; all modified modules included |
| SYN-04 | Generated memory wrappers pass interface contract checks against a locked RAM implementation; if physical macros or FPGA mappings are not yet available, label strictly as memory-inferred, without claiming SRAM area or Fmax achieved |
| SYN-05 | Synthesis report separately lists data memory capacity, logic gates / FFs, model clock, and actual STA results; without actual libraries, Fmax/power/physical area are marked NOT MEASURED, without converting gate depth to GHz |

SYN-04 requires interface contracts and synthesizable memory-preserving structures; purchasing a PDK or taping out is not required in this phase. If real macro or FPGA RAMs are introduced, primitives/libraries, RDW/mask behaviors, and constraints must be explicitly locked, and their evidence cannot be replaced by generic memory nodes.

By default, time-consuming full-chip flattening and generic gate mapping to estimate massive RF "area" are not performed. Structural synthesis has a single-task timeout of 1200 s; timeouts constitute incomplete deliverables, and missing modules cannot be recorded as 0 gates. Actual technology library STA represents optional supplementary evidence, and is not an implicit prerequisite in the absence of a PDK.

### R3-11: Budget Gates, Property Counterexamples, and Mutations

Responsibility: Verification / Tooling.

Deliverables: Budget linter, per-run and cumulative timers, trace limiters, standalone numerical/event checkers; unified registration of all newly added RTL runs.

| ID | Mandatory Acceptance Requirement |
|---|---|
| BUDGET-01 | Exceeding M/N/K/grid/row/repetition limits, implicit Cartesian products, or runs outside the 32 performance inventory rejected prior to launching RTL |
| BUDGET-02 | Single-run wall/cycle limits, cumulative 3600 s / 2,000,000 cycles, and component execution count limits caught by real gates; verified using reduced dummy budgets to avoid wasting real hours |
| BUDGET-03 | Each run records workload, implementation, profile, seed, actual wall/cycle time, and trace bytes; missing any item blocks signoff |
| BUDGET-04 | Standalone software/model cases do not launch RTL; waveforms disabled by default; failed runs and bounded diagnostic traces do not overwrite formal failure status |

The following mutations must all be detected, injected into isolated copies:

| ID | Intentional Defect | Mandatory Detecting Check |
|---|---|---|
| R3M-01 | Incorrectly advance synchronous read by one cycle, or return data from wrong bank | RF-02, FLOW-01 |
| R3M-02 | Remove a return/writeback backpressure or reservation | FLOW-03, FLOW-04 |
| R3M-03 | Reorder ARF accumulation or drop a masked write | UNIT-02, UNIT-05, NUM-01 |
| R3M-04 | Double-count overlapping stalls or drop a completion event | COUNT-01, COUNT-02 |
| R3M-05 | Model sets infinite RF ports or infinite queues | MODEL-01, MODEL-02 |
| R3M-06 | Benchmark secretly increases K, grid, or repetitions | BUDGET-01 |

The above counterexamples are detected preferentially within corresponding component tests, without replicating the entire GEMM/system workload matrix. The same checker must pass without mutation; mutation failure must be triggered by the designated semantic violation, rather than counting an unrelated build failure as success.

### R3-12: Dual Environments, Release Evidence, and Documentation

Responsibility: CI / Maintainers.

Stable entry points to implement:

```sh
make r3-doctor
make r3-check-budget
make r3-model
make r3-components
make r3-microbench
make r3-synth-structure
make r3-acceptance
make r3-verify-evidence EVIDENCE=<run-directory>
```

`r3-acceptance` uniformly deduplicates and logs reference/candidate 32 performance runs, component/calibration executions, models, structural synthesis, mutations, and inherited R1/R2 regressions; sub-targets must not re-run duplicate regressions. Two independent Linux environments are built according to locked toolchains; all mandatory R1/R2 regressions are executed once per environment on the candidate commit.

| ID | Mandatory Acceptance Requirement |
|---|---|
| REL3-01 | Newly added RTL workloads in both environments A and B satisfy budgets; 32 performance results complete; deterministic inputs yield identical cycles/traffic/numerical outputs; host wall times recorded separately without requiring identity |
| REL3-02 | R1/R2 candidate regressions all pass; new memory profile parameters match top instances; post-installation R2 entry points show no regression |
| REL3-03 | Missing evidence, altered values, cross-commit/profile/model version mixing, omitted slow cases, and fabricated budget summaries are rejected by the validator |
| REL3-04 | All work package assignees, test IDs, evidence paths, and support boundaries are complete; unmeasured PPA and model projections are clearly marked, without claiming large-matrix RTL benchmark results |

Mandatory archive structure:

```text
contracts/       profiles, unit_contracts, bank/port/RDW/reset rules, numerical domains
inventory/       16 workloads, 32 expanded performance runs, component inventory, budgets, seeds
build/           reference/candidate sources and tool locks, fingerprints, installation info
model/           inputs, prediction hashes, calibration/holdout partition, versions, errors, and sweep results
rtl/             real artifacts, parameters, counters, output oracles, actual wall/cycle times, short traces
storage/         memory instances and port/bitwidths, structural synthesis, measured/unmeasured physical metrics
mutations/       six counterexamples and corresponding failure evidence
inherited/       R1/R2 acceptance references and deduplication mappings on the same candidate
acceptance.json  machine verdict derived across work packages, test cases, and budgets
```

## 6. Execution Order and Final Signoff

| Order | Work Package | Non-Waivable Gate |
|---|---|---|
| 1 | R3-01, R3-02 | Freeze small workloads, budgets, unit contracts, and model boundaries first |
| 2 | R3-03, R3-04, R3-10 | Synchronous memory and finite ports verified real before integrating into large core; inspect structure early to avoid reverting to FF arrays |
| 3 | R3-05, R3-06, R3-07 | Unit timing, counters, and compiler resources jointly closed |
| 4 | R3-08, R3-09 | Execute only frozen 32 performance RTL runs, completing numerical, model holdout, and finite ablations |
| 5 | R3-11, R3-12 | Budget/counterexample proofs, dual environments, inherited regressions, and formal evidence |

The final verdict must simultaneously satisfy:

```text
R1 and R2 fully pass on the identical candidate commit
AND Real synchronous memory, capacity/port/latency/reset contracts established
AND Arbitration/collector/return resource conservation and progress checks pass
AND Unit throughput/latency conforms to pre-frozen structural analytical formulas
AND 16 small workloads × reference/candidate all numerically correct and drained
AND Model holdout errors individually satisfy thresholds, without predicting from measurements
AND All cycle, traffic, and useful work statistics verifiable, with no concealed degradations
AND All newly added RTL size, count, cycle, and wall-clock budgets satisfied
AND Structural synthesis genuinely preserves memory, with physical claims bounded by evidence
AND All mutations detected; dual-environment and evidence validation passes
AND Zero unexpected fail/skip/xfail/xpass/timeout/retry or orphaned processes
AND R3-01 through R3-12 all PASS
=> R3 ACCEPTED
Otherwise => R3 NOT ACCEPTED
```

| Work Package | Initial Status | Assignee / Evidence |
|---|---|---|
| R3-01 Baseline, Budget & Contracts | NOT VERIFIED | To be filled during implementation |
| R3-02 Fast Performance Model | NOT VERIFIED | To be filled during implementation |
| R3-03 Synchronous VRF/ARF | NOT VERIFIED | To be filled during implementation |
| R3-04 Collector, Arbitration & Return | NOT VERIFIED | To be filled during implementation |
| R3-05 Unit Throughput | NOT VERIFIED | To be filled during implementation |
| R3-06 Counters & ROI | NOT VERIFIED | To be filled during implementation |
| R3-07 Compiler Cost Feedback | NOT VERIFIED | To be filled during implementation |
| R3-08 Small Operator Numerical Closure | NOT VERIFIED | To be filled during implementation |
| R3-09 Calibration, Holdout & Ablation | NOT VERIFIED | To be filled during implementation |
| R3-10 Storage Structural Synthesis | NOT VERIFIED | To be filled during implementation |
| R3-11 Budget Gates & Mutations | NOT VERIFIED | To be filled during implementation |
| R3-12 Dual Environments & Release | NOT VERIFIED | To be filled during implementation |

**Small workload size does not lower correctness standards; rather, directed inputs and observable events are used to cover issues. Skipping mandatory properties due to RTL simulation slowness is forbidden, as is arbitrarily expanding RTL workloads under the guise of "greater realism".**
