# R38: Hardware DevOps and MLIR–ISA–Microarchitecture Co-Iteration Plan for ISCA

- Document Status: **Research Proposal / Local Review Draft; not yet implemented, no R38 performance conclusions.**
- Date: 2026-09-22; literature survey current as of this date.
- Baseline Starting Point: `r37-triton-coverage`, `1b2a9d6a394a760d6dc015a6baf6a519bdb19650`.
- Delivery Scope: Documentation only; no commits, no pushes, no modifications to R37 kernels, compiler, or RTL.
- All speedup multiples, error thresholds, sample counts, and schedules below are **proposed experimental designs or acceptance targets**; they do not represent existing measurements or mandatory requirements for positive results.

## Reading Navigation and Recent Decisions

- **Paper Mainline**: Sections 0, 3, 9: Hardware DevOps and the three-layer minimal runnable closed loop.
- **Overall Architectural Metric**: Key review in **Section 4.10 AQ**. It aggregates workloads, scales, and environments using a common abstract resource budget and candidate-independent ideal lower bound; Sections 4.1–4.9 provide foundational formulas and anti-distortion rules.
- **Concrete Implementation**: Sections 5–8, 13: 45 operators, cost model, profiler, cross-layer case studies, and interfaces.
- **Paper Experiments**: Sections 9–12, 15: Methodology controls, Agent, NVIDIA, PyTorch/LLM.
- **Execution Plan**: Sections 14, 16, 17: Work packages, timeboxes, risks, and acceptance criteria.

The metric hierarchy corresponding to the latest decisions is: **AQ measures normalized architectural family quality; S measures concrete deployment performance; E measures the distance of a concrete configuration from its own upper bound; design time/tokens measure iteration efficiency.** These four cannot replace one another or be arbitrarily multiplied into a single composite score. AQ serves as the primary architectural score; other metrics substantiate its validity, implementability, and acquisition cost.

## 0. Core Proposition and Paper Positioning

**Core Concept: Apply continuous integration, observability, and short feedback loops from DevOps to hardware design, keeping MLIR, ISA, and microarchitecture in a minimal runnable closed loop; under identical design budgets, explore superior architectures through rapid co-iteration.**

R38 should not be organized into five disjoint projects. The five thrusts correspond to a single chain of evidence:

```text
Real, minimally-modified Triton workloads
          ↓
MLIR / lowering ↔ ISA semantics and resource contract ↔ Executable microarchitecture
          ↓                                                      ↑
Common AOT + runtime → RTL / FPGA → profiler
          ↓                                                      ↑
ArchBench metrics → cost model → Next round of cross-layer changes
          ↓
Design quality, iteration efficiency, reliability under fixed budget
          ↓
NVIDIA external reference + PyTorch / FPGA LLM system validation
```

Proposed research proposition:

> Treating compiler, instruction set, and microarchitecture modifications as a single verifiable, measurable, and revertible co-iteration unit reduces cross-layer feedback latency and improves the performance-resource Pareto tradeoff of the final design under identical resource and engineering budgets; explainable cost models and profilers further accelerate human or Agent search efficiency.

Three causal chains must be verified independently:

1. **Faster loop**: The turnaround time from proposing a change to receiving credible correctness, performance, and resource feedback is reduced.
2. **More effective loop**: Exploring more valid designs within the same total budget, or reaching combinations unreachable when layers are fixed sequentially.
3. **Superior final design**: Improved actual performance under resource constraints on frozen held-out workloads, rather than merely improved predictive scores.

"Automated code generation", "utilizing MLIR", "passing 45 operators", or "Agent-authored RTL" are individually insufficient to prove these propositions. Final FPGA LLM execution screenshots demonstrate end-to-end completeness, but cannot substitute for the causal evidence above.

### 0.1 Proposed Organization of Contributions

| Contribution | Claim to Prove | Key Evidence |
|---|---|---|
| C1: Cross-layer executable iteration methodology | Compiler, ISA, and hardware co-evolve while remaining continuously executable, diagnosable, and comparable | Contracts, versioned artifacts, failure rollbacks, authentic cross-layer change traces |
| C2: ArchBench evaluation protocol | Analyze architectural efficiency across configurations without conflating scale and absolute performance | 45-operator scale space, resource upper bounds, frozen weights, anti-gaming validation |
| C3: Feedback-driven co-design | Outperforms single-layer or staged sequential optimization under identical design budgets | Controlled experiments, >= 3 cross-layer case studies, unseen shapes and configurations |
| C4: Agent as closed-loop executor | Cost model + profiler improves optimization quality and engineering cost | 2×2 factorial ablation, token/tool/wall-clock costs, replicated runs |
| System validation | Resulting designs support real-world applications | FPGA physical measurements, PyTorch graph execution, LLM prefill/decode |

The paper should prioritize C1 + C3. ArchBench serves as the common yardstick underpinning scientific credibility, aiming also for an independent benchmark contribution. The Agent serves as an experimental subject and productivity multiplier; overall conclusions must not depend entirely on a continuously updating closed-source model.

### 0.2 Novelty Boundaries

Agile hardware development, hardware-software co-design, and feedback-directed search are established fields. Chipyard, Gemmini, HASCO, CIRCT, and ACT cover many related capabilities. In particular, **HASCO is a direct prior work from ISCA 2021 that must be compared against head-to-head**, rather than only citing generic Roofline or LLM kernel papers.

Our intended distinction is: **Targeting broadly portable tensor programs, allowing MLIR lowering, ISA contracts, and microarchitectural structures to evolve concurrently, driven by empirical feedback from continuous execution, while accounting for design quality and iteration costs together as primary research findings.** This is a positioning statement to be proven, not a pre-emptive claim of absolute priority. See Section 2.

## 1. Project Facts: Existing R37 Foundations and Gaps R38 Must Bridge

### 1.1 Existing Reusable Assets

| Asset | Current Evidence and Path | Supported Conclusion |
|---|---|---|
| Original operator coverage | [r37_validation.md](r37_validation.md), [r37_tilebench.json](../spec/r37_tilebench.json) | 45/45 original operators; each O0/O1/O2 passes 149/149 numerical cases/stages, total 447/447 |
| Public entry points | [aot.py](../python/tnca/aot.py), [runtime.py](../python/tnca/runtime.py) | Tests exercise `compile_aot`, `.tnca`, `Session`, and RTL, rather than internal lowering in isolation |
| Upstream source constraints | TileBench commit `6e2cf155dfe01f4660540d187c81b09eb55f394f` | Original JIT function bodies remain unchanged; small-size launch provided via adaptation layer |
| MLIR foundation | [TNCADialect.td](../compiler/include/tnca/Dialect/TNCA/IR/TNCADialect.td), [VerifyAsyncOwnership.cpp](../compiler/lib/Dialect/TNCA/Transforms/VerifyAsyncOwnership.cpp) | Target dialect and ownership verifier foundation present; does not imply entire public AOT path is unified under MLIR passes |
| Compilation implementation | [tnca_cc](../compiler/python/tnca_cc), [compiler.py](../python/tnca/triton_backend/compiler.py) | Python lowering coexists with MLIR implementations; requires mapping real call graphs and logging pass provenance |
| Cycle measurement precedent | [perf.py](../tools/r35/perf.py) | Joint device cycle and correctness checks exist; strictly avoids substituting host wall time for device cycles |
| Counter contract | [r3_counters.json](../spec/r3_counters.json) | Distinguishes mutually exclusive stalls, overlapping stalls, operand collector, and testbench memory events |
| RTL structure | [tnca_top.v](../rtl/tnca_top.v), [core](../rtl/core), [nuca](../rtl/nuca), [noc](../rtl/noc) | RF, NUCA, and mesh implementation tiers are localizable; structural existence != validated full-scale performance |
| Tutorials | [08_triton_gemm.md](../tutorial/08_triton_gemm.md), [09_tensor_descriptors.md](../tutorial/09_tensor_descriptors.md) | Standard users can execute GEMM and descriptor cases from AOT / Session entry points |

**447 represents 149 cases/stages × 3 optimization levels, not 447 independent real-world model shapes, and not thousands of performance measurement points.** Stage counts of multi-stage operators cannot be converted directly into evaluation weights; specifically, the many stages of radix sort must not artificially inflate its contribution to aggregate scores.

### 1.2 Elements That Cannot Be Inherited Directly as Paper Conclusions

1. **R37 represents small-scale correctness coverage.** Arbitrarily large tiles, long reductions, large-K INT8, FP8 format boundaries, corner-case inputs, and all launch parameter spaces require extended validation.
2. **The public runtime currently dispatches kernels program by program.** `Session._run_one` loops linearly over the grid calling `dev.launch(..., role_mask=1)`, accumulating reported cycles. This validates grid semantics, but sequentially accumulated cycles do not represent the full-chip throughput of an on-device parallel grid scheduler.
3. **The multi-stage correctness harness is not a device-resident pipeline benchmark.** Host roundtrips for intermediate buffers and Python scheduling must be separated in a dedicated performance harness.
4. **The default simulated device memory is minimal.** `tb/device/tnca_device.v` configures 65,536 64-bit words (512 KiB total); the runtime default heap is `0x20000..0x80000`. This remains far from an external memory system capable of running full LLMs.
5. **Functional fallbacks must not be reported as native hardware capabilities.** [r37_isa_gaps.md](r37_isa_gaps.md) records discrepancies across formats, descriptors, and dot products. Vector expansions for FP32 dot and floating-point emulation paths for INT8 must be tracked independently; correctness on small K does not guarantee exact precision over arbitrary accumulation ranges.
6. **Existing synthesis results do not constitute PPA.** [synthesis.md](synthesis.md) explicitly states that without a PDK, gate/depth/flop metrics serve as structural proxies; they cannot be cited as mm², GHz, or watts.
7. **R36 does not equal completed FPGA board bring-up.** [r36_status.md](r36_status.md) records `IMPLEMENTED / NOT ACCEPTED` (detailed status `IMPLEMENTED / PENDING GATE`) with locked platform gate issues, and that phase explicitly excluded board selection, vendor IP, P&R, bitstreams, and physical debugging.
8. **16 tiles, 1.0 GHz, 8 MiB in architectural overviews are design proposals.** The [architecture draft](design/tnca_v1_architecture_draft.md) (formerly `architecture_overview.md`) represents an exploration baseline rather than implemented silicon or measured clock rates.

The first artifact of R38 must be an "Evidence Registry", categorizing each capability as `documented / implemented / functionally-validated / cycle-validated / synthesized / board-measured`, with each level linked to corresponding artifacts. A blanket `supported=true` spanning these distinct tiers is inadmissible.

## 2. Literature Context and Research Opportunities

The table below outlines our research positioning and does not represent an exhaustive re-implementation of each work. Preprints and dynamic documentation must be locked to specific versions during formal evaluation; preprints must not be cited as accepted conference papers.

| Direction / Literature | Applicable Concepts | Constraints or Distinctions for R38 |
|---|---|---|
| [Roofline](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2008/EECS-2008-134.pdf), Williams et al. | Compute and data movement bounds, operational intensity | Operating close to the roofline does not prove a machine is fast; DRAM and FLOPs alone cannot model all 45 operators |
| [ERT](https://amcr.lbl.gov/departments/computer-science-department/ppan/roofline-performance-model/empirical-roofline-tool-ert/), [Hierarchical Roofline](https://arxiv.org/abs/2009.05257) | Measured platform ceilings, hierarchical memory bottlenecks | Distinguish theoretical hard bounds from empirical sustained throughput; the latter is not an absolute upper bound |
| [Performance Portability metric](https://arxiv.org/abs/1611.07409) | Cross-platform efficiency aggregation and coverage | Harmonic means have methodological precedent; ArchBench's contribution cannot be just renaming an average |
| [Timeloop](https://timeloop.csail.mit.edu/), [Accelergy](https://accelergy.mit.edu/tutorial.html), [MAESTRO](https://maestro.ece.gatech.edu/) | Mapping, dataflow, resource, and energy modeling | Borrow analytical modeling principles; add TNCA dynamic cache, backpressure, and role dependencies; avoid duplicating tensor dataflow theory from scratch |
| [MLIR](https://arxiv.org/abs/2002.11054), [CIRCT](https://circt.llvm.org/) | Multi-level representations, composable transformations, hardware infrastructure | Using these tools does not constitute novelty; no requirement to rewrite existing RTL into CIRCT solely for paper writing |
| [Chipyard](https://people.eecs.berkeley.edu/~krste/papers/Chipyard-IEEEMicro2020.pdf) | Agile SoC workflows integrating generation, simulation, and implementation | Must quantify the specific cross-layer co-evolution mechanisms added and their quantitative gains, beyond CI automation |
| [Gemmini](https://dspace.mit.edu/entities/publication/f992c764-0107-4978-8ba8-6b4120c954cc) | Full-stack integration for DNN architecture evaluation | Must move beyond scanning fixed accelerator template parameters to demonstrate genuine ISA/lowering/structural co-evolution |
| [HASCO](https://arxiv.org/abs/2105.01585), ISCA 2021 | Agile hardware-software co-design for tensor computation | Direct neighbor; compare design spaces, executable feedback, dynamic microarchitectural modeling, and engineering cost accounting |
| [ACT](https://arxiv.org/abs/2510.09932) | Generating compiler backends from tensor accelerator ISA descriptions | Schema-driven backends have prior art; our focus is contract evolution and measured closed loops, not claiming first ISA-driven compilation |
| [Multi-level RISC-V Backend](https://arxiv.org/abs/2502.04063), CGO 2025 | Exploiting specialized ISA features across abstraction levels | Must demonstrate incremental gains from three-layer co-evolution beyond standard customized compiler backends |
| [Ansor](https://www.usenix.org/conference/osdi20/presentation/zheng) | Combining search with learned cost models | Serves as reference methodology for fixed-hardware schedule search; Agent search must be compared against non-LLM search |
| [FireSim](https://fires.im/publications/) | FPGA-accelerated cycle-accurate system simulation | Strictly distinguish simulated target cycles from host/FPGA simulation wall-clock time; no requirement to migrate the entire FireSim stack |
| [DCO](https://arxiv.org/abs/2512.07312), [LLaMCAT](https://arxiv.org/abs/2512.00083) | Predictive cache management, arbitration, and throttling mechanisms | Conceptual foundation and ablation target; inherited mechanisms must not be claimed as R38 innovations |
| [FlashAttention](https://arxiv.org/abs/2205.14135) | IO-aware exact attention computation | Semantic work differs from intermediate algorithmic traffic; writing back full attention matrices must not be counted as required traffic |
| [ArchEval](https://arxiv.org/abs/2607.03601), 2026 preprint | Evaluating LLM agent architecture design capabilities across simulators and feedback settings | Direct neighbor; contrast our real MLIR–ISA–RTL contract evolution, cost model/profiler factorial experiments, and AQ protocol, rather than claiming first LLM architect evaluation |
| [KernelBench](https://arxiv.org/abs/2502.10517) | Evaluating correct and performant kernel generation | Retain correctness gates, extend to ISA/RTL and engineering budgets; avoid comparing solely against PyTorch eager |
| [CudaForge](https://arxiv.org/abs/2511.01884), [CUDA Agent](https://arxiv.org/abs/2602.24286) | Hardware feedback, verification, and Agent kernel optimization | Providing profilers to agents is known art; requires our complete factorial ablation and cross-layer conclusions |
| [Compiler–LLM Cooperation](https://arxiv.org/abs/2604.04238) | Multi-abstraction compiler and Agent cooperation | Cross-layer agents are not unprecedented; hardware contract changes, RTL feedback, and cost attribution form key differentiators |
| [FlightLLM](https://arxiv.org/abs/2401.03868) | Full LLM mapping and execution on FPGAs | Running LLMs is not a new standalone contribution; focus is end-to-end viability of compiler paths and co-designed hardware |
| [PyTorch Accelerator Integration](https://docs.pytorch.org/docs/main/accelerator/index.html) | Dispatcher and runtime integration paths for new accelerators | Do not re-architect PyTorch; lock versions and stage graph backend before real device backend |
| [MLPerf Inference Rules](https://github.com/mlcommons/inference_policies/blob/master/inference_rules.adoc) | Explicit constraints on workloads, quality, and execution protocols | Borrow quality/scenario/disclosure guidelines; do not claim ArchBench is an official MLPerf result |

### 2.1 Nearest-Neighbor Cross-Examination Prior to Formal Publication

Construct a `related_work_matrix`, reviewing the text, artifacts, and limitations of HASCO, Gemmini, Chipyard, ACT, ArchEval, and DCO/LLaMCAT:

- Does the framework allow ISA semantic modifications, or merely parameter tuning?
- Is the compiler generated alongside hardware, or does it select schedules at runtime?
- Does feedback originate from analytical models, traces, RTL, FPGAs, or ASICs?
- Are correctness regressions and invalid designs tracked and reported?
- Are final outcomes evaluated under fixed engineering/design budgets?
- Are feedback latency, replication costs, and designer effort quantified?
- Are dynamic control flow, non-GEMM workloads, cache backpressure, and multi-stage operators covered?

The literature search conducted here establishes these as direct neighbors, **but does not suffice to claim they lack specific capabilities**. Final novelty claims must be locked only after completing this matrix.

## 3. Operationalizing "Hardware DevOps" into an Executable Research Methodology

### 3.1 Definition of the Minimal Runnable Closed Loop

The closed-loop state is formalized as:

\[
V=(C,I,H,R,W,M,P,E)
\]

where $C$ is the compiler/pass version, $I$ is the ISA/ABI contract, $H$ is the RTL and hardware configuration, $R$ is the runtime, $W$ is the workload/input suite, $M$ is the cost model, $P$ is the profiler schema, and $E$ is the toolchain and execution environment.

An experiment cannot record only a Git SHA: uncommitted patches, generated RTL, synthesis seeds, compilation flags, cached artifacts, and device firmware can all alter results. Every candidate must therefore produce a content-addressed manifest.

**Minimal Closed-Loop Passing Conditions:**

1. Ingest unmodified representative Triton source code into public AOT.
2. Record concrete TTIR/target IR/pass pipelines, producing version-matched binaries.
3. Submit via common runtime to designated RTL backend, yielding functionally correct outputs.
4. Obtain traceable device execution time and at least one explainable bottleneck event.
5. Execute resource proxy / synthesis checks, verifying configuration feasibility.
6. Generate complete diffs relative to parent; on failure, automatically preserve diagnostic evidence and rollback to the last known good candidate.

"Minimal" refers to selecting 6–8 representative points to shorten turnaround, not omitting runtime execution, hardware semantics, or numerical checks.

### 3.2 The Cross-Layer Change Unit

Each modification is expressed as a `design_change`:

```yaml
# Design draft, no corresponding command or schema implementation yet
id: rf-window-layout-001
parent: <candidate-content-hash>
hypothesis: "bank-aware lowering with window layout reduces operand collector conflicts"
touched_layers: [compiler, isa, microarch]
semantic_change: false
feature: rf_window_layout_v2
compiler_passes: [<real-pass-name>]
isa_contract: <schema-hash>
rtl_config: <config-hash>
required_tests: [encoding, ownership, randomized-backpressure, aot-smoke]
performance_cases: <development-suite-hash>
resource_budget: <budget-hash>
acceptance: <frozen-policy-hash>
```

Actual identifiers will be determined during implementation. Candidate modifications encompass new primitives, resource organization, and scheduling; changing descriptive strings in reports does not constitute a three-layer change.

### 3.3 Minimal Scope of the Contract

The ISA and resource contract must cover: instruction encoding, reserved fields, data types/rounding/overflow, masking/tail elements, operand shapes, memory read/write footprints, side effects, completion events, memory consistency, exceptions, resource consumption, and feature versioning.

- Semantic specifications must be jointly adhered to by the interpreter/oracle, RTL, and compiler verifier.
- Latency and throughput are properties of concrete microarchitectures, and must not be codified into ISA correctness requirements.
- Asynchronous primitives must explicitly distinguish: *request accepted*, *data readable*, *computation completed*, and *registers reusable*.
- Performance hints must not alter computational results; ignoring hints must remain semantically correct.
- AOT artifacts must declare required features; the runtime must reject incompatible hardware prior to launch, rather than silently substituting semantics.
- Schemas may generate decoder tables, disassemblers, legality suites, and compiler capability tables; however, core semantics must have independent oracles to avoid mutually validating co-generated errors.

### 3.4 Continuous Validation and Candidate Promotion

| Gate Level | Target Feedback Time | Scope | Usable as Final Performance Evidence |
|---|---:|---|---|
| L0 | Seconds to minutes | Schema/encoding/IR verifiers, compilation smoke tests | No |
| L1 | 5–15 minutes | Representative AOT + RTL runs, targeted randomized tests | Cycle metrics for evaluated points only |
| L2 | Hours | Full R37 O0/O1/O2 suite, expanded correctness, cycle regressions | Verified points usable |
| L3 | Tool-dependent | Synthesis, resource utilization, static timing, representative full-system RTL | Explicitly tagged with evidence tier |
| L4 | Board flow-dependent | Place & Route, bitstreams, full FPGA hardware measurements | Board measurements usable |
| Release | Fixed window | Frozen held-out suite, complete resource and protocol audits | Primary source for paper tables |

These durations represent target SLO proposals; baselines must be measured before final adjustment, and tests must never be omitted to meet time limits. Candidates failing L2 must not serve as credible baselines for subsequent iterations.

**Continuous execution does not imply that every exploratory branch succeeds.** Failed candidates are permitted, provided a reproducible last-known-good state is maintained; failure rates and recovery times constitute valid research data.

### 3.5 DevOps Metrics

- `feedback_latency`: Elapsed time from candidate manifest submission to the first gate providing correctness feedback.
- `validated_lead_time`: Elapsed time from hypothesis formulation to passing target verification gates, including bug fixes and tooling retries.
- `valid_candidates / wall-hour`: Throughput of valid designs per wall-clock hour, and valid candidates discovered per 1M tokens.
- `change_failure_rate`: Proportion of candidates failing correctness, compatibility, or resource gates.
- `recovery_time`: Duration from failure to restoring a runnable candidate.
- `reproduction_success`: Proportion of clean-room environments successfully reproducing identical artifacts and numerical results.
- `best_feasible_quality(budget)`: Pareto curve of optimal valid design quality as a function of engineering budget.

These metrics adapt established DevOps methodologies; their event boundaries are anchored in manifests, and Git commit counts or generated lines of code cannot substitute for genuine productivity measures.

## 4. ArchBench: Architecture Quality Evaluation Protocol

### 4.1 Avoiding an Unconstrained "Single Architecture Score"

An architecture can improve performance by adding compute units, expanding SRAM, or increasing bandwidth; conversely, it can inflate utilization metrics by artificially depressing theoretical peak throughput. No universal metric exists that neutralizes process, area, frequency, bandwidth, and scale discrepancies simply by dividing by peak throughput.

ArchBench results should be reported as a multi-dimensional evidence tuple; the specific definition of the Architecture Quality metric (AQ) appears in **Section 4.10**, serving as the primary metric for cross-deployment, cross-scale evaluation:

\[
\boxed{(\text{Correctness/Coverage},\;\text{Fixed-Budget Performance},\;\text{Upper-Bound Efficiency},\;\text{Resources and Energy},\;\text{Iteration Cost})}
\]

Within a fixed hardware configuration, upper-bound efficiency effectively isolates compiler and ISA overheads; across architectures, absolute execution wall time and physical resource constraints must be assessed simultaneously. The paper may employ a primary ranking metric, but its constraints and accompanying metrics must be fully disclosed.

### 4.2 Workload Definitions Precede Implementation

A workload point $i$ comprises: operator semantics, input tensor shapes, data types, layouts/strides, value distributions, numerical precision contracts, execution context, and call frequencies. Multiple stages within an operator are internal implementation details of a single semantic instance, and do not constitute independent scoring targets.

Two evaluation tracks are distinguished:

- **Portable-Source Track (Primary Track)**: TileBench JIT source code hashes are frozen; backends may alter lowering passes, tile mappings, and hardware implementations. Permissible launch configuration adaptations must be documented; replacing function bodies per platform is prohibited.
- **Semantic-Best Track (External Performance Reference)**: Preserves mathematical semantics and numerical precision while allowing platform-tuned libraries or specialized kernels; enables fair comparison against cuBLASLt, cuDNN, CUTLASS, etc. Results across differing algorithms must be reported independently.

TNCA must not employ loose approximations or fused implementations while forcing NVIDIA baselines into un-fused eager execution; conversely, the inability of native GPU launch wrappers to run on new hardware must not be mischaracterized as unsupported kernel semantics.

### 4.3 Theoretical Lower Bounds per Workload Point

The traditional formulation is:

\[
P_{roof}=\min(P_{compute}, B_{DRAM}\cdot I),\qquad
T_{LB}=\max(F/P_{compute},Q/B_{DRAM})
\]

derived from [Roofline](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2008/EECS-2008-134.pdf). For TNCA, this is generalized into a **multi-resource necessary condition**:

\[
L_{i,c}=\max\left(
\max_r \frac{W^{min}_{i,r}}{C^{max}_{c,r}},
\max_\ell\frac{Q^{min}_{i,\ell}(S_c)}{B^{max}_{c,\ell}},
D^{min}_{i,c}
\right)
\]

- $c$: Concrete realizable configuration; $S_c$: Associated memory capacities.
- $r$: Service resources including MMA, SIMD, SFU, integer/comparison, reduction, atomics, instruction issue.
- $\ell$: Provable structural bottlenecks including DRAM, NUCA banks, required NoC bisection cuts.
- $W, Q$: Provable minimal service and transfer requirements within the specified algorithm/semantic class.
- $D$: Provable necessary lower bounds on dependency chains or pipeline startup latencies; omitted if unproven, never fabricated.

**Not every hardware resource exhibits a non-zero lower bound at the semantic level.** For instance, RF read counts depend on operand forwarding, staging latches, and instruction scheduling; without formal proofs, observed RF traffic must not be inserted into primary lower bounds, remaining confined to mapping diagnostic models.

Shared issue slots or shared execution units prevent multiple peak throughputs from being attained concurrently. This can be formalized via small Linear Programs (LP): solving for minimal execution time subject to port, link, and execution resource constraints under a given instruction/service mix. LP constraints must reflect strict necessary conditions; empirical penalties belong to cost models and must not be conflated with mathematical lower bounds.

Lower bounds are rarely perfectly tight and may not be simultaneously achievable across all resources. Papers must refer to "necessary resource lower bounds / efficiencies", avoiding claims of "known optimal execution times". Because bound tightness varies across operator classes, sensitivity analyses must be reported per category.

### 4.4 Accounting for Computational Work and Transferred Bytes

| Operator Category | Explainable Service Volume | Pitfalls to Avoid |
|---|---|---|
| GEMM / Conv | Useful MACs, essential I/O, declared dense/direct algorithmic scope | Counting padding, redundant computations, or spills as useful FLOPs, rewarding inefficient implementations |
| Activation / SFU | Elements processed, exp/log/reciprocal service counts | Arbitrarily converting transcendental functions into FLOP counts for cross-machine comparisons |
| Reduction / Norm | Elements processed, reduction/comparison operations, essential I/O | Assuming all reduction operations execute at peak matrix MMA rates |
| Copy / Transpose | Useful payload bytes, essential reads and writes | Defining efficiencies via FLOPs for memory-bound workloads lacking floating-point arithmetic |
| Gather / Scatter / Histogram | Indices and effective transactions, conflict/skew parameters | Benchmarking only uniform distributions while omitting contention and atomic serialization |
| Sort / Top-k | Keys/items processed per second, comparison/data transfer bounds within declared algorithmic scope | Treating operation counts of specific bitonic/radix sorting implementations as algorithmic requirements |
| Attention | Effective attention operations, softmax service, theoretical I/O bounds | Counting $O(N^2)$ intermediate activation read/write traffic of un-fused implementations as mandatory |

For GEMM, conventional dense MAC counts ($2MNK$) must be explicitly adopted under declared algorithmic families; they must not be claimed as absolute mathematical lower bounds allowing arbitrary fast matrix algorithms. Structured sparsity and dense execution must be separated; unexecuted zeros cannot simultaneously count as executed operations and computational savings.

At every memory tier, two distinct fields must be recorded: `required_bytes_lower_bound` and `measured_bytes`. They serve separate roles: the former benchmarks efficiency, while the latter diagnoses cache reuse, redundancy, and spills. Re-inflating lower bounds using measured bytes masks memory waste as "operating near the roofline".

### 4.5 Separating Theoretical Peaks from Empirical Ceilings

Two capacity metrics must be maintained:

1. `spec_peak`: Maximum service rate verified by microarchitectural implementation and configuration; used for verifiable, auditable lower bounds.
2. `empirical_sustained`: Calibrated via independent microbenchmarks; used to diagnose practical bottlenecks.

Empirical bandwidth fluctuates with access patterns, concurrency, and measurement noise; workloads surpassing empirical ceilings do not violate physical laws. Reports may draw on [ERT](https://amcr.lbl.gov/departments/computer-science-department/ppan/roofline-performance-model/empirical-roofline-tool-ert/), but must never describe empirical ceilings as "absolute theoretical upper limits". For GPUs, rely on [Nsight Compute](https://docs.nvidia.com/nsight-compute/ProfilingGuide/) hierarchical profiles, fixing tool versions and metric definitions.

Primary efficiency is defined as:

\[
\eta_{i,c}=L_{i,c}/T_{i,c},\qquad \delta_{i,c}=1-\eta_{i,c}
\]

If a strict lower bound calculation yields $\eta > 1$ (beyond measurement uncertainty), the record fails audit: verify timing intervals, measurement units, MAC/FLOP definitions, concurrency, compression, cache warm-up states, and numerical precision. **Values must not be clipped to 1.0 to force passing.** Ratios relative to empirical ceilings may exceed 1.0, but must be labeled as empirical normalized figures and excluded from strict theoretical efficiency tables.

### 4.6 Weighting by Execution Time: Concrete Schemes for Frozen Weights

Reference architectures, implementations, and application invocation distributions $a_i$ must be frozen prior to evaluation. Given reference latencies $T_i^{ref}$:

\[
w_i=\frac{a_iT_i^{ref}}{\sum_j a_jT_j^{ref}},\qquad \sum_iw_i=1
\]

For general benchmark suites lacking end-to-end application traces, operator mass is evenly partitioned across the 45 operators, and subdivided within each operator according to pre-registered shape distributions; this is designated the **balanced track**, and must not be described as an authentic application trace. Weights derived from real model execution traces form the **application track**, documenting call counts and operator fusion boundaries.

The aggregate efficiency score is computed as the harmonic mean weighted by reference execution time:

\[
E_c=\left(\sum_i\frac{w_i}{\eta_{i,c}}\right)^{-1}
\]

The harmonic mean is sensitive to efficiency bottlenecks; raw point-wise $\eta$, quantiles, and per-class scores must be reported alongside it. The arithmetic mean $A_c = \sum_i w_i \eta_{i,c}$ may be provided as an auxiliary metric to offer intuitive insight. Harmonic aggregation follows established methodologies in [performance portability](https://arxiv.org/abs/1611.07409); this formula is not claimed as an original contribution.

The **primary performance score** is independently defined as the speedup across the identical aggregate workload:

\[
S_c=\frac{\sum_i a_iT_i^{ref}}{\sum_i a_iT_{i,c}}
=\left(\sum_i w_i\frac{T_{i,c}}{T_i^{ref}}\right)^{-1}
\]

This represents the reduction in total elapsed time for the specified invocation mix. Arithmetic averages of point-wise speedups lack this physical interpretation and must not serve as the primary performance score.

Another well-defined diagnostic metric is:

\[
U_c=\frac{\sum_i a_iL_{i,c}}{\sum_i a_iT_{i,c}}
\]

This equals $\eta$ weighted by candidate runtimes, describing aggregate necessary lower-bound efficiency across sequential executions. However, because its numerator and weights vary with candidate configurations, **$U_c$ cannot substitute for $S$ and $E$ in comparing architectures against a fixed standard**. In concurrent systems, end-to-end makespan across the workload DAG and corresponding DAG lower bounds must replace simple scalar summations.

### 4.7 Illustrative Numerical Example

Consider two workloads with frozen weights of 0.5 each. Design A has lower bounds of 1 μs and measured latencies of 2 μs for both; Design B has lower bounds of 10 μs and measured latencies of 11 μs for both.

- Design A achieves an efficiency of 0.5; Design B achieves an efficiency of ~0.909.
- However, Design A completes the identical work 5.5× faster than Design B.
- Operating closer to one's own upper bound does not imply superior absolute performance.

Now consider efficiencies $\eta = (0.9, 0.1)$: the arithmetic mean is 0.5, while the harmonic mean is 0.18. The former reflects average ratio, whereas the latter penalizes catastrophic bottlenecks. Methodological choices must be justified upfront, rather than selected post-hoc to present favorable numbers.

### 4.8 Handling Missing Results, Errors, and Timeouts

- Numerical failures or unsupported operators in full benchmark suites cannot be discarded prior to averaging; doing so constitutes a failure of the release gate.
- Exploratory reports must state both count coverage and weighted coverage $\sum_{\text{supported}} w_i$; supported-only scores may be reported strictly as diagnostic metrics.
- For strict unified rankings, unsupported efficiencies must be treated as 0, yielding an aggregate $E = 0$; predefined timeout penalties may generate auxiliary research scores, but must not be conflated with verified runs.
- Execution times for timed-out runs represent right-censored data rather than true latencies; timeout thresholds and statuses must be reported explicitly.
- Compilation failures, runtime crashes, numerical discrepancies, out-of-memory errors, and missing measurements must be recorded as distinct failure categories rather than aggregated into "slow execution".

### 4.9 Configuration Tiers and Hardware Budgets

Pre-freeze at least three **realizable hardware budget tiers**: small, medium, and large. Concrete resource bounds must be established following FPGA synthesis and board-level pilot studies, rather than directly adopting preliminary parameters from architecture drafts.

Resource constraint vectors encompass LUTs, FFs, DSPs, BRAMs/URAMs, available DDR/HBM capacity and bandwidth, physical I/O interfaces, timing closure targets, and power dissipation limits. FPGA resources constitute multi-dimensional constraints; DSPs cannot be converted into equivalent LUT counts to synthesize a single scalar area metric.

Each budget tier must document: concrete configuration, maximum operating frequency meeting timing closure, resource consumption, performance $S$, efficiency $E$, and per-category distributions. Iso-frequency baselines must be evaluated alongside to isolate cycle reductions from clock frequency variations. If adding register ports reduces execution cycles but degrades $F_{max}$, net gains must be evaluated via $\text{cycles} / \text{achieved\_frequency}$.

Explicitly distinguish:
- Strong scaling across a fixed problem size;
- Weak scaling where workload sizes scale with hardware capacity (retained as an auxiliary study);
- A single static configuration running the full benchmark suite;
- Multi-configuration portfolios (accounting explicitly for switching and reconfiguration costs);
- The oracle envelope picking optimal configurations per point (reported solely as an upper bound, not as a deployable system).

The primary evaluation prohibits free per-point bitstream reconfiguration. Small configurations lacking sufficient memory must record the workload as unsupported, rather than quietly truncating problem dimensions.

### 4.10 Core Metric: Architecture Quality (AQ) Independent of Concrete Deployments

**R38 promotes the primary metric for architectural comparison to Architecture Quality (AQ), evaluated under standardized counterfactual protocols.** The previously defined $E$ measures single-configuration utilization, and $S$ measures concrete deployment speedups; both remain essential diagnostics, but neither represents AQ.

The objective is not to derive a single conversion factor between ASIC and FPGA metrics, but rather:

> Place candidate architectures into a common, standardized abstract resource, timing, and environmental framework; evaluate their ability to convert hardware resources into useful computational throughput across identical budget tiers; and aggregate these results across frozen workload, scale, and environmental distributions into a single architectural score.

AQ does not incorporate FPGA device part numbers, ASIC process nodes, nominal clock frequencies, or board retail prices as scoring bonuses. If an architecture ported across distinct substrates exhibits identical abstract behaviors, it receives an identical AQ; differences in physical deployment quality are captured via auxiliary metrics.

#### 4.10.1 Why Absolute Invariant Rankings Are Impossible

Consider Architecture A using larger local SRAM to reduce DRAM traffic, and Architecture B using a larger compute array with smaller local SRAM. In bandwidth-constrained environments Architecture A excels; in high-bandwidth environments Architecture B dominates. Similarly, an architecture with simple control may excel at 1 core, whereas a network-centric design scales better at 64 cores.

One cannot simultaneously demand preserving authentic architectural tradeoffs while expecting rankings to remain invariant across all environments and scales. A rigorous metric protocol must balance:

1. **Substrate independence**: Architectural scores do not fluctuate purely due to process advantages when mapped to FPGA vs. ASIC.
2. **Scale fairness**: Architectures do not automatically score higher simply by scaling up hardware; evaluations must compare designs at identical budget tiers before aggregating across scales.
3. **Environmental coverage**: Aggregate across standardized environmental domains while disclosing rank inversions.
4. **Conditional consistency**: Explicitly bound workload scopes and analytical assumptions, without claiming unconstrained universal truth.

#### 4.10.2 Evaluation Target: Architecture Families Rather Than Specific Boards

Define an architectural family $\mathcal{A}$ encompassing: ISA semantics, execution organization, dataflow patterns, memory/interconnect topologies, resource arbitration, and an explicit parameterized scaling rule. Let $x_{\mathcal{A}}(b)$ denote the concrete configuration under budget tier $b$.

- $x_{\mathcal{A}}(b)$ is selected using development/validation workloads and frozen prior to final evaluation.
- For each budget tier $b$, a single static hardware configuration must execute the entire benchmark suite; per-operator hardware redesigns are prohibited.
- In the primary track, $x_{\mathcal{A}}(b)$ remains static across environmental conditions $e$; environmental sensitivity is an evaluated property.
- Adaptive architectural families that reconfigure across environments must be reported in a separate `adaptive-family` track.
- Compilers may generate legal schedules for given workloads, but optimization search budgets and algorithms must be normalized.

Because empirical studies evaluate $\text{AQ}(\mathcal{A}, \text{compiler})$, claiming unconstrained optimal architectural potential independent of compilation quality is impossible. Auxiliary metrics such as $\text{AQ}_{\text{best-found}}$ (under standardized mapping search budgets) and small-scale oracle comparisons should be used to estimate compiler mapping overheads.

#### 4.10.3 Unified Abstract Resource Ledger

Adopt an "organization vs. realization" separation. The organizational ledger specifies:

| Resource | Abstract Representation | Costs That Must Be Accounted For |
|---|---|---|
| Compute | Arithmetic/logic primitives and service rates per dtype | Distinct SIMD/MMA/SFU costs; instructions cannot be treated as uniformly priced |
| Storage | Bits, banks, widths, capacities, read/write ports | Tags, ECC (if present), replication, multi-porting, and accumulator overheads |
| Interconnect | Links, channel widths, hop counts, topologies, queues | Routers/crossbars, buffers, arbitration, and distance-dependent latency assumptions |
| Control | Contexts, issue logic, scoreboards, decoders, schedulers | Control overhead is non-zero; area and power differences between lightweight and complex control must be preserved |
| Concurrency | MSHRs, target entries, in-flight transactions, flow credits | State storage, comparator logic, selection logic, and return routing |
| Timing | Standardized primitive latencies and pipeline rules | Cannot bypass timing constraints by declaring deep combinational logic as "single-cycle" |

Cost models rely on an open, canonical primitive library specifying standardized arithmetic costs, storage/port functions, and interconnect rules. This represents an **evaluation convention** rather than an immutable physical constant, and does not claim exact correspondence with arbitrary ASIC cell libraries or FPGA LUT counts.

Resource constraints are expressed as:

\[
K_\theta(x_{\mathcal{A}}(b)) \le B_b
\]

where $K$ represents multi-dimensional resource costs and $\theta$ specifies the cost model parameter version. Initial versions enforce hard caps across logic, storage, and communication dimensions, alongside an aggregate canonical cost ceiling to prevent substituting uncounted control logic for computational throughput. Tradeoff rules among dimensions must be pre-registered.

Normalizing solely by raw MAC operations masks the silicon overhead of specialized matrix arrays; conversely, normalizing by FPGA DSP slices biases evaluations toward FPGA-specific mappings. Retaining primitive composition is essential to evaluating resource allocation strategies.

#### 4.10.4 Environmental and Scale Domains

Standardized environments $e$ define external DRAM bandwidth relative to compute throughput, DRAM latency relative to logic delays, concurrency limits, external memory capacity, and standardized interconnect cost assumptions.

These ratios are normalized against **common reference primitives and budgets**, rather than candidate-specific peaks, ensuring that changing internal hardware does not alter the evaluation baseline.

Pilot studies should explore 1×, 2×, 4×, and 8× budget points before freezing formal ranges. These represent budget multipliers within the resource ledger rather than core counts; candidate architectures may select their own core counts and memory balances within each tier.

Environments should sample low/medium/high bandwidth and low/high latency regimes. Workload weights $w_i$, scale weights $p_b$, and environment weights $q_e$ must be frozen prior to testing, with scale tiers weighted evenly in log-budget space.

**The primary track employs fixed-size workloads.** A secondary scaled-workload track scales problem sizes according to common rules to evaluate whether large architectures suffer purely from under-utilization; scaling rules apply uniformly across all candidates.

#### 4.10.5 Common Ideal Lower Bounds Independent of Candidate Architectures

For each tuple $(i, b, e)$, define a common lower bound:

\[
L^*_{i,b,e}=\operatorname{LB}\bigl(\text{semantic workload }i,
\text{canonical budget }B_b,\text{environment }e\bigr)
\]

$L^*$ **does not depend on candidate architecture $\mathcal{A}$**. It is derived from optimistic resource allocation and necessary service constraints under standard budget $B_b$:

- Assuming optimal data reuse and ideal arbitration, preserving only mandatory computation and minimal data movement;
- Allocating compute, storage, and communication resources under canonical cost constraints;
- Formulating relaxed LPs or analytical conditions to find the theoretical minimum execution time;
- Omitting unprovable bank conflicts or dynamic queuing penalties to ensure optimistic bounds;
- Restricting algorithmic families for sorting, sparse, and approximate computation to prevent invalid lower bounds.

Because $L^*$ represents an optimistic relaxation across all allowable architectures, physical designs are not expected to reach 100% of $L^*$. AQ measures how closely an architecture approaches common physical potential, rather than promising a score of 100.

If a workload category lacks a rigorous, informative $L^*$, it must be flagged as a methodological gap; it may be evaluated under empirical reference tracks, but unproven bounds must not be incorporated into formal AQ calculations.

#### 4.10.6 Recommended Aggregate AQ Formulation

Let $\tau_{\mathcal{A},i,b,e}$ denote the candidate's execution time in the canonical substrate across the region of interest (ROI). Let $w_i$, $p_b$, and $q_e$ denote normalized frozen weights.

\[
\eta^{common}_{\mathcal{A},i,b,e}=\frac{L^*_{i,b,e}}{\tau_{\mathcal{A},i,b,e}}
\]

The recommended aggregate Architecture Quality metric is:

\[
\boxed{
AQ(\mathcal{A})=100\left[
\sum_b p_b\sum_e q_e\sum_i w_i
\frac{\tau_{\mathcal{A},i,b,e}}{L^*_{i,b,e}}
\right]^{-1}
}
\]

This computes the **reciprocal of the weighted average slowdown relative to the common ideal lower bound across workloads, budget scales, and standardized environments**.

Assuming valid lower bounds and verified functional correctness, AQ lies in the interval $(0, 100]$. A score of 10 denotes a weighted 10× slowdown relative to the theoretical lower bound; it does not imply the design achieves 10% of a specific GPU, nor does it guarantee an achievable 10× optimization margin.

Key difference from $E$: Because $L^*$ is anchored to common resource budgets, candidates cannot inflate their scores by lowering their own internal peak specifications. Adding resources transitions the candidate to a higher budget tier $b$, where it is benchmarked against higher baselines.

Release gates follow Section 4.8: formal AQ certification requires all points in the registered domain to be functionally correct and fully evaluated. If a point is fundamentally unexecutable, its efficiency is treated as 0, yielding an aggregate AQ of 0 while reporting coverage diagnostics. Missing data due to incomplete measurements must be marked `NOT CERTIFIED`.

To distinguish simulation models from physical validation, explicitly tag scores as `AQ-model`, `AQ-RTL-canonical`, or `AQ-certified-domain`. Absent sufficient physical calibration, cross-deployment equivalence must not be asserted.

#### 4.10.7 Secondary Cross-Check: Relative Scoring Against Reference Families

While theoretical lower bounds remain under refinement, candidates may be compared against a frozen reference family $\mathcal{R}$:

\[
Q_{ref}(\mathcal{A})=100\exp\left[
\sum_{i,b,e}w_ip_bq_e\log
\frac{\tau_{\mathcal{R},i,b,e}}{\tau_{\mathcal{A},i,b,e}}
\right]
\]

The reference family scores 100; this metric has no upper bound of 100. It enables relative comparisons among concrete implementations without asserting theoretical optimality. Under fixed weights, the relative score ratio between two candidates is independent of reference selection, following [SPEC principles on weighted geometric means](https://www.spec.org/gwpg/gpc.static/geometric.html). If weights are regenerated upon switching references, rankings can shift; weights must therefore be frozen independently.

$Q_{ref}$ does not replace AQ: they evaluate implementation parity and theoretical potential, respectively. Discrepancies in ranking between AQ and $Q_{ref}$ require investigating lower-bound tightness and weighting schemes, rather than selectively publishing favorable results.

#### 4.10.8 Accounting for Hardware Overhead, Energy, and Timing

Architectural evaluations must account for control logic, register porting, and interconnect hardware; otherwise, designs with infinite crossbars and unlimited ports emerge as fictitious optima. Canonical delay models for $\tau$ must capture multiplexer depths, port arbitration, and pipeline stages, rejecting assumptions that all RTL cycles are of uniform physical duration.

Energy lacks an absolute substrate-independent unit. Initial evaluations establish canonical action energy accounts: multiplying primitive activity counts by standardized coefficients, accompanied by parameter sensitivity analyses. Energy may be introduced as a hard budget constraint in energy-constrained AQ evaluations, but arbitrary composite products of delay, energy, and area must be avoided.

Physical implementations must report mapping residuals: discrepancies between canonical predictions and FPGA/ASIC implementations attributed to clock distribution, logic packing, SRAM macros, and physical interfaces. If an architectural topology repeatedly fails to achieve timing closure across multiple substrates, the abstract model must be revised or the architecture judged unrealizable.

#### 4.10.9 Validating Substrate and Scale Invariance

The metric itself requires formal empirical validation:

1. Executing an identical architecture across varying FPGA clock frequencies must yield an invariant canonical AQ, while physical board execution times vary.
2. Porting an identical architecture across two FPGA families or an ASIC flow must yield consistent canonical AQ scores. Absent ASIC physical access, state compatibility rather than verified physical equivalence.
3. Increasing hardware resources moves the design to a higher budget tier $b$, preventing it from scoring within smaller budget classes.
4. Replicating non-interacting compute units and workloads without shared bottlenecks should maintain constant lower-bound efficiency; if multi-core interconnect overheads grow, AQ should reflect this scalability penalty.
5. Depressing internal peak throughputs, injecting redundant memory traffic, or synthesizing wide multi-cycle instructions must not improve scores.
6. Conduct sensitivity analyses perturbing cost parameters $\theta$, environment weights $q$, scale weights $p$, and workload weights $w$, reporting ranking stability and inversion regimes.
7. Benchmark intentionally degraded designs (excessive synchronization barriers, constrained NoC widths, reduced RF ports, underfed compute arrays); AQ must penalize these structural bottlenecks.
8. Verify that Pareto-optimal candidates under canonical models remain competitive in physical implementations.

Accompany AQ with `AQ-tail` (scoring the lowest quantile of environments and scales), alongside coverage and sensitivity bounds. The primary composite score remains AQ; composite numbers incorporating token costs or confidence bounds directly into AQ are prohibited.

#### 4.10.10 Alignment with the DevOps Mainline

The primary objective of cross-layer iteration is to **maximize held-out AQ under fixed design budgets**, while tracking concrete deployment performance $S$ to ensure designs do not overfit abstract metrics. Optimization agents must be decoupled from evaluators: agents cannot alter lower bounds, canonical costs, or environmental weights.

"Discovering equal AQ with fewer tokens" demonstrates engineering efficiency; "achieving higher AQ under identical token budgets" demonstrates co-design efficacy. Physical FPGA measurements and NVIDIA baselines provide external validation for AQ, rather than defining the scoring metric itself.

**AQ is an empirical research protocol proposed herein, not an established standard from prior literature.** Validating its lower bounds, canonical cost models, cross-substrate calibration, and ranking stability is a primary goal of R38.

## 5. From 45 Operators to Thousands of Meaningful Workload Points

### 5.1 Workload Scale and Tiering

The initial benchmark suite comprises 45 operators × ~64 semantic instances per operator, totaling **~2,880 points**. This represents a sampling budget: small operators must not be duplicated to hit 64 points; complex operators may sample more densely, using weighting to prevent sample counts from distorting aggregate scores.

Optimization levels (O0/O1/O2), hardware configurations, and experimental repetitions represent orthogonal experimental dimensions rather than distinct workloads. Evaluating 2,880 points × 3 optimization levels × 3 hardware configurations yields 25,920 combinations; exhaustive RTL simulation is cost-prohibitive, necessitating tiered evaluation.

| Suite Tier | Proposed Scale | Objective |
|---|---:|---|
| R37-contract | 149 stages × 3 levels | Baseline backward-compatibility gate |
| Smoke | 6–8 points across key bottlenecks | Rapid turnaround per cross-layer iteration |
| Correctness-expanded | Thousands of shapes and boundary conditions | Functional and numerical boundary verification via fast behavioral models |
| Perf-development | ~60% of shape families | Compiler heuristics tuning, cost model training |
| Perf-validation | ~20% of shape families | Hyperparameter selection, candidate pruning |
| Perf-held-out | ~20% of shape families | Final frozen evaluation; no per-point tuning feedback |
| RTL/FPGA anchors | 100–300 points initially | Physical calibration and ground-truth validation |

Final evaluation tables must tag every entry as `measured-RTL`, `measured-FPGA`, or `predicted`. Analytical predictions may span the full space, but must never be aggregated into physical measurement totals. Primary tables prioritize fully evaluated frozen subsets, reporting coverage relative to the complete suite.

### 5.2 Complete 45-Operator Planning Inventory

| Category | Operator (Retaining Manifest Identifier) | Key Evaluation Dimensions |
|---|---|---|
| Basic Elementwise (8) | `vector_add`, `mul2`, `relu`, `leaky_relu`, `sigmoid`, `fused_activation`, `swiglu`, `dropout` | Tensor length $N$, alignment/tails, strides, SFU utilization, RNG semantics and seeds |
| Quantization & Positioning (4) | `quantize_global`, `dequantize_rowwise`, `weight_dequant`, `rope` | Group sizes, quantization scales, saturation, dimensionality, offsets, formats |
| Reduction / Norm / Loss (10) | `softmax`, `layernorm`, `rmsnorm`, `l2_norm`, `mean_reduction`, `argmax`, `cross_entropy`, `kl_divergence`, `batch_normalization`, `moe_topk_gating` | Rows/widths, long reductions, numerical epsilon, ties handling, extreme values, top-$k$ |
| Atomic Statistics (1) | `histogramming` | Bin counts, uniform/skewed/identical distributions, atomic contention |
| Matrix (4) | `matmul_fp32_fp16_fp8`, `matmul_int8`, `batched_matmul`, `streamk_matmul` | $M/N/K$, batch size, non-square tiles, tails, large $K$, precision, grid imbalance |
| Attention (4) | `flash_attention`, `flash_decode`, `block_sparse_attention`, `linear_self_attention` | Batch/heads/seqlen/head-dim, causality, KV length, sparsity patterns, prefill vs. decode |
| Conv / Stencil (6) | `1d_conv`, `2d_conv`, `3d_conv`, `2d_max_pooling`, `gaussian_blur`, `jacobi_stencil_2d` | Channel counts, kernel dimensions, spatial resolution, padding boundaries, working set |
| Data Layout (5) | `matrix_copy`, `matrix_transpose`, `interleave`, `reverse_array`, `destindex` | Geometries, strides, alignment, spatial locality, bank conflicts |
| Sorting & Selection (3) | `bitonic_sort`, `radix_sort`, `top_k_selection` | Problem size $N$, key distributions, duplicate keys, $K/N$ ratios, stage counts, stability |

Total: 45 operators. Specific supported parameters follow original operator specifications; table dimensions do not introduce non-existent semantics.

### 5.3 Workload Generation Rules

Evaluation spaces must systematically cover:

- Exact tile multiples, alongside `tile - 1` and `tile + 1` boundaries;
- Startup-dominated regimes, compute-bound saturation, RF capacity limits, NUCA boundaries, and DRAM-bound regimes;
- Non-square matrix geometries: tall-and-skinny, wide-and-short, batch=1, multi-batch, and large-$K$;
- Contiguous layouts, supported strided tensors, transposed layouts, and irregular access patterns;
- Uniform random data, zeros, repeated values, high-contention distributions, and extreme ranges; NaN/Inf handling follows upstream semantics;
- Attention workloads spanning short-batch long-sequence prefill and long-KV single-token decode.

Training and validation splits must be partitioned by shape families (e.g., holding out entire ranges of $K$, specific head dimensions, or specific cache working-set ratios); random splitting across adjacent shapes risks data leakage. A held-out hardware configuration must be maintained to verify the cost model's generalization across unseen microarchitectures.

### 5.4 Correctness and Precision Verification Protocol

Every evaluated point must log: reference implementation version, input seeds/hashes, data types, accumulation precisions, rounding modes, rtol/atol thresholds, relative $L_2$ error, maximum absolute error, and out-of-bounds guard checks. Error tolerances are fixed upfront by mathematical semantics and reference distributions; they must not be relaxed upon candidate failures.

For INT8 large-$K$ operations, integer accumulation boundaries and floating-point fallback thresholds must be verified. Dropout and RNG benchmarks require statistical verification if deterministic seeds are not mandated. Argmax, top-$k$, and sorting operations require explicit rules for tie-breaking and numerical stability.

For asynchronous and atomic primitives, test suites must inject randomized delays, out-of-order responses, network congestion, credit exhaustion, and timeout watchdogs across multiple seeds. Correctness gates apply equally across O0, O1, and O2; primary performance evaluations pre-register a deployment level (recommended O2), while O0 and O1 isolate compiler optimization effects.

## 6. Cost Model: Prediction, Explanation, and Experiment Selection

### 6.1 Strict Separation from ArchBench Lower Bounds

ArchBench theoretical lower bounds provide an auditable scoring standard; the cost model is a predictive tool subject to estimation error. The cost model's internal predictions must not serve as both optimization target and final evaluation proof.

The cost model employs a three-tier architecture:

1. **Static Analysis Layer**: Extracts operation counts, instruction mixes, live ranges, RF/accumulator allocations, register spills, access patterns, tile padding, and dependency DAGs directly from IR or machine code.
2. **Microarchitectural Analysis Layer**: Models execution units, bank/port availability, NoC bandwidth, MSHRs, queues, barriers, role assignments, and grid scheduling to estimate overlap and identify primary bottlenecks.
3. **Residual and Uncertainty Layer**: Calibrates analytical estimates against measured hardware cycles, outputting prediction intervals, error attributions, and candidate acquisition values for active learning.

The initial implementation prioritizes an explainable analytical model calibrated with lookup tables. It adapts mapping principles from [Timeloop](https://timeloop.csail.mit.edu/) and analytical techniques from [MAESTRO](https://maestro.ece.gatech.edu/), while validating dynamic cache behavior and network backpressure against empirical TNCA data.

### 6.2 Rejecting Additive Stall Models

Coarse execution models assume:

\[
\hat T=T_{launch}+T_{scheduled\ DAG}+T_{drain}+\epsilon
\]

Here, $T_{scheduled\ DAG}$ must be derived from resource-constrained dependency scheduling or calibrated models, rather than naively summing compute, memory, and stall counters. Data movement and computation overlap dynamically, and multiple architectural stalls can occur in the same cycle.

Cost models must be constructed and calibrated for single-core execution before extending to multi-core scheduling and interconnect traffic. Models trained on serialized grid dispatches must not be asserted to predict parallel full-chip behavior.

### 6.3 Required Modeling Dimensions

| Subsystem | Input Parameters | Predicted / Explained Behavior |
|---|---|---|
| Compute | Data type, tile dimensions, instruction mix, tail elements | Effective throughput, padding overhead, MMA/SIMD/SFU balance |
| RF | Layouts, banking, port counts, live ranges, window configurations | Port conflicts, operand staging stalls, register spills, port area costs |
| Cache / NUCA | Reuse distance, home slice mapping, capacity, banking, MSHRs | Working-set thrashing, hit/miss service times, coalescing, slice hot-spotting |
| NoC | Request/response routing, injection rates, bisection cuts | Remote access overheads, network backpressure, link saturation |
| Async | Producer/consumer splits, credit tokens, buffer stages | Pipeline overlap efficiency, critical synchronization delays |
| Runtime | Grid geometry, kernel launch, descriptor updates, code/arg staging | Small-kernel launch overheads vs. batch pipelining gains |
| Physical | Architecture parameters, synthesis records, P&R samples | Implementation feasibility, resource bounds, $F_{max}$ risks; no power/area claims without physical calibration |

### 6.4 Model Acceptance Criteria

Baselines must be measured before freezing target criteria:

- Target median relative error $\le 15\%$ and P90 error $\le 30\%$ on held-out test points; report absolute errors on short kernels to prevent percentage distortion.
- Target Spearman rank correlation $\ge 0.8$; quantify real performance regret among the top-$k$ predicted candidates.
- Demonstrate that model-guided search discovers superior designs compared to random and pure analytical search under identical measurement budgets.
- Evaluate unseen shapes and unseen hardware configurations independently; do not aggregate them into a single error metric.
- Track 90% prediction interval coverage, overconfidence rates, and out-of-distribution (OOD) rejection frequencies.
- Explicitly log "false-positive" candidates (predicted fast, measured slow) and diagnose their structural causes.

These represent proposed evaluation criteria; if pilot studies indicate target thresholds are inappropriate, they may be revised prior to formal experiments, but must not be altered post-hoc.

### 6.5 Calibration and Online Learning Boundaries

All cost model training data, counter calibration runs, training/inference time, and simulation costs must be accounted for. Agent evaluations report two protocols:

- Frozen Model: All runs utilize an identical pre-trained model to isolate information gain.
- Online Updating: Models absorb new measurements during optimization, with all training overhead charged against the run's budget; reading data across experimental groups or from test splits is prohibited.

Report one-time model construction costs alongside amortization curves over $N$ subsequent design iterations.

## 7. Profiler: From Counters to Cross-Layer Attribution

### 7.1 Preserving Existing Counter Contracts

Extend [r3_counters.json](../spec/r3_counters.json) event definitions, bitwidths, reset rules, and overflow handling. New counters must specify source modules, trigger logic, measurement units, and ROI boundaries.

Note that `perf_cycles` tracks cycles where any role context is active, which may not encompass the complete kernel makespan including final memory drain. Implement a standardized metric tracking `launch_accepted → final_store_visible & drained`, cross-verifying it against runtime completion timestamps.

### 7.2 Initial Required Profiler Metrics

| Domain | Metric | Diagnostic Purpose |
|---|---|---|
| Timing | Host enqueue, device accept, first issue, last store, completion drain | Identify whether overheads stem from launch, execution, or drain |
| Compute | Instructions issued/completed, active vector lanes, MMA busy, SFU busy | Diagnose compute stalls vs. tail-element under-utilization |
| RF | Read/write grants, bank conflicts, operand staging stalls, spill bytes | Determine whether to modify layouts, porting, or tile dimensions |
| Memory | Request/response bytes, cache hits/misses, MSHR/target occupancy, writebacks | Identify data reuse efficiency vs. memory-level concurrency limits |
| NoC | Link byte throughput, queue occupancy, backpressure cycles, hop distance | Detect remote slice hotspots and bisection bottlenecks |
| Async | In-flight DMA requests, window lifecycles, barrier stalls | Diagnose producer/consumer synchronization mismatch vs. resource starvation |
| Grid | Active concurrent programs, per-core work distribution, tail latency | Quantify load imbalance and scheduling overheads |

Only counters physically implemented in RTL and exposed to the public backend may be tagged `available`; others must be labeled `planned` or `unavailable`. Testbench memory events must not be reported as physical DRAM controller statistics.

### 7.3 Trace Correlation

Construct the following mapping:

```text
operator/case → source span → TTIR/target IR op
             → machine PC range → instruction/event → RTL resource
```

Static debug mappings are stored in AOT metadata; dynamic traces record only PC, timestamp, event ID, and transaction tags to minimize overhead. Provide summary, sampled, and detailed trace collection modes.

The profiler outputs a unified structured schema for both humans and agents. Natural language diagnostics must reference concrete metrics; observing high hazard stalls must not automatically trigger assertions that adding RF ports is the optimal solution. Compiler layout ablations or hardware feature toggles are required to establish causality.

### 7.4 Conservation Laws and Measurement Overhead

- The sum of mutually exclusive stall cycles must match total stall time; overlapping stalls must not be added together.
- Conservation laws must hold between issued, completed, flushed, and faulted instructions.
- Data volumes across DRAM, NUCA, and NoC must use consistent accounting regarding metadata, retries, and writebacks.
- Cycles across multiple cores must not be summed to represent wall-clock elapsed time.
- Cache and queue warm-up states must be defined prior to ROI start; ROI completion must require pipeline drain.
- Verify that profiler instrumentation does not alter functional behavior, cycle counts, or timing closure.
- GPU profiling and timing runs must be executed independently to prevent Nsight instrumentation overhead from distorting runtime measurements.

## 8. Three Priority Cross-Layer Optimization Case Studies

These serve as **testable hypotheses** rather than mandatory architectural additions. Profiler data must substantiate performance bottlenecks before implementing minimal modifications. Each case study requires single-layer, combined, and resource-normalized ablations.

### 8.1 Case A: RF Bank Organization, Windowing, and Matrix Operand Delivery

- Hypothesis: In architectures lacking software-managed scratchpads, RF port contention and layout conversions constrain matrix execution throughput.
- Compiler: Bank-aware register layout, fragment reuse, live-range minimization, optimal tile/stage selection.
- ISA: Introduce versioned window layout or operand reuse specifiers if necessary; prioritize existing semantic encodings.
- Microarchitecture: Operand collector arbitration, operand staging latches, bank/port structural redesign.
- Metrics: Sustained MMA throughput, RF conflicts, memory traffic, register spills, silicon area, $F_{max}$.
- Ablations: Layout-only, hardware-only, combined compiler+hardware, and full ISA-extended co-design.
- Termination: Halt if cycle improvements fail to offset frequency/area penalties, or if compiler tuning achieves equal gains.

### 8.2 Case B: Asynchronous Descriptor → RF → Compute Pipeline

- Hypothesis: Single-role serialized execution, descriptor initialization, and synchronization barriers introduce substantial pipeline bubbles.
- Compiler: Producer/consumer role partitioning, pipeline stage selection, asynchronous ownership verification.
- ISA: Codify explicit completion events, window reuse preconditions, and resource admission rules.
- Microarchitecture: DMA concurrency tracking, return buffer reservation, role-context scheduling, completion signaling.
- Metrics: Compute/transfer overlap efficiency, in-flight transaction distributions, barrier wait cycles, drain latency.
- Stress Testing: High-latency out-of-order memory, minimal queue depths, slow consumer execution, transaction cancellations.
- Safety Invariants: Request throttling must never impede in-flight completions, consumer processing, or resource release; infinite buffering cannot be relied upon to prevent deadlock.

### 8.3 Case C: Multi-Core Grid Scheduling, NUCA Locality, and Request Admission

- Precondition: Bridge the public runtime's serialized grid dispatch with on-device hardware schedulers; single-core data cannot be extrapolated to 16 tiles.
- Compiler: Tile working-set analysis, optional home-slice and cache-reuse hints.
- ISA / Runtime: Optional locality hints, grid capabilities, ABI version contracts.
- Microarchitecture: On-device program dispatch, NUCA home-aware scheduling, MSHR admission control.
- Metrics: Full-grid makespan, per-core idle cycles, inter-tile NoC traffic, cache hot-spotting, scaling efficiency.
- Baselines: Round-robin dispatch, hint-free baseline, static locality scheduling, feedback-driven admission control.
- Boundary with Prior Art: Explicitly distinguish inherited mechanisms from DCO/LLaMCAT; R38 contributions must stem from executable cross-layer co-evolution and novel mechanisms.

### 8.4 Extension Candidates and Prioritization

Candidates include SFU/reduction expansion, native integer matrix arithmetic, instruction fetch bandwidth, and small-kernel fast paths. Priority is given to mechanisms addressing primary bottlenecks across multiple operators with modest verification overhead, avoiding simultaneous architectural changes that prevent causal attribution.

At least one negative or neutral case study must be documented, analyzing where the cost model failed, how the profiler diagnosed the issue, and the rollback cost incurred. Documenting negative results provides critical evidence for the value of closed-loop iteration.

## 9. Controlled Experiments Proving "Co-Iteration" Causality

### 9.1 Controlled Experimental Setup

All groups share an identical starting state $V_0$, common workload development splits, identical hardware budget limits, identical compute/tooling resources, identical correctness gates, and identical held-out evaluators. Compiler-only baselines receive equivalent profiler and cost model feedback to ensure fair comparison.

| Group | Modifiable Layers | Objective |
|---|---|---|
| B0 | Baseline R37, standardized deployment optimization level | Historical baseline |
| B1 | Compiler-only, fixed ISA and RTL | Quantify gains achievable via compilation alone |
| B2 | Hardware-only, fixed ISA and compilation strategy | Quantify gains achievable via microarchitectural tuning alone |
| B3 | Compiler + Hardware, fixed ISA | Quantify co-design potential without ISA modifications |
| B4 | Staged sequential (Compiler → ISA → Hardware), integration at phase boundaries | Baseline for long-feedback, staged co-design |
| B5 | Alternating optimization within common search space, periodic integration | Control for expanded design space size |
| B6 | Three-layer rapid-feedback co-iteration | Proposed methodology |

Additional baselines may include model-driven DSE and human-engineered designs. B4, B5, and B6 should share identical candidate modifications, isolating feedback frequency and iteration granularity as the evaluated variables.

### 9.2 Two Complementary Budget Frameworks

1. **Iso-Wall-Clock / Tooling Budget**: Evaluates whether rapid feedback discovers superior designs within identical engineering time, accounting for compilation, simulation, queuing, and debugging delays.
2. **Iso-Evaluated-Candidate Budget**: Normalizes for search trial counts, assessing the decision-making quality of co-iteration.

Track human engineering hours or agent token usage, without assuming direct equivalence between the two. Compute infrastructure, GPU/FPGA allocations, and cache policies must be kept identical across runs, reporting cold-cache and warm-cache costs separately.

Because candidate counts alone do not fully reflect engineering complexity, report patch diff sizes, touched modules, resource utilization deltas, and build times, avoiding lines of code as a quality metric.

### 9.3 Results and Interaction Effects

Primary metric: Optimal feasible AQ (Section 4.10) discovered under fixed design budgets, accompanied by deployment speedup $S$, internal efficiency $E$, resource utilization, correctness, and iteration cost. Environmental domains and resource budgets are held static across groups. Present anytime optimization trajectories and Pareto frontiers, computing hypervolume against a reference point frozen prior to testing.

For modular features, evaluate full $2^3$ factorial combinations across compiler, ISA, and hardware. Invalid combinations must be marked unexecutable rather than approximated. If new ISA instructions cannot run on legacy hardware, provide functionally equivalent software implementations as baselines and document their overheads.

Cross-layer interactions should be quantified via log-speedup interaction terms rather than summing individual speedups. Compelling evidence takes the form: "hardware tuning alone yields minimal gains, compiler tuning alone is constrained, but joint co-design resolves the operand bottleneck", supported by counter traces.

### 9.4 Statistical Rigor and Generalization

- Pilot studies should run at least 5 independent search trials per group; final sample sizes are determined by variance and target effect sizes.
- Compare matched trials sharing identical starting points and budgets; quantify agent stochasticity separately from measurement noise.
- Aggregate workload metrics using stratified bootstrapping across operator and shape families; adjacent shapes cannot be treated as statistically independent architectures.
- Report confidence intervals, effect sizes, failure counts, and all trial runs, avoiding selective reporting of optimal seeds.
- Validate findings across small, medium, and large configurations on held-out shape families; evaluate a secondary hardware configuration if time permits.
- If evaluations are limited to a single architecture family and three case studies, conclusions must be bounded accordingly.

## 10. Agent Experiments: Evaluating Whether Model and Profiler Benefits Justify Costs

### 10.1 2×2 Factorial Experimental Design

| Group | Cost Model | Profiler | Information Visible to Agent |
|---|---|---|---|
| A00 | None | None | Source code, contracts, compiler diagnostics, correctness, end-to-end latency |
| A10 | Enabled | None | Above + static predictions, uncertainty bounds, resource estimates |
| A01 | None | Enabled | Above + counter/trace data with source code mappings |
| A11 | Enabled | Enabled | Both information streams exposed via structured schemas |

A00 retains end-to-end latency and correctness feedback, providing a necessary baseline for optimization. "Profiler disabled" specifically denotes the absence of fine-grained architectural counters and traces, not the omission of runtime feedback.

Prevent data contamination: A00 must not access profiler artifacts in sibling directories; A10 must not inspect execution counters; all groups operate under identical measurement budgets, ensuring A11 does not receive uncounted simulation runs. Test oracles and held-out suites remain strictly read-only outside the agent's workspace.

### 10.2 Experimental Task Suite

Curate 8–12 representative optimization tasks: resolving RF contention, mitigating SFU bottlenecks, expanding memory concurrency, optimizing descriptor overheads, handling long reductions, balancing grid execution, etc. Tasks provide identical starting codebases and resource targets, stratified across compiler-only, hardware-only, and cross-layer categories.

Primary experiments center on cross-layer tasks modifying compiler, ISA, and microarchitecture. Pure scheduling tasks should be evaluated separately against search algorithms like Ansor, avoiding mixed rankings between agent RTL modifications and automated tile-size autotuning.

Prompts and logs must not contain hints derived from known human patches. Each trial executes in an isolated workspace; complete, sanitized interaction logs must be preserved for publication.

### 10.3 Tracking and Cost Accounting

Lock model checkpoints (or record API versions and dates), inference parameters, context windows, prompts, tool versions, retry policies, and seeds. If models are non-deterministic, document limitations and interleave runs across experimental groups.

Log the following parameters:
- Prompt, completion, and cached token counts; log reasoning tokens where reported by the API.
- Actual API expenses under billing rules at the time of execution.
- Tool runtimes (compiler, RTL simulation, synthesis, FPGA builds), invocation counts, tool errors, and timeouts.
- Model training/calibration overhead, profiling overhead, and human intervention minutes.
- Evaluated performance, resource utilization, correctness, and reproduction rates of the best candidates.

Total cost is formalized as $\text{API} + \text{compute rental} + \text{amortized calibration}$; human engineering hours must be listed separately.

### 10.4 Defining Token Efficiency

Report the following metrics:

1. Peak feasible speedup achieved within fixed token budgets.
2. Tokens required to achieve pre-registered improvement thresholds (treating unsuccessful trials as censored data).
3. Area under the improvement-vs-token curve, integrated across a standardized token interval.
4. Token and monetary cost incurred per valid improvement candidate.
5. Achieved design quality under identical wall-clock budgets, ensuring token-efficient runs that trigger excessive synthesis runs are not mislabeled as superior.

When integrating over $\log(S_{best})$, document that curves update only upon discovering verified improvements, with $\log(1) = 0$ and failed trials contributing zero gain. Calculating token metrics solely over successful trials introduces survivor bias; success rates and censored runs must be reported.

Employ factorial analysis to estimate main effects and interaction terms for cost models and profilers. Rich feedback can consume large context windows or flawed models can mislead search; negative interactions constitute valid research findings.

### 10.5 Additional Baselines and Stopping Rules

Compare against random search, static heuristics, analytical model-directed search, and Bayesian/evolutionary search where applicable. Exposing the identical cost model to non-LLM search baselines clarifies whether performance gains stem from the predictive model itself or from cross-layer reasoning by the Agent.

Each trial enforces joint limits on tokens, wall-clock time, and tool invocations, halting whenever the first limit is reached. Agents may debug errors within budget, but cannot modify evaluation harnesses, loosen numerical tolerances, drop slow test cases, or introduce hidden CPU fallbacks. Final candidates must be re-evaluated by an independent evaluation harness.

## 11. NVIDIA Comparison: Disaggregating "Beating NV" into Testable Claims

### 11.1 Four Distinct Levels of Inquiry

| Question | Defensible Conclusion | Invalid Over-Generalization |
|---|---|---|
| Does TNCA achieve higher efficiency relative to its own roofline? | Better utilizes available resource upper bounds on evaluated workloads | TNCA is faster or more area-efficient than modern GPUs |
| Does TNCA achieve superior absolute latency/throughput on specific points? | Wins on specific hardware configurations, precisions, and scenarios | Broad superiority across all 45 operators or across all problem scales |
| Does TNCA exhibit lower board-level energy consumption? | Consumes less total energy per task under identical physical measurement setups | TNCA ASIC core power is superior to NVIDIA silicon dies |
| Is TNCA architecturally superior under normalized technological assumptions? | Supported by calibrated models, implementation constraints, and sensitivity analyses | FPGA LUT counts convert directly into GPU die area ($\text{mm}^2$) |

The absolute throughput gap between an FPGA prototype and a modern datacenter GPU can be substantial. The paper's core contributions must not be predicated on outperforming NVIDIA GPUs across the board. Analysis should examine latency-sensitive, bandwidth-constrained, or irregular memory regimes, reporting the entire pre-registered workload distribution rather than selectively highlighting winning cases.

### 11.2 Baseline Matrix

1. Native GPU backend executing identical unmodified Triton kernels, given reasonable autotuning budgets.
2. PyTorch eager mode, serving as a framework baseline rather than a competitive hardware reference.
3. PyTorch compiled mode (`torch.compile`), with frozen versions and captured graph boundaries.
4. Vendor-tuned libraries (cuBLASLt, cuDNN, CUTLASS) where applicable, logging exact algorithmic and precision configurations.
5. TNCA R37 baseline, TNCA R38 compiler-only, and final TNCA co-designed architecture.

GPU hardware must reflect accessible physical devices; do not project hypothetical H100/B200 configurations. Benchmark at least one datacenter GPU; if a second GPU is available, verify whether conclusions hold across GPU architectural generations. Unsupported libraries must be marked N/A rather than substituted with mismatched baselines.

### 11.3 Unified Experimental Protocol

- Align mathematical semantics, inputs, outputs, accumulation precisions, data formats (TF32/FP16/FP8), sparsity, and approximations.
- Report device-resident kernel execution time separately from end-to-end host-device-host latency.
- Measure latency and throughput under distinct execution regimes; do not compare batched TNCA throughput against single-sample GPU latency.
- Standardize warm-up iterations, trial counts, CUDA streams, synchronization points, CUDA Graph usage, and memory allocator states.
- Report medians, tail distributions, and measurement variance; for microsecond-scale kernels, evaluate repeated invocations to isolate launch overheads.
- Document clock frequencies, power limits, temperatures, driver versions, and toolchains; record dynamic operating clocks rather than nominal specs.
- Power measurements must run sufficiently long to establish steady-state power, specifying idle subtraction rules; TDP must not be cited as measured power.
- Profiling runs and timing runs must be executed independently to prevent profiling instrumentation from distorting execution timing.

### 11.4 FPGA vs. ASIC Projections

The main text may present three distinct tiers of evidence, but must not conflate them:

- **RTL Target Cycles**: Simulated cycle counts on defined memory and network models, distinct from host simulation time.
- **FPGA Physical Measurements**: Measured clock frequencies, memory controller latencies, DMA transfers, and board power.
- **ASIC Projections (Optional)**: Permissible only with accessible standard-cell libraries, SRAM compilers, static timing, and power analysis; must disclose all modeling assumptions and uncertainties.

Generic Yosys gate and depth metrics support structural trends only. R38 must not convert structural proxies into speculative TOPS/W metrics for GPU comparisons.

## 12. PyTorch and FPGA LLM: System-Level Validation of the Closed Loop

### 12.1 Staged Two-Phase Integration Roadmap

**Phase P1: Graph-Level Integration (Initial Target).** Export inference subgraphs via `torch.export` or registered compiler backends, lower them to Triton/TNCA AOT, and execute them via `Session`. This phase permits host-managed tensor wrappers, but must account for memory transfer boundaries without claiming native `device='tnca'` integration.

**Phase P2: Native Device Backend (Subsequent Target).** Implement device allocators, memory copies, device guards, streams/events, dispatcher registrations, and tensor aliasing following the [PyTorch Accelerator Integration Guide](https://docs.pytorch.org/docs/main/accelerator/index.html) and [PrivateUse1](https://docs.pytorch.org/tutorials/advanced/privateuseone.html). Focus strictly on inference; autograd and distributed execution remain outside R38's required scope.

Lock PyTorch release versions during implementation, avoiding unstable APIs from unpinned branches.

### 12.2 Public AOT as the Sole Execution Mainline

All profilers, benchmarks, and PyTorch integrations must execute through `compile_aot → artifact → Session`, avoiding internal backdoor code-generation paths. New parameters must be managed via explicit ABI/capability interfaces, preserving standard execution workflows demonstrated in tutorials.

Retain Tutorials 08 and 09 as entry points, planning four new tutorial documents:

- `10_archbench_profile.md`: Compiling GEMM/softmax and inspecting correctness, cycle, and bottleneck reports.
- `11_cross_layer_iteration.md`: Stepping through baseline, optimization, and rollback of a toggleable architectural feature.
- `12_pytorch_inference.md`: Executing small computational graphs on simulation and board backends, documenting fallbacks.
- `13_fpga_decoder.md`: Running an isolated decoder layer progressing to a complete small LLM prefill/decode pipeline.

These represent planned additions rather than currently implemented files.

### 12.3 Memory Footprint and Model Scale Calculations

Model weight memory footprint:

\[
M_{weights}=N_{params}\cdot bits/8 + M_{scales}+M_{metadata}
\]

Standard KV cache footprint (excluding alignment and management overhead):

\[
M_{KV}=2\cdot L\cdot B\cdot S\cdot H_{kv}\cdot d\cdot bytes_{element}
\]

Total memory allocations must also accommodate activations, workspace memory, compiled code, DMA descriptors, and fragmentation. Target models must fit within physical board memory; assess the feasibility of 0.1B–1B parameter decoder models first. Model selection, licenses, and weight hashes must be finalized based on board capacity and numerical precision support. Do not attempt to run full models within the default 512 KiB simulation memory.

If quantization is employed, evaluate accuracy and numerical stability systematically; memory reductions must not alter underlying numerical contracts.

### 12.4 Staged Acceptance Criteria

| Stage | Milestone Scope | Required Evidence |
|---|---|---|
| P0 | Single ATen / custom operator → AOT | Numerical correctness, shape/stride support, hardware execution |
| P1 | Small subgraph (GEMM + Norm + Activation) | Elimination of intermediate host roundtrips, execution boundaries |
| P2 | Single Transformer decoder block | Attention, RoPE, KV cache updates verified against PyTorch reference |
| P3 | Full small-model prefill | Supported prompt lengths, logit error bounds, peak memory profile |
| P4 | Multi-step autoregressive decode | KV cache reuse, generation fidelity, token/s throughput |
| P5 | Reproducible FPGA board demonstration | Bitstream hash, execution command, machine logs, screenshots, power/latency |

Host-side tokenization, print functions, and sampling may run on the CPU; all model operator fallbacks to CPU must be logged. Demonstrating that "the model runs on hardware" requires that the designated computational subgraph executes without implicit CPU fallbacks. Unsupported operators must raise explicit errors.

Measure Time-To-First-Token (TTFT), prefill throughput (tokens/s), decode throughput (tokens/s), per-token latency, scaling across KV sequence lengths, and peak memory consumption. Generated text serves only as a qualitative demonstration; logit errors and perplexity metrics against reference outputs must be provided to verify numerical correctness.

Appendix screenshots must display model ID, configuration, board details, bitstream hash, execution command, and measured throughput; machine-readable log files must accompany screenshots.

## 13. Engineering Organization, Interfaces, and Experimental Data

### 13.1 Proposed Directory Layout (Planned for Implementation)

```text
spec/r38/
  workload.schema.json       # Semantic points and precision
  architecture.schema.json   # Configuration, resources, and implementation versions
  experiment.schema.json     # Complete experiment manifest
  counter.schema.json        # Inherits r3 counter contract
  budgets.json               # Fixed resource and search budgets
  suites/                    # Splits, weights, and provenance

tools/r38/
  validate_manifest.py
  run.py                     # Via public AOT/runtime only
  calibrate.py
  score.py
  compare.py
  agent_experiment.py
  reproduce.py

python/tnca/                 # Extends existing public API, no duplicate runtime
compiler/                    # Real passes/lowering and provenance
rtl/                         # Switchable, verifiable architectural changes
sim/build/r38/               # Generated reports, traces, artifacts (ignored by Git)
```

Target Makefile commands include `make r38-smoke / r38-correctness / r38-calibrate / r38-bench / r38-report`; these represent planned targets rather than existing commands.

### 13.2 Single Measurement Record Schema

```yaml
# Field design example; null indicates unknown, never zero
schema: archbench-measurement/1
candidate_hash: <hash>
parent_hash: <hash>
workload_id: <operator-shape-dtype-layout-distribution>
source_hash: <hash>
input_hash: <hash>
compiler: {hash: <hash>, opt_level: 2, passes_hash: <hash>}
isa_hash: <hash>
rtl_hash: <hash>
artifact_hash: <hash>
profile: <exact-profile>
evidence_level: measured-rtl
correctness: {status: pass, oracle_hash: <hash>, max_abs_error: null}
time:
  roi: launch_accept_to_final_store_and_drain
  device_cycles: null
  achieved_clock_hz: null
  device_seconds: null
  host_seconds: null
traffic: {required_lower_bound_bytes: null, measured_bytes: null}
bounds: {version: <hash>, strict_seconds: null, empirical_seconds: null}
resources: {lut: null, ff: null, dsp: null, bram: null, power_w: null}
repeat: {seed: 0, index: 0}
status: <pass-or-explicit-failure>
```

Formal schemas omit bracketed placeholder values; units must be embedded in field names or enumerated values. Every record must include provenance hashes; scoring utilities must reject records with mismatched schema versions.

### 13.3 Caching and Reproducibility

Cache keys must incorporate source code, parameters, shape specializations, compiler version, ISA version, hardware profile, feature toggles, and optimization flags. Modifying hardware or ABI configurations invalidates previously cached binaries. Report cold-build and warm-build costs, tracking cache hit rates.

Binaries, bitstreams, execution traces, and model weights reside in external artifact storage or local build directories; Git tracks schemas, scripts, lightweight manifests, and verification hashes, following existing repository conventions for `.tnca` artifacts.

Release packages must provide reproduction scripts, locked dependencies, raw experimental outputs, analysis pipelines, and failure logs. Reproducibility must not require manually transcribing numbers from screenshots.

## 14. Staged Roadmap and Acceptance Milestones

### 14.1 Dependency Graph and Priorities

```text
R38-00 Baseline/provenance
       ↓
R38-01 Minimal three-layer loop ──→ R38-02 ArchBench protocol
       ↓                                   ↓
R38-03 Profiler ←─────────────────→ R38-04 Cost model
       ↓                                   ↓
R38-05 Three cross-layer cases ───→ R38-06 Method/Agent controlled experiments
       ↓                                   ↓
R38-07 FPGA/NVIDIA measurements ──→ R38-09 Paper and artifact
       ↓
R38-08 PyTorch/LLM system demo ───→ Appendix / System validation
```

Board selection, EDA tool licensing, memory controllers, and host interfaces should be validated early, ensuring hardware feasibility is established before concluding simulation studies. ArchBench evaluation rules must be frozen prior to large-scale optimization.

### 14.2 Work Package Contracts

| ID | Work Package | Acceptance Criteria | Fallback Strategy |
|---|---|---|---|
| R38-00 | Baseline and Provenance Audit | Reproduce R37 correctness across all three levels; isolate R36 pending gate; map AOT/MLIR paths; lock environment | Fix baseline first; never cite stale reports as current passes |
| R38-01 | Minimal Three-Layer Loop | Implement one real cross-layer feature from source to RTL; demonstrate compatibility rejection, rollback, and provenance tracking | Reduce evaluated points while keeping full stack intact |
| R38-02 | ArchBench v0 | 45-operator manifests, splits, precisions, weights, canonical resource ledger, common $L^*$, AQ metric, and self-checks | Use absolute latency for classes lacking rigorous lower bounds |
| R38-03 | Profiler v0 | ROI and counter conservation, IR-to-PC mapping, diagnosis of RF, compute, and memory bottlenecks | Use verified core counters rather than unvalidated traces |
| R38-04 | Cost Model v0 | Held-out accuracy, ranking correlation, regret bounds, uncertainty estimates, and construction costs | Bound model applicability and add anchor points if dynamic modeling fails |
| R38-05 | Cross-Layer Optimization | Three case studies across at least two distinct bottlenecks; resource bounds, full ablations, documented negative results | Refocus paper on compilation if gains stem purely from software |
| R38-06 | Controlled Empirical Studies | Compiler-only vs. staged vs. co-iteration; 2×2 Agent factorial; full budget and reproducibility tracking | Report as case studies rather than broad generalizations if sample size is limited |
| R38-07 | Hardware Measurement Platform | P&R/bitstream execution on specified FPGA via public runtime; NVIDIA baselines; timing/energy data | Narrow paper scope to RTL methodology if FPGA deployment stalls |
| R38-08 | PyTorch / LLM Integration | Graph-level execution, decoder block, small model on board without implicit operator fallbacks | Defer full PrivateUse1 backend; retain verified subgraph execution |
| R38-09 | ISCA-Ready Evidence Package | Held-out evaluation frozen, all main tables reproducible, related work matrix, threats and negative results disclosed | Continue research iterations rather than masking incomplete evidence with demos |

### 14.3 Proposed Timeboxes

The timeline below assumes an experienced engineering team spanning compilers, RTL design, and FPGA systems; **it does not represent a schedule guarantee for a single individual or agent**. Target submission deadlines must be verified independently.

| Timeframe | Primary Focus | Milestone Gate |
|---|---|---|
| Weeks 1–2 | Baseline audit, literature review, platform feasibility, ArchBench protocol pilot | G0: Verified starting state and resource constraints |
| Weeks 3–4 | Minimal three-layer change, version contracts, ROI counters, smoke gate | G1: Functional minimal closed loop |
| Weeks 5–7 | Workload expansion, calibration, cost model v0, simulation throughput pilot | G2: Reliable scoring and predictive models established |
| Weeks 8–11 | Case studies A/B/C, full ablations, ongoing FPGA bring-up | G3: Cross-layer performance gains and root causes verified |
| Weeks 12–14 | Controlled experiments, Agent 2×2 trials, NVIDIA baselines | G4: Fixed-budget causal evidence established |
| Weeks 15–17 | PyTorch graph → decoder block → full model; frozen held-out evaluation | G5: Real-world application and generalization verified |
| Weeks 18–20 | Paper writing, artifact packaging, independent reproduction, audits | G6: Complete submission-ready evidence package |

Actual scheduling must benchmark runtimes for regression suites, synthesis, FPGA builds, and Agent iterations to establish realistic compute budgets.

### 14.4 Concrete Action Sequence for the First Two Weeks

1. Log branch/commit hashes and artifact signatures; rerun baselines in a clean environment, listing pending gates separately.
2. Trace and map public AOT call graphs, establishing explicit boundaries between Python lowering and C++ MLIR passes.
3. Run 8 representative smoke points, verifying ROI cycle counters against host elapsed time; profile tool runtimes and memory footprints.
4. Cross-verify analytical lower bounds, scripts, and RTL cycle data on a single operator; run anti-gaming checks with injected memory waste.
5. Freeze ArchBench v0 operator definitions, input distributions, weighting policies, and train/validation/test splits.
6. Select the initial minimal cross-layer case study (prioritizing RF layout/operand collector or asynchronous pipeline features).
7. Finalize target FPGA board specifications, available resources, EDA tools, and host interfaces; validate an initial GEMM pipeline.
8. Complete the literature matrix for HASCO, ACT, Chipyard, and Gemmini, refining paper contribution claims accordingly.

## 15. Paper Structure and Key Figures

### 15.1 Working Title (Provisional)

**Closing the Loop: Continuous Co-Evolution of Compiler IR, ISA, and Microarchitecture for Tensor Accelerators**

The title avoids premature speedup claims or references to outperforming NVIDIA; these will be finalized based on empirical results. If the final contribution emphasizes the co-iteration methodology rather than specific microarchitectural mechanisms, the abstract must state this directly.

### 15.2 Proposed Section Outline

1. **Introduction / Motivation**: How rigid architectural abstraction layers cause cross-layer mismatches and long feedback delays.
2. **Background & Related Work**: Agile hardware methodologies, hardware-software co-design, compiler/ISA generation, Agent-driven optimization.
3. **The Executable Co-Iteration Loop**: Versioned contracts, content-addressed manifests, multi-tiered validation, feedback mechanisms, and rollback protocols.
4. **ArchBench Evaluation Protocol**: Workload formulation, common resource lower bounds, frozen weighting, budget constraints, and anti-gaming rules.
5. **Cross-Layer Co-Evolution Case Studies**: Three detailed case studies analyzing compiler, ISA, and microarchitectural interactions.
6. **Evaluation**: Design quality Pareto frontiers, iteration engineering costs, Agent factorial ablations, scale generalization, and NVIDIA comparisons.
7. **Discussion & Limitations**: Substrate constraints, lower-bound tightness, Agent reproducibility, and threats to validity.
8. **Appendix**: PyTorch/LLM integration, complete workload shape lists, counter definitions, board screenshots, and artifact reproduction guides.

### 15.3 Core Figures and Tables to Prioritize in the Main Text

| Figure / Table | Research Question Addressed |
|---|---|
| Closed-loop architecture and contract diagram | What is the concrete methodology and how is executability maintained? |
| Feedback latency distribution across iteration styles | Does co-iteration genuinely shorten feedback turnaround times? |
| Best feasible performance vs. total engineering budget | Does the methodology yield superior designs under identical budgets? |
| Cross-scale/environment AQ, resource Pareto frontiers, and Roofline efficiency | Are performance improvements achieved without resource inflation or peak depression? |
| Case study ablations and counter-based attribution | Why is cross-layer co-evolution necessary to unlock these performance gains? |
| 45-operator performance heatmap / CDF | Are improvements general or confined to GEMM and isolated operators? |
| Agent 2×2: Quality-vs-tokens and quality-vs-time curves | Do cost models and profilers improve Agent optimization efficiency? |
| Cost model prediction vs. actual execution and regret | How accurate is the cost model and under what conditions does it fail? |
| Tiered FPGA and NVIDIA comparative results | What conclusions are supported by cross-platform empirical data? |

Detailed raw data belongs in the Appendix; the main text must contain sufficient self-contained evidence to substantiate core claims.

## 16. Risks, Stopping Criteria, and Acceptable Negative Results

| Risk | Early Warning Indicator | Mitigation and Scope Adjustment |
|---|---|---|
| Engineering integration lacks research novelty | Mechanisms already exist in prior art; evaluation demonstrates only that the pipeline runs | Emphasize cross-layer interactions and fixed-budget studies; adjust paper positioning |
| Roofline efficiency metric is vulnerable to gaming | Lowering peak throughput or adding redundant memory traffic improves scores | Rely on frozen reference weights, semantic data volumes, and joint reporting with absolute speedup $S$ |
| Loose lower bounds prevent meaningful efficiency comparisons | Efficiency $\eta$ is extremely low across all categories, matching mature libraries | Report per-category sensitivity, mark bound tightness, prioritize absolute performance ranking |
| Performance gains stem entirely from bug fixes | Resolving a functional fallback yields outsized speedups | Disentangle bug fixes from co-design gains; compare against strong, fully functional baselines |
| Co-iteration gains stem purely from increased trials | Performance advantages vanish under iso-candidate budgets | Disentangle raw search throughput from decision quality; report both findings |
| Agent advantage is driven by data contamination | Prompts or logs contain solutions or cross-group leakage | Enforce isolated workspaces, frozen task descriptions, access controls, and trace audits |
| Cost model misleads optimization trajectory | Rank correlation is low; overconfidence on unseen configurations | Implement prediction intervals, trigger active physical measurements, report negative results |
| Profiler counter attribution is unreliable | Sum of stall cycles exceeds ROI makespan or triggers are ambiguous | Correct counter contracts and conservation laws before evaluating optimizations |
| RTL simulation throughput is too slow | Multi-thousand point evaluations take weeks | Implement multi-fidelity pruning, evaluate anchors on RTL, expand fast models; never conflate predictions with physical measurements |
| FPGA timing closure or resource exhaustion | High register porting, wide FMAs, or large SRAMs exceed board capacity | Conduct early place-and-route on small configurations; tag infeasible designs in cost models |
| Inability to outperform NVIDIA GPUs | High internal efficiency but lower absolute throughput | Accurately report efficiency and methodological gains; core research propositions do not require beating GPUs |
| PyTorch / LLM integration consumes excessive resources | Vast engineering time diverted into framework compatibility | Prioritize small subgraphs and isolated decoder blocks; treat full PrivateUse1 backend as an extension |

**Key Stopping / Pivot Gates:**

- If Gate G1 cannot execute an ISA change through public AOT to RTL: Resolve contracts and tool integration before launching large-scale automated search.
- If Gate G2 lower bounds, timing metrics, or counters fail self-consistency audits: Suspend performance evaluations while maintaining correctness development.
- If Gate G3 reveals all gains stem from compiler tuning alone: Honestly reframe the contribution around compiler optimization, or identify new structural bottlenecks.
- If Gate G4 reveals A11 does not outperform A00/A01: Retain the data; analyze why model overhead or agent context limits outweighed predictive benefits.
- If physical FPGA or full LLM deployment is incomplete: Avoid presenting simulation logs as physical measurements; adjust paper claims to match verified evidence.

## 17. Definition of Done and Local Review Decision Points

### 17.1 Core Definition of Done

Completion of the primary research line requires fulfilling all of the following:

- [ ] Real minimal MLIR/lowering–ISA–microarchitecture closed loop with versioned cross-layer contracts.
- [ ] 45-operator specifications, multi-thousand point workload manifests, precision rules, frozen splits and weights.
- [ ] Auditable ArchBench theoretical lower bounds and performance metrics, strictly separating measured points from predictions.
- [ ] Architecture profiler with verified ROI/counter contracts, alongside a calibrated cost model reporting error intervals.
- [ ] At least three documented cross-layer case studies, including analysis of failed or ineffective explorations.
- [ ] Controlled experiments comparing single-layer, staged, and rapid-feedback co-iteration under identical budgets.
- [ ] Agent 2×2 factorial evaluation logging token consumption, tooling expenses, failure rates, and reproduction data.
- [ ] Demonstrated AQ improvements under canonical resource constraints, cross-scale/environment sensitivity analyses, and physical deployment validation (or full reporting of negative findings and appropriate scope adjustment).
- [ ] Fully reproducible artifacts, raw datasets, and evaluation scripts executable via public AOT/runtime interfaces.

Complete roadmap deliverables also encompass:

- [ ] Physical board-level measurements on target FPGAs compared against strong NVIDIA GPU baselines.
- [ ] PyTorch inference integration, decoder block / small LLM execution on FPGA, and machine-readable execution logs.

These two objectives should be pursued within R38, but the definition of completion must not be diluted to conceal incomplete milestones. Project dashboards must track core research, physical hardware, and ecosystem integration as separate progress tracks.

### 17.2 Design Decisions for Local Review

The following items represent strategic review decisions:

1. Formal endorsement of "Hardware DevOps with three-layer executable co-iteration" as the primary thesis.
2. Adoption of Section 4.10 AQ as the primary architectural metric, with $S$ and $E$ as accompanying deployment and diagnostic metrics; approval of canonical resource libraries, common lower bounds, and balanced/application evaluation tracks.
3. Selection of target FPGA boards and GPU comparison baselines, alongside compute and API budget allocations.
4. Selection of the initial cross-layer case study (RF layout vs. async pipelining), determined by initial bottleneck profiling.
5. Confirmation of target conference submission windows and team resourcing, adjusting the 20-week reference timeline accordingly.
6. Staging PyTorch integration (graph backend first, PrivateUse1 second); determining LLM model scale based on physical memory and precision constraints.

Baseline audits and minimal closed-loop bring-up can proceed immediately; hardware procurement, commercial API budgets, and final protocol freezes require explicit project authorization.

## Appendix A: Evaluation Protocol Self-Check Test Cases

The ArchBench evaluation harness must be validated against the following test cases:

1. Halving execution times uniformly across all points must scale $S$ and $\eta$ exactly per mathematical definition under a fixed architecture.
2. Depressing advertised peak throughput must not alter absolute performance $S$; resulting changes in efficiency $E$ must be flagged and not interpreted as performance gains.
3. Injecting redundant DRAM traffic must not alter the semantic lower bound; calculated efficiency must degrade or remain constant, never improve.
4. Duplicating identical problem shapes or subdividing an operator into more stages must not alter the operator's total weight or aggregate execution time.
5. Removing a slow or failing workload point must cause the completeness gate to fail, rather than increasing aggregate scores.
6. Unit conversions across cycles, nanoseconds, seconds, and MACs/FLOPs must remain consistent; identical inputs under different units must yield identical scores.
7. Workloads executing faster than theoretical lower bounds must trigger audit failures rather than passing via numerical clipping.
8. When multi-core aggregate active cycles exceed total makespan, the evaluation harness must still compute metrics against the correct makespan ROI.
9. Modifying reference weights or hashes within a candidate submission must be detected and rejected by the evaluator.
10. In the absence of physical power sensors, energy fields must remain null rather than populated with TDP values or zeros.

## Appendix B: Research Card Template for Individual Cross-Layer Case Studies

```text
Problem: Which workload, profile, and resource exhibit losses?
Evidence: Raw counters/traces, timing, precision, and resource reports.
Hypothesis: Why is modifying the compiler / ISA / microarchitecture combination necessary?
Change: Minimal patch per layer, contract versions, and compatibility policy.
Prediction: Cost model expected speedup, domain, and uncertainty.
Verification: Functionality, asynchronous/concurrency, boundary inputs, O0/O1/O2.
Ablation: Layer toggles, legal software fallbacks, iso-resource controls.
Outcome: Actual performance, resources, frequency, error, holdout, and negative results.
Cost: Time, tokens, tooling, and manual interventions from hypothesis to validation.
Verdict: Accept / revert / retain for research; transferable design insights.
```

## Appendix C: Citations and Evidence Guidelines

All citations point to primary authors, institutions, arXiv preprints, or official project documentation. Section 2 establishes primary literature references, with subsequent sections linking relevant methodologies where applied. The literature review conducted here is non-exhaustive; formal publication requires completing detailed nearest-neighbor reviews against published papers and released artifacts, avoiding reliance on paper abstracts to deduce competitor limitations.

Dynamic APIs, software versions, GPU specifications, and conference dates must be re-verified at submission time. This proposal omits unverified acceptance claims, unconfirmed submission deadlines, hypothetical board resources, and speculative performance figures.

Descriptions of existing repository capabilities are derived from local source code and recorded verification data; this document work did not re-execute the complete R37 regression suite or perform board bring-up. All proposed formulas, work packages, experimental designs, and timelines represent R38 proposals, and must not be interpreted as implemented features or measured results.
