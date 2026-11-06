# Description
This document is the compiler architecture design draft for TNCA-v1 (originally drafted with GPT-6-astra (max) in Chinese and translated into professional technical English). It outlines the compiler strategy, dialect layering, Warp/Context Specialization adaptation, register file and asynchronous resource lowering, code generation, and implementation milestones.

# Contents

This design first examines the actual status of Warp Specialization (WS) in upstream Triton mainline, along with the dependency boundaries of related IRs and passes, before formulating the TNCA dialect and implementation roadmap. The primary focus is on identifying what code can be directly reused, what requires abstracting target interfaces, and how pipelining across four Role Contexts (RCTXs) avoids copying GPU warp and shared-memory assumptions. **Compiler development should proceed in parallel immediately; the first milestone must focus on allowing real Triton kernels to continuously validate hardware interfaces, rather than attempting to deliver a fully automated optimization backend covering all operators up front.**

The overall recommendation is:

> **Retain the Triton frontend, TTIR, and reusable layout and loop transformation infrastructure; introduce a thin TNCA resource and asynchronous execution layer; decompose Warp Specialization into three parts—partitioning algorithms, communication protocols, and physical context mapping—to be reused or replaced independently.**

In particular, avoid two extremes:
- Redefining all tensors, arithmetic, and control flow into `tnca.*` starting from TTIR.
- Lowering first to NVIDIA shared-memory/TMEM programs to reuse existing WS, and then forcefully reinterpreting them as Register File (RF) programs.

The following sections cover: **Upstream Status, IR Design, Context Specialization, Layout and Resource Allocation, Code Generation, Implementation Roadmap, and Hardware Feedback Points.**

Regarding RTL status, this document relies strictly on the provided ISA and Status without treating self-reported "complete" statuses as independently verified implementation facts.

---

# 1. Upstream Status: WS Is in Mainline with Multiple Implementation Paths

## 1.1 No Longer an Unmerged Experimental Branch

As of the **September 19, 2026** search baseline:
- The Triton official releases page marks the latest release as **3.8.0**;
- At least since the **3.4.0** release notes, Automatic Warp Specialization has been explicitly listed;
- Upstream `main` contains complete WS IR, automatic partitioning, cross-partition communication construction, and target-dependent lowering ([github.com](https://github.com/triton-lang/triton/releases/tag/v3.8.0)).

However, several layers must be clearly distinguished:

| Level | Current Upstream Content | Significance to TNCA |
|---|---|---|
| High-level Triton entry | `tl.range(..., warp_specialize=True)` | Can be preserved as an automatic specialization request |
| Explicit programming entry | Gluon `warp_specialize(...)` | Region/capture model can be referenced, but cannot promise execution of arbitrary NVIDIA Gluon programs |
| Generic IR representation | `ttg.warp_specialize`, partition regions | Structural design can be reused, but physical execution semantics do not map directly to RCTX |
| Automatic WS midend | Partition scheduling, partition loops, Aref communication | Most valuable components to extract and reuse |
| GPU target backend | Shared memory, TMEM, GPU barrier, warp allocation | Must be replaced for TNCA |

The semantics of `ttg.warp_specialize` encompass concurrent execution of multiple regions, explicit capture of worker arguments, and convergence at the end; its execution domain remains warp groups of varying sizes. Gluon's explicit interface also includes worker warp counts and register budgets, rather than functioning as a completely target-agnostic task system ([triton-lang.org](https://triton-lang.org/main/dialects/TritonGPUOps.html)).

## 1.2 Hopper and Blackwell Automatic WS Paths Coexist in Current Mainline

This point requires special emphasis.

Documentation for `tl.range` still states that it "only supports Blackwell and is primarily intended for simple matmul loops." However, current NVIDIA `compiler.py` already:
- Invokes `add_hopper_warpspec` on the corresponding architecture branch;
- Invokes a separate automatic WS pipeline on Blackwell and subsequent branches.

Therefore, **one cannot conclude that "mainline lacks Hopper WS" based solely on API documentation**; evaluation must rely on actual pipelines and tests in pinned versions ([triton-lang.org](https://triton-lang.org/main/python-api/generated/triton.language.range.html)).

The Hopper path itself carries explicit constraints: the inspected implementation imposes restrictions on warp counts and control flow containing `else` branches. It is not a general-purpose partitioner capable of handling arbitrary control flow ([raw.githubusercontent.com](https://raw.githubusercontent.com/triton-lang/triton/main/third_party/nvidia/hopper/lib/Transforms/WarpSpecialization.cpp)).

## 1.3 Most Critical Upstream Files for TNCA

Priority reading should focus on the following modules:

```text
lib/Dialect/TritonGPU/Transforms/WarpSpecialization/
    AutomaticWarpSpecialization.cpp
    PartitionScheduling.cpp
    PartitionLoops.cpp
    Partition.cpp

include/triton/Dialect/TritonGPU/Transforms/
    Partition.h
    PartitionSchedulingUtility.h
    PipeliningUtility.h

third_party/nvidia/include/Dialect/NVWS/IR/
    NVWSOps.td
    NVWSTypes.td

third_party/nvidia/.../NVWS/.../
    InsertAref
    InsertTmemAref
    LowerAref
```

The current automatic WS pipeline chains:

```text
Partition scheduling
→ Aref insertion
→ Aref lowering
→ Partition loop construction
→ WarpGroup lowering
→ Subsequent loop scheduling
```

However, its driver directly depends on NVIDIA NVWS; one cannot assume the pass is target-agnostic merely because files reside under the `TritonGPU` directory ([raw.githubusercontent.com](https://raw.githubusercontent.com/triton-lang/triton/main/lib/Dialect/TritonGPU/Transforms/WarpSpecialization/AutomaticWarpSpecialization.cpp)).

Specifically:
- `PartitionScheduling` still embeds TMEM, CLC, and MMA categories along with specific clustering rules;
- `PartitionLoops` allocates shared-memory temporary objects for certain cross-partition tensor results;
- `NVWS Aref` operations currently operate on `TTG_MemDescType`, and their lowering targets GPU barriers.

**These are precisely the locations where target interfaces must be extracted, rather than boundaries that can be invoked unmodified** ([raw.githubusercontent.com](https://raw.githubusercontent.com/triton-lang/triton/main/lib/Dialect/TritonGPU/Transforms/WarpSpecialization/PartitionScheduling.cpp)).

---

# 2. Overall Strategy: Hybrid IR Rather than Full Translation of TTIR to TNCA

The recommended structure is as follows:

```text
Triton Python
    │
    ▼
Upstream TTIR
    tt + arith + math + scf
    │
    │ Reuse canonicalization, constant propagation, access analysis, etc.
    ▼
Layout-Annotated Tensor IR
    tt + restricted ttg layout/convert_layout + arith + scf
    │
    │ TNCA fragment selection
    │ Context partition / channel planning
    ▼
TNCA Execution IR
    Original arithmetic and control flow
    + TNCA RF, accumulator, channel, async operations
    │
    │ Resource allocation, protocol materialization, role outlining
    ▼
TNCA Low-level IR
    Fixed window / accumulator / barrier allocation
    Independent CFG per RCTX
    │
    ▼
Machine Instructions and Register Allocation
    │
    ▼
Existing assembler / Future LLVM TNCA target
    │
    ▼
Binary + Kernel Manifest
```

The core design principle here is:

> **The TNCA dialect expresses only what upstream IR cannot accurately express; it does not reinvent what upstream IR already represents cleanly.**

For example:

| Category | Proposed Representation |
|---|---|
| Standard integer and floating-point arithmetic | `arith` / `math` |
| Loops and conditional control flow | `scf`, later lowered to `cf` |
| Tensor shape, broadcast, reduce, dot | Retained as `tt` in early stages |
| Tensor layout | Maximize reuse of `ttg` layout representations and LinearLayout |
| RF window ownership | `tnca` |
| Asynchronous read source lifetimes | `tnca` |
| Special accumulator accesses | `tnca` |
| Inter-role communication and synchronization | `tnca` |
| Physical instruction encoding | Final backend phase; omitted from high-level TNCA IR |

Premature generation of operations such as:

```text
tnca.add
tnca.mul
tnca.for
tnca.if
tnca.reshape
tnca.broadcast
```

must be avoided. Doing so merely to support a new hardware target forfeits extensive existing optimizations and analysis capabilities.

---

# 3. Specific Reuse Inventory: Direct Reuse, Target Interfaces, and New Code

## 3.1 Category 1: Direct Reuse with Minimal Modification

### Frontend and Compilation Framework

- Triton Python AST to TTIR lowering;
- `constexpr` specialization;
- JIT compilation caching;
- Kernel metadata passing;
- Backend discovery mechanism;
- Python compilation stages framework;
- MLIR parser, printer, ODS, pass manager, and diagnostics infrastructure.

Upstream `BaseBackend` supports custom backend stage sequences, while backend discovery accommodates out-of-tree entry points. Consequently, TNCA does not need to masquerade as a CUDA backend or patch NVIDIA's `compiler.py` to integrate ([raw.githubusercontent.com](https://raw.githubusercontent.com/triton-lang/triton/main/python/triton/backends/compiler.py)).

### Common Analyses and Transformations

Prioritize direct reuse of:
- Canonicalization, CSE, SCCP;
- Dominance, post-dominance, and slice analysis;
- SCF structural transformations;
- Triton AxisInfo;
- LinearLayout composition, inversion, and mapping utilities.

AxisInfo provides contiguity, divisibility, and constancy information, serving as a solid foundation for LSU vs. Tensor DMA decisions. LinearLayout provides the mathematical framework for register layout transformations ([raw.githubusercontent.com](https://raw.githubusercontent.com/triton-lang/triton/main/include/triton/Analysis/AxisInfo.h)).

However, note:

> AxisInfo is not a complete alias analysis, nor is it proof that "this load can be hoisted."

It characterizes memory access footprints, but cannot independently prove the legality of reordering accesses across stores.

## 3.2 Category 2: Reuse Algorithm or Body, Replace Target Policy

| Module | What to Retain | What TNCA Replaces |
|---|---|---|
| Coalescing | Contiguity analysis, access clustering, candidate layouts | Line/word transaction costs, 16-lane LSU capabilities |
| Layout propagation | Encoding propagation, redundant conversion elimination, partial rematerialization | Conversion costs, set of legal layouts |
| Partition scheduling | Dataflow graph construction, dependence propagation, partitioning infrastructure | Operation classification, clustering rules, RCTX budget constraints |
| Partition loops | SCF cloning, loop-carried value reconstruction | Role containers, cross-role result forwarding |
| Aref insertion | Producer/consumer identification, last-use analysis | RF/ARF channels and release events |
| Software pipelining | Prologue/kernel/epilogue structural rewriting | Stage scheduling, async dependences, RF buffering |
| Concurrency checking | Analysis principles for visibility and outstanding memory accesses | RF footprints, TNCA epoch/completion protocols |

For instance, upstream `Coalesce` considers subsequent shared-memory accesses when selecting layouts for descriptor load/store operations, and `RemoveLayoutConversions` embeds GPU-specific cost heuristics. Accordingly, these passes must be **whitelisted and verified on a case-by-case basis**, rather than enabled globally without qualification ([raw.githubusercontent.com](https://raw.githubusercontent.com/triton-lang/triton/main/lib/Dialect/TritonGPU/Transforms/Coalesce.cpp)).

## 3.3 Category 3: Truly New Components for TNCA

These requirements cannot be satisfied by renaming opcodes:

1. Cross-RCTX physical RF/ARF resource planning;
2. Asynchronous read-source lifetime tracking;
3. Ready/Free channel abstractions;
4. Mapping command completions to mbarrier generations;
5. TNCA fragment selection and layout lowering;
6. Legalization based on concrete RTL capabilities;
7. TNCA ISA emission;
8. Kernel launch manifest generation and simulation runtime integration.

**New development must concentrate here, rather than rewriting tensor arithmetic or loop transformations.**

---

# 4. Layout IR: Prioritize Upstream Compatibility Without Treating Warps as RCTXs

## 4.1 A Practical Starting Point: TTGIR Subset with a Single Vector Execution Domain

The initial layout domain configuration should be parameterized as:

```text
lane = 16
warp = 1 logical layout group
CTA = 1 program layout domain
```

to evaluate reusing:
- `BlockedEncoding`;
- `SliceEncoding`;
- `LinearEncoding`;
- `ttg.convert_layout`.

Here, `warp = 1` is **solely a representation adaptation for reusing layout utilities**, not an assertion that TNCA possesses GPU warps, nor that execution is confined to a single RCTX.

The existing TTIR-to-TTGIR conversion accepts `numWarps`, `threadsPerWarp`, and `numCTAs` parameters, providing an entry point for this adaptation; however, this does not guarantee that downstream passes accommodate 16 lanes without regression. Each pass must be verified through unit tests ([raw.githubusercontent.com](https://raw.githubusercontent.com/triton-lang/triton/main/lib/Conversion/TritonToTritonGPU/TritonToTritonGPUPass.cpp)).

The governing architectural rule is:

> **The tensor distribution managed by an individual RCTX and the division of roles across multiple RCTXs are two orthogonal axes.**

Never conflate:

```text
num_warps = 4
```

with:

```text
RCTX = 4
```

## 4.2 Avoid Inventing a Full `#tnca.tensor_layout` Up Front

The recommended progression is:
- Retain upstream encodings for generic tensors;
- Introduce TNCA-specific layout attributes only for RF windows and MMA fragments;
- Transition to native TNCA layouts only when modeling packed half-precision, fixed fragment row orderings, and RF bank conflict constraints becomes necessary.

This maximizes the viable reach of upstream layout optimization passes.

Note that `ttg.linear` imposes specific constraints on input dimensions and mapping dimensionality; not every mathematical LinearLayout can be directly represented by this encoding. **The underlying LinearLayout mathematical infrastructure should be reused directly without forcing all TNCA physical layouts to masquerade as `ttg.linear`** ([raw.githubusercontent.com](https://raw.githubusercontent.com/triton-lang/triton/main/include/triton/Dialect/TritonGPU/IR/TritonGPUAttrDefs.td)).

## 4.3 Establish an Explicit Layout Legality Gate

Prior to lowering into TNCA Execution IR, the compiler must enforce a legality verification gate:

```text
Every tensor element has an unambiguous RF/ARF mapping
No unlowered operations depend on cross-warp shared-memory exchange
Every convert_layout has an executable lowering path
Every MMA fragment complies with ISA byte-layout specifications
```

Permitted transformation strategies should be prioritized as follows:
1. Metadata-only transformations;
2. Register renaming;
3. Lane shuffle / permute;
4. Explicit cross-vector register data movement;
5. Pack / unpack;
6. Tensor DMA transfer tile re-shaping;
7. Global temporary memory fallback.

Strategy 7 must explicitly account for spill/temporary memory overhead rather than treating layout conversion as cost-free.

---

# 5. TNCA Dialect: Three Conceptual Layers Without Three Bulky Dialects

The `tnca.*` operation and type names below represent **proposed design constructs**, not assertions of existing upstream IR.

## 5.1 Layer 1: Value Semantics Handled by TTIR and Standard Dialects

At this level:

```mlir
%a = tt.load ...
%b = tt.load ...
%c = tt.dot %a, %b, %acc ...
```

remains the optimal representation for algebraic simplifications and memory access analyses.

Values must not be prematurely assigned physical identifiers such as `W3`, `A7`, or `B12` at this stage. Introducing physical resource indices prematurely impedes:
- Common Subexpression Elimination (CSE);
- Rematerialization;
- Tiling;
- Role partitioning;
- Pipelining stage depth selection.

## 5.2 Layer 2: Resource and Asynchronous Execution Semantics

The proposed core type system includes:

| Type | Meaning |
|---|---|
| `!tnca.rf_ref<...>` | A virtual RF slice and layout descriptor, not a general pointer |
| `!tnca.acc_ref<...>` | A virtual accumulator resource handle |
| `!tnca.desc<...>` | A semantically snapshotted TensorMap descriptor |
| `!tnca.event<kind>` | A specific completion condition token |
| `!tnca.channel<...>` | A compile-time communication contract, realized via RF/ARF and barriers |
| `!tnca.lease<read/write>` | An access lease conferring read or write authority over a channel slot |

Event kinds must distinguish at least:

```text
load_ready
compute_done
source_reusable
store_visible
```

Even if the current hardware merges two conditions into a single physical signal, the IR must not conflate them into a single type. For example:

```text
Current MMA:
    source_reusable → Bound to compute_done

Future MMA:
    source_reusable → Independent hardware completion event
```

Maintaining this distinction ensures that future support for early source release requires modifying only lowering and scheduling logic, without overhauling the high-level ownership model.

### Channels Are Not a New Storage Tier

Here, a channel represents:

```text
channel
    = A set of RF/ARF slots
    + Ready/Free synchronization protocols
    + Producer/Consumer sets
```

It does not correspond to an additional physical SRAM bank, nor does it require software-addressable hardware FIFOs.

## 5.3 Proposed Operation Set

### Resources and Views

```text
tnca.rf.reserve
tnca.rf.view
tnca.rf.read
tnca.rf.write

tnca.acc.reserve
tnca.acc.read
tnca.acc.write

tnca.desc.snapshot
```

`rf.view` must be restricted to statically analyzable register sub-ranges and fragment views; it does not yield a generic byte-addressable RF pointer.

### Asynchronous Operations

```text
tnca.dma.load
tnca.dma.store
tnca.mma.async
tnca.await
```

### Roles and Communication

```text
tnca.specialize
tnca.role

tnca.channel.create
tnca.producer.acquire
tnca.producer.publish
tnca.consumer.acquire
tnca.consumer.release
```

### Performance Metadata

```text
tnca.prefetch
tnca.use_complete
tnca.progress
```

This final group may be eliminated or diverted to manifests when the underlying hardware does not implement corresponding instructions; they must never carry functional correctness dependencies.

## 5.4 Do Not Represent Ownership Merely as Annotations

Relying solely on metadata attributes such as:

```text
rf_lifetime = ...
async_read_set = ...
```

is insufficient.

Three distinct verification layers must operate concurrently:
1. **SSA Events and Leases**: Render dependences explicit in the def-use chain;
2. **`MemoryEffectsOpInterface`**: Prevent erroneous DCE, code motion, and duplication;
3. **Dedicated Asynchronous Resource Verifier**: Validate outstanding memory accesses across instructions, roles, and loop iterations.

While MLIR's side-effect interfaces describe abstract resource access, official MLIR documentation explicitly notes that they do not provide fine-grained address-range disambiguation or alias analysis. Adding `MemRead`/`MemWrite` alone cannot guarantee sound management of asynchronous RF lifecycles ([mlir.llvm.org](https://mlir.llvm.org/docs/Rationale/SideEffectsAndSpeculation/)).

Two core dialect interfaces are proposed:

```text
TNCAAsyncOpInterface
    getReadFootprints()
    getWriteFootprints()
    getCompletionKinds()
    getSourceReleaseCondition()

TNCAExecutionResourceInterface
    getExecutionUnit()
    getIssueDemand()
    getRFPortDemand()
    getCommandSlotDemand()
```

These interfaces are introduced by the TNCA dialect, while standard MLIR analyses continue to operate over standard traits.

---

# 6. Context Specialization: Reuse Partitioning Infrastructure, Redefine Communication and Mapping

This represents the architectural core of the backend.

## 6.1 Distinguish Three Distinct Concepts First

```text
Logical partition
    Functional separation on the algorithmic dependency graph

Physical RCTX
    Control context possessing its own PC, scalar registers, and wait state

Execution engine
    Shared execution hardware (SIMD, Matrix, DMA, SFU, etc.)
```

These three concepts do not map 1:1.

Specifically:

> Two mainloop RCTXs do not imply two Matrix engines, nor do they automatically double matrix throughput.

The "1 prologue + 1 epilogue + 2 mainloop" structure should serve as a **preferred scheduling template**, rather than an immutable IR constraint. If RTL issue logic or port arbitration favors this template, such preferences should be formalized in TargetInfo.

Furthermore, the prologue/epilogue roles here are distinct from the loop prologue/epilogue generated during software pipelining: the former can execute persistently throughout the kernel's lifetime.

## 6.2 Two Recommended Four-Role Templates

### Template A: GEMM

```text
R0: Input transfer and descriptor management

R1: Mainloop A
    Computes a partition of output sub-tiles across the full K dimension

R2: Mainloop B
    Computes another partition of output sub-tiles across the full K dimension

R3: Epilogue and output stores
```

The two mainloops can:
- Process separate M/N sub-tiles independently;
- Interleave independent output tiles within persistent kernels.

**Do not automatically perform split-K along the reduction dimension merely to populate two mainloops in the first version.**

Doing so introduces:
- Additional reductions;
- Complex accumulator lifetimes;
- Altered floating-point accumulation order;
- Requirements for atomics or secondary reduction kernels.

### Template B: FlashAttention

```text
R0: K/V transfers

R1: Matrix mainloop
    QKᵀ / PV computations

R2: Vector mainloop
    Masking / max / exp / sum / O rescale

R3: Final normalization, store output, or prepare next task
```

Here, the two mainloops assume distinct compute responsibilities rather than mirroring homogeneous matrix pipelines.

The compiler must preserve the dependency sequence:

```text
QK[t] → softmax[t] → O rescale[t] → PV[t]
```

while exploiting valid overlap opportunities between:

```text
QK[t+1] and softmax[t]
```

## 6.3 Mandatory Support for 2-RCTX Profile

The provided Status indicates **2 Role Contexts complete**, rather than 4.

Two target profiles must be defined from inception:

```text
tnca-s3-rctx2
tnca-rctx4
```

Current regression suites must treat only the former as a verified target.

For 2-RCTX execution:
- Deploy producer + compute/store partitioning;
- Or fall back to single-RCTX software pipelining.

The compiler must never generate four mutually dependent roles and mechanically concatenate pairs onto a single PC.

**Merging logical partitions requires regenerating the schedule.** Otherwise, a deadlock condition inevitably arises:

```text
Task A on RCTX 0 stalls waiting for an event,
but Task B responsible for signaling that event is also mapped to RCTX 0.
```

This deadlock cannot be resolved by adding barriers.

---

# 7. Concrete Algorithms and Reuse Boundaries for Automatic CS

The automatic Context Specialization pass should be structured in six stages:

## 7.1 Build Dataflow Graph with Loop-Carried Dependences

Graph edges must not be restricted to standard SSA def-use chains; they must incorporate:
- Loop-carried dependences;
- Potentially aliasing memory dependences;
- Control dependences;
- Accumulator update orderings;
- Resource conflicts between asynchronous sources and destinations.

Upstream partition scheduling already contains dataflow graph construction and partition merging infrastructure; `PartitionSet` also tracks cross-partition and cross-iteration inputs and outputs. These represent high-priority components for direct reuse ([raw.githubusercontent.com](https://raw.githubusercontent.com/triton-lang/triton/main/lib/Dialect/TritonGPU/Transforms/WarpSpecialization/PartitionScheduling.cpp)).

## 7.2 Replace Operation Classification and Partition Policy

The recommended classification for TNCA comprises:

```text
Address/control
Load/transfer
Matrix
Vector/reduction/SFU
Store/epilogue
Synchronization
```

Avoid inheriting policies such as:

```text
TMEM and MMA partitioning mandates
CLC dedicated worker requirements
NVIDIA warpgroup register budgets
```

among others.

Target differences should be encapsulated within an adaptation interface rather than dispersed throughout upstream algorithms:

```cpp
// Proposed interface; not an existing upstream API
struct SpecializationTargetInfo {
  OpClass classify(Operation *);
  bool canRematerialize(Operation *);
  Cost estimateCrossPartitionUse(Value);
  bool canMerge(Partition, Partition);
  ResourceBudget getBudget();
};
```

Upstream currently lacks a unified interface of this form ready for TNCA implementation; a controlled source extraction/refactoring is required.

## 7.3 Select Communication Mechanisms for Cross-Partition Values

Select mechanisms in the following order:
1. Constants: Duplicate across contexts;
2. Inexpensive pure address calculations: Rematerialize locally;
3. Kernel arguments: Each role reads independently from the ABI;
4. Tensors: Pass via shared RF/ARF windows;
5. Dynamic scalars: Synchronized RF mailbox slots;
6. Unregisterable values: Global temporary memory fallback.

Note especially:

**Scalar Register Files are strictly private to each RCTX.**

Consequently, a barrier token or dynamic loop bound generated by R0 does not implicitly exist in R1's `Sx` registers.

In static pipelines, epochs can generally be calculated independently by each role based on iteration indices, eliminating the need to transmit explicit tokens across roles. Truly dynamic runtime scalars require explicit communication.

## 7.4 Insert RF Channels Instead of Shared-Memory Arefs

Upstream Aref constructs (producer-enter/exit, consumer-enter/exit) align well conceptually, but their implementation relies on MemDesc and shared/TMEM allocations. TNCA should reuse their dependence analysis and insertion logic while replacing the underlying storage constructs and synchronization lowering ([raw.githubusercontent.com](https://raw.githubusercontent.com/triton-lang/triton/main/third_party/nvidia/include/Dialect/NVWS/IR/NVWSOps.td)).

Illustrative IR:

```mlir
// Illustrative syntax; not directly executable upstream MLIR.

%inputs = tnca.channel.create
    stages = 2
    consumers = [@main0, @main1]
    payload = <A_fragment, B_fragment>

tnca.role @producer {
  scf.for %i = ... {
    %slot, %write_lease =
        tnca.producer.acquire %inputs[%i]

    %a_ready = tnca.dma.load %desc_a, %coord_a, %slot.a
    %b_ready = tnca.dma.load %desc_b, %coord_b, %slot.b

    tnca.producer.publish %write_lease
        after(%a_ready, %b_ready)
  }
}

tnca.role @main0 {
  scf.for %i = ... {
    %slot, %read_lease =
        tnca.consumer.acquire %inputs[%i] consumer(0)

    %result_done, %source_free =
        tnca.mma.async %slot.a, %slot.b, %acc0

    tnca.consumer.release %read_lease after(%source_free)
    // Subsequent use of acc0 must still await result_done.
  }
}
```

`main1` maintains an independent read lease on the shared input buffer. The producer may overwrite the slot only after all consumers have released their leases.

## 7.5 Construct Intra-Role Loops

SCF cloning and operand/result reconstruction logic from upstream `PartitionLoops` can be reused, replacing:
- Region container abstractions;
- Shared-memory temporary buffers for cross-partition tensor results;
- Role parameter capture;
- Join convergence and resource drain mechanisms.

Version 1 should restrict support to:
- Structured `scf.for` loops;
- Uniform iteration counts statically derivable across all roles;
- Provably uniform conditional branches;
- A fixed count of communication channels.

Dynamic `while` loops, irregular conditional production/consumption, and nested specialization regions should be deferred.

## 7.6 Map to Physical RCTXs Last

The final mapping pass emits:

```text
logical partition → RCTX
role entry PC
private scalar/vector allocation
shared RF aliases
barrier allocation
channel generation formula
```

Physical RCTX count must not constrain the initial dataflow graph representation; otherwise, 2-role and 4-role targets would require bifurcated compiler pipelines.

---

# 8. Protocol Lowering: Prioritize Before Automatic Partitioning

## 8.1 `publish` Must Not Mechanically Lower to `mb.arm` In-Place

High-level IR may generate events before invoking publish operations.

However, low-level ISA execution requires:

```text
Arm barrier
→ Issue associated asynchronous commands
→ Arrive / signal barrier
```

Protocol lowering must aggregate the full event group and hoist `mb.arm` prior to issuing any corresponding asynchronous commands.

Proposed protocol mappings:

| High-Level Protocol | Low-Level ISA Implementation |
|---|---|
| Producer acquire | Await Free generation |
| Producer publish | Ready generation + associated async completions |
| Consumer acquire | Await Ready generation |
| Consumer release | Await source-reusable condition, then arrive on Free barrier |
| Role join | Await completion of all outstanding async operations and memory stores |

## 8.2 Operation Count Must Be Calculated After Command Expansion

A single high-level tensor load may decompose into:
- Multiple Tensor DMA transfers;
- DMA transfers plus LSU tail transfers;
- Multiple fragment staging operations.

Therefore:

> **Do not fix `operations = 2` at the high level based on "two tensor loads" and subsequently decompose them into eight asynchronous instructions.**

The required lowering order is:

```text
Transfer formation
→ Fragment command expansion
→ Completion group construction
→ Freeze barrier operation count
```

If subsequent passes alter the number of asynchronous commands, the completion group must be reconstructed.

## 8.3 Branches and Predication Must Preserve Completion Counts

A legally predicated-off asynchronous command:
- Performs no memory accesses;
- Does not modify register state;
- Still signals exactly one hardware completion event.

Conversely, standard branching:

```text
if (...) {
  issue_async
}
```

is fundamentally different: an un-predicated branch path that skips issuing an async command while leaving the barrier armed with an unconditional `operations = 1` causes permanent deadlock.

Version 1 must:
- Convert conditional transfers into fully predicated instructions;
- Or construct path-complete barrier generations with explicit control flow branches;
- Fall back to non-specialized sequential plans when balanced completions cannot be proven.

## 8.4 Generations Must Be Part of the Verifier

The verifier must enforce:
- Every generation has exactly one designated arming owner;
- Expected completion counts match the exact number of issued asynchronous commands;
- Every consumer contributes exactly one arrive to the Free barrier per buffer slot;
- Late responses from prior generations cannot corrupt newly acquired buffer slots;
- Epoch initialization and calculation formulas match across all collaborating roles;
- Awaiting future generations follows unambiguous semantics;
- Re-arming or initialization does not drop active waiters.

**These protocol tests should enter RTL regression suites before automatic WS optimizations are implemented.**

---

# 9. Immediate Hardware/Compiler Boundary Case: Large Tiles Do Not Naturally Match MMA Fragments

The assumption that:

> Transfer writes flat row-major and MMA reads flat row-major, so no repacking is needed,

holds for a **single natively matched fragment**, but cannot be generalized to arbitrary GEMM tile dimensions.

For example:

```text
B tile = 32 × 64
MMA B fragment = 32 × 16
```

If the entire B tile is loaded into RF in contiguous row-major order, the row stride between elements of a `32×16` column slice is:

```text
64 × sizeof(dtype)
```

However, the MMA fragment decoder expects:

```text
16 × sizeof(dtype)
```

Changing only the fragment base register cannot bridge this stride discrepancy.

Under the current ISA, the compiler must select among the following alternatives:
1. Decompose matrix B into four distinct `32×16` transfers, staging each as a contiguous fragment;
2. Perform explicit layout repacking within the RF;
3. Introduce controlled strided-fragment addressing in future hardware iterations;
4. Modify internal microtiling geometries.

**Version 1 should adopt Strategy 1, avoiding premature revisions to the MMA ISA.**

However, this alters:
- DMA command counts;
- Barrier operation counts;
- Register window allocations;
- Descriptor and block-shape planning;
- RF live ranges;
- Transfer dispatch overhead.

This demonstrates that:

> **Layout selection, communication protocols, and pipeline resource budgets must be co-designed rather than handled by decoupled passes.**

Additionally, `W`, `V base`, and `A` operands are largely encoded as static immediate indices. Dynamic indexing such as:

```text
slot = i % 2
```

cannot dynamically compute arbitrary vector-register base operands at runtime.

Version 1 must resolve this by:
- Unrolling loops across the pipeline stage depth;
- Or generating minimal branched variants;

to ensure all W/V/A register indices are fully resolved at compile time.

---

# 10. Resource Allocation: Standard Liveness Intervals Are Insufficient for TNCA

## 10.1 Whole-Program Allocation vs. Role-Local Allocation

### Whole-Program Resources

Allocated globally:
- Shared RF windows;
- Accumulator slots;
- Barriers;
- Descriptor table slots;
- Command/completion tracking budgets;
- RF alias mappings.

### Role-Local Resources

Allocated independently per role:
- Scalar RF;
- Predicate RF;
- Private vector temporaries;
- Standard control-flow state.

Roles cannot be passed independently to a standard register allocator with the expectation that shared RF allocations will harmonize automatically.

## 10.2 Asynchronous Liveness Ends at Completion Events

For example, issuing an asynchronous instruction:

```text
m.mma.async issue
```

does not mark the last-use point of the source fragment.

The actual physical lifetime spans:

```text
First write
→ DMA completion
→ Consumer acquires read authority
→ Last asynchronous consumer reaches source-reusable
→ Buffer slot can be safely reused
```

The compiler may conservatively bind `source-reusable` to `compute-done` for current silicon/RTL targets, but static cycle estimations must never substitute for explicit completion events.

## 10.3 Fallback Order Under Resource Exhaustion

When physical resources are exhausted, the compiler should fall back in the following order:
1. Reduce RF pipeline stages;
2. Reduce concurrently active independent output tiles;
3. Apply finer legal microtiling;
4. Select an alternative role partitioning strategy;
5. Rematerialize inexpensive compute expressions;
6. Spill explicitly to global memory;
7. Emit a resource exhaustion diagnostic error.

Note: The compiler must not silently modify the user's `BLOCK_M` after TTIR generation, as that alters the work distribution governed by `program_id`. The compiler may perform valid internal sub-tiling or guide autotuning to select a different kernel launch configuration.

## 10.4 Cost Model Must Accumulate Across Shared Execution Resources

For a steady-state pipelined iteration:

\[
II \ge
\max_r
\frac{\sum_{\text{all roles}} \text{Demand on resource } r}
     {\text{Capacity of resource } r}
\]

subject to recurrence loop bounds and event dependency latencies.

According to the repository `Readme.md`, the current Matrix unit is a **16 MAC/cycle** implementation. Counting each MAC as two floating-point operations, the theoretical throughput ceiling is **32 FLOP/cycle**; Context Specialization performance cannot be evaluated using early reference assumptions of 512 FLOP/cycle.

Consequently, the utility of dual mainloops must be substantiated by concrete measurements:
- Does it eliminate Matrix engine starvation?
- Does it achieve effective overlap with vector operations?
- Or does it merely introduce RF bloat and synchronization overhead?

---

# 11. Code Generation: Neither Rewrite LLVM nor Let a Full LLVM Target Block Bring-Up

A two-phase code generation strategy is recommended:

## 11.1 Bring-Up Path: Reuse Existing Assembler

The earliest executable bring-up path:

```text
TNCA low-level IR
→ Conservative scheduling and restricted static allocation
→ TNCA assembly text
→ Existing tnca_asm.py
→ RTL simulation / board
```

Advantages:
- Immediately reuses verified instruction encodings and reserved-bit validation;
- Allows instruction-by-instruction differential comparison with hand-crafted assembly;
- Clearly decouples compiler lowering bugs from binary encoding errors.

Complex general-purpose machine optimization passes should not be built during this stage. When physical resources are exceeded, emit an explicit error rather than compromising asynchronous lifetime correctness.

## 11.2 Long-Term Path: Thin LLVM TNCA Target

To support production-grade:
- General register allocation;
- Spill/reload code generation;
- Machine instruction scheduling;
- Peephole optimizations;
- Native instruction encoding and disassembly;

evaluate a thin LLVM backend target rather than reinventing machine-level compiler infrastructure within MLIR. LLVM's target architecture cleanly integrates register classes, instruction patterns, and generic register allocators ([llvm.org](https://llvm.org/docs/WritingAnLLVMBackend.html?utm_source=openai)).

However, note:

> Standard LLVM register allocators do not understand cross-RCTX, asynchronous RF window ownership.

The clean separation of concerns is:

```text
TNCA whole-program planner
    Locks down shared windows, accumulators, and barriers

LLVM target
    Allocates role-local S/P/V temporary registers and emits machine instructions
```

Until the LLVM target matures, conservative scheduling barriers must guard asynchronous boundaries, ensuring generic machine schedulers never hoist RF overwrites before `source-reusable` events.

## 11.3 Single Source of Truth for ISA Specification

The repository already provides machine-readable definitions in:

```text
tools/asm/tnca_isa.py
```

This should be leveraged or normalized into a shared schema to generate:
- Assembler and disassembler lookup tables;
- Compiler opcode definitions;
- Encoding legality validators;
- Future TableGen input files;
- Conformance test cases.

Avoid manually maintaining disparate, independent tables of opcode numbers.

---

# 12. Standard Triton Lowering Boundaries That Must Be Handled Early

These issues are orthogonal to Warp Specialization, but dictate whether broad Triton compatibility is practical:

| Issue | TNCA Lowering Recommendation |
|---|---|
| Tensor size > 16 elements | Decompose across multiple vector registers while preserving original tensor indexing |
| Uniform FP32 scalars | Do not assume Scalar ALU has FP hardware; execute in vector domain or pass raw bit patterns |
| Masked load `other=` operand | Emit explicit passthrough / select; never assume unwritten RF registers retain target values |
| 64-bit tensor pointers | Recognize uniform base pointer + bounded offset representation |
| Gather offsets exceeding signed 32-bit | Address rebasing, chunking, or per-lane scalar fallback |
| Custom reductions | Retain combiner region; emit shuffle + combine trees |
| Floating-point reductions | Select deterministic reduction tree matching target numeric contract; avoid arbitrary reassociation |
| Missing hardware SFU | Software mathematical expansion or explicit unsupported diagnostic; avoid trapping SFU opcodes |
| Missing atomics / fences | Emit capability diagnostic; never lower unsafely to plain loads/stores |
| TensorDescriptor rank > 2 | Flatten dimensions when provably contiguous; otherwise fall back to loops / LSU |
| Descriptor `nan` padding | TNCA zero/clamp/fault padding modes are not equivalent; pad explicitly or fall back |

In particular, software math approximations for SFU functions need not be designed from scratch: MLIR includes polynomial expansions that serve as viable candidates. However, target-dependent corner cases, error bounds, and special-value semantics (NaN, inf) must be validated against the ISA precision contract ([raw.githubusercontent.com](https://raw.githubusercontent.com/llvm/llvm-project/main/mlir/lib/Dialect/Math/Transforms/PolynomialApproximation.cpp)).

Similarly, Triton's `TensorDescriptor` semantic interface can be reused, but NVIDIA's binary descriptor format cannot. Existing Python descriptor and runtime code decomposes shape, stride, and padding metadata, providing an ideal interception point for lowering to TNCA's 256-byte format ([raw.githubusercontent.com](https://raw.githubusercontent.com/triton-lang/triton/main/python/triton/tools/tensor_descriptor.py)).

---

# 13. Immediate Feedback Items for Hardware / ISA Design

The following items should be resolved prior to compiler bring-up rather than deferred until FlashAttention:

## 13.1 Is Shared RF "Visibility" Truly Addressable?

The hardware specification must clarify:

```text
R0 writes W0 mapped to a physical V range.
How does R1's MMA instruction address that identical physical range?
```

If each RCTX operates over strictly partitioned, non-overlapping register address spaces, stating that "RF is physically shared" is insufficient to realize zero-copy inter-role communication.

The specification must define:
- Whether W base represents a logical or physical index;
- How each RCTX's logical view establishes shared aliases;
- How compiler manifests specify these shared mappings;
- Whether standard vector instructions and MMA share an identical addressing scheme.

## 13.2 What Exactly Does `mb.arrive.release` Cover?

The hardware contract must distinguish between:
- Instruction issued;
- Register writeback complete in RF;
- Asynchronous operation complete;
- Global memory store visible to memory hierarchy.

The compiler cannot assume that in-order scalar instruction issue implies completion of preceding multi-cycle vector writebacks or asynchronous memory requests. While the compiler explicitly awaits async completion, hardware must fulfill its defined release semantics.

## 13.3 Bootstrapping Initialization Must Not Race with Waiters

When multiple roles launch concurrently:
- Which role initializes barriers?
- Which role defines register windows?
- Which role configures descriptors?
- When are secondary roles permitted to wait on or consume these resources?

A deterministic bootstrapping protocol is required to eliminate race conditions such as:

```text
R1 executes mb.wait
R0 subsequently executes mb.init, clearing R1's waiter state
```

## 13.4 Wide Implicit Register Reads Must Enter the Compiler Model

The compiler's liveness model must account for wide implicit register ranges:
- MMA reading contiguous fragment registers starting from base;
- Tensor DMA reading `rank` scalar coordinate registers starting from `Sb`;
- Asynchronous stores reading source windows over deferred intervals.

Register liveness cannot be modeled purely from single explicit register indices in opcode bitfields.

## 13.5 BF16 Numerical Specification Requires Correction

The statement:

> "BF16×BF16 product is exactly representable in binary32"

does not hold unconditionally. Because BF16 and FP32 share the same 8-bit exponent range, their product can overflow or underflow the FP32 exponent range; having sufficient significand bits does not guarantee sufficient exponent headroom ([docs.riscv.org](https://docs.riscv.org/reference/isa/extensions/bf16min/_attachments/riscv-bfloat16.pdf)).

The numeric contract should be formally defined as:

\[
C_{k+1}
=
\operatorname{RNE}_{f32}
\left(
\operatorname{exact}(A_k)\operatorname{exact}(B_k)+C_k
\right)
\]

representing true widened-input FP32 fused multiply-accumulate semantics without intermediate rounding of the product.

A test case to add to the numerical oracle:

```text
A = 2^127   (representable in BF16)
B = 2
C = -2^127

Fused result: 2^127
If product is rounded prematurely to FP32: overflows to +inf
```

This is primarily a specification correction; reusing the `tnca_fp32_fma` implementation is likely already numerically sound.

---

# 14. Recommended Pass Pipeline

The following pass names outline the recommended engineering modularization:

| Stage | Input → Output | Core Tasks |
|---|---|---|
| 1. `tnca-check-target` | TTIR | Target capability, dtype, numeric mode, and control-flow legality checks |
| 2. Upstream normalization | TTIR → TTIR | Inlining, canonicalization, CSE, constant propagation |
| 3. `tnca-analyze-access` | TTIR | Regular access patterns, masks, aliasing, descriptor candidates |
| 4. Layout & whitelisted opt | TTIR → Layout IR | Reused layout encodings and layout propagation |
| 5. `tnca-select-fragments` | Layout IR | MMA microtiling, packed layouts, DMA transfer geometries |
| 6. `tnca-plan-contexts` | Layout IR | Partitioning, rematerialization, RCTX candidate generation |
| 7. `tnca-build-channels` | → Execution IR | RF/ARF communication channels and ownership materialization |
| 8. `tnca-plan-pipeline` | Execution IR | Pipelining stage depth, command overlap, dependency scheduling |
| 9. `tnca-plan-resources` | Execution IR | RF/ARF, barrier, and descriptor feasibility validation |
| 10. `tnca-expand-commands` | → Low-level async IR | Concrete DMA and MMA command expansion |
| 11. `tnca-lower-protocols` | → Low-level execution IR | Arm, count, epoch, wait, arrive protocol materialization |
| 12. `tnca-outline-roles` | → Per-role CFG | Parameter capture, entry points, termination drainage |
| 13. Instruction selection & RA | → Machine layer | S/P/V register allocation, instruction constraints, branch lowering |
| 14. `tnca-verify-machine` | Machine layer | Final validation of encoding, resources, and async lifetimes |
| 15. Emission | Binary + manifest | Emitted via assembler, transitioning to LLVM target |

Stages **5 through 9 do not form a strict one-way linear pipeline**.

A bounded candidate exploration should be employed:

```text
Layout / microtile / role / stages candidates
    ↓
Resource feasibility and cost estimation
    ↓
Select optimal valid plan
    ↓
Freeze plan before materializing synchronization protocols
```

Do not alter pipeline stage counts unconstrained once physical barriers have been instantiated.

For structural loop prologue/epilogue rewriting, MLIR utilities such as `pipelineForLoop` can be reused; their interfaces accept caller-specified schedules and predication, though they do not verify schedule soundness ([raw.githubusercontent.com](https://raw.githubusercontent.com/llvm/llvm-project/main/mlir/include/mlir/Dialect/SCF/Transforms/Transforms.h)).

---

# 15. Implementation Roadmap: Six Verifiable Milestones

Development should proceed across six concrete milestones:

## C0: Fixed Toolchain and Target Contract

Deliverables:
- Pinned Triton commit;
- Pinned corresponding LLVM revision;
- Out-of-tree backend skeleton;
- `tnca-opt` driver tool;
- `TargetInfo` specification;
- Integration with unified ISA schema;
- Per-pass IR dumping infrastructure.

Do not track upstream `main` continuously. Use Triton 3.8.0 as a stable baseline candidate, moving to pinned commits only when specific WS fixes are required. Upstream Triton requires building against its pinned LLVM revision rather than arbitrary LLVM versions ([raw.githubusercontent.com](https://raw.githubusercontent.com/triton-lang/triton/main/README.md)).

The target configuration must specify at least:

```text
RCTX count
VLEN
physical/logical V register limits
accumulator count
DMA rank / dtype / layout capabilities
MMA throughput and command constraints
barrier count / count width
source-release capability
SFU / atomics / fence capabilities
```

## C1: First Real Triton → RTL Closed Loop

Initial coverage:
- Vector add;
- Masked elementwise operations;
- Integer reductions;
- FP32 reductions;
- Basic control flow;
- Supported subset of gather / scatter.

Requirement:

> Compiled binaries must run directly on RTL without manual assembly edits or manually inserted barriers.

Bring-up should rely on Ahead-of-Time (AOT) compilation and a simulation launcher without requiring complete PyTorch device integration as a prerequisite.

## C2: Single-RCTX Tensor DMA + MMA

Before introducing automatic Context Specialization, complete:
- FP16 GEMM;
- BF16 GEMM;
- Arbitrary memory strides;
- N/M/K tail dimensions;
- Microfragment layout generation;
- Accumulator epilogue handling.

This milestone must surface and resolve issues where large B tiles diverge from contiguous B fragment layouts.

## C3: Explicit Two-Role Pipeline

Role partitioning should initially be explicitly declared via tests or compiler flags rather than derived automatically.

Deliverables:
- Ready / Free double buffering;
- Handling of 0, 1, 2, and odd iteration counts;
- Predicated-off asynchronous commands;
- Out-of-order memory responses;
- Deferred completions;
- Error propagation;
- Termination draining.

**This is the most critical milestone of Phase 1.**

## C4: Integrate Automatic Partitioning Infrastructure

At this stage, extract upstream:
- Dataflow graph construction;
- Partition propagation;
- Loop cloning;
- Producer / consumer dependence analysis.

Validate first on GEMM-style loops before expanding control-flow coverage. Once the 4-RCTX hardware profile becomes available, verify:

```text
P + M0 + M1 + E
```

## C5: Attention and Performance Feedback

Incorporate:
- Software math fallback implementations;
- Native hardware SFU lowering;
- Overlapping QK / softmax / PV pipelines;
- Decoupled K and V channels;
- ARF and vector register interactions;
- Empirical comparison of role templates.

Finally, derive Stage 4 cache/flow metadata directly from the materialized schedule.

---

# 16. Verification and Diagnostics Must Be Formal Backend Outputs

## 16.1 Resource and Bottleneck Report for Every Kernel

The compiler should emit detailed resource profiles:

```text
active contexts:          2
RF stages:                2
shared RF windows:        ...
private vector bytes:     ...
accumulator slots:        ...
barriers:                 ...
max async commands:       ...

layout conversion bytes: ...
DMA command expansion:   ...
predicted issue pressure: ...
source-release policy:   compute_done

rejected 4-context plan:
    insufficient accumulator capacity
```

This diagnostic report provides actionable guidance for hardware/software co-design compared to coarse register usage metrics.

## 16.2 Three-Tier Correctness Verification

### IR Unit Tests

Validate:
- Layout mappings;
- Role capture semantics;
- Channel instantiation;
- Generation calculation formulas;
- Barrier operation counts;
- Diagnostics for unsupported operations and types.

### Protocol Model Tests

Construct lightweight ownership and event models rather than maintaining a full Triton interpreter. Exhaustively enumerate:
- 1–3 pipeline stages;
- 1–2 consumers;
- Arbitrary completion orderings;
- Short loop iteration bounds;
- All-false predicated paths;
- Fault conditions.

### RTL Differential Verification of Compiled Artifacts

Leverage:
- Existing assembler toolchain;
- Exact FP reference oracle;
- Randomized memory latency and interconnect backpressure injection;
- Existing NUCA/mesh testbench environment.

Separate numerical equivalence testing from bit-exact instruction verification; TNCA's `dot` accumulation order cannot be assumed bit-identical to arbitrary GPU targets.

## 16.3 ConSan Deserves Study, but Cannot Be Enabled Out-of-the-Box

Upstream includes TritonInstrument/ConSan, using target hooks to model asynchronous memory accesses and barrier semantics; however, its buffer analysis and instrumentation target shared memory and TMEM ([raw.githubusercontent.com](https://raw.githubusercontent.com/triton-lang/triton/main/include/triton/Dialect/TritonInstrument/IR/TritonInstrument.md)).

The recommended adoption strategy:
1. Implement RF/ARF ownership validators on simulation execution traces first;
2. Adapt ConSan's visibility analysis and test methodology;
3. Extend TNCA target hooks and RF footprint tracking if interfaces align.

Never model RF as software-addressable shared memory merely to enable ConSan unmodified.

---

# Final Recommendations: Make These Five Engineering Decisions First

The following five architectural decisions should be frozen immediately:

1. **Adopt a hybrid IR: Retain `tt/arith/math/scf`; let TNCA introduce only resource and asynchronous execution semantics.**
2. **Prioritize reusing TTGIR and LinearLayout for standard layouts; never encode Role Context counts into warp layout dimensions.**
3. **Model Context Specialization via channels, leases, and typed completion events rather than passive role annotations.**
4. **Achieve an end-to-end closed loop with explicit two-role pipelining before attempting to port automatic partitioning algorithms.**
5. **Reuse the existing assembler in the near term; target a lean LLVM backend long term rather than building a redundant machine compiler.**

For this accelerator architecture, the primary challenge is not distributing load, MMA, and softmax across distinct roles.

The true challenge is:

> **Determining RF slice ownership post-partitioning; identifying when asynchronous operations cease reading sources; coordinating release across multiple consumers; ensuring physical registers can be safely recycled; and maintaining these invariants across layout optimizations, loop transformations, and register allocation.**

Therefore, the first TNCA IR specification to formally design and review must govern **RF ownership, channels, and completion semantics**. Once this protocol is established, Triton's existing layout, partitioning, and loop infrastructure can serve as reusable assets rather than sources of invalid GPU-centric assumptions.
