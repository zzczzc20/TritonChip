# Description
This document is the architectural and ISA design draft for TNCA-v1 (originally drafted with GPT-6-astra (max) in Chinese and translated into professional technical English). It details the architectural principles, microarchitecture, register file organization, TensorMap specification, NUCA cache mechanisms, ISA definition, compiler lowering strategy, and verification methodology.

# Contents

## Executive Summary

Before detailing the proposal, the complete mechanisms of both research baselines (LLaMCAT and DCO) were verified alongside practical Triton/MLIR requirements regarding data layout, asynchronous memory access, and synchronization. The proposal distinguishes between **mandatory semantics** and **optional performance mechanisms**, focusing specifically on whether an efficient, deadlock-free pipeline can be established among registers, NUCA, TMA, and a small number of dedicated execution contexts after eliminating Scratchpad Memory (SPM).

Two critical architectural boundaries directly impact this proposal:
1. **DCO's datapath still contains core-private SPM.** Completely eliminating SPM requires reorganizing operand reuse and layout transformations.
2. **A small number of dedicated contexts cannot use coarse "halt the fastest core" throttling.** Doing so risks halting consumers or resource-freeing paths, creating deadlock. Throttling must instead regulate the admission of new memory requests, while strictly distinguishing three distinct events: *data arrived in registers*, *computation complete*, and *registers reusable*.

### Architectural Positioning

This chip is positioned as:

> **A Tensor-Native Accelerator (TNCA) utilizing TensorMap as its dataflow contract, register files as its computational working set, a single-level NUCA as shared on-chip storage, and a small number of dedicated execution contexts driving asynchronous execution pipelines.**

TNCA is neither a "GPU stripped of Shared Memory" nor an "NPU renaming SPM as Cache".

**Four core interfaces are redesigned:**
1. **Triton tensor/layout semantics → registers and instructions.**
2. **TensorMap → NUCA cache management and request arbitration.**
3. **Asynchronous data movement → register ownership and completion events.**
4. **Inter-core progress → request admission, prefetch distance, and shared working set regulation.**

The research baselines referenced are **LLaMCAT v1 (November 26, 2025)** and **DCO v2 (September 10, 2026)**. The architecture, ISA, and parameters presented below constitute a design draft built on these works, and **do not represent verified silicon implementations or measured performance figures** ([arxiv.org](https://arxiv.org/abs/2512.00083)).

Crucially, **the datapath in the DCO paper still includes private SPM; eliminating SPM entirely is an architectural step beyond that paper** ([arxiv.org](https://arxiv.org/html/2512.07312v2)). Consequently, compute-side bandwidth, layout transformation, and pipelining assumptions from the paper cannot be inherited directly.

This proposal is designated **TNCA-v1**.

---

# 1. Six Architectural Principles

## 1.1 The Sole Data Storage Hierarchy is RF ↔ NUCA ↔ DRAM

```text
                 Compute Units
                      │
          Scalar / Vector / Accumulator RF
                      │
          Tensor DMA / Standard Load-Store
                      │
                 Mesh Network
                      │
       Single-Level Distributed Shared NUCA Cache
                      │
               DRAM Controllers
                      │
                  HBM / DDR
```

Here, the Register File (RF) hierarchy encompasses:
- Scalar registers;
- Vector registers;
- Matrix accumulator registers;
- Register groups directly addressed by instructions.

**Explicitly excluded:**
- Software-addressable Shared Memory;
- Software-allocated on-chip Tensor SRAM;
- Cache-as-SPM regions requiring software residency management;
- Cache objects requiring pin/unpin for correct functional execution.

Pipeline FIFOs, MSHRs, descriptor tables, and instruction buffers are permitted, but they are microarchitectural structures rather than software-addressable data storage tiers.

> If independent instruction buffers are disallowed, instruction fetch traffic must be accounted for against NUCA and mesh bandwidth. This document bounds the **data storage hierarchy**, allowing small instruction buffers.

## 1.2 Matrix Array Must Not Rely on Cycle-by-Cycle Remote Cache Supply

The first version must avoid the following execution paradigm:

```text
MMA reads the next operand set directly from NUCA every compute cycle
```

The recommended model is:

```text
NUCA → Asynchronous load into RF → MMA reuses RF data across multiple cycles
```

The rationale is direct:
- NUCA hit latency is variable;
- Mesh networks experience backpressure;
- Cache bank conflicts occur dynamically;
- Fill, writeback, and hit traffic share storage ports;
- Matrix inner loops require completely predictable operand delivery.

**Cache manages reuse across tiles, cores, and iterations; RF manages reuse within the compute inner loop.**

Eliminating SPM does not eliminate local operand reuse.

## 1.3 "≤4 Threads" Defined as Four Vector Control Contexts (RCTX)

These execution contexts are termed **Role Contexts (RCTX)**.

Each RCTX maintains independent architectural state:
- Program Counter (PC);
- Scalar register file;
- Predicate state;
- Instruction issue queue;
- Event wait state.

All RCTXs on a tile share:
- SIMD execution units;
- Matrix execution units;
- Tensor DMA engine;
- Physical vector and accumulator register pools;
- NUCA network interface.

An RCTX controls an entire SIMD execution group (vector lanes), rather than executing a single scalar thread.

Typical role decomposition:

| RCTX | Typical Responsibilities |
|---|---|
| R0 | Descriptor handling, address generation, prefetch, input data movement |
| R1 | Matrix compute execution |
| R2 | Reductions, softmax, activations, epilogues |
| R3 | Output data movement, or secondary compute pipeline |

**These roles represent compiler scheduling conventions rather than rigid hardware bindings.**

For example, a small elementwise sigmoid kernel should not be forced into four active contexts or complex barrier synchronization.

## 1.4 Strict Distinction in TensorMap: Semantics vs. Hints

Semantic requirements and performance hints must be cleanly separated.

### Mandatory Semantics
- Base address;
- Shape / dimensions;
- Strides;
- Data type (dtype);
- Boundary handling / out-of-bounds padding;
- Effective memory addresses;
- Register target destinations;
- Asynchronous completion signaling;
- Memory ordering.

### Optional Performance Hints
- Expected reuse count;
- Last-use indication;
- Cache priority / tier;
- Bypass preferences;
- Prefetch distance;
- Sharing group ID;
- Progress dimension markers.

**Cache prediction errors or hint mispredictions must affect only performance, never functional correctness.**

In particular:
> `dead` must only indicate "prioritize for cache line replacement", never "directly drop dirty data without writeback".

## 1.5 ISA Should Align with TTIR/TTGIR Without Hardware Ossification

The compilation layering should be structured as:

```text
Triton Source Program
    ↓
TTIR: Tensor, operator, and control-flow semantics
    ↓
Target-specific layout / pipeline / role partitioning
    ↓
TNCA IR: RF windows, asynchronous transactions, MMA fragments, barriers
    ↓
TNCA ISA
```

TTGIR inherently contains layout conversions, shared-memory allocations, asynchronous copies, and warp-specialization primitives; many carry explicit GPU memory and execution assumptions that must not be mapped mechanically 1:1 into hardware ([triton-lang.org](https://triton-lang.org/main/dialects/TritonGPUOps.html)).

## 1.6 Decomposing "Broad Triton Support" into Three Commitments

| Tier | Commitment |
|---|---|
| High-level operators | Native support for standard workloads: GEMM, Conv, FlashAttention, LayerNorm, Softmax, Sigmoid, etc. |
| Portable Triton semantics | Broad feature completeness, allowing software expansion for unsupported primitives. |
| Vendor-specific code | No native execution of NVIDIA/AMD target-specific intrinsics (e.g., PTX inline assembly); requires rewriting or target lowering. |

For example, `inline_asm_elementwise` embedding PTX assembly is vendor-specific and is not inherently portable across architectures ([triton-lang.org](https://triton-lang.org/main/python-api/generated/triton.language.inline_asm_elementwise.html)).

---

# 2. Most Likely Performance Bottlenecks

| Bottleneck | Architectural Root Cause | Mitigation Strategy |
|---|---|---|
| RF Bandwidth | Without SPM, DMA, MMA, and vector units all compete for RF ports | Banked RF, window layouts, operand retention latches, port scheduling budgets |
| RF Capacity | Asynchronous double buffering, attention intermediates, and accumulators live concurrently | Decouple accumulator RF, reduce tile size, restrict pipeline depth |
| NUCA Hit Bandwidth | Cache hits still traverse tag/data banks and mesh interconnect | Intra-core RF reuse, bank-aware layout, proximate scheduling |
| Mesh Bisection Bandwidth | Distributed capacity does not equal uniform bisection bandwidth | Home-aware placement, sharing group scheduling, restrict remote traffic |
| Insufficient MSHR Entries | 4 RCTXs do not inherently provide memory-level parallelism (MLP) | DMA expands bulk requests into numerous independent cache-line transactions, credit flow control |
| Insufficient MSHR Targets | Shared accesses (e.g., GQA) coalesce onto few misses | Dedicated target budgets, merge arbitration, response fanout control |
| Return Path Blocking | Returning DMA data blocked when RF windows or input FIFOs fill | Pre-reserve destinations prior to request issue, separate return credit path |
| TensorMap Processing Overhead | Complex multi-dimensional tensor address math per cache line is costly | Descriptor snapshots, hierarchical metadata, separate address gen from policy lookup |
| Layout Conversion | Absence of on-chip shared memory buffer | In-RF shuffle, TMA destination layout swizzling, controlled memory fallback |
| SFU / Reductions | FlashAttention includes extensive non-GEMM math | Provision sufficient exp2, reduction trees, and vector-accessible accumulators |
| Throttling Oscillation | Multiple distributed controllers adjusting cache and injection policies | Timescale decoupling, hysteresis thresholds, unified pressure metrics |
| Barrier / Ownership Overhead | Few contexts are vulnerable to synchronization critical paths | Fine-grained completion events, sleep/wakeup scheduling, static lifetime analysis |
| Small Kernel Overhead | Descriptor setup and multi-role pipelines exceed compute time | Single-RCTX fast path, standard vector load/store instructions |
| Irregular Memory Access | TMA cannot address arbitrary gather/scatter patterns | Retain general-purpose LSU with coalescing |
| Temporary Spill Traffic | Cache absorbs spills previously kept in shared memory or registers | Compiler cost model, dedicated low-priority spill streams |

---

## 2.1 Hazard 1: RF Replaces SPM as the New Bottleneck

Assuming a matrix unit processes a \(16\times16\) outer product each cycle:
- Input A: 16 FP16/BF16 elements (32 bytes);
- Input B: 16 FP16/BF16 elements (32 bytes).

Input bandwidth requirement:

\[
B_{\text{MMA-input}} = (16+16)\times 2 = 64\ \text{B/cycle}
\]

This excludes:
- SIMD operand reads;
- SIMD result writes;
- DMA writeback fills;
- Accumulator-to-SIMD transfers;
- Layout conversion shuffle traffic.

One cannot simultaneously assume: small RF, few ports, full-speed MMA, full-speed SIMD, full-speed DMA fills, and conflict-free overlap.

**Recommendation: Design realistic RF porting and banking first before finalizing theoretical compute throughput.**

## 2.2 Hazard 2: High Cache Hit Rate Does Not Guarantee High Performance

In this architecture, a NUCA cache hit undergoes:

```text
Request injection wait
→ Multi-hop mesh traversal
→ Slice request queue wait
→ Tag lookup
→ Data bank conflict resolution
→ Response network traversal
→ RF write-port arbitration
```

Thus:

\[
\text{hit rate} \quad\neq\quad \text{hit throughput} \quad\neq\quad \text{useful operand delivery rate}
\]

The primary optimization objective is:
> **Effective tensor bytes delivered to active compute register windows per unit time, and completed logical tiles per second.**

## 2.3 Hazard 3: Four Contexts May Fail to Hide Memory Latency

Four contexts are not inherently a bottleneck; **having only four outstanding cache misses is the bottleneck**.

Consider:
- DRAM effective bandwidth: 256 GB/s;
- Average round-trip read latency: 300 ns;
- Cache line size: 64 B.

The minimum number of concurrent in-flight cache misses required to saturate memory bandwidth is:

\[
N_{\text{miss}} \geq \frac{256\times 10^9 \times 300\times 10^{-9}}{64} = 1200
\]

This is a chip-wide requirement that cannot be sustained by four scalar threads issuing blocking requests.

Therefore:
- RCTX count can remain small;
- Issued DMA command count can remain modest;
- **Each DMA command must expand into numerous independent cache-line transactions.**

## 2.4 Hazard 4: Flawed Throttling May Block Resource Release

LLaMCAT employs MSHR-aware arbitration, progress balancing, and hierarchical throttling; in GPUs, throttling adjusts active thread block allocation ([arxiv.org](https://arxiv.org/html/2512.00083v1)).

In TNCA:

```text
R0: Produces data
R1: Consumes data
R2: Releases intermediate resources
```

Halting an entire core or indiscriminately pausing the context with the most requests can halt the consumer or releaser:

```text
Consumer halted
→ RF window cannot be released
→ Incoming return data cannot land
→ Misses cannot retire
→ Cache congestion exacerbates
```

**Rule: Throttle admission of new work, new prefetches, and new misses; never throttle in-flight completion, consumer execution, or resource release paths.**

---

# 3. Recommended Overall Chip Architecture

## 3.1 Reference Configuration: 16 Tiles, 4×4 Mesh

The configuration below serves as an architectural baseline for design space exploration rather than a finalized silicon implementation.

| Parameter | TNCA-v1 Reference Value |
|---|---:|
| Tile Count | 16 |
| Interconnect Mesh | 4×4, 2D mesh, bidirectional |
| RCTX per Tile | Up to 4 |
| SIMD Width | 16 lanes |
| Matrix Base Shape | M16×N16, K16/32/64 |
| Vector RF | 32 KiB per tile |
| Accumulator RF | 16 KiB per tile |
| Scalar RF | 4 × 64 × 64-bit |
| NUCA Capacity | 512 KiB per slice (8 MiB total chip-wide) |
| Cache Line Size | 64 B |
| Associativity | 8-way set associative |
| Data Banks per Slice | 4 |
| MSHR Entries per Slice | 96 (baseline) |
| Max In-Flight Line Transactions per Core | 128 (baseline) |
| Active Tensor DMA Commands per Core | 8 (baseline) |
| Hardware mbarriers per Core | 32 |
| Command / Completion Slots per Core | 64 |
| TensorMap Snapshots per Core | 16 |
| Mesh Data Link Width | 512-bit payload per direction, plus control |
| Target Clock Frequency | 1.0 GHz (for modeling) |
| External Memory Interface | Parameterized (independent of specific HBM generation) |

### Top-Level Physical Organization

```text
┌───────────────────────────────────────────────────────────┐
│ Host Interface / Command Processor / Grid Scheduler        │
│ Launch descriptors / Argument buffers / Resource admission │
└───────────────────────────┬───────────────────────────────┘
                            │
       ┌──────────┬──────────┬──────────┬──────────┐
       │ Tile 00  │ Tile 01  │ Tile 02  │ Tile 03  │
       ├──────────┼──────────┼──────────┼──────────┤
       │ Tile 10  │ Tile 11  │ Tile 12  │ Tile 13  │
       ├──────────┼──────────┼──────────┼──────────┤
       │ Tile 20  │ Tile 21  │ Tile 22  │ Tile 23  │
       ├──────────┼──────────┼──────────┼──────────┤
       │ Tile 30  │ Tile 31  │ Tile 32  │ Tile 33  │
       └──────────┴──────────┴──────────┴──────────┘
                            │
               Distributed Memory Controllers
```

### Tile Internal Microarchitecture

```text
┌────────────────────────────────────────────────────────────┐
│  RCTX0    RCTX1    RCTX2    RCTX3                            │
│    └──────── Ready/Scoreboard Scheduler ────────┘            │
│                │          │          │                      │
│             Scalar      SIMD       Matrix                   │
│                │          │          │                      │
│        Scalar RF    Banked Vector RF    Accumulator RF      │
│                         │                                   │
│          Tensor Address/Transfer Engine + General LSU        │
│                         │                                   │
│              Credits / Injection / Return Unit              │
├─────────────────────────┼───────────────────────────────────┤
│                       Router                                │
├─────────────────────────┼───────────────────────────────────┤
│  NUCA slice: tags / data / MSHR / targets / WB / local TMU    │
└────────────────────────────────────────────────────────────┘
```

## 3.2 Avoid a Centralized Chip-Wide TMU Hotspot

TensorMap management is partitioned across three tiers:
1. **Core-side TensorMap Frontend**: Handles descriptor parsing, coordinate expansion, and target RF window allocation.
2. **Slice-local TMU**: Handles cache line reuse hints, dead markers, bypass decisions, and local slice pressure.
3. **Low-frequency Group Controller**: Manages sharing-group progress tracking, injection rate budgets, and tile placement affinity.

Global request streams must not bottleneck on a single monolithic TensorMap unit.

---

# 4. Core Microarchitecture: Small Control Core, High Async Transaction Capacity

## 4.1 Frontend and Issue

Recommended microarchitecture:
- In-order execution per RCTX;
- No Reorder Buffer (ROB);
- No speculative register renaming;
- Scoreboard tracking data hazards;
- Up to dual-issue per cycle across complementary resource classes:
  - Slot 0: Control / Scalar operation;
  - Slot 1: Vector, Matrix command, or memory access command.

Matrix engines, DMA channels, and SFU pipelines advance autonomously once initiated. This enables asynchronous latency hiding without the area and power overhead of out-of-order execution logic.

### Recommended Issue Priority
1. Consumers capable of releasing occupied RF windows;
2. Instructions committing final results or advancing barrier state;
3. Computation instructions whose operands are ready;
4. Demand memory load/store operations;
5. Speculative prefetch commands.

This order optimizes for throughput and forward progress without replacing fairness guarantees.

## 4.2 Four Contexts Need Not Belong to Four Separate Programs

The architecture supports two dispatch modes:

### Mode A: Single Program, Multiple Roles
```text
One Triton Program Instance
    ├── R0: Producer (TMA / address generation)
    ├── R1: Matrix compute engine
    ├── R2: Vector epilogue / activations
    └── R3: Store / control flow
```
Optimal for GEMM, FlashAttention, and Convolutions.

### Mode B: Multiple Independent Small Programs
```text
R0 → Program p
R1 → Program p+1
R2 → Program p+2
R3 → Program p+3
```
Optimal for elementwise kernels and simple reductions.

Hardware resources are statically partitioned at launch; provisioning four program counters does not replicate underlying physical execution units.

## 4.3 Roles Are Not Hardware Thread Block Counts

> **Triton's `num_warps` configuration parameter must not be equated with the RCTX count.**

In Triton, `num_warps` governs logical tiling, collective communication, and parallel partitioning. RCTX represents distinct execution roles within a tile. The compiler backend must map logical program decomposition to physical role contexts.

---

# 5. Register Organization: The Critical Post-SPM Challenge

## 5.1 Three RF Classes

### Scalar RF
Per RCTX:
- 64 × 64-bit registers;
- Used for address arithmetic, loop indices, descriptor handles, and synchronization tokens;
- `s0` is hardwired to zero;
- Floating-point scalar values use the lower 32 bits or execute on lane 0 of the SIMD unit.

### Vector RF
Per tile:

\[
512\times 16\times 32\text{ bit}=32\text{ KiB}
\]

Organization:
- Each architectural vector register spans 16 lanes of 32-bit elements;
- FP16/BF16 data can be packed (32 elements per register);
- Each RCTX possesses a logical register mapping view;
- Physical registers are partitioned across active contexts at program launch;
- Producer contexts do not require an equal 1/4 allocation.

### Accumulator RF
Per tile:

\[
16\times 16\times 16\times 32\text{ bit}=16\text{ KiB}
\]

Comprising:
- 16 accumulator slots;
- Each slot stores a 16×16 FP32 submatrix tile.

Matrix internal pipeline stage registers are tracked independently from software-visible accumulator storage.

## 5.2 RF Window Is Register Aliasing, Not SPM

TNCA-v1 introduces the **RF Window** abstraction:

```text
W0 = Set of physical vector registers allocated at compile time
W1 = Another set of physical vector registers
```

A window descriptor specifies:
- Base physical register index;
- Allocation length;
- Element data format;
- Supported tile layout;
- Permitted accessing roles;
- Static stage/fragment partition boundaries.

Restrictions:
- No byte-addressable pointers are exposed;
- No arbitrary pointer arithmetic (`load [W0 + offset]`) is allowed;
- Arbitrary software objects cannot reside in windows;
- No physical "window SRAM" is added.

TMA and MMA instructions operate on windows and fragments; vector instructions reference architectural register identifiers directly.

**Without these constraints, the window would devolve into a re-labeled SPM.**

## 5.3 Implementable Starting Point for RF Ports

A conservative, area-efficient baseline configuration:
- 16 physical memory banks;
- 64-bit width per bank;
- 1 Read / 1 Write (1R1W) port per bank;
- 32 KiB total capacity.

Aggregate conflict-free bandwidth:

\[
B_\text{read}=128\text{ B/cycle}
\]

\[
B_\text{write}=128\text{ B/cycle}
\]

A 16-lane FP32 FMA instruction requires:
- Three input operands: $3 \times 64 = 192\text{ B}$;
- One output result: $64\text{ B}$.

Because single-cycle 3-operand reads exceed 128 B/cycle, single-cycle FMA throughput across all banks is not universally possible without conflict.

Recommended v1 mitigations:
- Two-cycle operand staging buffers;
- Operand latch registers;
- Interleaving FMA issues with independent instructions.

A SIMD width of 16 does not imply that every arithmetic operation achieves 16 retired results per clock cycle under all port configurations. If profiling demonstrates RF port contention as the dominant bottleneck, enhancements may follow:
- 2R1W banked SRAMs;
- Replicated read ports;
- Dedicated matrix operand staging banks.

## 5.4 Accumulator Must Be Directly Accessible by Vector Unit

Vector accessibility to accumulators is mandatory for FlashAttention. The ISA provides:

```text
a.vread      Vd, A_slot, row
a.vwrite     A_slot, row, Vs
a.mul.row    A_slot, row, scale
a.fma.row    A_slot, row, Vx, scale
```

`a.mul.row` may be implemented as a vector-microcode sequence without requiring dedicated matrix-side multipliers.

This enables:
- Online softmax rescaling of accumulator matrix $O$;
- Epilogue operations;
- Bias addition;
- Elementwise activations;
- Normalization post-processing;

all without round-tripping intermediate accumulators through NUCA.

---

# 6. Matrix Unit: Fixed Small Shapes to Avoid ISA Bloat

## 6.1 Basic Instruction Shapes

Version 1 fixes the hardware matrix instruction primitives:

```text
m.mma.f16.f32    M16 N16 K16/32/64
m.mma.bf16.f32   M16 N16 K16/32/64
```

Formal semantics:

\[
C_{16\times 16} \leftarrow C_{16\times 16} + A_{16\times K}B_{K\times 16}
\]

Larger $M$ and $N$ dimensions are tiled by the compiler.
Supporting $K \in \{16, 32, 64\}$ via unified internal loop counters eliminates dedicated datapaths for disparate tile geometries.

## 6.2 Accounting for FP32 Accumulation Feedback Latency

A 16×16 outer-product systolic or spatial array with 256 multiply-accumulate elements cannot assume that every PE updates the exact same FP32 accumulator register on consecutive cycles without accumulation latency.

Pipeline mitigations:
1. Interleaving independent accumulator matrix fragments ($C_0, C_1$);
2. Staggering partial sums within processing elements;
3. Performing reduction along the $K$ dimension prior to final accumulation.

The execution state machine follows:

```text
CAPTURE → MAC_RUN → PARTIAL_REDUCE → WRITE_C → COMPLETE
```

Assuming 16 tiles at 1.0 GHz with 256 MACs/cycle per tile, the theoretical peak arithmetic throughput is:

\[
P_\text{upper} = 16\times 256\times 2\times 10^9 = 8.192\text{ TFLOP/s}
\]

Sustained throughput depends on array utilization, RF port availability, and pipeline efficiency.

## 6.3 FP32 Dot Precision Cannot Be Silently Degraded

Triton's `tl.dot` interface defines explicit precision control parameters that backends must respect ([triton-lang.org](https://triton-lang.org/main/python-api/generated/triton.language.dot.html)).

Execution guarantees:
- FP16/BF16 inputs with FP32 accumulation: Native matrix execution datapath;
- Strict FP32 inputs: SIMD decomposition or dedicated emulation path;
- Downcasting modes: Explicitly requested by software;
- FP64: Software emulation;
- FP8 / INT8: Future architectural extensions.

Precision modes must be explicitly documented; native acceleration must not silently compromise IEEE numerical guarantees.

---

# 7. TensorMap: Dual-Targeted for Transfer and Cache, Maintaining Clean Boundaries

## 7.1 Decomposing Descriptors into Three Objects

A monolithic descriptor creates coupling between transfer mechanics and caching policies. Descriptors are decomposed into:

### A. Tensor Descriptor: What the Data Is
- Physical base address;
- Dimensions / extents;
- Strides;
- Data type;
- Boundary padding modes.

### B. Transfer Configuration: How Data Is Moved
- Sub-tile extents;
- Target register file layout;
- Packing / transposition modes;
- Destination RF window identifier;
- Fragment slicing rules.

### C. Cache/Flow Hint: How Data Is Expected to Be Used
- Expected reuse factor;
- Streaming classification;
- Sharing group affiliation;
- Logical progress dimension;
- Prefetch lookahead distance;
- Cache retention priority.

Triton's native tensor descriptor interfaces provide upstream source semantics, while TNCA binary descriptors maintain an independent hardware format ([triton-lang.org](https://triton-lang.org/main/python-api/generated/triton.language.make_tensor_descriptor.html)).

## 7.2 Recommended 256-Byte TensorMap Object

The primary descriptor format allocates 256 bytes, with the initial 128 bytes reserved for mandatory architectural semantics.

| Byte Range | Field | Description |
|---|---|---|
| 0–7 | `base_address` | 64-bit physical or base virtual address |
| 8–47 | `extent[5]` | Five 64-bit dimension extents |
| 48–87 | `byte_stride[5]` | Five signed 64-bit byte strides |
| 88–107 | `block_shape[5]` | Five 32-bit default block shape parameters |
| 108 | `rank` | Active tensor rank (1 to 5) |
| 109 | `dtype` | Data type encoding |
| 110 | `padding_mode` | Out-of-bounds fill rule (zero, replicate, clamp) |
| 111 | `semantic_flags` | Coherence, ordering, and synchronization flags |
| 112–119 | `storage_uid` | Unique storage allocation identifier |
| 120–127 | `descriptor_version` | Monotonic version counter |
| 128–191 | `cache_flow_hints` | Performance and flow control hints (Section 7.3) |
| 192–255 | `reserved` | Architectural expansion |

Coordinate address mapping:

\[
addr(i_0,\ldots,i_{r-1}) = base+\sum_{d=0}^{r-1}i_d\cdot stride_d
\]

Hardware natively supports ranks 1 through 5. Higher-rank tensors are flattened by the compiler where affine, or lowered to iterative scalar/vector loops.

## 7.3 Recommended Fields for Hint Section

| Field | Purpose |
|---|---|
| `tensor_id` | Identifier of tensor within current kernel domain |
| `execution_epoch` | Prevents stale hints across kernel invocations |
| `use_site_id` | Distinguishes distinct access sites referencing the same tensor |
| `reuse_class` | Category: `streaming`, `temporal`, `shared`, or `unknown` |
| `expected_uses` | Expected consumption count for each logical tile |
| `sharing_group` | Identifies cooperating cores / contexts |
| `progress_axis` | Identifies the outer loop iteration axis tracking progress |
| `live_window_hint` | Expected concurrent live tiles |
| `cache_priority` | Relative eviction retention priority |
| `allocation_hint` | Policy: `allocate`, `no-allocate`, or `adaptive` |
| `confidence` | Compiler analysis confidence level |
| `prefetch_distance` | Suggested lookahead distance in tiles |

**These fields do not modify target addresses, synchronization requirements, or memory access permissions.**

## 7.4 Descriptor Versioning Is Mandatory

Rules governing descriptors:
- Instructions capture a snapshot or immutable version tag upon issue;
- In-flight memory transactions must never observe torn (partially updated) descriptors;
- Updating a descriptor does not alter previously issued memory transactions;
- Reclaiming a descriptor handle requires that all associated in-flight requests have retired.

## 7.5 Physical Address Is the Identity for Cache Coherence

The same physical cache line may be referenced simultaneously by:
- Distinct TensorMaps;
- Different reshaped tensor views;
- General-purpose scalar/vector load/store instructions;
- Concurrent program instances.

Therefore:

> **Data tag matching and home slice selection must be derived strictly from the physical address, never from the `tensor_id`.**

`tensor_id` serves solely as performance metadata. ASID/PASID govern virtual-to-physical translation and access permissions; shared physical lines must map to a single coherent home location.

---

# 8. NUCA Cache: Scaling DCO Principles into a Concrete Mechanism

DCO incorporates tensor/tile metadata, dead-block detection, coordinated anti-thrashing, and selective cache bypassing ([arxiv.org](https://arxiv.org/html/2512.07312v2)). TNCA-v1 preserves these principles while tracking completion via explicit tensor transfer lifecycles rather than guessing when a tile is done based on which line address arrives last.

## 8.1 Basic Slice Organization

```text
Request queues
    │
Predictive request arbiter
    │
Tag pipeline ──────────────── Hit response queue
    │
Miss/merge handling
    ├── MSHR entries
    ├── Target storage
    ├── Fill buffers
    └── Writeback buffers
    │
Data banks
    │
Response/forwarding interface
```

Per-slice configuration (512 KiB, 4 banks, 8-way, 64 B line):
- 128 KiB per bank;
- 256 sets per bank;
- 8 ways per set.

The 96 MSHR entries can be partitioned into $4 \times 24$ banks to limit tag comparison and arbitration logic, subject to monitoring for capacity imbalance across banks.

## 8.2 Explicit Tile-Use Completion Replaces "Last Address Completes Last"

For each TMA command:

```text
logical_use_id = {execution_epoch, tensor_id, tile_coord, use_site}
```

The hardware tracks:
- Total sub-requests expanded for the logical tile;
- Completed sub-requests;
- Padding elements initialized;
- RF target write completions.

Only when the entire logical tile access completes is a `use-complete` notification dispatched to the TMU.

This handles:
- Out-of-order response arrivals;
- Masked boundary tiles;
- Zero/constant padding;
- Arbitrary memory layout address strides;
- Multiple active tensor views;
- Strided gather patterns.

Irregular LSU accesses lacking tile metadata simply fall back to conventional LRU/cache management policies.

## 8.3 Avoid Attempting Precise "Last Use" Tracking in Dynamic Programs

Three tracking tiers are established:

| Program Characteristics | Tracking Strategy |
|---|---|
| Affine, regular, compile-time provable reuse | Utilize explicit `expected_uses` |
| Partially regular with dynamic control masks | Conservative lifetime hints |
| Data-dependent addressing or irregular dynamic loops | Mark `expected_uses = unknown` |

When `expected_uses = unknown`:
- Aggressive `dead` block marking is disabled;
- Caching falls back to priority tiers, streaming hints, and local pressure metrics.

Predictive benefits should not be pursued at the cost of turning the TMU into an overly complex runtime program analyzer.

## 8.4 Replacement and Bypass Share a Unified Priority Hierarchy

Eviction and admission priority is computed from:

```text
reuse class
+ compiler priority
+ sharing status
+ stable tile/line hash tier
+ local pressure adjustment
```

Eviction precedence order:
1. Invalid cache lines;
2. Lines explicitly marked `dead`;
3. Streaming / low-reuse lines;
4. Lowest priority retention tier;
5. Age-based selection (LRU/FIFO) within the same tier.

Miss admission rule:
```text
if (new_data_priority < current_admission_threshold):
    execute no-allocate bypass
else:
    allocate and fill line normally
```

Unifying admission and eviction priorities prevents the pathological behavior where lines are allocated at high cost only to be immediately evicted.

## 8.5 Bypass Does Not Mean Bypassing Coherence

### Read No-Allocate
- Request is routed to the designated home slice;
- Checks for active dirty lines;
- Observes memory consistency ordering for matching addresses;
- Bypasses installation into the cache data array on a miss.

### Write No-Allocate
- Coordinates with existing cached copies;
- Must invalidate or update stale cached lines rather than writing directly to DRAM;
- Reports explicit memory visibility completion.

### Bypass Still Consumes Miss Tracking
Omitting allocation in the cache data array does not eliminate:
- MSHR entry tracking;
- Target register routing entries;
- Return data buffering;
- Memory controller queue occupancy.

Performance models must account for bypass tracking resources.

## 8.6 No Routine Cache Pinning

Routine software cache pinning is excluded:

```text
Pin entire tensor tile → wait for consumer completion → unpin
```

Cache pinning introduces severe liabilities:
- Pinned footprints can exceed cache capacity;
- Replacement flexibility is compromised;
- Programs require complex proofs of cache capacity;
- Inter-producer circular dependencies induce deadlock;
- It re-introduces the software capacity management burdens of SPM.

Allowed exceptions:
- Transient cache line locks during atomic read-modify-write operations;
- Protocol reservations during active fill/writeback sequences.

**Cache residency is strictly a performance optimization, never a precondition for correct program execution.**

---

# 9. Upgrading LLaMCAT Concepts: Controlling Transactions Rather than Coarse Threads

## 9.1 Distinguish at Least Five Credit Types

| Credit Name | Protected Hardware Resource |
|---|---|
| `new_miss_credit` | Free MSHR entry allocation |
| `merge_credit` | MSHR target coalescing slots |
| `return_credit` | Network return paths and receiver buffers |
| `rf_destination_credit` | Destination register file window buffers |
| `writeback_credit` | Dirty line victim writeback queues |

Coarse request limits (`outstanding_requests < N`) are insufficient because allocating a new miss and merging into an existing miss consume entirely different hardware resources.

## 9.2 Recommended Request Arbitration Order

For requests with verified resource credits:
1. Transactions that release blocked hardware resources;
2. Starved requests exceeding an aging threshold;
3. Predicted cache hits;
4. MSHR merges with available target slots;
5. Demand new cache misses;
6. Speculative prefetches.

Operational guarantees:
- Hit/merge predictions determine arbitration priority; actual hits are verified at tag lookup;
- Merged requests respect address ordering hazards with concurrent writes;
- Aging counters prevent starvation of new misses.

## 9.3 Cache Hits Must Be Served Even When MSHR Is Full

The slice must not stall completely on MSHR exhaustion:

```text
if MSHR_full:
    stall_entire_cache_pipeline // REJECTED DESIGN
```

Architecture requirement:
- Decouple hit service paths from miss allocation logic;
- Requests encountering full MSHR pools enter bounded replay queues;
- Incoming hits from other sources continue processing;
- Independent virtual channels/queues prevent head-of-line blocking by stalled misses.

Hits still consume data bank and response bandwidth, which must be tracked.

## 9.4 Compare Progress by Sharing Group, Not Chip-Wide Request Counts

Global chip-wide request comparisons are invalid:

```text
core 0 issued 1000 requests
core 1 issued 500 requests
therefore core 0 is too fast // INVALID
```

Disparities arise naturally because cores may execute different workloads, process different tile sizes, or serve distinct producer/epilogue roles.

Progress tracking must be localized:

```text
group_id
logical_iteration
completed_tile_count
window_id
```

Example (Grouped Query Attention, GQA):

```text
Query-head programs sharing the same KV head
        ↓
Form a sharing group
        ↓
Bound key-block advance of fastest over slowest program
```

This regulates the shared working set in cache rather than enforcing artificial request-count parity.

## 9.5 Two Control Timescales

### Fast Control Loop
Sampled every 64–128 cycles (programmable baseline):
- Regulates new miss injections;
- Regulates MSHR merges;
- Monitors response network backpressure;
- Monitors RF window availability.

Primary input metrics:
```text
MSHR entry occupancy
MSHR target occupancy
response queue occupancy
writeback pressure
RF destination availability
```

### Slow Control Loop
Sampled every 1024–4096 cycles (programmable baseline):
- Adjusts prefetch lookahead distance;
- Adjusts active sharing group window sizing;
- Adjusts new program launch admission;
- Dynamically updates cache allocation thresholds;
- Balances sharing-group progress skews.

These intervals represent evaluation baselines and must be tuned across workloads.

## 9.6 Simple Control Laws Outperform Complex Learners

Baseline control law:

```text
if (pressure_high sustained over multiple sample periods):
    new_miss_budget = max(min_budget, new_miss_budget / 2)

elif (pressure_low and consumer_starved):
    new_miss_budget += 1
```

Stabilization rules:
- Asymmetric hysteresis thresholds for throttle-down and ramp-up;
- Minimum state dwell times;
- Fair allocation across active groups;
- Minimum guaranteed forward-progress quotas.

Avoid simultaneously varying bypass policies, prefetch distances, active task counts, and miss budgets to prevent control loop oscillations.

---

# 10. Mesh, Data Placement, and Coherence

## 10.1 Version 1 Adopts Static-Home NUCA

The home slice is statically determined:

\[
home=H(\text{physical line address})
\]

Architectural properties:
- Every physical cache line maps to a unique home slice;
- Cores maintain no private L1 data caches;
- No transparent replication of dirty lines;
- No dynamic runtime cache line migration across slices.

This eliminates complex directory protocols and migration state machines.

Required protocol support:
- In-flight fill and store ordering;
- Writeback and concurrent access hazards;
- Atomic operations serialized at the home slice;
- Partial-line writes;
- No-allocate memory operations;
- DRAM return routing and line state transitions.

**The absence of private L1 caches does not eliminate the need for a rigorous coherence protocol.**

## 10.2 Home-Aware Placement Independent of Mutable TensorID

Placement techniques:
- Physical page coloring;
- Fixed address hashing algorithms;
- Runtime memory allocation policies;
- Program-to-cluster affinity mapping.

Example:
- A $2\times2$ cluster of tiles processes kernels sharing KV heads;
- Corresponding pages are mapped to NUCA slices proximate to that cluster;
- Standard non-shared tensors remain uniformly striped across all slices.

Two descriptors accessing the same physical address must never compute divergent home slice locations.

## 10.3 Mesh Bandwidth Must Be Accounted Per Individual Link

In a $4\times4$ mesh, the bisection cut spans exactly 4 links in each direction.
Assuming 64 B/cycle payload per link:

\[
64\text{ B/cycle}
\]

The unidirectional bisection upper bound at 1.0 GHz is:

\[
4\times 64\times 10^9=256\text{ GB/s}
\]

This represents a physical ceiling, not the aggregate bandwidth of all tile ports.
Deductions include:
- Packet header overhead;
- Arbitration bubbles;
- Cross-traffic contention;
- Request and control messages;
- Spatial traffic imbalances.

Performance models must calculate link load:

\[
\max_{\text{link }l} \frac{\text{bytes traversing }l}{B_l}
\]

Aggregate slice bandwidth (16 slices × bandwidth per slice) must not be assumed as deliverable system throughput.

## 10.4 Request and Response Paths Must Isolate Resources

Traffic isolation requires dedicated networks or virtual channels:
- Read requests;
- Write and writeback data payloads;
- Read responses;
- Completion and synchronization control.

Routing utilizes deterministic Dimension-Order XY routing in Version 1.

> **Deadlock-free routing algorithms resolve only channel-dependency cycles; end-to-end protocols can still deadlock at endpoints.**

Classic deadlock analysis builds upon channel dependency graphs ([comp.anu.edu.au](https://comp.anu.edu.au/courses/comp4300/assets/lectures/refs/dallyDeadlockAvoid87.pdf)); TNCA protocol verification must include resource dependencies spanning caches, RF buffers, writebacks, and barriers.

## 10.5 Five Forward-Progress Rules

1. Before issuing a memory read, the destination RF window must be allocated.
2. The arrival of return data must never depend on the software executing a subsequent allocation instruction.
3. Completion and control traffic must never be starved by speculative prefetches.
4. Each cache slice must maintain reserved buffer space to drain dirty victim lines.
5. Flow control throttling must never prevent admitted transactions from completing or releasing resources.

Hardware multicast is omitted from Version 1; initial designs utilize home-slice request merging, multi-target tracking, and unicast delivery.

---

# 11. ISA Overall Design

## 11.1 ISA Does Not Use Monolithic Operators as Basic Units

The ISA avoids hardcoding high-level operator primitives such as:

```text
FLASH_ATTENTION
LAYERNORM
SOFTMAX
CONV
```

Instead, primitive building blocks are provided:
- Scalar arithmetic;
- Vector SIMD;
- Predication and masking;
- Shuffles and reductions;
- Tensor DMA transfers;
- Matrix multiplication fragments;
- Accumulator row access;
- Asynchronous completion signaling;
- Cache and flow hints.

This ensures longevity across novel operators and fusion patterns.

## 11.2 Architecturally Visible State

| State Element | Allocation |
|---|---|
| Program Counter (PC) | One 64-bit PC per RCTX |
| Scalar Registers (S) | 64 × 64-bit registers per RCTX |
| Vector Registers (V) | Up to 256 logical names per RCTX (mapped to shared physical pool) |
| Predicate Registers (P) | 16 × 16-bit registers per RCTX (`P0` is hardwired to all-true) |
| Accumulator Slots (A) | 16 slots per tile (each holds 16×16 FP32 elements) |
| Window Handles (W) | Up to 32 RF window handles per tile |
| Descriptor Handles (D) | Up to 16 TensorMap snapshots per tile |
| Barrier Objects (B) | Up to 32 hardware mbarriers per tile |
| Command Slots | 64 asynchronous command tracking slots per tile |
| Special Registers | `program_id`, `grid_dim`, `role_id`, `core_id`, `cycle_count`, etc. |

Physical vector registers are dynamically partitioned across active contexts at kernel launch; all four RCTXs cannot assume simultaneous non-overlapping ownership of 256 physical registers.

## 11.3 Data Types

### Version 1 High-Speed Types
- Predicates (1-bit masks);
- Integer: `i32`, `u32`;
- Addresses: `i64`, `u64`;
- Floating Point: `FP32`;
- Packed Floating Point: `FP16`, `BF16` (load/store/convert);
- Native Matrix: `FP16` and `BF16` inputs with `FP32` accumulation.

### Version 1 Software-Expanded Types or Operations
- Complex 64-bit SIMD integer math;
- `FP64` double-precision arithmetic;
- Non-native FP8 variants;
- Integer division and remainder;
- High-precision transcendental functions;
- Non-native matrix dot products.

Software-expanded operations must preserve exact numerical semantics without silent approximations.

## 11.4 Instruction Length

Instruction encodings:
- Base instructions: 64 bits;
- Extended instructions (large immediates, tensor configurations): 128 bits;
- Alignment: 8-byte boundary aligned;
- Branch targets must align to instruction boundaries.

### Base 64-Bit Instruction Encoding
```text
63       56 55       48 47       40 39       32
+----------+----------+----------+----------+
| major_op | function |    d     |    a     |
+----------+----------+----------+----------+
31       24 23       16 15  12 11 10 9      0
+----------+----------+------+--+--+---------+
|    b     |    c     | pred | X|!P| modifier|
+----------+----------+------+--+--+---------+
```

Field definitions:
- `major_op` [63:56]: Opcode group;
- `function` [55:48]: Specific functional operation;
- `d`, `a`, `b`, `c`: Destination and source register specifiers;
- `pred` [15:12]: Predicate register;
- `X` [11]: Extension flag (`1` denotes an accompanying 64-bit extension word);
- `!P` [10]: Predicate inversion bit;
- `modifier` [9:0]: Type, rounding mode, or memory semantic flags.

### Tensor Transfer Extension Word (64-bit)
```text
63:48   transfer_cfg_id
47:32   stream_id
31:16   use_site_id
15:0    policy / extension flags
```

Full multi-dimensional shape and stride information is referenced through descriptors rather than embedded directly into instruction encodings.

## 11.5 Major Opcode Groupings

| Major Opcode | Family | Description |
|---|---|---|
| `0x00` | System | Control, special registers, traps, execution termination |
| `0x01` | Scalar Integer | Arithmetic, logical, shift, comparison |
| `0x02` | Control Flow | Unconditional branches, conditional jumps, calls, returns |
| `0x10` | Vector Integer | SIMD integer arithmetic, bit manipulation |
| `0x11` | Vector Float | SIMD floating-point arithmetic (FP32, packed FP16/BF16) |
| `0x12` | Convert/Pack | Type conversion, packing, unpacking, saturation |
| `0x13` | Predicate | Mask manipulation, ballot, condition evaluation |
| `0x14` | Permute | Vector shuffle, butterfly permute, element extract/insert |
| `0x15` | Reduction | Intra-vector reductions, parallel scans |
| `0x16` | SFU | Special function approximations (`exp2`, `rcp`, `rsqrt`) |
| `0x20` | Standard Load/Store | Scalar and vector contiguous/gather/scatter memory access |
| `0x21` | Atomic/Fence | Memory atomics and synchronization fences |
| `0x30` | Tensor Descriptor | TensorMap loading, snapshot creation, field update |
| `0x31` | Tensor Transfer | Asynchronous tensor loads, stores, and prefetches |
| `0x32` | Cache/Flow Hints | Cache management, reuse hints, progress marking |
| `0x40` | Matrix | Asynchronous matrix multiply-accumulate operations |
| `0x41` | Accumulator Access | Row-level vector reads, writes, and scalings of accumulators |
| `0x50` | Mbarrier | Hardware barrier initialization, arming, arrival, and waits |
| `0x51` | Role Control | Context yield, role activation, suspension |
| `0x60` | Debug/Perf | Performance counter access, trace event markers |

---

# 12. Detailed ISA Instruction Families

Implementation categories:
- **N**: Native hardware execution;
- **L**: Lowered / software library emulation;
- **E**: Future architectural extension.

## 12.1 Scalar and Control Flow

| Instruction | Semantics | Implementation |
|---|---|---|
| `s.mov` | Scalar register move | N |
| `s.movi` | Load immediate | N |
| `s.add / s.sub` | Integer addition and subtraction | N |
| `s.mul.lo` | Integer multiplication (low bits) | N |
| `s.mul.hi` | Integer multiplication (high bits, at least u32) | N |
| `s.and / s.or / s.xor / s.not` | Bitwise logical operations | N |
| `s.shl / s.lshr / s.ashr` | Logical and arithmetic shifts | N |
| `s.min / s.max` | Signed and unsigned minimum / maximum | N |
| `s.cmp` | Scalar comparison setting predicate | N |
| `s.select` | Conditional selection based on predicate | N |
| `s.div / s.rem` | Integer division and remainder | L |
| `s.read.special` | Read special registers (`pid`, `role_id`, `grid`, etc.) | N |
| `br` | Unconditional relative branch | N |
| `br.nz / br.z` | Conditional branch based on scalar register | N |
| `call / ret` | Subroutine call and return | N (inlined where feasible) |
| `yield` | Relinquish instruction issue slot to other RCTX | N |
| `trap` | Raise exception to runtime | N |
| `exit` | Terminate execution of current RCTX | N |

The architecture omits hardware SIMT divergence stacks; vector divergent execution relies on predication, while uniform control flow uses scalar branching.

## 12.2 Vector Integer and Floating Point

| Instruction | Semantics |
|---|---|
| `v.mov` | Vector register move |
| `v.splat` | Broadcast scalar element across all vector lanes |
| `v.iota` | Generate consecutive lane indices $[0, 1, \ldots, 15]$ |
| `v.add / v.sub / v.mul` | Elementwise vector addition, subtraction, multiplication |
| `v.mul.hi` | High-order integer product |
| `v.and / v.or / v.xor / v.not` | Vector bitwise operations |
| `v.shl / v.lshr / v.ashr` | Vector logical and arithmetic shift |
| `v.min / v.max` | Vector minimum and maximum (integer and float) |
| `v.fma` | Fused multiply-add ($d = a \times b + c$) |
| `v.neg / v.abs` | Floating-point negation and absolute value |
| `v.cmp` | Vector comparison generating predicate mask |
| `v.select` | Elementwise selection driven by predicate mask |
| `v.clz / v.popcount` | Count leading zeros and population count |

Floating-point `min`/`max` instructions must adhere to strict IEEE 754 specifications regarding NaN propagation, signed zeros ($\pm 0.0$), and numeric priority.

## 12.3 Predicate

| Instruction | Purpose |
|---|---|
| `p.and / p.or / p.xor / p.not` | Boolean logic across predicate masks |
| `p.from.cmp` | Form predicate mask from comparison results |
| `p.any / p.all` | Reduce predicate mask to scalar uniform boolean |
| `p.ballot` | Compact 16-lane predicate mask into 16-bit integer scalar |
| `p.from.bits` | Expand 16-bit scalar integer into 16-lane predicate mask |

Under masked vector execution:
```text
active lane: write result
inactive lane: preserve previous value
```

Memory loads must strictly respect Triton's `other` fill semantics rather than implicitly zeroing inactive lanes ([triton-lang.org](https://triton-lang.org/main/python-api/generated/triton.language.load.html)).

## 12.4 Convert, Pack, and Low Precision

| Instruction | Purpose |
|---|---|
| `v.cvt` | Numerical type conversion (e.g., `i32` to `f32`, `f32` to `f16`) |
| `v.bitcast` | Reinterpret bit representation without numeric conversion |
| `v.pack.f16 / v.pack.bf16` | Pack pairs of 32-bit registers into packed 16-bit formats |
| `v.unpack.f16 / v.unpack.bf16` | Unpack packed 16-bit vectors into widened 32-bit registers |
| `v.trunc / v.extend` | Integer truncation, sign-extension, and zero-extension |
| `v.cvt.sat` | Type conversion with saturation arithmetic |
| `v.cvt.fp8.*` | FP8 conversion routines (E or L) |

Default FP32 rounding uses Round-to-Nearest-Even (RNE). Flush-to-Zero (FTZ) and reduced-precision modes require explicit modifier flags.

## 12.5 Shuffle, Permute, and Layout Transformation

| Instruction | Purpose |
|---|---|
| `v.shuffle.xor` | Butterfly exchange across lanes (XOR index permutation) |
| `v.shuffle.up / v.shuffle.down` | Shift elements up or down across neighboring lanes |
| `v.permute` | Arbitrary lane-indexed vector permute |
| `v.zip / v.unzip` | Interleave and de-interleave even/odd vector elements |
| `v.extract / v.insert` | Extract or insert scalar elements within vector registers |

Arbitrary multi-dimensional transpositions are synthesized by combining shuffles, packing/unpacking, permutes, and TMA target destination layout configurations. Complex dynamic transformations fall back to temporary cache-backed staging buffers. Linear Layout analysis from Triton provides the mapping framework ([triton-lang.org](https://triton-lang.org/main/getting-started/tutorials/gluon/layouts.html)).

## 12.6 Reduction and Scan

Native vector reductions:
```text
v.reduce.add
v.reduce.max
v.reduce.min
v.reduce.and
v.reduce.or
v.reduce.xor
```

Semantics:
- Computes reduction across all 16 lanes of a vector register;
- Supports predicate masking (inactive lanes evaluate to the operation identity);
- Optionally broadcasts the scalar result across all lanes of the destination.

Cross-register reductions are scheduled by the compiler. Complex reductions (tuple reductions, `argmax`, Welford's algorithm, associative scans) are lowered using comparison, selection, and shuffle primitives ([triton-lang.org](https://triton-lang.org/main/dialects/TritonOps.html)).

## 12.7 SFU

Native approximations:
```text
v.exp2.approx
v.rcp.approx
v.rsqrt.approx
```

Optional in v1:
```text
v.log2.approx
```

Extended functions (`exp`, `sigmoid`, `tanh`, `erf`, trigonometric functions, strict IEEE division/square root) are synthesized via polynomial expansion in runtime libraries.
The hardware provisions a 4-lane SFU pipeline per tile as a baseline. The ISA specification must publish error bounds, subnormal handling, and special-value behaviors (NaN, $\pm\infty$, $\pm 0.0$).

## 12.8 Standard Load/Store

General-purpose memory access provides essential architectural completeness:

```text
s.ld / s.st
v.ld.contiguous
v.st.contiguous
v.ld.gather
v.st.scatter
```

Specification:
- 64-bit global memory addresses;
- Inactive lanes (`mask == false`) generate no memory transactions;
- Coalescing logic merges accesses falling within common 64-byte physical cache lines;
- Unaligned memory accesses are decomposed by the LSU;
- Masked vector stores update only specified bytes.

Standard loads block the issuing RCTX via scoreboard tracking; explicit asynchronous tokens are reserved for TMA transfers.

## 12.9 Atomics and Fence

Native atomic operations executed directly at the home slice:

```text
atom.cas.u32/u64
atom.xchg.u32/u64
atom.add.u32/u64
atom.and/or/xor.u32/u64
atom.min/max.s32/u32
```

Floating-point atomic addition in Version 1 is supported via a compare-and-swap loop, with dedicated home-slice FP32 adders planned for future revisions.

Memory ordering and synchronization instructions:
```text
atomic.load
atomic.store
fence.acquire
fence.release
fence.acq_rel
```

Supported scopes:
- `core`: Visibility within local tile;
- `device`: Coherence across all tiles on chip.

System-wide scope (`system`) spanning the host CPU requires hardware-coherent interconnect support; non-coherent PCIe DMA with manual cache flushing does not satisfy arbitrary host-device atomics ([triton-lang.org](https://triton-lang.org/main/python-api/generated/triton.language.atomic_cas.html)).

## 12.10 Tensor Descriptor Instructions

```text
td.load       Dd, [descriptor_address]
td.create     Dd, scalar_fields...
td.update     Dd, field, value
td.release    Dd
```

Operational rules:
- `td.load` captures an immutable versioned snapshot into descriptor register `Dd`;
- `td.update` generates a new version without altering active in-flight transfers;
- `td.release` invalidates the handle once associated in-flight operations complete;
- Descriptor tables cannot silently overwrite active descriptors when full.

## 12.11 Tensor Transfer Instructions

Core asynchronous movement instructions:
```text
t.ld.async
t.st.async
t.prefetch
```

Recommended format:
```text
t.ld.async Wdst, Dsrc, Scoord_base, Stoken, CFG
```

Operational flow for `t.ld.async`:
1. Reads descriptor snapshot `Dsrc`;
2. Fetches multi-dimensional coordinates from consecutive scalar registers starting at `Scoord_base`;
3. Validates tensor boundaries and determines padding requirements;
4. Decomposes the tile into cache-line read requests;
5. Routes arriving return data through `CFG` layout logic into destination window `Wdst`;
6. Completes padding and element packing;
7. Signals arrival completion to the barrier identified by token `Stoken`.

Additional transfer primitives:

| Instruction | Status | Notes |
|---|---|---|
| `t.prefetch` | N | Pure performance hint; does not guarantee subsequent cache residency |
| `t.copy` | L | Tiled register-mediated tensor copy |
| `t.gather.rows` | E | Lowered to repeated strided loads in v1 |
| `t.scatter.rows` | E | Lowered to repeated strided stores in v1 |
| `t.reduce.store` | L to atomics, then E | Lowered to home-slice atomic updates |
| `t.im2col.load` | E | Reserved for future convolution acceleration |

Triton IR natively distinguishes descriptor load/store, gather/scatter, and reducing stores; these must be lowered independently ([triton-lang.org](https://triton-lang.org/main/dialects/TritonOps.html)).

## 12.12 Matrix and Accumulator Instructions

```text
m.zero
m.mma.async
m.wait             // Assembly alias for mb.wait on matrix token

a.vread
a.vwrite
a.mul.row
a.fma.row
```

`m.mma.async` explicitly specifies:
- Source operand windows / fragments;
- Destination accumulator slot;
- M/N/K tile dimensions;
- Input operand types;
- Accumulation precision mode;
- Completion token.

Until the associated token signals completion:
- Source operand registers cannot be overwritten;
- Accumulator slot `A_slot` cannot be read by vector instructions;
- The accumulator slot cannot be reallocated to another context.

## 12.13 Cache and Flow Hints

```text
tm.bind
tm.unbind
tm.use_hint
flow.mark
flow.prefetch_hint
```

`flow.mark` logs progress points for sharing-group flow control:
```text
flow.mark group, logical_iteration
```

The ISA avoids exposing fragile physical cache implementation details:
```text
Must be placed in slice 3, way 7 // REJECTED
Permanently lock set S // REJECTED
Directly modify MSHR quota // REJECTED
```
Exposing such primitives tightly couples the compiler to a single microarchitecture generation.

---

# 13. Asynchrony and mbarrier: Establishing a Precise Protocol

NVIDIA's TMA and `mbarrier` architectures provide a proven foundation: asynchronous transactions and arrival counts coordinate barrier fulfillment ([docs.nvidia.com](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html)). TNCA adapts these concepts to register-file targets without requiring binary compatibility with CUDA shared-memory formats.

## 13.1 Four Completion Types Must Not Be Confused

The runtime protocol strictly differentiates:

### Load-Ready
Data has been fully written into the target destination register window (`Wdst`), and is available for compute consumption.

### Compute-Done
Matrix or SIMD functional units have finished computing; output accumulator or vector registers hold valid results.

### Source-Reusable
A functional unit or DMA engine has completed reading source operand registers, permitting the producer to overwrite them.

### Store-Visible
Asynchronous store data has propagated to the designated coherence point (e.g., home NUCA slice or memory controller).

In Version 1, store completion may conservatively combine source release and visibility into a single event:
```text
Single done for store
= Source is no longer in use + all writes are visible at home
```
This occupies registers slightly longer but keeps the initial protocol simple. Dual-completion optimizations can be added subsequently.

## 13.2 Barrier Object

Each hardware barrier tracking register contains:
```text
current_epoch
completed_epoch
pending_arrivals
pending_operations
waiter_mask
error_status
```
Rather than maintaining only a single phase bit.

A synchronization token is packed into a 64-bit scalar value:
```text
token = {epoch, barrier_id}
```
Epoch rollover must be handled when the barrier is completely quiescent.

## 13.3 Instructions

```text
mb.init
mb.arm
mb.arrive.release
mb.test
mb.wait.acquire
mb.invalidate
```

In Version 1, `mb.arm` initializes transaction counters for a specific generation:
```text
mb.arm Stoken, B, arrivals, operations
```
Tracking granularity is defined as:

> **One complete asynchronous instruction command, rather than an individual cache line or indeterminate byte count.**

Example:
- Two `t.ld.async` instructions set `operations = 2`;
- One `m.mma.async` instruction sets `operations = 1`;
- Each hardware command engine tracks its internal sub-transactions.

## 13.4 Completion Conditions

A barrier generation completes when:

\[
pending\_arrivals=0 \land pending\_operations=0
\]

Invariants:
- Completion triggers exactly once per generation;
- Counter underflows trigger hardware exceptions;
- Faulted transactions immediately wake waiting contexts with error flags set;
- Faulted commands must never cause infinite hangs;
- `mb.wait.acquire` suspends the calling RCTX into a low-power sleep state without busy-polling.

Predicated-off asynchronous commands emit a no-op completion to prevent counter mismatches.

## 13.5 Double Buffering Requires Both Ready and Free Event Pairs

Every double-buffered register slot transitions through a four-phase lifecycle:

```text
FREE
  ↓ Producer acquires slot
FILLING
  ↓ TMA completes write
READY
  ↓ Consumer begins compute
IN_USE
  ↓ Consumer finishes reading source
FREE
```

Tracking `READY` alone without tracking `FREE` is an architectural defect:

```text
TMA completed
≠
MMA has finished reading input
```

## 13.6 GEMM Dual-Role Pipeline Example

The following pseudocode illustrates double-buffered matrix multiplication across producer and consumer contexts; `B@epoch` denotes compiler-generated generation tokens.

### Initialization
```text
Empty[0..1] barriers armed for epoch 0 (marked complete)
Ready[0..1] barriers initialized
Register windows WA[0..1] and WB[0..1] allocated
```

### Producer
```text
for k_tile in 0 .. K_tiles-1:
    slot  = k_tile & 1
    epoch = (k_tile >> 1) + 1

    mb.wait.acquire Empty[slot]@(epoch-1)

    mb.arm Empty[slot]@epoch, arrivals=1, operations=0
    mb.arm Ready[slot]@epoch, arrivals=1, operations=2

    t.ld.async WA[slot], DA, coordA(k_tile),
               Ready[slot]@epoch, CFG_A

    t.ld.async WB[slot], DB, coordB(k_tile),
               Ready[slot]@epoch, CFG_B

    mb.arrive.release Ready[slot]@epoch
```

### Consumer
```text
for k_tile in 0 .. K_tiles-1:
    slot  = k_tile & 1
    epoch = (k_tile >> 1) + 1

    mb.wait.acquire Ready[slot]@epoch

    mb.arm ComputeDone, arrivals=0, operations=N_mma

    for each output fragment:
        m.mma.async A_fragment,
                    WA[slot].fragment,
                    WB[slot].fragment,
                    ComputeDone

    mb.wait.acquire ComputeDone

    mb.arrive.release Empty[slot]@epoch
```

This represents a robust, provably correct baseline. Future optimizations may decouple "source read finished" from "full MMA result written" to signal `Empty` earlier, but this requires explicit protocol additions rather than heuristic cycle assumptions.

---

# 14. Memory Model: Simple Yet Unambiguous

## 14.1 Standard Memory
- Conflicting accesses to the same physical address are serialized at the designated home slice;
- Accesses to distinct physical addresses are weakly ordered;
- Sequential consistency is not guaranteed across distinct addresses without synchronization;
- Explicit `fence.acquire` and `fence.release` establish happens-before relationships.

## 14.2 Inter-RCTX Communication

Communication across contexts within a tile is established through register windows:

```text
producer writes RF window
→ release completion
→ consumer acquire wait
→ consumer reads RF window
```
establishing a happens-before relationship.

Instruction retirements, asynchronous memory completions, and device visibility must follow defined synchronization protocols. Triton's asynchronous tokens and cooperative synchronization abstractions must not be collapsed into an undifferentiated barrier ([triton-lang.org](https://triton-lang.org/main/dialects/TritonGPUOps.html)).

## 14.3 Inter-Core Synchronization
- Local `mbarrier` structures coordinate contexts within the same tile;
- Inter-tile synchronization uses home-slice atomic operations and memory fences;
- Grid-wide synchronization across kernels is coordinated via host dispatch boundaries;
- Arbitrary hardware barriers across the entire grid are unsupported in Version 1.

Cooperative cluster execution requires that all participating programs be gang-scheduled simultaneously with guaranteed forward progress.

## 14.4 Host Visibility
When operating without hardware cache coherence with the host CPU:

```text
kernel completion
→ device write drain
→ required cache clean
→ host notification
```

The host CPU must not access device memory buffers concurrently without explicit ownership handover protocols.

---

# 15. Triton/MLIR Compiler Strategy

## 15.1 Reuse Frontend, Do Not Replicate Backend Mechanically

The compilation toolchain reuses Triton's Python frontend, AST parsing, and TTIR infrastructure, while replacing downstream code generation with a target-specific pipeline ([github.com](https://github.com/triton-lang/triton/blob/main/third_party/amd/backend/compiler.py)).

```text
Triton Python
    ↓
TTIR
    ↓
Canonicalization / Type Legalization / Control Flow Simplification
    ↓
Tensor Access Analysis
    ↓
Layout Selection
    ↓
TMA Formation
    ↓
Role Partitioning
    ↓
Async Pipeline Construction
    ↓
RF / Accumulator / Barrier Allocation
    ↓
Cache-Flow Metadata Generation
    ↓
TNCA Machine IR
    ↓
Instruction Scheduling / Register Allocation
    ↓
Binary + Kernel Resource Manifest
```

## 15.2 Introduce a Lightweight TNCA Dialect

Target-specific MLIR operations:
```text
tnca.tensor_map
tnca.rf_window
tnca.tensor_load_async
tnca.tensor_store_async
tnca.mma_async
tnca.acc_read
tnca.acc_write
tnca.barrier_arm
tnca.barrier_wait
tnca.barrier_arrive
tnca.role_region
tnca.progress
```

Core operation attributes:
```text
layout
rf_lifetime
async_read_set
async_write_set
completion_token
reuse_group
logical_use_id
```

Asynchronous operations must model side effects and lifetimes accurately to prevent invalid MLIR CSE, LICM, or dead-code elimination.

## 15.3 Six Critical Compiler Passes

### Pass A: Access Identification
Classifies memory access patterns into:
1. Contiguous bulk transfers;
2. Multi-dimensional affine slices;
3. Regular row gathers;
4. Dynamic, unaligned gather/scatter;
5. Atomic read-modify-write.

Only classes 1 through 3 are mapped to the TMA fast path.

### Pass B: Layout Legalization
Legalizes transitions across:
- Vector layout;
- Matrix operand layout;
- Accumulator layout;
- TMA destination layout.

The pass eliminates redundant transformations while ensuring unavoidable transpositions are materialized explicitly.

### Pass C: Role Partitioning
Decomposes dependency graphs into roles:
- Data load / prefetch;
- Matrix compute;
- Vector compute / reduction;
- Data store.

Kernels are partitioned into 1, 2, or 4 roles depending on workload complexity.

### Pass D: Asynchronous Pipeline Construction
Determines pipeline parameters:
- Prefetch lookahead distance;
- Register window buffering stages;
- Hardware barrier assignment;
- Window ownership transfers;
- Outstanding command slot limits.

### Pass E: RF Lifetime Analysis and Allocation
Tracks concurrent live intervals:
```text
Window currently being filled by producer
Window currently being read by consumer
In-flight unretired accumulator
Vector temporaries
Descriptor/token scalars
```

Source operand lifetimes extend until the `source-reusable` event, rather than terminating at instruction issue.

### Pass F: Cache/Flow Metadata Generation
Derives runtime caching hints from finalized schedules:
- Expected tile reuse counts;
- Sharing-group boundaries;
- Outer progress iteration axes;
- Streaming classifications;
- Rough categories of reuse distance.

These hints are generated after tiling and role pipelining are finalized.

## 15.4 Handling TTGIR Shared-Memory Operations

`ttg.local_alloc`, `ttg.local_load`, and `ttg.local_store` represent shared-memory objects in GPU IR ([triton-lang.org](https://triton-lang.org/main/dialects/TritonGPUOps.html)). TNCA handles these through three strategies:

### Path 1: Eliminate Shared Memory Objects
Where shared memory serves merely as a staging buffer (`global → shared → registers`), the pass bypasses intermediate storage:
```text
global → shared → registers
```
becomes:
```text
global/NUCA → RF window
```

### Path 2: Registerization
Small shared-memory allocations with static, analyzable access patterns are promoted into vector registers with intra-register shuffles.

### Path 3: Cache-Backed Temporary Objects
Complex shared allocations fall back to private, globally addressable memory buffers backed by NUCA and DRAM:
```text
global addressable temporary
    ↕
NUCA
```

Properties of cache-backed temporaries:
- Backed by DRAM address space;
- Lines can be evicted under cache pressure;
- Program correctness does not depend on permanent cache residency;
- They do not represent dedicated physical SPM.

Snapshot copy-on-write semantics must be preserved: if data is staged to a local buffer and the global source is subsequently mutated, promoting the local buffer to an alias of the global buffer is invalid without proving read-only isolation:
```text
copy global A → local B
modify A
read B
```

## 15.5 Triton Feature Coverage Table

| Triton Feature | TNCA Lowering Strategy |
|---|---|
| `program_id/num_programs` | Read special registers |
| `arange/full/splat` | `v.iota`, constants, scalar broadcasts |
| `reshape/expand_dims` | Metadata reinterpretation; register re-indexing |
| `trans` | Layout transformation, shuffles, or TMA configuration |
| `load/store` | TMA fast path for bulk tensors; standard LSU for irregular patterns |
| Block pointers | Lowered to TensorMap descriptors and coordinate arithmetic |
| Tensor descriptors | Direct mapping to native TensorMap |
| `dot` | Native Matrix engine or vector fallback |
| `sum/max/min` | Vector reduction instructions (`v.reduce.*`) |
| Custom reductions | Vector shuffles combined with elementwise operator trees |
| Associative scan | Hierarchical parallel prefix scan using shuffles |
| Gather / scatter | LSU instructions with coalescing; regular TMA extensions |
| Atomics | Home-slice atomic operations (`atom.*`) |
| Sort / Top-K | Bitonic sort networks using vector compare and exchange |
| Histogram | Small register bins or cache-backed temporaries with atomics |
| Random number generation | Pseudorandom integer arithmetic implementations |
| Math (`exp`, `sigmoid`, etc.) | SFU approximations (`exp2`, `rcp`) combined with math library |
| Assert / Print | Device traps and debug queues |
| Inline assembly | Native TNCA target assembly; arbitrary PTX is rejected |
| Warp-specialize regions | Mapping to distinct RCTX role partitions |
| Multi-CTA / Cluster features | Capability checks or specialized cluster lowering |

Frontend operations map to appropriate lowering paths; hardware is not required to provide 1:1 matching instructions for every high-level Python API ([triton-lang.org](https://triton-lang.org/main/python-api/triton.language.html)).

## 15.6 Do Not Promise Native Automatic Mapping for Arbitrary Worker Partitions

If target-specific programs require more than four concurrently active partitions:
- Partitions are coalesced when dataflow allows;
- Remaining partitions are lowered to software continuations / state machines;
- Otherwise, the compiler emits an explicit diagnostic rejecting the kernel.

Over-subscribing contexts sequentially without deadlock analysis is invalid when partitions depend on mutually dependent barriers.

---

# 16. Primary Operator Mapping and Real Bottlenecks

## 16.1 GEMM

Baseline exploration parameters:
```text
BLOCK_M = 64
BLOCK_N = 64
BLOCK_K = 32
```

Data footprints for FP16/BF16 inputs:

\[
A=64\times 32\times 2=4096\text{ B}
\]

\[
B=32\times 64\times 2=4096\text{ B}
\]

Double-buffered inputs:

\[
2(A+B)=16\text{ KiB}
\]

FP32 accumulator output:

\[
C=64\times 64\times 4=16\text{ KiB}
\]

Therefore:
- Inputs fit within the Vector RF (taking 16 KiB of 32 KiB);
- Output fully occupies the Accumulator RF (16 KiB).

Compute workload per $K$-tile:

\[
F=2\times 64\times 64\times 32=262144
\]

At a peak arithmetic capability of 512 FLOPs/cycle per tile, execution requires a minimum of 512 cycles.
Required operand delivery rate:

\[
\frac{8192}{512}=16\text{ B/cycle}
\]

> **By enabling intra-RF operand reuse across the matrix computation, the required operand delivery rate from NUCA drops from 64 B/cycle down to 16 B/cycle for this tile configuration.**

This reuse efficiency is the foundational premise that makes eliminating SPM viable.

### Capacity Trade-Off in Accumulator
Because a 64×64 FP32 $C$ matrix fully consumes the 16 KiB Accumulator RF:
- The accumulator cannot be double-buffered;
- Epilogue processing (bias, activation, quantization) serializes with the start of the next output tile.

Alternative configuration:
```text
BLOCK_M=32, BLOCK_N=64
```
This permits double-buffering two 8 KiB accumulator tiles, enabling overlap between epilogue execution and the subsequent GEMM compute, at the cost of reduced input reuse.

Tiling must simultaneously optimize:
- NUCA traffic;
- Accumulator capacity;
- Epilogue overlap;
- RF bank conflicts;

rather than simply pursuing maximum $M/N$ tile dimensions.

---

## 16.2 FlashAttention

Baseline tile parameters:
```text
BLOCK_M = 16
BLOCK_N = 16
HEAD_DIM = 128
```

Online softmax formulation:

\[
m'=\max(m,\max_j s_j)
\]

\[
\alpha=e^{m-m'}
\]

\[
p_j=e^{s_j-m'}
\]

\[
l'=\alpha l+\sum_j p_j
\]

\[
O'=\alpha O+PV
\]

Finally:

\[
O_{\text{final}}=O/l
\]

### Resource Estimation

| Data Structure | Dimensions and Format | Footprint |
|---|---|---:|
| $Q$ Tile | 16 × 128 BF16 | 4 KiB |
| $K$ Tile | 16 × 128 BF16 | 4 KiB |
| $V$ Tile | 16 × 128 BF16 | 4 KiB |
| Double-Buffered $K/V$ | Two sets of $K$ and $V$ | 16 KiB |
| Attention Probabilities $P$ | Two 16 × 16 BF16 buffers | 1 KiB |
| Attention Scores $S$ | Two 16 × 16 FP32 buffers | 2 KiB |
| Output Matrix $O$ | 16 × 128 FP32 | 8 KiB |
| Statistics ($m, l$) | 16 rows × 2 FP32 | 128 B |

Allocation mapping:
- $Q$, $K$, $V$, $P$, and vector temporaries reside in the Vector RF;
- Intermediate scores $S$ and output matrix $O$ reside in the Accumulator RF.

### Recommended Division Across Four Roles

```text
R0: K/V prefetch + TMA
R1: QK^T / PV Matrix scheduling
R2: Mask + max + exp2 + sum + O rescale
R3: Auxiliary transfer / epilogue / next task control
```

$QK^T$ and $PV$ matrix operations share the tile's unified matrix engine rather than requiring separate dedicated datapaths.

### True Bottlenecks
Performance is bounded by non-GEMM operations:
1. Transferring attention scores $S$ from accumulator storage to vector registers;
2. `v.exp2.approx` SFU throughput;
3. Vector row-reduction latency;
4. Rescaling accumulator $O$ via `a.mul.row`;
5. RAW dependencies between softmax results and $PV$ compute;
6. Live intervals of concurrent score and probability buffers.

If updating $O$ required round-tripping through NUCA:
```text
Accumulator → NUCA → Vector → NUCA → Accumulator
```
system throughput would collapse. Direct vector access to the accumulator (`a.mul.row`, `a.fma.row`) is an architectural prerequisite.

FlashAttention-3 emphasizes overlapping asynchronous data movement, matrix multiplication, and softmax computation ([arxiv.org](https://arxiv.org/abs/2407.08608)); TNCA preserves this pipeline concurrency without requiring Hopper-specific shared-memory hardware paths.

### Distinguishing Two Attention Phases
- **Prefill**: Compute-dense; concurrently bounded by Matrix, RF bandwidth, and SFU throughput.
- **Decode / GQA**: Memory-bound; throughput is governed by DRAM bandwidth, MSHR tracking, target merging, and sharing-group progress control.

A single static throttling configuration cannot serve both execution phases.

---

## 16.3 LayerNorm / RMSNorm

Norm operations do not require a dedicated monolithic normalization accelerator:

```text
TMA/LSU load
→ vector sum / sum-of-squares or Welford
→ reduction
→ rsqrt
→ affine transform
→ store
```

### Rows Fit Within RF
The entire row is retained in registers, reduced, normalized in place, and written out in a single pass.

### Rows Exceed RF Capacity
Requires a two-pass execution model:
1. Pass 1: Stream data from memory to compute reduction statistics;
2. Pass 2: Re-stream data, applying normalization and affine weights.

Capturing the second pass in the NUCA cache is a prime caching optimization opportunity. However, entire rows must not be forcibly pinned in cache to simulate a single pass. Complex tuple reductions (Welford) are lowered through standard instruction sequences.

---

## 16.4 Softmax

Small rows:
```text
In-RF max
→ exp2
→ sum
→ reciprocal
→ scale
```

Large rows:
- Multi-pass execution;
- Or tiled statistical calculation followed by re-reading;
- With the second access captured in NUCA.

Performance is governed by `v.exp2.approx` throughput, reduction tree latency, and RF port bandwidth rather than peak matrix FLOPs.

---

## 16.5 Conv

First revisions should avoid complex im2col TMA hardware:
- $1\times1$ Convolutions: Lowered directly to GEMM;
- Standard Convolutions: Lowered to implicit GEMM;
- Strides, dilations, and padding: Handled via scalar/vector address generation;
- Irregular borders: Lowered to vector LSU gather instructions;
- Depthwise Convolutions: Lowered to pure SIMD vector pipelines.

Specialized `t.im2col.load` instructions represent future performance optimizations rather than initial functional prerequisites.

---

## 16.6 Sigmoid and Standard Elementwise

Recommended mapping:
- Execute across 1 or 2 RCTXs;
- Use standard vector loads and stores;
- Optional small TMA transfers;
- Avoid full 4-role pipeline setup and barrier synchronization.

Sigmoid evaluation:

\[
\sigma(x)=\frac{1}{1+e^{-x}}
\]

Evaluated primarily via `exp2`, addition, and reciprocal. Simple elementwise kernels must not incur the latency and resource overhead of complex multi-role pipelines.

---

# 17. Prefetch Strategy: Decoupling Cache Lookahead from RF Buffering

## 17.1 Two Distinct Distances

### NUCA Prefetch Distance
Anticipates future data into the on-chip NUCA cache:
```text
Lookahead span = upcoming logical tiles
```

### RF Pipeline Depth
Governs active double buffering within local registers:
```text
Active span = current tile + next tile
```

For example:
```text
NUCA lookahead: 4–8 tiles
RF stages: 2
```
This decouples the time horizons of cache prefetching from physical register allocation without creating an additional storage tier.

## 17.2 Why Decoupling Is Effective

If register double-buffering were tasked with hiding full external memory latency:
- Register capacity would need to scale with round-trip DRAM latency;
- Physical RF area would expand beyond feasibility.

By using the NUCA cache to absorb DRAM latency, register double buffering only needs to hide local NUCA hit latency.

However:
> `t.prefetch` completion provides no guarantee of subsequent cache residency, nor does it establish ownership or snapshot semantics.

Under heavy cache pressure:
- Prefetch lookahead is dynamically shortened;
- Low-priority prefetches are dropped;
- Demand fetches take absolute priority;
- Hot working sets are protected from eviction.

## 17.3 Pipeline Depth Constraints

Conservative estimate:

\[
S_{\text{RF}} \gtrsim 1+\left\lceil\frac{L_{\text{ready}}}{T_{\text{iteration}}}\right\rceil
\]

Subject to hard physical capacity bounds:

\[
S_{\text{RF}}\cdot B_{\text{input tile}} + B_{\text{live vector}} \leq C_{\text{VRF}}
\]

and:

\[
B_{\text{live accumulators}}\leq C_{\text{ARF}}
\]

Compilers must respect capacity constraints rather than blindly increasing pipeline buffering stages based on latency alone.

---

# 18. Runtime and Kernel ABI

## 18.1 Kernel Manifest

Every compiled kernel binary embeds a structural manifest:
```text
ISA version
required capabilities
entry PC per role
active role mask
grid dimensions
argument layout
RF allocation plan
RF window table
accumulator usage
barrier count
descriptor count
maximum command slots
spill bytes per program
preferred sharing group
preferred placement
initial prefetch / injection hints
```

## 18.2 Launch Admission

The grid scheduler validates resource availability prior to launching a program on a tile:
- Vector register capacity;
- Accumulator slots;
- Hardware mbarriers;
- Descriptor slots;
- Minimum transaction credits.

Programs are never launched partially; all required role contexts must be admitted concurrently to prevent deadlock.

## 18.3 Program Completion

Executing `exit` does not immediately free resources. Complete retirement requires:
- All active roles have exited;
- All required asynchronous commands have completed;
- Output writes meet device completion visibility semantics;
- No late memory responses remain in flight to target register addresses;
- Descriptor and window handles have been reclaimed.

Network packets carry transaction generation identifiers to prevent late-arriving responses from corrupting newly allocated register files.

## 18.4 Address Translation

Version 1 execution baselines:
- Buffers are pinned by the runtime prior to launch;
- Addressing uses physical addresses or simple static IOMMU translations;
- Demand paging is unsupported during kernel execution.

Addressing violations raise a kernel trap, propagate errors to barrier waiters, and drain in-flight transactions safely. Recoverable page faults, preemption, and migration are excluded from initial revisions unless explicitly mandated.

---

# 19. Verilog RTL Module Breakdown

Hardware control and compute RTL can be implemented in Verilog-2001, while verification harnesses utilize SystemVerilog.

```text
tnca_top.v
├── host_if.v
├── command_processor.v
├── grid_scheduler.v
├── resource_admission.v
├── memory_controller_if.v
└── tile.v
    ├── role_frontend.v
    ├── instruction_decode.v
    ├── issue_scheduler.v
    ├── scalar_rf.v
    ├── scalar_alu.v
    ├── vector_rf_bank.v
    ├── vector_rf_crossbar.v
    ├── vector_alu.v
    ├── vector_shuffle.v
    ├── vector_reduce.v
    ├── sfu.v
    ├── matrix_engine.v
    ├── accumulator_rf.v
    ├── accumulator_vector_port.v
    ├── scoreboard.v
    ├── rf_window_table.v
    ├── tensor_descriptor_table.v
    ├── tensor_address_generator.v
    ├── tensor_transfer_engine.v
    ├── load_store_unit.v
    ├── coalescer.v
    ├── barrier_table.v
    ├── completion_unit.v
    ├── flow_controller.v
    ├── network_interface.v
    ├── mesh_router.v
    └── nuca_slice.v
        ├── request_queues.v
        ├── request_predictor.v
        ├── request_arbiter.v
        ├── tag_array.v
        ├── data_bank.v
        ├── mshr_bank.v
        ├── target_pool.v
        ├── fill_buffer.v
        ├── writeback_buffer.v
        ├── atomic_unit.v
        ├── slice_tmu.v
        └── replacement_admission.v
```

## 19.1 Explicit Handshaking Across All Inter-Module Paths

Inter-module communication uses explicit four-way handshaking:
```verilog
valid
ready
payload
```

Design invariants:
- When `valid && !ready`, `payload` must remain stable;
- Combinational `ready` loops spanning multiple mesh routers are prohibited;
- Traffic classes maintain independent queue buffering;
- Flow control credit updates occur exactly once per transaction.

## 19.2 Tensor DMA State Machine

```text
IDLE
→ CAPTURE_DESCRIPTOR
→ VALIDATE_DESTINATION
→ EXPAND_COORDINATES
→ ISSUE_SUBREQUESTS
→ RECEIVE_AND_WRITE_RF
→ PAD_REMAINING_ELEMENTS
→ COMPLETE
```

Concurrent commands advance through pipeline phases independently, tracking separate descriptor versions, destination windows, in-flight counts, errors, and completion targets.

## 19.3 Cache Line Transaction State Machine

Cache line protocol states:
```text
INVALID
VALID_CLEAN
VALID_DIRTY
MISS_PENDING
FILL_PENDING
WRITEBACK_PENDING
ATOMIC_LOCKED
```

Every transient state defines rules for:
- Admissible incoming requests;
- MSHR coalescing eligibility;
- Required stall conditions;
- Handling arriving memory fill data;
- Handling transaction errors and cancellations.

## 19.4 Features Excluded from Version 1

- General-purpose out-of-order instruction execution;
- High-count simultaneous multithreading (SMT);
- Transparent cache line dirty replication;
- Dynamic runtime cache line migration;
- Adaptive mesh packet routing;
- Hardware network multicast;
- Hardware cache compression;
- Arbitrary dynamic hardware layout reshaping;
- Sparse matrix hardware acceleration;
- Demand paging;
- Fine-grained mid-kernel preemption;
- Dedicated FP8/FP4 matrix multiplier arrays;
- Mandatory software cache pinning.

These features may offer value in future revisions, but are not essential for validating the core architectural thesis.

---

# 20. Performance Model and Verification Strategy

## 20.1 Beyond a Simple Roofline Model

Execution time lower bounds are modeled across all critical hardware resources:

\[
T\geq
\max\left(
\frac{F_{\text{mma}}}{P_{\text{mma}}},
\frac{F_{\text{vec}}}{P_{\text{vec}}},
\frac{N_{\text{sfu}}}{P_{\text{sfu}}},
\frac{D_{\text{DRAM}}}{B_{\text{DRAM}}},
\max_b\frac{D_b}{B_b},
\max_l\frac{D_l}{B_l},
\frac{D_{\text{RF}}}{B_{\text{RF}}},
T_{\text{critical-path}}
\right)
\]

Where:
- $F$: FLOP count for matrix and vector units;
- $P$: Peak computational capability;
- $D_{\text{DRAM}}, B_{\text{DRAM}}$: DRAM traffic and bandwidth;
- $D_b, B_b$: Cache bank traffic and bandwidth;
- $D_l, B_l$: Individual mesh link traffic and bandwidth;
- $D_{\text{RF}}, B_{\text{RF}}$: Register file traffic and port bandwidth;
- $T_{\text{critical-path}}$: Dependency chain latency.

Detailed modeling incorporates data-dependency DAGs, synchronization barriers, register window live ranges, finite queue depths, bank conflicts, MSHR allocations, injection backpressure, and realistic execution overlap.

## 20.2 Key Performance Counters

### Core Counters
- RCTX active cycles vs. stall cycles (`RCTX ready/blocked cycles`);
- RF read/write bank conflict stalls;
- Matrix operand starvation cycles;
- Accumulator dependency stall cycles;
- SFU busy cycles;
- Reduction unit busy cycles;
- Barrier wait suspension duration (`barrier wait cycles`);
- Register window allocation stall cycles (`window unavailable cycles`).

### Cache Counters
- Tag hits (`hit`);
- New misses (`new miss`);
- MSHR coalesced merges (`MSHR merge`);
- MSHR entry pool exhausted cycles (`entry-full`);
- Target pool exhausted cycles (`target-full`);
- Fill blocked cycles;
- Writeback blocked cycles;
- No-allocate bypass events;
- Dead-hint evictions (`dead-hint victim`);
- Useful vs. discarded prefetches (`prefetch useful/useless`);
- Bytes served per sharing group.

### Network Counters
- Link utilization per direction;
- Virtual channel / virtual network buffer occupancy;
- Injection stalls;
- Response stalls;
- Average and maximum hop counts;
- Packet and header bandwidth overhead.

### Flow Control Counters
- Progress divergence across sharing groups;
- Duration under credit throttling;
- Controller gear changes;
- Conflicting simultaneous control actions within the same cycle.

## 20.3 Mandatory Correctness Tests

1. Zero-mask loads generating no physical memory transactions;
2. Boundary tile accesses correctly applying padding;
3. Robustness against arbitrary, out-of-order sub-request returns;
4. Multiple TensorMaps accessing overlapping physical addresses;
5. Interleaved execution of standard LSU and asynchronous TMA traffic;
6. Read no-allocate instructions correctly observing cached dirty lines;
7. Conflict resolution between dirty victim evictions and incoming fills;
8. Uninterrupted hit servicing under complete MSHR entry exhaustion;
9. Independent transaction progress under complete MSHR target pool exhaustion;
10. DMA writebacks encountering destination RF bank conflicts;
11. Barrier epoch rollover and stale generation token rejection;
12. Predicated-off asynchronous commands generating valid completions;
13. Concurrent descriptor updates during active command execution;
14. Late-arriving memory responses after program termination;
15. Functional correctness when all soft performance hints are delayed or dropped;
16. Faulted memory transactions successfully waking waiting contexts;
17. Forward progress under cyclic inter-role resource contention;
18. Cache-evicted temporary objects correctly reloaded from memory.

Floating-point and atomic tests must evaluate numerical outcomes against defined language precision tolerances rather than expecting bit-identical bit patterns across distinct schedules.

## 20.4 Ablation Experiments

To isolate architectural contributions, evaluation must compare:
```text
Baseline NUCA
+ TensorMap descriptors
+ Dead-block eviction hints
+ Coordinated bypass and replacement
+ MSHR-aware request arbitration
+ Dedicated per-source transaction credits
+ Sharing-group progress control
+ Home-aware placement affinity
+ RF-aware compiler instruction scheduling
```

Evaluations must benchmark against an equivalent baseline featuring:
```text
RF + SPM + Cache
```
This isolates whether throughput gains stem from tensor-flow coordination or simply expanded RF, MSHR, or memory port capacities.

---

# 21. Recommended Implementation Sequence

## Phase 1: Prove Functionality First, Avoid Premature 16-Core Scaling
Baseline configuration:
- 1 core;
- 1 cache slice;
- 1–2 RCTXs;
- 16-lane SIMD execution unit;
- Standard load/store LSU;
- Basic atomic instructions;
- Hardware mbarriers;
- Unadorned baseline cache without predictive logic.

Target workloads:
- Elementwise operations;
- Vector reductions;
- LayerNorm;
- Vector-based GEMM.

Validates fundamental ISA execution and memory consistency semantics.

## Phase 2: Integrate Tensor DMA, RF Window, and Matrix Units
Target milestones:
- Asynchronous GEMM;
- Register double-buffering pipelines;
- Accumulator epilogue processing;
- FlashAttention on small tile dimensions.

Key validation goals:
- Register window ownership protocols;
- Source register lifetime management;
- Barrier generation synchronization;
- SPM-free layout legalization.

## Phase 3: 2×2 Mesh Integration
Architectural additions:
- Multi-slice NUCA;
- Static-home packet routing;
- MSHR entry and target management;
- Separation of request and response networks;
- Credit-based forward-progress flow control.

Establishes a stable multi-core baseline prior to introducing dynamic policies.

## Phase 4: Incrementally Incorporate Research Concepts
Sequential integration:
1. TensorMap metadata support;
2. Explicit tile-use completion signaling;
3. Dead-block eviction hints;
4. Coordinated cache replacement and bypass;
5. MSHR-aware request arbitration;
6. Fast-loop request admission control;
7. Sharing-group progress regulation.

Each mechanism is evaluated via independent ablation benchmarks.

## Phase 5: Scale to 4×4 Mesh, Optimize Compiler and Layouts
Final evaluation:
- Assessing RF port pressure under full load;
- Sizing matrix array dimensions;
- Sizing SFU pipeline lane widths;
- Evaluating hardware multicast trade-offs;
- Validating NUCA cache scaling efficiency;
- Identifying mesh bisection bottlenecks.

---

# Conclusion: Twelve Decisions to Freeze First

Before entering detailed hardware implementation, the following architectural decisions should be frozen:

1. **Storage Hierarchy**: The only data storage tiers are RF ↔ NUCA ↔ DRAM; software-addressable on-chip Tensor SRAM is excluded.
2. **Execution Contexts**: Up to four RCTXs per core, defined as vector control contexts rather than scalar threads.
3. **Dual Memory Paths**: Regular bulk tensor traffic flows through Tensor DMA; irregular memory accesses use a general-purpose coalescing LSU.
4. **Decoupled Matrix Supply**: Matrix units fetch operands from local RF windows rather than depending on cycle-by-cycle remote cache delivery.
5. **Accumulator Organization**: Vector RF and Accumulator RF are partitioned, with direct row-level vector access provided to accumulators.
6. **TensorMap Separation**: Strict architectural separation between mandatory memory semantics and ignorable performance hints.
7. **Cache Architecture**: Single-level NUCA with static physical-address home slices; no transparent dirty replication or runtime migration.
8. **Throttling Model**: Regulate admission of new memory requests; never pause completion paths, consumer execution, or resource release.
9. **Fine-Grained Flow Credits**: Concurrently track credits for MSHR entries, MSHR targets, return network paths, RF windows, and writebacks.
10. **Dual-Event Synchronization**: Both `ready` and `free` events are explicitly represented in the ISA and compiler lifetime models.
11. **Compiler Lowering Strategy**: Build down from TTIR and affine layout analysis; avoid mechanical 1:1 lowering from GPU shared-memory TTGIR passes.
12. **Verification Staging**: Prove functional completeness on a single-core baseline before scaling to multi-tile meshes and dynamic flow control.

**The defining characteristic of this architecture is the coordinated execution model:**

> **The compiler declares tensor access patterns and lifecycles; the cache determines data retention; flow control regulates transaction admission; and the register/barrier protocol guarantees that admitted work completes without deadlock.**

Harmonizing these four components enables an SPM-free architecture to sustain the computational efficiency required by GEMM and FlashAttention while preserving the general programmability expected by Triton.
