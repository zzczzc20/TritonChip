# TNCA — Tensor-Native Compute Array

An open, from-scratch AI accelerator built to run [Triton](https://triton-lang.org)
kernels, with **no software-addressable scratchpad anywhere in the data
hierarchy**. The only levels are register file → NUCA cache → DRAM.

| Document | |
|---|---|
| [`doc/architecture_overview.md`](doc/architecture_overview.md) | the design rationale — the normative source for everything else |
| [`doc/isa/TNCA-ISA-v0.1.md`](doc/isa/TNCA-ISA-v0.1.md) | the frozen instruction encoding |
| [`doc/conformance.md`](doc/conformance.md) | point-by-point audit of what is built against the architecture document, deviations included |
| [`doc/synthesis.md`](doc/synthesis.md) | synthesis results and the timing findings they produced |

---

## 项目简介（中文）

TNCA 是一个从零开始、面向 Triton 的开源 AI 芯片设计。它最核心的取舍是：
**彻底取消软件可寻址的 Shared Memory / SPM**，数据层次只有

```
寄存器堆 (RF)  ←→  单级分布式共享 NUCA Cache  ←→  DRAM
```

由此带来的四个接口需要重新设计：Triton 的 tensor/layout 语义如何落到寄存器、
TensorMap 如何驱动缓存管理、异步搬运如何定义寄存器所有权与完成事件、
以及跨核心进度如何控制请求准入。这些在
[`doc/architecture_overview.md`](doc/architecture_overview.md) 中有完整论证。

**当前进度：阶段 3。** 4 个 tile 组成 2×2 mesh，每个 tile 含一个核心与一个 NUCA
slice。物理行的 home slice 由地址哈希唯一确定，无私有数据缓存、无复制、无迁移，
因此 slice 之间不存在一致性协议。完整链路为

```
向量寄存器 → LSU beat → core NI → mesh → home NUCA slice → DRAM
```

其中除两端外没有任何软件可寻址的存储层次。已实现：流水化 FP/乘法器、
多未完成事务且支持乱序返回的 LSU、带 MSHR/target pool/脏行回写的 NUCA slice、
XY 路由与请求/响应分离虚拟网络的 mesh。

**尚未实现**：Matrix 单元、TensorMap/Tensor DMA、mbarrier、SFU、原子操作，
以及 Stage 4 的缓存策略（dead-block、bypass、准入 credit）。路线图见下文。

> 本仓库是一个诚实的在建工程，不是一颗流片过的芯片。凡是没做的，
> 下面都写明了没做；凡是测过的，都给出了测试规模。

---

## Status

**Stage 3 — four tiles on a 2×2 mesh over a single-level distributed NUCA cache.**

| Area | State |
|---|---|
| ISA v0.1 encoding | frozen, machine-readable, enforced by the decoder |
| Scalar integer + control flow | complete |
| Vector integer, predicate, shuffle, reduction | complete |
| IEEE-754 binary32 add / sub / mul / fma / min / max / compare / convert | complete, 2-stage pipeline |
| Pipelined 64×64 multiplier, multi-latency writeback scheduling | complete |
| FP reduction (bit-reproducible balanced tree) | complete |
| Load/store unit with a physical-word coalescer | complete |
| 2 Role Contexts on a shared datapath | complete |
| Assembler, disassembler, exact FP oracle | complete |
| Non-blocking LSU, many outstanding transactions, out-of-order returns | complete |
| NUCA slice: MSHRs, target pool, dirty writeback, hits past a blocked miss | complete |
| 2×2 mesh: XY routing, separate request/response virtual networks | complete |
| 4-tile system: cores, slices, mesh, DRAM, static-home addressing | complete |
| Matrix unit, TensorMap, Tensor DMA, mbarrier | **not started** (Stage 2) |
| SFU (`v.exp2.approx` et al) | **not started**; the capability bit reads 0 and the encodings trap |
| Synchronous instruction fetch | **not started** (Stage 1c) |
| Atomics, fences | **not started** (Stage 2) |
| Cache/flow policy: dead-block hints, bypass, admission credits | **not started** (Stage 4) |
| Triton/MLIR backend | **not started** (Stage 5) |

Nothing here has been through a real PDK, static timing, or power analysis.
The synthesis numbers below are relative proxies and are labelled as such.

---

## The one design decision everything follows from

The architecture removes the scratchpad that every comparable design relies on.
That forces four interfaces to be redesigned rather than inherited:

1. **Triton tensor/layout semantics → registers and instructions.**
   Without shared memory there is no staging buffer for layout conversion, so
   the register file has to absorb both the compute working set and the
   reuse that a scratchpad would have provided.
2. **TensorMap → cache management.** Descriptors carry two strictly separated
   kinds of field: semantics that must be executed exactly, and hints that may
   be ignored. A mispredicted hint may only cost performance.
3. **Asynchronous transfer → register ownership and completion events.**
   "Data has arrived", "compute is done" and "the source may be overwritten"
   are three different events and the ISA keeps them distinct.
4. **Cross-core progress → request admission.** Throttling applies to *new*
   work entering the system, never to the paths that complete work already in
   flight or release resources.

Stages 1 to 3 in this repository build the substrate those four interfaces
attach to: the register files and completion protocol for (1) and (3), and a
real single-level NUCA cache behind a mesh for (2) and (4). Stage 4 is where
the descriptors and the flow control land, and it can now be measured against a
baseline rather than argued about.

---

## Quick start

```bash
# Requirements: iverilog >= 11, yosys >= 0.40, python3 >= 3.9. No PDK needed.
make test          # lint + unit tests + core tests + slow-memory reruns
make synth         # Yosys synthesis and a PPA proxy report
make help
```

A full `make test` run:

```
lint: all 10 files elaborate
  fp32_fma               PASS   41224 vectors, 0 failures
  decoder                PASS   627 vectors, 0 failures
  asm round-trip         PASS   2964 instructions, 6 programs, 0 mismatches
  t00_scalar             PASS   cycles=541  issued=290
  t01_vector             PASS   cycles=876  issued=492
  t02_mem                PASS   cycles=1171 issued=353
  t03_fp                 PASS   cycles=2868 issued=1431
  t04_roles              PASS   cycles=133  issued=74
  t05_fault              PASS   cycles=455  issued=228
  t06_alu                PASS   cycles=5347 issued=2099
  t00_scalar (slow mem)  PASS   ...
  ... every core test again at memory latency 7 with random back-pressure

==============================================
 TNCA regression: all tests passed
==============================================
```

Write and run your own kernel:

```bash
cat > /tmp/hello.s <<'EOF'
.org 0
    v.iota  v1
    v.reduce.add s1, v1
    .assert_eq s1, #120, #1
    exit
EOF
python3 tools/asm/tnca_asm.py /tmp/hello.s -o /tmp/hello.hex --size 8192
vvp sim/build/tb_core.vvp +hex=/tmp/hello.hex +maxcycles=10000
```

---

## Repository layout

```
doc/
  architecture_overview.md    the design rationale; the normative source
  isa/TNCA-ISA-v0.1.md        the frozen instruction encoding
  conformance.md              audit against the architecture document
  synthesis.md                PPA proxies and the findings from them
rtl/
  tnca_top.v                  NX x NY tiles, mesh, memory arbiter
  common/
    tnca_defs.vh              encoding constants, derived from the ISA doc
    tnca_lzc.v tnca_popc.v    log-depth leading-zero and population count
    tnca_fifo.v               valid/ready FIFO
  core/
    tnca_core.v               4-stage in-order pipeline, N Role Contexts,
                              multi-latency writeback scheduling
    tnca_decoder.v            encoding -> control, and encoding legality
    tnca_scalar_alu.v         64-bit integer ALU
    tnca_mul64.v              3-stage pipelined 64x64 multiplier
    tnca_vector_alu.v         16-lane integer / predicate / shuffle / reduce
    tnca_vector_fpu.v         16-lane binary32, plus the reduction sequencer
    tnca_fp32_fma.v           IEEE-754 binary32 FMA, 2-stage pipeline
    tnca_lsu.v                coalescer, many outstanding tagged transactions
    tnca_scalar_rf.v          64 x 64 bit per RCTX, 3R1W, write-first
    tnca_vector_rf.v          physical vector pool, per-lane write enables
    tnca_pred_rf.v            16 x VLEN bit per RCTX
  nuca/
    tnca_nuca_slice.v         tags, data, MSHRs, targets, writeback, arbiter
    tnca_mem_arbiter.v        N slice memory ports onto one DRAM port
  noc/
    tnca_router.v             5-port XY router, N virtual networks
    tnca_mesh.v               NX x NY fabric
    tnca_core_ni.v            home-slice hashing and request injection
    tnca_slice_ni.v           request delivery and response addressing
    tnca_noc_pkg.vh           packet layout
  tile/
    tnca_tile.v               one core, one slice, their interfaces
tb/
  top/tb_tnca_top.v           the whole 2x2 system running a program
  nuca/tb_nuca_slice.v        randomised cache test against a shadow memory
  noc/tb_mesh.v               randomised all-to-all mesh test
  core/tb_tnca_core.v         runs a self-checking program
  core/tb_fp32_fma.v          differential test against the exact oracle
  core/tb_decoder.v           encoding legality
  core/tb_bitops.v tb_mul64.v unit tests for the shared primitives
  core/progs/*.s              the self-checking test programs
  common/tnca_mem_model.v     word memory, parameterised latency/stalls/order
  common/tnca_dram_model.v    line memory behind the slices
tools/
  asm/tnca_isa.py             machine-readable encoding tables
  asm/tnca_asm.py             assembler
  asm/tnca_dis.py             disassembler and round-trip checker
  model/fp32.py               exact binary32 semantics (Fractions, not floats)
  model/gen_*_vectors.py      test-vector generators
flow/
  synth/synth.py              Yosys synthesis + PPA proxy report
  synth/calibration.v         reference structures for reading the depth numbers
```

---

## The Stage-1 core

```
 F    pick a Role Context, read the instruction
 D    decode; check encoding legality and implementation limits
 R    scoreboard check (register indices only)
 X    read the register files, execute, hand the result to WB
 WB   drive the register file write port
```

Role Contexts interleave freely; program order is maintained per RCTX by the
scoreboard, and there is no ordering *between* RCTXs except through memory.
Branches are predicted not-taken, and a redirect in X squashes only the F, D
and R entries belonging to the same RCTX — other contexts keep making progress
across a mispredict.

A few decisions worth knowing about, because they were each forced by a bug
that testing found:

* **The register-file write port is driven by a register, not by X.** Driving
  it combinationally from X closes a loop through the write-first read bypass
  for any instruction that reads and writes the same register (`s.movk` does).
* **The WAW check does not use the "X is producing it now" bypass.** The
  scoreboard is one bit per register and cannot represent two in-flight
  writers; a second writer that claimed the bit early had it cleared out from
  under it by the first writer's release.
* **A unit is claimed from X, so R must ask whether it is free *next* cycle.**
  Asking whether it is free now dropped a load whose predecessor was accepted
  in the very cycle R looked.
* **Out-of-band completions (the LSU, the FP reduction sequencer) take the
  write port by stalling the front end for one cycle** so a bubble reaches X.
  X therefore never repeats a side effect, and a completion is never starved
  for more than one cycle.

## The memory system

```
  lane  ->  vector register  ->  LSU beat  ->  core NI  ->  mesh
                                                              |
                                                     home NUCA slice
                                                              |
                                                            DRAM
```

Nothing in that chain is software-addressable except the registers at one end
and DRAM at the other. That is the architecture's central claim and this is the
first configuration in which it is actually testable.

**Home selection.** `home = H(physical line address)`, and nothing else —
no descriptor, no TensorID, no hint. Two views of one physical line cannot
disagree about where it lives (§7.5, §10.1). A line exists in exactly one slice,
so there is no replication, no migration, and consequently no coherence protocol
between slices at all. The interleave is 64 bytes, so a contiguous buffer spans
every slice: in the 2×2 test, **1,792 of 2,048 requests left their tile**.

**The slice does not stall on a miss.** §9.3 requires that a miss which cannot
allocate an MSHR must not block hits queued behind it. The request buffer is
therefore not a FIFO — each cycle the arbiter picks the oldest entry that is
neither blocked nor shadowed by an older entry to the same line. The shadow rule
is what serialises conflicting accesses to one address at its home slice
(§14.1): a correctness property, not a heuristic. A counter records how often a
hit was selected past a blocked miss, and the test *fails if that never happens*
— so the property cannot quietly rot into an intention.

**Why the network cannot deadlock.** XY routing removes cyclic channel
dependencies within a virtual network, and requests and responses have separate
virtual networks so a backed-up request path cannot stop a response. But §10.4
is explicit that routing is only half the problem — the endpoint protocol can
still deadlock. The rule that closes it: a core's response path is
unconditionally ready, because the LSU reserved a beat-table slot *before*
issuing, so the landing site already exists and accepting the response needs no
further allocation (§10.5, rules 1 and 2). The response network therefore always
drains, which lets the requests occupying the slices retire.

### Deliberate Stage-3 simplifications

| Simplification | Why it is acceptable now | What replaces it |
|---|---|---|
| Memory controllers sit below the slices, not on the mesh edge | §10.3 wants every byte crossing a link counted, and DRAM traffic on the mesh changes those numbers; nothing about the cache protocol depends on which it is | MCs on the mesh edge, with the per-link bandwidth model (Stage 4) |
| Single-flit packets | a reassembly path would not change anything this stage establishes | multi-flit serialisation and per-link accounting (Stage 4) |
| Two virtual networks, not four | requests and responses are separated, which is the property forward progress needs | the four classes of §10.4 arrive with tensor traffic |
| Invalid-first then round-robin replacement | a baseline with a policy in it cannot serve as the baseline a policy is measured against (§21) | §8.4's priority scheme (Stage 4) |
| Instruction fetch is tile-local | §1.1 allows a small instruction buffer explicitly and restricts the *data* hierarchy | fetch through the NUCA, with its traffic counted |

### Deliberate Stage-1 simplifications

These are simplifications, not oversights. Each is local to fix.

| Simplification | Why it is acceptable now | What replaces it |
|---|---|---|
| Instruction fetch is a single-cycle asynchronous read | Stage 1 is about ISA and memory semantics | synchronous SRAM + fetch buffer (Stage 2) |
| One memory instruction in flight | the scoreboard makes it correct, just slow | MSHRs and credits (Stage 3) |
| Single-issue | the arch doc's dual-issue needs the matrix unit to be worth it | dual issue across resource classes (Stage 2) |
| FP units are combinational | correctness first; the depth is measured and reported below | pipelined FP (Stage 2) |
| Vector RF is flop-based | fine for simulation and small FPGA bring-up | banked SRAM + operand collector (Stage 2) |
| Vector register windows are a static per-RCTX partition | no tensor DMA yet to need them | RF window table (Stage 2) |

---

## Verification

The approach is that **a test must be able to fail**. Every self-checking
program was run once with a deliberately broken expectation to confirm the
harness reports it, and `tools/run_test.sh` treats "printed neither PASS nor
FAIL" as a failure rather than a pass.

| Test | What it covers | Scale |
|---|---|---|
| `tb_fp32_fma` | binary32 FMA against an exact `Fraction`-based oracle: every corner cross-product, subnormals, NaN/Inf, cancellation sweeps, ftz | 41,224 vectors in the default run; **516,224** in `make fma-deep`, 0 failures |
| `tb_decoder` | every assemblable encoding is accepted; mutations that ISA §2.2/§2.3 forbid are rejected — reserved fields, the X bit, predicates on unpredicated families, out-of-range indices, illegal condition codes and sizes, unallocated opcodes | 627 vectors, 0 failures |
| `t00_scalar` | integer ALU, immediates, compares, select, bit counting, loops, call/ret, special registers | 53 checks |
| `t01_vector` | vector integer, masked writes and inverted masks, predicate logic, shuffles, integer reductions including masked ones | 80 checks |
| `t02_mem` | every access size, sign/zero extension, unaligned scalar and vector accesses that straddle memory words, partial stores, masked loads and stores, gather, scatter, and a fully masked-off access proving no memory traffic | 56 checks |
| `t03_fp` | FP arithmetic, min/max NaN modes and signed zero, ordered/unordered compares, all four FP→int rounding modes, bit-reproducible reductions — all expectations generated from the same oracle | 193 checks |
| `t04_roles` | two Role Contexts, private scalar state, producer/consumer through memory | 4 checks |
| `t05_fault` | an unallocated encoding stops exactly one Role Context; the other keeps running and exits, and the core still reaches idle | fault containment |
| `t06_alu` | randomised integer datapath against a Python model, weighted toward `mulh` and sign boundaries | 186 checks |
| `tb_bitops` | `tnca_lzc` / `tnca_popc` against the ISA definitions: every single-bit position at three widths, plus random | 60,471 checks |
| `tb_mul64` | the pipelined 64×64 multiplier against Verilog's own 128-bit multiply, weighted to sign boundaries | 60,000 checks |
| assembler round-trip | assemble → disassemble → re-assemble reproduces the original word | 5,249 instructions |
| slow-memory rerun | every core test again at memory latency 7 with pseudo-random `ready` de-assertion | catches any fixed-latency assumption |
| out-of-order rerun | every core test again with responses delivered in pseudo-random order | once requests reach different slices through a mesh, ordered returns stop being a property anything guarantees (§20.3) |
| `tb_nuca_slice` | randomised reads and writes against a shadow memory, address range 8× the slice capacity so eviction is the common case, under three different memory behaviours | 800 requests: 568 reads checked, 367 hits, 403 misses, 99 dirty writebacks, MSHR file full 295 times, **233 hits selected past a blocked miss** |
| `tb_mesh` | randomised all-to-all on the 2×2 fabric with random receiver back-pressure; checks delivery, destination, virtual network, and per-flow ordering | 20,000 packets, none lost, none misrouted, none reordered |
| `t10_mesh` | the whole 2×2 system: per-tile private buffers, a shared table every tile hammers at once, and a streaming pass that forces eviction under cross-tile pressure | 4 tiles × 4,039 instructions; fails if the mesh carried no remote traffic or the slices never missed |

Every claim above is also checked in the other direction. Each self-checking
program was run once with a deliberately broken expectation, and `t06_alu` was
run against a deliberately broken `mulh` derivation, to confirm the harness
reports the failure rather than passing quietly.

The FP oracle itself is cross-checked: 59,550 multiplications and 11,435
additions against the host's own IEEE arithmetic, 0 mismatches. It uses exact
rational arithmetic rather than host floats, because host `a*b+c` rounds twice
and would not be a valid FMA reference.

---

## Synthesis

`make synth` runs Yosys over each module and reports two proxies:

* **gates** — cell count after mapping to a two-level gate set. A fair area
  proxy for random logic; *not* a fair proxy for register files, which map to
  flops here but belong in SRAM macros.
* **depth** — longest topological path between registers, in gate levels. At
  roughly 15–25 ps per level, a 1 GHz target wants depth under about 50.

| module | gates | flops | depth | floor | |
|---|---:|---:|---:|---:|---|
| `tnca_decoder` | 682 | 0 | 17 | — | ok |
| `tnca_lzc` / `tnca_popc` | 217 / 275 | 0 | 16 / 22 | — | ok |
| `tnca_scalar_alu` | 3,741 | 0 | 61 | 64 | **at the library floor** |
| `tnca_mul64` | 23,057 | 768 | 98 | 64 | 3 stages |
| `tnca_fp32_fma` | 7,120 | 203 | 115 | 71 | 2 stages |
| `tnca_vector_alu` | 145,334 | 0 | 99 | 65 | the largest block |
| `tnca_scalar_rf` | 31,033 | 8,192 | 10 | — | flops; belongs in SRAM |
| `tnca_fifo` | 127 | 137 | 4 | — | ok |
| `tnca_router` | 10,204 | 6,028 | 18 | — | **well under budget** |

**Read `depth` against the floor, not against 50.** The generic gate set has no
carry-lookahead cells, so every addition maps to ripple carry: a bare 32-bit
adder is already 32 levels and a 64-bit one is 64. Any module containing wide
arithmetic is over a flat budget *by construction*, and no amount of pipelining
fixes that, because the floor belongs to the library and not to the design.
`make synth` prints the calibration block above every run so the numbers can be
read honestly. The useful question is how far a module sits above its own floor.

[`doc/synthesis.md`](doc/synthesis.md) has the full table and the reasoning.

Measuring has paid for itself four times over, each one an application of §6.2
— decide the arithmetic after looking at what it costs:

| change | before | after |
|---|---|---|
| scalar ALU: multiplier moved to its own pipelined unit | 28,035 gates / 135 levels | **3,741 / 61** |
| FMA: cut into a 2-stage pipeline | 207 levels | **115** |
| vector ALU: one multiplier, one crossbar, one comparator | 205,205 gates | **145,334** |
| `clz`/`popc`: log-depth structures instead of scan loops | W-deep chains | 16 / 22 levels |

Every one of these was verified behaviour-identical by the tests that already
existed — the 516 k-vector FMA sweep, 266 vector checks, 186 randomised integer
checks — before and after.

---

## Roadmap

Stages follow `doc/architecture_overview.md` §21.

* **Stage 1a — done.** Single core, ISA v0.1 subset, verified.
* **Stage 1b — done.** Pipelined FP (2 stages) and scalar multiply (3 stages),
  log-depth `clz`/`popc`, and a writeback reservation register in the core so
  units with different latencies cannot collide on the register-file write port.
  Library calibration added to the synthesis flow so depth numbers mean
  something.
* **Stage 3 — done.** Non-blocking LSU with many outstanding tagged
  transactions and out-of-order returns; NUCA slice with MSHR entries, a shared
  target pool, dirty writeback and an arbiter that keeps hits flowing past a
  blocked miss; 2×2 mesh with XY routing and separate request/response virtual
  networks; the four-tile system running a program end to end. No adaptive
  policy — that is the point of a baseline.
* **Stage 2 (next).** Tensor descriptors and Tensor DMA, RF windows, mbarrier
  with the ready/free two-event protocol, the matrix unit (`m.mma.async`,
  M16 N16 K16/32/64), accumulator registers with vector access. Target
  workloads: asynchronous double-buffered GEMM, small FlashAttention. It comes
  after Stage 3 rather than before it because Tensor DMA's whole purpose is to
  keep many line transactions in flight, and there was nothing to keep them in
  flight *to* until the cache existed.
* **Stage 1c.** SFU — `v.exp2.approx`, `v.rcp.approx`, `v.rsqrt.approx`,
  `v.log2.approx` — with the accuracy **measured over the full input range and
  the measured bound written into the ISA** before `CAPABILITY0.SFU` is allowed
  to read 1. §12.7 requires a published error bound, NaN/Inf/zero behaviour and
  subnormal handling; setting a capability bit without them is exactly the
  dishonesty §1.6 warns about. Also: atomics and fences, synchronous
  instruction fetch, dual issue.
* **Stage 4.** Fold in the paper mechanisms one at a time with an independent
  ablation for each: TensorMap metadata, use-complete tracking, dead-block
  hints, coordinated replacement/bypass, MSHR-aware arbitration, request
  admission, sharing-group progress control.
* **Stage 5.** Scale to 4×4 and build the Triton/MLIR backend from TTIR and
  early layout analysis, not from shared-memory-dependent TTGIR lowering.

---

## References

The design draws on two works, and on the Triton language specification:

* LLaMCAT (arXiv:2512.00083) — MSHR-aware arbitration, progress balancing,
  hierarchical dynamic throttling.
* DCO (arXiv:2512.07312) — tensor/tile-level cache metadata, dead-block
  prediction, coordinated anti-thrashing and bypass.

Both PDFs are in `doc/`. Note that DCO's own datapath retains a private
scratchpad; removing it entirely, as this design does, goes further than either
paper and is the reason the register-file and completion protocols here are not
simply inherited.

---

## Contributing

The instruction encoding is the contract. If you change
`doc/isa/TNCA-ISA-v0.1.md`, change `rtl/common/tnca_defs.vh` and
`tools/asm/tnca_isa.py` in the same commit — `tb_decoder` and the assembler
round-trip exist to catch it when you do not.

`make test` must pass before a change lands. A new mechanism needs a test that
fails without it.

## License

Apache License 2.0 — see [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).

It covers the RTL, the toolchain and the documentation alike. Apache-2.0 was
chosen over a bare permissive licence for its Section 3 patent grant, which
matters more for a hardware design than for software; it was chosen over
Solderpad-2.1 and CERN-OHL for ecosystem fit, since the toolchain here lives
next to Triton and MLIR.
