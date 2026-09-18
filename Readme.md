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

**当前进度：阶段 1a。** 单核、2 个 Role Context、16 lane SIMD、完整标量/向量/
谓词/shuffle/归约指令、IEEE-754 binary32 运算、带合并器的访存单元。已通过
51 万组 FMA 差分向量、627 组编码合法性向量和 5 个自检核心程序（含慢速、
带反压的存储器变体）。**尚未实现**：Matrix 单元、TensorMap/Tensor DMA、
mbarrier、SFU、原子操作、mesh 与 NUCA。路线图见下文 Roadmap。

> 本仓库是一个诚实的在建工程，不是一颗流片过的芯片。凡是没做的，
> 下面都写明了没做；凡是测过的，都给出了测试规模。

---

## Status

**Stage 1a — a single core that executes the frozen Stage-1 ISA subset.**

| Area | State |
|---|---|
| ISA v0.1 encoding | frozen, machine-readable, enforced by the decoder |
| Scalar integer + control flow | complete |
| Vector integer, predicate, shuffle, reduction | complete |
| IEEE-754 binary32 add / sub / mul / fma / min / max / compare / convert | complete |
| FP reduction (bit-reproducible balanced tree) | complete |
| Load/store unit with a physical-word coalescer | complete |
| 2 Role Contexts on a shared datapath | complete |
| Assembler, disassembler, exact FP oracle | complete |
| Matrix unit, TensorMap, Tensor DMA, mbarrier | **not started** (Stage 2) |
| SFU (`v.exp2.approx` et al) | **not started**; the capability bit reads 0 and the encodings trap |
| Atomics, fences | **not started** (Stage 2) |
| Mesh, NUCA slices, MSHRs, flow control | **not started** (Stage 3–4) |
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

The Stage-1 core in this repository is the foundation for (1) and (3): it
establishes the register files, the completion protocol and the memory
semantics that the tensor machinery will be built on.

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
  common/tnca_defs.vh         encoding constants, derived from the ISA doc
  core/
    tnca_core.v               4-stage in-order pipeline, N Role Contexts
    tnca_decoder.v            encoding -> control, and encoding legality
    tnca_scalar_alu.v         64-bit integer ALU
    tnca_vector_alu.v         16-lane integer / predicate / shuffle / reduce
    tnca_vector_fpu.v         16-lane binary32, plus the reduction sequencer
    tnca_fp32_fma.v           IEEE-754 binary32 fused multiply-add
    tnca_lsu.v                load/store with a physical-word coalescer
    tnca_scalar_rf.v          64 x 64 bit per RCTX, 3R1W, write-first
    tnca_vector_rf.v          physical vector pool, per-lane write enables
    tnca_pred_rf.v            16 x VLEN bit per RCTX
tb/
  core/tb_tnca_core.v         runs a self-checking program
  core/tb_fp32_fma.v          differential test against the exact oracle
  core/tb_decoder.v           encoding legality
  core/progs/*.s              the self-checking test programs
  common/tnca_mem_model.v     unified memory, parameterised latency/stalls
tools/
  asm/tnca_isa.py             machine-readable encoding tables
  asm/tnca_asm.py             assembler
  asm/tnca_dis.py             disassembler and round-trip checker
  model/fp32.py               exact binary32 semantics (Fractions, not floats)
  model/gen_*_vectors.py      test-vector generators
flow/
  synth/synth.py              Yosys synthesis + PPA proxy report
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
| assembler round-trip | assemble → disassemble → re-assemble reproduces the original word | 2,925 instructions |
| slow-memory rerun | every core test again at memory latency 7 with pseudo-random `ready` de-assertion | catches any fixed-latency assumption |

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

| module | gates | flops | depth | |
|---|---:|---:|---:|---|
| `tnca_decoder` | 682 | 0 | 17 | ok |
| `tnca_scalar_alu` | 26,017 | 0 | 128 | **over budget** — the 64×64 multiplier |
| `tnca_scalar_rf` | 31,033 | 8,192 | 10 | flops; belongs in SRAM |
| `tnca_pred_rf` | 1,834 | 512 | 6 | ok |
| `tnca_fp32_fma` | 6,781 | 0 | 207 | **over budget by the largest margin** |
| `tnca_vector_alu` | 145,334 | 0 | 99 | **over budget**; the largest block |

[`doc/synthesis.md`](doc/synthesis.md) has the full table and the reasoning.
Two findings matter now:

* **The FP datapath cannot be clocked at 1 GHz as written.** 207 gate levels is
  3–5 ns of logic. The FP problem here is *depth, not area*: the FMA is only
  6,781 gates, so sixteen lanes is about 109 k gates, which is affordable —
  sixteen lanes at 207 levels is not buildable at the target frequency. This is
  the first Stage-1b task, and the core is already shaped to absorb it.
* **The register files must not stay flop-based.** 31 k gates for 8 kbit of
  scalar state is the cost of three asynchronous read ports built from
  multiplexer trees. §5.3 of the architecture document already specifies the
  replacement: 16 banks of 64 bit, 1R1W each, with an operand collector, sized
  *before* the arithmetic is sized.

Neither is a surprise — §6.2 of the architecture document says not to treat a
MAC-count ceiling as achievable throughput, and §5.3 says to make the RF ports
real before deciding the compute. The value of measuring is that the numbers
are now on the record rather than assumed.

Measuring has already paid for itself twice. The vector ALU was first written
the obvious way and synthesised to **205,205 gates**; applying §6.2 literally —
deriving the signed high product from the unsigned one instead of building a
second multiplier array, and sharing one shuffle crossbar and one subtractor
instead of five and six — brought it to **145,334 gates, −29 %, for identical
behaviour**, checked by the same 266 vector tests before and after. The scalar
ALU dropped 23 % the same way.

---

## Roadmap

Stages follow `doc/architecture_overview.md` §21.

* **Stage 1a — done.** Single core, ISA v0.1 subset, verified.
* **Stage 1b — next.** Close the gap between the ISA that is verified and the
  clock the design claims. In order:

  1. **Pipeline the FP unit.** Cut `tnca_fp32_fma` into 3–4 stages
     (unpack/align → multiply/add → normalise/round). Target: FMA depth under
     60 levels. The core already has a per-unit latency notion and an
     out-of-band completion path, so this adds a writeback reservation shift
     register and nothing else. The 516 k-vector differential test is the
     safety net, and it must still pass unchanged.
  2. **Pipeline the scalar multiplier**, leaving the rest of the ALU at one
     cycle. The signed-high correction lands in the last stage, where it costs
     nothing.
  3. **Synchronous instruction fetch**: SRAM plus a fetch buffer, removing the
     one deviation in the front end (`doc/conformance.md` D1).
  4. **SFU** — `v.exp2.approx`, `v.rcp.approx`, `v.rsqrt.approx`,
     `v.log2.approx`. Table plus polynomial, with the accuracy **measured over
     the full input range and the measured bound written into the ISA** before
     `CAPABILITY0.SFU` is allowed to read 1. §12.7 requires a published error
     bound, NaN/Inf/zero behaviour and subnormal handling; a capability bit set
     without those is exactly the dishonesty §1.6 warns about.
  5. **Dual issue** across resource classes, once the pipelined units make it
     worth anything.
  6. **Shrink the integer vector unit** along the lines in `doc/synthesis.md`
     §3 (shared funnel shifter, shared `clz`/`popc`).
* **Stage 2.** Tensor descriptors and Tensor DMA, RF windows, mbarrier with
  the ready/free two-event protocol, the matrix unit (`m.mma.async`,
  M16 N16 K16/32/64), accumulator registers with vector access. Target
  workloads: asynchronous double-buffered GEMM, small FlashAttention.
* **Stage 3.** 2×2 mesh, static-home NUCA slices, MSHR entries and targets,
  separated request/response networks, forward-progress credits. No adaptive
  policy yet — establish a trustworthy baseline first.
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
