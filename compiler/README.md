# TNCA compiler backend

Built to `doc/compiler_overview.md`. This directory is milestone **C0**: the
pinned toolchain, the out-of-tree backend skeleton, `tnca-opt`, TargetInfo, the
ISA schema, and the per-stage IR dump points.

## What runs today, and what does not

The plan splits code generation into two paths on purpose (§11.1, §11.2), and
that split decides what is usable right now.

| Path | Status |
|---|---|
| Bring-up: TNCA IR → assembly → `tools/asm/tnca_asm.py` → RTL (§11.1) | Python. Runs with nothing installed. |
| MLIR: the `tnca` dialect, `tnca-opt`, the async verifier | **Builds and runs.** |
| Triton frontend → TTIR → TNCA IR → RTL | **Closed.** `make triton-e2e` |
| Triton → context-specialised (multi-role) ISA | **Closed.** |
| Thin LLVM target (§11.2) | Encodings generated; the target itself is deliberately not started. |

### Building Triton

Triton 3.8.0 builds on macOS arm64 against the pinned LLVM, though it is not
officially supported there. What it needed:

```
GIT_CONFIG_COUNT=2 GIT_CONFIG_KEY_0=http.proxy GIT_CONFIG_VALUE_0= \
GIT_CONFIG_KEY_1=https.proxy GIT_CONFIG_VALUE_1= \
TRITON_APPEND_CMAKE_ARGS="-DTRITON_BUILD_UT=OFF" \
TRITON_BUILD_PROTON=OFF MAX_JOBS=10 \
  pip install -e . --no-build-isolation
```

Leave `LLVM_SYSPATH` unset — Triton finds `llvm-<rev>-macos-arm64-1` in
`~/.triton/llvm` by name. Pointing it at the `llvm-macos-arm64` symlink makes
`setup.py` link that name onto itself, and every later configure then fails
with "Too many levels of symbolic links" while reporting only that it could
not find MLIR.

### Getting the MLIR half

```
make llvm-toolchain     # fetch the pinned LLVM (about 355 MB, sha256 verified)
make tnca-opt           # configure and build
make compiler           # unit tests + IR tests
```

`fetch_toolchain.sh` downloads the prebuilt LLVM that Triton itself uses for
the pinned revision, into `~/.triton/llvm` — the directory Triton's own build
searches — so a later Triton build reuses it rather than fetching a second
copy. The URL is not invented: it is the scheme `python/build_helpers.py`
constructs, and the sha256 is the one `cmake/llvm-info.json` publishes for the
platform. A mismatch refuses to unpack.

`make compiler` skips the IR tests rather than failing when `tnca-opt` has not
been built, so a checkout with no LLVM still runs everything else.

## Single ISA source (§11.3)

> 不要再手工维护第三套独立 opcode 数字。

There is one set of opcode numbers. Everything else is generated:

```
tools/asm/tnca_isa.py            authoring form; assembler and RTL tests use it
  └─ isa/tnca-isa-v0.1.json      canonical schema   (make isa)
       ├─ compiler/include/tnca/ISA/TNCAOpcodes.inc    C++ x-macro table
       ├─ compiler/include/tnca/ISA/TNCAInstrInfo.td   TableGen, for §11.2
       └─ doc/isa/opcode-map.md                        the encoding map
```

`rtl/common/tnca_defs.vh` is hand-written, because the RTL is what the ISA
describes rather than a consumer of it. `tools/isa/check_consistency.py` holds
the two together — it compares all 198 shared constants and fails the build on
any disagreement.

## Target contract (§15 C0)

`compiler/targets/tnca-sim-v0.1.json` is generated from the RTL parameters by
`tools/isa/gen_target.py`, never typed by hand, and
`tb/core/progs/t15_target.s` asserts the hardware agrees with it. The chain is:

```
RTL parameter → target file → t15_target.s → special register read
```

A break anywhere in that chain fails the regression, so the file a compiler
trusts is the file the silicon confirms.

## Layout

```
pin/toolchain.json       pinned Triton + LLVM, verified against upstream
pin/fetch_toolchain.sh   fetch the prebuilt LLVM, sha256-checked
targets/                 generated target descriptions
python/tnca_cc/          the bring-up compiler: target, ISA, diagnostics,
                         implicit footprints (§13.4), protocol model (§16.2)
backend/                 the Triton out-of-tree plugin
include/, lib/           the tnca dialect (§5.3) and its two interfaces (§5.4)
lib/Dialect/TNCA/Transforms/  the async-resource verifier of §5.4
tools/tnca-opt/          the IR driver every §16.2 unit test runs against
test/                    Python unit tests and `test/IR/*.mlir`; `make compiler`
```

## Commands

```
make isa        regenerate every ISA artifact from the single source
make target     regenerate the target description from the RTL
make compiler   unit tests + staleness checks for every generated file
make pin        re-verify the pinned toolchain against upstream (needs network)
```

## What building this found in the hardware

The point of doing the compiler now is that it asks questions no RTL test was
asking. Nine so far, all recorded in
[`../doc/conformance.md`](../doc/conformance.md):

From the plan itself:

- `CAPABILITY0.TENSOR_DMA` read 0 while the transfer engine executed TXFER.
- The four whole-program resources (§10.1) were not reportable by the hardware,
  so a target description could not be checked against the machine.
- A transfer with its coordinate base at `s63` silently ran at coordinate 0 on
  its second axis, because the implicit `Sb+1` read wrapped onto `S0`.

From C2, compiling a GEMM with an N tail:

- The transfer engine's source-alignment requirement was documented in three
  places and checked in none, so an odd row stride returned data shifted by two
  bytes per row and raised nothing.
- An engine fault recorded the trap but never stopped the Role Context that
  issued the command, so a `DESCRIPTOR_FAULT` surfaced as a later `TRAP_USER`.

From C3, propagating a failure through a pipeline:

- A faulting command signalled an *ordinary* completion, releasing its consumer
  to read a window nothing had written.
- A waiter woken by a barrier error resumed instead of faulting, so a located
  fault became a hang on the next generation.
- `t.ld.async` accepted a lane mask the engine ignored, so a predicated-off
  stage performed its transfer anyway.

From C4:

- A taken branch failed to squash a younger instruction of the same context
  when the front end was stalled. Only reachable with two Role Contexts.
- The launch contract had no race-free way for a role to wait for another's
  `mb.init`, so the compiler polled a memory flag — and the polling starved the
  producer's transfer engine until the whole kernel deadlocked. Barriers are
  now guaranteed quiescent at kernel entry (ISA §9A.7).
- `mb.wait` on an already-complete generation livelocked: the sleep predicate
  required the barrier to have been armed and the wake predicate did not, so a
  waiter oscillated instead of returning.
- Both asynchronous engines shared one completion port into the barrier unit.
  When the matrix unit and the transfer engine finished in the same cycle, the
  transfer's completion was silently dropped and its barrier waited for ever.

The last one is worth a note on method: it presented as a clean deadlock under
a slow memory model, and the fix was only defensible after instrumenting both
`done` signals and finding exactly one coincident cycle in the failing run.
Without that measurement it would have been a plausible story rather than a
demonstrated cause.

The third is worth separating from a related case that was *not* changed.
Overwriting an `m.mma.async` source fragment while the matrix unit is still
reading it is also silently wrong — but ISA §9C.3 states that and assigns it to
the compiler, so it is undefined by design, and `footprints.py` models it. The
`s63` case was undefined by accident: nothing said what happened, and what
happened was a wrong address. That is the difference between a contract the
compiler can honour and a trap it cannot see.

## Milestones

| | State |
|---|---|
| **C0** toolchain, TargetInfo, ISA schema, `tnca-opt` | done |
| **C1** Triton → TTIR → TNCA IR → assembly → RTL | done |
| **C2** single-RCTX Tensor DMA + MMA | done |
| **C3** explicit two-role pipeline | done |
| **C4** automatic Context Specialization | done |
| **C5** attention and performance feedback | not started |

Each milestone's cases run on the RTL: `make compiler-rtl`.

## C1: the first closure

§15 C1 sets the bar plainly — 不修改输出汇编，不人工补 barrier，直接运行编译
产物. The first case now clears it.

`tnca_cc.lower.lower_elementwise` emits TNCA assembly for a masked elementwise
kernel. `compiler/test/test_rtl_closure.py` assembles that output with the
existing `tools/asm/tnca_asm.py` and runs it on the Icarus model of the core,
appending only data and assertions — never an edit to an emitted instruction.
It is checked two ways: instruction-for-instruction against the hand-written
`tb/core/progs/t16_vecadd.s` (§11.1 keeps that comparison available so a
compiler bug and an encoding bug can be told apart), and by the arithmetic
itself on the RTL, masked tail included.

Still to do for C1: integer and FP32 reduction, gather/scatter, blocks wider
than one vector register, and the Triton frontend in front of it. The lowering
takes a small kernel description rather than TTIR today, which is why it is a
slice and not the milestone.

Anything not lowered raises rather than approximating. §11.1 is explicit:
资源不够可以先明确报错，不能为了快速"支持"而偷偷破坏异步生命周期.

## C2: Tensor DMA and MMA

`tnca_cc.gemm` plans and lowers a GEMM tile. The planner exists because of
§9: a 32×64 B tile loaded row-major puts 64·sizeof(dtype) between the rows of
any 32×16 column sub-block where the MMA fragment wants 16·sizeof(dtype), and
moving the fragment base cannot close that. §9 recommends splitting into
separate transfers, each landing compact — and notes that the choice changes
the DMA command count, the barrier operation count, the window count, the
descriptor shapes and the RF live ranges at once. So one planner decides all
of them, which is §9's conclusion: 布局选择、通信协议和流水资源预算必须联合
规划.

FP16 and BF16, different strides, and M/N/K tails all run against
`fp32.mma_ref`, which accumulates in increasing k order with one rounding per
step as ISA §9C.1 requires.

## C3: the two-role pipeline

§15 calls this 第一阶段最重要的里程碑. `tnca_cc.pipeline` builds a producer and
a consumer over an RF channel — not a shared-memory Aref, because there is no
scratchpad for one to live in. Ready and free are separate barriers: ISA
§9A.1 warns that a transfer completing is not the consumer having finished
reading, and a pipeline with one barrier per slot implements only load-ready.

Covered: 0/1/2/odd/6 iterations, two- and three-slot rings, predicated-off
async, delayed completion and out-of-order returns (against the slow and
reordering memory models), error propagation, and draining at exit.

## C4: automatic partitioning

`tnca_cc.partition` follows §7's six steps. The graph carries more than SSA
def-use — loop-carried, memory, control, accumulator-order and async-resource
edges — because a partitioner that sees only def-use will split an
accumulator's updates across roles or move a write past a command still
reading the register.

The physical RCTX count is applied last, by merging, per §7.6:
不能让物理 RCTX 数决定最开始的数据流图表示. The same graph maps to two contexts
(P | M+E) and to four (P | M | E, nothing merged). Transfer and matrix are
never merged: they are the two that overlap in time.

§15 C4 defers the four-context check to 四 RCTX 硬件 profile 可用后. That
profile now exists — `make core4` elaborates the same core with `NUM_RCTX=4`
and half the logical vector registers each — with its own generated target
description, and a one-producer/two-consumer channel runs on it. The property
it tests is §7.4's: 只有两个 consumer 都释放，producer 才能覆盖该 slot.

Not done: the literal `P + M0 + M1 + E` split. The mapper *merges* partitions
when contexts are scarce; splitting the matrix partition across two roles for
throughput is a different operation, and it does not perform it.

### From Triton to context-specialised ISA

`split_roles` closes the gap between the two halves. It takes the single
region the Triton frontend produces and emits one region per Role Context,
following §7.3's order:

- a value whose **whole dependency chain is pure** is recomputed in every role
  that needs it rather than communicated. This is recursive on purpose: a
  predicate is one `v.cmp`, but it comes from `pid*BLOCK + arange`, and
  recomputing the comparison while communicating its inputs would be worse
  than recomputing all of it;
- a tensor that cannot be recomputed — anything rooted in memory or in an
  engine — crosses on a **shared RF window**, which §7.3 ranks above every
  other option and which is the only one TNCA has;
- the handoff is ordered by a barrier: the producer arrives after its last
  write, each consumer waits before its first read, and ISA §9A.5 makes that
  arrival cover the outstanding register write.

For `add_kernel` that is one crossing (the sum), five operations
rematerialised, 20 instructions in the producer and 12 in the consumer. The
test checks the split is structurally real — the epilogue role contains no
load, the producer contains no store — not merely that the arithmetic comes
out right.

Limit: straight-line regions, which is what the frontend emits for one program
id. §7.5's structured-loop cloning exists for the GEMM shape
(`specialize_gemm_rolled`) but is not yet driven from TTIR.

Communication follows §7.3's order, and on this target nothing falls back to
a global temporary — which is the point for a machine with no scratchpad.

Both role-loop forms are built. `specialize_gemm` unrolls the k loop, which
§9 recommends as the first version so every `W`, `V` and `A` number stays a
static encoding field. `specialize_gemm_rolled` clones the loop into each role
instead (§7.5): the body covers one trip through the slot ring, so the numbers
stay static while the code stops growing with K — 105 instructions at K=64 and
at K=128, against 110 and 166 unrolled. Each role derives its barrier
generation from its own trip counter, because it cannot read the other's token:
the scalar file is per-RCTX private (§7.3).
