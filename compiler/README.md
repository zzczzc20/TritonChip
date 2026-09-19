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
asking. Three so far, recorded in [`../doc/conformance.md`](../doc/conformance.md):

- `CAPABILITY0.TENSOR_DMA` read 0 while the transfer engine executed TXFER.
- The four whole-program resources (§10.1) were not reportable by the hardware,
  so a target description could not be checked against the machine.
- A transfer with its coordinate base at `s63` silently ran at coordinate 0 on
  its second axis, because the implicit `Sb+1` read wrapped onto `S0`.

The third is worth separating from a related case that was *not* changed.
Overwriting an `m.mma.async` source fragment while the matrix unit is still
reading it is also silently wrong — but ISA §9C.3 states that and assigns it to
the compiler, so it is undefined by design, and `footprints.py` models it. The
`s63` case was undefined by accident: nothing said what happened, and what
happened was a wrong address. That is the difference between a contract the
compiler can honour and a trap it cannot see.

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
