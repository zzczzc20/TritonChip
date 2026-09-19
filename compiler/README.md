# TNCA compiler backend

Built to `doc/compiler_overview.md`. This directory is milestone **C0**: the
pinned toolchain, the out-of-tree backend skeleton, `tnca-opt`, TargetInfo, the
ISA schema, and the per-stage IR dump points.

## What runs today, and what does not

The plan splits code generation into two paths on purpose (§11.1, §11.2), and
that split decides what is usable right now.

| Path | Status |
|---|---|
| Bring-up: TNCA IR → assembly → `tools/asm/tnca_asm.py` → RTL (§11.1) | Python only. Runs here with nothing installed. |
| MLIR: the `tnca` dialect, `tnca-opt`, the Triton plugin | Written, **not built**. Needs the pinned LLVM. |
| Thin LLVM target (§11.2) | Encodings generated; the target itself is deliberately not started. |

This host is macOS arm64 with no LLVM, no MLIR and no Triton, and Triton
publishes no macOS wheels. So nothing in `lib/`, `tools/tnca-opt/` or the
TableGen files has been compiled — they are unverified beyond review. The
Python half, the ISA schema, the target contract and the protocol model are
tested, and `make compiler` runs them.

Getting to a built MLIR pipeline needs either the prebuilt LLVM that Triton
publishes for the pinned revision (`compiler/pin/toolchain.json` records the
per-platform sha256 sums, `macos-arm64` among them) or a source build, and
then a Linux host for Triton itself.

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
targets/                 generated target descriptions
python/tnca_cc/          the bring-up compiler: target, ISA, diagnostics,
                         implicit footprints (§13.4), protocol model (§16.2)
backend/                 the Triton out-of-tree plugin
include/, lib/           the tnca dialect (§5.3) and its two interfaces (§5.4)
tools/tnca-opt/          the IR driver every §16.2 unit test will run against
test/                    unit tests; `make compiler`
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

## Where C0 stops

No lowering exists yet. `TNCABackend.add_stages` raises rather than returning
a half pipeline, because a backend that compiles something almost right is
worse than one that refuses: §11.1 asks for an explicit error when a resource
does not fit, not a quiet break of an asynchronous lifetime.

C1 is the first real Triton→RTL closure — vector add, masked elementwise,
integer and FP32 reduction, simple control flow — with the requirement that
the emitted assembly runs unmodified, with no hand-added barrier.
