# TNCA Compiler

The TNCA compiler lowers Triton kernels to TNCA machine code. It accepts Triton
IR (TTIR) produced by Triton's own frontend — from an ahead-of-time call, a
native `kernel[grid](...)` launch, or PyTorch Inductor — and emits a sealed
`.tnca` artifact that the public runtime (`tnca.runtime.Session`) loads and
runs on the RTL simulator or the Alveo U50 / U280 FPGA board.

Design, status and next steps: [`doc/compiler_overview.md`](../doc/compiler_overview.md)
(the original C0–C5 plan is [`doc/design/tnca_v1_compiler_draft.md`](../doc/design/tnca_v1_compiler_draft.md)).
Instruction-level support (what the ISA defines, what each profile implements,
what the compiler emits): [`doc/support_matrix.md`](../doc/support_matrix.md),
generated from the ISA schema, the target files and the compiler.

## Status

| Area | State | Evidence |
|---|---|---|
| Triton coverage | All 45 original TileBench operators (149 numerical cases/stages) compile **and** run correctly on the RTL at O0, O1 and O2 — 447/447 | `make r37`; [`doc/r37_validation.md`](../doc/r37_validation.md) |
| Native Triton launch | The same 149 cases through `kernel[grid](...)`; plugin and AOT code segments byte-identical for identical TTIR | `make r38-torch` (`plugin_identity.py`, `triton_native_run.py`) |
| PyTorch | `torch.compile` → Inductor → Triton → TNCA with zero host fallback on 22 offloading graph tests (23rd fallback detection test + device-assert trap check pass: 24/24) and all 52 tiny/small ArchBench points (operators and end-to-end LLaMA / GPT-2 / MLP / ResNet blocks) | `make r38-torch`, `make archbench-coverage`, `make archbench-tiny`; [`archbench/`](../archbench/README.md); [`doc/r38_pytorch_p1plus.md`](../doc/r38_pytorch_p1plus.md) |
| Tensor descriptors | Ranks 1–5 for TMA-style access (`tl.make_tensor_descriptor` with dynamic shapes/strides, host `TensorDescriptor`, zero/NaN padding, atomics, gather/scatter) executed via transfer engine or LSU. Inductor codegen supported (`configure(tensor_descriptors=True)`) | `make r38-tdesc` (`tools/r38/tdesc_tests.py`: 10/10 bit-exact); RTL `t33`–`t35` |
| Board path | Artifacts run unchanged through the U50 register protocol; Alveo U280 loaded and passed ladder B1–B6 (ArchBench tiny 27/27 and small 25/25); tiny cycles launch-by-launch identical to RTL simulation | `fpga-u50-sim` backend, `fpga-u280`; [`doc/r38_board_bringup.md`](../doc/r38_board_bringup.md) |
| Three-layer feature | Transposed tensor-DMA write (ISA §9E `mod[5]`, capability `tensor_transpose`) selected by the compiler on profile `core-xpose-v1` | `tools/r38/loop_tests.py`, `make r38-xpose-rtl` |
| Performance | Correctness-first baseline without instruction scheduling. Strict-bound efficiency η on tiny points is ~0.1 % (conv2d) to 22 % (fp32 add); harmonic mean E = 0.0072 / arithmetic mean A = 0.021 on U280 (models around 0.5 %–1.4 %; see Known limits) | `python -m archbench.report`; [`doc/compiler_overview.md`](../doc/compiler_overview.md) §13 |

## Compilation pipeline

Every TTIR entry point calls one lowering function, `tnca.aot._lower_ttir`, so the
AOT path and the Triton plugin cannot drift apart:

```text
@triton.jit ──► Triton frontend ──► TTIR
                                     │  legalize: upstream inliner (all levels; tl.* library calls)
                                     │  O1+: upstream canonicalize + CSE on the TTIR      (tnca_cc.upstream)
                                     ▼
                       frontend.TTIRImporter ──► TNCA IR                                    (frontend, descriptors,
                                     │                                                      tiles, packed, wide, mathlib)
                                     │  layout plans; O2: composed with upstream LinearLayout (layout)
                                     │  O1+: strength reduction, pure-scalar CSE, dead-value
                                     │       removal, lane-0 store narrowing                (optimization)
                                     ▼
                       codegen ──► TNCA assembly                                            (codegen, schedule,
                                     │  linear-scan allocation, async-ownership checks      footprints, protocol)
                                     ▼
                       tools/asm/tnca_asm.py ──► machine code ──► .tnca (manifest, ABI, provenance)
```

Entry points:

- **AOT** — `tnca.aot.compile_aot(fn, signature=..., constexprs=...)` (default `opt_level=0`).
- **Triton backend plugin** — `tnca.triton_backend`, registered as the
  `triton.backends` entry point `cpu` (and device type `tnca`). It serves `kernel[grid](...)` and PyTorch
  Inductor (`tnca.inductor.configure()`); default `TNCA_OPT_LEVEL=2`. See
  [`doc/r38_pytorch_p1plus.md`](../doc/r38_pytorch_p1plus.md).
- **TNCA IR builders** — `kernels.py`, `gemm.py`, `pipeline.py`, `partition.py`
  construct IR directly for the C1–C4 milestone tests.

Optimization levels:

| Level | Adds |
|---|---|
| O0 | Legalization only (inlining). Every program that compiles at O2 compiles at O0. |
| O1 | Upstream canonicalization and CSE on TTIR; TNCA-IR scalar optimizations. |
| O2 | O1 plus layout reasoning with the upstream `LinearLayout`; `no_ll` is the ablation. |

## Lowering model

The importer maps Triton's value model onto the machine explicitly; a construct
it cannot map raises a diagnostic naming the operation and is never skipped
(§11.1, §12).

| Triton value | TNCA representation |
|---|---|
| 1-D block | One vector register, VLEN = 16 lanes; tails are masked. Blocks wider than VLEN are rejected. |
| 2-D block | A tile (`tiles.py`) in one of six forms: a transfer-engine **window** in fragment layout (`tl.dot` operands, written by `t.ld.async`), an **accumulator** (16×16 binary32, ISA §9B) for computed 32-bit tiles, **rows** held in registers, **lazy** rules (recomputed rows, e.g. all `i1` tiles), **spilled** rows in scratch (non-4-byte elements, `i32` tiles, or ≥8 live accumulators), or **col**. A tile whose rows all alias (row stride 0, e.g. GEMM with M = 1) is one row load reused for every row. |
| Small N-D tensor | Packed into one vector in row-major order (`packed.py`); broadcasts and reductions permute lanes explicitly. |
| `i1` values | Predicate registers; converted via `vselect` when integers are required. 2-D `i1` splats lower to predicate tiles. |
| `tl.dot` | FP16 / BF16 / FP8 → `m.mma.async` with binary32 accumulation; FP32 → exact vector FMAs (no narrowing); INT8 → FP16 MMA, exact for K ≤ 1024. Mixed input types are rejected. |
| Transcendentals, division | SFU approximations (≤ 2⁻²² relative) with Newton refinement where needed; each function states its accuracy (`mathlib.py`). |
| Tensor descriptors | Rank 1–5, as Triton defines them for TMA: `tl.make_tensor_descriptor` (run-time shapes and strides), host `TensorDescriptor` arguments, `load`/`store`/`atomic_*`/`gather`/`scatter`, zero/NaN padding, signed coordinates. A box goes through the transfer engine (`t.ld.async`/`t.st.async`) when the result can live in a window, else through masked LSU accesses (`descriptors.py`). |
| 64-bit integers | Exact, as pairs of 32-bit lanes (`wide.py`) for 1-D blocks and one-row 2-D blocks. A 64-bit offset reaching a pointer is narrowed to a 32-bit gather index **with a check** that traps (`TRAP_USER 0xA5564`) instead of wrapping. |
| `tl.device_assert` | A real check: any failing active lane raises `TRAP_USER 0xA55E7`. |
| `tt.reduce` / `tt.scan` | ISA §8.7 reductions; multi-operand folds (argmax/argmin) and scans unrolled across lanes; NaN-propagating `maximumf`/`minimumf` built from specified behaviour. |
| Transpose | Row-by-row gather via LSU by default; on profiles declaring `tensor_transpose` (`core-xpose-v1`), a descriptor load whose only use is `tt.trans` lowers to a transposed transfer (`t.ld.async` with `mod[5]`). |
| Cache orchestration | Direct Cache Orchestration (`dco.py`): interprets loop trip counts, computes read access counts (nAcc), and emits TMU window configuration stores at kernel entry. |
| Asynchrony | Tensor DMA, MMA and mbarriers, with ownership checked at compile time (`footprints.py`, `protocol.py`). Automatic context specialization splits a kernel into producer/consumer roles (`partition.py`). |

## Module map

| Module | Responsibility |
|---|---|
| `frontend.py` | TTIR walk and lowering to TNCA IR: control flow, addressing, loads/stores, `tl.dot`, reductions, conversions |
| `descriptors.py` | TMA-style tensor descriptors (ranks 1–5), dynamic shapes/strides, LSU/transfer-engine routing, padding, atomics, gather/scatter |
| `tiles.py`, `packed.py`, `wide.py` | 2-D tile descriptions and placement; packed small tensors; exact 64-bit integers |
| `mathlib.py` | Transcendentals and division on the SFU |
| `ir.py` | The low-level TNCA IR and its builder |
| `optimization.py`, `layout.py`, `upstream/` | TNCA-IR passes; layout plans; the pinned upstream inliner, canonicalizer, CSE and `LinearLayout` |
| `codegen.py`, `asm.py` | Register allocation and assembly emission |
| `schedule.py` | Basic-block list scheduling against the measured machine latency and hazard model (`spec/r39_sched_model.json`) |
| `footprints.py`, `protocol.py`, `bankcost.py` | Implicit register footprints, the async ownership/event model, VRF bank-conflict cost |
| `partition.py`, `pipeline.py`, `gemm.py` | Context specialization (C4), the explicit two-role pipeline (C3), GEMM planning (C2) |
| `rctx_spmd.py` | Role Context SPMD execution across contexts for targets supporting sliced resources (`rctx_slice`) |
| `dco.py` | Direct Cache Orchestration (DCO): tile access analysis and TMU window configuration |
| `target.py`, `isa.py`, `triton_options.py` | The target contract, the ISA schema, what the backend reports to Triton |
| `report.py`, `diagnostics.py` | Per-kernel resource/bottleneck report; capability and resource diagnostics |

The PyTorch integration (`python/tnca/inductor.py`) and the Triton plugin
(`python/tnca/triton_backend/`) sit outside this package and call into it only
through `tnca.aot._lower_ttir`.

## Known limits

- **Block width.** Each block dimension is at most VLEN (16). Inductor is
  configured accordingly (blocks ≤ 16, one row per reduction program, one
  16×16×16 GEMM configuration); a wider tensor split across registers is not
  built yet.
- **Raw half-precision arithmetic.** Hand-written FP16/BF16 arithmetic without
  explicit upcasts emits binary32 instructions on half-precision bit patterns
  (miscompile). Inductor always upcasts to FP32 before arithmetic
  (`codegen_upcast_to_fp32`), so Inductor kernels are unaffected.
- **Control flow.** `scf.if` (dynamic branches on runtime values) and control-flow
  edges carrying block arguments are rejected. Inductor kernels do not emit them.
- **2-D tiles and descriptors.** TensorMap transfers need a unit column stride and a positive
  row stride; other access patterns use the LSU row by row. Multi-operand folds
  and 64-bit integers are supported on one-row 2-D blocks only. Descriptor box rows
  wider than 16 are rejected.
- **64-bit integers** are emulated lane by lane in scalar registers: exact, but
  slow. They appear mainly in embedding and token lookups.
- **Tensor residency.** Every launch round-trips through host memory; tensors do not
  yet remain resident on device across kernel launches (planned for PrivateUse1 in P2).
- **Instruction scheduling and allocation.** Instructions are emitted in program order
  unless list scheduling is enabled; the register allocator does not spill (resource
  exhaustion raises `ResourceError`).
- **Performance** has not been tuned. On the tiny ArchBench points, time is dominated by
  back-to-back register hazards (~30 %), single 8-byte memory port traffic, and VRF bank
  conflicts (`doc/r38_stage1_status.md`, `doc/compiler_overview.md` §13); the profiler
  and cost model of R38 exist to attribute it.
- **Execution model.** Single core; a grid's programs run serially (`serial-grid`),
  except where `rctx_spmd` distributes grid programs across role contexts on sliced targets.

## Build and test

Python 3.12 and the pinned Triton 3.8.0 (`TRITON_PYTHON`) cover everything
except the MLIR half. PyTorch lives in its own environment (`TORCH_PYTHON`,
`spec/r38/torch.lock.json`).

```sh
make compiler            # ISA/target staleness checks, unit tests, IR tests
make compiler-rtl        # the compiler's milestone kernels on the RTL
make triton-e2e          # Triton → TTIR → TNCA → RTL
make r37                 # 45 TileBench operators at O0/O1/O2 (447 cases)
make r38-identity        # every artifact byte-identical to the frozen set
make r38-torch           # plugin identity, native launches, PyTorch graph tests
make r38-tdesc           # tensor descriptors (ranks 1–5, TMA, host descriptors) on RTL
make archbench-coverage  # every tiny/small ArchBench point compiles for TNCA
make archbench-tiny      # run ArchBench tiny tier on the RTL simulator
```

Debugging a kernel:

```sh
TNCA_KERNEL_DUMP=1 TNCA_DUMP_DIR=/tmp/dump python -m archbench.run --platform tnca --point <id> --compile-only
python tools/r38/ttir_lower.py /tmp/dump/<kernel>/<kernel>.ttir --asm   # same lowering, full traceback
```

Per-stage IR dumps (TTIR, inlined TTIR, optimized TTIR, TNCA IR, assembly,
machine code) are described in [`doc/dumping_ir.md`](../doc/dumping_ir.md).

### MLIR half

The `tnca` dialect, its async-resource verifier
(`lib/Dialect/TNCA/Transforms/VerifyAsyncOwnership.cpp`) and the `tnca-opt`
driver build against the pinned LLVM:

```sh
make llvm-toolchain    # fetch the prebuilt LLVM Triton pins (sha256-checked)
make tnca-opt
```

`make compiler` skips the IR tests when `tnca-opt` is absent.

## Single sources of truth

- **ISA.** `tools/asm/tnca_isa.py` → `isa/tnca-isa-v0.1.json` (`make isa`) →
  `include/tnca/ISA/TNCAOpcodes.inc`, `TNCAInstrInfo.td` and
  `doc/isa/opcode-map.md`. The RTL's `rtl/common/tnca_defs.vh` is hand-written
  and `tools/isa/check_consistency.py` holds the two together.
- **Target.** `targets/*.json` are generated from the RTL parameters,
  `spec/r1_profiles.json`, and the ISA schema (`make target`, `tools/isa/gen_target.py`)
  and read back by the hardware in `tb/core/progs/t15_target.s`: RTL parameter → target file → test → special
  register.

| Target file | Profile | Elaboration |
|---|---|---|
| `tnca-sim-v0.1.json` | `core-sim` | 2 role contexts, 64 logical vector registers each |
| `tnca-sim-vrf-v1.json` | `core-vrf-sync-v1` | banked VRF; initial U50 / U280 board image |
| `tnca-sim-vrf-v2.json` | `core-vrf-sync-v2` | `core-vrf-sync-v1` + VRF bypass, no-bubble FE, prefetch; R39 board image |
| `tnca-sim-vrf-v3.json` | `core-vrf-sync-v3` | `core-vrf-sync-v2` + RCTX switch, 2 LSU in-flight, FPU overlap, `rctx_slice` |
| `tnca-sim-vrf-v4.json` | `core-vrf-sync-v4` | 4 role contexts, 64 registers each (256 physical registers) |
| `tnca-sim-vrf-v5.json` | `core-vrf-sync-v5` | `core-vrf-sync-v4` + transposed transfer, context pairs (`RPAIR`), MMA A buffer |
| `tnca-sim-xpose-v1.json` | `core-xpose-v1` | `core-sim` + transposed tensor-DMA write (`tensor_transpose`) |
| `tnca-sim-4rctx.json` | `core-4r` | 4 role contexts, 32 registers each |
| `tnca-sim-2x2.json` | `system-2x2` | 2×2 tiles, no SFU |

## Layout

```text
python/tnca_cc/    the compiler (Python)
targets/           generated target descriptions
include/, lib/     the tnca MLIR dialect, its interfaces and the async verifier
tools/tnca-opt/    the MLIR driver
pin/               pinned Triton/LLVM revisions and the LLVM fetch script
test/              unit tests, IR tests, RTL closure tests
```

## Milestones

| Milestone | State |
|---|---|
| C0 Toolchain, target contract, ISA schema, `tnca-opt` | Done |
| C1 Triton → TTIR → TNCA IR → assembly → RTL | Done |
| C2 Single-context tensor DMA + MMA | Done |
| C3 Explicit two-role pipeline | Done |
| C4 Automatic context specialization | Done |
| C5 Attention overlap and performance feedback | Not started as planned; profiler, cost model and ArchBench provide feedback |
| R3.7 Original TileBench, 45/45 operators at O0/O1/O2 | Done |
| R3.8 PyTorch via Inductor; first three-layer feature; profiler-driven work | In progress ([`doc/r38_stage1_status.md`](../doc/r38_stage1_status.md)) |

## Hardware issues found by the compiler

Compiling real kernels has repeatedly exposed RTL and ISA defects that no
directed RTL test reached — among them a dropped completion when the matrix
unit and the transfer engine finished in the same cycle, a barrier `wait` that
livelocked on an already-complete generation, an engine fault that recorded a
trap without stopping its context, a transposed-transfer mode that was
silently ignored, and an instruction counter (`done_retired`) accumulating
across grid programs. Each is recorded with its evidence in
[`doc/conformance.md`](../doc/conformance.md),
[`doc/r37_isa_gaps.md`](../doc/r37_isa_gaps.md), and
[`doc/r38_stage1_status.md`](../doc/r38_stage1_status.md).
