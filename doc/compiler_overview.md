# TNCA Compiler and Software Stack Overview

Status as of 2026-10-08 (branch `r38`). This document describes how a PyTorch or Triton program reaches the TNCA hardware today: the Triton-to-TNCA compiler, the runtime and its backends, and the PyTorch integration. It also covers how this stack is verified, what it cannot do yet, and what comes next. Its companion, [`architecture_overview.md`](architecture_overview.md), describes the hardware.

For day-to-day use (build targets, module map, debugging commands), see [`compiler/README.md`](../compiler/README.md).

> **Supersedes the September 2026 compiler plan.** The previous `compiler_overview.md` was a design plan for a hybrid MLIR/Triton backend with milestones C0–C5 (in Chinese). It is preserved unchanged at [`design/tnca_v1_compiler_draft.md`](design/tnca_v1_compiler_draft.md). References such as "`compiler_overview.md` sec 11.3" or "sec 15 C0" in tool docstrings (for example `tools/isa/*.py`) and older documents refer to that plan, whose section numbers are unchanged.

---

## Contents

1. [Summary](#1-summary)
2. [The stack end to end](#2-the-stack-end-to-end)
3. [Front ends](#3-front-ends)
4. [The lowering pipeline](#4-the-lowering-pipeline)
5. [Value model: from Triton blocks to registers](#5-value-model-from-triton-blocks-to-registers)
6. [Operation coverage and the error model](#6-operation-coverage-and-the-error-model)
7. [Code generation, artifact and ABI](#7-code-generation-artifact-and-abi)
8. [Runtime and launch path](#8-runtime-and-launch-path)
9. [PyTorch integration](#9-pytorch-integration)
10. [One ISA source](#10-one-isa-source)
11. [The MLIR dialect and the C1–C4 builders](#11-the-mlir-dialect-and-the-c1c4-builders)
12. [Verification](#12-verification)
13. [Performance today](#13-performance-today)
14. [Known limitations](#14-known-limitations)
15. [Next steps](#15-next-steps)
16. [Plan versus delivery](#16-plan-versus-delivery)

---

## 1. Summary

| Capability | State | Evidence |
|---|---|---|
| Triton kernels (AOT) | All 45 original TileBench operators, 149 cases, compile and run correctly on the RTL at O0, O1 and O2: 447/447 | `make r37`, [`r37_validation.md`](r37_validation.md) |
| Native `kernel[grid](...)` launches | Through the `triton.backends` plugin; the plugin's code is byte-identical to AOT for the same TTIR | `make r38-torch` (`plugin_identity.py`, `triton_native_run.py`) |
| PyTorch | `torch.compile` → Inductor → Triton → TNCA with zero host fallback on 22 graph tests (a 23rd checks that a real host fallback, `torch.sort`, is detected) and on all 52 tiny/small ArchBench points (operators plus LLaMA prefill/decode, GPT-2 block, MLP, ResNet block) | `make r38-torch`, `make archbench-coverage` |
| Tensor descriptors | Ranks 1–5, as Triton defines them for TMA on NVIDIA GPUs: `tl.make_tensor_descriptor` with run-time shapes and strides, host `TensorDescriptor` arguments, `load`/`store`/`atomic_*`/`gather`/`scatter`, zero or NaN padding, signed coordinates. Executed by the transfer engine (`t.ld.async`/`t.st.async`, rank 1–5) or the LSU. Inductor's own tensor-descriptor code generation is available with `configure(tensor_descriptors=True)` | `tools/r38/tdesc_tests.py` (10/10 on the RTL, bit-exact); RTL `t33`–`t35` |
| Execution | RTL simulator (`core-sim`, `core-vrf-sync-v1`, `core-xpose-v1`), the board protocol in simulation (`fpga-u50-sim`), and the Alveo U280 board | ArchBench tiny 27/27 on RTL and on the U280; small 25/25 on the U280 |
| Cycle fidelity | Board cycles equal simulator cycles launch by launch (tiny tier) | [`architecture_overview.md`](architecture_overview.md) §10.1 |
| Compiler stability | 447 R3.7 artifacts are byte-identical before and after the R38 compiler changes | `make r38-identity` |
| Performance | Correctness-first; no instruction scheduling; η ≈ 1 % on models | §13 |

## 2. The stack end to end

```text
 PyTorch program
   │  torch.compile(model)            tnca.inductor.configure(): device "cpu" kernels go to Triton,
   ▼                                  blocks ≤ 16, one config per kernel, SDPA decomposed, no FX cache
 Inductor ── generated Python wrapper + @triton.jit kernels
   │
   ▼
 Triton 3.8.0 frontend ──► TTIR
   │  backend plugin "tnca" (python/tnca/triton_backend): stages ttir → tnca
   ▼
 tnca.aot._lower_ttir  ─────────────────────────────── the one lowering every entry point uses
   │  upstream inliner; O1+: canonicalize + CSE      (compiler/python/tnca_cc/upstream)
   │  TTIR → TNCA IR                                 (frontend, tiles, packed, wide, mathlib)
   │  layout plans; O2: upstream LinearLayout        (layout)
   │  O1+: TNCA-IR scalar passes                     (optimization)
   │  register allocation, ownership checks, asm     (codegen, asm, footprints, protocol)
   ▼
 tools/asm/tnca_asm.py ──► machine code ──► .tnca artifact (manifest + ABI + provenance)
   │
   ▼
 tnca.runtime.Session  ── per launch: upload argument storages, build argument block, LAUNCH, wait,
   │                       download, free                    (driver.py launcher)
   ▼
 backend: sim-file │ sim-socket │ fpga-u50-sim │ fpga-u50 │ fpga-u280
   ▼
 TNCA device (RTL simulator or FPGA)
```

## 3. Front ends

Three entry points produce TTIR. All of them call `tnca.aot._lower_ttir`, so they cannot drift apart:

| Entry point | Use |
|---|---|
| `tnca.aot.compile_aot(fn, signature=..., constexprs=...)` | Ahead-of-time compilation to a `.tnca` file; the `tnca-aot compile` CLI. Default `opt_level` 0. |
| Triton backend plugin `tnca.triton_backend` | `kernel[grid](...)` on CPU tensors, and PyTorch Inductor. Registered as the `triton.backends` entry point `cpu` (`pyproject.toml`): it is the backend Inductor's `cpu_backend = "triton"` uses, and Inductor recognises it by that name. Active when `TNCA_TRITON=1`, or when CUDA is unavailable. Default `TNCA_OPT_LEVEL` 2. |
| TNCA IR builders (`kernels.py`, `gemm.py`, `pipeline.py`, `partition.py`) | Build TNCA IR directly for the milestone tests C1–C4 (§11). Not reachable from Triton. |

**Toolchain pin:**
- Triton 3.8.0 (`spec/r1_toolchain.lock.json`), with the upstream commit and LLVM revision in `compiler/pin/toolchain.json`.
- PyTorch 2.14.0 (`spec/r38/torch.lock.json`).
- The plugin consumes TTIR as Triton's frontend produces it. The `ttir` stage is a pass-through, so no Triton GPU passes (TTGIR) run.

## 4. The lowering pipeline

| Step | O0 | O1 | O2 | Module |
|---|---|---|---|---|
| Upstream inliner (resolves `tl.*` library calls) | ✓ | ✓ | ✓ | `upstream/passes.py` |
| Upstream canonicalize + CSE on TTIR | | ✓ | ✓ | `upstream/passes.py` |
| TTIR → TNCA IR import | ✓ | ✓ | ✓ | `frontend.py` (+ `tiles`, `packed`, `wide`, `mathlib`) |
| Layout plans | ✓ | ✓ | ✓ | `layout.py` |
| Upstream `LinearLayout` composed into layout plans | | | ✓ (ablation: `no_ll`) | `layout.py`, `upstream/layout.py` |
| TNCA-IR passes: broadcast-compare folding, dead `other` removal, lane-0 store narrowing, strength reduction, pure-scalar CSE, dead pure-value removal | | ✓ | ✓ | `optimization.py` |
| Narrow constant materialisation | | ✓ | ✓ | `codegen.py` |
| Register allocation and assembly | ✓ | ✓ | ✓ | `codegen.py`, `asm.py` |

Details:

- **Upstream passes are mandatory where enabled.** If the pinned upstream bindings are missing at O1/O2, compilation fails; it never falls back silently to a weaker pipeline.
- **Every program that compiles at O2 compiles at O0.** O0 is legalisation only. The R3.7 suite checks this on all 149 cases.
- **No machine scheduling.** Instructions are emitted in program order. This is the largest known performance gap (§13).

## 5. Value model: from Triton blocks to registers

The importer maps Triton's value model onto a 16-lane machine (VLEN = 16, 32-bit lanes) explicitly. A construct it cannot map raises a diagnostic naming the operation; nothing is skipped.

### 5.1 Blocks and tiles

| Triton value | TNCA representation |
|---|---|
| 1-D block | One vector register. Blocks smaller than 16 are masked; blocks larger than 16 are rejected. |
| Small N-D tensor (≤ 16 elements) | Packed row-major into one register (`packed.py`); broadcasts and reductions permute lanes explicitly |
| 2-D block | A tile (`tiles.py`) in one of six forms, chosen by how the value is produced and used (see below) |
| `i1` values | Predicate registers. Converted to 0/1 (or −1 for signed conversions) with `vselect` when an integer is needed. |
| `i64` values | Exact, as lo/hi pairs of 32-bit lanes (`wide.py`) |

The six tile forms:

| Form | Used for |
|---|---|
| `window` | A tensor-transfer-engine fragment, written by `t.ld.async`; the operand form for `tl.dot` |
| `accumulator` | A computed 16×16 32-bit tile in the accumulator RF |
| `rows` | One register per row |
| `lazy` | A rule that recomputes a row when it is read; all `i1` tiles take this form |
| `spilled` | Rows kept in scratch memory: elements that are not 4 bytes, `i32` tiles, or when 8 or more accumulators are live |
| `col` | Column vectors |

Addressing:
- A 2-D load or store is recognised as affine (`row_stride·row + col_stride·col + c`). It uses the transfer engine when `col_stride == 1`, and the LSU row by row otherwise.
- A tile whose rows all alias (row stride 0, e.g. GEMM with M = 1) is one row load reused for every row.

### 5.2 Operations with non-trivial lowerings

- **`tl.dot`:**
  - FP16, BF16 and FP8 (E4M3/E5M2) inputs → `m.mma.async` into an FP32 accumulator, completed through an mbarrier.
  - FP32 × FP32 → exact vector FMAs. Precision is never narrowed.
  - INT8 → FP16 MMA, exact for K ≤ 1024.
  - Mixed input types are rejected.
  - The minimum dot shape reported to Triton is 16×16×16.
- **Reductions and scans:**
  - ISA §8.7 lane reductions.
  - Multi-operand folds (argmax/argmin) are unrolled across lanes; 2-D folds only along axis 1.
  - Scans are inclusive, along lanes, not reversed.
  - NaN-propagating `maximumf`/`minimumf` are built from a reduction plus a NaN test.
- **Transcendentals and division** (`mathlib.py`): SFU exp2/log2/reciprocal/rsqrt (≤ 2⁻²² relative error), with Newton refinement where a function's stated accuracy needs it.
- **64-bit integers** (`wide.py`):
  - Covers 1-D blocks and one-row 2-D blocks. Integer arithmetic is per lane in scalar registers; reductions and scans support add only.
  - A 64-bit offset reaching a pointer is narrowed to a 32-bit gather index with a check: unless every lane's high word equals the sign of its low word, the kernel traps (`TRAP_USER`, code 0xA5564) instead of wrapping.
- **`tl.device_assert`** is a real check. Any failing active lane raises `TRAP_USER` 0xA55E7, which the runtime reports as a kernel fault.
- **Transpose:**
  - By default, one gather per output row.
  - On profiles with `tensor_transpose` (`core-xpose-v1`), a descriptor load whose only use is `tt.trans` becomes one transposed `t.ld.async`. This is the R38 three-layer case: ISA bit, RTL and compiler selection together.

### 5.3 Tensor descriptors (rank 1–5)

`compiler/python/tnca_cc/descriptors.py` implements the contract Triton states for TMA descriptors on NVIDIA GPUs:
- rank 1–5, dimension 0 outermost, the last dimension contiguous;
- a box of `block_shape` elements at signed coordinates, in bounds when `0 <= coord + index < extent` in every dimension;
- out-of-bounds elements of a load are zero, or NaN under `padding_option="nan"`;
- stores, atomic reductions and scatters do not touch out-of-bounds elements.

| Source | What the compiler knows |
|---|---|
| `tl.make_tensor_descriptor` | Everything. The shape and strides may be run-time values: the TensorMap is written to scratch from registers. Padding comes from the op's `padding` attribute. |
| Host `TensorDescriptor` argument | Only the rank, block shape and element type. Base, extents, strides and padding are read from the host-built TensorMap at run time. Its geometry need not suit the transfer engine (the TNCA plugin accepts a superset of NVIDIA's alignment rules), so it is always read through the LSU. |

Each operation picks one of two machines and the result's representation:

| Box | Representation | `load` | `store` |
|---|---|---|---|
| one row (`block` is `[1, …, 1, n]`) | one register; for rank ≥ 2 also a one-row tile, so tile and packed lowerings both accept it | one masked contiguous load | one masked contiguous store |
| ≤ 16 elements, several rows | a packed N-D value (one register, row-major lanes) | one masked gather with per-lane element offsets | one masked scatter |
| > 16 elements | a tile of `prod(block[:-1])` rows × `block[-1]` | the transfer engine when the descriptor is this kernel's, the element type is 16/32-bit, the box fits 1 KiB, and every use can read a window (an MMA operand, a 32-bit tile one row per register, or another descriptor store); otherwise one masked load per row | `t.st.async` for a window tile of the box's shape; otherwise one masked store per row |

- **Atomics.** `atomic_add`/`min`/`max`/`and`/`or`/`xor` are one scalar atomic per in-bounds element, the way `tl.atomic_rmw` does it. FP32 add is a compare-and-swap loop; other floating-point atomics are refused.
- **Gather and scatter.** `gather` and `scatter` need a 2-D descriptor with a one-row block, and address one row per index.
- **`tl.trans` of an N-D value:**
  - On a packed value it is a compile-time lane permutation.
  - On a tile it relabels when the non-unit dimensions keep their order, and is the 2-D transpose when it swaps them.
  - The permutation is read from the printed TTIR, because the Python binding does not expose it, and is checked against the operand and result shapes.

## 6. Operation coverage and the error model

| Family | Handled |
|---|---|
| Control flow | `scf.for`, `scf.while`, `cf.br`, `cf.cond_br` |
| Triton | `get_program_id`, `get_num_programs`, `make_range`, `splat`, `addptr`, `load`, `store`, `reshape`, `join`, `bitcast`, `expand_dims`, `broadcast`, `trans`, `dot`, `reduce`, `scan`, `assert`, `atomic_rmw`, `make_tensor_descriptor`, `descriptor_load`, `descriptor_store`, `descriptor_reduce`, `descriptor_gather`, `descriptor_scatter`, `fp_to_fp` |
| `arith` | Integer add/sub/mul/div/rem/and/or/xor/shifts/min/max; float add/sub/mul/div/neg, `maximumf`/`minimumf`/`maxnumf`/`minnumf`; extensions, truncations, int↔float conversions; `cmpi`, `cmpf`, `select`, `constant` |
| `math` | `absf`, `exp`, `exp2`, `log`, `log2`, `sqrt`, `rsqrt` |
| Other | `llvm.intr.assume` (no-op), `ttg.barrier` (fence) |

Not supported today:
- **Raw FP16/BF16 arithmetic in a hand-written kernel — currently a silent wrong answer.** `x * 2` on an FP16 tensor that was loaded and never upcast is emitted as binary32 `vf.mul` on FP16 bit patterns. Inductor upcasts to FP32 before arithmetic (`codegen_upcast_to_fp32`), and so do the TileBench kernels, so no tested path reaches it. It was found while testing descriptors (2026-10-08). The fix needs one consistent register representation for half-precision values: freshly loaded values hold FP16 bits, but `_fconst` builds FP32 bits under an FP16 label (§15.6).
- **`scf.if`.** A Triton `if` on a runtime value that becomes `scf.if` is rejected. Inductor's kernels do not produce it, but hand-written kernels may.
- Control-flow edges that carry block arguments.
- Some loop-carried predicate or pointer tensors.
- Multi-row `i64` tiles.
- Atomics other than 4/8-byte integer and FP32 add.
- The libdevice functions `pow`, `log1p`, `expm1`, `erf`, `lgamma` and trigonometric functions.

**Error model:**
- An unsupported operation raises `UnsupportedError` (code `UNSUPPORTED`). The diagnostic names the TTIR function and operation, the reason, and a remedy.
- Other compilation failures raise `CompileError` (`COMPILE_FAILED`).
- Resource exhaustion raises `ResourceError`. The register allocator does not spill.
- A capability the profile lacks (no SFU, no transposed transfer, no atomics) is a compile-time diagnostic, not a run-time fault.
- Through Inductor, these errors surface as compile failures. ArchBench classifies them as `compile-failure`, `capacity-exceeded`, `runtime-fault`, `timeout` or `unsupported`, never as slow.

## 7. Code generation, artifact and ABI

- **Register allocation** (`codegen.py`, `asm.py`):
  - Shared resources (RF windows, accumulators, barriers, descriptors) are placed first. Windows are packed from the top of the vector RF down.
  - Role-local scalar, vector and predicate registers then use a linear scan over last uses, with loop-aware liveness.
  - `footprints.py` accounts for instructions that read register ranges implicitly.
  - `protocol.py` checks asynchronous ownership: a window or accumulator is not reused before its completion event.
- **Encoding:** mnemonics come from the ISA schema. `tools/asm/tnca_asm.py` emits 64-bit little-endian words; the `.li` pseudo-instruction expands to four words.
- **Container** (`python/tnca/artifact.py`, `spec/r2_container_v1.json`): `"TNCA-AOT" | major | minor | header_len | manifest | payload`.
- **Manifest:**

  | Section | Contents |
  |---|---|
  | `identity` | Versions, kernel name, source/compiler/toolchain hashes, constexprs, options |
  | `target` | Profile, configuration hash, the profile's capability set |
  | `parameters.signature` | The ABI description and permitted aliases |
  | `execution` | Role mask and program-to-role map |
  | `resources` | Registers, accumulators, barriers, descriptors and scratch bytes measured from the emitted code (`'unknown'` where not measured) |
  | `semantics`, `optional_metadata.provenance` | |
- **Argument ABI v1** (`spec/r2_abi_v1.json`, `python/tnca/abi.py`):
  - One 8-byte little-endian slot per non-constexpr parameter, in declared order. 32-bit values must have a zero high half.
  - Pointers are device addresses. A kernel reads slot *i* with `sload(arg_base, 8·i)`.
  - If the kernel needs scratch, a scratch pointer follows the slots.
  - TensorMaps (256 bytes each, packed by `tnca.tensor.TensorDescriptor`) follow on a 256-byte boundary, then scratch.

## 8. Runtime and launch path

### 8.1 Session

`tnca.runtime.Session(profile, backend, heap_base=0x20000, heap_limit=0x80000, deadline_cycles, host_timeout, counters, **backend_kwargs)` is the public runtime.

- **Calls:** `allocate`, `upload`, `download`, `free`, `load`, `submit(module, args, grid, depends_on)`, `wait`, `drain`, `recover`, `close`.
- **Event states:** SUBMITTED → VALIDATED → QUEUED → RUNNING → DRAINING → SUCCEEDED. On a fault the path goes through FAILING → DRAINING → FAILED. Events carry `device_cycles`, per-program cycles and, optionally, counter deltas.
- **Queue rules:**
  - At most 8 unfinished tasks.
  - Grids of up to 4096 programs, 16 bits per axis.
  - A buffer named by an unfinished task answers `BUFFER_BUSY`; it is not silently waited on.
  - Overlapping buffers that the manifest does not declare as aliases are refused (`ALIAS_UNDECLARED`).
- **Epochs:** a device timeout or loss poisons the session and fails queued tasks. `recover()` restarts the device, bumps the epoch, and invalidates every handle from the previous epoch. A freed or stale handle is refused at the API.
- **Grids** are launched program by program (`serial-grid`): device time is the sum of program cycles.

### 8.2 Backends

The registry (`python/tnca/backend.py`) maps names to implementations. It never falls back to another backend:

| Backend | Reaches |
|---|---|
| `sim-file`, `sim-socket` | The RTL simulator, built per profile and cached by `tnca.device.prepare` (`~/.cache/tnca/devices/`) |
| `fpga-u50-sim` | The FPGA top and the board register protocol, on RTL |
| `fpga-u50`, `fpga-u280` | The board over XRT; no simulator needed (`needs_elaborator = False`) |

Common to all backends:
- Kernels select a backend with `TNCA_BACKEND` and `TNCA_BACKEND_ARGS`, for example `{"xclbin": "…"}`.
- The transport protocol (UPLOAD, DOWNLOAD, LAUNCH, QUIESCE, SHUTDOWN, PING, PERF_READ) is described in [`architecture_overview.md`](architecture_overview.md) §9.1.
- The conformance suite `tools/r36/conformance.py` checks that every backend implements the protocol with the same observable behaviour.

### 8.3 Per-launch path and the heap allocator

The Triton launcher (`python/tnca/triton_backend/driver.py`) does the following for each kernel launch:

1. Binds parameters by name.
2. Allocates one device buffer per distinct CPU storage and uploads it.
3. Binds each tensor as (buffer, byte offset).
4. Converts Triton `TensorDescriptor`s into runtime descriptors.
5. Submits the grid and waits.
6. Downloads every storage back and frees all buffers.

There is one `Session` per (profile, backend, arguments), kept for the life of the process and closed at exit.

**The heap allocator** (`_Allocator` in `runtime.py`) is first-fit with 64-byte alignment; argument blocks use 256-byte alignment.
- **2026-10-08 fix:** freed blocks are now merged with their neighbours, and a block that ends at the untouched top of the heap is returned to it. Alignment gaps are recorded as free space.
- Before the fix, a long graph fragmented the heap. LLaMA small failed on the U280 with "out of device memory: 131072 bytes needs 131072 and 98304 remain", although every individual launch fits.
- After the fix, the heap returns to empty after every launch (tested over 2,000 launch-shaped allocation cycles). Tiny-tier board cycles are unchanged.

**Environment switches:**

| Variable | Default | Meaning |
|---|---|---|
| `TNCA_PROFILE` | `core-sim` | Compile and run profile |
| `TNCA_OPT_LEVEL` | 2 | Optimisation level (plugin) |
| `TNCA_BACKEND` | `sim-file` | Backend name |
| `TNCA_BACKEND_ARGS` | — | JSON keyword arguments for the backend |
| `TNCA_COUNTERS` | off | Bracket every program with counter reads |
| `TNCA_DEADLINE` | 5 × 10⁶ | Device cycles per launch before a TIMEOUT |
| `TNCA_HOST_TIMEOUT` | 600 s | Host-side wait budget |
| `TNCA_LAUNCH=skip` | — | Compile every kernel, launch none (reports would-be transfer sizes) |
| `TNCA_KERNEL_DUMP`, `TNCA_DUMP_DIR` | — | Dump every stage (TTIR, inlined, optimised, TNCA IR, assembly, hex), also for failing kernels |
| `TNCA_TENSOR_DESCRIPTORS` | off | `1`: `tnca.inductor.configure()` turns on Inductor's tensor-descriptor code generation (§9) |

## 9. PyTorch integration

`tnca.inductor.configure()` uses only Inductor's public extension points: device registration, heuristics, template configurations and decompositions. It changes no PyTorch source.

| Setting | Why |
|---|---|
| `cpu_backend = "triton"` | CPU tensors are compiled as Triton kernels, which the TNCA plugin serves |
| Pointwise and reduction heuristics cap every block dimension at 16 | The compiler's block limit (§14) |
| Reductions: one row per program (`XBLOCK = 1`), `persistent_reductions = False` | A persistent reduction needs a block spanning the row |
| Exactly one configuration per kernel | Inductor's autotuner would otherwise launch candidates, and ArchBench audits that the launch count equals the wrapper's |
| GEMM and conv through Triton templates, one 16×16×16 configuration; `autotune_fallback_to_aten = False`; `benchmark_epilogue_fusion = False` | Keeps every matmul and convolution on the device |
| SDPA decomposed (`_scaled_dot_product_flash_attention_for_cpu`) | Otherwise attention would stay on the host |
| `fx_graph_cache`, remote cache and AOTAutograd cache off | Their keys do not cover these registrations; a stale entry once served a host-fallback graph |
| `compile_threads = 1` | Worker processes would not see the registrations |

`tnca.inductor.host_fallbacks(code)` and `triton_kernels(code)` scan Inductor's generated wrapper. A point with any host fallback is recorded as `unsupported`.

**Status:** zero fallback on the 22 offloading graph tests of `tools/r38/torch_tests.py` (its 23rd graph checks that a real fallback is detected, and a separate case checks the device-assert trap: 24/24) and on all 52 tiny/small ArchBench points. The large tier (Llama-3.2-1B shapes) exceeds device memory and is not run on TNCA.

**Tensor descriptors from Inductor.** `configure(tensor_descriptors=True)` (or `TNCA_TENSOR_DESCRIPTORS=1`) turns on Inductor's `use_tensor_descriptor` code generation, the same code it produces for TMA on NVIDIA GPUs:
- Pointwise and reduction kernels load and store through `tl.make_tensor_descriptor` at ranks 1–5, with `tl.trans`/`tl.reshape` around permuted accesses.
- Inductor offers this for CPU tensors only when a Triton backend is registered under the name `cpu`, which is why the plugin's entry point has that name.
- Inductor's CPU branch skips the 16-byte rules that Triton's `make_tensor_descriptor` itself enforces, so `configure()` installs a checker through `TritonKernel.tma_compatibility_checker_cls`. It applies Inductor's CUDA rules instead, refuses descriptors whose innermost block is `XBLOCK` in a reduction (TNCA runs reductions one row per program), and keeps to the element types the compiler takes through a descriptor.
- With these rules, a graph gets descriptors on TNCA exactly where it would on an NVIDIA GPU.
- Off by default; the pointer-based code generation is unchanged.

**Multi-step generation:** `python -m archbench.generate` chains prefill and decode steps on one model. Each step compiles for its cache length. The device's K/V outputs feed the next step, and each step's logits are checked against an FP32 reference given the same inputs.

On the U280 (LLaMA small, 64-token prompt, 8 new tokens):
- All 8 steps are within tolerance, and the token sequence equals a pure-FP32 generation.
- Device time: 15.4 tokens/s prefill, 1.01 tokens/s decode.
- Host wall time: 327 s. Most of it is per-step compilation and per-launch transfers, which §15.2 removes.

## 10. One ISA source

There is one authoring source for opcodes and encodings, `tools/asm/tnca_isa.py`, and everything else is generated or checked against it (draft plan sec 11.3):

| Generator | Output |
|---|---|
| `tools/isa/export_schema.py` | `isa/tnca-isa-v0.1.json` |
| `tools/isa/gen.py` | `TNCAOpcodes.inc`, `TNCAInstrInfo.td`, `doc/isa/opcode-map.md` |
| `tools/isa/gen_target.py` | `compiler/targets/*.json`: per-profile target files from the schema, `spec/r1_profiles.json` and RTL parameters |
| `tools/isa/gen_support_matrix.py` | [`support_matrix.md`](support_matrix.md): what the ISA defines, what each profile implements, what the compiler emits (111 of 169 instructions) |
| `tools/isa/check_consistency.py` | Checks the hand-written RTL decode defines against the tables |

`make isa` regenerates everything. `make compiler` checks that nothing is stale.

## 11. The MLIR dialect and the C1–C4 builders

The original plan put asynchrony and role partitioning in an MLIR dialect. That half exists, but it is not on the product path:

- **MLIR dialect.**
  - `compiler/include/tnca/Dialect/TNCA/IR/TNCAOps.td` defines `rf.reserve`, `dma.load`, `mma.async`, `await`, `specialize`, channels, and producer/consumer acquire/publish/release.
  - `VerifyAsyncOwnership.cpp` implements `--tnca-verify-async`.
  - `compiler/tools/tnca-opt` is its driver, tested with lit/FileCheck (`compiler/test/IR`). CI builds it against LLVM 20.
- **Python IR builders** for the plan's milestones:
  - C2: GEMM with tensor DMA and MMA (`gemm.py`).
  - C3: an explicit two-role pipeline (`pipeline.py`).
  - C4: automatic context specialisation into producer/consumer roles (`partition.py`).
  - They run on the RTL (`make compiler-rtl`), including four-role elaboration on `core-4r`.

The Triton path uses neither. Every Triton or PyTorch kernel runs as one role context, with asynchrony confined to a single program (transfer, then mbarrier wait, then compute). Connecting the role-partitioning work to TTIR is a next step (§15.4).

## 12. Verification

| Suite | What it establishes |
|---|---|
| `make compiler` | ISA/target staleness, consistency, compiler unit tests, IR tests |
| `make triton-e2e`, `tools/r1/ir_tests.py` | The real Triton frontend through TNCA to the RTL |
| `make r37` (R3.7) | 45 TileBench operators, 149 cases × O0/O1/O2 = 447, numerics checked on the RTL with tails and guards intact. Inventory checked against `spec/r37_tilebench.json`. |
| `make r38-identity` | Every artifact byte-identical to a frozen baseline (`tools/r38/artifact_identity.py`). Used to show that compiler changes for PyTorch alter no existing kernel. |
| `make r38-torch` | Plugin/AOT identity at O0–O2, the 149 cases launched natively, 23 PyTorch graph tests, and the device-assert trap (24/24) |
| `make archbench-coverage` | Every tiny/small ArchBench point compiles with zero fallback (also run on x86_64) |
| `make r38-tdesc` (`tools/r38/tdesc_tests.py`) | Tensor descriptors through the real Triton front end and the plugin, on the RTL: ranks 1–5, run-time shapes and strides, host descriptors, negative and past-the-end offsets, zero/NaN padding, atomics, gather/scatter, a rank-3 MMA operand. Each case is checked bit-exactly against PyTorch computing the box from the definition, and records which path the compiler chose. |
| ArchBench runs | Numerical agreement with an FP32 reference on RTL and on the board, per point |
| R2 tests (`r2-abi/compile/cache/dump/runtime/fault-tests`) | ABI, container, compilation cache, dumps, runtime state machine, faults |

**CI coverage:** CI runs `make compiler`, the Triton frontend and R2 tests, and R3.7 at O0/O1/O2. It does not run `r38-torch`, `r38-identity` or ArchBench, which need PyTorch or long RTL runs (§15.6).

## 13. Performance today

Strict-bound efficiency η = L/T per point:

| Tier | η range | E (harmonic) | A (arithmetic) |
|---|---|---|---|
| Tiny | 0.0011 (conv2d) to 0.22 (fp32 add) | | |
| Tiny + small on the U280 | | 0.0072 | 0.021 |

Model points sit around 0.005–0.014. The board counters ([`architecture_overview.md`](architecture_overview.md) §10.2) attribute the time; on LLaMA decode:

- **Register hazards: 30 % of cycles.** Dependent instructions issue back to back because nothing reorders them.
- **LSU: 13 %.** Gathers, row-by-row tiles and 64-bit integer emulation.
- **VRF bank conflicts: 12 %.**
- **IPC: 0.26.**

The matrix unit is almost idle: 16-wide tiles and LSU-fed operands keep it below about 0.5 % busy. These numbers come from a compiler with no scheduling, no tiling beyond 16 and no data reuse across launches, so they are a baseline, not a ceiling.

## 14. Known limitations

1. **Block width ≤ 16 per dimension.** A tensor wider than 16 is not split across registers. Inductor is configured around this, which caps tile size, matrix-unit use and data reuse.
2. **No `scf.if`** (§6), and the other unsupported constructs listed there.
3. **No instruction scheduling,** and an allocator that cannot spill.
4. **Every launch round-trips through the host.** Arguments are uploaded and downloaded per launch; tensors do not stay on the device (PyTorch P2).
5. **One role context per Triton program.** The partitioning and pipelining work (C3/C4) is not connected to TTIR.
6. **`i64` is exact but slow** (scalar per-lane emulation), and limited to one-row tiles.
7. **Tile access patterns.** Transfer-engine tiles need a unit column stride; non-uniform `[R, 1]` broadcasts with R > 1 are not covered.
8. **Raw FP16/BF16 arithmetic** in hand-written kernels is miscompiled (§6). Upcast with `.to(tl.float32)`, as Inductor does.
9. **Descriptors.** A box row wider than 16 elements is refused (one row is one register). An N-D value with more than two non-unit dimensions is a row tile: elementwise operations, stores and reshapes that keep the last dimension work, but a `trans` that moves more than the last two dimensions does not. Descriptor atomics on FP16/BF16 are refused (the only floating atomic is FP32 add). `round_f32_to_tf32` is refused rather than ignored.
10. **Stale text:**
   - `tnca_cc/__init__.py` calls the plugin C++ (it is Python).
   - The `r38_pytorch_p1plus.md` diagram names `_compile_ttir` (it is `_lower_ttir`) and counts 12 graph tests (now 23, plus the trap check).
   - The `_op_tt_trans` docstring predates the transposed-transfer RTL.
   - `tools/r38/tdesc_tests.py` records the path each case took from the dumped assembly; the R3.7 cases do not.

## 15. Next steps

Ordered by what unblocks the most, with the evidence each produces.

### 15.1 Board-scale PyTorch (roadmap §12.4 P4–P5)

- Multi-step LLaMA generation runs on the U280 (§9). Next is a reproducible demo bundle: bitstream hash, commands, the JSON record, logs and board power.
- Cache compiled steps across cache lengths, through dynamic shapes or a fixed-capacity cache with a length mask, so that generation does not recompile per token. Today compilation dominates host time.

### 15.2 Device-resident tensors (P2)

A PyTorch `PrivateUse1` device whose tensors live in device memory:
- Kernels then bind device buffers instead of uploading storages, so the per-launch round trip disappears.
- Host time becomes comparable with device time.
- The KV cache stays on the device across decode steps.

This depends on device memory capacity beyond 512 KiB for anything larger than the small tier ([`architecture_overview.md`](architecture_overview.md) §13.2).

### 15.3 Wider tiles

Split blocks larger than 16 across several registers or windows, so Inductor can use 32/64-wide blocks and GEMM tiles that keep the matrix unit busy. This is the prerequisite for any matrix-unit-bound result. It also changes the Inductor configuration: more than one block size, and possibly autotuning with launches excluded from measurement.

### 15.4 Asynchrony on the Triton path

Connect context specialisation (`partition.py`, C4) and the two-role pipeline (C3) to TTIR. Transfer-engine loads for the next tile would then overlap compute on the current one. Start with GEMM and attention (the draft's C5). The MLIR async-ownership verifier (`tnca-opt --tnca-verify-async`) can become the checker for that path.

### 15.5 Instruction scheduling and allocation

These are the cheapest performance work:
- **List scheduling** within basic blocks, aimed at the 30 % hazard stalls.
- **Bank-aware register allocation** (`bankcost.py` already models VRF bank conflicts).
- **Spilling,** so resource exhaustion is not a compile failure.

Each is an R38 cross-layer case, attributed by the profiler and predicted by the cost model before it is measured.

### 15.6 Coverage and engineering

- Coverage:
  - Half-precision arithmetic (§6): one register representation for FP16/BF16 values, computed in binary32 (exact for + − × ÷, since binary32 has 2p+2 bits for both). Highest priority, because the current failure is silent.
  - `scf.if`.
  - Multi-row `i64`.
  - The missing libdevice functions (`pow`, `erf`, trigonometric).
- CI:
  - A job for `r38-torch`, `r38-identity` and `archbench-coverage`. They need a PyTorch environment, not RTL time.
  - A scheduled job for the tiny tier on RTL.
- Specs:
  - Bump the transport spec to 1.1 (op 7 `PERF_READ`).
  - Fix the stale text in §14.10.

## 16. Plan versus delivery

The September plan ([`design/tnca_v1_compiler_draft.md`](design/tnca_v1_compiler_draft.md) §15) defined six milestones.

| Milestone | Plan | State |
|---|---|---|
| C0 | Pinned toolchain and target contract | Done: toolchain lock, generated target files, one ISA source |
| C1 | First real Triton → RTL closure | Done, then extended to all 45 TileBench operators (R3.7) and PyTorch (R38) |
| C2 | Single-RCTX tensor DMA + MMA | Done; used by `tl.dot` on the Triton path |
| C3 | Explicit two-role pipeline | Done in the IR builders and on RTL; not on the Triton path |
| C4 | Automatic partitioning | Done in the IR builders, including four-role elaboration; not on the Triton path |
| C5 | Attention overlap and performance feedback | Not started as planned. Performance feedback exists through the R38 profiler, cost model and ArchBench. |

What the plan did not foresee, and the project added:
- **PyTorch as the front door** (Inductor rather than hand-written ATen mappings, decided 2026-10-07).
- **The board as a backend** behind the same runtime.
- **ArchBench** as the measurement contract against GPUs.
