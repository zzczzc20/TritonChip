<p align="center">
  <img src="doc/design/triton_chip.png" alt="TritonChip · TNCA" width="300">
</p>

# TritonChip

TritonChip is a hardware/software co-design of **TNCA (Tensor-Native Compute
Array)**, an accelerator that runs standard PyTorch through the upstream
compiler stack: `torch.compile` → Inductor → Triton → a TNCA backend → the
device. The repository holds the RTL, the ISA, the Triton backend and
compiler, the runtime and native PyTorch device, the FPGA flow, and ArchBench,
the PyTorch-level benchmark used to measure it against GPUs.

```text
PyTorch model ─► torch.compile ─► Inductor ─► Triton ─► TNCA backend ─► .tnca artifact ─► Session ─► RTL simulator │ Alveo U280
```

## Status

| | |
|---|---|
| Hardware | Three physical TNCA cores, one per SLR of an Alveo U280, with a shared NUCA, three HBM outlets and a 4 GiB HBM window; 90 MHz |
| PyTorch | `tnca` is a native PyTorch device (real device storage, views, copies, streams/events); graphs compile with `fullgraph=True` and zero host fallback |
| Workloads | ArchBench: 77 tiny/small/medium points, 24 large points at Llama-3.2-1B shapes, 14 HBM-tier points, and a 4-layer decoder (prefill 512, 64 decode steps) |
| Last signed-off milestone | R39: the standard Torch path at **5.5×** S_geo over its frozen baseline on the board, 77 points ([report](doc/r39_report.md)) |
| Closed milestone | R40: native device, residency, HBM, multicore and NUCA; closed with failures shown under the [v3 contract](doc/r40_tasks_and_acceptance_v3.md) ([report](doc/r40_report.md)) |
| In progress | R41: 100 MHz timing closure, dual memory paths, and compiler-hardware co-iteration ([tasks](doc/r41_execution_tasks.md), [draft contract](doc/r41_tasks_and_acceptance.md)) |

This is a research prototype: an FPGA implementation and RTL, without silicon
or ASIC physical-design sign-off. Every reported number comes from the board or
from RTL simulation, with its evidence under [`evidence/`](evidence/).

## Architecture

- **Data hierarchy:** register files ↔ shared NUCA cache ↔ DRAM/HBM. There is
  no software-addressable scratchpad; a tensor transfer engine moves tiles
  described by TensorMap descriptors into register windows.
- **Core:** 16-lane SIMD, a matrix unit fed from register fragments, separate
  vector and accumulator register files, and up to four Role Contexts that
  form asynchronous producer/consumer pipelines under explicit ownership and
  completion rules.
- **Memory:** static-home NUCA slices with miss tracking, request merging,
  dirty writeback and out-of-order responses; a Tensor Management Unit for
  DCO cache orchestration (replacement, dead-block and bypass hints).
- **System:** a global grid dispatcher spreads Triton programs over the
  physical cores; host DMA hands buffers to and from the cache with an
  ownership protocol.

Details: [architecture overview](doc/architecture_overview.md),
[compiler overview](doc/compiler_overview.md), [ISA v0.1](doc/isa/TNCA-ISA-v0.1.md),
[support matrix](doc/support_matrix.md).

## Getting started

Run from the repository root. Requirements: Linux, GNU Make, Icarus Verilog 12,
Python 3.12 for the verification environment, and torch 2.14 with Triton 3.8
for the PyTorch path.
Exact versions are locked in [`spec/r1_toolchain.lock.json`](spec/r1_toolchain.lock.json)
and [`spec/r40/api_lock.json`](spec/r40/api_lock.json).

```sh
make device     # build the simulated device once
make compiler   # compiler tests and generated ISA/target checks
make core       # single-core RTL tests
make test       # development regression: compiler, core, cache, mesh, system
```

A PyTorch model on the device (the RTL simulator by default; `TNCA_BACKEND`
selects the board):

```python
import torch
import triton                     # discovers the tnca Triton backend
import tnca.inductor, tnca.torch_device

tnca.inductor.configure()         # Inductor -> Triton -> TNCA for the tnca device
tnca.torch_device.register()      # `tnca` as a PyTorch device

model = torch.nn.Sequential(torch.nn.Linear(64, 64), torch.nn.SiLU()).half().to("tnca")
x = torch.randn(16, 64, dtype=torch.float16).to("tnca")
y = torch.compile(model, fullgraph=True)(x)
```

ArchBench runs any benchmark point on TNCA or a GPU under one timing protocol:

```sh
python -m archbench.run --platform tnca --point op.matmul/tiny/fp16-m16-k32-n16 --out tnca.jsonl
python -m archbench.run --platform gpu  --point op.matmul/tiny/fp16-m16-k32-n16 --out gpu.jsonl
```

The [tutorial](tutorial/README.md) walks through compiling kernels, reading
the compiler's output, `torch.compile`, and running on the U280; every number
in it is produced by the script beside it.

## Repository layout

| Directory | Contents |
|---|---|
| `rtl/`, `tb/` | Core, NUCA, NoC, multicore shell and board RTL; testbenches and assembly programs |
| `isa/`, `spec/` | ISA schema, profiles, toolchain and API locks, frozen benchmark suites and contracts |
| `compiler/` | Triton frontend, TNCA IR, register allocation, scheduling and code generation; MLIR dialect |
| `python/tnca/` | Artifact/ABI, runtime and Session, backends, the PyTorch device and Inductor integration |
| `archbench/` | PyTorch-level benchmark: operators, models, tiers, protocol, GPU and TNCA platforms |
| `flow/`, `fpga/` | Synthesis, Vivado/Vitis flows and U280 platform configuration |
| `tools/` | Assembler, numerical models, generators, acceptance and evidence tooling |
| `xls/`, `rtl/gen/` | DSLX arithmetic sources and generated Verilog |
| `doc/`, `evidence/` | Design documents, milestone contracts and reports; recorded evidence |
| `tutorial/` | Worked examples from Python |

## Milestones

| Milestone | Delivered |
|---|---|
| R1–R3.6 | Reproducible RTL/compiler gates, AOT compiler and ABI, banked VRF, LinearLayout reuse, shared runtime |
| R37 | All 45 original TileBench Triton operators through AOT at O0/O1/O2 |
| R38 | ArchBench; PyTorch through Inductor; rank-5 tensor descriptors; DCO; U50 and U280 bring-up |
| R39 | Standard-path performance on the board: scheduling, code shape, four Role Contexts, v5 core |
| R40 | Native PyTorch device, residency, HBM, three physical cores, NUCA with three outlets; closed with failures shown ([report](doc/r40_report.md)) |
| R41 | 100 MHz timing closure, dual memory paths, and compiler-hardware co-iteration (in progress) |

Each milestone is specified by a tasks-and-acceptance document in [`doc/`](doc/);
a specification defines acceptance criteria, not completed features.

## Contributing

Keep ISA, profile, compiler and RTL contracts consistent. Add a regression test
for every behaviour change and a negative test for every rejected input. Report
missing coverage explicitly, and update the support matrix and status documents
when capabilities change.

## License

[Apache-2.0](LICENSE). See [NOTICE](NOTICE) for attribution.
