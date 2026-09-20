# TritonChip

TritonChip develops **TNCA (Tensor-Native Compute Array)**, an experimental AI
accelerator and compiler targeting a subset of Triton kernels. The repository
includes Verilog RTL, a custom ISA, compiler infrastructure, numerical reference
models, and simulation-based verification.

The data hierarchy is **register files ↔ shared NUCA cache ↔ DRAM**, with no
software-addressable scratchpad. Tensor transfers load register windows;
asynchronous matrix operations consume them under explicit ownership and
completion rules.

## Architecture

- **Compute:** 16-lane SIMD, a matrix unit, separate vector and accumulator
  register files, and up to four Role Contexts sharing execution resources.
- **Memory:** a static-home, non-uniform cache architecture (NUCA), with miss
  tracking, request merging, dirty writeback, and out-of-order responses.
- **Interconnect:** a 2×2 mesh with XY routing and separate request/response
  virtual networks.
- **Execution:** TensorMap descriptors, register windows, and barriers coordinate
  data movement, computation, and resource reuse.

The current simulation profiles are defined in
[`spec/r1_profiles.json`](spec/r1_profiles.json):

| Profile | Configuration | SFU |
|---|---|---|
| `core-sim` | One core, two Role Contexts | Enabled |
| `core-4r` | One core, four Role Contexts | Enabled |
| `system-2x2` | Four tiles, two Role Contexts per core | Disabled |

## Current scope

RTL tests cover arithmetic, conversions, tensor transfers, matrix operations,
barriers, atomics, and the cache/network path. Compiler tests include masked
vector operations, reductions, gather, and GEMM with explicit or specialized
multi-role pipelines.

Selected Triton kernels run through the real frontend to checked results on
single-core and four-tile RTL. The GEMM pipeline uses the TNCA IR path; this is
not yet general Triton GEMM support. The MLIR dialect and resource verifier
exist, while the unified AOT compiler and persistent simulation runtime remain
under development.

The [support matrix](doc/support_matrix.md) distinguishes ISA definitions,
profile capabilities, compiler lowering, and whole-system test coverage.
R1 has a signed-off historical baseline; R2 remains incomplete. See the
[R1 status](doc/r1_status.md) and [R2 status](doc/r2_status.md) reports.
This is an RTL/compiler prototype, without silicon or physical-design signoff.

## Getting started

Run commands from the repository root. The locked Linux verification environment
uses Python 3.12 and Icarus Verilog 12; GNU Make and Bash are also required.
See the [toolchain lock](spec/r1_toolchain.lock.json) for exact dependencies.
MLIR builds additionally require LLVM/MLIR, CMake, Ninja, and a C++ compiler;
Triton tests require a compatible Triton environment.

```sh
make lint       # Elaborate RTL modules
make core       # Run single-core assembly tests
make compiler   # Check compiler tests and generated ISA/target files
```

For the Triton-to-RTL tests, point to the Python interpreter in your Triton
installation:

```sh
make triton-e2e e2e \
  TRITON_PYTHON=/path/to/triton-env/bin/python R1_STRICT=1
```

See the [compiler guide](compiler/README.md) for toolchain setup. Generated
arithmetic RTL is checked in; XLS is needed only to regenerate it.

| Command | Purpose |
|---|---|
| `make test` | Development regression across compiler, core, cache, mesh, and system |
| `make compiler-rtl` | Execute compiler-generated programs on RTL |
| `make core-slowmem core-ooo` | Exercise latency, backpressure, and response reordering |
| `make r1-check-generated` | Check generated artifacts without regenerating them |
| `make r1-gate-selftest` | Verify that injected failures are detected |
| `make r1-acceptance` | Run the strict acceptance gate and collect evidence |
| `make r1-verify-evidence EVIDENCE=<dir>` | Independently validate an acceptance run |

Development tests may skip unavailable dependencies. A successful development
run is not release acceptance; the strict gate requires complete evidence.

**RTL simulation is slow.** Use focused tests during development and small
kernels for performance measurements. R3 v2 limits new single-core GEMM workloads
to 16×16×32 and focuses on finite-port VRF integration. Simulation wall time
is not hardware execution time.

## Development milestones

| Milestone | Deliverable | Specification |
|---|---|---|
| R1 | Reproducible builds, reliable test gates, consistent profiles, and async resource safety | [Tasks and acceptance](doc/phase1_tasks_and_acceptance.md) |
| R2 | Installable AOT compiler, versioned artifacts and ABI, simulation runtime, and grid scheduling | [Tasks and acceptance](doc/phase2_tasks_and_acceptance.md) |
| R3 v2 | Single-core banked VRF integration, writeback flow control, and small RTL validation workloads | [Tasks and acceptance](doc/phase3_tasks_and_acceptance.md) |
| R3.5 | LinearLayout and MLIR optimization reuse, with measured compiler gains on accepted R3 hardware | [Tasks and acceptance](doc/phase3_5_tasks_and_acceptance.md) |
| R3.6 | Shared compiler/runtime/RTL integration, backend interfaces, and handoff for separate FPGA/ASIC implementation branches | [Tasks and acceptance](doc/phase3_6_tasks_and_acceptance.md) |

These specifications define acceptance criteria, not completed features.

## Repository layout

| Directory | Contents |
|---|---|
| `rtl/`, `tb/` | Hardware modules, testbenches, and assembly programs |
| `compiler/` | Triton frontend adapter, TNCA IR/code generation, and MLIR dialect |
| `python/tnca/` | AOT artifact and ABI infrastructure under development |
| `isa/`, `spec/` | ISA schema, profiles, toolchain locks, and acceptance contracts |
| `tools/` | Assembly tools, numerical models, generators, and verification runners |
| `xls/`, `rtl/gen/` | DSLX arithmetic sources and generated Verilog |
| `flow/` | Synthesis tooling |
| `doc/` | Design documents, specifications, audits, and implementation status |

## Documentation

- [Architecture design](doc/architecture_overview.md) and [compiler design](doc/compiler_overview.md)
- [ISA v0.1](doc/isa/TNCA-ISA-v0.1.md) and [support matrix](doc/support_matrix.md)
- [Production-readiness audit](doc/production_readiness_audit_2026-09-19.md)
- [Synthesis results](doc/synthesis.md) — structural proxies, not signed-off PPA
- [XLS generation workflow](doc/xls.md)

## Contributing

Keep ISA, profile, compiler, and RTL contracts consistent. Add regression tests
for behavior changes and negative tests for rejected inputs. Run the relevant
checks and report missing coverage explicitly; update the support matrix and
status documents when capabilities change.

## License

[Apache-2.0](LICENSE). See [NOTICE](NOTICE) for attribution.
