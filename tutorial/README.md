# Using the TNCA compiler and runtime from Python

A working tour of the toolchain: compile a kernel, run it on the
simulated hardware, check the answer, compare optimisation levels, read
what the compiler did, and handle the things that go wrong.

Every number and every block of output in these pages was produced by
running the script beside it. Nothing is illustrative.

## Before you start

```sh
# the locked Triton, which the frontend needs
python3 -m venv ~/tritonenv
~/tritonenv/bin/pip install "triton==$(python3 -c "import json;print(json.load(open('spec/r1_toolchain.lock.json'))['triton']['version'])")"

# the simulated device, built once
make device
```

Run any example with the repository on the path:

```sh
PYTHONPATH=python:compiler/python ~/tritonenv/bin/python tutorial/examples/01_hello_vector_add.py
```

Installed as a package (`make r2-package` and `pip install` the wheel),
you do not need `PYTHONPATH` — that line is for working in a checkout.

[10](10_torch_compile.md) needs PyTorch at its locked version in a separate
environment; [11](11_running_on_the_u280.md) runs anywhere for its simulated
half and needs the host with the card for the rest. Each page says how.

*Re-run on 2026-10-09 against the RTL of that day (r38-dco): word counts,
cache keys and the GEMM's cycles changed since the pages were first written,
and the pages now show what the scripts print today.*

## The pages

| | what it covers | example |
|---|---|---|
| [01](01_first_kernel.md) | compile a Triton kernel, run it, check it | `01_hello_vector_add.py` |
| [02](02_optimisation_levels.md) | B0/O0/O1/O2, and why fewer instructions is not fewer cycles | `02_opt_levels.py` |
| [03](03_layout_views.md) | where O2's layout reasoning earns its keep, measured | `03_layout_views.py` |
| [04](04_reading_the_compile.md) | the IR at each stage, the assembly, what each pass did | `04_inspect_the_compile.py` |
| [05](05_matrix_multiply.md) | a 16×16×16 FP16 GEMM on the matrix unit, through the planner | `05_matrix_multiply.py` |
| [06](06_errors_and_faults.md) | what is refused, what faults, and what you are told | `06_errors_and_faults.py` |
| [07](07_backends_and_cache.md) | profiles vs backends, and what is in the cache key | `07_backends_and_cache.py` |
| [08](08_triton_gemm.md) | tiled `tl.dot`, runtime K loops, masked tails, and O0/O1/O2 | `08_triton_gemm.py` |
| [09](09_tensor_descriptors.md) | host descriptors, bounds and descriptor GEMM through AOT; rank 1-5, padding, stores | `09_tensor_descriptors.py` |
| [10](10_torch_compile.md) | PyTorch: `torch.compile` -> Inductor -> TNCA, and how a host fallback is reported | `10_torch_compile.py` |
| [11](11_running_on_the_u280.md) | one artifact on the simulator, the board's twin and the U280, cycle for cycle | `11_on_the_board.py` |

## The three ideas worth having before you read on

**A profile is a machine, not a program.** `core-vrf-sync-v1` is the
single core with the banked, finite-port register file. It decides what
the compiler may emit, and it is part of an artifact's identity, so
compiling the same source for a different profile gives you a different
artifact with a different key.

**A backend is how this host reaches a device.** It is *not* part of the
artifact. The same compiled kernel runs on any backend, which is why
changing the connection medium never requires a recompile.

**Correctness is checked against something that is not the device.**
Every example computes its expected answer in Python and compares. Two
implementations agreeing with each other is not evidence; one of them
agreeing with an oracle is.
