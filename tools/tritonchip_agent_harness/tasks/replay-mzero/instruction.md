# Task: make matrix-multiply-like PyTorch workloads faster on TNCA

The repository in your working directory is TritonChip: the TNCA accelerator's
ISA, Verilog RTL, and the compiler that takes Triton (and PyTorch, through
Inductor) to TNCA code.

Reduce the device cycles that matrix-multiply-like PyTorch workloads take on
TNCA's `core-sim` profile, without changing any numerical result. The whole
chain stays: PyTorch → `torch.compile` (Inductor) → Triton → the TNCA compiler
→ ISA → RTL. You may change the compiler, the ISA and the hardware: the RTL,
and the profiles in `spec/r1_profiles.json` that the simulated device is built
from. A method has to be general; one that only helps particular kernels is not
a result.

## What you are measured on

A benchmark you are not shown. You will not find it in your copy of the
repository, and you are not told its programs. What you are told is this:

- **The workloads** are matrix-multiply-like PyTorch programs, at shapes small
  enough for the RTL simulator, in the floating-point types TNCA's compiler
  supports.
- **How a program runs:** it is given to `torch.compile` with TNCA as the
  backend (`TNCA_OPT_LEVEL=2`), on the `core-sim` profile as your changes leave
  it, compiled once and run once.
- **Device time** is the sum of the device cycles of every launch the program
  makes.
- **Each program is judged on its own.** It fails if its numbers differ from
  PyTorch's on the CPU beyond the usual tolerance for its type, if any of it
  falls back to the host, or if it produces no result. The other programs still
  count.
- **The score** compares each program's device time with the base design's:
  how many programs passed, the geometric mean of the speedups over the
  programs that passed, and the share of all programs that passed and got
  faster.
- **Hardware changes** are allowed and recorded, and are reported apart from
  software-only changes.

## Measuring your own work

The simulator in your environment is cycle-accurate. Write your own test
cases. Once at the start, and again after changing a profile or a compiler
target, stage the package data and build the simulated device:

    make -s python/tnca/data/.staged
    tnca-sim prepare --profile core-sim

Then a test case of your own looks like this (the model and inputs are yours
to choose):

```python
import os
os.environ.setdefault("TNCA_PROFILE", "core-sim")
os.environ.setdefault("TNCA_OPT_LEVEL", "2")
import torch
import triton                      # first: importing it discovers the TNCA backend
import tnca.inductor
from tnca.triton_backend.driver import launch_log
from torch._inductor.utils import run_and_get_code

tnca.inductor.configure()
model = ...                        # a torch.nn.Module
inputs = (...)                     # its inputs
with torch.no_grad():
    expected = model(*inputs)      # eager, on the CPU
    with launch_log() as log:
        got, code = run_and_get_code(torch.compile(model), *inputs)
print("host fallbacks:", tnca.inductor.host_fallbacks(code))   # must be empty
print("device cycles:", sum(e["device_cycles"] or 0 for e in log))
torch.testing.assert_close(got, expected)
```

`python/tnca/triton_backend/driver.py` documents what `launch_log()` records
for each launch, and `python/tnca/inductor.py` how Inductor is pointed at
TNCA.
