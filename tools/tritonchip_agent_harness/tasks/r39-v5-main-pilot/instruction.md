# Task: improve TNCA from the R39 v5 starting point

The repository in your working directory is TritonChip: the TNCA accelerator's
ISA, Verilog RTL, and the compiler that takes Triton (and PyTorch, through
Inductor) to TNCA code.

Start from TNCA's already optimized `core-vrf-sync-v5` design. Improve the
device cycles of matrix-multiply-like PyTorch workloads without changing any
numerical result. The whole
chain stays: PyTorch → `torch.compile` (Inductor) → Triton → the TNCA compiler
→ ISA → RTL. You may change the compiler, the ISA and the hardware: the RTL,
and the profiles in `spec/r1_profiles.json` that the simulated device is built
from. A method has to be general; one that only helps particular kernels is not
a result. The starting compiler includes the existing zero-accumulator
lowering, and the starting hardware includes the R39 v5 matrix buffer and
context-switch policy. Keep an improvement only when it works on your own
independent tests as well as the measurements available during this run.

## What you are measured on

A benchmark you are not shown. You will not find it in your copy of the
repository, and you are not told its programs. What you are told is this:

- **The workloads** are matrix-multiply-like PyTorch programs, at shapes small
  enough for the RTL simulator, in the floating-point types TNCA's compiler
  supports.
- **How a program runs:** it is given to `torch.compile` with TNCA as the
  backend (`TNCA_OPT_LEVEL=2`), on the `core-vrf-sync-v5` profile as your changes leave
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

## The machine

The device is an FPGA: an AMD Alveo U280 with its kernel clock fixed at
100 MHz, so device cycles are latency. A submission that changes the hardware
(an RTL file, or the profile's parameters) is built for that card at 100 MHz
and its programs are run on it. If the design does not fit the card, does not
close timing at 100 MHz, or any program counts other cycles on the card than
in simulation, every program fails. The card's top level passes the profile's
parameters to the core by name: a profile parameter it does not pass cannot be
built, and a new switch of your own reaches the card through its default in
the RTL.

## Measuring your own work

The simulator in your environment is cycle-accurate. Write your own test
cases. Once at the start, and again after changing a profile or a compiler
target, stage the package data and build the simulated device:

    make -s python/tnca/data/.staged
    tnca-sim prepare --profile core-vrf-sync-v5

Then a test case of your own looks like this (the model and inputs are yours
to choose):

```python
import os
os.environ.setdefault("TNCA_PROFILE", "core-vrf-sync-v5")
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
