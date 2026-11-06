# Task: improve R41's three-core TNCA machine

The repository in your working directory is TritonChip: the TNCA accelerator's
ISA, Verilog RTL, and the compiler that takes Triton (and PyTorch, through
Inductor) to TNCA code.

Start from R41's three-core machine: three `core-vrf-sync-v5-x8-g2c4-dp-board`
cores, one per SLR of the FPGA, each with a home slice of a shared NUCA cache
and a line-wide transfer engine, three 512-bit HBM outlets striped across the
memory, at a 100 MHz kernel clock. Its matrix unit executes the grouped
dot-product opcode `m.mma.dp.async` (ISA sec 9C.4): four E4M3 products per lane
and cycle, two for the other formats. Improve the device cycles of small
PyTorch workloads on it without changing any numerical result. The whole chain
stays: PyTorch → `torch.compile` (Inductor) → Triton → the TNCA compiler → ISA
→ RTL. You may change the compiler, the ISA and the hardware: the RTL, the
profiles in `spec/r1_profiles.json` the cores are built from, and the machine's
uncore in its description, `spec/devices/u280-3core-r41.json`. A method has to
be general; one that only helps particular kernels is not a result. Keep an
improvement only when it works on your own independent tests as well as the
measurements available during this run.

## The machine, as a description

`spec/devices/u280-3core-r41.json` is the whole machine, and everything that
builds or models it is generated from it: the FPGA image's parameters and the
simulation you test with.

- `per_core_profile` names the core, a profile of `spec/r1_profiles.json`. The
  name stays `core-vrf-sync-v5-x8-g2c4-dp-board`; what that profile says may
  change. It is the author's `-dp` profile as the board image built it: one
  word-path load/store unit (`LSU_INFLIGHT` 1, chosen for the SLR budget), the
  rebuilt FMA and the registered matrix operand capture.
- `platform` is fixed by the card and the flow: three cores, 512-bit HBM
  outlets, 8 GiB of HBM, the 100 MHz clock, the runtime's memory controls and
  cache maintenance. A description that changes any of it, or its name, is not
  measured.
- `uncore` is open: how many memory outlets the cores use (`MEM_PORTS`), the
  outstanding requests per outlet (`AXI_OUT_N`), the NUCA's sets and ways, the
  register stages on the links, the dispatch path and the home path, the NUCA
  and DCO switches, and R41's one-lane memory path: the striped HBM layout,
  the pipelined LUTRAM-backed outlets, the hybrid load/store unit, the
  line-wide tensor engine, the line slots and the home's MSHRs and fill pipe.
  `python/tnca/devices.py` lists every field's legal values and derives the
  rest (the set and way index widths, the AXI ID width, the physical register
  file). One thing to know about this machine: with the hybrid load/store
  unit and the chained home fill pipe (`HYBRID_LSU` 1, `HOME_LINE_PIPE` 2)
  the tile builds the pipelined home (`rtl/nuca/tnca_home_pipe.v`), which
  takes no `NUCA_SHARED_DATA_READ`, `NUCA_CACHE_DATA_PIPELINE`,
  `NUCA_REGISTER_VICTIM_RANK`, `NUCA_REGISTER_VICTIM_SELECT`,
  `HOME_BACKGROUND_READS` or `DCO_REGISTERED_RANGES`: those six fields change
  nothing unless the home is switched back to the older slice.

`python -m tnca.devices params u280-3core-r41` prints every parameter your
description builds the machine with; a description it refuses cannot be built
or measured. The image's top level (`rtl/board/`) and the FPGA control block
(`rtl/fpga/`) are the platform's: changes there are not taken. A switch of
your own reaches the hardware through its default in the RTL, or as a field the
top level already passes.

## What you are measured on

A benchmark you are not shown. You will not find it in your copy of the
repository, and you are not told its programs. What you are told is this:

- **The workloads** are small PyTorch programs of the kinds language and
  vision models are made of: element-wise operations, reductions and
  normalisations, data movement, and matrix multiplication, at shapes small
  enough for the RTL simulator, in the floating-point types TNCA's compiler
  supports, the OCP 8-bit formats included (`torch._scaled_mm` and the
  linear layers quantisation code writes with it).
- **How a program runs:** it is given to `torch.compile` with TNCA as the
  backend (`TNCA_OPT_LEVEL=2`), on the `core-vrf-sync-v5-x8-g2c4-dp-board`
  profile as your changes leave it, on the machine your description builds
  with all three cores active, compiled once and run once.
- **Device time** is the sum, over every launch the program makes, of the
  cycles from the launch's start to its end as the device's control block
  counts them. A launch that reports completion while its memory traffic has
  not drained fails.
- **Each program is judged on its own.** It fails if its numbers differ from
  PyTorch's on the CPU beyond the usual tolerance for its type, if any of it
  falls back to the host, or if it produces no result. The other programs still
  count.
- **The score** compares each program's device time with the base machine's:
  how many programs passed, the geometric mean of the speedups over the
  programs that passed, and the share of all programs that passed and got
  faster.

## The card

The device is an FPGA: an AMD Alveo U280 with its kernel clock fixed at
100 MHz, so device cycles are latency. A submission that changes the hardware
(an RTL file, a profile's parameters, or the description's uncore) is built for
that card at 100 MHz and its programs are run on it. If the design does not fit
the card, does not close timing at 100 MHz, or any program counts other cycles
on the card than in simulation beyond the tolerance the base machine shows,
every program fails. The three cores sit in three SLRs: a path that crosses
between them is the hardest to close, and the base machine already fills most
of SLR0; the second load/store unit it gave up costs about 36k LUTs.

## Measuring your own work

**The simulation is cycle-accurate**, and it is the image's own top level,
built from your description. Stage the package data once at the start, and
again after changing a profile or a compiler target:

    make -s python/tnca/data/.staged

Then run a test case of your own with the three-core device:

    export TNCA_BACKEND=fpga-multicore-sim TNCA_ACTIVE_CORES=3
    export TNCA_BACKEND_ARGS='{"board_top": true, "device": "u280-3core-r41", "memory_bytes": 16777216, "hbm_dma": true}'

<!-- without-skill -->
```python
import os
os.environ.setdefault("TNCA_PROFILE", "core-vrf-sync-v5-x8-g2c4-dp-board")
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
<!-- /without-skill -->

The simulator elaborates the whole three-core machine with its caches and
memory outlets, so it is slow: a few tens of device cycles per second of wall
time, minutes for a program of a few thousand cycles. Each test builds its own
simulator from the description as it is when the test starts.
`python/tnca/triton_backend/driver.py` documents what `launch_log()` records for
each launch (its `device_cycles` is the count the device reports, which the
evaluator checks against its own), and `python/tnca/inductor.py` how Inductor
is pointed at TNCA.
