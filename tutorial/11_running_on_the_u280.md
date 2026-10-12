# 11 — Running on the U280

[`examples/11_on_the_board.py`](examples/11_on_the_board.py)

The device in every earlier page was the RTL in a simulator. The same RTL is
also built into an FPGA image for an Alveo U280. This page runs one artifact
on the simulator, on the board's simulated twin and on the card, and checks
that all three agree to the cycle.

## One artifact, three backends

On a machine with iverilog:

```sh
PYTHONPATH=python:compiler/python ~/tritonenv/bin/python \
    tutorial/examples/11_on_the_board.py --backend sim-file fpga-u50-sim
```

```text
one artifact, key 67e954274adc: AXPY over 1000 elements, 63 programs
  sim-file      SUCCEEDED     7044 cycle(s)  host wall   9.043 s  correct=True
  fpga-u50-sim  SUCCEEDED     7044 cycle(s)  host wall  10.853 s  correct=True
  the same cycle count on every backend: True
```

On the host with the card, with the image already loaded:

```sh
python tutorial/examples/11_on_the_board.py --backend fpga-u280 --xclbin tnca_u280.xclbin
```

```text
one artifact, key 67e954274adc: AXPY over 1000 elements, 63 programs
  fpga-u280     SUCCEEDED     7044 cycle(s)  host wall   0.009 s  correct=True
```

It is the same key, so it is the same compiled kernel. It took the same 7044
cycles on all three, because the board runs the RTL the simulators run. A
board result that matched the answer but not the cycle count would mean the
board path had changed something.

The wall times differ by three orders of magnitude, from about 10 s on a
simulator to 9 ms on the card. That gap is the reason to have a board.

The only difference in the code is how the session is opened:

```python
Session(profile='core-vrf-sync-v1', backend='fpga-u280', xclbin='tnca_u280.xclbin')
```

The image has to be named because it is a fact about the card, not about
the profile or the artifact. `fpga-u50-sim` is the same register protocol
the card speaks, driven against the RTL in a simulator. That is how the
board's host side is tested without a board.

## The clock

The device runs at the kernel clock divided by `DEV_DIV`
(`rtl/board/tnca_u50_xo.v`). The run above was on the image with a 150 MHz
kernel clock and **`DEV_DIV = 2`, so the device ran at 75 MHz**. That was
measured, not assumed:

- the shell's frequency counter reads 150 MHz for the kernel clock;
- `flow/fpga/measure_clock.py` times two spin kernels of different
  lengths on the host and divides the difference in device cycles by the
  difference in wall time: **74.97 and 75.02 MHz**.

So 7044 cycles is 94 µs of device time. The 9 ms of wall time is almost
all host and PCIe.

The same day an undivided image followed: kernel clock 100 MHz,
`DEV_DIV = 1`, measured at 100.07 and 100.01 MHz. It ran ArchBench with
every cycle count equal to the simulator's. That is now the flow's default;
`evidence/r38/u280_100mhz/` has the record. Cycle counts do not depend on
the clock, so the 7044 above is the same on either image. Only the time
they take changes.

## Building and loading an image

```sh
# package the RTL as an XRT kernel, then link it for the U280: about 4 hours
vivado -mode batch -source flow/fpga/package_xo.tcl -tclargs . <build> xcu280-fsvh2892-2L-e
TNCA_BOARD=u280 flow/fpga/link.sh <build>

# load it, through the gates
TNCA_BOARD=u280 flow/fpga/program.sh --check <build>/tnca_u280.xclbin
TNCA_BOARD=u280 flow/fpga/program.sh --go    <build>/tnca_u280.xclbin
```

What each step does:

- **`link.sh`** fails if v++ lowered the kernel clock to make timing. An
  image at a frequency nobody asked for is not the design that was measured.
  At the end it reads the frequency back out of the `.xclbin`.
- **`program.sh --check`** loads nothing. It checks the platform and
  interface identity, that the device is idle, that no other load is in
  progress, and the FPGA and HBM temperatures.
- **`program.sh --go`** loads only if those all pass. The card is shared:
  loading an image is a decision, not a side effect of running something.

## Checking a board against the simulator, in bulk

ArchBench runs the same points on the board and on `fpga-u50-sim`, and
`tools/r38/board_vs_sim.py` compares them point by point. On the 75 MHz image
all 27 tiny points matched the simulator to the cycle, from 1776 to
5 323 042. The files are in `evidence/r38/u280_75mhz/`.

`doc/r38_board_bringup.md` section 6 records how the device went from
37.5 MHz to 75 MHz: what limited the clock, what was changed, and what
each change cost or did not cost in cycles.

Back to the [index](README.md).
