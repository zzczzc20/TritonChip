# U50 adapter contract

`U50Adapter` is the boundary between the generic harness worker and the
repository's existing U50 flow. It does not reimplement Vivado or Vitis: it
runs the checked-in `flow/fpga/package_xo.tcl` and `flow/fpga/link.sh`, then
verifies their files and reports.

```python
from pathlib import Path
from tools.tritonchip_fpga_harness.adapters.u50 import U50Adapter, U50Config

adapter = U50Adapter(U50Config(repo=Path("/path/to/TritonChip")))
plan = adapter.run(Path("/scratch/runs/u50/<run-id>"), dry_run=True)
assert plan.preflight.ok
# A real run is intentionally explicit:
run = adapter.run(Path("/scratch/runs/u50/<run-id>"), timeout_s=8 * 3600)
if not run.artifacts or not run.artifacts.ok:
    raise RuntimeError(run.as_dict())
```

The package command is:

```text
vivado -mode batch -source flow/fpga/package_xo.tcl \
  -tclargs <repo> <build-dir>
```

The link command is the existing script:

```text
flow/fpga/link.sh <build-dir>
```

A full run is accepted only if all of these hold:

- `tnca_u50.xo` and `tnca_u50.xclbin` exist and are non-empty;
- `link.log` reports `VPP_RC=0` and `LINK_OK`;
- no `AUTO-FREQ-SCALING` marker is present;
- `xclbin.info` contains the clock mapped from shell clock ID 0 and its
  frequency is exactly the configured 150 MHz;
- artifact SHA256 values are recorded in the result.

An absent `tnca_u50.xclbin` fails even when a wrapper accidentally reports
`VPP_RC=0`. This is a deliberate fail-closed check. The adapter never calls
`program.sh` and therefore never changes board state.

`dry_run=True` resolves the commands and records their order without creating
or deleting a build directory. Vendor tools and platform files become warnings
in this mode, while source/flow path and configuration errors still fail.

## U280 PR5 adapter

`U280FlowAdapter` reuses the same validator but binds the board-specific PR5
contract: `TNCA_BOARD=u280`, Vitis 2023.2, platform
`xilinx_u280_gen3x16_xdma_1_202211_1`, part `xcu280-fsvh2892-2L-e`, kernel
clock 100 MHz, divider 1, and output `tnca_u280.xclbin`. PR5 keeps the package
top and XO name as `tnca_u50.xo`; this is why the profile has different XO and
XCLBIN names. The package command receives the U280 part as its third Tcl
argument, while the link command remains the checked-in board-selectable
`flow/fpga/link.sh`.

The adapter accepts only the documented `AUTO-FREQ-SCALING-08` advisory marker.
Any other automatic frequency-scaling rule, a missing output, nonzero
`VPP_RC`, missing `LINK_OK`, or a clock readback other than 100 MHz fails the
run. It never programs the card.

## Post-link board stage

The generic Harness owns the post-link boundary. When a manifest contains a
`board` block, it passes the exact linked XCLBIN to the profile programmer,
runs conformance with `--require-measured`, and then invokes ArchBench. The
board stage is separate from this adapter so it can run in a later board-access
window. It records `board_measurement.json`, raw conformance/ArchBench output,
and `archbench_summary.json` containing `done_cycles`, `done_retired`, and the
`perf_*` attribution counters. The U280 programmer requires an explicit
`TNCA_BDF` (or manifest `bdf`) so a worker cannot silently program the wrong
card.
