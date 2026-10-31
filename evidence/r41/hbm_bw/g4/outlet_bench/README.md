# Multicore outlet board bench

**Domain: RTL simulation on GB10.** These results are functional bench results only; they make no board, timing, resource, or bandwidth claim. No card was accessed.

## Implementation

`tnca_multicore_board_regs` now exposes `AXI_OUT_N` and `AXI_PIPELINED_ISSUE`, derives `AXI_ID_W` with the same expression as `tnca_multicore_hbm_xo`, and uses that width for every AXI ID wire and slice. The existing `tnca_axi_line_model` already provides generic `OUT_N`/`ID_W` storage; the bench now connects those parameters without changing its latency, random stall, or response-selection logic.

`image_geometry_tests.py` accepts repeatable `--rtl-param NAME=VALUE` overrides. A minimal forwarding fix in `tnca_multicore_fpga` repairs the pre-existing `HOME_LINE_WRITE_NOFILL` elaboration mismatch; its default remains unchanged.

## Evidence

- `default_image_geometry.log`: the default image-geometry command passes, including both full grids, all memory-control configurations, coherent host-window ownership, and final quiesce.
- `outlet32_pipelined_image_geometry.log`: the command with `AXI_OUT_N=32`, `AXI_PIPELINED_ISSUE=1`, and `AXI_LINE_LUTRAM=1` passes the same checks. Its memory-control grids pass at `1/32` and `3/32`.
- `default_r40_line_axi.log`: `make r40-line-axi` passes, including the logged 78 line-AXI configurations plus 8 stream cases, 7 multicore configurations, and 2 delayed-write multicore configurations.
- `default_image_geometry_elab_failure.log`: the preserved initial failure caused by the missing `HOME_LINE_WRITE_NOFILL` forwarding parameter. The successful default run above followed the forwarding fix and did not replace this log.

Each run log records its command, UTC clock, base commit, worktree-diff hash, and image-TCL SHA-256. The commands are:

```sh
python3 tools/r40/image_geometry_tests.py --image-tcl evidence/r41/stage2/integration/image.tcl
make r40-line-axi
python3 tools/r40/image_geometry_tests.py --image-tcl evidence/r41/stage2/integration/image.tcl \
  --rtl-param AXI_OUT_N=32 --rtl-param AXI_PIPELINED_ISSUE=1 --rtl-param AXI_LINE_LUTRAM=1
```
