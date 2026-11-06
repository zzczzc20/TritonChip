# Board-top RTL at the candidate image's parameters

**Domain: RTL simulation (Icarus Verilog) on GB10.** Functional results only: no timing, resource, bandwidth or board claim. No card was accessed.

`tools/r40/image_geometry_tests.py` elaborates the board top (`tnca_multicore_hbm_xo` inside `tb/fpga/tnca_multicore_board_regs.v`) with every parameter of an image's `image.tcl` and runs the MMIO v2 protocol through it: code upload and readback, two grids of 455 programs with every result checked, the runtime memory controls (1 outlet / all outlets × 1 / maximum outstanding) with a grid under each, the coherent host window and the ownership handoff.

## Tool changes
- The G4 parameters (`HYBRID_LSU`, `HOME_BACKGROUND_READS`, `HOME_LINE_WRITE_NOFILL`, `HOME_LINE_PIPE`, `HOME_MSHR_N`, `LINE_SLOTS`, `AXI_OUT_N`, `AXI_PIPELINED_ISSUE`) are forwarded to the bench. `AXI_ID_W` is checked against the value the bench derives from `AXI_OUT_N`. Before this, the run-2 image was refused (`run2_refused_axi_id_w.log`).
- A failed grid now reports how many result slots are wrong and which.
- `HBM_STRIPED` is still refused: the bench's 1 MiB window cannot hold the striped 8 GiB layout. The run uses `image_s4.tcl`, which is the run-2 candidate without striping.

## A bench defect found and fixed
The s4 image failed under the single-outlet control with 32 outstanding lines (`s4_attempt1_fail.log`). The detailed run (`s4_attempt2_fail_detail.log`) showed one whole line (result slots 296–303) read back as zero. Both bisection runs passed: `HOME_LINE_PIPE=0` (`s4_bisect_home_pipe0_pass.log`) and `AXI_PIPELINED_ISSUE=0` (`s4_bisect_pipelined_issue0_pass.log`).

A trace of that line (`s4_trace_line_0x30940.log`; the trace blocks are kept as `trace_*.vh.txt` and are not part of the RTL) shows that the device was correct:
- home 2 wrote the line back during the ownership handoff, with the right data;
- outlet 0 issued AW and W and received B before the handoff completed;
- the host then read outlet 2's view, the line's canonical outlet, and that view held zero.

The bench keeps one memory model per outlet. `tb/fpga/tnca_multicore_shared_hbm.vh` copies each committed write into the other outlets' views, as the board's three aliases of one BO would show it. That copy sliced the B ID as a fixed 3 bits. With `AXI_OUT_N` 32 the ID is 5 bits, so a write on slot 8 (`01000`) was mirrored from slot 0's stale buffer. The failure therefore depended on which slot a write happened to use. That is why it appeared only when the timing put this writeback on a slot of 8 or above. It also explains why the earlier 32-outlet bench run (`../outlet_bench/`) passed.

Fix: the mirror uses `AXI_ID_W` (the board bench derives it from `AXI_OUT_N`; `tb/fpga/tnca_multicore_regs.v` now names its fixed 3-bit width). No RTL changed.

## Results after the fix
| Run | Result | Log |
|---|---|---|
| s4 image (`image_s4.tcl`, profile `core-vrf-sync-v5-x8-g2`) | **pass**: both grids, memory controls 1/1, 1/32, 3/1, 3/32, coherent window and handoff | `fixed_s4_geometry.log` |
| run-1 integration image (`evidence/r41/stage2/integration/image.tcl`), regression | pass | `fixed_default_geometry.log` |
| `make r40-multicore-dma` (benches that include the mirror) | pass | `fixed_r40_multicore_dma.log` |
| `make r40-multicore-mmio` | pass | `fixed_r40_multicore_mmio.log` |

The first `make r40-multicore-mmio` attempt ran without the torch interpreter and stopped at its first check (`No module named 'torch'`). That is an environment error, not a test result. It was rerun with `PYTHON`/`TRITON_PYTHON` set to the torch environment, as the gate script does.

Commands (GB10, `source ~/tc_env/env.sh`):
```sh
python3 tools/r40/image_geometry_tests.py --image-tcl evidence/r41/hbm_bw/g4/images/image_s4.tcl \
    --image-profile core-vrf-sync-v5-x8-g2
python3 tools/r40/image_geometry_tests.py --image-tcl evidence/r41/stage2/integration/image.tcl
make r40-multicore-dma
make r40-multicore-mmio PYTHON=$TORCH_PYTHON TRITON_PYTHON=$TORCH_PYTHON
```
