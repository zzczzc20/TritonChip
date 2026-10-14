# Three-tile platform and physical link pipeline checkpoint

2026-10-14. Functional evidence; R40-05/06 are not accepted by this checkpoint.

The U280 wrapper exposes three 256-bit AXI masters, one common global buffer,
MMIO v2 through the real AXI-Lite adapter, and platform reset with synchronized
release. Argument offsets A8/100/108 alias the same allocation base. The board
host refuses xclbins whose three arguments select different allocation groups,
before allocating a BO. DMA ownership transfer remains unsupported.

NoC virtual networks, dispatcher commands/completions, each home-to-outlet link,
and each outlet-to-AXI link have two stages. Every request/command stage uses a
two-entry FIFO with ready determined by registered local occupancy. Completion
waits for an actual tile start. Buffered home writebacks and AXI replies must
drain before a launch completion can be consumed. Reset requires resetting the
attached AXI fabric/controllers too.

Jump-host commands (locked Torch/Triton interpreter, `source ~/tc_env/env.sh`,
`PYTHONPATH=python:compiler/python:tools/asm`):

```sh
python3 tools/r40/multicore_mmio_option_tests.py
python3 tools/r40/multicore_mmio_synth_tests.py
python3 tools/r40/line_axi_tests.py
$TRITON_PYTHON tools/r36/backend_tests.py
$TRITON_PYTHON tools/r40/multicore_mmio_tests.py --cores 3 --ports 3 --pipeline 2 --board-top
python3 tools/r40/mesh_pipeline_tests.py
```

Results: five synthesized control/link modules, nine rejected invalid geometries,
35 AXI configurations, 13 host boundaries, backend registry 3/3, legacy FPGA 8/8, and four pipeline/seed configurations
with 20,000 packets each, both virtual networks and all 32 source/destination/VN
combinations covered. The board-top protocol test passes two 455-program 3D grids,
code upload/readback, data readback, base aliases, stale/unknown requests,
zero-work launch, deadline drain, device timestamps, and traffic on every outlet.

`package.log` records successful Vivado 2023.2 XO packaging with the three inferred
masters and the 100 MHz clock association. OOC synthesis, linked placement/route,
board programming, graph makespan speedups, and HBM bandwidth acceptance are
separate gates. `place_r40_multicore.tcl` is a strict placement input requiring
three tile/router hierarchies and two-stage links; its actual placement must be
verified from the linked routed DCP. It does not constitute placement evidence.

Native three-core board-top Torch, descriptor and pair runs have now passed.
`native_3.log` records 257 programs, all three cores active, zero measured tensor
staging, and real read/write traffic on every outlet. `descriptor_3.log` and
`pair_3.log` record the generated artifacts, actual descriptor loads, physical
chunk widths, resource partitioning and three-core activity. Eight-core pipelined
Torch is still live; no pass is inferred from progress.

Packaged XO SHA256: `df9eeff80746c5d5e3df101d03c1887777b45f3ae196218fc61096419af47e49`.
`packaged_sources.sha256` records the exact RTL input tree. The committed
`tnca_top.v` subsequently replaces the two SystemVerilog size casts of home IDs
with an equivalent width-qualified localparam, to preserve the legacy Verilog
parser gate; other packaged RTL inputs match this checkpoint. Repackage the
committed tree before linking a deliverable image.

Final XO repackaging is recorded in `package_final.log`, SHA256 `c10beabaf7c89a9f93d50cf1b5c59d2576e0c63343e11130e72f7254b78e2d60`.
`packaged_final_sources.sha256` now matches the committed RTL, including the
Verilog-compatible home ID localparam. The packager emits `tnca_variant.txt`;
link.sh uses it to select multicore connectivity and placement and omit the
legacy whole-kernel SLR1 constraint. Actual linked placement/route is pending.

`ooc/` contains diagnostic Vivado 2023.2 synthesis resource/timing/check reports.
Synthesis has zero errors and zero critical warnings. OOC utilization is 911882
LUTs, 314376 FFs, 192 RAMB36, 45 RAMB18 and 531 DSPs. This excludes the platform,
HBM fabric and routed implementation costs; it is not a timing signoff.

`pblock_check.log` records the constraints successfully applied to the real
synthesized DCP and written to `placement_pblocks.dcp`. Private code RAM,
tile/router groups, command/completion FIFOs and both request/response memory
pipeline storage are assigned with Pblocks. Vivado ignores USER_SLR_ASSIGNMENT
on primitive cells, so that earlier hint-only approach has been replaced.
This checks constraints, not actual placed or routed locations.
