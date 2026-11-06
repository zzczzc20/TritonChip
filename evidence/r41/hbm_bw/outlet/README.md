# Pipelined line-outlet experiment

## Functional result

`PIPELINED_ISSUE=1` adds a one-entry pending-issue buffer to `tnca_line_axi`. Requests reserve their outstanding slot on acceptance. The current AXI issue can complete and be replaced on the same edge, while a stalled issue can accept one successor into the pending buffer. Request-ready remains independent of AXI ready. Same-line hazard checks still include every reserved slot, and faults stop further admission while accepted issues retain their state.

The option defaults to zero and is passed as `AXI_PIPELINED_ISSUE` through the multicore AXI shell, FPGA binding and U280 wrapper. It does not change instruction encodings or arithmetic. There is no new image yet.

Functional checks:

- `python3 tools/r40/line_axi_tests.py`: 78 configurations pass, covering both issue modes, 64/128/256/512-bit AXI, one/three/eight slots, independent random stalls, interleaved responses, same-line ordering, write-response modes, elastic link stages, runtime outstanding limits, error/stray/malformed replies and reset.
- Eight sustained stream cases pass: read and write, 32 and 64 slots, both issue modes. For 256 independent 512-bit lines, the AXI issue span falls from 510 to 255 cycles. This is an RTL interface throughput test, not an HBM measurement.
- GB10 `python3 tools/r40/multicore_axi_tests.py --cores 3 --pipelined-issue`: both configurations pass (two and three outlets, two link stages), including real instruction execution, coherent host accesses and expected fault/deadline completions.

Logs and source hashes are retained in this directory. These checks do not establish the full 113-point regression or board performance.

## Physical screens and area decision

The x86 Vivado host rejected both SSH attempts with a connection reset. The free CPU-only screens therefore run on the FPGA host through the required shared slot/memory guard. No card handle was opened and no full-run allocation was consumed.

The first screen script incorrectly read the product RTL as plain Verilog, whereas the packaging flow uses SystemVerilog. Both runs terminated during parsing. Their original script, logs and exit codes are retained in `screen_v1/`. Version 2 reads SystemVerilog and uses the same RTL, 512-bit data, write responses enabled and pipelined issue.

The 32-slot synthesis reports 17,514 LUTs, 21,746 registers and zero BRAM. Replicating 24 such adapters costs 420,336 LUTs before stream logic and interconnect. The existing routed image already uses 927,245 LUTs across its SLRs. Their sum, 1,347,581, exceeds the device's 1,303,680 LUT capacity; replacing the three old adapters would recover some area, but does not resolve the much tighter CLB packing constraints (96.74%, 88.67%, 85.11%). Therefore the generic CAM/line-register adapter is not selected as the replicated stream datapath.

The synthesis result changes the next implementation step: use burst-based, ordered per-PC stream engines with BRAM buffering and no per-line same-address CAM. The generic issue experiment establishes the cycle-rate requirement and remains an optional tested component; it does not solve the area or core path requirements.

Both version-2 routed screens have completed successfully under `~/r41_hbmbw_outlet_v2/`. The 32-slot screen has WNS +1.658 ns, WHS +0.011 ns, zero TNS/THS and 17,443 routed LUTs. The 64-slot screen has WNS +0.163 ns, WHS +0.021 ns and zero TNS/THS. Full routed reports for both are retained in `screen_v2/`. The 64-slot synthesis uses 43,656 LUTs and 42,448 registers, reinforcing the area rejection. The OOC script applies a 10 ns clock, 0.2 ns uncertainty and 1 ns input/output delay assumptions. It includes no shell or full-image placement pressure.
