# R40-04 checkpoint — 2026-10-11, not accepted

`gates.json` preserves the simulation checks available at this checkpoint.
The package remains not-measured pending real HBM, timing/CDC, calibrated
model fidelity and frozen bandwidth/performance gates.

- AXI bridge: 42/42 (six geometries, seven cases each).
- Semantic mutations: 11/11 detected.
- Address planning: 9/9, including PC31 and a nonzero window base.
- Board wrappers: r36-fpga-tests 8/8 on gx10-6646.
- Protocol: 34/34 on the local tree.
- Vivado 2023.2 packages the v5 HBM configuration with OOC-only clock
  constraints and all requested profile parameters exposed by the wrappers.
- First HBM link failed: u280_r40a, WNS -0.683 ns; no xclbin was produced.
  Diagnostics and a fresh link are in progress; see doc/r40_hbm.md §7.

Raw simulation logs are under ~/tritonchip-r40-next/sim/build/r40 on
gx10-6646. Vivado outputs are under ~/tnca_build/u280_r40next on acf3019.

HBM device group revalidated 8/8 on gx10-6646: default native 16/16,
adverse native 5/5, capacity 24 MiB into the device, all 32 MiB model
samples, delayed/reordered samples, and line8 adverse/capacity/samples.
This is RTL simulation, not the physical 32-PC or board bandwidth gate.
