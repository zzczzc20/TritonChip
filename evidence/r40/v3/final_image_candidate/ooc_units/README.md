# OOC unit timing: attempt-3 RTL vs final-image candidate (10 ns, xcu280-2L)

Post-synthesis timing of every changed module, read as SystemVerilog like the board build
(`tools/r40/final_image_ooc.sh units`, `flow/synth/vivado_ooc_timing.tcl` with `OOC_SV=1`), on the FPGA host,
2026-10-19. Old: `~/tnca_src_r40_attempt03_54249f7` (A3 image source). New: candidate `d2cc850` RTL.

| Unit (change) | A3 WNS | Candidate WNS | Failing endpoints |
|---|---:|---:|---:|
| tnca_vector_fpu (M4) | +3.175 ns | +3.184 ns | 0 / 0 |
| tnca_line_axi (M3) | +7.427 ns | +7.523 ns | 0 / 0 |
| tnca_memory_mode (M3) | +8.864 ns | +8.855 ns | 0 / 0 |
| tnca_multicore_mmio (M3) | +7.378 ns | +7.378 ns | 0 / 0 |
| tnca_idle_host_mux (M1) | +9.204 ns | +9.194 ns | 0 / 0 |

No changed unit develops a new local critical path. These are module-level estimates: the 100 MHz failure is the
cross-SLR cone through these modules, which only the kernel synthesis, the structural cone audit and the
routed census can judge. Raw reports: `units_reports.tgz`.
