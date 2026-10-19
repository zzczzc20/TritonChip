# Kernel synthesis and cross-SLR cone audit: attempt 3 vs final-image candidate

Candidate RTL `d2cc850` (flow/tools at `c745ea1`), synthesized with the attempt-3 image configuration
(`evidence/r40/v2/convergence/attempt03/image.tcl`; M1-M4 add no image parameter) by
`flow/fpga/synth_r40_multicore.tcl` on the FPGA host, 2026-10-19 (about 30 min). No errors; the only
port-count warning (`tnca_multicore_fpga` 43 declared / 36 connected) is identical in attempt 3's synthesis.

Post-synthesis estimate at 10 ns (no routing): WNS +1.740 ns, TNS 0, WHS +0.058 ns, 0 failing endpoints.

`tools/r40/audit_cross_slr_cones.tcl` on the unplaced netlists (connectivity and logic depth, not routed timing):

| Combinational reach into tile registers | A3 (`ulp_tnca_u50_1_0.dcp`) | Candidate (`kernel/synth.dcp`) |
|---|---:|---:|
| from `device/control/memory_active_ports` | 3,955 | 0 |
| through the home-link idle nets | 3,955 | 0 |
| through `host_allow` | 3,955 | 3 (the registered copy in tile 0 and its two host-mux registers, 1 level each) |
| placement hook dry-run | ok | ok |

The reported 100 MHz failure chain of attempt 3 (control `memory_active_ports` -> memory-mode response
check / home-link idle -> `all_idle` -> `host_allow` -> tile 0 host mux -> LSU pending/partial) is cut by
registers in the candidate. Remaining paths from these sources end at registers in 4-7 levels (estimated
slack +8 to +9 ns). Whether other failing groups hide behind this chain is the routed census's question
(`tools/r40/census_failing_endpoints.tcl`, pending a host with >= 30 GB free).
Raw: `kernel_audit.tgz`.
