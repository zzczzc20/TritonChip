# Hybrid local/HBM image — audited and loaded, not R40 accepted

FPGA build `~/tnca_build/u280_r40hybrid/xo/tnca_u280.xclbin` completed successfully.
The XO's 46 Verilog sources match immutable commit `c920a1f`; full hashes and
paths are in `xo_sources.json`. Packaged WORDS=65536 and LOCAL_LIMIT=524288 retain
the frozen B39 512 KiB RAM budget while making the existing HBM window available.
This is one physical core in SLR1, not the multicore ownership candidate.

The routed image meets 100 MHz kernel/device and 450 MHz HBM clocks. Setup/hold
slack is +0.016/+0.006 ns with no failing endpoints; bus-skew has zero violations,
and there are zero unclocked registers or unconstrained internal endpoints.
The seven inputs/three outputs without I/O delays are platform satellite control,
PERST, UART and HBM CATTRIP ports, explicitly listed in `checks.rpt.gz`.
All 58 critical CDC paths are platform infrastructure: zero TNCA endpoints.
No blanket RTL waiver or clock reduction was applied.

`preflight.log.gz` and `program.log.gz` record an idle board, no competing loader,
FPGA/HBM temperatures 31/26 C and successful loading of UUID
`826e9146-9f38-6edb-a226-255a4071ea25`. Source/config/xclbin and uncompressed report
hashes are retained in `manifest.json`. Reports reproduce with
`flow/fpga/audit_r40_image.tcl`; source bytes reproduce with
`tools/r40/audit_xo_sources.py --commit c920a1f`.

Three native diagnostic points pass numerically, but the initial transpose
measurement differs from later repeats. The add point remains slower than B39.
These results are retained separately under R40-01; no device-time performance
pass, full 77-point rerun or final R40 sign-off is claimed.
