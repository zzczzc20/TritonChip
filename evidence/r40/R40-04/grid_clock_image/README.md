# Common-clock/grid32 single-core image: post-route audit

Build: FPGA host ~/tnca_build/u280_r40grid/tnca_u280.xclbin. Source commit 5d3e3f359056fbb1031f685a86bf9eb298373abc; all 41 packaged Verilog files match that commit. SHA256 and UUID are in manifest.json. The image retains the requested 100 MHz kernel/device and 450 MHz HBM clocks, implements common-clock endpoints and the 48-bit barrier epoch, and has one physical core in SLR1.

Routed setup slack +0.006 ns, hold slack +0.006 ns; no failing setup/hold endpoints, no unclocked registers and no unconstrained internal endpoints. Bus-skew report has zero VIOLATED entries; minimum detailed reported slack is +1.545 ns. The CDC report contains 58 critical paths (CDC-1:1, CDC-7:13, CDC-10:8, CDC-11:32, CDC-13:4), all in the platform clock/reset/interconnect infrastructure, with zero TNCA endpoints. Full path details and source/report hashes are retained; no blanket RTL waiver or clock reduction was applied.

The seven inputs lacking I/O delay are platform satellite-control bits, PERST and platform UART RX; the three outputs are HBM CATTRIP and platform UART TX. They are recorded explicitly in checks.rpt; they are not TNCA data/control ports. SLR1 consumes 296985 CLB LUTs (68.75%), 253.5 BRAM tiles and 171 DSPs for the complete routed region, including infrastructure. This is not a three-core resource estimate.

Large original reports are stored with deterministic gzip compression; manifest report_sha256 values hash their uncompressed bytes. flow/fpga/audit_r40_image.tcl reproduces all reports from the routed checkpoint in one Vivado session. The recorded audit used two sessions because the first timing report requested an additional bus-skew check, which was then completed.

The image has not been loaded or functionally accepted. Next: load with the board idle, validate native common-clock endpoints and standard-path HBM correctness, then complete the frozen timing/performance protocol. This evidence does not close R40-04, R40-05 or R40. It contains none of the newly implemented multicore AXI/MMIO v2 RTL.
