# Corrected candidate: routed timing before packaging

At 2026-10-30 16:16 UTC, the ongoing x86 implementation generated
`hw_bb_locked_timing_summary_routed.rpt`. The full original report is
[`timing_summary.rpt.gz`](timing_summary.rpt.gz); its decompressed SHA-256 is
`0d84dc0350499debd96132d084f542abe55fc0e2b4d1bfa39c5da5b76330c340`.
The candidate source commit is `8d237223b9ab9cf6afc4228d6bc7149bf4a96086`.

This routed, pre-packaging report records design setup WNS -0.085 ns, TNS
-0.856 ns and 22 failing endpoints; all 22 are in the `hbm_aclk` intra-clock
row. The `clk_kernel_00_unbuffered_net` row records setup WNS +0.111 ns and
zero failing endpoints. Design hold and pulse-width summaries report zero
failing endpoints. The worst setup path shown is within `hmss_0`, between HBM
platform registers. These are routed timing results, not board measurements.

Implementation, clock metadata, bus-skew and CDC review, actual placement
audit, final image identity and board verification remain outstanding. The
old image's HBM-clock exception and load approval do not apply to this
candidate. No board load or acceptance is implied by this report.

During packaging, Vitis estimated and selected a 433.4 MHz HBM system clock
instead of the requested 450 MHz. The retained
[`hbm_scaling_log.txt`](hbm_scaling_log.txt) has SHA-256
`794a573524f1c9a072dfc7fac14b13297f7999f7e03a391488ed989e85221751`.
This tool estimate is not the completed xclbin metadata or an author waiver.
