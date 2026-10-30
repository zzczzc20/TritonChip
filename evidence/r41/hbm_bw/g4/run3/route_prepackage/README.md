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

At 2026-10-30 16:38 UTC the third full link produced
`xo/tnca_u280.xclbin`, SHA-256
`791b8af28f2583e4b29cd4bb191d44e8692bed41b7132c493d86818cb0c7ee09`,
UUID `ea1fa1af-cf1a-fda3-a5d2-e4b88a63abe3`. Its xclbin metadata records
100 MHz DATA_CLK and 433 MHz hbm_aclk. The runner retained exit 3
(`LINK_FAILED freq-moved`); it is not a clean timing closure. The original
routed checkpoint used by the independent post-link review is SHA-256
`e2e9f6880f04c22a0200f43128eae41037e406dcc5297fa309ddbb7b1f595b94`.
Post-link reports and individual CDC review are in progress. A new,
image-specific author decision is needed both for the 433 MHz HBM deviation
and before loading this image on the shared card.

The independent routed-checkpoint analysis records 1,278,501 fully routed
out of 1,278,501 routable nets, with zero routing errors:
[`route_status.rpt`](route_status.rpt), SHA-256
`2f9a29b957406fb7356caf23aa467a9322238dc828b64937374db166e9caa06b`.
The detailed [`cdc.rpt.gz`](cdc.rpt.gz) has decompressed SHA-256
`8c98866634215f258f21c6baf2b1a3c848425ae65b50289a7ccaf4679dcae529`.
Its 56 critical rule/source/destination triples are exactly the same as the
original run-2 CDC report. One of those differs from the earlier individually
reviewed platform baseline, so this comparison is not a disposition; that
source still needs an individual circuit and constraint review.
