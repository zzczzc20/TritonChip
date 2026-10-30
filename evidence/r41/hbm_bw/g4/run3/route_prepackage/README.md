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
reviewed platform baseline; the comparison alone is not a disposition.

The closed early reports additionally show the three cores and three local
TMUs physically placed in their respective SLRs, with no failures:
[`actual_cores.rpt`](actual_cores.rpt), SHA-256
`edef2c1486e20da4b73f1a54129f2f8bb6648751a08dee7299d4f94552a710f4`.
All 22 rows in [`failing_setup_all_groups.tsv`](failing_setup_all_groups.tsv)
are HBM platform paths; SHA-256
`a2d778630f4d11e6296ebd7dbdda592eb2befff34d33f03ff182e574a9f4553d`.
The [`bus_skew.rpt.gz`](bus_skew.rpt.gz) report contains 145 MET and zero
VIOLATED entries; decompressed SHA-256
`e897d0999580e6c17d177d47b5261f4cf3d48125c8f3046fb6075ca329067082`.
The candidate's [`platform_irq_trace.txt`](platform_irq_trace.txt) has
SHA-256 `2b13c1d5191c0693b057ca5cce743357900d1a8c394b51d0096bea17f66f2f59`.
Its separate implemented source/D and IRQ/D cone trace has SHA-256
`c2018fea38f9eac440dbf444192db5eb3828fc326fd9c3ea4e39d8f0deacf682`,
identical to the original run-2 trace. Structural identity alone does not
establish safe asynchronous operation.

A second read-only Vivado trace resolves the candidate source register's
hierarchical net segments: [`platform_irq_segments.txt`](platform_irq_segments.txt),
SHA-256 `ff70d990058bae3cabb246ad2cacc60e27ce015ec9ea7893d8dde54e70de9976`;
the corresponding Tcl is [`platform_irq_segments.tcl`](platform_irq_segments.tcl).
It connects source Q to I5 of the actual D-driving LUT6
(`INIT=64'h5554FFFF55545554`) and its I4 to a LUT4
(`INIT=16'h0008`) of `r_state` and `unblock`. The exhaustive 16-case check
`python3 tools/r41/run2_postlink/check_irq_sticky.py` passes: if Q is asserted,
reset is inactive and that clear-control LUT is zero, D remains asserted for
every combination of the other inputs. Clear-control is one only for
`r_state[2:0]=011` and `unblock=0`. This is implemented-netlist evidence of
a local sticky invariant; it does not prove the minimum assertion duration,
glitch-free multi-source aggregation, or dispose the CDC finding.

The subsequent individual review is recorded in
[`candidate_cdc_review.json`](candidate_cdc_review.json), SHA-256
`adec6741f5082aac7992cfe244bbf1b77fd8308f5cd832f6db16c8bd11f48129`.
The 55 unchanged findings retain their earlier platform-scoped explanations.
The new finding is the same vendor MI firewall read-error level into the same
management IRQ synchronizer as the prior reviewed SI error. The traced
asserted-state retention and the documented [AXI Firewall v1.2 port contract](https://docs.amd.com/r/en-US/pg293-axi-firewall/Port-Descriptions)
support its sticky-error interpretation; the [recovery protocol](https://docs.amd.com/r/en-US/pg293-axi-firewall/Recovery)
recommends detecting and handling a blocked condition before unblocking. The
[AXI INTC v4.1 guide](https://docs.amd.com/v/u/en-US/pg099-axi-intc) describes
synchronization of asynchronous interrupt inputs. The raw finding remains
Critical. This is conditional platform-scoped operation under the vendor
recovery and coordinated-reset assumptions, not a proof of minimum pulse width
or glitch-free simultaneous assertion and clear, and not a blanket waiver.
