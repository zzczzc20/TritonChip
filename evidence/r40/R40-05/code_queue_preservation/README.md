# Physical code queues — implementation checkpoint, not accepted

The ownership/code2 candidate synthesized from `025cf62` completed with zero
errors and zero critical warnings. The actual DCP has 954070 LUTs, 315553 FFs,
192 RAMB36, 45 RAMB18 and 531 DSPs. These are OOC resources, not linked timing.

`placement_before_fix.log` records the placement-contract failure on that DCP:
tile one's command pipeline is missing. The broadcast links have identical
inputs and unconditional output readiness, allowing synthesis to merge queue
copies. The intended SLR assignments require separate physical command/reply
queues beside each private instruction RAM.

The shell now marks each code command/reply link DONT_TOUCH to preserve these
copies for board placement. The same RTL protocol passes the six real Icarus
configurations in `code_link_matrix.log`: 1/3/8 cores, direct and 1/2/4 stages,
snapshot/byte masks/private RAM and drain checks. Simulation ignores physical
synthesis attributes; it does not prove that synthesis retained the queues.

`flow/fpga/audit_r40_multicore_placement.tcl` validates all pipeline and private
RAM assignments on an actual DCP, checks ownership hierarchy against the image
capability, and fails missing stages. A new immutable candidate must complete
synthesis and this check before it proceeds to route. The old DCP failure is
retained; no physical placement or performance pass is claimed here.

The immutable `09ebe06` candidate has now completed synthesis and this
actual-DCP check. `dcp_audit/placement.log.gz` ends with
`R40_MULTICORE_PLACEMENT_CONTRACT_PASS`; all three private code queue
copies are present. The assignment report explicitly states that no placement
or route was performed. DCP SHA256:
`3402a97b6cc2899c027491a560bcaedca6f7bb48164cef7cafc5e4dd09099931`.
The new OOC resources are 954722 LUTs, 315709 FFs, 192 RAMB36,
45 RAMB18 and 531 DSPs. The separate older route-congestion failure remains
retained in `../route_congestion/`; this successful contract audit does not
clear that physical routing risk.

The new XO SHA256 is
`8461d27318a69ef12c9d9b0fe71c8f283b3b207d6d89eaa90bd0caeb09d1b2de`.
`dcp_audit/xo_sources.json` matches all 60 packaged RTL sources to `09ebe06`.
The image parameters and exact launch wrapper are retained alongside it.
The detached link started on 2026-10-15, PID 566545 on the FPGA host,
with `SSI_SpreadSLLs` and 100 MHz. Local Vivado 2023.2 command help confirms
the directive; its stdout is retained as `place_help.log.gz`. This is a
placement-strategy experiment against the recorded localized SLL congestion,
not a routed/timing result. Authoritative completion is
`~/tnca_build/u280_r40ownership_preserved09/build/link.exit` and
`xo/link.log`; preserve the existing run while it is live.
