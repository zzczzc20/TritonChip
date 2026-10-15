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
actual-DCP check. `dcp_audit/placement.log` ends with
`R40_MULTICORE_PLACEMENT_CONTRACT_PASS`; all three private code queue
copies are present. The assignment report explicitly states that no placement
or route was performed. DCP SHA256:
`3402a97b6cc2899c027491a560bcaedca6f7bb48164cef7cafc5e4dd09099931`.
The new OOC resources are 954722 LUTs, 315709 FFs, 192 RAMB36,
45 RAMB18 and 531 DSPs. The separate older route-congestion failure remains
retained in `../route_congestion/`; this successful contract audit does not
clear that physical routing risk.
