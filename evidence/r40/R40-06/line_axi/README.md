# R40-06 full-line AXI and elastic multicore integration

Parent: 8fb64fa. Functional validation on the jump host, locked Icarus environment. Full matrix snapshot ~/tritonchip-r40-ports-final; final source/check snapshot ~/tritonchip-r40-axi-final; final independent line oracle ~/tritonchip-r40-line-final. These are RTL functional results, not native Torch AXI/HBM performance acceptance.

Implemented together:
- Full 64-byte NUCA line to AXI INCR bursts on 64/128/256/512-bit buses, independent outstanding slot IDs and original source/tag returns, read interleaving, stable AW/W/AR under backpressure, and same-line read/write ordering.
- Per-outlet elastic request links with two-entry FIFOs; ready uses registered occupancy and does not propagate combinationally across stages. Response registers carry one complete line per cycle.
- Public persistent multicore shell with independent AXI outlets, private instruction RAM, global grid IDs/telemetry and physical-memory fault input. Public completion, next-command admission and all_idle wait for every adapter and link to drain; AXI fault overrides completion status.
- Sticky error/stray/malformed-burst/address faults and explicit reset contract. Lost replies or queued requests blocked behind a fault require platform recovery. No fabricated successful completion.

Passing evidence:
- 35 line configurations: bus widths 64/128/256/512, outstanding slots 1/3/8, elastic stages 0/1/2/4, read and write AXI errors, stray replies, malformed bursts, invalid addresses and shared-fabric reset. Four cases explicitly verify the NUCA no-write-response contract. Mixed traffic checks complete 512-bit contents, source/tag exactly once, counters and simultaneous R/B arbitration.
- Seven real-core configurations (cores, outlets, stages): (1,1,0), (2,2,1), (3,2,2), (3,3,2), (4,4,2), (8,3,2), (8,8,4). Each uploads/reads code through the public host port, checks invalid PC/address, deadline drain, two 455-program 3-D memory grids, all 455 coherent host readbacks and error propagation stopping a subsequent grid. Every outlet has real reads and writes; eight-outlet counts are reads 21/21/21/21/24/21/21/21 and writes 7/7/7/7/8/7/7/7.
- Final-source three-core two/three-outlet cases pass again after wrapper formatting and model window guards. Two deliberately late write-response cases (300-cycle additional B delay) pass and hold global completion for 84/93 drain cycles.
- Legacy persistent shell on 1/3/8 cores passes, and backend registry passes 3/3. Python scripts compile and git diff --check passes.

The initial integrated second grid trapped numerically despite AXI/tag faults being zero: sending AXI B responses to NUCA homes corrupted an MSHR awaiting a read. The retained diagnostic log records that failure. NUCA writebacks are fire-and-forget at the home interface; the corrected adapter consumes B internally, and the public wrapper waits for B drain before completion. WRITE_RESPONSES=1 exists only as an explicitly selected independent-port diagnostic mode.

Source differences between the first full matrix and final snapshot are wrapper width formatting, explicit constant-zero fault ties in legacy benches, model range guards, and the late-B test parameter/monitor. Production AXI adapter/link and fault-enabled shell are identical; the final wrapper and model pass the final three-core and late-B cases. Source hashes here identify the committed batch.

This closes the functional AXI/link integration subtask. It does not close R40-05/06: AXI-Lite/runtime binding, real HBM capacity/connectivity, three-SLR placement/timing/CDC, native Torch board execution, locality/DCO, bandwidth accounting and same-resource performance gates remain required. The single-core common-clock board image audited separately contains none of this new multicore AXI integration.
