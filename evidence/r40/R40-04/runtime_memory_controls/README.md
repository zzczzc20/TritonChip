# Same-image runtime memory controls: functional evidence

These are mechanism diagnostics, not board performance or final-image gate passes.
MMIO capability bit 5 advertises the extended register map. Idle-only writes
select one/all physical outlets and one/eight outstanding transactions. Legacy
images refuse explicit controls. The backend performs no extra configuration
MMIO reads per graph launch. Line admission counts actual occupied AXI slots,
including writes awaiting B; it does not count upstream queued requests.

`functional_reports.tar.gz` retains source-hashed reports from snapshots 07–11.
Snapshots 07, 10 and 11 completed all four resident native Torch configurations
with numerical checks and actual outlet traffic. Snapshot 08 caught mutations
that bypass slot admission, alias source IDs, or permit unsafe configuration
changes. Snapshot 09 exposed an unresolved core reservation-X assertion in the
three-outlet/one-outstanding scenario after the extra per-launch probes were
removed. Subsequent passing runs do **not** establish that this failure is fixed.

`failure_capture.tar.gz` (SHA256
`745070cdfc91b0c510aa146515f955e8f66760544668cf2b7e7742ee86d8a8ed`)
retains the failed source, compiled simulator, register transcript and first
assertions. The run was deliberately terminated after observing the assertion.
The simulator now stops at the first reservation mismatch and prints the decode
state; this failure-only diagnostic is excluded from synthesis.

Historical pre-journal blocker: the undefined reservation source was not explained. Next
experiment: run `python3 tools/r40/memory_control_tests.py --native --python
$HOME/tc_env/torchenv/bin/python --out /tmp/r40-memory-controls-next` on the jump
host with its environment loaded and a fresh output directory. On failure,
preserve the first assertion and identify the producing instruction/state;
on success, vary controlled backpressure to reproduce snapshot 09. Do not add
MMIO delays or hide unknown state to obtain a pass.

## Follow-up backpressure experiments

`followup_reports.tar.gz` retains snapshot12 and controlled AXI-model seed7
and seed2026 runs. All completed the same four native configurations with
correct values and active/disabled outlet checks. The seed experiments change
only the test model PRNG initialization in separate immutable directories;
their model hashes and rationale are retained. These passes do not explain
snapshot09 and do not close the reservation-X blocker. No production RTL was
changed to clamp unknown state or add host delays.

## Timed failure capture

[Timed mailbox diagnostics](timed_journal/README.md) now retain a new reservation-X
failure on the same four-configuration native graph, its simulated input schedule,
responses, abnormal stop and exact harness. Normal replay and response/time/partial
mutations validate the reproducer boundary. Later strict replay locates the illegal-opcode classification source and validates
a candidate repair through the failing input prefix. Full native regression and
final-image validation remain separate requirements.
Use the next command and success/failure branches in that package rather than
adding live MMIO probes that change the host schedule.
