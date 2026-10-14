# Synthesized dispatcher telemetry

Parent: 148442b. Jump host, real Icarus mesh RTL, locked Torch/Triton.

The dispatcher counts accepted programs and core completion, occupied cycles for each physical core, and peak simultaneously occupied physical cores. Counters include memory stalls and retain the last accepted dispatcher launch until the next. Mailbox operations 12/13 expose these public RTL ports; the testbench does not manufacture activity counters. Opcode 15 stays refused.

Resident native Torch graph: 4097 FP32 elements, 257 programs, one whole-grid launch, zero measured tensor staging. 1/2/3-core numerical and hardware telemetry checks passed. Three cores have 85/85/87 completed programs and peak concurrency 3. See native logs for individual occupied cycles.

Protocol/admission checks: 15 passed, including refusal/malformed/inconsistent telemetry. Public Session Triton tail and exactly-once 455-program 3-D atomic oracle passed. Registry 3/3; conformance all three simulation backends measured and passed. Mailbox 455-program launch, count totals and invalid telemetry index refusal passed.

The dispatcher testbench independently samples accepted-core occupancy and checks peak/per-core cycles across 1/2/3/4/8 cores, zero/invalid work, mixed latency, backpressure, each-core fault and global abort/relaunch. Those protocol configurations passed; the extended real-core memory and shell matrices are still running at this commit, as are native 4/8-core and 6145-program mailbox tests. Their terminal results will be recorded separately.

Native runs started before the final additional host-side telemetry validation; the unchanged counter readers and hardware path were exercised by native runs, and final validation branches were separately covered by option tests. This is simulation evidence, not three-SLR placement or board performance acceptance. Full R40-05 remains open.
