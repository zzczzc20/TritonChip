# Dual DRAM paths: validation checkpoint

`kickoff/harness-smoke-v4-evidence.tgz` is the complete source-fingerprinted
remote smoke campaign after the slow-path domain guards changed. Its clean
SystemC/Ramulator build, fast and slow standard Torch graphs, real-artifact
execution, unsupported-op refusal, and off/aggregate/trace invariance pass.
This is functional harness evidence; timing remains uncalibrated.

The additional DRAM-path test checks the fast path's read and write timing
boundaries directly from the configuration, slow fallback read/write
completion, request identity across port/rw/repeated occurrences, reordered
lookup, and explicit address/port/occurrence/table/latency refusals.
The slow path now refuses port >=4, repeated occurrences beyond the 24-bit
key domain, and replay latencies beyond 65,535 core cycles. It no longer
aliases ports, saturates occurrence keys or silently clips long latency.
An inconsistent latency table or duplicate request key also refuses.

The bridge's seven tests pass for idle reads, hit/miss/conflict, bank
parallelism, refresh, idle skip, writes and exact event-skip equivalence
over 200,000 requests. These exercise the configured Ramulator controller;
they do not prove a physical U280 controller's queue or mapping behavior.
The separate `dram-guards-v1-evidence.tgz` retains the initial build/test
logs and binary hashes; the smoke campaign supplies full source identity.

`kickoff/slow-process-lifetime-v1-evidence.tgz` adds the next clean build and
six-step experiment. The actual slow pass now rearms `PR_SET_PDEATHSIG`
after fork and checks its expected parent to close the startup race. Both
Python model launchers use the existing parent-death exec helper; backend
failure cleanup also waits for the killed model. Three tests first observe
live real model/pass processes running an infinite ISA loop, then SIGKILL
only their owner. Direct model, backend and standalone runner ownership
all terminate their active descendants without a process-group kill.
Native artifact and fast/slow standard Torch regression also pass.
`slow-process-old-binary-negative-v1-evidence.tgz` preserves the same test
detecting a surviving pass with the preceding binary, followed by explicit
test cleanup. This is a reproduced defect and its correction, not a dummy
subprocess test. The full smoke manifest now includes this lifetime check
for future clean-archive execution.

`kickoff/harness-smoke-v5-evidence.tgz` contains 18 completed steps after
the fixed-point checks changed. Every traffic-bearing pass now goes through
Ramulator replay, including the final pass. Acceptance requires the same
request keys, per-key latency stability and makespan stability, followed by
reproduction using the installed table. A failed fork pass explicitly
refuses the prediction instead of retrying numerical execution silently.
The native summary, standalone runner, graph runner, fragment and primitive
reports carry timing qualification separately from numerical status.
`predictions.py` rejects inconclusive and historical unchecked slow results.

Actual tests retain a converged memory copy, a numerically correct one-pass
copy whose timing is explicitly inconclusive, downstream residual exclusion,
and a real unsupported instruction causing a fork-pass refusal. The extended
DRAM-path test covers request reordering, changed/missing keys and unstable
per-request latency. The complete smoke reruns actual parent-death cleanup,
native artifacts, monitors and standard Torch graphs.

The unchanged three-pass default reveals a preserved mismatch: eight larger
GEMM commands change request sets (2-41 unmatched keys). Several makespan
changes are below 0.5%, which the old test could have called converged.
They now remain `inconclusive-fixed-point`; all four numerical fragment
cases pass. The ArchBench add graph also passes numerically and retains an
inconclusive timing status. These failures are not dropped or promoted to
calibrated predictions. Small traffic-bearing commands do converge.

`address_map.md` and `kickoff/physical-dram-map-v1-evidence.tgz` add the
recorded B40 BO-to-PC translation, native fast/slow hookup and ten-step
boundary/standard-graph campaign. Logical addresses now subtract the actual
local address limit and add BO base before looking up topology-tagged PC
ranges. The native PC counters use that same map. Map hashes and domains
are explicit; a different allocation requires its own evidence.

`kickoff/physical-dram-map-v2-evidence.tgz` extends the mapped campaign to
eleven steps and nine bridge mechanisms. A 96-line write burst accepts and
completes exactly 192 half-lines with 3,878 send refusals and a 32-entry
write-queue peak. The single-line write test binds completion to both
32-byte burst callbacks and the final WR + CWL + BL timestamp. Independently
sequenced read/write probes observe command gaps of 14 and 13 DRAM cycles,
matching the configured nRTW and CWL+BL+nWTRS constraints. The 200,000-request
event-skip comparison remains exact.

The first simultaneous-arrival turnaround fixture incorrectly assumed an
execution order that the controller was allowed to reorder. Its failure,
source hashes and old fixture are retained in
`kickoff/ramulator-queues-v1-negative-evidence.tgz`. The corrected fixture
injects the second direction after independently observed command issue;
the timing constraints are unchanged. Negative gaps are printed signed.

The native model additionally reports accepted/completed 32-byte bursts,
queue-refusal attempts and maximum queued read/write bursts per PC from the
last replay of each command. These do not sum every fixed-point iteration
or represent physical FPGA counters. Actual mapped execution verifies
accepted = completed = 2 × replayed 64-byte lines. Latencies outside the
bridge's 32-bit DRAM-cycle representation now refuse instead of clipping.

`kickoff/timeline-drain-negative-v1-evidence.tgz` reproduces a maintenance
completion defect: an actual admitted outlet write completes at cycle 26,
but a flush with no dirty cache lines previously returned cycle zero.
Maintenance now includes already accepted outlet transactions in its
completion maximum. `kickoff/harness-smoke-v6-evidence.tgz` passes the
additional accepted-write/read drain case and the full 19-step model smoke,
including all nine bridge mechanisms and native fast/slow graphs.
This corrects that maintenance boundary; it does not certify every physical
controller or host-handoff timing boundary.

`kickoff/host-handoff-v1-evidence.tgz` adds an explicit diagnostic host policy.
The historical coherent host window does not invalidate cache tags, whereas
MMIO DMA hands ownership off before AXI transfers, including ABI uploads
inside native graph calls. No explicit inter-call CPU download therefore does
not imply a matched warm-cache boundary. `dma-maintenance` now uses native
maintenance at that split, with an explicit BO map and zero fitted host costs.
Native fast and slow tests each pass the actual copied output, local-only and
straddling transfer checks, all-slice maintenance completion, and three policy
refusals (unknown policy, missing map, opaque fitted host costs).

On the same standard Torch 4097-element graph and artifact, zero-host-fit
coherent mode retains 13,699 cycles, 335 read lines and 104 write lines.
One in-call ownership handoff changes these to 17,800 cycles, 776 reads and
258 writes. The independent three-outlet MMIO RTL diagnostic observes
21,623 cycles, 860 reads and 257 writes. This establishes a partial cache-boundary
explanation, not full reconciliation or physical calibration. Remaining
per-core/NoC/controller differences must still be investigated. The historical
default and its results are preserved; the diagnostic does not retune the R40 fit.
The quoted cycles are sums of completed launch windows, so prelaunch ABI
maintenance changes cache state but its own elapsed cycles lie outside that
metric. Counter snapshots surround the native call and include its maintenance
traffic. These boundaries are distinct; neither result is whole-call latency.

Remaining Required work includes actual bank/row mapping proof,
controller/AXI/NoC timing-boundary reconciliation, and calibration/validation
residuals. The generic controller's mechanisms do not establish physical
controller equivalence.
Per-command Ramulator state reset remains an explicit applicability limit.
No physical-board memory prediction is qualified by this checkpoint.
