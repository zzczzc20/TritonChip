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

Remaining Required work includes actual BO-to-PC and bank/row mapping proof,
controller/AXI/NoC timing-boundary reconciliation, read/write switching and
queue-backpressure qualification, fixed-point request/latency convergence,
fixed-point failure propagation and calibration/validation residuals.
Current convergence uses makespan agreement and reports unmatched requests;
that is insufficient to establish a stable request/latency fixed point.
Per-command Ramulator state reset remains an explicit applicability limit.
No physical-board memory prediction is qualified by this checkpoint.
