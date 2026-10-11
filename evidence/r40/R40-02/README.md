# R40-02 control checkpoint, 2026-10-11 — not accepted

On gx10-6646, core-vrf-sync-v5 RTL, locked torch/Triton:
`tools/r40/residency_tests.py --evidence` passed 6/6.

The standard Torch graphs produce identical TNCA artifacts under resident
and per-kernel staging. Resident launches transfer zero tensor bytes.
Staging records both original-to-shadow and shadow-to-original host byte
round trips, deduplicates backing storage and preserves strided offset
aliases and in-place updates. Unknown policy values are rejected.

The next checkpoint adds bounded asynchronous submission with dependency
failure propagation, actual completion timing, logical streams, deferred
storage recycling and allocation tokens independent of reused addresses.
A held RTL worker tests retirement while dependent tasks still own leases;
three forced recovery epochs test stale tensor/event access and destructors.
Transport payload/request/reply sequences are serialized per backend.

Validation: lifetime 2/2, recovery 3/3, payload concurrency/mutation 2/2,
original runtime lifetime 7/7, runtime batch 7/7, backend regression 3/3,
native device regression 16/16, native resident/staging control 6/6,
queued graph/stream/concurrent logging and 10-call smoke 5/5.

Guarded graph replay passed 4/4: a standard compiled multi-kernel graph
retains intermediate storage, replays unchanged artifacts after explicit
input updates, and rejects changed shape/stride/storage alias or recovery
epoch before any admission. Invalid graphs require reset and recapture.
The concurrent batch/stream regression with this change passed 5/5 (10 calls).

Native batched 1000-call stress passed 4/4 with the allocation token and
stream changes: correct final output, bounded unfinished queue, unchanged
live storage total and no outstanding leases at the end. The separate
concurrent-thread smoke passed 5/5.

The frozen chained-graph/decode audit, latency thresholds, replay stress
and explicit high-water ledger remain unmeasured. No R40-02 gate is
claimed passing by this checkpoint.
