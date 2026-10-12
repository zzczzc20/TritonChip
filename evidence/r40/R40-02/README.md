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

Allocation pressure checkpoint: 12 native batched calls in the real
393216-byte on-chip heap passed 2/2. All 12 waits occurred below queue
capacity, proving allocation backpressure rather than the 8-task limit.
Logical storage peaked at 196608 B; actual reserved heap peaked at 328704 B
(including retired leases/task arguments). Final reserved heap was 65536 B,
the sole input; no outstanding leases remained. Session.memory_stats exposes
actual reserved/high-water heap and code bytes per recovery epoch.
Recovery 3/3 and deferred lifetime 2/2 revalidated with this allocator.
Concurrent-stream/thread 1000-call stress passed 5/5; logical storage peaked
at 4608 B and ended at 2560 B, four buffers, zero leases/unfinished tasks.

Physical heap ledger in the 1000-call stress: actual reserved heap peaked at
9216 B and ended at 2560 B; code used/peak 5632 B, capacity 393216 B.
This replaces the earlier missing high-water observation, not the frozen
latency or replay stress acceptance. See batch_stress_physical.log.

Frozen transfer-audit checkpoint: the unchanged attention_decode tiny
point (validated against R40-02.graphs3's definition hash) passes two
resident calls and two staging calls, 4/4. Each actually launches three
kernels. Successful Session uploads/downloads involving device tensor
storage or staging shadows are intercepted without modifying payloads.
The physical copy ledger matches the launcher ledger: resident 0/0 bytes,
staging 5440/5440 bytes per call. Both policies use identical artifacts
and match the frozen FP32 reference/tolerance. Explicit output downloads
provide a positive control for the copy observer outside the graph audit.
Included in make r40-residency; the full frozen graph subset, model weight
retransfer, KV growth and latency gates remain pending.
See frozen_residency_attention_tiny.log and tools/r40/frozen_residency_tests.py.

Replay stress checkpoint: 1000 actual retained graph replays pass, 5/5
including shape/stride/storage and recovery invalidation. Replays cannot
reenter the Triton wrapper; every retained command has positive device
cycles and zero resident launch transfers. Output, unchanged live storage
and zero outstanding tasks/leases are verified after completion. The real
393216-byte heap peaks at 11264 bytes and ends at 3072 bytes; code stays
at 5312 bytes. make r40-residency now requires this full replay stress.
See replay_stress1000.log.

The frozen Llama decode tiny point also passes the copy observer's four
calls: 42 kernels per call, resident 0/0 actual tensor bytes and staging
691456/691456 bytes. Parameter storage addresses/sizes stay stable, both
policies keep identical artifacts and meet the unchanged frozen tolerance.
This repeats the fixed-cache-length point, not growing KV or generating
new tokens. Full subset and latency gates remain pending.
See frozen_residency_llama_tiny.log.
