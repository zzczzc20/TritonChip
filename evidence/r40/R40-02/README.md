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

2026-10-12 frozen inventory runner checkpoint: the independent child
audit on /hbm passes 4/4 for the frozen tiny attention_decode point,
with actual resident 0/0 B versus staging 5440/5440 B on each call.
The report explicitly says full_inventory=false; see frozen_suite_smoke_hbm.*.
The runner rejects unknown points and nonempty output directories,
retains per-point logs/checkpoints, and kills simulator descendants on
timeout. New reports hash software, audit tools and raw logs; children
verify the parent software digest before running.

The complete unchanged 24-point inventory is running with two workers
on gx10-6646, under sim/build/r40/frozen_residency_full24_20261012.
That initial run has its source digest in source_manifest.json. It is
pending, not passed; repeated fixed-cache audits do not establish KV
growth, copy overlap or any latency threshold. The public full entry
is make r40-frozen-residency; choose a fresh R40_RESIDENCY_OUT for reruns.

2026-10-12 native copy checkpoint: same-device Tensor.copy_ now executes
a general Triton transfer kernel, including broadcast, dtype conversion
and strided destinations; raw same-dtype copies preserve bytes. Disjoint
views of one storage use a native intermediate; overlapping copies are
refused before writing. The active Triton driver is restored after each
copy. Allocator resize also copies its prefix on the device using a
profile/epoch-bound byte-copy module. No model-specific code is used.

Actual Session transfer audits pass 15/15 on CUDA-visible and CUDA-hidden
runs, covering 10 dtypes, stride gaps, broadcasting, conversion, alias
views and storage resize, with zero tensor host transfers. The complete
CUDA-visible device regression after the Tensor.copy_ change passes 16/16.
See native_copy15_resize*.log and native_device_after_copy_cuda.log.
Growing decode remains under validation; this does not close latency,
copy-overlap, replay or KV-growth gates. Copy indexing currently explicitly
refuses byte counts or stride extents requiring 64-bit indexing.

2026-10-12 growing KV checkpoint: resident_growing_decode3.log passes
3/3 guarded standard fullgraph calls, KV length 16 -> 19, 45 actual
kernels per steady call, fixed cache addresses, unchanged weight storage
and zero tensor/staging Session transfers. Compile initialization and
explicit correctness downloads are outside the steady audit. This is
correctness evidence, not online attention or latency acceptance.
The complete 64-step run is pending. make r40-decode-residency defaults
to 64 steps and records actual memory high water at completion.

The first full frozen transfer audit completed 10/24; all 14 failures
were the default 600-second host wait. Retain its report in
frozen_residency_full24_600s.json. The isolated runner now exposes and
records --host-timeout (default 172800s), independently of its child
process timeout and the unchanged device cycle deadline. Full rerun
is pending; no failed record has been replaced with a pass.
