# R40-02 control checkpoint, 2026-10-11 — not accepted

On gx10-6646, core-vrf-sync-v5 RTL, locked torch/Triton:
`tools/r40/residency_tests.py --evidence` passed 6/6.

The standard Torch graphs produce identical TNCA artifacts under resident
and per-kernel staging. Resident launches transfer zero tensor bytes.
Staging records both original-to-shadow and shadow-to-original host byte
round trips, deduplicates backing storage and preserves strided offset
aliases and in-place updates. Unknown policy values are rejected.

The frozen chained-graph/decode audit, latency thresholds, batched/replay
submission and asynchronous stress remain unmeasured. No R40-02 gate is
claimed passing by this checkpoint.
