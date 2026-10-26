# Native cold-call audit checkpoint — not accepted

Locked Torch FakeTensor constant propagation clones device literals during
compilation. The first native add call executes 27 kernels; the next calls
execute only the generated graph's one kernel. The real-device negative
case adds a second physical graph: correct output does not bypass refusal.
See native3.log (3/3).

The ArchBench cumsum smoke passes on native core-vrf-sync-v2 RTL with two
warmup calls and three timed samples, [2768,2768,2768] cycles. It records
one primary graph launch and the complete 27-launch cold call separately;
CPU fallback is zero and the FP32 reference matches. Source/file provenance
and both bundle paths remain in cumsum_rtl.jsonl. Protocol checks pass 34/34.
This smoke is not a frozen performance comparison or R40-01 acceptance.

Only verified steady samples contribute to the timed median/p95. Every
warmup and timed call still requires the exact generated wrapper count,
device time, and numerical correctness. A deterministic cold-only run
still refuses a count mismatch. Raw startup work and its cost are retained.
The full 77-point initial board run on software 576b310/ac580b0 was numerically
correct but not-measured under the old cold-count check; it will be rerun
with this audit. See doc/r40_cold_call_audit.md.
