# Bounded home-slice and core-service study

## Scope and observations

This closes a bounded study of the existing run-1 architecture. It does not
qualify the separate hbm_bw candidate or request an image implementation.
The current-source native SystemC graph is standard Torch `(a+b)*0.5`, 4,097
FP32 elements, fixed seed 4006, with three active cores. Raw inputs, placements,
artifacts, configurations, counters and output identities are in
`service-transfer.tgz`. `closeout.py` verifies identical inputs, placements,
source, artifact text, model binary and output across settings; only active
outlets and outstanding depth may differ in the model configuration.

The 1/1, 1/8 and 3/8 completed-launch cycle observations are in `metrics.json`.
Increasing depth helps; increasing outlet count does not help this diagnostic.
These are completed-launch windows with prelaunch ownership maintenance outside
launch timing. They are not whole-call latency or measured physical bandwidth.
The earlier model/real board-top RTL service cohorts in
`evidence/r41/stage1/memory_service.md` independently show the same direction,
with different timing domains explicitly preserved. They are not pooled with
the current cohort.

## Resource and service structure

The run-1 parameters (`evidence/r41/stage2/integration/image.json`) select
32 sets, eight ways and three home slices. Each slice's data capacity is
32 × 8 × 64 bytes = 16 KiB, before tags, replacement metadata and queues.
`rtl/nuca/tnca_nuca_slice.v` separately bounds the request buffer (eight),
MSHRs (eight), targets (16) and response queue (four). Thus cache data capacity,
miss concurrency and request admission are separate resources. This is a
structural count, not a LUT/BRAM estimate.

The RTL engine has selection, lookup, hit, miss request, fill, wake and dirty
writeback states. Its age matrix prevents younger same-line requests passing
older requests. Blocked requests can yield selection to eligible requests;
returning MSHR service competes with new lookup work. The F07 cycle miter covers
partial writes, atomics, backpressure, reordered DRAM returns, dirty writeback,
metadata epochs and a high address base, including a detected mutation. Those
bounded tests do not prove unbounded starvation freedom.

The detailed SystemC slice separates lookup-engine occupancy from waiting for
miss completion and serializes same-line predecessors. It records hits, merges,
MSHR/target/set pressure, evictions and writebacks. It is intentionally not a
cycle-exact implementation of RTL engine arbitration; the previous RTL/model
residuals remain material. No claimed fairness improvement follows from it.

The existing F07 candidate snapshots the selected MSHR descriptor on an existing
edge without adding a service state. Its independent oracle/cycle miter is in
`evidence/r41/stage2/candidates/F07/functional_report.json`; its physical unit
screen is in `evidence/r41/stage1/f07_ooc/`. That screen reports +574 LUT and
+305 registers per slice and a 19-to-16-level address path reduction. It is
unrouted; the integrated and final-image evidence must be used for placement
and timing conclusions. The candidate's original README predates integration;
`APPLIED.md` and the integration image parameters establish inclusion.

## Decision and limits

Retain the existing geometry and admission bounds for this acceptance campaign.
Do not select additional outlets, MSHRs or a wider bus from this one graph.
More concurrency at an existing outlet is a useful diagnostic, not proof that
the home engine or core-side path has gained throughput. The three core ports,
three external AXI outlets and physical HBM ceiling remain distinct, as in the
corrected Stage 2 roofline report.

The command-queue comparison is separately recorded in README.md: immutable
admission is implemented, while one controller serializes setup and transfer.
No queue-only throughput optimization is selected. Per-candidate uncertainty
experiments and their limitations are recorded in candidate-sensitivity/.
Standard Torch integrated-candidate board comparisons are still pending the
lead commands in STATUS.md. No board pass is inferred from this study.

The hbm_bw owner retains the hardware redesign. Its unmerged working-tree
results are not copied into this acceptance verdict. If that work changes the
final image, the relevant studies and final acceptance measurements must be
rebound to its identity. This bounded negative/inconclusive study does not
require a competing implementation or another full run.
