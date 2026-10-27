# P1 corrective compiler and preflight evidence

The X8 profile now selects the banked scheduling model. Later register chunks
preserve nonnegative and alignment proofs; projected load users permit safe
same-block deferral. Multiple consumers and intervening stores refuse deferral.
These corrections address the extra pointer arithmetic and materialization
observed in the failed wide-fragment hypothesis. They do not establish a board
speedup or erase prior failures.

`python3 evidence/r41/acceptance/p1/closeout.py` verifies the raw archives and
regenerates metrics/index hashes. Actual GB10 runs provide 32 bitwise SystemC
rows, eight small bitwise RTL rows and 32 bitwise sensitivity rows. The two
profile audits and the post-correction audit each compile all 113 original
points; profile/default comparisons have zero changed artifacts. Aligned wide
fragments improve relative to narrow fragments in this model, while tail cases
remain slower. Wide fragments therefore receive no universal default-on claim.

Sensitivity consists of eight registered parameter configurations of the real
SystemC executable, with identities/configurations preserved in the archives.
They are uncalibrated hypotheses. Several effects are nonmonotonic; these data
do not isolate the timing contribution of each RTL candidate. The mandatory
per-candidate sensitivity obligation remains not measured in the ledger.

Four manifests under `spec/r41/stage2/acceptance/` prepare the matched old/fixed
GEMM, software-policy leave-one-out and same-artifact 90/100 MHz comparisons.
The latter remains a joint hardware/frequency migration comparison. Each
experiment has a three-hour total bound, not three hours per step. The queue
requires the appropriate approved image already loaded and 900 seconds idle;
it never builds, programs or resets a card. All manifests passed local
preparation checks without device access. Lead measurements remain pending.

## Memory service and command queue follow-up

Fresh current-source SystemC standard Torch `(a+b)*0.5` runs pass for all three
outlet/depth settings. Completed-launch cycles are 20,250 (1/1), 13,520 (1/8),
and 13,532 (3/8). These measurements are extracted from raw reports, not entered
as an oracle. Increasing outstanding depth helps this graph; adding outlets
does not. Cache/outlet counters, actual placements and model identities remain
in `service-transfer.tgz` and `metrics.json`. This is not a physical HBM result.

The model home slice releases its lookup engine before waiting for a miss,
serializes same-line predecessors, bounds MSHRs and targets, and records hits,
merges, capacity stalls and writebacks. This explains how admission and miss
completion are separated in the existing model; it does not establish RTL
fairness or a new home-service throughput capability. The separate hbm_bw owner
retains the core-side hardware investigation; no competing image is requested.

A real artifact-executing queue-depth 1/2 model comparison passes eight bitwise
mm/addmm rows with identical output hashes. Changes in makespan are mixed and
small, not a general speedup. `TensorEngine::issue` bounds admission and retains
immutable descriptor/window ownership, but one controller holds both setup
and transfer until completion. Increasing queue capacity alone cannot overlap
setup or data transfers in this implementation. Waiting can move to the queue
without improving critical-path throughput. No queue-throughput optimization
is selected. Model observations and this source audit do not substitute for
RTL counterfactuals or the pending standard Torch board comparison.

The lead queue now additionally caps the sum of execution time across manifests
to three hours; a caller can lower but cannot increase a registered experiment
budget. Idle waits do not consume the execution budget. The final launch
commands allocate 9,000 seconds to 100 MHz and 1,800 seconds to 90 MHz.
