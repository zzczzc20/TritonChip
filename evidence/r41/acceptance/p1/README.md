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
