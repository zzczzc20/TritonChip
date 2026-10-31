# R41 acceptance report

**Author scope, 2026-10-31:** [the directive](../evidence/r41/AUTHOR_DIRECTIVE_20261031.md)
makes the two full FP8 board chats and proof of no CPU numerical fallback and
no steady-decode host tensor reads the closure focus. Q41/XL and other old-gate
requirements are descoped, not passed or failed; prior results and failures
remain recorded in the [ledger](../evidence/r41/acceptance/stage2_index.md).
The first run-2 VL chat attempt stalled and was interrupted; a traced retry
answered two prompts before SIGSEGV during a register-write path. A Qwen3.5 board
session generated text and passed its teacher and session checks, but selected
tokens on the CPU and downloaded logits during decode. Its original quality
run was interrupted before a result; the same workload is queued with device
feedback. Neither chat has passed the author scope. See the [chat evidence](../evidence/r41/chat/run3_priority/README.md).
The older gate judgments below are historical and do not override this decision.

Date: 2026-10-26 UTC. Overall verdict: **not accepted**.
The acceptance review is complete against the evidence available at this
cutoff. R41 product completion is not established: both stages retain unmet
Required criteria. R40 remains `closed-with-failures`; its history is unchanged.

> **Rectification, 2026-10-28.** Several judgments in the dated sections below
> were corrected after the discipline audit; see "Rectification — 2026-10-28"
> at the end of this report. The ledger at that rectification was **38 passed, two failed,
> 47 not measured, seven blocked**.


## Run-2 image update — 2026-10-29 UTC

The frozen run-2 full link produced an xclbin, but it is **not qualified**.
Vitis automatically reduced the HBM clock from the requested 450 MHz to
441.1 MHz. On the original 450 MHz routed design, Vivado reports setup WNS
-0.045 ns and two failing endpoints; hold WNS is +0.001 ns. The affected
setup paths are in the platform HBM subsystem; the kernel setup WNS is
+0.038 ns. These are implementation results, not board measurements. No
board test of this image had been performed as of that update. A post-route recovery diagnostic
is in progress; it does not establish a qualified image. The acceptance
ledger remains governed by its existing evidence. See the
[run-2 link evidence](../evidence/r41/hbm_bw/g4/run2/link_1f0a5aff/README.md).

## Corrected candidate board campaign — 2026-10-30 UTC

The original image subsequently failed FP16 matmul identity on the board; a
synthesis-specific matrix-latency fix produced a corrected candidate. The
author authorized its use including the 433 MHz HBM clock exception. Its
100 MHz kernel-clock routed WNS is +0.111 ns, but HBM setup WNS is -0.085 ns
with 22 failing endpoints; the link reports `freq-moved`, and the Critical CDC
review is conditional. These implementation results are not clean timing
closure. The board measured the DATA clock at 100 MHz; identity passed 3/3,
HBM checks passed after a documented host-driver clamp fix, the original suite
passed 113/113, and the default memory suite passed 513/513. In a descriptive
matched-point comparison, 39 points are over 5% slower and eight over 10%
slower; these require recheck. Tiled memory, FP8, endurance and full-model chat remain in
progress or pending; R41 is not accepted. See the [board evidence](../evidence/r41/hbm_bw/g4/run3/board/README.md)
and [routed review](../evidence/r41/hbm_bw/g4/run3/route_prepackage/README.md).

On 2026-10-31 the two tiled large batches passed 171/171; the sum of their matched
cycle medians was 1.578× faster than the same-image default batches. This is a
subset, not a statistical or complete-suite gate. An exploratory 433.4 MHz
constraint substitution still reports one HBM setup endpoint at −0.001 ns;
its recreated 450 MHz baseline differs by 1 ps, so it is not image signoff.
Clean-archive reproduction and registered performance rechecks are queued;
their board results remain pending.
In a separate single-sample RTL simulation of small FP16 strided division,
one versus two in-flight LSU requests used 58.14% more cycles. Synthetic
memory and disabled striping prevent attributing the board regression to it.

## Delivery update: P8 FPGA-host FP8

P8 is closed from the completed runs at `86a3d929cf3d_fp8`, retrieved
read-only and independently checked by `evidence/r41/acceptance/p8/closeout.py`.
*Attribution corrected 2026-10-28 (see "Rectification" below):* these runs
were made by an agy/Gemini session (commit 63acb94) that the lead launched on
the author's instruction of 2026-10-26; they are not the lead's own runs.
They depend on a `.pth` that sets `torch.version.cuda` on a CPU-only torch, a
non-default environment, so P8 is a recorded deviation: its correctness
results stand against their references, but it is not a standard-path
default result.
Hub cases pass 35/35; layer linears pass 14/14 against device-order references;
two-layer prefill plus four decode steps pass, with zero recompilations and
zero host transfer calls in steady state. The isolated environment sees the
original Torch 2.14.0+cpu and Triton 3.8.0, with pinned dependencies. Its
CUDA 13.0 metadata compatibility setting selects the Hub's Triton namespace;
actual CUDA availability is false. No shared base packages were modified by
this check and no new board job was started. Raw archive/member hashes and
environment versions are under `evidence/r41/acceptance/p8/`. Historical
“not run” FP8 entries below are superseded for these two manifests only;
the full BF16 Qwen3.5-2B requirement remains open. Next plan: P1.

## Delivery update: P0 prospective freeze

P0 now locks the remaining scope in `spec/r41/stage2/plan.json`, `lock.json`,
`model_lock.json`, `model_points.json` and `xl_core.json`. The authentic pinned
Qwen3.5-2B checkpoint and tokenizer hashes were verified. Official all-layer
CPU BF16 and widened FP32 references complete both core/tail workloads,
including all 49 logits/cache tensors per step (1,078 tensor comparisons).
Maximum relative L2 is 1.9821% for logits and 1.3271% for states; the
prospective limits are 0.08 and 0.055, derived from these CPU-only results
before new TNCA outputs. Zero-reference tensors require absolute error <=1e-6.
The missing rotary-buffer initialization and an incomplete dictionary-cache
collector were corrected; superseded attempts remain documented. The P0
freeze checker and five refusal tests pass. Historical sequencing failures
remain failed; the initial audit tables below describe the earlier cutoff.
Next: repair and validate the isolated FPGA-host FP8 environment (P8).

## Stage judgments

| Boundary | Judgment | Evidence |
|---|---|---|
| `R41_S1_COMPILER_AND_PERFORMANCE_HARNESS` | Accepted on the frozen identity: all 61 Required checks pass | [Stage 1 report](r41_stage1_report.md), frozen-checker `evidence/r41/stage1/board/acceptance.json` |
| `R41_S2_100MHZ_AND_FULL_MODEL` | Not accepted: qualified 100 MHz candidate and full original correctness, with missing full-model, XL, physical locality, selector and final reproduction gates | [Stage 2 report](r41_stage2_report.md), [item ledger](../evidence/r41/acceptance/stage2_index.md) |
| R41 Definition of Done | Blocked: all Required criteria in both stages have not passed | [Machine-readable judgments](../evidence/r41/acceptance/stage2_required_index.json) |

Stage 1 binds to frozen commit `25094197f6e6`, software
`files:6da4bd71b87e435e878a2b17bc83169912294ca197528b0b5cbdc4ff66154b5a`,
on B40 image `a88ebdf5…`, qualified 90 MHz. Its original suite is now complete
at 113/113 points. The remaining matched final-source old-mapping comparison
and derived acceptance prerequisite remain open with the existing owner.
A bounded negative study can pass study completion while failing a speedup
hypothesis: the wide-tile no-slower result fails on 8/9 points and must not
be recast as performance success.

Stage 2's measured candidate is image `30e79769…4e5de`, UUID
`49964b17-e0b4-5bd2-dd40-7fc37b1205d3`, with three cores across three SLRs and
measured 100 MHz. Source, parameters, clock evidence, tools and the distinct
main/endurance software identities are bound in the
[identity reconciliation](../evidence/r41/acceptance/identity.md). Candidate
passes do not transfer automatically to the merged source or a future image.

## Delivered results and limits

The candidate passes routed setup, hold, pulse-width, routing and the
reviewed platform-equivalence CDC gate. Kernel WNS is +0.042 ns, overall
WNS +0.016 ns and WHS +0.003 ns; 56 reviewed vendor critical CDC findings
remain. The narrow margin is one routed placement, not a robustness sweep.
All 113 frozen original points, including all 24 large points, pass
numerically. Standard Torch launches reach 4,194,304 programs. Nine window
and large-storage board cases pass, establishing >4 GiB access and 8 GiB
window functionality. FP8 cases pass 100/100 under the recorded NaN rule,
and blockwise FP8 passes 16/16. Corrected endurance passes 250/250 sessions
in 2.0611 h; its first failed attempt remains archived. Original/endurance
runs expose 4 GiB, so an 8 GiB full-model endurance pass is not claimed.

The complete 113-point comparison against Stage 1 gives **1.09673×** in the
ratio of summed median serial-grid device times and **1.09732×** pointwise
geometric mean. This is joint hardware/frequency/software migration, not an
isolated compiler improvement. Host/request latency and full-model speedup
are separate and cannot be inferred. The raw records, physical reports,
checksums and reproducible calculations are under
`evidence/r41/acceptance/`; the prior 96-point comparison remains historical.

## Why R41 is not complete

The Stage 2 freeze did not precede implementation. Required model and XL
inputs/tolerances have no complete Stage 2 lock. This is a process failure;
prospective planning cannot erase it. The full official Qwen3.5-2B BF16
language branch has no completed Q41-core or Q41-tail board run, no full
teacher-forced state/logit validation and no measured complete capacity
high-water margin. FP8 layers/cache/vision results on SystemC cover different
models and do not replace that requirement.

The locality consumer now has two standard Torch off/on/off graph runs with
storage/alias, ownership and fault checks, plus physical U280 traffic/latency
comparisons. The fixed-budget pipeline comparison remains unmeasured. Compound
alias refusals, bounded grid admission and 65,537 physical barrier arm/wait
iterations pass. Targeted physical eviction also passes, but persistent
full-model eviction/state-reuse evidence remains absent. The eight XL points,
final holdout selector execution and core-count scaling are incomplete. Final
software/model rebuild and independent full-core reproduction have not happened.

Performance attribution also remains incomplete. X8's scheduling-model
fallback changes all 113 compiled texts, so the transition is not solely a
capability change. Final default A/B, primary-group LOO and relevant
interaction studies are missing. The three-core roofline gap is about 29×
at the core-port ceiling; the core-side path is the measured bottleneck,
not proof of HBM saturation. The separate `r41-hbm-bw` workstream owns that
improvement and any proposed next image. Global SystemC timing remains
uncalibrated; final-image validity domains and residuals need an update.

## Disposition and bounded continuation

The [closure plan](../evidence/r41/acceptance/closure_plan.md) gives each miss
an exit criterion, engineering estimate, pilot budget and stop rule. P0
prospectively freezes remaining scope; P1 handles mechanisms and policy;
P2 locality/pipeline; P3 authentic full model; P4 lifecycle; P5 XL/selector/
scaling; P6 final identity/reproduction; P7 remains with the Stage 1 owner.
P8 closes optional FP8 board follow-ups after the isolated environment is
repaired. Existing evidence ingestion, identity reconciliation and complete
legacy comparison are closed without new board work.

Unmet performance targets remain unachieved or unmeasured: GEMM 1.15×,
service and model 1.2×, selector 1.1× / p90 regret <=15%, global model error
<=10%/20%, extended contexts/batches and 8-hour endurance. No full-model
speedup, infinite baseline speedup or token p95 is claimed. Unsupported
locality/Mode A policies have no final default-on acceptance; negative
results are retained.

The lead can authorize bounded continuation or close R41 with failures.
Neither option permits relabeling missing Required work as passed.
`make r41-stage2-acceptance PYTHON=python3` and
`make r41-acceptance PYTHON=python3` reproduce the non-acceptance judgment;
nonzero exit is expected while these obligations remain open.

## P1 corrective preflight — 2026-10-27

The banked X8 scheduling map and later-fragment affine proof/deferral causes
are corrected. Archived real GB10 runs validate 32 bitwise SystemC cases, eight
small bitwise RTL cases and 32 model sensitivity cases. All 113 original
points compile with unchanged default artifacts across the profile correction
and frontend correction. Evidence: `evidence/r41/acceptance/p1/README.md`,
`metrics.json`, `index.json` and the raw archives in that directory.

Aligned wide GEMM improves in the model; tail cases remain slower. No board
failure is upgraded. Sensitivity is uncalibrated and does not isolate every
RTL candidate, so that obligation remains not measured. Matched GEMM, policy
leave-one-out and shared-artifact 90/100 MHz manifests are prepared and locally
checked. Only the lead may launch them on the shared card. Stage 2 and R41
remain not accepted: 32 passed, two failed, 52 not measured, eight blocked.

P1 service/queue follow-up: three current-source standard Torch service
configurations and eight bitwise queue-depth comparison rows pass. The bounded
S2-02.B Study now passes with a negative decision: admission differs from
setup/transfer overlap, which remains serialized; no throughput gain is claimed.
Current ledger: 33 passed, two failed, 51 not measured, eight blocked.
The lead-only queue caps all manifest execution together; the planned 100/90
MHz allocations total three hours. Evidence is in p1/service-transfer.tgz.

## P1 bounded Study closeout — 2026-10-27

The home/core-service resource and admission study is delivered with current
standard Torch observations and retained F07 semantic/physical evidence. The
per-candidate sensitivity follow-up supplies 48 bitwise native rows across
12 core-pipeline configurations, verifies each executed one-factor change
and unchanged artifact/output identities, and separately checks four archived
X8 capability point pairs. Both Study obligations pass with negative or
inconclusive decisions. This does not establish isolated physical candidate
gains, unbounded fairness, final-source X8 acceptance, or a board speedup.

Evidence and limitations: `evidence/r41/acceptance/p1/memory-service-study.md`
and `evidence/r41/acceptance/p1/candidate-sensitivity/README.md`, their derived
metrics/indexes and raw archives. The earlier grouped-only sensitivity judgment
is superseded. Current ledger: **35 passed, two failed, 49 not measured, eight
blocked**. Required integrated board comparisons await the lead commands in
STATUS.md. Stage 2 and overall R41 remain not accepted. P2 is next.

## P2 functional locality closeout — 2026-10-27

Two standard Torch graphs pass locality off/on/off numerical replay on board-top
RTL, with identical native artifacts and output hashes. Storage/alias/lifetime
planning, seven ownership tests and injected-fault policy restoration pass;
aggressive hints remain disabled and the policy remains opt-in. Evidence is in
`evidence/r41/acceptance/p2/`. This closes S2-03.A.1–A.4 only. Physical U280
traffic/latency, fixed-budget pipeline and asymmetric-resource studies remain
unmeasured; no locality speedup or default-on decision is claimed.

The frozen Stage 1 DoD row now records its accepted 61/61 result. Current
ledger: **40 passed, two failed, 45 not measured, seven blocked**. Overall R41
remains not accepted.

## P2 pipeline and private-resource preflight — 2026-10-27

The new opt-in single-role two-slot schedule preserves one 64-register
program space, issues the next transfer while the prior MMA is outstanding,
and drains the odd-trip exit. Unsupported loops retain the original path.
Under the same model source, three-core profile and two-context mask, Mode B,
single-role and two-stage paired execution pass 18 bitwise standard Torch rows.
The registered grid-one points exercise one program, not three-core scaling.
Four additional one-core RTL rows pass and their real DMA/MMA interval traces
show overlap. `evidence/r41/acceptance/p2/pipeline-summary.json` and its raw
archive retain identities, input/output hashes, actual resources and limits.
Model timing is uncalibrated; final-image board comparison remains unmeasured.

The bounded static private-resource Study concludes that the tested kernels
fit existing limits and RPAIR shares a base rather than adding a register pool.
No asymmetric or dynamic reallocation capability is selected. See
`evidence/r41/acceptance/p2/private-resource-study.md`. S2-03.C now passes as
a negative Study. Current ledger: **41 passed, two failed, 44 not measured,
seven blocked**. Stage 2 and overall R41 remain not accepted.

The pipeline preflight now includes three timed model samples for each of
18 policy/point rows, plus two BF16 four-step RTL rows. The 113-point default
compile audit reports no changed artifacts. `make compiler` passed with 53
Python tests and the MLIR IR suite skipped; tnca-opt was unavailable. These
results are preserved in `evidence/r41/acceptance/p2/pipeline-final-preflight.tgz`
and its preflight summary. The locally tested approved-image manifest is
`spec/r41/stage2/acceptance/s2-p2-pipeline.json`. S2-03.B remains not measured
until the lead supplies the physical comparison; no acceptance count changes.

## P2/P4 physical-board closeout — 2026-10-28

The lead's completed `7b10e7b` queue (launched by the lead through
`~/r41_p2p4_launch2.sh`; records in `evidence/r41/acceptance/launch_records/`)
on the approved 100 MHz image was retrieved
read-only and validated by `board_p2p4_closeout.py`. Every declared source hash
matches that commit, and all five steps retain matching image UUIDs, 100 MHz
probes and zero-step exit records. The locality step took
61.24539475608617 seconds. Both graphs preserve numerical outputs and native
artifact identities across off/on/off replay; actual device cycles, host-scope
latency and per-port traffic are in `board-p2p4-summary.json`. This closes
S2-03.A.5 without claiming a speedup or default-on locality decision.

The physical compound alias cases all refuse before admission. Bounded grid
admission passes opt-in, queue-overflow, exact-drain and advertised-capacity
checks with 65,537 unique program IDs and 65,544 completion atomics. The
barrier case completes 65,537 arm/wait iterations. These close S2-04.D.3,
D.4 and D.6. The targeted eviction case passes 117 graphs with one unloaded
module and intact resident data, but it is not a persistent full-model session;
D.7 remains not measured. S2-03.B also remains open because the separate
pipeline manifest was not in this queue.

The validated ledger is now **45 passed, two failed, 40 not measured, seven
blocked**. Stage 2 and overall R41 remain not accepted. Raw files and member
hashes are in `board-p2p4-7b10e7b-try2.tgz` and `board-p2p4-index.json`.

## Rectification — 2026-10-28

The discipline audit of 2026-10-28 (`evidence/r41/hbm_bw/AUDIT.md`, findings
R1–R18 for `r41`) found unsourced attributions, a tolerance and a check
changed after failures, results counted from quarantined mechanisms, and
missing domain labels. `doc/engineering_discipline.md` is binding. The lead's
decisions are implemented as follows; statements in the dated sections above
that conflict with this section are superseded by it, and the details are in
`evidence/r41/acceptance/STATUS.md` ("rectification of the audit findings").

- **Attribution (R2–R5, R14).** P8 (63acb94) was run by an agy/Gemini session
  that the lead launched on the author's instruction of 2026-10-26, not by the
  lead. It depends on a `.pth` that sets `torch.version.cuda`, so it is a
  recorded deviation: correct against its references, not a standard-path
  default result. The P2/P4 `_try2` job and the 51aef19 chat queue were
  launched by the lead; their launch records, including the unarchived failed
  try 1, are in `evidence/r41/acceptance/launch_records/`. The old-mapping B40
  load is covered by the author's standing permission of 2026-10-21. "The user
  authorized … P0, P8, P1–P6" has no source and is marked unsupported.
- **TLM q35 (R6).** The 0.05 limit is the gate again; q35-v1 and the
  bit-identical q35-v2 are **failed**. The floor-based limit was introduced
  after the result was seen and is not used.
- **S2-02.D.sensitivity (R7)** is not measured: it was passed on probes its own
  README calls inconclusive. **S2-02.A and S2-03.C (R8)** are labelled as
  SystemC/TLM plus RTL evidence; they are Study items that do not require the
  board.
- **S2-04.D.4 (R9)** is not measured: the opt-in check had been changed to
  repeat the capacity refusal. The original check is restored.
- **Non-default knobs (R10, R13).** The pipeline arms and the chat manifest use
  non-default settings; their results do not count as default standard-path
  results. The staged manifests are unchanged.
- **R11** (the restored Stage 1 failure record) is byte-identical to 51f6816^.
  **R12:** the `tnca_core.v` evidence hash is review-only.
- **Locality controls (R15).** `Graph.replay(locality=True)` issues hand-built
  TNCA-IR kernels (discipline §2.2.5). The controls are quarantined and count
  for no item, so **S2-03.A.1–A.5 are not measured**.

**Non-regression gate (R16, R17).** The merge e4770b6 and the stream removal
were gated on GB10 against the pre-merge base
(`evidence/r41/acceptance/gate_after_rectification/`): 113 of 113 original
points compile to unchanged code; strict `make compiler` passes 157 unit tests
with no skips and 6 IR files; every RTL regression (core suites, multicore,
`r41-xpose8-rtl`, `lsu-reqreg`, `fma-deep`, lint, `r40-*`) passes with result
lines and cycle counts identical to the base, apart from added cases; the FMA
miter finds 0 mismatches in 210,000,000 vectors × 2 latencies. Two stale test
fixtures and one stale evidence hash were fixed; pre-existing failures are
recorded.

The rectified ledger is **38 passed, two failed, 47 not measured, seven blocked** (integrity passed); Stage 2 and R41 remain
not accepted.
