# R41 Stage 2 acceptance report

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
feedback. The graph-preserving VL retry timed out after four text prompts,
three passed and the registered reason prompt failed. A full-checkpoint CPU
diagnostic reproduced the failure with a one-token prefill tail; grouping that
tail passed all six VL and five Qwen3.5 CPU quality prompts. These are CPU
diagnostics, not board acceptance. The grouped-tail VL board retry is running,
with Qwen3.5 queued. Neither chat has passed the author scope. See the [chat evidence](../evidence/r41/chat/run3_priority/README.md).
The older gate judgments below are historical and do not override this decision.

Date: 2026-10-26 UTC. Sign-off: `R41_S2_100MHZ_AND_FULL_MODEL`.
Verdict: **not accepted**. The acceptance audit is complete; product acceptance
is not. Required work remains unimplemented or unmeasured, and the
pre-implementation Stage 2 freeze was missed.

> **Rectification, 2026-10-28.** Several judgments in the dated sections below
> were corrected after the discipline audit; see "Rectification — 2026-10-28"
> at the end of this report. The ledger at that rectification was **38 passed, two failed,
> 47 not measured, seven blocked**.

The [item ledger](../evidence/r41/acceptance/stage2_index.md) judges every
mandatory Stage 2 delivery and the Definition of Done, including study
obligations without treating a negative study as a functional pass. Its
[machine-readable index](../evidence/r41/acceptance/stage2_required_index.json)
uses passed, failed, not measured and blocked. A passed row applies only to
its stated evidence identity. Aggregate final-release rows remain blocked.


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

## What is established

The run-1 image, SHA256
`30e79769c128668ac8f1d0d9017f6ea8d22b8335b8fee18cbe374f4bdef4e5de`,
UUID `49964b17-e0b4-5bd2-dd40-7fc37b1205d3`, operates at measured 100 MHz.
Its three cores occupy SLR0, SLR1 and SLR2. Routed kernel WNS is +0.042 ns;
overall WNS +0.016 ns, WHS +0.003 ns and WPWS 0, with no failing endpoints.
All 1,153,083 routable nets are fully routed, with zero routing errors.
The CDC gate passes by equivalence to the reviewed platform findings:
56 vendor critical findings remain; this is not a zero-critical-CDC claim.
Timing margin is narrow and comes from one placement. Source, parameters,
toolchain, measured clock and campaign software are reconciled in the
[identity record](../evidence/r41/acceptance/identity.md).

The raw evidence was retrieved through file reads only. No card access,
programming, reset, new board campaign or full implementation was performed
by this audit. `physical-audit.tgz` retains the actual routed audit and loaded
clock capture. `board-records.tgz` retains the main campaign JSON/JSONL,
including the failed first endurance attempt; member hashes and provenance
are in `board_index.json`. Large executable bundles remain on the host.

| Measurement | Judgment and scope |
|---|---|
| Original suite | Passed: 113/113 frozen points, including all 24 large points, matching point hashes, numerical tolerances, no fallback and qualified 100 MHz board records |
| Large standard Torch grid | Passed: up to 4,194,304 programs, beyond the required 65,535 boundary |
| FP8 cases | Passed: 100/100 cases; six plumbing comparisons permit NaN-sign encoding differences under the recorded rule |
| Blockwise FP8 | Passed: 16/16 cases with zero bit mismatches |
| 8 GiB window | Passed: 7 cases / 159 checks, plus 2 large-storage cases / 14 checks; high bits, bounds, aliases and reserved-page behavior covered |
| Mixed endurance | Passed on corrected second attempt: 250/250 sessions over 2.0611 h, with session and host resource logs; first attempt failed due to preserved harness defects |
| FP8 Hub and Qwen layer board campaigns | Not measured: environment preparation failed before these manifests ran |

Original-suite and endurance campaigns expose 4 GiB; separate high-address
and FP8 cases expose 8 GiB. No 8 GiB persistent full-model endurance result
is claimed. A separate physical run proves 65,537 barrier arm/wait iterations
and targeted eviction; persistent full-model code eviction/reuse remains open.

## Performance and causal limits

`python3 evidence/r41/acceptance/closeout.py` revalidates archived original
records using the existing board-record refusal rules and recomputes the
complete comparison against the now-completed Stage 1 suite. Across all
113 matched points, the ratio of summed median serial-grid device times is
**1.09673×**; the pointwise geometric mean is **1.09732×**. The old 96-point
comparison in the campaign results remains historical. These are descriptive
device-time results for a joint hardware, frequency and software migration.
They are not iso-frequency compiler or memory-service gains and do not
measure request latency or full-model throughput. Host wall-time gains are
not inferred from device cycles.

The original campaign records 202 runs, including qualification and repeated
timed samples. The locally checked `formal.jsonl` contains 113 accepted
formal records; these counts refer to different populations. Tiny-point
single-sample differences do not establish individual performance effects.

The [roofline analysis](../evidence/r41/stage2/roofline/README.md) places TNCA
about 29× below its three-core core-port ceiling, rather than the earlier
one-core 9.7× figure. The bottleneck is the core-side service path, not proof
of exhausted HBM bandwidth. The separate bandwidth workstream owns that
investigation and any next image. It does not retroactively improve run 1.

The integration audit found that X8's target name falls back to the
`flipflop` scheduling model instead of the corresponding `banked2` model,
changing scheduled code for all 113 points. The changed profile therefore
cannot be described as a capability-only change. Final default-mechanism
A/B, LOO and interaction evidence is incomplete. F06's +4 cycles per grid
and F10's isolated +1 cycle per access are documented costs, not independent
speedup measurements. Global SystemC timing remains uncalibrated; semantic
FP8 execution does not qualify final-image timing prediction.

## Unmet requirements and bottlenecks

| Package | Acceptance judgment | Bottleneck and closure |
|---|---|---|
| S2-00 | Failed process sequencing; freeze fields not measured | No Stage 2 plan/lock, model tolerance lock or XL manifest preceded implementation. Freeze remaining work prospectively; preserve the process failure (P0) |
| S2-01 | Eight numbered candidate checks passed | Candidate identity reconciled; final B41-F still unselected. A future image needs its own affected qualification and measurements (P6) |
| S2-02 | RTL MMA/FMA candidate and physical screens evidenced; study/comparison gaps remain | Complete SystemC sensitivity and integrated-mechanism board comparisons; coordinate bandwidth work and address scheduling-model confounder (P1) |
| S2-03 | Functional locality and physical traffic/latency passed; pipeline comparison not measured | Two standard Torch graphs pass physical off/on/off numerical, traffic and latency comparisons. The final fixed-budget pipeline comparison remains absent (P2) |
| S2-04 | Full model blocked; capacity and lifecycle partially passed | No authentic all-layer BF16 Qwen3.5-2B core/tail board inference. Actual full-model peak, state/logit evolution, persistent lifecycle and steady-state host traffic remain unmeasured (P3/P4) |
| S2-05 | Original suite passed; XL/selector/scaling incomplete | Eight XL points lack a freeze and complete execution; sealed holdout lacks final selector campaign; no final-image 1/2/3-core model comparison (P5) |
| S2-06 | Blocked | Final identity, complete functional gates, model-domain report, clean software/model rebuild and independent full-core reproduction remain absent (P6) |

FP8 software evidence includes Qwen3-VL-2B-FP8 and Qwen3.5-4B-FP8 layers,
decode/cache checks and a vision tower on SystemC. Those are different
checkpoints, precisions and execution scopes from the required official
Qwen3.5-2B BF16 full language branch. They cannot close Q41-core or Q41-tail.
The 8 GiB functionality pass removes a window obstacle but does not establish
`usable >= 1.25 × actual peak` for a model that has not run.

## Decisions, targets and next work

The [bounded closure plan](../evidence/r41/acceptance/closure_plan.md) assigns
P0–P8 to every unmet item, with engineering estimates, pilot limits, stop
rules, dependencies and board/image/user-decision needs. Documentation,
identity reconciliation, raw ingestion, the full comparison and the
acceptance checker are closed in this audit. New functional implementation
and board campaigns are explicitly proposed work, not completed delivery.

Unmet or unmeasured Targets remain visible: 8-hour endurance; repaired GEMM
1.15×, memory-service 1.2×, full-model prefill/decode 1.2×, final holdout 1.1×
and p90 regret <=15%, global model median/p95 error <=10%/20%, and extended
Q41-512/2k/8k/b4 workloads. No full-model throughput or reliable token p95 is
available. Stage 1's wide-tile no-slower hypothesis failed on 8/9 points;
that negative result is preserved. Mode A and locality have no evidenced
final default-on decision. No dynamic register-capacity feature is claimed.

One of five authorized full implementation runs is documented used at this
cutoff. Remaining allowances do not authorize this audit to launch a run.
The lead must choose continuation budgets or close Stage 2 with failures;
the missing Required items cannot become Targets during closeout.

## Reproduction of this judgment

Run `python3 evidence/r41/acceptance/closeout.py`, then
`make r41-stage2-acceptance PYTHON=python3`. The latter deliberately exits
nonzero while mandatory rows are not passed. It verifies evidence hashes
and index coverage, and never upgrades an author's judgment merely because
a command returned zero. `make r41-acceptance PYTHON=python3` also includes
the frozen Stage 1 result and Definition of Done. These commands reproduce
the audit; they do not constitute independent product reproduction.

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

The native graph consumer now executes residual and projection standard Torch
graphs with locality off/on/off on the board-top MMIO RTL binding. Both preserve
captured Inductor artifact identities and output hashes, with no host upload or
download during replay. Plans follow actual Session storage handles and record
aliases and captured lifetimes; aggressive dead/bypass hints remain disabled.
Seven real-Session ownership/alias tests pass. Injected numerical admission
faults after real locality registration propagate, restore policy zero and make
subsequent off replay refuse the invalid graph. Raw logs and the validated
comparison ledger are in `evidence/r41/acceptance/p2/`.

S2-03.A.1–A.4 pass for this functional scope. S2-03.A.5 remains not measured:
no physical U280 locality traffic/latency run has occurred and no speedup is
claimed. The fixed-budget pipeline and asymmetric-resource studies also remain
open. An approved-image board manifest is prepared but not launched.

The frozen Stage 1 DoD row is reconciled to its accepted 61/61 result. Current
ledger: **40 passed, two failed, 45 not measured, seven blocked**. Stage 2 and
overall R41 remain not accepted.

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
