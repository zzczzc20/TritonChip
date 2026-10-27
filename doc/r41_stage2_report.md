# R41 Stage 2 acceptance report

Date: 2026-10-26 UTC. Sign-off: `R41_S2_100MHZ_AND_FULL_MODEL`.
Verdict: **not accepted**. The acceptance audit is complete; product acceptance
is not. Required work remains unimplemented or unmeasured, and the
pre-implementation Stage 2 freeze was missed.

The [item ledger](../evidence/r41/acceptance/stage2_index.md) judges every
mandatory Stage 2 delivery and the Definition of Done, including study
obligations without treating a negative study as a functional pass. Its
[machine-readable index](../evidence/r41/acceptance/stage2_required_index.json)
uses passed, failed, not measured and blocked. A passed row applies only to
its stated evidence identity. Aggregate final-release rows remain blocked.

## Delivery update: P8 FPGA-host FP8

P8 is closed from the lead's completed runs at `86a3d929cf3d_fp8`, retrieved
read-only and independently checked by `evidence/r41/acceptance/p8/closeout.py`.
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
is claimed. More than 65,536 barrier arm cycles and targeted persistent code
eviction/reuse are not proved by elapsed endurance time.

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
| S2-03 | Not measured | Graph-level locality consumer is a design, with no two-graph off/on board ownership/fault/traffic comparison. Final fixed-budget pipeline and resource study also absent (P2) |
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
