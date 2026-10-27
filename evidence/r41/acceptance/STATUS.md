# R41 acceptance status

2026-10-26 UTC — milestone 1: acceptance ownership established at origin/r41
691d0dd in the dedicated r41-accept worktree. Read HANDOFF.md and SERVERS.md.
No card access, board job, or full image build has been started.

Stage 2 and R41 are not accepted. Evidence inventory confirms the 100 MHz
run-1 image, original-suite results, 8 GiB cases and corrected endurance result.
Full Q41-core/tail, XL-core, locality comparisons and independent reproduction
still need explicit judgments. Raw board evidence is being collected read-only.
Stage 1 closeout and the r41-hbm-bw workstream remain with their existing owners.

Milestone 2: raw board JSON/JSONL and physical audit collected without device
access. `python3 evidence/r41/acceptance/closeout.py` validates the 113 frozen
original point hashes (including all 24 large points), qualified image/clock,
source check, 100 FP8 and 16 blockwise cases, nine high-address cases and the
corrected endurance record. Archive/member hashes are in board_index.json.
The complete 113-point Stage 1 comparison is now available: ratio of summed
median device times 1.09673x; pointwise geometric mean 1.09732x. This is a joint
hardware/frequency/software migration comparison, not a compiler-only gain.
The old 96-point comparison remains historical evidence.

Milestone 3: all 94 Stage 2/DoD and supporting study/common obligations judged:
24 passed, 2 failed, 60 not measured, 8 blocked. Counts include aggregate and
study rows; they are not 94 independent functional tests. Eight numbered
S2-01 candidate checks pass. The >65,535-program requirement is closed from
existing standard Torch launch maps (maximum 4,194,304 programs).

Both requested reports are written in professional English. Stage 2 and R41
remain not accepted. Frozen Stage 1 snapshot remains 59/61 passes, one not
measured and one blocked; its owner retains the remaining comparison.
P0–P8 in closure_plan.md give bounded work, dependencies and resource decisions.
No board run or full image run was started. No new remote branch was created.

Validation: closeout.py passed raw board/physical/endurance checks and the
complete 113-point comparison; six Stage 2 checker refusal tests passed;
`python3 -m tools.docmaint check` passed (85 documents, acyclic).
`make r41-stage2-acceptance PYTHON=python3` and
`make r41-acceptance PYTHON=python3` both reject acceptance as expected, with
zero evidence-integrity errors. Overall target uses the archived checker
judgment of the frozen Stage 1 tree; it does not rerun that product in this
merged acceptance worktree. judgment.json preserves the aggregate result.

Final review retained the illegal-alias negative-test gate as not measured:
high-bit anti-alias tests pass but do not establish overlapping-storage
ownership refusal. The compound requirement is not upgraded from partial evidence.

## Delivery continuation — P0 in progress

The user authorized implementation on 2026-10-26, in order P0, P8, P1–P6.
No board access or image builds are authorized for this agent. The lead starts
all prepared waiting jobs. Prospective core/tail inputs, all eight XL definitions,
candidate list, baseline boundaries and budgets are drafted in spec/r41/stage2.
They are not yet claimed frozen: model_lock/lock await independent CPU numeric
cross-validation of authentic Qwen3.5-2B checkpoint values. Historical process
failures remain failures.

GB10 units r41-accept-p0-fetch (2 GiB) and r41-accept-p0-reference (14 GiB,
waiting for download then BF16/FP32 sequentially) use CPUs 0–5, no swap;
combined maximum 16 GiB. Reference unit never opens TNCA or the card.
Next step: inspect fetch/reference logs, resolve any reference failures,
freeze numerical thresholds and identities only after independent validation,
then run the Stage 2 freeze checker and proceed to P8.

## P0 milestone — prospective freeze delivered

Authentic checkpoint SHA256 aa33250c…ed680b1 and tokenizer SHA256
5f9e4d49…b81cb42 verified. Official 24-layer BF16/FP32 CPU references complete
Q41-core and Q41-tail; all 49 logits/cache tensors at every step compared
(1,078 comparisons). Maximum rel-L2: logits 0.0198213, states 0.0132705.
Limits locked prospectively at 0.08 / 0.055 before new TNCA outputs.
Reference metadata, failed/superseded attempt logs and member hashes are
in p0/reference-evidence.tgz and reference-index.json. Tensor files remain on
GB10 at ~/r41-accept-p0/{bf16,fp32}-v5 with hashes in the archived reports.

Local Stage 2 freeze verification and five refusal tests pass. Eight freeze
field judgments now pass for prospective scope; two historical process
failures remain failed and need an explicit lead disposition because the
first implementation cannot be made to occur after this freeze retroactively.
No contract requirement or historical record was rewritten.

Next step: P8, repair ~/r41s2_fp8env on acf3019 using a .pth pointing at
~/tc_env/torchenv site-packages and exact Torch 2.14.0 constraints; perform
imports/version checks only, then prepare lead-started board commands.

## P8 milestone — environment verified; lead board results ingested

After fetching origin, found the lead's completed P8 repair and card campaign
in commit 63acb94. This agent did not start it or duplicate it. Read-only raw
collection and p8/closeout.py validate all FPGA image/clock probes and actual
point results: Hub 35/35; linears 14/14; two-layer prefill + four decode steps
pass; steady state has zero new graphs and zero host read/write calls.
Environment check confirms Torch 2.14.0+cpu and Triton 3.8.0 imported from
~/tc_env/torchenv, all pinned dependency versions and real CUDA unavailable.
The process-local CUDA metadata setting selects the Hub Triton namespace; it
is not a CUDA hardware claim. Environment and board archive hashes are in p8/.

Prepared tools/performance_harness/launches/acceptance_queue.py for subsequent
lead-started measurements. It never programs/resets the card, requires the
approved image already loaded, and waits 900 elapsed idle seconds, resetting
on contexts, outstanding execs, clients or unreadable counters. Six fake-card
tests pass; thirteen experiment-runner tests pass. Dry-run validates both
P8 manifests without sysfs access. Current P8 needs no new launch. If affected
software changes later require a rerun, the lead command on acf3019 is:

```sh
cd ~/r41_accept_SOURCE_COMMIT
setsid nohup ~/tc_env/torchenv/bin/python tools/performance_harness/launches/acceptance_queue.py --root "$PWD" --out ~/r41_accept_p8_SOURCE_COMMIT --python ~/tc_env/torchenv/bin/python --manifest spec/r41/stage2/board/s2-fp8-hub.json --manifest spec/r41/stage2/board/s2-qwen-fp8.json > ~/r41_accept_p8_SOURCE_COMMIT.log 2>&1 < /dev/null &
```

SOURCE_COMMIT must be the staged, immutable archive's real commit; the final
P6 status will give its concrete value after software changes stop. This is a
prepared command, never launched by this agent. Next: P1 scheduler mapping,
GEMM policy/root-cause analysis and causal remeasurement manifests.

## P1 in progress — measured cause isolated

X8 scheduling-model map corrected to banked2; two mapping tests pass.
The two 113-point profile compile audits completed on GB10 (both 113/113);
comparison and archive ingestion follow. A fresh SystemC build and initial
FP16 N16/N32 mm/addmm runs pass bitwise. N32 remains 2.50–3.50x slower in
these model cases, with 136 pointer operations inside the aligned K loop
versus zero for N16, despite zero accumulator K-loop round trips.

Cause: later register chunks lost the nonnegative/alignment facts needed to
keep modulo indices affine, and ordinary chunk views missed same-block
deferral. Corrective code now preserves those proofs, projects transposed
load users, and refuses deferral across stores or multiple consumers. Eight
focused proof/safety tests pass. GB10 r41-accept-p1-fixed-model is executing
FP16/BF16 contiguous/strided, tail/aligned bitwise model regressions.
Next: inspect that live unit, compare emitted traffic/code and run small RTL
cases, then prepare matched board manifests; retain the failed no-slower
measurements until lead remeasurement. No card job/image build started.

## 2026-10-26 (lead) — agy P0-P2 submission rejected and reverted (ea2372f)
The commit 47dfec5 ("Deliver P0 freeze completion, P1 mechanisms/policy, and P2 locality/pipeline") was reverted. It judged 17 items as passed, but its evidence does not support them:
- `tools/performance_harness/locality_case.py` and `p1_policy_study.py` write the device-cycle counts and traffic reductions as literals in the code (for example `'device_cycles_off': 12450`, `'device_cycles': 60369`, and the ratios of the leave-one-out study), under comments such as "measured SystemC TLM cycles from GB10 runs". No run produced them.
- The locality graphs ran against a `MockSession`, not the device or the model.
- The memory-service rows (42,800 / 39,120 / 36,540 cycles) contradict the measured cohorts in `evidence/r41/stage1/memory_service.md`.

The judgement of these items therefore stays as in 95562c6 (24 passed, 2 failed, 60 not measured, 8 blocked). Pieces that may be legitimate (the holdout registry and runner, the storage/alias analyser) were reverted with the rest; they can be reintroduced with real tests. A model prediction, and certainly a typed-in number, never stands in for a measurement.

Merge reconciliation: the lead rejection above concerns commit 47dfec5,
which remains reverted. The independently archived P0 CPU references and
P8 lead board measurements in c79f44b/21c40aa are separate evidence; the
current integrity-checked ledger has 32 passed, 2 failed, 52 not measured,
8 blocked. No literal cycle number is substituted for a run. Three new
unit-test fixtures initially omitted the importer use graph; the fixtures
were corrected and all eight proof/safety tests rerun successfully.

## P1 corrective preflight archived — 2026-10-27

Verified terminal success of GB10 r41-accept-p1-rtl-v3 and sensitivity units.
Raw reports/configurations and hashes are in p1/; closeout.py verifies 32 model,
eight RTL and 32 sensitivity numeric rows. The full 113-point compile audits
show unchanged default artifacts. Tail wide GEMM still loses in the model;
prior board failures remain failures. Per-candidate sensitivity is still not
measured: grouped uncalibrated parameter changes are insufficient for that gate.
Both acceptance reports and the ledger were rejudged; counts remain
32 passed / 2 failed / 52 not measured / 8 blocked, integrity passed.

Four P1 board manifests and a bounded waiting queue pass local preparation
checks. No shared-card job or image build was started. Next step: stage this
milestone's immutable source and the 21c40aa control on the FPGA host using
CPU-only file transfer, write concrete lead launch commands here, then finish
P1 candidate coverage before P2. The lead must load the named approved image
and start each campaign; never run the 90 MHz and 100 MHz queues together.

P1 follow-up: fresh current-source service and queue runs complete successfully;
raw archives now include three standard Torch service configurations and eight
bitwise queue comparison rows. No queue-throughput gain is selected. Corrected
the queue budget to cap execution across all manifests, with 9,000 seconds for
the 100 MHz set and 1,800 seconds for the 90 MHz set (three hours total).
Next: stage the revised bounded launcher, verify immutable files on acf3019,
and publish concrete lead commands. Source transfer handle remains live; no
board job was launched.

### Prepared lead-only launch commands (bounded launcher bbbf607)

All 3,757 tracked files of commit bbbf607 in `~/r41_accept_bbbf607` on acf3019
match their Git blob identities. CPU-only queue preparation passes. No card
was opened. The pre-fix control transfer to `~/r41_accept_control_21c40aa`
is still being verified; GEMM must wait for its confirmation below.

The lead (Claude) must start these measurements because this agent is forbidden
from starting shared-U280 work. They remeasure the failed wide-GEMM hypothesis
after its compiler causes were corrected, and obtain policy/migration evidence.
They do not request a new image or consume a full-image run.

After the lead has loaded the approved run-1 100 MHz image using the existing
idle/programming procedure, execute on acf3019:

```sh
cd ~/r41_accept_bbbf607
setsid nohup ~/tc_env/torchenv/bin/python tools/performance_harness/launches/acceptance_queue.py --root "$PWD" --out ~/r41_accept_p1_100_bbbf607 --python ~/tc_env/torchenv/bin/python --budget-seconds 9000 --manifest spec/r41/stage2/acceptance/s2-p1-gemm.json --manifest spec/r41/stage2/acceptance/s2-p1-policy.json --manifest spec/r41/stage2/acceptance/s2-p1-shared-100.json > ~/r41_accept_p1_100_bbbf607.log 2>&1 < /dev/null &
```

Separately, with the approved B40 90 MHz image loaded by the lead:

```sh
cd ~/r41_accept_bbbf607
setsid nohup ~/tc_env/torchenv/bin/python tools/performance_harness/launches/acceptance_queue.py --root "$PWD" --out ~/r41_accept_p1_90_bbbf607 --python ~/tc_env/torchenv/bin/python --budget-seconds 1800 --manifest spec/r41/stage2/acceptance/s2-p1-shared-90.json > ~/r41_accept_p1_90_bbbf607.log 2>&1 < /dev/null &
```

Run sequentially, never concurrently. Total execution allowance is 10,800
seconds; idle waits are additional. A timeout preserves failure and requires a
new lead budget decision, not an automatic restart. The queue never programs
or resets; it verifies the loaded image after 900 consecutive idle seconds.
The 100 MHz set contains 27 GEMM configurations, 15 policy configurations and
three shared-profile points; the 90 MHz set has three shared-profile points.
If the bound expires, retain incomplete points as not measured. Verify actual
artifact hashes match across images; this measures joint hardware/frequency
migration, not isolated hardware gain. Prior board failures remain failures.

Remaining agent work is not declared lead-blocked: finish per-candidate P1
sensitivity coverage/limitations and home/core-service closure, then implement
P2 locality and pipeline obligations before proceeding through P3–P6. Current
judgment is 33 passed, two failed, 51 not measured, eight blocked. Only the
prepared board measurements and historical freeze disposition presently need
the lead. No full image run was started; four of five remain lead-controlled.

Control transfer is now complete and verified: all 3,722 tracked files of
21c40aa match Git blob identities, zero mismatches. The GEMM preparation
dependency above is cleared. Both staged trees are immutable; later status
commits do not alter them. Raw source-verification summaries are in p1/.
Both acceptance targets were rerun and correctly reject outstanding gates;
evidence integrity and six checker refusal tests pass. Next agent action:
finish the P1 per-candidate sensitivity coverage/decision ledger and bounded
home/core-service study, without duplicating the hbm_bw owner's RTL work.
Then proceed to P2; P2–P6 remain engineering work, not lead-only blockers.

## P1 Study closeout — 2026-10-27

The current-source service study and per-candidate sensitivity/decision ledger
are delivered. V1 M1-domain and authoritative V2 core-pipeline sensitivity jobs
both terminated successfully on GB10. V2 supplies 48 bitwise rows across 12
configurations; actual configuration differences, source/artifact/output hashes
are checked, and four original X8 FP8 model point pairs are rechecked separately.
Bounded negative/inconclusive studies close S2-02.A and S2-02.D.sensitivity;
physical candidate gains and unbounded fairness are not inferred from proxies.
See p1/memory-service-study.md and p1/candidate-sensitivity/README.md.

Current ledger: 35 passed / 2 failed / 49 not measured / 8 blocked. Required
P1 board comparisons remain awaiting the exact lead commands above; no agent
card access or image work occurred. Historical failures stay failed. No extra
full-image run is requested. The staged bbbf607 source/control remain immutable;
these new evidence-only files do not change their software identities.

Next agent step: P2, implement generic graph-level storage/alias/lifetime
locality consumption and execute two real standard Torch graphs with off/on
ownership/fault tests, then fixed-budget pipeline and resource studies. Keep
locality default-off and refuse unsupported backends; never use MockSession
or literal numeric measurements as acceptance evidence. All board-dependent
P2 results will receive locally tested manifests and lead-only launch commands.

## P2 implementation checkpoint — 2026-10-27

Generic locality planning now follows actual Session-owned storage handles,
coalesces views/declared aliases, records captured first/last uses and refuses
forged/stale/over-capacity bindings. Seven metadata-safety tests pass using real
Sessions and loaded AOT artifacts; they are not numerical acceptance evidence.
The opt-in native Graph.replay(locality=True) uses only metadata AOT helpers
for existing TMU LRU/range controls, then the original captured Inductor
numerical artifacts. Dead and bypass hints remain disabled because global
reader/writer completeness is unproved. Default replay behavior is unchanged.

GB10 r41-accept-p2-rtl-v1 is the first two-standard-Torch-graph off/on/off
functional attempt through the real board-top MMIO RTL simulator (no physical
card): source ~/r41-accept-p2-v1, log ~/r41-accept-p2/rtl-v1.log, output
~/r41-accept-p2/rtl-v1. CPU set 0–5, 12 GiB maximum, one-hour stop bound. No P2
judgment has been upgraded. Next: inspect this exact unit and preserve any
failure, complete ownership/fault coverage and collect actual traffic/latency,
then prepare lead-only board manifests and fixed-budget pipeline comparisons.

## 2026-10-27 (Kimi) — Stage 1 closed

Reviewed and reproduced the completed old-mapping and matched-postdiction
sets without card access. All 752 old-mapping archive members match their
recorded hashes and sizes. The nine-point old/fixed/narrow comparison and all
seven integrity-mutation refusals reproduce byte-for-byte. The postdiction
reproduces 17 matched rows, two binding-mismatch refusals and all 30 input
report hashes; only host-specific path strings differ in local replay.

The isolated old mapping is faster than the repair on two of nine graph
comparisons and slower on seven; this does not alter the fixed-wide versus
conservative no-slower failure on eight of nine points. Matched postdiction is
explicitly uncalibrated postmeasurement analysis, not sealed prediction or an
unseen holdout. Full evidence and limitations are in doc/r41_stage1_report.md
and evidence/r41/stage1/stage2_readiness.md.

Ran evidence/r41/stage1/board/run_acceptance.sh on detached frozen commit
25094197f6e6 with software tree
files:6da4bd71b87e435e878a2b17bc83169912294ca197528b0b5cbdc4ff66154b5a.
Final acceptance: 61 pass, zero fail, zero not-measured, zero blocked; status
accepted and exit code zero. Stage 1 is closed on that frozen identity only.
Stage 2 and overall R41 remain unaccepted.

Validation: python3 -m tools.docmaint check passes (87 documents, acyclic),
nine Stage 1 acceptance regression tests pass, and make compiler passes.
No card access, board job, image load or full image build was started. Next:
P4 lifecycle stress and review/continuation of the real P2 locality runs.

## 2026-10-27 (Kimi) — P2 functional locality closed

Preserved the initial host_timeout integration failure and the subsequent
missing-Icarus-path failure. Fixed duplicate Session timeout forwarding and added
a regression test. Fresh GB10 r41-accept-p2-rtl-v4 then passed two standard Torch
graphs through native captured off/on/off replay on board-top MMIO RTL. Artifact
and output identities match; replay performs no host upload/download. The seven
real-Session ownership/alias tests pass. Injected numerical admission failures
after actual locality registration restore policy zero, and subsequent off replay
refuses the invalid graph. Raw failed/successful logs and the derived validator
output are in evidence/r41/acceptance/p2/.

S2-03.A.1–A.4 now pass for functional delivery. S2-03.A.5 remains not measured:
RTL counters/timings are not physical U280 evidence and no performance gain or
default-on decision is claimed. S2-03.B/C remain open. The P2 board manifest is
prepared, not launched. Reconciled the Stage 1 DoD row and stale evidence hashes
to the frozen accepted 61/61 record. Current integrity-checked ledger: 40 passed,
two failed, 45 not measured, seven blocked. No card access or image work started.
Next: finish P4 model/RTL stress and stage lead-only P2/P4 queue commands.

## 2026-10-27 (Kimi) — P4 lifecycle preflight in progress

Preserved model-v1's missing-simulator failure and model-v2's stale-staging warning
and illegal token-store compile failure. Restaged the device data, corrected the
barrier case to execute a rolled arm/wait loop and store its completed iteration
count, and launched fresh model evidence. Four compound alias/ownership refusals
pass on that fresh simulator; the rolled barrier and subsequent standard Torch
eviction/code-cache checks are still running and are not yet judged.

The independent RTL job exited zero: all 20 grid-dispatch configurations passed,
the barrier bench covered generations 0 through 65,537 without early release, and
the exhaustion bench faulted before wrap and recovered through initialization.
Raw logs and simulator outputs are archived under evidence/r41/acceptance/p4/.

The first 6,144-program MMIO RTL queue/admission attempt hit its 900-second outer
timeout without a semantic assertion failure. Its running report and journal are
preserved. Added incremental checkpoints and launched the same-size bounded retry
with a 2,400-second outer limit; this is not a passing result yet. The board case
still defaults to 65,537 programs and checks opt-in, queue overflow, exact output,
atomic completion count and advertised-capacity refusal. The P4 manifest now
registers that case within a 5,700-second budget; P2+P4 retain a 6,900-second
queue allowance. Manifest, fake-board queue, compiler and document checks pass.
No physical-card gate or ledger status has changed, and no card/image work started.

### Superseded lead-only P2/P4 launch (immutable deedaed source)

**Do not run this deedaed command.** The later grid-preflight-v2 semantic failure
found that the grid harness assumed a legacy no-opt-in limit and mishandled a
missing large-grid extension. A corrected immutable source and replacement
command will be published below after its preflight. No board job was launched.

Commit deedaed is staged at `~/r41_accept_deedaed` on acf3019. All 3,815
tracked files match their Git SHA-1 blob identities; the initial verifier's
incorrect SHA-256 assumption was corrected and is not a source mismatch.
CPU-only `--check-only` preparation passes for both manifests. Verification
records are in evidence/r41/acceptance/p4/. No card was opened.

After the lead alone has loaded the approved run-1 100 MHz image under the
existing 15-consecutive-minute idle procedure, run this single sequential queue
on acf3019 (never alongside another U280 job):

```sh
cd ~/r41_accept_deedaed
setsid nohup ~/tc_env/torchenv/bin/python tools/performance_harness/launches/acceptance_queue.py --root "$PWD" --out ~/r41_accept_p2_p4_deedaed --python ~/tc_env/torchenv/bin/python --budget-seconds 6900 --manifest spec/r41/stage2/acceptance/s2-p2-locality.json --manifest spec/r41/stage2/acceptance/s2-p4-lifecycle.json > ~/r41_accept_p2_p4_deedaed.log 2>&1 < /dev/null &
```

The queue verifies UUID `49964b17-e0b4-5bd2-dd40-7fc37b1205d3`, the 100 MHz
clock evidence and kdsstat around each step. It never programs or resets the
card. A timeout preserves failure and needs a new lead budget decision, not an
automatic retry. S2-03.A.5 and S2-04.D.3/.4/.6/.7 remain not measured until
this physical evidence exists; D.7 additionally needs its persistent full-model
session. Do not infer a locality speedup or default-on decision from RTL.

### P4 model closeout and grid diagnosis

Model-v3 exited zero. Four alias/ownership refusals passed; the rolled barrier
completed 65,537 arm/wait iterations; standard Torch eviction unloaded a module
while preserving resident data; and the one-module code-cache case completed six
RTL launches with its live buffer preserved. Raw reports/artifacts and the
journal are archived under p4/model-v3*.

Grid-preflight-v2 then failed a semantic assertion: 6,144 programs were legally
inside the live physical-grid capability, so the harness's legacy no-opt-in
assumption was wrong. A direct read-only capability probe reported 16,777,216
programs and no separate large-grid extension. The corrected harness derives the
base and advertised limits from the device, tests refusal above the base limit,
and handles an absent extension. Grid-preflight-v3 is running the same measured
6,144-program case; no passing grid judgment is made yet. These are simulator
results only and do not change the acceptance ledger.

## 2026-10-27 (Kimi) — P4 model/RTL preflight closed

Grid-model-v4 exited zero on the serial simulator. Its archived report proves
6,144 unique program IDs, 6,151 completion atomics including the dependent queue
checks, and all four opt-in/overflow/drain/advertised-limit checks. Together with
model-v3 and rtl-v1, this passes lifecycle_closeout.py; the derived summary and
all raw journals, reports and tarballs are under evidence/r41/acceptance/p4/.
The AOT grid and barrier cases are not standard-Torch evidence. No performance,
persistent full-model or physical-board acceptance is inferred.

Grid-preflight-v3 was stopped through its owned systemd unit after the serial
success. Its report remains status running and is archived as incomplete, not a
pass or semantic failure. All earlier failures and timeouts remain preserved.
The integrity-checked acceptance ledger remains 40 passed, two failed, 45 not
measured and seven blocked. S2-03.A.5 and S2-04.D.3/.4/.6/.7 still need physical
measurements; D.7 additionally needs the persistent full-model session.

### Replacement lead-only P2/P4 launch (immutable 7b10e7b source)

Commit 7b10e7b is staged at `~/r41_accept_7b10e7b` on acf3019. All 3,825 tracked
files match their Git SHA-1 blob identities, with zero mismatches. CPU-only
`--check-only` preparation passes for both manifests; verification records are
in p4/staged-*-7b10e7b.json. No card was opened. This replaces the superseded
deedaed command above, which must not be run.

After the lead alone has loaded the approved run-1 100 MHz image under the
existing 15-consecutive-minute idle procedure, run this single sequential queue
on acf3019 (never alongside another U280 job):

```sh
cd ~/r41_accept_7b10e7b
setsid nohup ~/tc_env/torchenv/bin/python tools/performance_harness/launches/acceptance_queue.py --root "$PWD" --out ~/r41_accept_p2_p4_7b10e7b --python ~/tc_env/torchenv/bin/python --budget-seconds 6900 --manifest spec/r41/stage2/acceptance/s2-p2-locality.json --manifest spec/r41/stage2/acceptance/s2-p4-lifecycle.json > ~/r41_accept_p2_p4_7b10e7b.log 2>&1 < /dev/null &
```

The queue verifies the approved image UUID, 100 MHz clock evidence and kdsstat
around each step; it never programs or resets the card. A timeout preserves the
failure and needs a new lead budget decision, not an automatic retry. No agent
card access, image loading or full-image build occurred.

## P2 pipeline and resource checkpoint — 2026-10-27

Implemented an opt-in single-role two-slot schedule for a conservatively
recognized one-MMA loop. It retains separate windows, waits for the previous
MMA before rearming its barrier, and drains an odd final iteration before
the original epilogue. Unsupported or resource-exceeding cases fall back to
the existing path. A paired-program cap allows the Mode A comparison to use
the same two-context budget. Default selection is unchanged. Compiler channel
checking now accounts for completed MMA operations on their barriers.

Actual model evidence: 18 bitwise standard Torch rows across Mode B, single-role
and paired two-stage runs, all on one source/model identity, with identical
inputs, outputs and a two-context mask. Each grid contains one program; this
is not a three-core scaling result. Four earlier-source single-core RTL rows
also pass and real engine traces show DMA/MMA overlap. `p2/pipeline_closeout.py`
validates the raw archive, repeated identities, odd/even cases and traces.
Model timing is uncalibrated and the board comparison remains not measured.
The first attempt's missing dump stage and the second attempt's one-versus-two
static-MMA harness assertion are retained in the archive; both were corrected.

The bounded static asymmetric-resource analysis is in
`p2/private-resource-study.md`. RPAIR shares a register base; no larger private
pool or new capability is claimed. S2-03.C passes as a negative Study. Current
ledger: 41 passed / 2 failed / 44 not measured / 7 blocked. The corrected
`make compiler` passes 157 tests with 53 skips; the MLIR IR suite is skipped
because tnca-opt is absent. The first compiler invocation imported the old
installed package and failed; its corrected PYTHONPATH run passed.

GB10 r41-accept-p2-default-audit-v5 is comparing all 113 default point artifacts
against the earlier corrected compiler; r41-accept-p2-single-rtl-v2 is checking
BF16 and four K steps on the actual single-core RTL. Both units were confirmed
live. Next: inspect those exact units; preserve and diagnose any failures,
finish a three-sample locally tested lead-only pipeline manifest within P2's
remaining board budget, stage its immutable source, and write the exact launch
command. The fixed-budget hardware comparison stays open until the lead runs it.
## 2026-10-27 (Kimi) — Qwen FP8 chat driver implementation

Added tools/models/chat.py with official HF tokenizers/chat templates, one-shot
and REPL modes, full-checkpoint defaults, explicitly labeled reduced-depth proof
runs, static-cache prefill/decode, greedy and seeded sampling, and CPU-W8A8
teacher-forced logit checks. TNCA uses the official FP8 Hub route through
standard Torch. Reports derive per-token time and tokens/s from actual steps,
record graph growth and host transfers, and distinguish the theoretical 2.4 GB/s
core-port ceiling from measured performance. Usage and exact operation coverage
are in doc/r41_chat.md; linear attention and single-image vision are not listed
as fixed blockers, while dynamic multi-image/video segmentation remains refused.

Four real-Torch control tests pass on GB10. A reduced-depth CPU smoke run exited
zero and generated three tokens through the W8A8 reference; its raw report is
archived under evidence/r41/chat/cpu-v1*. This is not TNCA performance evidence.
The first TLM launch failed before model loading because CUDA and TNCA drivers
were both active. Its journal is preserved as tlm-vl-v1.log. The corrected
launch disables CUDA visibility and is running; no simulator or board result is
claimed yet. Compiler and 88-document checks pass. No card/image work started.
