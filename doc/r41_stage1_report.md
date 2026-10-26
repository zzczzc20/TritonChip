# R41 Stage 1 closeout report

Date: 2026-10-26 UTC. Sign-off name: `R41_S1_COMPILER_AND_PERFORMANCE_HARNESS`.
Status: **not accepted; matched old-mapping comparison remains open**.

This report judges the frozen Stage 1 product, commit `25094197f6e6`, software
`files:6da4bd71b87e435e878a2b17bc83169912294ca197528b0b5cbdc4ff66154b5a`.
It does not sign off the merged product or Stage 2. R40 remains
`closed-with-failures`.

## Evidence and identity

The seven completed manifests ran on acf3019 using the B40 U280 image
`a88ebdf59cfbbb2001ebd5b1b2c739c5d8cd33931f1e9202f18cac07c4c83083`,
UUID `1bada9db-78f3-5bc8-7ba6-cce062e7ade8`, qualified 90 MHz, three cores,
three physical outlets and a 4 GiB exposed window. Qualified clock evidence
SHA256 is `8ad4b3bd30a3f5fe8819fe59ef8ad160fd0de1478aaa506dc33d82785fb9f487`.

The [board evidence index](../evidence/r41/stage1/board/README.md) identifies
the selected archive, member hashes and host paths. Large raw logs stay on
`acf3019:/home/zzhouch/r41s1_board_25094197f6e6`; their SHA256 values are retained.
Collection used only file reads and tar streams. No device, queue or host script
was changed. The interrupted original-suite record is preserved separately and excluded from final statistics; its one point was successfully re-run.

The local `board/closeout.py` verifies archive hashes, checks the 35 study formal
records with the contract's board-record validator, verifies the frozen software
identity, loads the actual GEMM/prospective executable artifacts, and recomputes
both sealed prediction comparisons. Numerical tolerances, point hashes, image,
clock, configuration, descriptors, fallback absence and completion timing are
checked. Board artifact hashes match the formal bundles; executable identities
match the sealed predictions despite different provenance/file hashes.
The resulting `metrics.json`, `gemm-tile-residuals.json` and
`prospective-residuals.json` are reproducible without the board. Diagnostic
memory/pipeline cases use their own numerical and cohort checks, separate from
the 35 study ArchBench records and 113 original-suite records.

Experiment exit 0 establishes command completion. Several experiment reports
still say `result: not-recorded` or `performance_gate: not-measured`; this report's
judgements below come from the measurements, not those exit codes.

## Manifest judgements

| Manifest | Evidence-backed judgement | Limits and decision |
|---|---|---|
| `board-anchors` | Numerical and reproduction checks pass: 9/9 sessions across 3 frozen anchors | Descriptive stability supports bounded sampling; no population confidence claim |
| `board-gemm-tile` | Numerical capability passes, 18/18 configurations; **no-slower hypothesis fails on 8/9 points** | Close the study with a negative performance result; wide tile must not become a general performance default |
| `board-prospective` | Fast-path absolute residuals recorded and small for both arms | Slow path remains inconclusive; A/B ratio error is +5.23%; global model remains uncalibrated |
| `board-memory-service` | 9/9 diagnostic cases functional; depth benefit observed | No reliable additional outlet benefit; matched-source absolute model residual unavailable |
| `board-pipeline-admission` | 48/48 workload rows functional across 12 case processes; tail-mm stage-3 slowdown reproduced | Physical overlap itself not measured; stage-3 addmm bindings differ, so causal comparison is confounded |
| `board-tune-qualification` | 6/6 formal records, 3 device-ranked kernels, 3/3 cache hits; poison/recovery checks pass | Qualification of selection and recovery; no speedup/default nonregression claim |
| `board-original-suite` | **113/113 numerical passes; 202/202 accepted runs; device-time nonregression passes** | Interrupted record preserved; only its one point re-run; no >5% regression against 113 B40-OBS points |

## Anchors and timing boundaries

Each anchor has three independent interleaved processes, one formal timed call
per process. Cold calls are excluded. Range is `(max-min)/median`, not a confidence
interval. Graph timing ends at the last device completion; serial-grid timing sums
launch windows. Their difference includes gaps and must not be discarded.

| Anchor | Serial-grid median cycles | Graph median cycles | Graph range/median |
|---|---:|---:|---:|
| HBM add, fp16 n67108864 | 243,997,720 | 243,997,722 | 0.0348% |
| Medium fp16 matmul 128×192×128 | 231,745 | 231,747 | 1.5431% |
| HBM four-layer decode, b1 p512 | 170,603,213 | 204,863,817 | 0.6142% |

The long anchors are stable at this descriptive scale. Host latency has not been
qualified by these device samples. No borderline performance claim is promoted
without the registered supplementary sampling.

## GEMM repair, performance failure and prospective residuals

The measured wide candidate really executes the repaired path: native fp16
matmul artifacts declare two accumulators and match the sealed executable
identities. Earlier indexed RTL/structural regressions establish residency and
K-order semantics. A numerical capability pass is separate from a speedup pass.

The following ratios use median measured **graph makespan**, candidate `16,32,16`
divided by control `16,16,16`. Values above one mean slower. Ratio residual is
`model_ratio / board_ratio - 1`. No point is removed.

| Frozen GEMM subset point | Board ratio | Fast model ratio | Ratio residual |
|---|---:|---:|---:|
| `model.mlp/medium/fp16-t128-d128` | 3.2464 | 2.8613 | -11.86% |
| `op.bmm/medium/fp16-b16-m64-k32-n64` | 0.9270 | 0.9251 | -0.20% |
| `op.gemv/hbm/fp16-k4096-n11008` | 5.2996 | 6.3839 | +20.46% |
| `op.gemv/medium/fp16-k320-n448` | 3.2974 | 3.0870 | -6.38% |
| `op.linear/hbm/fp16-m512-k4096-n11008` | 8.6344 | 7.6545 | -11.35% |
| `op.linear/medium/fp16-m128-k320-n256` | 5.4708 | 5.1019 | -6.74% |
| `op.matmul/hbm/fp16-m4096-k4096-n4096` | 3.0402 | 3.0381 | -0.07% |
| `op.matmul/medium/fp16-m128-k192-n128` | 3.1234 | 3.0286 | -3.04% |
| `op.matmul/medium/fp32-m128-k192-n128` | 1.7482 | 1.6011 | -8.42% |

Only batched GEMM improves (about 1.079×); the subset ≥1.15× Target is not
established. There are no ranking inversions or false gain predictions in these
nine A/B pairs, but their ratio errors range from −11.86% to +20.46%.
The GEMM fast-path absolute residual median is 6.4432%, p95/max 21.9548%
(nearest-rank p95, 18 configurations). The historical 10%/20% qualification is
not met even within this small subset's p95; global calibration is not claimed.
Sixteen slow predictions are inconclusive-fixed-point and two were not run.
They are preserved without inclusion in qualified accuracy statistics.

For the separately registered prospective three-core resident medium GEMM:

| Arm | Measured graph cycles | Fast prediction | Signed residual | Slow prediction | Slow signed residual |
|---|---:|---:|---:|---:|---:|
| Wide candidate | 894,322 | 911,645 | +1.9370% | 918,103 | +2.6591% |
| Control | 304,701 | 295,176 | −3.1260% | 295,965 | −2.8671% |

Residual is `(model-board)/board`. Fast median absolute error is 2.5315%; the
board A/B ratio is 2.9351 versus model 3.0885, ratio error +5.2264%. There is no
ranking inversion or false gain assertion. The slow residuals are arithmetic
comparisons of **inconclusive predictions**, not qualified model results.
The hypothesis is supported for fast absolute timing at these configurations;
slow-path qualification is inconclusive. No numeric uncalibrated error band was
registered, so an invented threshold is not used to call the whole hypothesis a pass.

The sealed files are `kickoff/prospective-prediction-v2.json` (SHA256
`eff75a74abb9e7d5a49e2375313343ade18d3a922b9651e7622f4c5eeccd7766`)
and `kickoff/gemm-subset-predictions-v2.json` (SHA256
`49f4171d8c5815a99f88fb1f6077a260082faf7137ada5860830d183fa675e0a`).
They predate board measurements. No parameters were fitted during closeout.
These diagnostics cannot subsequently be described as unseen holdout.

Detailed-path cost does not buy demonstrated qualification here. For the
prospective candidate/control, sealed process wall times are 25.15/5.97 s fast
and 7.95/6.71 s slow; model launch host times are 0.312567/0.120590 s fast and
1.794893/0.859140 s slow. Process startup/build effects prevent interpreting
whole-process ratios as simulator throughput. The complete per-arm cost ledger
is in `metrics.json`. Slow fixed-point failure prevents claiming which residuals
Ramulator2 resolves; the fast path remains the screening default.

## Memory service and pipeline studies

Memory completed-launch cycle medians across three repetitions are 17,000
(1 outlet/depth 1), 15,870 (1/depth 8), 15,836 (3/depth 8). Depth 8 reduces
cycles by 6.65% (1.071×); the additional outlet change is 0.21%, with repetition
ranges overlapping and one repetition reversing its ordering. Thus the depth
part of the hypothesis is supported; there is no reliable additional outlet
benefit. All cases have equal inputs, shader and bindings; physical counter
deltas confirm all three outlets carry traffic in the 3/8 arm. Counter windows
include native-call maintenance whereas launch timing excludes prelaunch work.
Peak-counter differences are not occupancy measurements. The 1.2× Target is not
established. Existing defaults are retained.

Pipeline cycle medians are:

| Graph / shape | Baseline | Stage 1 | Stage 2 | Stage 3 |
|---|---:|---:|---:|---:|
| mm 16×16×48 | 3,843 | 3,884 | 3,219 | 3,248 |
| mm 17×31×65 | 124,660 | 125,446 | 123,201 | 127,634 |
| addmm 16×16×48 | 4,394 | 4,478 | 3,810 | 3,650 |
| addmm 17×31×65 | 123,920 | 122,925 | 120,949 | 128,584 |

Tail-mm stage 3 is 3.60% slower than stage 2, consistently across all three
repetitions. This supports the registered tail hypothesis and preserves a
negative result. The earlier RTL study measures actual transfer/MMA overlap;
the board comparison has no overlap trace, so board timing does not itself
prove overlap or its cause. Stage-3 addmm changes bindings relative to baseline
in all repetitions; its timings are reported but do not establish an isolated
stage-count gain. Mode A remains default-off.

The board cohorts explicitly say that absolute model residuals need matching
product-source model cases. Earlier model/RTL cohorts differ in source and
memory/ownership domains. Their numerical cycle differences are not accepted
as physical prediction residuals. Both bounded case studies are closed with
these limitations and decisions; no model accuracy or new default performance
pass follows from study completion.

## Tuning, recovery and outstanding Required items

The public path selects three pointwise kernels within four-candidate and
600-second limits, reuses all three winners in new processes, and keeps tuning
launches outside formal samples. The deadline test forces a 19,885,260-cycle
limit on a 159,082,080-cycle graph, poisons the session without advancing epoch,
restores limits, advances epoch 0→1 only on explicit recovery, refuses stale
storage and verifies recovered output. Its classifier says `expiry: other`,
while the preserved exception explicitly reports `DEVICE_LOST`; the accepted
result concerns the observed poison/recovery semantics, not that label.

Original-suite correctness now passes on all 113 frozen points (77 native,
12 HBM-tier and 24 large), with no failed final runs or fallback. Maximum error
against the independent references is 0.058543 of tolerance. The experiment
status is `completed-experiment`; the batch is `complete`, with 202 successful
qualification/formal processes and no failed configurations or missing formal data.

The original attempt completed 112 points. Another user's card reload interrupted
`op.attention_decode/large/fp16-b1-h32-l2048-d64` during qualification on October 25
19:33:02–19:33:13 UTC. Its rejected record remains in `runs.interrupted/`, and
`state.json.before-retry-of-interrupted-point` preserves the 201-entry state.
Only that point was re-run on the same software and B40 image: seed 0, warmup 0,
qualification timed 1 on October 26 09:04:21–09:04:40 UTC, then a separate formal
timed-3 process at 09:04:40–09:04:59 UTC. The other 112 points were retained.
The archive and member hashes are in `board/original-suite-index.json`.

The locally reproducible `board/original-suite-comparison.json` compares all 113
points against B40-OBS `R40-08.attention/board_native_fe1b` records. Image, 90 MHz
clock, backend arguments, compiler settings, knobs and original point hashes
match. Use the 77 native records, 12 HBM records (exclude the two duplicated
large attention points in `hbm14`) and 24 `large24` records. Cold calls are
excluded; the metric is median serial-grid device cycles, divided by 90 MHz.
Geometric mean R41/R40 is 0.998400; summed device-time ratio is 1.000322
(+0.0322%). No point regresses >5%, so the itemized >5% and >10% lists are empty.
Section 6.5 device-time nonregression passes its 5% budget. Historical references
have one timed sample each; these comparisons are descriptive, without noise
confidence intervals. No host-latency or graph-makespan nonregression is inferred.

The matched final-source old-mapping comparison remains **not-measured**.
The measured 16×16 control and historical B40-OBS suite are not an isolated
old-mapping experiment. Severe wide-tile slowdowns remain failures of the
no-slower hypothesis; the candidate must not become a general performance default.

## Frozen acceptance reproduction

The merged tree at `3bbe981` has software identity
`files:a86af4ae352dd47d9910b35fbe411fcaef920c483bd6dcc7db330972584e1303`.
Its contract document differs from the Stage 1 indexed SHA256
`2e0e598b9331dca5e922ce901ccd0d801792cefb41ca7fbf9b7bc4a41de3ddfd`.
Running the pre-closeout index there gave 37 pass, 2 fail, 21 not-measured,
1 blocked: the indexed contract mismatch and its derived prerequisite failed.
The frozen worktree with the existing evidence overlay gave 39 pass,
0 fail, 20 not-measured, 2 blocked before this closeout.

A full checkout in `/tmp` exceeded its tmpfs capacity and was discarded by git.
The successful detached worktree is `/home/ubuntu/r41s1-frozen-25094197f6e6`,
created with `git worktree add --detach ... 25094197f6e6`. No branch was created.
`board/run_acceptance.sh` copies only Stage 1 evidence and this report to that
worktree, extracts both archived record bundles, asserts the product identity and the
original indexed contract digest, then runs:

```sh
make -C /home/ubuntu/r41s1-frozen-25094197f6e6 r41-stage1-acceptance \
  PYTHON=python3 \
  R41_ACCEPTANCE_OUT=/home/ubuntu/TritonChip/evidence/r41/stage1/board/acceptance.json
```

The checker is unchanged. The updated index supplies real measured board
records and scoped study decisions; it never upgrades missing measurements.
The frozen contract is left intact, and the merged contract's index hash is not
silently redefined. Evidence/report overlays are outside the product hash's
source directories. Final `acceptance.json` and `acceptance.log` record **59 pass, 0 fail,
1 not-measured, 1 blocked** out of 61 Required items, and every open prerequisite.
The zero failed Required items does not erase the failed GEMM performance
hypothesis: bounded study completion and performance success are distinct. Acceptance exits nonzero while Required
items are open; **Stage 1 remains not accepted**.

The [Stage 2 readiness ledger](../evidence/r41/stage1/stage2_readiness.md)
records the board-derived inputs and their limits. The independent
[100 MHz Stage 2 results](../evidence/r41/stage2/board/results.md) retain their
own image/software boundaries and remaining full-model issues.

## Separate Stage 2 board results

The [Stage 2 results](../evidence/r41/stage2/board/results.md) concern the
100 MHz run-1 image `30e79769…4e5de`, UUID `49964b17-e0b4-5bd2-dd40-7fc37b1205d3`,
not the frozen Stage 1 product. They establish 113/113 original numerical points
(202 accepted runs), 100/100 FP8 board cases accepted by bitwise gates (six
plumbing comparisons allow NaN sign encoding differences), and 16/16 blockwise
cases with zero bit mismatches. The FP8 reports' stale RTL evidence label is a
harness defect; the backend, image and launch cycles establish board execution.

The 8 GiB window study passes 7/7 cases and 159 checks, plus 2/2 large-storage
cases and 14 checks, including all 32 PCs, addresses above 4 GiB and near 8 GiB,
a single storage above 2 GiB, and offsets beyond int32. Bandwidth is unmeasured.
Original-suite and endurance sessions themselves use a 4 GiB window; do not
combine these distinct configuration claims.

The first endurance attempt failed acceptance because of harness identity binding
and FP8 output-directory/report-field bugs; that failure remains recorded.
After the harness fix, commit `9bc8990`, the second attempt ran 2.0611 h
(October 26 06:12:27–08:16:07 UTC) and passed 250/250 sessions: 240 ArchBench
sessions and 10 FP8 sessions. Its software is `files:a86af4ae…1303` on the same
100 MHz image. The read-only `stage2/board/endurance-rerun.tgz` preserves its
report, endurance summary, sessions and host samples. This is a Stage 2 endurance
pass, not a Stage 1 sign-off. FP8 hub and Qwen FP8 execution remain unrun.
The earlier 96-point Stage 2 speed comparison remains a historical partial
comparison; it has not been recomputed against the newly completed Stage 1 set.

## Local verification

`python -m tools.docmaint check` passes: 76 documents, acyclic. The system has
`python3` but no `python` command, so the exact requested invocation used a
temporary PATH alias to `/usr/bin/python3`. Nine acceptance regression tests
pass with TMPDIR on the workspace disk; the initial rerun exhausted the `/tmp`
quota, so it was repeated with that directory changed. One initial test assumed that the live index could never pass any board
item; it was changed to construct an explicit fixture with board evidence
removed, preserving the refusal assertion as real board evidence arrives.
The checker itself and the frozen product sources are unchanged. Initial and
final test logs are retained. The residual recomputation also passes archive,
manifest, executable, numerical-status and board-record validation. Frozen
acceptance intentionally remains nonzero because the Required items above
remain unmeasured.
