> Non-normative English translation. The authoritative frozen contract is
> [r40_tasks_and_acceptance_v3.md](r40_tasks_and_acceptance_v3.md); only its original bytes are covered by
> the [v3 lock](../doc/r40_v3/lock.json). This translation does not amend the contract.

# R40 v3: A One-Time Relaxed Delivery Contract on the On-Board 90 MHz Image

Version 3, 2026-10-19. Status: **FROZEN CONTRACT / NOT ACCEPTED**.
Sign-off name: `R40_NATIVE_DEVICE_SCALABLE_NUCA_HBM_V3`.

Pursuant to user authorization on 2026-10-19 ("if unworkable clauses are found in the v2 contract, perform a one-time relaxation to v3 to accelerate delivery"; goal "deliver R40v2 on the basis of the currently successful on-board 90 MHz image"; "first close all gates that do not require a new bitstream, and combine items requiring a new bitstream with frequency optimization"), and the user's explicit choices on two decisions (performance gates adopt "significant gain + target reporting"; DCO/locality and `TNCA_TUNE` are deferred to R41). This document and [`doc/r40_v3/plan.json`](r40_v3/plan.json) are fixed by [`doc/r40_v3/lock.json`](r40_v3/lock.json), with the lock file hash embedded in [`tools/r40/contract_v3.py`](../tools/r40/contract_v3.py). Freezing was completed before any formal v3 performance samples were collected; after freezing, no further modifications to contracts, thresholds, candidate values, workloads, or baselines are permitted.

## Inheritance

Except for the seven modifications below, this contract fully inherits v2 ([r40_tasks_and_acceptance_v2.md](r40_tasks_and_acceptance_v2.md), with the [`spec/r40/v2/lock.json`](../spec/r40/v2/lock.json) hash embedded in the v3 plan) and all terms of v1 inherited by it: work packages, gates, dependencies, the standard Torch/Inductor/Triton path, descriptors, zero numerical fallback, frozen points and subsets, dtypes/shapes/tolerances, models, B39/B40-S/B40-N/B40-CPU baseline definitions, timing boundaries, and rejection rules ([`tools/r40/ablate.py`](../tools/r40/ablate.py)). Criteria for all capability gates remain unchanged.

## Modification 1: Delivered Image and Operating Frequency

The R40 delivery image is fixed to `a88ebdf59cfbbb2001ebd5b1b2c739c5d8cd33931f1e9202f18cac07c4c83083` (three cores SLR0/1/2, three memory outlets, 4 GiB exposed HBM window), with a measured operating clock of 90 MHz (at 90 MHz: WNS +0.016 ns, TNS 0, WHS 0; 100 MHz routing reported failure with WNS −1.002 ns). R40 makes no claim of 100 MHz. All gates are evaluated on this image; no further implementation or bitstream programming will be performed within R40. Items requiring a new bitstream are merged into the next implementation alongside frequency optimization (see Modification 3).

## Modification 2: Software Identity

The delivery software identity is the product tree `files:80dd1a823e4be49e5e40d375d842c17e46c362d298e37f389c5e225621b6e4ca` (covered by `archbench/provenance.software_tree` spanning `archbench/`, `python/`, `compiler/`, `tools/asm/`, `spec/`, `rtl/`, `tb/device/`). All on-board evidence must bind to this tree, this image, and proof of 90 MHz clock operation. Contracts, audits, and collection tools reside outside the tree (`doc/`, `tools/r40/`, `evidence/`); modifications to them do not alter product identity, but every piece of evidence records collection/audit tool hashes. If the product tree must be modified before delivery, all affected evidence must be re-collected under the new identity (as authorized and tracked in [`doc/r40_v3/identity_amendment.json`](r40_v3/identity_amendment.json)).

Pure RTL/simulation evidence may be collected on trees byte-identical in `rtl/` and relevant source files, with files and hashes listed in the audit. Capability evidence audited under v2, bound to the v2 lock, and with software identity matching the delivery tree remains valid in v3 (as capability criteria are unchanged).

## Modification 3: All Gates Not Requiring a New Bitstream Closed on this Image; Deferred Gates Moved Out of R40

| Gate | Reason | Destination |
|---|---|---|
| `R40-07.matrix` | Requires new MMA RTL (`TNCA_RTL_MMA` has no product implementation; image only contains v5 `MTX_ABUF`) | Merged with the next implementation alongside frequency optimization |
| `R40-04.ingress` ablation | v2 already specified this as required only when wide ingress is implemented; this image does not implement 32 B/core/cycle | Same as above |
| `R40-06.locality` | `TNCA_LOCALITY`/`TNCA_DCO` have no standard-path consumer (TMU hardware is in the image, but the compiler does not issue them) | R41 (user decision) |
| `R40-09.autotune`, `R40-09.holdout` | Measured selection via `TNCA_TUNE` is not connected to the standard path | R41 (user decision) |

Deferred gates are marked as `deferred` in v3, not pass; R40 delivery makes no claim on these capabilities. Mechanisms in the ablation matrix with unimplementable switches are likewise designated as "unimplemented, unclaimed". Point-by-point automatic selection depending on these mechanisms (v1 §3.2 item 5 "automatic selection/correction") transfers to R41 along with `TNCA_TUNE`.

## Modification 4: Performance Gate Criterion = Significant Gain + Target Reporting

A performance gate passes (`perf-pass`) if and only if: every v3 experiment for that gate completely measures all points of the frozen subset on the delivery bitstream image and delivery software identity; every point passes numerically with zero fallback; the comparison is accepted by [`tools/r40/ablate.py`](../tools/r40/ablate.py) (refusing missing points, zero-filling, mixed configurations, configuration discrepancies, missing provenance, mismatched timing definitions, clock-only differences, and warmup contamination); and the 95% paired bootstrap confidence interval lower bound of S_geo is > 1. The v2 thresholds are retained as **targets**, reporting "met/unmet" along with S_geo, confidence interval, and balanced-weighted geometric mean for each gate; unmet items along with bottleneck attribution transfer to R41 targets. Point-by-point regressions > 2% and > 5% are itemized with root causes, but do not serve as gate criteria (automatic selection transfers out with `TNCA_TUNE`). If the confidence interval lower bound is ≤ 1, the gate is marked `no-significant-gain` and does not pass.

`R40-01.device-time` remains a capability non-regression gate with unchanged criteria (paired bootstrap CI lower bound ≥ 1/1.05, per-point slowdown ≤ 5%).

The experiments, baseline, control, and candidate values for each performance gate are frozen in [`doc/r40_v3/plan.json`](r40_v3/plan.json):

| Gate | Experiment (Subset) | Baseline | Control → Candidate | v2 Target |
|---|---|---|---|---|
| R40-02.resident | TNCA_TRANSFER_POLICY (`graphs3`) | B40-S | per-kernel → resident | 1.5 |
| R40-02.batched | TNCA_SUBMIT (`small-kernel-graphs`) | B40-S, resident | single → batched | 1.2 |
| R40-03.wide-blocks | TNCA_MAX_BLOCK (`pointwise-reduction`) | B40-S | 16 → 64 (compile preflight rule below) | 1.3 |
| R40-03.gemm-tile | TNCA_GEMM_TILE (`gemm-reuse`) | B40-S | 16,16,16 → 16,64,16 (same as above) | 1.15 |
| R40-04.bandwidth | TNCA_RTL_OUTSTANDING (`streaming`) | B40-S | 1 → 8 | 1.5; directional bandwidth ≥ 50% bottleneck peak as target |
| R40-05.scaling | TNCA_ACTIVE_CORES (`compute`) | B40-N | 1 → 3; 1 → 2 | 2.1; 1.5 |
| R40-06.streaming-scaling | TNCA_ACTIVE_CORES, TNCA_RTL_MEM_PORTS (`streaming`) | B40-N; B40-S | 1 → 3 cores; 1 → 3 outlets | 1.8; 1.3 |
| R40-07.long-k | TNCA_PIPELINE (`long-k`) | B40-S | 1 → 2 | 1.2 |
| R40-08.prefill | TNCA_SDPA (`long-prefill`) | B40-S | decomposed → fused | 1.5; intermediate HBM bytes reduction ≥ 50% as target |
| R40-09.model | TNCA_ACTIVE_CORES (frozen four-layer model `prefill512` and `decode` points) | B40-N | 1 → 3 | 1.5; decode non-regression > 5% as target |

The timing definition for `R40-02.batched` is resident call latency, which must be measured under residency; hence both control and candidate set `TNCA_TRANSFER_POLICY=resident` with all other settings at B40-S; this is a clarification of measurement scope, not a relaxation.

Wide block and GEMM tile candidates are determined strictly by the pre-declared **compilation coverage** rule (without considering performance): offline compilation on the delivery tree (device access prohibited) adopts 64 if all points in the subset compile under 64, otherwise 32; GEMM tile sequentially evaluates 16,64,16 then 16,32,16. Preflight results are bound in [`evidence/r40/v3/compile_preflight/`](../evidence/r40/v3/compile_preflight/). If no candidate covers all points in a subset, the gate is marked `fail`; dropping points is prohibited.

## Modification 5: On-Board Timing Sampling

On-board device cycle counts are deterministic: the B39 rerun matches cycle-for-cycle on 77/77 points; repeated measurements for the same point across software identities and processes differ by ≤ 0.2% (e.g., raw GEMM4096 at 4,251,182,858 vs. 4,251,016,301 cycles); and device cycles between cold first call and steady state differ by ≤ 0.2% (across all 14 HBM points, the maximum difference is 0.19% on `attention_decode/large`). The v2 sampling protocol of 5 warmup runs + 30 timed runs (10 for long tasks) × 3 blocks provides no additional information for device cycle scopes while requiring tens to hundreds of hours of FPGA board time (a single R40-05 set alone exceeds 50 hours). Therefore, for device cycle scopes (`graph-makespan`, `device-graph-makespan`, `device-cycles-serial-grid`):

1. Each configuration for each point runs in an independent process. ArchBench's first (cold) invocation is recorded separately and never pooled into statistics; subsequent runs have 0 additional warmups.
2. Qualification measurement: 1 timed sample per configuration. If qualification samples for all configurations of that point take ≤ 1 s, each configuration collects 3 additional timed samples in independent processes as formal records; otherwise, the qualification record serves as the formal record (1 sample).
3. Single measurement block: all configurations for a given point run consecutively in an order randomized using frozen seed 20261010. Identical (point, complete knob set) configurations required across multiple experiments are measured only once and shared.
4. Statistics: each point takes the median of its formal samples; S_geo uses 2,000 paired bootstrap resamples (resampling points with seed 20261010).

Host wall-clock scopes (`request-latency`, `resident-call-latency`) exhibit host noise and retain the v2 sampling protocol: 5 warmup runs, 30 timed runs, 3 independent measurement blocks, and randomized interleaved A/B sampling within blocks.

Transmission watchdogs are declared parameters and must be identical on both sides; long tasks may increase `TNCA_DEADLINE` and `TNCA_HOST_TIMEOUT`, documented in the collection provenance.

## Modification 6: Leave-One-Out

Leave-One-Out (LOO: disabling individual features from B40-F, targeting ≥ 3% on target subsets) falls under "target reporting": it is measured and reported when board time permits; unmeasured items are marked `not-measured` and do not serve as R40 v3 sign-off criteria; the complete LOO table transfers to the R41 baseline.

## Modification 7: R40-10 Final Acceptance

`R40-10.acceptance` requires: all v3 required gates must pass; the delivery report [`doc/r40_report.md`](r40_report.md) must contain per-gate evidence, the performance target achievement table, deferred items with rationales, resource/timing/CDC closure, and the 90 MHz exception; and the final PR must be non-draft targeting `main`. Items 4 (GPU comparison) and 5 (independent clean-environment reproduction) of v1 §R40-10 remain mandatory requirements.

## Unchanged Provisions

Numerical tolerances and full workloads; correctness across all original points (`native77`, `HBM14`, `large24`, and the frozen models); three physical cores and real memory outlet traffic; HBM ownership and completion visibility; raw samples and provenance; missing or unmeasured items cannot PASS.

`make r40-v3-contract-check` verifies the contract freeze; `make r40-v3-acceptance` audits v3 evidence.
