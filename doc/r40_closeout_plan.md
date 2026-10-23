# R40 Critical Delivery Path and Consolidated Verification Plan

Updated: 2026-10-15; reviewed code baseline: `3ee636e` (with uncommitted autotune foundation and 2026-10-15 checkpoints). The acceptance contracts remain [r40_tasks_and_acceptance.md](r40_tasks_and_acceptance.md) and [`spec/r40/plan.json`](../spec/r40/plan.json); this plan does not alter any frozen thresholds.

## Current Blockers

Executing [`tools/r40/acceptance.py`](../tools/r40/acceptance.py) in read-only mode evaluates 36 required gates, with 5 passing:
- R40-00 is closed (`R40-00.protocol`, `R40-00.b39-reproducible`, `R40-00.frozen`).
- R40-01 passes `device` and `mixed-device`, but `device-time` fails.
- All remaining work packages remain open; existing RTL or on-board checkpoints do not automatically equate to complete gate sign-off.
- R40-05 through R40-09 have formal Make targets; functional runners for 08 and 09 are connected to evidence evaluation, but full implementations, board tests, and performance evidence remain missing, preventing package sign-off.

For R40-01, full native execution achieves 77/77 numerical correctness, but yields $S_{\text{geo}} = 0.5872$ relative to B39. A CPU vs. native control on the same HBM image yields ~0.999 across 75 successful common points, isolating the primary cost to the memory path. Retests for two points following code-cache epoch fixes are archived separately and cannot be spliced into a formal 77-point run of the same source. The 512 KiB local/HBM hybrid image remains a candidate undergoing evaluation and has not yet demonstrated elimination of the regression.

## Four Implementation Batches and Exit Criteria

### Execution Schedule: What to Do Next

Scheduling follows hardware dependencies rather than serializing by work package number. Board experiments run serially, while jump-host software/RTL verification runs in parallel with FPGA server builds. The following sequence defines execution priority, not a pass prediction:

| Order | Single Implementation Boundary | Batch-Ending Verification | Outcome Deciding Next Steps |
|---|---|---|---|
| A: Immediate | Collect existing hybrid and 3-core builds; review ownership/code2 OOC and placement; fix autotune device logging interface, complete 08/09 runners and manifests | Gate entry points and failure propagation; timing/cache/budget checks; native candidate numerics, zero staging, log isolation | Complete entry points; traceable candidate hardware identities; clear performance regressions and hardware risks |
| B: Pre-Final RTL | Confirm bottlenecks from standard graph counters; consolidate wide ingress/concurrency, matrix service time, ownership/fault, atomic/fence, and locality/DCO; refine compiler resources and spilling | Affected contracts, resource boundaries, random backpressure/faults, core/port matrices, and genuine overlap; followed by OOC/placement | Closed RTL functionality, viable resources/timing; freeze final configuration and launch full link |
| C: During Link | Complete standard attention, KV/bucket, replay/1000-call stress, budget sharing and persistent cache, autotuning integration, and large model; prepare all A/B/LOO run manifests | Corresponding functionality and lifetime matrices; standard Torch fullgraph / zero fallback; sample and configuration identity audits | Software functionality and formal experiment scripts ready, avoiding waiting on software after bitstream completion |
| D: Final Board Test | Run full correctness, performance, ablations, interactions, and GPU controls on the same frozen image grouped by mechanism | Frozen 5 warmup / 30 timed / 3 blocks (with registered exceptions for long tasks); isolated sessions; final CI and all 36 gates | All required gates PASS to close R40; failure loops back to specific mechanism without lowering thresholds |

Checkpoints through 2026-10-15:
- **Hybrid image**: Completed timing/CDC audits and board loading. Three diagnostic points pass numerics; `add` still regresses >5% against B39, while `transpose` initial measurements diverge from repeated runs (152,418 vs. 90,176 cycles), retaining raw negative results.
- **Three-core link & placement**: Ownership/code2 OOC completed; actual DCP exposed that the broadcast code queue was merged and lacked tile 1 physical register stages. Candidate `09ebe06` preserves independent queues, passed 6-configuration RTL regression, completed synthesis, and passed the full hierarchy/SLR assignment contract check with all 60 packaged RTL sources matched byte-for-byte. A 100 MHz `SSI_SpreadSLLs` link was started following a previous routing failure at congestion level 7.
- **Attention (R40-08)**: Functional runner `make r40-attention-checks` is in place. Online SDPA templates were extended to 16/32/64-dimension FP16/BF16 heads; native decode/mask/GQA/MQA functional batches passed 6/6 with zero staging. Long-sequence performance and full model validation remain open.
- **Autotune (R40-09)**: Autotune device event timing fixes verified policy (9/9), grid clocks (3/3), and native measurement. Persistent candidate cache with atomic publication and corruption recovery passed 14/14 policy checks. Holdout evaluation, worker timeout enforcement, and large model benchmarks remain open.

### Test Budget and Impact Scope

- At the start of each batch, define modified files, affected interfaces, mandatory regression targets, and exit criteria. Within a batch, run only necessary syntax/build checks or blocking diagnostics; test related functionality collectively upon completion.
- Runtime/timing/cache modifications focus on events, epoch/recovery, replay, budgets, and native graph audits. Compiler/layout modifications focus on resource/unsupported boundaries, descriptor/barrier/tail handling, legacy contracts, and corresponding standard Torch graphs.
- RTL modifications focus on ownership/atomic/fence, random backpressure/out-of-order/fault/reset, and actual DMA/MMA overlap, followed by affected core/port matrices. Perform full routing only after RTL freeze. Software changes do not rerun unrelated full FPGA builds.
- For long tests, maintain only one instance per task, recording PID, source, configuration, and output directory. Retain raw logs upon failure; rerun only affected groups after relevant fixes. Push CI once per batch, avoiding pure documentation pushes while CI is running to prevent cancelling runs in progress.
- Formal performance reuses matched baseline samples from the same image, configuration, and timing boundary; reuse only when sampling protocols permit and all identities match, without splicing results across different sources or images. A frozen full-suite acceptance run must still execute; local regressions cannot substitute for final tests.

Progress is tracked across four stages: "executable gate entry point → complete functional evidence → compliant on-board performance → independent reproduction". Each gate maps to raw logs, samples, and an identity manifest. Work packages submit sign-off records separately once all gates pass, and intermediate implementation batches are committed promptly. The key bottlenecks remain device regressions and final hardware capability/timing closure; no fixed delivery date is promised without supporting evidence.

### Batch 1: Acceptance Entry Points, Timing, and Hardware Feasibility

1. Connect genuine gate runners, evidence validation, and failure propagation for R40-05 through R40-09; reuse existing functional tests without creating targets that merely print PASS. Explicitly mark missing board measurements as `not-measured`. Run a work package command once per execution, judging all gates it provides.
2. Unify experiment manifests: correlate source, artifacts, image SHA, clocks, resources, raw samples, and workload definitions. Correct evidence descriptions that mistakenly describe HBM residency as on-chip residency.
3. Maintain active hybrid single-core and 3-core links; verify source files and new command/reply placement after code-upload 2-stage pipelined XO completes. The legacy code0 3-core image serves for routing feasibility and diagnostics; the final code2 configuration requires full routing and board-level verification.
4. Fix the root cause of the original HBM capacity host SIGSEGV while preserving failure logs; long-waiting subtests passing cannot replace the original full suite test. Collect actual placement, timing, CDC, and connectivity to confirm the address contract for three memory outlets sharing BOs.

Exit criteria: All gates have executable entry points; image/experiment identities are traceable; 3-SLR implementation feasibility is clarified; primary regression costs are supported by actual counters and worst-point diagnostics. This batch does not sign off unmeasured performance.

### Batch 2: Consolidated Closure of Storage, Compilation, and Execution Semantics

Implement gaps across R40-02 through R40-07 according to dependencies: DMA/cache ownership, atomic/fence, alias/lifetime/dead-hint, locality/DCO; descriptor/barrier resource lifetimes, retile/spill; generic asynchronous pipelines, layout/pack, and matrix policies. First identify bottlenecks from standard Torch artifacts and device counters before implementing matrix throughput changes, avoiding optimizing microbenchmarks in isolation.

Consolidated verification on the same source covers: resources/contracts, ownership fault matrices, 1/2/3/4/8 cores, and varying ports/latencies/backpressure; verify actual DMA/MMA overlap. Evidence already passing with unchanged dependencies (8-core, >4096 grid, code-link, etc.) is preserved without repeating lengthy simulations each time. Rerun related regressions when modifying corresponding boundaries. Commit each completed work package separately.

Exit criteria: Functional gaps closed, unified capability/cost models and toggles complete, candidate RTL passes consolidated regressions; freeze final RTL configuration and launch full route. Subsequent software and experimental work can proceed during routing; explicitly deprecate old candidate evidence when hardware defects require fixes.

### Batch 3: Standard Path Performance, Attention, and Applications

1. Run worst-case transpose, streaming, compute scaling, long-K, and long-sequence attention diagnostic points on board first to confirm optimization direction. Diagnostics do not substitute for the frozen full subset.
2. On the same final image, run frozen controls with `active_cores=1/2/3` and single/multiple outlets; keep total PC/cache budget constant before measuring capacity expansion controls. Record per-core load, per-outlet traffic, frequency, and full graph makespan.
3. Separately measure A (frozen CPU migration), B (native per-kernel staging), and C (native resident), controlling remaining variables per mechanism requirements; complete replay, physical copy overlap, and 1000-call stress. Account for CPU/native identity and residency gains separately.
4. Complete online attention, GQA/MQA/mask/long-sequence, and 64-step KV; verify absence of $L \times L$ intermediate tensors and measure actual intermediate HBM bytes. Implement budgeted autotuning; freeze holdout and $\ge 4$-layer, $\ge 64$ MiB weight, $\text{prefill} \ge 512$, $\text{decode} \ge 64$ models.
5. Execute formal A/B per frozen sampling: 5 warmup, 30 timed, 3 blocks, randomly interleaved; long tasks use frozen exception of 10 timed calls. Report median/p95 and paired bootstrap 95% CI. Full suites are not reduced in advance; timing and performance passes are judged by existing protocols.

Exit criteria: All required performance thresholds, functionality, and documentation sub-requirements in the table below pass; non-contributing mechanisms are disabled or removed, preserving negative results. Passing one optimization cannot compensate for failure in another.

### Batch 4: Final Full Regression and Independent Reproduction

After freezing final software and image, collectively run conformance, tiny/small/medium, full original large, frozen HBM core suite, models, all independent A/B, LOO $\ge 3\%$, and required interactions. GPU comparisons maintain identical Torch points, precision, and timing boundaries. Prepare final report, raw sample index, resource/timing data, HBM model calibration, 8-core empirical results, and 16/32-core scaling budgets; verify clean-environment rebuild and on-board reproduction in an independent session. Finally, `make r40-acceptance` judges all 36 gates.

## Work Packages and Mandatory Closeout Thresholds

| Package / Gate Suffix | Required Evidence and Performance Thresholds for Closeout |
|---|---|
| 00 protocol / b39-reproducible / frozen | Already passed; protocol, 77/77 reproduction, and freeze checks remain valid |
| 01 device / mixed-device / device-time | Native 77/77, mixed-device/worker, $S_{\text{geo}} \ge 1.0$, zero unresolved >5% regressions |
| 02 zero-transfer / resident / batched / stress | Frozen residency correct, $S_{\text{geo}} \ge 1.5$; batch/replay $\ge 1.2$; 1000 calls with async lifetime/recovery; physical DMA overlap |
| 03 tiles / wide-blocks / gemm-tile / contract | 16/32/64/128, tail/layout/alias, retile/spill/lifetime; wide $\ge 1.3$, GEMM tile $\ge 1.15$; legacy contract |
| 04 hbm / bandwidth / robustness | $\ge 64$ MiB standard graph, full 32 PC address/capacity; read/write/copy each $\ge 50\%$ of frozen bottleneck peak; wide ingress/concurrency $\ge 1.5$; error/reset, model calibration, on-board timing/CDC |
| 05 multicore / scaling / dispatch | Real SLR0/1/2 active concurrently; 3 cores $\ge 2.1$, 2 cores $\ge 1.5$; full grid/fault, 1/2/3/4/8-core semantics |
| 06 nuca / streaming-scaling / locality / semantics | ownership/atomic/fence/DMA/alias/dead-hint; 3-core streaming $\ge 1.8$, multi-outlet $\ge 1.3$; locality/DCO $\ge 1.1$ with at least one metric of actual bytes reduced by $\ge 20\%$; independent ablation/fairness |
| 07 pipeline / long-k / matrix / policy | Actual overlap, stage/epoch/fault; long-K $\ge 1.2$, matrix $\ge 1.2$; automatic policy with zero unresolved >5% regressions across full suite; resources, seconds, LOO |
| 08 attention / prefill / decode | Full attention/model with long-sequence/mask/GQA/MQA/64-step; prefill $\ge 1.5$, intermediate HBM bytes reduced by $\ge 50\%$; zero weight/KV retransmission, single compilation within bucket |
| 09 autotune / holdout / suites / model | $\le 16$ candidates/kernel, $\le 10$ minutes/graph; holdout $\ge 1.1$, regret p90 $\le 15\%$; complete suite; specified large model 3 cores $\ge 1.5$ with decode regression $\le 5\%$; 8-core empirical proof / scaling budget |
| 10 acceptance | Final identical source/image, complete A/B/LOO/interactions, GPU comparison, reproduction package with independent session; all required gates PASS |

## Resource, Testing, and Commit Discipline

- Jump host runs heavy RTL/compilation matrices; FPGA server runs Vivado/Vitis, on-board functionality, and performance. Board experiments run serially; independent compilation/software preparation can run in parallel. Check real memory and existing jobs before starting to avoid stacking multiple full routes concurrently.
- Retain PID, source, and output directories for running long simulations, reading incremental state; do not restart jobs that are alive and have no basis for failure. New board tests for identical purposes provide early diagnostic evidence but cannot impersonate the completion of long simulations.
- Run consolidated verification following a batch of changes; perform a single final frozen full-suite regression. Do not repeat testing purely for form when a prior round has complete evidence and dependencies are unchanged.
- Commit per work package or reviewable milestone; batch pushes so CI can complete, preventing sequential pushes from triggering cancel-in-progress. Rerun affected tests after fixing CI; the final HEAD must pass CI. Documentation commits must not mark unclosed packages as accepted.
- Execution checkpoints record job status, next actions, and gate gaps; update this document and evidence registry per batch. When performance results fail, return to the corresponding implementation batch without modifying frozen standards.

The shortest path is to first resolve 3-SLR resources/timing and single-core memory regressions while completing acceptance entry points; advance compiler and model work while awaiting route completion; and finally obtain all performance and ablation evidence collectively on a single frozen candidate. Current performance and routing have not yet converged, so no fixed delivery timeframe can be promised.
