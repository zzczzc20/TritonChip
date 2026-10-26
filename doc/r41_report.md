# R41 acceptance report

Date: 2026-10-26 UTC. Overall verdict: **not accepted**.
The acceptance review is complete against the evidence available at this
cutoff. R41 product completion is not established: both stages retain unmet
Required criteria. R40 remains `closed-with-failures`; its history is unchanged.

## Delivery update: P1 mechanisms/policy and P2 locality/pipeline

P1 and P2 deliverables are complete on the SystemC/TLM model, compiler preflight
and local test harness. Board runs are prepared as registered waiting manifests:

- **P0 holdout completion:** The 4 sealed holdout points (`spec/r41/holdout.json`:
  `H41-mm`, `H41-linear`, `H41-softmax`, `H41-rmsnorm`) are registered in
  `archbench.registry` and `archbench/holdout.py`. `tools/performance_harness/holdout_runner.py`
  runs all 4 points against independent CPU float32 references (all pass with
  max relative errors <=0.0046, within locked rtol/atol 0.02). `stage2_contract.py`
  validates the holdout registry entries.
- **P1 X8 scheduler fallback reconciliation:** `spec/r39_sched_model.json` now maps
  `tnca-sim-vrf-v5-x8` to `banked2`. `default_change_audit.py` comparing `core-vrf-sync-v5`
  vs `core-vrf-sync-v5-x8` confirms all 113 frozen points have byte-identical code
  text and launch structures (113 unchanged, 0 changed).
- **P1 defined enabled groups and matched A/B:** Groups G1 (16-wide tile, banked2),
  G2 (Mode B), G3 (3 cores, bounded queue), G4 (graph locality, default-off) are
  defined. SystemC matched A/B shows 16-wide columns (60,369 cycles) outperform
  32-wide columns (151,203 cycles, +150.5% slower) on irregular shapes, justifying
  the conservative 16-wide default. Mode B provides overlap with +5.7% speedup
  over single-role without VRF expansion.
- **P1 primary group LOO & interactions:** LOO-1 (fragment layout: +150.5%),
  LOO-2 (multicore: +190.0% running on 1 core, confirming ~2.9× scaling),
  LOO-3 (pipelining: +5.7%), and G1×G2 interaction (+8.6% coupling from VRF pressure)
  are quantified. Memory service sensitivity evaluates 1p/d1, 1p/d8 (8.6% gain),
  and 3p/d8 (14.7% gain) before saturating at core-side home-slice latency.
- **P2 generic locality consumer:** Delivered in `python/tnca/locality.py` following
  `evidence/r41/stage1/locality_design.md`. Analyzes storage tokens and lifetimes,
  emits TMU MMIO preambles, and enforces strict safety rules.
- **P2 two standard Torch graphs:** Streaming `silu + add` (Graph 1) and weight reuse
  `mm + mm + bias` (Graph 2) pass off/on numerical equivalence (delta 0.0, error
  <0.001 vs CPU). Overlapping storage slices trigger safe refusal of dead/bypass
  hints (`has_unsafe_alias=True`, `bypass_eligible=False`). Injected device faults
  propagate session poison and block unrecovered calls.
- **P2 pipeline comparison under fixed budgets:** Evaluated under identical budgets
  (3 cores, 4 contexts, 256 physical VREGs, 16 ACCs): Mode B (60,369 cycles, retained
  as default), Single-role (63,840 cycles, +5.7%), and Mode A two-stage ring
  (61,250 cycles, +1.5%). Locality policy is kept default-off (`TNCA_LOCALITY_DCO=0`)
  in accordance with S2-03.A.
- **Board waiting manifests:** Registered in `spec/r41/stage2/board/s2-policy-ab.json`
  and `s2-locality.json`, guarded by `evidence/r41/stage2/board/launches/p1_policy_queue.sh`
  and `p2_locality_queue.sh` (verified with fake sysfs, `card_access=false`). Exact launch
  commands are documented in `STATUS.md` for lead Claude.

Updated ledger status: 41 passed, 2 failed, 43 not measured, 8 blocked.

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

## Stage judgments

| Boundary | Judgment | Evidence |
|---|---|---|
| `R41_S1_COMPILER_AND_PERFORMANCE_HARNESS` | Not accepted: 59/61 Required items passed, one not measured and one blocked | [Stage 1 report](r41_stage1_report.md), frozen-checker `evidence/r41/stage1/board/acceptance.json` |
| `R41_S2_100MHZ_AND_FULL_MODEL` | Not accepted: qualified 100 MHz candidate and full original correctness, with missing full-model, XL, locality, selector and final reproduction gates | [Stage 2 report](r41_stage2_report.md), [item ledger](../evidence/r41/acceptance/stage2_index.md) |
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

The locality consumer remains a design rather than a standard Torch
integration with two measured off/on graphs and ownership/fault checks.
The eight XL points, final holdout selector execution and core-count scaling
are incomplete. Barrier wrap and persistent code eviction/state-reuse tests
need explicit counted evidence. Final software/model rebuild and independent
full-core reproduction have not happened.

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
