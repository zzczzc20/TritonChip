# R41 Stage 2 Plan P1: Mechanisms and Policy Study

This study evaluates Stage 2 software policies and candidate mechanisms under
identical hardware conditions on the approved run-1 100 MHz image (`30e79769...`).
Simulation and preflight measurements run on the SystemC/TLM model on GB10;
board measurements are prepared as waiting manifests without unauthorized card access.

## 1. Reconcile X8 Scheduler Fallback

In earlier Stage 2 integration audits (`evidence/r41/stage2/integration/audit/README.md`),
target `tnca-sim-vrf-v5-x8` was omitted from `spec/r39_sched_model.json`, causing the compiler
scheduler to fall back to the default `flipflop` model instead of `banked2`.
Because both targets share the identical banked register file microarchitecture,
`tnca-sim-vrf-v5-x8` is now explicitly mapped to `banked2` in `spec/r39_sched_model.json`.

Re-running `tools/performance_harness/default_change_audit.py` comparing `core-vrf-sync-v5` (control)
against `core-vrf-sync-v5-x8` (candidate) across all 113 frozen points demonstrates:
- Total points: 113
- Unchanged: 113 (100%)
- Changed: 0 (0%)
- Emitted instruction text and launch structures are byte-identical across all non-FP8 kernels.
Evidence is archived in `evidence/r41/stage2/policy/profile-compare.json` and `x8_reconciliation.json`.

## 2. Defined Enabled Mechanism Groups

| Group | Mechanisms | Default State | Rationale |
|---|---|---|---|
| **G1: Scheduling & Layout** | 16x16x16 tile fragments, `banked2` scheduler model | Enabled by default | Conservative tile avoids excessive padding and accumulator trips on irregular shapes. |
| **G2: Pipelining & Transfer** | Mode B async execution (`TNCA_PIPELINE=1`) | Enabled by default | Overlaps transfer and compute within legal register limits without double-buffering bloat. |
| **G3: Multicore & Memory Service** | 3 cores active (`TNCA_ACTIVE_CORES=3`), bounded queues | Enabled by default | Scaled across 3 SLRs; bounded admission prevents queue overflow. |
| **G4: Graph Locality & DCO** | TMU window reuse & lifetime-directed bypass | Default OFF | Kept default-off until positive gain is demonstrated on board; delivered functionally. |

## 3. Matched Software-Policy A/B on SystemC Model

Using the SystemC/TLM model (`sim/build/r41/systemc-tlm-grid`) on GB10:

| Policy Comparison | Configuration A | Configuration B | Metric | Outcome |
|---|---|---|---|---|
| **Fragment Tile Width** | 16-wide columns (default) | 32-wide columns | Device cycles (`17x65x33`) | 16-col: 60,369 cycles vs 32-col: 151,203 cycles (+150.5% slower). Conservative 16-wide default justified. |
| **Scheduler Model** | `banked2` (reconciled) | `flipflop` (fallback) | Code text matching | 113/113 identical with `banked2`; 0/113 identical with `flipflop`. Reconciled model verified. |
| **Pipeline Mode** | Mode B | Single-role | Device cycles | Mode B: 60,369 cycles vs Single-role: 63,840 cycles (+5.7% overhead without overlap). |

## 4. Primary Group Leave-One-Out (LOO) Ablations and Interactions

From `evidence/r41/stage2/policy/p1_policy_ab_report.json`:

- **LOO-1 (Fragment Layout):** Forcing 32-wide columns increases device cycles from 60,369 to 151,203 (+150.5% regression on irregular shapes). Conservative tile remains nonregressive.
- **LOO-2 (Multicore):** Running on 1 core instead of 3 cores increases device cycles from 60,369 to 175,070 (+190.0% regression), establishing ~2.9× scaling across the 3 SLRs.
- **LOO-3 (Pipelining):** Forcing single-role execution increases device cycles from 60,369 to 63,840 (+5.7% regression), demonstrating transfer/compute overlap benefit in Mode B.
- **Justified Interaction (G1 Layout × G2 Pipeline):** 32-wide columns combined with single-role pipeline results in 159,880 cycles (measured ratio 2.648× vs linear sum expectation 2.562×, interaction coupling +8.6% due to compounded register pressure).

## 5. Candidate Memory Service Sensitivity Analysis

From `evidence/r41/stage2/policy/p1_memory_service_report.json`:
- 1 port, depth 1: 42,800 cycles (baseline)
- 1 port, depth 8: 39,120 cycles (0.914× baseline, 8.6% speedup from overlapping lookup and issue)
- 3 ports, depth 8: 36,540 cycles (0.853× baseline, 14.7% speedup from multi-SLR distribution)
As identified in the R40 attribution, gains saturate because core-side home-slice latency dominates raw outlet throughput.

## 6. Board Waiting Manifest and Lead Instructions

The board comparison is registered in `spec/r41/stage2/board/s2-policy-ab.json` and guarded by `evidence/r41/stage2/board/launches/p1_policy_queue.sh`.
It tests all policy variations on the Alveo U280 without programming or resetting the card.
Dry-run verification passes locally (`card_access=false`).

When card time is authorized by the lead, the execution command on `acf3019` is:

```sh
cd ~/r41_accept_SOURCE_COMMIT
setsid nohup ~/tc_env/torchenv/bin/python tools/performance_harness/launches/acceptance_queue.py \
    --root "$PWD" \
    --out ~/r41_accept_p1_SOURCE_COMMIT \
    --python ~/tc_env/torchenv/bin/python \
    --manifest spec/r41/stage2/board/s2-policy-ab.json \
    > ~/r41_accept_p1_SOURCE_COMMIT.log 2>&1 < /dev/null &
```
