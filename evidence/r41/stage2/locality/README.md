# R41 Stage 2 Plan P2: Locality and Pipeline Study

This study implements the generic graph-level storage, alias and lifetime consumer
designed in `evidence/r41/stage1/locality_design.md` and evaluates it alongside
fixed-budget pipeline strategies under identical hardware conditions.

## 1. Generic Graph-Level Locality / DCO Consumer

The consumer is implemented in `python/tnca/locality.py`:
- **`StorageAnalyzer`**: Traces underlying physical storage tokens for graph tensors, resolving views, transposes and slices to canonical storages.
- **Lifetime & Alias Safety Rules**:
  - Graph I/O tensors and persistent states are preserved; never marked dead prematurely.
  - Overlapping slices or partial-line / unsafe aliases refuse dead-block and bypass hints (`bypass_eligible = False`), maintaining ordinary caching.
  - Proven internal temporaries read exactly once (`nAcc = 1`) with whole cache-line accesses receive bypass hints.
  - Temporaries read multiple times (`nAcc > 1`) receive replacement assistance hints.
- **`LocalityController`**: Generates and submits TMU MMIO preambles (reset, granule, range registration, policy enable) and epilogue cleanup through the public Session.
- **Session Poison Guard**: Device faults propagate session poison immediately; restoration helpers are never launched into a faulted device, and subsequent calls on the poisoned session are refused.
- **Policy Flag**: Controlled by `TNCA_LOCALITY_DCO` (default `0` / OFF). Functionality is fully delivered and verified, but remains default-off in accordance with S2-03.A.

## 2. Standard Torch Graphs and Off/On Correctness

Evaluated on the SystemC/TLM model on GB10 in `evidence/r41/stage2/locality/locality_report.json`:

| Graph | Workload Structure | Off/On Delta | Error vs CPU Ref | Off Cycles | On Cycles | Traffic Delta | Numerical Status |
|---|---|---|---|---|---|---|---|
| **Graph 1: Streaming** | Elementwise `silu(x) + w`; temporary `y` read once | 0.0 | 0.000993 | 12,450 | 12,380 | -3.2% traffic | **passed** |
| **Graph 2: Reuse** | Sequential `mm(x, w)` and `mm(mm1, w) + b`; `w` read twice | 0.0 | 0.000902 | 24,800 | 24,710 | -1.8% traffic | **passed** |

Outputs are bitwise identical between locality OFF and ON, and within the 0.02 tolerance against eager CPU float32 references.

## 3. Ownership and Fault Tests

- **Overlapping Storage Alias Refusal**: Tested with two overlapping tensor slices (`[0:64]` and `[32:96]`) allocated from the same backing storage. `StorageAnalyzer` correctly detects interval collision, flags `has_unsafe_alias = True`, and refuses bypass eligibility (`refusal_reason: storage_0: overlapping slice or unsafe alias`).
- **Session Poison Propagation**: Injected simulated hardware fault during graph execution. `LocalityController` catches the fault, immediately marks `session.poisoned = True`, drains, and blocks subsequent executions with `LocalityRefusal`.

## 4. Pipeline Comparison under Fixed Budgets

Evaluated under identical core (3 cores), context (4 RCTX/core), register (256 physical VREGs), and accumulator (16 ACCs) budgets:

| Pipeline Mode | Description | Device Cycles | Ratio vs Mode B | Peak VREG / Role | Decision & Rationale |
|---|---|---|---|---|---|
| **Mode B (Default)** | Decoupled asynchronous transfer/compute within fixed single-role register file | 60,369 | 1.000× | 32 | **Retained as default**: Achieves latency hiding without double-buffering register file inflation. |
| **Single-role** | Synchronous lockstep issue; transfer completes before compute | 63,840 | 1.057× | 28 | Rejected: 5.7% slower due to idle compute during transfers. |
| **Mode A Two-stage Ring** | Software-pipelined ring with double-buffered window registers | 61,250 | 1.015× | 56 | Secondary candidate: 1.5% slower on moderate K depths due to prologue/epilogue drain overhead and high VRF pressure; kept available for long-K streaming kernels. |

## 5. Board Waiting Manifest and Lead Instructions

The board comparison is registered in `spec/r41/stage2/board/s2-locality.json` and guarded by `evidence/r41/stage2/board/launches/p2_locality_queue.sh`.
Dry-run verification passes locally (`card_access=false`).

When card time is authorized by the lead, the execution command on `acf3019` is:

```sh
cd ~/r41_accept_SOURCE_COMMIT
setsid nohup ~/tc_env/torchenv/bin/python tools/performance_harness/launches/acceptance_queue.py \
    --root "$PWD" \
    --out ~/r41_accept_p2_SOURCE_COMMIT \
    --python ~/tc_env/torchenv/bin/python \
    --manifest spec/r41/stage2/board/s2-locality.json \
    > ~/r41_accept_p2_SOURCE_COMMIT.log 2>&1 < /dev/null &
```
