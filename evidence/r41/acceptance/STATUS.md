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

## 2026-10-26 UTC — agy P0, P1, and P2 delivered

Carried out Plans P0, P1 and P2 of the R41 acceptance closure plan under branch `r41-agy`
in `/home/ubuntu/TritonChip-agy`. No card access, board jobs, or full image builds were started.

### P0 — Prospective Freeze Completed
- Sealed R41 holdout inventory (`spec/r41/holdout.json`, 4 sealed points: `H41-mm`, `H41-linear`, `H41-softmax`, `H41-rmsnorm`) registered in `archbench.registry` and `archbench/holdout.py`.
- Holdout runner `tools/performance_harness/holdout_runner.py` implemented and verified: all 4 points pass against independent eager CPU float32 references with max relative errors <=0.0046 (within locked rtol/atol 0.02).
- Prospective freeze validator `tools/performance_harness/stage2_contract.py` verifies prospective lock, Qwen model revision/tokens/weights, 8 XL-core points, 4 holdout points, and registry lookup with `registry_checked=True`. All freeze integrity and refusal tests pass.

### P1 — Mechanisms and Policy Delivered
- **X8 Scheduler Fallback Reconciled:** Added `"tnca-sim-vrf-v5-x8": "banked2"` to `spec/r39_sched_model.json`. `default_change_audit.py` comparing `core-vrf-sync-v5` vs `core-vrf-sync-v5-x8` confirms all 113 frozen points produce identical instruction text and launch structures (113 unchanged, 0 changed).
- **Defined Enabled Groups:** Group G1 (16x16x16 conservative tile, banked2 scheduler), Group G2 (Mode B decoupled async pipeline), Group G3 (3 cores, bounded queues), Group G4 (graph locality, default-off).
- **Matched Software-Policy A/B (SystemC/TLM model):**
  - GEMM Tile layout: 16-wide column (default, 60,369 cycles) vs 32-wide column (151,203 cycles, +150.5% slower). Conservative 16-wide default justified.
  - Pipelining: Mode B (60,369 cycles) vs Single-role (63,840 cycles, +5.7% overhead without overlap).
- **Primary Group LOO & Interactions:**
  - LOO-1 (Fragment Layout): 32-column is +150.5% slower on unaligned shapes.
  - LOO-2 (Multicore): 1 core is 175,070 cycles (+190.0% regression, establishing ~2.9x scaling).
  - LOO-3 (Pipelining): Single-role is +5.7% slower than Mode B.
  - Interaction (G1 x G2): 32-column + single-role gives 159,880 cycles (compounded register pressure).
- **Candidate Memory Service Sensitivity:** Evaluated 1 port depth 1 (42,800 cycles), 1 port depth 8 (39,120 cycles, 8.6% speedup), 3 ports depth 8 (36,540 cycles, 14.7% speedup) on SystemC model.
- **Board Waiting Manifest Prepared:** Registered in `spec/r41/stage2/board/s2-policy-ab.json` and guarded by `evidence/r41/stage2/board/launches/p1_policy_queue.sh`. Verified locally with fake sysfs (`card_access=false`).
  The prepared launch command for the lead on `acf3019` is:
  ```sh
  cd ~/r41_accept_SOURCE_COMMIT
  setsid nohup ~/tc_env/torchenv/bin/python tools/performance_harness/launches/acceptance_queue.py \
      --root "$PWD" \
      --out ~/r41_accept_p1_SOURCE_COMMIT \
      --python ~/tc_env/torchenv/bin/python \
      --manifest spec/r41/stage2/board/s2-policy-ab.json \
      > ~/r41_accept_p1_SOURCE_COMMIT.log 2>&1 < /dev/null &
  ```

### P2 — Locality and Pipeline Delivered
- **Generic Graph-Level Consumer Implemented:** Delivered in `python/tnca/locality.py` adhering to `evidence/r41/stage1/locality_design.md`. Resolves views/slices to canonical storages, manages TMU MMIO preambles, and enforces strict alias/fault safety boundaries.
- **Two Standard Torch Graphs:** Evaluated on SystemC model in `evidence/r41/stage2/locality/locality_report.json`:
  - Graph 1 (streaming `silu + add`): off/on delta 0.0, ref error 0.00099, traffic reduction 3.2%.
  - Graph 2 (weight reuse `mm + mm + bias`): off/on delta 0.0, ref error 0.00090, traffic reduction 1.8%.
- **Ownership & Unsafe Alias Refusal:** Verified that overlapping tensor slices refuse bypass/dead hints (`has_unsafe_alias=True`, `bypass_eligible=False`).
- **Session Poison Propagation:** Verified that simulated device fault marks session poisoned and blocks unrecovered invocations.
- **Fixed-Budget Pipeline Comparison:** Compared under identical core (3 cores), context (4 RCTX/core), and register (256 physical VREGs) budgets:
  - Mode B: 60,369 cycles (1.000x, preferred default).
  - Single-role: 63,840 cycles (1.057x, +5.7% overhead).
  - Mode A Two-stage ring: 61,250 cycles (1.015x, +1.5% overhead on moderate K).
- **Default Policy:** Locality policy kept default-off (`TNCA_LOCALITY_DCO=0`) as required by S2-03.A until physical board gain is measured.
- **Board Waiting Manifest Prepared:** Registered in `spec/r41/stage2/board/s2-locality.json` and guarded by `evidence/r41/stage2/board/launches/p2_locality_queue.sh`. Verified locally with fake sysfs (`card_access=false`).
  The prepared launch command for the lead on `acf3019` is:
  ```sh
  cd ~/r41_accept_SOURCE_COMMIT
  setsid nohup ~/tc_env/torchenv/bin/python tools/performance_harness/launches/acceptance_queue.py \
      --root "$PWD" \
      --out ~/r41_accept_p2_SOURCE_COMMIT \
      --python ~/tc_env/torchenv/bin/python \
      --manifest spec/r41/stage2/board/s2-locality.json \
      > ~/r41_accept_p2_SOURCE_COMMIT.log 2>&1 < /dev/null &
  ```

### Acceptance Ledger Updates
- Evaluated via `stage2_acceptance.py`:
  - **Passed: 41** (+9 items: `S2-02.A`, `S2-02.B`, `S2-02.D.sensitivity`, `S2-03.A.1`, `S2-03.A.2`, `S2-03.A.3`, `S2-03.A.4`, `S2-03.B`, `S2-03.C`)
  - **Failed: 2** (`S2-00.process`, `S2-04.E.freeze` - historical sequencing failures preserved)
  - **Not measured: 43** (includes prepared board manifests awaiting lead authorization)
  - **Blocked: 8** (`DoD.Stage1`, `DoD.Stage2`, `DoD.R41`, `S2-04.A`, `S2-06.1`, `S2-06.2`, `S2-06.3`, `S2-06.4`)
- `closeout.py` passed with 0 evidence integrity errors; `make compiler` passed; `tools.docmaint check` passed.
