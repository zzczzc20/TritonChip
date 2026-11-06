# R3 Status: v2 Integration and v1 Historical Progress

Current contract: [Phase 3 (R3 v2)](phase3_tasks_and_acceptance.md). Status: **ACCEPTED** (`R3_V2_CORE_VRF_ACCEPTED`).
Replan basis: [2026-09-20 replan review](r3_replan_review_2026-09-20.md).
Prerequisites: [R1 sign-off](r1_signoff.md) (**R1 is ACCEPTED**), [R2 status](r2_status.md).

R3 v2 focused on the real single-core integration of the synchronous finite-port VRF.
All Gates G0–G7 (baseline, VRF, client integration, return/writeback, E2E, counters/micro-model, structural synthesis/regressions, mutations/evidence/release) are **VERIFIED / PASS**.
The milestone was achieved on the locked Linux primary environment (`Linux-6.17.0-1019-aws-x86_64`, Icarus Verilog 12.0, Yosys 0.33, Triton 3.8.0, Python 3.12.3) at commit `33dbb375828e` on a clean working tree, run ID `1789939808-33dbb375828e`, recorded in [`doc/evidence/r3v2/acceptance-linux.json`](evidence/r3v2/acceptance-linux.json).

All ten checkers passed: G0 plan 3/3, G1 rf 3/3, G2 core 3/3, G3 flow 3/3, G4 e2e 3/3, G5 count 2/2 + model 1/1, G6 syn 3/3, G7 mut 2/2 + rel 1/1. All 24 `R3V2-*` criteria have independently readable verification results.

**The verdict is verified independently.** `tools/r3v2/verify_evidence.py` evaluates the evidence without relying on the `verdict` or `passed` fields, recomputing the outcome from the frozen inventory and raw checker outputs: all 24/24 criteria are complete, summary and raw results match, frozen file hashes match, and commit/clean-tree provenance is verified. The checkers required 1629s total elapsed time, with an additional 304s for mutation runs accounted for separately.

## R3 v2 (Current Contract) Gate Status

Contract: [phase3_tasks_and_acceptance.md](phase3_tasks_and_acceptance.md) v2.0.
Sign-off identifier: `R3_V2_CORE_VRF_ACCEPTED`. **Status: ACCEPTED**, executed on the locked Linux primary environment (`Linux-6.17.0-1019-aws-x86_64`, iverilog 12.0, yosys 0.33, Triton 3.8.0) at commit `33dbb375828e` on a clean tree, run ID `1789939808-33dbb375828e`, recorded in [`doc/evidence/r3v2/acceptance-linux.json`](evidence/r3v2/acceptance-linux.json).

| Gate | Status | Verified Criteria |
|---|---|---|
| G0 Baseline / Inventory / Budget | **DONE** | R3V2-BASE-01/02/03. `spec/r3v2_contract.json` and `spec/r3v2_inventory.json` frozen; checker rejects eight malformed manifests before launching RTL; Linux scout measured 409–518 cycles/s, with projected execution of ~139k cycles / ~370s within budget |
| G1 VRF Wrapper / Initialization | **DONE** | R3V2-RF-01/02/03. Per-lane valid bits allow architectural reads of zero while underlying data array holds X; three seeds match standalone model lane-by-lane with zero discrepancies; differential testing caught two real bugs (lost destination when response held; port receiving a second unconsumed response) |
| G2 Full Client Integration into Core | **DONE** | R3V2-CORE-01/02/03. `make core-vrf` 23 programs all pass; elaborated hierarchy contains only `tnca_vrf_banked` + `tnca_rf_bank`, with no `tnca_vector_rf` shadow and no hierarchical direct reads; `t29_vrfbank` directed test triggered 4 same-bank serializations, 8 cross-bank parallel accesses, 12 real rejections; matrix 544 requests waited 272 cycles |
| G3 Writeback / Return / Ownership | **DONE** | R3V2-FLOW-01/02/03. Two write clients targeting same bank in same cycle: one commits, rejected one holds payload and retries, both commit exactly once; four requesters contending for same bank wait at most 4 cycles each, matching the frozen bound `(C-1)*B + A` — read arbitration converted from fixed-priority to round-robin to prevent matrix starvation behind execute port; flush and in-flight return isolation is unreachable at latency 1 (return arrives at N+1 and is immediately consumed), reachable and verified at latency 2 |
| G4 Six Programs and Post-Installation Closed Loop | **DONE** | R3V2-E2E-01/02/03. C01–C06 × reference/candidate (12 runs total) all pass: numerical values match independent oracle (not merely comparing implementations against each other), tail block sentinel untouched, canaries on both sides intact, fault=0, resources drained. C03/C04/C05 are genuine FP16 GEMMs compiled via production driver (`source_kind=tnca-ir`), C04/C05 on slow/reorder memory. Post-installation: 3 dynamic alpha launches share one artifact, one session, one device process, with 0 compiler invocations after initial compile and code hash invariant. Reference/candidate artifact cross-mismatches rejected in zero device requests |
| G5 Counters and Micro-Model | **DONE** | R3V2-COUNT-01/02, MODEL-01. Counters do not compare against themselves: core prints bounded event traces, Python independently counts events, matching item-by-item on req/accept/conflict/mtx_req/mtx_wait/wb_retry; nine primary + two VRF mutually exclusive stall counters sum exactly to `perf_stall_total` across 6 programs (integrating banked VRF added two stall sources that were initially omitted from attribution, discovered and fixed here). Six directed graph predictions sealed with hashes before simulation, zero discrepancy cycle-by-cycle on accept/response, with a single-bank control model proving disagreement detection. Simulation wall time, hardware cycles, and clock frequency reported in three separate columns, Fmax marked NOT MEASURED |
| G6 Structural Synthesis / Inherited Regressions | **DONE** | R3V2-SYN-01/02, REG-01. Candidate VRF data is exactly 65,536 bits, 100% in synchronous memory objects, four banks each with 1 read port and 1 write port, synchronous read, matching profile; valid/collector/FIFO listed separately (512 valid bits per bank), ARF's 16 KiB excluded from ratio and marked behavioral. Full machine elaboration + connectivity check passed with 0 unexplained warnings; hierarchy contains `tnca_vrf_banked` and no `tnca_vector_rf`. Physical area/Fmax/power marked NOT MEASURED and verified not claimed. 12 inherited regressions all pass with 0 skips |
| G7 Negative Cases / Evidence / Sign-off | **DONE** | R3V2-MUT-01, EV-01, REL-01. Thirteen sub-mutations all caught by contract-designated checkers, all injected into isolated copies, candidate source hashes unchanged. Two mutations were themselves incorrect and disproved during testing: M01a originally modified `READ_LATENCY==2` branch while design instantiates 1 (dead code), changing it to shift `rvalid` hung the bench (caught by watchdog, not checker); finally fixed to read current address instead of accepted address from previous cycle (true off-by-one read port behavior). M04b originally removed redundant guard where no stimulus asserted `consume` on unready entry (equivalent mutation); replaced with genuine test. Two mutations discovered real blind spots: `t29_vrfbank` originally used `v.iota` setting all registers to identical 16 values across 4 banks so cross-bank corruption would still pass; now each register has distinct values read back and verified. FLOW-03 originally asserted ownership only in a flush scenario where flush cleared entries anyway; now also tests same-bank three-source case verifying slot remains busy and not ready while waiting for remaining operands |

Candidate vs. reference cycle overhead (identical program, parameters, seed): overall +13%, matrix +42…46% (extra cycle per matrix VRF read), tensor +36%, pure vector +16%, pure scalar +0%, `t18_squash` -14% (stricter interlock altered branch resolution timing). The candidate is not required to be faster; all cycle increases have documented causes.

## Gates A–D

| Gate | Status | Basis |
|---|---|---|
| Gate A: G0/G1 | **PASS** | Profile, initialization, ports, and budget verifiable; scout measured 406 cycles/s on this commit, projected 373s / 138,635 cycles, below 1,800s and 1,000,000 cycles |
| Gate B: G2/G3 | **PASS** | Four read clients and two write clients all connected to banked file; hierarchy contains no behavioral model or direct reads; return, writeback, flush, and ownership bounded and verified cycle-by-cycle |
| Gate C: G4/G5 | **PASS** | C01–C06 × reference/candidate (12 runs) all pass against independent oracle; post-installation 3 dynamic launches share one artifact; counters match independent event counts; micro-model sealed with hash before simulation |
| Gate D: G6/G7 | **PASS** | 65,536 data bits 100% in synchronous memory objects; 12 inherited regressions with zero skips; 13 mutations caught by designated checkers; evidence recomputed by standalone verifier; second environment macOS/arm64 agrees element-by-element with primary environment |

The non-premature claims boundaries from Section 7 of the contract still hold: this sign-off certifies **R3 v2 single-core synchronous VRF integration**, not "project production readiness" nor the legacy R2/R3 contracts. Four-tile persistent runtime, synchronous ARF, and full kernel model calibration are deferred to subsequent phases with separate tasks & acceptance contracts.

## R3 v2 Entry Points, Evidence, and Verification

The entry points specified in Section 6 of the contract exist:

```sh
make r3v2-plan            # Contract/inventory/budget, no RTL launched
make r3v2-components      # Bank/collector/return directed tests
make r3v2-core            # VRF integration on core
make r3v2-runtime         # Post-installation candidate profile, 3 dynamic launches
make r3v2-synth           # Memory-preserving synthesis + integration connectivity checks
make r3v2-mutations       # Thirteen sub-mutations against designated checkers
make r3v2-release         # Primary vs second environment element-by-element comparison
make r3v2-acceptance      # Sole target producing verdict
make r3v2-verify-evidence EVIDENCE=<run-directory>
```

`r3v2-acceptance` writes into a unique run directory (run ID containing timestamp and commit); failed historical runs remain in place without being overwritten, with `latest` acting as a pointer. Checker outputs are archived according to the Section 6 directory partitions: `contracts/`, `inventory/`, `build/`, `runs/`, `runtime/`, `model/`, `synthesis/`, `mutations/`, `inherited/`.

`r3v2-verify-evidence` is executed independently from evidence generation: it recomputes the verdict from the frozen inventory and raw results, treating `verdict` and `passed` fields in the summary as claims rather than evidence. The mutation suite verifies this capability: M07c forges a passing summary over a failing raw result, and M07d removes a required result; both must be rejected, while unmodified valid evidence must be accepted.

The second environment is macOS/arm64 ([`evidence/r3v2-second-env.json`](../evidence/r3v2-second-env.json)), verified against the primary locked Linux x86_64 environment. The comparison binds by source fingerprint rather than commit hash. Wall times are recorded separately without cross-machine performance comparisons.

## v1 Historical Work Packages

Historical contract: [Phase 3 (R3 v1)](phase3_tasks_and_acceptance_v1.md).
Prerequisites: [R1 sign-off](r1_signoff.md) (**R1 is ACCEPTED**), [R2 status](r2_status.md).

**R3 v1 is NOT signed off** (superseded by R3 v2). The DELIVERED designations below record the work package state under the original plan; they cannot be used to claim full acceptance under the original contract. Re-examination in the [replan review](r3_replan_review_2026-09-20.md) identified semantic gaps in PLAN-01/03/04 and noted that SYN-02 did not verify ARF synchronous ports. Legacy IDs and annotations in `tools/r3/` belong to v1; v2 uses independent `R3V2-*` criteria and checkers.

This section records only components that were executed with verified evidence archived. Statuses are strictly classified as **DELIVERED**, **PARTIAL**, or **NOT STARTED**; there are no completion percentages. The table records three DELIVERED, five PARTIAL, and four NOT STARTED packages:

| Work Package | Status | Actual Deliverables | Unmet Acceptance Criteria |
|---|---|---|---|
| R3-01 Rate measurement and budget | **DELIVERED** | `tools/r3/scout.py` empirically measures simulation rate before setting budget; 16 workloads in `spec/r3_workloads.json` expand to 32 performance runs; contract caps checked by `tools/r3/plan.py` prior to launching RTL | None |
| R3-02 Finite-resource performance model | **DELIVERED** | `tools/r3/model.py` cross-checked against hand-calculated micro-graphs; MODEL-01…05 executed via `tools/r3/model_tests.py` | None |
| R3-03 Synchronous VRF/ARF and storage wrappers | **PARTIAL** | RF-01…05 all pass. `rtl/core/tnca_rf_bank.v` (1 read port, 1 write port, selectable 1/2 cycle latency, per-lane write mask, explicit RDW write-first bypass cost accounting) and `rtl/core/tnca_vrf_banked.v` (low-order interleaved 4 banks, exactly 8 KiB capacity, invertible mapping). RF-02 uses `tools/r3/rf_model.py` (written against specification rather than RTL) across three seeds, two latencies, and cold/hot address distributions with zero lane discrepancies; control group verifies that latency-1 responses checked against latency-2 model produce expected disagreement | ARF (16 KiB accumulator file) lacks equivalent wrapper |
| R3-04 Operand collector / arbitration | **PARTIAL** | FLOW-01/02/04/05 pass. `rtl/core/tnca_operand_collector.v`: 1 entry per instruction, up to 3 sources, issues only when all sources arrive (same-bank 3 reads serialize over 3 cycles, cross-bank gathered in 1 cycle, with values v0/v4/v8 vs v0/v1/v2); rejected requests retried rather than dropped, with `n_conflict`/`n_wait` recording real waits; round-robin ready sequence across entries; 12 required reads produce exactly 12 grants, at most 1 grant per bank per cycle; flush discards unissued entries while absorbing in-flight accepted reads as orphans without corrupting reallocated slots | FLOW-03 requires writeback unit from R3-05 |
| R3-05 DMA/MMA/SIMD real throughput | **NOT STARTED** | — | UNIT-01…05 |
| R3-06 Counters / ROI / trace | **PARTIAL** | COUNT-01/02/05 pass. `spec/r3_counters.json` freezes event definitions, reset/epoch rules, overflow behavior, and widths for 23 counters. Discovered and resolved stall counting bug: core previously had only hazard/lsu/wb overlapping counters and omitted six other causes (t16_vecadd overcounted by 48 cycles, t22_atomic overcounted by 143 cycles, t12_tensor undercounted by 343 cycles). Established nine fixed-priority mutually exclusive stall counters whose sum strictly equals `perf_stall_total` across 9 programs, with overlapping view retained separately. COUNT-01 matches 12 required reads against `t28_count` | COUNT-03 requires per-link/VN mesh counters; COUNT-04 requires ROI timestamps |
| R3-07 Compilation layout and cost feedback | **PARTIAL** | R2-12 OBS-01 records measured resources in manifest (AXPY uses 4/64 vector registers) rather than profile upper bounds; unmeasured costs marked `unknown` | CC-01…05 require bank mapping exposure in TargetInfo |
| R3-08 Small operator numerical path | **NOT STARTED** | — | NUM-01…05 |
| R3-09 Model calibration and holdout validation | **NOT STARTED** | — | CAL-01…05 |
| R3-10 Storage structural synthesis | **DELIVERED** | SYN-01…05 all pass. `flow/synth/mem_synth.py` runs memory-preserving structural synthesis via `proc; opt; memory -nomap; opt`. Results: `tnca_vrf_banked` 4 memory objects totaling 65,536 bits = exactly 8 KiB, each with 1 read port and 1 write port, synchronous read; compared against legacy `tnca_vector_rf` (4 read ports, 130 write ports, asynchronous read). Data array not cleared on reset; collector's 162 flip-flops and 136-bit control array itemized separately and excluded from data capacity. Labeled strictly memory-inferred; physical area/Fmax/power marked NOT MEASURED and verified not claimed | None |
| R3-11 Budget gating and mutation | **PARTIAL** | R3M-01a/01b/04a/04b/05/06 six mutations caught by designated checkers in isolated copies with unchanged source hashes: early synchronous read and cross-bank data corruption caught by RF-02; duplicate stall counting caught by COUNT-02; single-port grant undercounting caught by COUNT-01; infinite RF ports in model caught by MODEL-02; workload secretly expanding K to 4× cap rejected by PLAN-02 prior to launching RTL | Detectors for R3M-02/03 (FLOW-03, UNIT/NUM) not present; BUDGET-02/03/04 not implemented |
| R3-12 Dual environment and release evidence | **NOT STARTED** | — | R3REL-01…05 |

## Core Properties Established in Storage Modeling

The storage wrapper matches the independent protocol model request-by-request, and the comparison is proven to detect discrepancies:

```text
    seed 1 spread latency 1 (139 request(s), 40 warm-up write(s)):
        654 lane comparison(s), 1186 against never-written state, 0 problem(s)
    control: latency-1 responses checked against a latency-2 model -> DISAGREE (good)
    four requests to one bank, every cycle: at most 1 accepted per cycle
    a register never written reads as X (unknown)
    100 registers in 8 banks of 12: refused at elaboration
```

## Boundaries and Architectural Clarifications

In the v1 exploratory phase, `tnca_vrf_banked` replaced only the standalone storage macro implementation, while `tnca_core.v` continued to instantiate the behavioral register file. In R3 v2, this boundary was resolved: the active core datapath instantiates `tnca_vrf_banked` and `tnca_operand_collector`, connecting all read/write clients without behavioral shadow models or hierarchical direct reads.

However, the non-premature claims boundaries remain in force:
- ARF (accumulator register file) remains behavioral and asynchronous; it is not converted to synchronous memory in this phase and does not count toward synchronous storage completion.
- MMA reads from VRF pass through the banked operand collector, but ARF accumulation retains its legacy order.
- Multi-tile persistent runtime, NoC/mesh interconnect models, and full-kernel holdout calibration belong to subsequent phases (Phase 3.5 / R3B / R3C) and are not certified by this milestone.
