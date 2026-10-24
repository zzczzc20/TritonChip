# R3 Replan: Current State and Basis for Convergence

Date: 2026-09-20. Inspected commit: `71317b35afb9d2602b0510da84188a0fbde15e29`.
Scope: Source code, contracts, status documents, committed evidence, and four sets of targeted local checks.
**This is not full R1/R2/R3 acceptance of the current commit, nor does it re-verify original remote dual-environment artifacts.**

## 1. Conclusions That Currently Hold

| Phase | Evidence & Implementation | Current Assessment |
|---|---|---|
| R1 | [Historical combined verdict](evidence/r1-combined-verdict.json) records `97c0cf90` with 173/173 passing in both environments across 19 normalized artifact comparisons | Historical baseline is signed off; cannot be generalized to claim current HEAD is signed off |
| R2 | `python/tnca` already includes AOT, artifact, ABI, cache, session, and transport; status document records post-installation dynamic parameters, multiple launches, and chained in-device execution | Usable single-core product path exists, but original R2 is not yet complete; these product tests were not rerun during this check |
| R2 gaps | Device table in `device.py` contains only `core-sim`; grid/memory tests explicitly list unimplemented system items; status document lists MLIR main path and release as not started | Four-tile persistent runtime, completion boundaries, MLIR, and dual-environment release cannot be treated as readily reusable prerequisites for R3 |
| R3 storage | `tnca_vrf_banked`, `tnca_rf_bank`, and collector exist; local RF tests pass | Component foundations exist, but no finite-port single-core closed loop exists |
| R3 integration | `tnca_core.v` still instantiates `tnca_vector_rf`; matrix VRF interface remains address/combinational data; `tnca_acc_rf.v` still uses asynchronous reads and full-array reset | New VRF is not connected to core; ARF synchronization has also not been delivered |
| R3 performance | Model hand-calculation tests pass; counters and structural synthesis tooling exist | Model accuracy, port costs, and resource progression on real programs remain unproven |

## 2. Gaps Between Status, Tests, and Contracts

1. **PLAN-01 check is insufficient.** `tools/r3/plan.py` passes simply by reading `verdict` from the historical combined file; it does not verify the current commit, R2 prerequisites, or evidence completeness. An exit code of 0 here does not prove that legacy PLAN-01 holds.
2. **The meaning of 32 runs has changed.** The legacy contract specified 16 workloads × reference/candidate; `spec/r3_workloads.json` and `check_expansion()` actually count workloads × fast/slow/ooo. Equal counts do not prove that a comparison implementation exists.
3. **Unit contracts are not frozen.** PLAN-04 allows 7 parameters to be unmeasured and returns PASS. While suitable as exploration progress, this fails to prove that candidate ports, latencies, queues, and throughput formulas required by the original contract are complete.
4. **Synthesis scope is insufficient.** `synth_tests.py` SYN-01 checks VRF/ARF memory bit widths, but SYN-02 synchronous read and port constraints only check banked VRF. Retaining memory objects for ARF does not mean ARF has become a synchronous finite-port implementation; DELIVERED for R3-10 cannot cover all requirements of the original contract.
5. **Component semantics cannot replace architectural semantics.** RF-03 verifies that unwritten banks read as X; the original core initialization behavior still requires valid/zero bits or a bounded scrub implementation. The risk of directly integrating this and altering software-visible behavior has not been closed out.
6. **Model self-testing is not model calibration.** Finite-resource checks on hand-crafted DAGs are valuable, but no complete mapping from actual compilation artifacts to the model has been established, nor is there RTL error evidence.
7. **Document summaries are outdated.** The R2 status header initially claimed 11 packages met, whereas the table actually recorded 7 DELIVERED, 5 PARTIAL, 2 NOT STARTED; the R3 header claimed 4 packages partially met, whereas the table recorded 5. Package-by-package status and actual test cases remain the true basis.

These gaps primarily concern integration paths, acceptance semantics, and dependency structure. The solution is to narrow the scope of a single delivery and ensure checkers cover the claimed properties, rather than merely adjusting PASS counts or editing status text.

## 3. Current Execution and Limitations

Execution record: [checks.json](audit/2026-09-20-r3-replan/checks.json). Raw output:

| Entry Point | Result | What It Proves |
|---|---|---|
| `python3 tools/r3/plan.py` | exit 0, 4 PASS | Current script accepts current manifest; semantic gaps listed in previous section remain |
| `python3 tools/r3/model_tests.py` | exit 0, 5 PASS | Existing self-tests of finite-resource model pass |
| `python3 tools/r3/rf_tests.py` | exit 0, 5 PASS | Banked VRF component currently passes directed/random comparisons |
| `python3 tools/r3/flow_tests.py` | exit 1, 4 PASS, FLOW-03 NOT RUN | Implemented parts of collector pass; return/writeback closed loop remains missing |

Raw logs are available at [plan](audit/2026-09-20-r3-replan/plan.log), [model](audit/2026-09-20-r3-replan/model_tests.log), [RF](audit/2026-09-20-r3-replan/rf_tests.log), and [flow](audit/2026-09-20-r3-replan/flow_tests.log). Synthesis, system RTL, Triton product matrix, and remote release were not rerun in this check.

The existing [timing scout](evidence/r3-timing-scout-macos.json) records core-matrix at 3513 cycles / 6.316 s and system at 18673 cycles / 76.306 s. These are historical macOS measurements, not throughput guarantees for the current candidate. The new phase no longer treats full-system performance matrices or new operators as prerequisites for finite-port single-core integration.

## 4. Reasonable Next-Phase Objectives

The new [R3 v2 tasks & acceptance](phase3_tasks_and_acceptance.md) focuses on: **an experimental single-core implementation connecting the existing banked VRF/collector to all VRF clients, closing the loop on writeback backpressure, ownership, a limited set of small programs, and post-installation runtime.**

- R1: Retain historical sign-off; execute blast-radius regressions on candidate commits during this phase. Production release must still be signed off according to the original contract.
- R2: Continue completing MLIR, four-tile runtime, host completion boundaries, and release separately. This phase depends only on the already-implemented single-core path and does not require all of R2 to finish first.
- R3 v2: Deliver the aforementioned single-core experimental implementation without modifying default implementations of established stable profiles.
- Follow-up R3B: Synchronous ARF, full compute unit throughput, closing the loop on model and compilation costs.
- Follow-up R3C: Await completion of R2 system runtime before pursuing four-tile integration, Norm operators, system calibration, and full release.

Legacy [R3 v1](phase3_tasks_and_acceptance_v1.md) is preserved for traceability; unmet requirements are explicitly deferred to follow-up phases. Legacy R3 ACCEPTED cannot be claimed retroactively, nor can partial sign-off under the new version be used to claim production readiness for the project.
