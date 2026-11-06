# R1 Status: Actual Progress by Work Package

Contract: [phase1_tasks_and_acceptance.md](phase1_tasks_and_acceptance.md).
Audit: [production_readiness_audit_2026-09-19.md](production_readiness_audit_2026-09-19.md).
Sign-off table and completion report: [r1_signoff.md](r1_signoff.md).
Support matrix: [support_matrix.md](support_matrix.md).

This document records only components that have **already been executed with verified evidence archived**. Contract §1.3 strictly prohibits percentage-based acceptance; therefore, no completion percentages are provided. Each work package is classified solely into one of three statuses: **DELIVERED**, **PARTIAL**, or **NOT STARTED**. The final machine verdict is recorded in `acceptance.json` for each run, re-derived directly by `make r1-verify-evidence EVIDENCE=<dir>` rather than reading textual assertions from the file.

**R1 is ACCEPTED.** On candidate commit `97c0cf90`, two mutually independent Linux x86_64 environments each completed 173/173 tests. Configuration hashes for all three profiles match across both environments (`core-4r`: `26692e8a4a394edd`, `core-sim`: `d4b739271bb463ba`, `system-2x2`: `d3188d732796426d`), and all 19 built artifacts agree completely under the three pre-declared normalization rules. The final verdict was determined by `tools/r1/compare_envs.py` across both sets of evidence:

```text
  r1-compare-envs   PASS   the two environments agree
  R1-09 and R1-12 resolved here: {'R1-09': 'PASS', 'R1-12': 'PASS'}
  FINAL VERDICT: ACCEPTED
```

The underlying evidence resides in [`evidence/r1-combined-verdict.json`](evidence/r1-combined-verdict.json) and [`evidence/r1-environments.json`](evidence/r1-environments.json). An individual environment's own `acceptance.json` still records NOT ACCEPTED because a single run cannot assess whether two separate environments agree; this is by design rather than a defect. Similarly, running `make r1-acceptance` on a development host (macOS) continues to exit non-zero, which is also expected behavior.

## Work Package Status

| Work Package | Status | Actual Deliverables | Unmet Acceptance Criteria |
|---|---|---|---|
| R1-01 Specifications / Baseline | **DELIVERED** | 173 mandatory test cases in `spec/r1_required_cases.json`, each with entry, profile, expect, source, and work_package fields; [support_matrix.md](support_matrix.md) generated with four separated columns derived from ISA exports, three target definitions, lowering code, and full-machine programs; read/write footprints, hold durations, release events, error completion behavior, barrier generation, and predicated-off semantics defined item-by-item for asynchronous operations | None |
| R1-02 Runner / Failure Propagation | **DELIVERED** | Strict execution core in `tools/r1/r1lib.py`; failure injections RUN-01...RUN-10 all detected (10/10 PASS in `tools/r1/selftest.py`, verified identically on macOS and Linux) | None |
| R1-03 Build / Generated Artifacts | **DELIVERED** | BUILD-01...BUILD-07 executed in isolated checkouts without dirtying the candidate tree; candidate source tree hashes remain unchanged before and after execution; benchmark dependencies track `rtl/gen`, `.vh` headers, and build fingerprints; decoupled `make r1-check-generated` without pre-generation; artifact provenance detects when artifacts from another profile are newer than prerequisites | None |
| R1-04 Profiles / Capabilities | **DELIVERED** | CFG-01...CFG-06; capabilities and resource parameters read back from elaborated instances across all three profiles; audit F03 remediated (`HAS_SFU` propagated from `tnca_top`, system target SFU=false); audit F05 topology boundaries enforced by rejecting unsupported mesh configurations prior to execution | None |
| R1-05 Resource Footprint Calculation | **DELIVERED** | FP-01...FP-06; MMA footprints derived dynamically from instruction K dimension, datatype, and layout across 4 dtypes (FP16/BF16, E5M2/E4M3) and 3 K values (16/32/64); independent oracle formula distinct from tested functions; audit F01 K/dtype footprint underestimation defect remediated | None |
| R1-06 Asynchronous Ownership Verification | **DELIVERED** | OWN-01...OWN-12 executed across static and post-allocation checkers; every negative test pairs with a valid positive case differing solely by events, ordering, or ranges; `t.st.async` core deadlock remediated to raise `ILLEGAL_INSTRUCTION` with barrier error bit; `BASE-CORE-t26_stfault-{fast,slow,ooo}` executed across fast/slow/ooo memory models; multi-consumer last release and predicated-off async verified | None |
| R1-07 IR / Frontend Boundaries | **DELIVERED** | IR-01...IR-07; single-block safe subset accepted; multi-block CFG, nested regions, branch with single-sided await, events forwarded via block arguments, and unknown resource effects explicitly rejected with structured diagnostics; audit F07 boundary remediated (only designated target function compiled, arguments bound by signature name/index via `triton_bridge.param_names`); MLIR verifier covers accumulator tracking | None |
| R1-08 RTL / E2E Regressions | **DELIVERED** | REG-01...REG-11 and 109 baseline semantic IDs preserved and passing (173 total test executions); 36 pipeline combinations (iterations ∈ {0,1,2,3,5,6} × stages ∈ {2,3} × memory ∈ {fast,slow,ooo}) pass; three PRNG seeds {1, 7, 42} passed to stimulus and stall/reorder PRNG under slow/ooo memory, producing three distinct traces; 7 skipped compiler-unit tests executed via real entry points with unique IDs | None |
| R1-09 Reproducible Toolchain | **DELIVERED** | Lockfile `spec/r1_toolchain.lock.json` pins AWS AMI, Ubuntu package versions, MLIR 20.1.2, Triton wheel, and XLS release sha256; `tools/r1/bootstrap.sh` builds dedicated venv, `tnca-opt`, and local XLS extracted via `tools/xls/fetch_xls.sh`; two independent Linux x86_64 environments A and B each pass 173/173 tests, with 19/19 built artifacts matching under declared normalization rules | None (runner host specs below reference runner documented in lockfile) |
| R1-10 CI / Artifact Verification | **DELIVERED** | All five required entry points implemented (`r1-doctor`, `r1-check-generated`, `r1-gate-selftest`, `r1-acceptance`, `r1-verify-evidence`); evidence archives include `profiles/`, `build/`, `cases/`, `mutations/`, `run.json`, `inventory.json`, `results.json`, `junit.xml`, `acceptance.json`, `checksums.sha256`; tampering injections EV-01...EV-06 verified; CI gate status derived from `r1-verify-evidence` | None |
| R1-11 Mutation Verification | **DELIVERED** | MUT-01...MUT-07 executed in isolated clones; each mutation alters exactly one invariant, baseline passes, mutation fails on the specific designated test case; candidate git hash unaffected before and after execution | None |
| R1-12 Documentation / Handoff | **DELIVERED** | Phase completion report in [r1_signoff.md](r1_signoff.md) documenting assignees (`zzczzc20`), implementation commits, case IDs, evidence paths, and status across all 12 work packages (12/12 PASS); six newly discovered correctness/verification blockers outside F01-F04 remediated with regressions; `tools/r1/signoff.py` structural and run consistency check passes | None |

## Audit Findings

| Audit ID | Status in R1 | Resolution & Evidence |
|---|---|---|
| F01 Asynchronous ownership and footprint calculation | **Fixed** | Dynamic K/dtype footprint calculation; unified write range checking; FP-01...FP-06, OWN-01...OWN-12, MUT-01, MUT-02 |
| F02 Failure propagation and result formats | **Fixed** | Direct subprocess management with pipefail, timeout process-group termination, and zero-skip/zero-xfail strict checking; RUN-01...RUN-10, MUT-05 |
| F03 System target SFU configuration | **Fixed** | Profiles derived from elaborated hardware instances; `HAS_SFU` propagated from `tnca_top`; system target sets SFU=false; CFG-01, CFG-03, MUT-04 |
| F04 Incremental builds and stale RTL validation | **Fixed** | Header and generated RTL dependencies added (`rtl/gen`, `.vh`, build fingerprints); `r1-check-generated` decoupled from code generation; BUILD-01...BUILD-07, MUT-03 |
| F05 4×4 topology parameterization | **Boundary Closed** | Unsupported topologies (non-2×2 mesh, 1×1, 4×4, non-positive dimensions) explicitly rejected at configuration/elaboration time; CFG-05 (full 4×4 deferred to Phase 4) |
| F06 MLIR asynchronous safety verification | **Boundary Closed** | Single-block safe subset enforced; multi-block CFG, nested regions, branch with single-sided await, and forwarded block arguments rejected; IR-01...IR-03, IR-07, MUT-06 |
| F07 Production backend and importer parameter binding | **Boundary Closed** | Importer restricted to designated target function; parameters bound by signature name/index (`triton_bridge.param_names`); structured diagnostics on type/shape errors; IR-04 |
| F08 Reproducible toolchain and release infrastructure | **Fixed** | Pinned lockfile (`spec/r1_toolchain.lock.json`) with AMI, package versions, and sha256 digests; local XLS release via `tools/xls/fetch_xls.sh` (`R1_STRICT=1` rejects remote SSH); dual clean environments; EV-01...EV-06 |
| F09 Memory progress verification and finite workloads | **Baseline Verified** | Stimulus and stall/reorder PRNG driven by three seeds {1, 7, 42} producing distinct traces; finite workload regressions executed; REG-04, REG-06, REG-07 |
| F10 Physical register files, storage macros, and timing | **Deferred** | Out of scope for R1 baseline per contract §1.2; behavioral models retained; proxy gate-level synthesis documented in [synthesis.md](synthesis.md) |
| F11 Documentation and empirical evidence alignment | **Fixed** | Four-column support matrix in [support_matrix.md](support_matrix.md); compiler roadmap contradictions resolved; audit baseline preserved alongside sign-off report in [r1_signoff.md](r1_signoff.md) |

## Linux x86_64 Acceptance Environment

Host: Ubuntu 24.04.4 LTS (AMI `ami-0a99fe8dc4fc1f359`), x86_64, 4 vCPUs / 15 GiB RAM, workspace located on mounted volume `/mnt/triton-viz-data/r1` (primary disk 85% full, unused). Icarus Verilog 12.0 (development machine uses 11.0), MLIR 20.1.2, Triton 3.5.0 wheel.

`tools/r1/bootstrap.sh <env-root> <archive>` extracts a clean tree from the candidate commit's `git archive`, building its own virtual environment, `tnca-opt`, and local XLS extraction inside the environment. The two environments share only a verified read-only download cache. This is what mutual independence between A and B means.

## Non-Waivable Platform Limitation

Contract R1-09 explicitly states that macOS cannot substitute for Linux dual-environment acceptance. Running `make r1-doctor` on the development host (macOS arm64) faithfully reports this platform mismatch and refuses to imply otherwise. This rule remains unchanged: macOS is a development platform.
