# Phase 3.6 Status: Multi-Backend Engineering Integration

Contract: [phase3_6_tasks_and_acceptance.md](phase3_6_tasks_and_acceptance.md).
Handover guide: [r36_handoff.md](r36_handoff.md).

Status: **IMPLEMENTED / NOT ACCEPTED**. Handover notes and the single unresolved failure are documented in [r36_open_questions.md](r36_open_questions.md).

Detailed status: **IMPLEMENTED / PENDING GATE**. All 30 criteria have checkers and pass in the development environment; the sign-off verdict is produced on the locked platform by `make r36-acceptance` and `make r36-verify-evidence`. This document does not substitute for that verdict.

Following checker implementation, a clause-by-clause audit was conducted and over a dozen "claimed but unperformed" checks were repaired; see [Clause-by-Clause Audit and Its Findings](#clause-by-clause-audit-and-its-findings) below.

The milestone target is `R36_ENGINEERING_READY`. **Does not specify FPGA; not responsible for physical board bring-up.** Board selection, vendor tools and IP, placement and routing, bitstreams, programming, physical board debugging, and ASIC process implementation are performed in independent follow-up branches; `platform-verified` is a reserved state rejected by the registry.

| Work Package | Status | Checker / Evidence |
|---|---|---|
| R36-01 Integration Baseline and Capability Inventory | PASS (3/3) | `make r36-check-plan` |
| R36-02 Backend Abstraction and Dependencies | PASS (3/3) | `make r36-backend-tests` |
| R36-03 Configuration and Compatibility | PASS (3/3) | `make r36-config-tests` |
| R36-04 Common RTL and Export | PASS (3/3) | `make r36-rtl-tests`, `make r36-export` |
| R36-05 Storage and Clock/Reset Adaptation | PASS (3/3) | `make r36-mem-tests` |
| R36-06 Host Protocol and Completion | PASS (3/3) | `make r36-io-tests` |
| R36-07 Build, Installation, and Delivery | 2/3 (BUILD-02 requires locked platform) | `make r36-build-tests` |
| R36-08 Conformance and Regression | PASS (standalone) | `make r36-conformance`, `make r36-mutation-tests`, `make r36-regression` |
| R36-09 Downstream Branch Handover | PASS (3/3 standalone) | `make r36-handoff-tests` |
| R36-10 Evidence and Sign-off | Executed; verdict NOT ACCEPTED | `make r36-acceptance`, `make r36-verify-evidence` |

On the locked platform's most recent gate run: 9/11 checkers passed (plan, backend, config, rtl, memory, io, build, conformance, mutation all full marks), and Section 5.2 fidelity passed (six frozen workloads bit-identical in machine code, 11 cycle differences fully attributed). Unpassed were `R36-08 regression` and `R36-09 handoff`, both of which pass standalone but failed in the gate; see [r36_open_questions.md](r36_open_questions.md) for details.

The fresh-environment test case of R36-BUILD-02 is **NOT RUN and therefore judged FAIL** on macOS: the locked Triton 3.8.0 wheel is manylinux x86_64 only, and `spec/r1_toolchain.lock.json` explicitly states that macOS cannot substitute for the acceptance platform. Per R1-02, this is not counted as a skip; the verdict is governed by runs on the locked platform.

## What Hardware Was Actually Modified in This Phase

Device behavior was moved from the testbench into the synthesizable public top `rtl/tnca_shell.v`: command port, completion record, host access port and arbitration, deadline, and fault latching. `tb/device/tnca_device.v` retains only clocks, reset, storage, and a single protocol translation layer, with no hierarchical reads or writes.

Defects exposed and fixed as a result (separated from directory/interface restructuring per Section 5.2 requirements):

- The core's `f1_valid`/`fb_valid` registers were never reset, leaving `core_idle` as X from power-up until flushed by the first kernel launch (recorded in `spec/r36_baseline_delta.json` as DELTA-03).
- Staged copies under `python/tnca/data/` shadowed the working tree, and `rtl/` itself was absent from the device fingerprint, together causing the runtime to execute against stale RTL over extended periods.
- The socket backend lacked parameter validation, allowing illegal addresses to trigger connection shutdown via relay error paths, converting recoverable errors into device loss.

## Section 5.2 Fidelity

Using R3.5 frozen commit `0e7c34795cd6` as baseline, measured in a worktree at that commit:

- **Machine code: all six frozen workloads are bit-identical under both B0 and O2.**
- Cycles: 11 differences, all attributed to DELTA-01 (the shell samples on clock edges, whereas the legacy testbench read in the active region; grid `device_cycles` accumulates, so grid=2 differs by 2). L06 on the slow reordering model has an additional systematic shift because host access now shares memory ports with the core, recorded by comparing medians against the measured spread.
- DELTA-02 and DELTA-03 are declared, but not touched by these workloads; the tooling explicitly reports this.

## Clause-by-Clause Audit and Its Findings

After all checkers were written, an audit was performed against all 30 standards, clause by clause: comparing each clause in the contract against its checker implementation to verify whether every requirement was genuinely executed. A common failure pattern identified was exercising the first requirement while silently skipping subsequent ones. The following items, **where code or docstrings claimed checks that were not actually performed**, have all been fixed:

- In IO-01's illegal request brackets, the sequence counter was read and then discarded via `del`, whereas the docstring required both halves to remain unchanged.
- MUT-ASSET-HASH did not invoke any checker; it recalculated the hash inline and forged a results dictionary.
- The frozen mutation list had 7 entries, but the harness only had 6; M07 was never implemented while `plan.py` asserted a count of 7 ("count matched the contract, work did not").
- The mutation suite lacked clean controls: a checker failing in the lab for its own reasons would be credited as detecting a mutation.
- BE-01 only grepped `runtime.py`; vvp was launched by `transport.py`, and iverilog was invoked by `device.py`.
- RTL-02's docstring claimed random stalling was enabled, but the smoke bench hard-coded `RANDOM_STALL(0)`.
- CFG-03 printed prose about an "acceptance matrix" when no such matrix existed in the repository tree.
- HAND-01 launched the rehearsal with `PYTHONPATH=<repo>/python`, whereas Section 5.3 explicitly requires using the **installed package**.
- BUILD-02's runtime-only half was claimed by two files to be covered by R2's PKG-03; no `pkg03` function existed.
- IO-02's concurrency rejection was represented by two blocks of prose.
- `gemm_program(..., roles=2)` accepted and ignored the parameter, returning machine code byte-identical to `roles=1` for both callers—making "two-role pipeline" merely a label.

Beyond the fixes, 4 items remain knowingly partial, each recorded in `spec/r36_required_cases.json` under `known_partial_coverage` (stating what is not checked, why, and what would close it), with R36-BASE-01 asserting this record is present and complete:
R36-RTL-03, R36-EV-02, R36-CFG-02, R36-REL-01.

## Work Not Done

`technology_mapping` for every array in `spec/r36_memories.json` remains `not started`; the ARF remains a behavioral model; cache coherence is absent; gaps remain in the four-tile runtime. These are documented in the "What is NOT a board task" section of the [handover guide](r36_handoff.md) and belong to mainline responsibilities.

`CAP-TWO-ROLE`: The core features two role contexts (`NUM_RCTX=2`, and `tb/core/progs/t04_roles.s` runs producer/consumer on real RTL), but **the product path cannot reach them**—`runtime.py` hard-codes `role_mask=1`, and the GEMM builder only emits single-role tiles. The "two-role pipeline" path required by Section 5.1 is therefore recorded as NOT RUN.

R35-OPT-02 was open in the baseline; it is not closed in this phase and makes no claim of closure.
