# R1 completion report and sign-off

The contract is `phase1_tasks_and_acceptance.md`; the audit it answers is
`production_readiness_audit_2026-09-19.md`. This file is R1-12's deliverable:
one row per work package with its owner, the commits that implemented it, the
case IDs that test it, where the evidence lives, and its status.

**Owner for every work package: the repository maintainer (git `zzczzc20`).**
This is a single-maintainer project; recording one name twelve times is
honest, and leaving the column blank or writing "TBD" is what the contract
forbids.

Status is one of **PASS**, **PARTIAL** or **NOT RUN**, and nothing else. In
particular there is no "not applicable": R1-01 says an undelivered
requirement inside the delivery scope is a FAIL or a NOT RUN, never an
absence.

## Sign-off table

| Work package | Owner | Commits | Cases | Evidence | Status |
|---|---|---|---|---|---|
| R1-01 spec, support matrix, baseline cases | maintainer | `e31709b`, `974a3d5` | the whole inventory; `spec_problems` is checked before anything runs | `spec/r1_required_cases.json`, `doc/support_matrix.md`, `make r1-inventory` | PASS |
| R1-02 failure propagation, timeouts, result format | maintainer | `e31709b`, `ff5f14c`, `6f8a6df` | RUN-01..RUN-10 | `tools/r1/selftest.py`, evidence `logs/gate-selftest/` | PASS |
| R1-03 build graph and generated artefacts | maintainer | `d824616` | BUILD-01..BUILD-07 | `tools/r1/build_tests.py`, evidence `build/` | PASS |
| R1-04 profiles and capabilities | maintainer | `3dcc0c1` | CFG-01..CFG-06 | `tools/r1/profile_tests.py`, evidence `profiles/` | PASS |
| R1-05 resource footprints | maintainer | `501d152` | FP-01..FP-06 | `tools/r1/resource_tests.py` | PASS |
| R1-06 async ownership | maintainer | `501d152`, `3f137a3`, `14401ec` | OWN-01..OWN-12, `BASE-CORE-t26_stfault-{fast,slow,ooo}` | `tools/r1/resource_tests.py`, `tb/core/progs/t26_stfault.s` | PASS |
| R1-07 IR support boundary | maintainer | `66da4c8`, `5fa415c` | IR-01..IR-07 | `tools/r1/ir_tests.py`, `compiler/test/IR/ir0*.mlir` | PASS |
| R1-08 regression matrix | maintainer | `136849f`, `b424f9d` | REG-01..REG-11 and the 109 baseline IDs | `tools/r1/regression.py`, evidence `cases/regression.json` | PASS |
| R1-09 reproducible toolchain, two environments | maintainer | `db82824`, `2f801a0`, `5fa415c`, `061d2b2` | `r1-doctor`, `compare_envs` | `spec/r1_toolchain.lock.json`, `tools/r1/bootstrap.sh`, `tools/r1/compare_envs.py` | PARTIAL |
| R1-10 CI gate and evidence | maintainer | `3de733c` | EV-01..EV-06 | `.github/workflows/r1-acceptance.yml`, `tools/r1/verify_evidence.py` | PASS |
| R1-11 mutation testing | maintainer | `14401ec`, `66da4c8` | MUT-01..MUT-07 | `tools/r1/mutation_tests.py`, evidence `mutations/` | PASS |
| R1-12 documentation and sign-off | maintainer | `974a3d5`, this file | structural check in `tools/r1/signoff.py` | `doc/r1_signoff.md`, `doc/r1_status.md` | PARTIAL |

A row is PASS only when its cases are PASS **in the run being signed off**.
The table above records what each work package delivered; the machine verdict
for a given candidate is `acceptance.json` in that run's evidence directory,
and `make r1-verify-evidence EVIDENCE=<dir>` re-derives it rather than reading
the verdict out of the file.

## Why two rows are PARTIAL

**R1-09.** One Linux x86_64 acceptance environment exists and is locked: the
AMI, every Ubuntu package version, MLIR 20.1.2 and the Triton wheel by
sha256. `bootstrap.sh` builds an environment's own venv and its own
`tnca-opt` from a clean candidate archive, so A and B share only a read-only
download cache. What is not closed:

- the second environment's full run and the artefact comparison had not both
  completed at the time this file was last updated; `tools/r1/compare_envs.py`
  is the check and `.github/workflows/r1-acceptance.yml` is where it runs;
- `rtl/gen` is still regenerated through a personal SSH host, so
  `make r1-check-generated` reports it as not re-checked unless `R1_XLS=1`.
  R1-09 asks for a local or containerised XLS runner at a fixed revision;
- the runner is below the contract's reference runner on all three counts
  (4 vCPU against 8, 15 GiB against 32, 58 GiB free against 100). Recorded in
  the lock rather than rounded up.

**R1-12.** This file and `doc/r1_status.md` exist and the documentation points
at the support matrix, the three profiles, the strict entry points and the
known limits. It cannot be PASS until R1-09 is, because the contract's final
verdict requires "独立环境 A 和 B 均完整通过，规定的产物确定一致" and a
sign-off that claimed otherwise would be the thing this whole contract exists
to prevent.

## Found during R1, outside F01-F04

The contract requires new correctness or verification blockers found inside
this phase's scope to be fixed with a regression rather than waved through
for not being in the original findings. Six were:

| What | Where it was found | Fix and case |
|---|---|---|
| `t.st.async` deadlocked the core: encodable, excluded from the memory-issue condition and nothing else, so it entered the transfer state and never left | OWN-03 | raises ILLEGAL_INSTRUCTION with the barrier error bit; `BASE-CORE-t26_stfault-*` |
| the out-of-order memory model never reordered: a fixed latency meant at most one entry was ever eligible | REG-04 | latency varies with the LFSR under REORDER; ~50 reorderings per pipeline run, counted and reported |
| two completions on one barrier in one cycle had never occurred in any test, so the three-port sum was unexercised | REG-10 | `t27_collide.s.in`, with the delay solved for from a measured run |
| `NUM_ROLES` reported the architectural RCTX count rather than the launched one (ISA sec 5.1) | `t24_amo_mesh` hanging | latched `launched` mask; `t15_target` runs with `+allroles` |
| the TTIR importer segfaulted on a function name not in the module | R1-07 | `module.has_function` check raising a Diagnostic; IR-04 |
| MLIR prints infinities as raw hex, which reached `float()` and raised ValueError | IR-06 | hex patterns read as bits; an unreadable literal is a Diagnostic |

## Not accepted, and named

Carried into the next phase rather than closed quietly. None of these is
marked "not applicable".

- transfer rank 3-5 (`DESCRIPTOR_FAULT` above rank 2)
- byte-granular window writes
- `t.st.async` execution (the engine has no vector read port)
- general CFG in the MLIR verifier -- a frozen, stated boundary
- `tnca_vector_fpu` synthesis, which yosys did not finish; see
  `doc/synthesis.md`
- the XLS regeneration path, which still needs a host
