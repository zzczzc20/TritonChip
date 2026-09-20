# R1 completion report and sign-off

The contract is `phase1_tasks_and_acceptance.md`; the audit it answers is
`production_readiness_audit_2026-09-19.md`. This file is R1-12's deliverable:
one row per work package with its owner, the commits that implemented it, the
case IDs that test it, where the evidence lives, and its status.

## R1 是 ACCEPTED

在候选 commit `97c0cf90` 上，两套相互独立的 Linux x86_64 环境各自从同一个干净
archive bootstrap、各自建自己的 venv 与 `tnca-opt`、各自跑完全套：

```text
A commit 97c0cf909c47c908be0564fcafa6809c0ec1a677  B commit 97c0cf90...
A 173/173 pass, B 173/173
profile core-4r:     26692e8a4a394edd  in both
profile core-sim:    d4b739271bb463ba  in both
profile system-2x2:  d3188d732796426d  in both
0/19 built artefact(s) are byte-identical
19/19 agree after the 3 declared normalisation rule(s)
  r1-compare-envs   PASS   the two environments agree
  R1-09 and R1-12 resolved here: {'R1-09': 'PASS', 'R1-12': 'PASS'}
  FINAL VERDICT: ACCEPTED
```

证据：[`evidence/r1-combined-verdict.json`](evidence/r1-combined-verdict.json)、
[`evidence/r1-environments.json`](evidence/r1-environments.json)。

两个环境**各自**的 `acceptance.json` 都写着 NOT ACCEPTED，这是设计如此而不是
缺陷：一次 run 无权回答"两套环境是否一致"。R1-09 与 R1-12 由
`tools/r1/compare_envs.py` 在两份证据之上判定，最终判决是它给出的那一个。
零字节相同是 iverilog 按对象地址命名符号的结果；`spec/r1_normalization.json`
事先声明了可去除的三个字段，去除后 19 个产物全部一致。

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
| R1-09 reproducible toolchain, two environments | maintainer | `db82824`, `2f801a0`, `5fa415c`, `061d2b2`, `90e9a24`, `97c0cf9` | `r1-doctor`, `compare_envs` | `spec/r1_toolchain.lock.json`, `tools/r1/bootstrap.sh`, `doc/evidence/r1-combined-verdict.json` | PASS |
| R1-10 CI gate and evidence | maintainer | `3de733c` | EV-01..EV-06 | `.github/workflows/r1-acceptance.yml`, `tools/r1/verify_evidence.py` | PASS |
| R1-11 mutation testing | maintainer | `14401ec`, `66da4c8` | MUT-01..MUT-07 | `tools/r1/mutation_tests.py`, evidence `mutations/` | PASS |
| R1-12 documentation and sign-off | maintainer | `974a3d5`, `f9f8c2d`, this file | structural check in `tools/r1/signoff.py` | `doc/r1_signoff.md`, `doc/r1_status.md` | PASS |

A row is PASS only when its cases are PASS **in the run being signed off**.
The table above records what each work package delivered; the machine verdict
for a given candidate is `acceptance.json` in that run's evidence directory,
and `make r1-verify-evidence EVIDENCE=<dir>` re-derives it rather than reading
the verdict out of the file.

## 两条必须继续说明的限制

R1 已签收，下面两点不改变判决，但属于必须留在记录里的事实。

**运行机器低于合同的参考规格。** 4 vCPU 对 8、15 GiB 对 32、58 GiB 可用磁盘
对 100。锁文件如实记录而不是四舍五入：在更小的机器上跑得更慢，这不等于参考
规格不必要。

**XLS 现在在环境内本地运行，从按 checksum 固定的发行包解出。**
`tools/xls/fetch_xls.sh` 在解包前校验 sha256，`R1_STRICT=1` 直接拒绝 ssh
runner。两套环境的 `make r1-check-generated R1_XLS=1` 都报告
`PASS: rtl/gen matches the DSLX sources`，即受控 RTL 确实由 DSLX 重新生成而
不是被信任。

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
