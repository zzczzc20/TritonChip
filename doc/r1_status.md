# R1 状态：逐工作包的实际进度

合同：[phase1_tasks_and_acceptance.md](phase1_tasks_and_acceptance.md)。
审计：[production_readiness_audit_2026-09-19.md](production_readiness_audit_2026-09-19.md)。
签收表与完成报告：[r1_signoff.md](r1_signoff.md)。
支持矩阵：[support_matrix.md](support_matrix.md)。

本文件只记录**已经跑过并留下证据**的部分。合同第 1.3 条禁止按百分比验收，
所以这里不给完成度百分比；每个工作包只有 DELIVERED、PARTIAL、NOT STARTED
三种。最终机器判决在每次 run 的 `acceptance.json` 里，由
`make r1-verify-evidence EVIDENCE=<dir>` 重新推导，不读文件里的结论。

**R1 已签收（ACCEPTED）。** 候选 commit `97c0cf90`，两套相互独立的 Linux
x86_64 环境各 173/173 通过，三个 profile 的配置 hash 两边一致，19 个构建产物
按事先声明的规范化规则全部一致。最终判决由
`tools/r1/compare_envs.py` 在两份证据之上给出：

```text
  r1-compare-envs   PASS   the two environments agree
  R1-09 and R1-12 resolved here: {'R1-09': 'PASS', 'R1-12': 'PASS'}
  FINAL VERDICT: ACCEPTED
```

证据在 [`evidence/r1-combined-verdict.json`](evidence/r1-combined-verdict.json)
与 [`evidence/r1-environments.json`](evidence/r1-environments.json)。
单个环境自己的 `acceptance.json` 仍写 NOT ACCEPTED——一次 run 无权回答"两套
环境是否一致"，这是设计而不是缺陷。开发机（macOS）上 `make r1-acceptance`
仍然退出非零，同样是正确行为。

## 逐包状态

| 工作包 | 状态 | 实际交付 | 未满足的验收条件 |
|---|---|---|---|
| R1-01 规范/基线 | **DELIVERED** | 173 条必跑用例，每条带 entry/profile/expect/source/work_package；`doc/support_matrix.md` 由 ISA 导出、三个 target、lowering 本身和整机程序四个来源生成，四列分开；异步操作的 read/write footprint、保持时刻、释放事件、错误完成、barrier generation、predicated-off 逐条写出 | 无 |
| R1-02 runner/失败传播 | **DELIVERED** | `tools/r1/r1lib.py` 严格执行核心；RUN-01…RUN-10 注入全部被检出（10/10，macOS 与 Linux 一致） | 无 |
| R1-03 构建/生成物 | **DELIVERED** | BUILD-01…BUILD-07 在隔离 checkout 中执行，候选树前后 hash 不变；bench 依赖含 `rtl/gen`、`.vh` 与 fingerprint；provenance 记录抓到"别的 profile 的产物比所有前置都新" | 无 |
| R1-04 profile/能力 | **DELIVERED** | CFG-01…CFG-06；三个 profile 的能力与资源从真实实例读回；F03（system target 宣告 SFU 而 tile 关闭 SFU）已修 | 无 |
| R1-05 footprint | **DELIVERED** | FP-01…FP-06，oracle 独立于被测函数；F01 的 K/dtype footprint 缺陷已修 | 无 |
| R1-06 ownership | **DELIVERED** | OWN-01…OWN-12 全部执行；每个负例都有只差一个 wait 的正例；`t.st.async` 的死锁改为 ILLEGAL_INSTRUCTION，`t26_stfault` 在 fast/slow/ooo 三种内存下运行 | 无 |
| R1-07 IR/前端边界 | **DELIVERED** | IR-01…IR-07；多 block、嵌套 region、经 block argument 转发的事件、未知资源效果全部按名拒绝；F07 的两条（只遍历目标函数、按声明签名绑定参数）已修；accumulator 进入 MLIR verifier | 无 |
| R1-08 RTL/E2E | **DELIVERED** | REG-01…REG-11；36 组 pipeline 组合、{1,7,42} 三个 seed 传到真正控制 stall 与 reorder 的 PRNG 并产生三条不同 trace；`compiler-unit` 的 7 个 skip 按真实入口逐条执行并各有 ID | 无 |
| R1-09 可复现环境 | **DELIVERED** | 锁文件记录 AMI、每个 Ubuntu 包版本、MLIR 20.1.2、Triton wheel 与 XLS 发行包的 sha256；`bootstrap.sh` 在环境内自建 venv、`tnca-opt` 与 XLS；A/B 两套独立环境各 173/173，19 个产物规范化后一致 | 无（runner 低于参考规格已如实记录在锁文件中） |
| R1-10 CI/工件 | **DELIVERED** | 五个入口齐备；证据含 profiles/、build/、cases/、mutations/、junit.xml；EV-01…EV-06 六种篡改逐条注入，验证器必须指名道姓地报出来；CI 的状态取自 `r1-verify-evidence` | 无 |
| R1-11 mutation | **DELIVERED** | MUT-01…MUT-07，每项在隔离副本里只破坏一件事，基线必须通过、变异必须失败、失败必须落在指名的 case 上，候选树 hash 前后不变 | 无 |
| R1-12 文档/交接 | **DELIVERED** | [r1_signoff.md](r1_signoff.md)：每包负责人、commit、case ID、证据路径、状态；R1 期间新发现的六个问题各配回归；`tools/r1/signoff.py` 校验签收表与该次 run 不矛盾（12/12 PASS） | 无 |

## 审计问题

| 审计 ID | 状态 | 证据 |
|---|---|---|
| F01 footprint 与 ownership | **已修复** | FP-01…FP-06、OWN-01…OWN-12、MUT-01、MUT-02 |
| F02 端到端失败被吞 | **已修复** | 三个独立成因全部修掉；RUN-01…RUN-10、MUT-05 |
| F03 system profile 的 SFU | **已修复** | `HAS_SFU` 由 `tnca_top` 传下；CFG-01、CFG-03、MUT-04 |
| F04 check 先再生成 | **已修复** | `r1-check-generated` 无再生成前置；BUILD-01…BUILD-07、MUT-03 |
| F05 构建失效 | **已修复** | BUILD-01…BUILD-07；fingerprint + provenance |
| F06 profile 与实例不一致 | **已修复** | CFG-01…CFG-06 |
| F07 importer 段错误 / 参数绑定 / 遍历整个 module | **已修复** | IR-04；`triton_bridge.param_names` |
| F08 工具链不可复现 | **已修复** | 锁文件具体到 AMI、包版本与 sha256；XLS 由 `tools/xls/fetch_xls.sh` 按 checksum 取得并在环境内本地运行，`R1_STRICT=1` 拒绝 ssh runner |
| F09 skip 被当作覆盖 | **已修复** | REG-01；7 个 skip 各有 ID |
| F10 证据不可核验 | **已修复** | EV-01…EV-06 |
| F11 文档与实现不符 | **已修复** | `doc/support_matrix.md`；`compiler/README.md` 的 C0/C4 矛盾已消 |

## Linux x86_64 验收环境

主机：Ubuntu 24.04.4 LTS（AMI `ami-0a99fe8dc4fc1f359`），x86_64，
4 vCPU / 15 GiB RAM，工作区在挂载盘 `/mnt/triton-viz-data/r1`
（主盘已 85% 满，不使用）。iverilog 12.0（开发机是 11.0），
MLIR 20.1.2，Triton 3.5.0 wheel。

`tools/r1/bootstrap.sh <env-root> <archive>` 从候选 commit 的 `git archive`
解出干净树，在环境内部建自己的 venv 和 `tnca-opt`，两套环境之间只共享一个
只读下载缓存。这就是 A 与 B 相互独立的含义。

## 一条不能靠改代码消掉的限制

合同 R1-09 明确写着 macOS 不能替代 Linux 双环境验收。开发机（macOS arm64）
上的 `make r1-doctor` 会如实报告这一点并拒绝暗示相反的结论。这条不变：
macOS 是开发平台。
