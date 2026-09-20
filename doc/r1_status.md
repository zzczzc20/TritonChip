# R1 状态：逐工作包的实际进度

合同：[phase1_tasks_and_acceptance.md](phase1_tasks_and_acceptance.md)。审计：[production_readiness_audit_2026-09-19.md](production_readiness_audit_2026-09-19.md)。

本文件只记录**已经跑过并留下证据**的部分。合同第 1.3 条禁止按百分比验收，所以这里不给完成度百分比；每个工作包只有 DELIVERED（该包自身的验收条件已满足）、PARTIAL（有可用交付但验收条件未全满足）、NOT STARTED 三种。

**R1 未通过。** 151 个必跑用例中 66 个属于合同新增、尚未实现，按合同第 1.3.3 条记为 NOT RUN 即 FAIL。`make r1-acceptance` 因此退出非零，这是正确行为，不是缺陷。

## 逐包状态

| 工作包 | 状态 | 实际交付 | 未满足的验收条件 |
|---|---|---|---|
| R1-01 规范/基线 | **PARTIAL** | `spec/r1_required_cases.json`：151 个必跑用例 ID，每条带 entry/profile/expect/source/work_package；`tools/r1/inventory.py` 校验无空字段、无 TBD/TODO/N-A、无重复 ID | 四列支持矩阵（ISA 定义 / profile 实现 / 编译器可生成 / 整机验证）未建；异步操作的 read/write footprint、保持时刻、释放事件尚未逐条写出 |
| R1-02 runner/失败传播 | **DELIVERED** | `tools/r1/r1lib.py` 严格执行核心；`tools/r1/selftest.py` RUN-01…RUN-10 全部注入并全部被检出（10/10） | 无。合同要求的最终 PASS 仍取决于 R1-09 的环境 |
| R1-03 构建/生成物 | **PARTIAL** | `make r1-check-generated` 与生成器分离，且**没有会先行再生成的前置依赖**——这正是审计 F04 指出的问题 | BUILD-01…BUILD-07 七项构建失效试验未实现；bench 依赖仍缺 `rtl/gen`、`.vh`、生成器 Python 模块与 fingerprint |
| R1-04 profile/能力 | **NOT STARTED** | — | CFG-01…CFG-06 全未实现；F03（system target 宣告 SFU 而 tile 关闭 SFU）未修 |
| R1-05 footprint | **NOT STARTED** | — | FP-01…FP-06 全未实现；F01 的 K/dtype footprint 缺陷未修 |
| R1-06 ownership | **NOT STARTED** | — | OWN-01…OWN-12 全未实现；F01 的写入检查缺口未修 |
| R1-07 IR/前端边界 | **PARTIAL** | TTIR importer 不再因函数名不存在而段错误（已改为受控诊断，commit `6c90063`） | IR-01…IR-07 全未实现；参数按 dict 插入顺序绑定、遍历整个 module 未修 |
| R1-08 RTL/E2E | **PARTIAL** | 基线 85 条语义用例已冻结进清单并由 gate 实际执行；`compiler-ir` 现按 RUN 行计数而非按文件 | REG-01…REG-11 未实现：seed 矩阵、trace hash、cover 计数、compiler-unit 中 7 个 skip 未转为实跑 |
| R1-09 可复现环境 | **NOT STARTED** | `make r1-doctor` 会明确拒绝把 macOS 当作验收环境 | 无锁文件、无 Linux x86_64 双环境、XLS 仍依赖个人 SSH 与远程 `cur` 目录 |
| R1-10 CI/工件 | **PARTIAL** | 五个入口全部存在；`tools/r1/gate.py` 产出 run.json / inventory.json / results.json / acceptance.json / checksums.sha256 / logs；`tools/r1/verify_evidence.py` 重算 hash 并重新推导集合关系，不信任 acceptance.json | 无 CI 工作流；证据篡改注入（六种）未作为测试执行；profiles/ build/ cases/ mutations/ 目录目前为空 |
| R1-11 mutation | **NOT STARTED** | — | MUT-01…MUT-07 全未实现 |
| R1-12 文档/交接 | **PARTIAL** | 本文件；`Readme.md` 的能力表已随实现更新 | 完成报告、负责人、每包证据路径未填；`compiler/README.md` 的 C0/C4 声明与实际支持矩阵仍不一致 |

## 已修复的审计问题

| 审计 ID | 状态 | 证据 |
|---|---|---|
| F02 端到端失败被吞 | **已修复** | `make triton-e2e e2e TRITON_PYTHON=/usr/bin/false` 由 exit 0 变为 exit 2；RUN-01 会在回归时重新检出 |
| F02 IR runner 的 shell pipeline | **已修复** | `compiler/test/IR/run_ir_tests.py` 改用 `bash -o pipefail`；RUN-09 静态扫描仓库中未加保护的 pipeline |
| F02 跳过被当作通过 | **已修复** | `R1_STRICT=1` 下缺工具为失败；RUN-02 覆盖三种缺失方式 |
| F07 importer 段错误 | **已修复** | 函数名不在 module 中时返回诊断而非崩溃解释器 |
| F04 check 先再生成 | **部分修复** | `r1-check-generated` 无再生成前置；`make compiler` 的旧行为未动 |
| F01、F03、F05、F06、F08、F09、F10、F11 | **未修复** | 见上表 |

## 第一个 Linux x86_64 验收环境（envA）

主机：Ubuntu 24.04.4 LTS，x86_64，4 vCPU / 15 GiB RAM，工作区在挂载盘 `/mnt/triton-viz-data/r1/envA`（主盘已 83% 满，不使用）。iverilog 12.0（开发机是 11.0）。

从候选 commit 的 `git archive` 解出干净树，不复制开发机的任何构建产物：

| 步骤 | 结果 |
|---|---|
| `make r1-doctor` | PASS，并如实报告低于参考 runner 规格（4 vCPU < 8，57 GiB < 100） |
| `make r1-check-generated` | PASS——干净 checkout 上 ISA schema、TableGen 表、opcode map、三个 target、218 个共享常量、DSLX 与浮点 oracle 全部一致 |
| `make r1-gate-selftest` | PASS，10/10 注入全部检出（与开发机一致） |
| `tools/r1/gate.py` | **NOT ACCEPTED**：required=151，pass=81，not-run=66，fail=4 |

81 项通过的都是真跑出来的：21 个 core 程序 × 3 种内存模型、8 个 unit bench、NUCA、mesh、四 tile 整机两种 launch 模式，全部在 Linux + iverilog 12 上通过——开发机用的是 iverilog 11，这是一次真实的跨平台结果。

4 项失败全部来自 envA 缺 Triton/LLVM，且在 `R1_STRICT=1` 下**正确地记为失败而不是跳过**：`BASE-CC-IR`（缺 tnca-opt）、`BASE-TRITON-CORE`、`BASE-TRITON-SYSTEM`（缺 Triton）、以及 `BASE-CC-UNIT`。

envA 还缺：Triton + LLVM 的锁定安装、tnca-opt 构建、envB。这些是 R1-09 的内容。

## gate 发现的一个此前看不见的缺口

`compiler-unit` 长期报告 “PASS 76 tests, 7 skipped”。严格模式下逐条打印后，7 个全部是 `triton not importable`——整个套件跑在系统 `python3` 下，而 Triton 只装在 `TRITON_PYTHON` 指向的 venv 里。也就是说 **compiler-unit 里所有 Triton 相关用例从来没有在这个入口执行过**，而总数 76 让它看起来跑了。

这正是合同 REG-01 点名的那 7 项。修法不是把 skip 改成 pass，而是让这些用例在真正能 import Triton 的解释器下运行；尚未修复。

## 一个不能靠改代码消掉的限制

合同 R1-09 要求在 **Linux x86_64** 上用两套互相独立、从锁定工具链 bootstrap 的环境完成验收，并明确写着 macOS 不能替代。本仓库目前在 macOS arm64 上开发，`make r1-doctor` 会如实报告这一点并拒绝暗示相反的结论。

平台这一条现在有解了：envA 是真正的 Linux x86_64，合同要求的两套环境已经有第一套。仍然缺的是 envB、锁定的 Triton/LLVM 安装、XLS 的本地/容器化执行方式，以及 OS image digest——**这些是环境与锁文件的工作，不是更多测试代码**。

开发机（macOS arm64）上的 `make r1-doctor` 会继续拒绝把自己当作验收环境。这条不变：macOS 是开发平台。
