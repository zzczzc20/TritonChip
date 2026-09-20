# R3 重规划：当前状态与收敛依据

日期：2026-09-20。检查提交：`71317b35afb9d2602b0510da84188a0fbde15e29`。
范围：源码、合同、状态文件、已提交证据，以及四组本地定向检查。
**这不是当前提交的 R1/R2/R3 全量验收，也没有重新验证远端双环境原始工件。**

## 1. 当前可以成立的结论

| 阶段 | 证据与实现 | 当前判断 |
|---|---|---|
| R1 | [历史联合判决](evidence/r1-combined-verdict.json) 记录 `97c0cf90` 两环境各 173/173，通过 19 个规范化工件比较 | 历史基线已签收；不能推广成当前 HEAD 已签收 |
| R2 | `python/tnca` 已有 AOT、artifact、ABI、cache、session、transport；状态文件记载安装后动态参数、多次 launch 和设备内链式执行 | 已有可用单核产品路径，尚未完成原 R2；本次没有重跑这些产品测试 |
| R2 缺口 | `device.py` 的设备表只有 `core-sim`；grid/memory tests 明确列出未实现系统项；状态文件列 MLIR 主路径和发布未开始 | 四 tile 持久 runtime、完成边界、MLIR 和双环境发布不能当作 R3 随手可复用的前置 |
| R3 存储 | `tnca_vrf_banked`、`tnca_rf_bank`、collector 已存在；本地 RF 测试通过 | 有组件基础，尚无有限端口单核闭环 |
| R3 集成 | `tnca_core.v` 仍例化 `tnca_vector_rf`；matrix 的 VRF 接口仍为地址/组合数据；`tnca_acc_rf.v` 仍异步读、全数组 reset | 新 VRF 未接 core；ARF 同步化也未交付 |
| R3 性能 | 模型手算测试通过；已有计数器与结构综合工具 | 尚未证明真实程序的模型精度、端口成本和资源进展 |

## 2. 状态、测试与合同之间的差距

1. **PLAN-01 检查不足。** `tools/r3/plan.py` 读取历史联合文件中的 `verdict` 即通过；没有验证当前提交、R2 前置或证据完整性。此次 exit=0 不证明旧 PLAN-01 成立。
2. **32 次的含义变了。** 旧合同规定 16 workload × reference/candidate；`spec/r3_workloads.json` 与 `check_expansion()` 实际统计 workload × fast/slow/ooo。数量相等不能证明对照实现存在。
3. **unit 合同未冻结。** PLAN-04 允许 7 个参数未测量并返回通过。这适合作为调研进度，不能证明原合同要求的 candidate 端口、延迟、队列和吞吐公式齐备。
4. **综合范围不足。** `synth_tests.py` SYN-01 检查 VRF/ARF memory 位数，但 SYN-02 的同步读和端口约束只检查 banked VRF。ARF 保留 memory 对象不等于 ARF 已成为同步有限端口实现；R3-10 的 DELIVERED 不能覆盖原合同全部要求。
5. **组件语义不能代替架构语义。** RF-03 验证未写 bank 读为 X；原 core 的初始化行为还需要 valid/zero 或有界 scrub 实现。直接接入会改变软件可见行为的风险尚未闭合。
6. **模型自测不是模型校准。** 手工 DAG 有限资源检查有价值，但没有建立实际编译产物到模型的完整映射，也没有留出 RTL 误差证据。
7. **文档汇总过时。** R2 状态开头称 11 包已满足，而表内实际为 7 DELIVERED、5 PARTIAL、2 NOT STARTED；R3 开头称 4 包部分满足，表内为 5。逐包状态和实际 case 才是依据。

这些差距主要是集成路径、验收语义和依赖组织问题。解决方式是缩小一次交付的范围，并让 checker 覆盖所声明的性质；不能只改 PASS 数量或改状态文字。

## 3. 本次执行与限制

执行记录：[checks.json](audit/2026-09-20-r3-replan/checks.json)。原始输出：

| 入口 | 结果 | 能证明什么 |
|---|---|---|
| `python3 tools/r3/plan.py` | exit 0，4 PASS | 当前脚本接受当前清单；存在上节列出的语义缺口 |
| `python3 tools/r3/model_tests.py` | exit 0，5 PASS | 有限资源模型的现有自测通过 |
| `python3 tools/r3/rf_tests.py` | exit 0，5 PASS | banked VRF 组件当前定向/随机比对通过 |
| `python3 tools/r3/flow_tests.py` | exit 1，4 PASS、FLOW-03 NOT RUN | collector 已实现部分通过；返回/写回闭环仍缺失 |

原始日志分别为 [plan](audit/2026-09-20-r3-replan/plan.log)、[model](audit/2026-09-20-r3-replan/model_tests.log)、[RF](audit/2026-09-20-r3-replan/rf_tests.log)、[flow](audit/2026-09-20-r3-replan/flow_tests.log)。本次没有重跑综合、系统 RTL、Triton 产品矩阵或远端发布。

已有 [timing scout](evidence/r3-timing-scout-macos.json) 记录 core-matrix 3513 cycles/6.316 s、system 18673 cycles/76.306 s。它们是历史 macOS 测量，并非当前 candidate 的速率保证。新阶段不再把整机性能矩阵和新增算子作为有限端口单核集成的前置。

## 4. 合理的下一阶段目标

新的 [R3 v2 tasks & acceptance](phase3_tasks_and_acceptance.md) 聚焦：**一个实验性单核实现，把现有 banked VRF/collector 和全部 VRF 客户端接通，完成写回反压、所有权、有限小程序与安装后 runtime 闭环。**

- R1：保留历史签收；本阶段对候选提交做影响范围回归。生产发布仍须按原合同重新签收。
- R2：继续单独完成 MLIR、四 tile runtime、host 完成边界与发布。本阶段只依赖已实现的单核路径，不要求整个 R2 先结束。
- R3 v2：交付上述单核实验实现，不修改既有稳定 profile 的默认实现。
- 后续 R3B：同步 ARF、全部计算单元吞吐、模型与编译成本闭合。
- 后续 R3C：等 R2 系统 runtime 完成，再做四 tile、Norm、系统校准和完整发布。

旧 [R3 v1](phase3_tasks_and_acceptance_v1.md) 保留供追溯；未完成要求明确转入后续，不能追认旧 R3 ACCEPTED，也不能据新版局部签收宣称项目已生产级。
