# Phase 3.5 状态：编译器复用与可观测优化

合同：[phase3_5_tasks_and_acceptance.md](phase3_5_tasks_and_acceptance.md)。
状态：**PLANNED / NOT ACCEPTED**。本次只交付计划，没有实现或测量新增优化。

阶段启动前提：**R3 v2 已验收**。使用其固定的 `core-vrf-sync-v1` 硬件、有限端口、计数和 runtime；本阶段只改变编译器，不把硬件改进计入编译器收益。此处是阶段假设，不是对撰写时 R3 状态的签收声明。

| 工作包 | 状态 | 负责人 / 完成提交 / 证据 |
|---|---|---|
| R35-01 基线、依赖与 inventory | NOT VERIFIED | 实施时填写 |
| R35-02 上游 adapter 与构建 | NOT VERIFIED | 实施时填写 |
| R35-03 LL 语义与转换 | NOT VERIFIED | 实施时填写 |
| R35-04 MLIR 优化与 AxisInfo | NOT VERIFIED | 实施时填写 |
| R35-05 分配与调度 | NOT VERIFIED | 实施时填写 |
| R35-06 安全差分与 mutation | NOT VERIFIED | 实施时填写 |
| R35-07 RTL 收益与消融 | NOT VERIFIED | 实施时填写 |
| R35-08 API、cache 与报告 | NOT VERIFIED | 实施时填写 |
| R35-09 回归、复现与证据 | NOT VERIFIED | 实施时填写 |

签收须满足合同的全部 27 条标准，包含至少两项小 workload 的实际周期收益、至少一项 LL 驱动收益的消融证据、逐项不退化约束。链接上游库、模型预测或只减少静态指令均不能代替实测。
