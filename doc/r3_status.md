# R3 状态：v2 集成计划与 v1 历史进展

当前合同：[R3 v2](phase3_tasks_and_acceptance.md)。状态：**NOT ACCEPTED**。
重规划依据：[2026-09-20 复查](r3_replan_review_2026-09-20.md)。

v2 聚焦同步有限端口 VRF 的真实单核集成。G0–G7（基线、VRF、客户端接入、
返回写回、E2E、计数、综合回归、证据）目前全部 **NOT VERIFIED**。
已有组件可以复用，但旧用例的 PASS 不自动转换成新标准的 PASS。

本次本地复查：plan 4 PASS、model 5 PASS、RF 5 PASS；flow 的四项已实现检查
通过，FLOW-03 未实现，整套 exit=1。详见复查文档和原始日志。

## 以下为 v1 历史工作记录

历史合同：[R3 v1](phase3_tasks_and_acceptance_v1.md)。以下 DELIVERED 是当时的
工作包记录，不能据此宣称原合同完整验收。复查发现 PLAN-01/03/04 与合同语义
有差距，SYN-02 没有核验 ARF 同步端口；相应包仍需按具体性质重新评估。
现有 `tools/r3` 的旧 ID 和注释归属于 v1；v2 使用独立 `R3V2-*` ID，待实现新 gate。
前置：[R1 签收](r1_signoff.md)（**R1 已 ACCEPTED**）、[R2 状态](r2_status.md)。

本文件只记录**已经跑过并留下证据**的部分。状态只有 DELIVERED、PARTIAL、
NOT STARTED 三种；没有完成度百分比。

**R3 v1 未签收。** 下表记录三个 DELIVERED、五个 PARTIAL、四个 NOT STARTED；
DELIVERED 的证据范围限制见上文，不等于已满足原合同全部条件。

## R3 v2（当前合同）逐 Gate 状态

合同：[phase3_tasks_and_acceptance.md](phase3_tasks_and_acceptance.md) v2.0。
签收名 `R3_V2_CORE_VRF_ACCEPTED`。**当前 NOT ACCEPTED。**

| Gate | 状态 | 已通过的标准 |
|---|---|---|
| G0 基线/清单/预算 | **DONE** | R3V2-BASE-01/02/03。`spec/r3v2_contract.json` 与 `spec/r3v2_inventory.json` 冻结；checker 拒绝八种畸形清单，全部在启动 RTL 前；Linux scout 实测 409–518 cycles/s，计划投影 ~139k cycles / ~370s，在预算内 |
| G1 VRF 包装/初始化 | **DONE** | R3V2-RF-01/02/03。per-lane valid 位让架构读零而数据阵列仍是 X；三个 seed 逐 lane 与独立模型零分歧；差分比对抓出两个真 bug（响应被 hold 时丢失目的地；一个端口可以拿到第二个未取走的响应） |
| G2 全客户端接入 core | **DONE** | R3V2-CORE-01/02/03。`make core-vrf` 23 个程序全过；elaborated 层级只含 `tnca_vrf_banked`+`tnca_rf_bank`，无 `tnca_vector_rf` shadow、无层级直读；`t29_vrfbank` 定向制造同 bank 串行 4 次、跨 bank 并行 8 次、12 次真实拒绝；matrix 544 次请求等 272 次 |
| G3 写回/返回/所有权 | NOT STARTED | R3V2-FLOW-01/02/03 |
| G4 六程序与安装后闭环 | NOT STARTED | R3V2-E2E-01/02/03 |
| G5 计数与微模型 | 部分 | core 已有 per-client 计数（req/accept/conflict/stall/serial/parallel/mtx_wait/wb_retry）；R3V2-COUNT-01/02、MODEL-01 未做 |
| G6 结构综合/继承回归 | 部分 | 继承回归在 candidate 上通过（23 程序 + slow/ooo + top + lint）；R3V2-SYN-01/02、REG-01 未正式记录 |
| G7 负例/证据/签收 | NOT STARTED | R3V2-MUT-01、EV-01、REL-01 |

候选相对参考的周期代价（同程序、同参数、同 seed）：整体 +13%，矩阵类
+42…46%（每次 matrix VRF 读多一拍），tensor +36%，纯向量 +16%，纯标量 +0%，
`t18_squash` -14%（更严的 interlock 改变了分支解析时机）。**不要求 candidate
更快**；每一项增加都有公开的原因。

## 逐包状态

| 工作包 | 状态 | 实际交付 | 未满足的验收条件 |
|---|---|---|---|
| R3-01 速率测量与预算 | **DELIVERED** | `tools/r3/scout.py` 先测真实仿真速率再定预算；`spec/r3_workloads.json` 的 16 个 workload 展开为 32 次性能运行，尺寸上限是合同的，由 `tools/r3/plan.py` 在任何 RTL 开跑**之前**核验 | 无 |
| R3-02 有限资源性能模型 | **DELIVERED** | `tools/r3/model.py` 与手算微图对照；MODEL-01…05 由 `tools/r3/model_tests.py` 执行 | 无 |
| R3-03 同步 VRF/ARF 与存储包装层 | **PARTIAL** | RF-01…05 全通过。`rtl/core/tnca_rf_bank.v`（一读一写端口、1/2 拍可选延迟、per-lane 写 mask、write-first 旁路显式计费）与 `rtl/core/tnca_vrf_banked.v`（低位交错 4 bank，容量正好 8 KiB，映射可逆）。RF-02 用 `tools/r3/rf_model.py`（照合同而非照 RTL 写）在三个 seed、两种延迟、冷热两种地址分布下逐 lane 比对，零分歧；比对自带对照组：latency-1 的响应对 latency-2 的模型**必须**不一致 | ARF（16 KiB 累加器文件）尚未做同样的包装层 |
| R3-04 Operand collector/仲裁 | **PARTIAL** | FLOW-01/02/04/05 通过。`rtl/core/tnca_operand_collector.v`：每条指令一个 entry、最多三个源，收齐才 ready（同 bank 三读串行 3 拍，跨 bank 一拍收齐，数值分别是 v0/v4/v8 与 v0/v1/v2）；被拒的请求重发而不是丢弃，`n_conflict`/`n_wait` 记录真实等待；round-robin 让四个 entry 在第 10/11/12/13 拍依次 ready（独占端口会是 9 拍的跨度）；12 次必需读恰好 12 次 grant，一个 bank 每拍最多 1 次；flush 丢弃未发射的 entry，但**不**丢弃已被 RF 接受的读——它们被标记为 orphan 吸收掉，槽位立刻复用也不会被写坏 | FLOW-03 需要 R3-05 的写回单元 |
| R3-05 DMA/MMA/SIMD 真实吞吐 | **NOT STARTED** | — | UNIT-01…05 |
| R3-06 计数器/ROI/trace | **PARTIAL** | COUNT-01/02/05 通过。`spec/r3_counters.json` 冻结 23 个计数的事件定义、reset/epoch 规则、溢出行为与宽度。**COUNT-02 发现并修掉一个真问题**：core 原来只有 hazard/lsu/wb 三个 stall 计数，它们互相重叠且另外六个 stall 原因根本没计，所以在真实程序上**两个方向都错**——t16_vecadd 多算 48 拍、t22_atomic 多算 143 拍，而 t12_tensor 少算 343 拍。现在有九个按固定优先级互斥归因的计数，在 9 个程序上其和**恒等于** `perf_stall_total`，重叠视图另列保留。COUNT-01 用 `t28_count`（12 条指令，每条编码正好一个字，直线无分支）与 operand collector 的 12 次必需读逐一对上 | COUNT-03 需要 mesh 每 link/VN 的计数；COUNT-04 需要 ROI 时间戳 |
| R3-07 编译布局与成本反馈 | **PARTIAL** | R2-12 OBS-01 已让 manifest 记录**实测**资源（AXPY 用 4/64 向量寄存器）而不是 profile 上限，未测量的成本标 `unknown` | CC-01…05 需要 bank mapping 进入 TargetInfo |
| R3-08 小算子数值路径 | **NOT STARTED** | — | NUM-01…05 |
| R3-09 模型校准与留出验证 | **NOT STARTED** | — | CAL-01…05 |
| R3-10 存储结构综合 | **DELIVERED** | SYN-01…05 全通过。`flow/synth/mem_synth.py` 用 `proc; opt; memory -nomap; opt` 做**保留 memory** 的结构综合（旧的 `synth.py` 会把寄存器堆映射成几十万个触发器，回答的是别的问题）。结果：`tnca_vrf_banked` 4 个 memory 对象共 65536 bit = **正好 8 KiB**，每个 **1 读口 1 写口、同步读**；作为对照，被它替换的 `tnca_vector_rf` 是 **4 读口 130 写口、异步读**。数据阵列不在 reset 里清零；collector 的 162 个触发器位与 136 bit 控制阵列**单独列账**，不计入容量。无 PDK，所以只写 memory-inferred，Fmax/面积/功耗一律 NOT MEASURED，且测试会检查报告**没有**声称 SRAM 面积或频率 | 无 |
| R3-11 预算门禁与 mutation | **PARTIAL** | R3M-01a/01b/04a/04b/05/06 六个变异全部被指定检查抓住，全部注入在隔离副本，候选源码哈希不变：同步读提前一拍与跨 bank 串数据被 RF-02 抓住；stall 重叠重复计数被 COUNT-02 抓住；少数一个端口的 grant 被 COUNT-01 抓住；模型 RF 端口无限被 MODEL-02 抓住；workload 偷偷把 K 撑到 4 倍上限被 PLAN-02 在**启动 RTL 之前**拒绝 | R3M-02/03 的检测器（FLOW-03、UNIT/NUM）尚不存在；BUDGET-02/03/04 未做 |
| R3-12 双环境与发布证据 | **NOT STARTED** | — | R3REL-01…05 |

## 这一阶段已经成立的核心性质

存储包装层与一个**独立**模型逐请求一致，并且这个比对被证明会失败：

```text
    seed 1 spread latency 1 (139 request(s), 40 warm-up write(s)):
        654 lane comparison(s), 1186 against never-written state, 0 problem(s)
    control: latency-1 responses checked against a latency-2 model -> DISAGREE (good)
    four requests to one bank, every cycle: at most 1 accepted per cycle
    a register never written reads as X (unknown)
    100 registers in 8 banks of 12: refused at elaboration
```

## 一条必须继续说清楚的边界

`tnca_vrf_banked` 替换的是 `tnca_vector_rf` 的**存储实现**，不是它在 core 里
的位置：core 目前仍然例化行为模型。把 core 切到 banked 实现需要 operand
collector 与仲裁（R3-04），在那之前说"core 跑在真实端口上"是不成立的。
