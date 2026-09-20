# R3 状态：逐工作包的实际进度

合同：[phase3_tasks_and_acceptance.md](phase3_tasks_and_acceptance.md)。
前置：[R1 签收](r1_signoff.md)（**R1 已 ACCEPTED**）、[R2 状态](r2_status.md)。

本文件只记录**已经跑过并留下证据**的部分。状态只有 DELIVERED、PARTIAL、
NOT STARTED 三种；没有完成度百分比。

**R3 未签收。** 十二个工作包中两个已满足全部验收条件，三个部分满足。

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
| R3-10 存储结构综合 | **NOT STARTED** | `tnca_rf_bank.v` 写成可综合形式（`SYNTHESIS` 下不含 `initial`/`$display`） | SYN-01…05 |
| R3-11 预算门禁与 mutation | **NOT STARTED** | — | GATE-01…05 |
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
