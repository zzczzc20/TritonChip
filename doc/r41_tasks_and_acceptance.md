# R41：标准 Torch 推理与编译—硬件—性能模型协同迭代

**版本：** 0.2  
**日期：** 2026-10-20  
**状态：** PROPOSED / NOT FROZEN / NOT ACCEPTED

**阶段签收名：**

- `R41_S1_COMPILER_AND_PERFORMANCE_HARNESS`
- `R41_S2_100MHZ_AND_FULL_MODEL`

本文中新增的文件、工具和 Make target 均为拟交付内容，不表示已经实现。

---

## 0. 合同原则

### 0.1 两个阶段，两个明确交付边界

| 阶段 | 主要运行平台 | 必须交付 | 不作为本阶段签收前提 |
|---|---|---|---|
| **Stage 1** | R40 已加载的三核、90 MHz、4 GiB 镜像 | 编译器关键修复；有预算的选择机制；完整 performance_harness；100 MHz 候选设计与模型可行性准备 | 新 bitstream、100 MHz、完整 Qwen 上板 |
| **Stage 2** | 新三核镜像，目标且签收频率为 100 MHz | 真实时序闭合；容量与长生命周期；完整 Qwen 文本推理；原套件及有限 XL；选中优化的板上验证 | 所有研究机制都产生收益、所有目标倍数都达到、8/16/32 核物理交付 |

Stage 1 可以独立发布和签收。

Stage 2 未完成时，不得把整个 R41 标为完成；但不得撤销已经成立的 Stage 1 交付。

### 0.2 区分三种要求

| 类型 | 含义 | 关闭方式 |
|---|---|---|
| **Required** | 必须具备的功能、正确性、安全性或证据 | 必须通过 |
| **Study** | 必须完成的有限研究或候选评估 | 可以是正结果、负结果或预算内无法确认；必须提交证据和决策 |
| **Target** | 性能、效率或覆盖面的努力目标 | 报告达到与否，不单独阻断阶段签收 |

特别规定：

- `closed-negative / default-off` 可以关闭研究任务，**不是性能通过**。
- 未实现、未运行、缺证据不能写成 capability-pass。
- Required 不允许通过写一份负结果报告替代。
- 因预算停止的未完成 Required，保持 `blocked` 或 `fail`；不得在收尾时改成 Target。

### 0.3 分阶段冻结，而不是反复修改同一合同

共同原则与 Stage 1 范围先冻结。

Stage 2 的具体新增负载、数值容差、资源预算和测量计划，在 Stage 1 可行性检查完成后冻结。此时允许使用：

- 官方参考实现；
- R40 已有测量；
- 编译规模与容量预检；
- 不涉及候选收益选择的功能性试运行。

不得根据 Stage 2 候选优化的收益挑选有利验收点。

**两次冻结约束不同阶段，不是把失败合同从 v1 改成 v2、再改成 v3。**

---

## 1. R40 起点与转入事项

### 1.1 历史状态不改写

R40 保持以下状态：

> `closed-with-failures`，`R40-10.acceptance` 未通过。

R41 不要求先使 R40 的历史合同全部通过，也不追认 R40 已通过。

R40 基线命名为 **B40-OBS**，不再使用暗示正式签收的 `B40-F`：

| 项目 | B40-OBS |
|---|---|
| 软件 | `files:fe1b5199db515440d2d3407aed61ef1eed5edb627e5baf9cda10be5a7198cf75` |
| 镜像 | `a88ebdf59cfbbb2001ebd5b1b2c739c5d8cd33931f1e9202f18cac07c4c83083` |
| 频率 | 实测 90 MHz |
| 物理配置 | 三核、三个 HBM 出口、4 GiB 暴露窗口 |
| 证据 | R40 v3 最终 campaign、gate records、报告及身份修订 |

这些身份来自 R40 最终报告；R41 冻结时保存不可变引用。([github.com](https://github.com/zzczzc20/TritonChip/blob/r40/doc/r40_report.md))

### 1.2 转入台账

新增 `spec/r41/carryover.json`，至少包含：

| R40 项目 | Stage 1 | Stage 2 | R41 处理原则 |
|---|---|---|---|
| `R40-01.device-time` | 修正不必要的同步；测量地址放置敏感性 | 最终镜像复验相关点 | 不追改 R40 判定 |
| `R40-03.gemm-tile` | 修复非原生 tile 的 fragment lowering | 与新 MMA 联合验证 | 编译修复优先于扩阵列 |
| `R40-03.wide-blocks` | 按算子与资源合法性选择；分开设备时间与 host gap | 最终策略回归 | 不强制全局宽块 |
| `R40-02.batched` 未达目标 | 对长 launch 链做一次定向诊断 | 有明确修复时复验 | 不无期限追求 1.2× |
| `R40-04.bandwidth`、`R40-06.ports` | 建立 home-slice/核侧服务账本 | 评估服务流水化 | 不把出口数量当作收益 |
| `R40-06.locality` / DCO | 设计标准路径消费者和安全边界 | 最小可用接入、板测与默认策略 | 功能接入 Required，收益 Target |
| `R40-07.long-k` | 保守选择 Mode B、单 role、Mode A | 配合 transfer-engine 候选重测 | Mode A 不强制默认开启 |
| `R40-07.matrix` | 候选建模、资源和语义预检 | RTL 候选及选中实现的验证 | 不强制指定阵列规模 |
| `R40-08.prefill` | 查清寄存器、搬运和计算成本 | 通用 D256 与混合模型实现 | 不强制使用更慢的融合路径 |
| `R40-09.autotune/holdout` | 接通有预算选择与缓存 | 最终策略的冻结 holdout | 不要求所有点都调优 |
| M1–M4 与频率工作 | 合并候选、更新路径 census | 新镜像实施与上板 | 未上板不算硬件交付 |
| SystemC M2/M3 | 工具化、校准流程与双内存路径 | 新镜像配置和适用域更新 | 模型结果不替代板测 |

R40 合同明确转出的 locality、autotune、holdout、MMA，以及条件性的宽入口消融，均保留原始来源和状态。([raw.githubusercontent.com](https://raw.githubusercontent.com/zzczzc20/TritonChip/r40/doc/r40_tasks_and_acceptance_v3.md))

---

## 2. 两阶段共同的不可变要求

### 2.1 唯一正式执行路径

```text
官方 PyTorch 模型 / 模型无关的 Torch 图
  → torch.Tensor(device="tnca")
  → torch.compile(backend="inductor", fullgraph=True)
  → Inductor 通用分解、调度、融合、Triton template
  → Triton frontend / TNCA backend
  → 共同 lowering / ISA / artifact
  → 同一 Session、allocator、event、grid dispatcher
  → TNCA / NUCA / NoC / HBM
```

要求：

- tensor descriptors 保持开启。
- 允许有明确理由的 pointer lowering，不要求所有访问都变成 descriptor。
- 禁止模型名、层名、checkpoint hash、benchmark ID 特判。
- 禁止 CPU/CUDA 数值回退和隐式 tensor staging。
- 禁止通过模型 adapter 直接调用专属 TNCA kernel。
- 允许公共 Inductor 选择机制使用工程师实现的通用 Triton template。
- 合法回退必须仍在 TNCA 标准路径内，例如缩小 tile、使用通用分解或保守调度。

模型的 prefill 和固定 bucket decode 分别 fullgraph；Python 生成循环可以作为 host orchestration。

### 2.2 数值和协议

- 原冻结点的 shape、dtype、容差、point hash 不变。
- 现有 MMA 保持逐 K 的 FP32 FMA 舍入语义。
- 不通过 split-K、树形归约或降低累加精度暗中改变旧指令。
- source-free 与 result-ready 不得混用。
- alias、异步 lease、masked store、ownership、fault drain 和 completion visibility 必须保持。
- 编译资源不足须提前拒绝或选择合法配置，不能运行到一半才发现 code store 或寄存器溢出。

当前矩阵单元是 16 MAC/cycle，现有四种 MMA 输入格式通过 widening 使用共同 FMA；五格式 FMA 支持不等于五格式 MMA 支持。([raw.githubusercontent.com](https://raw.githubusercontent.com/zzczzc20/TritonChip/r40/rtl/core/tnca_matrix.v))

### 2.3 保留原套件

两阶段最终软件身份均须保留：

- native tiny/small/medium：77 点；
- large：24 点；
- 冻结 HBM inventory；
- R40 四层模型原 workload；
- 与修改有关的 ISA/compiler/runtime/protocol 回归。

完整板测放在每阶段最终候选上执行一次。开发期间只跑相关子集，不要求每次提交重跑全部大负载。

---

## 3. 技术主线：先统一执行资源描述，再局部改实现

R41 不建设第二套编译器或第二套 runtime。

建议增加一个轻量的 **execution resource plan**，由现有编译阶段产出 sidecar，连接现有布局、分配、调度、模型和选择器：

```text
逻辑 tile / valid mask
    ↓
物理 fragment、pack、conversion
    ↓
VRF / window / accumulator 生命周期
    ↓
DMA / MMA / LSU 命令及依赖
    ↓
资源需求、合法候选与性能诊断
```

至少记录：

- 物理 fragment 数与有效 MAC / padding MAC；
- K 循环中的 accumulator 往返；
- VRF 峰值、window、ACC、barrier、code bytes；
- descriptor 与 pointer 传输；
- transfer/MMA 命令数、依赖与潜在重叠；
- 选择或拒绝某配置的原因。

**这个 plan 是共同的资源描述，不是共同的数值 oracle。**  
数值验证仍使用独立参考；SystemC 仍执行真实 artifact，不能只执行编译器给出的成本摘要。

当前 LinearLayout 已用于 view 组合和转换分类，物理 bank 分配另有实现；R41 应连接这些现有部分，而不是重新证明“调用过 LinearLayout”。([raw.githubusercontent.com](https://raw.githubusercontent.com/zzczzc20/TritonChip/r40/compiler/python/tnca_cc/layout.py))

---

# Stage 1：90 MHz 上的软件修复与快速性能迭代闭环

## S1-00：冻结基线、范围和预算

### 交付

- B40-OBS 身份与只读证据索引。
- 原套件 inventory 和 hash 审计。
- Stage 1 required/study/target 清单。
- 固定的性能诊断子集、选择器开发集和 holdout 划分。
- 编译、模型、板测与 worker 预算。
- 转入台账。

### Required

1. 不依赖 R40 历史终验重新通过。
2. 不能把旧 PASS 直接复制成 R41 PASS。
3. 相同身份、point、配置、计时边界的记录允许复用，避免一个基线为多个 gate 重跑。
4. 不同镜像、代码、cache 状态或计时边界不得错误去重。
5. 冻结文件中 required workload、参考、容差、预算不得为 `null`。

---

## S1-01：把 performance_harness 交付为完整工作流

### 目标

不是只增加一个 SystemC simulator，而是形成：

```text
问题与假设
  → 编译产物/资源账本
  → 分析下界
  → SystemC 快速筛选
  → 必要时详细内存模型
  → 小型 RTL 验证
  → FPGA 验证
  → 预测残差、归因与下一步决策
```

现有 harness 已包含成本约束、计数器、Proton、RTL 实验与物理报告接口；R41 应复用并整合，而不是再建立平行工具链。([raw.githubusercontent.com](https://raw.githubusercontent.com/zzczzc20/TritonChip/r40/tools/performance_harness/README.md))

### A. 统一实验与证据格式

一个实验定义应包括：

- 假设、竞争解释、预期计数器变化；
- workload、输入、参考和配置；
- 软件、artifact、镜像或 model 身份；
- 时钟、memory map、工具版本；
- 测量边界与预算；
- 结果、残差、结论和下一步。

统一支持：

- compile-only；
- analytical screening；
- SystemC native execution；
- RTL diagnostic；
- FPGA measurement；
- compare/report。

支持断点续跑、精确去重和失败记录保留。

### B. SystemC-3/TLM-2 native execution

Required：

1. 继续经共同 Session 路径执行真实编译器输出。
2. 不在执行路径中嵌入 RTL simulator。
3. 不用捕获的板上 stall trace 驱动完整芯片预测。
4. 未实现 opcode、协议或 capability 明确拒绝。
5. 功能结果、时序预测、模拟器 wall time 分开记录。
6. `off/aggregate/trace` 模式不改变架构结果；trace 有界并报告丢弃。
7. 新 RTL 语义必须更新对应 model profile，不能继续套用旧芯片配置。

M2 已经加入核心流水、LSU、VRF bank、transfer engine 和 home-slice 等结构建模；这些不应再次列为“从零实现”。但其现有整体时序判定仍为未校准：holdout median 21.5%、p95 34.7%。([raw.githubusercontent.com](https://raw.githubusercontent.com/zzczzc20/TritonChip/r40/evidence/r40/v3/systemc_m2/README.md))

### C. 快速内存路径与 Ramulator2 路径

交付两个可切换路径：

| 路径 | 用途 |
|---|---|
| 快速、RTL 结构化服务模型 | 日常编译筛选、发射/寄存器/引擎瓶颈分析 |
| Ramulator2 详细 DRAM 路径 | 对 bank、row、读写切换、refresh 等敏感的问题 |

要求：

- 固定依赖 revision 和配置。
- 明确 AXI、NoC、home engine、controller、DRAM 的边界。
- 不能在详细路径中重复计入快速模型已经包含的延迟。
- 正确传递 backpressure、完成事件和写响应。
- 使用实际可核对的 PC/address map。
- 详细路径不作为所有实验的默认路径。

**接入 Ramulator2 不自动意味着模型更准确。**必须比较它解决了哪些残差、增加了多少模拟成本。R40 报告中的 M3 仍是继续推进的工作，不能提前记成已完成校准。([github.com](https://github.com/zzczzc20/TritonChip/blob/r40/doc/r40_report.md))

### D. 校准与适用域

分别记录：

- structural constant；
- fitted parameter；
- calibration set；
- validation set；
- sealed holdout；
- 适用镜像、配置和 workload family。

规定：

1. 不用 holdout 拟合。
2. 已被反复用于诊断的数据不继续称为未见 holdout。
3. 分别报告绝对周期误差、A/B 比值误差、排序错误和无收益判断。
4. 对 out-of-domain 情况返回明确状态，不输出伪精确结论。
5. 模型不确定性大于候选差异时，不据此淘汰合法候选。
6. 全局未达到原 10%/20% 精度标准时，仍保持 `uncalibrated`；允许另外报告某个明确域的验证结果。

**harness 可交付，不等于全芯片时序模型已经全面合格。**

### E. Stage 1 验收

Required：

- 一个干净环境可以构建并运行 smoke campaign。
- 至少完成三个闭环案例：
  1. GEMM fragment/codegen；
  2. memory service / ports；
  3. pipeline / engine admission。
- 至少包含一个负结果。
- 至少一个未参与拟合的候选，在板测前保存预测，板测后保存残差。
- 模拟器相对 RTL 的速度在相同小负载上实测，不用超时结果编造精确倍数。
- 现有明确的模型失配点保留，不通过删除它们提高精度。

不设置“所有模型点 median ≤10%”作为 Stage 1 阻断 gate。

---

## S1-02：修复宽 GEMM tile，并接通资源驱动布局

### Required 交付

首先修复 `16×32` 逻辑 tile：

- 用两个合法 `16×16` accumulator fragment 表示。
- K 循环内保持 accumulator 驻留。
- 不再为维持非原生逻辑 tile，反复逐行读回、计算和写回 accumulator。
- epilogue 中必要的读回不视为违规。
- K 顺序不变。
- tail、mask、非连续输入和资源不足路径有测试。

R40 的问题不是“宽 tile 天生不适合 TNCA”，而是现有 lowering 产生了约 12 倍指令和大量逐行 accumulator 命令。([raw.githubusercontent.com](https://raw.githubusercontent.com/zzczzc20/TritonChip/r40/evidence/r40/v3/R40-03.gemm-tile/model_diagnosis/README.md))

### 范围限制

Stage 1 不要求一次支持所有宽 tile。

- `16×32` 是必需修复范围。
- 更大 tile 按资源合法性逐步支持。
- 不适合的 shape 保留原合法实现。
- 不能始终退回 `16×16`，却宣称已经修复多 fragment 路径。

### 布局和分配

把 execution resource plan 接到现有：

- LinearLayout 转换分类；
- fragment/pack；
- window lifetime；
- bank-aware allocation；
- retile/refusal。

不要求本阶段建设通用最优调度器或完整 spill 系统。

### 验收

1. 至少两个标准 Torch 图真正执行修复后的多 fragment 路径。
2. 数值、异步 ownership、资源计数通过。
3. 生成代码和动态计数证明病态搬运已移除。
4. 原 GEMM-tile 子集完整复测。
5. 分别报告：
   - 相对旧错误映射的修复收益；
   - 相对原保守 `16×16` 产品路径的真实收益。

不得把“修复一个很慢的候选”直接包装成整个产品同倍数提速。

---

## S1-03：同步、宽块与有预算自动选择

### A. 同步修复

审计 `tl.debug_barrier()` 到 TNCA ordering 的映射。

- 只在能够证明所需顺序的情况下弱化或消除同步。
- 不能全局把 barrier 改成 no-op。
- 跨 role、store/reload、alias 和异步读写均有负例。

R40 的 softmax 回退包含额外 device fence；部分 tiny 点则是同机器码下的 staged-buffer 放置差异，不能都归因于编译器。([raw.githubusercontent.com](https://raw.githubusercontent.com/zzczzc20/TritonChip/r40/evidence/r40/v3/R40-01.device-time/regression_diagnosis.md))

### B. TNCA_TUNE 最小可用实现

Required：

- 进入公共 Inductor/TNCA 选择路径。
- 候选依据语义、shape、stride、dtype、资源和 capability，不依据 benchmark 名称。
- 保守合法配置始终存在。
- 默认优先少量候选，建议最多 4 个；硬上限仍为每 kernel 16 个实测候选。
- 每图 tuning 增量预算最多十分钟；基线编译成本另记。
- worker 超时实际终止；设备任务正确 drain/recover。
- tuning 样本与正式测量隔离。
- cache key 覆盖会影响合法性和选择的代码、配置、硬件和内存映射身份。
- 不用 simulator wall time 选择配置。

可以先支持以下有限候选：

- 窄/宽 block；
- 原 tile/修复后的多 fragment tile；
- Mode B/单 role 流水/现有 Mode A；
- 合法 fused/decomposed 路径；
- single/batched submission。

不要求一次搜索所有开关的笛卡尔积。

### C. 策略安全

- 测量预算不足、结果不确定或模型 out-of-domain 时，保留保守配置。
- 选择失败不能变成 CPU fallback。
- 未证实有效的机制默认关闭。
- Stage 1 不要求 autotune 总体达到 1.1×，但必须交付真实选择、缓存、预算和错误处理。

### D. Locality/DCO 准备

Stage 1 完成：

- 通用消费者设计；
- storage/alias/lifetime 与 policy 的对应关系；
- 将执行到哪些已有控制或指令；
- 不安全情况的拒绝规则；
- Stage 2 板测方案。

仅设置环境变量而没有实际消费者，不算接入。

---

## S1-04：100 MHz 候选设计与完整模型可行性准备

本包并行推进，但不要求 Stage 1 加载新镜像。

### A. 硬件候选

以 `r40-final-image` 的 M1–M4 为起点：

- 注册 host grant；
- 注册 host_allow；
- 本地化并注册 memory controls；
- 按 ISA 修复 float-to-int。

这些修复已经有候选验证，但没有作为新镜像上板。([raw.githubusercontent.com](https://raw.githubusercontent.com/zzczzc20/TritonChip/r40-final-image/evidence/r40/v3/final_image_candidate/README.md))

进一步按**结构类别**处理时序，而不是逐条修补数百条路径：

- 全局 `stall_front` 与高扇出 enable；
- scalar read → barrier/scoreboard → issue；
- VRF bank/collector/writeback；
- FMA 输入选择和 pipeline 平衡；
- LSU/mesh 接口；
- slice RAM 控制。

优先考虑：

- 控制信号就地复制；
- 有明确 valid/ready 语义的寄存边界；
- 局部 credit/queue；
- 减少跨单元组合反馈。

**不能简单延迟 stall 一拍而不处理在途指令、重放、flush 和 ownership。**

### B. Required 交付

- 已知关键路径类别、责任模块、候选修改及验证计划。
- 合并 M1–M4 后的相关 RTL 回归。
- 单元综合/OOC 与区域资源预检。
- 下一次全设计实现的固定候选清单。

不要求 OOC 提供统一的某个正 slack 余量，更不把 OOC 当作路由通过。

### C. Qwen 可行性准备

在 Stage 1 完成：

- 官方权重与语言分支 distinct storage 统计；
- 所需算子和状态 dtype inventory；
- D256 attention、depthwise conv、FP32 recurrence 的代表图编译与功能检查；
- code size、编译时间、IR 和内存峰值估算；
- 公共模型接口与 cache adapter 差异清单。

完整 Qwen 可以提前试跑，但不是 Stage 1 的阻断项。

---

## S1-05：Stage 1 签收

`make r41-stage1-acceptance`

Required：

1. S1-00 至 S1-04 的 Required 交付完成。
2. 最终软件在原 90 MHz 镜像上完成原完整套件数值回归。
3. 多 fragment 修复实际执行，不能只交付未使用代码。
4. tuning 正式接入且保守回退可用。
5. performance_harness 三个闭环案例完成。
6. 新默认策略通过第 6 节非退化规则。
7. 干净 archive、fresh caches 重建软件和模型，并复跑少量冻结板上点。
8. 发布 Stage 1 报告和 Stage 2 ready/blocker 清单。

**Stage 1 不等待新 bitstream。**

---

# Stage 2：100 MHz 新镜像、完整模型与规模验证

## S2-00：冻结 Stage 2 最小交付集

进入正式实施前冻结：

- 最终模型 revision、权重与 tokenizer；
- 新增核心模型点；
- XL-core；
- 数值容差；
- 必需硬件改动；
- 候选研究列表；
- 编译、板时、tuning 和实现预算；
- 最终性能对照。

默认最多安排：

1. 一次集成实现；
2. 一次依据路由结果的修正实现。

第一轮实现后不继续加入无关功能。

达到实现预算仍未满足 100 MHz 时，提交明确失败状态和下一步，不通过降频改写合同。

---

## S2-01：三核 100 MHz 物理交付

### Required

1. 三个物理核继续位于三块 SLR。
2. 新镜像包含 M1–M4 及选中的时序修复。
3. 实际运行 100 MHz。
4. post-route setup、hold、pulse-width 和项目 CDC 审核通过。
5. 不删除既有 SFU、精度或语义范围换取时序。
6. 最终镜像 hash、源码、参数、工具版本和实际时钟绑定。
7. 不存在未解释的路由或物理错误。
8. 完成至少两小时连续混合运行；可与正式 campaign 合并，但须保留 session 和资源增长记录。

八小时 endurance 为扩展目标，不作为首次交付前提。

### 频率与性能分开

在设备周期完全不变的理想情况下，90→100 MHz 只对应约 11.1% 的速度提升；新增流水级可能改变周期数。

因此必须同时报告：

- 实际完成时间；
- 设备周期；
- 新增流水和资源代价。

不能把频率提升自动计为编译或内存优化收益。

---

## S2-02：内存服务与 MMA——有限候选、择优集成

### A. 内存服务

优先研究 home-slice 和核侧服务：

- lookup/hit/fill/wake/writeback 的资源占用；
- 请求接纳与完成的解耦；
- 小型、有界队列；
- hit-under-miss 与公平性；
- transfer engine 的命令接纳；
- 实际消费者的 outstanding 需求。

不要首先把工作变成“更多出口、更多 MSHR、更宽总线”。

R40 的端口实验中，模型显示单出口并未被充分使用，增加出口未改变 home engine 与核侧等待；这应作为候选选择的起点，而不是已经证明所有新 workload 都具有同一瓶颈。([raw.githubusercontent.com](https://raw.githubusercontent.com/zzczzc20/TritonChip/r40/evidence/r40/v3/model_attribution/README.md))

### B. Transfer engine

允许先实现小型命令队列，但必须区分：

- 能接纳多个命令；
- 能重叠 setup；
- 能重叠数据传输；
- 是否真正缩短图的关键路径。

**只把等待从 issue 端移动到队列里，不算吞吐优化。**

### C. MMA 与 XLS

至少评估一个针对明确瓶颈的 RTL 候选，例如：

- operand buffering / reuse；
- 沿独立 M/N 输出增加并行；
- accumulator banking/端口；
- FMA pipeline 平衡；
- skinny-M/GEMV 的有效工作率。

首轮不要求从 16 MAC/cycle 跃迁到某个大阵列规模。

候选必须保持：

- 既有四种 MMA fragment；
- 五格式 FMA/转换；
- 逐 K 舍入；
- source-free/result-ready；
- 独立数值 oracle。

XLS 是实现手段，不是签收成果。必须说明生成结构改善了什么，并报告 FPGA 实际映射；没有更好折中时允许不合入。

### D. Study 关闭条件

每个候选给出：

- 假设与支持证据；
- SystemC 敏感性实验；
- 相关 RTL 语义验证；
- 面积/时序预检；
- 集成或不集成的理由。

进入最终镜像的候选，必须增加标准 Torch 板上对照。

**不要求内存、MMA、流水三个方向都独立达到 1.5×。**

---

## S2-03：Locality、流水与寄存器资源再分配

### A. Locality/DCO 最小可用交付

Required：

- 至少两个模型无关的标准 Torch 图真实使用支持的 locality/DCO 操作。
- policy 能追溯到 storage、alias 和生命周期。
- 不安全的 alias、未完成 writer/reader 不使用激进 dead/bypass 提示。
- off/on 数值、ownership 和 fault 路径通过。
- 有板上流量与时间对照。

若没有性能收益，功能可以交付，策略默认关闭。

### B. Pipeline 比较

在固定 core、context 和总寄存器预算下比较：

- Mode B；
- 单 role 软件流水；
- Mode A 两级 ring。

三级 ring 仅在模型或计数器显示有必要时增加，不列为必做组合。

### C. 私有资源再平衡

将 producer/consumer 非对称资源需求纳入研究，但优先顺序为：

1. 改善现有合法 window/lifetime 分配；
2. 评估静态、按 kernel 的非对称资源分配；
3. 必要时才研究运行期寄存器再分配。

NVIDIA 的 `setmaxnreg` 依赖实际寄存器池、执行一致性和同步规则；不能把它理解成 TNCA 编译器改一个预算数字就能获得同样能力。([docs.nvidia.com](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html))

新增 TNCA 资源分配能力若涉及 RTL/ABI，必须显式声明 capability。现有 RPAIR alias 不等于新增寄存器容量。

本项为 Study，不强制新增动态资源分配 ISA。

---

## S2-04：容量、长生命周期与完整 Qwen 文本推理

### A. 模型范围

Required：

> 官方 Qwen3.5-2B checkpoint 的完整语言分支，真实权重、全部 24 层、完整词表、BF16 主干及参考要求的 FP32 状态，完成 prefill 和连续 autoregressive decode。

不包含：

- 视觉编码器和多模态推理；
- MTP；
- speculative decoding；
- 量化主验收。

官方配置包含 18 层 linear attention、6 层 full attention，full-attention head dimension 为 256，recurrent-state dtype 为 FP32。不能用普通 Llama + KV 支持代替。([huggingface.co](https://huggingface.co/Qwen/Qwen3.5-2B/raw/main/config.json))

官方实现提供文本模型类，也包含公共 Torch 计算和外部 kernel 替换钩子；正式路径须审计实际执行分支。([raw.githubusercontent.com](https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/models/qwen3_5/modeling_qwen3_5.py))

### B. 核心模型负载

为控制首次交付规模，新建明确 ID，不沿用原草案 Q0–Q3 的名称：

| ID | Batch | Prefill | 连续 decode | 级别 |
|---|---:|---:|---:|---|
| `Q41-core` | 1 | 128 | 16 | Required |
| `Q41-tail` | 1 | 129 | 4 | Required |
| `Q41-512` | 1 | 512 | 128 | Target |
| `Q41-2k` | 1 | 2048 | 256 | Target |
| `Q41-8k` | 1 | 8192 | 64 | Target |
| `Q41-b4` | 4 | 512 | 128 | Target |

这是对**尚未冻结的新增模型范围**进行分级，不修改任何 R40 原冻结点。

完整层数、权重和词表不减少；首次签收只不强制同时完成长上下文、长 decode 和 batch 扩展。

### C. 实现顺序

先完成合法 B41-C，再优化：

1. primitive；
2. 单层和状态更新；
3. 完整 `Q41-core`；
4. 完整 `Q41-tail`；
5. 可选扩展。

允许正确性基线使用通用分解，不强制一开始就完成最大融合。

特别注意：

- FP32 recurrent state 不能为了使用现有 BF16 MMA 而偷偷降精度。
- 不强制使用 R40 中已知更慢的 fused SDPA。
- 声称分块/线性内存 attention 时，必须另外证明其分配和流量。
- 不把每个 token 的完整 recurrent state 历史长期保存。
- 固定 decode bucket 内，position 更新不应逐 token 重新编译。

### D. 容量

以 6–8 GiB 可达容量作为实现规划，硬要求按实际峰值判断：

\[
C_{\text{usable}} \ge 1.25\,H_{\text{peak}}
\]

峰值包括：

- distinct weights 与 tied alias；
- KV、recurrent state、conv state；
- activation、scratch、pack cache；
- code、descriptor、保留区和 allocator 开销。

不把 checkpoint 文件总大小直接当语言分支显存，也不把它当每步 decode 的读流量。官方索引包含语言分支之外的权重。([huggingface.co](https://huggingface.co/Qwen/Qwen3.5-2B/raw/main/model.safetensors.index.json))

Required：

- 真实覆盖大 size/offset 和 >4 GiB 地址；
- memory-map 保留区不被 tensor 覆盖；
- 高位截断、越界和 alias 负例；
- bounded grid admission；
- 超过 65535 programs 的实际标准 Torch 执行；
- 跨 65536 次 barrier arm 的验证；
- code eviction、allocator reuse、连续调用无状态污染；
- 不发生稳态权重或 state 的隐性 host 重传。

### E. 数值

在 Stage 2 冻结前，用独立参考交叉校验确定新容差；不能看 TNCA 输出后放宽。

检查：

- logits；
- 各类 state；
- teacher-forced 连续演化；
- reset 与再次调用；
- tail、mask 和极端 gate。

TNCA 和 reference 在 teacher-forced 测试中接收同一 token，但各自维护自己的 state。

自由生成另做功能展示，不用字符串相同代替数值验证。

---

## S2-05：有限 XL、原套件与最终选择器

### A. XL-core

不建设无边界的大型新 benchmark 矩阵。

建议冻结 8 个主点：

| 类别 | 建议主点 |
|---|---|
| 大 storage streaming | 单输入 storage 超过 2 GiB，实际读写覆盖高地址 |
| transpose | `16384×16384 BF16`，真实物化输出 |
| GEMM | `M=512,K=2048,N=6144,BF16` |
| GEMV | `M=1,K=2048,N=6144,BF16` |
| 完整输出投影 | `M=1,K=2048,N=248320,BF16` |
| embedding/gather | 大表、int64 index、重复索引、跨区域访问 |
| attention | D256、GQA、causal、非整齐长度 |
| gated recurrence | FP32 state、chunk 边界、连续更新 |

精确输入、shape、stride、容差和 hash 写入 Stage 2 manifest。

更多 dtype、长度、batch 的交叉组合为扩展，不自动变成 required。

### B. 原套件

原 native77、large24、HBM inventory、四层模型全部保留。

任何原 large 点未通过，不能声称 large 全套通过。

### C. Holdout

- 保留 R40 原 holdout inventory 和历史状态。
- 若其中一些点已经参与策略开发，明确标记。
- 在 Stage 1 冻结新的小型、未参与开发的 R41 holdout。
- 最终比较保守配置与实际选择器输出。
- regret 只针对实际评估过的合法候选集合，不冒充全局最优 regret。

Required 是选择正确、预算有效、结果完整。`S_geo≥1.1` 和 p90 regret≤15% 保留为 Target。

### D. 多核

- 同一最终镜像验证 active cores 1/2/3。
- 只在预登记代表图和一个核心模型点做主要缩放比较。
- 不要求所有 XL、所有模型点、所有候选遍历全部核数。
- 保留既有参数化回归；新增 8/16 核完整功能 campaign 和 32 核规划不阻断三核交付。

---

## S2-06：最终签收与独立复现

`make r41-stage2-acceptance`

Required：

1. 100 MHz 新镜像真实合格。
2. 全部 Required 功能和数值检查通过。
3. 原完整套件与 XL-core 完成。
4. 完整 `Q41-core`、`Q41-tail` 完成。
5. 所有已启用的新性能机制有对应板测和默认策略依据。
6. 更新最终镜像的 SystemC profile，重新报告适用域和误差；不要求所有域都达到同一精度。
7. 干净 archive、fresh caches 重建软件与 model。
8. 独立 session 复跑代表旧点、一个高地址点和完整 `Q41-core`。
9. 保存最终硬件构建链和报告，不额外强制再做一次完整 place-and-route 来获得相同 bitstream。
10. 报告所有未达目标、负结果和未启用机制。

---

## 6. 基线、计时与性能判定

### 6.1 基线

| 名称 | 定义 |
|---|---|
| **B40-OBS** | R40 v3 实际交付的软件、镜像和 90 MHz |
| **B41-S1** | Stage 1 最终软件，在原镜像上 |
| **B41-C** | 新负载合法运行所必需的通用正确性/容量基线 |
| **B41-F** | Stage 2 最终软件、镜像和策略 |

B41-C 不能故意关闭已有有效优化制造慢分母。

B40-OBS 不能运行的新模型，不计算相对它的“无限 speedup”。

### 6.2 对照

- 编译 A/B：同镜像、同频率、同资源。
- 硬件 A/B：同软件策略，尽可能使用相同 artifact。
- 若不能建立同频率硬件对照，报告“硬件与频率联合迁移”，不伪称独立硬件贡献。
- 不要求为了补一项漂亮消融，再增加一个昂贵 bitstream。
- 只对确实相互作用的主要机制做小型交互矩阵。

### 6.3 计时

分开报告：

- cold start；
- resident graph makespan；
- resident call latency；
- request latency / TTFT；
- 完整 decode step。

completion 必须是真完成。

计入实际发生的：

- code reload；
- repacking；
- state 更新；
- token selection；
- 必要同步；
- 正式边界内的数据 copy。

模型 wall time只用于衡量 harness 效率，不能用于排名 FPGA 候选。

### 6.4 采样预算

- 每阶段先用少量锚点验证设备周期稳定性。
- 稳定的长设备任务允许少量完整样本，不机械执行几十次重复。
- 接近结论边界的点按预登记规则补测。
- host latency 使用交错重复，不能借用设备周期的确定性假设。
- 完整模型至少有两个独立 session，其中一个可作为独立复现。
- token 样本不足时只报告描述统计，不强行宣称可靠 p95。
- 达到预算仍无法确认收益，记 `inconclusive`，不继续采样直到显著。

对点 bootstrap 的区间与重复运行噪声区间分别说明，不能混为一谈。

### 6.5 新默认优化的非退化规则

本规则只约束 R41，不改变 R40 的历史判定。

- 原套件数值正确性全部 Required。
- 在冻结的匹配性能集合上，整体允许不超过 5% 的非退化预算。
- 单点 >5% 回退逐一报告。
- 经复测确认的 >10% 新回退，阻止对应性能优化默认启用，须修复、选择合法保守路径或撤回。
- 冻结前列明的必要语义修复，单独记录代价，不冒充性能优化。

不得仅以“已经解释原因”作为保留严重默认回退的理由。

### 6.6 性能目标，不设置逐包倍数硬门槛

建议保留以下 Target：

| 项目 | 目标 |
|---|---:|
| 修复后 GEMM 策略，相对合法保守路径 | 目标子集 ≥1.15× |
| 内存服务改进 | 目标子集 ≥1.2× |
| Qwen 优化，相对合法 B41-C | prefill/decode 分别争取 ≥1.2× |
| 最终选择器 holdout | ≥1.1× |
| SystemC 整体预测资格 | 继续争取 median≤10%、p95≤20% |

只有实测支持的部分可以声称提速。

不设置：

- 每个优化都必须 ≥3% LOO；
- 每类内存方向都必须达到理论峰值固定比例；
- Mode A 必须胜出；
- fused attention 必须胜出；
- XLS 候选必须合入；
- 新矩阵单元必须达到指定倍数。

---

## 7. 消融最低集：保留因果关系，控制实验数量

必须有：

1. 每个主要默认启用机制的匹配 A/B。
2. 最终系统中最主要两至三个机制组的 leave-one-out。
3. 对实际存在交互的机制做小型矩阵。

优先交互：

- fragment/layout × allocation；
- pipeline × matrix；
- cores × memory service。

只有涉及相应主张时，才增加：

- batch × weight reuse；
- locality × cache policy；
- role resource allocation × pipeline。

允许预先登记必要交互组，不强求组内每个辅助改动单独达到收益门槛。

不允许：

- 把所有机制打包成一个开关而不解释；
- 把各项 speedup 相加；
- 为每个小改动重跑全部大套件。

---

## 8. 防止阶段再次失控的执行规则

### 8.1 限制同时进行的主线

Stage 1 主线：

1. fragment/codegen；
2. performance_harness；
3. 有预算选择。

时序和模型可行性并行准备，不阻断上述主线发布。

Stage 2 主线：

1. 时序与集成镜像；
2. 完整模型与容量；
3. 已选中的少量服务/MMA 改进。

### 8.2 候选准入

新候选必须说明：

- 针对哪个已测瓶颈；
- 什么结果会否定它；
- 预计修改和验证范围；
- 是否需要新 bitstream；
- 是否能在当前阶段预算内完成。

没有明确瓶颈证据的“大重构”，不进入当前阶段关键路径。

### 8.3 实验 stop rule

研究达到预登记预算时，必须选择：

- 合入；
- 默认关闭；
- 放弃；
- 保留未完成事项进入下一阶段。

不得通过不断增加候选和测量，把收尾变成无限探索。

### 8.4 冻结后变更

- 可以修复实现和增加诊断。
- 修改产品后重采受影响证据。
- 最终主张必须绑定最终身份。
- 不能删除慢点、缩短 required workload、放宽容差或替换分母。
- Required 未完成时可以关闭阶段为失败，但不能修改合同使其变成通过。

**可交付合同的意义是把必须做的事限制在合理范围，不是保证任何结果都能标 PASS。**

---

## 9. 文件、命令与证据

建议最小新增结构：

```text
doc/r41_tasks_and_acceptance.md
doc/r41_stage1_report.md
doc/r41_report.md

spec/r41/carryover.json
spec/r41/stage1/plan.json
spec/r41/stage1/lock.json
spec/r41/stage2/plan.json
spec/r41/stage2/lock.json
spec/r41/stage2/model_lock.json
spec/r41/stage2/xl_core.json
spec/r41/holdout.json

evidence/r41/stage1/
evidence/r41/stage2/
```

performance_harness 的公共实现继续位于：

```text
tools/performance_harness/
tools/performance_harness/systemc/
```

不要把可复用的模型、runner、比较器继续全部堆入 `tools/r41/`。

拟新增入口：

```sh
make r41-freeze-check STAGE=1
make r41-harness-check
make r41-stage1-acceptance

make r41-freeze-check STAGE=2
make r41-stage2-acceptance

make r41-acceptance
```

验收器至少拒绝：

- 缺失原 large 点；
- 截层或替换模型权重；
- CPU 数值 fallback；
- descriptor 开启但正式路径未执行且无理由；
- 90 MHz 记录冒充 100 MHz；
- model prediction 冒充 board measurement；
- 提交时间冒充完成时间；
- tuning 污染正式样本；
- 在 holdout 上拟合；
- 错误镜像、错误配置或历史 PASS 复制。

---

## 10. 完成定义

### Stage 1 完成

> 在原 90 MHz 三核镜像上，关键编译映射得到修复，原负载保持正确；有预算选择真实接入；performance_harness 能把假设、真实编译产物、模型、RTL 与板测连接成可重复的快速迭代闭环。

### Stage 2 完成

> 三核新镜像真实通过 100 MHz 物理要求；完整官方 Qwen3.5-2B 语言分支经标准 Torch 路径完成冻结核心负载；原完整套件与 XL-core 正确；已启用优化有实际依据；性能模型的适用范围与不足被明确记录。

### R41 完成

> 两阶段 Required 均通过。性能 Target 可以未达到，但未达到的部分必须明确报告，不能包装成已经实现的性能提升。

---

**这版最重要的取舍是：不削弱真实性和正确性，削减的是“所有研究都必须成功”和“所有维度都必须一次测全”。**

从架构上看，最有希望同时帮助多个问题的也不是某个万能开关，而是三个相互配合的改动：

1. **让逻辑 tile 真正映射成驻留的物理 fragment**，同时改善 GEMM、寄存器压力、代码规模和 attention 的基础；
2. **把长组合反馈改成局部、有界、可观测的服务接口**，同时服务时序、home-slice 吞吐和 transfer admission；
3. **让编译器、SystemC、选择器使用一致的资源描述，但保留独立正确性验证**，使每次优化都能更快发现“瓶颈只是被转移了”。

这比同时承诺更宽入口、更大 MMA、三级 Mode A、动态寄存器分配、全局高精度模型和完整长上下文矩阵，更有可能得到一个真正可交付的 R41。