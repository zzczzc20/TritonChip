# R38：面向 ISCA 的硬件 DevOps 与 MLIR–ISA–微架构联合迭代计划

- 文档状态：**研究计划 / 本地审阅稿，尚未实施、尚无 R38 性能结论**。
- 日期：2026-09-22；文献检索截至该日。
- 起点：`r37-triton-coverage`，`1b2a9d6a394a760d6dc015a6baf6a519bdb19650`。
- 本次交付仅为文档；不提交、不推送，不改变 R37 kernel、编译器或 RTL。
- 下文所有倍数、误差门限、样本数和工期均为**建议实验设计或验收目标**，不代表已有测量，也不作为必须得到正结果的要求。

## 阅读导航与最新决策

- **论文主线**：第 0、3、9 节，硬件 DevOps 与三层最小可运行闭环。
- **架构总体 metric**：重点审阅 **第 4.10 节 AQ**。它使用公共抽象资源预算与候选无关的理想下界，汇总 workload、规模与环境；第 4.1–4.9 节提供公式基础和防失真规则。
- **落地实现**：第 5–8、13 节，45 算子、cost model、profiler、跨层案例与接口。
- **论文实验**：第 9–12、15 节，方法对照、Agent、NVIDIA、PyTorch/LLM。
- **执行计划**：第 14、16、17 节，工作包、时间盒、风险和验收。

最新要求对应的指标层级为：**AQ 衡量标准化后的架构族质量；S 衡量具体部署性能；E 衡量具体配置距自身上界的距离；设计时间/token 衡量迭代效率。** 这四者不能相互替代，也不任意相乘成总分。架构主分为 AQ，其他指标用于说明它为什么成立、是否能落地，以及怎样获得。

## 0. 核心命题与论文定位

**核心思路：把 DevOps 的持续集成、可观测性和短反馈周期应用到硬件设计，使 MLIR、ISA 和 microarchitecture 始终形成一个最小可运行闭环；在相同设计预算下，通过三者的快速联合迭代寻找更好的架构。**

R38 不应组织成五个相互独立的项目。五项工作对应一条证据链：

```text
真实、尽量不改源码的 Triton 工作负载
          ↓
MLIR / lowering ↔ ISA 语义与资源契约 ↔ 可执行微架构
          ↓                              ↑
公共 AOT + runtime → RTL / FPGA → profiler
          ↓                              ↑
ArchBench 度量 → cost model → 下一轮跨层变更
          ↓
固定预算下的设计质量、迭代效率、可靠性
          ↓
NVIDIA 外部参照 + PyTorch / FPGA LLM 系统验证
```

拟研究命题：

> 将编译器、指令集和微架构的变更作为一个可验证、可测量、可回退的共同迭代单元，能够降低跨层反馈延迟，并在相同资源与研发预算下改善最终设计的性能—资源折中；可解释的 cost model 和 profiler 进一步提高人或 Agent 的搜索效率。

这里有三个必须分别验证的因果链：

1. **闭环更快**：从提出变更到得到可信正确性、性能和资源反馈的时间降低。
2. **闭环更有效**：在相同总预算内，得到更多有效设计，或探索到分层固定设计不能到达的组合。
3. **最终设计更好**：冻结的未知工作负载上，资源约束内的实际性能提高，而非仅预测分数提高。

“能自动生成代码”“使用了 MLIR”“跑通 45 个算子”“Agent 写过 RTL”都不足以单独证明这些命题。最终 FPGA LLM 截图可以展示完整性，但不能代替上述证据。

### 0.1 建议的贡献组织

| 贡献 | 要证明什么 | 最关键的证据 |
|---|---|---|
| C1：跨层可执行迭代方法 | 编译器、ISA、硬件共同演进时仍能持续运行、诊断和比较 | 契约、版本化制品、失败回退、真实跨层变更轨迹 |
| C2：ArchBench 评估协议 | 可跨配置分析架构效率，同时不混淆规模和绝对性能 | 45 算子规模空间、资源上界、权重冻结、反作弊实验 |
| C3：反馈驱动的联合设计 | 同等预算下比单层/分阶段优化更有效 | 对照实验、至少三个跨层案例、未见形状与配置的结果 |
| C4：Agent 作为闭环执行者 | cost model + profiler 改善优化质量与成本 | 2×2 消融、token/工具/墙钟成本、重复实验 |
| 系统验证 | 所得设计能服务真实应用 | FPGA 实测、PyTorch 图执行、LLM prefill/decode |

论文应优先讲 C1 + C3；ArchBench 是支撑研究可信度的共同标尺，并争取独立的基准贡献。Agent 是研究对象和放大器，不应让整个结论依赖一个不断更新的闭源模型。

### 0.2 新颖性的边界

敏捷硬件、软硬协同、基于反馈的搜索并不是新概念。Chipyard、Gemmini、HASCO、CIRCT、ACT 已经覆盖其中多项能力。尤其 **HASCO 是必须正面比较的 ISCA 2021 相关工作**，不能只引用通用 Roofline 或 LLM kernel 论文。

我们的潜在区别应是：**面向广泛可移植 tensor 程序，允许 MLIR lowering、ISA 契约和微架构结构共同改变，以持续可执行的实证反馈驱动迭代，并把设计质量与迭代成本一起作为研究结果。** 这只是待验证定位，不能提前声称“首次”。详见第 2 节。

## 1. 项目事实：R37 已有基础与 R38 必须补齐的缺口

### 1.1 已有可复用基础

| 项目 | 当前证据与路径 | 能支持的结论 |
|---|---|---|
| 原始算子覆盖 | [r37_validation.md](r37_validation.md)、[r37_tilebench.json](../spec/r37_tilebench.json) | 45/45 原始算子；每个 O0/O1/O2 为 149/149 数值 cases/stages，总 447/447 |
| 公共入口 | [aot.py](../python/tnca/aot.py)、[runtime.py](../python/tnca/runtime.py) | 测试经过 `compile_aot`、`.tnca`、`Session`、RTL，而非只调用内部 lowering |
| 上游源码约束 | TileBench commit `6e2cf155dfe01f4660540d187c81b09eb55f394f` | 原始 JIT 函数体保持不变；小尺寸 launch 由适配层提供 |
| MLIR 基础 | [TNCADialect.td](../compiler/include/tnca/Dialect/TNCA/IR/TNCADialect.td)、[VerifyAsyncOwnership.cpp](../compiler/lib/Dialect/TNCA/Transforms/VerifyAsyncOwnership.cpp) | 有目标方言和所有权 verifier 基础；不能由此推断全部公共 AOT 路径都已经过统一 MLIR pass pipeline |
| 编译实现 | [tnca_cc](../compiler/python/tnca_cc)、[compiler.py](../compiler/backend/compiler.py) | Python lowering 与 MLIR 相关实现并存；需绘制实际调用图并记录 pass provenance |
| 周期测量先例 | [perf.py](../tools/r35/perf.py) | 已有 device cycles 与正确性联合检查，明确不以 host wall time 冒充周期 |
| counter 契约 | [r3_counters.json](../spec/r3_counters.json) | 已区分互斥 stall、重叠 stall、operand collector 和 testbench memory 事件 |
| RTL 结构 | [tnca_top.v](../rtl/tnca_top.v)、[core](../rtl/core)、[nuca](../rtl/nuca)、[noc](../rtl/noc) | 可以定位 RF、NUCA、mesh 等实现层；功能存在不等于全规模性能已验证 |
| 教程 | [08_triton_gemm.md](../tutorial/08_triton_gemm.md)、[09_tensor_descriptors.md](../tutorial/09_tensor_descriptors.md) | 普通用户可从 AOT / Session 入口运行 GEMM 与 descriptor case |

**447 是 149 个 case/stage × 3 个优化级别，不是 447 个独立真实模型形状，更不是几千个性能测量点。** 多阶段算子的 stage 数不能直接当权重，尤其 radix sort 的许多阶段不能使该算子在总分中自动获得更大占比。

### 1.2 不能直接沿用为论文结论的部分

1. **R37 是小规模正确性覆盖。** 任意大 tile、长 reduction、大 K INT8、FP8 格式边界、极端输入和全部 launch 参数仍需扩展验证。
2. **公共 runtime 当前逐 program 发射。** `Session._run_one` 对 grid 的 `linear` 循环调用 `dev.launch(..., role_mask=1)`，累加返回周期。它可以验证 grid 语义，但这些相加周期不是已经具备设备端并行 grid scheduler 的全芯片吞吐。
3. **多阶段正确性 harness 不是设备驻留 pipeline benchmark。** 中间结果的 host 搬运和 Python 调度必须在新性能 harness 中单独处理。
4. **默认模拟设备容量很小。** `tb/device/tnca_device.v` 的 65,536 个 64-bit word 合计 512 KiB；runtime 默认 heap 为 `0x20000..0x80000`。这与可运行真实 LLM 的外存系统差距很大。
5. **功能 fallback 不能写成原生硬件能力。** [r37_isa_gaps.md](r37_isa_gaps.md) 记录了格式、descriptor、dot 等落差。FP32 dot 的向量展开和部分 INT8 的浮点路径必须单独记录；小 K 正确不意味着任意累加范围都精确。
6. **现有综合不是 PPA。** [synthesis.md](synthesis.md) 明确没有 PDK，gate/depth/flop 是结构代理；不能当作 mm²、GHz 或 watt。
7. **R36 未等于 FPGA 上板完成。** [r36_status.md](r36_status.md) 仍记录 `IMPLEMENTED / NOT ACCEPTED` 与锁定平台 gate 问题，而且该阶段明确不负责具体板卡、P&R 和 bitstream。
8. **架构概览中的 16 tile、1 GHz、8 MiB 等是设计建议。** [设计草案](design/tnca_v1_architecture_draft.md)（原 `architecture_overview.md`）不能作为当前已实现配置或实测频率来源。

R38 的第一项产物应是“证据注册表”，将每项能力标为 `documented / implemented / functionally-validated / cycle-validated / synthesized / board-measured`，每级都链接对应制品。不允许一个笼统的 `supported=true` 跨越这些层次。

## 2. 文献脉络与我们应补出的研究空间

下表是研究定位，不是对每项工作的完整复现结论。预印本与动态文档在正式实验时需固定版本；不要把预印本自动写成已被某会议录用。

| 方向、文献 | 可借鉴的已有内容 | 对 R38 的约束或区别 |
|---|---|---|
| [Roofline](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2008/EECS-2008-134.pdf)，Williams 等 | 计算与数据搬运上界、operational intensity | 距 roofline 近不等于机器快；不能只用 DRAM 和 FLOPs 描述全部 45 算子 |
| [ERT](https://amcr.lbl.gov/departments/computer-science-department/ppan/roofline-performance-model/empirical-roofline-tool-ert/)、[Hierarchical Roofline](https://arxiv.org/abs/2009.05257) | 实测平台特性、层次化存储瓶颈 | 区分理论硬上界与经验可持续吞吐；后者不是绝对上界 |
| [Performance Portability metric](https://arxiv.org/abs/1611.07409) | 跨平台效率聚合与支持覆盖 | 调和平均已有前例；ArchBench 的贡献不能只是换个平均数名字 |
| [Timeloop](https://timeloop.csail.mit.edu/)、[Accelergy](https://accelergy.mit.edu/tutorial.html)、[MAESTRO](https://maestro.ece.gatech.edu/) | 映射、数据流、资源及能耗建模 | 借鉴分析模型，补 TNCA 的动态缓存、反压、角色依赖；不应从零重复所有张量数据流分析 |
| [MLIR](https://arxiv.org/abs/2002.11054)、[CIRCT](https://circt.llvm.org/) | 多层表示、可组合变换与硬件工具基础设施 | 使用这些工具本身不构成创新；不要求为写论文而把现有 RTL 全部重写成 CIRCT |
| [Chipyard](https://people.eecs.berkeley.edu/~krste/papers/Chipyard-IEEEMicro2020.pdf) | 集成生成、仿真和实现的敏捷 SoC 工作流 | 必须说明我们增加的是哪种跨层演进机制及定量收益，而非仅 CI 自动化 |
| [Gemmini](https://dspace.mit.edu/entities/publication/f992c764-0107-4978-8ba8-6b4120c954cc) | 面向 DNN 架构评估的全栈集成 | 需要超越固定加速器模板参数扫描，证明真实 ISA/lowering/结构共变 |
| [HASCO](https://arxiv.org/abs/2105.01585)，ISCA 2021 | 张量计算的敏捷软硬协同 | 直接近邻；比较设计空间、可执行反馈、动态微架构建模及研发成本口径 |
| [ACT](https://arxiv.org/abs/2510.09932) | 从 tensor accelerator ISA 描述生成编译后端 | schema 驱动后端已有近邻；我们的重点是契约演进及实测闭环，不宣称首次 ISA 描述驱动编译 |
| [多层 RISC-V 后端](https://arxiv.org/abs/2502.04063)，CGO 2025 | 在不同抽象层利用专用 ISA 能力 | 需要展示三层联合改变的额外收益，而非普通定制后端 |
| [Ansor](https://www.usenix.org/conference/osdi20/presentation/zheng) | 搜索与学习 cost model 配合 | 应作为固定硬件调度搜索的方法参照；Agent 必须与非 LLM 搜索比较 |
| [FireSim](https://fires.im/publications/) | FPGA 加速的周期精确系统仿真 | 清楚区分目标周期与仿真主机/FPGA 执行速度；无需强制迁移整套 FireSim |
| [DCO](https://arxiv.org/abs/2512.07312)、[LLaMCAT](https://arxiv.org/abs/2512.00083) | 缓存预测管理、仲裁与节流相关机制 | 作为项目已有思想来源和消融对象；不能把继承的机制重新算作 R38 创新 |
| [FlashAttention](https://arxiv.org/abs/2205.14135) | IO-aware 精确 attention | 语义工作量与算法中间量不同；不能把完整 attention matrix 写回当作必需流量 |
| [ArchEval](https://arxiv.org/abs/2607.03601)，2026 预印本 | 在多种模拟器与反馈条件下评估 Agent 架构设计能力 | 直接近邻；重点比较我们真实 MLIR–ISA–RTL 契约演进、model/profiler 因子实验及 AQ 协议，而非泛称首次评估 Agent 架构师 |
| [KernelBench](https://arxiv.org/abs/2502.10517) | 正确且更快的 kernel 生成评估 | 保留正确性门槛，扩展到 ISA/RTL 和设计预算；不能只比较 PyTorch eager |
| [CudaForge](https://arxiv.org/abs/2511.01884)、[CUDA Agent](https://arxiv.org/abs/2602.24286) | 硬件反馈、验证与 Agent kernel 优化 | “给 Agent profiler”已有相关工作；需要本项目的完整 factorial 与跨层结论 |
| [Compiler–LLM cooperation](https://arxiv.org/abs/2604.04238) | 多抽象层的编译器与 Agent 协作 | “跨层 Agent”也不是空白；硬件契约变化、RTL 反馈及成本归因是关键区别 |
| [FlightLLM](https://arxiv.org/abs/2401.03868) | FPGA 上完整 LLM 映射与执行 | 跑 LLM 本身不是新贡献；我们的重点是通用编译路径与迭代所得硬件的可用性 |
| [PyTorch accelerator integration](https://docs.pytorch.org/docs/main/accelerator/index.html) | 新设备的 dispatcher/runtime 集成路径 | 不从头设计 PyTorch；固定版本，分图后端与真实 device backend 两步做 |
| [MLPerf Inference rules](https://github.com/mlcommons/inference_policies/blob/master/inference_rules.adoc) | workload、质量和运行协议的明确约束 | 借鉴质量/场景/披露规则，不把 ArchBench 自称 MLPerf 合规结果 |

### 2.1 正式论文前必须完成的近邻复核

建立 `related_work_matrix`，逐篇读 HASCO、Gemmini、Chipyard、ACT、ArchEval、DCO/LLaMCAT 的正文、artifact 和限制：

- 是否允许 ISA 语义改变，还是只改已有参数？
- 编译器是否随硬件生成，还是运行时选择 schedule？
- 反馈来自模型、trace、RTL、FPGA 还是 ASIC？
- 是否报告正确性回归与失败设计？
- 是否用固定设计预算比较最终结果？
- 是否量化反馈周期、复现实验开销及设计者成本？
- 是否覆盖动态控制、非 GEMM、缓存反压和多阶段算子？

本稿检索足以确定这些是近邻，**不足以声称它们缺少某项具体能力**。正式新颖性措辞必须在完成该表后确定。

## 3. 把“硬件 DevOps”变成可执行研究方法

### 3.1 最小可运行闭环的定义

闭环状态定义为：

\[
V=(C,I,H,R,W,M,P,E)
\]

其中 C 是 compiler/pass 版本，I 是 ISA/ABI 契约，H 是 RTL 与硬件配置，R 是 runtime，W 是 workload/输入，M 是 cost model，P 是 profiler schema，E 是工具链与测量环境。

一次实验不能只记录 Git SHA：未提交 patch、生成 RTL、综合 seed、编译 options、缓存制品、设备固件都可能影响结果。因此每次 candidate 必须生成内容哈希 manifest。

**最小闭环通过条件：**

1. 从未修改的代表性 Triton 源码进入公共 AOT。
2. 记录真实 TTIR/目标 IR/pass 路径，产生版本匹配的二进制。
3. 同一 runtime 提交到指定 RTL backend，得到正确结果。
4. 获得有来源的 device 时间与至少一个可解释瓶颈事件。
5. 运行资源代理/综合检查，判定配置是否合法。
6. 产生相对于 parent 的完整差异；失败自动保留证据并回到上一个有效候选。

“最小”是先选 6–8 个代表点缩短反馈，不是允许跳过 runtime、硬件语义或数值核验。

### 3.2 跨层变更单元

每个变更以一个 `design_change` 表达：

```yaml
# 设计草案，尚无对应命令或 schema 实现
id: rf-window-layout-001
parent: <candidate-content-hash>
hypothesis: "bank-aware lowering 配合窗口布局可减少 operand collector 冲突"
touched_layers: [compiler, isa, microarch]
semantic_change: false
feature: rf_window_layout_v2
compiler_passes: [<real-pass-name>]
isa_contract: <schema-hash>
rtl_config: <config-hash>
required_tests: [encoding, ownership, randomized-backpressure, aot-smoke]
performance_cases: <development-suite-hash>
resource_budget: <budget-hash>
acceptance: <frozen-policy-hash>
```

实际名称由实现决定。候选变更包括新原语、资源组织和 scheduling；不得仅改报告字符串就算三层变更。

### 3.3 契约的最小内容

ISA/资源契约至少覆盖：encoding、保留位、dtype/舍入/溢出、mask/尾块、operand shape、读写 footprint、副作用、完成事件、内存顺序、异常、资源占用与 feature version。

- 语义项必须由 interpreter/oracle、RTL 和 compiler verifier 共同遵守。
- latency/throughput 是具体硬件配置属性，不能错误地固化成 ISA 正确性条件。
- 异步操作明确“已接收”“数据可读”“计算完成”“寄存器可复用”的区别。
- performance hint 不得改变结果；忽略 hint 仍须正确。
- AOT artifact 携带 required features，runtime 在 launch 前拒绝不兼容硬件，不能静默换语义。
- schema 可以生成 decoder 表、disassembler、合法性测试、编译器 capability 表；但核心语义应有独立 oracle，避免共同生成的错误互相验证。

### 3.4 持续验证与候选晋级

| 层级 | 建议目标反馈时间 | 内容 | 是否可作为最终性能证据 |
|---|---:|---|---|
| L0 | 秒至分钟 | schema/encoding/IR verifier、编译 smoke | 否 |
| L1 | 5–15 分钟 | 代表点 AOT + RTL、针对性随机测试 | 仅这些点的周期结果 |
| L2 | 小时级 | 全 R37 O0/O1/O2、扩展 correctness、周期回归 | 已测点可用 |
| L3 | 依实际工具 | 综合、资源、时序、代表性全系统 RTL | 清楚标注证据级别 |
| L4 | 板卡流程决定 | P&R、bitstream、FPGA 全路径测量 | 板级实测可用 |
| Release | 固定窗口 | 冻结 held-out 集、完整资源与协议审计 | 论文主表来源 |

这些时间是 SLO 提案；先测基线再修订，不能因超时而少跑必需测试并宣称通过。未通过 L2 的候选不得成为下一轮可信性能 baseline。

**连续运行不等于每个探索分支一直正确。** 允许失败候选，但始终保留一套可重建的 last-known-good；失败率与恢复时间是研究数据。

### 3.5 DevOps 指标

- `feedback_latency`：提交候选 manifest 到第一份有正确性门槛的反馈。
- `validated_lead_time`：提出假设到通过目标 gate，包含修复与工具失败。
- `valid_candidates / wall-hour`，以及每 1M token 的有效候选数。
- `change_failure_rate`：未通过 correctness/compatibility/resource gate 的候选比例。
- `recovery_time`：失败到恢复可运行候选的时间。
- `reproduction_success`：从干净环境重建相同制品/结果的比例。
- `best_feasible_quality(budget)`：随预算变化的最佳有效设计质量曲线。

这些指标借鉴 DevOps 的工程思路；它们在这里的事件边界由 manifest 定义，不能直接用 Git commit 数或生成代码行数代替生产率。

## 4. ArchBench：架构质量的度量协议

### 4.1 不强求一个脱离约束的“架构总分”

一个架构可以通过加计算单元、加 SRAM、加带宽提高性能，也可以通过降低峰值让利用率更好看。不存在一个仅除以峰值就能自动消除工艺、面积、频率、带宽、规模差异的万能分数。

ArchBench 的主结果应为以下证据组合；其中架构总分 AQ 的具体定义见 **第 4.10 节**，它是跨部署、跨规模评估的主指标：

\[
\boxed{(\text{正确性/覆盖},\;\text{固定预算性能},\;\text{上界效率},\;\text{资源与能耗},\;\text{迭代成本})}
\]

在同一个硬件配置内，上界效率非常适合定位 compiler/ISA 的损失；跨架构时必须同时看绝对完成时间和资源约束。论文可使用一个主排序指标，但其约束和伴随指标必须公开。

### 4.2 Workload 的定义必须先于实现

一个点 i 包含：算子语义、shape、dtype、layout/stride、输入分布、精度契约、运行场景和调用次数。一个算子的多个 stage 是同一语义实例内部的实现，不自动成为独立评分对象。

区分两条轨道：

- **Portable-source track（主轨）**：冻结 TileBench JIT 源码 hash，后端可以改变 lowering、合法 tile mapping 和硬件实现。公开哪些 launch 参数允许适配；不能针对某平台替换函数体。
- **Semantic-best track（外部性能参照）**：保持数学语义和精度，允许成熟库或专门 kernel；与 cuBLASLt/cuDNN/CUTLASS 等可比。不同算法的结果独立报告。

不能让 TNCA 使用宽松近似、融合实现，而 NVIDIA 只能用逐算子 eager reference；也不能把原始 GPU launch wrapper 无法在新设备运行误记为 kernel 语义不支持。

### 4.3 每点的理论下界

传统形式为：

\[
P_{roof}=\min(P_{compute}, B_{DRAM}\cdot I),\qquad
T_{LB}=\max(F/P_{compute},Q/B_{DRAM}).
\]

其来源是 [Roofline](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2008/EECS-2008-134.pdf)。在 TNCA 上建议扩展为**多资源必要条件**：

\[
L_{i,c}=\max\left(
\max_r \frac{W^{min}_{i,r}}{C^{max}_{c,r}},
\max_\ell\frac{Q^{min}_{i,\ell}(S_c)}{B^{max}_{c,\ell}},
D^{min}_{i,c}
\right).
\]

- c：具体可实现配置；S：相关存储容量。
- r：MMA、SIMD、SFU、整数/compare、reduction、atomic、issue 等服务资源。
- ℓ：DRAM、NUCA bank、必要 NoC cut 等可证明需通过的边界。
- W、Q：在规定算法/语义范围内有依据的最低服务需求。
- D：已证明的依赖链/启动延迟必要下界；不知道就留空，不能编造。

**不是每个资源都天然存在语义层面的非零下界。** 例如 RF 读次数取决于 bypass、operand holding 和指令组织；没有证明时不把“当前实现的 RF 流量”放进主下界，而只放入 mapping 诊断模型。

共享 issue slot 或复用执行单元使多个峰值不能同时达到。进一步可建立小型 LP：在给定 instruction/service mix 下寻找满足各端口、链路和执行资源约束的最短时间。LP 的约束必须是必要条件；不确定的经验惩罚属于 cost model，不能伪装成严格下界。

该下界一般不紧，也未必可同时达到。论文用“必要资源下界/效率”，不要称“已经知道的最优实现时间”。不同算子的下界紧度不同，按类别报告敏感性。

### 4.4 Work 与 bytes 如何记

| 算子类型 | 可解释服务量 | 要避免的错误 |
|---|---|---|
| GEMM/Conv | 有效 MAC，必要输入输出，声明 dense/direct 算法范围 | 将 padding、重复计算、spill 算入 useful FLOPs，奖励低效实现 |
| 激活/SFU | 元素、exp/log/reciprocal 等服务次数 | 把一个 exp 任意折成若干 FLOPs 后跨机器比较 |
| Reduction/Norm | 元素、reduce/compare 服务、输入输出 | 假定所有 reduce 都可按 GEMM 峰值执行 |
| Copy/Transpose | 有效 payload bytes、必要读写 | 用 FLOPs 定义没有浮点计算的算子效率 |
| Gather/Scatter/Histogram | 索引和有效事务，冲突/分布参数 | 只测均匀随机、不披露热点和 atomic 串行化 |
| Sort/Top-k | keys/items/s，已声明算法范围的比较/传输下界 | 用当前 bitonic/radix 的操作数当作所有排序算法的必需工作 |
| Attention | 有效 attention 计算、softmax 服务、IO 下界 | 把某个 unfused 实现的 N² 临时张量读写当作语义必需流量 |

对 GEMM 的 `2MNK` 明确采用 conventional dense MAC 计数约定，并在主轨限制算法族；不声称它是允许任意快速矩阵算法时的数学工作下界。结构稀疏与 dense 分轨，未执行的零不能同时算作实际服务和有效节省。

在每级边界同时存两个字段：`required_bytes_lower_bound` 与 `measured_bytes`。它们用于不同目的：前者比较效率；后者解释 cache reuse、冗余和 spill。若用 measured bytes 重新抬高下界，会把访存浪费洗成“已经接近 roofline”。

### 4.5 理论峰值与经验 ceiling 必须分开

建立两套容量：

1. `spec_peak`：由实现与配置确认的最大服务率；用于有效性可审计的下界。
2. `empirical_sustained`：通过独立 microbenchmark 标定；用于诊断现实差距。

经验 bandwidth 会受 access pattern、并发度和计时噪声影响，某个 workload 超过它并不违反物理定律。报告可以借鉴 [ERT](https://amcr.lbl.gov/departments/computer-science-department/ppan/roofline-performance-model/empirical-roofline-tool-ert/)，但不可把经验值叫“绝对理论上限”。GPU 则利用 [Nsight Compute](https://docs.nvidia.com/nsight-compute/ProfilingGuide/) 的层次化统计，固定工具版本与指标定义。

主效率：

\[
\eta_{i,c}=L_{i,c}/T_{i,c},\qquad \delta_{i,c}=1-\eta_{i,c}.
\]

若严格下界算出 η > 1（超过测量不确定性），该记录进入审计失败：检查计时窗口、单位、MAC/FLOP、并发、压缩、缓存初始状态和精度。**不能简单 clip 到 1 后继续评分。** 经验 ceiling 的比值允许超过 1，但标为经验归一化值，不进入严格效率主表。

### 4.6 按运行时间加权：冻结权重的具体方案

先冻结参考机器与实现，以及应用调用分布 a_i。参考延迟为 T_i^ref：

\[
w_i=\frac{a_iT_i^{ref}}{\sum_j a_jT_j^{ref}},\qquad \sum_iw_i=1.
\]

没有真实应用 trace 的通用 suite，先按 45 算子均衡分配 operator mass，再在算子内部按预注册 shape 分布分配权重，称为 **balanced track**；它不能自称真实应用运行时间分布。来自具体模型 trace 的权重另称 **application track**，保留调用次数和融合边界来源。

建议主效率分数使用固定参考时间权重的调和平均：

\[
E_c=\left(\sum_i\frac{w_i}{\eta_{i,c}}\right)^{-1}.
\]

它对低效率点敏感；原始逐点 η、分位数与按类分数必须同报。辅助给出更直观的算术平均 `A_c = Σ w_i η_i,c`，避免读者只能看到一个聚合数字。调和聚合在 [performance portability](https://arxiv.org/abs/1611.07409) 已有方法先例，本项目不将该公式本身宣称为原创。

**主性能分数**独立定义为同一工作组合的总时间加速比：

\[
S_c=\frac{\sum_i a_iT_i^{ref}}{\sum_i a_iT_{i,c}}
=\left(\sum_i w_i\frac{T_{i,c}}{T_i^{ref}}\right)^{-1}.
\]

它具有明确“相同调用组合耗时”的解释。按点 speedup 的算术平均一般没有这个解释，不用作主性能分数。

另一个合法但含义不同的量是：

\[
U_c=\frac{\sum_i a_iL_{i,c}}{\sum_i a_iT_{i,c}}.
\]

它等于按候选自身耗时加权的 η，描述这个串行工作组合的整体必要下界效率；可作为诊断。但分子随架构配置改变，且权重随候选改变，**不能代替跨架构固定标准的 S 与 E**。如果系统实际并发，端到端时间不能再简单求和，应使用 workload DAG 的实测 makespan 与相应下界。

### 4.7 一个防止误读的数值例子（纯示意）

两个点，冻结权重各 0.5。设计 A 的下界均为 1 μs、实测均为 2 μs；设计 B 的下界均为 10 μs、实测均为 11 μs。

- A 的效率 0.5，B 的效率约 0.909。
- 但 A 完成相同工作比 B 快 5.5 倍。
- 因而 B 更接近其自身上限，并不代表 B 是更好的性能设计。

再考虑 η = (0.9, 0.1)：算术平均 0.5，调和平均 0.18。前者表达平均比例，后者强调短板。论文必须公开选择原因，而不是在看完结果后挑使自己最好看的均值。

### 4.8 缺失、错误与超时

- 完整套件的数值失败或 unsupported 不能剔除后平均；正式 release gate 不通过。
- 探索报告显示 count coverage 和 `Σ_supported w_i` 两种覆盖；可附 supported-only 分数，但明确为诊断。
- 若必须统一排名，将缺失效率设为 0，因此全套 E 为 0；也可用事先定义的超时惩罚形成研究辅助分数，但不能与完整通过结果混为主分。
- timeout 的耗时是右删失数据，不等于真实耗时；保留 timeout 阈值和状态。
- compilation failure、runtime fault、数值错误、超过容量、未测量分别记录，不合并成“慢”。

### 4.9 配置大小与硬件预算

至少设置小、中、大三个**先冻结的可实现预算档位**。具体数值在板卡和综合 pilot 后确定，不直接采用架构概览中的建议参数。

预算向量包括 LUT、FF、DSP、BRAM/URAM、可用 DDR/HBM 容量与带宽、接口、时序和可测功率。FPGA 资源是多维约束；不能任意把一块 DSP 折成固定 LUT 得出普适面积。

每档报告：实际配置、最高满足时序的工作频率、资源、性能 S、效率 E、每类分布。另做固定频率对照区分周期改善与 Fmax 变化。若加端口减少周期却降低频率，要用 `cycles / achieved_frequency` 判断最终收益。

区分：

- 固定 workload 的 strong scaling；
- workload 随配置增大的 weak scaling（仅作扩展实验）；
- 一套静态配置跑完整 suite；
- 多配置 portfolio（单独计切换、重配置成本）；
- 每点最佳配置的 oracle envelope（只能当上限，不是可部署主结果）。

不允许主分逐点免费换 bitstream。容量不够的小配置在相应场景记录 unsupported，不悄悄缩小问题。


### 4.10 核心补充：独立于具体部署的架构总指标 AQ

根据进一步明确的目标，**R38 应将“架构本身”的主指标提升为标准化反事实评估下的 Architecture Quality，简称 AQ**。前文 E 是单配置自身利用率，S 是具体部署性能；二者保留为诊断与验证，但都不等于 AQ。

目标不是给 ASIC 和 FPGA 实测数字寻找一个神奇换算系数，而是：

> 把不同架构放到同一套公开的抽象资源、时序与环境协议中，比较它们在多个相同规模预算下，把资源转化为真实有用工作完成能力的程度；再按冻结的 workload、规模和环境分布形成一个总分。

AQ 不读取 FPGA 型号、ASIC node、标称 GHz 或板卡价格作为直接加分项。同一架构部署到不同载体，若反推得到的抽象行为一致，应得到同一 AQ；部署质量用另一个结果解释。

#### 4.10.1 为什么“完全无条件的不变排名”不成立

考虑 A 用更多本地存储减少外存读取，B 用更大算阵换更少本地存储。在低带宽环境 A 可能快，在高带宽环境 B 可能快。再考虑一种架构在 1 core 上控制简单，另一种在 64 core 上通信扩展性更好，规模变化也会反转排名。

因此不能同时要求：保留这些真实架构差异，又让任何环境和规模变化都不影响排名。合理目标应区分：

1. **载体无关**：不因同一架构映射到 FPGA 或 ASIC 的工艺优势而改变架构评分。
2. **规模公平**：不因把同一设计简单做大就自动拿更高分；在相同预算点比较，再汇总尺度范围。
3. **覆盖环境而非依赖单一环境**：在公共环境域上汇总，并披露排名反转。
4. **协议有条件、结果可复现**：声明 workload 与抽象模型适用域，不能称脱离一切假设的普适真值。

这个边界不是放弃总体 metric，而是使它能够经受审稿人的反例检验。

#### 4.10.2 对象：架构族，而非某块板卡

定义架构族 A，包括：ISA 语义、执行组织、数据流、存储/互连拓扑、资源仲裁以及一条公开的参数化 scaling rule。设 x_A(b) 是预算档 b 下的具体配置。

- x_A(b) 用 development/validation 选择，在 final suite 前冻结。
- 每个 b 只能有一套硬件配置跑整套 workload；不能对每个算子免费重新造芯片。
- 主轨 x_A(b) 对环境 e 也保持不变，环境敏感性是被测属性。
- 允许按环境选择不同配置的 `adaptive-family` 轨单独报告，不混入主 AQ。
- 编译器可按已知 workload 产生合法 schedule，但调优预算和算法必须统一记录。

实际只能评估 `AQ(A, compiler)`，无法从有限实验证明脱离编译器的最优架构潜力。额外给 `AQ_best-found`（统一 mapping 搜索预算）和小案例 exhaustive/oracle 对照，估计编译器映射损失，不将 best-found 写成真正最优。

#### 4.10.3 统一抽象资源账本

采用“组织层 + 实现层”分离。组织层描述：

| 资源 | 抽象表示 | 必须计入的代价 |
|---|---|---|
| 运算 | 各 dtype 的算术/逻辑原语与服务率 | SIMD/MMA/SFU 的内部实际算子，不能所有 instruction 同价 |
| 存储 | bits、bank、宽度、容量、读写端口 | tag、ECC（若有）、复制、多端口结构与 accumulator |
| 互连 | links、宽度、hop、topology、队列 | router/crossbar、buffer、仲裁和距离相关延迟假设 |
| 控制 | contexts、issue、scoreboard、decoder、scheduler | 不是零成本；少控制与复杂调度的差异应保留 |
| 并发 | MSHR、targets、outstanding、tokens | entry bits、比较/选择逻辑及返回路径 |
| 时间 | 统一原语延迟与流水规则 | 不能通过声明更长组合逻辑仍“一周期”来作弊 |

成本模型使用公开 canonical primitive library：固定的算术逻辑成本、存储与端口成本函数、连线/仲裁规则。它是**评测约定**，不是天然物理常数，不宣称等于任意 ASIC 面积或 FPGA LUT。

资源约束可以写成：

\[
K_\theta(x_A(b))\le B_b,
\]

其中 K 是多维资源成本/约束，θ 是该协议的成本系数版本。初版至少保留 logic、storage、communication 三维 hard caps；另有总体 canonical cost cap，防止用未计价的端口/控制换算力。各维是否可互换、按什么成本互换必须预注册。

统一 raw MAC 数会掩盖 SIMD/MMA 组织选择的面积代价；统一 FPGA DSP 数又会偏向该器件映射。故两者都不能单独作为“等规模”。保留 primitive composition 才能研究资源分配本身。

#### 4.10.4 环境与规模域

在抽象世界中定义环境 e，例如外存带宽/参考运算服务率、外存延迟/参考原语延迟、请求并发限制、外部内存容量，以及预注册的存储/互连成本假设。

这些比值使用**公共参考原语和预算**作分母，不能使用候选自己的峰值归一化，否则候选改峰值就等于换考试题。

初版建议以 1×/2×/4×/8× 四个预算点进行 pilot，再根据合法配置和测量代价冻结正式区间。它们不是 1/2/4/8 cores，而是同一资源账本的预算倍数。每档都允许架构选择不同的合法 core 数与存储分配。

环境可先做低/中/高 bandwidth 与低/高 latency 的小网格；规模与环境域的权重 p_b、q_e 在测试前冻结，建议规模按 log-budget 区间等权。不能多复制一个有利规模点就提高总分。

**主轨采用固定语义 workload。** 另给按公共预算规则缩放问题的 scaled-workload 轨，以研究大架构是否只因问题太小而闲置；缩放规则对所有候选相同。不能按每个候选自己的 SRAM 容量分别生成有利问题。

#### 4.10.5 公共理想下界，而非候选自身 roofline

对每个 i、b、e 定义公共下界：

\[
L^*_{i,b,e}=\operatorname{LB}\bigl(\text{semantic workload }i,
\text{canonical budget }B_b,\text{environment }e\bigr).
\]

L* **不读取候选架构 A**。它来自统一预算下的乐观资源分配与多资源服务约束，例如：

- 允许理想复用、理想仲裁，保留必需计算和不可避免的数据传输；
- 在 canonical 成本约束下分配算术、存储和搬运资源；
- 用 relaxed LP/解析必要条件求最短时间的下界；
- 不计无法证明的 bank-conflict/动态排队惩罚，保留其乐观性质；
- 针对 sort、稀疏与近似算法明确算法族，不能套不成立的工作下界。

这是所有允许组织的乐观 relaxation，不要求一台实际机器同时达到所有 workload 的 L*。因此 AQ 衡量“接近公共资源潜力”的程度，而不是承诺能达到 100 分。

若某类工作负载无法得到合法且有信息量的 L*，先标为该 bound 的研究缺口；可以在参考比值轨评估性能，但不能把未经证明的 lower bound 填进正式 AQ。

#### 4.10.6 推荐总体公式

设 τ_A,i,b,e 为候选在 canonical substrate 下的完整 ROI 时间，w_i 为从公共参考场景一次性冻结的 workload 时间权重，p_b 与 q_e 为规模/环境权重，均归一化。

\[
\eta^{common}_{A,i,b,e}=\frac{L^*_{i,b,e}}{\tau_{A,i,b,e}}.
\]

推荐总体架构分数：

\[
\boxed{
AQ(A)=100\left[
\sum_b p_b\sum_e q_e\sum_i w_i
\frac{\tau_{A,i,b,e}}{L^*_{i,b,e}}
\right]^{-1}
}
\]

也即：**跨工作负载、跨相同规模预算、跨标准环境，对“相对公共理想下界的 slowdown”取固定权重平均，再取倒数。**

若下界合法且所有点正确，AQ 在 (0,100]。10 分表示该协议下的加权相对下界 slowdown 为 10×；不是性能只有某 GPU 的 10%，也不是还有确定可实现的 10× 提升。

与前文 E 的关键区别：L* 来自公共资源预算，候选降低自己的算力或 bandwidth 不会使分母标准变宽松。加大硬件只能进入更大的 b，同档必须满足相同预算，不能用一颗更大的设计来争取小预算的分数。

覆盖 gate 沿用第 4.8 节：正式 AQ 要求注册域全部合法、正确并完成所需证据；若某点真实不可运行，效率记 0，完整域 AQ 为 0，同时保留 coverage/条件分数。未测量属于证据缺失，不伪装成已知不支持；此时总分标 `NOT CERTIFIED`，可以另报模型预测 AQ。

为了区分模型与实测，使用 `AQ-model`、`AQ-RTL-canonical`、`AQ-certified-domain` 标签。ASIC/FPGA 校准域不足时不授予跨部署已验证结论。

#### 4.10.7 次级交叉检查：相对公共 reference 的分数

公共下界紧度尚在研究时，可同时用冻结 reference family R：

\[
Q_{ref}(A)=100\exp\left[
\sum_{i,b,e}w_ip_bq_e\log
\frac{\tau_{R,i,b,e}}{\tau_{A,i,b,e}}
\right].
\]

reference 为 100；这个分数没有 100 上限。它便于已有实现之间比较，不等于理论最优效率。采用固定权重时，两候选分数之比不受 reference times 的代数选择影响，类似 [SPEC 对加权几何比值的说明](https://www.spec.org/gwpg/gpc.static/geometric.html)。**若换 reference 同时重新生成 w，排名仍可能改变**，因此权重必须独立冻结。

Q_ref 不是 AQ 的偷偷替代：两者分别衡量公共理论资源潜力与相对实现水平。若 AQ 与 Q_ref 排序明显冲突，需要检查下界紧度及权重，而不是挑有利者发表。

#### 4.10.8 硬件代价、能耗与时序如何体现

架构评分需要计入控制、端口和通信资源，不能仅算运算与容量；否则“无限交叉开关、无限端口”会成为虚假最佳架构。τ 的 canonical delay 模型必须反映选择器深度、端口组织、链路与流水，不直接采用“所有 RTL cycle 等长”的假设。

能耗无绝对载体无关单位。初版以 canonical action energy 建立独立的能量约束/效率轨；同一原语活动计数乘公开系数，并做系数敏感性。可在 energy-budget constrained AQ 中把能耗作为硬约束，但不要任意将速度、焦耳、面积相乘再宣称它有唯一正确权重。

物理实现报告映射残差：同一设计 canonical 预测与 FPGA/ASIC 实现的偏差，归因到硬件时钟、packing、SRAM macro、外部接口和工具优化。**不能把不利差异全部归为“部署噪声”**；若某拓扑在多个载体上都无法达到抽象时序/资源假设，就应修订公共模型或判架构不可实现。

#### 4.10.9 如何证明它确实消除了部署和规模偏差

需要专门的 metric validation，而非只有公式：

1. 同一架构不同 FPGA clock 运行，换算回 canonical 模型后 AQ 不变；board performance 应变化。
2. 同一架构在两个 FPGA/一个可获得 ASIC 实现流程上提取原语行为，检查反推出的 AQ 区间是否一致。没有 ASIC 流程时只能声称抽象定义与 ASIC 兼容，不能声称已验证一致性。
3. 同一架构简单增加资源，不允许继续以原 b 评分；不同 b 的结果都进入冻结聚合。
4. 复制不互相作用的工作与资源且无共享瓶颈时，相对下界效率应保持；存在真实跨 core 通信增长时 AQ 可以下降，这正是需要捕捉的扩展性问题。
5. 降候选自身峰值、增加冗余读写、伪造一次完成更大指令均不得提高公共评分。
6. 对成本系数 θ、环境权重 q、规模权重 p 和 workload 权重 w 做预注册扰动；报告排名稳定率、置信区间与反转区域。
7. 使用明显劣化的控制设计：额外 barrier、窄 NoC、减少 RF ports、过大算阵但供数不足。AQ 应解释劣化来源，不能只奖励峰值。
8. 检查 canonical 最优候选在真实实现上是否仍处于好 Pareto 区；若经常选出不能实现的设计，metric 模型不合格。

建议伴随 `AQ-tail`（预注册最差一部分环境/规模的分数）、覆盖与敏感性区间。主总分仍然是 AQ，不把 tail、置信度或 token 乘进去形成难解释的复合数。

#### 4.10.10 与 DevOps 主线的关系

联合迭代的**架构目标改为固定预算下提高 held-out AQ**；同时记录具体部署 S，确保设计没有只适应抽象评测器。Agent 的目标函数和 evaluator 分开，模型不能自行改 L*、canonical costs 或环境权重。

“更少 token 找到同样 AQ”属于设计过程效率，“相同 token 得到更高 AQ”属于固定设计预算收益。这两种结论清楚区分。FPGA 和 NVIDIA 实测是对 AQ 的外部有效性检验，而非定义 AQ 的计分基准。

**AQ 是本稿提出、需要验证的研究协议，不是已有文献证明过的普适架构度量。** R38 的重要研究任务之一，正是验证其下界、抽象成本、跨载体校准和排名稳定性。不能在这些工作完成前写“已经获得与任何平台和规模无关的客观真分数”。

## 5. 从 45 个算子到几千个有意义的点

### 5.1 规模与层次

建议第一版 45 算子 × 每算子约 64 个语义实例，约 **2,880 点**。这是采样预算，不能为了凑 64 给小算子复制相同点；某算子覆盖需求更复杂可以增加，并通过权重防止数量改变总分。

O0/O1/O2 是同一点的编译实验维度，硬件配置与重复次数也是实验维度，均不重复计算 workload 数量。2,880 × 3 levels × 3 configs 已有 25,920 组合，若全做 RTL 会很昂贵，因此必须先测吞吐并采用分层测量。

| 套件层 | 建议规模 | 用途 |
|---|---:|---|
| R37-contract | 现有 149 stages × 3 levels | 不退化的兼容性门槛 |
| Smoke | 6–8 个点，覆盖不同瓶颈 | 每次跨层反馈 |
| Correctness-expanded | 数千形状及输入性质 | 功能与精度边界；功能快模型可分担，但不能冒充 RTL 已测 |
| Perf-development | 约 60% shape families | 开发调优、cost model 训练 |
| Perf-validation | 约 20% shape families | 超参数和候选选择 |
| Perf-held-out | 约 20% shape families | 最终冻结评估；不返回可调优的逐点反馈 |
| RTL/FPGA anchors | 分层选取约 100–300 点起步 | 校准与真实证据；最终范围按容量和时长扩大 |

“几千点在不同配置下”的最终报告必须逐行注明 `measured-RTL / measured-FPGA / predicted`。预测可以覆盖整个空间，但论文不得把它们统称几千点实测。主表优先使用实际完成测量的冻结集合，并同时公开其相对于完整集合的覆盖。

### 5.2 全部 45 算子的规划清单

| 类别 | 算子（保持 manifest ID） | 重点维度 |
|---|---|---|
| 基础逐元素（8） | vector_add、mul2、relu、leaky_relu、sigmoid、fused_activation、swiglu、dropout | N、对齐/尾部、stride、SFU、RNG 语义与种子 |
| 格式与位置（4） | quantize_global、dequantize_rowwise、weight_dequant、rope | 分组长度、量化 scale、饱和、维度、offset、格式 |
| Reduction/Norm/损失（10） | softmax、layernorm、rmsnorm、l2_norm、mean_reduction、argmax、cross_entropy、kl_divergence、batch_normalization、moe_topk_gating | rows/width、长归约、epsilon、ties、极值、top-k |
| 原子统计（1） | histogramming | bin 数、均匀/热点/全相同分布、原子冲突 |
| 矩阵（4） | matmul_fp32_fp16_fp8、matmul_int8、batched_matmul、streamk_matmul | M/N/K、batch、矩形、tail、长 K、精度、grid 不均衡 |
| Attention（4） | flash_attention、flash_decode、block_sparse_attention、linear_self_attention | B/H/seq/head-dim、causal、KV 长度、稀疏 pattern、prefill/decode |
| Conv/Stencil（6） | 1d_conv、2d_conv、3d_conv、2d_max_pooling、gaussian_blur、jacobi_stencil_2d | channels、kernel size、空间尺寸、边界、工作集 |
| 数据布局（5） | matrix_copy、matrix_transpose、interleave、reverse_array、destindex | 形状、stride、对齐、索引 locality、bank 冲突 |
| 排序选择（3） | bitonic_sort、radix_sort、top_k_selection | N、key 分布、重复值、K/N、阶段数、稳定性要求 |

合计 45。具体支持的输入参数以原始 operator 定义为准；表中维度不是强迫原算子增加不存在的语义。

### 5.3 形状生成规则

每个适用维度涵盖：

- 对齐值和 `tile−1 / tile+1`，不只测 2 的幂。
- 小启动开销区、计算饱和区、RF 容量边界、NUCA 容量两侧、DRAM 区。
- 非方 GEMM：skinny/tall、batch=1 与多 batch、长 K。
- 连续、支持的非连续 stride、transpose、低 locality。
- 均匀随机、零、重复值、热点、正负极值；NaN/Inf 按上游语义明确支持或排除。
- Attention 的小 batch 长序列与 decode 长 KV，防止只优化 GEMM。

以 shape family 分割训练与测试，例如把一段 K 范围、一个 head-dim 或一种缓存容量比例整组留出；随机拆相邻 shape 容易泄漏。另设未见硬件配置 holdout 检查 cost model 的跨配置能力。

### 5.4 正确性与精度协议

每点记录 reference 实现与版本、输入 seed/hash、dtype、累计精度、rounding、rtol/atol、相对 L2/最大绝对误差，以及 guard/非法地址检查。容差由数学语义和 reference 分布预先确定，不因候选失败而放宽。

INT8 长 K 必须检查整数累加范围与浮点 fallback 的精确性边界。Dropout/RNG 若要求相同算法就比确定输出；若只要求分布语义，应另有统计核验，不能拿不同随机输出做逐元素相等。Argmax/top-k/sort 明确 ties、稳定性和索引规则。

对异步/atomic 算子，加入随机延迟、乱序返回、拥塞、极限 credit、超时 watchdog 和多 seed。O0/O1/O2 都保留 correctness gate；主性能 track 预注册一个 deployment level（建议 O2），O0/O1 用于编译优化消融，不能逐点选最快 level 后仍称固定 pipeline。

## 6. Cost model：预测、解释和选择实验

### 6.1 与 ArchBench 下界严格分工

ArchBench 的下界是相对稳定、独立审计的评分标准；cost model 是会犯错、会更新的预测器。不能用 cost model 的预测时间既当优化目标又当最终评分证明自己的成功。

建议三层结构：

1. **静态分析层**：从真实 IR/机器码提取工作量、指令 mix、live range、RF/accumulator 占用、spill、访问模式、tile padding、依赖 DAG。
2. **架构分析层**：描述执行单元、bank/port、NoC、MSHR、queue、barrier、角色映射和 grid 调度，估计可重叠执行与瓶颈。
3. **残差与不确定性层**：用实测校准 analysis 的误差，输出预测区间、失配原因和下一项最值得测的候选。

第一版先做可解释的 analytical + lookup calibration，不必一开始训练大模型。借鉴 [Timeloop](https://timeloop.csail.mit.edu/) 的映射与数据流思路和 [MAESTRO](https://maestro.ece.gatech.edu/) 的分析方法，但动态 cache/反压必须由 TNCA 数据验证。

### 6.2 不能简单把所有时间相加

粗略模型可以写成：

\[
\hat T=T_{launch}+T_{scheduled\ DAG}+T_{drain}+\epsilon.
\]

`T_scheduled DAG` 应来自资源约束下的依赖调度或校准模型；不是 `compute + memory + stall counters` 的机械求和。部分搬运与计算可重叠，多个 stall 也可同周期发生。

先建单 core、单 program 的模型，再加入多 program 分配与跨 core 通信。每步都设独立校准 gate。以 host 串行 grid 数据训练的模型不得直接宣称可预测并行全芯片。

### 6.3 必需建模维度

| 子系统 | 输入 | 预测 / 解释 |
|---|---|---|
| Compute | dtype、tile、instruction mix、tail | 有效吞吐、padding 浪费、MMA/SIMD/SFU 平衡 |
| RF | layout、bank、ports、live range、window | conflict、等待、spill、额外端口代价 |
| Cache/NUCA | reuse distance、home、容量、bank、MSHR | 工作集容量风险、hit/miss 服务、合并和热点 |
| NoC | request/response 路径、注入与 cut | 远端访问、反压、链路容量限制 |
| Async | producer/consumer、tokens、buffer stages | 流水可重叠程度、同步关键路径 |
| Runtime | grid、launch、descriptors、code/arg upload | 小 kernel 启动成本与批处理收益 |
| Physical | 参数、综合数据、P&R 样本 | 可实现性、资源区间、Fmax 风险；无校准不输出确定 watt/mm² |

### 6.4 模型验收

先记录 baseline，再冻结建议门槛：

- held-out 点 median relative error ≤15%、P90 ≤30% 作为第一目标；同时报告小延迟点的绝对误差，避免相对误差失真。
- Spearman 排名相关建议 ≥0.8；更重要的是 top-k 候选的真实 regret。
- 给定测量预算，model-guided 搜索相对 random 与 analytical-only 的最佳实际性能。
- 未见形状与未见配置分别报告，不混成一个平均误差。
- 90% 预测区间的实测覆盖率、过度自信比例和 OOD 拒绝率。
- 特别记录“预测更好但实测更差”的候选及其资源/流量原因。

这些不是论文必须达到的任意硬门槛：pilot 若表明门槛不合理，可以在正式实验前改版；正式测试后不得调门槛隐藏失败。

### 6.5 校准与在线学习的实验边界

cost model 的训练集、counter calibration、训练/推理时间和测量费用全部记账。Agent 实验给两种口径：

- 冻结模型：所有 run 使用同一训练好的 model，便于解释信息增益。
- 在线更新：允许每轮吸收自己的测量，训练费用计入该 run；不能读取其他实验组或 test split 的结果。

同时报告一次性模型构建成本，以及重复使用 N 次后的摊销曲线。没有报告建设成本，就不能称整体更便宜。

## 7. Profiler：从计数到跨层归因

### 7.1 延续已有 counter 契约

沿用 [r3_counters.json](../spec/r3_counters.json) 的事件、位宽、reset 和 overflow 规则，新增字段必须有来源模块、触发条件、单位和 ROI 边界。

尤其注意当前 `perf_cycles` 是“任一 role 正在运行”的周期，未必等于包含排空的完整 kernel makespan。增加或明确 `launch_accepted → final_store_visible & drained` 的独立时间基准，并与 runtime 返回值交叉验证。

### 7.2 第一批必要指标

| 领域 | 指标 | 可回答的问题 |
|---|---|---|
| 时间 | host enqueue、device accept、first issue、last store、drain | 损失发生在启动、执行还是收尾？ |
| Compute | issued/completed、active lanes、MMA busy、SFU busy | 算力闲置还是尾块浪费？ |
| RF | read/write grant、bank conflict、等待周期、spill bytes | 是否该改 layout、端口或 tile？ |
| Memory | 请求/响应 bytes、hits/misses、MSHR/target 占用、writeback | 数据复用还是并发度不足？ |
| NoC | 每 link bytes、queue occupancy、backpressure、路径 | 远端热点或 bisection 是否主导？ |
| Async | outstanding DMA、window 生命周期、barrier 等待 | 生产消费不同步还是资源无法释放？ |
| Grid | 并发 program、每 core 工作量、尾部空闲 | 调度与负载不均衡损失多少？ |

只有 RTL 中真正存在并已接到公共后端的 counter 才标 `available`；其余 `planned` 或 `unavailable`。不能把 testbench memory 的事件当真实板卡 DRAM controller 计数。

### 7.3 Trace 关联

构造以下映射：

```text
operator/case → source span → TTIR/target IR op
             → machine PC range → instruction/event → RTL resource
```

静态 debug map 放在 AOT sidecar 或 artifact metadata；动态只记录 PC、时间、event ID 和必要 tag，减少 trace 量。支持 summary、采样 trace、完整定向 trace 三种级别。

Profiler 输出给人和 Agent 共用同一个结构化 schema。自然语言解释只能引用已有数据，不能把“hazard stall 最大”直接断言为“加 RF 端口一定有效”。需要编译器 layout 消融或 hardware feature toggle 验证因果。

### 7.4 计数守恒与测量扰动

- 互斥 stall 的和应与定义中的 total 相等；重叠组不可求和成总时间。
- issued/completed/flush/fault 的关系有明确守恒式。
- DRAM/NUCA/NoC bytes 的定义包含或排除 metadata、重试、writeback 时必须一致。
- 多 core cycles 不能相加当墙钟周期。
- ROI 开始前清楚规定 cache 和 queue 状态；ROI 结束包括必要 drain。
- 当前 counter 通常常开；若新增 trace enable，验证开关对功能、周期和 P&R 的影响。
- GPU timing 与 counter collection 分开运行，避免 Nsight replay/插桩时间混入正常 latency。

## 8. 三个优先的跨层优化案例

这些是**待测假设**，不是预先决定一定要加的硬件。先用 profiler 证实损失，再实施最小变更。每个案例必须有单层、组合与资源消融。

### 8.1 案例 A：RF bank / window / dot 供数

- 现象假设：没有软件可寻址 SPM 的结构中，RF 供数和 layout 转换可能限制 MMA。
- Compiler：bank-aware layout、fragment reuse、live range 缩短、tile/stage 选择。
- ISA：若现有指令不足，加入版本化 window/layout 或 operand reuse 表达；优先复用现有语义。
- Microarch：operand collector 仲裁、holding/bypass、bank/port 组织。
- 测量：有效 MMA 服务率、RF 冲突、搬运字节、spill、资源和 Fmax。
- 核心消融：只改 layout；只改 hardware；两者；加入 ISA 表达后的整体。
- 停止条件：周期改善不足以抵消频率/资源代价，或普通 compiler 调整已获得全部收益。

### 8.2 案例 B：异步 descriptor → RF → compute 流水

- 现象假设：单角色顺序执行、descriptor 设置与同步占据大量空闲。
- Compiler：producer/consumer 分区、stage 数选择、所有权 verifier。
- ISA：明确完成事件、window 可复用条件和资源准入；不能只靠编译器“猜延迟”。
- Microarch：DMA outstanding、返回资源预留、角色调度、完成路径保活。
- 测量：计算与搬运重叠、outstanding 分布、barrier 等待、drain 时间。
- Stress：慢速乱序内存、最小 queue、消费者延迟、故障/取消。
- 安全不变量：节流新请求不阻止在途返回、消费者和资源释放；不能靠无限 buffer 避免死锁。

### 8.3 案例 C：跨 core grid / NUCA locality / admission

- 前提：先把公共 runtime 串行 grid 与真实 device scheduler 的边界打通，不能从当前单 core 数据外推 16 tile。
- Compiler：tile 工作集估计、可选 home/reuse hint。
- ISA/runtime：可忽略的 locality hint、grid capability 和版本契约。
- Microarch：设备端 program 分配、NUCA home-aware scheduling、MSHR admission。
- 测量：整 grid makespan、每 core 空闲、remote bytes、热点、扩展性。
- 基线：round-robin、无 hint、静态 locality、反馈 admission。
- 相关工作边界：明确继承 DCO/LLaMCAT 的部分；R38 的新意必须来自可执行跨层演进及其新机制/结果。

### 8.4 扩展候选及优先级

SFU/reduction 宽度、原生整数 dot、code size/指令供给、小 kernel 快路径都是候选。优先做能够覆盖多个算子类别、预计能改善主瓶颈且验证代价较低的变更。避免一次加十种机制后无法归因。

至少保留一个失败或无收益案例，展示 cost model 错在哪里、profiler 如何纠正、回退成本多少。这比只展示成功故事更能说明闭环的用途。

## 9. 证明“联合迭代”的因果实验

### 9.1 固定哪些条件

所有组同一起始状态 V0、相同 workload development split、同一硬件预算、工具资源、正确性要求与最终 held-out evaluator。编译器单层组也获得合理 profiler/model 信息，不能故意让对照组盲调。

| 组 | 可改变范围 | 目的 |
|---|---|---|
| B0 | 固定 R37，统一部署优化级别 | 起点 |
| B1 | Compiler-only，固定 ISA/RTL | 常规编译优化能做到多少 |
| B2 | Hardware-only，固定语义与编译策略 | 只调硬件参数/实现的效果 |
| B3 | Compiler + hardware，ISA 固定 | ISA 演进的增量价值 |
| B4 | 分阶段 C→I→H，只允许阶段末集成 | 长反馈、阶段式协同基线 |
| B5 | 同一空间的交替优化，定期集成 | 排除“只是允许更多变量” |
| B6 | 三层短反馈联合迭代 | 核心方法 |

可再增加 model-only DSE 与人设计基线。B4/B5/B6 尽量共享可选变更集合与工具，区别是反馈粒度和变更组织，避免把更强实现能力误归因给 DevOps。

### 9.2 两种互补预算口径

1. **等总墙钟 / 工具资源预算**：验证短反馈能否在相同时间获得更好结果；编译、仿真、排队、修复都计入。
2. **等有效测量次数 / 等候选数**：减少“只是多试了几次”的影响，检查信息和联合决策的质量。

另报告人工小时或 Agent token，但不能用二者直接等价替换。综合机器并行度、GPU/FPGA 占用、缓存命中政策要一致；warm-cache 与 cold-cache 分别报告。

“等候选数”本身也不能完全控制设计复杂度，因此同时公开 patch 范围、受影响模块、资源变化和 build 时间，不用代码行数作质量指标。

### 9.3 结果与交互效应

主结果：相同设计预算下最佳 feasible AQ（第 4.10 节）；同时报告具体部署 S、自身效率 E、资源、正确性与迭代成本。AQ 的抽象环境和资源域在各组之间固定。展示 anytime 曲线和 Pareto frontier，使用共同的参考点计算 hypervolume，参考点必须在看测试结果前冻结。

对一个可开关机制，可测 2³ compiler/ISA/hardware 因子组合；不合法组合标不可执行，不能硬凑不存在的设计。如果 ISA 新原语在旧硬件不能运行，提供有相同语义的合法软件实现作为对照并明确其开销。

跨层交互可以用 log-speedup 的差异分析，而不能只把三个独立 speedup 相加。典型证据应是“硬件 alone 收益小，compiler alone 受限，组合后供数瓶颈解除”，并用 counter 支持。

### 9.4 统计与泛化

- pilot 建议每组至少 5 个独立搜索 run；正式样本量由方差和最小有意义效应确定。
- 同一任务、起点和预算配对比较；agent 不确定性与测量噪声分别估计。
- 对 workload 聚合采用 operator/shape-family 分层 bootstrap；几千相邻 shape 不是几千独立架构样本。
- 报告 confidence interval、效应量、失败数与所有 run，避免只挑最好 seed。
- 在小/中/大配置和未见 shape family 上复核；时间允许再迁移到第二个硬件 profile。
- 若只有一种架构、三个案例，结论限制为该范围，不能声称对所有硬件设计普遍有效。

## 10. Agent 实验：model 与 profiler 的收益是否值得成本

### 10.1 主实验为 2×2，而非只有“有/无”

| 组 | Cost model | Profiler | 共同可见信息 |
|---|---|---|---|
| A00 | 无 | 无 | 源码、契约、编译错误、正确性、端到端 latency |
| A10 | 有 | 无 | 同上 + 静态预测、区间、资源估计 |
| A01 | 无 | 有 | 同上 + counter/trace 与来源映射 |
| A11 | 有 | 有 | 两类信息与一致的结构化接口 |

A00 仍有 latency 和 correctness，否则它无法优化，也不是公平基线。“无 profiler”明确为没有瓶颈 counter/trace，不是没有运行结果。

必须控制信息泄漏：A00 不能通过读取其他目录的 profile JSON 获得隐藏数据；A10 不能读取本轮真实 counter；所有组使用相同搜索预算，不让 A11 获得免费额外硬件测量。测试 oracle 与 held-out 结果对所有组只读且不进入工作区可修改范围。

### 10.2 实验任务集

先选择 8–12 个任务：RF 冲突、SFU 占用、访存并发、descriptor 开销、长 reduction、grid imbalance 等。每项给出同一起点与目标预算，按编译器、ISA/RTL、跨层三类分层。

主实验以允许改 compiler/ISA/microarch 的跨层任务为中心；另设纯 schedule 调优小任务与 Ansor-style search 比较，不拿 Agent 修改 RTL 的大任务和 random tile-size search 混成一张排名。

不得把开发者已知最终 patch 放在 prompt、历史记录或可读日志。各 run 使用独立干净工作区；保留完整对话与工具轨迹的可发布脱敏版本。

### 10.3 记录与费用

固定模型标识/快照（若服务不提供快照，记录服务版本与实验日期）、推理配置、上下文预算、prompt、工具版本、重试政策和随机 seed。模型不可完全复现时明确限制，并尽量在紧邻时间随机交错运行各组。

至少记录：

- input/output/cached tokens；平台可提供时记录 reasoning usage，不反推不可见内容。
- API 实际费用，记录当时计费规则；不能只用未来动态价格重算后声称当时成本。
- compiler/RTL/synthesis/GPU/FPGA 时间、工具调用次数、失败调用和超时。
- 模型训练/校准成本、profile 获取成本、人工干预分钟数。
- 最佳候选的实际性能、资源、正确性与复现成功率。

总成本可写成 `API + compute rental + amortized calibration`；人工成本单列，除非明确公开折算假设。

### 10.4 Token efficiency 的定义

建议同时报告：

1. 固定 token 预算下的最佳 feasible speedup。
2. 达到预注册 improvement threshold 所需 token，未达到则作为 censored run。
3. improvement–token 曲线的面积，所有组在相同区间积分。
4. 每个有效改善候选的 token 与美元成本。
5. 相同墙钟预算下的质量，防止“省 token、耗费十倍综合时间”被称整体更高效。

若用 `log(S_best)` 积分，应说明曲线只更新为已验证最佳候选；初始 log(1)=0，失败不产生正收益。达到阈值的只统计成功 run 会产生幸存者偏差，应同时给成功率和删失处理。

采用 factorial 分析估计 model、profiler 的主效应与交互项；不要预设 A11 一定最好。结构化 feedback 可能耗费更多上下文，或 model 错误误导 Agent，这也是有效研究结果。

### 10.5 额外基线与停止规则

比较 random search、固定启发式、分析模型引导搜索、条件允许时 Bayesian/evolutionary search。把同一个 cost model 提供给非 LLM 搜索，判断收益来自模型本身还是 Agent 的跨层推理。

每个 run 预设 token、墙钟、工具次数三重预算；先达到哪项就停止。允许在预算内修复错误，但不能修改评测脚本、降低精度、删除慢点或隐藏 CPU fallback。最后由独立 evaluator 重新运行所选制品。

## 11. NVIDIA 对比：把“打 NV”拆成可检验的命题

### 11.1 四个层次分别回答

| 问题 | 合理结论 | 不能推出的结论 |
|---|---|---|
| TNCA 的自身 roofline 效率更高吗？ | 对规定 workloads，资源上界利用更充分 | 比 GPU 更快、更省面积 |
| 某些点的绝对 latency/throughput 更好吗？ | 对具体设备、精度与场景胜出 | 所有 45 算子或所有规模全面胜出 |
| 实测 board-level energy 更好吗？ | 同口径测量下该系统每任务能耗更低 | ASIC 核心功耗优于 NVIDIA die |
| 相同技术/资源假设下架构更好吗？ | 经校准模型、实现约束与敏感性分析支持的架构判断 | FPGA LUT 可以直接换算 GPU mm² |

FPGA 与先进工艺 GPU 的绝对吞吐差距可能很大。R38 的核心方法贡献不应以全面击败 NVIDIA 为成败条件。可重点观察小 batch、带宽受限、launch-sensitive 或特殊访存场景，但必须报告预注册完整分布，不能事后只保留获胜点。

### 11.2 基线矩阵

1. 同一原始 Triton kernel 的 GPU 后端，固定源码与合理 autotune 预算。
2. PyTorch eager，仅作为生态 reference，不作为唯一强基线。
3. PyTorch compiled 路径，固定版本和图边界。
4. 适用算子的 cuBLASLt/cuDNN/CUTLASS 或专用成熟实现；每个库的精度/算法选择有记录。
5. TNCA 的 R37 baseline、R38 compiler-only 与最终联合设计。

GPU 型号按实际可获得设备冻结，不在计划阶段虚构 H100/B200 的可用性或价格。至少一款数据中心 GPU；若有第二款，检查结论是否受代际特性影响。不适用的库标 N/A，不用错误库替代。

### 11.3 统一实验协议

- 数学语义、输入、输出、累计精度、TF32/FP16/FP8、稀疏、approximation 明确匹配。
- device-resident kernel latency 与 host→device→host end-to-end 分开，二者都可有意义。
- latency 与 throughput 用不同场景测；禁止 TNCA batch 吞吐对 GPU 单次 latency。
- warmup、重复次数、stream、同步、CUDA Graph 使用、cache 状态、allocator 状态冻结。
- 报 median、尾部与测量误差；对极短 kernel 用重复执行并核算循环/launch 开销。
- 记录时钟、功耗限制、温度、驱动、工具链；动态频率用实测状态，不只报标称频率。
- 功率测量要足够长、说明 idle subtraction 与传感器边界；TDP 不等于实测能耗。
- NVIDIA profiling 与普通 timing 独立运行；统计工具本身可能 replay 或扰动缓存。

### 11.4 FPGA vs ASIC 投影

主文可以同时给三层证据，但不能混合：

- **RTL target cycles**：指定内存/网络模型的目标周期，非主机仿真耗时。
- **FPGA board measurements**：实际频率、控制器、DMA 与电源边界。
- **ASIC projection（可选）**：只有获得公开可复现工艺库、SRAM 假设、时序/功率流程和校准后才考虑；报告不确定性和敏感性。

现有 generic Yosys gate/depth 数据只能支持结构趋势。R38 不应为了对标 GPU 而把它们换算成缺乏依据的 TOPS/W。

## 12. PyTorch 与 FPGA LLM：作为闭环的系统级验证

### 12.1 两条路线分阶段做

**路线 P1：图级集成（先完成）。** 从 `torch.export` 或注册的编译后端取得受支持的 inference graph，lower 到已有 Triton/TNCA AOT，使用 Session 执行。此阶段可以有 host-owned tensor 包装，但必须承认传输边界，不能声称完整 `device='tnca'` 已实现。

**路线 P2：真实 device backend（后续完成）。** 按 [PyTorch accelerator integration](https://docs.pytorch.org/docs/main/accelerator/index.html) 与 [PrivateUse1](https://docs.pytorch.org/tutorials/advanced/privateuseone.html) 路径实现 allocator、copy、device guard、events/streams、dispatcher、必要 tensor metadata/alias 行为。先做 inference，不把 autograd/distributed 放入 R38 必选范围。

接口版本随 PyTorch 演进；实施前固定 release 和官方接口，不照抄本稿检索时的 `main` 到另一版本。

### 12.2 公共 AOT 必须是唯一执行主路径

所有 profiler、benchmark 和 PyTorch 适配最终复用 `compile_aot → artifact → Session`，不要为每篇实验建立绕过 runtime 的内部 codegen 小路。新增参数通过明确的 profile/capability/ABI 管理；用户应像 tutorial 一样运行。

保留原 tutorial 08/09 作为简单入口；新增教程草案：

- `10_archbench_profile.md`：编译一个 GEMM/softmax，读取正确性、周期和瓶颈报告。
- `11_cross_layer_iteration.md`：一个可切换 feature 的 baseline/改进/回退全过程。
- `12_pytorch_inference.md`：固定小图在模拟与板卡后端运行，列出 fallback。
- `13_fpga_decoder.md`：一个 decoder block → 完整小模型 prefill/decode。

这些是计划新增文件，不是当前已经存在的教程或 CLI。

### 12.3 内存与模型规模先算清楚

权重内存近似：

\[
M_{weights}=N_{params}\cdot bits/8 + M_{scales}+M_{metadata}.
\]

普通 KV cache（不含对齐/管理额外开销）：

\[
M_{KV}=2\cdot L\cdot B\cdot S\cdot H_{kv}\cdot d\cdot bytes_{element}.
\]

总预算还包括 activation、workspace、code、DMA buffer 和碎片。先选择实际能装入板卡外存的模型；建议先做 0.1–1B 级 decoder 的可行性评估，最终型号、许可证、权重 hash 由板卡和精度能力确定。不要在当前 512 KiB 默认 simulator 配置上承诺完整模型。

若要量化，从精确支持和质量评估开始；不能因为省内存就默默改变整套比较的数值契约。

### 12.4 分级验收

| 级别 | 内容 | 证据 |
|---|---|---|
| P0 | 单个 ATen/custom op → AOT | 数值、shape/stride、设备结果 |
| P1 | GEMM + norm + activation 小图 | 无多余 host 中间往返、时间边界 |
| P2 | 一个 decoder block | attention、RoPE、KV 更新与 logits reference |
| P3 | 完整小模型 prefill | prompt batch/length、logit 误差、内存峰值 |
| P4 | 多步 autoregressive decode | KV cache 复用、生成一致性/质量、tokens/s |
| P5 | FPGA 上可复现演示 | bitstream/hash、运行命令、日志、截图、功率/延迟 |

host tokenizer、打印和采样可明确留在 CPU；模型算子 CPU fallback 必须逐个披露。主“模型在 FPGA 运行”结论要求约定的模型计算子图没有静默 fallback。动态图/unsupported op 应报清晰错误或列在 fallback 日志。

测 TTFT、prefill tokens/s、decode tokens/s、每 token latency、KV 长度曲线、峰值内存。生成文本只是展示；额外比较固定 token 序列的 logits、误差和适用的 perplexity/质量指标，避免文本“看起来正常”替代正确性。

Appendix 截图应包含模型 ID、配置、板卡、bitstream hash、命令和 tokens/s；同目录提供机器可读日志，截图本身不作为性能原始数据。

## 13. 工程组织、接口与实验数据

### 13.1 建议目录（均为未来实施建议）

```text
spec/r38/
  workload.schema.json       # 语义点与精度
  architecture.schema.json   # 配置、资源及实现版本
  experiment.schema.json     # 完整实验 manifest
  counter.schema.json        # 延续 r3 counter 契约
  budgets.json               # 固定资源与搜索预算
  suites/                    # split、权重与来源

tools/r38/
  validate_manifest.py
  run.py                     # 只经公共 AOT/runtime
  calibrate.py
  score.py
  compare.py
  agent_experiment.py
  reproduce.py

python/tnca/                 # 扩展既有公共 API，不另起 runtime
compiler/                    # 真实 pass/lowering 与 provenance
rtl/                         # 可切换、可验证的架构变更
sim/build/r38/               # 生成的报告、trace、artifact，不加入 Git
```

命令目标可设计为 `make r38-smoke / r38-correctness / r38-calibrate / r38-bench / r38-report`，但当前尚未实现；本稿不应被当作这些命令已经可运行的说明。

### 13.2 单次测量记录

```yaml
# 字段设计示例；null 表示未知，绝不是零
schema: archbench-measurement/1
candidate_hash: <hash>
parent_hash: <hash>
workload_id: <operator-shape-dtype-layout-distribution>
source_hash: <hash>
input_hash: <hash>
compiler: {hash: <hash>, opt_level: 2, passes_hash: <hash>}
isa_hash: <hash>
rtl_hash: <hash>
artifact_hash: <hash>
profile: <exact-profile>
evidence_level: measured-rtl
correctness: {status: pass, oracle_hash: <hash>, max_abs_error: null}
time:
  roi: launch_accept_to_final_store_and_drain
  device_cycles: null
  achieved_clock_hz: null
  device_seconds: null
  host_seconds: null
traffic: {required_lower_bound_bytes: null, measured_bytes: null}
bounds: {version: <hash>, strict_seconds: null, empirical_seconds: null}
resources: {lut: null, ff: null, dsp: null, bram: null, power_w: null}
repeat: {seed: 0, index: 0}
status: <pass-or-explicit-failure>
```

正式 schema 不使用尖括号占位值；所有单位入字段名或枚举。每个结果附 provenance 和 raw-data hash，score 工具只接受版本与计时协议兼容的记录。

### 13.3 缓存与实验复现

缓存 key 至少包含源、参数、shape specialization、compiler、ISA、profile、feature 与 optimization options。硬件/ABI 改变时旧制品不得静默复用。报告 cold 与 warm build 成本，记录 cache 命中。

二进制、bitstream、trace、模型权重等大产物放 artifact storage 或本地 build 目录；Git 保存 schema、脚本、小型基准描述与校验哈希。延续 `.tnca` 的生成物忽略策略。

发布包包含一条最小重建路径、依赖锁定、原始结果、分析脚本、失败结果和 license。不能要求复现者从截图手抄数据。

## 14. 分阶段 Roadmap 与验收目标

### 14.1 依赖关系与优先级

```text
R38-00 baseline/provenance
       ↓
R38-01 最小三层 loop ───────────→ R38-02 ArchBench 协议
       ↓                                  ↓
R38-03 profiler ←──────────────→ R38-04 cost model
       ↓                                  ↓
R38-05 三个跨层案例 → R38-06 方法/Agent 对照实验
       ↓                                  ↓
R38-07 FPGA/NVIDIA 测量 ────────→ R38-09 论文与 artifact
       ↓
R38-08 PyTorch/LLM 系统演示 ────→ Appendix / 系统验证
```

板卡选型、工具许可、外存控制器和 host transport 可尽早做可行性验证；不要等所有模拟优化结束才发现目标板卡无法实现。ArchBench 规则在大规模优化之前冻结，避免为已知赢家定规则。

### 14.2 工作包合同

| ID | 工作包 | 完成条件 | 失败时的调整 |
|---|---|---|---|
| R38-00 | 基线与证据审计 | 复现 R37 三档正确性；区分 R36 pending gate；记录实际 AOT/MLIR 路径；环境/配置锁定 | 先修复基线，不拿旧报告冒充当前通过 |
| R38-01 | 最小三层闭环 | 一个真实跨层 feature 从源码到 RTL；compatibility 拒绝、回退、provenance 可演示 | 缩小代表点，保留完整路径 |
| R38-02 | ArchBench v0 | 45 算子 manifest、split、精度、权重、canonical 资源账本、公共 L*、AQ 及 metric 自检 | 无严谨下界的类先用绝对性能，不伪造 roofline |
| R38-03 | Profiler v0 | ROI 与 counter 守恒、IR→PC 映射、至少 RF/compute/memory 三类瓶颈可定位 | 用真实少量指标，不虚构完整 trace |
| R38-04 | Cost model v0 | 独立 holdout 精度、ranking、regret、不确定性与构建费用 | 若动态模型差，限制适用范围并增加 anchor |
| R38-05 | 跨层优化 | 三个案例，至少两个不同瓶颈；资源约束、完整消融、保留负结果 | 若全是 compiler 收益，调整论文核心结论 |
| R38-06 | 受控研究 | 单层/阶段/联合比较；2×2 Agent；预算与复现记录完整 | 若样本不足，只称 case study，不泛化 |
| R38-07 | 实测平台 | 指定 FPGA P&R/bitstream/公共 runtime 跑通；NVIDIA 强基线；时间/能耗分层 | FPGA 未就绪时明确收缩为 RTL 方法论文，不声称上板 |
| R38-08 | PyTorch/LLM | 图级入口、decoder block、可行小模型实板运行、无隐式算子 fallback | 可将完整 device backend 推后；保留真实小图 |
| R38-09 | ISCA-ready evidence | held-out freeze、所有主表可重建、近邻比较、威胁与失败公开 | 未达核心证据则继续研究，不用演示堆砌结论 |

### 14.3 建议时间盒

以下按具备编译器、RTL 与板卡工程能力的项目团队估算，是相对周计划；**不是对一个人或单个 Agent 的工期保证**。投稿年份/官方截止时间另行确认，不凭空填写 ISCA deadline。

| 时间 | 主要工作 | 阶段 gate |
|---|---|---|
| 第 1–2 周 | baseline 审计、近邻正文复核、平台可行性、ArchBench 公式/协议 pilot | G0：知道实际起点与资源约束 |
| 第 3–4 周 | 最小三层变更、版本契约、ROI/counter、smoke gate | G1：真正可运行的短闭环 |
| 第 5–7 周 | workload 扩展、校准、cost model v0、测量吞吐 pilot | G2：可信评分与预测，不是只会跑 |
| 第 8–11 周 | 案例 A/B/C、逐项消融；FPGA bring-up 持续推进 | G3：跨层收益及其原因 |
| 第 12–14 周 | 方法对照、Agent pilot/正式 runs、NVIDIA 强基线 | G4：固定预算的因果证据 |
| 第 15–17 周 | PyTorch 小图→decoder→模型，冻结 held-out 全测 | G5：真实应用与外推验证 |
| 第 18–20 周 | 论文、artifact、独立复现、资源/统计审计 | G6：可投稿证据包 |

实际排程先测 `单次 full regression / synthesis / FPGA build / agent run` 时长，再形成 CPU/GPU/FPGA 小时预算。若一个完整回归需要数小时，不能把“每天上百次完整三层试验”写成计划。

### 14.4 前两周具体行动顺序

1. 记录 branch/commit、生成物哈希、锁定环境；从干净环境重跑基线，单列 pending gate。
2. 绘制公共 AOT 的真实 IR/pass/后端路径，标记 Python 与 C++ MLIR 的责任边界。
3. 运行 8 个代表点，核对 ROI 周期与 host 时间，测工具耗时和容量限制。
4. 给一个算子的下界手算、脚本算和 RTL 数据做交叉检查；加入故意多搬运/降低峰值的 metric 测试。
5. 冻结 ArchBench v0 的算子定义、输入分布、weight policy 和 split 方法。
6. 选择最小跨层案例，优先 RF layout/operand collector 或已有异步 feature 的一致演进，避免先造整套新 ISA。
7. 明确 FPGA 板卡、可用资源、P&R 环境、外存/host 接口；先测一条 GEMM 路径。
8. 完成 HASCO/ACT/Chipyard/Gemmini 的近邻矩阵，据此更新论文贡献措辞。

## 15. 论文结构与关键图表

### 15.1 工作标题（暂定）

**Closing the Loop: Continuous Co-Evolution of Compiler IR, ISA, and Microarchitecture for Tensor Accelerators**

中文：**面向张量加速器的编译器、指令集与微架构持续联合演进**。

标题先不放“打败 NVIDIA”或具体 speedup；结果出来后再定。若最终突出的是迭代方法而非特定新硬件机制，摘要要坦率写清楚。

### 15.2 建议章节

1. Motivation：分层冻结导致哪些具体跨层失配与反馈延迟。
2. Background/related work：敏捷硬件、co-design、MLIR/ISA 生成、Agent 优化。
3. Executable loop：契约、manifest、验证晋级、反馈与回退。
4. ArchBench：工作负载、资源下界、权重、预算与评分约束。
5. Co-evolution case studies：三个可解释案例。
6. Evaluation：设计质量、迭代成本、Agent 消融、规模泛化与 NVIDIA。
7. Limitations：平台、下界紧度、模型/Agent 复现、研究范围。
8. Appendix：PyTorch/LLM、完整 shape 列表、counter 定义、截图与 artifact 指南。

### 15.3 主文必须优先保留的图表

| 图表 | 回答的问题 |
|---|---|
| 闭环与变更契约图 | 方法具体是什么，如何保持可执行？ |
| 不同迭代方式的 feedback latency 分布 | 是否真的缩短反馈周期？ |
| best feasible performance vs 总设计预算 | 相同成本下是否更好？ |
| 跨规模/环境 AQ、资源—性能 Pareto + 各配置 roofline 效率 | 是否靠加资源或降峰值得分？ |
| 三案例的消融及 counter 归因 | 收益为什么需要跨层？ |
| 45 算子热图 / CDF | 是否只在 GEMM 或少数点有效？ |
| Agent 2×2：质量—token / 质量—墙钟曲线 | model/profiler 是否提高效率？ |
| model prediction vs actual 与 regret | cost model 是否可靠、何时失败？ |
| FPGA/NVIDIA 分层结果 | 跨设备比较到底支持何种结论？ |

Appendix 放全量数据，主文保留足够证据使论点独立成立。不要把所有失败和误差都藏到补充材料。

## 16. 风险、停止条件与可以接受的负结果

| 风险 | 早期信号 | 对策与结论边界 |
|---|---|---|
| 只有工程集成，没有研究增量 | 近邻已有同样机制，实验只有“跑通” | 强化跨层交互与固定预算研究；必要时调整定位 |
| Roofline 得分可被操纵 | 降峰值/加冗余后分数上升 | 固定权重、semantic bytes、伴随 S 与资源约束 |
| 下界过松导致效率不可比 | 不同类别 η 极低但成熟实现也如此 | 分类别分析、标紧度、绝对性能主排序 |
| 全部收益来自修 correctness fallback | 一次补原生 dot 即巨大提升 | 将能力补全和设计方法收益分开，使用同功能强 baseline |
| 联合组只是试得更多 | 等候选数后优势消失 | 分清生产率收益与搜索质量收益，二者都可报告 |
| Agent advantage 是信息泄漏 | prompt/history 含最终答案或其他组结果 | 独立工作区、冻结任务、数据权限与轨迹审计 |
| cost model 误导优化 | 排名错误、未见配置过度自信 | OOD/区间、主动测量、报告负效应 |
| counter 归因不可靠 | stall 合计大于 ROI 或 source 不明 | 先修事件契约和守恒再谈优化 |
| RTL 太慢 | 几千点一次需数周 | 多保真筛选、代表实测、扩大快速后端；预测不能冒充实测 |
| FPGA 资源/时序失败 | RF 端口、FMA 或 SRAM 推断代价过高 | 提前小配置 P&R，模型标记不可实现候选 |
| 无法绝对击败 GPU | 有较高利用率但实际慢 | 如实报告效率和方法收益，不影响核心假设的可检验性 |
| PyTorch/LLM 抢占主线 | 大量时间用于生态适配 | 先小图与 decoder，完整 backend 作为扩展 |

**关键停止/转向 gate：**

- G1 后仍不能把一次 ISA 变更通过公共 AOT 跑到 RTL：先解决契约和集成，不启动大规模 Agent 搜索。
- G2 下界、计时或 counter 不自洽：暂停性能结论，保留 correctness 开发。
- G3 只有 compiler-only 有收益：诚实将结论改为编译器优化，或寻找有证据的新跨层瓶颈。
- G4 A11 不比 A00/A01 更好：不删除该实验；说明 model 的增量不足或成本不值得。
- FPGA 或完整 LLM 未完成：不能把模拟截图写成 FPGA 实测；重新界定 artifact 和论文主张。

## 17. R38 完成定义与本地审阅决策点

### 17.1 核心完成定义

R38 的研究主线完成需要同时具备：

- [ ] 真实最小 MLIR/lowering–ISA–microarch 闭环及跨层版本契约。
- [ ] 45 算子定义、几千实例 manifest、精度规则、冻结 split/权重。
- [ ] 可审计的 ArchBench 下界与性能评分，未测点和预测明确分离。
- [ ] 有 ROI/事件契约的 profiler，校准过且报告误差的 cost model。
- [ ] 至少三个有过程记录的跨层案例，并包含失败/无效探索。
- [ ] 单层、阶段式与短反馈联合迭代的公平预算比较。
- [ ] Agent model/profiler 2×2、token 与工具成本、失败和复现数据。
- [ ] 固定 canonical 资源约束下 AQ 的改进、跨规模/环境敏感性，以及真实部署验证；如果无改进则完整报告负结果并调整论文主张。
- [ ] 可复建制品、原始数据和分析脚本，结果从公共 AOT/runtime 得到。

用户完整 roadmap 还包括：

- [ ] 具体 FPGA 的板级证据与 NVIDIA 强基线对比。
- [ ] PyTorch inference 路径、decoder/LLM FPGA 演示与机器可读日志。

这两项应尽量在 R38 完成，但不能通过改写“完成”的定义来隐藏未完成工作。项目状态要分别列核心研究、实板、生态验证三条进度。

### 17.2 需要本地审阅确定的设计选择

以下是计划决策项，不是实施前要求重复授权：

1. 是否认可“硬件 DevOps 的三层可执行联合迭代”为论文第一主线。
2. ArchBench 架构主指标采用第 4.10 节 AQ，S/E 为部署与瓶颈伴随指标；审阅 canonical resource library、公共下界及 balanced/application 双轨。
3. 哪块 FPGA 与哪款 GPU 可获得，开发机器和实验机器的工具预算。
4. 首个跨层案例选 RF/layout，还是 async/window；由 pilot 瓶颈决定。
5. 论文投稿窗口与团队能力，据此收缩或扩展 20 周参考计划。
6. PyTorch 先图后端、后 PrivateUse1；LLM 模型规模由内存与精度能力决定。

不需要等这六项全部讨论完才能做 baseline 审计与最小闭环；但硬件采购、大规模付费 Agent 实验和最终协议冻结需要具体项目资源决策。

## 附录 A：评测协议自检用例

ArchBench 的公式也需要验证，以下是未来实现应具备的有意义测试：

1. 时间全部缩短 2×，固定硬件下 S 和 η 按定义变化。
2. 仅降低 advertised peak 不得改变 S，E 的变化有标注，不能被判为性能改善。
3. 注入冗余 DRAM 流量，不改变 semantic bound；效率应下降或保持，不应因 bound 被抬高而虚假改善。
4. 复制同一个 shape 或拆更多 stages，不改变原 operator 权重和语义总时间。
5. 删除一个慢点或失败点，完整覆盖 gate 应失败，而非总分上升。
6. cycles/ns/seconds 与 MAC/FLOP 单位转换一致；同一输入两种记录方式得同结果。
7. 比严格下界更快的记录触发审计，不能截断后通过。
8. 多 core 累计 active cycles 大于 makespan 时仍使用正确 ROI。
9. candidate 修改 reference weight/hash 后 evaluator 拒绝。
10. 无功率传感器时 energy 留空，不用 TDP 或零值补齐。

## 附录 B：单个跨层案例应交付的研究卡

```text
问题：哪个工作负载、哪个 profile、哪个资源出现损失？
证据：原始 counter/trace、计时、精度和资源报告。
假设：为何需要改变 compiler / ISA / microarchitecture 的组合？
变更：每层最小 patch、契约版本与兼容策略。
预测：cost model 的预期收益、范围、不确定性。
核验：功能、异步/并发、极端输入、O0/O1/O2。
消融：各层开关、合法软件 fallback、同资源对照。
结果：实际性能、资源、频率、误差、holdout 与负结果。
成本：提出假设到验证的时间、token、工具、人工干预。
结论：接受/回退/保留研究；可迁移的设计经验。
```

## 附录 C：引用与证据使用说明

本文引用均指向论文作者/机构、arXiv 或项目官方文档。第 2 节给出主要文献入口，其余章节在使用具体方法时再次链接。检索不是系统性穷尽综述；正式论文需完成近邻正文与 artifact 复核，不使用搜索摘要推断对方没有某能力。

动态 API、软件版本、GPU 规格和会议日期必须在实施/投稿时重新核对。本稿没有填写未经验证的录用状态、投稿 deadline、板卡资源或性能数字。

本文对项目已有能力的描述来自本地代码和已有核验记录；本次文档工作没有重新执行完整 R37 测试或上板验证。新的公式、工作包、实验和工期均为 R38 提案，不能当成已经实现或已得到的结果。
