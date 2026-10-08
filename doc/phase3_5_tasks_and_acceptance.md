# Phase 3.5：复用上游编译基础设施，交付可观测的 TNCA 编译优化

版本：1.0，2026-09-20。状态：**PLANNED / NOT ACCEPTED**。
检查基线：`03c88fad3f9482a22b1ded1aaf183b84709b6abc`。
前置：**假设进入本阶段时 [R3 v2](phase3_tasks_and_acceptance.md) 已验收**；接口约束：[R2 interfaces](r2_interfaces.md)；设计依据：[compiler overview](design/tnca_v1_compiler_draft.md)。

**目标：复用 Triton LinearLayout 和合适的 Triton/MLIR 优化基础设施，在不增加 SMEM、不修改 ISA 或硬件资源的前提下，让同一份程序产生更好的 TNCA 机器码，并在当前小规模 RTL 上测到可归因的周期下降。**

这是新增阶段，不修改 R1/R2/R3 的签收标准。完成名称为 `R35_COMPILER_ACCEPTED`，不等于完整 MLIR 后端、通用 Triton GEMM 或生产发布。

## 1. 当前起点与目标边界

本次源码检查发现：

- `python/tnca/aot.py` 的 `opt_level=0..2` 明确标为 reserved，当前不改变 emission。
- 主路径为 Triton TTIR → Python TNCA IR → codegen；没有直接使用 `LinearLayout` / `toLinearLayout`。
- C++ TNCA dialect、async interface 和 ownership verifier 已存在，但不能据此认为 MLIR 优化结果已进入产品机器码。
- 现有 `partition.py`、`pipeline.py`、`gemm.py`、`report.py` 可复用；其中 TNCA 专用调度策略不等于已复用上游 partition/pipeline pass。
- `compiler/pin/toolchain.json` 记录 Triton 源码与配套 LLVM pin，而 R1 验收另有 Triton wheel / distribution MLIR 20.1.2 路径。新增 C++ 依赖首先要解决版本边界。
- 撰写时 R3 v2 的同步 VRF 集成仍在推进；但本交付计划按用户明确前提，假设阶段启动时 R3 v2 已验收。采用其已签收的 `core-vrf-sync-v1` 和计数/排空接口，不把完成硬件作为本阶段任务。工作区中正在新增的 R3 v2 contract/inventory 不在本次修改范围。

### 1.1 必须交付

1. 直接复用锁定版本的 upstream LinearLayout 实现，作为 TNCA 布局推导的实际计算组件。
2. 产品编译路径真正执行 MLIR canonicalization/CSE，并消费其优化后的结果。
3. 在已支持的 TTIR 子集上复用 Triton AxisInfo，向 TNCA 传递经过验证的连续性、整除性等事实。
4. TNCA 专用布局选择、受 bank 约束的寄存器分配和有限范围的调度；全部保留异步资源安全。
5. 固定基线、优化模式、消融实验、数值校验、实际 RTL 周期与资源计数；至少两项真实 workload 达到第 6 节收益门槛，其中一项收益可归因于 LL 驱动优化。
6. 安装后的公开编译 API、artifact/cache/report/dump 和复现入口。

### 1.2 本阶段不增加

不增加软件可寻址 scratchpad，不把 NUCA 区域偷偷定义为 SMEM，不增加 bank/端口/MAC/role，不改变浮点数值合同。也不要求一次移植整个 TritonGPU pipeline、GPU warp specialization、SMEM swizzle lowering、通用 LICM、全循环软件流水或系统级自动调优。

编译器可修改数据的合法摆放和指令调度，但必须为实际 copy、搬运、等待付出成本。仅更改报告、删除必要 wait、修改基线或增加硬件都不算本阶段性能收益。

## 2. 复用哪些组件，哪些必须自己实现

| 组件 | 复用范围 | TNCA 自己承担的部分 | 本阶段等级 |
|---|---|---|---|
| Triton LinearLayout | 上游布局表示及组合/映射运算，通过薄 adapter 调用 | VRF/lane/fragment/ARF 的语义、合法转换和物理分配 | 必须进入产品路径 |
| MLIR canonicalizer + CSE | 直接调用 upstream pass，对受支持的纯值 IR 工作 | 导入/导出、effect 边界、数值与资源检查 | 必须有改变机器码的正例 |
| Triton AxisInfo | 锁定版本的 TTIR 分析，保守导出 proven facts | TNCA 地址/向量化选择、ABI 假设约束 | 必须有实际 lowering 决策与负例 |
| MLIR pass manager、verifier、诊断/dump | 管理上述受控 pass pipeline | 阶段记录、来源追踪、优化后 ownership 检查 | 必须 |
| TNCA allocation / local scheduling | 使用 LL 映射和已验证的依赖信息 | bank-aware allocation、窗口连续性、return/credit 成本 | 必须 |
| LLVM/MLIR liveness、dominance 或其他分析 | 仅在生产 IR 可直接应用时采用 | 与异步租约存活区间取并集 | 可选；不能拿未使用库算成果 |
| LICM、SCF 软件流水、Triton remove-layout-conversions/partition pass | 可做兼容性调研 | target policy、loop/effect/communication 合法性 | 延后，不作为本阶段关键依赖 |

上游 LL 将硬件位置映射到张量索引，支持重复值位置；它不是通用地址算术，也不是寄存器分配器。具体接口以仓库锁定版本的 [LinearLayout.h](https://github.com/triton-lang/triton/blob/c01b6774b1865984607d89d89d3a10833de92037/include/triton/Tools/LinearLayout.h) 为准。

AxisInfo 的分析结果需由 adapter 转为明确的 proven/unknown 信息，不能把 hint 当作对任意动态指针均成立的事实；参见锁定版本的 [AxisInfo.h](https://github.com/triton-lang/triton/blob/c01b6774b1865984607d89d89d3a10833de92037/include/triton/Analysis/AxisInfo.h)。

通用优化依赖正确的 operation 语义。实现需遵循 [MLIR canonicalization](https://mlir.llvm.org/docs/Canonicalization/)、[CSE 与 pass 文档](https://mlir.llvm.org/docs/Passes/#-cse) 和 [side effects/speculation 接口](https://mlir.llvm.org/docs/Rationale/SideEffectsAndSpeculation/)。本文件的具体选择、收益指标和测试矩阵是 TNCA 的工程合同，不是上游对 TNCA 的性能保证。

## 3. 无 SMEM 架构中的 LL 合同

### 3.1 分清三个层次

```text
logical tensor indices + valid mask
              ↑ LL mapping
virtual register / lane / packed element / MMA fragment position
              ↓ TNCA allocation + window constraints
physical VRF register → bank / row; ARF slot / row
```

- **Role Context 是独立指令流和资源所有者，不是 warp。** 不把 role 数塞进分布式 tensor layout 维度，更不能直接用 warp shuffle 语义解释跨 role 通信。
- VRF 的 16 lane 和 FP16 packed element 是不同坐标；它们的对应关系必须显式定义。MMA fragment 与 ARF 输出坐标也分别建模。
- 物理 bank 由实际分配后的 register ID 确定。不能只看虚拟 LL 就声称 bank 冲突消失；若 allocation 不是线性映射，使用显式分配表组合求值，不强行表示成 GF(2) 线性函数。
- 一般 stride、动态 base、地址进位、非二次幂边界仍由普通 index/address IR 处理。LL 可作用于有界 padded tile，合法 mask 单独保存，禁止把 padding 位置当作可访问内存。
- broadcast/复制可能不是双射；不能无条件求普通逆矩阵。需显式选择等价来源，并验证所有有效输出的覆盖和一致性。

### 3.2 转换合法性与成本

每个布局转换必须归入以下类别并有 lowering 记录：

1. **纯编译期重命名/视图**：不用移动值，仅当所有消费者所见元素与打包顺序一致时允许，额外数据指令为 0。
2. **现有指令可实现的转换**：寄存器 copy、合法 lane 操作或既有 tensor transfer；列出实际指令、临时 VRF、bytes 和等待。
3. **当前 ISA 不支持**：返回有位置的诊断或保留合法原布局。禁止无声生成 SMEM、虚构 lane exchange 指令或调用 host 重排输出。

本阶段先覆盖 identity、合法 reshape/transpose 组合、broadcast、masked tail、FP16 MMA A/B fragment 和 ARF result 对应。LL 可以描述的置换不自动进入受支持范围。对不支持的跨 lane 置换必须有拒绝用例。

### 3.3 布局和 bank 优化策略

候选数每个 kernel ≤8；同一源输入只选一个正式 O2 计划。候选不触发 RTL 调优。评分至少包含：新增转换指令/bytes、实际 bank 请求冲突、live register 峰值、window 连续性/对齐、barrier/credit 占用和关键依赖长度。

保留合法 baseline 候选作为回退。评分未知项不能填 0；预测改善没有证据时保守选择 baseline 并记录理由。不得按 benchmark 名称、输出数据或实测该 case 的周期查表选择计划。

共享 window 的固定连续寄存器要求、MMA fragment 次序、跨 role alias 和 async lease 都是硬约束。局部 bank 分配不能破坏这些条件。没有符合约束的候选就返回原计划，不靠增加资源容纳它。

## 4. 产品编译路径和依赖组织

### 4.1 先接通窄路径，不等待 R2 全面迁移

允许分两段接入：

```text
Triton TTIR → upstream analysis/canonicalization → supported TNCA IR
TNCA IR pure regions → small MLIR bridge → canonicalize/CSE → TNCA IR
                      ↓
       LL layout planning → allocation/scheduling → safety checks → emission
```

TNCA IR 输入也必须能到达 LL 与适用的纯值优化。无需将整个低层控制流在本阶段迁往 MLIR，但桥接后的结果必须替换实际 emission 输入；不能仅对副本运行 pass 后丢弃。

纯值 region 有明确输入/输出与 effect 边界。load/store、DMA、MMA、barrier、window ownership、trap、可能 fault 的操作默认不可跨越，不得标成 Pure 来获得 CSE。只先允许不改变数值语义的常量折叠、整数地址表达式简化、重复纯表达式消除和死纯值消除。跨内存 effect 的 load CSE、推测访问、浮点重结合或新 FMA contraction 不在默认优化内。

优化前检查输入合法性，优化后检查语义/effects，完成物理分配后再次执行现有 ownership/footprint/resource 检查。降低寄存器占用时，liveness 必须包括 DMA/MMA 在 issue 之后仍持有的异步租约。

### 4.2 upstream 依赖和版本

- 首选独立 `tnca-layout-opt` 编译期工具或等价隔离 adapter，直接构建并调用 pinned upstream LL/分析；只传版本化 MLIR 文本或数据协议，不跨 ABI 传 MLIR 对象。
- 工具自身使用一套匹配的 Triton/LLVM；不要在一个进程内混链 distribution MLIR 20 与另一套 pinned LLVM 库。若改为进程内 binding，必须证明工具链一致并冻结支持矩阵。
- G0 记录准确 upstream commit、最小源码/库依赖闭包、补丁、许可证/NOTICE、build 命令与 hash。不自动升级 R1/R2 工具链，不依赖 `main` 或未导出的私有 Python 符号。
- TTIR frontend 与优化工具若版本不同，必须有明确文本 dialect 兼容验证；不兼容时锁定同一配套 frontend/compiler 环境或给出受控诊断，不能假设 wheel 内部 MLIR 对象可直接互通。
- 编译器可依赖 upstream 工具；runtime 加载 artifact 不得 import Triton/LLVM/LL，也不需要其安装。
- 只有上游原实现真实参与布局计算才可宣称“复用 LL”。独立 Python oracle 只用于验证，不可成为产品中的悄悄替代实现。

### 4.3 优化级别、缓存和报告

| 模式 | 定义 |
|---|---|
| B0 | G0 冻结的现有产品编译行为，含其已有优化；作为公平基线，不故意删除现有优化 |
| O0 | 新 driver 的兼容路径；在冻结用例上机器码与 B0 一致，除版本化 metadata 外不得人为变差 |
| O1 | upstream canonicalization/CSE、经过证明的 AxisInfo lowering；保留 legacy 布局/分配策略 |
| O2 | O1 + LL 驱动布局消除/选择 + bank-aware allocation + 保守局部调度 |
| O2-no-LL | 消融：保留 O1 和可独立工作的调度，关闭 LL 驱动布局/分配决策，使用 legacy 合法计划；不能删计算 |

模式是待实现合同，当前 `opt_level` 尚未具备这些行为。显式请求 O1/O2 而依赖缺失时必须失败；某个优化不适用时允许在该模式内保留原代码，报告 not-applicable 原因。

cache key 必须包含优化级别、pass pipeline 版本、LL upstream revision、target/实现 fingerprint、会影响代码的假设；动态 buffer 地址和运行期 alpha 不进入编译 key。诊断路径不影响 code hash。

报告输出阶段 IR、layout bases/有效 mask、转换分类、实际物理 bank 分配、live peak/异步占用、copy/指令数、选中与拒绝计划、pass remarks、编译耗时。模型预测、静态指令计数、RTL 实测 cycle 三者分列。

## 5. 与 R3 的衔接及实施顺序

- **启动基线**：R3 v2 已签收的 `core-vrf-sync-v1`、端口/队列合同、计数器、runtime 和排空语义。G0 核验签收证据并冻结硬件资产，不重新承担 R3 的实施工作。
- **实施顺序**：LL adapter 与工具链隔离 → 纯值优化/AxisInfo → 布局与 bank-aware allocation → 安全差分 → 固定硬件上的收益门禁。组件工作可并行，最终验收没有“硬件未就绪”的豁免。
- 所有主收益和 LL 消融均在已验收的 banked profile 上测量。旧 `core-sim` 只作兼容回归，不能代替实际有限端口测量。
- 性能配对必须使用相同的已验收 RTL 二进制/硬件 fingerprint；B0 与 O2 只改变编译器行为。硬件若发生变化，必须先补齐对应 R3 验证、建立新基线，再重做正式配对；不能将硬件收益归给编译器。
- 不等待 R2 四 tile runtime、全部 MLIR 控制流或双环境生产发布。本阶段只使用已验证单核能力，并保留真实 source_kind。

## 6. 小规模性能实验与硬验收门槛

### 6.1 固定六项 workload

G0 从现有程序/可支持算子构造以下源，冻结 source hash、输入种子 42、内存映射、模式、mask、源路径和 oracle。不得手写差的 B0 汇编或添加死 copy 制造胜利；优化前先证明源表达的是实际计算需求。

| ID | 固定 workload | 输入与尺寸 | 主要观察 |
|---|---|---|---|
| L01 | masked AXPY | 真实 Triton；N=17、BLOCK=16、grid=2；fast | dynamic scalar、尾块、地址开销 |
| L02 | 双输出共享中间值的逐元素计算 | 真实 Triton；N=16、grid=1；两输出均校验；fast | CSE/地址与 mask 复用，不能删一个输出 |
| L03 | 合法 tensor view/布局组合后的逐元素消费 | TNCA IR；4×4 FP32、grid=1；fast | LL 转换消除和实际 bank 分配；独立枚举元素映射 |
| L04 | masked reduction | 既有受支持输入路径，G0 固定；一行16存储元素、15有效；fast | 死纯值消除、live range、归约顺序不变 |
| L05 | FP16 GEMM | TNCA IR；16×16×16、单 role；fast | TensorMap→VRF→MMA fragment 布局与转换成本 |
| L06 | FP16 GEMM pipeline | TNCA IR；16×16×32、两 role/2-slot；固定 slow/reorder | 实际 bank conflict、窗口生命周期、调度 |

L02 的公式在 G0 固定，例如 `t=x*alpha; y=t+b; z=t-b`，共享需求从源获得，不通过插入重复汇编人为扩大收益。L03 的具体合法 view 组合须先与当前 ISA 对齐；若纯 rename 已经零成本，仍保留为不退化对照，不能强迫基线搬运。L03/L06 为预定消融对象，无论有没有赢都必须报告。

### 6.2 唯一执行矩阵

| 分组 | 次数 | 说明 |
|---|---:|---|
| 主比较 | 12 | 六项 × B0/O2，全部在同一 banked candidate 上 |
| 消融 | 4 | L03/L06 各运行 O1、O2-no-LL；复用主比较 O2 |
| 旧 profile 兼容回归 | 4 | L01/L02 × B0/O1，在相同 `core-sim` 二进制上；仅保留兼容性，不作为主收益门槛 |
| 安装后动态参数 | 3 | L01 O2，同 artifact、同 session，三组 alpha / buffer；编译一次 |
| 合计 | **23 次 kernel launch** | grid 中的 program 全部计入 cycles；无隐藏循环重复 |

另允许最多 24 次组件/负例 RTL invocation，全部登记。其余布局、IR、变异和随机差分在 host 上执行，不把三个 seed 乘到整套 RTL。新增 RTL 总计 ≤47 次、累计 ≤2,400 秒/环境、≤1,500,000 cycles；core 单次 ≤180 秒/50,000 cycles，组件单次 ≤60 秒/10,000 cycles。默认无波形，失败 trace ≤256 cycles/10 MiB。继承回归独立计账，不能通过预算删除。

RTL 仿真很慢；本阶段不增加 GEMM K64/K128、大矩阵、四 tile 或大型 autotune。软件候选搜索每 kernel ≤8，总默认 sweep ≤48，≤60 秒且不启动 RTL。

### 6.3 性能通过标准

定义 `C` 为**launch accepted 到全部结果 store 提交且资源 drain 完成**的 device cycles；ROI 另报，仅作解释，不替代总 C。memory/reset/缓存初态、输入、输出、硬件、grid、资源容量完全相同。不同计划可有不同实际资源占用和请求时序，必须报告。

主比较必须同时满足：

1. **六项全部正确**：B0、O2 与独立 oracle 一致，tail/canary/错误/排空检查通过。GEMM 和 reduction 保留既有累加/归约顺序，禁止 fast-math 换精度。
2. **至少两个不同 workload 有实际收益**：各自 `C_B0 - C_O2 ≥ max(8 cycles, ceil(0.05×C_B0))`；其中至少一个不是纯布局微例 L03。
3. **至少一项有 LL 因果证据**：在预定 L03 或 L06，`C_noLL - C_O2 ≥ max(4 cycles, ceil(0.03×C_noLL))`；并有 copy 指令减少或 accepted 请求/冲突等待变化的匹配证据。两边其余开关相同，不能把 O1 的收益算给 LL。
4. **无明显退化**：每项 `C_O2 - C_B0 ≤ max(4 cycles, ceil(0.02×C_B0))`；六项总 cycles 不增加。慢项全部展示，不能只报 geometric mean。
5. **收益来源可解释**：每个赢的 case 至少关联一项真实变化——动态指令、copy bytes、bank conflict stall、依赖等待或外部请求；静态指令变少但 C 未降，不算第 2 条收益。

5%/3% 是待达到的阶段目标，不是已测加速或收益保证。G0 先评估现有代码是否存在机会；达不到时保留测量与失败，标 `OPTIMIZATION_DELIVERED / PERFORMANCE_NOT_MET`，继续修编译器或提出下一版有依据的范围调整。不能事后改 workload、增大尺寸、降低阈值或故意削弱 B0 将失败转绿。

本阶段不要求两倍加速。短 kernel 上一次真实端口等待或几条动态指令的减少就有价值，但必须达到上述可复验门槛。

### 6.4 编译成本

工具链和 simulator 构建另记。warm helper、cache miss 的六项 O2 编译各 ≤30 秒；中位数不超过 `max(2×O1 中位数, 1秒)`；首次 helper 启动单独记录，不混入硬件 cycles。每项重复三次仅测编译，不运行 RTL；取中位数并保留全部记录。cache hit 必须通过调用计数证明没有重新运行 frontend/LL/优化器。

## 7. 全部工作包与验收标准

全部 ID 前缀 `R35-`；每个包负责角色在实施时落实到人。当前全部 **NOT VERIFIED**。每项的全部子条件必跑，无百分比验收。

### R35-01 — 依赖、基线、机会和 inventory（维护者 / 编译器）

| ID | 必须通过 |
|---|---|
| R35-BASE-01 | 锁定 B0 源码/版本、六个 workload、23-run 矩阵、硬件 hash、预算和工具链；候选实现源码未修改 benchmark oracle；清单由独立 checker 验证完整 |
| R35-BASE-02 | 建立每种 pass 的适用 IR、上游真实符号/库、输入输出和 effect 边界表；列出预期收益机制与当前阻塞；未知吞吐不记零 |
| R35-BASE-03 | baseline 可在 candidate 硬件执行；O0 与 B0 machine code 一致；硬件变更/漏消融/放大尺寸/混 profile 在启动仿真前拒绝 |

### R35-02 — 上游组件 adapter 与可安装构建（编译器 / 构建）

| ID | 必须通过 |
|---|---|
| R35-UP-01 | 直接执行 pinned upstream LL/MLIR/AxisInfo，实现来源可追溯；最小构建和安装自测通过，不依赖开发者源码绝对路径 |
| R35-UP-02 | Triton/LLVM 版本错配、bridge 协议错版、helper 缺失/异常各报稳定错误；O2 不悄悄退回无 LL；无跨版本 MLIR ABI 混链 |
| R35-UP-03 | bridge 输出确实进入 emission：受控变异其布局/纯值结果，指定 checker 或代码差分必须检出；只调用但丢弃输出不算通过 |

### R35-03 — LL 语义和合法 lowering（布局 / 验证）

| ID | 必须通过 |
|---|---|
| R35-LL-01 | 第 3 节布局逐元素与独立枚举 oracle 一致；固定 seed {1,7,42}、每 seed ≤64 个 host case，覆盖二次幂 tile、有效 tail、broadcast、packed f16 和 A/B/ARF 映射 |
| R35-LL-02 | composition/转换结果覆盖全部有效元素，无重复写/缺输出；非双射处理、动态 stride/mask 与不可支持 lane 置换有明确正负例；role 不作为 warp/layout 维度 |
| R35-LL-03 | 每个转换有 rename/真实指令/拒绝分类，实际 copy/bytes/live 临时值可从发射代码核对；禁止 SMEM allocation、伪造 shuffle 或 host 输出修正 |

### R35-04 — 纯值优化与 AxisInfo（MLIR / 前端）

| ID | 必须通过 |
|---|---|
| R35-OPT-01 | canonicalization/CSE 在产品 O1/O2 路径真实执行，至少一个非测试专用源的重复纯值/地址运算消失并改变 machine code；pass 前后 dump 与 remarks 对得上 |
| R35-OPT-02 | AxisInfo 至少驱动一个合法地址/向量访问 lowering 选择；alignment/contiguity 无法证明时走保守计划；动态未对齐、tail、alias 输入不会沿用错误假设 |
| R35-OPT-03 | CSE/DCE 不删除或越过 store/atomic/DMA/barrier/trap/潜在 fault；浮点 NaN/Inf/signed-zero/subnormal 和既有 FP 合同回归通过；不启用额外 reassociation/contraction |

### R35-05 — bank-aware 分配与有限调度（后端）

| ID | 必须通过 |
|---|---|
| R35-ALLOC-01 | cost 使用最终 physical register→bank 映射；同/异 bank 定向例符合 R3 接口；纯 rename 无额外 move，实际 move 全部计费 |
| R35-ALLOC-02 | window 连续/对齐、alias、ARF 顺序、register peak、barrier/credit 都在原预算内；异步 lease 延长 liveness；压力超限有诊断/合法原计划，无隐藏 spill |
| R35-SCHED-01 | 调度保留 def-use、memory/effect、控制、accumulator-order 和 async-resource 边；仅重排已证明独立的局部指令；无法证明或收益未知时保留合法 baseline |

### R35-06 — 安全差分和检查器反例（验证）

| ID | 必须通过 |
|---|---|
| R35-SAFE-01 | 所有受影响 R1 OWN/FP、前端拒绝、R2 ABI/cache/artifact 及 R3 单核安全 case 在新编译器上通过；各优化级别都调用最终物理 ownership verifier |
| R35-SAFE-02 | host 差分验证每种 pass 的正例及边界负例；动态参数不被错误 constexpr 化；重命名函数/寄存器/符号不改变优化选择或结果 |
| R35-MUT-01 | 下述所有变异都被指定 detector 检出：错 LL basis→LL-01；去 tail mask→LL-02；把 effect op 当 Pure→OPT-03；提前释放异步值→SAFE-01；忽略 bank→ALLOC-01；优化级别漏 cache key→PKG-02；模型周期冒充实测→PERF-01；伪造 PASS/漏慢 case→EV-01 |

变异用隔离副本，clean 基线必须通过；不能用无关 build failure 算检出。需要 RTL 的变异占组件额度，不复制整个性能矩阵。

### R35-07 — 真实收益与消融（性能 / 验证）

| ID | 必须通过 |
|---|---|
| R35-PERF-01 | 23 次 launch 按清单运行；raw cycle、数值、canary、drain、硬件/代码 hash、实际 elapsed 齐全；R3 banked 前置有效；没有用 host wall 或模型预测代替设备 C |
| R35-PERF-02 | 第 6.3 节五条全部满足，六项 B0/O2 完整表与 L03/L06 消融公开；至少两项实测赢、至少一项 LL 归因成立；失败项不可删 |
| R35-PERF-03 | 第 6.2/6.4 节尺寸、次数、cycles、wall、候选搜索和编译成本上限通过；低预算注入可证明 watchdog/进程清理生效 |

### R35-08 — 产品 API、cache、报告和 dump（产品编译器）

| ID | 必须通过 |
|---|---|
| R35-PKG-01 | 非 editable 安装、任意 cwd、仓库不在 sys.path；公开 API 的 O0/O1/O2 正确执行；L01 O2 三次动态发射结果正确，同 code hash/同 device 进程、额外 compiler 调用 0 |
| R35-PKG-02 | opt level、pass/LL revision、硬件 profile、代码假设变化各使 key 改变；相同输入重编确定；损坏 cache 拒绝；runtime-only 安装可用 artifact，不加载编译器依赖 |
| R35-OBS-01 | dump 包含输入/优化后 IR、LL 决策、实际分配和最终汇编；report 与真实 emission 对应；debug/dump 不改变代码；未应用优化显示理由，GEMM source_kind 不伪标 Triton |

### R35-09 — 回归、复现和证据签收（CI / 维护者）

| ID | 必须通过 |
|---|---|
| R35-REG-01 | G0 冻结的影响范围回归全 PASS；共享编译器变动涉及旧 core-4r/system 路径时补跑对应既有 case，不把新增性能矩阵扩成整机跑分；原阶段证据不自动替代候选回归 |
| R35-EV-01 | verifier 按冻结 inventory 从原始数值/计数/退出码重算收益、退化、预算和判决；拒绝混源码/硬件/工具链、改 cycle、漏消融、篡改日志和仅修改 summary PASS |
| R35-REL-01 | 主锁定 Linux 环境完成全门禁；第二独立 Linux 构建重现 LL/IR/安全测试及 L03/L06 的 B0/O2/O2-no-LL，确定性代码/数值/cycles 一致；签收表、原始证据、已知边界齐全 |

第二环境不共享构建产物，可共享只读下载缓存；不要求完整复制 23 次矩阵，也不能据此宣称原 R1/R2 的双环境生产发布完成。主环境的其他赢家不在第二环境新增矩阵，但必须保留完整可重放证据。

## 8. 交付文件、入口和验收顺序

建议落地位置（均为待实现，不宣称已经存在）：

- `compiler/.../layout_adapter`：上游 LL 隔离层；`analysis_adapter`：AxisInfo/proven facts。
- `compiler/.../optimization`：纯值 bridge、pass pipeline、TNCA 布局与分配策略。
- `spec/r35_{toolchain,layouts,workloads,required_cases}.json`：版本化合同和清单。
- `tools/r35/`：runner、差分/变异、性能计算、独立 verifier。
- `doc/r35_status.md`：按本文件 27 条标准逐项记录，初始 NOT VERIFIED。

待实现入口：

```sh
make r35-doctor
make r35-check-plan
make r35-layout-tests
make r35-opt-tests
make r35-safety-tests
make r35-perf
make r35-package-tests
make r35-acceptance
make r35-verify-evidence EVIDENCE=<run-directory>
```

运行 registry 统一去重；缺工具、缺前置、未实现 detector、fail/skip/xfail/timeout 都不能签收。研发重跑保留旧失败并产生新 run ID，禁止覆盖记录。

证据至少包含：

```text
contracts/     本文件hash、B0来源、工具链、硬件/布局/数值/资源合同
inventory/     主23-run、组件、消融、继承回归、第二环境子集
build/         compiler/upstream/helper/RTL/wheel hashes及构建命令
ir/            各阶段输入输出、LL bases、有效mask、分配、remarks
code/          B0/O0/O1/O2/no-LL代码、artifact、静态资源比较
runs/          原始命令、数值oracle、退出码、计数、cycles、elapsed
performance/   逐case收益/退化、LL消融、计数归因、编译成本
mutations/     clean与变异结果、对应detector、源码未改证明
inherited/     当前候选的影响回归、R3前置验证引用
acceptance.json  contract_id=r35-compiler；逐项判决、预算、环境
```

| 关口 | 完成条件 | 不满足时 |
|---|---|---|
| A：依赖与 LL 语义 | R35-01/02/03 | 不接自动布局优化 |
| B：真正优化且安全 | R35-04/05/06 | 不跑正式收益门禁；先修语义/ownership |
| C：收益可观测 | R35-07，在已验收 R3 硬件上完成 | 未运行标 NOT VERIFIED；未达收益标 PERFORMANCE_NOT_MET，不签收 |
| D：产品与证据 | R35-08/09 | 不宣称可交付 |

**27 条标准及其全部子条件、两项实测收益、LL 因果收益、不退化约束、预算与复现同时通过，才输出 `R35_COMPILER_ACCEPTED`。**

验收后的直接工程衔接为 [R3.6：多后端工程整合与原型分支交接准备](phase3_6_tasks_and_acceptance.md)。R3.6 整合公共产品链和后端接口，不指定 FPGA，也不承担上板；具体平台实现由其他人在独立分支开展。

编译与硬件能力的后续扩展：证明窄路径有效后，再扩大 MLIR 主路径覆盖、循环优化和软件流水；结合 R3B 的同步 ARF 做更完整布局/成本模型；等 R2 的系统 runtime 成立后考虑跨 tile。当前阶段不借这些远期目标延迟单核编译收益的交付。
