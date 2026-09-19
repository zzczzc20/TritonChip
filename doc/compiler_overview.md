我会先核实 Triton 主线中 Warp Specialization 的实际状态、相关 IR 和 pass 的依赖边界，再据此规划 TNCA 方言与实现路线。重点会放在“哪些代码可以直接复用、哪些需要抽出目标接口”，以及四个 RCTX 的流水如何避免照搬 GPU 的 warp/shared-memory 假设。**现在就应该并行启动编译器，而且第一阶段的目标应是“让真实 Triton kernel 持续检验硬件接口”，而不是先完成一个覆盖所有算子的自动优化后端。**

我的总体建议是：

> **保留 Triton 的前端、TTIR、可复用的布局与循环变换基础设施；新增一个很薄的 TNCA 资源／异步执行层；将 Warp Specialization 拆成“分区算法、通信协议、物理上下文映射”三部分，分别复用和替换。**

尤其不要走两个极端：

- 从 TTIR 开始，把所有 tensor、算术、控制流重新定义成 `tnca.*`；
- 为了复用现有 WS，先生成 NVIDIA shared-memory/TMEM 程序，再把它强行解释成 RF 程序。

下面分为：**上游现状、IR 设计、Context Specialization、布局与资源分配、代码生成、实施路线和硬件反馈点。**

关于你的 RTL，我以下只依据你贴出的 ISA 和 Status，不把文档中的“complete”当作已经独立验证过的实现事实。

---

# 一、先回答检索结果：WS 已经在主线，而且不止一条实现路径

## 1.1 已经不是“某个尚未合并的实验分支”

以 **2026 年 9 月 19 日**为检索口径：

- Triton 官方 release 页面目前标记的最新版本是 **3.8.0**；
- 至少在 **3.4.0** 的发布说明中，已经明确列出了 Automatic Warp Specialization；
- 当前 `main` 中有完整的 WS IR、自动分区、跨分区通信构造和目标相关 lowering。([github.com](https://github.com/triton-lang/triton/releases/tag/v3.8.0))

但必须区分以下几件事：

| 层次 | 当前上游已有内容 | 对 TNCA 的意义 |
|---|---|---|
| Triton 高层入口 | `tl.range(..., warp_specialize=True)` | 可以保留作为自动专门化请求 |
| 显式编程入口 | Gluon `warp_specialize(...)` | 可参考其 region/capture 模型，但不能承诺执行任意 NVIDIA Gluon 程序 |
| 通用一些的 IR | `ttg.warp_specialize`、partition regions | 可复用结构设计，不代表其物理执行语义直接适合 RCTX |
| 自动 WS 中端 | partition scheduling、partition loops、Aref 通信 | 最值得抽取复用的部分 |
| GPU 目标后端 | shared memory、TMEM、GPU barrier、warp allocation | TNCA 必须替换 |

`ttg.warp_specialize` 的语义包括并发执行多个 region、显式捕获 worker 参数以及末尾汇合；它仍然以不同大小的 warp group 为执行域。Gluon 的显式接口也包含 worker warp 数量和寄存器预算，而不是一个完全目标无关的任务系统。([triton-lang.org](https://triton-lang.org/main/dialects/TritonGPUOps.html))

## 1.2 当前主线同时存在 Hopper 和 Blackwell 相关自动 WS 路径

这点值得特别强调。

`tl.range` 的文档仍有“只支持 Blackwell、主要面向简单 matmul loop”的说明；但当前 NVIDIA `compiler.py` 已经：

- 在相应架构分支调用 `add_hopper_warpspec`；
- 在 Blackwell 及之后的分支调用另一套自动 WS pipeline。

因此，**不能只根据那一句 API 文档得出“主线没有 Hopper WS”**。应以固定版本的实际 pipeline 和测试为准。([triton-lang.org](https://triton-lang.org/main/python-api/generated/triton.language.range.html))

Hopper 路径本身还有明确限制，例如所检索实现对 warp 数和带 `else` 的控制流存在限制。它不是一个任意控制流都能处理的通用分区器。([raw.githubusercontent.com](https://raw.githubusercontent.com/triton-lang/triton/main/third_party/nvidia/hopper/lib/Transforms/WarpSpecialization.cpp))

## 1.3 对 TNCA 最重要的上游文件

建议第一轮重点阅读这些模块：

```text
lib/Dialect/TritonGPU/Transforms/WarpSpecialization/
    AutomaticWarpSpecialization.cpp
    PartitionScheduling.cpp
    PartitionLoops.cpp
    Partition.cpp

include/triton/Dialect/TritonGPU/Transforms/
    Partition.h
    PartitionSchedulingUtility.h
    PipeliningUtility.h

third_party/nvidia/include/Dialect/NVWS/IR/
    NVWSOps.td
    NVWSTypes.td

third_party/nvidia/.../NVWS/.../
    InsertAref
    InsertTmemAref
    LowerAref
```

当前自动 WS pipeline 确实连接了：

```text
分区调度
→ Aref 插入
→ Aref lowering
→ 分区循环构造
→ WarpGroup lowering
→ 后续循环调度
```

但其组织者直接依赖 NVIDIA NVWS，不能因为文件位于 `TritonGPU` 目录，就认为整个 pass 已经目标无关。([raw.githubusercontent.com](https://raw.githubusercontent.com/triton-lang/triton/main/lib/Dialect/TritonGPU/Transforms/WarpSpecialization/AutomaticWarpSpecialization.cpp))

更具体地说：

- `PartitionScheduling` 中仍有 TMEM、CLC、MMA 类别及合并规则；
- `PartitionLoops` 会为部分跨 partition tensor 结果创建 shared-memory 临时对象；
- `NVWS Aref` 当前操作实际使用 `TTG_MemDescType`，lowering 面向 GPU barrier。

**这些就是应当抽出 target interface 的位置，而不是可以原封不动调用的边界。**([raw.githubusercontent.com](https://raw.githubusercontent.com/triton-lang/triton/main/lib/Dialect/TritonGPU/Transforms/WarpSpecialization/PartitionScheduling.cpp))

---

# 二、总体路线：不是“TTIR 全部翻译成 TNCA”，而是混合 IR

我建议采用以下结构：

```text
Triton Python
    │
    ▼
上游 TTIR
    tt + arith + math + scf
    │
    │ 复用规范化、常量传播、访问分析等
    ▼
布局化 Tensor IR
    tt + 受限 ttg layout/convert_layout + arith + scf
    │
    │ TNCA fragment selection
    │ Context partition / channel planning
    ▼
TNCA Execution IR
    原有算术与控制流
    + tnca 的 RF、accumulator、channel、async operations
    │
    │ 资源分配、协议实体化、role outlining
    ▼
TNCA Low-level IR
    固定 window / accumulator / barrier 分配
    每个 RCTX 的独立 CFG
    │
    ▼
机器指令与寄存器分配
    │
    ▼
现有 assembler / 后续 LLVM TNCA target
    │
    ▼
Binary + Kernel Manifest
```

这里最关键的设计是：

> **TNCA dialect 只表达上游 IR 无法准确表达的事情，不重新发明上游已经表达清楚的事情。**

例如：

| 内容 | 建议表示 |
|---|---|
| 整数、浮点普通算术 | `arith` / `math` |
| 循环和条件控制流 | `scf`，后期降到 `cf` |
| tensor shape、broadcast、reduce、dot | 前期继续保留 `tt` |
| tensor layout | 尽量复用 `ttg` 的布局表示与 LinearLayout |
| RF window 所有权 | `tnca` |
| 异步读取源的生命周期 | `tnca` |
| accumulator 特殊访问 | `tnca` |
| role 间通信和同步 | `tnca` |
| 物理指令编码 | 最末端，不放进高层 TNCA IR |

不要过早生成：

```text
tnca.add
tnca.mul
tnca.for
tnca.if
tnca.reshape
tnca.broadcast
```

否则为了引入一个新硬件目标，会失去大量现成优化和分析能力。

---

# 三、具体复用清单：哪些直接用，哪些抽接口，哪些必须新写

## 3.1 第一类：直接复用，尽量不修改

### 前端和编译框架

- Triton Python AST 到 TTIR；
- constexpr specialization；
- JIT 编译缓存；
- kernel metadata 传递；
- backend discovery；
- Python 编译 stages 框架；
- MLIR parser、printer、ODS、pass manager、diagnostics。

当前 `BaseBackend` 支持后端自定义 stage 序列，而 backend discovery 支持 out-of-tree entry points。因此 TNCA 不需要伪装成 CUDA backend，也不需要修改 NVIDIA `compiler.py` 才能接入。([raw.githubusercontent.com](https://raw.githubusercontent.com/triton-lang/triton/main/python/triton/backends/compiler.py))

### 通用分析与变换

优先复用：

- canonicalization、CSE、SCCP；
- dominance、post-dominance、slice analysis；
- SCF structural transformations；
- Triton AxisInfo；
- LinearLayout 的组合、反演和映射工具。

AxisInfo 已经提供 contiguity、divisibility、constancy 等信息，适合成为 LSU/Tensor DMA 选择的基础；LinearLayout 则适合承担寄存器布局变换的数学表示。([raw.githubusercontent.com](https://raw.githubusercontent.com/triton-lang/triton/main/include/triton/Analysis/AxisInfo.h))

但要注意：

> AxisInfo 不是完整的别名分析，更不是“这个 load 可以提前执行”的证明。

它能帮助判断访问形状，不能单独证明跨 store 重排的合法性。

## 3.2 第二类：复用算法或主体，替换目标策略

| 模块 | 保留什么 | TNCA 替换什么 |
|---|---|---|
| Coalescing | 连续性分析、访问分组、布局候选 | line/word 事务成本、16-lane LSU 能力 |
| Layout propagation | 编码传播、冗余转换消除、部分 rematerialization | 转换成本、合法布局集合 |
| Partition scheduling | 数据流图、依赖传播、分区基础设施 | 操作分类、合并规则、RCTX 预算 |
| Partition loops | SCF 克隆、loop-carried value 重建 | role 容器、跨角色结果传递 |
| Aref 插入 | producer/consumer 识别、最后使用位置 | RF/ARF channel 与 release 事件 |
| Software pipelining | prologue/kernel/epilogue 结构改写 | stage 调度、异步依赖、RF 缓冲 |
| 并发检查 | 可见性和未完成访问的分析思想 | RF footprint、TNCA epoch/completion 协议 |

例如，上游 `Coalesce` 对 descriptor load/store 的布局选择仍考虑 shared-memory 后续访问；`RemoveLayoutConversions` 的部分成本判断也具有 GPU 假设。因此这些 pass 应进入**白名单复用与测试**，不能整体无条件启用。([raw.githubusercontent.com](https://raw.githubusercontent.com/triton-lang/triton/main/lib/Dialect/TritonGPU/Transforms/Coalesce.cpp))

## 3.3 第三类：TNCA 的真正新增部分

这些无法靠改几个 opcode 名称解决：

1. RF/ARF 跨 RCTX 的物理资源规划；
2. 异步源生命周期；
3. ready/free channel；
4. command completion 到 mbarrier generation 的映射；
5. TNCA fragment 选择与布局转换；
6. 基于真实 RTL 能力的合法化；
7. TNCA ISA emission；
8. kernel launch manifest 与仿真 runtime。

**新增代码应该主要集中在这里，而不是集中在重写 tensor 算术和循环优化。**

---

# 四、布局 IR：优先兼容上游，但不把 warp 当作 RCTX

## 4.1 一个可行的起点：单向量执行域的 TTGIR 子集

建议第一版布局域按：

```text
lane = 16
warp = 1 个逻辑布局组
CTA = 1 个 program 布局域
```

来尝试复用：

- `BlockedEncoding`；
- `SliceEncoding`；
- `LinearEncoding`；
- `ttg.convert_layout`。

这里的 `warp=1` **仅仅是复用布局工具时的表示适配**，不是说 TNCA 具有 GPU warp，也不是说只有一个 RCTX。

现有 TTIR→TTGIR conversion 接收 `numWarps`、`threadsPerWarp`、`numCTAs` 参数，给了这种适配一个入口；但这不证明所有后续 pass 都支持 16 lanes。必须逐个建立测试。([raw.githubusercontent.com](https://raw.githubusercontent.com/triton-lang/triton/main/lib/Conversion/TritonToTritonGPU/TritonToTritonGPUPass.cpp))

建议规则：

> **一个 RCTX 控制的 tensor 分布，和多个 RCTX 之间的角色分工，是两条互相独立的轴。**

不要把：

```text
num_warps = 4
```

解释为：

```text
RCTX = 4
```

## 4.2 不一开始发明完整的 `#tnca.tensor_layout`

建议先：

- 普通 tensor 保留上游 encoding；
- RF window 和 MMA fragment 引入 TNCA 专用 layout 属性；
- 到必须表达 packed half、固定 fragment 行序、RF bank 约束时，再进入 TNCA layout。

这样可以尽量延长上游布局优化的可用区间。

需要注意，`ttg.linear` 对输入维度和映射形态有约束，并不等于任意 LinearLayout 都能直接作为该 encoding。**可以直接复用底层 LinearLayout 数学工具，不必强迫所有 TNCA 物理布局都伪装成 `ttg.linear`。**([raw.githubusercontent.com](https://raw.githubusercontent.com/triton-lang/triton/main/include/triton/Dialect/TritonGPU/IR/TritonGPUAttrDefs.td))

## 4.3 建立明确的 layout legality gate

在进入 TNCA Execution IR 前，强制验证：

```text
所有 tensor element 都有明确 RF/ARF 映射
不存在依赖跨 warp shared-memory exchange 的未降级操作
所有 convert_layout 都有可执行 lowering
所有 MMA fragment 都符合 ISA 字节布局
```

允许的转换路径建议按优先级排列：

1. metadata-only；
2. 寄存器重命名；
3. lane shuffle / permute；
4. 跨 vector register 的显式搬运；
5. pack/unpack；
6. 调整 Tensor DMA 搬运形状；
7. 普通全局临时对象兜底。

第 7 条必须显式计入 spill/temporary 成本，而不是假装它是免费的“布局转换”。

---

# 五、TNCA dialect：建议分成三个抽象层，但不要做三个庞大方言

下面的 `tnca.*` 名称都是**建议新增的设计**，不是声称上游已经存在。

## 5.1 第一层：值语义仍由 TTIR 和标准 dialect 承担

在这里：

```mlir
%a = tt.load ...
%b = tt.load ...
%c = tt.dot %a, %b, %acc ...
```

仍然是最适合做数学与访问分析的表示。

不要在这一阶段就给所有值分配 `W3`、`A7`、`B12`。

物理资源编号过早进入 IR，会限制：

- CSE；
- rematerialization；
- tiling；
- role partition；
- stage 数选择。

## 5.2 第二层：资源与异步执行语义

建议核心类型如下：

| 类型 | 含义 |
|---|---|
| `!tnca.rf_ref<...>` | 一段虚拟 RF 存储及布局，不是通用指针 |
| `!tnca.acc_ref<...>` | 虚拟 accumulator 资源 |
| `!tnca.desc<...>` | 已取得语义快照的 TensorMap |
| `!tnca.event<kind>` | 某一种完成条件 |
| `!tnca.channel<...>` | 编译期通信协议，最终由 RF/ARF 和 barriers 实现 |
| `!tnca.lease<read/write>` | 对某个 channel slot 的访问权限 |

其中 event 至少区分：

```text
load_ready
compute_done
source_reusable
store_visible
```

即使当前硬件把两个事件合并，也不要在 IR 中把它们混成一种。

例如：

```text
当前 MMA:
    source_reusable → 绑定到 compute_done

未来 MMA:
    source_reusable → 独立硬件完成事件
```

这样未来增加早释放能力，主要修改 lowering 和调度，不需要重写上层所有权模型。

### Channel 不是新增存储层次

这里的 channel：

```text
channel
    = 若干 RF/ARF slot
    + ready/free 同步关系
    + producer/consumer 集合
```

它不对应一块新 SRAM，也不要求硬件增加软件可寻址 FIFO。

## 5.3 建议操作集合

### 资源与视图

```text
tnca.rf.reserve
tnca.rf.view
tnca.rf.read
tnca.rf.write

tnca.acc.reserve
tnca.acc.read
tnca.acc.write

tnca.desc.snapshot
```

`rf.view` 应限制为编译器可分析的寄存器子范围／fragment 视图，不提供通用 RF byte pointer。

### 异步操作

```text
tnca.dma.load
tnca.dma.store
tnca.mma.async
tnca.await
```

### 角色和通信

```text
tnca.specialize
tnca.role

tnca.channel.create
tnca.producer.acquire
tnca.producer.publish
tnca.consumer.acquire
tnca.consumer.release
```

### 性能元数据

```text
tnca.prefetch
tnca.use_complete
tnca.progress
```

最后这一组在当前硬件不支持相应指令时可以被删除或只进入 manifest；它们不能承担正确性依赖。

## 5.4 不要把所有权只写成 annotation

例如，仅仅：

```text
rf_lifetime = ...
async_read_set = ...
```

不够。

建议三个层次同时存在：

1. **SSA event/lease**：让依赖可见；
2. **MemoryEffectsOpInterface**：阻止错误删除、复制和移动；
3. **专用异步资源 verifier**：检查跨指令、跨角色、跨循环的未完成访问。

MLIR 的副作用接口可以描述资源访问，但官方文档也明确指出，它不是细粒度地址范围和别名消歧机制。因此不能期待加上 `MemRead`/`MemWrite` 就自动解决异步 RF 生命周期。([mlir.llvm.org](https://mlir.llvm.org/docs/Rationale/SideEffectsAndSpeculation/))

建议增加两个核心接口：

```text
TNCAAsyncOpInterface
    getReadFootprints()
    getWriteFootprints()
    getCompletionKinds()
    getSourceReleaseCondition()

TNCAExecutionResourceInterface
    getExecutionUnit()
    getIssueDemand()
    getRFPortDemand()
    getCommandSlotDemand()
```

它们是 TNCA 新增接口；普通 MLIR analyses 继续复用。

---

# 六、Context Specialization：应复用分区基础设施，但重新定义通信与映射

这是整个后端的核心。

## 6.1 先区分三个不同概念

```text
Logical partition
    算法依赖图上的职责划分

Physical RCTX
    真正拥有 PC、标量寄存器和等待状态的控制上下文

Execution engine
    SIMD、Matrix、DMA、SFU 等共享执行资源
```

这三者不能一一等同。

尤其：

> 两个 mainloop RCTX，不代表两个 Matrix engine，也不代表矩阵吞吐自动翻倍。

你的“1 prologue + 1 epilogue + 2 mainloop”应成为**首选调度模板**，而不是 IR 的不可变结构。

如果 RTL 的发射或端口仲裁已经偏向这个模板，可以在 TargetInfo 中表达偏好。

另外，这里的 prologue/epilogue role，和软件流水改写产生的 loop prologue/epilogue 不是同一概念。前者可以在整个 kernel 生命周期内持续运行。

## 6.2 推荐的两种四角色模板

### 模板 A：GEMM

```text
R0：输入搬运与描述符管理

R1：mainloop A
    计算一部分输出子块，覆盖完整 K

R2：mainloop B
    计算另一部分输出子块，覆盖完整 K

R3：epilogue 与输出
```

两个 mainloop 可以：

- 分别负责不同 M/N 子块；
- 在 persistent kernel 中交错处理独立输出 tile。

**第一版不要为了得到两个 mainloop，自动沿 K 做 split-K。**

因为这会引入：

- 新的归约；
- 新的 accumulator 生命周期；
- 浮点累加顺序改变；
- 可能需要 atomics 或额外 kernel。

### 模板 B：FlashAttention

```text
R0：K/V 搬运

R1：Matrix mainloop
    QKᵀ / PV

R2：Vector mainloop
    mask / max / exp / sum / O rescale

R3：最终归一化、输出或下一任务准备
```

这里两个 mainloop 是不同计算职责，不是两个固定矩阵流水。

编译器仍须维护：

```text
QK[t] → softmax[t] → O rescale[t] → PV[t]
```

同时寻找合法的：

```text
QK[t+1] 与 softmax[t]
```

重叠机会。

## 6.3 当前必须同时支持 2-RCTX profile

你贴出的 Status 是 **2 Role Contexts complete**，而不是 4。

建议从一开始定义两个目标 profile：

```text
tnca-s3-rctx2
tnca-rctx4
```

当前回归只能把前者当作已实现目标。

对于 2 RCTX：

- 可以选择 producer + compute/store；
- 或选择单 RCTX 软件流水。

但不能先生成四个互相等待的 role，再简单把其中两个顺序拼到同一个 PC 上。

**逻辑 partition 的合并必须重新生成调度。**否则很容易出现：

```text
同一个 RCTX 上的任务 A 睡眠
但唤醒 A 所需的任务 B 也在这个 RCTX 上
```

这种死锁不是增加 barrier 能修复的。

---

# 七、自动 CS 的具体算法与复用边界

建议拆成以下六步。

## 7.1 建立包含循环依赖的数据流图

图中的边不能只有 SSA def-use，还应包括：

- loop-carried dependence；
- 有别名可能的内存依赖；
- 控制依赖；
- accumulator 更新顺序；
- 异步源与目的的资源冲突。

上游 partition scheduling 已经有数据流图和分区合并基础设施，`PartitionSet` 也考虑跨分区及跨迭代输入输出。这些是应优先复用的部分。([raw.githubusercontent.com](https://raw.githubusercontent.com/triton-lang/triton/main/lib/Dialect/TritonGPU/Transforms/WarpSpecialization/PartitionScheduling.cpp))

## 7.2 替换 operation classification 和 partition policy

TNCA 的分类建议为：

```text
Address/control
Load/transfer
Matrix
Vector/reduction/SFU
Store/epilogue
Synchronization
```

不要继承诸如：

```text
TMEM 与 MMA 必须如何分区
CLC 必须独占 worker
NVIDIA warpgroup 寄存器预算
```

等策略。

建议把目标差异集中为一个适配接口，而不是到处修改上游算法：

```cpp
// 建议接口，不是现有上游 API
struct SpecializationTargetInfo {
  OpClass classify(Operation *);
  bool canRematerialize(Operation *);
  Cost estimateCrossPartitionUse(Value);
  bool canMerge(Partition, Partition);
  ResourceBudget getBudget();
};
```

当前上游并没有完整提供这样一个 TNCA 可直接实现的统一接口；需要做一次受控的源代码抽取／重构。

## 7.3 对跨分区值选择通信方式

按以下顺序：

1. 常量：复制；
2. 廉价纯地址运算：rematerialize；
3. kernel 参数：各 role 从 ABI 读取；
4. tensor：共享 RF/ARF window；
5. 动态标量：受同步保护的 RF mailbox；
6. 实在无法寄存器化：普通全局临时对象。

特别注意：

**Scalar RF 是每 RCTX 私有的。**

因此，R0 生成的 barrier token 或动态循环界限，不会自动出现在 R1 的 `Sx` 中。

对于静态 pipeline，epoch 通常可以由各 role 根据迭代编号独立计算；不必真的跨角色搬运每个 token。

对于真正动态产生的标量，则需要明确通信。

## 7.4 插入 RF channel，而不是 shared-memory Aref

上游 Aref 的 producer-enter/exit、consumer-enter/exit 思路很适合这里；但当前实现以 MemDesc 和 shared/TMEM buffer 为基础。建议复用其依赖分析与插入位置计算，替换承载对象和同步 lowering。([raw.githubusercontent.com](https://raw.githubusercontent.com/triton-lang/triton/main/third_party/nvidia/include/Dialect/NVWS/IR/NVWSOps.td))

示意 IR：

```mlir
// 示意语法，不是可直接运行的现有 MLIR。

%inputs = tnca.channel.create
    stages = 2
    consumers = [@main0, @main1]
    payload = <A_fragment, B_fragment>

tnca.role @producer {
  scf.for %i = ... {
    %slot, %write_lease =
        tnca.producer.acquire %inputs[%i]

    %a_ready = tnca.dma.load %desc_a, %coord_a, %slot.a
    %b_ready = tnca.dma.load %desc_b, %coord_b, %slot.b

    tnca.producer.publish %write_lease
        after(%a_ready, %b_ready)
  }
}

tnca.role @main0 {
  scf.for %i = ... {
    %slot, %read_lease =
        tnca.consumer.acquire %inputs[%i] consumer(0)

    %result_done, %source_free =
        tnca.mma.async %slot.a, %slot.b, %acc0

    tnca.consumer.release %read_lease after(%source_free)
    // 后续使用 acc0 仍必须满足 result_done。
  }
}
```

`main1` 对共享输入也持有独立 read lease。

只有两个 consumer 都释放，producer 才能覆盖该 slot。

## 7.5 构造 role 内部循环

这里复用上游 `PartitionLoops` 的 SCF 克隆、参数和结果重建逻辑，但替换：

- region container；
- 跨分区 tensor 结果的 shared-memory 临时对象；
- 角色参数捕获；
- join 与资源排空。

第一版建议只支持：

- 结构化 `scf.for`；
- 所有 role 可共同推导的迭代次数；
- 可证明一致的条件控制；
- 固定数量 channel。

动态 `while`、不规则条件生产／消费、多个相互嵌套 specialization 区域放到后续。

## 7.6 最后才进行物理 RCTX 映射

输出：

```text
logical partition → RCTX
role entry PC
private scalar/vector allocation
shared RF aliases
barrier allocation
channel generation formula
```

不能让物理 RCTX 数决定最开始的数据流图表示；否则两角色和四角色版本会形成两套编译器。

---

# 八、协议 lowering：这是比“自动分区”更应该先完成的部分

## 8.1 `publish` 不能原地机械翻译成 `mb.arm`

前面的高层 IR 可以先产生 event，再 publish。

但低层 ISA 必须：

```text
先 arm
→ 再发出相关异步命令
→ 最后 arrive
```

所以 protocol lowering 应先收集完整 event group，再把 `mb.arm` 放到所有命令提交之前。

建议映射：

| 高层协议 | 低层实现 |
|---|---|
| producer acquire | 等待 Free generation |
| producer publish | Ready generation + 对应 async completions |
| consumer acquire | 等待 Ready generation |
| consumer release | 等待 source-reusable 后 arrive Free |
| role join | 等待其所有必要异步操作和输出完成 |

## 8.2 operation count 必须在命令展开后计算

一个高层 tensor load 可能被拆成：

- 多条 Tensor DMA；
- DMA + LSU tail；
- 多个 fragment 搬运命令。

因此：

> **不能在高层按“两个 tensor load”固定 `operations=2`，之后再随意把它们拆成八条异步指令。**

建议：

```text
transfer formation
→ fragment command expansion
→ completion group construction
→ barrier operation count 固定
```

如果后续还会改变异步命令数量，就必须重新构建 completion group。

## 8.3 分支和 predication 必须保持完成计数

合法的 predicated-off async command：

- 不访问内存；
- 不修改目标；
- 仍产生一次 completion。

但普通：

```text
if (...) {
  issue_async
}
```

不是同一回事。

不能让某条路径跳过 async command，却仍保留无条件 `operations=1`。

第一版优先选择：

- 将条件搬运变成整体 predication；
- 或分别构建路径完整的 generation；
- 无法证明时退回较简单的执行计划。

## 8.4 generation 必须成为 verifier 的一部分

至少验证：

- 每个 generation 有唯一 arm owner；
- 完成数与最终发出的命令数一致；
- 每个 consumer 对 Free 恰好贡献一次；
- 旧 generation 的迟到响应不会修改新 slot；
- epoch 初始化和各角色公式一致；
- 等待未来 generation 的语义明确；
- arm/reinit 不会清除合法 waiter。

**这些测试应先于自动 WS 优化进入 RTL 回归。**

---

# 九、一个会立即暴露硬件／编译器问题的例子：大 tile 并不天然符合 MMA fragment

你当前的表述：

> Transfer 写 flat row-major，MMA 读 flat row-major，因此不需要 repack。

对**完全匹配的一块原生 fragment**成立，但不能自动推广到任意 GEMM tile。

例如：

```text
B tile = 32 × 64
MMA B fragment = 32 × 16
```

如果整个 B tile 按 row-major 装入 RF，那么某个 `32×16` 列子块的相邻行间隔是：

```text
64 × sizeof(dtype)
```

而当前 MMA fragment 解释要求：

```text
16 × sizeof(dtype)
```

只改变 fragment base，不能消除这个差别。

依据你给出的 ISA，编译器至少要在以下路径中选择：

1. 把 B 拆成四次 `32×16` 搬运，分别形成紧凑 fragment；
2. 在 RF 中显式 repack；
3. 未来增加受控的 strided-fragment 能力；
4. 改变内部 microtiling。

**建议第一版选择 1，不急着改 MMA ISA。**

但这样会改变：

- DMA command 数量；
- barrier operation count；
- window 数量；
- descriptor/block-shape 规划；
- RF live range；
- 搬运发射开销。

这正说明：

> **布局选择、通信协议和流水资源预算必须联合规划，不能作为三个互不相关的 pass。**

另外，当前 `W`、`V base`、`A` 等大多是指令中的静态编号。动态：

```text
slot = i % 2
```

不能直接变成一个任意动态选择的 vector-register base。

第一版应通过：

- 按 stage 数展开；
- 或生成少量分支版本；

把每个 stage 的 W/V/A 编号静态化。

---

# 十、资源分配：普通活跃区间不足以覆盖 TNCA

## 10.1 分成全 program 分配与 role-local 分配

### 全 program 资源

统一分配：

- 共享 RF windows；
- accumulator slots；
- barriers；
- descriptor slots；
- command/completion 预算；
- RF alias mapping。

### Role-local 资源

分别分配：

- Scalar RF；
- Predicate RF；
- 私有 vector 临时值；
- 普通控制流状态。

不能把每个 role 独立交给普通寄存器分配器，然后期待它们自然协调共享 RF。

## 10.2 异步生命周期以完成事件结束

例如：

```text
m.mma.async issue
```

不是源 fragment 的最后使用点。

真实生命周期是：

```text
第一次写入
→ DMA 完成
→ consumer 取得读取权
→ 最后一个异步 consumer source-reusable
→ 可以复用
```

当前可以把 source-reusable 保守绑定到 compute-done，但不能用估计周期数代替完成事件。

## 10.3 资源不足时的回退顺序

建议：

1. 减少 RF stages；
2. 减少同时活动的独立输出 tile；
3. 使用更细的合法 microtiling；
4. 重新选择 role 划分；
5. rematerialize 廉价纯计算；
6. 使用显式 global spill；
7. 给出资源不足诊断。

注意：编译器不能在 TTIR 已生成后，偷偷改变用户的 `BLOCK_M`，进而改变 `program_id` 对应的工作范围。

可以做内部合法细分，或者让 autotuning 选择另一份源程序配置。

## 10.4 成本模型必须按共享执行资源累计

对于一个稳态迭代：

\[
II \ge
\max_r
\frac{\sum_{\text{所有 roles}} \text{该资源上的需求}}
     {\text{该资源的容量}}
\]

同时还要满足 recurrence 和事件依赖约束。

按你贴出的 README，当前 Matrix 是 **16 MAC/cycle** 的实现。若每个 MAC 按两次浮点运算计，其理想上限是 **32 FLOP/cycle**，不能继续沿用早期参考配置的 512 FLOP/cycle 来评估 CS。

因此两个 mainloop 的价值应由实际测量回答：

- 是否减少 Matrix 饥饿；
- 是否与向量计算重叠；
- 是否只是增加了 RF 占用和 barrier 开销。

---

# 十一、代码生成：既不要重写 LLVM，也不要让完整 LLVM target 阻塞启动

这里建议采用明确的两阶段策略。

## 11.1 Bring-up 路径：复用现有 assembler

最早的可运行路径：

```text
TNCA low-level IR
→ 保守调度和受限静态分配
→ TNCA assembly
→ 现有 tnca_asm.py
→ RTL
```

好处是：

- 立即复用已测试的编码和 reserved-bit 检查；
- 可以与手写汇编逐条比较；
- 编译器问题和机器码编码问题容易分离。

这一阶段不建设复杂的通用机器优化器。

资源不够可以先明确报错，不能为了快速“支持”而偷偷破坏异步生命周期。

## 11.2 长期路径：瘦 LLVM TNCA target

如果要实现成熟的：

- 普通寄存器分配；
- spill/reload；
- machine scheduling；
- peephole；
- 指令编码与反汇编；

应优先评估一个瘦 LLVM target，而不是在 MLIR 中再造同类基础设施。LLVM 的 target 框架本来就将寄存器类、指令描述与通用寄存器分配连接起来。([llvm.org](https://llvm.org/docs/WritingAnLLVMBackend.html?utm_source=openai))

但必须明确：

> LLVM 普通寄存器分配器不会自动理解跨 RCTX、跨异步命令的 RF window 所有权。

建议边界为：

```text
TNCA whole-program planner
    先固定共享 windows、accumulators、barriers

LLVM target
    再处理每个 role 的普通 S/P/V 临时值和机器指令
```

在 LLVM 路径成熟前，异步边界使用保守调度约束；不要让通用机器调度把 RF 覆盖移动到 source-reusable 之前。

## 11.3 单一 ISA 信息源

你已经有机器可读的：

```text
tools/asm/tnca_isa.py
```

应复用它或将其规范化为共同 schema，生成：

- assembler/disassembler 表；
- compiler opcode 定义；
- encoding legality；
- 后续 TableGen 输入；
- conformance tests。

不要再手工维护第三套独立 opcode 数字。

---

# 十二、普通 Triton lowering 也有几个必须提前处理的边界

这些不是 WS，但会决定“广泛支持 Triton”是否真实。

| 问题 | TNCA lowering 建议 |
|---|---|
| tensor 大于 16 elements | 拆成多个 vector registers，保持原 tensor 索引映射 |
| uniform FP32 标量 | 不假定 Scalar ALU 有浮点能力；用向量域或合适的位表示搬运 |
| masked load 的 `other=` | 显式 passthrough／select，不能依赖 RF 旧值恰好正确 |
| 64-bit tensor pointers | 优先识别 uniform base + 有界 offset |
| gather offset 超过 signed 32-bit | rebasing、分组或逐 lane scalar fallback |
| 自定义 reduction | 保留组合 region，生成 shuffle + combine |
| FP reduction | 按目标数值契约选择确定树形，不随意 reassociate |
| SFU 不存在 | 软件数学 lowering 或明确不支持，不发会 trap 的 SFU opcode |
| atomics/fence 不存在 | capability 诊断，不能降成普通 load/store |
| descriptor rank > 2 | 证明可 flatten 时 flatten，否则循环/LSU fallback |
| descriptor `nan` padding | 当前 TNCA zero/clamp/fault 不等价，必须显式补齐或 fallback |

尤其是 SFU，不需要从零设计整套软件数学算法。MLIR 已有数学多项式展开，可作为候选；但其中某些路径带目标依赖，误差和特殊值行为也必须重新验证，不能直接声称满足你 ISA 的 SFU 精度契约。([raw.githubusercontent.com](https://raw.githubusercontent.com/llvm/llvm-project/main/mlir/lib/Dialect/Math/Transforms/PolynomialApproximation.cpp))

同样，Triton 的 TensorDescriptor 可以复用其语义接口，但不能复用 NVIDIA descriptor 的二进制格式。现有 Python descriptor 和 driver 代码已经有 shape、stride、padding 等分解与封装机制，可在这层接入 TNCA 的 256-byte 格式。([raw.githubusercontent.com](https://raw.githubusercontent.com/triton-lang/triton/main/python/triton/tools/tensor_descriptor.py))

---

# 十三、现在就应该反馈给硬件／ISA 的几个问题

我会把以下项目列为 compiler bring-up 的前置核对，而不是等到 FlashAttention 才发现。

## 13.1 共享 RF 的“可见”是否真的可寻址

需要明确回答：

```text
R0 的 W0 写入物理 V 范围
R1 的 MMA 如何命名同一物理范围？
```

如果每 RCTX 仍是严格不重叠的 RF 是共享的”还不足以实现零拷贝 role 通信。

必须明确：

- W base 是逻辑还是物理编号；
- 每 RCTX 的逻辑视图如何建立共享 alias；
- compiler manifest 如何描述；
- 普通 vector 指令和 MMA 是否采用相同映射。

## 13.2 `mb.arrive.release` 的 release 到底覆盖什么

必须区分：

- 已发射；
- 已写回 RF；
- 异步操作已完成；
- global store 已到达可见性点。

不能因为 scalar 指令顺序执行，就默认它之前的多周期向量写回和异步访存都已经完成。

编译器会显式等待异步完成，但硬件仍需兑现它承诺的 release 语义。

## 13.3 启动初始化不能和 waiter 竞争

多角色同时启动时：

- 谁初始化 barriers；
- 谁定义 windows；
- 谁加载 descriptors；
- 其他 role 在什么时候允许等待／使用；

需要一个确定协议。

特别要避免：

```text
R1 先进入 wait
R0 随后 mb.init 清掉 waiter 状态
```

这种启动时序问题。

## 13.4 大范围隐式寄存器读必须进入编译器模型

包括：

- MMA 从 base 开始读取整个 fragment；
- Tensor DMA 从 `Sb` 开始读取 rank 个 scalar coordinates；
- 异步 store 延迟读取源窗口。

这些都不能只靠 opcode 显式字段中的单个寄存器 index 建 liveness。

## 13.5 BF16 的数值描述需要修正

你给出的：

> BF16×BF16 product is exactly representable in binary32

不能无条件成立。BF16 和 FP32 的指数范围相同，乘积仍可能超出 FP32 指数范围；只有有效位数量够，并不意味着指数范围也够。([docs.riscv.org](https://docs.riscv.org/reference/isa/extensions/bf16min/_attachments/riscv-bfloat16.pdf))

建议将契约直接写成：

\[
C_{k+1}
=
\operatorname{RNE}_{f32}
\left(
\operatorname{exact}(A_k)\operatorname{exact}(B_k)+C_k
\right)
\]

即真正的 widened-input FP32 FMA 语义，不先把乘积舍入成 FP32。

一个建议加入 oracle 的例子：

```text
A = 2^127   （BF16 可表示）
B = 2
C = -2^127

融合计算结果：2^127
若先将乘积舍入为 FP32：可能先变成 +inf
```

这主要是规范表述修正；你复用 `tnca_fp32_fma` 的实现路线可能已经是正确的。

---

# 十四、推荐的 pass pipeline

以下 TNCA pass 名称是建议的工程拆分。

| 阶段 | 输入→输出 | 主要任务 |
|---|---|---|
| 1. `tnca-check-target` | TTIR | capability、dtype、数值模式、控制流检查 |
| 2. 上游规范化 | TTIR→TTIR | inlining、canonicalize、CSE、常量传播 |
| 3. `tnca-analyze-access` | TTIR | 规则访问、mask、alias、descriptor 候选 |
| 4. 布局化与白名单优化 | TTIR→布局 IR | 复用布局表示和传播 |
| 5. `tnca-select-fragments` | 布局 IR | MMA microtile、packed layout、DMA 形状 |
| 6. `tnca-plan-contexts` | 布局 IR | partition、rematerialization、RCTX 候选 |
| 7. `tnca-build-channels` | →Execution IR | RF/ARF 通信和 ownership |
| 8. `tnca-plan-pipeline` | Execution IR | stage 数、command 重叠、依赖调度 |
| 9. `tnca-plan-resources` | Execution IR | RF/ARF/barrier/descriptor 可行性 |
| 10. `tnca-expand-commands` | →低层 async IR | 最终 DMA/MMA 命令序列 |
| 11. `tnca-lower-protocols` | →低层执行 IR | arm/count/epoch/wait/arrive |
| 12. `tnca-outline-roles` | →每 role CFG | 参数捕获、入口、结束排空 |
| 13. 指令选择与普通 RA | →机器层 | S/P/V、指令约束、branch lowering |
| 14. `tnca-verify-machine` | 机器层 | 编码、资源、async lifetime 最终检查 |
| 15. emission | Binary + manifest | 复用 assembler，后续 LLVM target |

其中 **5–9 不是严格单向的一次流水**。

建议使用有限候选搜索：

```text
布局 / microtile / role / stages 候选
    ↓
资源可行性与成本评估
    ↓
选择一个计划
    ↓
冻结后再实体化协议
```

不要在物理 barrier 已经生成之后，继续无约束地改变 stage 数。

对于循环 prologue/epilogue 的机械改写，可复用 MLIR `pipelineForLoop` 一类工具；其接口允许调用者提供 schedule 和 predication，但并不替调用者证明 schedule 正确。([raw.githubusercontent.com](https://raw.githubusercontent.com/llvm/llvm-project/main/mlir/include/mlir/Dialect/SCF/Transforms/Transforms.h))

---

# 十五、实施顺序：六个可验收里程碑

建议把原来的 Stage 5 拆出来，与硬件 Stage 4 并行。

## C0：固定工具链与目标契约

交付：

- 固定 Triton commit；
- 固定其对应 LLVM revision；
- out-of-tree backend skeleton；
- `tnca-opt`；
- TargetInfo；
- ISA schema 接入；
- 每阶段 IR dump。

不追实时 `main`。可以用 3.8.0 作为稳定基线候选，再根据需要选择包含特定 WS 修复的明确 commit。Triton 官方也要求与其指定 LLVM revision 配套构建，而不是任意 LLVM 版本。([raw.githubusercontent.com](https://raw.githubusercontent.com/triton-lang/triton/main/README.md))

目标配置至少包含：

```text
RCTX count
VLEN
physical/logical V register limits
accumulator count
DMA rank / dtype / layout capabilities
MMA throughput and command constraints
barrier count / count width
source-release capability
SFU / atomics / fence capabilities
```

## C1：第一个真实 Triton→RTL 闭环

先完成：

- vector add；
- masked elementwise；
- integer reduction；
- FP32 reduction；
- 简单控制流；
- gather/scatter 可支持子集。

要求：

> 不修改输出汇编，不人工补 barrier，直接运行编译产物。

先用 AOT + 仿真 launcher，不必把完整 PyTorch device integration 作为前提。

## C2：单 RCTX 的 Tensor DMA + MMA

先不做自动 CS。

完成：

- FP16 GEMM；
- BF16 GEMM；
- 不同 stride；
- N/M/K tail；
- microfragment 布局；
- accumulator epilogue。

这一阶段就应该发现“大 B tile 不等于紧凑 B fragment”的问题。

## C3：显式两角色 pipeline

角色计划先由测试或编译选项明确指定，不急于自动推导。

完成：

- ready/free 双缓冲；
- 0、1、2、奇数次迭代；
- predicated-off async；
- 乱序返回；
- 延迟 completion；
- 错误传播；
- 程序退出排空。

**这是第一阶段最重要的里程碑。**

## C4：接入自动 partition 基础设施

这时才抽取上游：

- graph construction；
- partition propagation；
- loop cloning；
- producer/consumer 分析。

先支持 GEMM 类循环，再扩大控制流范围。

四 RCTX 硬件 profile 可用后，再验证：

```text
P + M0 + M1 + E
```

## C5：Attention 和性能反馈

加入：

- 软件数学 fallback；
- 后续 SFU 原生 lowering；
- QK/softmax/PV 重叠；
- K/V 分离 channel；
- ARF 与 vector 交互；
- 多种 role 模板比较。

最后再让 Stage 4 cache/flow metadata 从最终 schedule 生成。

---

# 十六、验证和诊断必须成为后端的正式输出

## 16.1 每个 kernel 输出资源与瓶颈报告

例如：

```text
active contexts:          2
RF stages:                2
shared RF windows:        ...
private vector bytes:     ...
accumulator slots:        ...
barriers:                 ...
max async commands:       ...

layout conversion bytes: ...
DMA command expansion:   ...
predicted issue pressure: ...
source-release policy:   compute_done

rejected 4-context plan:
    insufficient accumulator capacity
```

这比只输出“寄存器用了多少”更能指导硬件修改。

## 16.2 正确性测试分三层

### IR 单元测试

检查：

- layout 映射；
- role capture；
- channel 构造；
- generation 公式；
- count；
- unsupported capability 诊断。

### 协议模型测试

只建立小型 ownership/event 模型，不另写一整套 Triton 解释器。

枚举：

- 1–3 stages；
- 1–2 consumers；
- 不同完成顺序；
- 短循环；
- 全 false；
- 故障。

### 编译产物 RTL 差分

复用：

- 现有 assembler；
- 现有 exact FP oracle；
- 随机 memory latency/back-pressure；
- 原有 NUCA/mesh 测试环境。

数值语义比较和 bit-exact 指令语义比较分开，不能要求 TNCA 和任意 GPU 的 `dot` 累加位模式完全一致。

## 16.3 ConSan 值得研究，但不是直接打开就能用

上游现在已有 TritonInstrument/ConSan，并通过 target hooks 建模异步访问与 barrier；但其 buffer analysis 和插桩主要围绕 shared memory/TMEM。([raw.githubusercontent.com](https://raw.githubusercontent.com/triton-lang/triton/main/include/triton/Dialect/TritonInstrument/IR/TritonInstrument.md))

建议：

1. 先在仿真 trace 上实现 RF/ARF ownership 检查；
2. 再复用 ConSan 的可见性分析和测试思路；
3. 若接口适合，扩展 TNCA hooks 和 RF footprint analysis。

不要为了使用 ConSan，反过来把 RF 假装成可寻址 shared memory。

---

# 最后的建议：最先做这五个工程决定

我会立即冻结以下五点：

1. **采用混合 IR：保留 `tt/arith/math/scf`，TNCA 只增加资源和异步语义。**
2. **普通布局优先复用 TTGIR/LinearLayout；角色数不进入 warp 布局维度。**
3. **用 channel + lease + typed completion 建模 CS，不只使用 role annotation。**
4. **先完成显式两角色端到端闭环，再移植自动 partition 算法。**
5. **短期复用现有 assembler，长期复用 LLVM 通用机器后端能力，不自建第二套通用编译器。**

对你这颗芯片而言，最难的并不是“把 load、MMA、softmax 分到不同角色”。

真正难的是：

> **分区以后，谁拥有哪一段 RF；异步命令何时停止读取；两个消费者怎样共同释放；同一个物理寄存器何时能安全复用；这些事实如何在布局优化、循环变换和寄存器分配之后仍然成立。**

因此，第一份值得认真设计和审查的 TNCA IR 文档，应当是 **RF ownership、channel 与 completion semantics**。这份协议稳定后，Triton 现有的布局、分区和循环基础设施才能真正成为可复用资产，而不是新的 GPU 假设来源。