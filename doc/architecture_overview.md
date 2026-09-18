我先核对两篇论文的完整机制，以及 Triton/MLIR 对布局、异步访存和同步的实际要求。然后把方案分成“必须保证的语义”和“可选的性能机制”，重点检查：取消 SPM 后，寄存器、NUCA、TMA 和少量专职上下文之间能否形成高效且不会死锁的流水线。核对后，有两个关键点会直接影响方案：DCO 正文中的数据通路仍包含核心私有 SPM，因此这次取消 SPM 后，必须重新安排操作数复用和布局转换；另外，少量专职上下文不能照搬“暂停最快核心”的节流方式，否则可能暂停掉消费者或释放资源的路径。下面我会把节流落到新访存请求的准入上，并明确区分“数据已到寄存器”“计算已完成”和“寄存器可以复用”这三个事件。citeturn1view2turn1view3## 先给出结论

我建议把这个芯片定位为：

> **以 TensorMap 为数据流契约、以寄存器为计算工作集、以单级 NUCA 为共享数据存储、以少量专职执行上下文驱动异步流水的 Tensor-Native Accelerator。**

它不应是“删掉 Shared Memory 的 GPU”，也不应是“把 SPM 改名为 Cache 的 NPU”。

**真正需要重新设计的是四个接口：**

1. **Triton 的 tensor/layout 语义 → 寄存器与指令。**
2. **TensorMap → NUCA 的缓存管理与请求仲裁。**
3. **异步数据搬运 → 寄存器所有权与完成事件。**
4. **跨核心进度 → 请求准入、预取距离和共享工作集控制。**

我核对的是 **LLaMCAT v1，2025 年 11 月 26 日**，以及 **DCO v2，2026 年 9 月 10 日**。下面的架构、ISA 和参数是我在这两篇工作基础上提出的设计草案，**不是论文已经验证过的芯片实现或性能数据**。([arxiv.org](https://arxiv.org/abs/2512.00083))

还有一个非常重要的边界：**DCO 正文中的核心数据通路仍包含 private SPM；你现在要求彻底取消 SPM，是比论文架构更进一步的设计。**因此不能直接沿用原来的计算侧带宽、布局转换和流水线假设。([arxiv.org](https://arxiv.org/html/2512.07312v2))

以下暂称该方案为 **TNCA-v1**。

---

# 一、首先确定六条架构原则

## 1.1 唯一的数据存储层次是 RF ↔ NUCA ↔ DRAM

```text
                 计算单元
                    │
          Scalar / Vector / Accumulator RF
                    │
          Tensor DMA / 普通 Load-Store
                    │
               Mesh Network
                    │
        唯一一级、分布式共享 NUCA Cache
                    │
             DRAM Controllers
                    │
                HBM / DDR
```

这里的 RF 包括：

- 标量寄存器；
- 向量寄存器；
- 矩阵累加器寄存器；
- 指令直接命名的寄存器组。

**不包含：**

- 软件可寻址的 Shared Memory；
- 软件显式分配的片上 Tensor SRAM；
- 需要软件维持驻留的 Cache-as-SPM 区域；
- 必须依赖 pin/unpin 才能正确运行的缓存对象。

允许存在流水 FIFO、MSHR、描述符表和指令缓冲，但它们不是软件可寻址的数据存储层次。

> 如果连独立指令缓冲也不允许，指令取指流量就必须计入 NUCA 和 mesh 带宽。本文默认限制的是**数据存储层次**，允许小型指令缓冲。

## 1.2 不让矩阵阵列直接依赖远端 Cache 的逐周期供数

我不建议第一版提供这样的执行方式：

```text
MMA 每个计算周期向 NUCA 读取下一组操作数
```

推荐：

```text
NUCA → 异步装入 RF → MMA 多次复用 RF 数据
```

原因很直接：

- NUCA 命中延迟不固定；
- mesh 可能反压；
- Cache bank 可能冲突；
- fill、writeback、hit 共享存储端口；
- 矩阵内部循环需要稳定的操作数供应。

**Cache 负责跨 tile、跨核心、跨迭代复用；RF 负责计算内循环复用。**

取消 SPM 不等于取消局部复用。

## 1.3 “≤4 个线程”应定义为四个向量控制上下文

建议称为 **Role Context，简称 RCTX**。

每个 RCTX 有独立：

- PC；
- 标量寄存器；
- 谓词状态；
- 指令队列；
- 等待事件状态。

但共享：

- SIMD 单元；
- Matrix 单元；
- Tensor DMA；
- 寄存器物理资源；
- NUCA 接口。

一个 RCTX 控制一组 SIMD lanes，而不是只执行一个标量元素。

典型分工：

| RCTX | 常见职责 |
|---|---|
| R0 | 描述符、地址计算、预取、输入搬运 |
| R1 | Matrix 运算 |
| R2 | reduction、softmax、activation、epilogue |
| R3 | 输出搬运，或第二条计算流水 |

**这些只是编译器角色分配，不应成为不可改变的硬件绑定。**

例如小型 sigmoid 不应该被迫启动四个上下文和一整套 barrier 流水。

## 1.4 TensorMap 中必须区分“语义”与“提示”

两类信息绝不能混在一起。

### 必须严格执行的语义

- 基地址；
- shape；
- stride；
- dtype；
- 边界处理；
- 实际访问地址；
- 寄存器目标；
- 异步完成；
- 内存顺序。

### 可以忽略的性能提示

- 预计复用次数；
- 最后一次使用；
- 缓存优先级；
- bypass 倾向；
- 预取距离；
- 共享组；
- 进度标记。

**所有缓存预测错误都只能影响性能，不能影响结果正确性。**

尤其：

> `dead` 只能表示“优先考虑替换”，不能表示“可以直接丢弃 dirty data”。

## 1.5 ISA 应亲和 TTIR/TTGIR，但不能把 IR 直接固化成硬件

建议保留这样的分层：

```text
Triton 源程序
    ↓
TTIR：tensor、算子、控制流语义
    ↓
目标相关 layout / pipeline / role partition
    ↓
TNCA IR：RF window、异步事务、MMA fragment、barrier
    ↓
TNCA ISA
```

TTGIR 中确实存在布局转换、共享存储对象、异步拷贝和 warp-specialization 等操作；其中部分操作已经带有明确的 GPU 存储和执行假设，不能机械地逐条映射到你的硬件。([triton-lang.org](https://triton-lang.org/main/dialects/TritonGPUOps.html))

## 1.6 “广泛支持 Triton”应拆成三个承诺

| 层次 | 建议承诺 |
|---|---|
| 高层算子 | 支持你列出的 GEMM、Conv、FlashAttention、LayerNorm、softmax、sigmoid 等 |
| 可移植 Triton 语义 | 尽量完整，允许部分操作用软件展开 |
| NVIDIA/AMD 特定代码 | 不承诺原样执行，需要重写或目标专用 lowering |

例如嵌入 PTX 的 `inline_asm_elementwise` 本身就是目标相关代码，不能因为前端叫 Triton 就自动变得可移植。([triton-lang.org](https://triton-lang.org/main/python-api/generated/triton.language.inline_asm_elementwise.html))

---

# 二、最可能出现的性能瓶颈

先给总表，再展开最重要的几个。

| 瓶颈 | 为什么你的架构容易遇到 | 主要解决方向 |
|---|---|---|
| RF 带宽 | 没有 SPM 后，搬运、MMA、向量运算都竞争 RF | 分 bank、窗口布局、操作数保持寄存器、端口预算 |
| RF 容量 | 异步双缓冲、Attention 中间量、累加器同时存活 | 分离累加器 RF、缩小 tile、控制流水深度 |
| NUCA 命中带宽 | 命中也要经过 tag/data bank 和网络 | 核内 RF 复用、bank-aware layout、近端调度 |
| Mesh 截面带宽 | 分布式容量不等于可均匀使用的带宽 | home-aware placement、共享组调度、限制远程流量 |
| MSHR entry 不足 | 四个 RCTX 不代表内存级并行度足够 | DMA 展开大量独立请求、端到端 credit |
| MSHR target 不足 | GQA 等共享访问可能集中合并到少数 miss | 独立 target 预算、合并仲裁、响应扇出控制 |
| 返回路径阻塞 | DMA 返回时 RF/window 或 FIFO 无法接收 | 发出前预留目标、返回路径独立保活 |
| TensorMap 处理吞吐 | 每个 cache line 都做复杂 tensor 运算会很贵 | 描述符快照、分层元数据、地址生成与策略查询分离 |
| 布局转换 | 没有 shared memory 中转 | RF shuffle、TMA 入 RF 重排、受控慢速兜底 |
| SFU/reduction | FlashAttention 并非只有两次 GEMM | 足够的 exp2、归约、累加器向量访问能力 |
| 节流振荡 | 多个控制器同时改变缓存和注入行为 | 分时间尺度、滞回、统一压力指标 |
| Barrier/所有权开销 | 少量上下文更容易被同步关键路径限制 | 完成事件细分、sleep/wakeup、静态生命周期分析 |
| 小 kernel 开销 | 描述符和多角色流水可能比计算还贵 | 单 RCTX 快路径、普通向量访存 |
| 不规则访存 | TMA 无法覆盖全部 gather/scatter | 必须保留通用 LSU |
| 临时值 spill | 缓存承载原本 shared memory 和寄存器溢出 | 编译器成本模型、独立低优先级 spill 流 |

---

## 2.1 第一危险：RF 会代替 SPM 成为新的瓶颈

假设矩阵单元每周期处理一个 \(16\times16\) 外积：

- A：16 个 FP16/BF16；
- B：16 个 FP16/BF16。

输入需求是：

\[
B_{\text{MMA-input}}
=(16+16)\times2=64\ \text{B/cycle}
\]

这还没包括：

- SIMD 读操作数；
- SIMD 写结果；
- DMA 回填；
- accumulator 与 SIMD 交换；
- layout conversion。

因此不能同时声称：

- RF 很小；
- RF 只有很少端口；
- MMA 满速；
- SIMD 满速；
- DMA 每周期满速写入；
- 这些又能全部无冲突重叠。

**我的建议：先把 RF 端口做真实，再决定算力。不要反过来。**

## 2.2 第二危险：高 Cache hit rate 不等于高性能

对这个设计，一个 NUCA hit 仍可能经历：

```text
请求注入等待
→ 多跳 mesh
→ slice 请求队列
→ tag 查询
→ data bank 等待
→ 响应网络
→ RF 写端口等待
```

所以至少要区分：

\[
\text{hit rate}
\quad\neq\quad
\text{hit throughput}
\quad\neq\quad
\text{useful operand delivery rate}
\]

建议主要优化目标改为：

> **单位时间送达可计算寄存器窗口的有效 tensor bytes，以及完成的逻辑 tile 数。**

而不是只优化命中率。

## 2.3 第三危险：四个上下文可能掩盖不了延迟

四个上下文不是问题，**只有四个 outstanding load 才是问题**。

对于假设的：

- DRAM 有效带宽：256 GB/s；
- 平均读延迟：300 ns；
- cache line：64 B；

维持带宽所需的独立在途请求下界约为：

\[
N_{\text{miss}}
\geq \frac{256\times10^9\times300\times10^{-9}}{64}
=1200
\]

这是全芯片量级，不是四个线程可以自然提供的。

因此：

- RCTX 数量可以少；
- DMA 命令数量可以不多；
- **DMA 展开的 cache-line 事务必须很多。**

## 2.4 第四危险：错误的节流可能阻止资源释放

LLaMCAT 的关键内容包括 MSHR-aware 仲裁、进度平衡和分层动态节流；原始节流还涉及调整核心内运行 thread block 数量。([arxiv.org](https://arxiv.org/html/2512.00083v1))

但在你的设计中：

```text
R0：生产数据
R1：消费数据
R2：释放中间资源
```

如果笼统地“暂停一个核心”或“暂停所有请求最多的上下文”，可能连消费者也暂停：

```text
消费者不运行
→ RF window 不释放
→ 返回数据不能落地
→ miss 不能退休
→ Cache 更拥堵
```

因此我建议：

> **节流新工作、新预取和新 miss 的准入；不要节流已经进入系统的事务完成、消费者运行和资源释放。**

---

# 三、推荐的芯片总体架构

## 3.1 参考配置：16 个 tile，4×4 mesh

下面是设计空间探索的起点，不是最终 PPA 结论。

| 项目 | TNCA-v1 参考值 |
|---|---:|
| Tile 数量 | 16 |
| Mesh | 4×4，双向 |
| 每 tile RCTX | 最多 4 |
| SIMD 宽度 | 16 lanes |
| Matrix 基本形状 | M16×N16，K16/32/64 |
| Vector RF | 32 KiB/tile |
| Accumulator RF | 16 KiB/tile |
| Scalar RF | 4×64×64 bit |
| NUCA 容量 | 512 KiB/slice，共 8 MiB |
| Cache line | 64 B |
| Associativity | 8-way |
| 每 slice data banks | 4 |
| 每 slice MSHR | 起点 96 |
| 每 core 在途 line 事务上限 | 起点 128 |
| 每 core Tensor DMA 活跃命令 | 起点 8 |
| 每 core mbarrier | 32 |
| 每 core command/completion slots | 64 |
| 每 core TensorMap 快照 | 16 |
| Mesh 数据链路 | 512-bit payload/方向，另计控制开销 |
| 评估频率 | 先用 1 GHz 做模型参数 |
| 外存接口 | 参数化，不绑定某一代 HBM |

逻辑结构：

```text
┌───────────────────────────────────────────────────────────┐
│ Host Interface / Command Processor / Grid Scheduler        │
│ Launch descriptors / Argument buffers / Resource admission │
└───────────────────────────┬───────────────────────────────┘
                            │
       ┌──────────┬──────────┬──────────┬──────────┐
       │ Tile 00  │ Tile 01  │ Tile 02  │ Tile 03  │
       ├──────────┼──────────┼──────────┼──────────┤
       │ Tile 10  │ Tile 11  │ Tile 12  │ Tile 13  │
       ├──────────┼──────────┼──────────┼──────────┤
       │ Tile 20  │ Tile 21  │ Tile 22  │ Tile 23  │
       ├──────────┼──────────┼──────────┼──────────┤
       │ Tile 30  │ Tile 31  │ Tile 32  │ Tile 33  │
       └──────────┴──────────┴──────────┴──────────┘
                            │
               Distributed Memory Controllers
```

每个 tile：

```text
┌────────────────────────────────────────────────────────────┐
│  RCTX0    RCTX1    RCTX2    RCTX3                            │
│    └──────── Ready/Scoreboard Scheduler ────────┘            │
│                │          │          │                      │
│             Scalar      SIMD       Matrix                   │
│                │          │          │                      │
│        Scalar RF    Banked Vector RF    Accumulator RF      │
│                         │                                   │
│          Tensor Address/Transfer Engine + General LSU        │
│                         │                                   │
│              Credits / Injection / Return Unit              │
├─────────────────────────┼───────────────────────────────────┤
│                       Router                                │
├─────────────────────────┼───────────────────────────────────┤
│  NUCA slice: tags / data / MSHR / targets / WB / local TMU    │
└────────────────────────────────────────────────────────────┘
```

## 3.2 不设置中心化全芯片 TMU 热点

建议分成：

1. **核心侧 TensorMap 前端**：处理 descriptor、地址展开和 RF destination。
2. **slice-local TMU**：处理缓存复用、dead hint、bypass、局部压力。
3. **低频组级控制器**：处理共享组进度、注入预算和调度亲和性。

不要让所有 cache-line 请求都经过一个全局 TensorMap 查询单元。

---

# 四、核心微架构：小控制核，大异步事务能力

## 4.1 前端与发射

建议采用：

- 每 RCTX 顺序执行；
- 无 ROB；
- 无通用寄存器重命名；
- scoreboard 检查数据相关；
- 最多每周期发射两条，来自不同资源类别；
- 一个控制/标量槽；
- 一个向量、矩阵命令或访存命令槽。

Matrix、DMA、SFU 在指令发出后自行推进。

这样保留异步重叠，但不需要实现一个复杂乱序处理器。

### 调度优先级建议

1. 可以释放 RF/window 的消费者；
2. 可以提交结果或完成 barrier 的指令；
3. 已 ready 的计算；
4. demand 访存；
5. speculative prefetch。

这是性能优先级，不代替公平性保证。

## 4.2 四个上下文不一定属于四个 program

建议支持两种 launch 模式。

### 模式 A：一个 program，多角色

```text
一个 Triton program
    ├── R0 producer
    ├── R1 matrix
    ├── R2 vector
    └── R3 store/control
```

适合 GEMM、FlashAttention、Conv。

### 模式 B：多个独立小 program

```text
R0 → program p
R1 → program p+1
R2 → program p+2
R3 → program p+3
```

适合 elementwise、小 reduction。

资源仍需静态分区，不能因为有四个 PC 就默认有四份完整计算资源。

## 4.3 角色不是硬件线程块数量

特别注意：

> **Triton 的 `num_warps` 不能直接等同于 RCTX 数量。**

`num_warps` 可能影响逻辑布局和并行划分；RCTX 表示不同控制流角色。这两者应该由后端分别处理。

---

# 五、寄存器组织：这是取消 SPM 后最关键的部分

## 5.1 三类 RF

### Scalar RF

每 RCTX：

- 64×64-bit；
- 用于地址、循环变量、descriptor handle、token；
- `s0` 固定为零；
- 浮点标量可使用低 32 位，或通过 SIMD lane 0 计算。

### Vector RF

每 core：

\[
512\times16\times32\text{ bit}=32\text{ KiB}
\]

建议：

- 每个 architectural vector register 是 16×32 bit；
- FP16/BF16 可以 packed 存储；
- 每 RCTX 使用自己的逻辑寄存器视图；
- 物理 RF 在 launch 时静态分配；
- producer 不必占用四分之一 RF。

### Accumulator RF

每 core：

\[
16\times16\times16\times32\text{ bit}=16\text{ KiB}
\]

即：

- 16 个 accumulator slots；
- 每个 slot 是 16×16 FP32。

Matrix 内部流水寄存器另计，不能把它们假装成零面积。

## 5.2 RF Window 是寄存器别名，不是 SPM

建议引入 **RF Window**：

```text
W0 = 一组编译期分配的物理向量寄存器
W1 = 另一组物理向量寄存器
```

window descriptor 包含：

- 物理 RF 起始位置；
- 长度；
- 元素格式；
- 支持的布局；
- 允许访问的角色；
- 静态 stage/fragment 划分。

但是：

- 不提供任意 byte pointer；
- 不提供通用 `load [window + arbitrary_byte_offset]`；
- 不允许任意软件对象驻留其中；
- 不单独增加一块“window SRAM”。

TMA 和 MMA 可以按 window/fragment 操作这些寄存器，普通向量指令仍访问寄存器名字。

**否则它就只是重新命名的 SPM。**

## 5.3 RF 端口的可实现起点

一个相对保守的设计：

- 16 个 bank；
- 每 bank 64-bit；
- 每 bank 1R1W；
- 共 32 KiB。

总理论带宽：

\[
B_\text{read}=128\text{ B/cycle}
\]

\[
B_\text{write}=128\text{ B/cycle}
\]

但这是 bank 无冲突时的聚合值。

例如一条 16-lane FP32 FMA：

- 三个源：192 B；
- 一个结果：64 B。

所以不能保证每周期一条三源 FMA。

第一版可以：

- 用两拍收集三个源；
- 使用操作数锁存寄存器；
- 允许 FMA 与别的独立操作交错。

**SIMD 宽度是 16，不代表任何操作都能每周期吞吐 16 个结果。**

后续如果测得 RF 而非计算单元成为主要瓶颈，再考虑：

- 2R1W bank；
- RF 复制读端口；
- 操作数 bank 分区；
- 专门的 Matrix operand banks。

不要一开始就默认昂贵的多端口 RF。

## 5.4 Accumulator 必须能被向量单元直接访问

这是 FlashAttention 的关键。

建议至少提供：

```text
a.vread      Vd, A_slot, row
a.vwrite     A_slot, row, Vs
a.mul.row    A_slot, row, scale
a.fma.row    A_slot, row, Vx, scale
```

`a.mul.row` 可以是由向量单元执行的小型微序列，不要求另造专用乘法器。

目的：

- online softmax 重缩放 O；
- epilogue；
- bias；
- activation；
- norm 等后处理；

都不需要先把整个 accumulator 写到 NUCA 再读回来。

---

# 六、Matrix 单元：固定小形状，避免 ISA 膨胀

## 6.1 基本指令形状

建议第一版：

```text
m.mma.f16.f32    M16 N16 K16/32/64
m.mma.bf16.f32   M16 N16 K16/32/64
```

语义：

\[
C_{16\times16}
\leftarrow
C_{16\times16}
+
A_{16\times K}B_{K\times16}
\]

M、N 大形状由编译器组合。

K16/32/64 可共用同一个计数器式执行控制器，避免为大量形状设计独立 datapath。

## 6.2 不要忽略 FP32 accumulation 的反馈延迟

一个 16×16 外积阵列即使有 256 个乘法位置，也不代表：

```text
每个 PE 每周期都能更新同一个 FP32 accumulator
```

浮点反馈可能有多周期延迟。

可选处理方式：

1. 多个独立 C fragment 交错；
2. PE 内部交错 partial sums；
3. K 维局部归约后写回 C。

建议第一版使用一个明确的微架构状态机：

```text
CAPTURE
→ MAC_RUN
→ PARTIAL_REDUCE
→ WRITE_C
→ COMPLETE
```

并在 ISA 中明确累加顺序/精度契约。

**不能把“256 MAC/cycle 的算术资源上限”直接当作所有形状的有效吞吐。**

在 16 tile、1 GHz、每 tile 256 MAC/cycle 的理想资源上限下：

\[
P_\text{upper}
=16\times256\times2\times10^9
=8.192\text{ TFLOP/s}
\]

实际吞吐还需乘上阵列、RF、形状和流水利用率。

## 6.3 FP32 dot 的精度不能偷偷降级

Triton 的 dot 接口存在精度相关参数，后端必须尊重对应模式。([triton-lang.org](https://triton-lang.org/main/python-api/generated/triton.language.dot.html))

建议：

- FP16/BF16 → Matrix 快路径；
- FP32 严格路径 → SIMD 或专门精确实现；
- 允许降精度的模式 → 目标相关加速；
- FP64 → 第一版软件展开；
- FP8/INT8/低比特 Matrix → 后续扩展。

“支持一个 dtype”可以是功能支持，但必须诚实区分是否有高速原生 datapath。

---

# 七、TensorMap：同时面向搬运和 Cache，但保持边界清晰

## 7.1 描述符分为三个对象

不要让一个巨大 descriptor 同时承担所有功能。

建议：

### A. Tensor Descriptor：描述数据是什么

- 地址；
- shape；
- stride；
- dtype；
- 边界。

### B. Transfer Configuration：描述这次如何搬

- tile extents；
- RF layout；
- transpose/pack；
- 目标 window；
- fragment 解释。

### C. Cache/Flow Hint：描述预计如何用

- reuse；
- streaming；
- sharing group；
- progress dimension；
- prefetch distance；
- priority。

Triton 已有包含 shape、stride 和 block shape 的 tensor descriptor 接口，可以作为前端语义来源；但你的二进制格式完全可以独立设计。([triton-lang.org](https://triton-lang.org/main/python-api/generated/triton.language.make_tensor_descriptor.html))

## 7.2 建议的 256-byte TensorMap 对象

其中前 128 B 是语义部分。

| 字节范围 | 字段 |
|---|---|
| 0–7 | base address |
| 8–47 | extent[5]，每项 64 bit |
| 48–87 | byte_stride[5]，每项 signed 64 bit |
| 88–107 | 默认 block_shape[5]，每项 32 bit |
| 108 | rank |
| 109 | dtype |
| 110 | padding mode |
| 111 | semantic flags |
| 112–119 | storage object UID |
| 120–127 | descriptor version |
| 128–191 | cache/flow hints |
| 192–255 | 扩展或保留 |

地址公式：

\[
addr(i_0,\ldots,i_{r-1})
=
base+\sum_{d=0}^{r-1}i_d\cdot stride_d
\]

第一版硬件支持 rank 1–5；更高维：

- 可以安全 flatten 的由编译器 flatten；
- 不能 flatten 的走循环或普通向量访存。

## 7.3 Hint 部分建议字段

| 字段 | 用途 |
|---|---|
| tensor_id | 当前执行域中的 tensor 标识 |
| execution_epoch | 防止跨 kernel/跨复用的旧提示混入 |
| use_site_id | 区分同一 tensor 的不同访问位置 |
| reuse_class | streaming / temporal / shared / unknown |
| expected_uses | 预计逻辑 tile 使用次数；允许 unknown |
| sharing_group | 哪些核心/program 共享 |
| progress_axis | 哪个逻辑迭代维度表示进度 |
| live_window_hint | 预计同时活跃 tile 数 |
| cache_priority | 缓存优先级提示 |
| allocation_hint | allocate / no-allocate / adaptive |
| confidence | 推导可信度 |
| prefetch_distance | 预取距离提示 |

**这些字段不改变地址、不改变同步、不改变内存权限。**

## 7.4 Descriptor version 是必需的

要求：

- 指令提交时取得 descriptor 快照或版本；
- 命令执行期间不得看到“前半旧、后半新”的描述符；
- 更新 descriptor 不能改变已经发出的事务；
- 销毁 descriptor handle 前必须处理 outstanding 引用。

这类规则比选择某种 replacement policy 更基础。

## 7.5 物理地址才是 Cache 一致性的身份

同一物理 cache line 可能被：

- 不同 TensorMap；
- 不同 view；
- 普通指针 load；
- 不同 program；

访问。

因此：

> **数据 tag 与 home 选择必须基于物理地址，不能基于 TensorID。**

TensorID 是性能元数据，不是另一套数据身份。

ASID/PASID 用于地址转换与权限管理；如果不同地址空间合法共享同一物理页，也不能因此生成互不一致的数据副本。

---

# 八、NUCA Cache：把 DCO 的思想变成可扩展的机制

DCO 的核心包括 tensor/tile 级元数据、dead-block 判断，以及协调的 anti-thrashing 和 bypass。我的建议是保留这些思想，但把信息来源升级为显式 tensor-transfer 生命周期，而不是依赖某一个 cache line 恰好代表整个 tile 的访问结束。([arxiv.org](https://arxiv.org/html/2512.07312v2))

## 8.1 每个 slice 的基本组织

建议：

```text
Request queues
    │
Predictive request arbiter
    │
Tag pipeline ──────────────── Hit response queue
    │
Miss/merge handling
    ├── MSHR entries
    ├── Target storage
    ├── Fill buffers
    └── Writeback buffers
    │
Data banks
    │
Response/forwarding interface
```

512 KiB、4 banks、8-way、64 B line 时：

- 每 bank 128 KiB；
- 每 bank 256 sets；
- 每 set 8 ways。

可把 96 个 MSHR 分成 4×24，缩小比较和仲裁规模，但必须测量 bank 间不均衡造成的容量浪费。

## 8.2 用显式 tile-use 完成替代“最后地址一定最后完成”

对于 TMA 命令：

```text
logical_use_id = {execution_epoch, tensor_id, tile_coord, use_site}
```

硬件跟踪：

- 该逻辑使用展开了多少实际子请求；
- 哪些已经完成；
- padding 是否完成；
- RF 目标是否全部有效。

当整个逻辑使用完成时，才向 TMU 发一个 use-complete 提示。

这可以正确应对：

- 返回乱序；
- masked tile；
- 边界 padding；
- 不同地址排序；
- 多个 tensor view；
- gather 产生的非连续访问。

普通不规则 load 没有足够信息时，就回退到普通缓存策略。

## 8.3 不要试图精确跟踪所有动态程序的“最后一次使用”

建议三档：

| 情况 | 策略 |
|---|---|
| 仿射、规则、编译器能推导复用 | 使用 expected_uses |
| 部分规则、带动态 mask | 使用保守提示 |
| 数据相关地址/循环 | expected_uses=unknown |

在 unknown 情况下：

- 不使用激进 dead 判断；
- 仍可使用优先级、streaming、负载压力等信息。

**宁愿丢失一点预测收益，也不要让 TMU 变成一个复杂的通用程序分析器。**

## 8.4 Replacement 与 bypass 共享同一个优先级体系

建议优先级由如下信息组成：

```text
reuse class
+ compiler priority
+ sharing status
+ stable tile/line hash tier
+ local pressure adjustment
```

替换顺序：

1. invalid；
2. 已标记 dead 的候选；
3. 低复用/streaming 候选；
4. 低优先级 tier；
5. 同 tier 内按简单年龄选择。

Miss allocation：

```text
若新数据的优先级低于当前准入阈值
    → no-allocate
否则
    → 正常填充
```

准入与替换使用相同排序，避免：

```text
一边高成本装入
另一边马上优先驱逐
```

## 8.5 Bypass 不等于绕过一致性

必须规定：

### Read no-allocate

- 仍到 home slice；
- 仍检查已有 dirty line；
- 仍遵守同地址顺序；
- miss 时才选择不安装新数据。

### Write no-allocate

- 必须与已有 cached copy 协调；
- 不得一边写 DRAM，一边留下旧的 valid cache line；
- 必须有明确的可见性完成点。

### Bypass 也可能占用 miss tracking

不分配 data array，并不意味着：

- 不需要 MSHR；
- 不需要 target；
- 不需要返回缓冲；
- 不需要内存控制器队列。

这是设计性能模型时很容易漏掉的一项。

## 8.6 不做常态化 Cache pinning

第一版不建议：

```text
pin 整个 tensor tile
等待 consumer 完成
再 unpin
```

因为这会导致：

- pinned working set 超容量；
- 替换失去自由度；
- 程序必须证明驻留可满足；
- 多 producer 相互等待；
- 最后重新引入 SPM 式容量管理。

建议只允许：

- 原子操作期间的短暂 line 锁；
- fill/writeback 协议要求的 transient reservation。

**Cache 中的驻留始终是性能属性，不是计算正确性的前提。**

---

# 九、LLaMCAT 思想的升级：控制事务，而不是笼统控制线程

## 9.1 至少区分五种 credit

| Credit | 保护的资源 |
|---|---|
| `new_miss_credit` | 新建 MSHR entry |
| `merge_credit` | 增加 MSHR target |
| `return_credit` | 返回路径和接收缓冲 |
| `rf_destination_credit` | 目标寄存器窗口 |
| `writeback_credit` | 脏数据排出 |

不能只设置一个：

```text
outstanding_requests < N
```

因为“一个新 miss”和“合并到现有 miss”消耗的资源不同。

## 9.2 请求仲裁的推荐顺序

对于已经具备所需资源的请求，建议：

1. 能释放被阻塞资源的事务；
2. 已 aging 到阈值的请求；
3. 预测 cache hit；
4. 有 target 空间的 MSHR merge；
5. demand new miss；
6. speculative prefetch。

其中：

- hit/merge 预测仅影响选择顺序；
- 真正命中情况必须在实际 lookup 后验证；
- merge 也必须检查写/原子操作造成的同地址顺序约束；
- aging 防止 new miss 被无限饿死。

## 9.3 MSHR 满时也应尽量继续服务 hit

不能设计成：

```text
if MSHR_full:
    stall_entire_cache_pipeline
```

建议：

- hit path 与 miss admission 分离；
- 新 miss 无法分配时进入有限 replay/等待队列；
- 不阻塞其他来源的 hit；
- request queue 分流，避免单个 miss 卡住整条 FIFO。

同时要注意：hit 仍消耗 data bank 和 response bandwidth，不是完全免费的。

## 9.4 进度按“共享组”比较，不按全芯片请求数比较

不建议全局比较：

```text
core 0 发了 1000 次请求
core 1 发了 500 次请求
因此 core 0 太快
```

因为：

- 两个核心可能执行不同算子；
- 工作量不同；
- 一个是 producer，一个是 epilogue；
- 请求大小不同。

建议：

```text
group_id
logical_iteration
completed_tile_count
window_id
```

只在同一个共享组内衡量进度差。

例如 GQA：

```text
共享同一 KV head 的若干 query-head programs
        ↓
组成一个 sharing group
        ↓
限制最领先者超前最落后者的 key-block 数
```

这样才是在控制共享工作集，而不是追求没有意义的请求数相等。

## 9.5 两个时间尺度

### 快控制环

每约 64–128 cycles 采样一次，起点可调。

控制：

- 新 miss 注入；
- merge 注入；
- 返回拥塞；
- RF window 可用性。

主要依据：

```text
MSHR entry occupancy
MSHR target occupancy
response queue occupancy
writeback pressure
RF destination availability
```

### 慢控制环

每约 1024–4096 cycles 更新一次，起点可调。

控制：

- prefetch distance；
- group active window；
- 新 program 准入；
- cache allocation threshold；
- sharing-group 进度差。

这些周期只是参数扫描起点，不应直接继承某一工作负载上的最优值。

## 9.6 简单控制律优于复杂学习器

第一版可以用：

```text
if pressure_high 持续多个采样周期:
    new_miss_budget = max(min_budget, new_miss_budget / 2)

elif pressure_low 且消费者缺数据:
    new_miss_budget += 1
```

加上：

- 上下阈值不同；
- 最小保持时间；
- group fairness；
- 最小 forward-progress 配额。

**不要同时快速改变 bypass、prefetch、active programs 和 miss budget。**否则很难解释性能，也容易振荡。

---

# 十、Mesh、数据放置与一致性

## 10.1 第一版采用 static-home NUCA

即：

\[
home=H(\text{physical line address})
\]

特点：

- 同一物理 line 始终有唯一 home；
- 无核心私有 data cache；
- 不做透明 dirty replication；
- 不做运行中 line migration。

这样可以省掉很大一部分目录和迁移协议复杂性。

但仍然需要处理：

- pending fill 与 store；
- writeback 与新访问；
- 同地址 atomics；
- partial write；
- no-allocate 访问；
- DRAM 返回与 line 状态变化。

**没有 L1 coherence，不代表没有一致性协议。**

## 10.2 Home-aware placement 不要依赖可变 TensorID

可以使用：

- 物理页 coloring；
- 固定地址 hash；
- runtime 的分配策略；
- program-to-cluster 亲和性。

例如：

- 一个 2×2 tile 小组处理共享 KV 的 programs；
- 相关页优先映射到该组附近的 home slices；
- 普通 tensor 仍进行均衡 striping。

不要允许两个 descriptor 对同一物理地址计算出不同 home。

## 10.3 Mesh 带宽必须按链路逐条计数

4×4 mesh，中间切面只有 4 条同向链路。

假设每条链路理想 payload：

\[
64\text{ B/cycle}
\]

则 1 GHz 下单向截面上限为：

\[
4\times64\times10^9=256\text{ GB/s}
\]

这不是全芯片所有端口带宽之和。

实际还要扣除：

- packet header；
- 仲裁空洞；
- 转发冲突；
- request/control traffic；
- 不均匀流量。

因此，性能模型必须统计：

\[
\max_{\text{link }l}
\frac{\text{bytes traversing }l}{B_l}
\]

不能只用：

```text
16 slices × 每 slice 带宽
```

估计系统吞吐。

## 10.4 请求与返回必须隔离资源

建议至少区分：

- read request；
- write/writeback data；
- read response；
- completion/control。

可实现为独立 virtual networks 或带独立资源保证的虚拟通道。

路由第一版用 deterministic XY。

但要明确：

> **无环路由只解决网络通道依赖的一部分问题；端点协议仍可能形成死锁。**

路由死锁分析的经典依据是 channel dependency graph，而你的设计还需要加入 cache、RF、writeback 和 completion 的资源依赖。([comp.anu.edu.au](https://comp.anu.edu.au/courses/comp4300/assets/lectures/refs/dallyDeadlockAvoid87.pdf))

## 10.5 五条 forward-progress 规则

1. 发出读取前，目标 RF window 已经合法分配。
2. 返回数据的接收不依赖软件再次执行一条分配指令。
3. completion/control 不被普通预取完全占满。
4. 每个 slice 保留能排出 dirty victim 的资源。
5. 节流不阻止已准入事务完成和资源释放。

第一版先不做网络级 multicast。

先通过：

- home slice 合并；
- 多 target；
- endpoint unicast；

获得正确的基础系统，再评估 multicast 是否值得增加协议复杂性。

---

# 十一、ISA 总体设计

下面给出一套足够支撑功能模拟器、编译器 lowering 和 RTL 分解的 ISA 草案。

## 11.1 ISA 不以完整算子为基本单位

不建议硬编码：

```text
FLASH_ATTENTION
LAYERNORM
SOFTMAX
CONV
```

建议基本单元是：

- scalar；
- vector；
- mask；
- shuffle/reduce；
- tensor transfer；
- matrix fragment；
- accumulator row；
- async completion；
- cache/flow hint。

这样才能应对新算法和不同 fusion。

## 11.2 架构可见状态

| 状态 | 建议 |
|---|---|
| PC | 每 RCTX 一个，64-bit |
| S registers | 每 RCTX 64×64-bit |
| V registers | 每 RCTX 最多 256 个逻辑名字，映射到共享物理池 |
| P registers | 每 RCTX 16×16-bit，P0 为全 true |
| A registers | core 内 16 个 16×16 FP32 slots |
| W handles | 最多 32 个 RF window handle |
| D handles | 最多 16 个 TensorMap 快照 |
| B objects | 最多 32 个 mbarrier |
| Command slots | 64 |
| Special registers | program id、grid size、role id、core id、clock 等 |

V 的物理资源在整个 core 范围内分配，不能四个 RCTX 各自假定拥有 256 个不重叠寄存器。

## 11.3 数据类型

### 第一版高速类型

- predicate；
- i32/u32；
- 地址用 i64/u64；
- FP32；
- FP16/BF16 的 load/store/convert；
- FP16/BF16 Matrix。

### 第一版软件展开类型或操作

- SIMD i64 的部分复杂算术；
- FP64；
- 非原生 FP8 格式；
- 整数除法；
- 高精度 transcendental；
- 非原生 dot 类型。

软件展开必须保持前端要求的语义，不能静默替换为近似低精度算法。

## 11.4 指令长度

建议：

- 基本指令：64 bit；
- 带大立即数、tensor 配置的指令：128 bit；
- 8-byte 对齐；
- 分支目标只能指向指令首字。

基础格式：

```text
63       56 55       48 47       40 39       32
+----------+----------+----------+----------+
| major_op | function |    d     |    a     |
+----------+----------+----------+----------+
31       24 23       16 15  12 11 10 9      0
+----------+----------+------+--+--+---------+
|    b     |    c     | pred | X|!P| modifier|
+----------+----------+------+--+--+---------+
```

其中：

- `X=1`：跟随一个 64-bit extension word；
- operand 的寄存器类型由 opcode 决定；
- modifier 根据 family 表示 dtype、rounding、memory semantics 等；
- 不支持的编码触发 illegal-instruction。

Tensor transfer 扩展字建议：

```text
63:48   transfer_cfg_id
47:32   stream_id
31:16   use_site_id
15:0    policy / extension flags
```

完整 shape/stride 不塞进每条指令，而是引用 descriptor/configuration。

## 11.5 主 opcode 分组

| Major | 指令族 |
|---|---|
| 0x00 | 系统、特殊寄存器、trap |
| 0x01 | Scalar integer |
| 0x02 | 控制流 |
| 0x10 | Vector integer |
| 0x11 | Vector floating point |
| 0x12 | Convert/pack |
| 0x13 | Predicate |
| 0x14 | Shuffle/permute |
| 0x15 | Reduction |
| 0x16 | SFU |
| 0x20 | 普通 Load/Store |
| 0x21 | Atomic/Fence |
| 0x30 | Tensor descriptor |
| 0x31 | Tensor transfer |
| 0x32 | Tensor/cache/flow hints |
| 0x40 | Matrix |
| 0x41 | Accumulator access |
| 0x50 | Mbarrier |
| 0x51 | Role control |
| 0x60 | Debug/performance |

---

# 十二、ISA 详细指令族

以下区分：

- **N**：建议原生；
- **L**：编译器或运行库展开；
- **E**：后续扩展。

## 12.1 Scalar 与控制流

| 指令 | 语义 | 实现 |
|---|---|---|
| `s.mov` | 寄存器复制 | N |
| `s.movi` | 加载立即数 | N |
| `s.add/sub` | 整数加减 | N |
| `s.mul.lo` | 低位乘积 | N |
| `s.mul.hi` | 高位乘积 | N，至少 u32 |
| `s.and/or/xor/not` | 位运算 | N |
| `s.shl/lshr/ashr` | 移位 | N |
| `s.min/max` | 有符号/无符号 min/max | N |
| `s.cmp` | 标量比较 | N |
| `s.select` | 条件选择 | N |
| `s.div/rem` | 除法/余数 | L |
| `s.read.special` | 读取 pid、role、grid 等 | N |
| `br` | 无条件跳转 | N |
| `br.nz/br.z` | 标量条件跳转 | N |
| `call/ret` | 函数调用 | 可 N，常见路径内联 |
| `yield` | 让出发射机会 | N |
| `trap` | 错误上报 | N |
| `exit` | 当前角色结束 | N |

不建议实现通用 SIMT divergence stack。

向量条件通过 mask 处理；uniform 控制流通过 scalar branch 处理。

## 12.2 Vector 整数和浮点

| 指令 | 语义 |
|---|---|
| `v.mov` | 向量复制 |
| `v.splat` | 标量广播 |
| `v.iota` | 生成 lane/连续索引 |
| `v.add/sub/mul` | 算术 |
| `v.mul.hi` | 高位整数乘积 |
| `v.and/or/xor/not` | 位运算 |
| `v.shl/lshr/ashr` | 移位 |
| `v.min/max` | 整数或浮点 min/max |
| `v.fma` | 浮点融合乘加 |
| `v.neg/abs` | 取负/绝对值 |
| `v.cmp` | 产生 predicate |
| `v.select` | 按 predicate 选择 |
| `v.clz/popcount` | 可微序列实现 |

浮点 min/max 必须区分：

- NaN-propagating；
- number-preferred；
- signed-zero 行为。

不要让不同 lowering 随意选用不同语义。

## 12.3 Predicate

| 指令 | 用途 |
|---|---|
| `p.and/or/xor/not` | mask 运算 |
| `p.from.cmp` | 比较结果 |
| `p.any/all` | 转 uniform 条件 |
| `p.ballot` | mask 转位图 |
| `p.from.bits` | 位图转 mask |

普通 masked 运算建议采用：

```text
active lane：写结果
inactive lane：保留旧值
```

但 load 的 `other` 必须按前端语义生成，不能一律假设为零。Triton 的 load 明确区分 pointer 类型、mask、other 和边界处理。([triton-lang.org](https://triton-lang.org/main/python-api/generated/triton.language.load.html))

## 12.4 Convert、pack 与低精度

| 指令 | 用途 |
|---|---|
| `v.cvt` | 数值类型转换 |
| `v.bitcast` | 位解释转换，通常无指令 |
| `v.pack.f16/bf16` | 将多个窄元素打包 |
| `v.unpack.f16/bf16` | 解包并扩展 |
| `v.trunc/extend` | 整数截断、符号/零扩展 |
| `v.cvt.sat` | 饱和转换，语义明确 |
| `v.cvt.fp8.*` | E，或 L |

默认 FP32 基础运算建议采用 RNE；FTZ、近似和降精度必须是显式模式，而不是隐式实现细节。

## 12.5 Shuffle、permute 与布局转换

| 指令 | 用途 |
|---|---|
| `v.shuffle.xor` | butterfly 交换 |
| `v.shuffle.up/down` | 相邻 lane/scan |
| `v.permute` | lane-indexed permutation |
| `v.zip/unzip` | 元素交织 |
| `v.extract/insert` | 小粒度元素或子向量操作 |

不需要为所有 tensor transpose 单独设计指令。

一般布局转换由：

- 寄存器重命名；
- pack/unpack；
- shuffle；
- 多条 permute；
- TMA 目标布局；

组合实现。

较大、不规则的动态布局转换可使用 cache-backed 临时对象兜底，但编译器必须把这条路径标记为昂贵。

Triton 的布局描述本身包含 register、thread、warp 等映射；Linear Layouts 提供了适合复用的统一映射表示。你的后端应复用这种分析思路，而不是为每对 layout 写一套独立规则。([triton-lang.org](https://triton-lang.org/main/getting-started/tutorials/gluon/layouts.html))

## 12.6 Reduction 与 scan

建议原生：

```text
v.reduce.add
v.reduce.max
v.reduce.min
v.reduce.and
v.reduce.or
v.reduce.xor
```

定义为：

- 对一个 16-lane 向量归约；
- 支持 mask；
- inactive lane 使用该运算 identity；
- 结果可广播到所有 lanes。

跨多个 V 寄存器的归约由编译器组合。

下列功能不必做专用硬件：

- tuple reduction；
- argmax/argmin；
- Welford；
- 自定义 associative scan；
- sort/top-k。

它们可以由 compare、select、shuffle 和基础算术展开。Triton 的 reduce/scan 允许通用组合函数，因此只实现 `sum/max` 专用指令并不足以覆盖这些语义。([triton-lang.org](https://triton-lang.org/main/dialects/TritonOps.html))

## 12.7 SFU

建议至少：

```text
v.exp2.approx
v.rcp.approx
v.rsqrt.approx
```

可选：

```text
v.log2.approx
```

其余：

- exp；
- sigmoid；
- tanh；
- erf；
- sin/cos；
- 高精度 division/sqrt；

由运行库展开。

建议从每 core 4-lane SFU 起步，评估 FlashAttention 后再决定是否扩到 8 或 16。

必须发布：

- 正常值误差界；
- NaN/Inf/zero 行为；
- overflow/underflow 行为；
- 是否处理 subnormal。

没有这些契约，编译器无法可靠选择 fast/strict 路径。

## 12.8 普通 Load/Store

这是功能完备性的底座，不能删除。

```text
s.ld / s.st
v.ld.contiguous
v.st.contiguous
v.ld.gather
v.st.scatter
```

建议：

- 全局地址 64-bit；
- mask=false 的 lane 不产生数据内存访问；
- gather 地址可以是 base+offset，也可以是完整地址；
- coalescer 按 physical cache line 合并；
- response 能将子结果写回正确 lane；
- 非对齐访问由 LSU 拆分；
- partial store 保留未写字节。

第一版可让普通 vector load 对所属 RCTX 表现为 scoreboard 等待；不必所有普通 load 都暴露显式 async token。

**TMA 是规则大块访问的快路径，不是唯一访存路径。**

## 12.9 Atomics 与 fence

建议原生 home-slice 操作：

```text
atom.cas.u32/u64
atom.xchg.u32/u64
atom.add.u32/u64
atom.and/or/xor.u32/u64
atom.min/max.s32/u32
```

FP32 atomic add：

- 初版可以 CAS loop；
- 后续增加 home-side FP32 atomic 单元。

另外需要：

```text
atomic.load
atomic.store
fence.acquire
fence.release
fence.acq_rel
```

作用域至少：

- program/core；
- device。

Triton atomics 的 memory semantics 和 scope 是接口的一部分，不能只实现“原子更新数值”而忽略其顺序含义。([triton-lang.org](https://triton-lang.org/main/python-api/generated/triton.language.atomic_cas.html))

`system` scope 是否支持必须成为设备 capability：

- 有系统一致性/原子接口才能真正承诺；
- 普通非一致性 PCIe DMA 加一次 flush，不等同于支持任意 host/device 并发 system atomics。

## 12.10 Tensor descriptor 指令

```text
td.load       Dd, [descriptor_address]
td.create     Dd, scalar_fields...
td.update     Dd, field, value
td.release    Dd
```

规则：

- `td.load` 建立版本化快照；
- update 不改变旧命令；
- release 要等待引用清空或延迟回收；
- metadata table 满时可以等待或走普通 LSU，不能静默覆盖活跃 descriptor。

## 12.11 Tensor transfer 指令

核心指令：

```text
t.ld.async
t.st.async
t.prefetch
```

推荐形式：

```text
t.ld.async Wdst, Dsrc, Scoord_base, Stoken, CFG
```

语义：

1. 读取 Dsrc 快照；
2. 从连续标量寄存器取得各维坐标；
3. 计算访问；
4. 检查边界；
5. 展开为 cache-line 请求；
6. 将返回值按 CFG 写入 Wdst；
7. 完成 padding/pack；
8. 完成一次与 Stoken 关联的异步操作。

其他指令：

| 指令 | 建议 |
|---|---|
| `t.prefetch` | N，纯性能操作，不保证未来驻留 |
| `t.copy` | L，使用寄存器窗口分块搬运 |
| `t.gather.rows` | E，初版可展开为多次搬运/LSU |
| `t.scatter.rows` | E |
| `t.reduce.store` | L 到 atomics；后续 E |
| `t.im2col.load` | E，不作为第一版必要条件 |

原生 descriptor load/store、gather/scatter/reducing store 等在 Triton IR 中有不同操作，应分别 lowering，而不是假设所有 descriptor 操作都是普通二维 memcpy。([triton-lang.org](https://triton-lang.org/main/dialects/TritonOps.html))

## 12.12 Matrix 与 accumulator 指令

```text
m.zero
m.mma.async
m.wait             // 可作为 mb.wait 的汇编别名

a.vread
a.vwrite
a.mul.row
a.fma.row
```

`m.mma.async` 应明确：

- 源 window/fragment；
- 目标 accumulator slot；
- M/N/K 形状；
- 输入类型；
- accumulation 模式；
- 完成 token。

没有完成之前：

- 不允许覆盖仍被读取的输入；
- 不允许向量指令读取尚未完成的 accumulator；
- 不允许把该 accumulator 分配给下一个 program。

## 12.13 Cache 与流量提示

建议只有少量：

```text
tm.bind
tm.unbind
tm.use_hint
flow.mark
flow.prefetch_hint
```

其中：

```text
flow.mark group, logical_iteration
```

表示一个逻辑进度点。

不要提供大量暴露物理 Cache 实现的指令，例如：

```text
必须放到第 3 slice 第 7 way
永久锁住某 set
直接修改 MSHR quota
```

这些会把编译器与某一具体微架构绑死。

---

# 十三、异步与 mbarrier：必须先写成精确协议

这是整个设计最应该先冻结的部分。

NVIDIA 的 TMA/mbarrier 给出了有价值的参考：异步操作与 arrival/transaction 完成共同驱动 barrier；但你的寄存器目标和无 SPM 架构不需要二进制兼容它的 shared-memory 对象格式。([docs.nvidia.com](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html))

## 13.1 三种完成不能混淆

必须区分：

### Load-ready

数据已经全部写入目标 RF，后续消费者可以读取。

### Compute-done

Matrix/SIMD 的结果已经完成，输出可以读取。

### Source-reusable

某操作已经不再读取源 RF，producer 可以覆盖它。

对于 store，还可能区分：

### Store-visible

写入已经到达定义的 device 可见性点。

第一版可以保守合并部分事件，例如：

```text
store 的单一 done
= 源已经不再使用 + 所有写在 home 可见
```

这会多占用一点 RF，但协议简单。

后续再增加双完成事件优化。

## 13.2 Barrier 对象

建议每个 barrier 包含：

```text
current_epoch
completed_epoch
pending_arrivals
pending_operations
waiter_mask
error_status
```

不是只有一个 phase bit。

建议 token：

```text
token = {epoch, barrier_id}
```

可以放进一个 64-bit 标量寄存器。

epoch 回绕只能在完全 quiescent 时处理。

## 13.3 指令

```text
mb.init
mb.arm
mb.arrive.release
mb.test
mb.wait.acquire
mb.invalidate
```

建议第一版：

```text
mb.arm Stoken, B, arrivals, operations
```

一次性确定该 generation 的计数。

相比允许任意时刻增加 expected bytes，这更容易验证。

计数单位定义为：

> **一个完整的异步命令，而不是一个 cache line，也不是不明确的字节数。**

例如：

- 两条 `t.ld.async` → operations=2；
- 一条 MMA → operations=1；
- 每条命令自己跟踪内部所有子事务。

## 13.4 完成条件

\[
pending\_arrivals=0
\land
pending\_operations=0
\]

则 barrier 完成。

要求：

- completion exactly once；
- 不能 underflow；
- 错误必须唤醒 waiter 并上报；
- 不能让错误命令导致无限等待；
- `mb.wait` 让 RCTX sleep，不做高频忙轮询。

整体谓词为 false 的异步命令建议产生一次 no-op completion，目标不变。这样静态 operation count 不会因 predication 丢失，但编译器仍必须保证消费者控制流正确。

## 13.5 双缓冲必须使用 ready/free 两组事件

对每个 slot：

```text
FREE
  ↓ producer 获得使用权
FILLING
  ↓ TMA 完成
READY
  ↓ consumer 开始
IN_USE
  ↓ consumer 不再读取
FREE
```

只实现 READY，不实现 FREE，是最常见的错误之一。

因为：

```text
TMA 完成
≠
MMA 已经读取完输入
```

## 13.6 GEMM 双角色流水示意

下面是语义伪汇编；`B@epoch` 表示编译器生成的 generation token。

### 初始化

```text
两个 Empty barrier 的 generation 0 已完成
两个 Ready barrier 已初始化
两个 RF slot 已分配
```

### Producer

```text
for k_tile in 0 .. K_tiles-1:
    slot  = k_tile & 1
    epoch = (k_tile >> 1) + 1

    mb.wait.acquire Empty[slot]@(epoch-1)

    mb.arm Empty[slot]@epoch, arrivals=1, operations=0
    mb.arm Ready[slot]@epoch, arrivals=1, operations=2

    t.ld.async WA[slot], DA, coordA(k_tile),
               Ready[slot]@epoch, CFG_A

    t.ld.async WB[slot], DB, coordB(k_tile),
               Ready[slot]@epoch, CFG_B

    mb.arrive.release Ready[slot]@epoch
```

### Consumer

```text
for k_tile in 0 .. K_tiles-1:
    slot  = k_tile & 1
    epoch = (k_tile >> 1) + 1

    mb.wait.acquire Ready[slot]@epoch

    mb.arm ComputeDone, arrivals=0, operations=N_mma

    for each output fragment:
        m.mma.async A_fragment,
                    WA[slot].fragment,
                    WB[slot].fragment,
                    ComputeDone

    mb.wait.acquire ComputeDone

    mb.arrive.release Empty[slot]@epoch
```

这是保守而正确的第一版。

后续可以把“输入读取完”和“完整 MMA 结果完成”分开，让 Empty 更早到达，但必须显式增加协议，而不是猜测固定若干周期后可以覆盖输入。

---

# 十四、内存模型：尽量简单，但不能含糊

## 14.1 普通内存

建议：

- 同一物理地址由 home slice 串行化冲突访问；
- 不同地址允许乱序；
- 不自动保证所有普通 load/store 的全局顺序；
- acquire/release 显式建立同步关系。

不要不加限定地宣称“所有程序都顺序一致”。

## 14.2 RCTX 间通信

通过：

```text
producer 写 RF window
→ release completion
→ consumer acquire wait
→ consumer 读 RF window
```

建立 happens-before。

普通 scalar/vector 指令完成、异步操作完成、内存可见性必须按协议分别定义。

Triton 的异步等待与 CTA 同步也不是同一个概念，后端不能把所有 wait 都替换成一个模糊的“全核心 barrier”。([triton-lang.org](https://triton-lang.org/main/dialects/TritonGPUOps.html))

## 14.3 跨 core 同步

第一版建议：

- local mbarrier 只覆盖同 core 的 program；
- 跨 core 用 device atomics/fence；
- 大范围阶段同步优先拆 kernel；
- 不支持普通 grid 任意全局 barrier。

如果未来支持 cooperative cluster：

- 必须保证所有参与者同时 resident；
- 必须有明确 group admission；
- 不能让尚未调度的 program 成为已占满芯片的 program 的 barrier 前驱。

## 14.4 Host 可见性

若 Host 与 accelerator 不硬件一致：

```text
kernel completion
→ device write drain
→ 必要的 cache clean
→ host notification
```

这是 runtime/driver 协议的一部分。

CPU 不能在没有 ownership/synchronization 协议时随意读写正在被 accelerator 使用的同一 buffer。

---

# 十五、Triton/MLIR 编译器方案

## 15.1 推荐复用前端，不照搬后端

现有 Triton backend 已经将 TTIR、TTGIR 和后续目标代码生成分阶段组织，因此你不需要先实现一个完整 LLVM target，才能开始做新硬件后端。可以先增加自有 lowering 和机器码发射器。([github.com](https://github.com/triton-lang/triton/blob/main/third_party/amd/backend/compiler.py))

建议流水：

```text
Triton Python
    ↓
TTIR
    ↓
规范化 / 类型合法化 / 控制流简化
    ↓
Tensor access analysis
    ↓
Layout selection
    ↓
TMA formation
    ↓
Role partition
    ↓
Async pipeline construction
    ↓
RF / accumulator / barrier allocation
    ↓
Cache-flow metadata generation
    ↓
TNCA machine IR
    ↓
Instruction scheduling / register allocation
    ↓
Binary + kernel resource manifest
```

## 15.2 新增一个小型 TNCA dialect

建议操作：

```text
tnca.tensor_map
tnca.rf_window
tnca.tensor_load_async
tnca.tensor_store_async
tnca.mma_async
tnca.acc_read
tnca.acc_write
tnca.barrier_arm
tnca.barrier_wait
tnca.barrier_arrive
tnca.role_region
tnca.progress
```

关键属性：

```text
layout
rf_lifetime
async_read_set
async_write_set
completion_token
reuse_group
logical_use_id
```

异步操作必须正确声明 memory effects 和资源生命周期，否则普通 CSE、LICM、DCE 就可能破坏程序。

## 15.3 六个最关键的 pass

### Pass A：访问识别

将访存分为：

1. 连续块；
2. 多维仿射；
3. 规则行 gather；
4. 完全动态 gather/scatter；
5. atomic。

只有前几类进入 TMA 快路径。

### Pass B：布局合法化

目标是：

- vector layout；
- matrix operand layout；
- accumulator layout；
- TMA destination layout；

之间有明确转换。

优先消除不必要转换，但不能把需要搬动数据的 transpose 当作 metadata-only。

### Pass C：角色划分

依据依赖图划分：

- load；
- matrix；
- vector/reduction；
- store。

不必总是四份。

### Pass D：异步流水构造

同时选择：

- prefetch lookahead；
- RF stage 数；
- barrier；
- window 所有权；
- command 数。

### Pass E：RF 生命周期与分配

需要统计：

```text
producer 正在填充的 window
consumer 正在读取的 window
尚未完成的 accumulator
vector 临时值
descriptor/token 标量
```

异步源的生命周期延续到 source-reusable，而不是 instruction issue。

### Pass F：Cache/flow 元数据

从最终 tiling 和 schedule 推导：

- expected uses；
- sharing group；
- progress axis；
- streaming；
- reuse distance 的粗略类别。

**必须在 tiling 和 role/pipeline 基本确定后生成。**

## 15.4 TTGIR 的 shared-memory 操作如何处理

这是最大兼容性问题之一。

`ttg.local_alloc/local_load/local_store` 确实表达 shared-memory 对象；不能直接把它们全部解释成“引用原始 global 数据的 cache line”。([triton-lang.org](https://triton-lang.org/main/dialects/TritonGPUOps.html))

建议三条路径：

### 路径一：优化掉共享对象

如果只是规则 staging：

```text
global → shared → registers
```

可以改为：

```text
global/NUCA → RF window
```

### 路径二：寄存器化

若对象：

- 小；
- 访问静态可分析；
- 无复杂动态索引；

则变成寄存器和 shuffle。

### 路径三：cache-backed 临时对象

为每个 program 分配普通全局地址范围：

```text
global addressable temporary
    ↕
NUCA
```

它：

- 有 DRAM backing；
- 可被替换；
- 正确性不依赖驻留；
- 不是片上 SPM。

这是兼容性兜底路径，可能很慢。

**尤其不能忽略 copy 的快照语义。**

如果原程序：

```text
copy global A → local B
随后修改 A
然后读取 B
```

把 B 直接别名到 A 就是错误的。只有证明整个相关生命周期只读，才能做这种消除。

## 15.5 Triton 功能覆盖表

以下是目标后端方案，不是声称现有 TNCA 软件已经存在。

| Triton 功能 | TNCA lowering |
|---|---|
| `program_id/num_programs` | special registers |
| `arange/full/splat` | iota、常量、广播 |
| `reshape/expand_dims` | 优先 metadata/寄存器重解释 |
| `trans` | layout 变换或 shuffle |
| `load/store` | TMA 或普通 LSU |
| block pointer | descriptor/地址分析 |
| tensor descriptor | 原生 TensorMap |
| `dot` | Matrix 或 vector fallback |
| `sum/max/min` | reduction |
| 自定义 reduce | shuffle+组合函数 |
| associative scan | 分层扫描 |
| gather/scatter | LSU 或规则 TMA 扩展 |
| atomics | home-slice atomic |
| sort/top-k | compare-exchange 展开 |
| histogram | 小规模寄存器或临时对象+atomic |
| random | 整数算术实现 |
| exp/log/sigmoid 等 | SFU+数学库 |
| device assert/print | trap/debug queue |
| inline asm | TNCA 目标版本；不接受任意 PTX |
| warp-specialize 区域 | role partition |
| 目标特定 multi-CTA | capability 检查或专门 lowering |

上述前端操作类别在 Triton API 中已有明确区分；硬件无需为每个 API 提供一条同名指令。([triton-lang.org](https://triton-lang.org/main/python-api/triton.language.html))

## 15.6 不要承诺任意数量 worker partitions 自动原生映射

如果一个目标特定程序需要超过四个同时活动的 partition：

- 能证明可合并时合并；
- 能生成 continuation/state-machine 时软件化；
- 否则给出明确不支持诊断。

不能把它们简单顺序执行，因为其中可能有互相等待的 barrier。

---

# 十六、主要算子的映射与真实瓶颈

## 16.1 GEMM

建议从：

```text
BLOCK_M = 64
BLOCK_N = 64
BLOCK_K = 32
```

探索。

BF16/FP16 下：

\[
A=64\times32\times2=4096\text{ B}
\]

\[
B=32\times64\times2=4096\text{ B}
\]

双缓冲输入：

\[
2(A+B)=16\text{ KiB}
\]

FP32 accumulator：

\[
C=64\times64\times4=16\text{ KiB}
\]

因此：

- 输入适合放 Vector RF；
- 输出刚好占满 Accumulator RF。

每个 K tile 的计算量：

\[
F=2\times64\times64\times32=262144
\]

若按每 core 512 FLOP/cycle 的理想算术上限，至少 512 cycles。

输入供数需求：

\[
\frac{8192}{512}=16\text{ B/cycle}
\]

这说明：

> **通过 RF 内复用，可以把 NUCA 的需求从矩阵逐周期输入的 64 B/cycle，降低到这个 tile 下的理想 16 B/cycle。**

这正是无 SPM 设计能够成立的关键。

### 但有一个容量问题

64×64 C 占满 16 KiB accumulator：

- 下一块 C 不能同时完整双缓冲；
- epilogue 可能阻塞下一块 GEMM。

备选：

```text
BLOCK_M=32, BLOCK_N=64
```

用两个 8 KiB accumulator 集合交错。

代价是输入复用下降。

因此 tile 选择应该同时优化：

- NUCA 流量；
- accumulator 容量；
- epilogue 重叠；
- RF bank 冲突；

而不是只追求最大的 M/N。

---

## 16.2 FlashAttention

建议第一版从较小 query block 开始：

```text
BLOCK_M = 16
BLOCK_N = 16
HEAD_DIM = 128
```

采用 online softmax：

\[
m'=\max(m,\max_j s_j)
\]

\[
\alpha=e^{m-m'}
\]

\[
p_j=e^{s_j-m'}
\]

\[
l'=\alpha l+\sum_j p_j
\]

\[
O'=\alpha O+PV
\]

最后：

\[
O_{\text{final}}=O/l
\]

### 资源估算

| 数据 | 大小 |
|---|---:|
| Q，16×128 BF16 | 4 KiB |
| K，16×128 BF16 | 4 KiB |
| V，16×128 BF16 | 4 KiB |
| 双缓冲 K/V | 16 KiB |
| 两份 P，16×16 BF16 | 1 KiB |
| 两份 score，16×16 FP32 | 2 KiB |
| O，16×128 FP32 | 8 KiB |
| m/l | 128 B |

可将：

- Q、K/V、P、向量临时值放 Vector RF；
- O 和 score 放 Accumulator RF。

这个形状在前述资源配置下有可行性，但仍需检查具体布局和 bank 冲突。

### 四个角色的推荐分工

```text
R0：K/V prefetch + TMA
R1：QK^T / PV 的 Matrix 调度
R2：mask + max + exp2 + sum + O rescale
R3：辅助搬运 / epilogue / 下一任务控制
```

**不建议把 QK 和 PV 永久绑定到两个不同 Matrix 单元。**第一版一个共享 Matrix engine 即可。

### 真正的瓶颈

不是只有 QK 和 PV：

1. score 从 accumulator 到向量域；
2. exp2 吞吐；
3. 行归约；
4. O 重缩放；
5. softmax 与 Matrix 的依赖；
6. 多个 score/P buffer 的存活区间。

例如 O 是 8 KiB，如果每个 key block 都：

```text
Accumulator → NUCA → Vector → NUCA → Accumulator
```

这套架构基本失去意义。

所以前面提出的 accumulator-row 向量访问是必要能力。

FlashAttention-3 的一个重要设计方向就是异步搬运/矩阵计算与 softmax 的重叠；你的方案应该保留这种依赖图级重叠，而不必复刻 Hopper 的 shared-memory datapath。([arxiv.org](https://arxiv.org/abs/2407.08608))

### 特别注意两类 Attention

- **Prefill**：Matrix、RF、SFU 可能共同限制。
- **Decode/GQA**：外存、MSHR、target 合并、共享组进度更重要。

不能用一个统一的“Attention 最优节流配置”覆盖两者。

---

## 16.3 LayerNorm / RMSNorm

不要假设存在一个大型专用 norm 单元。

使用：

```text
TMA/LSU load
→ vector sum / sumsq 或 Welford
→ reduction
→ rsqrt
→ affine transform
→ store
```

两条路径：

### 行能放进 RF

保留输入，完成归约后直接归一化。

### 行不能放进 RF

两遍处理：

1. 第一遍归约；
2. 第二遍重新读取并归一化。

第二遍能否命中 NUCA，是 Cache 管理的重要机会。

但不能为了“保证一遍读取”，把整个行强行 pin 在 Cache。

Welford 等 tuple reduction 应由通用组合 lowering 支持，而不是必须增加专用 ISA。

---

## 16.4 Softmax

小行：

```text
RF 内 max
→ exp2
→ sum
→ reciprocal
→ scale
```

大行：

- 多 pass；
- 或分块计算统计量后重读；
- 由 Cache 捕获第二次访问。

这里通常需要的是：

- exp2；
- reduce；
- RF 带宽；

而不是更多 Matrix 峰值。

---

## 16.5 Conv

第一版不要急着加入复杂 im2col TMA。

先做：

- 1×1 Conv → GEMM；
- 常规 Conv → implicit GEMM；
- stride/dilation/padding → scalar/vector 地址生成；
- 不规则情况 → gather；
- depthwise → vector 路径。

后续若发现大量时间消耗于规则 convolution 地址生成，再增加：

```text
t.im2col.load
```

这应是性能扩展，不是功能正确性的前提。

---

## 16.6 Sigmoid 与普通 elementwise

推荐：

- 单 RCTX 或双 RCTX；
- 普通向量访存；
- 可选较小 TMA；
- 不强制建立四角色流水。

sigmoid：

\[
\sigma(x)=\frac{1}{1+e^{-x}}
\]

主要由 exp2、加法和 reciprocal 构成。

这里最容易犯的错误是：

> 为了展示 TensorMap 和角色专门化，让一个本来很简单的 kernel 付出过大的 setup/barrier 开销。

---

# 十七、预取策略：Cache lookahead 与 RF buffering 分开

这是我非常推荐的一个设计。

## 17.1 两种不同的距离

### NUCA prefetch distance

提前把未来数据带到 Cache：

```text
未来若干个逻辑 tile
```

### RF pipeline depth

真正占用寄存器的：

```text
当前 tile + 下一 tile
```

例如：

```text
NUCA lookahead：4–8 个 tile
RF stages：2
```

这不是增加存储层次，而是在已有的 Cache 与 RF 上使用不同时间尺度。

## 17.2 为什么有用

如果全部延迟都靠 RF 双缓冲掩盖：

- RF 要随着 DRAM latency 增大；
- 容量很快失控。

如果较远的预取先吸收 DRAM latency，真正的 TMA-to-RF 主要面对 NUCA hit latency，就可以使用较小 RF stage 数。

但：

> `t.prefetch` 完成不保证之后仍然命中，也不能建立数据快照或所有权。

Cache 压力高时：

- 缩短 lookahead；
- 丢弃低价值 prefetch；
- 优先 demand；
- 避免把热点活跃数据挤出去。

## 17.3 流水深度约束

保守估计：

\[
S_{\text{RF}}
\gtrsim
1+\left\lceil\frac{L_{\text{ready}}}{T_{\text{iteration}}}\right\rceil
\]

同时必须满足：

\[
S_{\text{RF}}\cdot B_{\text{input tile}}
+
B_{\text{live vector}}
\leq C_{\text{VRF}}
\]

以及：

\[
B_{\text{live accumulators}}\leq C_{\text{ARF}}
\]

不能只根据延迟自动增加 `num_stages`。

---

# 十八、运行时与 Kernel ABI

## 18.1 Kernel manifest

每个 kernel binary 携带：

```text
ISA version
required capabilities
entry PC per role
active role mask
grid dimensions
argument layout
RF allocation plan
RF window table
accumulator usage
barrier count
descriptor count
maximum command slots
spill bytes per program
preferred sharing group
preferred placement
initial prefetch / injection hints
```

## 18.2 Launch admission

调度器在启动 program 前检查：

- RF 是否足够；
- accumulator 是否足够；
- barrier 是否足够；
- descriptor slots 是否足够；
- 是否满足最小进展资源。

不要先启动一半角色，再等待永远不足的另一半角色资源。

## 18.3 Program 结束

`exit` 不能直接让资源立即复用。

Program complete 必须满足：

- 所有角色结束；
- 所有必须完成的 async commands 已完成；
- 输出满足 device completion 语义；
- 不会再有返回数据写入旧 RF；
- descriptor/window 引用已经回收。

返回包必须带有足够的 transaction/job generation 信息，避免旧响应误写入被重新分配的 RF。

## 18.4 地址转换

第一版可以选择：

- runtime 固定/pin buffer；
- 物理地址或简单 IOMMU 映射；
- 不支持执行中 demand paging。

地址错误：

- 上报 kernel fault；
- 完成错误传播；
- 安全回收或排空相关事务。

不要一开始加入可恢复 page fault、抢占和迁移，除非产品需求明确要求。

---

# 十九、Verilog RTL 的模块拆分

建议全部计算与控制 RTL 可以用 Verilog-2001 编写；验证环境可以独立使用更丰富的语言。

```text
tnca_top.v
├── host_if.v
├── command_processor.v
├── grid_scheduler.v
├── resource_admission.v
├── memory_controller_if.v
└── tile.v
    ├── role_frontend.v
    ├── instruction_decode.v
    ├── issue_scheduler.v
    ├── scalar_rf.v
    ├── scalar_alu.v
    ├── vector_rf_bank.v
    ├── vector_rf_crossbar.v
    ├── vector_alu.v
    ├── vector_shuffle.v
    ├── vector_reduce.v
    ├── sfu.v
    ├── matrix_engine.v
    ├── accumulator_rf.v
    ├── accumulator_vector_port.v
    ├── scoreboard.v
    ├── rf_window_table.v
    ├── tensor_descriptor_table.v
    ├── tensor_address_generator.v
    ├── tensor_transfer_engine.v
    ├── load_store_unit.v
    ├── coalescer.v
    ├── barrier_table.v
    ├── completion_unit.v
    ├── flow_controller.v
    ├── network_interface.v
    ├── mesh_router.v
    └── nuca_slice.v
        ├── request_queues.v
        ├── request_predictor.v
        ├── request_arbiter.v
        ├── tag_array.v
        ├── data_bank.v
        ├── mshr_bank.v
        ├── target_pool.v
        ├── fill_buffer.v
        ├── writeback_buffer.v
        ├── atomic_unit.v
        ├── slice_tmu.v
        └── replacement_admission.v
```

## 19.1 所有跨模块通路使用明确握手

```verilog
valid
ready
payload
```

规定：

- `valid && !ready` 时 payload 保持稳定；
- 不允许跨多个 router 形成组合 ready 链；
- packet/class 有独立 buffering；
- credit 更新 exactly once。

## 19.2 Tensor DMA 状态机

```text
IDLE
→ CAPTURE_DESCRIPTOR
→ VALIDATE_DESTINATION
→ EXPAND_COORDINATES
→ ISSUE_SUBREQUESTS
→ RECEIVE_AND_WRITE_RF
→ PAD_REMAINING_ELEMENTS
→ COMPLETE
```

允许多个命令处于不同阶段，但每个命令有独立：

- version；
- destination；
- outstanding count；
- error；
- completion target。

## 19.3 Cache line 事务状态机

建议先写清：

```text
INVALID
VALID_CLEAN
VALID_DIRTY
MISS_PENDING
FILL_PENDING
WRITEBACK_PENDING
ATOMIC_LOCKED
```

不必照抄 MESI，但每个 transient state 必须定义：

- 哪些新请求允许进入；
- 哪些可以 merge；
- 哪些必须等待；
- 返回数据如何处理；
- 错误和取消如何处理。

## 19.4 不建议第一版加入的功能

- 通用乱序执行；
- 大量硬件线程；
- dirty cache replication；
- transparent line migration；
- adaptive routing；
- 网络 multicast；
- 通用压缩缓存；
- 动态任意形状硬件重排；
- 稀疏 Matrix ISA；
- demand paging；
- 任意点抢占；
- FP8/FP4 专用矩阵 datapath；
- 强依赖 cache pinning 的执行。

这些功能都可能有价值，但不是验证你的核心思想所必需的。

---

# 二十、性能模型与验证方案

## 20.1 不能只做一个简单 Roofline

建议至少建立以下下界：

\[
T\geq
\max\left(
\frac{F_{\text{mma}}}{P_{\text{mma}}},
\frac{F_{\text{vec}}}{P_{\text{vec}}},
\frac{N_{\text{sfu}}}{P_{\text{sfu}}},
\frac{D_{\text{DRAM}}}{B_{\text{DRAM}}},
\max_b\frac{D_b}{B_b},
\max_l\frac{D_l}{B_l},
\frac{D_{\text{RF}}}{B_{\text{RF}}},
T_{\text{critical-path}}
\right)
\]

但这只是下界，不是直接的执行时间公式。

还需要：

- 数据依赖 DAG；
- barrier；
- RF/window 生命周期；
- 有限队列；
- bank 冲突；
- MSHR entry/target；
- 注入与返回网络；
- 实际 overlap。

如果两个操作必须串行，不能因为使用不同硬件单元就假定完全重叠。

## 20.2 关键性能计数器

### Core

- RCTX ready/blocked cycles；
- RF read/write bank conflict；
- Matrix operand starvation；
- accumulator dependency stall；
- SFU busy；
- reduction busy；
- barrier wait cycles；
- window unavailable cycles。

### Cache

- hit；
- new miss；
- MSHR merge；
- entry-full；
- target-full；
- fill blocked；
- writeback blocked；
- no-allocate；
- dead-hint victim；
- prefetch useful/useless；
- bytes served per sharing group。

### Network

- 每条 link 利用率；
- 每个 VN/VC 队列占用；
- injection stall；
- response stall；
- average/max hop count；
- packet/header overhead。

### 流量控制

- 每个 group 的进度差；
- 每个 credit 类别的限制时间；
- controller gear 变化次数；
- 同一周期内相互冲突的控制动作。

## 20.3 必须做的正确性测试

1. 全 false mask，不产生越界访问。
2. 部分边界 tile，padding 正确。
3. 任意子请求返回顺序。
4. 同一个物理地址通过多个 TensorMap 访问。
5. 普通 LSU 与 TMA 交错读写。
6. no-allocate 读取已有 dirty line。
7. dirty victim 与新 fill 竞争。
8. MSHR entry 满但 hit 仍能进展。
9. MSHR target 满但其他独立请求能进展。
10. DMA 回包遇到 RF bank 冲突。
11. barrier stale generation。
12. predicated-off async command。
13. descriptor 更新与旧命令并行。
14. program 退出后迟到响应。
15. 所有软 hint 关闭、延迟或丢失后，语义仍正确。
16. 异常事务能唤醒等待者。
17. 四个角色构成循环资源压力时仍有明确进展路径。
18. 临时对象被 Cache 驱逐后仍可正确重读。

浮点和 atomics 测试应按允许的数值/执行语义比较，不能要求所有不同调度都得到完全相同的浮点累加位模式。

## 20.4 消融实验

至少比较：

```text
普通 NUCA
+ TensorMap
+ dead-block hints
+ coordinated bypass/replacement
+ MSHR-aware arbitration
+ per-source credits
+ group progress control
+ placement affinity
+ RF-aware compiler scheduling
```

另外还要有公平的：

```text
RF + SPM + Cache
```

参考架构。

否则无法回答：

> 你的收益来自更好的 cache/flow 协同，还是仅仅来自更多 RF、更多 MSHR、更多端口或更好的编译器？

---

# 二十一、建议的实现顺序

## 阶段 1：先证明功能，不先做 16 核

配置：

- 1 core；
- 1 cache slice；
- 1–2 RCTX；
- 16-lane SIMD；
- 普通 load/store；
- 基础 atomics；
- barrier；
- 不带预测的普通 Cache。

跑通：

- elementwise；
- reduction；
- LayerNorm；
- vector GEMM。

这一阶段验证 ISA 和内存语义。

## 阶段 2：加入 Tensor DMA、RF Window、Matrix

跑通：

- 异步 GEMM；
- 双缓冲；
- accumulator epilogue；
- 小形状 FlashAttention。

重点验证：

- RF 所有权；
- source lifetime；
- barrier generation；
- 无 SPM 的 layout lowering。

## 阶段 3：2×2 mesh

加入：

- 多 slice；
- home routing；
- MSHR entry/target；
- request/response 分离；
- forward-progress credits。

先不加 DCO/CAT 动态策略，建立可靠基线。

## 阶段 4：逐项融合论文思想

按顺序：

1. TensorMap metadata；
2. use-complete；
3. dead hints；
4. 协调 replacement/bypass；
5. MSHR-aware 仲裁；
6. request admission；
7. sharing-group 进度控制。

每加入一项都做独立消融。

## 阶段 5：扩展到 4×4，优化编译器与布局

这时再决定：

- RF 是否需要更多端口；
- Matrix 是否需要更大；
- SFU 是否需要扩宽；
- multicast 是否值得；
- Cache 容量是否真的有效；
- mesh 是否才是主要限制。

---

# 最后：我会优先冻结的十二个决定

如果现在就准备开始设计，我建议先确定：

1. **数据层次只有 RF—NUCA—DRAM，不新增可寻址 Tensor SRAM。**
2. **每 core 最多四个 RCTX，每个是向量控制上下文，不是标量线程。**
3. **规则搬运走 Tensor DMA，非规则访问保留完整 LSU。**
4. **Matrix 从 RF 取数，不直接依赖远端 Cache 逐周期供数。**
5. **Vector RF 与 accumulator RF 分离，但 accumulator 可被向量单元高效访问。**
6. **TensorMap 的正确性字段与缓存/流控提示严格分离。**
7. **Cache 采用唯一 static home，第一版不做复制和迁移。**
8. **节流新事务准入，不暂停完成路径和资源释放路径。**
9. **同时管理 MSHR entry、target、return、RF 和 writeback credit。**
10. **ready 与 free 两类事件都进入 ISA/编译器生命周期模型。**
11. **从 TTIR/早期布局分析建立新后端，不照搬依赖 shared memory 的末端 TTGIR lowering。**
12. **先完成小规模可验证系统，再扩展 mesh 和动态控制。**

**最值得做成你这颗芯片核心特征的，不是“Cache 能读 TensorMap”这一件事，而是：**

> **编译器声明 tensor 的访问与生命周期；Cache 决定保留什么；流控决定让哪些访问进入系统；RF/barrier 协议保证已经进入系统的工作能够完成。**

把这四者设计成一个一致的执行模型，才有机会在取消 SPM 的同时保住 GEMM 和 FlashAttention 的效率，并保留 Triton 所需要的通用性。

*Thinking...*

> **Evaluating mapping options**
> 
> I need to focus on the "tl.sum" reduction order for the backend target. There are different known errors to consider, like specified safety and measuring tolerances. It’s essential I don't guarantee strict compliance if there are Triton quirks. So, I should choose a strict mode as a precise fallback for important f64 calculations. It's a bit complicated, but I want to ensure everything is handled correctly!
> 
> **Updating key insights**
> 
> I have enough research to provide a commentary update on the key insights. The SPM paper may already be full time, and it feels lengthy considering we have only a few tools. I should check if there's any extra update needed. I could mention a key difference, like “核对后最关键差异是DCO仍...”. This is a factual citation, so I can include that. It seems like I'm on the right track here!