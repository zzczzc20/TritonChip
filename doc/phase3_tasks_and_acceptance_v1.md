# 第三阶段 R3：计算与存储实现、快速性能模型及小规模 RTL 校准

版本：1.0，2026-09-20。状态：**任务与验收规范；尚未验收，不宣告 R1/R2 已完成。**

前置合同：[R1](phase1_tasks_and_acceptance.md)、[R2](phase2_tasks_and_acceptance.md)。设计依据：[架构原始设计](design/tnca_v1_architecture_draft.md)第五、六、十六、二十节及[生产化路线](production_readiness_audit_2026-09-19.md)中的 R3。

**RTL 仿真很慢。本阶段只用小尺寸、少组合的 RTL 微基准验证正确性、真实周期数和资源争用；参数搜索由快速模型完成。禁止把 R3 扩成大型 GEMM、Attention 或模型推理的 RTL benchmark。**

历史审计证明小 GEMM 与多角色流水能够在 RTL 上运行，但不能由此推导当前机器的仿真速度。阶段开始必须实际测量每类 bench 的 cycles/second 和 wall time，并据此控制执行成本。**仿真 wall time、硬件周期数、物理时钟频率是三个不同指标，任何报告都必须分开。**

## 1. 阶段目标与完成含义

R3 必须交付：

1. 有明确 bank/端口/同步读延迟合同的 VRF/ARF，实现有限端口仲裁、operand collection、返回写入和正确反压。
2. DMA、MMA、SIMD、SFU、RF、barrier、NUCA/NoC 的可核验周期/流量计数，能解释小程序慢在哪里。
3. 一个独立于 RTL 的有限资源性能模型，用于有限数量的候选配置探索；用预先冻结的 RTL 微基准校准和检验预测误差。
4. 一组尺寸严格受限的 GEMM、reduction、softmax、RMSNorm、LayerNorm 样例，使用真实编译产物执行；保留 R1/R2 的正确性和 runtime 合同。
5. 存储结构可综合、数据确实映射为同步 memory 而非全数组 reset 的 FF 堆，以及诚实标明物理库边界的综合报告。

本阶段完成表示“资源受限的实现可用，关键吞吐/延迟有合同，模型有小规模 RTL 证据”，**不表示原设计的 1 GHz 已实现，不表示 16×16 MMA 指令每周期执行 256 MAC，也不表示大规模 GEMM 性能已在 RTL 验证。**

### 1.1 范围与前置依赖

| 必须在 R3 完成 | 本阶段不增加 |
|---|---|
| VRF/ARF 同步 memory、bank 仲裁、collector、延迟可变时的正确 issue/写回 | RF/ARF 容量扩张、增加 RCTX、4×4 mesh |
| RF 端口预算与 DMA/MMA/SIMD 的测量和排程反馈 | 为追跑分增大 MAC 阵列、双发射、通用乱序 |
| 小型数值 workload 和两角色 overlap 校准 | 长序列 Attention、整模型推理、GEMM 128×128 及以上 |
| 模型探索、实际实现与选定配置的对照 | 每个模型候选都生成 RTL、每个 kernel 做全参数扫描 |
| memory-preserving 综合与结构检查 | PDK/封装/电源完整性/DFT/量产签核 |
| R2 产品入口和事件/排空语义的回归 | 新 host driver、PCIe/FPGA 板级产品 |

性能模型与存储接口设计可提前进行；最终 R3 验收必须在同一候选 commit 上完整通过 R1/R2。已有 R1/R2 的 K128、grid=17 等用例仍按其原合同运行，**本文件的小尺寸上限仅约束 R3 新增的 RTL 工作负载，不删除旧回归，也不为旧回归增加新的全组合。**

R2 的 profile 名称/架构资源数量继续保留；同步 memory/端口/延迟变化必须更新实现 fingerprint、编译成本描述和运行资产。旧 artifact 的兼容性仍按 R2 精确 fingerprint 规则处理，不能隐瞒硬件变化。

## 2. 先冻结仿真预算，再写 benchmark

### 2.1 三层验证职责

| 层级 | 负责回答的问题 | 必须避免 |
|---|---|---|
| 独立数值/协议模型、组件性质验证 | 算法/浮点/所有权是否正确；端口冲突时是否守恒 | 用自己的输出作 oracle；只覆盖快乐路径 |
| 快速有限资源性能模型 | 哪些候选配置值得实现；周期、流量和瓶颈如何变化 | 只算 roofline；读取 RTL 总周期后回填预测 |
| 小规模 RTL 微基准 | 指令实际用几拍；资源争用、overlap、模型误差是否真实 | 大矩阵扫描；用几小时模拟一个统计上没有新信息的样本 |

组件级测试优先；只有必须经过真实 core/NUCA/mesh 的性质才上整机。memory bank 冲突用几十到几百次请求即可触发，不用大矩阵“等它自然出现”。NUCA 的替换压力用同 set 的选定地址制造，不流式扫遍大容量。

### 2.2 R3 新增 RTL 的硬上限

| 项目 | 允许范围/上限 |
|---|---|
| 单核 GEMM | M≤16，N≤16，K≤64；尾块固定为13×11×19 |
| 整机 GEMM | 每 program M≤16，N≤16，K≤32；grid≤4，最多2×2 tile |
| reduction/softmax/Norm | 1行、最多16个有效元素；masked case 15个；不增加 batch |
| 新增纯 elementwise | N≤64，grid≤4 |
| GEMM 性能测量 repetitions | 每次仿真1次 kernel；不在 RTL 内循环100/1000次求平均 |
| 稳态端口微测试 | ≤256个 accepted 请求；初始化/排空另计，不能包装大工作负载 |
| 矩阵单元微测试 | 最多2个连续 MMA 命令，单命令K≤32；K64在指定 core workload 测 |
| 组件随机 seed | {1,7,42}；仅限指定组件和故障注入，不乘到全部 workload |
| 性能 workload seed | 固定42，确定性输入；性能测量不需要大量随机 seed |
| 内存场景 | workload 清单逐项指定；不默认乘 fast/slow/ooo |
| 默认波形 | 关闭；失败定位只截取有界窗口，禁止全程 VCD/FST |

R3 新增性能 inventory 固定为第3节 **12个 core + 4个 system workload**。每个 workload 只运行 `reference` 和 `candidate` 两个实现：**每个验收环境32次性能 RTL 执行**。

此外，新增正确性/反压组件 RTL 执行总数≤48，新增 unit 吞吐/计数校准 RTL 执行总数≤24。因此每环境 R3 新增 RTL 总次数≤104。mutation 用受控 unit/host 测试检出；如需 RTL 必须占上述组件额度，不额外复制16个 workload。baseline 是同一候选树中冻结的旧实现，不借用另一 commit 的测试结果。

### 2.3 时间与周期上限

| 运行类别 | 单次 wall-clock 上限 | 单次 RTL cycle 上限 |
|---|---:|---:|
| 新增组件正确性/吞吐 | 60秒 | 10,000 |
| 新增 core workload | 180秒 | 50,000 |
| 新增 system workload | 300秒 | 50,000 |
| 新增 RTL 全部执行累计 | **3,600秒/环境** | **2,000,000 cycles/环境** |
| 快速模型默认 sweep | **60秒/环境** | 不启动RTL |

wall-clock 累计指每个子进程实际 elapsed time 之和，不能靠并行掩盖预算；重复请求/进程/批次不能绕过 cycles 和工作量统计。baseline 与 candidate 均计入预算。每个模拟进程还必须继承 R1 的进程组清理和外层 watchdog。

参考环境沿用已锁定的 R1/R2 Linux x86_64 runner，记录 CPU/内存/工具版本/并发度。第一次 timing scout 只运行 P01、P04、S01、S03 的 reference，共4次；全部计入正式32次清单，可作为本轮该条结果，不能另算重复跑分。若发现预算无法满足，阶段保持未完成：先解决 simulator 构建/运行效率、bench 初始化、日志开销或实现的无谓循环；不能默默放宽时间、减少必测性质或超时后换 seed 转绿。

允许评估 Verilator 等编译型仿真器作为实现选择，但不承诺未经测量的加速倍数。切换前必须完成组件与选定既有小程序的数值/事件一致性检查，保留 reset/X/未初始化状态专门检查，不能用两态仿真掩盖初始化错误。已通过的 R1/R2 独立验证配置不能被无证据替换。

R1/R2 原有运行预算保持不变，最终签收各执行一次，不因有两个 RF 实现而各翻倍。R3 两套验收环境各满足本节预算；不得把两环境结果拼成一套完整矩阵。

## 3. 唯一默认 workload 清单

所有 workload 都是小程序；无隐含 batch、隐藏多轮重复或 shape 笛卡尔积。`cold` 表示规范化 reset/内存初态，不能只刷新软件计数而保留未知 cache 状态。编译/装载/host 上传时间与设备周期分别报告。

### 3.1 Core：12项

| ID | workload | 固定输入与配置 | 目的 |
|---|---|---|---|
| P01 | GEMM FP16 | 16×16×16，单role，fast，cold | 最小基线与timing scout |
| P02 | GEMM BF16 | 16×16×16，单role，fast，cold | dtype及数值路径 |
| P03 | GEMM FP16 | 16×16×32，单role，fast，cold | K增长预测，留出验证 |
| P04 | GEMM FP16 | 16×16×64，两role、2-slot，固定slow/reorder场景，cold | ready/free overlap与bank争用 |
| P05 | GEMM FP16 tail | 13×11×19，单role，fast，cold | 尾块与填充开销，留出验证 |
| P06 | GEMM FP16 strided | 16×16×16，A/B每行stride=64 B，单role，fast，cold | 布局与搬运，留出验证 |
| P07 | GEMM FP16 serial | 与P04相同shape、内存/seed/输入；显式串行load→MMA，2-slot资源预算相同 | P04的配对消融，不能换小任务比较 |
| P08 | FP32 reduction sum | 1×16，单role，fast | reduction吞吐与加法顺序 |
| P09 | masked FP32 reduction | 存储16元素，仅前15有效，单role，fast | lane mask与零副作用 |
| P10 | FP32 stable softmax | 1×16，core-SFU开启，fast | max/exp/reduce/normalize依赖链 |
| P11 | FP32 RMSNorm | 1×16，epsilon=1e-5，gamma=1，core-SFU开启，fast | rsqrt/reduction，留出验证 |
| P12 | FP32 LayerNorm | 1×16，epsilon=1e-5，gamma=1、beta=0，core-SFU开启，fast | 均值/方差与临时值生命周期 |

P04/P07 的 slow/reorder 参数沿用已验证的 R1 内存模型参数，并在 manifest 中固定 latency/stall/seed；不用两个不同的随机过程比较 overlap。P04的K64可以是4个K16 step，保持 ISA 的累加合同。

P10–P12 可以由正式的 TNCA IR 输入 adapter 编译，不要求本阶段新增通用 Triton softmax/Norm lowering；必须由 R2 产品编译器/loader/runtime执行，报告真实输入路径。禁止调用 Python 计算输出后写入 buffer。

### 3.2 System：4项

| ID | workload | 固定配置 | 目的 |
|---|---|---|---|
| S01 | 动态masked AXPY | N=17、BLOCK=16、grid=2，FP32，SFU关闭，cold | 小型真实Triton→system链、尾tile |
| S02 | GEMM FP16 | 每program16×16×16，grid=1，单role，cold | core到NUCA/NoC的单program增量 |
| S03 | GEMM FP16 | 每program16×16×16，grid=4，单role，独立输出、cold | 四tile有限争用；总计4个小tile |
| S04 | GEMM FP16 | 16×16×32，grid=1，两role、2-slot，cold | 系统流水、返回路径，留出验证 |

系统仍为 `system-2x2-nosfu`，不得为了Norm样例偷偷开启SFU。S03 的四个 program 共享只读输入可以作为固定选定映射，但必须在测量前声明；reference/candidate 使用相同地址/映射。其他共享/放置候选只在模型探索，不再新增整机矩阵。

### 3.3 数值与内存验收

- GEMM FP16/BF16 遵循原ISA的 increasing-k fused FP32累加契约，与独立 exact oracle 按其NaN/zero规则比较；不能为了换bank或改pipeline重排加法。
- P08/P09 按既有 reduction 顺序精确比较；masked元素不参与计算且不得触发其越界访问。
- P10的有限输入限制为[-8,8]，包含等值、正负混合、偏置到大/小端的输入；P11/P12为[-4,4]，包含常量行、近常量行和交替符号。性能case各固定一组输入，其余数值组合在独立软件/组件测试中验证，不能扩成额外整机case。
- P10对高精度数学oracle：每元素 `abs_error ≤ 2e-5 + 2e-4*abs(ref)`，输出非负、有限，`abs(sum(y)-1) ≤ 1e-4`。
- P11/P12对至少50位十进制精度的独立oracle：每元素 `abs_error ≤ 3e-5 + 3e-4*abs(ref)`，输出有限；LayerNorm用中心化方差，不允许为满足测试把合法近常量输入删除。
- 这些误差界是R3新增组合算子的验收要求，不放宽SFU指令自身的R1误差界。超出这些输入域的组合算子支持不在本阶段宣称，原SFU NaN/Inf/subnormal测试仍保留。
- 全部RTL workload检查完整输出、sentinel/canary、预期fault=0、资源排空；性能变好但一个值错也失败。

## 4. 存储与性能合同

### 4.1 两个实现的关系

`reference`：同一候选树中冻结的R2行为RF/ARF实现，维持其原异步读/端口语义，用于数值与周期差异解释。

`candidate`：R3唯一选定的同步、有限bank/端口实现；必须通过真实top实例。两者容量、ISA、算子、输入与软件算法一致。编译器可按各自已声明的延迟/端口合同调度，但报告必须保留两边的代码、资源和指令数；不能通过删计算获得速度提升。

旧RF相当于理想化的多端口实现，candidate并不保证比它周期更少。**不以“任何代价都要GEMM提速”为目标，而以真实存储端口下实现可用、吞吐合同达成、退化能逐项解释为验收。** 相对baseline的周期增加必须有归因，不能隐藏。

### 4.2 必须冻结的硬件资源参数

- VRF容量仍为128×16×32 bit=8 KiB；ARF仍为16×16×16×32 bit=16 KiB；不复制整个data array来伪造免费读端口。
- candidate VRF从4或8 bank中选一个；每bank最多1个read、1个write accepted/cycle，读延迟选择1或2拍。bank映射与每bank实际word宽/深度写入profile，容量求和精确一致。
- ARF可按row/lane切bank，但必须给出真实每bank端口与返回延迟；MMA与向量row操作共享的端口竞争不能消失在组合逻辑中。
- byte/lane mask、同地址read/write、同周期DMA与SIMD写、zero与普通写优先级、invalid数据读、反压时payload稳定性必须逐项定义。
- 如果memory原语不支持某种write mask/RDW语义，必须用明确的subbank、bypass或read-modify-write实现，其额外周期/寄存器/流量全部计入。不得在模拟RAM里增加实现库不存在的端口。
- FIFO、collector、operand holding寄存器是有限资源，不可软件任意寻址；容量、满/空和返回预留进入profile与模型。
- reset必须可实现为清valid元数据或有界scrub；未初始化数据的架构可见语义必须保持R1/R2合同。不得直接删除data reset后把随机/X值当允许结果。

### 4.3 可计算的吞吐/进展合同

在实现前提交 `unit_contracts`：每个unit的请求粒度、bank/port上限、latency L、最大接受率 q、pipeline/queue深度、setup S、drain D、仲裁参与者数C与最大占用长度B。参数必须由硬件结构或既有单元测量导出，不能在测试失败后从失败结果反算一个更宽门槛。

无外部反压、无数据依赖的定向微测试中，N≤256个请求的完成上界为其明确调度模型的 `S + schedule(N,ports,banks,L) + D`。简单固定rate端口使用 `S + ceil(N/q) + L + D`；若不能达到所声明q，验收失败，不可只报告“利用率偏低”。

仲裁采用固定可验证公平策略：若每个grant非抢占地最多占B拍且下游持续ready，C个持续请求者中任一者至多等待 `(C-1)*B + 仲裁流水延迟`。完成/返回所需资源必须在发出前预留或有独立可进展路径；不能靠永远优先新DMA请求保证释放。

MMA的M16×N16是指令形状，unit_contract必须写出实际MAC lane数、每k/m步的读取/计算/等待、ARF往返和FMA依赖；不能直接使用256 MAC/cycle作为模型峰值。

### 4.4 固定测量口径

每个workload同时报告：

- host编译/build/仿真wall time；RTL总cycle与ROI cycle；不相互替代。
- launch accepted、首条有效指令、首个数据请求、首个MMA issue、最后结果store完成、全任务drain完成的时间戳。
- useful MAC/FLOP/bytes与实际发射的padding工作；GEMM一个MAC按2 FLOP统计，但不把padding算成有效工作。
- unit busy/ready/stall、RF每bank accepted读写与冲突、MMA operand starvation、ARF hazard、barrier/window等待、MSHR entry/target压力、link/VN实际传输及DRAM bytes。

stall原因允许同时成立，但必须分别提供重叠bitset与一个固定优先级的互斥主因计数；不能把多个重叠stall相加冒充总时间。短kernel包含固定launch开销，不能通过只截取最好的一拍声称稳态吞吐。

## 5. 全部工作包与必验收用例

用例ID表示验收语义，不等于每个ID都新开一次整机仿真；共享已运行的同一trace核验多个性质是允许的，必须记录case→run映射。必须执行的子条件不能因此消失。

### R3-01：冻结基线、预算与性能合同

责任：微架构/验证。

交付：两个实现的profile、16个workload manifest、unit_contracts、组件/微测试清单、运行预算与责任人；reference源码/hash与所有测量口径。

| ID | 必须验收 |
|---|---|
| PLAN-01 | R1/R2前置证据有效，候选回归清单完整；没有通过给新profile增加资源规避错误 |
| PLAN-02 | workload自动校验M/N/K、grid、row数、repetitions、seed、内存模式；超上限在启动RTL前拒绝 |
| PLAN-03 | timing scout实际执行，记录cycles/second和wall time；清单展开后恰为32次性能运行，组件/微测试和总预算不超限 |
| PLAN-04 | unit_contracts完整；bank/端口/L/q/S/D/队列大小有来源；未知不是0，所有验收公式在candidate测量前冻结 |

### R3-02：独立的有限资源性能模型

责任：性能模型/编译器。

交付：显式建模依赖DAG、issue限制、RF bank/port、ARF更新、DMA请求/返回、MMA/SFU/reduction占用、barrier/channel、有限队列、NUCA entry/target、NoC/DRAM路径的离散事件或等价周期模型。输入为编译后的schedule/操作与profile，不是反向读取该workload的RTLstall trace来“预测”。

| ID | 必须验收 |
|---|---|
| MODEL-01 | 纯串行链、完全独立操作、同bank冲突、不同bank并发、队列满、ready/free复用六种小图，对手算周期/资源计数精确一致 |
| MODEL-02 | 减少端口/增加延迟的组件模型按合同改变结果；不出现零成本布局转换、无限队列或compute与依赖数据到达前重叠 |
| MODEL-03 | 自动检查resource conservation、无环事件推进、未完成租约；非法配置/死锁返回受控错误而非预测零周期 |
| MODEL-04 | 同输入确定重现；默认sweep≤128个候选-工作负载组合，60秒内完成，不启动RTL或数值大矩阵计算 |
| MODEL-05 | 输出预测cycle、bottleneck、流量、拒绝配置原因和模型适用范围；大型shape结果只标model projection，不能标RTL measured |

可探索4/8 bank、1/2拍读、1/2-stage、4/8 credit等有限候选；总资源不能越过profile预算。模型候选不等于受支持RTL profile。只选一个candidate进入实现；不得自动生成128套RTL。

### R3-03：同步VRF/ARF与存储包装层

责任：RTL。

交付：同步read request/response接口、有限bank memory wrapper、mask/RDW语义、valid/reset机制和可综合实现；与旧行为模型差分。

| ID | 必须验收 |
|---|---|
| RF-01 | bank容量总和与8 KiB VRF/16 KiB ARF精确相等；地址映射可逆；边界/跨bank/role物理别名正确 |
| RF-02 | 1/2拍所选同步读延迟、mask、同地址RDW、同周期双客户端写、zero与write竞争，独立模型逐请求比较 |
| RF-03 | reset与relaunch后invalid内容不泄漏；清零/未初始化读的架构语义不退步，不以2-state零初始化掩盖 |
| RF-04 | 每周期accepted读写不超真实port；任一pending响应有保留目的地，无丢失、重复或静默覆盖 |
| RF-05 | 写mask不被支持的memory配置明确拒绝或走已计费RMW；不能在仿真wrapper中使用额外无限端口 |

RF-02/04随机序列每seed≤256请求，三个seed；相同地址热点由定向输入制造。非法同一逻辑数据的并发写若由编译器合同禁止，必须明确assert/reject，不用测试确定一个偶然赢家。

### R3-04：Operand collector、仲裁和返回保活

责任：RTL/协议验证。

交付：有限operand collector、RF request arbitration、writeback/return保留、公平策略、backpressure，以及core pipeline相应hazard更新。

| ID | 必须验收 |
|---|---|
| FLOW-01 | 同bank三读和跨bank读的收集次序/数据正确；未收齐不发射，重放不重复执行 |
| FLOW-02 | SIMD、MMA、DMA同时争用RF；各grant符合冻结公平上界；计数说明真实等待而非丢请求 |
| FLOW-03 | 返回时RF忙、队列满、writeback冲突：已发事务可以完成，无需再分配不存在的资源 |
| FLOW-04 | 随机stall时valid/payload稳定；request/response/tag守恒；drain后occupancy和租约归零 |
| FLOW-05 | branch squash、fault、epoch变化时尚未发射operand/已发事务分别正确处理；R2旧响应隔离仍有效 |

只在声明下游最终ready/内存最终响应的环境下检查有界进展；同时测试R2规定的永不返回→deadline/poison路径，不能把环境假设写成所有错误都不可能发生。

### R3-05：DMA/MMA/SIMD与ARF的真实吞吐

责任：微架构/RTL。

交付：适配同步RF/ARF的新单元时序；可测的accepted/issued/completed接口；必要的operand保持与流水调整。默认维持既有计算lane数，不新增大阵列。

| ID | 必须验收 |
|---|---|
| UNIT-01 | 无冲突RF读写与ARF row访问达到unit_contract接受率；完整setup/steady/drain周期满足公式 |
| UNIT-02 | ≤2条MMA小命令与独立exact oracle比较，报告真实每步MAC/读写/stall；K变化不使用默认K16 footprint |
| UNIT-03 | DMA到RF持续小搬运与有限credit，payload bytes/命令数/完成数精确；不把beat数当barrier operation数 |
| UNIT-04 | 同周期DMA/MMA完成、ARF向量row更新与MMA竞争均不丢完成；源reusable与compute-done不混淆 |
| UNIT-05 | SIMD/SFU/reduction在新RF延迟下数值和mask一致；FMA累加顺序不因重新流水化改变 |

指令正确性与端口协议先通过，再调吞吐。任何新增数据暂存都要计入寄存器/存储结构，不能以“实现细节”为由从资源报告中消失。

### R3-06：计数器、ROI与可验证trace

责任：RTL/性能工具。

交付：第4.4节所有计数的来源、事件定义、reset/epoch/溢出规则和读取接口；对小ROI提供有界trace。

| ID | 必须验收 |
|---|---|
| COUNT-01 | 手工构造小trace中读写、grant、stall、MAC、DMA bytes、barrier完成各精确匹配；窄计数器注入溢出有明确行为 |
| COUNT-02 | 主stall互斥计数总和与被阻塞cycle相等；重叠计数另列，不重复累计总体时间 |
| COUNT-03 | NoC每link/VN计数和实际flit宽/控制字段匹配；DRAM专用路径单列，不假称其经过mesh |
| COUNT-04 | ROI时间戳有统一clock与边界；launch/输出/drain开销可重建total，成功读数不消费上一任务残留 |
| COUNT-05 | 关闭性能采样不改变数值/协议；启用默认计数不新增软件可见存储层；trace大小与窗口受限 |

计数器不需要把短workload天然没有的每种cache事件都触发；未发生显示0。验证事件定义的组件测试必须定向触发对应分支，不能仅因整机计数为0就声称分支已验证。

### R3-07：编译布局、资源报告和成本反馈

责任：编译器。

交付：bank mapping/端口/读延迟进入TargetInfo；布局和pipeline计划知道真实RF成本；分配后保留R1 ownership；R2 manifest/report记录新实现fingerprint和实际资源。

| ID | 必须验收 |
|---|---|
| CC-01 | 同一地址/fragment按真实bank映射预测冲突；布局转换、padding、额外copy的bytes与指令数全部显示 |
| CC-02 | 1/2-stage候选超RF/ARF/barrier/credit预算时拒绝；合法计划不是靠增资源才接受 |
| CC-03 | 异步租约跨RF读延迟/collector延迟仍安全；不能把source free提前到指令发射 |
| CC-04 | 编译器选择/拒绝计划可解释；两个对照实现的代码、资源、动态参数ABI不混淆；未知成本明确标unknown |
| CC-05 | P04/P07用相同数值工作和资源预算作overlap比较；输出准备/等待/搬运/计算的时间线，不能仅凭“两role”推定加速 |

不要求P04在所有小任务上快于P07。若同步/短任务开销抵消overlap，必须如实报告并由模型解释；不得改大K直到只剩好看的结果。

### R3-08：小算子数值路径与产品闭环

责任：编译器/数值验证。

交付：16个workload的受控源、独立oracle、R2 artifact、安装后重放命令；P10–P12组合算子的稳定算法与资源预算。

| ID | 必须验收 |
|---|---|
| NUM-01 | P01..P09和S01..S04按既有精确合同/产品结果检查通过；tail的padding不能增加有效输出区域 |
| NUM-02 | P10..P12满足第3.3节误差与域要求；constant/near-constant/alternating输入的软件和组件检查全部通过 |
| NUM-03 | softmax稳定减max、Norm的epsilon/方差/缩放语义明确；不通过修改参考公式掩盖误差 |
| NUM-04 | 三个非SFU system负例在编译/loader能力边界拒绝，不能用SFU关闭的整机跑出伪结果 |
| NUM-05 | workload经安装后的正式API，dynamic buffers、event、download、canary、drain按R2验证；无测试helper拼装正确答案 |

本阶段样例的有限数值域不代表通用数学库；未验证域必须在支持矩阵中明确。已经由R1/R2承诺的数值域不得缩小。

### R3-09：模型校准、留出验证与有限消融

责任：性能模型/验证。

冻结校准集：P01/P02/P04/P07/P08/P09/P10/P12及S01/S02/S03。**留出集为P03/P05/P06/P11/S04**，在模型参数定版前禁止用这些case的RTL周期拟合参数。所有case都做正确性比较，只有性能模型拟合用途被分开。

| ID | 必须验收 |
|---|---|
| CAL-01 | 模型与RTL使用相同artifact/schedule、profile、初始状态和外部内存约束；预测在读取对应测量结果前保存hash |
| CAL-02 | candidate每个core留出case：`abs(pred-measured) ≤ max(16 cycles, 0.10*measured)`；S04：`≤ max(64 cycles, 0.20*measured)`；逐项满足，不能用平均值掩盖 |
| CAL-03 | deterministic组件的读写/命令/完成计数精确；memory请求/有效bytes对workload精确；外部排队延迟不明时明确建模边界，不填入实际RTL总周期 |
| CAL-04 | P04/P07和S02/S03的资源/时间线消融解释一致；性能偏差给出可定位的事件，不仅写“缓存影响” |
| CAL-05 | reference/candidate逐case报告数值、总cycle/ROI、RF stall、traffic、变化原因；任何退化都公开，不只选择最快case |

若留出误差超限，阶段不能签收。允许修模型后以新模型版本重新做正式验收，但必须保留之前预测与失败记录；不能为P03/P05/P06/P11/S04加入case-name修正系数。重跑仍限本轮32条性能清单，失败run不可被覆盖转绿。

强制消融只有P04↔P07、S02↔S03以及reference↔candidate。其他bank/credit/stage候选只做模型消融，不额外执行全芯片RTL。

### R3-10：存储结构综合与物理声明边界

责任：RTL/综合。

交付：memory-preserving综合流程、memory实例/端口/位数报告、外围控制逻辑面积/FF、与旧data reset结构对照；全部top在选定参数下可综合/elaborate，不靠不可见blackbox隐去逻辑。

| ID | 必须验收 |
|---|---|
| SYN-01 | 在展开算术前的memory结构检查中，VRF/ARF数据位≥90%保留为同步memory对象；remaining valid/bypass/collector bits独立列账，数据容量正确 |
| SYN-02 | 每个memory的真实read/write port、clock、mask、latency与profile一致；不存在异步read替代、全data array reset或隐式额外读复制 |
| SYN-03 | top的结构综合/lint无未解释latch、组合环、宽度截断、多驱动、未连接必需端口；所有变更模块均纳入 |
| SYN-04 | 生成memory wrapper与一种锁定的RAM实现契约对接检查通过；若尚无物理macro/FPGA mapping，只可标memory-inferred，不可写SRAM area/Fmax已达标 |
| SYN-05 | 综合表将data memory容量、逻辑门/FF、模型时钟、实际STA结果分别列出；无实际库则Fmax/功耗/物理面积为NOT MEASURED，不从gate depth换算GHz |

SYN-04要求接口契约及综合可保持的memory结构，不要求本阶段采购PDK或流片。若引入真实macro/FPGA RAM，必须额外锁定primitive/库、RDW/mask行为和约束，其证据不能用通用memory节点代替。

默认不做耗时的全芯片flatten+通用门映射来估计巨大RF的“面积”。结构综合单任务上限1200秒；超时就是未完成，不能将缺失模块记录为0 gate。实际技术库STA属于可附加证据，不作为无PDK条件下本合同的隐含前置。

### R3-11：预算门禁、性质反例与mutation

责任：验证/工具。

交付：预算lint、单次与累计计时、trace限制、独立数值/事件checker；所有新增RTL run统一登记。

| ID | 必须验收 |
|---|---|
| BUDGET-01 | M/N/K/grid/row/repetitions超限、隐式笛卡尔积、32条性能清单外新增运行，启动RTL前拒绝 |
| BUDGET-02 | 单次wall/cycle超限、累计3600秒/200万cycles、组件执行次数超限被真实gate捕获；测试用缩小的假预算验证，不浪费真实小时 |
| BUDGET-03 | 每个run记录workload、implementation、profile、seed、实际wall/cycle、trace bytes；缺一项不能签收 |
| BUDGET-04 | 独立软件/模型用例不启动RTL；waveform默认关闭；failed run与其有界诊断trace不覆盖正式失败状态 |

以下mutation全部必须检出，注入在隔离副本：

| ID | 故意破坏 | 必须检出的检查 |
|---|---|---|
| R3M-01 | 将同步read错误地提前一拍，或返回错bank数据 | RF-02、FLOW-01 |
| R3M-02 | 去掉一条return/writeback反压或预留 | FLOW-03/04 |
| R3M-03 | ARF累加改顺序或丢一个masked write | UNIT-02/05、NUM-01 |
| R3M-04 | stall重叠重复计数或丢一次完成计数 | COUNT-01/02 |
| R3M-05 | 模型让RF端口无限或队列无限 | MODEL-01/02 |
| R3M-06 | benchmark偷偷增加K/grid/repetitions | BUDGET-01 |

上述反例优先在对应组件测试内检出，不复制整个GEMM/system workload矩阵。未变异时同一checker通过，变异失败必须由指定语义触发，不能用无关build失败算成功。

### R3-12：双环境、发布证据和文档

责任：CI/维护者。

待实现的稳定入口：

```sh
make r3-doctor
make r3-check-budget
make r3-model
make r3-components
make r3-microbench
make r3-synth-structure
make r3-acceptance
make r3-verify-evidence EVIDENCE=<run-directory>
```

`r3-acceptance` 对reference/candidate的32条性能运行、组件/校准执行、模型、结构综合、mutation、R1/R2继承回归统一去重登记；不能每个子目标重新跑同一套大回归。两套独立Linux环境按锁定工具链构建；候选上的R1/R2全部必跑，各环境执行一次。

| ID | 必须验收 |
|---|---|
| REL3-01 | A/B新RTL工作量均符合预算、32条性能结果齐全；确定性输入的cycle/流量/数值一致，host wall time分别记录，不要求相同 |
| REL3-02 | R1/R2候选回归全通过；新memory profile参数与top实例相符，安装后的R2入口无退步 |
| REL3-03 | 证据缺失、改值、跨commit/profile/model版本混用、漏掉慢case、伪造预算汇总都被验证器拒绝 |
| REL3-04 | 所有工作包负责人、测试ID、证据路径、支持范围完整；未测PPA和模型外推清楚标记，不宣称大矩阵RTL跑分 |

必须归档：

```text
contracts/       profiles、unit_contracts、bank/port/RDW/reset规则、数值域
inventory/       16个workload、32个展开性能run、组件清单、预算、seed
build/           reference/candidate来源与工具锁、fingerprint、安装信息
model/           输入、预测hash、calibration/holdout划分、版本、误差与sweep结果
rtl/             真实产物/参数/计数、输出oracle、实际wall/cycle、短trace
storage/         memory实例与端口位数、结构综合、已测/未测物理指标
mutations/       六项反例和对应失败证据
inherited/       同候选R1/R2验收引用与去重映射
acceptance.json  按工作包/用例/预算推导的机器判决
```

## 6. 实施顺序与最终签收

| 顺序 | 工作包 | 不可跳过的关口 |
|---|---|---|
| 1 | R3-01、R3-02 | 先固定小workload、预算、unit合同与模型边界 |
| 2 | R3-03、R3-04、R3-10 | 同步memory与有限端口真实成立，再接大core；先查结构，避免回到FF堆 |
| 3 | R3-05、R3-06、R3-07 | 单元时序、计数与编译资源共同闭合 |
| 4 | R3-08、R3-09 | 只跑冻结的32条性能RTL，完成数值、模型留出与有限消融 |
| 5 | R3-11、R3-12 | 预算/反例证明、双环境、继承回归和正式证据 |

最终判决必须同时满足：

```text
R1/R2在同候选上全部通过
AND 真实同步memory、容量/端口/延迟/复位合同成立
AND arbitration/collector/return资源守恒与进展检查通过
AND unit吞吐/延迟符合事先冻结的结构公式
AND 16个小workload × reference/candidate 全部数值正确且排空
AND 模型留出误差逐项满足阈值，没有读测量结果伪造预测
AND 所有周期/流量/有效工作统计可核验，退化未隐藏
AND 新增RTL尺寸、次数、cycle和wall预算全部满足
AND 结构综合真实保留memory，物理声明不超过证据
AND 所有mutation被检出，双环境与证据检查通过
AND 无非预期fail/skip/xfail/xpass/timeout/retry/孤儿进程
AND R3-01 ... R3-12 全部PASS
=> R3 ACCEPTED
否则 => R3 NOT ACCEPTED
```

| 工作包 | 初始状态 | 负责人/证据 |
|---|---|---|
| R3-01 基线/预算/合同 | NOT VERIFIED | 实施时填写 |
| R3-02 快速模型 | NOT VERIFIED | 实施时填写 |
| R3-03 同步VRF/ARF | NOT VERIFIED | 实施时填写 |
| R3-04 collector/仲裁/返回 | NOT VERIFIED | 实施时填写 |
| R3-05 单元吞吐 | NOT VERIFIED | 实施时填写 |
| R3-06 计数/ROI | NOT VERIFIED | 实施时填写 |
| R3-07 编译成本反馈 | NOT VERIFIED | 实施时填写 |
| R3-08 小算子闭环 | NOT VERIFIED | 实施时填写 |
| R3-09 校准/留出/消融 | NOT VERIFIED | 实施时填写 |
| R3-10 存储结构综合 | NOT VERIFIED | 实施时填写 |
| R3-11 预算门禁/mutation | NOT VERIFIED | 实施时填写 |
| R3-12 双环境/交付 | NOT VERIFIED | 实施时填写 |

**工作负载小不是降低正确性标准，而是用定向输入和可观测事件覆盖问题。不得以RTL慢为由跳过必测性质，也不得以“更真实”为由无限增加RTL负载。**
