# 第一阶段 R1：完整任务清单与不可豁免的验收合同

版本：1.0，2026-09-19。状态：**规范已制定，实现尚未验收。**

依据：[当前代码审计](production_readiness_audit_2026-09-19.md)、[架构原始设计](design/tnca_v1_architecture_draft.md)、[编译器原始设计](design/tnca_v1_compiler_draft.md)。审计基线 commit 为 `6c90063`；当时的测试记录在 [audit/2026-09-19](audit/2026-09-19/verification_summary.json)。旧回归通过不表示本合同已通过。

本文件把审计中的 R1 落成执行合同。**本阶段完成的唯一判据是：本文所有必交付项、所有必跑用例、所有证据要求，在同一个候选 commit 上全部满足。没有百分比完成验收，没有带问题通过，没有临时豁免。**

## 1. 阶段目标、边界和禁止退让的规则

### 1.1 唯一阶段目标

交付一个可在干净环境重建、可可靠发现失败、编译器能阻止已支持程序中的异步资源违规、编译目标与实际硬件一致的 **2×2 实验性 AOT 基线**。

这里的 AOT 基线指已有的提前编译到汇编/镜像并在 RTL 上执行的流程。R1 不要求完成 R2 的可安装产品 CLI、动态参数 ABI、通用 grid scheduler、生产 loader 或真实 host driver。R1 必须提供统一的**验收入口和编译安全检查入口**，不能用产品入口尚未完成作为关闭检查的理由。

### 1.2 范围冻结

| 必须在 R1 完成 | R1 不新增的能力 |
|---|---|
| F01–F04 的根因修复；K16/K32/K64 与 dtype 的实际 footprint 校验 | 新的 Attention/Conv 算子或完整 Triton 支持 |
| F05 的边界封闭：不支持的 topology 在运行前拒绝 | 4×4 实现、任意 grid 调度 |
| F06 的安全子集实现与显式拒绝；保住已有合法循环/多角色路径 | 通用跨 CFG、任意嵌套 region 的完整优化编译器 |
| F07 中当前 importer 的函数选择、参数绑定、支持边界检查 | 正式 Triton 插件的全部 pass、完整 LLVM target、生产 runtime |
| F08 中完成本基线所需的锁定环境、CI、生成物再生成与证据 | FPGA/ASIC 签核、真实 SRAM、host 一致性、生产多租户隔离 |
| F09 中现有有限 workload 的进展/资源守恒回归 | 无限流量下的全系统活性证明、新 cache/flow policy |
| F11 中当前状态、能力矩阵与验证记录同步 | 完整性能调优、全芯片 PPA 承诺 |

此边界是阶段分工，不是对阶段内失败的豁免。审计概述中的“先修 F01–F05”在 R1 的确切含义是：F01–F04 完整修复，F05 通过禁止不支持的配置关闭当前错误入口；4×4 功能实现留给 R4。

### 1.3 验收规则

1. **零失败、零错误、零跳过、零 xfail、零 xpass、零超时、零失联子进程。** 缺工具、没有硬件 profile、FileCheck 未安装、Triton import 失败都使验收失败。
2. 负例必须作为正式测试执行：预期的编译拒绝/硬件 trap 是测试的成功条件。负例不能以 unittest skip、xfail 或忽略退出码实现。
3. 缺失证据、没有执行、结果未知、结果格式错误均等价于未通过；不能默认 PASS。
4. 禁止删除或禁用基线测试、降低随机规模、提高数值容差、增加资源以掩盖原 profile 的错误、移除负例、删除功能声明以规避本文件冻结的必保留能力。
5. 禁止“重跑直到成功”：release run 内 retry=0。任一次失败均保留该 run 的 FAIL；修复后用新的候选 commit 重新执行完整验收。
6. 不得将另一个 commit、另一个 profile、单核测试或模型测试的结果替代当前候选的整机结果。
7. 不得修改生成汇编、人工补 barrier、预填正确输出、绕过真实前端来满足相应 E2E 用例。
8. 所有强制用例必须有唯一 ID；发现的用例集合必须等于验收清单的集合。仅比较总数不够，重命名/拆分必须保留原语义 ID 的映射。
9. 候选 commit 的任一受控源码、配置、锁文件、测试或本合同有变化，都使旧签收失效。不得在已知失败后仅修改清单或门槛将同一候选转为通过。
10. 没有 waiver 字段、允许失败列表或“仅开发环境问题”的例外。若不能达到本合同，阶段状态保持 **NOT ACCEPTED**。

## 2. 必须保留的配置和能力

### 2.1 配置清单

下列为 R1 规范 ID，实施时必须形成机器可读 profile；旧名字允许保留为无歧义的别名，不能把不同实例混为同一 target。

| Profile ID | 硬件范围 | 必须冻结的关键参数 | 必跑验证 |
|---|---|---|---|
| `core-2r-sfu` | 独立 core | 2 RCTX，VLEN=16，ELEN=32，128 physical VREG，64 logical VREG/RCTX，SFU=1 | fast/slow/ooo、原 core 程序、编译器 RTL、能力读回 |
| `core-4r-sfu` | 独立 core | 4 RCTX，VLEN=16，ELEN=32，128 physical VREG，32 logical VREG/RCTX，SFU=1 | 能力读回、一个 producer/两个 consumer、多消费者最后释放 |
| `system-2x2-nosfu` | 四 tile 整机 | NX=NY=2，每 tile 2 RCTX，SFU=0；每 slice 64 sets、8 ways、64 B line、8 MSHR、16 targets | 每 tile 能力读回、原 top 四种运行、真实 Triton 整机 E2E |

其余资源数量以冻结的统一 profile 文件为准，必须和实际 elaboration 逐项相符。不能把 `core-4r-sfu` 写成已经验收的四 core/四 RCTX 整机。fast/slow/ooo 是内存行为变体，必须具有不同 build fingerprint，但不伪造不同 ISA 能力。

R1 的 system profile 不支持 NX/NY 改为 1×1、1×2、2×1、4×4 或非正值；这些配置必须在配置校验/elaboration 阶段失败，不能启动仿真后错误路由。独立 core profile 不等于 1×1 mesh。

### 2.2 不能退步的功能集合

- 所有审计基线中的 core 程序、浮点/转换/SFU/decoder/assembler 用例。
- Python TNCA IR 的 masked elementwise、integer/FP32 reduction、gather；真实 Triton 的 add/mul-add 及已存在的角色分区闭环。
- FP16/BF16 GEMM，现有 stride 和 M/N/K tail；原专用 GEMM planner 与 lowering。
- 两角色 ready/free 流水、2/3-slot ring、0/1/2/3/5/6 次迭代、predicated-off async、错误传播、slow/ooo。
- 已有 unrolled 与 rolled 专用 GEMM 路径，包括 rolled K64/K128 和三槽 K96；四 RCTX 的双消费者释放。
- NUCA hit-past-block、dirty writeback、MSHR/target、mesh 背压、设备原子操作与既有异常隔离。

本清单按**具体入口**保留能力：手工构造 TNCA IR 的 GEMM 不得标成真实 Triton GEMM；单核 SFU 不得标成整机 SFU。通用 TTIR 控制流未实现可以明确拒绝，但不能顺便拒绝已经能正确运行的专用 rolled pipeline。

## 3. 全部工作包及独立验收门槛

每个工作包都必须交付实现、测试、文档和证据；责任字段必须填写实际负责人。下表中的角色只是实施职责，不代替最终负责人记录。

### R1-01：冻结规范、支持矩阵和基线用例

责任：维护者/验证。

必须完成：

- 导出基线的全部语义用例 ID、入口、profile、输入、期望结果与旧测试位置；建立 `r1_required_cases.json` 或等价受版本控制清单。
- 将“ISA 定义、profile 实现、编译器可生成、整机验证”分为四列，另列 dtype/K/shape/stride/control-flow 限制。
- 为所有异步操作写出 read/write footprint、资源保持时刻、释放事件、错误完成行为、barrier generation 和 predicated-off 行为。
- 给每个新用例绑定本文件工作包/审计问题 ID；把两份原始设计标为设计意图，不覆盖成现状说明。

验收：基线所有语义 ID 均可追踪；不存在只有“支持”二字而无入口/profile/测试的条目；必跑清单不得为空；没有未确定的边界、TBD、TODO 或未填写责任人。实现暂缺但属于必交付范围的条目只能标 FAIL/NOT RUN，不能标不适用。

### R1-02：统一失败传播、超时和结果格式（F02）

责任：构建/验证。

必须完成：

- 修复 Make、Python runner、IR runner、assembler/simulator 调用链；上游任一非零退出不得被 `tail`、管道或日志格式掩盖。
- `rtl_harness.run/run_system` 同时检查进程返回码、预期结果、完整输出；禁止仅用 `'PASS' in stdout` 判成功。
- 所有子进程有 wall-clock timeout；RTL 另有 cycle timeout。超时终止整个该测试进程组，回收子孙进程并写出 TIMEOUT。
- strict 模式任何 skip/xfail/xpass 都失败；开发模式可保留可选测试，但不能产生 R1 PASS 证据。
- 测试构建、输入、输出、dump 使用每次 run/用例独立目录；不共享 `cc_<name>` 文件，不能读取上次运行残留。

下列失败注入逐项必跑，测试驱动必须检测到预期失败；驱动自身最后返回 0 才表示此负例通过：

| ID | 注入 | 必须观察到的结果 |
|---|---|---|
| RUN-01 | `TRITON_PYTHON=/usr/bin/false` 调两个 E2E Make 入口 | 两个入口分别非零退出 |
| RUN-02 | interpreter 不存在、Triton import 异常、FileCheck 缺失 | 明确缺失依赖，strict 非零；不是 SKIP |
| RUN-03 | Python assertion/error、assembler 非零 | 顶层 gate 非零，原 traceback/stderr 保留 |
| RUN-04 | simulator 输出 PASS 后退出 7 | 判失败，记录 exit=7 |
| RUN-05 | simulator 退出 0，但无结果、同时出现 PASS/FAIL、结果格式损坏 | 全部判失败 |
| RUN-06 | 子进程挂起并派生一个挂起子进程 | 到截止时间后终止整个测试进程组；5 秒内无存活子孙进程 |
| RUN-07 | suite 空集合、少发现一个必跑 case、重复 case ID、skip、xfail、xpass | 全部判失败 |
| RUN-08 | 删除本次 dump，但保留上次正确 dump | 判失败，不消费旧输出 |
| RUN-09 | IR shell pipeline 第一段失败、末段成功 | 判失败 |
| RUN-10 | 同一用例两个并发 run，输入/输出不同 | 两个结果各自正确，工件无覆盖/交叉读取 |

固定超时上限：单个 metadata/unit/IR 用例 120 秒，单个 RTL 仿真 600 秒，单个 native build 步骤 3600 秒，已配置环境中的整次验收 7200 秒，独立 bootstrap 14400 秒。测试管理器必须在每个截止时刻后 5 秒内结束相关子进程并报告失败；其本身也受外层 watchdog 管理。发布运行不允许用环境变量提高或关闭这些上限。

失败注入使用 1 秒的受控假进程超时，不必为测试 runner 等待实际 600 秒。原 RTL 的 cycle limit 不得调大来掩盖进展回归。

### R1-03：构建依赖、增量失效与生成物只读检查（F04）

责任：构建/RTL。

必须完成：统一每个 bench 的 source/include 依赖；纳入 `rtl/gen`、所有 `.vh`、testbench、内存模型、Python 生成器的导入文件、工具版本、profile、seed、向量数量。为输出写入构建 fingerprint，编译参数变化必须失效。

`generate` 与 `check-generated` 分离。后者不能先自动生成或修复，再声称一致。检查路径必须覆盖 ISA schema、TableGen/C++ 表、文档 opcode map、所有 target、XLS 生成 RTL 和其相关表。

| ID | 必须执行的试验 | 成功条件 |
|---|---|---|
| BUILD-01 | 修改 core RTL、公共 `.vh`、生成 RTL、内存模型各一次 | 所有受影响 bench 确实重新调用编译器；非仅打印“最新” |
| BUILD-02 | 切换 2/4 RCTX、SFU、slow/ooo、seed、向量数量和工具 fingerprint | 新输出对应新配置；不得复用旧 build |
| BUILD-03 | 改生成器或其导入 oracle/常量文件 | 依赖输出重新生成；需要的编译随之发生 |
| BUILD-04 | 人为制造 stale schema/target/XLS RTL 后只运行 check | 非零；整个受控源树与检查前逐文件 hash 相同 |
| BUILD-05 | 相同输入做 clean build 与合法 incremental build | binary、生成文件、规范化报告、数值结果、确定性 cycle 计数相同 |
| BUILD-06 | 无输入变化重复 build | 不重复编译/生成依赖目标；允许执行本来就要求每次运行的测试 |
| BUILD-07 | 故意放入另一 profile 的旧仿真/编译产物 | fingerprint 检查在执行前拒绝 |

BUILD-01 至 BUILD-04 在隔离 checkout 执行，不污染候选树。不能只用 `make -n` 作为最终证据；必须归档实际执行和产物来源。

### R1-04：统一 profile 与真实实例能力（F03、F05 边界）

责任：RTL/编译器。

必须完成：以唯一配置源生成 elaboration 参数、target JSON、能力期望和测试命令；参数覆盖必须传到实际实例。修复 system SFU=false 及独立 core 的 num_cores/mesh 描述；编译结果附上 profile ID、配置 hash、ISA/schema 版本，现有仿真 launcher 在启动前校验。

| ID | 用例 | 必须结果 |
|---|---|---|
| CFG-01 | 三个 profile 的顶层能力读回 | 所有可读资源/能力逐项匹配；system 四个 tile 都执行读回 |
| CFG-02 | 无硬件读回寄存器的配置字段 | 通过 elaboration 导出的实例参数清单核验；不是再次读取同一 JSON 自证 |
| CFG-03 | core SFU 正例与 system SFU 负例 | core 正常数值执行；compiler 对 system 提前拒绝；直接注入非法指令触发规定的硬件 fault |
| CFG-04 | target 配置 hash、ISA 版本或必需能力故意不匹配 | launcher 执行第一条指令前拒绝，无内存副作用 |
| CFG-05 | 非法 topology、RCTX 数/位宽不一致、RF 基址越界、参数组合不合法 | 配置/elaboration 非零；不得截断索引或静默修正 |
| CFG-06 | core-4r profile 检查 | 实际四 RCTX；不能从 system-2x2 JSON 拷贝出虚假的四 core 信息 |

系统 SFU 保持关闭是正确 profile 的一部分，不能为了消除不一致而无验证地把 SFU 打开。所有参数正例和负例都通过同一配置入口。

### R1-05：按实际指令计算资源 footprint（F01）

责任：编译器/ISA。

必须完成：建立统一的显式及隐式资源效果描述，供分配器、所有权检查、lowering 和诊断使用。MMA 范围从实际 K/dtype/layout 计算；DMA 坐标、源/目标 window、accumulator、barrier、descriptor 的生命周期按冻结合同处理，不从 target 默认形状猜测。

VLEN=16、ELEN=32 时一个 VREG=64 B，M=N=16 的每个 A/B fragment 必须满足：

| 输入格式 | K16 | K32 | K64 |
|---|---:|---:|---:|
| FP16/BF16 | 8 VREG | 16 VREG | 32 VREG |
| E5M2/E4M3 | 4 VREG | 8 VREG | 16 VREG |

| ID | 用例 | 必须结果 |
|---|---|---|
| FP-01 | 上表 4 dtype × 3 K 的 12 个组合 | read footprint 精确等于独立公式；不能过小或以扩大资源掩盖 |
| FP-02 | 每个组合分别让 A、B 短 1 个寄存器 | 编译拒绝；诊断给出实际 K/dtype、所需/可用范围 |
| FP-03 | 每个组合的精确长度、长 1、越过逻辑/物理 RF 末端 | 合法范围接受；越界拒绝；超出 profile 总预算报告 resource error |
| FP-04 | K64 + 8-register FP16 windows 的审计负例 | 编译阶段拒绝，不能交给 RTL fault |
| FP-05 | rank=1/2 坐标连续范围，末端越界，非法 dtype/K/rank/window offset | 合法值精确建模；非法值统一诊断，不发生 wrap |
| FP-06 | role-local 名称不同但映射到重叠物理 RF | 分配后识别真实重叠；不能仅比较 SSA identity |

FP-01 是局部资源公式测试；它不强迫 32 logical VREG 的 profile 容纳两个各 32-register 的窗口。容量不够时必须在整体资源分配阶段拒绝，不能跳过 footprint 用例。独立公式 oracle 不能调用被测 footprint 函数。

### R1-06：所有编译入口强制执行异步安全检查（F01、F06）

责任：编译器。

必须完成：

- 分配前验证逻辑生命周期，分配后验证物理范围；所有发射操作的读写统一经过检查，禁止只修 `_op_window_put`。
- 支持 reader 集合、独占 writer、accumulator 访问顺序、event/barrier generation 和多角色物理别名；每个资源仅在相应完成/释放时回收。
- 与 ISA/RTL 核对 DMA 坐标的真实 snapshot/read 时刻并冻结合同：如果发射即快照，证明快照后不再读原寄存器；如果延迟读取，分配器必须持有至规定事件。禁止用未经证明的“应该已读完”提前 free。
- 缺少必须完成事件、未知操作的资源效果、未验证控制流都必须拒绝，不得默认无副作用。
- `check_protocol=False` 不能关闭发射安全性。可保留调试模型开关，但所有已支持编译路径仍必须执行不可绕过的静态/分配后验证；低层 raw assembler 不属于安全编译器入口，文档必须明确区分。
- 同一个 IR 重复编译以及以不同顺序编译 roles 不得被 `Value.reg` 等共享可变状态污染。

下列每个负例都必须有一个仅修正事件/顺序/范围的合法配对正例；正例不能通过删除计算实现：

| ID | 必测错误或约束 |
|---|---|
| OWN-01 | MMA outstanding 时通过 window_put、普通向量写、DMA refill 三种路径覆盖 A/B |
| OWN-02 | DMA 写入未完成时读取/计算其目标；两个 writer 同时占用重叠范围 |
| OWN-03 | 异步 store 源在 source-reusable 前被覆盖；其语义至少在有该操作的 MLIR/协议层验证 |
| OWN-04 | DMA 坐标在合同允许时刻前重用；完成之后允许正常重用 |
| OWN-05 | accumulator 在不允许的 outstanding 状态下被读、清零、重分配；合法的硬件串行更新按明确合同接受 |
| OWN-06 | 缺 await、等待错误事件/错误 generation、重复释放、过期事件释放新租约 |
| OWN-07 | 两消费者仅一方释放便重用 window；必须等最后一方释放 |
| OWN-08 | A/B 部分重叠、多角色物理别名、隐式读范围超出显式基寄存器 |
| OWN-09 | predicated-off async 仍完成一次；错误完成不发布有效数据且能唤醒等待者 |
| OWN-10 | 窗口、barrier、descriptor、accumulator、role 数及 RF 超预算；禁止部分成功发射 |
| OWN-11 | 已支持的 rolled/unrolled 循环跨迭代覆盖、遗漏释放与退出未排空 |
| OWN-12 | 不同 roles 编译顺序、同 IR 连续编译两次、切换 target 后再编译 | 

验收：所有违规在编译期拒绝；错误码稳定，包含操作位置（无源代码位置时给 IR op/role ID）、资源类型/范围、当前所有者、冲突事件和修复方向。OWN-12 的合法输入要求汇编/binary 与语义报告确定一致。

对于 OWN-09 的错误完成，编译器测试验证协议构造，RTL 测试验证运行时唤醒与 fault；不能要求编译器预测外存动态故障。所有无法在当前层表达的测试必须明确放到能表达它的层，不能记为 skip。

### R1-07：MLIR 与 TTIR 的支持边界不得默默越过（F06、F07 部分）

责任：编译器。

必须完成：MLIR verifier 使用统一资源语义并覆盖 accumulator/别名；R1 MLIR 路径允许限定为单 block，但必须显式拒绝不支持的多 block、嵌套 region 与跨 block lease。Python 的既有专用 rolled pipeline 必须通过模板协议不变量或数据流分析证明，不能只模拟第一轮后就声称所有迭代安全。

TTIR importer 必须只选择指定函数；参数按明确 signature 的名字/索引映射，不能依赖传入 dict 的插入顺序；类型、shape、操作和常量解析失败必须返回受控诊断，不能静默转换或崩溃。

| ID | 必须用例 | 判据 |
|---|---|---|
| IR-01 | MLIR 单 block 合法 DMA/MMA/store/await；OWN 系列适用负例 | 正例通过，负例按资源合同拒绝 |
| IR-02 | 多 block、分支只一侧 await、嵌套 region、事件经 block argument 转发 | 未实现的结构明确报 unsupported；不得不检查直接接受 |
| IR-03 | Python 已有 rolled K64/K128、三槽 K96及非法跨迭代重用 | 原正例仍通过；非法生命周期拒绝；不能统一禁用循环 |
| IR-04 | 两函数 module、目标不存在、不同 dict 插入顺序、缺/多参数、类型不符 | 仅编译目标函数；同一语义参数结果一致；非法输入受控拒绝 |
| IR-05 | BLOCK≠VLEN、未支持 scf/dot/layout/op/type、未知资源效果 | 在 lowering 前或当步明确拒绝，不生成可执行成功工件 |
| IR-06 | 浮点常量：±0、有限值、NaN/Inf 及前端允许的字面编码；解析损坏 | 支持者保留规定 bit 语义；未支持/损坏者诊断，不能读成另一个值 |
| IR-07 | 同一资源安全样例的 Python IR 与 MLIR 表达 | 二者对支持范围内合法性结论一致；差异必须修复而非屏蔽 |

没有实现通用 CFG 的情况下，IR-02 的明确拒绝是冻结的支持边界；不是删除已有能力。诊断不得使用 KeyError、assertion traceback 或解释器崩溃代替正常用户错误。

### R1-08：保留并扩充 RTL 与 E2E 回归

责任：验证/RTL/编译器。

必须完成：所有原测试迁移到统一 runner；把新的编译安全正例送到 RTL；增加同周期多 completion、延迟/乱序返回、循环 slot 重用及多消费者释放的可观测检查；失败 seed 可单独重放。

固定基础矩阵：

| ID | 必跑范围 | 不可降低的规模/结果要求 |
|---|---|---|
| REG-01 | 审计基线的所有语义用例 | 全部保留并通过；原 76 项 compiler-unit 中跳过的 7 项按实际入口执行，允许去重但不允许消失 |
| REG-02 | 原 compiler RTL、IR、真实 Triton 单核/整机 | 原 67 个 compiler RTL 测试语义、2 个 IR 文件中的全部 case、5 个单核和2个整机 Triton 测试语义全部执行 |
| REG-03 | 所有原 core 程序 | fast/slow/ooo 全跑；预期 fault 程序核对准确 fault 与未受影响角色 |
| REG-04 | Pipeline | iteration ∈ {0,1,2,3,5,6} × stages ∈ {2,3} × memory ∈ {fast,slow,ooo}，36 组合全部通过 |
| REG-05 | GEMM | 原 FP16/BF16、stride、全部 tail、rolled/unrolled cases 保留；新增 K16/32/64 指令的合法 footprint 计算/发射测试 |
| REG-06 | NUCA | fast/slow/ooo，每种每 seed 至少 800 请求；命中越过阻塞 miss、eviction/dirty WB、target/MSHR 压力相关计数真实触发 |
| REG-07 | Mesh | 每 seed 至少 20,000 包，双 VN、全部源目的对、随机接收背压；零丢包、零重复、零错路由、零同流乱序 |
| REG-08 | Top | t10_mesh/t24_amo_mesh × single-role/all-roles 四种运行全部通过 |
| REG-09 | 两消费者 | core-4r，最后释放语义在 slow/ooo 下各通过；延迟不同消费者验证不能提前重用 |
| REG-10 | 异步错误与完成碰撞 | DMA/MMA 同周期完成均被计数；错误唤醒、不发布错误数据、最终排空与 fault 精确符合合同 |
| REG-11 | 真实 Triton closure | 由固定版本 Triton 前端生成 TTIR；单核 masked tail、多角色结果、整机四 block 输出及 remote traffic 检查全部真实运行 |

REG-04 的 36 个组合各执行一次固定输入，slow/ooo 再执行下述额外 seed。REG-06/07、REG-09/10 及 REG-04 的 slow/ooo 使用 **{1, 7, 42} 三个 seed**；必须把 seed 传到真正控制 stimulus、stall、reorder 的 PRNG，而非只记在日志中。R1 首次验收不得缩短此矩阵。新增 seed 控制必须保留基线的原默认向量/程序语义，不能通过换 seed 消除失败。

所有 seed 路径记录输入/调度 trace hash；三种 seed 必须实际产生不同 trace，相同 seed 的重复运行必须可重放。对于组合覆盖，驱动必须报告“已触发”的计数或 assertion cover；只跑了 20,000 包却没有背压不能通过 REG-07。

浮点：精确 ISA 操作按照既有 exact oracle 与规定 NaN/zero 行为比较；SFU 按现有 ISA 的误差界比较。不得把精确比较改为容差，不得放宽 SFU 误差界。保留审计基线默认向量规模：FMA 41,240；decoder 773；转换 866,495 输入 × 两实例；SFU 单元 16,740；core SFU 1,042 批 × 16 lanes；multi-format FPU 5,012。重构生成器后数量可以增加，不能减少；同时保留原固定种子的输入语义，不能用重复向量充数。

### R1-09：可复现工具链与生成链（F08）

责任：构建/工具链。

必须完成：

- 受控锁文件固定 Triton/LLVM commit、Python 与依赖版本、Icarus、CMake/Ninja、生成工具和基础 OS/image digest；doctor 校验版本、hash、平台与必需能力。
- 建立 **Linux x86_64** 必验收环境；参考 runner 至少 8 vCPU、32 GiB RAM、100 GiB 可用磁盘，具体 OS/image digest 和工具锁必须在候选 commit 内固定。不是引用维护者现有虚拟环境路径。
- 给 XLS 提供固定 revision/依赖的本地 Linux 或容器执行方式；从全部 DSLX/表/参数重新生成受控 RTL，不能依赖个人 SSH、私钥或远程 `cur` 目录。
- 允许下载固定 checksum 的公开/授权工具分发包；不能把未校验的本地缓存或浮动 tag 当锁定版本。锁文件中的每个外部依赖记录来源、版本和 digest。
- 生成物 provenance 覆盖传递依赖，不只记录顶层 `.x` hash。运行后的受控源码不得发生任何自动修复。

验收：两套相互独立的临时环境 A/B，从同一个干净候选 checkout bootstrap 和构建；B 不可复制 A 的 build/venv/生成物，允许使用校验过的只读下载缓存。两个环境全部 R1 测试通过；ISA/profile/XLS 生成物、目标 binary 的 hash 相同。中间 host 可执行文件若含路径/时间戳可不要求 byte-identical，但规范化去除字段必须预先列入受控清单，不能去除语义配置、数值或 cycle 计数。

macOS 保留为开发平台；R1 不允许用 macOS 成功替代 Linux 双环境验收，也不能宣称 macOS 完整 R1 认证而没有同等证据。

### R1-10：CI 门禁与不可混用的验收工件

责任：CI/验证。

必须提供以下稳定入口。**这些命令是待实现的接口要求，本文件不声称它们已经存在：**

```sh
make r1-doctor
make r1-check-generated
make r1-gate-selftest
make r1-acceptance
make r1-verify-evidence EVIDENCE=<run-directory>
```

各入口失败均非零；`r1-acceptance` 固定 strict、retry=0，调用前置检查和全部必跑测试，不能由用户 flag 关闭其中一类。生成 fresh environment 与 source mutation 试验可作为受控 CI jobs，但其结果必须由同一次候选验收汇总，不能靠手工拼一个 PASS 文件。

每个 run 至少归档：

```text
run.json                    commit、tree、clean 状态、环境/工具锁 hash、开始/结束时间
profiles/                   配置输入、生成 target、elaboration 参数、能力读回
inventory.json              required/discovered/executed ID 集合与映射
results.json + junit.xml    每 case 状态、返回码、时间、seed、profile、证据路径
logs/                       全量 stdout/stderr；不只保留 tail
build/                      实际命令、依赖图/fingerprint、生成物/provenance hash
cases/                      输入、TTIR/TNCA IR、分配/footprint、汇编/binary、输出/计数
mutations/                  失败注入、构建失效、checker mutation 的实际结果
checksums.sha256            所有工件的完整性清单
acceptance.json             最终机器判决及每个工作包对应证据
```

验收：`r1-verify-evidence` 对缺文件、改一个输出字节、混入另一 commit/profile、删除一个测试、伪造 discovered count、缺失结束状态分别返回非零。检查器必须重新计算 hash 和集合关系，不能只信任 `acceptance.json` 的 PASS。CI 成功状态必须取自该验证器；上传工件失败也不能算完成。

### R1-11：证明检查器确实能失败

责任：验证，修改与运行均在隔离副本中。

必须逐个做受控 mutation，不能把变异提交到候选实现：

| ID | 故意破坏一处 | 必须将其检出的用例 |
|---|---|---|
| MUT-01 | 移除普通/window 写入的 ownership 检查 | OWN-01；不能依赖 RTL 数值恰好出错 |
| MUT-02 | 将实际 K footprint 恢复为 target 默认 K16 | FP-02/FP-04 |
| MUT-03 | 移除生成 RTL 或 `.vh` 的构建依赖 | BUILD-01 |
| MUT-04 | system target 的 SFU 改成 true | CFG-01/CFG-03 |
| MUT-05 | 恢复 `python ... | tail` 吞退出码 | RUN-01 |
| MUT-06 | 允许未知资源效果或不支持 CFG 无检查通过 | IR-02/IR-05 |
| MUT-07 | 最后消费者释放前回收 window | OWN-07/REG-09 |

验收：七种 mutation 各自至少使指定测试失败，失败原因与破坏点对应；基线未变异时同一测试通过；变异试验结束后候选源码 hash 不变。不接受“测试数量增加了”作为该工作包证据。

### R1-12：当前状态文档、交接和最终签收

责任：维护者。

必须更新 `Readme.md`、`compiler/README.md`、`doc/conformance.md` 和工具链状态说明，使其指向当前 support matrix、三种 profile、严格测试入口、已知限制与本合同。保留审计历史，不把过去缺陷从记录中删除。

必须附本阶段完成报告：每个工作包的负责人、实现 commit、用例 ID、证据路径与最终状态。对本阶段支持范围内的新发现正确性/验证阻断问题，必须补 regression 并修复；不能因为未写在最初 F01–F04 中就放行。

验收：不存在“C0/C1/C4 done”等无条件完成声明与实际支持矩阵矛盾；不将原型写成量产级；清楚记录未验收的后续阶段。签收表 12 项全为 PASS，所有证据由同一候选 commit 的机器验证器核验通过。

## 4. 实施顺序与依赖

| 批次 | 工作包 | 该批次完成后允许做什么 |
|---|---|---|
| 1 | R1-01、R1-02 | 有冻结用例和可信失败传播，可开始根因修复 |
| 2 | R1-03、R1-04、R1-09 | 构建/profile/工具链可信，可获取可比较的修复证据 |
| 3 | R1-05、R1-06、R1-07 | 实际编译入口实施统一安全边界 |
| 4 | R1-08、R1-11 | 正例、负例、压力和变异检测完整闭合 |
| 5 | R1-10、R1-12 | 双环境完整验收、证据验证、阶段签收 |

CI 框架与文档可提前开发，但不能在依赖未完成时提前签收。每个修复提交先跑相关正负例，候选发布必须从头运行全套，不能把逐提交结果相加替代最终完整 run。

## 5. 最终判决算法与签收表

最终 gate 必须执行以下判决，任一条件不成立即退出非零：

```text
候选 commit 固定且源树干净
AND 三种 profile 的配置与顶层实际实例匹配
AND 工具锁/生成链/依赖变更试验全部通过
AND required IDs == discovered IDs == executed IDs
AND 所有必跑 case == PASS
AND skip/error/fail/xfail/xpass/timeout/retry/orphan_process 均为 0
AND 所有负例和 mutation 的预期拒绝都被测试驱动验证
AND 现有合法功能无退步，新增错误入口均被关闭
AND 独立环境 A 和 B 均完整通过，规定的产物确定一致
AND 证据文件完整、hash 正确、commit/profile/run 归属一致
AND R1-01 ... R1-12 全部 PASS
=> ACCEPTED
否则 => NOT ACCEPTED
```

required/discovered/executed 的比较以每个环境/profile/seed 展开后的唯一执行 ID 为准；共享语义 case ID 保留用于追踪。不能把同一 case 跑两次当成另一个遗漏 case。

| 工作包 | 初始状态 | 负责人 | 证据与最终判决 |
|---|---|---|---|
| R1-01 规范/基线 | NOT VERIFIED | 实施时填写 | 未交付 |
| R1-02 runner/失败传播 | NOT VERIFIED | 实施时填写 | 未交付 |
| R1-03 构建/生成物 | NOT VERIFIED | 实施时填写 | 未交付 |
| R1-04 profile/能力 | NOT VERIFIED | 实施时填写 | 未交付 |
| R1-05 footprint | NOT VERIFIED | 实施时填写 | 未交付 |
| R1-06 ownership | NOT VERIFIED | 实施时填写 | 未交付 |
| R1-07 IR/前端边界 | NOT VERIFIED | 实施时填写 | 未交付 |
| R1-08 RTL/E2E | NOT VERIFIED | 实施时填写 | 未交付 |
| R1-09 可复现环境 | NOT VERIFIED | 实施时填写 | 未交付 |
| R1-10 CI/工件 | NOT VERIFIED | 实施时填写 | 未交付 |
| R1-11 mutation | NOT VERIFIED | 实施时填写 | 未交付 |
| R1-12 文档/交接 | NOT VERIFIED | 实施时填写 | 未交付 |

**本文件的写入不改变以上初始状态。只有实际完成全部任务并获得上述证据，才允许宣布“R1 完成”，并启动依赖 R1 验收结果的 R2 交付。**
