# 第三阶段 R3 v2：有限端口 VRF 的单核集成闭环

版本：2.0，2026-09-20。状态：**PLANNED / NOT ACCEPTED**。
检查基线：`71317b35afb9d2602b0510da84188a0fbde15e29`。

依据：[当前状态复查](r3_replan_review_2026-09-20.md)、[架构设计](architecture_overview.md)、[编译器设计](compiler_overview.md)。
本文件替代原第三阶段的执行计划；[v1 合同原文](phase3_tasks_and_acceptance_v1.md)保留。R1/R2 原合同不变。

**RTL 仿真很慢。新阶段只做一个单核 candidate、少量小程序和定向组件测试。缩小交付范围，不降低所保留功能的正确性、反压、所有权和证据标准。**

## 1. 阶段目标与签收边界

本阶段只回答一个问题：**现有同步 banked VRF 和 operand collector 能否接入真实 core，在全部 VRF 客户端争用、返回和写回受限的条件下，正确执行小程序，并经安装后的单核 runtime 使用？**

最终交付：

1. 一个显式 opt-in 的单核实验实现，使用真实同步、有限端口 VRF；旧实现保留为默认和差分参考。
2. SIMD、LSU、tensor/DMA、matrix 及其他实际 VRF 访问者全部经受控端口访问，无隐藏直读/直写路径。
3. 有限 collector、返回预留、写回仲裁、复位与故障排空；异步 ownership 不退步。
4. 六个小 workload 的数值、资源守恒、周期和端口证据，以及安装后动态参数多次发射。
5. 对实际集成 VRF 的结构综合证明和能拒绝缺失/篡改证据的阶段 gate。

签收名称固定为 **`R3_V2_CORE_VRF_ACCEPTED`**。它不等于旧 `R3 ACCEPTED`、R2 签收或生产发布。

### 1.1 做与不做

| 本阶段必须完成 | 后续再做；不计为本阶段 PASS |
|---|---|
| 8 KiB VRF 同步读、4 bank、每 bank 1R1W、1 拍读延迟 | ARF 同步 memory 改造；本阶段仍显式标为 behavioral |
| 单核、2 Role Context；全部 VRF 客户端经过同一资源合同 | 4 role candidate、四 tile candidate、mesh/NUCA 性能模型 |
| SIMD/LSU/tensor/MMA 的 VRF 时序适配和真实写回反压 | 新 MAC 阵列、双发射、吞吐优化和全面 unit peak 性能承诺 |
| 使用既有 TNCA IR / Triton adapter 与安装后的 runtime | 完整 MLIR 主路径、通用 Triton GEMM lowering |
| 小型 GEMM、AXPY、向量和 reduction 验证 | softmax/RMSNorm/LayerNorm 新组合算子和大形状扫描 |
| 端口/collector 微模型精确比对，core 周期实测 | 全 kernel 模型误差阈值、自动布局优化、系统留出集校准 |
| 一套锁定 Linux 全门禁 + 第二环境定向复现 | 原 R1/R2/旧 R3 的完整双环境生产发布 |

选择固定 4 bank / 1 拍，是复用已有组件并减少集成变量。已有组件的 2 拍测试保留，但不新增第二个可发布 candidate。ARF 延后不允许绕过 VRF 为 MMA 补一个无限读口。

### 1.2 依赖和回归规则

- R1 的历史签收只证明 `97c0cf90`。本阶段每份证据必须绑定新候选 commit，不能直接读取历史 verdict 当作候选通过。
- 不等待 R2 全部完成；依赖其已有单核 compile/artifact/ABI/session 路径。缺少本阶段必需能力必须实现，不能用未完成的 R2 作跳过理由。
- 新增实验 profile 建议 ID `core-vrf-sync-v1`，资源继承 `core-sim`：2 RCTX、相同 ISA/寄存器容量/SFU；实现差异写入 fingerprint。`core-sim`、`core-4r`、`system-2x2` 默认行为不变。
- 新 profile 仍须支持所继承的 ISA 功能。六个 workload 只是定向集成证据，不能据此静默删掉其他指令；已有单核 ISA 回归必须在 candidate 执行。
- 当前提交的旧 profile 必须跑既有核心 fast/slow/ooo 回归、compiler/ownership/footprint 检查、生成物只读检查和 profile 核验。candidate 必须跑相同单核适用回归及本文件用例。
- G0 必须冻结每个继承 case 的具体 ID/命令/实现/理由。共享文件发生变化时，相关 core-4r/system 回归也必跑；只有仅改变新 profile 且无共享语义变化的项可以不进入本阶段清单，必须有源码影响说明。不得因耗时或失败删除 case。
- 继承回归与新增预算分别计账、统一去重。阶段签收不宣称原 R1/R2 在当前候选全部再次签收；正式产品发布仍执行原合同。

## 2. 当前基础与真正的关键路径

当前已经有 bank、VRF、collector、微模型、计数和综合工具；core 仍使用行为 RF，ARF 仍异步读。`r3-flow-tests` 的 FLOW-03 明确失败于未实现返回/写回测试。

顺序固定为：**验收清单与接口冻结 → 复位语义和 VRF 包装 → 全部客户端接入 → 返回/写回与 ownership → 小程序差分 → 安装后闭环 → 证据签收。**

不要先增加模型候选、算子种类或整机规模。阶段最早可评审成果是“core 中真的使用有限端口”，不是更多独立模块的 PASS。

## 3. 冻结的资源和时序合同

G0 交付实现规格，至少包含以下字段并由 elaborated 实例读回核对：

| 项目 | 本阶段要求 |
|---|---|
| VRF 容量 | 128 × 16 × 32 bit = 65,536 bit = 8 KiB；2 role 各 64 logical registers；物理别名保持现有合同 |
| bank | 4 bank；每 bank 32 × 512 bit；物理寄存器低 2 bit 选 bank，其余选 row |
| 端口 | 每 bank 每拍最多 1 个 accepted read、1 个 accepted write；不复制数据阵列增加读口 |
| 同步读 | 请求在边沿 t 被接受，数据最早在 t+1 可消费；完整定义 valid/ready 与 downstream stall 时数据保持 |
| collector | 复用现有 4 entry / 每条至多 3 个源；源收齐才能 issue；每次 accepted read 有 tag 和返回归宿 |
| 写回 | 所有写客户端枚举；FIFO/holding 的固定深度、最大在途数、仲裁策略和 credit 逐项声明，不允许 unbounded |
| RDW/mask | 固定 write-first；仅对 accepted write 旁路，逐 lane 合并；拒绝的 write 不能提前生效 |
| 复位 | 保持既有架构的零初始化行为；允许 valid 元数据返回零或有界 scrub；不允许 reset 整个数据阵列，也不允许把原来读零改成 X |
| ARF | 保持现有行为实现与累加顺序，报告为 behavioral；不计入“同步存储已完成” |
| 排空 | halt/成功完成不能早于所有被接受读写、异步操作和持有租约完成；错误路径必须释放或明确毒化 session |

必须分别统计 data memory、valid、collector、bypass 和 FIFO 位数。valid/mask 的实现不得令部分写覆盖 lane 的初始化语义失效。

公平性以有界 downstream ready 为前提：冻结请求者数量 C、每次 grant 最长占用 B、仲裁流水 A；持续请求者的等待不超过 `(C-1)×B+A`。无外部进展不能承诺固定完成时间，此时必须触发 watchdog 并输出阻塞原因。若实现使用逐 beat 仲裁，B 按 beat 计算，不把整条 DMA 当作永久占用。

## 4. 小尺寸 workload 和执行成本

### 4.1 唯一新增 core 清单

每项恰好运行 reference 与 candidate 两个实现，**共 12 次**。seed 固定 42；不乘 fast/slow/ooo 全组合；每次只有一次 kernel。reference 为同一候选树中的旧实现，不能复用另一个提交的结果。

| ID | 固定 workload | 输入路径 / 内存 | 验证目的 |
|---|---|---|---|
| C01 | FP32 vecadd，N=16，grid=1 | 既有编译器；fast | 基础 operand collection、SIMD 写回 |
| C02 | FP32 masked AXPY，N=17，BLOCK=16，grid=2 顺序单核执行 | 真实 Triton adapter；fast | 尾块、动态 alpha、LSU/写回；不宣称双 tile |
| C03 | FP16 GEMM，16×16×16，单 role | TNCA IR；fast | tensor→VRF→MMA→store 数值闭环 |
| C04 | FP16 GEMM，16×16×32，串行调度 | TNCA IR；固定 slow/reorder | 两个 K16 step，返回争用基线 |
| C05 | FP16 GEMM，16×16×32，两 role / 2-slot pipeline | TNCA IR；与 C04 同地址、输入、内存配置和 seed | window 复用、并发客户端、资源归还 |
| C06 | FP32 reduction，16 存储元素、15 有效元素、1 row | TNCA IR；fast | mask、依赖链、读写归宿 |

C04/C05 均冻结相同可用 register/window/barrier 预算。不同调度可能改变外部请求时序；保存实际返回 trace，不宣称相同 seed 令两条 trace 逐拍相同。允许 C05 比 C04 慢，必须公开原因，不增大 K 追求好看的加速。

**新增 GEMM 上限 M=N=16、K≤32；无 batch、无隐藏重复。** reduction 最多一行 16 元素，elementwise N≤32、grid≤2。旧回归已有更大尺寸不删除，也不复制进新增性能矩阵。

数值门槛：GEMM 按现有 ISA increasing-k fused FP32 累加合同和独立 exact oracle 比较；vecadd/AXPY/reduction 按各自现有指令舍入与归约顺序比较。reference/candidate 相互一致仍不足够，二者均须与独立 oracle 一致。检查完整输出、尾部 sentinel、两侧 canary、fault=0、资源最终排空。

### 4.2 安装后 runtime 用例

只对 candidate 另执行 3 次 C02：同一 artifact，alpha 分别 2.5、-1.0、0.25；输出换 buffer；一次 session、一个 device 进程。code hash 不变，首编译后额外 compiler 调用数为 0。它们计入新增 RTL，共 **15 次 core kernel 执行**，不另跑完整 R2 矩阵。

### 4.3 预算规则

| 类别 | 次数上限 | 单次 wall / cycle 上限 |
|---|---:|---|
| core 对照 + 安装后执行 | 12 + 3 = 15 次 kernel | 180 秒 / 50,000 cycles |
| 组件与负例，包括 mutation 的 RTL 基线/变异执行 | 40 次 RTL invocation | 60 秒 / 10,000 cycles；每场景 ≤256 accepted 请求 |
| 新增 RTL 合计 | ≤55 次；持久进程也逐 kernel 计账 | elapsed 累计 ≤1,800 秒；累计 ≤1,000,000 cycles / 完整环境 |
| 微模型测试 | 不启动 RTL；≤32 配置组合 | 30 秒 |
| 结构综合 | 选定 VRF/collector/adapter 及集成结构检查 | 合计 1,200 秒 |

以上是工程预算，不是本次已测达成值。G0 在锁定 Linux 上先跑 C01/C03 的 reference 做 scout；相同 commit/artifact/config 的记录可直接占据正式 12 次中的两条。candidate 必须另测。记录 build 时间，build 与仿真预算分别计账；compile/setup 设独立总 watchdog，G0 冻结上限后才开始正式 run。

子进程 elapsed 求和，不能用并行后的墙钟总时长掩盖成本。持久进程的启动、上传、执行、下载只计一次开销；各 kernel cycles 独立登记。过程重试保留原失败，不覆盖；调试 run 独立目录、有界预算，不能用其 PASS 覆盖同一正式 run 的 FAIL。

默认关闭波形；诊断最多一个失败 case 的 256-cycle 窗口、10 MiB trace。超时清理整个进程组；缩小的假预算验证超时机制，无需真的等待半小时。

G0 若发现预算不够，先定位初始化/日志/无谓循环和仿真构建开销；仍不可满足则在正式运行前提交新版本的预算依据，保持当前版本 NOT ACCEPTED。不得事后修改阈值将失败改成成功。

## 5. 工作包和明确验收标准

所有新用例采用 `R3V2-*`，避免与旧脚本中的 PLAN/RF/SYN 等 ID 混用。每项要求必须映射到真实命令、断言、退出码和工件；初始状态全部 NOT VERIFIED。

### G0 — 基线、接口、清单和 gate 骨架

交付：版本化 profile、资源合同、15 次 kernel 清单、组件清单、继承回归清单、负责角色、工具锁和预算。负责：维护者 / 验证。

| ID | 不可退让的验收标准 |
|---|---|
| R3V2-BASE-01 | 所有 case 有 ID、实现、源码/artifact hash、profile、seed、尺寸、内存模式和预算；必需集合由独立冻结 inventory 决定，不能按结果文件里出现什么就验什么 |
| R3V2-BASE-02 | checker 拒绝漏 candidate、重复 reference、未知 profile、memory mode 冒充 implementation、零/负/超限尺寸、重复 case；全部在启动 RTL 前失败 |
| R3V2-BASE-03 | Linux scout 有真实命令/退出码/周期和 elapsed；candidate 资源合同无必需 null；历史 R1 仅作 provenance，当前 commit 的继承回归计划完整 |

### G1 — VRF 包装、初始化与存储语义

交付：同步 bank 接口、valid/zero 方案和独立协议模型。负责：RTL / 验证。

| ID | 不可退让的验收标准 |
|---|---|
| R3V2-RF-01 | 容量、4 bank、地址逆映射、role 物理别名和边界检查一致；每拍每 bank accepted read/write 各 ≤1，全部返回只出现一次 |
| R3V2-RF-02 | seed {1,7,42} 的有界序列覆盖 mask、RDW、冲突、连续请求、read response stall；已初始化 lane 与独立模型逐项相等；定向输入保证每种性质实际触发 |
| R3V2-RF-03 | cold reset、部分写后读取、relaunch、reset 时在途事务、zero/write 竞争符合冻结合同；旧 epoch 返回不能污染新状态；架构未写内容仍读零，底层 X 不被误当合法输出 |

### G2 — 全部客户端接入真实 core

交付：新 profile 的 top/参数、各读客户端时序适配和端口仲裁。负责：RTL。

| ID | 不可退让的验收标准 |
|---|---|
| R3V2-CORE-01 | elaboration/结构报告证明新 top 活跃数据路径例化 banked VRF；逐项枚举 SIMD/LSU/tensor/MMA/其他客户端；无旧 RF shadow copy、层级直读或额外组合数据通路 |
| R3V2-CORE-02 | 同 bank 三源读必须串行获 grant、跨 bank 可并行；所有源返回前不 issue；暂停 response 后 payload/tag 稳定，恢复后值和目的 entry 正确 |
| R3V2-CORE-03 | 每个读客户端至少一个定向程序触发真实 stall；各客户端都能独立完成并参与争用；不能通过屏蔽 MMA/tensor/第二 role 使 candidate 通过 |

### G3 — 写回、返回、所有权与有界进展

交付：所有写客户端仲裁、有限返回 holding、credit/epoch 和排空断言。负责：RTL / 编译器验证。

| ID | 不可退让的验收标准 |
|---|---|
| R3V2-FLOW-01 | 同周期 SIMD 与 DMA 等多客户端写同 bank，所有接受事务最终提交一次；不接受者保持 payload 并重试；full FIFO 时不能继续发出无返回空间的事务 |
| R3V2-FLOW-02 | 持续请求且 downstream ready 时，逐请求等待满足冻结公平界；固定有界反压后全部排空；issued/accepted/returned/committed/cancelled 守恒，credit 最终归还 |
| R3V2-FLOW-03 | 复用 collector entry、flush/fault 与 late return 交错不串数据；租约从真实最后消费/完成事件释放；继承 OWN/FP 正负例在 candidate 上成立，不能在 issue 时提前 free |

### G4 — 六个小程序与安装后单核闭环

交付：受控 workload 源、独立 oracle、artifact、运行记录和安装后脚本。负责：编译器 / runtime / 验证。

| ID | 不可退让的验收标准 |
|---|---|
| R3V2-E2E-01 | C01–C06 × 两实现共 12 条全部执行，通过第 4.1 节的数值、mask、canary、fault 和 drain 检查；GEMM source_kind 必须是实际的 tnca-ir |
| R3V2-E2E-02 | wheel 非 editable 安装，仓库不在 sys.path、任意 cwd；新 profile prepare/compile/run/download 成功；3 次动态 alpha 发射共一个设备进程，数值正确，code hash 与编译调用数满足第 4.2 节 |
| R3V2-E2E-03 | reference artifact 与 candidate device 双向错配在零次 launch 前拒绝；错误旧 cache 不被采用；只安装 runtime 和运行资产时可加载既有 artifact，不依赖源码 checkout |

### G5 — 可核验计数和有限微模型

交付：端口/collector/写回事件定义、程序时间线、微模型和差异报告。负责：性能 / 验证。

| ID | 不可退让的验收标准 |
|---|---|
| R3V2-COUNT-01 | bank 读写 accepted、conflict、collector wait、写回 wait、inflight/credit 与独立短 trace 逐事件相等；主 stall 归因互斥且求和等于总 stall，重叠视图单独列出 |
| R3V2-COUNT-02 | 每个 core case 给出 launch、首次 issue、最后 store committed、drain 的周期边界；计数按同一边界截取；输出 host/build/sim wall、总 cycles 和 ROI cycles，不从模拟速度推出 GHz |
| R3V2-MODEL-01 | 串行读、跨 bank、同 bank、队列满、返回反压、flush 后复用六种定向图，微模型与 RTL 的周期/计数精确一致；预测先保存 hash；不得读取测量结果回填模型 |

不要求 candidate 比 reference 快。所有周期增加公开列出等待原因；无法解释的残差标明 unknown。kernel 级预测只可作为带误差的实验记录，本阶段不签收全程序/NUCA/NoC 模型精度。

### G6 — 结构综合和继承回归

交付：真实集成层级的 memory/端口清单、组件综合报告、继承回归证据。负责：RTL / CI。

| ID | 不可退让的验收标准 |
|---|---|
| R3V2-SYN-01 | candidate VRF 数据正好 65,536 bit，≥90% 在同步 memory 对象中；全部 memory 端口/clock/mask 与 profile 相符；valid/collector/FIFO 单列，不把旧 ARF 位数加入该比率 |
| R3V2-SYN-02 | 新 bank、collector、adapter 和相关控制模块无未解释 latch、多驱动、组合环或宽度问题；集成 hierarchy/端口连接检查通过；只标 memory-inferred，物理面积/Fmax/功耗为 NOT MEASURED |
| R3V2-REG-01 | G0 冻结的 candidate 单核回归、旧 profile 影响回归全部 PASS；旧默认 profile 保留；无缺依赖 skip/xfail、无 ownership/数值/错误完成语义退步 |

不强制整个系统 flatten 后做通用门映射；局部综合不能代替集成连接检查，也不能宣称 ARF 已同步化。

### G7 — 负例、证据验证和阶段签收

交付：一个非交互 gate、独立 verifier、证据目录和签收表。负责：验证 / 维护者。

| ID | 不可退让的验收标准 |
|---|---|
| R3V2-MUT-01 | 下表全部变异均由指定语义 checker 检出；clean 基线通过；无关语法/编译失败不算检出；注入隔离副本，源码 hash 不变 |
| R3V2-EV-01 | 独立 verifier 从必跑 inventory 和原始结果重算判决；拒绝漏 case、改结果/日志、混 commit/profile/artifact、漏慢 case、伪造 cycle/wall 汇总；不能信任写入的 PASS 字段 |
| R3V2-REL-01 | 主 Linux 环境全部必跑项通过；第二独立构建环境完成规定 smoke；预算、工具锁、负责人和工件齐全；所有 G0–G7 条件同时满足才输出 R3_V2_CORE_VRF_ACCEPTED |

必须检测的变异：

| ID | 变异 | 指定 detector |
|---|---|---|
| M01 | read 提前一拍 / 错 bank 返回 | RF-02、CORE-02 |
| M02 | 丢 writeback reservation 或忽略 full | FLOW-01/02 |
| M03 | flush 后旧 tag 写入新 entry | FLOW-03 |
| M04 | 漏一个 masked write 或提前 source-free | RF-02 / FLOW-03；两个子变异都要执行 |
| M05 | stall 重复计数 / 少一次 accepted | COUNT-01 |
| M06 | 模型允许同 bank 无限读 | MODEL-01 |
| M07 | 漏 candidate、扩大 K、伪造 elapsed、漏证据 | BASE-02 / EV-01；四个子变异都要执行 |

表中 detector 缩写均指本文件 `R3V2-` 前缀 ID。M01/M05 的两个子变异也全部执行；RTL 开销计入组件 40 次，host-only 负例不启动 RTL。

## 6. 环境、入口和证据

主环境：锁定 Linux x86_64，沿用 R1 工具版本并记录本阶段新增依赖。第二环境可为另一独立 Linux 构建目录/容器或 macOS 开发机；只能共享只读下载缓存，不共享构建产物。第二环境跑 RF/collector 定向测试、C02/C03/C05 的 candidate 和安装后 3 次动态发射；数值/确定性计数应一致，wall time 分别记录。平台差异必须定位，不能忽略。

第二环境是本阶段跨环境 smoke，**不能冒充旧合同的完整双 Linux 发布验收**。其执行次数和预算按子清单独立核验，不与主环境合并凑覆盖。

下列入口是待实现接口，本次文档更新没有创建这些 Make targets：

```sh
make r3v2-plan                 # 校验合同/清单/影响范围，不启动 RTL
make r3v2-components           # bank/collector/return 定向测试
make r3v2-core                 # 六 workload × reference/candidate
make r3v2-runtime              # 安装后候选 profile，3 次发射
make r3v2-synth                # memory-preserving + 集成连接检查
make r3v2-acceptance           # 同一 run registry 去重执行所有必跑项
make r3v2-verify-evidence EVIDENCE=<run-directory>
```

各子目标通过不等于总签收；总 gate 必须识别未实现 checker，返回非零。缺少依赖、测试未运行、timeout、非预期 fail/skip/xfail/xpass 都不能签收。成功重跑必须产生新的 run ID，保留失败历史。

证据目录至少包含：

```text
contracts/     本合同版本/hash、profiles、端口/队列/复位/公平合同
inventory/     15 条 core、组件、mutation、继承回归、第二环境子清单
build/         commit/clean-tree、源码和运行资产 hash、工具锁、层级和实例参数
runs/          每次真实命令/退出码/预算/日志/oracle/计数/短 trace
runtime/       wheel 来源、安装位置、artifact、device/compiler 调用计数
model/         微图输入、先验预测 hash、逐事件比较
synthesis/     memory 端口/位数、控制状态位数、集成检查、未测物理指标
mutations/     clean 与变异结果、具体 detector、源码不变证明
inherited/     逐 case 结果及影响范围依据，不能只放历史 ACCEPTED
acceptance.json  contract_id=r3-v2-core-vrf；环境/每项结果/预算/推导判决
```

验收清单的 24 条 `R3V2-*` 标准均须有证据；一个测试可证明多个标准，但每个标准的全部子条件都必须有独立可读的检查结果。所有工作包在实施时填写实际负责人、完成提交和证据路径，不按百分比或“代码已写”签收。

## 7. 执行关口与后续衔接

| 关口 | 必须先成立 | 不可提前宣称 |
|---|---|---|
| Gate A：G0/G1 | profile、初始化、端口与预算可核验 | core 已使用 banked VRF |
| Gate B：G2/G3 | 全客户端集成、返回写回有界进展 | 小程序/产品入口已通过 |
| Gate C：G4/G5 | 数值对照、安装后运行、端口计数 | 全 kernel 模型已校准 |
| Gate D：G6/G7 | 结构、回归、负例、证据和环境齐全 | 项目生产级或原 R2/R3 已签收 |

旧合同要求的去向必须在后续计划中保持可追溯：

| 旧 R3 工作包 | 新阶段保留部分 | 延后交付 |
|---|---|---|
| R3-01/11/12 | G0/G7 的准确 inventory、预算、mutation、局部签收 | 全双环境系统发布 |
| R3-02/09 | G5 的端口/collector 微模型 | 完整编译 schedule 映射、kernel 留出精度、NUCA/NoC |
| R3-03/04/10 | G1/G2/G3/G6 的 VRF 集成与结构 | 同步 ARF 和全存储系统合同 |
| R3-05/06/07 | 客户端时序、局部计数、实现 fingerprint | 全单元吞吐、自动 bank-aware 布局与成本反馈 |
| R3-08 | 六个小 workload 与单核 runtime | BF16/tail/Norm 新性能矩阵、四 tile 产品闭环 |

本阶段验收后，新增 [Phase 3.5](phase3_5_tasks_and_acceptance.md)：在固定的已验收硬件上复用 LinearLayout 和编译优化工具，交付可观测的单核性能收益。它不依赖同步 ARF 或四 tile runtime 完成。

硬件后续仍为 R3B（ARF 和单核模型闭合），再在 R2 四 tile runtime 真正完成后做 R3C（系统校准与发布）。这些方向需各自独立 tasks & acceptance，不能拿本阶段或 Phase 3.5 的 PASS 自动填入旧合同。

**只有 Gate A–D、24 条标准、全部子条件、预算和环境证据同时通过，才能签收 R3 v2。当前状态保持 NOT ACCEPTED。**
