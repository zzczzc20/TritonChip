# 第二阶段 R2：统一 AOT 编译器与仿真 runtime 的任务及验收合同

版本：1.0，2026-09-19。状态：**规范已制定，尚未实施验收；不表示 R1 已完成。**

前置合同：[R1 tasks & acceptance](phase1_tasks_and_acceptance.md)。路线依据：[生产化审计](production_readiness_audit_2026-09-19.md)、[架构设计](architecture_overview.md)第十八节、[编译器设计](compiler_overview.md)第二、十、十四至十六节。

本文件在 R1 进行期间预先冻结 R2 的交付范围。R1 的实时进度以其实施证据为准，本文件不复制其完成状态，也不修改 R1 门槛。R2 设计、接口和隔离的原型可以提前进行；**最终验收必须以已通过的 R1 为基础，且在 R2 候选 commit 上重新通过 R1 全套。**

## 1. 阶段目标和不可替代的成果

用户在没有源码 checkout 的安装环境中，通过正式 CLI/Python API：

```text
Triton 源程序 + signature + constexpr + profile
  → 可检查、可缓存的 AOT artifact
  → 版本化 manifest + 固定机器代码 + 编译报告/IR dumps
  → runtime 分配/上传 buffer、绑定动态参数、提交 grid
  → 持续存在的 RTL 仿真设备执行若干批 program
  → 完成/错误事件
  → runtime 下载结果、回收资源、继续执行下一个 kernel
```

**同一 artifact 必须能在不重编译、不修改机器代码的情况下使用不同地址、数据、动态标量和 grid 重复运行。** 三个程序串联 A→B→C 时，中间结果必须留在同一个仿真设备的内存系统中，不能由 Python 计算或经 testbench 内部数组代传。

R2 的交付名称为“可安装的实验性 AOT 编译器与 RTL 仿真 runtime”。这不是 FPGA/PCIe 驱动、真实设备 host coherence 或量产芯片认证。

### 1.1 范围冻结

| R2 必须交付 | 后续阶段承担，本阶段不声称支持 |
|---|---|
| 非 editable 安装包，AOT CLI/Python API，正式编译 driver | 完整 PyTorch device/JIT runtime 集成 |
| 固定版本 Triton adapter，真实的窄 MLIR 编译主路径 | 任意 Triton/Gluon、任意 TTGIR 或 CUDA/PTX 特定语义 |
| manifest/ABI v1、动态参数、确定性 artifact 与缓存 | 完整 LLVM target、动态链接器、跨 profile binary 兼容 |
| 同一 2×2 RTL 设备上的任意受限 grid 与重复 launch | 4×4、硬件工作窃取、多 kernel 同时执行、抢占 |
| buffer/事件/资源准入、排空、已定义的故障恢复 | IOMMU、多租户隔离、demand paging、透明迁移 |
| 可实现为硬件协议的仿真 host 内存访问通路 | 真实 PCIe/SoC transport 或操作系统内核驱动 |
| 可消费的资源报告、确定性诊断与完整证据 | Attention、Conv、新 cache/flow policy、PPA/吞吐优化承诺 |

支持边界不可用来删除 R1 的功能。R1 的专用 GEMM、多角色、rolled pipeline 等路径必须保留；未迁移到真实 Triton 前端的能力必须如实标为 TNCA IR/AOT 输入，不得改写成“已经支持 Triton GEMM”。

### 1.2 延续 R1 的严格规则

- 所有必跑用例零非预期失败、错误、skip、xfail、xpass、timeout；retry=0，无 waiver。
- 故障注入测试预期出现的 kernel FAULT/TIMEOUT 由测试驱动验证，驱动 PASS 不等于故障 kernel SUCCESS；两类状态必须分开记录。
- 缺依赖、证据缺失、结果不完整、状态未知即未通过。不得以“R1 已测过”替代候选 commit 的 R1 回归。
- 禁止通过重编译、改汇编、插入测试专用 barrier、隐式下载/上传中间值、重启模拟器或预填正确输出实现验收正例。
- 所有数值合同、R1 profile 资源预算和必保留测试不得降低；不存在“先去掉检查让 runtime 跑起来”的路径。
- 仅同一干净候选 commit、其锁定工具链和受控 profile 的证据可以联合签收。改实现/合同/清单后必须产生新的完整验收 run。

## 2. R1→R2 接口合同及配置

### 2.1 进入最终验收前必须取得的 R1 输入

| 输入 | R2 如何使用 | 缺失时的判决 |
|---|---|---|
| R1 ACCEPTED 证据、commit/tree hash、工具锁和 normalization 规则 | 建立继承基线与回归清单 | NOT ACCEPTED |
| 三个真实 profile、capability readback、elaboration fingerprint | 编译、装载、运行一致性校验 | 拒绝编译/装载，不猜测能力 |
| 强制执行的 footprint/ownership 检查入口 | 所有 AOT 路径发射前、分配后复用 | 不得发布可执行 artifact |
| 明确的支持矩阵与全部语义 case ID | 确认迁移前后功能不退步 | 必跑集合不完整即失败 |
| 能正确失败的 runner、证据验证器及生成链 | R2 扩展而非绕过 | gate 自检失败则停止验收 |

R2 不在本文件固定 R1 尚在变化的内部函数签名。实施时必须提交 adapter 边界、公共数据结构与版本合同；使用内部接口的地方统一集中在 adapter 中，不散落到测试或 runtime。

### 2.2 Profiles 与允许的 RTL 变化

沿用 R1 的 `core-2r-sfu`、`core-4r-sfu`、`system-2x2-nosfu` 资源与 SFU 约束。三个 profile 都必须支持安装后的装载、参数绑定、基本执行与下载。通用 grid/多批调度以 `system-2x2-nosfu` 为强制整机验收对象；core profiles 仅承诺 grid=(1,1,1)，其他 grid 提前拒绝。

R2 为支持 per-tile PID/launch、host 内存访问、job generation、资源排空而修改 RTL 是本阶段任务。必须更新 profile 的实现/接口 fingerprint，并对修改后的顶层重跑 R1。**资源名可以不变，实现 hash 不得伪装成 R1 的旧 hash。** R1 旧 hash 的 artifact 在 R2 明确拒绝，或由重新编译生成新 artifact；不能默默装载。

不增加 RCTX、RF、NUCA、MSHR 或 SFU 来避开调度/分配错误。系统仍然只有 2×2；禁止把“更多 program”实现为“更多物理 tile”。

## 3. 产品合同：ABI、artifact、grid 和完成语义

本节是实现必须遵守的公共约定，不是当前代码状态描述。

### 3.1 Artifact 与 manifest v1

artifact 至少包含 `manifest.json`、不可变机器代码、只读常量区（若使用）、资源/编译报告、各段 SHA-256 和 provenance。容器格式及规范化序列化规则写入受控 schema；装载不能执行其中附带的 Python/pickle 或 shell。调试信息可以独立存储，但不得成为数值执行依赖。

manifest 的必填内容：

| 分类 | 必填字段/信息 |
|---|---|
| 身份 | artifact format/ABI/ISA 版本，kernel 名称，compiler/source/工具锁 hash，constexpr 与编译选项 |
| 目标 | profile ID、配置/实现 fingerprint、所需 capabilities |
| 代码 | code/constant 段的长度、hash、对齐、每 role entry 或明确的共享 dispatch entry |
| 参数 | 参数名/顺序/类型/offset/size/alignment、读写方向、buffer 最小范围表达式、允许的 alias 关系 |
| 执行 | active role mask、program→role 映射、支持 grid 维度、动态参数约束 |
| 资源 | 每 role 私有 RF、共享 window 的物理映射、ARF/barrier/descriptor/command 上界、spill/scratch bytes |
| 语义 | completion scope、错误合同、是否存在跨 program 依赖、所有权验证版本/结果 |

R2 artifact 仅支持一个精确 profile fingerprint；跨 profile portability 不在 v1 内。未知版本、未知必需字段/能力、重叠 code 段、越界 entry、非法 mask、冲突资源映射都拒绝。可忽略的扩展仅允许放在 schema 定义的 `optional_metadata` 中，且不能影响地址、资源、完成或数值语义。

运行时参数、buffer 地址、grid、job ID、随机 seed 不得成为编译缓存 key；明确列为 constexpr 的参数必须进入 key。编译选项、代码、依赖、profile 或工具锁改变则 key 必须改变。manifest 的 hash 用于完整性与兼容性，不宣称提供恶意生产者认证。

### 3.2 Kernel 参数 ABI v1

- 设备为 little-endian；支持 `i32/u32/i64/u64/f32/device_ptr64`。FP16/BF16 等本阶段通过 buffer 元素传递，其他标量 ABI 类型明确拒绝。
- 参数按冻结 signature 顺序占 8-byte slot，offset=8×index，argument block 8-byte 对齐。32-bit/f32 放低 4 byte，高 4 byte 必须为零；64-bit 参数占全部 8 byte。无参数的 block 长度为 0；load header/ABI 版本在 launch record 中传递，不混入参数区。
- `f32` 按 binary32 bit pattern 编码，保留 ±0 和 schema/数值合同允许的非有限值；不通过十进制字符串往返损失 bits。整数越界报错，不截断。
- API 的 pointer 参数接收本 session 的 buffer handle + byte offset；runtime 验证有效性、范围、对齐、读写权限和 alias 合同后写入设备地址。不能把任意 host pointer 当 device pointer。
- argument block 在 submit 时快照并保持不可变至任务排空；提交后修改原 Python dict/对象不能改变已提交任务。
- `launch_arg_base` 指向该参数区；kernel 从 ABI 加载动态参数。**禁止把动态地址/标量嵌入 code 或在每次 launch patch 指令。** 允许装载阶段按受控段表安排代码/常量位置，不能伪装成按参数重编译。
- v1 支持的 buffer 地址可以是模拟设备的物理地址；它不提供真实 IOMMU 隔离保证。非连续/gather 的动态越界访问必须在模拟设备内存边界检测处产生受控 fault，不能 wrap 或侵入其他已分配 buffer。

### 3.3 Grid 与调度合同

`grid=(gx,gy,gz)`，每轴是 `0..2^32-1` 的整数，负数/非整数拒绝；非空 grid 的总 program 数上限 **4096**，乘法必须检查溢出。任一维为 0 是空任务：完成正常、有唯一事件，但不启动 role、不发出 kernel 数据访问。

线性映射固定为 `linear = x + gx*(y + gy*z)`。`program_id(axis)` 和 `num_programs(axis)` 必须读到逻辑 grid 值，而不是 tile ID/当前批大小。同一 program 的所有 roles 使用同一 PID；一个 program 的所有角色及其共享资源作为一个整体驻留同一 tile。

R2 采用确定性批次调度：按 linear ID 递增，将至多四个 program 放到四个 tile；不足四个的尾批只启用需要的 tile。允许每批完全排空再进入下一批，不要求跨批 overlap；禁止每批重启模拟器或重新编译 kernel。grid 上限 4096 的静态校验与映射由 host/unit 测试穷尽验证，不要求 RTL 逐次执行 4096 个 program；RTL 的强制规模见 E2E 矩阵。

R2 不支持跨 program barrier 或依赖其他 program 先运行的自旋协议；编译器/API 对已识别的这类语义明确拒绝。测试不得使用没有合同的数据竞争。设备原子操作仍保留 R1 语义，但不使上述跨 program 进展依赖合法。

### 3.4 Session、事件和内存可见性

每个 session 同时最多一个 RUNNING kernel；submit 以 FIFO 排队，至少支持 **8 个未终结任务**（含 RUNNING）。第 9 个 submit 在队列满时返回明确 `QUEUE_FULL`，不得阻塞 API 无限等待或覆盖旧任务。

事件状态机：

```text
SUBMITTED → VALIDATED → QUEUED → RUNNING → DRAINING → SUCCEEDED
                 任意运行故障 → FAILING → DRAINING → FAILED
                 已知无法排空 → DEVICE_LOST（session POISONED）
```

装载/参数/admission 错误在任何角色启动前返回，不产生 RUNNING 事件。每个已接收任务恰有一个终态；job ID 在 session epoch 内唯一。`submit` 支持显式 `depends_on` 事件列表；只接受本 session 已提交的事件，不允许向未来事件引用，因此不能形成环。FIFO 保证执行顺序，依赖事件另决定能否执行：任一前驱失败，后继以 `FAILED/DEPENDENCY_FAILED` 终结且不启动任何 role。A→B→C 示例必须声明依赖；FAILED 任务写过的 buffer 标为结果无效，必须先由显式 upload 或新的完整覆盖任务重新建立有效内容，不能让普通读任务消费。

`wait(event)` 幂等；普通 host 等待超时只表示还没观察到终态，不能把任务标为成功或自动取消。设备执行 deadline 是另一项配置，超时按故障路径处理。

`SUCCEEDED` 的必要条件：所有 program/role 结束，DMA/MMA/LSU/return/writeback 和相关完成事件按合同排空，不会再写旧 RF/window；输出已能通过**正式 runtime 下载接口**读取；资源与参数块可以安全回收。

host 上传/下载只能经过定义明确的 transport 请求：普通受仲裁的 home-slice 访存，或可综合的 quiesce + clean/invalidate/维护端口协议。具体方案在实现提交中冻结，必须实测 dirty cache 和 host 覆盖场景。**不得用 testbench 层级引用读取/写入 cache tag/data/RF 数组来实现产品数据接口。** testbench 内部观察仍可用于验证断言，不能作为 runtime 结果来源。

host 内存访问在设备 quiescent 时进行；buffer 有未排空任务租约时，free/上传/下载返回 `BUFFER_BUSY`，不隐式等待、不改变数据。A→B kernel 依赖通过 FIFO/事件和设备内存实现，中间无需下载；需要 host 读取时显式 wait 后 download。FAILED 的输出可为部分结果，必须标为无有效结果保证；不承诺事务回滚。`close` 在有未终结任务时返回 `SESSION_BUSY`，不得隐式丢弃任务；设备已丢失时允许关闭并清理进程。正常 drain 后 close 释放全部资源，重复 close 幂等。

## 4. 全部工作包及必跑用例

各用例 ID 的所有子条件都必须执行；实现时生成受控 inventory，展开 profile、输入、seed 和环境，不能把一行中的多个场景只做一个。

### R2-01：R1 衔接、接口冻结和支持矩阵

责任：维护者/编译器/runtime。

交付：R1 输入锁定清单；R2 schema/ABI/API 文档；编译器、runtime、transport、test runner 四者依赖边界；语义 case 映射；实际负责人。

| ID | 验收 |
|---|---|
| DEP-01 | 缺失/伪造/不完整 R1 验收证据，R2 gate 非零；R1 的实时 PARTIAL 状态不能被当作前置通过 |
| DEP-02 | R2 候选上完整运行 R1，全部 case 保留且通过；实现 fingerprint 变化有可追踪映射 |
| DEP-03 | 每项对外能力都有入口、profile、参数约束、正负例；没有不加限定的“支持 Triton/GEMM/异步”声明 |

R1 验收后形成的修复可以在 R2 候选继续演进，但必须重新满足 R1；不要求 R2 候选与 R1 签收 commit 字节相同。

### R2-02：安装包、CLI 和公开 API

责任：构建/编译器/runtime。

交付：带锁定依赖和入口的非 editable wheel/sdist；独立编译包与运行包边界；正式 `tnca-aot`、`tnca-sim` 命令以及 Python API。以下是待实现的稳定入口合同：

```text
tnca-aot compile / inspect / doctor
tnca-sim run / doctor
Python: compile_aot, inspect_artifact
Python Session: allocate, upload, load, submit, wait, download, free, recover, close
```

`compile` 输出 artifact；`inspect` 无需启动 simulator；`run` 通过同一 runtime API 执行。精确参数、返回类型、JSON schema 与错误码必须受控记录。kernel fault/loader error/transport lost 的 CLI 返回非零。导入包不能运行测试、连接网络或启动 simulator。

| ID | 验收 |
|---|---|
| PKG-01 | 两个全新环境非 editable 安装；将仓库从 sys.path/PYTHONPATH 和 cwd 中移走后完成 compile→run→download |
| PKG-02 | 编译与运行均不 import `compiler/test`、不读取 checkout 内 `sim/build`；从任意 cwd 和含空格路径成功执行 |
| PKG-03 | 运行端只有 artifact、runtime、锁定 simulator/运行资产，没有 Triton/LLVM/编译器也能执行；不得在 launch 自动拉取依赖 |
| PKG-04 | 缺运行资产、依赖不匹配、只读/无权限输出路径给稳定诊断；CLI 与 Python API 成败和数据一致 |
| PKG-05 | 包内许可证、版本、profile schema、必需运行资产完整；卸载重装不依赖开发者缓存 |

允许在安装/显式 prepare 阶段构建 profile 的 simulator，不能在每个 kernel launch 做 build；prepare 工件必须有 R1 fingerprint 并归入证据。

### R2-03：统一编译 driver 与固定 Triton adapter

责任：编译器。

交付：`source + signature + constexpr + profile + options → artifact` 的单一产品入口；从 test bridge 移出必要逻辑；未知选项拒绝；source/function 选择及参数绑定统一。所有入口调用 R1 安全检查。

| ID | 验收 |
|---|---|
| CMP-01 | 真实 Triton add、mul-add、带动态 f32 alpha 的 AXPY 经公开入口编译；与独立参考结果相同 |
| CMP-02 | 不存在函数、多函数 module、dict 乱序、未知 option、dtype/signature 不符分别按合同处理，不捕获宽泛异常后悄悄降级 |
| CMP-03 | 将 R1 两个审计负例及全部适用 OWN/FP 负例经产品输入路径送入；必须拒绝，debug/offline/优化选项不能绕过 |
| CMP-04 | 编译失败不得留下可被 loader 当成功 artifact 消费的半成品；原有同路径 artifact 不被破坏 |
| CMP-05 | R1 专用 TNCA IR GEMM/多角色输入经明确的第二输入 adapter 生成同格式 artifact；入口标明 `tnca-ir` 而非冒充 Triton |

不要求 `@triton.jit` 直接通过 PyTorch launch；但编译端必须实际调用固定 Triton 的前端，不能用预存 TTIR 替代 CMP-01。

### R2-04：窄 MLIR 主路径与控制流边界

责任：编译器。

交付：真正消费 TTIR 的 pass pipeline；通用算术/控制流保留 `arith/math/scf` 等语义，TNCA 表达资源/异步；真实 pass 边界 dump、pre/post-allocation verifier 与最终机器发射连接。现有 Python emitter 可作为最末端机器编码/汇编工具，不能使 MLIR 仅成为不参与产物的旁路展示。

R2 新增的通用 TTIR 控制流限定为：**uniform scalar `scf.if`（含 else）和单层 `scf.for`，trip count 为动态 i32 且约束为 0..8**；只允许本阶段已支持的同步 elementwise/访存和循环携带值。跨该通用 CFG 的异步租约、任意嵌套循环、while、递归明确拒绝。R1 的已验证专用异步 rolled pipeline 独立保留，不能因本限制而退步。

| ID | 验收 |
|---|---|
| MLIR-01 | CMP-01 三类真实 Triton 程序经 MLIR 主路径输出机器代码并在 RTL 执行；删掉/破坏实际 lowering pass 必须使编译失败 |
| MLIR-02 | if 的 true/false、for trip={0,1,2,3,8} 全部执行；同一 artifact 用动态参数切换分支/次数且结果正确 |
| MLIR-03 | trip<0 或 >8 在 launch 约束处拒绝；不支持 CFG/跨 CFG async 在编译期拒绝，不能生成未检查的指令 |
| MLIR-04 | MLIR 主路径与保留的 Python reference 对共同支持输入做数值差分；包括 masked tail、alias、不同标量和角色映射 |
| MLIR-05 | dump 包含真实 TTIR、目标执行/协议表示、分配结果、机器/汇编；每份都有 stage ID/hash，禁止重复打印同一 IR 冒充多个 pass |

本阶段不要求全部 R1 专用 pipeline 都重写成 MLIR，也不要求完整 `TNCABackend.add_stages` 的 upstream JIT 集成；未迁移入口仍必须受 R1 检查并在报告中标明实际编译路径。

### R2-05：Artifact/manifest schema 与 loader

责任：编译器/runtime。

交付：第三节的格式 v1、schema 校验、完整性检查、profile 绑定、原子写入，以及执行前 loader/admission。配置由 manifest 与真实设备 readback共同校验，不能只比较两份来自同一 JSON 的拷贝。

| ID | 验收 |
|---|---|
| ART-01 | 正常 artifact 可 inspect、装载、执行，所有必填字段完整；manifest 中资源总量与实际分配逐项一致 |
| ART-02 | code/constant 改一字节、截断、段重叠、entry 越界、重复 JSON key、未知版本/必需字段各拒绝 |
| ART-03 | profile/hash/ISA/capability 不匹配、非法 role mask/资源重叠各在第一条指令前拒绝，无 kernel 写副作用 |
| ART-04 | 丢失文件、混入另一 artifact 段、嵌套路径越界等损坏拒绝；加载不会执行 artifact 内代码作为 host 脚本 |
| ART-05 | 对资源/参数/代码段大小做溢出检查；超出设备可用空间时拒绝而非分配一半后启动 |
| ART-06 | 编译报告/调试信息被移除时：必需执行合同缺失则拒绝；schema 明定的可选 debug 缺失不改变执行结果 |

v1 默认拒绝更高或未知 format/ABI major/minor；本阶段不承诺旧实验文件兼容。当前 v1 的 golden artifact 必须能在独立安装环境中装载，不能要求重新运行源码生成。

### R2-06：动态参数 ABI 与 buffer 合同

责任：编译器/runtime。

交付：ABI 编解码、代码中的参数加载、signature 约束和安全参数快照。loader 使用 typed buffer handle；资源租约和生命周期由 runtime 管理。

| ID | 验收 |
|---|---|
| ABI-01 | 六种 ABI 类型按独立 byte-level oracle 校验 offsets/大小/端序；整数边界、f32 ±0/有限值/允许的 NaN/Inf 均保持合同 |
| ABI-02 | 同 artifact 用至少三组不同 buffer 地址、N、f32 alpha、数据执行；code hash 不变，compiler 调用次数保持 1 |
| ABI-03 | 缺/多参数、越界整数、错误类型、坏对齐、buffer 太小、offset 加法溢出、过期/其他 session handle 全部拒绝 |
| ABI-04 | 提交后修改调用方 dict/标量容器及重用临时参数对象；已提交任务只看到 submit 快照 |
| ABI-05 | 合法 in-place alias 的 elementwise 正确；未声明可 alias 的输入输出重叠提前拒绝；不依赖执行调度碰巧安全 |
| ABI-06 | 非连续运行时 gather/scatter 越界产生准确设备内存 fault；不能写坏相邻 buffer 的 canary |

静态可验证的地址范围必须在 launch 检查；依赖设备计算的地址由明确的模拟内存边界检查负责。这个边界检查要在正式 transport/device model 中，不是只在测试 oracle 中。

### R2-07：编译缓存、确定性与不可变代码

责任：编译器/构建。

交付：内容寻址缓存、并发安全发布、hit/miss 原因及 trace；运行端不具备偷偷触发编译的依赖链。

| ID | 验收 |
|---|---|
| CACHE-01 | source、constexpr、选项、profile、工具锁分别改变时 cache miss；动态地址/值/grid 改变时仍复用原 artifact |
| CACHE-02 | 相同输入在两个干净环境构建，规定的 code/constant/manifest 规范化 hash 相同 |
| CACHE-03 | 两进程同时编译同 key，不生成半成品；注入中断/损坏缓存后明确拒绝或受控重建，不消费坏 code |
| CACHE-04 | runtime-only 环境连续 launch，编译/汇编/iverilog 调用计数为 0，code pages/hash 不变 |

允许缓存损坏后的**显式编译请求**重建；不允许 runtime launch 隐式修复。比较 normalization 只可去掉提前列出的非语义路径/时间戳。

### R2-08：Runtime session、buffer、队列和事件

责任：runtime。

交付：第三节 API/状态机、持久 simulator session、buffer allocation registry、引用/租约计数、FIFO queue 和显式错误类型。

| ID | 验收 |
|---|---|
| RT-01 | allocate→upload→load→submit→wait→download→free 的完整生命周期；0-byte allocation 明确拒绝，零参数 kernel 仍可执行 |
| RT-02 | 同 session 32 次成功 launch，地址/输入交替；不重启 simulator、不残留资源；中间含 A→B→C 设备内存链 |
| RT-03 | 同时接收 8 个未终结任务，FIFO 执行；满时第9个返回 QUEUE_FULL；任务终结释放槽后可再次 submit |
| RT-04 | wait 重复调用、事件乱序查询、host wait 超时再 wait：终态唯一、任务不重复执行、不自动取消 |
| RT-05 | 活跃租约下 free/upload/download 返回 BUFFER_BUSY；终结并排空后可用；double free、过期事件/handle 受控拒绝 |
| RT-06 | 两 session 使用同名 kernel/buffer 且不同数据，文件、地址空间、事件互不混淆；busy close 拒绝，排空/设备丢失后的 close 无进程泄漏，重复 close 幂等 |
| RT-07 | A 成功时 B/C 按 depends_on 执行；A 失败时 B/C 不启动并报告 DEPENDENCY_FAILED；跨 session/未知依赖拒绝；无效输出 buffer 不能被普通读任务消费 |

队列 full 测试通过可控 device hold 维持占用，不能依靠 host 调度时序碰巧形成队列。缓冲区的资源计数与执行事件状态必须分别记录。

### R2-09：多批 grid、PID 与整体资源准入

责任：runtime/RTL/编译器。

交付：per-tile launch mask、PID/NPID/arg_base、逻辑→物理映射、资源 reservation、尾批停用，以及批次之间的 drain。调度可以由 host runtime 控制，但每个被调度 program 的计算必须由 RTL 完成。

| ID | 验收 |
|---|---|
| GRID-01 | grids={(0,1,1),(1,1,1),(3,1,1),(4,1,1),(5,1,1),(9,1,1),(17,1,1),(2,3,1),(2,2,2)}，每个逻辑 PID 恰好执行一次 |
| GRID-02 | kernel 输出 PID/NPID 编码及 per-program 写次数；结果与独立映射 oracle 一致，证明未把 tile ID/批大小冒充逻辑值 |
| GRID-03 | 尾批 inactive tile 不取指、不发出该任务的数据请求；empty grid 无角色启动与 kernel 内存写 |
| GRID-04 | 同一多角色 artifact 在 grid=1/4/5 下执行，roles 不拆到不同 tile；共享 window/barrier 不被另一个 program 占用 |
| GRID-05 | 资源恰好够、任一资源差1、非法共享映射、角色数不足分别验证；不够时在任一 role 启动前拒绝 |
| GRID-06 | host 映射测试覆盖 program 总数0..4096、3D 边界；4097、负数、32-bit 轴越界、乘积溢出明确拒绝 |
| GRID-07 | 相邻批交替不同数据，所有上一批异步响应排空后才复用其 RF/descriptor/barrier；计数与 generation trace 可核验 |

GLOBAL grid scheduler 不承担核内 `num_warps` 映射；不得把它等同于 RCTX 数。新 launch 接口要有 busy/admission 握手，不能在执行中用 `launch_valid` 无条件清掉旧状态。

### R2-10：Host 内存 transport 与真正完成边界

责任：runtime/RTL/验证。

交付：第三节的正式 upload/download 协议、dirty-line 可见性、host overwrite 规则、全设备 quiescent 判定。必须覆盖 core idle 之外的 NI/mesh/slice/DRAM/返回与写回在途状态。

| ID | 验收 |
|---|---|
| MEM-01 | kernel 输出驻留 dirty cache，尚未自然 eviction；wait 成功后 download 得到新值，不从 DRAM 旧值或内部数组旁路读取 |
| MEM-02 | 已缓存 buffer 由 host 在 quiescent 时上传新值，再运行 kernel；读取新值而非 stale cache |
| MEM-03 | A 写入→B 读取/变换→C 校验，无 host 中间下载；同一 session 跨多批和重复 launch 数值正确 |
| MEM-04 | 对最后一个 store/写回/返回注入延迟；不能提前 SUCCEEDED/free/重用参数块；释放延迟后恰好终结一次 |
| MEM-05 | 运行接口审查与动态 hook 证明 upload/download 不引用层级 cache/RF/tag/data 数组；禁用旧 harness dump 后产品 E2E 仍通过 |
| MEM-06 | 每个 host 请求有地址/长度/方向/响应和完成记录；失败不伪装成功，upload/download 超范围拒绝且 canary 不变 |

MEM-01 不强制“kernel 完成时所有 dirty line 必须已写回 DRAM”：如果正式 host 访问经 home slice 保证一致可见即可。若选择 flush 方案，则 flush 完成是事件成功的前置条件，必须等待其写回应答。

### R2-11：故障、deadline、排空与恢复

责任：runtime/RTL/验证。

交付：带 job/epoch/role/PC/指令或地址信息的错误事件；known fault 的 drain；timeout/transport lost 的 poison 与显式 recover。正常 launch 不 reset 设备；失败恢复不得无声地把旧结果当新结果。

| ID | 验收 |
|---|---|
| ERR-01 | 非法指令、descriptor fault、越界内存、barrier error 分别注入，事件 FAILED，准确归属，不能 SUCCEEDED |
| ERR-02 | 故障时仍有 DMA/MMA/LSU 在途：禁止新工作，完成/释放路径仍进展；可排空时保留 session，后续独立合法任务成功 |
| ERR-03 | 注入不返回的事务触发设备 deadline：session POISONED，未运行任务以 FAILED 终结并给出 DEVICE_LOST 原因；不能让队列永久等待或继续使用有风险资源 |
| ERR-04 | `recover` 建立新 epoch，重置/重建设备及 transport，全部旧 buffer/module/event 句柄失效；重新上传装载后合法任务成功 |
| ERR-05 | 旧 epoch/旧 program 的迟到、重复响应或完成消息：被识别并丢弃/报错，不写新 RF/buffer、不改变新事件 |
| ERR-06 | simulator 异常退出、transport 消息截断/版本不匹配：DEVICE_LOST，保留日志，CLI 非零，无悬挂子进程 |
| ERR-07 | 16 次交替故障/正常提交，覆盖 drain 后复用和 recover 后重建；事件/资源计数无泄漏，未执行任务不被报告为执行过 |

允许不可排空时 recover 重启仿真进程，但这是显式、可观测的设备丢失恢复；不能用于 RT-02、GRID 多批或正常成功路径。FAILED 输出不保证回滚，API 必须禁止把它当成功结果；隔离不相关 buffer 的 canary 必须完整。

job epoch 可以通过显式 tag 或“完整 drain + transport epoch 隔离”实现；必须用 ERR-05 证明，而不是声称旧响应不会发生。

### R2-12：真实资源报告、诊断和用户可复现示例

责任：编译器/runtime。

交付：编译报告自动接入产品 driver；包含每 role RF/共享区、ARF/barrier/descriptor、实际 K/dtype footprint、DMA 展开计数、channel/role 计划、拒绝方案与原因；runtime 报告 grid/batch、job、执行/等待周期、fault、排空结果。

| ID | 验收 |
|---|---|
| OBS-01 | 报告资源与 manifest、allocation map、发射指令交叉核对；没有测量/模型支持的成本标为 unknown，不能填0冒充零开销 |
| OBS-02 | 强制不足资源、错误 ABI、错误 profile、bad IR，诊断给 source/IR/role/resource 位置及稳定错误码；正常用户错误无裸 traceback |
| OBS-03 | 保存失败 artifact、参数元数据、输入/seed、trace 与重放命令，在同锁定环境复现相同语义错误；不要求复现 host PID/时间戳 |
| OBS-04 | 至少提供三个安装后可直接运行的示例：动态 masked AXPY；grid>4 的三程序链；多角色 GEMM + 故障恢复示例 |

GEMM 示例允许使用明确标注的 TNCA IR 输入，但必须经产品 compiler/loader/runtime，不 import 测试模块。报告不承诺本阶段准确预测吞吐。

### R2-13：端到端矩阵、差分与 mutation

责任：验证。

所有比较使用独立参考计算，不得由 runtime 代算输出。seed 集合固定 `{1,7,42}`；值生成、内存 stall/reorder 的 seed 分别记录与传递。所有 mask/尾块输出区使用 sentinel 与邻接 canary 验证未触达范围。

| ID | 必跑矩阵 |
|---|---|
| E2E-01 | 真实 Triton add/mul-add/动态 AXPY，N={0,1,15,16,17,63,64,65}，BLOCK=16，grid.x=ceil(N/16)，三个 seed 全跑 |
| E2E-02 | GRID-01 九种 grid，用 PID/NPID 输出 kernel 和至少一种真实数值 kernel 各跑；empty grid 无计算输出，两个3D grid必须真执行 |
| E2E-03 | MLIR-02 两个分支与五种 trip count、动态参数复用、masked tail，在 core 与 system 各通过；profile 能力按实际支持核验 |
| E2E-04 | R1 TNCA IR GEMM：FP16/BF16，K={16,32,64,128}，M/N={16/16,13/11}，两角色，fast/slow/ooo；尾 K=19 另测；资源不变 |
| E2E-05 | RT-02/ERR-07 的持久 session 序列与 MEM-01..04 的一致性/排空场景，在三个 seed 下执行 |
| E2E-06 | core-4r 的双消费者程序经安装后的 compiler/runtime/API 执行；最后消费者释放前不能复用，保留 R1 数值结果 |

E2E-04 在 `core-2r-sfu` 上跑完整数值/内存矩阵；system 上另跑 FP16/BF16 K32、M=N=16、grid=5 的多角色多批 GEMM，以及一组 M=13/N=11/K=19 尾块。这里的 K128 是专用 GEMM 多个 K-step，不新增 K128 ISA 编码。动态 GEMM shape 不属本阶段新增要求，运行地址/输入仍必须动态。

为证明新 gate 有效，以下受控 mutation 各自必须被指定测试抓住，未变异正例通过：

| ID | 破坏 | 检出测试 |
|---|---|---|
| R2M-01 | 把动态 pointer/N 固化为第一次 launch 的值 | ABI-02、CACHE-04 |
| R2M-02 | PID 恢复为 tile ID 或 NPID 恢复为4 | GRID-01/02 |
| R2M-03 | 最后 store/return 未完成就发送成功事件 | MEM-04 |
| R2M-04 | download 改为只读 DRAM 绕过 dirty cache 一致性 | MEM-01 |
| R2M-05 | 移除 artifact/profile hash 或能力检查 | ART-02/03 |
| R2M-06 | 恢复允许旧 epoch 完成新事件/写资源 | ERR-05 |
| R2M-07 | 让 MLIR pass 只 dump 不参与实际编译 | MLIR-01/05 |
| R2M-08 | 在成功 launch 间偷偷重启 simulator 或重编译 | RT-02、CACHE-04 |

全部在隔离副本运行，变异不进入候选源码；检测必须对应故障根因，不能靠无关的 import/build 失败算检出。

### R2-14：双环境交付、CI 与最终证据

责任：构建/CI/维护者。

新增稳定入口（以下是待实现接口，并非宣告已存在）：

```sh
make r2-doctor
make r2-package
make r2-gate-selftest
make r2-acceptance
make r2-verify-evidence EVIDENCE=<run-directory>
```

两套独立 Linux x86_64 环境 A/B 均从同一干净候选及锁文件构建、非 editable 安装、执行全部 R1+R2 必跑矩阵；B 不复制 A 的 build/venv/安装产物。允许已校验只读下载缓存；另用 runtime-only 环境证明 PKG-03。不能用 macOS 或本机增量测试替代。

沿用 R1 的单测试、子进程回收、native build、bootstrap 上限。R1 子验收仍受其原 7200 秒上限；R2 新增测试部分上限 14400 秒；组合验收（不含 bootstrap/build）上限 21600 秒。单次模拟器调用仍为600秒，不得将长序列拆成重新初始化设备以掩盖该上限；允许在一次持久 session 中按用例分段计时，所有段共享 session/epoch trace。发布 retry=0，正常 host wait deadline 与注入的设备 deadline单独记录。

除了 R1 证据，必须归档：安装包/hash、安装日志、隔离路径证明、schema/ABI/golden artifact、编译调用和缓存 trace、simulator build/start 计数、所有 job/event 状态变化、PID→tile/批次映射、资源 reservation/release、transport 内存访问、generation/恢复 trace、各 E2E 的独立 oracle 输入输出、mutation 结果与完整日志。

| ID | 验收 |
|---|---|
| REL-01 | A/B 全套通过；code/constant/规范化 manifest 和确定性结果一致；任何环境缺依赖都不是 SKIP |
| REL-02 | 安装包可离开 checkout 运行；runtime-only golden artifact 正常装载，未知 ABI 版本拒绝 |
| REL-03 | 删除一个用例/输出、篡改一个字节、混入别的 commit/profile/epoch/session、缺日志、伪造计数分别令证据验证器失败 |
| REL-04 | required==discovered==executed（展开 profile/seed/环境后），全部 PASS；任何 mutation 未被检出、孤儿进程、缺终态都使 gate 非零 |
| REL-05 | 文档、API/schema、支持矩阵、示例、全部工作包责任人与证据路径齐全；不得留下影响必交付范围的 TBD/例外 |

## 5. 重构边界与实施顺序

R2 不以“移动了目录”验收。以下依赖必须实际成立：

```text
公开 CLI/API → compiler driver → Triton adapter / MLIR passes / 安全发射
公开 CLI/API → runtime → artifact/ABI/admission → scheduler → transport → RTL
verification → 调用上述公开接口 + 独立 oracle/故障注入
产品 compiler/runtime 不依赖 verification/test 模块
```

原 test harness 中可复用的产品逻辑搬到正式模块，测试反向调用。原 Python IR/planner 保留为明确的第二输入/reference adapter；不能让公共运行入口调用测试 helper 拼装代码/断言。

| 批次 | 工作包 | 完成后可依赖的结果 |
|---|---|---|
| 1 | R2-01、R2-05/06 的 schema 合同 | R1 衔接、profile fingerprint、artifact/ABI 不再由各模块自行猜测 |
| 2 | R2-02、R2-03、R2-04 | 安装后的真实编译入口与窄 MLIR 主路径 |
| 3 | R2-05、R2-06、R2-07、R2-08 | 不变 code + 动态参数，持久 session、buffer 和事件 |
| 4 | R2-09、R2-10、R2-11 | 实际 grid、整体资源准入、内存可见性、排空和恢复 |
| 5 | R2-12、R2-13、R2-14 | 可用报告/示例、差分与 mutation、双环境最终签收 |

批次间可以提前开发接口，但不能用 stub/mock 的成功结果验收依赖真实 RTL 的用例。unit/mock 用于定位问题，E2E/内存/恢复判据仍必须落到实际实现。

## 6. 最终判决与签收表

```text
已有有效 R1 基线证据
AND 同一 R2 候选 commit 的 R1 全套再次通过
AND 安装包在 checkout 外完成真实编译与执行
AND 动态 ABI、manifest、cache、loader 全部正负例通过
AND MLIR 主路径真正参与受支持程序的机器代码生成
AND 同一 artifact/session 跨参数、grid、program 批次重复运行
AND 正式 transport 保证 host 结果可见、旧 cache 不污染新输入
AND 所有任务恰有一个终态，排空后才回收，迟到响应不越代
AND 已知 fault 可 drain、不可 drain 则 poison/recover，后续任务正确
AND R1+R2 用例集合完整，无非预期 fail/skip/xfail/xpass/timeout/retry
AND 所有 mutation 被检出，双环境工件与证据验证通过
AND R2-01 ... R2-14 全部 PASS
=> R2 ACCEPTED
否则 => R2 NOT ACCEPTED
```

| 工作包 | 初始状态 | 负责人 | 最终证据 |
|---|---|---|---|
| R2-01 R1 衔接/合同 | NOT VERIFIED | 实施时填写 | 未交付 |
| R2-02 安装/API | NOT VERIFIED | 实施时填写 | 未交付 |
| R2-03 编译 driver | NOT VERIFIED | 实施时填写 | 未交付 |
| R2-04 MLIR 主路径 | NOT VERIFIED | 实施时填写 | 未交付 |
| R2-05 artifact/loader | NOT VERIFIED | 实施时填写 | 未交付 |
| R2-06 动态 ABI | NOT VERIFIED | 实施时填写 | 未交付 |
| R2-07 缓存/确定性 | NOT VERIFIED | 实施时填写 | 未交付 |
| R2-08 runtime 生命周期 | NOT VERIFIED | 实施时填写 | 未交付 |
| R2-09 grid/admission | NOT VERIFIED | 实施时填写 | 未交付 |
| R2-10 host transport/完成 | NOT VERIFIED | 实施时填写 | 未交付 |
| R2-11 故障/恢复 | NOT VERIFIED | 实施时填写 | 未交付 |
| R2-12 报告/诊断/示例 | NOT VERIFIED | 实施时填写 | 未交付 |
| R2-13 E2E/mutation | NOT VERIFIED | 实施时填写 | 未交付 |
| R2-14 双环境/发布 | NOT VERIFIED | 实施时填写 | 未交付 |

**编写本合同不改变 R1 或 R2 的验收状态。任何一项缺失，均不得用“主流程能跑”“只剩工程化”或“下一阶段再补”宣布 R2 完成。**
