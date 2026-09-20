# TritonChip 当前状态审计与生产化路线

审计日期：2026-09-19。基线：`6c90063`（工作区审计，非隔离的干净 checkout）。

设计依据：[架构原始设计](architecture_overview.md)、[编译器原始设计](compiler_overview.md)。审查范围涵盖 RTL、Python 编译路径、MLIR 方言与检查器、测试入口、生成链、综合流程及现有状态文档。本次只增加审计材料，不修改实现。开始时已有 `flow/synth/synth.py` 未提交修改，保留原样；涉及该文件的结论针对当前工作区。审计结束时另发现 `Readme.md` 和 `doc/synthesis.md` 被外部更新了综合数据；已核对差异并保留，本次没有写入这两个文件。没有重新核验原设计引用的论文、上游版本最新性，也没有访问 XLS 配置中的远程机器。

## 1. 结论与下一阶段目标

第一阶段的完整实施范围与不可豁免验收门槛见 [R1 任务与验收合同](phase1_tasks_and_acceptance.md)。该文件细化本报告的阶段建议；R1 的完成判定以该合同为准。

**项目已具备有价值的功能原型：真实 Triton 前端的一小部分程序能经 Python lowering 运行在四 tile RTL 上，GEMM、多角色 ready/free 协议、NUCA、NoC 也有专门测试。当前尚不具备生产编译器、生产 runtime 或可签核芯片的交付条件。**

最先需要解决的是可信度链条，而不是增加算子数量：

1. 异步所有权检查存在已复现的漏检；编译器能生成 ISA 明确禁止的源寄存器覆盖序列。
2. E2E 测试入口会掩盖失败退出码；部分构建依赖缺失可能让旧仿真产物参与“验证”。
3. 声称描述四 tile 系统的 target 文件宣告 SFU 可用，但 tile 实例关闭了 SFU。
4. 当前的网格参数化、Triton 插件、运行时和 MLIR lowering 均未形成完整产品闭环。
5. RF 端口、存储宏、时序和物理实现仍是模型与原型层面的能力；不能从功能测试或通用门深推导生产 PPA。

**建议下一阶段设为 R1：“可复现、会正确失败、异步资源安全的 2×2 AOT 系统基线”。** R1 结束前不把 4×4 扩展、复杂 cache policy、Attention 或完整 LLVM target 作为主线。先修复本报告 F01–F05，建立明确的支持矩阵和 release gate，再把已有功能接到统一 AOT 入口与 manifest。

生产化分三个独立验收层次：

| 层次 | 实际交付物 | 完成含义 |
|---|---|---|
| 软件可交付 | 安装包、AOT 编译器、版本化 manifest、仿真 runtime、支持矩阵 | 新环境可复现；不支持的输入明确失败；已支持程序正确且可诊断 |
| 板级可运行 | FPGA/原型板、真实内存与 host 接口、驱动、恢复机制 | 脱离 testbench 内部状态访问，持续执行多个 kernel 并正确回收资源 |
| 芯片可量产 | 指定工艺/IP 下的 RTL 与物理签核、DFT、测试与可靠性方案 | 时序、面积、功耗、可测性和系统验证闭合；不由软件测试替代 |

这三者不能合并为一个“production-ready”标签。工艺、封装、内存接口、功耗预算、目标频率和部署模式尚未在仓库中形成产品合同；下面给出的是依赖明确的路线，不是承诺流片日期。

## 2. 对照原设计的完成度

以下评价区分“已有受限实现”和“原设计整体完成”，不沿用 README 的 done 标签。

| 原设计要求 | 当前代码证据 | 审计判断 |
|---|---|---|
| RF → NUCA → DRAM，无软件可寻址 SPM | `rtl/core`、`rtl/nuca`、`rtl/noc`、`rtl/tnca_top.v` | 原型中贯彻；值得保留 |
| 最多四 RCTX，角色与 SIMD lane 分离 | core 参数、多角色 pipeline、`core4` profile | 已有 2/4 RCTX 单核验证；不能据此证明所有系统 profile |
| DMA/MMA 通过 RF window 通信，ready/free 分离 | `pipeline.py`、`partition.py`、barrier/tensor/matrix RTL | 核心机制已实现；编译检查存在缺口，见 F01/F06 |
| 小型 TNCA 执行层，保留上游算术与控制流 | `frontend.py` 转入自定义 Python IR；C++ dialect 独立存在 | bring-up 路径合理，但生产路径尚未迁移到原设计的混合 MLIR |
| C0：工具链、target、每阶段 IR dump | pin、生成脚本、`tnca-opt` 已有；backend 的 STAGES 只是声明 | 部分完成；target 不一致、真实 pass/dump pipeline 未闭合 |
| C1：真实 Triton → RTL | `triton_bridge.py`、`test_triton_e2e.py`、`test_e2e_system.py` | 受限闭环成立；frontend 只接受一个向量寄存器宽度的 block，不支持通用 scf/dot 路径 |
| C2：FP16/BF16 GEMM、tail、stride | `gemm.py` 和 GEMM RTL 测试 | 手工构造 TNCA IR 的 GEMM 路径已建立；不等于任意 Triton GEMM 已贯通 |
| C3：显式两角色异步流水 | `test_pipeline.py`，slow/ooo 测试 | 有实质覆盖；编译器安全性仍不能由正例测试推定 |
| C4：自动分区、循环克隆、P+M0+M1+E | `partition.py`，直线 TTIR 分区与专用 GEMM 循环 | 部分完成；README 自述尚无 literal P+M0+M1+E，也未从 TTIR 驱动通用循环 |
| C5：Attention 和性能反馈 | report 结构已有；未见完整 Attention 编译/系统测试 | 未完成 |
| Stage 4：cache hint、bypass、共享组进度与准入 | 当前 NUCA 基线；target 的 cache_hints=false | 尚未建立独立机制与消融结果，不应描述为论文机制已落地 |
| Stage 5：4×4 | 顶层 NX/NY，但 NI home 与 source ID 固定四 tile | 参数化不完整，见 F05 |
| launch manifest、资源准入、退出排空、job generation | launch wires、testbench、简单 launcher | 只有部分硬件接口；缺少真正版本化的 host/runtime 合同 |
| 可制造的 RF 与 datapath | 异步读、多端口、全数组复位的 RF；Yosys 门级代理 | 功能模型/原型，未完成生产存储与 PPA 闭环 |

当前默认系统与原始参考规模不同：4 tile、2 RCTX/tile、128 个物理 V 寄存器（8 KiB/tile）、64 sets × 8 ways × 64 B（32 KiB/slice）、8 MSHR/slice；原方案的 16 tile、32 KiB VRF、512 KiB/slice、96 MSHR/slice 是设计起点，不是现有实现参数。矩阵 RTL 的计算循环也不能按 M16×N16 的指令形状直接当成每周期 256 MAC 的阵列。

## 3. 按优先级排列的问题与解决方案

优先级：P0 为阻断可信发布的正确性/验证问题；P1 为支持范围扩展或系统运行前必须完成；P2 为规模、性能与量产工程。标记“复现”表示本次执行过；“静态确认”表示代码直接可见；“待验证风险”不等于已证明系统失效。

### F01 — P0：异步所有权记录未接入写入检查【复现】

证据：`compiler/python/tnca_cc/codegen.py:436` 的 `_op_window_put` 直接发射 `v.mov`；`:588` 的 MMA lowering 调用 `self.own.hold`，但本文件没有调用 `self.own.check_write` 或 `check_write_range`。`footprints.py` 中已有这些检查方法。DMA 坐标临时寄存器又在发射后立即归还 allocator，注释声称依赖 ownership hold 保护，但 hold 本身不会阻止后续分配或写入。

在默认 `check_protocol=True` 下，[复现脚本](audit/2026-09-19/reproduce_ownership.py) 生成并接受：

```asm
m.mma.async.k16.f16 a0, v56, v48, s1
v.mov              v56, v0
mb.wait            s1
```

这证明编译器未拒绝源 fragment 尚被读取时的覆盖；**本次没有把该故意非法程序跑到 RTL，也不声称已经测得某个错误数值**。ISA 把此类行为交给编译器避免，因此这已经是编译正确性边界的实际缺陷。

另一个已复现的范围计算缺陷：同一 `_op_mma_async` 虽按 `op.attrs['k']` 发射 K32/K64，却用 `target.mma_shape` 的 K16 计算 footprint。K64 FP16 的 A/B 各需 16×64×2 B，即 32 个 64 B 向量寄存器；当前仅分配 8 个寄存器的 window 也被接受。见 [K64 复现脚本](audit/2026-09-19/reproduce_k64_footprint.py)和[输出](audit/2026-09-19/k64_footprint_reproduction.log)。这同样是编译拒绝缺口，本次未执行其非法 RTL 产物。

解决方案：从实际指令的 K/dtype/布局推导 footprint，不能用 target 的默认形状代替；给每个机器操作统一描述显式/隐式读写范围及保持期限；所有普通写、window 写、DMA 写、accumulator 写均经过同一检查；allocator 在 source-reusable/compute-done 前不得释放被占用范围。不要只在 `window_put` 加一个特判。明确区分读租约、写租约、accumulator 顺序和角色间物理别名。最终分配后的检查必须覆盖实际生产入口，不可由 `check_protocol=False` 静默关闭。

验收：加入上述负例、DMA 坐标重用、DMA 与 MMA 同窗、accumulator 提前读写、多消费者最后释放、分配后重叠别名等负例；合法等待后的同类程序仍通过。负例应在编译阶段给出操作位置、物理范围、所有者和缺失的完成事件。

### F02 — P0：端到端失败被 shell pipeline 吞掉【复现】

证据：`Makefile:469`、`:480` 用 `python ... 2>&1 | tail -4`，未设置 pipefail。执行以下命令返回 **0**，虽然 `/usr/bin/false` 必定失败：

```sh
make triton-e2e e2e TRITON_PYTHON=/usr/bin/false
```

缺 Triton 时这些目标也以成功状态 SKIP。`compiler/test/run_tests.py` 输出的 tests 数包含 skipped 项；不能把“PASS 76 tests, 7 skipped”写成执行了 76 个有效测试。`rtl_harness.py` 仅据 stdout 中 PASS/FAIL 判断仿真成功，未检查 `proc.returncode`，也没有 wall-clock timeout。IR runner 使用 shell pipeline，也应统一审查退出码传播。

解决方案：优先让 Python runner 直接管理 subprocess，保留完整 stdout/stderr 与返回码；shell 使用明确的 pipefail。开发模式允许 skip，但 release 模式要求指定套件零跳过、零失败；测试总数异常下降也失败。返回码、结构化结果和期望完成标志共同决定成功。

验收：故意失败的 interpreter、test case、assembler、simulator、缺失依赖、无 PASS、超时分别导致 release gate 非零退出；保存失败日志而非只留末尾四行。

### F03 — P0：系统 target 宣告了系统没有的 SFU【静态确认】

证据：`compiler/targets/tnca-sim-v0.1.json` 描述四 tile 系统，`capabilities.sfu=true`；`tools/isa/gen_target.py:83` 从 core 默认 `HAS_SFU` 取值；但 `rtl/tile/tnca_tile.v:131` 实例化 core 时明确 `.HAS_SFU(0)`。`t15_target` 位于 core 回归，没有证明四 tile 的覆盖参数与 manifest 相同。

影响：编译器依据该 target 放行 SFU 程序，整机实际会拒绝该指令。单核 SFU 测试通过不能解除这个冲突。

解决方案：定义独立的 core-sim、system-2x2 和未来 fpga profile；由同一个配置实例生成 RTL 参数、target JSON、能力读回期望和仿真命令。target 描述 elaborated instance，而不是某个模块的默认值。编译产物携带 profile/hash，loader 启动前核验能力。

验收：每个发布 profile 在其顶层读回所有能力与资源；SFU 开/关两种 profile 都有正负例；系统明确拒绝不兼容 binary。

### F04 — P0：增量构建可能验证旧 RTL【复现构建决策】

证据：`Makefile:164` 的 `tb_core.vvp` prerequisite 没有 `RTL_GEN`，也没有 `.vh` include。以下 dry-run 都只输出建目录，没有重新调用 iverilog：

```sh
make -n -W rtl/gen/tnca_cvt_comb_xls.v sim/build/tb_core.vvp
make -n -W rtl/common/tnca_defs.vh sim/build/tb_core.vvp
```

这证明构建图漏边；本次没有修改这两个源文件。slow/ooo/其他 bench 也需逐个审查。另一个问题是 `compiler` 的 schema/target prerequisite 会先自动生成，再执行 `--check`，与“检查而不修复 stale 文件”的注释不完全一致。

解决方案：统一 RTL source/include manifest 或自动依赖文件；参数和工具版本纳入构建 fingerprint。分离 `generate` 与只读 `check-generated`，CI 对干净 checkout 检查后要求工作区无变化。生成程序/向量依赖也包括其 Python 模块与随机种子。

验收：对每种输入变更做构建失效测试；clean 与 incremental 输出一致；旧 schema 必须令 check 失败而不是被隐式重写。

### F05 — P1：4×4 不是改 NX/NY 即可得到【静态确认】

证据：`rtl/noc/tnca_core_ni.v:84` 使用 2-bit home，x/y 各取一个 bit；`rtl/tnca_top.v:177` 固定 `.N_W(2)`，外部 `mem_req_src`/`mem_rsp_src` 也为 2 bit。tile 的 PID 固定取 tile ID。增大 NX/NY 不会自动提供正确的 16-slice home 和 DRAM 响应路由。

解决方案：统一 node ID、x/y、home、source tag 编码；将宽度由受支持 topology 派生，处理 N=1；明确只支持哪些矩形/幂次配置。不支持的配置在 elaboration 时失败。grid program ID 与物理 tile ID 分离，由 scheduler 逐批派发。

验收：先 1×1、2×2，再 4×4；每个 slice 可访问、跨所有 source 的乱序返回正确、同一物理地址只有一个 home；grid 大小覆盖小于/等于/大于 tile 数。没有这些结果前，不声称支持 4×4。

### F06 — P1：MLIR 检查器不是完整的异步安全证明【静态确认】

`VerifyAsyncOwnership.cpp` 的状态在每个 block 重新建立；`collectTouches` 只对 load/store 的 window 与 MMA 的 A/B 取值，不记录 MMA accumulator；没有利用完整物理区间、别名、CFG fixed point 或跨角色 channel 生命周期。`TNCAAsyncOps.cpp` 虽定义了 footprint 接口，但该 pass 未据此统一检查。现有 IR suite 只有两个文件。

解决方案：短期明确只允许经验证的单 block 子集，其余保守拒绝；中期建立事件/租约的数据流分析，对 CFG join、循环回边、region capture 与取消路径给出明确规则。资源分析在分配前检查逻辑所有权，在分配后再检查物理别名。

验收：合法跨 block 生命周期不误报；分支只在一侧 await、循环重复覆盖、嵌套 region、重复释放、accumulator 冲突等错误均拒绝；Python 路径与 MLIR 路径共享语义样例，不能只各自通过自己的测试。

### F07 — P1：正式后端、参数 ABI 与运行时仍是骨架【静态确认】

`compiler/backend/compiler.py` 的 `load_dialects`、`add_stages` 仍抛 `NotImplementedError`。真实 TTIR 通过 `compiler/test/triton_bridge.py` 调 Triton 内部接口取得，再交给 Python importer。`driver.py` 只构造 vvp 命令。没有可独立安装的编译包入口和生产 loader。

`frontend.py:123` 按 `arg_values` 字典插入顺序绑定函数参数，且把地址/数值变为编译时常量；`:102` 遍历整个 module，而不是仅目标函数。多函数模块、参数顺序改变、不完整参数容易触发错误绑定或非目标函数处理；这些是待增加负例的边界。类型/浮点常量仍部分依赖文本提取。

系统测试 `test_e2e_system.py` 明确只有 BLOCK=16、四 block 的两个 elementwise kernel；`run_system` 通过 testbench 读 dirty cache/DRAM 的一致视图，这证明了仿真结果正确，**不证明 host DMA 在真实设备上能看见输出**。

解决方案：先做明确受限的 AOT CLI；发布 binary + manifest + 编译报告；参数 ABI 包含类型、宽度、偏移和对齐，按 signature 绑定。引入可测试的 runtime/loader 层，处理 admission、grid、错误完成与 drain；随后接正式 Triton backend。host 可见性需要明确定义 flush/一致性/完成通知路径。不能把整个设备集成藏在测试 helper 中。

验收：同一 binary 用不同参数和地址重复执行；非法参数/目标版本拒绝；grid>Ntiles；背靠背 launch、fault 后新 launch 和延迟回包；用户态拿到正确结果且不读 RTL 内部数组。

### F08 — P1：可重复构建与工程发布链不完整【静态确认】

仓库没有 `.github` CI、Python 打包/依赖锁定入口；默认 Triton 路径指向开发者 home 下的虚拟环境。LLVM/Triton 有 pin，这是好的基础，但 `toolchain.json.host_status` 与当前可运行的本机构建状态已不一致。XLS 驱动依赖配置中的个人远程地址、私钥路径和 `cur` 工具目录；生成文件头记录版本不等于固定再生成环境。

解决方案：开发环境与发布环境分离；增加 lock、tool doctor、安装/构建入口及 CI 工作流；XLS 提供可固定版本的 Linux 本地/容器执行方式，远程执行作为可选 transport，个人配置移出版本控制。所有生成物记录源集合、工具 digest、参数和命令。已有 LLVM 下载 checksum 机制保留。

验收：至少一个全新 Linux 环境从源码重建并运行强制回归；macOS 作为已测试开发 profile；不依赖维护者 SSH 账户。每次发布归档工具清单、日志、种子、binary 与 manifest 校验值。

### F09 — P1/P2：内存进展验证还不支持“不会死锁”的广泛结论【待验证风险】

已有 req/rsp 分离、beat reservation、MSHR/target pool、hit-past-block 与随机背压测试，基础正确。`tnca_mem_mux.v` 对 tensor 新请求固定优先，`a_req_ready = !pick_b && m_req_ready`；持续 tensor 请求下没有独立的 LSU 有界等待保证。优先保障完成路径，不等于任意新的 DMA 请求都应永久压过 LSU。

解决方案：把 completion/return、释放资源的控制访问、demand 与 prefetch 分开；保留完成路径资源，普通请求用可证明公平的仲裁。给 ready/valid、请求保留、tag 唯一、资源计数、barrier generation 建断言。活性证明必须列出外部内存最终响应等环境假设。

验收：长期争用、MSHR/target/WB/return 同时满、异常与排空时仍有进展；在声明的公平环境下给出 bounded wait 或 formal liveness 证据。新增策略时 hints 全关、丢失、延迟均不改变结果。

### F10 — P2：RF/存储与时序尚未进入物理实现阶段【静态确认】

`tnca_vector_rf.v` 为多路异步读、双写、全数组 reset 的行为存储；不能直接假定映射成普通 SRAM。`tnca_core.v` 约 1811 行，发射、hazard、写回、异步完成及异常逻辑集中，改变端口/延迟容易牵连协议。当前门级综合指标是 proxy，不是指定 PDK 的 STA；本次未重跑耗时的全芯片综合。

解决方案：先定义 SRAM/BRAM 可实现的 bank、端口、读延迟、写冲突与 bypass 语义，增加 operand collector、backpressure 和 valid-state reset。分别预算 DMA、SIMD、MMA、ARF 端口需求；根据带宽确定计算并行度。建立真实库下的综合、SDC、STA、存储宏与时钟复位流程，再做 floorplan/P&R。

验收：功能与原模型差分；bank 冲突随机注入；目标 profile 完整综合；指定时钟/工艺/角落下 WNS/TNS、面积、功耗证据齐备。ASIC 后续需要 CDC/RDC、DFT/MBIST、形式等价和物理签核，具体门槛随产品/工艺合同确定。

### F11 — P2：文档、性能声明与证据不同步【静态确认】

`Readme.md` 同时包含阶段完成与“Stage 2 next”等旧路线；`compiler/README.md` 开头仍称 C0，又列 C0–C4 done，后文仍列 C1 未完成项；`conformance.md` 的审计范围和偏差计数不能代表整个当前树。resource report 有结构，但 `KernelPlan/format_report/try_plans` 在当前仓库内除定义、导出外只有报告测试调用，尚未接入真实编译入口；也没有完整 benchmark、RF/NoC 带宽实测与预测误差闭环。

解决方案：保留两份原始设计为设计意图，增加一份当前规范和 capability matrix；每行记录 profile、实现路径、正例、负例、最近验证 commit。性能报告绑定固定 shape/dtype/seed、工具版本和参数；旧数字保留其版本，不自动继承为当前性能。

验收：新贡献者能区分“ISA 有此指令”“该 profile 实现”“编译器可生成”“整机验证过”；benchmark 报告含原始计数器与基线对比，不以 hit rate 或门深替代吞吐/频率。

## 4. 工程重构方案：按合同拆分，逐步迁移

先修可信度缺陷，再改变结构。不要把所有 Python 编译代码一次性翻译成 C++，也不要先搬目录再定义接口。现有 assembler、精确浮点 oracle、协议模型、RTL microbench 是迁移时的比较基线，应继续保留。

建议的职责边界如下；目录名为提案，不表示本次已创建：

```text
spec/                      ISA、target/profile、ABI/manifest schema
compiler/
  frontend/                固定版本 Triton adapter，入口与类型检查
  lib/.../Transforms/      layout、partition、channel、protocol、allocation
  python/tnca_cc/           CLI/API、调试与参考路径；逐步减少生产 lowering
  test/                    IR/diagnostic/编译器负例
runtime/
  common/                  manifest 校验、资源准入、launch/fault 生命周期
  sim/                     仿真 transport；不依赖 test 模块
  device/                  后续真实 host/device transport
rtl/
  core/                    fetch、issue/scoreboard、RF、writeback、completion
  mem/ 或现有 nuca/noc/    保留现有模块边界，统一接口和配置来源
verification/
  reference/               FP/ISA/protocol 独立参考模型
  unit/ integration/       从 core 到整机的分层回归
  formal/                  握手、所有权、资源守恒与进展性质
benchmarks/                workload、基线、计数器、消融与报告
flow/                      生成、仿真、FPGA、综合与物理实现
```

不要求立刻采用完整目录结构。建议依次提交以下可独立审查的变更：

| 重构单元 | 明确合同 | 迁移与回归要求 |
|---|---|---|
| 配置与 ISA | profile 同时决定 elaboration、capability、compiler target；ISA 只保留一个 authoring source | 先补一致性测试，再生成更多派生数据；暂不同时改 ISA 编码 |
| 构建/test runner | 每个执行步骤有 returncode、timeout、必跑/可选分类、日志与工具 fingerprint | 先保留 Make 入口，替换其内部执行器；验证失败注入 |
| compiler driver | `source + signature + target + options → binary + manifest + report + dumps` | 把 test bridge/harness 中必要产品逻辑搬出；旧测试调用新 API |
| 异步资源层 | event、lease、channel、physical footprint 为显式分析结果 | 不让每个 lowering 自己手写一套 hold/free；建立共用语义样例 |
| pass pipeline | 明确 input/output IR 与 invariant；lowering 后运行 verifier | 先接通窄 MLIR 路径，再逐算子替代 Python；每步与原路径/RTL 比较 |
| codegen/allocator | allocation map 与 IR 分离，避免 `Value.reg` 的共享可变状态扩散 | 重复编译、不同 role 编译顺序、不同 target 的确定性测试 |
| core RTL | issue/hazard、写回仲裁、async completion/fault、RF 接口可单独验证 | 每次只拆一个状态所有者；确保同周期多个 completion 不丢失 |
| runtime/bench | 产品完成语义与 testbench 观察接口分离 | 保留内部 dump 用于诊断，同时用真实可实现的 host 输出路径验收 |

`partition.py`（约 946 行）应按 dependency graph、代价/映射、channel 插入、loop cloning 四个职责拆；`codegen.py`（约 630 行）应按资源校验、分配、指令选择和发射拆。拆分目的在于让不变量只有一个实现位置，不以缩短文件为验收标准。

生产编译路径继续遵循原设计：上游 `tt/arith/math/scf` 保留通用语义，TNCA 只承担资源、布局与异步生命周期；不要把当前 bring-up 的自定义算术 IR 扩成另一套通用优化器。LLVM target 在稳定机器 IR、ABI 与调度需求出现后再建设，不作为 R1 的前置条件。

## 5. 分阶段执行计划与验收门槛

各阶段由退出条件推动，不按“代码已合并”判定完成。角色是建议责任归属，不假定当前已有相应团队人数。

### R1 — 可信且可重复的 2×2 基线（立即开始）

责任：编译器维护者与验证/构建维护者共同负责。

交付：修 F01–F04；落实 F06 的受支持子集；增加只读生成检查和 strict regression；整理 core/system profile 与当前支持矩阵；把本审计负例变为 CI 正式测试。

退出条件：

- 干净环境可重建，必跑套件零 skip；故意失败能够稳定使 CI 失败。
- 异步覆盖、遗漏 await、target capability 不符均在正确边界被拒绝。
- 每个对外声明的 profile 都从自己的顶层读回 target 合同。
- clean/incremental 构建一致；所有结果绑定 commit、dirty 状态、工具和参数。
- 当前正例回归不退步；已知失败均有 issue、owner 与是否阻断发布的结论。

R1 的成果应命名为“验证过的实验性 AOT 基线”，不能命名为量产版本。

### R2 — 统一 AOT 编译器与仿真 runtime

依赖 R1。责任：编译器/runtime。

交付：安装包与 CLI，固定 Triton adapter，signature/参数 ABI、binary manifest、资源 admission、日志/IR dump；正常 grid 调度与重复 launch；逐步接通 MLIR pipeline，Python 保留参考用途。

退出条件：同一 binary 改地址/形状参数运行；grid=1、tile 数、tile 数+1、多个批次；覆盖尾块、stride、分支与明确支持的循环；fault/timeout/drain 可观测；不支持的输入有稳定诊断。端到端程序不再靠 test helper 拼装产品入口。

### R3 — 有性能合同的计算与存储实现

依赖 R1 的配置和验证基础；可以与 R2 的部分工作交错推进，但新 RTL 必须走相同 gate。责任：微架构/验证/性能。

交付：真实 banked RF/ARF 接口、读延迟与仲裁模型、可测的 DMA/MMA/SIMD 吞吐；GEMM、reduction、softmax、LayerNorm/RMSNorm workload；分层计数器与资源报告相互校验。

退出条件：先冻结每个目标 workload 的最低吞吐/最大延迟/资源预算，再提交可复现测量；报告 RF bank stall、MMA starvation、barrier wait、MSHR entry/target 满、NoC link 流量和 DRAM bytes。给出相对于 R1 的正确性与性能变化；不能只报告最快样本。

Attention 在 GEMM+reduction+SFU/软件 fallback、accumulator↔vector、两类 channel 和资源预算全部闭合后接入；先一个可验证的 online-softmax 变体，再扩 causal/tail/不同 shape 与 QK/PV 重叠。

### R4 — 4×4 与 cache/flow 机制的受控扩展

依赖 R2/R3 的配置、网格和测量能力。责任：NoC/NUCA/编译器调度。

交付：修复全部 topology 编码，先扩规模，再依次加入 use-complete、replacement/bypass、MSHR-aware admission、sharing-group control；元数据从最终 schedule 生成。

退出条件：4×4 地址路由/返回/原子性/资源进展验证；每个新机制可以独立开关，分别做 baseline、单项、组合消融。关闭或扰动全部 hint 后数值仍正确。每链路记账包含实际数据/控制开销，外存路径纳入瓶颈模型。

### R5 — 真实板级/设备产品

前置产品决策：FPGA 型号或 SoC/PCIe 集成方式、内存接口、单任务还是多租户、允许的 reset 粒度和 host 一致性合同。责任：系统/驱动/硬件。

交付：真实内存控制与 host transport，驱动/loader、buffer 地址验证、完成队列、错误回收；必要的事务 generation 或等效的可靠 drain 隔离；性能与故障遥测。

退出条件：host 直接验证输出；连续 launch、错误恢复、长时间压力、背压、外存错误、复位边界通过；无旧任务响应写入新任务资源。根据实际隔离目标验证越界 DMA/不合法 binary；第一版可按原设计不支持 demand paging/抢占，但必须明确拒绝而非部分执行。

### R6 — ASIC/量产签核（若产品目标包含自研芯片）

依赖明确 PDK、SRAM/IO/内存 IP 和系统预算，不以 R5 软件 gate 代替。责任：RTL/DV、物理设计、DFT、系统与制造测试。

交付与退出条件包括：约束完整的综合/STA、多角落物理实现、CDC/RDC、复位与电源域、形式等价、存储测试/DFT、可靠性与错误上报、功耗/IR drop/EM、物理检查、封装/板级接口及量产测试计划。具体覆盖率、余量、测试时长由产品合同确定，并有 waiver 审核记录。

所有阶段应保留回滚点：配置化实验功能默认关闭；旧 benchmark 与 binary/profile 兼容性保留；资源语义或 ISA 变化必须版本化并提供明确拒绝行为。

## 6. 验证体系需要增加什么

| 层级 | 重点内容 | 何时执行 |
|---|---|---|
| 每次提交 | schema/profile 一致性、编译器负例、IR verifier、快速 RTL 与 lint | 必跑，无环境性 skip |
| 集成 | Triton→AOT→单核/整机，fast/slow/ooo、2/4 RCTX，能力开关 | 合并门禁，保留完整日志 |
| 长回归 | 多 seed、随机内存/背压、FP corner、fault/reset/退出排空、共享地址原子性 | nightly，失败 seed 可单独重放 |
| 性质验证 | 请求响应守恒、credit/target 不泄漏、tag 唯一、completion 不丢失、generation 不混淆 | 协议模块变更及周期性检查 |
| 性能回归 | shape/dtype/stride/尾块矩阵、计数器、相对基线差异 | 性能相关变更及发布 |
| 发布 | 全新环境重建、强制套件零 skip、安装与 ABI 兼容、生成物一致性 | 每个 release candidate |

优先增加能够反驳“安全”的测试，不只增加快乐路径。RTL 随机测试应有 coverage 目标和失败种子；形式验证应列出环境假设与覆盖边界；独立数值 oracle 与软件功能模型避免与被测代码共享同一错误。浮点分别报告精确 ISA 契约与近似算子的误差合同，不能混用逐位相等和容差判断。

现有 `tools/run_test.sh` 会拒绝无 PASS 或退出非零，这是可复用的做法；但它不能补救 Make 上层吞掉返回码，也不能替代 wall-clock timeout。随机背压已存在，也不能单独证明所有未来参数与策略下永不死锁。

## 7. 本次验证记录与边界

具体运行结果见下表与随附日志。测试在已有开发环境/构建目录执行，**不是干净环境可复现性验收**。本次未安装依赖、重建固定版本 Triton/LLVM、运行远程 XLS、进行物理签核或穷尽硬件状态空间。

| 命令/检查 | 本次结果 | 能说明什么 |
|---|---|---|
| `make test` | 返回 0；38 个 RTL 文件 elaboration 通过；compiler RTL 67 项通过；core fast/slow/ooo、三种 NUCA、20,000 包 mesh、四个 top 运行及 E2E 完成 | 现有回归通过；有 7 个 compiler-unit skip，且不消除 F01–F04 的验证盲区 |
| `make compiler` | 返回 0；unit 76 项，其中 7 skipped；2 个 IR 文件通过；169 条 schema 指令、218 个 shared constants 一致 | 当前环境中的限定检查通过，不证明系统实例能力一致 |
| `make triton-e2e e2e` | 日志分别为 5 tests OK、2 tests OK | 已有窄 Triton 闭环通过；Make 返回码本身有 F02 缺陷 |
| 使用 Triton 虚拟环境直接运行两个 E2E Python 文件 | 两个进程均返回 0，分别 5/2 项通过 | 绕过 Make 的 `tail`，确认本次 E2E 没有被其隐藏失败 |
| `TRITON_PYTHON=/usr/bin/false` 失败注入 | 两个 Make 目标仍返回 0 | 复现 F02 |
| `reproduce_ownership.py` | 非法覆盖序列被接受并输出 | 复现 F01 的编译拒绝缺口 |
| 两条 `make -n -W ... tb_core.vvp` | 未重新调用 iverilog | 复现 F04 的依赖缺口 |

附属材料：

- [本次验证结果汇总](audit/2026-09-19/verification_summary.json)。
- [环境、commit 与失败注入/dry-run 的返回码](audit/2026-09-19/environment_and_reproductions.json)。
- [所有权复现代码](audit/2026-09-19/reproduce_ownership.py)与[实际输出](audit/2026-09-19/ownership_reproduction.log)。
- [完整 make test 回归日志](audit/2026-09-19/full_regression.log)（进程退出码 0）。
- [compiler 回归](audit/2026-09-19/compiler.log)、[Make E2E 日志](audit/2026-09-19/e2e.log)。
- [直接运行单核 E2E](audit/2026-09-19/triton_e2e_direct.log)、[直接运行整机 E2E](audit/2026-09-19/system_e2e_direct.log)。

## 8. 第一批可直接立项的变更

| 顺序 | 变更 | 完成判据 |
|---|---|---|
| 1 | 测试返回码与 strict mode | `/usr/bin/false`、缺依赖、测试异常均使 release 命令失败；输出完整日志 |
| 2 | 修构建依赖与只读生成检查 | `.vh`/生成 RTL/参数变更正确重建；stale schema 在检查中失败 |
| 3 | 统一 core/system profile | 修 SFU 能力冲突；各顶层能力读回与 manifest 一致 |
| 4 | 接通实际 codegen 所有权验证 | 本报告负例拒绝；合法异步正例通过；无关闭检查的生产旁路 |
| 5 | 明确 MLIR/TTIR 支持边界 | CFG/别名/参数边界的负例完整；暂不支持的行为明确拒绝 |
| 6 | AOT CLI 与 manifest/runtime 最小闭环 | 脱离 test 模块，安装后可编译、装载、重复执行并获取结果 |

前五项构成 R1 的关键路径，第六项启动 R2。先获得这条可信交付链，再以 benchmark 决定 banked RF、Attention、4×4 和 cache/flow 机制的具体优先级。
