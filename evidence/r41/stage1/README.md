# R41 Stage 1：在途证据

状态：**未签收**。合同为 `doc/r41_tasks_and_acceptance.md`；本文件只索引实际证据，不替代或缩减合同。
所有本轮编译、RTL、SystemC 和 Torch 测试均在 GB10 `gx10-6646` 执行。

## 已有证据及其边界

| 证据 | 观察 | 不能据此声称 |
|---|---|---|
| `kickoff/merged-boundaries-summary.json`、同名 tgz | 合并 M1–M4 后 lint、核心、coherent host、memory controls、f2i 回归完成 | 新镜像实现、100 MHz 或上板通过 |
| `kickoff/harness-smoke-v3-summary.json`、同名 tgz | 干净构建 SystemC/Ramulator2；共同 Session 真产物；快慢路径；监控模式数值/周期/计数器不变；有界 trace 报告丢弃；worker 生命周期检查完成 | 全芯片周期模型已校准、三个闭环案例均已完成 |
| `fragments/` | 标准 Torch mm/addmm、tail、FP16/BF16、非连续输入；宽 tile 的独立 ACC 跨 K 保留；有旧映射及保守 16x16 对照 | 宽 tile 一定更快；这些小 tail 点中保守 tile 的模型周期更少 |
| `kickoff/depthwise-regression-v2-evidence.tgz` | 3 项谓词检查含实际 RTL 逐 lane 校验；3 项 fragment 检查；4 项 depthwise/broadcast 检查；D256 attention、depthwise、FP32 recurrence 代表图全部功能通过 | 完整 Qwen 文本模型或长期 cache 生命周期已经通过 |
| `kickoff/compiler-regression-after-depthwise-v2.log` | 生成文件一致性检查与 102 项 compiler unit 检查通过；MLIR `tnca-opt` 未安装，IR suite 明确 SKIP | 未执行的 MLIR suite 通过 |
| `kickoff/qwen-storage/` | 固定官方 revision，632 个 tensor 的 header/index 覆盖；语言分支 320 个 storage、3,763,655,360 字节；独立 dtype/shape/区间/文件大小审计通过 | 4 GiB 内完整运行时容量足够；权重未下载，执行未测量 |
| `kickoff/public-tuning-v2-evidence.tgz` | 公共 Inductor runtime hook；隔离候选编译及真实子进程超时终止；Session drain/poison；SystemC 保守拒绝；两个配置的 MMIO RTL 端点选择与样本隔离 | 物理镜像资格或板上收益；RTL 测试显式替换了物理资格检查 |
| `kickoff/archive-d1feb89-harness-evidence.tgz` | 从 git archive 导出检查点，远端 fresh caches 重建与 harness smoke 完成 | 后续 tuning 修改的最终 archive 或冻结板上锚点已经完成 |
| `kickoff/historical-ooc-audit.json` | 77 个 RTL 文件与历史 OOC 源码全部一致；changed/missing/extra mutation probes 均被拒绝 | 新综合、区域预检或新 route timing 通过 |
| `kickoff/matched-native-speed-v1-evidence.tgz` | 相同 AOT artifact、输入、地址、单 program；RTL/SystemC 各 1 次 warmup、3 次完成执行；中位 host 执行 3.357 s / 0.001852 s | 全工作负载的统一加速倍数；单点周期预测 1397 对 RTL 1281 仍保留为误差证据 |
| `kickoff/prospective-prediction.json`、`prospective-wide/` | 板测前保存候选预测及不可变源码身份 | 对应板测、残差或最终源码回归已经完成 |
| `kickoff/public-tuning-v4-evidence.tgz`、`tuning-policy-v4.log`、`tuning-boundary-v1.log` | 最大块 32 仍以合法 16 块起步；三次独立设备样本与保守不确定性策略；真实公共路径 cache 命中且不重测；17 项策略、6 项 worker、CUDA/TNCA/CPU 边界回归通过 | 物理镜像上的 tuning/drain/error 资格已经完成 |
| `kickoff/harness-smoke-v4-evidence.tgz`、`memory_paths.md` | 新慢路径域拒绝守卫后的干净构建、完整 smoke、快慢路径与监控一致性通过 | 实际 bank/row map、请求/延迟定点、控制器背压或板上模型资格已经完成 |
| `kickoff/barrier-ordering-v2-evidence.tgz`、`synchronization.md` | store/reload 与 alias 的三类 RTL 响应模式；跨 role fence 替代 rendezvous 的负例；异步 lease 负例 | 可以删除其他图的同步或替代最终板测 |
| `kickoff/frequency-preflight-v2-evidence.tgz`、`frequency_preflight.md` | 全 census 重建、93 MHz 历史差异保留、F01-F10 责任源码、原始 SLR packing 风险 | 新频率闭合或候选实现已经完成 |
| `kickoff/primitive-ir-v2-evidence.tgz` | 三个独立进程的代表图保留真实 IR、artifact 资源、TNCA lowering 时间与 Session heap/code 峰值 | 完整模型运行时峰值或完整模型编译已经通过 |
| `locality_design.md` | 通用消费者、storage/alias/lifetime、已有 TMU 控制、拒绝规则与 Stage 2 板测设计 | Stage 2 locality runtime 功能已经接入或已有收益 |
| `kickoff/slow-process-lifetime-v1-evidence.tgz`、旧二进制负例 tgz | 三种真实 owner 单独 SIGKILL 后活动模型/slow pass 均退出；旧版本遗留子进程的缺口复现并清理；新模型普通 artifact 与快慢路径回归通过 | 慢路径请求/延迟定点已经收敛或模型已经校准 |
| `kickoff/harness-smoke-v5-evidence.tgz` | 18 步完整 smoke；请求集合、逐请求延迟、makespan 与最终执行重现的定点判定；实际不收敛及子进程失败负例；残差工具明确拒绝不合格慢路径 | 所有图均收敛；较大 GEMM 的请求集合变化仍保留为不收敛，数值通过不等于周期预测通过 |
| `kickoff/physical-dram-map-v1-evidence.tgz`、`address_map.md` | B40 实际 BO/local limit/PC0-PC15 范围导出与原生模型接入；快慢路径同 artifact 的 PC 边界、数值和字节计数一致；映射后的标准 Torch 图通过 | bank/row 定制、控制器等价或新 Session 的 BO 映射已经验证 |
| `kickoff/physical-dram-map-v2-evidence.tgz`、队列夹具首轮负例 tgz | 11 步映射/原生图实验；9 项 Ramulator 机制；192 个半行完整接受/完成、拒收重试、写完成与双向切换约束；实际模型的 burst/line 计数一致 | 通用控制器队列与物理 FPGA 控制器等价或模型已校准 |
| `kickoff/harness-smoke-v6-evidence.tgz`、maintenance 原缺陷 tgz | 维护等待已接受的出口读写；原“写第 26 周期完成、维护第 0 周期完成”已复现并修复；19 步完整模型 smoke 通过 | 所有物理 host-handoff/控制器计时边界已经合格 |
| `kickoff/pipeline-admission-v2-evidence.tgz`、strict 比较补充 tgz、`pipeline_admission.md` | 4 组模型、3 组 RTL 与比较全部完成；实际 ring/transfer/MMA 轨迹；第三层增加 overlap 但尾块更慢；缺项/重复项比较拒绝 | artifact/内存域不同，不能宣称匹配的跨域预测准确率；板上闭合未完成 |
| `kickoff/host-handoff-v1-evidence.tgz` | 原生快慢路径均通过显式维护/local split/跨边界传输与三项拒绝；ABI 维护使模型流量更接近 RTL，残差仍保留 | 物理控制器等价、完整周期校准或 host latency 已预测准确 |
| `kickoff/language-meta-v1-evidence.tgz`、`model_feasibility.md` | 官方 24 层、320 个参数 storage 的精确形状/dtype/tied alias 核对；128/129 prefill 全分支 ATen 图和别名存储分析 | TNCA 完整模型编译、机器码工作集、设备 allocator 峰值或数值执行通过 |
| `kickoff/memory-service-v1-evidence.tgz`、`memory_service.md` | 三组 outlet/depth 的实际三核 MMIO RTL 全部数值通过；三套模型域与原始残差保留；维护/缓存几何解释、严格比较与拒绝检查完成 | 合成 AXI slave 等价于物理 HBM；更多出口必然获益；板上闭合或全局周期资格已经完成 |
| `fragments/bitwise-k-order/`（`systemc-bitwise.json`、`rtl-bitwise.json` 与原始 tgz） | 标准 Torch mm/addmm，FP16/BF16、连续与非连续、M/N tail、K=7/33/48/65/113：16x32 双 fragment 输出与 16x16 输出逐位相同，且都与独立逐 K FP32 FMA 参考逐位相同；SystemC 80 行、单核 iverilog RTL 16 行，无容差。含精确抵消的输入下，16 块归约或逆序会改变 SystemC 37/47 行、RTL 2/5 行的输出（每图最多 1452 个元素），实际输出仍逐位等于逐 K 参考 | 宽 tile 更快；未测 shape/dtype；FP32 输入路径 |
| `fragments/execution-plan-tests.log`、`kickoff/tuning-plan-v1-evidence.tgz` | execution resource plan v2：物理 fragment、有效/padding MAC（常量 TTIR 形状，按 launch）、K 循环 ACC 往返、VRF/window/ACC/barrier/code、descriptor/pointer 传输、transfer/MMA 命令与依赖、选择/拒绝原因；16x16、16x32、tail、mask 形状与手算值一致；Inductor retile/拒绝、code store 拒绝与 TNCA_TUNE 候选资源读取计划；公共 tuning 7 步（含两项 MMIO RTL）通过 | 计划是数值 oracle 或周期预测；运行时 mask 下的 MAC |
| `default_change_audit_narrowed.json`、`default_change_audit.md`、`kickoff/narrowed-broadcast-v1-evidence.tgz` | `[R,1]` 列广播修复收窄为仅在未证明行复制时复制 lane 0（`Tile.replicated`）：113 个原始点代码与启动结构全部与 B40-OBS 相同，原 56 个变化点全部恢复；depthwise 图仍有 1-3 处 lane-0 复制，强制去掉后 SystemC 数值失败，收窄版 SystemC/RTL 通过；primitive 三图、80 行 K 顺序逐位检查、`compiler/test/test_column_broadcast.py`（含 RTL 对照）通过 | 收窄前的 56 点代价已上板测量 |
| `default_change_audit.json` | 收窄前（bf6fd2b）记录：113 点中 57 点相同、56 点变化，全部由 `[R,1]` 列广播修复引起（消融还原）；predicate spill 与 pointwise retile 默认零影响；资源计划改动不改变代码 | 当前树状态（见上一行） |
| `kickoff/compiler-regression-plan-v1.log`、`kickoff/compiler-regression-narrowed-broadcast.log` | 生成文件检查与 110 项（收窄后 113 项）compiler unit（Triton 环境无 skip）通过；`tnca-opt` 以 Ubuntu LLVM/MLIR 20.1.2（CI 所用发行版本）构建，IR suite 6/6 通过 | RTL compiler suite 或 e2e 通过 |

谓词压力修复使用现有 `p.ballot`/`p.frombits` 保存精确 lane mask，不能增加架构容量。
pointwise 超出资源时依次尝试较小 block，保留共同 TNCA 执行路径；已有合法配置仍优先使用。
column broadcast 修复 `[R,1]` 加载后的每行 lane 0 复制语义，不按模型或算子名称特判；只对未被证明行复制（`Tile.replicated`）的来源执行复制。

原始失败保留：`make compiler` 首次发现生成的 support matrix 落后于新增的两条可生成指令；
由远端原生成器更新矩阵后复跑通过。depthwise 的初始 predicate/scalar 资源失败和 bias 仅首 lane 生效的诊断均未计作 PASS。

## 尚未满足的签收条件

公共 `TNCA_TUNE` 的物理验证、内存映射及计时边界资格、三个闭环案例的板上闭合、
完整模型运行时峰值与 code working-set 估算仍须完成。当前模型算子/状态/cache 接口分析见 `model_feasibility.md`。
最终 90 MHz 全套数值回归、正式非退化结果、prospective 板测残差、干净 git archive/fresh cache 重建及冻结板上锚点亦未完成。

卡上观测镜像曾为 UUID `428c00e5-6996-92c7-1f8f-ec41f2d037b3`、245 MHz，且有使用者；
与 B40-OBS 的 90 MHz 镜像不同。`kickoff/restore90-check.log` 为只读检查，usage gate 未通过，未执行加载。
恢复 R40 镜像须有明确授权、卡空闲且全部加载检查通过。当前不能把原 90 MHz 镜像当作仍在运行。
`kickoff/restore90-check-20261021T0517Z.log` 再次确认同一外部 UUID、一个活动 context，usage gate 仍失败，未执行加载。

模型保持 `uncalibrated`。sealed holdout 未用于开发或拟合；已有失配与负结果不删除。
