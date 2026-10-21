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

谓词压力修复使用现有 `p.ballot`/`p.frombits` 保存精确 lane mask，不能增加架构容量。
pointwise 超出资源时依次尝试较小 block，保留共同 TNCA 执行路径；已有合法配置仍优先使用。
column broadcast 修复 `[R,1]` 加载后的每行 lane 0 复制语义，不按模型或算子名称特判。

原始失败保留：`make compiler` 首次发现生成的 support matrix 落后于新增的两条可生成指令；
由远端原生成器更新矩阵后复跑通过。depthwise 的初始 predicate/scalar 资源失败和 bias 仅首 lane 生效的诊断均未计作 PASS。

## 尚未满足的签收条件

公共 `TNCA_TUNE` 的物理验证、内存映射及计时边界资格、三个闭环案例的板上闭合、
完整模型运行时峰值与 code working-set 估算仍须完成。当前模型算子/状态/cache 接口分析见 `model_feasibility.md`。
最终 90 MHz 全套数值回归、正式非退化结果、prospective 板测残差、干净 git archive/fresh cache 重建及冻结板上锚点亦未完成。

卡上观测镜像曾为 UUID `428c00e5-6996-92c7-1f8f-ec41f2d037b3`、245 MHz，且有使用者；
与 B40-OBS 的 90 MHz 镜像不同。`kickoff/restore90-check.log` 为只读检查，usage gate 未通过，未执行加载。
恢复 R40 镜像须有明确授权、卡空闲且全部加载检查通过。当前不能把原 90 MHz 镜像当作仍在运行。

模型保持 `uncalibrated`。sealed holdout 未用于开发或拟合；已有失配与负结果不删除。
