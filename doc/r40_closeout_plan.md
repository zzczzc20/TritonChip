# R40 交付关键路径与集中验证计划

更新：2026-10-14；盘点源码：`25c2049`。验收合同仍是
`doc/r40_tasks_and_acceptance.md` 和 `spec/r40/plan.json`，本计划不修改门限。

## 当前阻塞

只读运行 `tools/r40/acceptance.py`：36 个必需 gate，5 个通过。
R40-00 已关闭；R40-01 device/mixed-device 通过，device-time 失败。
其余工作包均未关闭。已有 RTL/板上 checkpoint 不自动等于完整 gate。
R40-05～09 的正式 Make target 缺失，必须补齐实际执行与证据汇总入口。

R40-01 原生全量 77/77 正确，但相对 B39 的 S_geo=0.5872。
同一 HBM 镜像的 CPU/native 对照在 75 个成功共同点上约为 0.999，
优先定位存储路径代价；两点 code-cache epoch 修复后的复测另存证据，
不能拼接成同源码的正式 77 点实验。512 KiB local/HBM hybrid 镜像正在
构建，仅为候选，尚未证明能消除回退。

## 四批实施及退出条件

### 第一批：验收入口、计时与硬件可行性

1. 接齐 R40-05～09 的真实 gate runner、证据验证与失败传播；复用已有
   功能测试，不建立仅打印 PASS 的目标。板测缺失应明确 not-measured。
   同一工作包命令在一次执行中只跑一次，再判断它提供的各 gate。
2. 统一实验 manifest：源码、制品、镜像 SHA、时钟、资源、原始样本与
   workload 定义关联；修正证据描述中把 HBM 驻留误写成片上驻留的问题。
3. 保持已启动的 hybrid 单核与三核 link；code-upload 两级流水 XO 完成
   后先核验源文件与新增 command/reply placement。旧 code0 三核镜像
   用于布线可行性与诊断，最终 code2 配置另需完整 route/板上验证。
4. 修复 HBM capacity 原始 host SIGSEGV 的根因并保留失败记录；长等待
   子点成功不能代替原全量用例。收齐真实 placement、timing、CDC 与
   connectivity，确认三个出口共享 BO 的地址合同成立。

退出条件：所有 gate 有可执行入口；镜像/实验身份可追溯；三 SLR 实现
可行性明确；回退的主要成本有实际计数与最差点诊断支持。此批不签收
尚未测量的性能。

### 第二批：一次集中补齐存储、编译与执行语义

按依赖合并实施 R40-02～07 的缺口：DMA/cache ownership、atomic/fence、
alias/lifetime/dead-hint、locality/DCO；descriptor/barrier 资源生命周期、
retile/spill；通用异步 pipeline、layout/pack 与矩阵策略。先从标准 Torch
制品及设备计数判定瓶颈，再实施矩阵吞吐改造，避免只优化微基准。

集中验证用同一源码的：资源/合同、ownership 故障矩阵、1/2/3/4/8 核、
不同端口/延迟/背压；检查 DMA/MMA 实际重叠。已通过且依赖未改变的
八核、>4096 grid、code-link 等证据保留，不每次重复完整长仿真。
修改对应边界时再跑关联回归。每个完成的工作包单独 commit。

退出条件：功能缺口关闭、统一能力/成本模型及开关完备，候选 RTL
通过集中回归；冻结最终 RTL 配置后启动完整 route。后续软件与实验
工作可在 route 期间进行；硬件缺陷需要修改时明确废弃旧候选证据。

### 第三批：标准路径性能、attention 与应用

1. 板上先跑最差 transpose、streaming、compute scaling、长 K 和长序列
   attention 诊断点，确认优化方向。诊断不代替冻结全子集。
2. 同一最终镜像先运行 active_cores=1/2/3、单/多出口的冻结对照；
   保持 PC/cache 总预算一致，再测扩容对照。记录每核负载、每出口
   流量、频率与完整 graph makespan。
3. 分开测 A（冻结 CPU 迁移）、B（native per-kernel staging）、C（native
   resident），按各机制要求控制其余变量；补齐 replay、物理 copy
   overlap 和 1000-call stress。CPU/native 身份与驻留收益分开计账。
4. 完成在线 attention、GQA/MQA/mask/长序列及 64-step KV；证明没有
   L×L 中间张量并测实际中间 HBM bytes。随后实现预算内 autotune，
   冻结 holdout 与 ≥4 层、≥64 MiB 权重、prefill≥512、decode≥64 模型。
5. 正式 A/B 按冻结采样执行：warmup 5，timed 30，3 blocks，随机交错；
   长任务按冻结例外 timed 10。报告 median/p95、配对 bootstrap 95% CI。
   全量样本不提前缩减；计时与性能通过由现有协议判定。

退出条件：下表各必需性能门限、功能及文档子要求全部通过；无收益
机制关闭或移除，并保留负结果。不能用优化一项抵偿另一项失败。

### 第四批：一次最终全量回归与独立复现

冻结最终软件/镜像后，集中跑 conformance、tiny/small/medium、原 large
全量、冻结 HBM 核心套件、模型、全部独立 A/B、LOO≥3% 与规定交互。
GPU 对照保持同 Torch point/精度/计时边界。准备报告、原始样本索引、
资源/时序、HBM 模型校准、8 核实证和 16/32 核扩展预算；独立干净
session 重建及上板复现。最后由 `make r40-acceptance` 判定全部 36 gate。

## 工作包与必须关闭的门限

| 包 / gate 后缀 | 关闭所需证据与性能门限 |
|---|---|
| 00 protocol / b39-reproducible / frozen | 已通过；协议、77/77 重现和冻结检查持续有效 |
| 01 device / mixed-device / device-time | 原生 77/77、混合设备/worker、S_geo≥1.0，未解决 >5% 回退为零 |
| 02 zero-transfer / resident / batched / stress | 冻结驻留正确、S_geo≥1.5；batch/replay≥1.2；1000 次与异步生命期/恢复；物理 DMA overlap |
| 03 tiles / wide-blocks / gemm-tile / contract | 16/32/64/128、tail/layout/alias、retile/spill/生命周期；wide≥1.3、GEMM tile≥1.15；旧合同 |
| 04 hbm / bandwidth / robustness | ≥64 MiB 标准图、全 32 PC 地址/容量；read/write/copy 各≥冻结瓶颈峰值50%；宽入口/并发≥1.5；错误/复位、模型校准、板上 timing/CDC |
| 05 multicore / scaling / dispatch | 真实 SLR0/1/2 同时工作；三核≥2.1、两核≥1.5；完整 grid/fault、1/2/3/4/8 核语义 |
| 06 nuca / streaming-scaling / locality / semantics | ownership/atomic/fence/DMA/alias/dead-hint；三核 streaming≥1.8、多出口≥1.3；locality/DCO≥1.1且实际 bytes 至少一项降20%；独立消融/公平性 |
| 07 pipeline / long-k / matrix / policy | 实际 overlap、stage/epoch/fault；long-K≥1.2、matrix≥1.2；自动选择与全量无未解决 >5% 回退；资源、秒数、LOO |
| 08 attention / prefill / decode | 全 attention/模型与长序列/mask/GQA/MQA/64-step；prefill≥1.5、中间 HBM bytes 降≥50%；零权重/KV 重传、bucket 内一次编译 |
| 09 autotune / holdout / suites / model | ≤16 候选/kernel、≤10分钟/graph；holdout≥1.1、regret p90≤15%；完整套件；规定大模型三核≥1.5且decode退化≤5%；8核实证/扩展预算 |
| 10 acceptance | 最终同源码/镜像、完整 A/B/LOO/交互、GPU对照、复现包与独立session；全部必需 gate PASS |

## 资源、测试与提交纪律

- 跳板机跑重型 RTL/编译矩阵；FPGA server 跑 Vivado/Vitis、板上功能与
  性能。板卡实验串行，独立编译/软件准备可并行。启动前按实际内存
  和现存作业定并发，不同时堆积多个完整 route。
- 已运行的长仿真保留 PID、源码与输出目录，读取增量状态；活着且
  没有失败依据时不重启。新增相同目的的板测可提前给诊断证据，
  不能冒充旧仿真的完成状态。
- 一批修改后跑相关集中验证；一次最终冻结全量回归。前一轮已有
  完整证据且相关依赖未变时不为形式重复测试。
- 工作包或可审查里程碑 commit；push 合并成批，让 CI 完成，避免
  连续 push 触发 cancel-in-progress。修复 CI 后重跑受影响测试；最终
  HEAD 必须 CI 通过。文档提交也不得把未关闭的包改成 accepted。
- 执行检查点保存各作业状态、下一动作及 gate 缺口；每批更新本表与
  evidence registry。性能结果失败时返回对应实现批，不修改冻结标准。

最短路径是先解决三 SLR 资源/时序与单核存储回退，同时补齐验收入口；
route 等待期间推进编译和模型；最后在同一个冻结候选上集中取得
全部性能及消融证据。当前性能和 route 尚未收敛，不能承诺固定交付时长。
