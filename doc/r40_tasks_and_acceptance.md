# R40：原生 TNCA 设备、可扩展多核 NUCA 与 HBM

版本：0.1，2026-10-10。状态：**PROPOSED / NOT ACCEPTED**。
执行状态（2026-10-11）：R40-00 已关闭（`evidence/r40/R40-00/`）；R40-01 功能完成，全量 gate 待 HBM；R40-04 进行中（首个 HBM link 时序失败，正在修复约束并重建；仿真复验进行中）。
起点：R39 尾声；本次审查为 `r39` 分支、HEAD `3a546702db3fda63547b4c1ff0a9d779a6c5cfbe`，工作区另有未提交的 R39 改动。正式起点由 R40-00 冻结，不能把当前工作区或尚未签收的镜像当作基线。
签收名：`R40_NATIVE_DEVICE_SCALABLE_NUCA_HBM`。
上位与前序：[R39 tasks & acceptance](r39_tasks_and_acceptance.md)、[R39 报告](r39_report.md)、[ArchBench 协议](../archbench/PROTOCOL.md)。仓库阶段文档目录为 `doc/`，本文沿用此惯例。

## 0. 阶段目标与标准路径

R40 把已经能跑的小规模单核系统推进到**原生设备张量驻留、通用 Triton 编译、多个物理核、分布式 NUCA、真实外存**的完整系统。U280 是首个实现载体；逻辑架构与软件接口必须能扩展到更多核、不同外存控制器和后续 ASIC 实现。

### 0.1 唯一可以签收性能的路径

```text
PyTorch model / op，输入与权重为真实 torch.Tensor(device="tnca")
  → torch.compile(..., backend="inductor", fullgraph=True)
  → Inductor 的 TNCA 设备调度、融合、Triton 模板（tensor descriptors 开）
  → Triton TNCA backend（上游 frontend → 经验证的 TNCA lowering）
  → .tnca → tnca.runtime.Session
  → grid dispatcher → 物理 core / role context → NUCA / NoC → 外存控制器
  → U280 HBM；确定性 RTL 用于功能与因果验证
```

- **R40 的性能收益必须从上述 PyTorch 入口产生。** 手写 Triton、`compile_aot`、IR builder、planner、汇编和 AXI traffic generator 可以定位问题、验证硬件或做诊断，数字不能替代任何工作包的性能验收。共享 `_lower_ttir` 内部生成 TNCA IR 是正常编译流程；禁止替换标准路径的 TTIR 或在测量时注入特调 IR。
- R39 的 CPU 张量 → Inductor `cpu_backend="triton"` → TNCA 路径保留为迁移基线。它已经生成 Triton；“CPU lowering 不经济”应落实为 CPU 设备语义导致的传输、启发式、wrapper 和缓存限制，不能把它描述成计算始终在 CPU 上执行。
- descriptor 配置必须开启。适合 descriptor 的 GEMM、连续读写等用例必须留下实际生成和执行 descriptor 传输的证据。间接索引、不能由 descriptor 表达的 stride 等允许同一 Triton kernel 内的合法 pointer lowering，记录原因及覆盖率；不能关闭全图 descriptor 来取得收益。
- 不支持的 op、dtype、layout、shape 必须明确拒绝。验收图 CPU/ATen 外部计算回退为 0，graph break 为 0，不能只扫描 wrapper 字符串判定通过；要结合 dispatcher / runtime 事件审计。CPU 上的元数据与 shape 计算允许，张量数值计算与隐式搬回 CPU 不允许。

### 0.2 “走 GPU lowering”的准确含义

注册 `PrivateUse1` 并命名为 `tnca`，建立 allocator、device guard、stream/event、FakeTensor/Meta、Dynamo DeviceInterface，再为 `tnca` 注册 Inductor 调度、wrapper、heuristics 和 descriptor 能力。复用上游适用于加速器的 Triton 代码生成与融合机制，TNCA 后端把这些张量操作映射到自身执行模型。

**设备注册不会自动选中 GPU 编译路径，也不会自动解除块宽、寄存器和异步能力限制。** 现有 CPU 路径本来就采用 TritonScheduling；R40 要证明新的设备身份、设备属性、模板与运行时确实被使用。上游 `register_backend_for_device` 是可用的接入机制，具体签名按锁定版本核对。[PyTorch PrivateUse1 教程](https://docs.pytorch.org/tutorials/advanced/privateuseone.html)、[Inductor backend registration 源码](https://github.com/pytorch/pytorch/blob/main/torch/_inductor/codegen/common.py)。

`tnca` 的 Triton target 必须保持自身身份。warp、shared memory、TMA、FP8 等属性只能声明已经实现的语义；不能伪装 CUDA compute capability，或直接套用 NVIDIA 的布局、PTX lowering 和硬件吞吐假设。

## 1. 已知起点与商用方案差距

### 1.1 本地代码与已有证据

下表来自本地代码审查与 R39 记录；本次文档工作没有重跑上板测量。

| 已有能力 / 缺口 | 证据 | R40 的处理 |
|---|---|---|
| R39 v4 标准路径明显提速 | [R39 报告](r39_report.md)：tiny + small 52/52，相对 R39-00 `S_geo=4.315`；medium 的 v4 25/25、`S_geo=4.309` 见 R39-13 | 以最终签收的 R39 状态为起点，不能重复计入旧收益 |
| v5 / Mode A 仍在收尾；不能假定所有新增能力有收益 | R39-12：转置传输等在 tiny 有收益，Mode A 在几个短 K 点反而慢；R39 报告仍有 attention_decode 板/仿真 1 拍未结项 | R40-00 清点、复测、冻结已验收开关；流水按 workload 选择 |
| CPU 设备语义与保守配置 | [inductor.py](../python/tnca/inductor.py)：CPU 注册、块宽 16、单一 16×16×16 GEMM、persistent reduction 关、图缓存关；[pyproject.toml](../pyproject.toml)：Triton entry point 为 `cpu` | 原生设备接入与模板能力分别验收，不能只改设备名称 |
| 每 kernel 的 storage 往返与同步 | [driver.py](../python/tnca/triton_backend/driver.py)：只接受 CPU 张量，上传全部 distinct storage、submit/wait、下载；设备属性含占位的大容量值 | 真实设备 storage、长生命周期 buffer、事件依赖、批量提交；按实际 profile 报属性 |
| 上板仍是片上 RAM 单核 shell | [tnca_fpga_top.v](../rtl/fpga/tnca_fpga_top.v)、[tnca_fpga_mem.v](../rtl/fpga/tnca_fpga_mem.v)：默认 65536×8 B；[kernel.xml](../fpga/kernel.xml) 只有 AXI-Lite 控制端口 | 数据 AXI master、HBM 与 bulk DMA；AXI-Lite 保留控制用途 |
| 多核 NUCA 已有 RTL，但没有公共软件闭环 | [tnca_top.v](../rtl/tnca_top.v)、[tnca_tile.v](../rtl/tile/tnca_tile.v)：mesh、slice、TMU；末端只有共享内存口；[架构说明 §3.2](architecture_overview.md#32-the-multi-tile-system-rtl-only) | 接入 Session、标准路径、R39 核能力与多个内存出口；不能只运行已有 top 测试即签收 |
| 多核与单核最新能力尚未形成统一 profile 契约 | tile/top 的参数与单核 shell 的 R39 开关不一致，`system-2x2` 默认无 SFU | 核能力参数贯穿所有层；既有算子覆盖不能因复制核而缩水 |
| 宽 tile、分配与矩阵吞吐仍有缺口 | R39 报告 §6：宽块、软件流水、bank 感知分配未完成，K16 MMA 约 256 拍 | 宽块与资源分配先闭环，再以标准路径计数器决定计算资源投入 |
| 大规模点受人工 heap 边界影响 | [ArchBench TNCA runner](../archbench/platforms/tnca.py)：384 KiB heap、逐 launch 周期累加、单次首次调用测量；medium 曾为此缩 shape | 新增 HBM 套件，保持原有点定义；并发与真实外存必须升级计时协议 |

### 1.2 商用方案给出的设计启示

以下是公开实现中的机制对照，右列为**本文提出的 TNCA 方案**，不代表已实现或已测得收益。商用 ASIC 与 FPGA 的峰值、工艺和功耗不能直接构成公平胜负结论。

| 方案与公开机制 | TNCA 的差距 | 本阶段方案 |
|---|---|---|
| Meta MTIA-2i：Triton 专用后端、Inductor 生成调整、异步固定功能单元、软件流水、PID 到 PE 映射；公开论文包含编译生成的模型评测。[MTIA 论文 §3–5、§7](https://arxiv.org/html/2608.00325v1) | TNCA 还缺原生设备系统、通用资源模型和物理多核 grid 映射 | 原生设备 + 公共编译能力契约 + 自动流水和 locality mapping；性能只采用 Torch 生成路径 |
| AWS Trainium2：多 NeuronCore、HBM 与 DMA；NKI 的 SBUF / PSUM 分别承载操作数与累加结果，编译器管理资源和溢出。[Trainium2 架构](https://awsdocs-neuron.readthedocs-hosted.com/en/v2.31.1/about-neuron/arch/neuron-hardware/trainium2.html)、[NKI programming model](https://awsdocs-neuron.readthedocs-hosted.com/en/v2.26.0/general/nki/programming_model.html) | window / accumulator / scratch 生命周期保守，扩大块容易耗尽资源；外存与计算串行 | 显式资源生命周期、分阶段缓冲、受控 spill / retile 和异步搬运；保留 TNCA 的 NUCA 设计 |
| Google TPU：矩阵、向量、标量分工，HBM 供给矩阵运算，矩阵内部复用数据。[TPU 架构](https://docs.cloud.google.com/tpu/docs/system-architecture-tpu-vm) | 提升单核发射后，GEMM 的单位吞吐与复用不足会更明显 | 先改善块复用、布局和融合，再增加经过资源审计的矩阵吞吐 |
| NVIDIA Hopper：descriptor 驱动 TMA、异步事务 barrier 支持搬运与计算重叠。[Hopper 架构说明](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/) | TNCA 有 descriptor / barrier 原语，标准路径未充分利用多级并行 | 验证真正的异步 overlap、buffer ownership、epoch 与背压；不能把 `num_stages` 字段存在当作实现 |
| FlashAttention：通过分块与在线 softmax 减少 attention 的外存 IO。[原始论文](https://arxiv.org/abs/2205.14135) | 数学分解容易把 score / probability 大张量落到外存，接 HBM 后仍浪费带宽与容量 | 为标准 Torch SDPA 选择可审计的 Triton 融合模板；KV cache 驻留，检验实际 HBM 字节数 |

据此，R40 优先补齐**软件设备与存储闭环、可扩展执行/存储接口、局部性、计算和搬运重叠、应用级 IO**。FP8/INT4、稀疏、训练和多卡互联列为后续阶段；已有 FP32/FP16/BF16 与整数语义不得为 FPGA 实现方便而削弱。

### 1.3 U280 的物理事实及带宽预算

U280 有三个 SLR，HBM 接口位于 SLR0；板卡提供 8 GB HBM、32 个 AXI 接口 / pseudo-channel，公开标称总带宽约 460 GB/s。[U280 资源布局](https://docs.amd.com/api/khub/documents/rD6Ydbexy9OTF9vem6wnwA/content)、[HBM Memory](https://docs.amd.com/r/en-US/ug1314-alveo-u280-reconfig-accel/HBM-Memory)、[U280 Introduction](https://docs.amd.com/r/en-US/ug1314-alveo-u280-reconfig-accel/Introduction)。

跨 SLR 访问必须计入流水、布线和时序成本。AMD 文档建议靠近对应的 HBM 接口、避免对角跨区，并对跨区路径流水化。HBM 原生 AXI 事务为 32 B 粒度，小写事务会浪费带宽。[跨 SLR 流水建议](https://docs.amd.com/r/2024.2-English/ug949-vivado-design-methodology/Pipelining-Considerations-for-Crossing-SLRs)、[HBM AXI considerations](https://docs.amd.com/r/en-US/pg276-axi-hbm/AXI-Considerations)。

当前 8 B/cycle × 100 MHz 的单核数据口仅为 0.8 GB/s，三个同样的数据口合计也仅为 2.4 GB/s（本项目接口的理论计算）。因此，“HBM 已接上”主要先解除容量限制；要有带宽收益，必须同时扩展 descriptor 搬运、缓存行通路、inflight 与内存出口。

每个候选都交付带宽账本：core ingress、transfer engine、slice、NoC、跨 SLR link、AXI adapter、实际连接的 PC，各自给出读/写峰值与可持续测量。串联瓶颈取最小值，共享链路与读写竞争单独建模；不能把整板 460 GB/s 直接填成三个核可用带宽。

## 2. 可扩展架构与兼容性合同

### 2.1 逻辑层与实现层的边界

| 层 | 应承担的内容 | 禁止泄漏到此层的 U280 限制 |
|---|---|---|
| PyTorch / Triton 语义 | tensor、dtype、shape、stride、alias、descriptor、program ID、显式同步 | SLR 号、HBM PC 号、512 KiB heap、要求模型改为固定小 shape |
| TNCA ISA / artifact / capability | 执行与内存语义、真实能力、资源需求、版本协商、可拒绝的能力请求 | 把三核或固定通道数编码为 ISA 语义、把特定 FPGA 延迟写死进程序 |
| 通用编译器与 runtime | symbolic shape guard、tile / role / core 分配、资源生命周期、事件依赖、逻辑地址、合法 retile / spill | 按 benchmark 名称选择 lowering、以 SLR 布局改变数值或 alias 语义 |
| 通用系统 RTL | 可参数化 core / slice / controller 数、ready/valid 与 tag、NoC 路由、home 映射、内存序 | 只有 2×2 或 power-of-two 才正确、所有控制器汇聚到固定单端口 |
| board / platform adapter | XRT、HBM AXI、PC 地址映射、SLR placement、register slices、CDC、clock/reset | 反向要求 frontend 缩小维度、改变精度、取消 tensor descriptor |

逻辑 tensor descriptor 只携带逻辑地址与张量布局。物理 PC 分配及访问路由由 memory map / runtime / memory adapter 解释；合法 strides 与 views 的语义保持一致。逻辑地址字段保留 64 位，所有缩窄、PC 边界与容量检查显式且可测试。地址稳定和物理连续性分别声明，不能用低位截断处理 HBM 地址。

### 2.2 核数、拓扑与能力扩展

- 通用参数至少包括 `NUM_CORES`、`NUM_RCTX`、`NUM_SLICES`、`NUM_MEM_PORTS`、line / flit 宽度、MSHR / tag 数、buffer 容量、每类单元吞吐和延迟。能力由 profile / target / runtime 共同校验，实际镜像返回指纹与限制。
- U280 首个正式目标为 **3 个物理核分布在 SLR0/1/2**，每核配本地 NUCA slice；按实测布局决定内存出口和 PC 分组。2 核是中间 gate，4 核是资源允许时的扩展。若三核无法收敛，要提出有证据的调整与 amendment，不能把多个 role context 算成多个物理核。
- home slice、program 分配和 requester tag 必须支持非二次幂核数。设计不得靠 `core_id & (N-1)`、固定两位 source 或隐含配对核假设工作。选择规则需可实现，避免在地址热路径上引入昂贵的通用除法。
- 功能矩阵至少覆盖 1/2/3/4 核及 8 核 RTL smoke；1/2/4 外存出口、快/慢/乱序响应、不同缓冲容量。大核数的性能结论必须有实际测量；参数化和 elaboration 成功只是扩展能力证据。
- 保持 R39 已支持算子、dtype、layout、alias 和错误语义。资源不足优先采用语义等价的 retile 或有归因的 spill；特殊优化不适用时走同一 TNCA 标准编译路径。禁止静默 CPU fallback、放宽精度或删测试点。
- 旧 ABI / profile / CPU 迁移路径保留受版本约束的回归；新系统请求旧镜像缺少的能力必须明确拒绝。新增容量不会自动许可旧代码访问新地址范围。

### 2.3 内存序与完成语义

采用全局唯一 home slice 的共享 NUCA，首先建立可验证的 producer → consumer 可见性：kernel completion、event wait、跨核依赖、host copy 与必要的 flush/invalidate。完成事件至少等到所有可观察的写入到达正确的可见性域，不能只看计算核 idle。无需对 host CPU cache 承诺硬件一致性，但必须保证显式 copy 和 wait 的 PyTorch 可见语义。

多核原子、store mask、跨行读改写、dirty eviction、同地址读写、DMA 与缓存的别名访问均有明确规则。绕过 NUCA 的 host DMA 必须有写回/失效协议或明确的 ownership 转移。发生 fault 时停止派发、drain 已接受事务、返回首个可归因故障并推进恢复 epoch；不能留下仍在写旧 tensor 的核或 DMA。

## 3. 共同验收与消融规则

### 3.1 四种基线，避免混淆收益

| 基线 | 定义 | 用途 |
|---|---|---|
| B39 | R39 最终签收的软件、片上 RAM 镜像与配置；CPU 迁移路径 | 迁移兼容性与原有规模的端到端对照 |
| B40-S | 原生设备、单核、相同 HBM / NUCA 系统与软件，性能开关关 | 软件与 HBM 优化的配对基线 |
| B40-N | 同一多核镜像，仅改变 active core mask；保持 slice / PC / cache 容量一致 | 测量物理核并行本身的收益 |
| B40-F | 最终候选全部获准开关开；逐项关闭作 leave-one-out | 验证每项改进在最终系统仍有贡献 |

另报告独立单核实现 → 多核实现的产品级对照，记录增加的核、cache、port 和资源。不能把 B40-N 的固定总资源消融与产品扩容混为同资源优化。

### 3.2 每一个改进必须有消融

1. 每个独立机制有 compile/runtime/RTL 开关及配置哈希。打包开发可以共用工作包，**签收必须拆出每个机制**；例如宽块、bank 分配、流水、MMA 加宽、command replay 不能只给一个全开/全关数。
2. 每项都有单因素 A/B 与最终系统 leave-one-out。跨层协同还须做合法的 2×2，例如单/多核 × 单/多出口、朴素/局部性映射 × DCO、流水关/开 × 单/多 context。非法组合记录拒绝原因，使用语义等价对照，不伪造结果。
3. freeze workload、seed、tolerance、weight、clock policy、资源约束和子集。每个子集按 family / shape / arithmetic intensity 规则在测量候选前冻结，不能事后只取提速点。holdout 不用于选策略或阈值。
4. 全量报告逐点正确性、A/B 时间、`S_geo`、balanced 加权几何平均、完整套件 E、最差回退、counter / traffic、资源、频率、失败。继承 R39 的 `S_geo` 定义，新增的加权量独立命名；E 不能充当速度。
5. 除工作包单列的阈值，任何点回退 >2% 必须有逐点原因与自动选择/修正；最终默认策略仍回退 >5% 的点不通过。功能性改动可以签收为“能力交付”，不能被称为性能提升。
6. 性能改进的 95% 配对置信区间下界必须 >1；最终 leave-one-out 的目标子集至少 3% 收益。达不到者记为“无显著贡献”或回退，不累计为有效优化。已冻结的更高阈值未达到则该工作包性能 gate 不通过。
7. 新开关关闭，在原 profile 和原存储器的确定性路径上必须保持既有逐点周期 / artifact 合同；更换为 HBM / 并发系统时以能力、语义及对应基线对照，不能要求不同存储系统仍逐周期相同。

### 3.3 计时、统计与计数器

- **kernel makespan**：一个 grid 从设备接受到全部核完成且写入到达规定可见性域；**graph makespan**：图首次提交到最后依赖完成，包含设备等待与调度空隙。多 stream / 多核禁止把累计 active cycles 或各 kernel 重叠时间之和当作完成时间。
- 全局时间戳使用共同、可读回的时基；多 clock domain 的局部周期分别记录频率并转换，不能直接相加。旧 B39 若无法提供 graph timestamp，只比较两侧可重建的同口径 kernel 时间和同步主机 latency，不把旧的周期和与新 graph makespan 配对。
- **resident call latency**：权重和输入已驻留，主机调用前后同步，用 monotonic clock 测完成延迟；**request latency**：还计入显式输入 H2D 与输出 D2H。两者各有配对 A/B，不能用一种口径作 A、另一种作 B。
- cold compile / autotune、权重初始化、稳态执行分别记录。板上每点至少 5 次 warmup、30 次计时，分 3 个独立 session / 测量块；A/B 交错或随机顺序，报告 median、p95、配对 bootstrap 区间。长任务至少 10 次、3 块，样本数由 R40-00 预先登记。
- warm resident 与 cold-cache 分开报告。cold-cache 清理必须覆盖实际 NUCA 容量，不能假设换 tensor 就清了 cache；用于 memory roofline 的 cache policy 和 output completion 明确冻结。
- 每核 / 每 context：issued、有效 active、stall 归因及 busy；每 slice：hit/miss/merge、MSHR/full、dirty writeback；每 link / port：flit/beat、读写字节、backpressure、请求/响应延迟、outstanding 与公平性。互斥 issue/stall 守恒在每核成立，矩阵/DMA busy 可重叠、不能加入同一守恒式。
- 更新 ArchBench 的 `serial-grid`、首次调用单次测量、单口 roofline 假设。下界按实际 compute / memory / interconnect 拓扑和语义字节数计算，版本化；测得的可持续带宽用于诊断，不代替硬件峰值下界。跨资源/新 bound 的 E 只作伴随量，主比较使用实际完成秒数和 `S_geo`。
- provenance 保存 PyTorch 图、Inductor wrapper / Triton 源、TTIR、TNCA IR / 汇编、artifact 哈希、profile/capability、软件/tree 哈希、镜像/平台/clock、launch/event/core 映射及原始样本。每个性能点能反查这条链。

### 3.4 正确性、板卡与仿真

- 合入闸门继承 R39：`make r37`、core / core-vrf / core-slowmem / core-ooo / top / nuca、`r38-baseline-check`、`r38-profiler`、`r38-torch`、`r38-tdesc` 及适用的 R39 role/pair 回归；以冻结时的实际清单和数量为准，不能只复用文档中的过期计数。
- 新原生设备路径对全部 tiny + small + medium 数值通过，descriptor 开/关只作兼容回归，性能仅用开；dtype tolerance 沿用原套件，整数精确、NaN/Inf/mask/alias 规则有定向测试。
- 原片上 RAM 的 B4 继续要求确定性逐点等周期；R39 的未结差异保留状态，不通过改成 HBM 规则自动关闭。
- HBM 不承诺板/仿真逐周期相等。先核验协议与数据正确；以固定 AXI trace replay 对 adapter / NoC 的相同输入证明事务和完成时刻一致，再以板上延迟分布校准内存模型。预先冻结至少 10 个校准点与独立至少 10 个 holdout；holdout graph makespan median 相对误差 ≤10%、误差 p95 ≤20%。未满足则标记模型未校准，禁止用它的周期代替板上验收。
- 硬件先 lint/elaboration、定向快/慢/乱序/背压测试及 OOC，再 link。最终报告每 SLR 和全芯片 LUT/FF/BRAM/URAM/DSP、SLR crossing、post-route setup/hold、CDC 与 DRC。核心 100 MHz 为首轮目标；更改时钟需 amendment，报告同频与实际频率结果，不能让工具静默降频后签收。
- 板上性能结论必须来自实际 U280 镜像；能耗只有实测传感器且说明范围时才报告，没有测量为 null。故障、超时、容量不足与未测留在完整套件里。

## 4. 工作包与具体验收

以下阈值为工程目标，**不是已测得结果或性能预测**。R40-00 可依据基线测量在实现优化前提交带理由的 amendment；优化后不得降低门槛追认通过。`tools/r40/*`、`spec/r40/*` 与 `make r40-*` 为本阶段待交付接口，本文不会声称它们目前可执行。

### R40-00：关闭起点不确定性，冻结可复现协议

交付：

- R39-12/13 最终状态、Mode A 选择规则及未结 B4 的证据；冻结 B39 commit/tree、全部开关、镜像、工具链。未结项标明责任与阻塞范围，不能把 R39 v4/v5 结果混合为一份基线。
- `spec/r40/plan.json` / `baseline.json` / `suites/` / `ablation_matrix.json`：逐包 target subset、holdout、样本计划、公式、阈值、依赖及配置身份。锁定仓库使用的 torch 2.14.0 / Triton 3.8.0 与实际可用扩展 API；上游 main 文档仅作参考，升级单独做兼容消融。
- `tools/r40/ablate.py` 在 R39 工具基础上增加多时间口径、manifest 完整性、置信区间、配置差异白名单、硬件资源对照和 provenance 审计；更新 ArchBench 协议版本。
- HBM 套件预先冻结：保留全部现有点；新增 working set 超过总 NUCA 容量至少 4× 的 streaming / GEMM / transpose / gather，以及原始 large 档和多步 decoder。分别记录容量、编译覆盖与实测，不缩原 large shape。

验收：

1. `make r40-protocol`：人为删除慢点、零填未测、混合镜像、改 tolerance/weight、漏 provenance、把 core cycles 相加冒充 makespan、只改频率、夹入 autotune/replay 预热，全部应拒绝；相同输入对照为 1。
2. B39 全量数值与现有 gate 可复现；device、host、传输及 compile 的时间边界可从日志核对。每个后续性能实验在实施前已有冻结子集。
3. `tools/r40/acceptance.py` 以机器可读状态区分 `capability-pass`、`perf-pass`、`no-significant-gain`、`fail`、`not-measured`，缺命令/缺证据不能 PASS。

进展（2026-10-10，执行记录；上面的合同未改动）：

- **R39 收尾**：R39 已签收，PR #6 合入 `r38-dco`（merge `b5c606d`）；`r40` 从这里分出。
  - 终版：软件 `be87779` + 镜像 u280_r39h（v5）。相对 R39-00 基线 77 点 S_geo 5.548，相对 v4 终版 1.287。
  - Mode A 带 `ring_estimate` 门控：77 点 S_geo 1.0045，套件周期 −0.37 %，记为「无显著贡献」，默认关。
  - 选择规则：`rctx_roles=1` 只在估计有收益、且两个 role 不会无序读写同一块内存时拆分；`2` 强制（测试用）。
  - 证据在 `doc/r39_report.md` §8.4–8.5。
- **B39 冻结**（`spec/r40/baseline.json`，`spec/r40/b39/`）：
  - 软件：commit `be87779`、tree，以及内容摘要 `files:04338f…`（`archbench/provenance.py`，git checkout 与无 .git 的副本算法相同）。
  - 默认开关与 Inductor 配置。
  - 镜像：sha256、profile、capabilities、RTL 参数、时钟策略与实测频率、时序。资源按 SLR 记录在 `spec/r40/b39/resources.json`。
  - 工具链：板卡主机与构建主机的 Vitis/Vivado/XRT/torch/Triton/Icarus/Yosys 版本。
  - 记录：77 条板上记录、B1、B4 的仿真参考。只有一种配置（v5、Mode A 关），不混 v4。
  - 未结项 5 条，各有责任与阻塞范围：
    - B4：llama_decode 与 attention_decode，都是跨 launch 状态造成的 1–2 拍。
    - softmax fp16 tiny 对时序开关敏感（交 R40-07）。
    - v5 上 fp32 matmul/conv2d 慢 0.7–1.4 %（交 R40-07）。
    - GitHub Actions 因账户计费不启动。
- **起点更正**：§1.1 表中「K16 MMA 约 256 拍」是 R39 早期估计。实测 v4 是 551 拍，v5 的 `MTX_ABUF` 降到 311 拍（R39 报告 §6、§8）。
- **API 锁**（`spec/r40/api_lock.json`，`tools/r40/api_lock.py --check`）：torch 2.14.0、Triton 3.8.0 下，R40-01 要用的 25 个扩展 API 都在，签名已记录。
  - 包括 `register_backend_for_device`（带 fx wrapper 与 custom config 参数）、`_setup_privateuseone_for_python_backend`、`register_interface_for_device`。
  - Triton 以 entry point 名 `cpu` 发现本后端（§1.1 所说）。
- **ArchBench 协议 2**（`archbench/PROTOCOL.md` 版本 2）：
  - 冷启动首次调用单独记录；warm-up 与计时调用分开，每次调用一个样本。
  - 计时口径：`device-cycles-serial-grid`，有计时调用时另有 `request-latency`。
  - `host.steady` 记录稳态的 upload / run / download。
  - 每条记录带 provenance：软件摘要、artifact sha256、profile 与 capabilities、镜像与时钟、launch map。
  - `--bundle-dir` 把 Dynamo 图、Inductor wrapper 与 Triton 源、TTIR / TNCA IR / 汇编、Triton cache 与 artifact 逐文件哈希存下。
  - 仿真上 tiny 的 add 与 mlp 已验证。
- **工具**（`tools/r40/`）：
  - `ablate.py`：多计时口径、配置差异白名单（knob 与 `platform.<字段>`）、bootstrap 置信区间、bundle 审计、资源对照（`resources.py`）。
  - `acceptance.py`、`protocol_tests.py`：34 个用例全部通过。每个反例都按正确的原因被拒绝；相同输入 S_geo = 1，结论为「无显著贡献」；缺命令、缺证据或不点名的 gate 都不能 PASS。
  - `b39.py`、`b39_gates.py`、`b39_repro.py`、`freeze.py`、`hbm_coverage.py`、`evidence.py`。
  - make 目标：`r40-protocol`、`r40-freeze[-check]`、`r40-api-lock`、`r40-hbm-coverage`、`r40-b39-gates`、`r40-b39`、`r40-acceptance`。
- **HBM 套件**（`archbench/hbm.py`、`archbench/suites/hbm.json`、`spec/r40/suites/hbm.json`）：
  - 新增 12 点，working set 都 ≥ 32 MiB，即架构全部 NUCA（8 MiB）的 4 倍：
    - streaming：add / silu / copy 2^26、softmax、rmsnorm。
    - GEMM：matmul 4096³、linear、gemv 对 86 MiB 权重。
    - transpose：8192² fp32。
    - gather：embedding，从 256 MiB 表取 8192 行。
    - R40 decoder：4 层，211 MiB 权重，prefill 512；`archbench.generate --tier hbm --new 64` 是多步 decoder 的 workload。
  - 套件 = 全部现有点 + 原 large + 这 12 点，共 113 点，加 decoder workload。原 large 未缩。
  - 覆盖记录（`spec/r40/hbm_coverage.json`）：large + hbm 36 点中 35 点经标准路径编译通过，0 点放得进 B39 的 heap，0 点实测。
  - 未编译的是 `op.layernorm/large`：循环 reduction 产生 `scf.if`，TNCA 没有 lowering。交 R40-03。
- **冻结子集**（`spec/r40/suites/`，21 个；`plan.json`、`ablation_matrix.json`）：
  - 每个子集都是对 family / tier / shape / 算术强度 / B39 记录的规则，解析成带定义哈希的点表。
  - 每个性能实验都有冻结的子集、可变开关、计时口径、阈值与基线。
  - R40-09 holdout 是 medium + large + hbm 按点 id 哈希取的 30 %（14 点）；HBM 模型的校准 / holdout 各 17 / 19 点。
  - 其余每包的子集与阈值见 `plan.json`。
- **复现**：
  - B39 软件（`be87779` 的导出）在 u280_r39h 上重跑：77/77 数值通过，**77/77 逐点周期与冻结记录相同**。
  - R40-00 的树按协议 2 在同一镜像上跑（5 次 warm-up、10 次计时）：
    - 77/77 通过；首次调用的周期与 B39 全部相同，10 次计时之间没有周期差。
    - 时间边界一致：host run ≥ 器件时间，request ≥ 各部分，首次调用 ≥ 稳态调用。
    - compile 中位 3.5 s；CPU 张量路径上器件时间只占 request 的中位 0.3 %。
  - B39 对它（经 `ablate.py`，只声明软件 tree 不同）：接受，S_geo 1.000000，CI [1, 1]，无显著贡献。
  - R39 的 27 个合入 gate 在本树上运行，计数按实际冻结（`spec/r40/b39/gates.json`）。
    - 其中 3 个在 B39 commit 上同样不以 0 退出，连同原因一起冻结：`r38-l0`（两项按设计 pending）、`r38-baseline-check`（R38 基线的 profile 配置；149/149 用例通过）、`r38-identity`（需先 `make r38-baseline`）。
- **R40-00 关闭**（2026-10-10）：`acceptance.py --wp R40-00` 的四个 gate 都是 `capability-pass`：
  - protocol 34/34。
  - frozen 33/33。
  - b39-gates 27/27。
  - b39-reproducible：上面的三次运行加 identity。
  - 证据在 `evidence/r40/R40-00/`。
  - 此后对 `spec/r40/` 的计划、子集或阈值的任何改动都是 amendment。

### R40-01：原生 PyTorch TNCA 设备与 Inductor 加速器路径

交付：

- PrivateUse1 backend、真实设备 storage / allocator、`empty/empty_strided`、显式 copy、view / offset / stride / alias、guard、设备模块、stream/event 基础接口；支持 FakeTensor / Meta 与编译所需 shape guards。
- TNCA DeviceInterface、Inductor Scheduling / wrapper / device op overrides、设备专属模板与启发式、descriptor capability、Triton `tnca` entry point 与 driver。明确隔离旧 CPU 迁移路径，使用同一 TNCA lowering 和 Session。
- 实际 profile 属性替代占位无限资源；多 compile worker 初始化、缓存 key、设备与版本身份一致；CPU/CUDA/TNCA 同进程使用不相互覆盖配置或劫持 driver。

消融：A 为 B39 CPU 迁移路径；B 为原生设备但 `transfer_policy=per-kernel` 的合法 staging 对照；C 为原生驻留。A/B 评估生成代码与设备调度变化，B/C 的 residency 收益归 R40-02。每个对照匹配输入/输出边界；不能把 A/C 的总收益全部记给注册设备。

验收：

1. `make r40-device`：`model.to("tnca")`、`torch.compile(fullgraph=True)`、输出 `.cpu()` 跑通全部 tiny + small + medium；tensor device/dispatch key、device scheduler、target 和 runtime buffer 证据一致，零数值 fallback。HBM 就绪前先跑容量可容纳的 smoke，完整签收在 R40-04 后补齐，不缩原点。
2. 有 CUDA 与无 CUDA 主机分别跑混合设备测试；导入/编译顺序、并发首次注册、进程 worker、缓存命中、checkpoint map_location、不支持 dtype 的错误均可复现。
3. descriptor 实际代码与设备事件覆盖 GEMM / reduction / pointwise 代表点；新增 device/codegen 在 tiny + small + medium 的设备时间 `S_geo ≥1.0`，无未处理 >5% 回退。此包以设备能力签收；若声称 codegen 提升须另满足 §3.2。

进展（2026-10-10，进行中；设计与审计见 `doc/r40_tnca_device.md`）：

- **设备**（`python/tnca/torch_device/`）：PrivateUse1 改名为 `tnca`。
  - C++ 扩展：allocator 的 DataPtr 是器件地址（runtime heap 里的 buffer），加上 empty / empty_strided / as_strided / view / `_reshape_alias` / set_ / resize_。
  - Python 部分：所有方向、任意 stride 与 dtype 的拷贝，fill_ / zero_ / item()；device module `torch.tnca`（stream / event 为 R40-01 的同步语义）。
  - 选择 C++ 的原因：torch 2.14 的纯 Python 后端只能建无 storage 的张量，view 与 alias 都得逐算子重写。
- **编译**：
  - Dynamo DeviceInterface 的属性取自 profile 的 target 文件。
  - `register_backend_for_device("tnca", TritonScheduling, ...)` 与 device op overrides；kernel 以 Triton target `tnca` 编译。
  - launcher 直接传 `tnca` 张量自己的 buffer 与偏移，0 字节上传；混合 host / 器件张量的 launch 被拒绝。
  - runtime 的代码区改为与 heap 同样的分配器，加 `Session.unload`：张量常驻时，代码区满就卸载最久未用的模块。
- **lowering 策略**（`tnca.torch_device.policy`）：
  - `tnca` 不加入 Inductor 的 GPU_TYPES，`is_gpu()` 处取 CPU 路径（B39）的答案。
  - 以 `device.type == "cpu"` 判断的 6 处给 `tnca` 同样的答案，各有开关：模板、方差、cat、小矩阵分解（含 bmm lowering）、realize 与融合、conv layout。SDPA 的 backend 选择取 CPU 自己的选择器（C++ stub）。
  - 先当作 GPU 实测：tiny 上 S_geo 0.834，layernorm 0.32、rope 0.31，原因见设计文档 §3。改用上述策略后对齐。
  - `scf.if` 的 lowering（if-conversion / 分支）补齐：Welford 与 `op.layernorm/large` 的编译缺口。
- **测试**（`tools/r40/device_tests.py`，16 项）：
  - 仿真与 U280 板上都通过：storage 与 alias、拷贝、module、拒绝、编译图（pointwise / reduction / norm / GEMM / SDPA，0 上传）、view、代码区卸载、属性、策略。
  - 验收 2：有 CUDA 主机（GB10）上 CUDA 与 tnca 图同进程交替运行；无 CUDA 主机（板卡主机）也测。另有三种 import 顺序、8 线程并发注册、spawn worker、cache 命中、checkpoint map_location（cpu / tnca）、复数与 fp64 计算的拒绝。
  - plugin identity O0/O1/O2 149/149 字节相同：改动不改变任何 B39 制品。
- **设备时间**（tiny，RTL 仿真，相对 B39 的 CPU 路径）：27 点中 20 点逐拍相同，S_geo 0.999。差别都在 2 % 以内，各有原因：
  - softmax fp16、gpt2_block、attention_prefill：Inductor 对非 cpu 设备在“reduction 循环里写回后再读”前加 `tl.debug_barrier()`，TNCA 编译为 fence，每处约 20 拍。这正是 B39 缺的同步（B39-OI-6）。
  - resnet_block −1.7 %：FakeTensor 的 conv backend 选择对 PrivateUse1 给出 contiguous 输出，CPU（MKLDNN）给 channels-last。
- **板上**（u280_r39h，`TNCA_TORCH_DEVICE=native`，协议 2）：
  - tiny 27/27、small 21/25、medium 20/25 通过。其余 9 点 capacity-exceeded：常驻张量放不下片上 384 KiB 的 heap。
  - 68 点相对 B39 S_geo 1.0015，52 点逐拍相同，最差 0.983。
  - 请求延迟在 tiny 上是 B39 CPU 路径的 3.59 倍快（几何平均，范围 1.66–17 倍）。
- **gate**（`evidence/r40/R40-01/`）：
  - `R40-01.mixed-device` capability-pass：有 CUDA 与无 CUDA 两台主机各 16/16。
  - `R40-01.device` 与 `R40-01.device-time` 为 not-measured（9 点 capacity），按验收 1 在 R40-04 之后补齐。
- **graph 输入上的 descriptor**（`TNCA_INPUTS_ALIGNED=1`，实测）：medium 上 rmsnorm 0.62、softmax 0.92、transpose 0.83。原因是一行一个 program 的 descriptor load 比 vector load 贵，所以默认关，留给 R40-03 的多行块。

### R40-02：张量驻留、bulk DMA、事件与批量提交

交付：

- storage 生命周期贯穿整图和多次调用；设备 allocator 可复用且 stream-safe。只在显式输入、输出、checkpoint 边界搬运；权重、临时结果与 KV buffer 不逐 kernel 往返。
- XRT bulk 数据搬运与通用 copy 接口；AXI-Lite 只发控制。依赖由事件连接，安全延迟回收 buffer；批量 command queue，支持受 guard 保护的图 replay，epoch / shape / alias 变化后正确失效。
- copy/compute overlap 是独立开关，不能把同步队列换成异步后只测提交耗时。跨 stream 使用、原地更新和 CPU 写入后的再次 copy 明确记录 ownership。

消融：同一原生设备图，`per-kernel / resident`、`single-submit / batched / replay`、`copy-overlap=0/1` 分别比较；host control 与 DMA 改进也分别测量。初始化和稳态同时报告。

验收：

1. `make r40-residency`：链式图与多步 decode 的中间张量 H2D/D2H 字节为 **0**；权重初始化后稳态重传为 **0**；输出下载仅限显式请求。
2. 在预冻结的 ≥3-kernel 图子集，resident 相对 staging 的 steady request latency `S_geo ≥1.5`；批量/replay 在独立冻结的小 kernel 图子集 resident call latency `S_geo ≥1.2`。全量退化按 §3.2 处理。
3. 1000 次图调用、非连续 alias/offset、异步 free/realloc、多 stream event、故障恢复后的 stale handle、replay 失效通过；buffer 高水位及末尾未释放 storage 可审计，不要求缓存 allocator 归零。

进展（2026-10-11，进行中）：

- `TNCA_TRANSFER_POLICY=resident/per-kernel` 已接入原生 launcher；两者使用同一标准路径制品。
  staging 为每个 distinct storage 分配临时 device buffer，host bytes 往返搬运，
  offset/view 别名共用该临时 buffer；完成后复制回原 storage，并记录全部实际传输字节。
- `tools/r40/residency_tests.py`：驻留零 kernel H2D/D2H、staging 实际往返、
  两策略制品身份、两策略非连续 offset/alias 原地更新，以及未知策略拒绝，6/6。
- `TNCA_SUBMIT=batched` 使用有界 runtime 队列和真实 completion event；
  中间 storage 在最后一个任务释放 lease 后回收。stream context 传递实际逻辑 ID，
  当前单核仍通过同一 FIFO 保证顺序，不声称物理 copy/compute overlap。
- 分配 token 与物理地址分离，恢复后同地址重分配不会让旧 tensor/deleter 命中新 storage。
  async lifetime 2/2、三次恢复 epoch 的 stale handle 隔离 3/3、payload 并发与 mutation 2/2；
  原生设备完整回归 16/16，runtime 生命周期与 batch 旧回归各 7/7。
- `make r40-residency` 当前为能力 checkpoint；冻结链式图、多步 decode、
  replay 完整压力、高水位账本及性能 gate 尚未完成，不能据此签收 R40-02。
- 原生 batched 模式 1000 次调用实测通过：有界队列、数值、末尾 storage 数与 leases 审计通过；
  stream context/事件及零 kernel 传输同组通过 4/4。并发线程审计另组 5/5。
- 标准 Torch launch 可通过 `torch.tnca.graph(Graph())` 捕获并保留 intermediate storage；
  replay 复用同一 module/参数/制品、跳过 wrapper，日志仍等到真实 completion。
  shape/stride/storage alias 和 recovery epoch 在首条 admission 前整图校验；
  失败后须 reset/重新捕获。实测 4/4；含更新输入与拒绝部分执行的负例。

- allocator 在容量不足且有未完成任务时，等最早任务释放 retired storage 后重试；
  容量检查与分配在同一 Session 锁内。393216 B 真实片上 heap 下 12 次调用通过 2/2：
  12 次容量背压均低于 8-task 队列上限，实际保留字节峰值 328704 B，最终 65536 B。
  `Session.memory_stats` 计入 retired lease 与 scratch，并报告每 epoch 的 heap/code 高水位。
  新 allocator 的恢复 3/3、lifetime 2/2 复验通过；并发 1000-call 5/5。

### R40-03：通用 tile、布局与资源分配

交付：

- 1-D 块至少 16/32/64/128 元素，跨多个向量寄存器；多行 reduction；至少两种大于 16×16 的逻辑 GEMM tile，由已有 MMA fragment 组合，K16/K32/K64 选择按能力校验。
- bank 感知分配、window / descriptor / barrier / accumulator / scratch 生命周期；压力过大有可归因 retile / spill，区分哪一级存储与实际 bytes，不能临时换成 IR builder 的 GEMM。
- shape/stride/tail/layout 转换均由通用规则处理；GPU 生成的布局若不适合 TNCA，在共享编译 pipeline 中转换。动态 shape 以 guard/重编译处理，不能变成永久小 shape 限制。

消融：`max_block`、多行 reduction、GEMM tile、bank allocation、lifetime reuse、spill 策略逐项；保证 CPU/GPU 设备迁移收益与这些算法变化分开。

验收：

1. `make r40-tiles`：跨寄存器 tail（17/31/33/65/127 等）、transpose/stride、mask、int64 index、alias，既有 dtype 均正确；压力不足触发合法 retile/spill，资源统计与机器需求一致。
2. pointwise/reduction 的冻结 medium + HBM 子集相对窄块 `S_geo ≥1.3`；更大 GEMM tile 的冻结 reuse 子集 `S_geo ≥1.15`。每个独立机制最终贡献仍按 §3.2 验证。
3. 开关关时旧 profile 制品合同通过；新的能力不能由 U280 PC / SLR 或 benchmark family 名称决定。

进展（2026-10-11，能力 checkpoint，未签收）：

- 标准 Torch 路径的 1-D block 跨 VLEN 寄存器分块；16/32/64/128 各 12/12，
  包含 tail、int64、fp16/bf16/i8/i16/i32、stride/offset/alias 与原地更新。
  测试检查制品真实宽度，避免只把 cap 字段当作执行宽度。
- predicate 按实际容量重建；直线 IR 的长寿命 vector 可用私有 per-program scratch。
  128-wide int64 图实测触发 896 B / 14 slot，制品为每 role 52 VRF / 两 role，数值正确。
  控制流 spill 仍拒绝；修复 RCTX fallback 漏掉 plain kernel 最后一次重试的问题。
- 单行 wide reduction 的 sum/amax/RMSNorm/softmax/LayerNorm 在 32/64/128 各 20/20；
  通用多输出 argmax fold 在 16/32/64/128 各 4/4，覆盖 NaN 与首个并列索引。
  scalar-carry loop 保持逐迭代内存顺序；实际宽度/kind 写入制品 provenance。
- 多行 reduction、大 GEMM tile、bank 分配、完整资源压力、旧制品合同与冻结性能 gate 待测。
  证据 `evidence/r40/R40-03/`，默认 block cap 仍 16。

### R40-04：单核真实 HBM 与可扩展外存接口

交付：

- 通用 line/block request-response 接口，带 source/transaction tag、byte mask、backpressure、错误与完成语义；板级 HBM AXI adapter 提供 burst、multiple outstanding、4 KiB/PC 边界分裂、读写独立跟踪、CDC。
- descriptor block 搬运路径目标至少 **32 B/core cycle** 的入口能力，标量 LSU 继续支持 8 B 与掩码。line/beat 宽度参数化；合并与分拆不改变可观察语义。burst、outstanding 深度、宽入口分别有消融。
- 全部 32 PC 的地址 map / capacity 有可审计覆盖，active PC 数按预算选择，不能分配到不可达区域；heap 来源于真实 device memory，代码/descriptor/control 有保护边界。
- XRT BO / bank placement 与设备地址对应，地址高位、存储别名和 host DMA 可见性测试覆盖 PC 边界；给通用接口提供确定性 memory model 和另一种配置 adapter / model，避免 HBM 协议进入核心。

消融：容量解除记为能力交付；相同驻留 HBM 系统上 `burst=0/1`、`outstanding=1/N`、窄/宽 descriptor ingress 分别作标准路径 A/B。片上 RAM 与 HBM 的 latency 对照只作迁移代价，不能凭容量变化宣称提速。

验收：

1. `make r40-hbm`：标准 Torch 图的权重或工作集 ≥64 MiB，明显超过片上 RAM，实际 HBM traffic 非零；全地址空间的 host/device 抽样与高位地址测试正确，不能只测第一个 PC。
2. 大于 cache 4× 的标准 Torch streaming 读、写、copy：各方向有效带宽 ≥冻结瓶颈峰值的 **50%**；独立宽入口/并发改进的目标子集 `S_geo ≥1.5`，同时报告 burst efficiency 和读写 traffic 放大。
3. 乱序 response、随机背压、相同 line 的 subword stores、错误、timeout/reset、跨边界事务通过；HBM 模型 fidelity 满足 §3.4，板上数据正确且 timing/CDC gate 通过。

进展（2026-10-10，进行中；设计、验证与带宽数据见 `doc/r40_hbm.md`）：

- **接口**：核心的数据口不变，后面接一个桥。
  - 低于 `LOCAL_LIMIT`（0x20000）的地址仍走片上 RAM，包括代码与参数。
  - 其余地址走 AXI 窗口：`axi_base + (A − LOCAL_LIMIT)`，由 host 写 `R_AXI_BASE`/`R_AXI_SIZE`。
  - 新寄存器 0xA8–0xD8：窗口、粘滞错误/无主响应、AXI 读/写/读延迟/R 拍计数器。旧镜像读为 0。
- **两个桥**（`burst` 机制的 A/B，`AXI_LINE`）：
  - `tnca_axi_bridge`：单拍，按字的 hazard 表；atomic 在桥里做读-改-写；复位后 1024 拍的静默窗口吸收在途的旧响应。
  - `tnca_axi_bridge_lb`：读按 64 B 行，INCR burst 进入全相联行缓冲，预取下一行。写直达，FIFO 化。`inv` 在每条命令时拉一拍，保证 host DMA 之后可见。
  - 顺序规则与变异：单拍桥 5 个变异，行缓冲桥 6 个，全部被抓到。
- **仿真**：内存模式 `core-vrf-sync-v5/hbm`。编译、能力与 artifact 按 base profile 走；bench 带 32 MiB 的 AXI 窗口与可调时序的 AXI 模型。
  - `make r40-hbm`：桥在 4 种几何下各 7 个用例，覆盖乱序、背压、SLVERR、在途复位、DMA 后失效。
  - device tests 16/16，恶劣时序下 5/5；张量在窗口 24 MiB 处的图正确；整窗 32 个样本先全部写入后再读回，正确。
  - `DATA_AXI=0` 不变：B4 sim tiny 27/27 与 B39 逐拍相同；R39 合入 gate 照冻结数复跑；`r36-fpga-tests` 以 `DATA_AXI=0/1` 两种方式展开。
  - 另修复了 `R36-FPGA-08` 在 HEAD 上的既有失败：bench 写死的“越界计数器”下标 20，在 R39-00 增加计数器后成了真实计数器。
- **板卡**：`rtl/board/tnca_hbm_xo.v` 与 `fpga/kernel_hbm.xml`；设备直接运行在 ap_clk 上。
  - link 时 `sp=tnca_u50_1.hbm:HBM[0:31]`。
  - runtime 按 xclbin 的 group connectivity 在所连 bank 中分配窗口 BO，写入基址；窗口内的传输走 XRT DMA，heap 以设备报告的大小为界。
  - 首个镜像 `u280_r40a`（单拍桥，v5 参数）：OOC 100 MHz 时 WNS +1.367 ns；link 于 2026-10-11 失败（WNS −0.683 ns），没有可装载 xclbin。OOC 主时钟误入整机实现，修复及复测见 `doc/r40_hbm.md` §7。板卡共享，装载前检查占用；作者已于 2026-10-11 授权持续推进到 R40 验收并 commit/push。
- **带宽**（仿真，AXI 延迟 30 拍）：
  - 行缓冲 16 行 / 32 waiter / OUT_N 16 时，read/write/copy 为片上 RAM 的 91% / 81% / 77%；延迟 1 拍时与片上 RAM 逐拍相同。
  - 天花板是核心本身：16 元素块下仅 2.3–3.7 B/cycle，约为 8 B 入口的 30–45%。所以验收 2 的 50% 门限依赖 R40-03 的宽块与 32 B 入口，只改桥达不到。
- **三核可行性**（§5 要求尽早验证）：单核约占 SLR1 的 64.6% LUT（27.9 万）。
  - 三个 CU（每 SLR 一个、各自接 HBM）的 link 已开始，但与单核 link 并行时共享主机内存只剩约 10 GB（另有其他用户的 Vivado），已停止，待单核镜像完成后再跑。
- **gate**：`R40-04.robustness` 与 `R40-04.hbm` 的仿真部分全部通过，均为 not-measured，等板上部分；`R40-04.bandwidth` 为 not-measured。

### R40-05：多 SLR 物理多核接入公共 runtime

交付：

- 统一 single-core / multi-core 的 capability 和核心参数贯穿，保留 R39 已验收的 SFU、descriptor、role-SPMD、pair / slice 等能力；每核私有代码、参数、scratch 与资源隔离。
- 全局 grid dispatcher：一个逻辑 launch 分发 global program IDs，每个 program 恰好执行一次；核内 role 是独立层。有限队列可连续领取任务，解除旧 4096-program 硬编码容量限制或按真实可流式能力报告。
- 1→2→3 核上板，跨 SLR 的 pipelined / elastic link；通用拓扑与路由不依赖 SLR，board 层定义 placement。三核、八核 requester tag / home / dispatch 正确。
- 全局 completion、multi-core perf、launch ID / event / fault / drain 协议，禁止 host 对每个 program 串行 MMIO launch。

消融：同一三核镜像 `active_cores=1/2/3`；role 数固定。另作保持总 PC/NUCA 不变与随核扩容的对照。静态/动态 grid 分配作为独立改进，不与复制核打包计数。

验收：

1. `make r40-multicore`：标准路径数值全通过；运行证据显示三个物理 core 同时活跃，post-route placement 覆盖 SLR0/1/2；不能用三次串行调用或多个 context 充数。
2. 预冻结且并行度足够的 compute 子集（grid ≥32×active cores），B40-N 三核 graph makespan 相对一核 `S_geo ≥2.1`，两核 ≥1.5；报告 `speedup/N`、每核负载与频率。
3. grid 1-D/2-D/3-D、非整除核数、最后不满批次、>4096 programs、零工作/错误输入、混合快慢 program、各核 fault 与恢复通过。1/2/3/4 核回归、8 核标准 Torch smoke 通过；elaboration-only 不满足 smoke。

### R40-06：NUCA、多个内存出口与自动局部性

交付：

- 去掉固定共享 DRAM 单出口，slice / controller / PC group 路由参数化，分布式仲裁和 tag 回程；避免先串行化再在末端扇出。credit 与虚通道的依赖证明、随机背压与持续公平性。
- 唯一 home、dirty ownership、原子与 fence、DMA ownership 转移；tensor home / program mapping / address placement 共用通用策略。稳定 layout、跨 kernel 复用与自动策略不能要求用户显式填 PC 或 SLR。
- DCO tensor metadata / lifetime 由标准图与 descriptor 推导并经 TMU 接入；普通 home mapping、locality、cache bypass、replacement / dead tile hint 为独立开关。无法证明 dead 的 tensor 不得提前回收，别名与异步依赖计入生命期。
- slice容量、MSHR、NoC flit/link 带宽按 §1.3 的账本定量调整；cache 和 interconnect 改动必须报告资源与 clock 代价。

消融：单/多核 × 单/多 memory出口；朴素/locality × DCO；cache on/bypass 在相同总容量与 PC 预算下。cold streaming 与 warm reuse 分开，所有带宽测量来自 Torch 图。

验收：

1. `make r40-nuca`：跨核 producer/consumer、same-line masked stores、atomic reduction、eviction 后 host copy、DMA 更新 cache、alias/dead hint、饱和读写与 fault drain 正确；无 deadlock、丢响应或饥饿。
2. 三核 HBM streaming 子集相对一核同资源基线 graph makespan `S_geo ≥1.8`；多出口相对同核单出口 `S_geo ≥1.3`。每个 active port / PC group 有实际流量，不能靠扩大 cache 容纳整个点获得“HBM 扩展”。
3. locality/DCO 在预冻结 reuse/remote-traffic 子集 `S_geo ≥1.1`，至少一项实际 remote bytes 或 HBM bytes 降低 ≥20%；两种机制单独消融。流式负载默认策略不得强制绕远或反复污染 cache。
4. 同一 Torch 图在 1/2/3/4/8 核功能矩阵与不同 memory latency / port count 下保持语义，参数调整不需要修改 model、ISA opcode 或 frontend 算子规则。

### R40-07：异步流水、布局搬运与矩阵吞吐

交付：

- 用真实标准路径 profile 分辨 DMA 等待、MMA service time、发射槽、VRF、accumulator、NoC/HBM 背压。延续 R39 的负结果：短 K、Mode B 已能覆盖等待时，Mode A 不自动开。
- 通用循环 eligibility、2/3-stage buffer、producer/consumer 或单 role pipeline；barrier/epoch/buffer ownership 验证，编译报告 overlap 计划和拒绝原因。
- descriptor transpose、寄存器 pack / layout conversion 与 epilogue 融合，减少经全局 scratch 的重排；MMA 计算/装载重叠或加宽按资源预算择优，不能为吞吐牺牲既有精度。
- 每个实现更新统一 latency/throughput/resource 模型。矩阵改造若无有效标准路径收益，保留明确负结果与资源分析，不把微基准提升写成工作包通过。

消融：stage 数、role pipeline、transpose/pack、epilogue fusion、MMA 改造逐项；`pipeline × context_count`、`pipeline × MMA` 做交互对照，生成代码与硬件两层都能回溯。

验收：

1. `make r40-pipeline`：长 K/tail/masked tile、不同 stage 数、快/慢/乱序 HBM、barrier epoch wrap、fault drain 正确；设备 trace 显示 DMA 与 MMA 的时间重叠，而非仅有 async opcode。
2. 长 K GEMM 标准路径子集的 pipeline 相对串行 `S_geo ≥1.2`；选定矩阵改造相对同 tile/同软件/同频基线 compute-bound 子集 `S_geo ≥1.2`。每项资源开销、实际秒级收益与最终 LOO ≥3% 同时满足。
3. 短 K 与低并行度点由冻结的自动策略选择，完整 tiny + small + medium 无未解决 >5% 回退。100 MHz 目标未收敛时按实际频率 gate，不能只报 cycles。

### R40-08：标准 Torch 的 attention / reduction 与驻留 KV

交付：

- `torch.nn.functional.scaled_dot_product_attention` 经设备 lowering / Inductor 模板生成融合 Triton，descriptor 分块加载、在线 softmax / 稳定 reduction；能力不足时走同一标准数学分解并明确说明。
- 不向 HBM 物化完整 score / probability 矩阵；GQA/MQA、causal、tail 与 mask 按冻结范围验证。R40 推理范围为 dropout=0，额外 unsupported 情况明确拒绝，原来支持的情况不能删掉。
- KV cache 真正驻留，支持多步写入/读取与 event 序，position 由 tensor/guard 支持；稳定 shape bucket / 容量策略限制重编译，禁止每 token 重新上传整份 KV。
- reduction/SFU 融合与在线 attention 分别消融，不能把 softmax 精度放宽带来的速度计作架构改进。

消融：同一标准 Torch SDPA，分解/融合；KV copy/resident；融合/未融合 reduction。相同精度、输入、mask、输出比较，同时提供中间 HBM bytes 与峰值设备分配。

验收：

1. `make r40-attention`：原有 attention / llama / GPT-2 图全通过；新长序列、奇数长度、全 mask 行、GQA/MQA、64-step cache 更新与参考一致，CPU 数值回退为 0。
2. 冻结 long-sequence prefill 子集相对标准分解 `S_geo ≥1.5`；分配与 traffic 记录证明无全量 `L×L` score/probability tensor，实测中间 HBM bytes 至少降低 50%。只看总 allocator 高水位不足以证明没有物化。
3. decode 稳态不重传权重/KV；计时包含所有层与依赖，缓存更新正确。不同 position 引起的编译次数在固定 bucket 内为 1；需要重编译的 guard 边界单独报告。

### R40-09：有预算的自动选择、应用与扩展验证

交付：

- template/tile/stage/core count/placement 的合法候选由 capability 与 cost model 过滤，允许板上 device event 计时的受预算 autotune；禁止以 RTL simulator 的 wall time 排序。
- 保守固定配置、模型预测与实测 autotune 三组对照，缓存键包括 shape/stride/dtype、全部 compile/runtime/RTL 能力、memory map、clock 与版本。测量候选与正式运行日志区分，预算开销单列。
- 原 tiny/small/medium 全量、原 large 全量结果和预冻结 HBM 套件；至少一个 ≥4 层 decoder inference，权重占用 ≥64 MiB、prefill ≥512、连续 decode ≥64 steps，真实 Torch 模型与标准编译图。模型大小、权重来源、初始化 seed、数值 reference 与 logits 验证在 R40-00 冻结。
- ASIC 延伸设计说明：相同逻辑接口的 8/16/32 核带宽/容量/NoC 预算、预计瓶颈与新增控制器需求；8 核功能实证，其余标为分析。FPGA LUT/DSP 或 OOC 时序不能被称为 ASIC PPA。

消融：固定配置/模型/实测选择分别测同一 holdout；与 B40-F 逐项 LOO 合并成最终机制表。端到端应用提供 single/multicore、residency、locality、pipeline、attention 和 replay 各项对照。

验收：

1. `make r40-autotune`：超过资源的候选在运行前拒绝，选择不依赖 benchmark 名称；缓存命中/配置失效/worker 行为正确。每 kernel ≤16 个实测候选、每图 ≤10 分钟 tuning，超预算明确记录，预先登记的长图预算例外不能事后补填。
2. 冻结 holdout 相对保守配置 `S_geo ≥1.1`，选择 regret 的 p90 ≤15%（相对同预算合法候选中实际最优）；编译+tuning 成本和达到摊销所需调用次数完整报告。
3. tiny + small + medium 全部正确；HBM 核心套件在 R40-00 冻结后全量正确且有板上测量，至少覆盖 streaming、GEMM、transpose、gather、norm、SDPA、decoder。原 large 的所有点保留结果，未通过则 large 全套件不得认证。
4. ≥4 层模型的 prefill、decode、logits 与显式边界 copy 全部正确；给出 TTFT、decode median/p95 ms/token、tokens/s、峰值 HBM、输入/输出/权重/KV 流量。相同最终软件/同一镜像三核相对一核模型 graph makespan `S_geo ≥1.5`，decode 不退化 >5%；失败不能用 kernel-only 提速补签收。

### R40-10：板上终验、完整消融与复现包

交付：`doc/r40_report.md`、完整 gate JSON、baseline amendments、标准路径 provenance、原始样本、逐机制 A/B / LOO / 交互表、负结果、资源/时序、HBM 模型校准和运行说明。大生成物放 `sim/build/r40/`；Git 保存 spec 与可重建摘要，长期板上证据沿用 `evidence/` 惯例及校验清单。

验收：

1. `make r40-acceptance` 执行工作包的能力/正确性/性能 gate，全部必需 gate 通过才签收。单独通过设备注册、接 HBM、RTL 多核或 compile-only 均不构成 R40 完成。
2. 三核 U280 最终镜像通过 conformance、全量原套件与 HBM 核心套件；每个性能主张都可从 Torch 源和样本重建，每个独立改进都有合格消融。无贡献机制明确移除或保留为关闭的实验配置。
3. 性能主表使用完成时间秒数：同最终软件三核/一核、B39/最终系统的共同点迁移、各机制 LOO；容量扩展、资源增加、代码生成改进分列，不能相加 speedup 或用旧 E 比值冒充新系统速度。
4. 与可获得的 GPU 运行同一 Torch point/精度，沿用强 Inductor/cudagraph 与 Triton-only 对照，采用相同 request/resident/makespan 边界；不同资源与功率范围写明。商用 ASIC 公开数据只用于机制对照，未经同任务实测不填胜负速度比。
5. 从干净环境重建与板上运行至少由一次独立 session 验证；缺板卡/缺测量时相应 gate 为 `not-measured`，不以 RTL 推算吞吐替代。

## 5. 执行顺序与完成定义

```text
R40-00 冻结协议与 R39 起点
  ├─ R40-01 原生设备 → R40-02 驻留/事件/提交 → R40-03 通用 tile/资源
  └─ R40-04 单核 HBM/通用接口 → R40-05 多 SLR 多核 → R40-06 NUCA/多出口
                         两条线合流 → R40-07 流水/矩阵 → R40-08 attention/KV
                                       → R40-09 自动选择/应用 → R40-10 终验
```

接口和 correctness 可以并行推进；依赖尚未验收的优化不得提前宣称性能通过。R40-01/02 的全量驻留与 bulk DMA 验收依赖 R40-04 的外存容量和数据接口，R40-04 的标准路径性能也依赖 R40-01；先用小图与保守配置完成两者的功能闭环，再做全量性能 gate。前期先验证单核 HBM 和三核资源/时序可行性，以免软件完成后才发现 SLR 或带宽预算不成立。

每个里程碑更新本文件状态、gate 清单和 evidence registry：

- **M0：** 起点、能力与计时合同可复现，旧路径正确。
- **M1：** 原生 device + resident tensor 在标准路径通过，单核 HBM 的大 tensor 可运行。
- **M2：** 三个物理核跨三个 SLR、公共 runtime、NUCA 与多个内存出口通过；1/2/3/4/8 核功能证据齐备。
- **M3：** 宽 tile、流水/矩阵、attention、自动选择与模型的收益有独立消融，负结果没有被隐藏。
- **M4：** 最终板上全量验收、复现包和报告完成，`R40_NATIVE_DEVICE_SCALABLE_NUCA_HBM` 才能签收。

R40 的完成必须同时满足原生设备、兼容性、可扩展接口、多 SLR 物理多核、实际 HBM、标准路径应用与逐项消融。减少核数、删功能、缩 workload、改 tolerance、取消某性能 gate，都需要公开 amendment；未完成项继续标为未完成，不能通过改写“标准路径”或“多核”含义关闭。

## 6. 首轮需要避免的具体失败

| 失败信号 | 必须采取的行动 |
|---|---|
| 改成 `tnca` 后仍使用 CPU heuristics / per-kernel IO | 回到设备调度与 storage 证据，R40-01/02 不签收 |
| 三个核活跃但最终都等同一个 memory arbiter | 单/多出口消融定位，扩展 slice/port/NoC；不能以总 active cycles 证明加速 |
| HBM 标称很高，实际仍被 8 B 入口限制 | 用带宽账本识别入口，完成宽 block 传输与 outstanding；标称带宽不进收益表 |
| FPGA 适配要求固定 shape / contiguous / 降精度 | 将限制收回 board adapter 或实现通用 retile；保留原兼容性 gate |
| 多核不确定性被要求与片上 RAM 仿真逐周期相同 | 分别保留旧 B4、协议 trace replay 与 HBM 统计校准，不混用证据等级 |
| 全开有效、逐项无贡献或短 K 退化 | LOO 与交互消融，选择策略或回退；资源开销随负结果公开 |
| 接 HBM 后运行 large 需要缩原 shape 才能通过 | 保留原 large 的真实失败，新增独立命名的开发点；不能污染冻结套件 |
| 加核、cache、频率后 E 上升却完成时间更慢 | 以实际秒数和同资源对照判定；更新 bound 不能产生“免费收益” |

本文提出的是 R40 合同与实现方向。所有商用机制引用为公开资料，本地状态以代码和 R39 记录为证；新工作包、工具、阈值、板上结果均待实施和验证。
