# R40 v3：在已上板 90 MHz 镜像上交付的一次性放宽合同

版本 3，2026-10-19。状态：FROZEN CONTRACT / NOT ACCEPTED。
签收名：`R40_NATIVE_DEVICE_SCALABLE_NUCA_HBM_V3`。

依据用户 2026-10-19 的授权（“发现 v2 合同不合理处可一次性 relax 为 v3 并加速交付”；目标“在目前成功上板的 90 MHz 镜像基础上交付 R40v2”；“首先 close 掉所有不需要 bitstream 的，新的 bitstream 的和优化频率一起做”），以及用户对两项决策的明确选择（性能 gate 采用“显著收益 + 达标汇报”；DCO/locality 与 TNCA_TUNE 转入 R41）。本文与 `doc/r40_v3/plan.json` 由 `doc/r40_v3/lock.json` 固定，锁文件哈希写入 `tools/r40/contract_v3.py`。冻结在任何正式 v3 性能样本采集之前完成；之后不得再修改合同、门限、候选值、负载或基线。

## 继承

除以下七条，完整继承 v2（`doc/r40_tasks_and_acceptance_v2.md`，`spec/r40/v2/lock.json` 的哈希写入 v3 plan）及其继承的 v1 全部条款：工作包、gate、依赖、标准 Torch/Inductor/Triton 路径、descriptor、零数值回退、冻结点与子集、dtype/shape/容差、模型、B39/B40-S/B40-N/B40-CPU 基线定义、计时边界与拒绝规则（`tools/r40/ablate.py`）。所有能力（capability）gate 的判定条件不变。

## 变更一：交付镜像与频率

R40 交付镜像固定为 `a88ebdf59cfbbb2001ebd5b1b2c739c5d8cd33931f1e9202f18cac07c4c83083`（三核 SLR0/1/2、三个内存出口、4 GiB 暴露 HBM 窗口），实测运行时钟 90 MHz，90 MHz 下 WNS +0.016 ns、TNS 0、WHS 0；100 MHz 路由报告失败（WNS −1.002 ns）。R40 不声称 100 MHz。全部 gate 在此镜像上判定，R40 内不再做实现或烧录。需要新比特流的项目与下一次频率优化实现合并进行（见变更三）。

## 变更二：软件身份

交付软件身份为产品树 `files:80dd1a823e4be49e5e40d375d842c17e46c362d298e37f389c5e225621b6e4ca`（`archbench/provenance.software_tree` 覆盖的 archbench/、python/、compiler/、tools/asm/、spec/、rtl/、tb/device/）。所有板上证据必须绑定该树、该镜像与 90 MHz 时钟证明。合同、审计和采集工具位于树外（`doc/`、`tools/r40/`、`evidence/`），其修改不改变产品身份，但每条证据记录采集/审计工具哈希。若交付前必须修改产品树，受影响的全部证据按新身份重新采集。

纯 RTL/仿真证据可在 rtl/ 与所涉源文件逐字节相同的树上采集，须在审计中列出文件与哈希。v2 已审计、绑定 v2 锁且软件身份等于交付树的能力证据，在 v3 中继续有效（能力条件未变）。

## 变更三：不需要新比特流的 gate 全部在本镜像关闭；以下转出 R40

| Gate | 原因 | 去向 |
|---|---|---|
| `R40-07.matrix` | 需要新的 MMA RTL（`TNCA_RTL_MMA` 无产品实现，镜像中仅 v5 `MTX_ABUF`） | 与频率优化合并的下一次实现 |
| `R40-04.ingress` 消融 | v2 已列为实现宽入口时才必需；本镜像未实现 32 B/core/cycle | 同上 |
| `R40-06.locality` | `TNCA_LOCALITY`/`TNCA_DCO` 无标准路径消费者（TMU 硬件在镜像中，编译器未下发） | R41（用户选择） |
| `R40-09.autotune`、`R40-09.holdout` | `TNCA_TUNE` 实测选择未接入标准路径 | R41（用户选择） |

转出的 gate 在 v3 中标为 `deferred`，不是通过；R40 交付不声称这些能力。消融矩阵中未实现开关的机制同样列为“未实现、不声称”。依赖这些机制的逐点自动选择（v1 §3.2 第 5 条的“自动选择/修正”）随 TNCA_TUNE 转入 R41。

## 变更四：性能 gate 判定 = 显著收益 + 达标汇报

性能 gate 通过（`perf-pass`）当且仅当：其每个 v3 实验在交付镜像与交付软件身份上完整测量冻结子集的全部点；每点数值通过、零回退；比较被 `ablate.py` 接受（缺点、零填、混合配置、配置越界、provenance、计时口径、仅时钟差、预热混入均拒绝）；且 S_geo 的 95% 配对 bootstrap 区间下界 > 1。v2 门限保留为**目标**，逐 gate 报告“达标/未达标”与 S_geo、区间、平衡加权几何平均；未达标项连同瓶颈归因进入 R41 目标。逐点 >2%、>5% 回退逐一列出并给出原因，不作 gate 判据（自动选择随 TUNE 转出）。区间下界 ≤1 记 `no-significant-gain`，gate 不通过。

`R40-01.device-time` 仍为能力非退化 gate，判据不变（区间下界 ≥ 1/1.05、每点 ≤5% 回退）。

各性能 gate 的实验、对照值与候选值在 `doc/r40_v3/plan.json` 冻结：

| Gate | 实验（子集） | 基线 | 对照 → 候选 | v2 目标 |
|---|---|---|---|---|
| R40-02.resident | TNCA_TRANSFER_POLICY（graphs3） | B40-S | per-kernel → resident | 1.5 |
| R40-02.batched | TNCA_SUBMIT（small-kernel-graphs） | B40-S，resident | single → batched | 1.2 |
| R40-03.wide-blocks | TNCA_MAX_BLOCK（pointwise-reduction） | B40-S | 16 → 64（编译预检规则见下） | 1.3 |
| R40-03.gemm-tile | TNCA_GEMM_TILE（gemm-reuse） | B40-S | 16,16,16 → 16,64,16（同上） | 1.15 |
| R40-04.bandwidth | TNCA_RTL_OUTSTANDING（streaming） | B40-S | 1 → 8 | 1.5；方向带宽 ≥50% 瓶颈峰值为目标 |
| R40-05.scaling | TNCA_ACTIVE_CORES（compute） | B40-N | 1 → 3；1 → 2 | 2.1；1.5 |
| R40-06.streaming-scaling | TNCA_ACTIVE_CORES、TNCA_RTL_MEM_PORTS（streaming） | B40-N；B40-S | 1 → 3 核；1 → 3 出口 | 1.8；1.3 |
| R40-07.long-k | TNCA_PIPELINE（long-k） | B40-S | 1 → 2 | 1.2 |
| R40-08.prefill | TNCA_SDPA（long-prefill） | B40-S | decomposed → fused | 1.5；中间 HBM 字节 −50% 为目标 |
| R40-09.model | TNCA_ACTIVE_CORES（冻结四层模型 prefill512 与 decode 两点） | B40-N | 1 → 3 | 1.5；decode 不退化 >5% 为目标 |

`R40-02.batched` 的计时口径是 resident call latency，必须在驻留下测量，故其对照与候选都设 `TNCA_TRANSFER_POLICY=resident`，其余为 B40-S；这是口径澄清，不放宽。

宽块与 GEMM tile 候选按预先声明的**编译覆盖**规则确定（不看性能）：交付树上离线编译（禁止打开设备）若子集全部点在 64 下编译通过则取 64，否则取 32；GEMM tile 依次取 16,64,16、16,32,16。预检结果绑定在 `evidence/r40/v3/compile_preflight/`。同一子集任一候选都无法覆盖全部点时，该 gate 记 `fail`，不删点。

## 变更五：板上计时采样

板上设备周期计数是确定的：B39 重跑 77/77 逐点周期相同；同一点跨软件身份与进程重复测量相差 ≤0.2%（例如原始 GEMM4096 4,251,182,858 与 4,251,016,301 周期）；首次调用与稳态调用设备周期相差 ≤0.2%（HBM14 全部 14 点，最大为 attention_decode/large 的 0.19%）。v2 的 5 次 warmup + 30（长任务 10）次 × 3 块对设备周期口径不提供额外信息，却需要数十至数百小时板卡时间（R40-05 一组即 >50 h）。因此，对设备周期口径（`graph-makespan`、`device-graph-makespan`、`device-cycles-serial-grid`）：

1. 每点每个配置在独立进程中运行。ArchBench 的首次（冷）调用单独记录，从不进入统计；随后 0 次额外 warmup。
2. 资格测量：每个配置 1 次计时样本。若该点所有配置的资格样本都 ≤1 s，则每个配置再以独立进程采 3 次计时样本作为正式记录；否则资格记录即正式记录（1 样本）。
3. 单一测量块；同一点的各配置按冻结种子 20261010 随机排序后相邻运行。多个实验需要的相同（点、全部开关）配置只测一次、共同使用。
4. 统计：每点取正式样本中位数；S_geo 与 2000 次配对 bootstrap（对点重采样，种子 20261010）。

主机墙钟口径（`request-latency`、`resident-call-latency`）有主机噪声，保持 v2 采样：5 次 warmup、30 次计时、3 个独立块、块内随机交错。

传输看门狗为声明参数，两侧相同；长任务可提高 `TNCA_DEADLINE` 与 `TNCA_HOST_TIMEOUT`，并记录在采集身份中。

## 变更六：leave-one-out

LOO（B40-F 逐项关闭，目标子集 ≥3%）属于“达标汇报”：在板卡时间允许时测量并报告，未测量记 `not-measured`，不作 R40 v3 签收判据；完整 LOO 表进入 R41 基线。

## 变更七：R40-10 终验

`R40-10.acceptance` 要求：全部 v3 必需 gate 通过；交付报告 `doc/r40_report.md` 含逐 gate 证据、性能目标达标表、转出项与原因、资源/时序/CDC、90 MHz 例外；最终 PR 为非 Draft、目标 `main`。v1 §R40-10 第 4 项（GPU 对照）与第 5 项（干净环境独立复现）保留为必需项。

## 不变

数值容差与完整负载；全部原有点（native77、HBM14、large24、冻结模型）的正确性；三个物理核与实际出口流量；HBM 所有权与完成可见性；原始样本与 provenance；缺失或未测项不得 PASS。

`make r40-v3-contract-check` 验证冻结；`make r40-v3-acceptance` 审计 v3 证据。
