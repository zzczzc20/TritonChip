# R38 Stage 1：无 FPGA 条件下的跨层闭环、Profiler 与 ArchBench v0

版本：1.0，2026-09-22。状态：**IN PROGRESS / NOT ACCEPTED**。
上位计划：[R38 ISCA roadmap](r38_isca_roadmap.md)。起点：`r38` 分支，`3b5b28f`（R37 447/447 之上）。
签收名：`R38_STAGE1_RTL_LOOP_READY`。它**不**表示任何 FPGA、板级、能耗、NVIDIA 对比或 LLM 结论成立。

## 0. 五项目标与两阶段的对应

R38 的主线按以下顺序推进，后一项以前一项为前提：

| # | 目标 | Stage 1（小算子 + RTL 仿真） | Stage 2（FPGA 大规模仿真/实测） |
|---|---|---|---|
| 1 | **ArchBench 标准**：45 算子、几千点、不同配置下“距 roofline 上限多远”，按算子运行时间加权 | 先行：协议、下界、η/E（按参考运行时间加权的调和平均）、AQ 公式、全部反作弊自检；在现有 RTL 配置（`core-sim`、`core-vrf-sync-v1`、`core-4r` 及 elaboration 参数变体）上算出 E_c | 全量几千点 × 小/中/大 FPGA 预算档；AQ 跨规模、跨载体验证 |
| 2 | **cost model + profiler** 驱动编译器 + ISA + 硬件联合优化，提高 ArchBench 分数 | profiler v0、单核 cost model v0、一个真实三层 feature 与跨层案例消融 | 多核/全 grid 模型、FPGA 校准、Fmax/资源纳入 |
| 3 | **Agent（Codex）自动调优**：有/无 cost model 与 profiler 的优化程度与 token 成本 | 2×2 协议、隔离工作区、token/工具记账、pilot | 正式 runs（每组 ≥5）、统计结论 |
| 4 | **ArchBench 上对比 NVIDIA** | 规定 GPU 如何用同一协议计分（spec_peak、下界、权重来源），不产生对比结论 | GPU 实测 + FPGA 实测，按 §11 分层结论 |
| 5 | **PyTorch 支持、FPGA 上跑 LLM**（截图放 Appendix） | P0/P1 小图在仿真器上 | decoder → 小模型 → FPGA 演示与截图 |

目标 1 的定量含义：对配置 c、点 i，`η_{i,c} = L_{i,c} / T_{i,c}`（L 为严格的多资源 roofline 下界，T 为实测设备时间），按冻结的参考运行时间权重 `w_i` 取调和平均 `E_c = (Σ w_i / η_{i,c})^{-1}` 得到“距上限的距离”；为了消除“同一设计做大/做小”的影响，再以公共预算下界得到 AQ（roadmap §4.10）。E 与 AQ 都报告，S（绝对加速比）作为伴随指标，防止靠降低峰值刷高 E。

## 0.1 为什么分两个阶段

R38 roadmap 的全部工作都依赖“可信的时间与正确性反馈”。当前环境只有 Icarus RTL 仿真（`core-sim`、`core-vrf-sync-v1`、`core-4r`、`system-2x2`）、yosys 结构代理和小规模 TileBench 用例；没有 FPGA、没有板级时钟/外存/功率，RTL 仿真吞吐大约是每个小 case 1–2 秒墙钟。

据此按**证据来源**而不是按工作包名称切分：一项任务的必需证据如果能由“小算子 + RTL 周期 + 结构代理”完整给出，就属于 Stage 1；如果它的结论本身依赖板级频率/资源/功率、几千点 × 多配置的测量吞吐、或大容量外存，就属于 Stage 2。Stage 1 可以为 Stage 2 准备协议和工具，但**不得以 Stage 1 数据给出 Stage 2 结论**。

### 0.2 任务分类总表

| roadmap 工作包 / 章节 | Stage 1（当前 RTL 环境） | Stage 2（需要 FPGA 大规模仿真/实测） |
|---|---|---|
| R38-00 基线与证据审计（§1, §14.4） | 全部：复现 R37、证据注册表、真实 AOT/MLIR 路径 | —（Stage 2 追加 `synthesized/board-measured` 级证据） |
| R38-01 最小三层闭环（§3） | 全部：一个真实 compiler+ISA+RTL feature、兼容拒绝、回退、manifest | 在 FPGA 上重复同一闭环、测 L3/L4 反馈时延 |
| R38-02 ArchBench（§4, §4.10） | 协议、schema、单核配置 `spec_peak`、严格下界 L、E/A/S/U、附录 A 全部自检、AQ 计算器 + `AQ-RTL-canonical`（1× 预算，`NOT CERTIFIED`）、§4.10.9 第 3/5/7 条 | 小/中/大预算档（LUT/DSP/BRAM/Fmax）、1×–8× 多预算 AQ、跨载体一致性（§4.10.9 第 1/2/8 条）、`AQ-certified-domain` |
| §5 2,880 点 workload | 45 算子 manifest 与 split 规则冻结；RTL 可承受的 dev/validation 子集（约 150–300 点） | 全量 2,880 点 × 配置 × O0/O1/O2 的测量、held-out 全测、100–300 FPGA anchor |
| R38-03 Profiler（§7） | 全部 v0：设备端 counter 可由公共 runtime 读出、ROI、守恒、IR→PC 映射、RF/compute/memory 三类瓶颈定位 | 板级 DRAM controller/NoC 实测 counter、trace 开关对 P&R 的影响 |
| R38-04 Cost model（§6） | v0：单核单 program 解析模型 + 校准，未见 shape family 上的误差/排名/区间 | 多 core/全 grid、未见硬件配置、FPGA 校准、物理层（Fmax/资源） |
| R38-05 跨层案例（§8） | 案例 A（RF/operand collector）、案例 B（async descriptor 流水）在单核 RTL 上做完整消融 | 案例 C（16-tile grid/NUCA/admission）、所有案例的 Fmax/资源代价 |
| R38-06 受控研究与 Agent（§9, §10） | 协议、任务集、隔离与记账工具；2×2 pilot（每组 1 run、2 个任务） | B0–B6 正式对照、每组 ≥5 run、held-out 评测、统计结论 |
| R38-07 实测平台（§11） | — | FPGA P&R/bitstream、NVIDIA 强基线、能耗 |
| R38-08 PyTorch/LLM（§12） | P0（单 op → AOT → Session）、P1（GEMM+norm+act 小图）在仿真器上 | P2–P5：decoder block、完整小模型、FPGA 演示 |
| R38-09 论文证据（§15） | 近邻矩阵（HASCO/ACT/Chipyard/Gemmini/ArchEval）文档 | held-out 冻结、主表重建、artifact 发布 |

### 0.3 Stage 1 明确不做

- 不报告 Hz、秒、瓦、mm²、LUT/DSP 实数；yosys 数字只标 `structural-proxy`。
- 不把 `Session` 串行 grid 的周期和写成全芯片吞吐；所有 grid 结果标 `serial-grid`。
- 不做多核扩展性结论（`system-2x2` 只做功能与 counter 正确性）。
- 不在 Stage 1 冻结正式 held-out 结果；只冻结**规则**（split、权重、下界版本）。

## 1. Stage 1 最终交付

1. 一个任何人可一条命令运行的 L1 闭环：`make r38-smoke` 从未改源码的 TileBench → 公共 AOT → `.tnca` → `Session` → RTL，给出正确性 + 设备周期 + counter + 下界效率 + 一条结构化测量记录。
2. 公共 runtime 可读的设备 profiler v0，counter 满足守恒并被证明不扰动执行。
3. ArchBench v0 计分库：可审计的严格下界、E/A/S/U/AQ，附录 A 十条自检全部可执行。
4. 一个真实的三层 feature（编译器 + ISA 契约 + RTL），带能力位、兼容拒绝与回退，以及至少一个跨层案例的完整消融（含资源代理和负结果记录）。
5. 单核解析 cost model v0 及其在未见 shape family 上的误差报告。
6. 证据注册表：每项能力标注 `documented / implemented / functionally-validated / cycle-validated / synthesized / board-measured`，Stage 1 结束时不得出现 `board-measured`。

## 2. 共同规则（适用于所有工作包）

- **R1 公共路径**：所有性能与正确性数据只经 `tnca.aot.compile_aot → .tnca → tnca.runtime.Session`；不调用内部 lowering/codegen 取数。
- **R2 不回退**：任何合入都必须保持 R37 `O0/O1/O2 = 447/447`，并且在 feature 关闭时设备周期与 S1-01 基线**逐 case 相同**（快内存模型确定性）。周期变化必须由某个 `design_change` 记录解释。
- **R3 null 不是 0**：未知字段写 `null`；schema 校验拒绝用 0 填充未测量。
- **R4 证据标签**：每条测量记录带 `evidence_level ∈ {predicted, measured-rtl}`，Stage 1 不产生其他值。
- **R5 失败保留**：失败 case、失败候选、被回退的 feature 均保留记录，不从报告删除。
- **R6 生成物**：报告写到 `sim/build/r38/`（不入 Git）；Git 只保存 schema、脚本、冻结的 suite 描述与哈希。

## 3. 工作包与验收

编号 `S1-xx`。每个工作包的“验收测试”都必须是可执行命令，返回非零即不通过；“人工核对”项单独列出。

### S1-00 — 阶段划分与证据注册表

- 交付：本文档；`spec/r38/evidence_registry.json`，列出 R37/R38 每项能力（算子覆盖、公共 AOT、counter、MLIR pass、异步、INT8/FP8 路径、grid、synthesis、board 等）及其证据级别和链接制品。
- 验收测试 `tools/r38/evidence_tests.py`：
  1. 每条记录的 `level` 属于六级枚举，且链接的文件存在于仓库或被声明为 `sim/build` 生成物。
  2. `level ≥ functionally-validated` 的条目必须引用一个存在的测试命令（Makefile target 或脚本）。
  3. 任何条目为 `board-measured` 时测试失败（Stage 1 不可能有）。
  4. `synthesized` 条目必须标 `structural-proxy: true`。
  5. 变异：人为把一条改成 `board-measured` 或删除其链接文件，测试必须失败（测试自带该变异用例）。

### S1-01 — 基线复现与周期基线

- 交付：在当前 commit 重跑 R37 三档；`sim/build/r38/baseline_O{0,1,2}.json` 记录每 case 的 `serial-grid` 周期、逐 program 周期、artifact hash、source hash、profile config hash、工具版本（iverilog、triton、python）。冻结一份小型 `spec/r38/baseline_cycles.json`（case → O2 周期），作为 R2 规则的比较基准。
- 验收测试：
  1. `make r37` 得到 447/447。
  2. `tools/r38/baseline.py --check`：连续两次运行 O2 全部 149 case，周期逐 case 完全一致（确定性）；与 `spec/r38/baseline_cycles.json` 完全一致。
  3. 基线 JSON 中每条都有非空 `artifact_sha256`、`source_sha256`、`config_hash`。

### S1-02 — Profiler v0：设备 counter 经公共 runtime 可读

- 交付：
  - RTL：`tnca_shell` 增加 counter 读端口 `perf_sel/perf_rdata`（组合读，不改变任何执行路径），覆盖 `spec/r3_counters.json` 中 `cycles`、`stall_exclusive`（含 `perf_x_vrf/vrfwb`）、`stall_overlapping`、`control` 各组。
  - 传输：`spec/r2_transport_v1.json` 版本升为 1.1，新增 op 7 `PERF_READ`（a0 = counter index → r0 = 低 32 位，r1 = 高 32 位）；旧设备对 op 7 回 `REFUSED`，runtime 据此报告 `counters: unavailable` 而不是 0。
  - runtime：`Session(..., counters=True)` 时对每个 program 在 launch 前后读快照并相减，`event.counters` 给出每 program 与整 grid 汇总的差值；默认关闭。
  - `spec/r38/counters.json`：index ↔ 名称 ↔ 组 ↔ 来源模块，与 RTL 和 runtime 由同一文件驱动。
- 验收测试 `tools/r38/profiler_tests.py`（在 `core-sim` 与 `core-vrf-sync-v1` 上各跑一次）：
  1. **守恒**：对每个 program，`Σ perf_x_* == perf_stall_total`（精确相等）。
  2. **身份**：`Δperf_issued == done_retired`（launch 返回的 retired）；`Δperf_cycles` 与 `done_cycles` 的关系写成一个精确等式或 `0 ≤ Δperf_cycles − done_cycles ≤ k`，`k` 由 RTL 结构推出并写入 `counters.json`，测试断言它。
  3. **包含**：`perf_stall_total ≤ Δperf_cycles`；每个 overlapping 计数 ≤ `Δperf_cycles`。
  4. **单调**：同一 session 内快照不减。
  5. **不扰动**：R37 O2 全部 149 case 在 `counters=True` 下数值全部通过，且周期与 S1-01 基线逐 case 相同。
  6. **兼容**：用旧版（无 op 7）设备镜像运行，runtime 报 `unavailable`，不抛异常、不返回 0。
  7. **变异**：测试内置两个 RTL 变异（交换两个 counter 的 index；把一个 exclusive 计数接成常 0），守恒/身份检查必须至少一条失败。
  8. R36 导出/端口检查（`make r36-rtl-tests r36-build-tests`）在更新端口清单后通过。

### S1-03 — 测量记录与候选 manifest

- 交付：`spec/r38/measurement.schema.json`（roadmap §13.2）、`spec/r38/candidate.schema.json`（§3.1 的 `V=(C,I,H,R,W,M,P,E)` 内容哈希）、`spec/r38/design_change.schema.json`（§3.2）；`tools/r38/manifest.py` 生成与校验。
- 验收测试 `tools/r38/manifest_tests.py`：
  1. 由 S1-04 生成的每条记录通过 schema。
  2. 缺字段、单位缺失、`evidence_level` 非法、把 `null` 换成 `0` 的未测量字段（依据 `measured_fields` 列表）均被拒绝。
  3. 修改任一 RTL 文件/编译器文件/ISA 表/runtime 文件后，候选哈希相应的 H/C/I/R 分量改变，其余不变。
  4. 相同输入在两个干净工作目录生成相同候选哈希（可复现）。
  5. `score` 入口拒绝 `bounds.version` 或计时协议不匹配的记录。

### S1-04 — 性能 harness 与 smoke / dev suite

- 交付：`tools/r38/run.py`（只用公共 AOT/runtime）；`spec/r38/suites/smoke.json`（8 个覆盖不同瓶颈的点：FP16 GEMM、INT8 GEMM、softmax、layernorm、histogram、flash_attention、2d_conv、matrix_transpose）；`spec/r38/suites/dev.json`（≥ 12 个算子、≥ 150 个参数化点，按 shape family 划分 `dev/validation/heldout` = 60/20/20，family 整组留出）；`make r38-smoke`、`make r38-dev`。
- 验收测试：
  1. `make r38-smoke` 8/8 数值通过，每点产出一条合法测量记录，含周期、counter、下界（S1-05）、效率。
  2. smoke 在快内存模型上重复 3 次周期完全一致；在 `slow` 模型上报告 median 与 range。
  3. `make r38-smoke` 墙钟 ≤ 15 分钟（L1 SLO），实际时长写入报告。
  4. dev suite 数值失败 = 0；任何失败以显式状态（compile/runtime/numeric/capacity）记录，不以“慢”合并。
  5. split 检查：同一 family 的点不跨 split；held-out 点在 Stage 1 只允许被 S1-06 的最终一次评估读取（读取次数写入日志并被测试检查）。

### S1-05 — ArchBench v0 计分库

- 交付：`tools/r38/archbench/`：
  - `peaks.py`：从 `compiler/targets/*.json` 与 profile elaboration 推导 `spec_peak`（每周期 MMA MAC、SIMD lane-op、SFU op、LSU bytes、host/memory 端口 bytes），**不**从测量推导。
  - `bounds.py`：严格下界 `L = max(max_r W_r/C_r, max_ℓ Q_ℓ/B_ℓ)`，按 roadmap §4.4 覆盖 elementwise、SFU、reduction/norm、GEMM/batched GEMM、conv、copy/transpose；sort/gather/attention 在 v0 标 `bound: unavailable`（不伪造）。
  - `score.py`：η、E（调和）、A（算术）、S、U、AQ（§4.10.6）；缺失/失败按 §4.8 处理。
  - `canonical.json`：canonical primitive library v0（成本系数版本 θ0），给出 1× 预算的 logic/storage/communication 三维上限。
- 验收测试 `tools/r38/archbench_tests.py`：
  1. roadmap **附录 A 第 1–10 条**逐条为一个测试。
  2. §4.7 数值例子：A/B 两设计的 η、S、算术与调和均值与文档一致。
  3. §4.10.9 第 3 条（加资源不得在原预算档评分）、第 5 条（降自身峰值/加冗余读写/伪造大指令不提高 AQ）、第 7 条（额外 barrier、减 RF 端口的劣化配置 AQ 下降，并由 counter 指向正确原因）。
  4. 对 smoke 全部有下界的点，实测 η ≤ 1；若 η > 1 测试失败并输出审计信息（不 clip）。
  5. GEMM 下界手算（一个 16×16×16 FP16 点）与脚本一致，写入测试常量。
  6. AQ 输出标签固定为 `AQ-RTL-canonical`，且在任一注册点缺失时输出 `NOT CERTIFIED`。

### S1-06 — Cost model v0（单核、单 program）

- 交付：`tools/r38/costmodel/`：从 artifact 的机器码与 manifest 提取静态指令 mix、访存字节、依赖链；按资源解析调度 + 查表校准残差；输出预测值与 90% 区间。
- 验收（在 dev suite 的 held-out family 上，一次性评估）：
  1. median 相对误差 ≤ 20%，P90 ≤ 40%，Spearman ≥ 0.7；90% 区间实测覆盖率 ∈ [80%, 98%]。
  2. 分别报告未见 shape family 与未见 O-level 的误差，不合并。
  3. 报告建模与校准的墙钟与测量次数。
  4. 达不到第 1 条：S1-06 记为 `NOT ACCEPTED (model-limited)`，写明失配类别；不得调整门槛或重新划分 held-out。

### S1-07 — 最小三层闭环（一个真实 feature）

- 交付：依据 S1-02/S1-04 的 counter 数据选择一个 feature（候选：RF 窗口/bank-aware 布局配合 operand collector，或 async descriptor 完成事件），同时改：
  - 编译器：一个真实 pass 或 lowering 选择，记录 pass provenance；
  - ISA 契约：版本化 feature 位（`compiler/targets/*.json` capabilities + ISA 表），artifact 声明 `required_features`；
  - RTL：一个 elaboration 参数开关的实现。
  以 `design_change` 记录提交。
- 验收测试 `tools/r38/loop_tests.py`：
  1. feature 关闭：447/447，周期与基线逐 case 相同。
  2. feature 打开：447/447，至少一个 smoke 点周期下降且由 counter 差值解释（对应 stall 组下降）。
  3. 兼容：需要 feature 的 artifact 在无 feature 的 profile 上 `Session.load` 以类型化错误拒绝；无 feature 的 artifact 在新硬件上照常运行。
  4. 回退：关闭 feature（`core-sim`）时，149 个 O2 编译产物与冻结基线逐字节相同、周期相同；RTL（`HAS_TENSOR_XPOSE=0`）的结构代理与父提交的引擎一致（gates、flops 偏差 ≤ 0.5%）。
  5. 资源：yosys 结构代理（`structural-proxy`）前后对比写入 design_change。

### S1-08 — 跨层案例 A/B 的消融

- 交付：对 S1-07 的 feature（以及若时间允许，第二个 feature）做 compiler-only / hardware-only / both 三组 + 父基线，在 smoke 与 dev 的 dev/validation split 上报告 S、E、counter 归因、结构代理；按 roadmap 附录 B 写研究卡 `doc/r38_case_<id>.md`，包含至少一个无收益或负收益的探索。
- 验收：
  1. 四组全部 447/447。
  2. 交互效应用 log-speedup 差分报告，不相加。
  3. 研究卡的每个数字可由 `sim/build/r38/` 下的记录重算（`tools/r38/verify_case.py` 重算并比对）。

### S1-09 — L0/L1/L2 门禁与 DevOps 计量

- 交付：`make r38-l0`（schema/encoding/IR verifier/compile smoke）、`make r38-smoke`（L1）、`make r38-l2`（R37 三档 + profiler + archbench 自检 + dev suite）；每次运行追加一条 `sim/build/r38/devops_log.jsonl`（feedback_latency、结果、候选哈希、失败原因）；`last-known-good` 指针只在 L2 通过后更新。
- 验收：
  1. 人为引入一个数值错误的候选，L1 失败、`last-known-good` 不变、报告保留失败证据。
  2. L0 ≤ 5 分钟，L1 ≤ 15 分钟，实测值写入日志。

### S1-10 — Agent（Codex）pilot 协议与 PyTorch P0/P1（后置）

- P0/P1：固定 PyTorch 版本，`torch.export` 小图（GEMM + rmsnorm + silu）→ Triton → AOT → Session，逐元素与 CPU reference 比较，中间张量不经 host 往返；fallback 清单为空或逐个披露。
- Agent：Codex 为被测 Agent，固定模型标识与日期；冻结 2 个任务、2×2（cost model × profiler）组的隔离工作区和 token/工具/墙钟记账脚本；每组 1 个 pilot run。**只作为协议验证，不给结论。**
- NVIDIA：写出 GPU 在 ArchBench 上计分的协议（spec_peak 来源、同一下界公式、同一权重），不做测量。
- 这两项不阻塞 `R38_STAGE1_RTL_LOOP_READY`，但须在状态中标明完成与否。

### S1-11 — 近邻矩阵

- 交付：`doc/r38_related_work_matrix.md`，按 roadmap §2.1 的七个问题逐篇记录 HASCO、ACT、Chipyard、Gemmini、ArchEval、DCO/LLaMCAT，引用具体章节。只写读到的事实。

## 4. 执行顺序与关口

```text
S1-00 ── S1-01 ──┬── S1-05 ArchBench v0（目标 1，最先完成的标准）
                 ├── S1-02 profiler ── S1-03 ── S1-04 ──┬── S1-06 cost model
                 │                                       └── S1-07 ── S1-08 跨层案例
                 ├── S1-09 门禁（与 S1-04 起并行）
                 └── S1-11 近邻矩阵          S1-10 Agent / PyTorch（后置）
```
S1-05 的公式、下界与自检不依赖 counter，可以先用 S1-01 的基线周期完成；S1-04 完成后再用 smoke/dev 点重新验证 η ≤ 1。

| 关口 | 条件 | 未通过时 |
|---|---|---|
| G1a | S1-00、S1-01、S1-05（标准先行）验收 | 不开始任何性能结论或 feature 选择 |
| G1b | S1-02、S1-03、S1-04 验收；smoke 全部 η ≤ 1 | 暂停 cost model 与 feature，先修计时/下界 |
| G1c | S1-07 验收 | 不做 Agent/方法对照 |
| Stage 1 签收 | S1-00 … S1-09 验收（S1-06 可为 model-limited 并如实记录）、`doc/r38_stage1_status.md` 逐条链接证据 | 继续 Stage 1，不进入 Stage 2 结论 |

## 5. 进入 Stage 2 的交接物

- 冻结的协议：workload/measurement/candidate schema、split 规则、下界版本、canonical θ0。
- 可在 FPGA backend 上直接复用的 harness（transport op 7 的 MMIO 等价实现即可接入 counter）。
- 单核 cost model 及其已知失效范围；Stage 2 需重新校准多核/板级。
- 跨层 feature 与案例研究卡：Stage 2 补 Fmax/资源实测与大规模 held-out。
