# R39：标准路径的性能——去掉 R38 及以前的妥协

版本：1.2，2026-10-10。状态：**R39 已关闭**。签收：软件 `be87779` + 镜像 u280_r39h（v5）。R39-01、R39-02a、R39-05/06/07/08/09/10、R39-12、R39-13 都在终版中；Mode A 记为「无显著贡献」，默认关。结果与未结项见 `doc/r39_report.md` §8.4–8.5。
起点：`r39` 分支，`74f841c`（r38-dco：U280 100 MHz 不分频上板、ArchBench B4 逐点等周期）。
签收名：`R39_STANDARD_PATH_PERF`。

## 0. 范围

### 0.1 标准路径，以及只作参考的路径

从 R39 起，**所有性能结论只来自标准路径**：

```text
torch.compile(model) → Inductor（tensor descriptors 开启）→ Triton → tnca Triton backend
  → .tnca → tnca.runtime.Session → 设备（U280 板卡；或与板卡逐点等周期的 RTL 仿真）
```

手写 Triton kernel 经同一 frontend（`kernel[grid](...)` 或 `aot.compile_aot` 的 Triton 源）也属于标准路径。

以下路径**只作参考**：必须继续通过回归，但其性能数字不构成任何 R39 结论，也不作为任何 R39 改动的收益证据：

- TNCA IR builder，例如 GEMM planner（`plan_gemm`/`build_gemm`）；
- 手写汇编（`tb/core/progs`）；
- TileBench 经 AOT（R37 447）；
- 关闭 descriptor 的 Inductor 路径。

### 0.2 起点（实测）

数据来源：`74f841c`，U280 100 MHz 镜像（`evidence/r38/u280_100mhz/`），ArchBench tiny 27 点，`--counters`，profile `core-vrf-sync-v1`。下表为 27 点合计：

| 量 | 数值 |
|---|---|
| 设备周期 | 15 460 084 |
| IPC（issued / cycles） | **0.279** |
| 寄存器冒险停顿（`perf_x_hazard`） | **32.6 %** |
| LSU 停顿（`perf_x_lsu`） | 11.6 % |
| VRF bank/端口停顿（`perf_x_vrf`） | 11.3 % |
| transfer engine / FP / 矩阵单元停顿 | 1.3 % / 0.2 % / 0.0 % |
| 未归因（既未发射也不在任何 `perf_x_*`，即前端气泡与 squash） | **14.0 %** |
| 数据口利用率（读 / 写 beat 每周期） | 0.136 / 0.097 |
| 单点 η = L/T | 0.001–0.216，大多数约 0.01 |
| Inductor 配置 | `max_block = 16`，**未开 tensor descriptors** |

两点结论：

- **存储器不是瓶颈。** 数据口只用了约四分之一。
- **核心没有被喂饱。** 七成多的周期在等寄存器、LSU、VRF 端口或前端。

ArchBench 本身还没有走标准路径：descriptor 默认关闭，记录的键里也不区分开与关。

## 1. 妥协清单

证据为 2026-10-09 对仓库的审查，行号以 `74f841c` 为准。C 表示编译器与运行时，M 表示微结构，B 表示疑似缺陷。

### 1.1 编译器与运行时

| # | 妥协 | 证据 | 代价 |
|---|---|---|---|
| C1 | **没有指令调度**：按程序顺序发射，target 里的延迟（`fp_result_cycles`、`mul_result_cycles`）没有任何代码读取 | `codegen.py:5-7` "scheduling is program order"；`asm.py:9-11`；`compiler_overview.md:107` "largest known performance gap" | 相关指令背靠背，是 32.6 % 冒险停顿的主因 |
| C2 | **块宽上限 16**：Inductor `STAGE1_MAX_BLOCK = 16`，编译器拒绝跨多个向量寄存器的块 | `inductor.py:53`；`frontend.py:1488-1514` | 每个元素分摊的固定开销大；矩阵单元忙碌 ≤ 0.5 %（`compiler_overview.md:411`） |
| C3 | 规约每 program 一行（`XBLOCK = 1`），关闭 persistent reduction | `inductor.py:99-112, 186` | 规约类算子 program 数和开销成倍 |
| C4 | GEMM/conv 只有一个 16×16×16 配置；MMA 恒用 K16，K32/K64 模式从不使用 | `inductor.py:119-148`；`frontend.py:2787` | GEMM 无法分块复用，矩阵单元空闲 |
| C5 | **没有异步重叠**：每个 `t.ld.async`/`t.st.async`/MMA 发出后立即等待；`num_stages`/`num_warps` 固定为 1，不跑 TTGIR pass | `frontend.py:2569-2587, 2785-2792`；`triton_backend/compiler.py:43-45` | DMA、MMA 与计算串行 |
| C6 | K 循环每次迭代重建 TensorMap（16 次 `sstore` 加 `td.load`，约 96 周期），累加器每次 MMA 前后拷贝约 3 次 | `frontend.py:2389-2399, 2776-2783, 595-608` | GEMM 内循环开销远大于 MMA 本身 |
| C7 | descriptor 加载多数不用 transfer engine：host descriptor 从不使用，单行 box 走 LSU，≤ 16 元素的 box 走 gather | `descriptors.py:296-378` | 标准路径的数据搬运落到最慢的通道 |
| C8 | FP32 `tl.dot` 不进矩阵单元（逐 k 的 `vextract`+`splat`+`vfma`）；计算出来的 f16 操作数经内存重打包（`v.pack` 保留未实现） | `frontend.py:2807-2834, 2271-2315`；`r37_isa_gaps.md:123-130` | FP32 GEMM 与融合 GEMM 走慢路径 |
| C9 | 寄存器分配是 linear scan，不感知 bank，不能 spill；窗口、累加器、barrier 按使用点静态分配、整个 kernel 不释放；tile 过早溢出到 scratch 并加 `fence` | `asm.py:49-58`；`codegen.py:121-147`；`frontend.py:3063-3070` | 11.3 % 的 VRF 停顿；寄存器压力迫使更小的块 |
| C10 | 没有循环展开、LICM、FMA 收缩 | `frontend.py:569-625`；`upstream/passes.py:11-13`；`compiler.py:58-61` | 循环开销与依赖链无法摊薄 |
| C11 | **只用 1 个 role context**；grid 逐 program 串行，每个 program 一次主机 LAUNCH | `ir.py:692`；`aot.py:828-829`；`runtime.py:736-750`（`role_mask=1`） | 硬件第二个 context 闲置；设计初衷的 4 context 延迟隐藏完全没有 |
| C12 | 每次启动都整块上传、下载参数，没有设备驻留张量 | `triton_backend/driver.py:9-16`；`architecture_overview.md:388` | 墙钟由主机往返主导（不影响设备周期，影响端到端） |

### 1.2 微结构

| # | 妥协 | 证据 | 代价 |
|---|---|---|---|
| M1 | **role context 只有 2 个且没有独立指令队列**。设计初衷是 4 个：R0 地址、预取与搬入，R1 矩阵，R2 规约与收尾，R3 搬出；每个 context 有自己的队列和 ready 调度。现实现中一个 context 的停顿冻结整个前端 | `tnca_core.v:55, 320-332, 1689`；`doc/design/tnca_v1_architecture_draft.md:91-122, 385, 445-466` | 无法用其他 context 的指令填补停顿 |
| M2 | **banked VRF 没有旁路**（`AVAIL_V` 恒假）：单周期向量生产者到消费者约 3 个气泡，FP 约 5 个；即使互不相关的指令也要在 R 级停 2 拍 | `tnca_core.v:866-872, 617-672`；`r3_status.md:55`（+16 % / +13 %） | 冒险停顿的硬件一半 |
| M3 | VRF 只有 4 个 bank，各 1 读 1 写，5 个请求者；已写好的 operand collector（4 项）没有接进核心；矩阵和 transfer engine 读口与取操作数抢 bank | `tnca_core.v:781-786`；`tnca_operand_collector.v` 只在 tb 中实例化；`spec/r3v2_contract.json:46-50` 声称 4×3 | 11.3 % VRF 停顿 |
| M4 | **LSU 同一时刻只允许一条指令**（`req_ready = (state == S_IDLE)`）。固定开销 S_SETUP、S_DRAIN 加一次注册握手，背靠背访存约 9 拍；store 也要等响应 | `tnca_lsu.v:30-33, 151, 422-466`；`conformance.md` D2 未关闭 | 11.6 % LSU 停顿 |
| M5 | 数据口 8 B/周期，由 LSU 与 transfer engine 共用，engine 绝对优先（`pick_b = b_req_valid`） | `tnca_mem_mux.v:69`；`tnca_core.v:81-82` | 搬运期间 LSU 饿死 |
| M6 | 单发射、写回口一个结果每周期、8 深预约；`wb_steal`、`stall_rsv` 冻结前端 | `tnca_core.v:1517-1523, 1676` | 结果密集时互相阻塞 |
| M7 | 前端：同步取指、一项 fetch buffer、predict-not-taken、X 级解析，taken 分支约 4 个气泡 | `tnca_core.v:1722-1735`；`conformance.md` D1（约 +8 %） | 14 % 未归因周期中的主要部分；该部分目前没有 counter |
| M8 | 矩阵单元 k 为外层，每个 k 先装 B 再算 16 拍，K16 约 272 拍；banked VRF 上每次端口 3 读再加 1 拍（实测 +42–46 %）；同一时刻只有一条命令 | `tnca_matrix.v:391-427`；`r3_status.md:55` | 即使编译器喂饱，单位吞吐也只有一半 |
| M9 | transfer engine 同一时刻只有一条命令，`td.load`、`w.def` 也会停；`t.prefetch` 被丢弃；descriptor 读取串行（32 字） | `tnca_tensor.v:288, 605-606`；`tnca_core.v:1335-1340, 1638-1640` | descriptor 流水无法重叠 |
| M10 | FP 规约顺序进行（4 级 × (FP_LAT+1)），期间普通 FP 指令停；SFU 每周期一个 lane | `tnca_vector_fpu.v:441-475`；`tnca_sfu.v:11-18`；`tnca_core.v:1601, 1662` | softmax、norm、激活类慢 |

### 1.3 疑似缺陷（先于一切性能工作确认）

| # | 内容 | 证据 |
|---|---|---|
| B1 | banked VRF 拒绝写（`vrf_wb_freeze`）的那一拍，scoreboard 仍可能被清除，后续读到旧值 | `tnca_core.v:2084-2088` 无 freeze 条件（审查推断，未经测试） |
| B2 | VRF 仲裁的 `rr_ptr` 只有 2 位，请求者却有 5 个，端口 4（transfer engine store 读）永远不会排在第一 | `tnca_vrf_banked.v:204`；`tnca_core.v:786` |
| B3 | 文档过期：`architecture_overview.md:406` 仍写板上镜像只有 rank 1–2；`r3_counters.json` 的优先级缺 vrf、vrfwb | 同左 |

## 2. 共同规则

- **R1 标准路径。** 性能数字只来自 §0.1 的标准路径，由 ArchBench 产生；参考路径的数字在报告中单独成表，标 `reference-path`。
- **R2 每项改动有开关，开关关闭时与之前逐周期相同。**
  - 编译器改动：compile option 或 Inductor 选项，例如 `schedule=0/1`、`max_block`、`num_stages`。
  - 硬件改动：RTL 参数加 profile 能力位，例如 `VRF_BYPASS`、`NUM_RCTX`、`LSU_INFLIGHT`。
  - 验收要求：开关关闭时，ArchBench tiny 每点设备周期与上一个冻结基线**逐点相同**，R2 冻结基线 149/149 相同。
  - 开关打开后的每一处周期变化，都由一条 `design_change` 记录（`spec/r38/design_change.schema.json`）解释。
- **R3 消融协议。** 每个工作包的收益只以一次 A/B 消融给出：`tools/r39/ablate.py`，套件为 ArchBench tiny + small，走标准路径。报告内容：
  - 逐点 Δ周期、η；
  - 套件 E（balanced 权重）；
  - 几何平均加速比 S_geo；
  - 停顿归因的变化（每个 `perf_x_*` 与前端气泡占比）；
  - 正确性。

  通过条件：
  - 达到该工作包写明的阈值；
  - 任何点慢于基线 2 % 以上时必须有逐点解释，没有解释则不通过；
  - 数值失败为 0。
- **R4 测量载体。**
  - 编译器和运行时改动在 U280 上测（tiny 约 6 分钟，small 约 8 分钟）。
  - 硬件改动先在 RTL 仿真上测 tiny（`fpga-u50-sim`，确定性，与板卡逐点等周期），再由最终镜像在板上确认。
  - 每个板上镜像都必须重做 B4（板卡与同一 RTL 仿真逐点等周期），才能用它的数字。
- **R5 时钟与资源。**
  - 硬件改动不得使板卡默认频率（100 MHz 不分频）失去时序收敛；先用 OOC 综合估计，最终以链接结果为准。
  - 每项硬件改动报告 LUT、FF、BRAM、DSP 的增减。
  - 若某项必须降频，报告「同频收益」和「按实际频率换算的秒级收益」两者。
- **R6 正确性闸门（每次合入）。**
  - 参考路径：`make r37` 447/447；core、core-vrf、core-slowmem、core-ooo、top、nuca；`r38-baseline-check`。
  - 标准路径：torch 24/24（descriptor 开、关两种模式）；tdesc 10/10；ArchBench 数值全部通过；板上 B1 conformance。
- **R7 冻结基线。** R39-00 冻结标准路径基线。此后每个通过验收的工作包更新一次冻结基线，并附 amendment，说明来自哪个工作包。
- **R8 沿用 R38。** null 不是 0；失败保留在报告中；生成物写到 `sim/build/r39/`，不入 Git。

## 3. 工作包与验收

编号 `R39-xx`。验收测试都是可执行命令，返回非零即不通过。阈值是**下限**；估计依据写在括号中。

### R39-00：标准路径默认化、消融工具、缺陷确认、冻结基线

交付：

1. **标准路径默认化。**
   - ArchBench 默认开启 tensor descriptors：`configure(tensor_descriptors=True)`，用 `--no-tensor-descriptors` 关闭。
   - `platform.inductor` 完整记录 Inductor 配置，开关显式写 `true`/`false`。
   - 记录键（`report.load`）加入 Inductor 配置哈希和 RTL/profile 配置哈希，使不同配置不会互相覆盖。
2. **消融工具。** `tools/r39/ablate.py`：输入两组 ArchBench JSONL（或在给定开关下各运行一次），输出 R3 规定的逐点表、E、S_geo、停顿归因差异和回退点清单；`--gate` 时按阈值返回非零。
3. **前端气泡计数。** 新增 counter `perf_x_frontend`，把「未发射也未停顿」的周期按取指空、squash、分支重定向归因，使 `issued + Σ perf_x_* = cycles` 精确成立。
4. **疑似缺陷。** B1 先写定向测试复现（或证伪），再修复；B2 修正指针位宽并测试 5 口公平性；B3 更新文档。
5. **冻结基线。** 在 U280 100 MHz 镜像（含 B1/B2 修复和新 counter 的重建镜像）上运行标准路径的 ArchBench tiny + small，冻结为 `spec/r39/baseline_archbench.json`（逐点周期、η、counter、配置哈希）。

验收测试：

1. `tools/r39/ablate_tests.py`：
   - 在一个点上人为加 5 % 周期，工具必须标出回退并在 `--gate` 下失败；
   - 同 profile 不同 Inductor 配置的两组记录不合并；
   - 两份相同输入的 S_geo 恰为 1.0。
2. 守恒：`profiler_tests.py` 扩展后，每个 program 都满足 `perf_issued + perf_stall_total + perf_x_frontend == perf_cycles`，在 core-sim 和 core-vrf-sync-v1 上各跑一次。
3. B1：定向测试 `t38_wbfreeze` 在修复前失败（若复现）、修复后通过；若无法复现，测试保留，结论写入文档。B2：`t29_vrfbank` 扩展的 5 口测试证明每个请求者都会被服务。
4. 基线：tiny + small 在板上数值全部通过；tiny 的 B4 逐点等周期；基线文件记录 commit、镜像 sha256、配置哈希。
5. 标准路径基线与 §0.2（无 descriptor）的对照表写入报告。这一步只是测量，不设阈值。

进展（2026-10-09）：

- 交付 1–4 已完成（`7121da0`、`db59522`）。
- **B4（新，读 RTL 时发现，已修复）：banked VRF 的写回冻结会丢结果。** transfer engine 抢到 bank 时写回级冻结并重试，但 FP、MUL 流水线不受停顿约束，冻结那一拍落下的结果被丢掉：寄存器保留旧值，scoreboard 位永不清除，下一个读者永远等待。
  - 复现：`t39_freeze_drop`，banked 核在 3 次 `wb_retry` 后挂死。
  - 修复：写回优先拿 bank（`tnca_vrf_banked` 注释本来就这么写）；transfer engine 被拒时保持该写、下拍重发（`tnca_tensor vw_ready`），有写被保持时不报告传输完成。
  - 验证：t39 通过；core、core-vrf、core-slowmem、core-ooo 各 35/35，xpose 3/3，count、fairness 测试通过。B1 由 `t38_wbfreeze` 复现后修复；B2 由 `tools/r39/vrf_fairness_test.py` 证明 4 口、5 口每个请求者都会被服务；B3 文档已更新。
- **标准路径第一次全量运行失败 8 点**（nd100c 镜像，descriptor 开）。原因有三类，都已修复：
  1. **标量指针未下降。** Inductor 在参数的切片上建 descriptor，写作 `tl.make_tensor_descriptor(p + k, ...)`；导入器只在 load 的指针链里处理 `p + k`，没有把它当作值。影响 gpt2_block、llama_prefill（tiny、small）。修复：标量 `tt.addptr` 在被当作值使用时才算出地址（`frontend._scalar_ptr_value`），因此只经由 load 使用的指针不多出指令，制品同一性 447/447 不变。
  2. **一个 range tree 被拆成两维。** Inductor 把 `y // 4`、`y % 4` 匹配成 descriptor 的两维，块形状为 `[(YBLOCK+3)//4, XBLOCK, Min(4, YBLOCK)]`；kernel 因此要把 tile 重排成 4 维再置换，导入器没有这种下降。即使有，搬运的行也只有 1/4 或 1/2 个向量宽。影响 conv2d、resnet_block。修复：`TNCATMACompatibilityChecker` 拒绝这种访问，Inductor 退回指针访问。
  3. **数值错误（llama_decode、llama_prefill）。** rank-1 descriptor 读出的权重经 `[16] → [16, 1] → [1, 16]` 重新标号。`tt.trans` 把它当作改名处理，tile 的表示仍是「每行一个值」，于是后续运算每个 lane 都读到第 0 个元素。反方向 `[1, 16] → [16, 1] → [16]` 再接 descriptor store 也是同一个问题。修复：`frontend._relabel` 在「每行一个值」与「一行」之间转换表示；从同一个寄存器拆出来的列，直接还原为该寄存器，不付代价。新增定向测试 `rank1_relabel_in_loop`、`rank1_store_relabel`，tdesc 12/12。
- **定位手段**，留作调试开关，默认关闭：
  - `TNCA_TDESC_BISECT=lo:hi`：按 Inductor 询问的顺序给 descriptor 访问编号，只对区间内的访问使用 descriptor，二分到出错的那一个。本次定位到第 69 个访问。
  - `TNCA_GUARD_BYTES` / `TNCA_GUARD_CHECK`：给每个缓冲区两侧加毒化边缘。前者检出越界写；后者用两种毒值各跑一次，检出越界读。本次排除了越界读写。
- **counter 回归。** 加入 `perf_x_frontend` 后，旧镜像对第 20 号 counter 回答 refused。`Session._snapshot` 因此丢掉了全部 counter，R39 分支上的 ArchBench 记录都没有 counter。修复：只跳过设备没有的那一项。
- **结果（nd100c 镜像，正式基线待 r39b 镜像）：**
  - 标准路径 tiny + small **52/52 数值通过**，全部带 counter。
  - tiny：IPC 0.277。停顿占比：hazard 32.1 %，LSU 12.6 %，VRF 11.2 %，未归因 13.9 %。
  - small：IPC 0.264。停顿占比：hazard 35.3 %，LSU 12.4 %，VRF 10.0 %，未归因 13.3 %。
  - 与 §0.2（descriptor 关）逐点对照，tiny 的 S_geo 为 0.998（0.925–1.068）。`max_block = 16` 时多数 kernel 只有一个向量宽，descriptor 与普通 load 几乎没有差别；差别要到 R39-02 的宽块才出现。

- **基线冻结（u280_r39b 镜像，2026-10-09）。**
  - 镜像：RTL `db59522`，B1/B2/B3 已修，R39 开关全关；xclbin sha256 `807d4014…48e2`；实测时钟 100.092 / 99.919 MHz。
  - 软件：`0b7ef29` 的代码树，`TNCA_SCHEDULE=0 TNCA_INDEX_OPT=0 TNCA_NO_X_DIM=0`，即 `d4ee950` 的标准路径。
  - tiny + small **52/52 数值通过**。tiny IPC 0.277，hazard 32.1 %，LSU 12.6 %，VRF 11.2 %，前端 13.9 %；small IPC 0.264，hazard 35.2 %，LSU 12.4 %，VRF 10.0 %，前端 13.3 %。
  - **B4 逐点等周期：27/27。** 板上周期与板卡 shell 的 RTL 仿真（u50sim 后端，同一软件、同一开关）逐点相同。
  - 冻结为 `spec/r39/baseline_archbench.json`（镜像、软件、配置哈希、逐点周期、η、counter、B4 结论）和 `spec/r39/baseline_archbench.jsonl`（52 条原始记录，作为 `tools/r39/ablate.py` 的 A 侧）。
- **R39-00 验收 1–5 全部满足，R39-00 关闭。**

### R39-01：指令调度（编译器）

交付：

- 基本块内的 list scheduler，用于标准路径的全部 kernel。延迟与资源模型从 target 读取，同一份模型给 cost model 用：
  - FP、MUL 延迟；
  - load-to-use 延迟；
  - banked VRF 读延迟和 bank 冲突代价（`bankcost.py`）；
  - 异步 MMA、DMA；
  - 写回口预约。
- 调度约束：遵守 effect、barrier 和内存依赖；在寄存器预算内工作（与 R39-04 联合）。
- 开关：compile option `schedule=0/1`，默认 1。

验收：

1. 开关关闭时，ArchBench tiny 逐点周期与 R39-00 基线相同；R2 冻结基线 149/149 相同。
2. R6 正确性闸门全部通过。
3. 消融（板上，tiny + small）：
   - 套件 `perf_x_hazard` 周期相对下降 ≥ 30 %；
   - S_geo ≥ **1.15×**；
   - 无未解释的回退点。

   估计依据：冒险停顿占 32.6 %，其中软件可重排的部分预计一半以上；硬件旁路属于 R39-06。

进展（2026-10-09）：

- **实现。**
  - `compiler/python/tnca_cc/schedule.py`：基本块内的 list scheduler。依赖规则：
    - 纯运算只受数据依赖约束；
    - 访存读可以越过纯运算、也可以相互越过，但不越过任何带 effect 的运算；
    - effect 之间保持原序；
    - `vinsert`、`assign` 视为原地写；
    - 窗口别名的读者不越过它后面的下一个 effect；
    - label、branch、exit 是块边界。
  - 选择规则：最早可发射优先，其次关键路径，再次原序。
  - 寄存器压力按 allocator 的活跃区间分别跟踪 vector、scalar、pred 三个寄存器堆（含跨 loop 的延长），上限为原序峰值加余量；候选限定在原序 48 条之内。超过上限的块退回原序。
  - `aot._emit` 依次尝试：默认余量与窗口 → 零余量、窗口 16 → 原序，并在 provenance 中记录回退。
  - 调度时顺带删去无人读的参数块 load：参数块由 runtime 为本次 launch 写入，读它不会 fault。
- **机器模型实测**，写在 `spec/r39_sched_model.json`：
  - 核心延迟来自 `tools/r39/latency_probe.py`（banked VRF 上）。整数向量依赖距离 4 拍，FP 6 拍，SFU 29 拍，标量 ALU 2 拍，标量 MUL 5 拍。
  - **banked VRF 上连互不相关的向量指令也只能每 2 拍发射一条**（`vrf` 停顿 1 拍；同 bank 再加 1 拍）。这是 R39-07 的硬件问题，调度无法消除。
  - 存储器代价来自 `tools/r39/load_latency.py`，在板上测得：标量 load 11 拍，64 字节向量 load 16–20 拍；LSU 一次只容一条访存。
- **合法性。** `tools/r39/schedule_tests.py` 用独立规则检查每个重排过的块：894 次编译，约 1 万个块，0 违规，模型估计没有一个变差。
- **消融**（nd100c 镜像，tiny + small，`TNCA_SCHEDULE=0/1`），52/52 数值通过：
  - S_geo **1.158**（tiny 1.164，small 1.151）≥ 1.15 ✓；
  - 套件 `perf_x_hazard` 周期降为 **0.633×（−37 %）** ≥ 30 % ✓；
  - 无回退点（最慢 op.silu/small 1.000，最快 op.rope/tiny 1.520）✓；
  - 套件总周期 0.931×。LSU 停顿周期随之升至 1.27×：计算被压紧以后，单发射的 LSU 成了瓶颈，属于 R39-08。
  - 第一版只跟踪向量压力，GEMM kernel 因标量寄存器不足退回原序，S_geo 1.138，未达标；修正后才通过。
- **调度暴露出的潜伏缺陷（已修复）。** 掩码运算接到了非 predicate 的操作数，codegen 把它打印成同号的 predicate 寄存器，读到什么算什么。原序下碰巧对，换一种分配就错：
  - Inductor 的 `xmask ^ true`：`true` 被物化成向量 `v.movi -1`。`index_assert_traps` 在调度后不再 trap。
  - argmax 中标量 0/1 进入 `p.xor`。
  - 注意力 kernel 中跨循环的 mask tile 以向量作为 `v.sel` 的条件。

  修复：`frontend._as_pred` 把向量、标量转成 predicate；循环携带的 mask 放在 predicate 寄存器中（`p.mov`）。并加了两道检查：`Builder.vselect` 与 codegen 的 `p.*`、`v.sel` 和执行掩码都拒绝非 predicate 操作数，同类错误以后编译时就报出。参考路径的制品不受影响（447/447 相同）。
- **默认值。** 标准路径（Triton/Inductor 插件）默认 `schedule=1`；`compile_aot`（参考路径）默认仍为 0，R2 冻结基线不变。`plugin_identity` 把插件的选择显式传给 AOT，两条入口生成同一份代码。
- **发现（交给后续工作包）：** Inductor 的 `[1, R]` 规约 kernel 代码量是计算本身的十倍以上：
  - 索引 tile 用 `iota*1 + splat` 逐次物化，且在循环内重复计算；
  - 连续 load 退化成 gather，并经 scratch 存取、加 fence；
  - 一行的 tile 经由累加器搬运。

  见 R39-02 进展。

### R39-02：宽块与多寄存器 tile（编译器 + Inductor）；R39-02a：代码形状

交付：

- 编译器支持跨多个向量寄存器的块：
  - 一维块 ≤ 1024 元素；
  - 二维 tile R×C，C 是 16 的倍数；
  - 规约 `XBLOCK > 1` 和 persistent reduction；
  - 宽 tile 上的 mask、尾部和 broadcast。
- Inductor 的 `max_block` 由 16 放开为可配置（默认按寄存器预算选，例如 128），每类 kernel 提供少量候选配置。
- GEMM/conv 模板提供 32×32、64×64 等配置，由多个 MMA fragment 拼成。
- 开关：`configure(max_block=16)` 恢复旧行为。

验收：

1. `max_block=16` 时与基线逐点相同。
2. torch 24/24（两种模式）。新增宽块定向测试：一维 1000 元素、二维 37×100、行宽 256 的规约，覆盖尾部与 mask，数值与 eager 一致。
3. 消融：
   - pointwise 和 reduction 类点 S_geo ≥ **1.5×**；
   - 全套件 S_geo ≥ **1.3×**。

   估计依据：每 program 的固定开销（参数加载、地址计算、规约树）按元素数摊薄。

进展（2026-10-09）：先做 **R39-02a**，再做宽块。

量了 R39-01 之后的代码才发现，比宽块更大的妥协在代码形状本身：
- 静态指令里绝大部分是索引和地址的物化，真正的运算只占很小一部分；
- Inductor 的 `[1, R]` 规约逐次物化索引 tile，并经 gather、scratch、fence 搬运；
- GEMM 模板每个 K 步约 2700 条指令。

所以把 R39-02 拆成两部分：
- **02a**：代码形状，已完成，下面是消融；
- **02b**：宽块与 program 合并，交付和验收仍按上文。

R39-02a 的改动都在两个开关之下，关闭时制品逐字节相同（447/447）：

- **`TNCA_NO_X_DIM`（Inductor）。** 循环规约去掉单位 x 维，生成一维 `[R0_BLOCK]` 块（`tnca.inductor._install_no_x_dim`）。导入器为此补上了以下支持：
  - `tensor<1>` 到 `tensor<N>` 的 broadcast；
  - 循环携带的 mask（放在 predicate 寄存器里）。
- **compile option `index_opt`（`TNCA_INDEX_OPT`）。**
  - 标量值上的 lane-uniform 格：`64*xindex + r0_index` 仍是连续访问。
  - 由 `assume(x >= 0)` 和乘以常数得出非负、倍数事实：`(u + arange) % M` 在 u 是块长倍数时仍连续。矩阵模板的 `rm % M` 因此不再走 gather。
  - 连续索引的 `expand_dims` 是仿射项。
  - 标量除法改在标量寄存器上做：常数除数用倒数乘法，除数为 1 走快路径，取代 32 轮向量恢复除法。
  - 指针 tile 的 TensorMap 在 kernel 入口只建一次，tile 原点作为坐标传入，不再每个 K 步重建 256 字节。
  - `acc + tl.dot(a, b, 0)` 直接在 acc 的累加器上累加；循环携带值已就位时 yield 不再复制。
  - `acc.to(f16)` 后接 store 时，逐行在出口转换，不再整块经 scratch 落地再读回。
  - 仿射行的各部分只算一次；相同的比较只做一次。
- **效果（模拟器上逐 kernel 计）。**
  - RMSNorm 4×64：14 332 → 4 668 拍（3.1×）。
  - 16×32×16 matmul：20 539 → 2 602 拍（7.9×），K 循环里只剩两次传输、一次 MMA。
  - 52 个点的静态指令：pointwise −50 %，规约 −38 %，模板 −56 %。
- **消融**（nd100c 镜像，tiny + small；A 为 R39-01 的默认配置，B 再加 `TNCA_NO_X_DIM=1 TNCA_INDEX_OPT=1`），52/52 数值通过：
  - S_geo **2.114**（tiny 2.129，small 2.098），套件总周期 0.409×；
  - E：tiny 0.0070 → 0.0126，small 0.0100 → 0.0199；
  - 无回退点。
- **仍待处理**，按收益排序：
  - f32 `tl.dot`（注意力里的 P@V、fp32 卷积）走向量 FMA，而不是矩阵单元；
  - 卷积模板仍有 gather；
  - 宽块、program 合并（02b）。

### R39-03：异步重叠与软件流水（编译器）

交付：

- K 循环和 attention 循环做软件流水：
  - `num_stages` 生效（≥ 2）：下一块的 `t.ld.async` 在当前块计算前发出，等待推迟到使用点；
  - MMA 异步发出，在需要结果时才等待。
- TensorMap 在循环外构建一次（LICM），循环内只改坐标。
- 累加器跨迭代原地保留，不再拷贝。
- descriptor 加载的通道选择以 transfer engine 为首选：
  - host descriptor 也可以用 engine；
  - 单行与小 box 由代价模型在 LSU、gather、engine 之间选择。
- FP32 `tl.dot`：给出能用矩阵单元的方案，例如 TF32/拆分为 bf16 的近似必须是选项、默认不改变 IEEE 结果；做不到的，在文档中说明原因。
- 开关：`num_stages=1` 恢复旧行为。

验收：

1. `num_stages=1` 时逐点相同；tdesc 10/10 和 torch 24/24 通过。
2. 新 counter 证明重叠真实发生：transfer engine 忙与发射同时成立的周期数、矩阵单元忙与发射同时成立的周期数。它们不加入 exclusive 组，只用作证明。
3. 消融：
   - GEMM、attention、linear、模型类点 S_geo ≥ **1.5×**；
   - 矩阵单元忙碌率 ≥ 10 %（基线 ≤ 0.5 %）。

### R39-04：寄存器分配（编译器）

交付：

- 按活跃区间复用窗口、累加器、barrier、descriptor 槽位。
- bank 感知分配：同一指令的源尽量分属不同 bank（`bankcost.py` 接入分配器）。
- 寄存器不足时 spill 到 scratch，而不是拒绝编译。
- tile 只有在真正需要时才溢出到 scratch，并去掉多余的 `fence`。
- 开关：`alloc=linear`。

验收：

1. 关闭时逐点相同；正确性闸门通过；R2 中原先因寄存器不足而拒绝的情形，现在能编译并数值正确（定向测试）。
2. 消融：套件 `perf_x_vrf` 相对下降 ≥ 40 %，S_geo ≥ **1.05×**。

### R39-05：4 个 role context 与 program 并发（硬件 + 运行时 + 编译器）

交付：

- **硬件。** 板上 profile 改为 `NUM_RCTX = 4`，每个 context 64 个向量寄存器，VRF 共 16 KiB，BRAM 实现。每个 context 有自己的指令槽和 ready 判断：一个 context 停顿时，前端从其他就绪的 context 发射，停顿不再冻结整个前端。
- **运行时与编译器（Mode B）。**
  - 一次 LAUNCH 用 `role_mask = 0xF` 同时启动 4 个 context；
  - 编译器把 kernel 体包成「pid 从 rctx 开始、步长 4、直到 grid 大小」的循环（persistent program）；
  - grid 在设备上并发执行，不再每个 program 一次主机往返；
  - 计时协议由 `serial-grid` 改为 `grid-concurrent`：一次 launch 从接受到所有 context 空闲的周期。
- **设计初衷的 Mode A**（一个 program 按角色拆分为生产者、消费者）列为可选的后续项，不进本轮验收。
- 开关：`NUM_RCTX` 参数为 2 或 4；运行时 `concurrency=1/2/4`。`concurrency=1` 与旧行为逐点相同。

验收：

1. 新定向测试：
   - 4 个 context 同时运行不同 program，结果正确；
   - 一个 context 长时间停顿（例如 LSU 长访存）时，其他 context 的发射不受影响，由 per-context issued counter 证明。
2. core、core-vrf 和全部回归在 `NUM_RCTX = 4` 上通过；conformance 通过。
3. 消融（RTL 仿真 tiny，加板上终版确认）：
   - grid ≥ 4 个 program 的点 S_geo ≥ **1.6×**；
   - 全套件 S_geo ≥ **1.3×**；
   - 套件 IPC ≥ 0.45。
4. R5：100 MHz 时序收敛，资源增量单列。

进展（2026-10-09）：

- **Mode B（编译器 + 运行时）。** `tnca_cc.rctx_spmd`，compile option `rctx_spmd`（`TNCA_RCTX_SPMD`）。消融通过后，标准路径默认开启；`compile_aot` 默认关。
  - kernel 体包成循环：context r 依次运行 program r、r + R、……（R 为 target 的 context 数），直到 `num_programs(0)`。`program_id(0)` 的值就是循环变量，循环尾是 4 条指令。
  - 参数 load、常量、其他轴的 pid 在循环外算一次。只提升标量；被 `assign` 原地写的值不提升（循环计数器的初值若被提升，第二个 program 的循环会从第一个结束处开始——ArchBench 的 cumsum、norm、softmax 先发现了这个错误，已修复，定向测试 `ragged_trip_counts` 覆盖）。
  - manifest：`execution.role_mask = 2^R − 1`，`program_to_role` 每个 context 一项，`entry_points` 每个 context 一项。运行时对每个 (y, z) 发一次 launch，启动 min(R, gx) 个 context；scratch 按 context 数放大。
- **RSLICE（硬件 capability `rctx_slice`）。** launch 的 role-mask 参数第 31 位要求按 context 切分 per-core 资源：指令立即数和 barrier token 里的资源编号在 D→R 和执行单元入口处 OR 上 context 号，context r 使用 accumulator、barrier、descriptor、window 各自的第 r 片。
  - 编译器保证每类编号小于片大小，OR 因而幂等：`mb.arm` 返回的 token 已经切过，再用一次结果不变。
  - 用到这些资源或 scratch 的 kernel 只在有 `rctx_slice` 的 target 上才包成 Mode B；scratch 的每次 load（导入器在循环之后会再 load 一次）都加上 ROLE_ID × scratch_bytes。
  - 旧镜像只取 role-mask 的低位，第 31 位被忽略；编译器不会为它们生成依赖切分的代码。
  - **切片数由 launch 决定。** role-mask 的第 30:29 位是 log2(切片数)；0 表示每个 context 一片。在 4 context 的核上，一个 kernel 可以只用 2 个 context，各分到一半资源。
    - 原因：v4 上按四份切，每个 context 只有 4 个 accumulator。主导模型点的 GEMM 模板要用 6–8 个，于是被拒绝 SPMD。同时，被接受的 GEMM 类 kernel 在 4 个 context 上反而比在 v3 的 2 个 context 上慢：gpt2_block 785k 对 667k 拍，mlp 468k 对 390k，attention_decode 13.0k 对 11.4k。四个 context 共用一个矩阵单元和一个 transfer engine。
    - 编译器的选择：取 2 的幂中能放下这个 kernel 资源的最大 context 数；持有任何 per-core 资源（accumulator、descriptor、window、barrier 或 scratch）的 kernel 最多 2 个（`TNCA_RCTX_SPMD_SHARED`，默认 2）。
    - 先只限制用矩阵单元的 kernel，attention 的规约 kernel 在 v4 上仍比 v3 慢（7141 对 6155 拍，3 个 32 个 program 的 kernel 都是这样）。它们经 TensorMap 读数据，不用 accumulator：4 个 context 各建自己的 TensorMap，共用同一条存储器通路。不持有资源的 kernel 仍用全部 4 个 context，比如 add 快 1.14×。
    - `t39_slice` 在 4 context 台架上用 `+slicek=1`（2 个 context 各一半）和默认的四份切分都通过。
  - 入口块里由不变量建成的 TensorMap（Inductor 的都是由参数建成的，tile 的位置在每次 load 时给出）也只建一次：写进本 context scratch 的 32 个 store，以及所有 store 都已移出循环后的 snapshot。只有 snapshot 会读 TensorMap，所以移动这些 store 不改变任何可观察的顺序。实测 `tdesc_rows`（5 个 program）：SPMD 开但每个 program 都重建 TensorMap 时为 0.99×，提到循环外后为 1.62×；`dot_grid` 为 1.17×，此后主要停顿在矩阵单元（R39-10）。
  - 拒绝按 kernel 起点计数 barrier epoch 的 token（`barrier_token`，只有软件流水的降低会生成）：同一 context 上的第二个 program 中，barrier 的 epoch 是接着第一个往下数的。标准路径不生成这种 token。
- **RCTX_SWITCH（硬件）。** R 级指令因寄存器 hazard 停顿、D 级是另一个 context 的指令时，丢弃 R 级指令（它还没有读、写或占用任何东西），让前端前进。该 context 从 `r_pc` 重新取指，并 park 到有写回到它的寄存器（或 accumulator 完成）为止。
  - 只对 hazard，不对 LSU 忙：t39_switch 上对 LSU 忙也让路反而慢 8 %，因为 LSU 本身就是瓶颈，被丢弃的访存在 LSU 空出后 3 拍才回到 R。
  - 计数：让路的那一拍计入 `perf_x_hazard`，`issued + stall + frontend = cycles` 仍精确成立；`perf_rctx_switch` 记让路次数（仿真可读）。
- **4 个 context。** profile `core-vrf-sync-v4`：`NUM_RCTX = 4`，每个 context 64 个向量寄存器，VRF 共 256 个（16 KiB）。
- **profile。** `core-vrf-sync-v3` = v2 + `RCTX_SWITCH` + `LSU_INFLIGHT=2` + `FPU_OVERLAP` + `RSLICE`（2 个 context）；`core-vrf-sync-v4` = v3 + 4 个 context。两者都只有仿真，板卡镜像待建。单个时序开关可以用 `TNCA_RTL_PARAMS="RCTX_SWITCH=0,..."` 在器件仿真上关掉做消融（进入缓存键，ArchBench 用 `--set` 记录）。
- **测试。**
  - `t39_switch`（新）：所有 context 同时跑指针追踪、MUL 链、SFU 链，以及 SFU pass 期间发出的普通 FP 与规约，结果与 role 相关、逐项断言。core、core-vrf、r39hw、r39sw、r39sw4 上都通过。
  - `t39_isolate`（新，验收 1 的第二条）：context 0 跑 24 个相互依赖的 SFU pass，其余 context 跑不碰 SFU、不碰存储器的整数与向量循环；`+ctxtrace` 打印每个 context 的退出拍和退休指令数。2 个 context 时，计算的那个 context 在没有 RCTX_SWITCH 时第 1794 拍退出，有时第 1392 拍。
  - `t39_slice`（新，`gen_t39_slice.py`）：每个 context 用同样的名字 `d1`、`w1`、`b0`、`b1`、`a1` 走 TensorMap → transfer engine → window → barrier → MMA → `a.vread`，数据按 context 不同、结果逐 context 断言。r39sw 上 2 个、r39sw4 上 2/3/4 个 context 都通过；没有 RSLICE 的台架上只启动 1 个 context。
  - `tools/r39/rctx_spmd_tests.py`（新）：奇数 grid、单 program、二维 grid、内层循环、每个 program 不同 trip count、cumsum、masked softmax；以及三个持有资源的 kernel（tl.dot、tensor descriptor、dot tile 网格），它们在有 `rctx_slice` 的 target 上必须被接受并数值正确，在其他 target 上必须被拒绝。
  - R3.7 全套（447 个，`TNCA_RCTX_SPMD=1`，与 R39-01/02a 的开关一起）：**447/447**，135/135 个算子 × 优化级别通过。
  - tensor descriptor 测试（`TNCA_RCTX_SPMD=1`，v3）：12/12。
  - `core-r39sw`（v3 全部开关）37 个程序 × 两种存储器模型全部通过。
- **消融（v2 RTL 仿真 tiny，13 个不持有资源的点，A 为 v2、SPMD 关）：**

  | B | S_geo | hazard 占比 | FPU 占比 |
  |---|---|---|---|
  | v2 + SPMD | 1.223 | 16.9 % → 4.2 % | 2.1 % → 5.6 % |
  | v3（只开 RCTX_SWITCH）+ SPMD | 1.226 | → 6.4 % | → 4.5 % |
  | v3（RCTX_SWITCH + LSU_INFLIGHT + FPU_OVERLAP）+ SPMD | 1.278 | → 8.3 % | → 1.8 % |

  v3 在 SPMD 关时与 v2 逐点同周期（S_geo 恰为 1.0000）：单 context 不受影响。SFU 密集的点（gelu、silu、swiglu）只开 RCTX_SWITCH 时比 v2 + SPMD 慢：普通 FP 在 SFU pass 期间整体停顿，被丢弃的指令又要多等一次重新取指。`FPU_OVERLAP` 去掉这个停顿后，这些点回到 1.29–1.45×。
- **时序与资源（Vivado OOC 综合后，整个器件 `tnca_fpga_top`，10 ns）：**

  | 配置 | WNS | 最差路径 | CLB LUT | 寄存器 | BRAM | DSP |
  |---|---|---|---|---|---|---|
  | v2（2 context） | 3.177 ns | FPU FMA 操作数选择，6.805 ns | 195 579 | 53 759 | 167.5 | 171 |
  | v4（4 context + RCTX_SWITCH + LSU_INFLIGHT=2 + FPU_OVERLAP + RSLICE） | 3.139 ns | 同一条路径，6.843 ns | 215 202（+10 %） | 64 146（+19 %） | 167.5 | 171 |

  新逻辑没有进入最差路径。256 个向量寄存器仍放在同样数量的 BRAM 里。板卡镜像 `u280_r39d`（v4，100 MHz）已开始链接。
- **未完成：** v4 与持有资源的 kernel 的全套件消融；板卡确认（B4 逐点等周期）；per-context issued counter 的硬件导出（仿真里用 `+ctxtrace` 打印每个 context 的退出拍和退休指令数）。

### R39-06：banked VRF 的旁路网络（硬件）

交付：

- X→R、W→R 的向量结果旁路，使相关指令不必等待写入寄存器文件。
- FP、MUL 延迟结果在可用的那一拍即可被消费，scoreboard 提前释放。
- 消除互不相关的指令在 R 级的 2 拍等待：读在 D 级发出，或者操作数预取。
- 开关：RTL 参数 `VRF_BYPASS = 0/1`。

验收：

1. 新定向测试 `t39_deplat`，在 banked 文件上测相关指令对的气泡：
   - 单周期向量生产者到消费者 = **0**（基线约 3）；
   - FP 生产者到消费者 = **FP_LAT − 1**（基线约 5）；
   - 测试同时在关闭 bypass 时断言旧值，证明测的是对的东西。
2. 回归全部通过；B1 的测试在 bypass 开启时仍然通过。
3. 消融：套件 `perf_x_hazard` 相对下降 ≥ 40 %（在 R39-01 之后测），S_geo ≥ **1.15×**。
4. R5：100 MHz 收敛。

进展（2026-10-09）：

- RTL 开关 `VRF_BYPASS`（正在写回的源经 bank 的 write-first 路径读出，不算 hazard）和 `VRF_PREFETCH`（R 不用读口时，D 级指令提前一拍读向量源），组成 profile `core-vrf-sync-v2` 的一部分，板卡镜像 `u280_r39c` 正在链接。
- core 台架实测（`tools/r39/latency_probe.py`）：互不相关的向量指令 1 拍/条（原 2），相关的 `v.add` 3 拍（原 4），`vf` 5 拍（原 6）。调度器的机器模型新增 `banked2`。
- `core-r39hw`（v2）37 个程序 × 两种存储器模型通过。
- **套件消融（v1 → v2，三个开关一起，RTL 仿真 tiny，软件为 R39-01 + R39-02a 默认）：** 25/27 点（两个 llama 点仍在跑），S_geo **1.158**，最慢的点 1.016×，没有点变慢。
  - 停顿占比：VRF 10.1 % → 5.1 %（绝对周期 −56 %，属于 R39-07 的指标，`VRF_PREFETCH` 达成）；前端 17.0 % → 10.7 %（绝对 −45 %，R39-09 的指标，`FE_NOBUBBLE` 达成）。
  - hazard 的绝对周期只降 4 %，**未达到 R39-06 的 −40 %**。bypass 只消除了「正在写回」的那一拍；剩下的 hazard 来自 load、SFU 和规约这类长延迟结果，由 R39-05 的多 context 处理，见 R39-05。
  - 获益最大的是 cumsum 1.68×、embedding 1.49×、matmul fp32 1.35×、argmax 1.27×。
- **板上确认（u280_r39c，v2，100 MHz，路由后 WNS +0.073 ns；xclbin sha256 `8d38ca27…5cd4`）。** 软件为当前默认（R39-01 调度、R39-02a 代码形状开，SPMD 与 sparse TensorMap 关）。
  - B1 conformance 通过；tiny 27/27 数值通过。
  - 相对冻结基线（`spec/r39/baseline_archbench.jsonl`，r39b 镜像，全部开关关）：tiny **S_geo 2.872**，E 0.00614 → 0.01522（2.48×），总周期 0.371×，最慢的点 1.148×，没有点变慢。
  - small 25/25 通过，S_geo 2.807。tiny + small 合计 52/52：**S_geo 2.840**，E 0.00725 → 0.01866（**2.57×**），总周期 0.334×；提升最少的是 silu small，1.086×。实测时钟 100.038 / 99.967 MHz。
  - 板上周期与 v2 的器件级 RTL 仿真（sim-file）逐点相同（例如 gpt2_block 899716、swiglu 3392），所以本轮的仿真消融数字就是板上数字。
  - 这个镜像在 B5 修复之前构建，含该缺陷；1–2 个 context 的运行没有触发。

### R39-07：VRF 端口与 bank 冲突（硬件）

交付：

- 把已有的 operand collector（4 项）接入核心，或者把 bank 数增加到 8、给 bank 增加读口。由综合代价决定，两条路线的数据都要记录。
- 矩阵和 transfer engine 读口与取操作数之间采用带饥饿保护的优先级。
- 开关：`VRF_BANKS`、`OPERAND_COLLECTOR`。

验收：

1. 定向测试：同一 bank 的三操作数指令由 3 拍降到 ≤ 2 拍；矩阵读不再让 MMA 速率减半（K16 MMA 周期回到接近 behavioural 文件的值，误差 ≤ 10 %）。
2. 回归通过。
3. 消融：`perf_x_vrf` 相对下降 ≥ 50 %，S_geo ≥ **1.08×**。
4. R5：100 MHz 收敛。

进展（2026-10-10）：

- **B5（新，R39-05 的 4 context 测试中发现，已修复）：banked VRF 读仲裁会让矩阵端口永远拿不到 bank。** 现象是 v4 上 3 个 program 的 dot tile 网格挂死：矩阵单元停在 MAC 状态，每拍请求一次读，却一直没有被接受；等它的 context 睡在 barrier 上，另两个 context 堵在 R、D 两级。
  - 原因：round robin 指针移到「本拍最后一个被服务的端口」之后。D 级预取（端口 1，与矩阵的端口 3 同一个 bank）和端口 0（另一个 bank）每拍都请求；按 1 2 3 4 0 的顺序服务 1 和 0 之后，指针又回到 1，于是端口 3 永远排在端口 1 后面。
  - 修复：指针移到本拍第一个被服务的端口之后。按顺序排第一的请求者一定拿到它的 bank，所以每个持续请求的端口在 NUM_PORTS 拍内必被服务。
  - 测试：`tools/r39/vrf_fairness_test.py` 新增 case 8（两个 bank 同一拍被服务），旧仲裁失败（端口 3 在 24 拍里只被服务 1 次），新仲裁通过（最长等 3 拍）。
  - 这是 v2 起（VRF_PREFETCH）就潜在的缺陷，只是 1–2 个 context 时 bank 的组合没有触发。u280_r39c（v2）镜像按旧 RTL 构建；v4 镜像 u280_r39d 已按修复后的 RTL 重新链接；仿真消融全部用修复后的 RTL 重跑。

### R39-08：非阻塞 LSU 与存储器端口（硬件）

交付：

- LSU 允许多条访存指令在途（深度 ≥ 4），完成经 scoreboard 回写；去掉 S_SETUP、S_DRAIN 中不必要的拍。
- store 为 posted write：顺序和 fence 语义不变，fence 等所有在途 store 完成。
- LSU 与 transfer engine 之间公平仲裁，不再让 engine 绝对优先。
- 评估数据口加宽（128 位）或分口的代价与收益，取舍需记录。
- 开关：`LSU_INFLIGHT = 1/4`、`MEM_ARB`。

验收：

1. 定向测试：
   - 背靠背独立 load 的间隔 ≤ 2 拍（基线约 9）；
   - load-to-use 延迟 ≤ 4 拍；
   - store 之后的 fence 正确；
   - 原子操作、fault、`t23_amofault` 不变；
   - 三种存储器模型都通过。
2. 消融：`perf_x_lsu` 相对下降 ≥ 50 %，S_geo ≥ **1.1×**。
3. R5：100 MHz 收敛。

进展（2026-10-09）：

- RTL 开关 `LSU_INFLIGHT = 2`（`rtl/core/tnca_lsu_pair.v`）：两个 `tnca_lsu` 共用一个存储器口。另一个单元在等最后的响应（drain）时，空闲单元就接受下一条 load 或 store。
  - 第二个单元只在第一个单元发出最后一个 beat 之后才被接受，所以前一个访问的请求总是先到达端口。本 profile 的存储器按到达顺序执行请求，只把响应延迟或打乱（板上 `tnca_fpga_mem` 是单个数据口；`tnca_mem_model` 在接受时执行请求），因此任何非原子访问之间的重叠都与程序顺序等价。
  - 原子操作仍等两个单元都空闲，`fence` 的含义不变（等 `req_ready`，即两个单元都空闲）。
  - 起初只让 load 重叠；放开 store 之后，store 序列（建 TensorMap）不再每条都等写回应。`tdesc_rows` 4565 → 3355 拍，`dot_grid` 10428 → 8712 拍（都是单 context）。
  - 若存储器会把不同地址的请求乱序执行（多 slice 的 NUCA），这里需要地址检查；带这个开关的 profile 都没有这样的存储器。
  - beat tag 的最高位是单元号；两个完成同时等待时，较早的先出。
- `t39_switch` 在 v3 台架上（存储器 1 拍返回）2266 → 2106 拍（与 FPU_OVERLAP 合计）；单 program 的 vector add 79 → 67 拍。
- `core-r39sw`（含此开关）全部通过。套件消融随 v3 一起报告（见 R39-05）。
- **软件侧：TensorMap 只存非零字。** compile option `sparse_tensormaps`（`TNCA_SPARSE_TENSORMAPS`）。消融通过后，标准路径默认开启。kernel 在 scratch 里建 TensorMap 时，每个字都要一条 `s.st.d`，而每条 store 占用 LSU 一整次访问（setup、beat、等写回应、完成）；32 个字里秩 2 的只有 7 个非零。运行时现在把 scratch 随参数块一起清零上传，于是零字不必再存。
  - 结果：tensor descriptor 测试 12/12；`tdesc_rows` 4565 → 2510 拍（SPMD 关），与 SPMD 合用为 1958 拍，相对两者都关 2.33×；`dot_grid` 10428 → 7848 拍，合用为 7201 拍（1.45×）。
  - 套件消融随 v4 一起报告（见 R39-05）。
- 未做：深度 4（需要更宽的端口 tag），store 的 posted write，与 transfer engine 的公平仲裁。

### R39-09：前端（硬件）

交付：

- 用 R39-00 的 `perf_x_frontend` 先把 14 % 的前端周期按原因分解。
- 按分解结果实施：
  - 加深 fetch buffer；
  - 简单的分支预测（向后跳转预测 taken，或 BTB）；
  - 缩短 taken 分支的重定向路径。
- 开关：`FETCH_DEPTH`、`BPRED`。

验收：

1. 定向测试：紧循环每次迭代的分支代价 ≤ 1 拍（基线约 4）。
2. 消融：`perf_x_frontend` 相对下降 ≥ 40 %，S_geo ≥ **1.05×**。
3. R5：100 MHz 收敛。

进展（2026-10-09）：

- RTL 开关 `FE_NOBUBBLE`：解码器本拍取走缓冲字时不再阻止取指，停顿不再额外损失一个取指槽。属于 profile `core-vrf-sync-v2`。
- **分支预测不做，数据如下。** v2 的 RTL 仿真 tiny 全套件中，taken 分支的重定向只有 3570 次，共 249 万拍。每次约 3 拍，合计约 0.4 %：热点 kernel 展开后几乎是直线代码。前端剩下的 10.7 % 主要是 `mb.wait` 睡眠和停顿后的取指空泡；前者由多 context 填补（R39-05）。

### R39-10：矩阵与规约单元的吞吐（硬件，按收益决定是否实施）

交付：

- 矩阵单元：去掉 banked 文件上的端口 3 额外一拍，与 R39-07 合并实现；B 装载与 MAC 重叠；允许下一条 MMA 在当前 MMA 收尾时开始。
- FP 规约：树规约流水化，不阻塞普通 FP 指令。
- 每个子项只有在 R39-02/03 之后的 counter 显示它确实是瓶颈时才实施。实施的必须给出消融；不实施的必须给出数据说明为什么不做。

验收：实施的子项，其目标点（GEMM 类或 softmax/norm 类）S_geo ≥ 1.2×；不实施的子项需有数据支持。

进展（2026-10-09）：

- RTL 开关 `FPU_OVERLAP`：SFU 有独立的数据通路并在 start 时锁存源操作数，只有规约借用 FMA 阵列。因此普通向量 FP 只在规约折叠时停顿，SFU pass 也不再等待在途的普通 FP。原来 SFU pass 的约 22 拍内，任何 context 的普通 FP 都不能发射。
- `t39_switch` 增加 E 段：SFU pass 期间发出普通 FP 与一次规约，结果逐 lane 断言。
- 与 SPMD 一起时，SFU 密集的点 FPU 停顿占比从 4.5 % 降到 1.8 %（见 R39-05）。

### R39-11：总消融与签收

交付：`doc/r39_report.md`，内容包括：

- **累计表。** 从基线起按工作包顺序逐项打开，记录每一步的 S_geo 和 E。
- **leave-one-out 表。** 全部打开，然后每次只关一项，量出每一项在全套件里的边际贡献。
- **板上确认。** 最终镜像的 ArchBench tiny + small、B4 逐点等周期、实测时钟。
- **代价。** 资源与频率代价，以及参考路径的数字（单独标注）。

验收：

1. 最终镜像在 U280 上满足：
   - ArchBench tiny + small 数值全部通过；
   - S_geo 相对 R39-00 基线 ≥ **3.0×**；
   - 设备时间 E ≥ 3× 基线；
   - 没有任何点慢于基线。
2. leave-one-out：每个已实施的工作包关闭时，套件周期增加 ≥ 3 %（否则在报告中降级为「无显著贡献」）。
3. 板卡 100 MHz 时序收敛，B4 逐点等周期，实测时钟 100 ± 0.5 MHz。
4. 全部 R6 闸门通过；每项周期变化都有 `design_change` 记录。

进展（2026-10-10，板上终版）：

- **镜像 u280_r39d**：core-vrf-sync-v4，4 个 context，开关 RCTX_SWITCH、RSLICE（可变切片数）、LSU_INFLIGHT=2（load、store 都可重叠）、FPU_OVERLAP，以及 v2 的三个开关和 B5 修复。100 MHz，路由后 ap_clk WNS +0.087 ns；整个平台 CLB LUT 25.6 %、BRAM 18.2 %。xclbin sha256 `335e48a2…24ce8`，实测时钟 100.017 / 99.992 MHz。
- **软件**：标准路径默认，即 R39-01 调度、R39-02a 代码形状、R39-05 SPMD、sparse TensorMap。
- B1 conformance 通过；ArchBench tiny 27/27、small 25/25 数值通过。
- **相对冻结基线（tiny + small，52 点）**：

  | | S_geo | E | 总周期 | 最慢的点 |
  |---|---|---|---|---|
  | tiny | 4.305 | 0.00614 → 0.02087（3.40×） | 0.251× | — |
  | small | 4.325 | 0.00884 → 0.03356（3.80×） | 0.222× | — |
  | **合计** | **4.315** | **0.00725 → 0.02576（3.55×）** | **0.224×** | embedding tiny 1.974× |

  `ablate.py --gate --min-speedup 3.0 --max-regress 1.0` 通过：S_geo ≥ 3.0，没有点变慢。IPC 0.265 → 0.411。
- 相对 v2 镜像（u280_r39c），也就是 R39-05/08/10 加 sparse TensorMap 的板上增益：S_geo 1.519，总周期 0.669×；提升最少的是 matmul fp32 tiny，1.095×。
- 停顿占比（tiny，基线 → 终版）：hazard 32.1 % → 16.4 %，VRF 11.2 % → 4.0 %，前端 13.9 % → 4.3 %。LSU 从 12.6 % 升到 22.0 %，成为最大的一项（绝对周期 0.44×）。
- **终版镜像 u280_r39e**：
  - 内容：u280_r39d 加上「launch 时 round robin 归零」。
  - 时序与时钟：100 MHz，路由后 ap_clk WNS +0.053 ns；实测时钟 100.003 / 99.990 MHz；xclbin sha256 `6a3e76a4…4b18cf`。
  - 在板上：B1 通过；tiny 27/27、small 25/25 数值通过。
  - B4：整套连跑 26/27 逐拍相同。不同的一点是 attention_decode：板上 10252，仿真 10251。
  - 在同一镜像上单独重跑两次、紧跟前一点 argmax 重跑，板上都是 10251。这 1 拍不能稳定复现，也不来自前一个 kernel，原因记为未结项。

### R39-12：Mode A——用 partition.py 让访存与计算在多个 role 间重叠（硬件 + 编译器 + 运行时）

背景：R39-05 的 Mode B 让多个 program 并发，但每个 program 内部仍是串行的。标准路径 GEMM 模板的 K 循环每一步依次是：load A 并等待；用 16 个 gather 把 B（转置的权重）搬到 scratch 再重排；建 TensorMap；load B 并等待；MMA 并等待。访存与计算从不重叠。

交付：

- **编译器。** 在导入器产出的 IR 上做自动 context 特化（`tnca_cc.partition`：classify → graph → partition → map_to_rctx），把带 MMA 的计数循环拆成两个 role：
  - **producer**：TRANSFER 分区。A、B 的 load、gather 与重排、TensorMap、`dma_load` 写入 window。
  - **consumer**：MATRIX 与 EPILOGUE。MMA、accumulator 读出、写回。
  - 两者之间是一个 2 级 window 环。每级有一对 full/empty barrier。各 role 用自己的迭代号算 epoch（sec 7.3），不交换 token。
  - producer 在 consumer 做第 k 步 MMA 时准备第 k+1 步。
  - 两个 role 各自复制循环控制（trip count 相同）。不在循环里的部分：地址计算在需要它的 role 里重算；accumulator 初始化与 epilogue 归 consumer。
- **寄存器。** 一个 program 的两个 role 共用 64 个逻辑向量寄存器：window 在顶部，producer 和 consumer 的私有寄存器分别在不相交的区间里。codegen 的分配器按 role 给出区间。
- **硬件（launch 的 pair 模式）。**
  - role-mask 的第 28 位置位时，context 2k 与 2k+1 组成一对：它们的 vbase 相同（都是 2k 的），RSLICE 也按对（rctx >> 1）切片。
  - 这样一个 program 的两个 role 看到同一组 window、barrier、descriptor 和 accumulator，而不同 program 之间仍然隔离。
  - 不置位时行为不变。
- **与 Mode B 组合。** v4 上 2 个 program × 2 个 role。每一对按 R39-05 的方式循环执行一行 grid 的 program；一对里的两个 role 迭代同样的 program 序列。
- **开关。** compile option `rctx_roles`（`TNCA_RCTX_ROLES`），以及硬件参数 `RPAIR`。
- **B 操作数用一次转置传输（同属本工作包）。**
  - linear 的 B 是 W 的转置，列方向连续。原来每个 K 步要 16 次 gather，再经 scratch 重排：32 次 store、一次 fence、16 次 load、建一个 TensorMap、做一次传输。
  - 这些只有在每步 producer 只剩两次传输时才值得去重叠。R38 S1-07 已有传输引擎的转置写（ISA sec 9E mod[5]，`HAS_TENSOR_XPOSE`），但 v4 没有打开它。
  - 现在导入器把这种 tile 当成它的转置来读：转置后是一个普通 tile，引擎转置后写入 window。限于只给 `tl.dot` 用的 16 位 tile。
- **`mb.arrive` 精确化（`RPAIR`）。**
  - 原来 arrive 要等所有引擎空闲，所以 consumer 释放一个 stage 时，要等 producer 已经发出的下一批传输做完。这正好把两个 role 串行化了。
  - 有 `RPAIR` 时，arrive 只等本 context 发出的矩阵和传输命令。ISA sec 9A.5 允许这样做。
- **新 profile core-vrf-sync-v5** = v4 + `HAS_TENSOR_XPOSE` + `RPAIR`，target `tnca-sim-vrf-v5`，capability `tensor_transpose`、`rctx_pair`。

验收：

1. 定向测试：
   - core 级：两个 role 经 window 环交接 K 步，逐元素正确。
   - 4 context：2 对同时运行不同 program，结果正确。
   - 对 Triton GEMM（`dot_grid`、K ≥ 2 步）数值正确，并由 `+ctxtrace` 证明 producer 与 consumer 的执行在时间上重叠。
2. 全部 core 回归、R3.7、torch 测试、`rctx_spmd_tests` 通过；`compile_aot` 默认关时制品不变（plugin identity）。
3. 消融（RTL 仿真 tiny + 新的 medium 档）：GEMM 类点（linear、gemv、matmul、bmm、mlp、gpt2_block、llama_*）的 S_geo ≥ **1.2×**（相对 R39-05 终版），全套件不慢于终版。
4. R5：100 MHz 收敛，板上终版确认，B4 逐点等周期。

进展（2026-10-10，RTL 仿真；板上镜像 u280_r39f 在链接中）：

- **硬件**（f9297a8）：
  - `RPAIR`，以及 RPAIR 下精确的 `mb.arrive`。
  - 缺陷 B6：software 构造的 barrier token 在 `mb.wait`/`mb.test` 上没有按 slice 改写。B6 的表现和发现经过见下面「定向测试」一条。
  - profile core-vrf-sync-v5。
- **编译器**（644482e）：
  - `tnca_cc.partition.specialize_loop`：在导入器的 IR 上跑上游的三步（PartitionScheduling → aref → PartitionLoops）。
  - `rctx_spmd` 的 `pair` 模式。
  - `codegen.emit_roles`。
  - B 操作数的转置传输。
  - masked 连续行的 epilogue。
  - 延迟到唯一使用者处计算的 tile。
- **定向测试**：
  - `t39_pair`（core 级，4 context、2 对）在快、慢两种存储器上通过。
  - 在乱序存储器上，它首先暴露了 B6：pair 1 的 consumer 在 pair 0 的 barrier 上醒来，提前做了 MMA。
  - `tools/r39/rctx_roles_tests.py`：
    - v5：6 个多步 GEMM 循环都被特化，结果正确，比只用 Mode B 快 1.008–1.132×。
    - 单步的 GEMM 被 Triton 折掉了循环，按预期被拒绝。
    - v4：全部被拒绝，结果正确。
- **回归**：core-r39sw5 84/84，R3.7 447/447，torch 24/24，plugin identity O0/O1/O2 149/149，rctx_spmd_tests v4/v5 10/10。
- **消融（tiny，RTL 仿真，相对 v4 终版）**：
  - v5 软硬件包（转置传输 + epilogue）：GEMM 类 S_geo 1.53。各点：linear 1.99、gemv 2.22、mlp 2.53、gpt2_block 1.89、llama_decode 1.57、llama_prefill 1.44。全套件 1.17，没有点变慢。
  - Mode A 再叠加在上面时，在 tiny 上不划算：linear 0.955、gemv 0.948、mlp 0.969、llama_decode 0.974，只有 matmul 1.084。
  - 原因：tiny 的 GEMM 只有 2–4 个 K 步。Mode B 的两个 program 已经互相掩盖了传输与 MMA，而 Mode A 让两个 role 各算一遍地址，在单发射的核上多占了发射槽。
  - Mode A 的价值要在 K 步多的 medium 档和板上量。
- **试过、未采用**：`ACC_RD2`，让 `a.vread` 走累加器文件的第二读口，不必等别的 context 的 MMA。
  - linear 1.065，mlp 1.02，gemv 0.97，几何平均约 1.01。
  - 不值得为它重新链接镜像，已回退。

进展（2026-10-10 下午，板上）：

- **镜像 u280_r39f**（v5，不含 MTX_ABUF 与 RCTX_SWITCH=2）：
  - 时序与时钟：100 MHz，WNS +0.101 ns，实测 100.002 / 99.947 MHz。
  - B1 通过；tiny 27/27、small 25/25、medium 25/25，Mode A 开、关都正确。
  - **B4 27/27 逐拍相同。**
  - v5 相对 v4 终版：77 点 S_geo 1.187（总周期 0.745×），没有点变慢。相对冻结基线 5.119。
- **Mode A 的消融**：
  - 板上 r39f：77 点 S_geo 0.997。只有单 program 的 matmul tiny 快 1.084；mlp、matmul small/medium、linear tiny 慢 2–5 %。
  - 仿真（tiny，v5 终版）：1.006。
  - 按 R39-11 的规则记为「无显著贡献」，**默认关**。
  - 原因：GEMM 受矩阵单元约束，Mode B 已经让两个 program 的传输与 MMA 互相掩盖；Mode A 两个 role 重复的地址计算在单发射核上占了发射槽。
  - Mode A 在 program 少、或 producer 重（gemv）时有收益，作为经测试的选项保留。
- **验收 3（GEMM 类 S_geo ≥ 1.2，相对 R39-05 终版）由 v5 整包达成**：
  - 仿真 tiny 上 GEMM 类 1.91（9 点），全套件 1.287，没有点变慢。
  - 贡献来自转置传输、epilogue 改造、`m.zero`、`MTX_ABUF`（MMA 551 → 311 拍）和 `RCTX_SWITCH=2`，**不是来自 Mode A**。
- **Mode A 的正确性修复**：
  - bmm 模板把指针 tile 经 scratch 在循环里携带，Mode A 在 medium 档上算错。
  - 现在循环前的写由两个 role 各做一份；一个 role 可能读到另一个 role 写的内存时不拆分。
  - `rctx_roles_tests` 加了这种情形的用例。
- **终版镜像 u280_r39h**（v5，含 `MTX_ABUF`、`RCTX_SWITCH=2`）在链接中；板上结果出来后补上。

最终状态（2026-10-10 晚，板上终版 u280_r39h）：

- 时序与时钟：100 MHz，WNS +0.088 ns，实测 100.022 / 99.989 MHz；LUT 336 334。
- B1 通过；Mode A 开、关：tiny 27/27、small 25/25、medium 25/25。
- **验收 3**：v5 终版相对 v4 终版（R39-05 终版）：
  - GEMM 类 27 点（tiny + small + medium）S_geo **1.880**。
  - 全套件 77 点 1.287（总周期 0.679×），没有点变慢。
  - 相对冻结基线 5.548。
- **Mode A**（带 `ring_estimate` 门控）：77 点 S_geo 1.0045，没有点变慢，套件周期 −0.37 %。
  - 变了的 9 点都变快：gemv 1.06–1.11，llama_decode 1.01–1.05，llama_prefill 1.000–1.009。
  - 其余 68 点逐拍相同。
  - 仍记为「无显著贡献」，**默认关**。
- **验收 4（B4）**：整套连跑 26/27。llama_decode 板上 545164，仿真 545162。
  - 单独跑两次是 545162。紧跟 gpt2_block 跑可复现 545164：是 gpt2_block 留下的跨 launch 状态。
  - 差别在一个 kernel（`..._neg_split_transpose_view_3`，grid 2）：161 → 162 拍。不是 buffer 周围的内存；单会话仿真复现不了。证据见 `doc/r39_report.md` §8.4。
  - 记为未结项，冻结进 B39（R40-00）。

### R39-13：评测负载适度扩大（ArchBench medium 档）

交付：

- 在 small 与 large 之间加 `medium` 档，覆盖全部 family（op 与 model）。
  - 约束是每个 kernel 的参数（输入、输出、权重）放得进器件 heap：512 KiB 器件存储器中的 384 KiB。runtime 每次 launch 只上传这个 kernel 的参数。
  - 在这个约束下，规模取 small 的 2–8 倍。gemv 的 small 已经占了 256 KiB 权重，只能小幅增大。
- 规模写进 `archbench`（`ops.py`、`models.py`、`registry.TIERS`），冻结 `suites/medium.json`；`ablate.py` 支持按档统计。
- 板上跑 medium 全档；仿真只用于定位，不作为 medium 的常规手段。

验收：

1. medium 全档在终版镜像上数值全部通过。
2. 冻结 R39-00 基线在 medium 档上的数字：在 r39b 镜像上，用基线软件跑一遍。
3. R39-11 的板上结论在 tiny + small + medium 上复核：S_geo、E、无点变慢。

进展（2026-10-10，板上）：

- **medium 档**（5a7eab4、2d51e3c）：25 点，24 个 family，每个 kernel 的最大一次 launch 的参数不超过 352 KiB。
  - runtime 每次 launch 还需要约 13 KB：每个 context 一份 scratch、TensorMap、对齐。
  - 第一次上板时 gpt2_block 因此超出 heap，transpose 正好用满 heap。两者都已缩小后重新冻结。
- **冻结基线的 medium 数字**（验收 2）：
  - 测法：u280_r39b 镜像，0b7ef29 软件，开关全关，profile core-vrf-sync-v1。
  - 结果：25/25 数值通过，记录在 `spec/r39/baseline_archbench_medium.jsonl`。
- **v4 终版在 medium 档上**（验收 3 的一部分）：u280_r39e，R39-05 终版软件，25/25 通过。相对基线：
  - S_geo **4.309**；E 0.00751 → 0.02669（3.55×）。
  - 没有点变慢，最少的是 embedding，1.983×。
  - 发射占比 0.277 → 0.515。
  - 这与 tiny + small 的 4.315、3.55× 一致，R39-11 的结论在更大的负载上成立。

最终状态（2026-10-10 晚）：

- **验收 1**：medium 25/25 在终版镜像 u280_r39h 上数值通过，Mode A 开、关都是。
- **验收 2**：冻结基线的 medium 数字已在 `spec/r39/baseline_archbench_medium.jsonl`。
- **验收 3**：v5 终版相对冻结基线，tiny 5.538、small 5.617、medium 5.491，没有点变慢；R39-11 的结论在三档上一致。

## 4. 执行顺序

1. **R39-00 先于一切。** 没有标准路径基线和消融工具，后续一切收益都无从证明。
2. **编译器侧 R39-01 → 04。** 它们能在板上分钟级测完，先做。
3. **硬件侧 R39-06、08、07、05、09。** 每项先 RTL 仿真加 OOC，再合并为少量板上镜像确认，以减少每次约 4 小时的链接。R39-05（4 context）改动面最大，排在旁路与 LSU 之后，好让它的消融反映的是「隐藏剩余延迟」，而不是掩盖本可以直接消除的停顿。
4. **R39-10 视数据决定。**
5. **R39-11 收尾。**

每完成一个工作包：更新冻结基线、`design_change` 记录、`doc/r39_tasks_and_acceptance.md` 的状态行，并推送。
