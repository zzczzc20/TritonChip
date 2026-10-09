# R39：标准路径的性能——去掉 R38 及以前的妥协

版本：1.1，2026-10-09。状态：**R39-00 进行中**（进展见各工作包末尾）。
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

- 交付 1–4 已完成（`7121da0`、`db59522`）。B1 由 `t38_wbfreeze` 复现后修复；B2 由 `tools/r39/vrf_fairness_test.py` 证明 4 口、5 口每个请求者都会被服务；B3 文档已更新。
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

### R39-02：宽块与多寄存器 tile（编译器 + Inductor）

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

### R39-10：矩阵与规约单元的吞吐（硬件，按收益决定是否实施）

交付：

- 矩阵单元：去掉 banked 文件上的端口 3 额外一拍，与 R39-07 合并实现；B 装载与 MAC 重叠；允许下一条 MMA 在当前 MMA 收尾时开始。
- FP 规约：树规约流水化，不阻塞普通 FP 指令。
- 每个子项只有在 R39-02/03 之后的 counter 显示它确实是瓶颈时才实施。实施的必须给出消融；不实施的必须给出数据说明为什么不做。

验收：实施的子项，其目标点（GEMM 类或 softmax/norm 类）S_geo ≥ 1.2×；不实施的子项需有数据支持。

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

## 4. 执行顺序

1. **R39-00 先于一切。** 没有标准路径基线和消融工具，后续一切收益都无从证明。
2. **编译器侧 R39-01 → 04。** 它们能在板上分钟级测完，先做。
3. **硬件侧 R39-06、08、07、05、09。** 每项先 RTL 仿真加 OOC，再合并为少量板上镜像确认，以减少每次约 4 小时的链接。R39-05（4 context）改动面最大，排在旁路与 LSU 之后，好让它的消融反映的是「隐藏剩余延迟」，而不是掩盖本可以直接消除的停顿。
4. **R39-10 视数据决定。**
5. **R39-11 收尾。**

每完成一个工作包：更新冻结基线、`design_change` 记录、`doc/r39_tasks_and_acceptance.md` 的状态行，并推送。
