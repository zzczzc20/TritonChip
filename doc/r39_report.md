# R39 报告：标准路径的性能

版本：2026-10-10。对应 `doc/r39_tasks_and_acceptance.md` 的 R39-11（v4 终版，镜像 u280_r39e）。R39-12/13（Mode A、v5、medium 档）的结果在完成后另列。

本轮所有数字都来自标准路径：torch.compile → Inductor（tensor descriptor 开）→ Triton → tnca → U280，或者与板卡逐点同周期的器件级 RTL 仿真。参考路径（无 descriptor、`compile_aot`）的数字只作对照，不进验收。

## 1. 结论

| 验收项（R39-11） | 要求 | 结果 |
|---|---|---|
| tiny + small 数值 | 全部通过 | 52/52 |
| S_geo 相对 R39-00 基线 | ≥ 3.0× | **4.315×** |
| 设备时间 E 相对基线 | ≥ 3× | **3.55×**（0.00725 → 0.02576） |
| 没有点慢于基线 | — | 最少的是 embedding tiny，1.974× |
| 100 MHz 时序 | 收敛 | u280_r39d：路由后 ap_clk WNS +0.087 ns，实测 100.017 / 99.992 MHz；终版 u280_r39e：WNS +0.053 ns，实测 100.003 / 99.990 MHz |
| B4 逐点等周期 | 27/27 | u280_r39e：整套连跑 26/27；attention_decode 板上 10252、仿真 10251。单独重跑两次、紧跟 argmax 重跑，板上都是 10251（§4） |
| leave-one-out 每项 ≥ 3 % | — | 6 项达到；4 个 context、RCTX_SWITCH、FE_NOBUBBLE、VRF_BYPASS、FPU_OVERLAP 不足 3 %，降级为「无显著贡献」（§3） |

## 2. 累计表（板上，tiny + small，52 点，相对冻结基线）

| 步骤 | 镜像 / 软件 | S_geo | E | 总周期 |
|---|---|---|---|---|
| R39-00 基线 | u280_r39b，全部开关关 | 1.000 | 0.00725 | 1.000 |
| + R39-01 指令调度 | nd100c 镜像上单独测 | 1.158（相对同镜像不开） | — | — |
| + R39-02a 代码形状 | u280_r39b，软件全开 | 2.448 | 2.13×（supported-only） | 0.381× |
| + R39-06/07/09 硬件 | u280_r39c（v2） | 2.840 | 2.57× | 0.334× |
| + R39-05/08/10 与 sparse TensorMap | u280_r39d（v4） | **4.315** | **3.55×** | **0.224×** |

v2 → v4 一步在板上是 S_geo 1.519、总周期 0.669×。

停顿占比（tiny，基线 → 终版）：

| | 基线 | 终版 |
|---|---|---|
| 发射（IPC） | 0.277 | 0.425 |
| hazard | 32.1 % | 16.4 % |
| VRF | 11.2 % | 4.0 % |
| 前端 | 13.9 % | 4.3 % |
| LSU | 12.6 % | 22.0 %（绝对周期 0.44×） |

## 3. leave-one-out（RTL 仿真，tiny，相对终版）

仿真与板卡逐点同周期（v2：27/27；v4 见 §4），所以下面的数字就是板上数字。每行只关一项，其余都开。

| 关掉的一项 | S_geo 变慢 | 套件周期 | 开着时最差的一点 |
|---|---|---|---|
| R39-02a 代码形状 | 2.378× | +148.4 % | 无变慢 |
| R39-05 role-SPMD | 1.314× | +26.3 % | 0.995 |
| sparse TensorMap | 1.037× | +5.6 % | 无变慢 |
| R39-01 指令调度 | 1.052× | +3.0 % | 0.977 |
| R39-05 4 个 context（改用 v3 的 2 个） | 1.076× | +0.5 % | 无变慢 |
| R39-06/07 VRF_PREFETCH | 1.147× | +12.2 % | 1.004 |
| R39-08 LSU_INFLIGHT | 1.084× | +10.7 % | 1.008 |
| R39-05 RCTX_SWITCH | 1.056× | +1.3 % | 0.982 |
| R39-09 FE_NOBUBBLE | 1.051× | +2.1 % | 0.883 |
| R39-10 FPU_OVERLAP | 1.035× | +0.8 % | 0.933 |
| R39-06 VRF_BYPASS | 1.032× | +2.4 % | 0.957 |

（最后一列：开着这一项时，最差的那一点与关掉时的周期比；小于 1 表示它让该点变慢。）

- **套件周期 ≥ 3 % 的工作包**：R39-02a、R39-05 role-SPMD、VRF_PREFETCH、LSU_INFLIGHT、sparse TensorMap、R39-01（恰好 3.0 %）。
- **记为「无显著贡献」（套件周期 < 3 %）**：4 个 context、RCTX_SWITCH、FE_NOBUBBLE、VRF_BYPASS、FPU_OVERLAP。
  - 它们的几何平均增益都在 3.2–7.6 %，但大周期的模型点主要受 GEMM 和 LSU 限制，这几项碰不到。
  - 这几项都不是 0：v1 → v2 那一步（VRF_BYPASS、VRF_PREFETCH、FE_NOBUBBLE 一起）在板上是 S_geo 1.158。在终版里关掉其中一项时，其余几项和多 context 已经把大部分停顿隐藏了。
- **一个对时序敏感的点**：softmax fp16 tiny。关掉 FE_NOBUBBLE、FPU_OVERLAP、VRF_BYPASS、RCTX_SWITCH 中任意一项，它都会快 2–12 %（全开 4489 拍，关 FE_NOBUBBLE 3965 拍）。原因还没查清，列为后续工作。即使如此，它相对基线仍快 4.36×。

4 个 context 一项的套件周期只多 0.5 %，按 R39-11 的规则记为「无显著贡献」：模型点由 GEMM 模板主导，而持有资源的 kernel 只用 2 个 context（§5）。按点的几何平均，4 个 context 仍快 7.6 %，全部来自不持有资源的 kernel。

## 4. 板上确认与 B4

- u280_r39d：tiny 27/27、small 25/25；B1 conformance 通过。
- B4（板上对同一 RTL、同一软件的器件级仿真）：24/27 逐点相同。add fp16 2391/2395、attention_decode 10252/10251、softmax fp32 1702/1710 不同。
- 原因：取指的 round robin 指针和 VRF 读仲裁的 round robin 指针在 launch 时不复位。板上的器件跨点持续运行（并且先跑了 B1），仿真每个点从复位开始。一个 context 时这两个指针不影响周期；多 context 时影响交织顺序。
- 修复：launch 时两个指针都回到 0。修复后在仿真中验证：同一个会话里，不管前面跑过什么，同一个 kernel 都是 369 拍。
- 终版镜像 u280_r39e 带这个修复，sha256 `6a3e76a4…4b18cf`。
  - 在板上：B1 conformance 通过；tiny 27/27、small 25/25 数值通过。
  - B4：add fp16、softmax fp32 已与仿真逐拍相同。
  - attention_decode 在整套连跑中是 10252，仿真是 10251。
- 复测 attention_decode：同一镜像上单独跑两次，以及紧跟它前一个点 argmax 跑，板上都是 10251，与仿真相同。
  - 所以这 1 拍不是前一个 kernel 留下的状态，也不能稳定复现。
  - 原因未查清，记为未结项。
  - 仿真没有模拟主机侧的异步行为。

## 5. 做了什么（按工作包）

详见 `doc/r39_tasks_and_acceptance.md` 各节的进展。要点：

- **R39-01**：每个基本块按实测的机器模型做 list scheduling（`tnca_cc.schedule`）。
- **R39-02a**：no_x_dim；lane-uniform 的下标留在标量寄存器里；入口建一次 TensorMap；`acc + dot` 原地累加。
- **R39-05**：
  - Mode B：每个 context 循环执行一行 grid 的 program。
  - `RCTX_SWITCH`：停顿的 context 让路给其他 context。
  - `RSLICE`：per-core 资源按 launch 给出的切片数分给各 context。
  - 4 个 context（v4）。
  - 编译器按 kernel 选 context 数：持有资源的最多 2 个，其余用全部。
- **R39-06/07/09**：`VRF_BYPASS`、`VRF_PREFETCH`、`FE_NOBUBBLE`。
- **R39-08**：`LSU_INFLIGHT=2`，load、store 都可重叠；sparse TensorMap。
- **R39-10**：`FPU_OVERLAP`。
- **缺陷**：
  - 缺陷 B4（与验收项 B4 同名，不是一回事）：写回冻结会丢 FP/MUL 结果。
  - B5：VRF 读仲裁会饿死矩阵端口。
  - launch 不复位 round robin。

## 6. 没做的

- R39-02b 宽块（多寄存器 1-D tile）：导入器明确拒绝 64 元素的块。
- R39-03 软件流水、R39-04 bank 感知分配。
- R39-09 分支预测：taken 分支的代价只占约 0.4 %，按数据不做。
- R39-10 矩阵单元加宽：MAC 阵列仍是每拍一行。
  - v4 上一个 FP16 K16 的 MMA 实测 551 拍，不是早先估的约 256 拍：每个 MAC 都在等一次读 A 元素的 VRF 访问。
  - v5 的 `MTX_ABUF` 把它降到 311 拍（§8）。
- per-context issued counter 的硬件导出：仿真里用 `+ctxtrace`。

## 7. 代价

| | v2（u280_r39c） | v4（u280_r39d） | v5（u280_r39f） |
|---|---|---|---|
| CLB LUT（含平台 shell） | 314 851 | 333 236（+5.8 %） | 336 181（+0.9 %） |
| 寄存器 | 216 965 | 226 423（+4.4 %） | 227 541（+0.5 %） |
| BRAM tile | 351 | 367 | 367 |
| 路由后 WNS（100 MHz） | +0.073 ns | +0.087 ns | +0.101 ns |

v5 的 RPAIR 与转置传输几乎不占资源。

OOC 综合（只算 `tnca_fpga_top`，10 ns）：v4 比 v2 LUT +10 %、寄存器 +19 %，最差路径相同（FPU FMA 的操作数选择）。

## 8. R39-12、R39-13：Mode A、v5 与 medium 档

### 8.1 做了什么

- **Mode A**（`rctx_roles`，默认关）：在导入器的 IR 上跑上游 warp specialization 的三步（`tnca_cc.partition.specialize_loop`）：
  1. PartitionScheduling：分区。
  2. aref：2 级 window 环，每级一对 full/empty barrier。
  3. PartitionLoops：每个 role 一份循环。

  一个 program 变成 producer（传输）和 consumer（MMA 与 epilogue），跑在一对 Role Context 上（`RPAIR`）；各对再按 Mode B 跑一行 grid。
- **v5 的硬件**（profile core-vrf-sync-v5）：
  - `RPAIR`。
  - RPAIR 下精确的 `mb.arrive`。
  - 转置传输 `HAS_TENSOR_XPOSE`。
  - `MTX_ABUF`：MMA 先把 A 片段读进缓冲，FP16 K16 从 551 拍降到 311 拍。
  - `RCTX_SWITCH=2`：等矩阵单元或传输引擎的 context 也让路。
- **编译器**：
  - linear 的 B（W 转置）用一次转置传输，代替 16 次 gather 加经 scratch 的重排。
  - masked 连续行的 load/store 每行一条带谓词的访存。
  - 只有一个使用者的 tile 延迟到使用处按行计算，GEMM epilogue 不再经过 scratch 和两个 accumulator。
  - 零初始化的 accumulator 用一条 `m.zero`。
- **缺陷**：
  - B6：程序自己构造的 barrier token 在 `mb.wait`/`mb.test` 上没有按 slice 改写，是 R39-05 留下的。
  - Mode A 自己的一个错：bmm 模板把指针 tile 在循环里经 scratch 携带，循环前的写原来只放在 consumer 里，结果错。
    - 现在循环前的写两个 role 各做一份。
    - 一个 role 可能读到另一个 role 写的内存时，不拆分这个循环。
- **medium 档**（R39-13）：25 点，每个 kernel 的参数都在 384 KiB heap 以内，规模是 small 的 2–8 倍。

### 8.2 板上结果（镜像 u280_r39f：v5，不含 MTX_ABUF 与 RCTX_SWITCH=2）

- 时序与时钟：100 MHz，路由后 ap_clk WNS +0.101 ns；实测 100.002 / 99.947 MHz；xclbin sha256 `7631515f9cde…`。
- B1 通过；tiny 27/27、small 25/25、medium 25/25。
- **B4：27/27 逐拍相同。**

| 比较（板上，Mode A 关） | tiny | small | medium | 合计 77 点 |
|---|---|---|---|---|
| v4 终版 / 冻结基线 | 4.304 | 4.325 | 4.309 | 4.313 |
| v5（r39f）/ v4 终版 | 1.193 | 1.195 | 1.173 | **1.187**（总周期 0.745×） |
| v5（r39f）/ 冻结基线 | 5.135 | 5.166 | 5.056 | **5.119** |

没有点比 v4 终版慢。

### 8.3 Mode A 的消融

- **板上**（r39f，开 / 关 `rctx_roles`）：77 点都正确，S_geo **0.997**。
  - 单 program 的 matmul tiny 快 1.084。
  - 已被 Mode B 掩盖的 GEMM 变慢，最多 5 %：mlp 0.956，matmul small 0.953，linear tiny 0.955。
- **仿真**（tiny，v5 终版软件与硬件）：S_geo **1.006**。
  - 变快的：gemv 1.112，matmul 1.072，llama_decode 1.047。
  - 变慢的：mlp 0.966，linear 0.980，gpt2_block 0.988。
- **结论**：按 R39-11 的规则，Mode A 记为「无显著贡献」，默认关。
  - 原因 1：这些 GEMM 受矩阵单元约束。Mode B 已让两个 program 的传输与 MMA 互相掩盖。
  - 原因 2：Mode A 让两个 role 各算一遍地址，在单发射的核上多占发射槽。
  - Mode A 只在 program 少（每行 grid 一个）或 producer 重（gemv 的 A 行要重排）时有收益。
  - 它作为功能完整、经测试的选项保留。

### 8.4 终版镜像 u280_r39h（v5，含 MTX_ABUF 与 RCTX_SWITCH=2）

仿真（tiny，相对 v4 终版，Mode A 关）：
- 全套件 S_geo 1.287，没有点变慢。
- GEMM 类 1.91：linear 3.01，mlp 3.41，gemv 2.71，gpt2_block 2.23，llama_decode 1.70，llama_prefill 1.62，matmul 1.39，bmm 1.36。

板上结果待镜像链接完成后补上。
