# R39 报告：标准路径的性能

版本：草稿，2026-10-10。对应 `doc/r39_tasks_and_acceptance.md` 的 R39-11。

本轮所有数字都来自标准路径：torch.compile → Inductor（tensor descriptor 开）→ Triton → tnca → U280，或者与板卡逐点同周期的器件级 RTL 仿真。参考路径（无 descriptor、`compile_aot`）的数字只作对照，不进验收。

## 1. 结论

| 验收项（R39-11） | 要求 | 结果 |
|---|---|---|
| tiny + small 数值 | 全部通过 | 52/52 |
| S_geo 相对 R39-00 基线 | ≥ 3.0× | **4.315×** |
| 设备时间 E 相对基线 | ≥ 3× | **3.55×**（0.00725 → 0.02576） |
| 没有点慢于基线 | — | 最少的是 embedding tiny，1.974× |
| 100 MHz 时序 | 收敛 | 路由后 ap_clk WNS +0.087 ns；实测 100.017 / 99.992 MHz |
| B4 逐点等周期 | 27/27 | u280_r39d：24/27，3 点相差 1–8 拍。原因已定位（见 §4），u280_r39e 修复后复测 |
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
- 修复：launch 时两个指针都回到 0。修复后在仿真中验证：同一个会话里，不管前面跑过什么，同一个 kernel 都是 369 拍。镜像 u280_r39e 已带此修复重新链接。

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
- R39-10 矩阵单元加宽：一个 K16 MMA 约 256 拍，是 GEMM 点剩下的主要瓶颈之一。
- per-context issued counter 的硬件导出：仿真里用 `+ctxtrace`。

## 7. 代价

| | v2（u280_r39c） | v4（u280_r39d） |
|---|---|---|
| CLB LUT（含平台 shell） | 314 851 | 333 236（+5.8 %） |
| 寄存器 | 216 965 | 226 423（+4.4 %） |
| BRAM tile | 351 | 367 |
| 路由后 WNS（100 MHz） | +0.073 ns | +0.087 ns |

OOC 综合（只算 `tnca_fpga_top`，10 ns）：v4 比 v2 LUT +10 %、寄存器 +19 %，最差路径相同（FPU FMA 的操作数选择）。
