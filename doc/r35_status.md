# Phase 3.5 状态：复用上游编译基础设施，交付可观测的编译优化

合同：[phase3_5_tasks_and_acceptance.md](phase3_5_tasks_and_acceptance.md) v1.0。
签收名 `R35_COMPILER_ACCEPTED`。**当前 NOT ACCEPTED。**

前置已成立：R3 v2 于 commit `33dbb375828e` 在锁定 Linux 上签收
`R3_V2_CORE_VRF_ACCEPTED`（见 [`doc/evidence/r3v2/acceptance-linux.json`](evidence/r3v2/acceptance-linux.json)），
本阶段的全部主收益都在它签收的 `core-vrf-sync-v1` 上测。

## G0 的第一件事：先证明机会存在

合同 6.3 写得很清楚，5% 和 3% 是**待达到的阶段目标，不是已测加速**。所以在
承诺任何百分比之前，`tools/r35/scout.py` 读产品编译器**当前真正发射的机器码**，
按名字数出哪些字是可以不花的，并且每一项都写出去掉它的机制和今天卡在哪里。

在 `core-vrf-sync-v1` 上，冻结 workload 中能编出来的四个：

| workload | 发射字数 | 机会 | 点位 | 可省字数 |
|---|---:|---|---:|---:|
| L01 masked AXPY | 30 | 宽常量 | 1 | 3 |
| | | 乘 2 的幂 | 1 | 4 |
| | | masked load 的 `other=0` 死值 | 2 | **unknown** |
| L04 masked reduction | 22 | 宽常量 | 2 | 6 |
| | | masked load 的 `other=0` 死值 | 1 | **unknown** |
| L05 FP16 GEMM k16 | 156 | 宽常量 | 23 | 69 |
| L06 FP16 GEMM k32 | 183 | 宽常量 | 27 | 81 |

合计 391 字里有 **163 字是有名字、有机制的浪费，占 42%**。

三条机制分别是：

- **宽常量**：`.li` 永远发四个字（`movz` + 三个 `movk`）。这是汇编器**故意**的：
  伪指令必须在展开前就能精确定长，否则分支目标会错。所以要改的不是 `.li`，
  而是编译器——一个装得下 16 位的常量应该直接发 `s.movz`，一个字。GEMM 的
  结果排空循环每行都为一个 0…15 的行号花四个字，这一项就占了它 44%。
- **乘 2 的幂**：`arith.muli %pid, 16` 现在原样落到 `.li` + `s.mul`，五个字。
  上游 canonicalization 会把它变成移位，常量随即没有用户。
- **masked load 的 `other=0`**：这一项**记 unknown，不记成收益**。它要的是一个
  mask 感知的活跃性事实，外加一个安全论证——未被观测的 lane 不会 fault、也不会
  影响被观测 lane 的浮点结果。合同 7 的 R35-OPT-03 管这件事，没证明之前它不是收益。

上游那一侧也确实有活干：真实 Triton 源编出的 TTIR 里有 **6 个 `arith.constant`、
只有 3 个不同值**，`16 : i32`、`0.0 : f32` 和 `dense<0.0> : tensor<16xf32>` 各重复一次。
这正是 R35-OPT-01 要的正例——重复纯值出现在真实源里，不是为了喂给 CSE 才写的测试。

今天卡在哪里，也照实记：

- 产品路径是 TTIR → Python TNCA IR → codegen，**没有任何 MLIR pass 跑在上面**。
- `opt_level` 0..2 在 `python/tnca/aot.py` 里明确标着 reserved，不改变 emission。
- LinearLayout **一次都没有被调用**，现在没有任何布局计算可谈"复用"。

## O0/O1 已经能跑，收益已经实测

`opt_level` 不再是 reserved：

| 模式 | 现在真的做什么 |
|---|---|
| B0 | 冻结的既有行为。不动 |
| O0 | 新 driver 的兼容路径。四个 workload 上机器码与 B0 **逐字节相同** |
| O1 | 锁定上游 canonicalizer + CSE 跑在**产品即将导入的那个 module 上**（就地，所以优化后的 module 就是发射输入）；常量按能装下的最小形式下降；TNCA IR 源走 §4.1 的桥 |
| O2 | 未实现 |

上游确实在跑，不是被调用了然后丢掉：真实 Triton AXPY 的 TTIR 从 24 个值操作
降到 20，去掉四个 `arith.constant`；TNCA IR 那条桥把 GEMM 每个 descriptor 重复
生成的传输坐标合并掉。**判等是上游做的**——桥把纯标量森林发成一个把每个值都
return 出来的 `tt.func`，CSE 合并之后读 return 列表，两个被证明相等的值会以
同一个 SSA 名字回来。第一版在 Python 里自己算等价类、只拿上游跑个 diff，那正是
§4.2 禁止的"悄悄替代实现"，而且更弱：上游会把 `16 + 0` 折成 `16`，于是这样算出来
的两个值直接塌到那个常量上，结构化 key 匹配则看不出来。

在 R3 v2 签收的硬件上实测，四个都对独立 oracle 正确：

| case | B0 周期 | B0 字数 | O1 周期 | O1 字数 | 省下 | §6.3 门槛 |
|---|---:|---:|---:|---:|---:|---:|
| L01 masked AXPY | 228 | 30 | 218 | 27 | **+10** | 12 |
| L04 masked reduction | 74 | 22 | 69 | 16 | **+5** | 8 |
| L05 FP16 GEMM k16 | 1215 | 156 | 1191 | 83 | **+24** | 60 |
| L06 FP16 GEMM k32 | 3241 | 183 | 3168 | 95 | **+73** | 162 |

**四个都是真实收益，四个都没达到 `max(8 cycles, 5%)`。** L01 差 2 个周期，
L04 差 3 个。照实记：§6.3 明确禁止靠移动门槛去够到它。

两条值得记住的发现：

- **从一个本来就在 stall 的周期里拿掉一条指令，等于没拿掉。** L05 少了 47% 的字，
  只换来 2% 的周期。这个形状的 stall 分解是 538 里有 **422 是 LSU**，另外 matrix
  为 VRF 读等了 **544** 次。O2 必须去缩短核**在等什么**，而不是它等的时候在发什么。
- **一次测量里 C05 少了 7 条指令却慢了 12 个周期**：把坐标合并之后活跃区间变长了。
  没有成本模型的 CSE 不是免费的——这正是 R35-ALLOC-01 存在的理由。

还有一个自己的错误一并记下：门槛计算里用了 `int()` 而不是 `ceil()`，
把 L01 的门槛从 12 算成了 11——往对自己有利的方向偏了一个周期。已修。

## 逐条标准

27 条标准全部 **NOT VERIFIED**，除下表另有记录者。

| 包 | 标准 | 状态 | 依据 |
|---|---|---|---|
| R35-01 | R35-BASE-01 | **PASS** | 六个 workload、23 次 launch（12 主 + 4 消融 + 4 旧 profile + 3 安装后）、47 次 RTL 上限、27 条标准、8 个变异各有指定 detector，全部冻结在 `spec/r35_workloads.json` 与 `spec/r35_required_cases.json` |
| R35-01 | R35-BASE-02 | **PASS** | 十种畸形各自被按名字拒绝，全部在启动任何 RTL 之前：尺寸事后放大、删 workload、消融对象不再消融、主比较缩短、总数改成配合短矩阵、收益门槛调低、去掉不退化条件、预算放大、删标准、变异没有 detector。另含上面的机会评估 |
| R35-01 | R35-BASE-03 | **部分成立** | R3 v2 前置成立（签收、commit、干净树、八个 gate、`core-vrf-sync-v1` 存在都已核对）。**O0 == B0 已实测逐字节相同**（L01/L04/L05/L06 四个）。checker 本身仍在读 `aot.py` 里那句 reserved 判定，待更新 |
| R35-02 | R35-UP-01/02/03 | NOT VERIFIED | 尚无 adapter。`spec/r35_toolchain.json` 冻结了它必须遵守的边界：wheel 与 pinned 源码树不在同一进程里相遇，不跨 ABI 传 MLIR 对象，runtime 不得 import 上游 |
| R35-03 | R35-LL-01/02/03 | NOT VERIFIED | — |
| R35-04 | R35-OPT-01 | **部分成立** | 上游 canonicalize/CSE 在产品 O1 路径真实执行并改变机器码（TTIR 24→20 个值操作，去掉四个 `arith.constant`；TNCA IR 桥合并重复坐标）。尚缺：pass 前后 dump 与 remarks 的逐项对照 |
| R35-04 | R35-OPT-02/03 | NOT VERIFIED | AxisInfo 尚未接入；effect 边界目前靠白名单（`PURE_KINDS` 只有 const/sbin/sbini），负例未建 |
| R35-05 | R35-ALLOC-01/02、R35-SCHED-01 | NOT VERIFIED | — |
| R35-06 | R35-SAFE-01/02、R35-MUT-01 | NOT VERIFIED | — |
| R35-07 | R35-PERF-01 | **部分成立** | `tools/r35/perf.py` 按 §6.3 的 C 读设备周期（不是 host wall、不是模型预测），四个 workload 的 B0/O1 齐全并对 oracle 校验。尚缺：L02/L03 没有源，23 次矩阵未跑满 |
| R35-07 | R35-PERF-02 | **NOT MET** | 四个都有实测收益，**没有一个达到 `max(8, 5%)`**。按 §6.3 这是 `PERFORMANCE_NOT_MET`，保留测量与失败，不改 workload、不改门槛 |
| R35-07 | R35-PERF-03 | NOT VERIFIED | — |
| R35-08 | R35-PKG-01/02、R35-OBS-01 | NOT VERIFIED | — |
| R35-09 | R35-REG-01、R35-EV-01、R35-REL-01 | NOT VERIFIED | — |

## 关口

| 关口 | 包 | 状态 |
|---|---|---|
| A：依赖与 LL 语义 | R35-01/02/03 | 进行中。adapter 已直接调用上游真实 LinearLayout 与 MLIR pass；R35-LL-01/02/03 的正负例矩阵未建 |
| B：真正优化且安全 | R35-04/05/06 | 进行中（O1 的纯值优化已进产品路径并实测；分配/调度与安全差分未开始） |
| C：收益可观测 | R35-07 | 已测，**未达标**：`PERFORMANCE_NOT_MET` |
| D：产品与证据 | R35-08/09 | 未开始 |

## 一条必须一直说清楚的边界

42% 是**静态字数**，不是加速。合同 6.3 第 5 条专门堵这个口子：静态指令变少而
device cycle 没降，不算收益。这些数字是"值得动手"的理由，不是结果。
