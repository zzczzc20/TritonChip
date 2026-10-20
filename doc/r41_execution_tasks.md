# R41 执行清单：频率与双内存路径

日期：2026-10-20。作者指令：先开 `r41` 分支、列任务，再实现。
本清单记录本轮执行范围；现有 `r41_tasks_and_acceptance.md` 是更广范围的未冻结草案。

## 起点与状态

- [x] 从 R40 关闭提交 `9235f5d` 创建 `r41`。
- [x] 列出两条主线的任务后，合并 `r40-final-image` 的 `6eed3e0`。
- [ ] 在合并后的 R41 源码上完成 M1–M4 相关回归。
- R40 保持 `closed-with-failures`，`R40-10.acceptance` 未通过。
- 已加载镜像仍为 `a88ebdf59cfbbb2001ebd5b1b2c739c5d8cd33931f1e9202f18cac07c4c83083`，实测 90 MHz、三核、三个 HBM 出口、4 GiB 窗口。
- 原工作区存在未提交的 SystemC/Ramulator2 实现及 R41 合同草案；保留原状，不将其自动当作已验证交付。

## 主线①：95–100 MHz

M1–M4 已合入，但尚未在新镜像上实现或上板：

| ID | 内容 | R41 验证 |
|---|---|---|
| M1 | host grant 寄存边界 | host mux 定向、随机及三核 ownership 测试 |
| M2 | 源端与 tile 本地 `host_allow` 寄存 | coherent-host、权限撤销与在途响应 |
| M3 | 本地 memory controls、aperture 预计算、MMIO settle | memory mode、MMIO、AXI limits/stages、实际镜像几何 |
| M4 | ISA §8.4 float-to-int 修复 | 独立 Fraction oracle、边界向量和核心指令回归 |

任务顺序：

1. 完成上述回归并保存当前源码身份和日志。
2. 从原始 census 重建路径分类，核对 12,799 endpoint / 753 family 的完整覆盖。
3. 按下表逐类确定寄存边界，先做单元功能与同约束 OOC 筛选。
4. 集成候选后，复跑关联核心/协议回归、资源与布局预检，更新 census。
5. 依据已验证的候选决定完整实现；完成 route timing 后再上板测量频率与正确性。

### 约十项结构改造候选

这是待设计和筛选的工作包，不能将十个条目等同于已证明足以闭合时序。

| ID | 残余结构 | 修改与验证重点 |
|---|---|---|
| F01 | `stall_front` / operand enable 高扇出 | 本地控制与有 credit 的边界；保持在途指令、flush、重放语义 |
| F02 | scalar read → barrier / scoreboard / issue | 读值与依赖检查分阶段；验证跨 role wakeup 和 RAW |
| F03 | VFPU `red_busy` → FMA stage-0 | 注册仲裁/输入边界；验证 reduction 与普通 FMA 交替 |
| F04 | writeback `w_rctx` / `w_en` → `w_sdata` | 上下文与数据同行；验证多上下文冲突和返回顺序 |
| F05 | scoreboard `sb_v` → VRF bank / collector | 局部有效位与采集边界；验证 bank 冲突与 forwarding |
| F06 | dispatch DSP `total` → `remaining` | 分离乘法结果与调度状态更新；验证维度边界、deadline |
| F07 | slice `cur_ms` / `ms_way` → data RAM | 审核已有 C1 候选 `658c811`；保持 commit、eviction、maintenance 可见性 |
| F08 | XLS FMA `p0_sticky_al` → `p1_sh_mag` | 重新平衡生成源与流水；保留 IEEE 舍入与逐 K 累加顺序 |
| F09 | XLS FMA `p1_sh_mag` → MTX accumulator | 累加回写边界与相关延迟；验证连续 MMA、ACC RAW、source-free/result-ready |
| F10 | LSU `pending` / FSM → mesh input FIFO | request/credit 寄存边界；验证背压、fault drain 与响应 ownership |

参考：[原始 census 与分析说明](../evidence/r40/v3/final_image_candidate/census/README.md)、
[频率 frontier](../evidence/r40/v3/final_image_candidate/census/frontier.json)。
12,799 是原始 routed DCP 在分析中排除 M1–M3 组合锥后的 what-if 结果，
不是合并后新设计的实测失败数。95 MHz 对应 170 个阻塞 family、8,959 个 endpoint；
100 MHz 对应全部 753 个 family。不可把该分析中的临时路径排除带入实现约束。

## 主线②：内存模型双 path

- [ ] 审核并构建已有实现；固定 Ramulator2 revision、配置与来源身份。
- [ ] 默认参数快路径与 R40 M2 做架构结果、周期及计数器对照。
- [ ] 慢路径验证 row hit/miss/conflict、bank 并行、读写切换、refresh、队列背压与写完成。
- [ ] 核对真实 address → PC/bank/row 映射和 AXI/NoC/home/controller/DRAM 的计时边界，避免重复计入。
- [ ] 验证路径切换及缺少 Ramulator2 依赖时的明确拒绝。
- [ ] 验证 replay/定点迭代的收敛、请求匹配和不收敛状态；审核逐命令重置 DRAM 状态的适用范围。
- [ ] 保存独立开发/校准/验证/holdout 数据，比较残差与模拟 wall time。
- [ ] 根据新 RTL 流水更新模型配置；每份结果记录模型与硬件身份。

实现入口：`tools/performance_harness/systemc/include/tnca_dram.hpp`、
`src/chip.cpp`、`ramulator/`（相对 `tools/performance_harness/systemc/`）。
现有慢路径是逐命令 fork/replay 定点模型；完成验证前不声称在线 DRAM 耦合或完整时序校准。

## 完成证据

每项保留源码身份、配置、命令、退出状态及结果。
功能通过、OOC 时序、整机 route timing、上板频率、模型校准分别判定。
R40 已有 PASS 是来源证据，不能直接复制成 R41 PASS。
