# Phase 3.6 状态：多后端工程准备

合同：[phase3_6_tasks_and_acceptance.md](phase3_6_tasks_and_acceptance.md)。
交接指南：[r36_handoff.md](r36_handoff.md)。

状态：**IMPLEMENTED / PENDING GATE**。30 条标准全部有 checker 并在开发环境
通过；签收判决由锁定平台上的 `make r36-acceptance` 与
`make r36-verify-evidence` 产生，本文件不代替该判决。

checker 写完后做过一次逐子句审计并据此修复了十余处"声称做了而没做"的检查，
见下文《一次逐条审计》。

目标为 `R36_ENGINEERING_READY`。**不指定 FPGA，不负责上板。** 板卡、厂商工具
与 IP、布局布线、bitstream、烧录、实板调试及 ASIC 工艺实现由独立后续分支完成；
`platform-verified` 是注册表拒绝接受的保留状态。

| 工作包 | 状态 | checker / 证据 |
|---|---|---|
| R36-01 集成基线与能力清单 | PASS (3/3) | `make r36-check-plan` |
| R36-02 后端抽象与依赖 | PASS (3/3) | `make r36-backend-tests` |
| R36-03 配置与兼容性 | PASS (3/3) | `make r36-config-tests` |
| R36-04 公共 RTL 与导出 | PASS (3/3) | `make r36-rtl-tests`、`make r36-export` |
| R36-05 存储/时钟复位适配 | PASS (3/3) | `make r36-mem-tests` |
| R36-06 host 协议与完成 | PASS (3/3) | `make r36-io-tests` |
| R36-07 构建安装与交付 | 2/3；BUILD-02 需锁定平台 | `make r36-build-tests` |
| R36-08 conformance 与回归 | PASS | `make r36-conformance`、`r36-mutation-tests`、`r36-regression` |
| R36-09 下游交接演练 | PASS (3/3) | `make r36-handoff-tests` |
| R36-10 证据与签收 | 待锁定平台运行 | `make r36-acceptance`、`make r36-verify-evidence` |

R36-BUILD-02 的 fresh-environment 用例在 macOS 上 **NOT RUN 并因此判 FAIL**：
锁定的 Triton 3.8.0 只有 manylinux x86_64 wheel，
`spec/r1_toolchain.lock.json` 明确 macOS 不能替代验收平台。按 R1-02，这不记为
skip；判决以锁定平台的运行为准。

## 这一阶段实际改了什么硬件

设备行为从 testbench 移入可综合的公共 top `rtl/tnca_shell.v`：命令口、完成
记录、host 访问口与仲裁、deadline、fault 锁存。`tb/device/tnca_device.v` 只剩
时钟、复位、存储和一层协议翻译，不再有层次化读写。

由此暴露并修复的缺陷（每一条都单独记录在 `spec/r36_baseline_delta.json`，
按 5.2 节要求与目录/接口整理分开）：

- 核心的 `f1_valid`/`fb_valid` 从未复位，`core_idle` 自上电起为 X，直到第一次
  launch 冲刷流水线才确定。
- `python/tnca/data/` 暂存副本会遮蔽工作树，且 `rtl/` 本身不在设备指纹里，
  两者合起来让 runtime 长时间对着旧 RTL 出结果。
- socket backend 不做参数校验，非法地址上线后由 relay 的错误路径关闭连接，
  把可恢复错误变成 device loss。

## 5.2 节保真度

以 R3.5 冻结提交 `0e7c34795cd6` 为基线，在该提交的 worktree 中实测：

- **机器码：六个冻结 workload 在 B0 与 O2 下全部逐位相同。**
- 周期：11 处差异，全部归因于 DELTA-01（shell 在时钟沿采样，而旧 testbench 在
  active region 读取；grid 的 device_cycles 累加，所以 grid=2 差 2）。L06 在
  慢速乱序模型上另有系统性偏移，因为 host 访问现在与 core 共用存储端口，
  以中位数对测得 spread 比较后记录。
- DELTA-02、DELTA-03 已声明但这些 workload 未触及，工具明确输出这一点。

## 一次逐条审计，以及它查出来的东西

在所有 checker 写完之后，对 30 条标准逐子句做了一次审计：读合同的一句，读
对应 checker 的实现，判断该句里每一项要求是否真的被执行。结论是"覆盖第一
项、静默跳过后面几项"是常见失败模式。以下是**代码或 docstring 声称做了而
实际没做**的部分，都已修复：

- IO-01 的非法请求括号里，序号计数器读了之后被 `del` 丢掉，而 docstring 写
  着两半都必须不变。
- MUT-ASSET-HASH 根本没有调用任何 checker，它自己内联重算了一遍 hash 并伪造
  了结果字典。
- 冻结变异表有 7 条，harness 只有 6 条；M07 从未实现，而 plan.py 断言数量是
  7，于是"数量和合同一致、工作不一致"。
- 变异套件没有 clean 对照：在 lab 里因自身原因失败的 checker 会被记为检出。
- BE-01 只 grep 了 `runtime.py` 一个文件；启动 vvp 的是 transport.py，调用
  iverilog 的是 device.py。
- RTL-02 的 docstring 说它打开了 random stalling，而 smoke bench 把
  `RANDOM_STALL(0)` 写死了。
- CFG-03 打印了一句关于"验收矩阵"的散文，而树里根本没有这样一份矩阵。
- HAND-01 用 `PYTHONPATH=<repo>/python` 启动演练，5.3 节要求用**安装包**。
- BUILD-02 的 runtime-only 一半被两处文件声称由 R2 的 PKG-03 覆盖；没有
  pkg03 这个函数。
- IO-02 的并发拒绝是两段散文。
- `gemm_program(..., roles=2)` 参数被接受且被忽略，两个调用方拿到的是与
  `roles=1` 逐字节相同的机器码——"两 role pipeline"只是一个标签。

修复之外，另有 4 条覆盖仍然是部分的，已逐条记入
`spec/r36_required_cases.json` 的 `known_partial_coverage`（写明未检查什么、
为什么、以及什么能关闭它），并由 R36-BASE-01 断言该记录存在且完整：
R36-RTL-03、R36-EV-02、R36-CFG-02、R36-REL-01。

## 尚未做的事

`spec/r36_memories.json` 中每个阵列的 `technology_mapping` 均为
`not started`；ARF 仍为行为模型；无 cache coherence；四 tile runtime 仍有缺口。
这些写在 [交接指南](r36_handoff.md) 的「不是板级任务」一节，属于主线责任。

`CAP-TWO-ROLE`：核心有两个 role context（`NUM_RCTX=2`，
`tb/core/progs/t04_roles.s` 在真实 RTL 上跑生产者/消费者），但**产品路径到
不了**——`runtime.py` 写死 `role_mask=1`，GEMM builder 只发单 role tile。
5.1 节点名的"两 role pipeline"因此记为 NOT RUN。

R35-OPT-02 在基线中即为 open，本阶段未关闭也不声称关闭。
