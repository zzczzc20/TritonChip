# Phase 3.6 状态：多后端工程准备

合同：[phase3_6_tasks_and_acceptance.md](phase3_6_tasks_and_acceptance.md)。
交接指南：[r36_handoff.md](r36_handoff.md)。

状态：**IMPLEMENTED / PENDING GATE**。30 条标准全部有 checker 并在开发环境
通过；签收判决由锁定平台上的 `make r36-acceptance` 与
`make r36-verify-evidence` 产生，本文件不代替该判决。

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

## 尚未做的事

`spec/r36_memories.json` 中每个阵列的 `technology_mapping` 均为
`not started`；ARF 仍为行为模型；无 cache coherence；四 tile runtime 仍有缺口。
这些写在 [交接指南](r36_handoff.md) 的「不是板级任务」一节，属于主线责任。

R35-OPT-02 在基线中即为 open，本阶段未关闭也不声称关闭。
