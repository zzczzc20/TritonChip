# R41 Stage 1：在途证据

状态：**未签收**。合同为 `doc/r41_tasks_and_acceptance.md`；本文件只索引实际证据，不替代或缩减合同。
所有本轮编译、RTL、SystemC 和 Torch 测试均在 GB10 `gx10-6646` 执行。

## 已有证据及其边界

| 证据 | 观察 | 不能据此声称 |
|---|---|---|
| `kickoff/merged-boundaries-summary.json`、同名 tgz | 合并 M1–M4 后 lint、核心、coherent host、memory controls、f2i 回归完成 | 新镜像实现、100 MHz 或上板通过 |
| `kickoff/harness-smoke-v3-summary.json`、同名 tgz | 干净构建 SystemC/Ramulator2；共同 Session 真产物；快慢路径；监控模式数值/周期/计数器不变；有界 trace 报告丢弃；worker 生命周期检查完成 | 全芯片周期模型已校准、三个闭环案例均已完成 |
| `fragments/` | 标准 Torch mm/addmm、tail、FP16/BF16、非连续输入；宽 tile 的独立 ACC 跨 K 保留；有旧映射及保守 16x16 对照 | 宽 tile 一定更快；这些小 tail 点中保守 tile 的模型周期更少 |
| `kickoff/depthwise-regression-v2-evidence.tgz` | 3 项谓词检查含实际 RTL 逐 lane 校验；3 项 fragment 检查；4 项 depthwise/broadcast 检查；D256 attention、depthwise、FP32 recurrence 代表图全部功能通过 | 完整 Qwen 文本模型或长期 cache 生命周期已经通过 |
| `kickoff/compiler-regression-after-depthwise-v2.log` | 生成文件一致性检查与 102 项 compiler unit 检查通过；MLIR `tnca-opt` 未安装，IR suite 明确 SKIP | 未执行的 MLIR suite 通过 |
| `kickoff/qwen-storage/` | 固定官方 revision，632 个 tensor 的 header/index 覆盖；语言分支 320 个 storage、3,763,655,360 字节；独立 dtype/shape/区间/文件大小审计通过 | 4 GiB 内完整运行时容量足够；权重未下载，执行未测量 |
| `kickoff/prospective-prediction.json`、`prospective-wide/` | 板测前保存候选预测及不可变源码身份 | 对应板测、残差或最终源码回归已经完成 |

谓词压力修复使用现有 `p.ballot`/`p.frombits` 保存精确 lane mask，不能增加架构容量。
pointwise 超出资源时依次尝试较小 block，保留共同 TNCA 执行路径；已有合法配置仍优先使用。
column broadcast 修复 `[R,1]` 加载后的每行 lane 0 复制语义，不按模型或算子名称特判。

原始失败保留：`make compiler` 首次发现生成的 support matrix 落后于新增的两条可生成指令；
由远端原生成器更新矩阵后复跑通过。depthwise 的初始 predicate/scalar 资源失败和 bias 仅首 lane 生效的诊断均未计作 PASS。

## 尚未满足的签收条件

公共 `TNCA_TUNE` 消费者、同步与 locality/DCO 审计、内存映射及计时边界资格、三个闭环案例、
匹配 RTL/SystemC 速度测量、候选 OOC/区域预检和完整模型接口差异分析仍须完成。
最终 90 MHz 全套数值回归、正式非退化结果、prospective 板测残差、干净 git archive/fresh cache 重建及冻结板上锚点亦未完成。

卡上观测镜像曾为 UUID `428c00e5-6996-92c7-1f8f-ec41f2d037b3`、245 MHz，且有使用者；
与 B40-OBS 的 90 MHz 镜像不同。`kickoff/restore90-check.log` 为只读检查，usage gate 未通过，未执行加载。
恢复 R40 镜像须有明确授权、卡空闲且全部加载检查通过。当前不能把原 90 MHz 镜像当作仍在运行。

模型保持 `uncalibrated`。sealed holdout 未用于开发或拟合；已有失配与负结果不删除。
