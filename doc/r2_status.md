# R2 状态：逐工作包的实际进度

合同：[phase2_tasks_and_acceptance.md](phase2_tasks_and_acceptance.md)。
接口冻结：[r2_interfaces.md](r2_interfaces.md)。
前置：[R1 合同](phase1_tasks_and_acceptance.md)、[R1 状态](r1_status.md)、
[R1 签收](r1_signoff.md)（**R1 已 ACCEPTED**）。

本文件只记录**已经跑过并留下证据**的部分。状态只有 DELIVERED、PARTIAL、
NOT STARTED 三种；没有完成度百分比。

**R2 未签收。** 十四个工作包中十个的验收条件已满足。最终判决还要求：在 R2
候选 commit 上重跑 R1 全套（DEP-02）、grid/多批（R2-09）、host transport 的
完成边界（R2-10）、报告与示例（R2-12）、E2E 与 mutation（R2-13）、双环境发布
（R2-14）。

## 这一阶段已经成立的核心性质

一次编译、一个 artifact、多次不同参数的执行，中间结果留在设备内存里：

```text
    compiler invocations: 1 for the first compile, 0 for four more
    alpha=2.5   SUCCEEDED  match=True
    alpha=-1.0  SUCCEEDED  match=True
    alpha=0.25  SUCCEEDED  match=True
    32 launches, alternating alpha: 32 correct
    device processes started: 1 (131 requests served in one)
    A->B->C with no host download in between: SUCCEEDED/SUCCEEDED, correct=True
```

安装后、仓库移出 `sys.path`、没有 Triton 也没有编译器：

```text
    $ tnca-sim prepare
      building core-sim -> ~/.cache/tnca/devices/core-sim-d4b7392.../tnca_device.vvp
    $ tnca-sim run axpy.tnca --arg n=16 --arg alpha=2.5 ...
      job 1: SUCCEEDED, 113 device cycles
      o_ptr: [6.0, 6.0, 6.0, ...]
```

## 逐包状态

| 工作包 | 状态 | 实际交付 | 未满足的验收条件 |
|---|---|---|---|
| R2-01 R1 衔接/合同 | **PARTIAL** | 84 条语义用例展开为 170 个执行 ID；`tools/r2/baseline.py` 实现 DEP-01（缺失/伪造/不完整/PARTIAL 的 R1 全部拒绝，复用 R1 自己的验证器）；`doc/r2_interfaces.md` 冻结依赖方向与 adapter 边界 | DEP-02 需要在 R2 候选上重跑 R1 全套；DEP-03 的支持矩阵待补 |
| R2-02 安装/API | **PARTIAL** | 非 editable wheel；`tnca-aot compile/inspect/doctor`、`tnca-sim run/prepare/doctor`；**PKG-03 已验证**：无 Triton、无编译器、仓库外执行成功；运行端零依赖 | PKG-01/02/04/05 尚未作为受控用例执行 |
| R2-03 编译 driver | **DELIVERED** | CMP-01…05 全通过：真实 Triton add/mul-add/动态 alpha AXPY 经产品入口编译并在 RTL 执行；六种畸形输入各按自己的错误码拒绝；R1 的负例经产品路径在 `-O2 --debug` 下仍被拒绝 | 无 |
| R2-04 MLIR 主路径 | **NOT STARTED** | — | MLIR-01…05 |
| R2-05 artifact/loader | **DELIVERED** | ART-01…06 全通过：一字节改动、截断、段重叠、重复 JSON key、未知版本、另一个 artifact 的段各被拒绝；profile/ABI/format 不匹配**在零次设备请求内**拒绝；容器无路径可遍历 | 无 |
| R2-06 动态 ABI | **PARTIAL** | ABI-01（独立 byte-level oracle，六类型 22 个值）、ABI-02（三组地址与标量，code hash 不变）、ABI-03（九种坏参数，无一启动任务）、ABI-04（提交后修改 dict 无效） | ABI-05 alias 正例、ABI-06 gather/scatter 越界 fault |
| R2-07 缓存/确定性 | **DELIVERED** | CACHE-01…04：七个维度改变 key，运行期输入无处可传；两个干净环境同 hash；损坏与截断各拒绝；**编译器被计数而非计时**，四次复用调用编译器 0 次 | 无 |
| R2-08 runtime 生命周期 | **DELIVERED** | RT-01…07：完整生命周期、32 次不重启、八个占位第九个 QUEUE_FULL、wait 幂等且超时不是判决、租约阻塞、两 session 互不干扰、依赖链与跨 session 依赖拒绝 | 无 |
| R2-09 grid/admission | **PARTIAL** | GRID-01/02（九种 grid，每个 PID 恰好出现一次，NPID 是逻辑 grid 大小，与独立 oracle 比对）、GRID-06（映射是双射，边界被拒绝）。`t.program_id(0)` 曾返回线性 id，transport 现在传全部六个 grid 值 | GRID-03/04/05/07 需要四 tile 持久设备 |
| R2-10 host transport/完成 | **PARTIAL** | MEM-03（A→B→C 全程留在设备内存）、MEM-05（静态：`python/tnca` 15 个模块无任何层级引用；动态：杀掉设备后 download 变成 DEVICE_LOST 而不是一个数）、MEM-06（每个请求都记录方向/地址/长度/状态；越界被拒；canary 完好）。设备的越界检查曾因 `integer` 是 32 位有符号而对 0xFFFFFFF0 失效 | MEM-01/02/04 需要四 tile profile 的脏 cache 行与延迟注入钩子 |
| R2-11 故障/恢复 | **DELIVERED** | ERR-01…07 全通过：trap 归因到 code/PC、可排空故障后 session 仍可用、设备 deadline 毒化并使排队任务 FAILED、recover 建立新 epoch、旧 epoch 的迟到回复被丢弃、设备被杀后 DEVICE_LOST 且无孤儿进程 | 无 |
| R2-12 报告/诊断/示例 | **DELIVERED** | OBS-01（资源从发射出的代码实测：AXPY 用 4/64 向量寄存器，未测量的成本标 `unknown` 而不是 0；机器指令数与 code segment 交叉核对）、OBS-02（五类用户错误各有稳定码、位置与补救，无裸 traceback）、OBS-03（`Session.capture_failure` 写出可重放 bundle，`tnca-sim replay` 重放出同一语义错误；对照组换成正常 kernel 必须**不**复现）、OBS-04（三个安装后可直接运行的示例） | 无 |
| R2-13 E2E/mutation | **PARTIAL** | E2E-01（216 个用例：3 kernel × 8 个 N × 3 值 seed × 3 内存 seed，参考值全部在 Python 侧独立算出，尾块 sentinel 与两侧 canary 每次核验）、E2E-02（九种 grid + 真实数值 kernel，空 grid 不计算，三个 3D grid 真执行）、E2E-05（三个 seed 下各 16 次发射加一条链）、R2M-01/02/05/05b/06/08 六个变异全部被指定测试抓住 | E2E-03/04/06 与 R2M-03/04/07 的检测器尚不存在 |
| R2-14 双环境/发布 | **NOT STARTED** | — | REL-01…05 |

## 已冻结的合同

| 合同 | 文件 | 关键规则 |
|---|---|---|
| manifest v1 | `spec/r2_manifest_v1.schema.json` | 未知/更高 major **或 minor** 拒绝；未知字段拒绝；重复 JSON key 拒绝；`optional_metadata` 是唯一可忽略扩展位 |
| 容器 v1 | `spec/r2_container_v1.json` | 单文件、无路径、每个长度都是"声明"并按实际文件大小核验；原子写入 |
| ABI v1 | `spec/r2_abi_v1.json` | 六种类型、8 字节 slot、`offset = 8×index`、32 位高半必须为零、f32 按 bit pattern |
| transport v1 | `spec/r2_transport_v1.json` | 单调序号的请求/响应信箱；迟到回复按序号丢弃；重启设备是显式的 epoch 事件 |
| 错误码 | `python/tnca/errors.py` | 码是公开接口：不改名、不复用 |

## 一条必须继续说清楚的边界

合同 1.1 禁止把未迁移到真实 Triton 前端的能力写成"已经支持 Triton GEMM"。
R1 的专用 GEMM、多角色和 rolled pipeline 经 **`tnca-ir` 输入**进入产品路径，
manifest 的 `identity.source_kind` 记录这一点，CMP-05 验证它写的是
`tnca-ir` 而不是 `triton`。
