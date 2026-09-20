# R2 状态：逐工作包的实际进度

合同：[phase2_tasks_and_acceptance.md](phase2_tasks_and_acceptance.md)。
接口冻结：[r2_interfaces.md](r2_interfaces.md)。
前置：[R1 合同](phase1_tasks_and_acceptance.md)与[R1 状态](r1_status.md)。

本文件只记录**已经跑过并留下证据**的部分。状态只有 DELIVERED、PARTIAL、
NOT STARTED 三种；没有完成度百分比。

**R2 未签收，且按合同现在也不可能签收**：R2 的最终判决要求"已有有效 R1 基线
证据 AND 同一 R2 候选 commit 的 R1 全套再次通过"。R1 的双环境验收正在执行中，
`tools/r2/baseline.py` 在拿到它之前对任何 R2 gate 返回非零——这是 DEP-01 要求
的行为，不是缺陷。

## 逐包状态

| 工作包 | 状态 | 实际交付 | 未满足的验收条件 |
|---|---|---|---|
| R2-01 R1 衔接/合同 | **PARTIAL** | `spec/r2_required_cases.json`：84 条语义用例，展开 profile 与 seed 后 170 个执行 ID；`tools/r2/inventory.py` 校验无空字段、无占位符、无重复；`tools/r2/baseline.py` 实现 DEP-01（缺失/伪造/不完整/PARTIAL 的 R1 全部拒绝，并复用 R1 自己的验证器而不是另写一个）；`doc/r2_interfaces.md` 冻结依赖方向与 adapter 边界 | DEP-02（在 R2 候选上重跑 R1 全套）需要 R1 先通过；DEP-03 的支持矩阵需要实现存在后才能逐条填 |
| R2-02 安装/API | **NOT STARTED** | — | PKG-01…05 |
| R2-03 编译 driver | **NOT STARTED** | — | CMP-01…05 |
| R2-04 MLIR 主路径 | **NOT STARTED** | — | MLIR-01…05 |
| R2-05 artifact/loader | **PARTIAL** | `spec/r2_manifest_v1.schema.json` 冻结 manifest v1；`python/tnca/manifest.py` 实现之（拒绝未知字段、重复 JSON key、更高 major **与** minor；canonical 序列化与 hash）；ART-02 的 schema/完整性半部分全部通过 | 容器格式与 loader、ART-01/03/04/05/06 需要设备 |
| R2-06 动态 ABI | **PARTIAL** | `spec/r2_abi_v1.json` 冻结 ABI v1；`python/tnca/abi.py` 实现之；**ABI-01 对独立 byte-level oracle 全部通过**（六种类型、±0、NaN/Inf、整数边界不截断、高半必须为零、被拒绝的标量类型逐个具名） | ABI-02…06 需要 runtime 与设备 |
| R2-07 缓存/确定性 | **NOT STARTED** | — | CACHE-01…04 |
| R2-08 runtime 生命周期 | **NOT STARTED** | — | RT-01…07 |
| R2-09 grid/admission | **NOT STARTED** | — | GRID-01…07 |
| R2-10 host transport/完成 | **NOT STARTED** | — | MEM-01…06 |
| R2-11 故障/恢复 | **NOT STARTED** | — | ERR-01…07 |
| R2-12 报告/诊断/示例 | **NOT STARTED** | `python/tnca/errors.py` 的 20 个稳定错误码是 OBS-02 的前提 | OBS-01…04 |
| R2-13 E2E/mutation | **NOT STARTED** | — | E2E-01…06、R2M-01…08 |
| R2-14 双环境/发布 | **NOT STARTED** | — | REL-01…05 |

## 批次

合同的实施顺序是五批。当前在**第 1 批**：R2-01 与 R2-05/06 的 schema 合同。

| 批次 | 工作包 | 状态 |
|---|---|---|
| 1 | R2-01、R2-05/06 的 schema 合同 | 进行中 |
| 2 | R2-02、R2-03、R2-04 | 未开始 |
| 3 | R2-05、R2-06、R2-07、R2-08 | 未开始 |
| 4 | R2-09、R2-10、R2-11 | 未开始 |
| 5 | R2-12、R2-13、R2-14 | 未开始 |

## 已冻结的合同

| 合同 | 文件 | 关键规则 |
|---|---|---|
| manifest v1 | `spec/r2_manifest_v1.schema.json` | 未知/更高 major **或 minor** 拒绝；未知字段拒绝；重复 JSON key 拒绝；`optional_metadata` 是唯一可忽略扩展位且 loader 不读 |
| ABI v1 | `spec/r2_abi_v1.json` | 六种类型、8 字节 slot、`offset = 8×index`、32 位高半必须为零、f32 按 bit pattern |
| 错误码 | `python/tnca/errors.py` | 码是公开接口：不改名、不复用；用户错误不得裸 traceback |
| 必跑清单 | `spec/r2_required_cases.json` | 84 条语义用例；比较以展开后的执行 ID 为准，同一 case 跑两次不算两条 |

## 一条现在就必须说清楚的边界

合同 1.1 禁止把未迁移到真实 Triton 前端的能力写成"已经支持 Triton GEMM"。
R1 的专用 GEMM、多角色和 rolled pipeline 是经 **`tnca-ir` 输入**进入产品路径
的，manifest 的 `identity.source_kind` 记录这一点，`doc/support_matrix.md`
的四列继续分开陈述。任何相反的措辞都是 DEP-03 要抓的。
