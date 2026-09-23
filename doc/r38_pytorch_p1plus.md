# R38 PyTorch P1+：经 Inductor 的 PyTorch 支持

版本 0.1，2026-09-22。状态：**IN PROGRESS / NOT ACCEPTED**。
上位：[R38 roadmap §12](r38_isca_roadmap.md)、[Stage 1](r38_stage1_tasks_and_acceptance.md)（替代 S1-10 中的 "P0/P1 小图"）。

## 0. 决定

不做"手写 ATen→kernel 映射"的 P1，直接走 Inductor：`torch.compile` → Inductor 生成的 Triton kernel → TNCA。理由：先手写映射再换 Inductor，最后必然留下两套执行路径。

范围限制：只保证**部分特性、小 shape** 的正确性；大 shape 与覆盖率在能上板之后做（Stage 2）。

## 1. 原则

1. **最小化自我实现**：图捕获、分解、融合、调度、Triton 代码生成全部用 PyTorch/Inductor/Triton 上游；我们只实现上游无法提供的部分。
2. **一条编译路径**：Triton 插件的编译阶段与 `tnca.aot.compile_aot` 调用同一个 TTIR→TNCA 函数；同一 kernel、同一参数经两条入口得到的 `.tnca` 必须**逐字节相同**。
3. **一条执行路径**：所有执行都经 `tnca.runtime.Session`，不另起 runtime。
4. **不静默回退**：任何在 CPU 上执行的计算都要被检测并报告；验收用例要求零回退。
5. **CI 严格**：每个改动都必须让现有全部正确性测试保持通过、周期不变（见 §4）。

## 2. 架构

```text
torch.compile(model)            ← 上游 Dynamo / AOTAutograd / Inductor（不修改）
   │  Inductor config: cpu_backend="triton"
   ▼
Inductor 生成的 @triton.jit kernel + wrapper   ← 上游
   │  triton.compile(...)  ← 上游 JIT / 缓存 / 参数特化
   ▼
Triton 后端插件 "tnca"（entry point triton.backends）    ← 我们：薄层
   compiler.py : BaseBackend，stages = ttir（上游 make_ir）→ tnca（共享 _compile_ttir）
   driver.py   : DriverBase，target=GPUTarget('tnca'), launcher → Session
   ▼
tnca.runtime.Session → RTL（core-sim / core-xpose-v1）   ← 已有
```

张量驻留：Stage 1 中张量是 host（CPU）上的 torch 张量。launcher 按存储（storage）去重，把每个参数的存储上传到设备缓冲，launch 整个 grid，再把存储下载回来。这是明确的 host 往返边界；中间张量留在设备上属于后续优化（P2 的 PrivateUse1 设备），不在本阶段。

## 3. 工作分解与验收

| ID | 内容 | 验收（可执行） |
|---|---|---|
| T0 | 环境：torch 2.14.0（Linux 依赖 `triton~=3.8.0`，与锁定的 3.8.0 配套）写入工具链锁；独立 venv | `python -c "import torch, triton"` 版本与锁一致；现有 CI 所用环境不变 |
| T1 | 共享编译函数 + Triton 插件（compiler/driver，entry point） | R37 O2 149 例经 `kernel[grid](...)` 原生 Triton 启动全部数值通过；其 `.tnca` 与 `compile_aot` 产物逐字节相同；周期与冻结基线相同 |
| T2 | launcher 的张量上传/下载 | 存储别名、非连续张量、storage offset、未写区域不被破坏、标量参数；故障以类型化错误返回 |
| T3 | Inductor `cpu_backend="triton"` 接入 | 小图（逐元素、融合逐元素、归约、softmax、rmsnorm、GEMM、GEMM+RMSNorm+SiLU）数值与 eager CPU 一致，零 CPU 回退 |
| T4 | 回退检测与报告 | 故意放入一个不支持的 op，测试必须报告它而不是静默在 CPU 上算 |

## 4. CI 规则

- 新目标 `make r38-torch`（T1–T4 的全部测试）。
- 任何改动合入前：`r38-l0`、`compiler`、`core`/`core-vrf`/`core-slowmem`/`core-ooo`、`r37`（447/447）、`r38-baseline-check`（周期逐例相同）、`r38-profiler`、`r38-xpose-rtl`、`loop_tests` 全部通过；Linux 打包检查（`r36-build-tests`）在 x86_64 主机上通过。
- torch 只进入 `torchenv`；`TRITON_PYTHON` 所指环境不引入 torch，避免改变现有测试的依赖。

## 5. 实现与状态（2026-09-22）

| 组件 | 位置 | 说明 |
|---|---|---|
| 共享 lowering | `tnca.aot._lower_ttir`、`tnca.aot._seal` | `compile_aot` 与插件共用；重构后 R37 447 个产物与冻结基线逐字节相同 |
| Triton 插件 | `python/tnca/triton_backend/`（entry point `triton.backends: tnca`） | compiler：TTIR 原样进入共享 lowering；driver：launcher 按存储上传/下载 torch CPU 张量，经 `Session` 执行；`libdevice` 只提供语义精确的函数 |
| Inductor 集成 | `tnca.inductor.configure()` | 全部通过上游扩展点：`cpu_backend="triton"`；为 device `cpu` 注册的 pointwise/reduction 启发式（块 ≤ 16）；单一 16×16×16 的 mm 模板配置；`persistent_reductions=False`；`compile_threads=1` |
| 测试 | `make r38-torch` | `plugin_identity.py`（O0/O1/O2 代码段逐字节相同）、`triton_native_run.py`（R37 149 例原生 `kernel[grid]` 启动）、`torch_tests.py`（12 张图） |

编译器为此补的内容（都属于"上游给不了、只能自己做"的部分）：

- `arith.maximumf/minimumf` 归约：用已规定语义的普通 max 归约加 NaN 检测构造（`_reduce_vec`）。
- `llvm.intr.assume`：显式下降为空操作。
- 修正 `arith.cmpf uno` 的条件码拼写（`u` → `uo`），此前从未被触达。

已知限制：

- 块大小上限 16（编译器尚不支持跨多个向量寄存器的宽张量），属于本阶段的小 shape 限制。
- 张量在每次 launch 时上传和下载，没有设备驻留。
- 插件的 Triton 缓存 key 覆盖编译器源码；`TNCA_PROFILE` / `TNCA_OPT_LEVEL` 进入选项哈希。
- JIT 特化会改变部分 kernel 的 TTIR，因此原生启动的周期约一半与 AOT 基线相同，其余不同但数值正确；逐字节相同的保证针对相同 TTIR。
- 含异步张量传输的 kernel 不声明指针别名，参数重叠仍会被 runtime 拒绝。
