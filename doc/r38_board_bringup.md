# R38：上板实验准备与执行清单（U50）

版本 0.1，2026-10-07。状态：**准备完成，等待 FPGA 服务器访问**。
上位：[Stage 1](r38_stage1_tasks_and_acceptance.md) §5“进入 Stage 2 的交接物”；板卡流程与既有实测：[flow/fpga/README.md](../flow/fpga/README.md)；测量协议：[archbench/PROTOCOL.md](../archbench/PROTOCOL.md)。

## 0. 现状：板上镜像落后于 r38 RTL

`flow/fpga/README.md` 记录的镜像（2026-09-26 加载，uuid `ccbbcf8b-…`，150 MHz 总线 / 37.5 MHz 设备）由 **fpga-u50 分支**的 RTL 构建。合并进 r38 之后，设备侧 RTL 有以下变化，旧镜像**不包含**：

| 变化 | 文件 | 对上板的影响 |
|---|---|---|
| counter 读口 `perf_sel/perf_rdata`、两个数据口 beat counter、`done_retired` 修正（只报本命令的指令数） | `rtl/tnca_shell.v` | 旧镜像上 grid 的第二个 program 起 `retired` 偏大；counter 不可读 |
| 邮箱 op 7 `PERF_READ`（与仿真 bench 同一契约：r0/r1 为低/高 32 位，越界 REFUSED） | `rtl/fpga/tnca_mmio.v`、`rtl/fpga/tnca_fpga_top.v` | 旧镜像对 op 7 回 REFUSED，runtime 报 `counters: unavailable`（不是 0） |
| 转置写回引擎（`HAS_TENSOR_XPOSE`，默认 0）及未实现 mod 位改为 ILLEGAL_INSTRUCTION | `rtl/core/tnca_tensor.v` | 板上 profile `core-vrf-sync-v1` 不声明该能力，编译器不会发出；关闭时引擎与父提交一致 |
| R3.7 编译器/ISA 覆盖 | 编译器 | fpga-u50 已含 R3.7（squash），无差异 |

因此上板分两条并行线：**A** 用已加载的旧镜像先做功能测试（无 counter）；**B** 用当前 RTL 重建 `.xclbin`（约 2 小时），得到带 counter 的镜像后做正式测量。

## 1. 上板前已完成的准备（本仓库内可验证）

| 项 | 证据 |
|---|---|
| fpga-u50 合并进 r38，板卡 RTL 门禁全过 | `make r36-fpga-tests`：R36-FPGA-01…08（08 为新增的寄存器协议 bench，含 PERF_READ） |
| PERF_READ 经完整板卡软件栈可用 | `tools/r38/profiler_tests.py --quick --backend fpga-u50-sim --profiles core-vrf-sync-v1`：守恒/身份/单调/数值全部 PASS |
| PyTorch → Inductor → Triton → tnca → 板卡协议 全流程 | `python -m archbench.run --platform tnca --backend fpga-u50-sim --profile core-vrf-sync-v1 …`（与板卡同一个 `U50Backend`，同一个 `tnca_fpga_top`，只把 XRT 换成仿真） |
| 后端可由环境切换，不改代码 | `TNCA_BACKEND=fpga-u50`、`TNCA_BACKEND_ARGS='{"xclbin": "<path>"}'`（`python/tnca/triton_backend/driver.py`） |
| 每个 kernel launch 的周期、程序数、传输字节都有记录 | `tnca.triton_backend.driver.launch_log()`，ArchBench 记录的 `time.per_launch`、`host` |
| “期望板卡周期”：板卡设计的 RTL 仿真结果 | `archbench` 在 `fpga-u50-sim` 上的 tiny/small 记录（见 §4 B4） |
| GPU 侧基线 | GB10 上 ArchBench tiny/small/large 记录（`archbench/PROTOCOL.md` §3.2） |

## 2. 拿到 FPGA 服务器后的步骤

所有命令在仓库根目录执行；Vivado/v++/XRT 环境按服务器既有方式加载。

### A. 旧镜像（若仍加载）：功能先行，约 30 分钟

```sh
flow/fpga/program.sh --status                       # 现在加载的是什么、温度
python tools/r36/conformance.py --xclbin <loaded.xclbin> --require-measured
python -m archbench.run --platform tnca --backend fpga-u50 --profile core-vrf-sync-v1 \
    --backend-args '{"xclbin": "<loaded.xclbin>"}' --tier tiny --jobs 1 --out results/u50_old_tiny.jsonl
```

预期：conformance 中 `fpga-u50` 从 N/A 变为 measured；tiny 档数值全部通过；counter 为 unavailable。**板卡只能串行使用：`--jobs 1`。**

### B. 用 r38 RTL 重建镜像，约 2.5 小时（与 A 并行）

```sh
vivado -mode batch -source flow/fpga/package_xo.tcl -tclargs $PWD <build>/pkg
flow/fpga/link.sh <build>/pkg                       # 150 MHz，设备 /4 = 37.5 MHz；link.sh 自己回读并核对频率
flow/fpga/program.sh --check <build>/pkg/tnca_u50.xclbin
flow/fpga/program.sh --go    <build>/pkg/tnca_u50.xclbin   # 唯一改变硬件状态的一步；温度门槛照旧
```

记录：link 报告的 WNS/WHS、资源（LUT/FF/BRAM/DSP）、`.xclbin` sha256 与 uuid，写入 `flow/fpga/README.md` 的新小节。时序不收敛时按该文件“If the link fails on timing again”处理，**不**单方面降频。

### C. 新镜像上的验收阶梯

| 步 | 命令 | 通过条件 |
|---|---|---|
| B1 | `python tools/r36/conformance.py --xclbin X --require-measured` | 5/5 backend measured |
| B2 | `python tools/r38/profiler_tests.py --quick --backend fpga-u50 --backend-args '{"xclbin":"X"}' --profiles core-vrf-sync-v1` | 守恒、身份、单调全部成立（与 `fpga-u50-sim` 同一检查） |
| B3 | `python -m archbench.run --platform tnca --backend fpga-u50 --profile core-vrf-sync-v1 --backend-args '{"xclbin":"X"}' --tier tiny small --counters --jobs 1 --out results/u50.jsonl` | 数值全部通过；无 host 回退 |
| B4 | `python -m archbench.report results/u50.jsonl results/u50sim.jsonl` 与逐 launch 周期对比 | **板卡周期 = `fpga-u50-sim` 周期，逐 launch 完全相同**（同一 RTL、同一固定延迟内存；任何差异都是板卡与仿真模型不一致，必须解释） |
| B5 | `python -m archbench.report results/u50.jsonl results/gpu.jsonl --tier tiny small` | 板卡记录带 `time.device_seconds`（37.5 MHz），首次可算与 GPU 的时间比 S |
| B6 | LLM 演示：`model.llama_decode/small` 连续多步、`model.llama_prefill/small`；截图含 bitstream hash、命令、tokens/s | roadmap §12.4 P2–P3 的板卡证据（P4/P5 需要更大内存，见 §3） |

每次运行保存：JSONL 记录、`program.sh --status` 输出（uuid、温度、时钟）、`xbutil examine` 的时钟计数。

## 3. 已知限制与下一步决策点

- **内存容量**：设备内存是 512 KiB 片上 BRAM（`WORDS=65536`），堆 384 KiB；每次 launch 的参数存储必须放得下，所以 small 档按此设计，large 档（1B 模型）在板上记 `capacity-exceeded`。扩大有两条路：URAM（U50 约 22.5 MB，改 `WORDS/ADDR_W` 与 `tnca_fpga_mem` 的推断，需要重新做时序）或 HBM（新增 AXI 主口与 DMA，工作量大）。需要你决定优先级。
- **传输开销**：Stage 1 每次 launch 经寄存器窗口逐字上传/下载参数存储；板卡上这是 host 时间（单独记录，不计设备时间），会让端到端时延远大于设备时间。张量常驻设备是 roadmap P2（PrivateUse1）的工作。
- **单核串行 grid**：设备时间是各 program 周期之和（`serial-grid`），不是多核并发的 makespan。
- **温度**：上次记录 FPGA 91–95 °C、HBM 86–89 °C，超过 90/85 °C 的停测门槛后经明确指示才继续；这次仍按该门槛执行，超限即停并报告。
