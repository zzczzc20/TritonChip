# R38：上板实验准备与执行清单（U50）

版本 0.4，2026-10-08。状态：**U280 已加载并完成验收阶梯 B1–B6**（acf3019，见 §4、§5）。
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

## 4. 实际板卡：acf3019 上的 Alveo U280（2026-10-07）

提供的 FPGA 主机是 U280 而不是 U50。设备 RTL、kernel（`tnca_u50`）与寄存器协议不变，只换卡相关的值，集中在 `flow/fpga/boards.sh`，由 `TNCA_BOARD` 选择：

| 项 | 值（在主机上读出） |
|---|---|
| 访问 | 经 GB10 主机跳板：`ssh zhongchun@143.89.78.224`，再 `ssh zzhouch@acf3019.ece.ust.hk`（跳板机上的密钥） |
| 主机 | Ubuntu 22.04，24 核，62 GB，XRT 2.16（2023.2），Vivado/Vitis 2021.1 与 2023.2；多人共用 |
| 卡 | `xilinx_u280_gen3x16_xdma_base_1`，BDF `0000:04:00.1`，接口 UUID `fb2b2c5a19ed63593fea95f51fbc8eb9` |
| 平台 / 器件 | `xilinx_u280_gen3x16_xdma_1_202211_1`，`xcu280-fsvh2892-2L-e` |
| 时钟 | 平台 kernel clock 0 默认 300 MHz；本设计同样请求 150 MHz、器件 /4 = 37.5 MHz |
| 后端名 | `fpga-u280`（与 `fpga-u50` 同一个 `U50Backend`） |

命令（在 acf3019 的仓库根目录）：

```sh
vivado -mode batch -source flow/fpga/package_xo.tcl -tclargs $PWD <build> xcu280-fsvh2892-2L-e
TNCA_BOARD=u280 XILINX_VITIS=/tools/Xilinx/Vitis/2023.2 VPP_JOBS=8 flow/fpga/link.sh <build>
TNCA_BOARD=u280 flow/fpga/program.sh --check <build>/tnca_u280.xclbin
TNCA_BOARD=u280 flow/fpga/program.sh --go    <build>/tnca_u280.xclbin      # 见下面的共用规则
python -m archbench.run --platform tnca --backend fpga-u280 --profile core-vrf-sync-v1 \
    --backend-args '{"xclbin": "<build>/tnca_u280.xclbin"}' --tier tiny small --counters --out results/u280.jsonl
```

**构建记录**（`<build>` = `~/tnca_build/u280`，RTL 与本分支 HEAD 相同）：

| 次 | 放置 | 结果 |
|---|---|---|
| 1 | 交给布局器 | kernel 跨 SLR0/SLR1；ap_clk WNS −1.919 ns，32 个端点，全是 AXI-Lite 适配器里的一条路径（`w_have` → `shadow_lo[*]/CE`：一级 LUT、8.2 ns 走线、一次 SLR0→SLR1 跨越）。失败 |
| 2 | `--connectivity.slr tnca_u50_1:SLR1`（`boards.sh`） | synth 44 分钟，impl 3 小时 13 分钟。routed 全设计 WNS +0.016 ns、0 个失败端点；kernel clock 域 WNS +4.229 ns。`.xclbin` 读回 DATA_CLK = 150 MHz，器件时钟 37.5 MHz。`tnca_u280.xclbin` sha256 `f8717adc…a77a52bec` |

第 2 次 v++ 退出 0，但 `link.sh` 当时报失败（退出 3）：它的检查 1 匹配任何 `AUTO-FREQ-SCALING` 规则，而这次打印的是 `AUTO-FREQ-SCALING-08`。这条 advisory 的意思是时序有余量、频率保持原值（“will select the original specified frequency of 150.0 MHz”）。检查 1 现在只放过 08，其余规则 ID（包括以后版本新增的）仍判失败。`LINK_RECHECK=1` 可以对已完成的链接重跑检查而不重跑 v++；该构建复检结果为 `link.sh: OK`。

**共用规则**：这块卡上加载着别人的镜像（2026-10-07 为一个 HBM 带宽测试，CU 空闲），主机上也有其他人的 Vivado 任务。`program.sh --go` 会替换别人的镜像，所以每次加载前确认没有进程占用该设备，并经负责人同意；构建（package/link）不碰卡，`VPP_JOBS` 限制并行度。

期望值：板卡周期应与 `core-vrf-sync-v1` 的仿真 bench 逐 launch 相同（`fpga-u50-sim` 已在 RTL 上证明板卡顶层与该 bench 周期一致）；tiny 档的期望记录由 `python -m archbench.run --platform tnca --profile core-vrf-sync-v1 --tier tiny` 产生。

## 5. U280 上板结果（2026-10-08）

经负责人同意，2026-10-08 09:38 用 `program.sh --go` 加载 `tnca_u280.xclbin`（sha256 `f8717adc…a77a52bec`，xclbin uuid `083faace-4e12-8434-c841-947e12a47ba8`），替换原镜像 `483a6e3f…`。加载前所有门检查通过：设备空闲、无其他加载进程、FPGA 32 °C、HBM 27 °C。加载后 HBM 传感器 2 s 内出现读数，为 26 °C。管理功能上的时钟频率计数器（`clock_counter_freqs`）读出 kernel clock 为 150 MHz，器件时钟因此是 37.5 MHz。

结果记录在 acf3019 的 `~/tnca_build/u280/results/`。

| 步 | 命令要点 | 结果 |
|---|---|---|
| B1 | `tools/r36/conformance.py --backends fpga-u280 --xclbin X --require-measured` | **5/5 PASS**，1 of 1 backend measured。成功 launch 23 周期，oracle 值正确。故障 launch 报 code 0x40 及其 PC，之后 session 仍可用。 |
| B2 | `tools/r38/profiler_tests.py --backend fpga-u280 …`（先 `--quick`，后完整验收） | **PASS**：守恒、单调、无扰动三项均成立，经 PERF_READ 读出；快速版和完整版都通过 |
| B3 tiny | `archbench.run … --backend fpga-u280 --tier tiny --counters` | **27/27 pass**，用时 6.3 分钟，27 个点全部带 counters |
| B3 small | 同上，`--tier small` | **25/25 pass**，用时 7.7 分钟。LLaMA prefill（S = 64）155.6 M 周期 = 4.15 s；decode（P = 128）43.9 M 周期 = 1.17 s |
| B4 | 板卡记录与 `core-vrf-sync-v1` 仿真记录逐 launch 对比 | tiny 27/27 点、132 次 launch，**周期逐 launch 完全相同**。small 档没有 RTL 参照，RTL 仿真太慢。 |
| B5 | `archbench.report u280_tiny u280_small gpu_gb10 --tier tiny small` | 52/52 对 52/52。TNCA U280：E = 0.0072，A = 0.021；GB10：E = 0.0051，A = 0.041。秒级时间比 T_GPU/T_TNCA 在 2×10⁻⁵ 到 6×10⁻³ 之间 |
| B6 | `python -m archbench.generate --tier small --new 8 …` | 64-token prefill 加 7 步 decode，每步都用设备自己产生的 KV cache。8 步全部在容差内，token 序列与纯 FP32 生成完全一致。设备时间：prefill 15.4 tokens/s，decode 1.01 tokens/s。host 墙钟 327 s，其中 194 s 在 launch 内（传输和等待），其余是逐步编译。 |

为跑通 small 档，上板时修了三处，均已提交：

- **`link.sh` 误判 `AUTO-FREQ-SCALING-08`**：见 §4。
- **`program.sh` 读不出 U280 的温度**：XRT 2.16 的 thermal 报告是表格格式，原解析只认 `FPGA : 45 C` 这种格式。现在两种格式都能读。
- **运行时堆分配器碎片化**：释放的块不合并，LLaMA small 跑到图的后半段时报 “out of device memory: 131072 bytes needs 131072 and 98304 remain”。合并后，每次 launch 结束堆都回到空，tiny 档周期与修复前完全相同。

另外，conformance 套件此前对每个后端都先构建仿真器，所以在没有 iverilog 的板卡主机上跑不起来。现在只为需要仿真器的后端构建。

**下一步**（详见 [architecture_overview.md](architecture_overview.md) §13 与 [compiler_overview.md](compiler_overview.md) §15）：

1. 从 U280 链接报告提取 kernel 单独的资源占用。
2. 板卡功耗（`xbutil examine -r electrical`）。
3. 把板卡证据登记进 evidence registry（Stage 2 等级）。
4. 内存容量路线（URAM 或 HBM）待负责人决定。

## 6. 器件时钟：37.5 MHz 的瓶颈与第一轮优化（2026-10-08）

### 6.1 瓶颈在哪里

U280 链接的 routed 报告（`hw_bb_locked_timing_summary_routed.rpt`）里，clk_dev（26.668 ns）最差路径的余量只有 2.689 ns：

- **起止点**：`u_mtx/g_mac[12].u_mac/u_fma` 中 `p1_mag_reg → p0_exp_common_reg`。
- **规模**：60 级逻辑（19 个 CARRY8、一个 DSP），数据路径 23.931 ns，其中走线占 73 %。
- **含义**：FMA 第 2 级（前导 1 搜索、规格化、舍入）的结果经累加器文件的写→读直通，进入下一次乘加的第 0 级。在第 0 级里，它经过 24×24 乘法（DSP）和「乘积是否为 0」的判断，最后决定移位方向和公共指数。整个过程在一个周期内完成。

### 6.2 方法：几分钟一次的 OOC 综合，而不是四小时的链接

- **工具**：`flow/synth/vivado_ooc.sh`（`make fpga-timing`）。对单个单元或整个器件（`tnca_fpga_top`）做 out-of-context 综合加时序估计。单元级 3–5 分钟，整器件约 25 分钟。
- **注意**：报告的是布线前的估计。矩阵这条路径综合后估计为 14.7 ns，链接后实测为 23.9 ns，比值约 1.6，下文都按这个比值折算。
- **yosys**：用 yosys 映射 UltraScale+ 原语后再数级数的办法试过了，不可用。它不推断 CARRY8，`ltp` 也不把 FDRE 当作寄存器边界。yosys 仍用于 `make synth` 的通用门结构代理。

### 6.3 改动（都不改变任何周期数）

| 改动 | 文件 | 综合后估计 |
|---|---|---|
| 去掉累加器文件的写→读直通。矩阵引擎读第 m 行后 FMA_LAT 拍才写回，同一行下一次被读是 ROWS+1 拍以后；行操作和 `a.vread` 都在上一条命令写回之后才开始，所以这条直通从未传过被使用的值。`tnca_matrix` 在仿真里检查这一点（`acc_collide`），违反时测试直接失败。 | `rtl/core/tnca_acc_rf.v`、`rtl/core/tnca_matrix.v` | 矩阵单元 14.7 → 8.6 ns |
| FMA 改由 XLS 生成（见 [`xls.md`](xls.md)）。`xls/tnca_fma.x` 不使用会把次正规数清零的 `apfloat`，算法与原手写版逐步对应，流水级由 XLS 排定。`LAT=2` 的实例与手写版的级数和延迟相同。 | `xls/tnca_fma.x`、`rtl/gen/tnca_fma_xls.v`、`rtl/core/tnca_fp32_fma.v` | 向量 FPU 9.2 → 6.8 ns |
| tensor 引擎的窗口检查原来是五个块维度连乘再比较，现在在 `td.load` 读入描述符尾部时每字一次乘法，预先算出 box 元素数（饱和到 2^20），传输时只做移位和比较。 | `rtl/core/tnca_tensor.v` | 整器件最差路径原为 10.1 ns |

验证：

- `unit`：XLS FMA 两个延迟各 41 240 条向量。
- `fma-deep`：两个延迟各 516 240 条向量，与精确 Fraction oracle 0 差异。
- `core`、`core-vrf`：各 31/31，包括 tensor 描述符 t12/t17/t33–t35。
- 其余：`r38-xpose-rtl`、`top`、`e2e` 全部通过，周期数与改动前逐一相同（例如 t10_mesh 18 591）。

### 6.4 频率

- **整器件结果**：三项都做完后，整器件（`tnca_fpga_top`，以 13.333 ns 为目标，U280 器件）综合后最差路径为 **10.19 ns，40 级**，WNS +3.08 ns，525 817 个端点全部满足。这条路径从 VRF bank 的 BRAM 读口出发，经矩阵单元选出 A 元素、做格式扩展，进入 FMA 第 0 级。原来 60 级、14.7 ns 的矩阵回路和 10.1 ns 的 tensor 窗口检查都已不在最差路径中。资源：LUT 242 474（18.6 %），寄存器 181 133（7.0 %），BRAM 167.5。
- **新构建默认 /3**：`rtl/board/tnca_u50_xo.v` 的 `DEV_DIV` 参数默认改为 3，即 150 MHz / 3 = 50 MHz，周期 20 ns。按 1.6 倍折算，布线后约 16.3 ns（约 61 MHz），余量约 3.7 ns。（2026-10-09 起默认改为 /2，见 §6.10。）
- **跨时钟域不受影响**：适配器用 toggle 加同步器跨域，与分频比无关；分频后时钟的边沿都与 ap_clk 对齐，跨域约束仍是一个 ap_clk 周期。
- **以链接结果为准**：`link.sh` 在时序不满足时会报失败，不会自行降频。
- **当前加载的镜像（uuid `083faace…`）仍是 /4、37.5 MHz**。`archbench/platforms/tnca.py` 的 `BOARD_CLOCK_HZ` 描述的是已加载的镜像，要等新镜像加载时一起改。

### 6.5 再往上：75 MHz（150 / 2，13.3 ns）

按 1.6 倍折算，75 MHz 需要综合后不超过约 8.3 ns。现在剩下的两处：

- **矩阵单元，10.2 ns（整器件最差）**：VRF 的 BRAM 读出后，经 A 元素选择和格式扩展直接进入 FMA 第 0 级；单独综合矩阵单元时，累加器读口进入第 0 级为 8.6 ns。可以在 A 操作数进入 FMA 前加一级寄存器（MAC 扫描只多 1 拍排空，累加回路不受影响），或者用 `LAT=3` 的 XLS FMA（单元级 7.8 ns）。
- **向量 FPU**：`LAT=3` 时为 5.8 ns。

`LAT=3` 只需改核心的 `FP_LAT`，FMA 本身只是换一个生成实例。代价是浮点结果晚一拍，需要重新冻结周期基线（`spec/r38/baseline_cycles.json`），并更新目标描述里的 `fp_result_cycles`。这属于下一步，应该作为一条 design change 记录在 `spec/r38/design_changes/`。

### 6.6 第二轮：向 75 MHz（2026-10-08 夜）

- **矩阵操作数寄存器（timing-002）。** 在 `tnca_fpu_multi` 的格式扩展与 FMA 之间加一级寄存器（`IN_REG`）；矩阵单元的 `OPND_REG` 默认打开，结果标签流水与排空相应延长一拍。整器件（75 MHz 目标）综合后估计从 10.19 ns 降到 **9.30 ns**，WNS +4.02 ns。原来 2 672 个端点的 VRF→MAC 一组已低于 8.9 ns。
- **周期代价。** 每条 `m.mma` 多 1 拍。R37 O2 的 149 个用例中有 14 个变化，各 +1 到 +4 拍，全部数值正确。
- **冻结基线修订。** 重新冻结时发现：r38-tdesc 改变了 17 个用例的周期，并改变了 profile 配置哈希（能力表新增 `tensor_store`），而当时只修订了 streamk 的 4 项。漏掉的原因是 R2 检查的输出被截到了最后几行。现在全部补齐，并在 `spec/r38/baseline_cycles.json` 中按来源分条记录。配置哈希的变化也解释了 `r38-identity` 的 0/447。
- **重新测量周期时要先重新暂存 RTL。** 直接运行 `tools/r38/baseline.py --run` 不会重新暂存 `python/tnca/data/rtl`，设备仿真器会沿用旧 RTL。`make r38-baseline-check` 会先暂存，不受影响。
- **剩下的最差一组**：VALU 的单周期带掩码整数归约（`x_pm_idx → w_sdata`，9.30 ns，37 级）。它在一个单元内部，扇出小，布线比例可能低于矩阵那条路径的 1.6。
- **75 MHz 链接。** 第一次以 `DEV_DIV = 2` 的链接已于 2026-10-08 23:58（HKT）在 acf3019 上启动，源码为本节的状态。布线后的结果决定归约是否也要拆成两拍。
- **整数归约与 FPU 操作数选择（timing-003）。** 归约改成三棵树，只在末端选一次：加法、按位运算、最大/最小各一棵。最大/最小统一为对「保序变换后的键」取无符号最大值，有符号时翻转符号位，取最小时整体取反，各运算的单位元都映射到键 0。向量 FPU 的归约操作数选择不再等待核心的发射判断 `start`。两项都不改变周期。新增 t36_reduce：8 种整数归约 × 8 组向量 × 8 种掩码（含空掩码），加上广播形式，共 544 项检查。它在旧版和新版 VALU 上都通过。整器件综合后估计从 9.30 ns 降到 **7.76 ns**（WNS +5.56 ns），最差路径变为写回选择。第二次 75 MHz 链接于 2026-10-09 00:45（HKT）用这个版本启动，与第一次并行。

### 6.7 再往上：100 MHz 与 150 MHz 需要什么

以 150 MHz（6.667 ns）为目标对整器件做综合后估计（timing-002 之后、timing-003 之前的 RTL），按「起点模块 → 终点模块」分组，得到每组的最差延迟：

| 延迟 (ns) | 端点 | 组 | 性质 |
|---:|---:|---|---|
| 9.30 | 544 | VALU 整数归约 → 标量写回 | timing-003 已降到 7.76，起点变为 W 阶段写谓词时的写后直通 |
| 8.25 | 3 184 | 发射译码 → 向量 FMA 第 0 级 | timing-003 已去掉 `start` 依赖 |
| 7.47 | 467 | X 阶段 → VRF bank 控制（停顿链） | 29 级，走线占 79 % |
| 7.18 | 1 917 | tensor 引擎 → 片上存储 | 访存地址与请求 |
| 7.00 | 1 062 | LSU → 片上存储 | 同上 |
| 6.64 | 4 | 核心 → VRF | |
| 6.35 | 1 488 | tensor 引擎内部 | 地址累加、窗口 |
| 6.17 | 1 136 | MAC 内 FMA 级 | `LAT=3` 的 XLS 实例可降 |
| 6.06 | 112 | 向量 FMA 级 | 同上 |
| 5.74 | 9 659 | MAC → 累加器写入 | |

按 1.6 倍布线比例换算：

- **100 MHz（10 ns）**：综合后需在约 6.25 ns 以内。还需要处理前 7 组，其中归约必须拆成两拍，停顿链和访存请求要加寄存器。这些都会改变周期，各自需要一条 design change，每组还要重新验证。
- **150 MHz（6.67 ns）**：综合后需在约 4.2 ns 以内。上表每一组都超出，FMA 需要 4–5 级流水（`FP_LAT` 4–5），存储、VRF、发射逻辑都要重新分级。这是全核心的重新流水化，不是一个夜里能完成的工作。

因此本轮以 /2（75 MHz）为目标在 U280 上验证。100 MHz 可以作为下一步，按上表逐组推进。

（这个判断过于悲观：§6.8 的第三轮没有改变任何周期，就把上表除 FMA 以外的各组都降到了 6.2 ns 以内。）

### 6.8 第三轮：不改周期，整器件到 6.84 ns（2026-10-09 凌晨）

仍以 150 MHz（6.667 ns）为目标做整器件综合后估计，五项改动都不改变周期，结果如下：

| 改动 | 处理的组 | 之前 (ns) | 之后 (ns) |
|---|---|---:|---:|
| timing-004：整数 min/max 归约改为所有车道两两并行比较，再用独热选择取出胜者 | VALU 归约 → 标量写回 | 7.76 | 6.71 |
| timing-005：FMA 阵列的控制（操作数形式、取负、只乘、归约覆盖）在 R 级译码，随操作码一起寄存进 X | 发射译码 → 向量 FMA 第 0 级 | 7.17–7.42 | 6.83 |
| timing-006：LSU 每个车道的字地址、字节使能和移位后的写数据在 S_SETUP 一次算好；规划器只选字、做或 | LSU → 片上存储 | 5.6–11.0 | < 5.9 |
| timing-007：VALU 的立即数选择在 R 级译码；有符号高半部的修正项与乘法并行形成，乘法后只剩一次减法 | VALU 乘法 → 向量写回 | 6.71 | 6.07 |
| timing-008：tensor 引擎的越界判断改为一个 36 位偏移与按传输寄存的常量比较；请求的地址、标签、使能按状态而不是按「哪个请求有效」选择 | tensor 引擎 → 片上存储 | 7.20 | 5.73 |

几点说明：

- **LSU 那一组的 11.0 ns 不是 RTL 变化造成的。** 加入 timing-004 的那次综合里，LSU 规划器在 16 个车道上带条件的累加被综合成了 25 级的串行链；同一份 LSU 代码在另外几次综合中只有 5.6 ns。这说明原结构依赖综合器的选择，timing-006 去掉了这种依赖。
- **等价性。** timing-008 的每个边界判断都是把原不等式两边移项，前提是没有操作数接近 2^63，地址和偏移远达不到这个范围。边界只依赖 S_TCHK 设置的 t_CB、t_E、t_row_bytes，S_TCHK 和数据搬运之间至少隔一个 S_TMUL 周期，所以每拍更新的寄存副本在搬运开始时已是最新值。t33（672 项）、t34（400 项）、t35 和转置路径都通过。
- **验证。** 三种内存模型下的 core、core-vrf、top、nuca、mesh、r38-xpose-rtl 都通过；R2 周期检查对 149 个用例两次运行都通过，周期不变。

之后又加了一项：**timing-009**。向量 FPU 的格式转换器原来是组合逻辑，接在 FP_LAT 延迟线第 0 级前面（6.44 ns，71 % 是走线）。现在它的输入在第 0 级旁寄存，结果从第 1 级进入延迟线。转换结果本来就要等 FP_LAT 拍，所以周期不变：t21_cvt 前后都是 1942 拍，R2 周期检查 149/149。

**timing-010** 处理 banked VRF。操作数读取原来要等整条停顿判断（`stall_other`）结束才向各 bank 发请求，这条路径在 bank 仲裁器之前，6.17 ns，77 % 是走线，而走线为主的路径在布线后增长最多。会让读到的值过期的只有一种情况：某个向量源的生产者还在流水线中，而所有更早的写者都在 scoreboard 里。所以读取现在只等向量源冒险。
- **周期。** core-sim 配置不变（R2 冻结基线所用，行为级寄存器文件）。banked 配置（板上用的）有 5 个 core 程序快 1–14 拍，没有变慢的。
- **板上与仿真对比。** 包含这项改动的镜像需要重新生成 B4 的仿真参考。
- **时序。** 加上 timing-009，VRF 和转换器两组都不再出现在 5.4 ns 以上的列表中。

现在整器件综合后最差为 **6.84 ns**（150 MHz 目标下 WNS −0.19 ns，538 791 个端点中 80 个失败）。6 ns 以上的组全部在生成的 FMA 内部或紧邻它：向量 FMA 第 0 级 6.84，MAC 的 FMA 级 6.17，FMA 输出到写回 6.15，FMA 内部 6.06。再往下就要加深 FMA 流水（`FP_LAT = 3`，MAC 的 `LAT = 3`），这会改变周期。按 1.6 倍布线比例，6.84 ns 约等于布线后 11 ns，即不分频约 90 MHz。实际比例由两次 75 MHz 链接的布线结果给出。

### 6.9 不分频的尝试：100 MHz（2026-10-09）

- **配置。** 用 5baba1e（第三轮）的 RTL，`DEV_DIV = 1`，kernel clock 设为 100 MHz：`package_xo.tcl` 的 `AP_CLK_HZ`、`ooc.xdc` 的周期、`link.sh` 的 `EXPECT_MHZ` 都改为 100。这些修改只在 acf3019 的构建副本 `~/tnca_srcND` 里，仓库里的流程仍是 150 MHz kernel clock 加分频。2026-10-09 02:57（HKT）开始链接，与两次 75 MHz 链接并行（`VPP_JOBS=4`，避免三个链接同时综合时内存不够）。
- **为什么不分频是安全的。** `tnca_axi_lite` 用 toggle 加同步器跨时钟域，clk_dev 的每个沿都是 ap_clk 的沿，所以分频比为 1 时跨域要求不变。
- **如果 100 MHz 没有收敛。** v++ 会自动把 kernel clock 降到能收敛的频率，仍然写出 `.xclbin`；`link.sh` 把这种情况判为失败，因为它不是所请求的设计。分频比为 1 时器件时钟就是 kernel clock，这样得到的镜像在降频后的频率上仍然可用，只要把 `BOARD_CLOCK_HZ` 设为读回的频率即可。
- **整器件布线后的 WNS。** 两次 75 MHz 链接在布局后都是 −0.05 ns，布线初始化估计都是 +0.015 ns。它们的器件 RTL 综合后相差 1.5 ns，这两个数却相同，说明这是平台自己的路径，与 37.5 MHz 那次链接的 +0.016 ns 是同一类。决定成败的是 clk_dev 域。
- **布线迭代中的负余量。** 进入 rip-up-and-reroute 后两次链接都出现负 WNS：第一次（timing-002）三轮迭代从 −1.087 降到 −0.757 ns，第二次（timing-003）首轮为 −1.124 ns。在最终报告出来前还不知道是哪个时钟。如果是 clk_dev，这两版 RTL 都可能被自动降频到 75 MHz 以下。
- **备用 /2 链接。** 04:50（HKT）停掉第一次链接，它的 RTL 最旧也最慢，另外两版都严格更好。腾出的内存用来启动 d72731d（第三轮加 timing-009，综合后 6.84 ns，比第二次链接的 RTL 快约 0.9 ns）的 /2 链接：构建目录 `~/tnca_build/u280_75c`，源码 `~/tnca_srcR3`。

### 6.10 75 MHz 在 U280 上跑通（2026-10-09）

- **镜像。** 第二次 /2 链接（timing-003 的 RTL）于 05:41 完成，`LINK_EXIT=0`，v++ 只报告 AUTO-FREQ-SCALING-08（时序有余量，保持请求的频率）。`.xclbin` 读回 DATA_CLK = 150 MHz，sha256 `16813fa9…8cc3cee`。
- **布线结果。**
  - 全设计 WNS +0.016 ns，即平台路径。
  - **clk_dev WNS +0.471 ns**（13.333 ns 周期），498 254 个端点无失败。
  - 最差路径是 `x_pm_idx → w_vdata`：12.68 ns，28 级，走线占 80 %。这条路径综合后约 7.8 ns，布线比例约 1.6，与此前的估计一致。
- **上板。** `program.sh --go` 的门检查全部通过：设备空闲、无其他加载、FPGA 31 °C、HBM 26 °C。
  - B1 conformance PASS：oracle、同一会话第二次启动、故障与恢复。
  - B2 profiler PASS。
  - B3 ArchBench tiny 27/27 通过。
  - B4：27 个点的器件周期数与 RTL 仿真（`fpga-u50-sim`）**逐点完全相同**，从 op.add 的 1 776 拍到 llama_prefill 的 5 323 042 拍。工具为 `tools/r38/board_vs_sim.py`。
- **时钟实测。**
  - 管理功能上的频率计数器读出 kernel clock 为 150 MHz。
  - `flow/fpga/measure_clock.py` 用两个长度不同的自旋 kernel，由器件自己的周期数除以主机计时，得到 **74.965 MHz 和 75.017 MHz**。循环每次迭代恰好 6 拍。
- **默认改为 /2。** `DEV_DIV` 默认 2，ArchBench 的 `fpga-u280` 时钟改为 75e6。`link.sh` 结束时的「device clock」原来固定按 /4 计算，现在从封装的包装层读出 `DEV_DIV`。
- **记录。** 所有文件、镜像哈希和布线路径都在 `evidence/r38/u280_75mhz/`。
- **之后的 RTL 更快。** 当前 RTL（第三轮加 timing-009）综合后为 6.84 ns，比上板的这一版快约 0.9 ns，周期完全相同。按同样的 1.6 倍，布线后约 10.9 ns，/2 的余量约 2.4 ns。不分频（100 MHz）和这版 RTL 的 /2 链接（`u280_75c`）仍在运行。

### 6.11 再往上要付周期：FMA 加深一级（`FP_LAT = 3`）的代价

timing-010 之后，综合后 6 ns 以上的组全部是生成的 FMA 本身：

| 组 | 延迟 (ns) |
|---|---:|
| 向量 FMA 第 0 级 | 6.84 |
| MAC 的 FMA 级 | 6.17 |
| FMA 输出到写回 | 6.13 |
| FMA 内部 | 6.06 |

FMA 的三级已经大致平衡（各约 6.1–6.8 ns），不改周期就压不下去。`FP_LAT` 同时决定向量 FPU 和矩阵单元的 FMA 延迟（`FMA_LAT(FP_LAT)`），XLS 生成的 4 级实例（`tnca_fma_l3_xls`）已经存在。下面在一个临时副本里把 `FP_LAT` 改为 3 测周期代价：

- **正确性。** 全部 core 程序通过。
- **R2 的 149 个用例。** 59 个变慢，90 个不变，没有变快的。变化的用例合计 +0.91 %，中位数 +1.8 %，最大 +9.0 %（moe_topk_gating；softmax/n37 为 +7.7 %，linear_self_attention/out 为 +7.3 %）。代价集中在 FP 依赖链较长的小 kernel 上。
- **收益（实测，综合后）。** 同一副本（含 timing-009/010）以 150 MHz 为目标做整器件综合后估计：**WNS +0.274 ns，543 528 个端点无失败**，即综合后已满足 150 MHz。最差为 6.22 ns，是 tensor 引擎在 S_TMUL 中的描述符乘法；其次是 core → core 6.09 ns 和向量 FPU 内部 6.00 ns。FMA 各组都降到 5.8 ns 以下。和 `FP_LAT = 2` 的 6.84 ns 相比约快 10 %，换来 1–2 % 的周期。
- **改它的成本。** 周期变化意味着冻结基线要修订 59 项，目标文件的 `fp_result_cycles` 和配置哈希都会变，B4 的仿真参考也要重新生成。

按 1.6 倍布线比例，6.22 ns 布线后约 10 ns，不分频约 100 MHz。为了把实测数据摆在决策前面，用这个配置（`FP_LAT = 3`、`DEV_DIV = 1`，kernel clock 请求 125 MHz）于 06:51 启动了一次链接（`~/tnca_build/u280_fl3`，源码 `~/tnca_srcFL3`）。同时在 GB10 上生成相应的 B4 仿真参考。仓库默认仍是 `FP_LAT = 2`：是否用周期换频率，由负责人决定。

150 MHz 不分频按 1.6 倍布线比例要求综合后约 4.2 ns。上面每一组都超出，包括片上存储自身的 BRAM 级联（`u_mem → u_mem` 5.37 ns，需要输出寄存器，访存延迟 +1）。这需要整核心的重新分级，不在本轮范围内。

