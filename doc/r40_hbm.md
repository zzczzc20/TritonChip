# R40-04：单核 HBM 与外存接口（设计与验证记录）

合同见 `doc/r40_tasks_and_acceptance.md` 的 R40-04；本文记录实现与证据，不改合同。

## 1. 接口：核心的数据口不变

核心一直只有一个数据口（`tnca_shell` 的 mem_req / mem_rsp，见 `rtl/fpga/tnca_fpga_mem.v`）。它的语义是：
- valid/ready 请求带 tag、byte enable、atomic；每个请求一定有一个响应。
- 按**接受顺序**生效；响应顺序任意，且不可反压。

R40-04 不改这个口，只在它后面放一个桥：
- 低于 `LOCAL_LIMIT`（0x20000）的地址留在片上 RAM：代码区、launch 参数区，以及 runtime 放在 heap 之下的东西。取指照旧只走片上 RAM。
- 从 `LOCAL_LIMIT` 起是 AXI 窗口，设备地址 A 对应 AXI 地址 `axi_base + (A − LOCAL_LIMIT)`。
  - `axi_base` 与 `axi_size` 由 host 写入 `R_AXI_BASE`（0xA8）和 `R_AXI_SIZE`（0xB0）。
  - 超出窗口的访问和片上 RAM 越界的答法相同：读得 0，写丢弃。

这样，HBM 协议止于板级 adapter，核心、ISA、artifact 都不知道 HBM 的存在（§2.1 的分层）。

新增寄存器（只在 `DATA_AXI` 镜像中有，旧镜像读为 0，runtime 据此识别镜像）：

| 偏移 | 名称 | 内容 |
|---|---|---|
| 0xA8 | AXI_BASE | 窗口在 AXI 空间的基址（host 写） |
| 0xB0 | AXI_SIZE | 窗口字节数（host 写；mailbox 的范围检查以 `LOCAL_LIMIT + size` 为界） |
| 0xB8 | AXI_STAT | bit0：粘滞的错误响应；bit1：有窗口；bit2：粘滞的无主响应；63:32：LOCAL_LIMIT，以 4 KiB 页计 |
| 0xC0 / 0xC8 | AXI_RD / AXI_WR | AR、AW 握手计数（自复位起计） |
| 0xD0 | AXI_LAT | 每拍在途读数的累加；除以读数即平均读延迟（Little 定律） |
| 0xD8 | AXI_RBEAT | R 拍数；与 AXI_RD 之比即 burst 长度 |

公共设备时钟扩展（2026-10-13，当前在建的 `ac580b0` 镜像尚不包含）：

| 偏移 | 名称 | 内容 |
|---|---|---|
| 0xE0 | TIME_CAP | 只读版本 1；旧镜像返回 0，表示不可用 |
| 0xE8 | TIME_START | 最近一次接受 LAUNCH 的公共时钟时间戳 |
| 0xF0 | TIME_END | 同一次 LAUNCH 的 completion 到达 MMIO 的时间戳 |
| 0xF8 | TIME_SEQ | 对应请求序号；新 LAUNCH 尚未完成时为 0 |

64 位计数器自复位起每个设备时钟递增，包括 idle、提交间隙和背压。
PING、QUIESCE 和 PERF_READ 不覆盖 LAUNCH 时间戳。读取必须先后两次
校验 TIME_SEQ，并与预期请求序号匹配；复位前后不可组合，同一次
backend epoch 内的差值按 64 位模数计算。`U50Backend.launch_timestamps`
实现此读取，旧镜像返回 None，序号过期、正在运行及未知版本明确拒绝。

`make r40-device-clock`：host 单测 6/6，FPGA register protocol 实际
RTL 测试通过，证明两次 launch 之间的 host 工作被包含在公共时钟中，
graph 时间跨度大于两次 launch 各自跨度之和。该检查点尚未连接
Session/ArchBench 的 graph makespan 采样，也没有板上计时证据。
原来的 serial-grid 周期和 request-latency 记录继续保持原义。

## 2. 两个桥（`burst` 机制的 A/B）

`tnca_fpga_top` 的 `AXI_LINE` 选桥：0 为单拍桥，>0 为行缓冲桥，两者端口相同。

### 2.1 `tnca_axi_bridge`：单拍，容量那一步

每次访问是一个 8 B 的 AXI beat（INCR，len 0）。
- **顺序**：在途的 AXI 访问记在 OUT_N 项的表里。新请求若与表中某项同字，且其中一方是写，就等它完成。两个同字的读可以重叠。
  - 写只用一个 ID，按发出顺序完成；读以 tag 为 ID，可乱序完成。
- **atomic**：AXI4 没有 `tnca_amo.vh` 那种原子操作，所以在桥里做读-改-写。
  - 先排空表，再读、算新值、写回，写完成后返回旧值。期间不接受新请求。
  - 新值用 `amo_new` 计算，与片上 RAM、home slice 是同一个函数。
- **复位**：复位会忘掉在途请求，但 AXI 侧可能没有一起复位，旧响应仍会回来。
  - 复位后 QUIET（1024）拍内不接受请求，并把 R/B 全部收下丢弃。
  - 此后任何对不上在途请求的 R/B 也丢弃，并置 `axi_stray`。
  - 请求一侧（AW 已被收、W 未被收）无法撤回，因此由平台把 interconnect 与本模块一起复位。U280 上只有装载镜像时复位。

### 2.2 `tnca_axi_bridge_lb`：按行读

窗口按 LINE_BEATS×8 B 的对齐行读：一行一个 INCR burst（ARLEN = LINE_BEATS−1，ARID = 槽号），进入 LB_N 行的全相联缓冲。

读的处理：
- 命中缓冲的读，由缓冲作答。
- 所在行正在取回的读，等这一行。
- 未命中的读，分配一个槽并发起取行；同时预取其后 PF 行（不越过窗口末端）。
- 三种情况都以 waiter（WAIT_N 个）接受请求，行到齐即作答，顺序任意。

保持接受顺序的规则（每条都有一个变异测试，见 §4）：
1. 写仍是单拍、写直达，并更新缓冲里该行的副本，之后的读看到它。
2. 写不进入正在取回的行，否则取回的数据会把写覆盖。
3. 写不越过同一行上未作答的 waiter：那个读先被接受，不能看到这次写。
4. 某行有写在途时不发起取这一行，否则 AXI 可能让读超过写。
5. atomic 等到全部在途访问完成、所有 waiter 作答，再把自己那一行移出缓冲，然后照 §2.1 做读-改-写。读的 ID 为 LB_N。
6. host 的 DMA 绕过桥直接写 HBM。`tnca_mmio` 每服务一条命令就拉一拍 `inv`：
   - 已到的行作废；在途的行到达时丢弃。若某行恰在 `inv` 这一拍完成，直接作废。
   - host 的 DMA 之后总跟着一条命令（UPLOAD，然后 LAUNCH），且从不与 kernel 并发。
7. 写经过每个通道一个 4 项的 FIFO，每拍可接受一个写。单拍桥要等上一个写的 AW 与 W 都被收下，最多两拍一个。

## 3. Runtime 与板卡

- **仿真**：内存模式 `core-vrf-sync-v5/hbm`（`python/tnca/device.py`）。
  - 架构与 v5 相同：编译、能力、artifact 都按 `base_profile` 走；内存模式只决定 session 在哪个 bench 上运行。
  - bench 的 `DATA_AXI=1`，`tb/common/tnca_axi_mem_model.v` 提供 32 MiB 窗口，时序由 `+axi_*` plusarg 控制（`TNCA_DEVICE_PLUSARGS`）。
  - `TNCA_RTL_PARAMS=AXI_LINE=8,...` 选行缓冲桥。
- **板卡**：`rtl/board/tnca_hbm_xo.v`、`fpga/kernel_hbm.xml`，由 `package_xo.tcl` 的 `hbm` 变体打包。
  - 设备直接运行在 ap_clk 上，不经 BUFGCE_DIV。m_axi 与平台 interconnect 同钟，接口是同步的。
  - kernel 参数 `hbm` 位于偏移 0xA8，即 R_AXI_BASE。
  - `link.sh` 的 `LINK_EXTRA="--connectivity.sp tnca_u50_1.hbm:HBM[0:31]"` 把 m_axi 接到全部 32 个 PC 组成的组。
- **runtime**（`python/tnca/backends/u50.py`）：看到 `AXI_STAT.bit1` 后：
  - 从 xclbin 的 (GROUP_)CONNECTIVITY 与 MEM_TOPOLOGY 找出该参数所连的 bank（最大的那个，即组），用 `xclAllocBO` 在其中分配窗口（`TNCA_HBM_BYTES`，默认 1 GiB）。
  - 写 `R_AXI_BASE` 与 `R_AXI_SIZE`，回读校验。Session 的 heap 以设备报告的 `LOCAL_LIMIT + size` 为界。
  - 完全落在窗口内的传输经 `xclWriteBO`/`xclSyncBO` 走 DMA，其余仍走寄存器窗口。
  - 每次 launch 后检查 `AXI_STAT` 的粘滞位，有错即报 DeviceLost。

## 4. 验证（不需要板卡的部分，`make r40-hbm`）

- **桥**：`tb/fpga/tb_axi_bridge.v`。随机流量对照按接受顺序更新的参考内存，地址池很小，以便频繁冲突。
  - AXI slave 尽量不配合：读数据在发 R 时才采样，写在发 B 时才生效；延迟随机，`+reorder` 时乱序并交织 burst，READY 按 `+bp`% 随机撤下。
  - 模式：`+err`（部分字返回 SLVERR）、`+rst`（在途时复位 4 次，slave 不复位）、`+dma`（静止时在桥背后写内存并拉 `inv`）。
  - 7 个用例 × 6 种构造（单拍桥 OUT_N 8/16；行缓冲 8/4/1、2/2/0、4/8/2，以及选定的 8/16/1、32 waiter、OUT_N 16，即 beats/槽/预取）。
- **变异**：单拍桥 5 个，行缓冲桥 6 个。每个都必须被所指定的用例抓到。
- **设备**：`tools/r40/device_tests.py` 在 `/hbm` 上全部 16 项。
  - 在恶劣时序（`+axi_lat 20..120 +reorder +bp 30`）下另跑 5 项，单拍桥与行缓冲桥各一遍。
  - 一张图的张量位于窗口 24 MiB 处。
  - 窗口内 32 个样本全部写入后再读回。地址若丢失高位，两个样本就会混在一起。
- **lint**：`r36-fpga-tests` 以 `DATA_AXI=0/1` 两种方式展开顶层，两个桥都在根层展开检查。

## 5. 尚待板卡的部分

- `R40-04.hbm`：在 U280 HBM 镜像上用 `tools/r40/hbm_board.py` 跑三项：窗口、整窗样本（按 PC 步进）、≥64 MiB 的标准 Torch 图（附计数器）。
- `R40-04.bandwidth`：同一工具的 `stream` 给出带宽账本的输入。50% 门限还取决于核侧，见 §6。
- `R40-04.robustness`：板上数据正确、post-route 时序与 CDC，以及用板上延迟分布校准内存模型（§3.4）。

## 6. 带宽：瓶颈在哪里

测量方法：`/hbm` 仿真上的三张标准 Torch 图，fp32，16384 元素。
- read：`x.sum()`，grid 4，加 1 个合并 kernel。
- write：`full_like`，1024 个 program。
- copy：`x * 1.5`，1024 个 program。

指标为设备周期内搬运的语义字节，单位 B/cycle。AXI 模型的读写延迟固定为 30 拍（按 HBM 经 SmartConnect 在 100 MHz 下的量级设定，尚未校准，见 §5）。

| 配置 | read | write | copy |
|---|---|---|---|
| 片上 RAM（`core-vrf-sync-v5`） | 2.27 | 2.51 | 3.68 |
| 单拍桥，OUT_N 8 | 1.91 | 1.69 | 1.92 |
| 单拍桥，OUT_N 16 | 1.96 | 1.85 | 2.22 |
| 行缓冲 8 拍 × 8 行，8 waiter，预取 1 | 1.54 | 1.77 | 2.55 |
| 行缓冲，同上，预取 2 | 1.54 | 1.77 | 2.59 |
| 行缓冲 8 拍 × 16 行，32 waiter，预取 1 | 2.07 | 1.76 | 2.52 |
| 行缓冲，同上，预取 2 | 2.05 | 1.76 | 2.57 |
| 行缓冲 8 拍 × 16 行，32 waiter，预取 1，OUT_N 16 | **2.07** | **2.04** | **2.84** |
| 同一配置，AXI 延迟 1 拍 | 2.27 | 2.51 | 3.68 |

读法：

1. **核心自身的上限才是天花板。** 即使用理想的片上 RAM，核心在 16 元素块下也只有 2.3–3.7 B/cycle，约为 8 B/cycle 入口的 30–45%。
   - 原因是每个 program 只搬 64 B，剩下的周期花在循环、地址和 launch 上。
   - 所以 R40-04 验收 2 的“≥冻结瓶颈峰值的 50%”需要 R40-03 的宽块；R40-04 的 wide ingress 还需要核心侧的 32 B 入口。只改桥达不到。
2. **行缓冲的大小要按并发的流来定。** read 图的 4 个 program 同时运行（4 个 role context），即 4 条流。
   - 每条流需要当前行与下一行，在途的每个字各占一个 waiter。
   - 8 行 / 8 waiter 不够，结果比单拍桥还差（1.54）；16 行 / 32 waiter 时为 2.07。
   - 预取深度 2 没有收益。
3. **写受在途数限制。** 写是单拍写直达，一个写占用一个表项直到 B 返回。OUT_N 16 把 write 从 1.76 提到 2.04。
4. **延迟为 1 拍时，行缓冲桥与片上 RAM 逐拍相同。** 桥本身不再带来开销。
   - 单拍桥在延迟 1 拍时，write 为 2.08、copy 为 2.74：每个写要等上一个写的 AW 与 W 都被收下，最多两拍一个写。行缓冲桥的写 FIFO 去掉了这个限制。

选定给板卡的下一个配置：`AXI_LINE=8, AXI_LB_N=16, AXI_WAIT_N=32, AXI_OUT_N=16, AXI_PF=1`。
- 在 30 拍延迟下，read / write / copy 分别为片上 RAM 的 91% / 81% / 77%。
- 时序方面，窗口请求先进两项 FIFO：v5 设备内 core→行查找的路径综合后为 9.1 ns（WNS +0.70 ns），routed 时很可能不过，FIFO 把它从 req_ready 上拿掉。

## 7. 2026-10-11 续接审计与重建

- acf3019 上 `u280_r40a/link.log` 的 `VPP_RC=1`；没有 xclbin。
  routed WNS −0.683 ns，TNS −2642.759 ns，7019 个 setup 端点失败；不是已通过的镜像。
- 最坏跨接口路径：HBM interconnect 的 RID 寄存器 → core TXF xbuf CE，
  `clk_kernel_00_unbuffered_net` → `ap_clk`，10 ns 周期，数据路径 5.540 ns，clock skew −5.025 ns。
  `ooc.xdc` 的主 `create_clock` 覆盖了 CU pin 上的平台时钟传播。
  `package_xo.tcl` 现在限定 `USED_IN {synthesis out_of_context}`；导出的 component.xml
  带 `USED_IN_out_of_context`。删除 routed checkpoint 中重复时钟只用于诊断；不能替代重新实现。
- 尚有核内真实失败路径（VRF address → vector write data，−0.546 ns），
  以及平台 interconnect 的 −0.064 ns；新 link 必须重新通过全部 timing/CDC gate。
- 构建参数现在由 `tools/r40/image_profile.py` 从 `spec/r1_profiles.json`
  产生 Tcl 与 JSON manifest。通过 `TNCA_IMAGE_CONFIG` 传给 packager，wrapper 显式逐层传参，
  不再需要在导出源码中改写 v5 默认值。HBM 下一配置为 §6 的 8/16/32/16/1。
- 板上采样修复：8 GiB 窗口现在包含 PC31，支持非零 BO 基址；
  `R40-04.hbm` 通过前必须有 HBM[0:31] 的实际覆盖，小窗口诊断不能通过该 gate。
  地址规划 9/9 回归通过（包含最后一个 PC、高地址、偏移窗口和非法部分 PC）。
  streaming 三个方向均新增 CPU 数值核对，容量图的读字节账本改用 R beats。
- 变异测试的编译失败不再被视作捕获语义缺陷；正常仿真还要求进程退出码为 0。
- 当前仿真运行在 gx10-6646 的独立快照 `~/tritonchip-r40-next`；
  Vivado 打包/诊断运行在 acf3019，构建目录 `~/tnca_build/u280_r40next`。
  全量 device/performance/板上结果尚未产生，不提前签收。

本轮可提交检查：桥 42/42、语义变异 11/11、地址规划 9/9、`r36-fpga-tests` 8/8、协议 34/34；设备/HBM 板上与性能 gate 仍未关闭。

## 8. 2026-10-12 核内边界判断时序修复

`u280_r40next` 重新 link 后仍未通过 100 MHz：routed WNS −0.209 ns、
TNS −156.731 ns、2446 个 setup 端点失败。工具自动改频到 97.9 MHz，
`link.sh` 正确拒绝该结果；这个 xclbin 没有作为验收镜像装载。
最差路径从 TXF `t_i1_reg[8]` 到 `xbuf_reg[460]/R`，经过 `in_dim` 返回值，
共有 31 级逻辑、16 个 CARRY8；数据延迟 9.496 ns，route 占 74.2%。

`in_dim` 改为先并行判断两个可能的高 32 位，再由低 32 位加法的进位选择。
保留任意 64 位坐标、无符号 32 位迭代器和原有 64 位回绕语义，
不增加状态或周期，也不依赖 frontend 的坐标范围。
`tools/r40/tensor_bounds_tests.py` 从实际 RTL 提取函数，由 Yosys SAT
对全部 160 位自由输入证明与原始 64 位加法/符号/比较公式等价。
形式证明已接入 CI 和 `make r40-hbm`；新的板上 timing/CDC 结果仍待重新实现。

`u280_r40bounds` 的新 routed 结果仍失败：WNS −0.336 ns、
TNS −226.125 ns、1905 个 setup 端点；自动 96 MHz 被拒绝，未装载。
最差路径已经移到 LSU `pending_reg[10]` → local BRAM `WEA[0]`，
数据延迟 9.968 ns，16 级逻辑，route 占 84.771%。
证据为 `evidence/r40/R40-04/bounds_link_timing.{txt,json}`；
JSON 保留完整 routed 报告的 SHA256，原件在板卡主机的构建目录。
下一步检查 burst bridge 的 local RAM 请求是否也应经过现有两项 FIFO；
该机制默认关闭，必须先通过乱序、背压、原子、错误和复位审计，
再重新进行 100 MHz link。

## 9. Local RAM 请求流水 checkpoint

`AXI_LOCAL_PIPE=1` 将 local RAM 与窗口请求一起放入 burst bridge 的
现有两项 FIFO。上游 ready 只依赖注册的 occupancy；local BRAM 从
注册的 head 读取全部 address/data/mask/tag/atomic 字段，并在本地
ready 握手后才出队。默认 0 保留旧 bypass，不改变片上配置。
镜像生成器的 `--local-pipeline` 只允许 HBM burst 模式，所有 wrapper
与仿真 override 均贯通这个独立开关。local 请求增加 FIFO 延迟，
这里只验证语义，实际性能必须重新测量。

跳板机验证：七种桥几何合计 49/49；13/13 语义变异均检出
（包括队列 head tag 被 live tag 替换、local 请求不出队）；
FPGA wrapper/协议 8/8；原生 storage/refusal/fullgraph/strided view
4/4，包含 pointwise/reduction/norm/GEMM/SDPA、驻留零 launch 上传。
日志在 `evidence/r40/R40-04/localpipe_*.log`，测试树为
`~/tritonchip-r40-ci`。这些检查不构成 timing/CDC 或 HBM 带宽签收。
新 100 MHz link 将使用显式开关并保留镜像配置。

## 10. 2026-10-13 行数据 RAM checkpoint

`u280_r40localpipe`（`ccede7a`、100 MHz）因路由拥塞失败：
2270 条信号未布通、2225 个 node overlap，错误集中在 `lb_data_reg`。
link 返回失败；无验收镜像、未装载，不能将未完成路由的 slack 当成结果。

可选 `AXI_RAM_CACHE=1` 将 line 数据放入同步双端口 RAM：
A 端写 fill beat；B 端读取等待完成的 line 或按 byte enable 更新 store hit。
同步读取保存 tag，回复因 local/null/atomic 优先级被阻塞时保持 data/tag；
该期间禁止覆盖 B 端输出。waiter 在读取捕获时释放，写同 line 的 hazard
仍受原有规则约束。默认 0 保留旧实现。增加一个 cache-hit 回复周期，
但连续读取可以每周期一次；实际性能须重新测量。

随机桥 70/70、变异 16/16、恶劣 AXI 下代表原生图 4/4、
wrapper/协议 8/8 通过。变异覆盖 live tag、覆盖 held 回复、
错用 full-byte mask。Yosys XCU 映射为 2 个 RAMB36E2、8 个 LUT2、
0 个数据触发器；这不是 Vivado timing 签收。
`--ram-cache` 只允许 HBM burst 配置；公共 `make r40-hbm` 加入映射核验。
新 100 MHz link 将显式开启 local pipeline 和 RAM cache，仍保留
16 line / 32 waiter / OUT_N16 / 8-beat / PF1 的容量与并发配置。
