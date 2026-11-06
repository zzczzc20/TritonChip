# R40-04: Single-Core HBM and External Memory Interface (Design and Verification Record)

Contract: see R40-04 in `doc/r40_tasks_and_acceptance.md`; this document records implementation and evidence without modifying the contract.

## Current NUCA address-map correction (2026-10-19)

The NUCA TMU decoder intercepts device addresses `[0x7fff0000, 0x7fff1000)`
before HBM translation. This is a **4 KiB reserved register page**, even when
its address falls inside the allocated XRT BO window. Runtime allocation must
never place or reuse a buffer across that page. The code and heap allocators
exclude it, retain valid free prefixes, and preserve the exclusion after recovery.
`Session.memory_stats.heap_capacity_bytes` subtracts intersecting reserved bytes;
`heap_reserved_ranges` exposes the exact excluded intervals. Physical BO size
and usable allocation capacity are distinct quantities.

The original 16-layer large Llama output weight crossed this page. The allocator
repair restores the original decode logits error from 0.11417 to 0.002714 on the
same 90 MHz image, without changing executable payloads or the 0.05 tolerance.
See `evidence/r40/v2/convergence/attempt03/reserved_original_decode_1a7c643_complete.json`
and its artifact comparison. Original prefill and complete repaired-source
regressions remain required. The sections below retain historical interface and
validation records; their older image/capacity claims are not final signoff.

## 1. Interface: Core Data Port Remains Unchanged

The core has always had exactly one data port (`mem_req` / `mem_rsp` of `tnca_shell`, see `rtl/fpga/tnca_fpga_mem.v`). Its semantics are:
- Valid/ready requests with tag, byte enable, and atomic flags; every request receives exactly one response.
- Effects occur in **acceptance order**; responses may return in arbitrary order and cannot be backpressured.

R40-04 does not alter this port; it simply places a bridge behind it:
- Addresses below `LOCAL_LIMIT` (0x20000) remain in on-chip RAM: the code area, launch parameter area, and runtime data below the heap. Instruction fetch continues to access only on-chip RAM.
- Starting from `LOCAL_LIMIT` is the AXI window, where device address A maps to AXI address `axi_base + (A − LOCAL_LIMIT)`.
  - `axi_base` and `axi_size` are written by the host to `R_AXI_BASE` (0xA8) and `R_AXI_SIZE` (0xB0).
  - Out-of-window accesses behave identically to out-of-bounds on-chip RAM accesses: reads return 0, writes are dropped.

In this way, the HBM protocol terminates at the board-level adapter; the core, ISA, and artifacts have no awareness of HBM (§2.1 layering).

New registers (present only in `DATA_AXI` images; reading 0 in older images, which the runtime uses to identify the image):

| Offset | Name | Content |
|---|---|---|
| 0xA8 | AXI_BASE | Base address of the window in AXI space (host-written) |
| 0xB0 | AXI_SIZE | Window size in bytes (host-written; mailbox range check bounds at `LOCAL_LIMIT + size`) |
| 0xB8 | AXI_STAT | bit 0: sticky error response; bit 1: window present; bit 2: sticky stray response; 63:32: LOCAL_LIMIT, in 4 KiB pages |
| 0xC0 / 0xC8 | AXI_RD / AXI_WR | Handshake count for AR and AW (since reset) |
| 0xD0 | AXI_LAT | Accumulation of in-flight reads per cycle; dividing by read count yields average read latency (Little's Law) |
| 0xD8 | AXI_RBEAT | R beat count; ratio to AXI_RD yields burst length |

Common device clock extension (2026-10-13, not yet included in the `ac580b0` image under construction):

| Offset | Name | Content |
|---|---|---|
| 0xE0 | TIME_CAP | Read-only version 1; older images return 0, indicating unavailable |
| 0xE8 | TIME_START | Common-clock timestamp when the most recent LAUNCH was accepted |
| 0xF0 | TIME_END | Timestamp when completion of that same LAUNCH reached MMIO |
| 0xF8 | TIME_SEQ | Corresponding request sequence number; 0 when a new LAUNCH has not completed |

The 64-bit counter increments on every device clock cycle since reset, including idle time, submission gaps, and backpressure. PING, QUIESCE, and PERF_READ do not overwrite the LAUNCH timestamps. Reading requires verifying `TIME_SEQ` twice and matching the expected request sequence number; endpoints across resets cannot be combined, and differences within the same backend epoch are computed using 64-bit modular arithmetic. `U50Backend.launch_timestamps` implements this read; older images return None, while stale sequence numbers, in-progress runs, and unknown versions are explicitly rejected.

`make r40-device-clock`: host unit tests 6/6 and FPGA register protocol RTL tests pass, proving host work between two launches is encompassed by the common clock, and graph time span exceeds the sum of the individual launch spans. Existing serial-grid cycles and request-latency records retain their original definitions.

Software integration: backend args `device_clock: true` explicitly enables reading MMIO timestamps upon completion of each program, disabled by default. Session attaches a session ID to timestamps; cross-clock domains, reset epochs, missing endpoints, and ambiguous spans are disallowed from forming full graph measurements; 64-bit wrap-around and out-of-order completion spans have independent tests. Asynchronous launch logs fill in endpoints upon wait completion. ArchBench timed calls add `device-graph-makespan` samples only when endpoints are complete, including submission gaps and backpressure; per-program MMIO read overhead is measurement instrumentation, and A/B comparisons must use identical settings. Currently, on-board timing evidence remains pending; protocol 2 sampling, performance thresholds, and new image verification are incomplete.

Native MMIO Torch checks also revealed edge cases in initialization copies. Dense transposed layouts covering the entire span write directly without reading uninitialized destinations; complete Storage copies only pad the allocator's trailing word padding (beyond `Storage.nbytes`, which cannot be addressed by valid aliases). Partial Storage copies and views with gaps preserve existing bytes. Simulation bindings encountering complete X / invalid value responses with `end` explicitly raise `DeviceLost`, no longer misclassifying them as host timeouts; incomplete responses continue waiting.

## 2. Two Bridges (A/B of Burst Mechanism)

`AXI_LINE` in `tnca_fpga_top` selects the bridge: 0 for the single-beat bridge, >0 for the line-buffered bridge; both share identical ports.

### 2.1 `tnca_axi_bridge`: Single-Beat, the Capacity Step

Each access is an 8-byte AXI beat (INCR, len 0).
- **Ordering**: In-flight AXI accesses are tracked in an `OUT_N`-entry table. If a new request targets the same word as an entry in the table and either access is a write, it waits until the in-flight access completes. Two reads to the same word may overlap.
  - Writes use a single ID and complete in issue order; reads use the tag as their ID and may complete out of order.
- **Atomics**: AXI4 lacks atomic operations like `tnca_amo.vh`, so read-modify-write is performed inside the bridge.
  - The table is drained first; the bridge then reads, computes the new value, writes back, and returns the old value upon write completion. No new requests are accepted during this period.
  - The new value is computed using `amo_new`, identical to the function used in on-chip RAM and the home slice.
- **Reset**: Reset discards in-flight requests, but the AXI side may not reset simultaneously, allowing stale responses to return.
  - For `QUIET` (1024) cycles following reset, no requests are accepted, and all incoming R/B responses are accepted and dropped.
  - Thereafter, any R/B response not matching an in-flight request is also dropped, setting `axi_stray`.
  - The request side (AW accepted, W not accepted) cannot be retracted, so the platform must reset the interconnect alongside this module. On the U280, reset occurs only when loading an image.

### 2.2 `tnca_axi_bridge_lb`: Line-Buffered Read

The window is read in aligned lines of `LINE_BEATS` × 8 B: one INCR burst per line (`ARLEN = LINE_BEATS − 1`, `ARID = slot number`), entering a fully associative buffer of `LB_N` lines.

Read handling:
- Reads hitting the buffer are answered directly by the buffer.
- Reads targeting a line currently being fetched wait for that line.
- Cache-miss reads allocate a slot and issue a line fetch; simultaneously, `PF` subsequent lines are prefetched (without crossing the window boundary).
- In all three cases, requests are accepted as waiters (`WAIT_N` total), answered in arbitrary order as soon as their line arrives.

Rules preserving acceptance order (each verified by a mutation test, see §4):
1. Writes remain single-beat and write-through, updating the buffered copy of that line so subsequent reads observe it.
2. Writes do not enter a line currently being fetched, preventing fetched data from overwriting the write.
3. Writes do not bypass unanswered waiters on the same line: that read was accepted earlier and must not observe the write.
4. Line fetches are not issued for a line with an in-flight write, preventing AXI from allowing the read to overtake the write.
5. Atomics wait until all in-flight accesses complete and all waiters are answered, invalidate their line from the buffer, and perform read-modify-write as in §2.1. The read ID is `LB_N`.
6. Host DMA writes HBM directly, bypassing the bridge. `tnca_mmio` asserts `inv` for one cycle for each command serviced:
   - Valid lines are invalidated; in-flight lines are discarded upon arrival. If a line completes on the exact cycle `inv` is asserted, it is immediately invalidated.
   - Host DMA is always followed by a command (UPLOAD, then LAUNCH) and never runs concurrently with the kernel.
7. Writes pass through a 4-entry FIFO per channel, accepting one write per cycle. The single-beat bridge must wait until both AW and W of the preceding write are accepted, accepting at most one write every two cycles.

## 3. Runtime and Board

- **Simulation**: Memory mode `core-vrf-sync-v5/hbm` (`python/tnca/device.py`).
  - Architecture identical to v5: compilation, capabilities, and artifacts follow `base_profile`; memory mode determines only which bench the session runs on.
  - Bench sets `DATA_AXI=1`; `tb/common/tnca_axi_mem_model.v` provides a 32 MiB window, with timing controlled by `+axi_*` plusargs (`TNCA_DEVICE_PLUSARGS`).
  - `TNCA_RTL_PARAMS=AXI_LINE=8,...` selects the line-buffered bridge.
- **Board**: `rtl/board/tnca_hbm_xo.v`, `fpga/kernel_hbm.xml`, packaged by the `hbm` variant of `package_xo.tcl`.
  - The device runs directly on `ap_clk` without BUFGCE_DIV. `m_axi` shares the platform interconnect clock, making the interface synchronous.
  - Kernel argument `hbm` is located at offset 0xA8 (`R_AXI_BASE`).
  - `LINK_EXTRA="--connectivity.sp tnca_u50_1.hbm:HBM[0:31]"` in `link.sh` connects `m_axi` to the memory bank group comprising all 32 pseudo-channels.
- **Runtime** (`python/tnca/backends/u50.py`): Upon observing `AXI_STAT.bit1`:
  - Identifies the bank connected to this argument from the xclbin's (GROUP_)CONNECTIVITY and MEM_TOPOLOGY (the largest one, i.e., the group), and allocates the window using `xclAllocBO` (`TNCA_HBM_BYTES`, default 1 GiB).
  - Writes `R_AXI_BASE` and `R_AXI_SIZE`, verifying with readback. Session heap bounds are set to the device-reported `LOCAL_LIMIT + size`.
  - Transfers wholly within the window use DMA via `xclWriteBO`/`xclSyncBO`; others continue through the register window.
  - Checks sticky bits in `AXI_STAT` after each launch, raising `DeviceLost` if errors are detected.

## 4. Verification (Simulation Suite, `make r40-hbm`)

- **Bridges**: `tb/fpga/tb_axi_bridge.v`. Random traffic checked against a reference memory updated in acceptance order, using a small address pool to force frequent collisions.
  - AXI slave maximizes adversarial timing: read data is sampled only when sending R, writes take effect only when sending B; random latency, burst reordering and interleaving under `+reorder`, and READY randomly deasserted with probability `+bp`%.
  - Modes: `+err` (selected words return SLVERR), `+rst` (reset 4 times with requests in flight, slave unreset), `+dma` (quiescent writes behind the bridge with `inv` asserted).
  - 7 testcases × 6 configurations (single-beat bridge OUT_N 8/16; line-buffered 8/4/1, 2/2/0, 4/8/2, and selected 8/16/1 with 32 waiters, OUT_N 16, i.e., beats/slots/prefetch).
- **Mutations**: 5 for single-beat bridge, 6 for line-buffered bridge. Each mutation must be caught by its designated testcase.
- **Device**: All 16 cases of `tools/r40/device_tests.py` on `/hbm`.
  - An additional 5 cases run under adversarial timing (`+axi_lat 20..120 +reorder +bp 30`), once each for single-beat and line-buffered bridges.
  - One graph places tensors at 24 MiB into the window.
  - All 32 samples across the window are written before readback. If high address bits are lost, two samples would alias and trigger a mismatch.
- **Lint**: `r36-fpga-tests` elaborates the top-level with both `DATA_AXI=0` and `DATA_AXI=1`; both bridges are elaborated and checked at root level.

## 5. Board-Level Scope and Gates

- `R40-04.hbm`: Runs three checks via `tools/r40/hbm_board.py` on the U280 HBM image: window, whole-window samples (stepped across pseudo-channels), and a standard Torch graph ≥ 64 MiB (with counters).
- `R40-04.bandwidth`: `stream` from the same tool provides inputs for the bandwidth ledger. The 50% threshold additionally depends on core-side ingress, see §6.
- `R40-04.robustness`: On-board data correctness, post-route timing and CDC, and memory model calibration against on-board latency distributions (§3.4).

## 6. Bandwidth: Identifying Bottlenecks

Measurement methodology: Three standard Torch graphs on `/hbm` simulation, fp32, 16384 elements.
- read: `x.sum()`, grid 4, plus 1 reduction kernel.
- write: `full_like`, 1024 programs.
- copy: `x * 1.5`, 1024 programs.

Metric is semantic bytes moved per device cycle (B/cycle). AXI model read and write latency is fixed at 30 cycles (order of magnitude for HBM via SmartConnect at 100 MHz, uncalibrated, see §5).

| Configuration | read | write | copy |
|---|---|---|---|
| On-chip RAM (`core-vrf-sync-v5`) | 2.27 | 2.51 | 3.68 |
| Single-beat bridge, OUT_N 8 | 1.91 | 1.69 | 1.92 |
| Single-beat bridge, OUT_N 16 | 1.96 | 1.85 | 2.22 |
| Line-buffered 8 beats × 8 lines, 8 waiters, prefetch 1 | 1.54 | 1.77 | 2.55 |
| Line-buffered, same as above, prefetch 2 | 1.54 | 1.77 | 2.59 |
| Line-buffered 8 beats × 16 lines, 32 waiters, prefetch 1 | 2.07 | 1.76 | 2.52 |
| Line-buffered, same as above, prefetch 2 | 2.05 | 1.76 | 2.57 |
| Line-buffered 8 beats × 16 lines, 32 waiters, prefetch 1, OUT_N 16 | **2.07** | **2.04** | **2.84** |
| Same configuration, AXI latency 1 cycle | 2.27 | 2.51 | 3.68 |

Interpretation:

1. **The core ceiling forms the overall limit.** Even with ideal on-chip RAM, the core achieves only 2.3–3.7 B/cycle with 16-element blocks, approximately 30–45% of the 8 B/cycle port capacity.
   - Each program moves only 64 B; remaining cycles are spent on loops, addresses, and launches.
   - Consequently, the R40-04 acceptance 2 criterion ("≥50% of frozen bottleneck peak") requires wide blocks from R40-03; R40-04 wide ingress also requires a 32 B port on the core side. Modifying the bridge alone cannot reach it.
2. **Line buffer sizing must match concurrent streams.** The read graph executes 4 programs concurrently (4 role contexts), representing 4 streams.
   - Each stream requires the current and next line, with each in-flight word occupying a waiter.
   - 8 lines / 8 waiters is insufficient, performing worse than the single-beat bridge (1.54); 16 lines / 32 waiters reaches 2.07.
   - Prefetch depth 2 provides no additional benefit.
3. **Writes are constrained by in-flight slots.** Writes are single-beat write-through, occupying an entry until B returns. OUT_N 16 increases write throughput from 1.76 to 2.04.
4. **At 1-cycle latency, the line-buffered bridge matches on-chip RAM cycle-for-cycle.** The bridge introduces zero overhead.
   - At 1-cycle latency, the single-beat bridge achieves 2.08 write and 2.74 copy: each write must wait for both AW and W of the previous write to be accepted (at most one write per two cycles). The line-buffered bridge write FIFO removes this restriction.

Selected board configuration: `AXI_LINE=8, AXI_LB_N=16, AXI_WAIT_N=32, AXI_OUT_N=16, AXI_PF=1`.
- At 30-cycle latency, read / write / copy achieve 91% / 81% / 77% of on-chip RAM, respectively.
- For timing, window requests enter a 2-entry FIFO: post-synthesis delay for the core→line lookup path in v5 was 9.1 ns (WNS +0.70 ns), which risked failing post-route; the FIFO decouples it from `req_ready`.

## 7. 2026-10-11 Resumption Audit and Reconstruction

- `u280_r40a/link.log` on acf3019 had `VPP_RC=1`; no xclbin was generated.
  Routed WNS −0.683 ns, TNS −2642.759 ns, with 7019 setup endpoints failing; not a passing image.
- Worst cross-interface path: HBM interconnect RID register → core TXF xbuf CE,
  `clk_kernel_00_unbuffered_net` → `ap_clk`, 10 ns period, datapath delay 5.540 ns, clock skew −5.025 ns.
  The primary `create_clock` in `ooc.xdc` overwrote platform clock propagation on CU pins.
  `package_xo.tcl` now restricts `USED_IN {synthesis out_of_context}`; exported `component.xml`
  includes `USED_IN_out_of_context`. Deleting duplicate clocks in the routed checkpoint served only for diagnosis, not replacing re-implementation.
- Genuine core-internal failing paths remained (VRF address → vector write data, −0.546 ns),
  along with −0.064 ns in the platform interconnect; new links must pass all timing/CDC gates cleanly.
- Build parameters are now generated by `tools/r40/image_profile.py` from `spec/r1_profiles.json`
  as Tcl and JSON manifests. Passed via `TNCA_IMAGE_CONFIG` to the packager, wrappers explicitly propagate parameters across hierarchy levels, eliminating source edits to v5 defaults. The next HBM configuration is §6's 8/16/32/16/1.
- Board sampling fix: 8 GiB window now covers PC31, supporting non-zero BO base addresses;
  passing `R40-04.hbm` requires actual HBM[0:31] coverage, which small-window diagnostics cannot satisfy.
  Address planning regression passed 9/9 (including final PC, high addresses, offset window, and invalid partial PCs).
  Streaming checks across all three directions added CPU numerical cross-checks, and capacity read accounting switched to R beats.
- Mutation test compilation failures are no longer counted as catching semantic bugs; normal simulation also requires a process exit code of 0.
- Simulation runs in independent snapshot `~/tritonchip-r40-next` on gx10-6646;
  Vivado packaging/diagnostics run on acf3019 in build directory `~/tnca_build/u280_r40next`.
  Full device/performance/board results were not yet generated; no premature signoff.

Submittable checks for this round: bridges 42/42, semantic mutations 11/11, address planning 9/9, `r36-fpga-tests` 8/8, protocol 34/34; device/HBM board and performance gates remained open.

## 8. 2026-10-12 Core Boundary Check Timing Fix

`u280_r40next` relink still failed 100 MHz: routed WNS −0.209 ns,
TNS −156.731 ns, with 2446 setup endpoints failing. The tool automatically downclocked to 97.9 MHz,
which `link.sh` properly rejected; this xclbin was not loaded as an acceptance image.
The worst path ran from TXF `t_i1_reg[8]` to `xbuf_reg[460]/R` through the `in_dim` return value,
spanning 31 logic levels and 16 CARRY8 primitives; datapath delay 9.496 ns, route accounting for 74.2%.

`in_dim` was restructured to evaluate both candidate upper 32-bit results in parallel, selected by carry-out from the lower 32-bit addition.
Arbitrary 64-bit coordinates, unsigned 32-bit iterators, and original 64-bit wraparound semantics are preserved,
adding zero state or cycles, and without relying on frontend coordinate range assumptions.
`tools/r40/tensor_bounds_tests.py` extracts the function directly from RTL, using Yosys SAT
across all 160 unconstrained input bits to prove equivalence to the original 64-bit addition/sign/comparison formulation.
The formal proof is integrated into CI and `make r40-hbm`; new board timing/CDC results awaited re-implementation.

The subsequent routed run `u280_r40bounds` still failed: WNS −0.336 ns,
TNS −226.125 ns, 1905 setup endpoints; automatic downclocking to 96 MHz was rejected, unloaded.
The worst path shifted to LSU `pending_reg[10]` → local BRAM `WEA[0]`,
datapath delay 9.968 ns, 16 logic levels, route accounting for 84.771%.
Evidence is recorded in `evidence/r40/R40-04/bounds_link_timing.{txt,json}`;
the JSON retains the SHA256 of the complete routed report, with original artifacts in the board host build directory.
Next step: evaluate whether burst bridge local RAM requests should also traverse the existing 2-entry FIFO;
this mechanism defaults to disabled and must pass out-of-order, backpressure, atomic, error, and reset audits
before attempting another 100 MHz link.

## 9. Local RAM Request Pipeline Checkpoint

`AXI_LOCAL_PIPE=1` places local RAM requests alongside window requests into the burst bridge's
existing 2-entry FIFO. Upstream ready depends solely on registered occupancy; local BRAM reads
all address/data/mask/tag/atomic fields from the registered head, dequeuing only upon local ready handshake.
Default 0 retains the legacy bypass, leaving on-chip configurations unchanged.
The image generator's `--local-pipeline` is permitted only in HBM burst mode; all wrappers
and simulation overrides route this dedicated switch. Local requests incur FIFO latency;
semantic correctness is verified here, while actual performance requires re-measurement.

Jump host verification: 49/49 across seven bridge geometries; 13/13 semantic mutations detected
(including queue head tag replaced with live tag, local requests failing to dequeue);
FPGA wrapper/protocol 8/8; native storage/refusal/fullgraph/strided view
4/4, covering pointwise/reduction/norm/GEMM/SDPA and resident zero-launch uploads.
Logs reside in `evidence/r40/R40-04/localpipe_*.log`, test tree at
`~/tritonchip-r40-ci`. These checks do not constitute timing/CDC or HBM bandwidth signoff.
The new 100 MHz link will use this explicit switch and retain the image configuration.

## 10. 2026-10-13 Line Data RAM Checkpoint

`u280_r40localpipe` (`ccede7a`, 100 MHz) failed due to routing congestion:
2270 unrouted nets and 2225 node overlaps, heavily concentrated around `lb_data_reg`.
The link returned failure; no acceptance image, unloaded, and incomplete routing slack cannot serve as a valid result.

Optional `AXI_RAM_CACHE=1` places line data into synchronous dual-port RAM:
Port A writes fill beats; Port B reads lines awaiting completion or updates store hits per byte enables.
Synchronous reads preserve tags; when responses are stalled by local/null/atomic priority, data and tags are held,
during which overwriting Port B output is prohibited. Waiters release upon read capture, while hazards from writes to the same line
remain governed by existing ordering rules. Default 0 retains the legacy implementation. It adds one cache-hit response cycle,
but pipelined reads achieve one per cycle; actual performance requires re-measurement.

Random bridge 70/70, mutations 16/16, representative native graphs under adversarial AXI 4/4,
and wrapper/protocol 8/8 passed. Mutations cover live tag leakage, held response overwriting,
and incorrect full-byte masks. Yosys XCU maps this to 2 RAMB36E2, 8 LUT2,
and 0 data flip-flops; this is not Vivado timing signoff.
`--ram-cache` is restricted to HBM burst configurations; top-level `make r40-hbm` incorporates mapping verification.
The new 100 MHz link will explicitly enable both local pipeline and RAM cache, retaining the
16 line / 32 waiter / OUT_N16 / 8-beat / PF1 capacity and concurrency configuration.

### 2026-10-13: First True HBM Image Meeting 100 MHz

`u280_r40ram` (hardware ac580b0) completed post-route: setup WNS +0.016 ns,
TNS 0, hold WHS +0.006 ns, THS 0, zero pulse-width violations, and zero unconstrained internal
endpoints. v++ maintained kernel/device at 100 MHz and HBM at 450 MHz. Full CDC reported
58 critical endpoints, all internal to platform PCIe/ERT/clock/HBM IP paths, with none in the TNCA
kernel; complete reports and path-by-path reviews are retained without misrepresenting platform CDC as zero critical.

After verifying the board was idle, without open sessions, and within thermal limits, `program.sh --check/--go`
passed, establishing new UUID `e576c850-23ec-61c0-f728-74a1c19676ea`. Real standard Torch
tests passed across the 8 GiB HBM window, 32/32 pseudo-channels, addresses above 4 GiB, DMA, and in-place
device updates. Independent audits of 64 MiB capacity / 96 MiB streaming were ongoing.
Evidence is archived in `evidence/r40/R40-04/ram_image/`; R40-04 was not closed, nor was bandwidth threshold
satisfaction claimed. This image does not include the newer common device clock registers.

### 2026-10-13: 64 MiB Capacity and Barrier Epoch Defect

Initial capacity relative error was 3.88%, exceeding the 2% threshold; failure logs are fully preserved. Inputs and weights
matched bit-for-bit before and after execution, with anomalies isolated to the final two output tiles. Disabling the existing general
`TNCA_RCTX_SPMD` switch reduced relative error on the same standard Torch graph to 0.03116%,
achieving 3/3 on-board passes for window / all 32 PC samples / 64 MiB capacity.
This constitutes evidence of capacity capability, not an SPMD-on pass or performance gain.

The barrier epoch previously stored only 16 bits; on the 65,536th arm, it wrapped to token 0,
allowing unfinished operations to pass wait prematurely. RTL was updated to the ISA's 48-bit epoch,
explicitly faulting on exhaustion; tests across 65,536, sleep/wake, late stale tokens, and post-exhaustion init
passed, with the legacy RTL caught by counterexample at 65,536 and FPGA regressions passing 8/8.
Re-implementation and re-verification of SPMD-on on a new image remain required; the current board image remains ac580b0.
96 MiB streaming was also rejected by legacy 4096-program admission, leaving the bandwidth gate open.
