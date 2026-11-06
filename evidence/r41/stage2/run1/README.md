# R41 Stage 2: full run 1/5 (measurement)

The full three-core U280 image was linked at 100 MHz with v++, through synthesis, placement, routing and bitstream. **It met timing.** The kernel clock routed at setup WNS **+0.042 ns** and hold WHS **+0.003 ns**, and every clock group has zero TNS and THS. `link.sh` returned `LINK_OK`. The kernel clock reads back as 100 MHz from the xclbin. The only frequency rule in the log is AUTO-FREQ-SCALING-08: the scaler found 100.4 MHz achievable and kept the 100 MHz that was asked for.

This was a measurement run. The tile screens had predicted a 0.3–1.0 ns F04 failure. **The image was not loaded, and it is not qualified.** Qualifying it would still need the CI on this exact source, the operating/placement audits and the board gates. The slack margin is thin, and it comes from one placement.

## Source, flow and identity

| Item | Value |
|---|---|
| Source | `r41s2-t2` at `c9f25787494a8414f9ab19e03bea6fb3fa794d20` (F01–F10, X8, T2 fixes 1+2) |
| Source archive | `git archive --format=tar.gz` of that commit, SHA-256 `4a2dcaf62b7b3c9d6ff94144dc51793003f8e49e51c9817a19361738abfadd48` (the uncompressed tar stream has SHA-256 `5413bcbc…38fc2f`) |
| Image parameters | `evidence/r41/stage2/integration/image.tcl` (sha256 `50843bee…1b58b`) / `image.json` (`1d185bbe…0318c`), profile `core-vrf-sync-v5-x8`, `HAS_TENSOR_XPOSE=2` |
| Flow | Attempt 3's runner, unchanged: `flow/fpga/run_r40_attempt01.sh BUILD image.tcl`. It runs `package_xo.tcl` (multicore-hbm), then `link.sh` (u280) with `VPP_JOBS=2`, `place_r40_multicore.tcl` as `PLACE_DESIGN.TCL.PRE`, `SSI_SpreadSLLs`, `u280_multicore_hbm.cfg` (`hbm0/1/2:HBM[0:31]`) and `--kernel_frequency 0:100`. `link.sh`, `package_xo.tcl`, `place_r40_multicore.tcl`, the runner and the cfg are byte-identical to attempt 3's source `54249f7` |
| Host / build | FPGA host acf3019, `~/tnca_build/r41s2_run1_c9f25787494a`, run with `setsid nice -n 10`. MemAvailable was 47 GB at launch |
| Timeline (HKT) | Packaging 22:55; link 22:55:19 → 07:58:42 (9 h 03 m). Phases: synthesis+linking 2 h 20 m, opt 16 m, place 2 h 30 m, route 3 h 03 m, bitstream 51 m. The coordinator's memory guard never paused the run (0 STOP events) |
| XO | `65222b9b…e34164` |
| xclbin | `tnca_u280.xclbin`, sha256 `30e79769c128668ac8f1d0d9017f6ea8d22b8335b8fee18cbe374f4bdef4e5de` |
| Routed DCP | `level0_wrapper_routed.dcp`, sha256 `311bd49cdfe5870056c9496804112edc263e6adc2ae9caeecd7a409967045215` |
| Kept on host | `~/tnca_images/r41s2_run1_c9f25787494a/` holds the xclbin, routed DCP, `xclbin.info`, launch record and `SHA256SUMS`, all read-only, as well as the full build directory. **The xclbin was never loaded.** The card and the queue scripts were not touched |

**Deviation (packaging only).** For about 15 s after extraction, the launcher had the source tree read-only. During that window, `package_xo.tcl`'s Vivado could not write `vivado.jou`/`vivado.log` in its working directory. It logged two CRITICAL WARNING 17-183 messages and nothing else. Its stdout went to `build/package.log` as usual. The XO was produced (`TNCA_PKG_DONE`, `HAS_TENSOR_XPOSE=2` applied). The tree was writable again before v++ started.

## Routed timing

| Clock / group | Setup WNS | TNS | Hold WHS | THS |
|---|---:|---:|---:|---:|
| **Design** (1,478,650 setup endpoints) | **+0.016** | 0 | **+0.003** | 0 |
| `clk_kernel_00` (kernel, 100 MHz, 1,040,689 endpoints) | **+0.042** | 0 | +0.003 | 0 |
| `hbm_aclk` (450 MHz, platform HMSS) | +0.086 | 0 | +0.010 | 0 |
| `dma_ip_axi_aclk` (platform) | +0.025 | 0 | +0.006 | 0 |
| `dma_ip_axi_aclk` → `GTYE4_TXOUTCLK[3]_7` (platform, sets the design WNS) | +0.016 | 0 | +0.006 | 0 |
| `clk_kernel_01` | +0.369 | 0 | +0.035 | 0 |

- Failing setup endpoints, all groups: **0**. Failing hold endpoints, all groups: **0**.
- Route status: 1,153,083 routable nets, all fully routed, 0 errors.
- The kernel clock has 1,111 endpoints below +0.5 ns.
- Implementation trajectory: the placer's post-placement estimate was WNS −0.054. The router's first intermediate summary showed −0.649 / TNS −298. Routing ended at +0.012 (estimated), and +0.042 for the kernel clock in the signed-off report.
- Congestion: initial routing congestion was level 5 (global) and level 6 (timing). The level-6 long-north window is in SLR0 (`CLEL_R_X81Y138..X112Y201`, 49 % LUTRAM).

## Per-SLR utilisation and CLB packing (routed)

| | SLR0 | SLR1 | SLR2 |
|---|---:|---:|---:|
| CLB packing | **96.74 %** | 88.67 % | 85.11 % |
| CLB LUTs | 322,921 (73.44 %) | 324,295 (75.07 %) | 280,029 (64.82 %) |
| CLB registers | 156,891 (17.84 %) | 141,661 (16.40 %) | 102,406 (11.85 %) |
| BRAM tiles / DSP | 139 / 171 | 153.5 / 178 | 121.5 / 175 |
| A3 CLB packing (for reference) | 95.24 % | 88.39 % | 85.41 % |

SLL use: SLR1↔SLR0 14,263 of 23,040 (61.9 %), and SLR2↔SLR1 8,081 (35.1 %). A3 used 14,224 and 8,071. SLR0 packing came out 1.3 points above the integration README's linear estimate of ~95.4 %.

## Link gates (as attempt 3)

- **Clock.** The xclbin reports `DATA_CLK` (clock ID 0) at 100 MHz, which is the expected value. The only rule in the log is AUTO-FREQ-SCALING-08. Raw readback: `run1-reports/host/.../xclbin.info`.
- **CDC** (`report_cdc -details`). It reports **56 Critical** findings: CDC-7 ×13, CDC-10 ×7, CDC-11 ×32 and CDC-13 ×4. All of them are in vendor platform structures (ulp_ucs, blp, HMSS, interconnect clock converters). This is the same count as attempt 3. **No TNCA kernel endpoint appears in the CDC report** (0 matches for `tnca_u50_1/inst`). The kernel reaches other clocks only through the platform's `auto_cc` converters.
- **Clock interaction.** Every kernel-clock row is Clean. The single "Partial False Path (unsafe)" row is `io_clk_pcie_00 → dma_ip_axi_aclk_1` (26 endpoints, platform PCIe), which is outside the kernel.
- `check_timing` and `report_bus_skew` are in the tgz.

## Families (kernel clock, endpoints below +0.5 ns)

Grouping: `run1_families.py`. It uses the census family key from `tools/r40/group_failing_paths.family` and the owner rules from `screen_families.owner`. Tiles map to SLRs as tile 0 = SLR0, tile 1 = SLR1 and tile 2 = SLR2. Per-family detail is in `run1_families.json`.

| Class | Owner | Tile/SLR | Endpoints | Worst (ns) | Worst path |
|---|---|---|---:|---:|---|
| in-tile | **F04 writeback** | t0 / SLR0 | 62 | **+0.042** | `of_d1` → `w_vdata` (23 levels); `w_sdata` → `w_vdata` (11 levels, 9.1 ns route) |
| in-tile | **F04 writeback** | t1 / SLR1 | 69 | **+0.043** | `x_imm` → `w_vdata` (22 levels) |
| in-tile | **F04 writeback** | t2 / SLR2 | 52 | +0.065 | `of_d1` → `w_vdata` (24 levels) |
| in-tile | **F07 NUCA slice** | t0 / SLR0 | 401 (211 fam) | **+0.065** | `u_slice/cur_ms_reg_rep` → `data_mem` LUTRAM write ports (15–16 levels, ~8.3 ns route) |
| in-tile | **F09 MTX / accumulator** | t0 / SLR0 | 143 | +0.079 | VRF bank `mem_reg` → `u_mac a_q` (16 levels); FMA `s1_q` → `u_acc lane_mem` (+0.105) |
| in-tile | F09 MTX | t1 / SLR1 | 16 | +0.148 | `x_sb_fw_reg_rep` → `u_mac a_q` |
| in-tile | F07 NUCA slice | t2 / SLR2 | 177 | +0.165 | `service_way_q/service_set_q` → `data_mem` |
| **out-of-tile** | **Quiescence / idle → MMIO control (cross-SLR)** | SLR0 → SLR1 | 22 | **+0.180** | tile 0 `u_mtx/st_reg` → `control/state_reg`, `launch_sequence_reg`, `cmd_valid_reg` (7–10 levels, 8.3 ns route) |
| in-tile | F03/F08 VFPU | t1 / SLR1 | 22 | +0.184 | `of_d1` → `u_vfpu g_fma s0_q` (18 levels) |
| in-tile | F09 MTX | t2 / SLR2 | 72 | +0.228 | |
| in-tile | F10 LSU / NoC | t1 / SLR1 | 25 | +0.239 | |
| **out-of-tile** | **DCO local TMU replica** | SLR1 | 31 | +0.281 | `g_local_tmu … tl_class_id_q` → `lv_valid` (12 levels) |
| in-tile | F02/F05, F03/F08, transfer engine, F10 | t0, t2 | 15 | +0.336 … +0.490 | |
| **out-of-tile** | **F06 grid dispatch** | — | 0 | ≥ +0.5 | none below +0.5 |
| **out-of-tile** | **Mesh / NoC links, router FIFOs** | — | 0 | ≥ +0.5 | none below +0.5 |
| **out-of-tile** | **Memory outlets / AXI, kernel-side** | — | 0 | ≥ +0.5 | none below +0.5 |
| shell | **Platform (HMSS, DMA, PCIe)** | — | — | +0.016 … +0.086 | vendor paths only; nothing to fix in TNCA |

The transfer-engine families that T2 closed stay closed: the worst is +0.451 (tile 0, 3 endpoints). The W-forward families are not below +0.5 anywhere.

## Screen-to-image calibration (T2 round 3 screens, `89a6441`, versus this image)

Offset = image worst − screen worst, per owner, with the same tile and SLR.

| Owner | Tile 0 / SLR0: screen → image (offset) | Tile 1 / SLR1: screen → image (offset) |
|---|---|---|
| F04 writeback | +0.159 → +0.042 (**−0.12**) | +0.507 → +0.043 (**−0.46**) |
| Transfer engine | +0.285 → +0.451 (+0.17) | +0.977 → ≥ +0.5 (≥ −0.48) |
| F03/F08 VFPU | +0.592 → +0.397 (−0.20) | ≥ +1.0 → +0.184 (≤ −0.82) |
| F02/F05 scoreboard/VRF | +0.833 → +0.336 (−0.50) | ≥ +1.0 → ≥ +0.5 |
| F07 NUCA slice | +0.841 → +0.065 (**−0.78**) | ≥ +1.0 → +0.377 (≤ −0.62) |
| F09 MTX | +0.951 → +0.079 (**−0.87**) | ≥ +1.0 → +0.148 (≤ −0.85) |
| F10 LSU/NoC | +0.993 → +0.490 (−0.50) | ≥ +1.0 → +0.239 (≤ −0.76) |

- **The offset depends on the family.** It is not a single shift.
  - Short, logic-heavy families (F04, the transfer engine) moved −0.1 to −0.5 ns.
  - Wide RAM/DSP fan-in families (F07 LUTRAM writes, F09 MAC operand capture, F10) moved −0.5 to −0.9 ns. These sit in SLR0's dense level-6 region and on the cross-tile fan-out that the screens cannot see.
- The integration README's −0.8 … −1.15 ns applied to everything. That overstated the F04 loss by about 0.3–0.7 ns, which is why the predicted F04 failure did not happen. It was about right for F07 and F09.
- **Rule for the next screens:** −0.5 ns for F04-like (logic-bound) families, −0.9 ns for RAM/DSP-capture families, and −1.0 ns for SLR0 where packing is ≥ 96 %.
- Placement variance is still large. T2 saw F04 swing 0.6 ns between placements. This image's +0.042 is one sample.

## Fix list (ordered by risk to a repeat link at 100 MHz)

Nothing fails. These are the families that hold the margin to within 0.25 ns. A rerun of the same source on a different placement could lose 0.3–0.6 ns on any of them.

1. **F04 writeback, all three tiles (+0.042 / +0.043 / +0.065; 183 endpoints).** Owner: core (`rtl/core/tnca_core.v`). The fix already in progress applies: the vector integer MUL/MULH and the reduce broadcast move to the scheduled-writeback path (`MUL_LAT`, `wb_rsv`). Also check `w_sdata → w_vdata`, 11 levels with 9.1 ns of route in SLR0. It is pure placement/fan-out, so register or replicate `w_sdata` for the vector-broadcast leg.
2. **F07 NUCA slice data RAM write (SLR0 +0.065, 401 endpoints in 211 families; SLR2 +0.165).** Owner: NUCA (`rtl/nuca/tnca_nuca_slice.v`). `cur_ms_reg_rep` and `service_way_q/service_set_q` fan out to every LUTRAM column's write address and enable, through 15–16 levels with about 8.3 ns of route. Candidate fix: register the decoded write enable and address per RAM column group, one stage before `data_mem` (this costs a cycle on fills), or replicate `cur_ms` per column group with `max_fanout`. This is the largest near-critical population. The screens put it at +0.84.
3. **F09 MTX operand capture and accumulator write (SLR0 +0.079/+0.105, 143 endpoints; SLR1 +0.148; SLR2 +0.228).** Owner: MTX (`rtl/core/tnca_matrix.v`, `tnca_acc_rf.v`). VRF bank `mem_reg` → `u_mac a_q` is a 16-level broadcast to all MACs. Candidate fix: a per-MAC-group capture register after the bank read (the `MTX_ABUF` path), and a registered FMA `s1_q` → `lane_mem` write.
4. **Quiescence / idle cone, out-of-tile and cross-SLR (+0.180, 22 endpoints).** Owner: F10's second family, the quiescence cone (`rtl/core/tnca_matrix.v` `st` → idle aggregation in `rtl/tnca_top.v` → `rtl/fpga/tnca_multicore_mmio.v` `control`). Tile 0's MTX state reaches `control/state_reg`, `launch_sequence_reg` and `cmd_valid_reg` in SLR1 through 7–10 levels, with 8.3 ns of route. Candidate fix: register the per-tile idle bit in the source SLR, add an SLL TX/RX register pair, and give the MMIO a one-cycle-later idle (quiescence detection +1–2 cycles).
5. **F03/F08 VFPU operand (SLR1 +0.184) and F10 LSU/NoC (SLR1 +0.239).** Owners: VFPU/FMA (`tnca_vector_fpu.v`) and LSU (`tnca_lsu*.v`). Both are `of_d1` fan-out. Defer; they ride on fix 1's `of_d1` work.
6. **DCO local TMU replica (SLR1 +0.281, 31 endpoints, out-of-tile).** Owner: DCO/TMU (`rtl/tnca_top.v` `g_local_tmu`). `tl_class_id_q` → `lv_valid` has 12 levels. Watch only.
7. **SLR0 density (96.74 % CLB packing, level-6 congestion).** This is cross-cutting. Any area added to tile 0 lands here. Prefer fixes 1–3 that add registers but not LUTs (registers are at 17.8 %). Consider moving SLR0-resident non-tile logic, if `place_r40_multicore.tcl` allows it.
8. **Not ours / no action:**
   - F06 grid dispatch, the mesh/NoC links and the kernel-side memory outlets are all ≥ +0.5.
   - The platform (HMSS, DMA, PCIe) sets the design WNS at +0.016.
   - Hold is fixed everywhere (WHS +0.003, THS 0).

## Files

- `run1-reports.tgz`:
  - `analysis/`: the read-only post-route analysis of the routed DCP (`analyze_run1.tcl`, one `open_checkpoint`, 2 threads). It contains `route_status`, the min/max `timing`, `utilization_slr`/`_hier`, `clock_interaction`, `cdc`, `bus_skew`, `checks`, `top_paths_routed` (2,000), `top_paths_full` (200), `top_hold_paths`, `near_endpoints_0p5.tsv` (census columns), the all-group failing setup/hold TSVs (empty), `design_analysis`, `logic_levels` and `congestion`.
  - `vpp_impl1/`: v++'s own routed and placed timing/utilisation reports.
  - `host/`: the launcher, the post-run chain and analysis Tcl, `link.log`, `xclbin.info`, the build markers, the impl `runme.log` and the launch record.
- `run1_families.py` and `run1_families.json`: the family census above.
