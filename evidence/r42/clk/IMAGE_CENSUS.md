# R42 clk: image-level path census of the run-3 image (routed, 10 ns) and what has to change for an 8.0 / 8.5 ns link

Domain: routed run-3 image (evidence/r41/hbm_bw/g4/run3, kernel clock `clk_kernel_00`, period 10 ns, setup WNS +0.111 ns), read-only analysis of the area worker's endpoint list
(`~/r42_area_out/dcp_run3/endpoints.csv` on the x86 host, one worst path per endpoint, 400,000 endpoints sorted by slack, complete up to slack 2.445 ns; tool `tools/r42/clk/image_census.py`; table
`evidence/r42/clk/image_census_run3.md`). No new Vivado run. An endpoint with slack below +2.0 ns is a path longer than 8 ns (fails an 8.0 ns link if nothing changes), below +1.5 ns longer than 8.5 ns.
Only endpoints captured by the kernel clock are counted (90,899 below +2.0 ns; the platform's own clocks have other periods).

## 1. Counts by class (endpoints below +2.0 ns / +1.5 ns; worst slack; implied period of the class as it is = 10 - worst slack)

| class | < 2.0 (8.0 ns) | < 1.5 (8.5 ns) | worst slack | implied period | what it is | needs |
|---|--:|--:|--:|--:|---|---|
| A tile-internal (start and end in the same tile) | 54,378 | 28,696 | +0.126 (LSU) | 9.87 | home slice 26,116 / 13,834, core own 6,348 / 3,117, VRF 5,788 / 1,920, LSU 5,760 / 5,178, txf 4,655 / 2,808, matrix 2,793 / 950, vFPU 2,097 / 572, barrier 659 / 208, valu 161 / 109 | CLKFIX / CLKFIX2 (the union tile closes 5.6 ns OOC; the image/tile derate 1.4-1.5 puts that at 7.8-8.4 ns) |
| reset synchroniser fan-out (`ulp/tnca_u50_1/inst/reset_sync_reg` -> registers in tiles, mesh RAMs, outlet adapters, TMU; counted inside the groups below) | 21,160 | 12,601 | +0.168 | 9.83 | 1-2 LUT levels, 8.4-9.2 ns of net: one global reset net crossing the SLRs; the reset enters data-register enables (CE) of every always block with a synchronous reset | **new RTL**: registered, replicated reset tree per tile and unit (0 cycles in operation; not visible in any OOC tile screen: the port has no source delay) |
| B tile boundary, start outside the tile | 13,929 (13,784 reset) | 9,590 (9,562 reset) | +0.168 | 9.83 | reset into ms_addr/valid/rq_addr/lane_addr/acc registers | the reset tree; the 145 / 28 non-reset ones: none |
| B mesh (`u_mesh`) | 5,637 (3,056 reset) | 1,931 (800 reset) | +0.706 | 9.29 | non-reset 2,581 / 1,131: mesh router logic into LUTRAM (16 levels, 9.0 ns) | mesh RTL or `NOC_LINK_STAGES` 3 (parameter exists; +1 cycle per hop) |
| B home links (SLR crossing) | 1,972 (452 reset) | 1,021 (276) | +0.343 | 9.66 | non-reset 1,520 / 745: home pipe `wbq_addr` -> memory request select -> link FIFO RAM, 17 levels, 8.8-9.0 ns | home slice (bit 9 closes this cone in the unit screen: wbq_addr -> o_mem_req_addr; to be confirmed in a tile/image screen) |
| B HBM outlet links (SLR crossing) | 526 | 156 | +0.801 | 9.20 | link FIFO write enables, 6 levels, 8.4 ns of net | `LINK_STAGES`/placement; small |
| B command links | 180 | 32 | +1.119 | 8.88 | | nothing at 8.5 ns |
| C HBM AXI outlet adapters | 8,348 (2,600 reset) | 3,815 (872) | +0.595 | 9.40 | `bus_address_reg`/`address_reg`/`tag_reg` enables from the outlet link's read pointer (`rptr_reg`), 10 levels, net 8.5 ns | **new RTL**: registered dequeue/issue enable in the outlet adapter (5,748 / 2,943 non-reset endpoints) |
| C TMU (shell, outside the tiles) | 1,505 (1,248 reset) | 1,302 (1,082) | +0.779 | 9.22 | reset into tb/te registers; `s3_v_reg` (home slice) -> `mw_data_q_reg` 3 levels 8.3 ns | reset tree; 192 non-reset: small RTL |
| C device top: control / MMIO | 1,470 | 810 | +0.111 | 9.89 | `fast/valid_reg` (LSU quiescence, three tiles in three SLRs) -> all_idle -> `rsp_r0/rsp_r1/launch_sequence/cache_maint_*` enables, 14-15 levels | **bit 7** (registered all_idle; it sits in front of every one of these enables: rsp_r0 126, rsp_r1 126, launch_sequence 100, cache_maint_base/size 128) |
| C dispatch / code / deadline | 1,284 | 657 | +0.650 | 9.35 | `g_outlet[].adapter/active_reg` -> `active_cycles/completed/issued/done_cycles` counters in the shell, 8 levels, net 7.3 ns | **new RTL**: registered capture of the per-outlet counters (perf/status registers; 0 or 1 cycle on status reads) |
| C platform / HBM subsystem | 1,640 | 1,115 | +0.671 | | asynchronous FIFO gray-pointer and XPM memory paths (0 levels, delay 0.5-1.6 ns) with a fixed max-delay requirement | none: not period dependent |
| C kernel registers, D other | 30 | 13 | +0.506 | | | none |

(Group totals: A 54,378 / 28,696; B 22,244 / 12,730; C 14,271 / 7,710; D 6 / 2; reset-started in all groups 21,160 / 12,601. Full per-subgroup table with top end registers: `image_census_run3.md`.)

## 2. Reading (derived)

- As it stands the run-3 image's classes B and C and the reset tree all have worst paths of 9.2-9.9 ns, i.e. the kernel clock cannot go below about 9.9 ns even with a tile that closes 5.6 ns OOC: no CLKFIX/CLKFIX2 bit touches them.
- 8.0 ns (slack < 2.0): about 34,700 endpoints outside the tile (reset tree 21,160, outlet adapters 5,748 non-reset, mesh 2,581, home links 1,520, TMU 257, device control 1,470, dispatch counters 1,284, outlet/command links 702) need RTL or link-stage changes, plus the tile-internal 54,378 through the union bits at an image derate of 1.4-1.5 (7.8-8.4 ns for a 5.6 ns tile): 8.0 ns is the edge of what the union tile alone supports and needs every non-tile class fixed.
- 8.5 ns (slack < 1.5): about 19,300 outside the tile (reset tree 12,601, adapters 2,943, mesh 1,131, device control 810, home links 745, dispatch 657, TMU 220, outlet/command links 188) and the tile at derate up to 1.5 (8.4 ns): the same classes, about 45 % fewer endpoints, still every class has members.
- The ranking by endpoints: reset tree (21k) > HBM outlet adapters (5.7k) > mesh (2.6k) > home-slice memory request select (1.5k, bit 9) = device control (1.5k, bit 7) > dispatch counters (1.3k) > TMU (0.3k).
- Which need NEW RTL before an 8.0 or 8.5 ns link could close: (1) the replicated registered reset tree, (2) the HBM outlet adapter enables, (3) the dispatch/status counter capture, (4) the TMU's two registers (small), (5) the mesh router cone (or one more mesh link stage), (6) bit 7 for the device control cone (built, default-off, tested only as +/-0.1 % cycles on three stand-in points), (7) bit 9 for the home slice request select. Items (1)-(4) are not covered by any bit built so far and none of them is visible in an OOC tile screen.
