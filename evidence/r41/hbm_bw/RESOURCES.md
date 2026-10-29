# Where the run-1 image spends its FPGA resources, and what can be removed

Status: G2 of `BANDWIDTH_PLAN.md`, 2026-10-28. **`doc/engineering_discipline.md` is binding** (§2.5: subtract as well as add; removing an architectural capability needs the author's approval).

This document answers three questions about the run-1 100 MHz three-core U280 image (`evidence/r41/stage2/run1/README.md`):
1. where the LUTs, LUTRAM, FFs, BRAM, DSP and CARRY8 go, per module, per tile and per SLR, platform shell included;
2. why the large consumers are as large as they are;
3. what can be removed or resized without losing an instruction, a numeric format, a precision or an SFU function, and how much each change saves when measured.

Every number below is read from a report produced by a recorded run (§8 lists the commands, hosts, commits and hashes). The domain of each number is stated: **routed image** (the run-1 routed checkpoint), **kernel synthesis** (out-of-context synthesis of the whole kernel at 10 ns), **tile screen** or **outlet screen** (out-of-context place and route at 10 ns, in SLR0), or **RTL simulation**. No number here is a board measurement, and no screen is a substitute for a linked image.

## 1. Summary

- **The platform is not the problem; the three tiles are.** Of the routed design's 927,245 LUTs, the TNCA kernel holds 798,355 and the three tiles 740,674 (246,868 / 247,348 / 246,458). Each tile sits entirely in its own SLR. SLR0 additionally holds all three memory outlets, one DCO/TMU replica, the HBM subsystem and part of the static shell.
- **Floating point is the largest single cost.** The kernel instantiates 96 binary32 FMAs (16 in each core's vector FPU and 16 in each core's matrix engine). In kernel synthesis of the run-1 source they take 229,008 LUTs, 28.3 % of the kernel; their normalisation stage alone takes 130,656 (16.1 %).
- **Next come the core's data-movement structures:** the two-unit LSU (25,435 LUTs and 13,639 FFs per core), the banked vector register file's read crossbar (13,784 LUTs per core outside the banks), the NUCA slice (30,647 LUTs per tile, 9,644 of them LUTRAM), the tensor transfer engine (14,494) and the barrier unit (7,450).
- **Two capability-neutral removals were built, regression-tested and screened** (both default-off, behind new image parameters):
  - **`AXI_LINE_LUTRAM 1`** (§4.2): the outlet's eight 512-bit slot lines move from reset registers into single-write-port distributed RAM. Outlet screen, per outlet: routed LUTs 6,942 → 1,389, FFs 6,131 → 2,027, CLBs 1,952 → 399, WNS +4.083 → +4.057 ns. Kernel synthesis: −6,153 LUTs and −4,106 FFs per outlet. All three outlets are in SLR0, so this is the largest SLR0 saving found.
  - **`NUCA_SHARED_DATA_READ 2`** (§4.1): one fewer LUTRAM copy of the slice's 128 Kbit line array. Kernel synthesis: −2,332 LUTRAM and −2,730 to −3,454 LUTs per slice. Tile-0 screen: −2,478 routed LUTs, −407 CLBs, WNS +0.838 → +0.772 ns, with the worst path in untouched SFU logic. Mode 1 (a single port) saves twice the LUTRAM but lost 0.552 ns to a structural path, so it is not recommended.
  - Together (kernel synthesis): −27,077 LUTs and −12,877 FFs of the kernel's 814,043 and 242,620.
  - Both changed nothing in the default build (identical kernel synthesis) and passed the lockstep, unit and board-top regressions (§7).
- **Measured costs of performance and timing features** (§5; removing one keeps every capability but trades performance or timing, so none is recommended). The second LSU (`LSU_INFLIGHT 2`) costs 35,925 kernel LUTs; the three DCO/TMU replicas cost 13,746 LUTs more than one shared TMU; `MTX_ABUF` costs 1,329; `VRF_PREFETCH` costs nothing measurable.
- **Options that would remove or narrow an architectural capability** (§6) went to the author. The decisions and their implementation are in §9:
  - two contexts instead of four, eight barriers instead of 32, 38-bit epochs and 35-bit home addresses, as image parameters and the new profile `core-vrf-sync-v5-x8-g2`;
  - accumulators stay at 16, because fewer lose matrix throughput (context slicing) and the Qwen FP8 kernels need 16;
  - MAC sharing and a smaller FMA were rejected.
- **Everything together** (the G2 parameters and both §4 changes), in kernel synthesis: −62,472 LUTs (7.7 %) and −35,807 FFs (14.8 %). Tile-0 screen: 247,579 → 232,925 routed LUTs, 41,878 → 39,974 CLBs (76.20 % → 72.73 % of SLR0), WNS +0.838 → +0.603 ns; the three outlets save their §4.2 amount in SLR0 on top.

## 2. Where the resources go

### 2.1 Whole design, routed image

From `report_utilization -hierarchical -hierarchical_depth 20` on the routed DCP (`resources/run1_routed/util_hier_d20.rpt.gz`; generated tables in `resources/tables/run1_routed_hierarchy.md`):

| Instance | LUT | LUTRAM | SRL | FF | RAMB36 | RAMB18 | DSP |
|---|---:|---:|---:|---:|---:|---:|---:|
| TNCA kernel (`ulp/tnca_u50_1`) | 798,355 | 55,472 | 162 | 226,528 | 192 | 45 | 520 |
| Static shell (`blp`: XDMA, PCIe, firewalls, CMC, ERT) | 106,153 | 4,564 | 1,413 | 129,945 | 193 | 5 | 4 |
| HBM memory subsystem (`ulp/hmss_0`) | 13,902 | 1,996 | 3,046 | 27,063 | 4 | 0 | 0 |
| `ulp/ulp_ucs` | 4,493 | 0 | 20 | 7,375 | 0 | 0 | 0 |
| `ii_level0_pipe` | 1,482 | 0 | 0 | 6,141 | 0 | 0 | 0 |
| Remainder (dynamic-region glue: `ulp` own logic, `ulp_cmp`, SLR crossings, resets) | 2,860 | | | | | | |
| **Design total** | **927,245** | **62,032** | **4,645** | **400,958** | **389** | **50** | **524** |

The kernel by block (routed):

| Block | LUT | LUTRAM | FF | RAMB36 | RAMB18 | DSP |
|---|---:|---:|---:|---:|---:|---:|
| Tile 0 (SLR0) | 246,868 | 15,216 | 65,345 | 32 | 15 | 171 |
| Tile 1 (SLR1) | 247,348 | 15,220 | 65,382 | 32 | 15 | 171 |
| Tile 2 (SLR2) | 246,458 | 15,220 | 65,303 | 32 | 15 | 171 |
| DCO/TMU, three local replicas (`g_local_tmu`) | 16,397 | 828 | 6,096 | 0 | 0 | 0 |
| Outlet adapters `tnca_line_axi` ×3 | 7,413 + 7,283 + 7,238 | 0 | 6,003 + 5,930 + 5,930 | 0 | 0 | 0 |
| Mesh | 5,130 | 2,628 | 264 | 0 | 0 | 0 |
| Memory-port arbitration (`g_multiple_memory.outlets`) | 2,136 | 0 | 8 | 0 | 0 | 0 |
| Grid dispatch | 1,312 | 0 | 1,407 | 0 | 0 | 7 |
| Pipeline links (outlet, home, dispatch, code, handoff) | under 1,300 each | | | | | |
| Code stores (`shell`, 3 × 16 K × 64 bit) | 117 | 0 | 139 | 96 | 0 | 0 |

### 2.2 Per SLR, including the platform

Site-level totals come from `report_utilization -slr` (`resources/run1_routed/util_slr.rpt`):

| | SLR0 | SLR1 | SLR2 |
|---|---:|---:|---:|
| CLBs used (packing) | 53,170 (96.74 %) | 47,884 (88.67 %) | 45,961 (85.11 %) |
| CLB LUTs | 322,921 | 324,295 | 280,029 |
| LUT as distributed RAM | 19,958 | 24,334 | 17,740 |
| LUT as shift register | 3,709 | 172 | 764 |
| CLB registers | 156,891 | 141,661 | 102,406 |
| CARRY8 | 10,264 | 10,432 | 9,687 |

To split each SLR by owner, every placed leaf cell of the routed design was dumped with its SLR (`resources/run1_routed/leaf_cells.tsv.gz`, 1,585,262 cells) and grouped by hierarchy (`tools/r41/util_hier.py slr`). LUT and LUTRAM columns count leaf cells (a site holding two LUT cells counts twice), so they exceed the site counts above; FF, CARRY8, RAMB and DSP counts are exact. SLR0:

| SLR0 group | LUT cells | LUTRAM/SRL cells | FF | CARRY8 | RAMB36 | RAMB18 | DSP |
|---|---:|---:|---:|---:|---:|---:|---:|
| TNCA tile 0 | 268,740 | 17,070 | 65,345 | 9,246 | 32 | 15 | 171 |
| TNCA outlets (three `tnca_line_axi` and their links) | 22,599 | 2,016 | 19,429 | 186 | 0 | 0 | 0 |
| TNCA DCO/TMU replica | 7,615 | 360 | 2,032 | 343 | 0 | 0 | 0 |
| TNCA mesh and other kernel logic | 2,060 | 2,848 | 861 | 72 | 32 | 0 | 0 |
| Platform: HBM subsystem | 10,736 | 5,042 | 17,062 | 58 | 4 | 0 | 0 |
| Platform: other dynamic region | 5,896 | 22 | 8,985 | 85 | 0 | 0 | 0 |
| Platform: static shell | 32,074 | 3,131 | 45,243 | 274 | 62 | 3 | 0 |
| **SLR0 total** | **349,720** | **30,489** | **158,957** | **10,264** | **130** | **18** | **171** |

SLR1 and SLR2 are in `resources/tables/run1_routed_slr.md`. Each tile is placed entirely in its own SLR. The static shell is mostly in SLR1 (68,396 LUT cells there). **What makes SLR0 different from SLR1 is the outlets (all three are in SLR0, next to the HBM controllers) and the HBM subsystem, not the tile.**

Register replication from timing fixes (`_rep` cells) is small: 1,559 of tile 0's 65,345 FFs.

### 2.3 Per tile and per module

Two views are given because they disagree in detail.
- **Kernel synthesis of the exact run-1 source** (`run1src`, commit `c9f2578`, `-flatten_hierarchy none`; `resources/synth/run1src_c9f25787494a/`) keeps the RTL module boundaries, so it attributes cost to the module that owns it.
- **The routed image** (v++'s own synthesis and implementation) moves logic across module boundaries. For example, `u_valu` shrinks from 23,153 to 4,459 LUTs, `u_core`'s own logic grows from 6,818 to 23,268, and `u_cni` grows from 20 to 4,196. The per-tile totals agree within 1.7 % (251,063 in synthesis, 246,868 routed).

Tile 0, kernel synthesis of the run-1 source (the three tiles are within 0.4 % of each other; full tables in `resources/tables/run1src_synth_hierarchy.md`):

| Module | LUT | LUTRAM | FF | RAMB36 | RAMB18 | DSP | Share of tile LUTs |
|---|---:|---:|---:|---:|---:|---:|---:|
| `u_vfpu` vector FPU (16 FMAs, 16 converters, SFU) | 62,834 | 0 | 8,490 | 0 | 15 | 46 | 25.0 % |
| `u_mtx` matrix engine (16 MACs, accumulator file) | 51,071 | 2,856 | 11,440 | 0 | 0 | 32 | 20.3 % |
| `u_slice` NUCA home slice | 30,647 | 9,644 | 5,028 | 0 | 0 | 0 | 12.2 % |
| `g_lsu_pair` two load/store units | 25,435 | 72 | 13,639 | 0 | 0 | 0 | 10.1 % |
| `u_valu` vector integer ALU | 23,153 | 0 | 0 | 0 | 0 | 64 | 9.2 % |
| `g_vrf_banked` vector register file | 21,288 | 0 | 6,291 | 32 | 0 | 0 | 8.5 % |
| `u_txf` tensor transfer engine | 14,494 | 600 | 8,312 | 0 | 0 | 13 | 5.8 % |
| `u_bar` barrier unit | 7,450 | 0 | 3,891 | 0 | 0 | 0 | 3.0 % |
| `u_core` own logic (pipeline, scoreboard, counters) | 6,818 | 0 | 5,933 | 0 | 0 | 0 | 2.7 % |
| `u_srf` scalar register file | 3,756 | 2,072 | 256 | 0 | 0 | 0 | 1.5 % |
| Remainder (`u_salu`, `u_prf`, decoders, `u_mul`, memory mux, NIs) | 4,117 | 0 | 1,730 | 0 | 0 | 16 | 1.6 % |
| **Tile 0** | **251,063** | **15,244** | **65,010** | **32** | **15** | **171** | |

The candidates in §4 and §5 are measured against a second baseline, `base` (commit `67a453b`, the branch head they were built on). Its RTL includes later T2 timing rounds, so its tile 0 has 251,804 LUTs and 68,613 FFs (`resources/tables/base_synth_hierarchy.md`).

By module type, over the whole kernel (run-1 source; `resources/tables/run1src_synth_bytype.md`):

| Module type | Instances | LUT | Share of kernel LUTs | FF | DSP |
|---|---:|---:|---:|---:|---:|
| `tnca_fp32_fma` (binary32 FMA) | 96 | 229,008 | 28.3 % | 25,632 | 192 |
| … of which `tnca_fma_norm_xls` (normalise) | 96 | 130,656 | 16.1 % | 0 | 0 |
| … of which `tnca_fma_align_xls` (multiply, align) | 96 | 51,024 | 6.3 % | 0 | 192 |
| … of which `tnca_fma_round_xls` (round) | 96 | 47,232 | 5.8 % | 0 | 0 |
| `tnca_nuca_slice` | 3 | 92,518 | 11.4 % | 15,044 | 0 |
| `tnca_lsu` | 6 | 74,178 | 9.2 % | 40,914 | 0 |
| `tnca_vector_alu` | 3 | 69,459 | 8.6 % | 0 | 192 |
| `tnca_vrf_banked` | 3 | 63,864 | 7.9 % | 18,873 | 0 |
| `tnca_tensor` | 3 | 43,476 | 5.4 % | 24,934 | 39 |
| `tnca_line_axi` (outlets) | 3 | 22,869 | 2.8 % | 18,009 | 0 |
| `tnca_cvt_comb_xls` (format conversion) | 48 | 22,560 | 2.8 % | 0 | 0 |
| `tnca_barrier` | 3 | 22,350 | 2.8 % | 11,673 | 0 |
| `tnca_acc_rf` (accumulator files) | 3 | 19,581 | 2.4 % | 12,288 | 0 |
| `tnca_tmu` (three DCO replicas) | 3 | 16,836 | 2.1 % | 7,632 | 0 |

## 3. Why the large consumers are large

**Binary32 FMAs (96 instances, 28.3 % of the kernel).** Every vector FP operation goes through one FMA per lane (`tnca_vector_fpu.v` header: add, sub, mul and fma all map onto `tnca_fp32_fma`), and the matrix engine has a second set of 16 per core for its MACs (`tnca_fpu_multi`). Each FMA costs about 2,400 LUTs and 2 DSPs. The DSPs only form the 24×24 product; the alignment, the 80-bit addition, the leading-zero count, the normalising shift and the rounding are in LUTs. The generated normalise stage (`rtl/gen/tnca_fma_norm_xls.v`) evaluates both subtraction orders and both rounding candidates in parallel (`sub_bs_squeezed`, `sub_sb_squeezed`, `dif_squeezed`, `dif_m1_squeezed`, 81 bits each), which is duplicated logic chosen for speed. The device has 9,024 DSPs and the image uses 524, so the arithmetic is LUT-heavy on a part where DSPs are almost unused.

**NUCA slice (30,647 LUTs, 9,644 LUTRAM).** The line array is 32 sets × 8 ways × 512 bits = 128 Kbit (`SETS 32`, `WAYS 8`). Synthesis mapped it to `RAM256X1D ×512` plus `RAM64M8 ×592` (`resources/synth/run1src_c9f25787494a/synth.log`, distributed-RAM report): one copy per distinct read address. The engine reads the array in five states with four distinct address expressions (lookup, maintenance, write-back/wake service), although only one state is active per cycle. The other large state is the MSHR fill buffer (`ms_line`, 8 × 512 bits) and the request, MSHR and target tables, which keep full 64-bit addresses.

**Load/store units (25,435 LUTs, 13,639 FFs per core).** `LSU_INFLIGHT 2` builds `tnca_lsu_pair`: two complete `tnca_lsu` units, so that a second access can start while the first waits for memory. Each unit holds 16 lanes × (64-bit address, 64-bit write data, 64-bit read data) and a beat planner (12,363 LUTs and 6,819 FFs per unit). Only one unit issues at a time, but both carry a full issue path. The second unit never receives an atomic (an atomic needs both units idle and goes to unit 0), yet it carries the atomic path.

**Vector register file (21,288 LUTs per core).** The 256 × 512-bit file (4 contexts × 64 registers) is in BRAM (32 RAMB36). The LUTs are the read side: five read requesters across four banks of 512-bit words, with per-lane valid bits (`ZERO_INVALID 1`) and two writers (13,784 LUTs outside the banks).

**Matrix engine (51,071 LUTs).** Sixteen MACs at about 2,630 LUTs each (2,629 for `g_mac[0]`) are about 42,000. The accumulator file (16 slots × 16 rows × 16 lanes × binary32 = 128 Kbit) is efficient LUTRAM (`RAM64M8 ×20` per lane, 2,560 LUTRAM), but its per-lane valid bits (256 × 16) are registers (4,096 FFs) so that a zeroed slot reads as zeros without clearing the RAM. `MTX_ABUF 1` adds an 8 × 512-bit operand buffer (296 LUTRAM).

**Tensor transfer engine (14,494 LUTs, 8,312 FFs).** Sixteen descriptors (base, five extents, five strides, boxel, rank, dtype, pad) sit in LUTRAM. The registers are the 32-entry 64-bit line buffer (`dl_buf`), the eight store slots, the window table and, with `HAS_TENSOR_XPOSE 2`, the corner-turn state.

**Barrier unit (7,450 LUTs, 3,891 FFs).** 32 barriers, each with two 48-bit epochs and two 8-bit counters, all held in registers because every barrier is compared every cycle.

**Scalar register file (3,756 LUTs, 2,072 LUTRAM).** 256 × 64 bits (4 contexts × 64 registers) with seven read ports (`ra`…`rg` in `tnca_scalar_rf.v`): `RAM64M8 ×280`, one LUTRAM copy per read port.

**Outlets (7,623 LUTs and 6,003 FFs each in the run-1 source synthesis, all three placed in SLR0).** `tnca_line_axi` keeps eight 512-bit slot lines in reset registers (4,096 FFs). Each line can be written by two sources (a request's write data on admission and the AXI R beats), and is read by the W channel (one 256-bit beat of the issuing slot) and by the response (the whole line of `m_rid`). This gives a write multiplexer per bit and two wide read multiplexers.

**DCO/TMU (three replicas, 16,836 LUTs).** `DCO_LOCAL_REPLICAS 1` keeps one TMU state copy next to each slice (a timing fix for the cross-SLR TMU paths); each replica holds the 256-entry live table and the tensor ranges.

**Debug, trace and counters: not a material cost.** All `$display` tracing and the scheduling checkers in `tnca_core.v` are under `ifndef SYNTHESIS` and do not reach the image. The RTL declares 20 × 64-bit performance counters per core, ten 32-bit counters per slice and four 64-bit counters per outlet. Most of the core counters are not connected above the tile (`tnca_tile.v` leaves the stall counters open, and `tnca_multicore_shell.v` connects none of `tnca_top`'s `perf_*` outputs), so synthesis removes them. Counting the registers that survive in the run-1 source synthesis (`resources/host/count_perf_cells.tcl`, log in `resources/synth/run1src_c9f25787494a/`) gives 1,100 `perf_*` registers in the whole kernel (128 in the cores, 972 in the three slices), plus 192 for the outlets' line counters (the beat counters are removed). Together that is about 0.6 % of the kernel's 231,691 registers.

## 4. Capability-neutral removals: built, regression-tested, screened

Both changes are default-off parameters plumbed to `tnca_multicore_hbm_xo` (the image top), so the delivered build is unchanged until the lead integrates them.

### 4.1 `NUCA_SHARED_DATA_READ`: fewer read ports on the slice's line array (SLR0, SLR1, SLR2)

**Change** (`rtl/nuca/tnca_nuca_slice.v`): the line array is read through a state-selected address instead of one address per read site. The engine reads the array in exactly one state per cycle, so the values read are unchanged.
- Mode 1: one read address for every state.
- Mode 2: the lookup (`E_LOOK`) keeps its own port; the reads with registered addresses (`E_HIT` without the data pipeline, `E_MREAD`, `E_WBRD`, `E_WAKE`) share the second one.

**Default-off identity.** A kernel synthesis of the source with the parameter present but 0 (`c1off`, commit `2edc9ab`) reproduces the base synthesis exactly: every row of `resources/tables/compare_c1.md` has a delta of 0.

**Mode 1, measured:**

| Domain | Scope | Base | Mode 1 | Δ |
|---|---|---:|---:|---:|
| Kernel synthesis | Kernel LUTs | 814,043 | 796,292 | −17,751 |
| Kernel synthesis | Kernel LUTRAM | 55,728 | 41,660 | −14,068 |
| Kernel synthesis | Kernel FFs | 242,620 | 241,826 | −794 |
| Kernel synthesis | Tile 0 slice LUTs (LUTRAM) | 30,491 (9,632) | 24,961 (4,948) | −5,530 (−4,684) |
| Tile screen, SLR0 | Tile 0 routed LUTs | 247,579 | 241,999 | −5,580 |
| Tile screen, SLR0 | Tile 0 CLBs (share of SLR0) | 41,878 (76.20 %) | 40,569 (73.82 %) | −1,309 |
| Tile screen, SLR0 | WNS at 10 ns | +0.838 ns | +0.286 ns | **−0.552 ns** |

The line array maps to `RAM256X1D ×512` alone (the `RAM64M8 ×592` copies are gone). **Mode 1 is not recommended.** Its worst path in the tile screen runs from `u_slice/cur_addr` through the 53-bit tag compare (six CARRY8), the lookup way, the shared read address and a RAM read into the atomic merge's write data (24 levels). This path is structural, not functional: the lookup and the wake never share a cycle, but static timing cannot know that. With the run-1 calibration for RAM-capture families in a ≥ 96 % SLR (−0.9 to −1.0 ns, `stage2/run1/README.md`), +0.286 ns would not close in the image.

**Mode 2, measured** (kernel synthesis `c1b2` and its tile-0 screen; that build also has `AXI_LINE_LUTRAM 1`, which lies outside the tile, so the tile and slice rows below are mode 2's alone):

| Domain | Scope | Base | Mode 2 | Δ |
|---|---|---:|---:|---:|
| Kernel synthesis | Slice LUTs, tiles 0 / 1 / 2 | 30,491 / 30,945 / 30,328 | 27,761 / 27,491 / 27,326 | −2,730 / −3,454 / −3,002 |
| Kernel synthesis | Slice LUTRAM, each tile | 9,632 / 9,644 / 9,632 | 7,300 / 7,292 / 7,300 | −2,332 / −2,352 / −2,332 |
| Tile screen, SLR0 | Tile 0 routed LUTs | 247,579 | 245,101 | −2,478 |
| Tile screen, SLR0 | Tile 0 LUT as memory | 15,269 | 12,947 | −2,322 |
| Tile screen, SLR0 | Tile 0 CLBs (share of SLR0) | 41,878 (76.20 %) | 41,471 (75.46 %) | −407 |
| Tile screen, SLR0 | WNS at 10 ns | +0.838 ns | +0.772 ns | −0.066 ns |
| Tile screen, SLR0 | Worst path ending in the slice | +1.267 ns | +1.334 ns | +0.067 ns |

The line array maps to `RAM256X1D ×512` plus `RAM64M8 ×296` (one of the two `RAM64M8` copies is gone). Mode 2's worst tile path is in the SFU's exp2 table address (`u_vfpu/g_sfu`), RTL that this change does not touch; the slice's own worst path is slightly better than in the base screen. One screen per build is a single placement, and run 1 saw the same family move by 0.6 ns between placements, so differences of this size are within placement variance.

**Regression** (§7): the lockstep bench `tb_nuca_shared_read` (both modes, with and without the cache data pipeline, two DCO policies, three seeds, 200,000 cycles each) compares every output of the new and the delivered slice on every cycle under random loads, stores, every atomic opcode, evictions, maintenance and an out-of-order stalling DRAM model. The deliberately broken variants listed in §7 (modes 1 and 2) fail it. `make nuca`, `make nuca-dco` and the board top at the image's parameters (`image_geometry_tests.py`) pass with each mode.

**Risk.** Mode 2: low; it adds one state-decoded multiplexer level on reads whose addresses are all registers. Neither mode changes the number of cycles of any engine state. Mode 1: the 0.552 ns structural path above.

### 4.2 `AXI_LINE_LUTRAM`: outlet slot lines in distributed RAM (SLR0 only)

**Change** (`rtl/fpga/tnca_line_axi.v`): with the parameter set, a write slot's line (written once, on admission, read by the W channel) and a read slot's beats (written by R, read once, on the last beat) are kept in separate distributed-RAM arrays, each with one write port and no reset. A slot's data is read only after the same tenure wrote it, so the bytes on every handshake and every response are unchanged. WDATA differs only while WVALID is low.

**Outlet screen** (one `tnca_line_axi` at the image's parameters: 256-bit data, 8 slots, dynamic limit, placed in SLR0, 10 ns; `flow/fpga/screen_r41_outlet_lines.tcl`, `resources/outlet_screen_1f6bfcb1f0a1/`):

| Routed, per outlet | Base (`LINE_LUTRAM 0`) | `LINE_LUTRAM 1` | Δ |
|---|---:|---:|---:|
| CLB LUTs | 6,942 | 1,389 | −5,553 |
| … LUT as logic / as memory | 6,942 / 0 | 965 / 424 | |
| CLB registers | 6,131 | 2,027 | −4,104 |
| CLBs | 1,952 | 399 | −1,553 |
| WNS | +4.083 ns | +4.057 ns | −0.026 ns |

Kernel synthesis of the image with `AXI_LINE_LUTRAM 1` (`c2_line_lutram`, commit `8017d7c`; `resources/tables/compare_c2.md`):

| Kernel synthesis | Base | `AXI_LINE_LUTRAM 1` | Δ |
|---|---:|---:|---:|
| Each outlet adapter: LUTs | 7,676 | 1,523 | −6,153 |
| Each outlet adapter: LUTRAM | 0 | 552 | +552 |
| Each outlet adapter: FFs | 6,003 | 1,897 | −4,106 |
| Kernel LUTs | 814,043 | 796,996 | −17,047 |
| Kernel FFs | 242,620 | 230,415 | −12,205 |

The kernel delta also contains synthesis noise in RTL that did not change: tile 0's `u_core` moved by +331 LUTs and the TMU by −258. Kernel synthesis of identical core RTL varies by a few hundred LUTs per tile from run to run of different tops (§8). The three outlets are all in SLR0, so the whole outlet saving lands there; the outlet screen puts it at 1,553 CLBs per outlet when routed on its own.

**Regression** (§7): `tools/r40/line_axi_tests.py --line-lutram` runs the 78-configuration outlet matrix with the new build and the delivered build in lockstep inside the bench (`tb_line_axi.LINE_LUTRAM`), comparing every output on every cycle; the board top at the image's parameters runs with the parameter set. The deliberately broken variants listed in §7 (a wrong read slot, a fixed W slot) fail the lockstep check.

**Risk.** Low. The read of `m_rid`'s beats is now an asynchronous LUTRAM read addressed by the AXI R channel's ID; in the screen it does not limit timing (+4.057 ns at 10 ns). A slot's previous contents are no longer reset to zero, which is not observable because a slot's data is read only after the same tenure wrote it (the lockstep run checks this across reset, fault and malformed-burst modes).

### 4.3 Both together (`NUCA_SHARED_DATA_READ 2`, `AXI_LINE_LUTRAM 1`)

Kernel synthesis (`c1b2`, commit `08c2601`; `resources/tables/compare_c1b_c2.md`):

| Kernel synthesis | Base | Both | Δ |
|---|---:|---:|---:|
| Kernel LUTs | 814,043 | 786,966 | −27,077 |
| Kernel LUTRAM | 55,728 | 50,368 | −5,360 |
| Kernel FFs | 242,620 | 229,743 | −12,877 |
| Tile 0 LUTs | 251,804 | 249,405 | −2,399 |
| Tile 0 slice LUTs (LUTRAM) | 30,491 (9,632) | 27,761 (7,300) | −2,730 (−2,332) |
| Outlet adapters, three | 23,028 | 4,569 | −18,459 |

Tile-0 screen of this build: §4.1 (mode 2). The outlets are outside the tile, so their saving is the outlet screen's (§4.2) times three, all in SLR0.

**SLR0 in one line** (screens, not an image): tile 0 −2,478 routed LUTs and −407 CLBs, plus three outlets at −5,553 routed LUTs and −1,553 CLBs each. Run 1's SLR0 used 53,170 CLBs (96.74 %). How much of the screens' CLB reduction a linked image keeps depends on placement and can only be known from a link.

## 5. Measured costs of performance and timing features (author's trade-off)

These parameters add hardware for performance or for timing closure. Turning one off keeps every instruction, format, precision and SFU function, but slows some programs or reopens a timing path, so the decision belongs to the author with a performance non-regression measurement (discipline §2.4). The costs are measured by kernel synthesis of the image with one parameter changed.

Kernel synthesis of commit `08c2601` with images that each differ from `image.tcl` in one parameter, compared with `base` (`resources/tables/compare_p1_p4.md`). The performance column quotes the R39 leave-one-out study (`doc/r39_report.md` §3, RTL simulation on the R39 `tiny` suite, relative to the final R39 configuration). It was not re-measured here.

| Parameter change | Kernel LUTs | Kernel FFs | Where the saving is | What it costs |
|---|---:|---:|---|---|
| `LSU_INFLIGHT 2 → 1` (one LSU instead of the pair) | −35,925 | −20,921 | −11,809 LUTs and −6,966 FFs in tile 0, similarly in each tile | R39-08: disabling it slowed the suite 1.084× (geometric) and +10.7 % in cycles. It is the core's only source of two accesses in flight, so it matters more, not less, for the bandwidth target |
| `DCO_LOCAL_REPLICAS 1 → 0` (one shared TMU instead of three replicas) | −13,746 | −4,921 | The three replicas (17,079 LUTs, one per SLR) are replaced by one TMU of 4,067 LUTs | The replicas were added to close the cross-SLR TMU paths. The shared TMU would sit in one SLR and broadcast its state to the other two; its timing is unscreened |
| `MTX_ABUF 1 → 0` | −1,329 | −15 | −443 LUTs per tile (−296 LUTRAM, the operand buffer) | A timing fix for the VRF-to-MAC operand capture (F09 in `stage2/run1/README.md`), which run 1 left at +0.079 ns in SLR0 |
| `VRF_PREFETCH 1 → 0` | +155 | +44 | None measurable: within kernel-synthesis noise | R39-06/07: disabling it slowed the suite 1.147× |

**Reading.** The only large saving among the performance features is the second LSU, and it buys the memory-level parallelism that the bandwidth plan needs. `DCO_LOCAL_REPLICAS` and `MTX_ABUF` are timing fixes on near-critical SLR0 families. None of these is recommended for removal; they are listed so that the author can weigh each against the area it costs.

## 6. Options that needed the author's approval

Each of these would remove or narrow an architectural capability, a semantic range or an observability interface, so none was built until the author decided. The author's decisions (source: the author, in the lead's chat, 2026-10-28) and what was done with them are in §9.

| Option | What it saves (from the tables above) | What it removes |
|---|---|---|
| Share the matrix engine's 16 MACs with the vector FPU's 16 FMAs | Up to one set of 16 FMAs per core (16 × about 2,400 LUTs and 32 DSPs per core) | Concurrent matrix and vector FP execution; the arch overview's separate matrix unit |
| `NUM_RCTX` 4 → 2 (`PHYS_VREGS` 256 → 128) | Half of the VRF BRAM and of the scalar file's LUTRAM (2,072 per core), smaller per-context state | Two of the four hardware contexts |
| `ACC_N` 16 → 8 accumulator slots | Half of the accumulator file (2,560 LUTRAM and 4,096 FFs per core) | Accumulator slots visible to `a.*` and the matrix instructions |
| `BAR_N` 32 → 16 barriers, or narrower epochs (`EPOCH_W` 48) | Up to half of the barrier unit (7,450 LUTs, 3,891 FFs per core) | Barrier names and epoch range visible to software |
| Narrow internal physical addresses (64 → 34 bits) in the slice, LSU and outlet tables | Part of the 64-bit address registers and comparators in `rq_addr`, `ms_addr`, `ms_wbadr`, `lane_addr`, `address`/`bus_address` | The 64-bit address range seen by hardware; out-of-range addresses would alias instead of faulting |
| A smaller FMA: one subtraction order plus a swap, or a DSP-based normalising shift | Part of the 130,656 LUTs of the normalise stage | Nothing architectural if bit-exact, but it touches the single rounding implementation of the core, so it needs exhaustive numeric verification and the author's agreement before it is attempted |

## 7. Regression results

All regressions ran on GB10 (RTL simulation, iverilog) through the scripts kept with their logs in `resources/regression/`. Each run records its commit in `summary.txt`.

| Run (commit) | Test | What it covers | Result |
|---|---|---|---|
| `g2_regress` (`082c9a2`) | `make nuca`, `make nuca-dco` | Slice benches, default build | pass |
| | `make nuca-shared-read` | Mode 1 against the delivered slice in lockstep: pipeline on/off × policies 0/46 × seeds 1/7/42, 200,000 cycles each | 12/12 pass |
| | `make r40-cache-maintenance` | Ownership handoff on the real-core grid, default build | pass |
| | `image_geometry_tests.py`, base and mode 1 | Board top at the image's parameters | **refused** (`HAS_TENSOR_XPOSE 2` not mapped to the bench); fixed in `d33456a` |
| `g2_regress2` (`1f6bfcb`) | `line_axi_tests.py` | Outlet matrix, default build | 78 configurations + 8 stream cases pass |
| | `line_axi_tests.py --line-lutram` | Same matrix with `LINE_LUTRAM 1` and the delivered outlet in lockstep | 78 + 8 pass |
| | `multicore_axi_tests.py` (1 and 3 cores) | Multicore AXI top, default build | pass |
| | `image_geometry_tests.py`, mode 1 + `AXI_LINE_LUTRAM` | Board top | **refused** (same cause) |
| `g2_regress3` (`d33456a`) | `image_geometry_tests.py` | Board top at the image's parameters: base, mode 1, `AXI_LINE_LUTRAM 1`, mode 1 + `AXI_LINE_LUTRAM 1` | 4/4 pass |
| `g2_regress4` (`08c2601`) | `make nuca`, `make nuca-dco` | Slice benches | pass |
| | `make nuca-shared-read` | Modes 1 and 2, as above | 24/24 pass |
| | `image_geometry_tests.py` | Board top: mode 2; mode 2 + `AXI_LINE_LUTRAM 1` | 2/2 pass |
| `g2_regress5` (`11e653d`) | `make top`, `make mesh`, `r40-grid-dispatch`, `r40-coherent-host`, `r40-multicore-shell`, `r40-line-axi` | Default build of every top that instantiates the changed modules | pass |
| | `make lint` | Every RTL file elaborates | **fails on `rtl/fpga/tnca_stream_bank.v`** (SystemVerilog size casts, last changed in `9ab0276`, the stopped stream unit). The files changed here elaborate. Not caused by G2 |
| | `make compiler` | Compiler tests | exit 0, but `compiler-ir` was **skipped** (no `tnca-opt` build in that tree). The changes are RTL only |

The image-geometry bench runs the MMIO v2 protocol through `tnca_multicore_hbm_xo` at the image's 32 sets × 8 ways and every image option. It covers code upload and readback, two 455-program grids with every result checked, four memory-control settings, coherent host writes, dirty-line download through the ownership handoff, a partial-line handoff and a deadline abort.

The lockstep benches were checked for sensitivity by breaking the candidates on purpose (`resources/regression/mutation/run_mutants.sh`, run on the local development host; output in `mutants.log`). Five mutants were tried: an off-by-one maintenance index (modes 1 and 2), a wrong service way in the shared slice read, a wrong read slot and a fixed W slot in the outlet. The benches reported every one as an output mismatch.

## 8. Method, provenance and preserved failures

**Routed image (read only).** FPGA host `acf3019`, Vivado 2023.2, through `~/r41s2_tools/vivado_slot.sh --min-gb 12`:
`vivado -mode batch -source hier_run1.tcl -tclargs ~/tnca_images/r41s2_run1_c9f25787494a/level0_wrapper_routed.dcp OUT`.
- The routed DCP's SHA-256 was checked on the host before use: `311bd49c…7045215`, the value recorded in `stage2/run1/README.md`.
- The script (`resources/host/hier_run1.tcl`) opens the checkpoint once, writes `report_utilization -hierarchical -hierarchical_depth 20`, `report_utilization -slr` and one row per placed leaf cell (SLR, REF_NAME, PRIMITIVE_TYPE, name), and never writes a checkpoint.
- Outputs, the Vivado log and `SHA256SUMS` are in `resources/run1_routed/`. The tables in §2 are generated from them by `tools/r41/util_hier.py` (`routed`, `slr`, `bytype`) into `resources/tables/`.

**Kernel syntheses.** x86 host (Vivado 2023.2, at most two jobs), `flow/fpga/synth_r40_multicore.tcl` on a `git archive` of the named commit with `evidence/r41/stage2/integration/image.tcl` or one of the screen images in `resources/screens/` (each differs from `image.tcl` only in the parameters its name says). Runs were queued by `resources/host/screen_chain.sh`. Reports and logs are in `resources/synth/<name>_<commit>/`; base versus candidate tables come from `util_hier.py compare` (`resources/tables/compare_*.md`).

| Run | Commit | Image | Purpose |
|---|---|---|---|
| `base` | `67a453b` | `image.tcl` | Baseline for every candidate: the `r41-hbm-bw` head the candidates were built on. Its RTL is not the run-1 source: 39 RTL files differ from `c9f2578` (later T2 timing rounds and default-off additions) |
| `run1src` | `c9f2578` | `image.tcl` | The exact run-1 source (archive SHA-256 `4a2dcaf6…`), for the synthesis-versus-routed comparison of §2.3 |
| `c1off` | `2edc9ab` | `image.tcl` | Default-off identity of `NUCA_SHARED_DATA_READ` |
| `c1` | `2edc9ab` | `image_shared_read.tcl` | Mode 1 |
| `off2` | `8017d7c` | `image.tcl` | Default-off identity of both parameters |
| `c2_line_lutram` | `8017d7c` | `image_line_lutram.tcl` | `AXI_LINE_LUTRAM 1` |
| `c1b2` | `08c2601` | `image_c1b_c2.tcl` | Mode 2 with `AXI_LINE_LUTRAM 1` |
| `p1`…`p4` | `08c2601` | `image_p*.tcl` | One performance or timing parameter off (§5) |
| `off3`, `off4` | `40ceb89`, `8057dea` | `image.tcl` | Default-off identity of the §9 parameters |
| `g2ctx2`, `g2epoch38` | `40ceb89` | `image_g2_ctx2.tcl`, `image_g2_epoch38.tcl` | §9 contexts, epochs |
| `g2addr35`, `g2bar8e38` | `5430243` | `image_g2_addr35.tcl`, `image_g2_bar8_epoch38.tcl` | §9 home address, barriers |
| `g2all` | `8057dea` | `image_g2_all.tcl` | Every §9 parameter with both §4 changes |

**Tile screens.** `flow/fpga/screen_r41_tile.tcl` cuts tile 0 from a kernel-synthesis checkpoint and places and routes it alone in SLR0 at 10 ns. These are the same screens that run 1 calibrated (`stage2/run1/README.md`, "Screen-to-image calibration"). The screens are optimistic, because they have no shell, no other tile and none of SLR0's static density. Reports are in `resources/tile_screens/`.

**Outlet screens.** FPGA host, through the slot helper: `flow/fpga/screen_r41_outlet_lines.tcl` on commit `1f6bfcb`, `LINE_LUTRAM` 0 and 1, one `tnca_line_axi` at the image's parameters, in SLR0 at 10 ns (`resources/outlet_screen_1f6bfcb1f0a1/`).

**Kernel-synthesis noise.** Kernels whose core RTL is identical still differ by a few hundred LUTs per core between syntheses with different tops (for example tile 1's `u_core` +552 in `c1`, tile 0's `u_core` +331 in `c2_line_lutram`). The deltas reported above are on the blocks that changed; the kernel totals include this noise.

**Preserved failures and deviations.**
- The first routed-analysis launch ran with an empty Tcl file: the script did not reach the host through the jump-host pipe, and Vivado exited at once. Its log is kept as `resources/run1_routed/attempt1_failed_*`; the second launch, with the file's SHA-256 checked on the host, is the one used.
- The first mode-1 synthesis (`c1_shared_read`, before the change was put under a `generate`) was stopped and renamed `c1_shared_read_aborted_pre_generate` on the x86 host; it produced no report. It was stopped with a pattern-matching `pkill` that matched only that job's command line, rather than by PID as the server rules ask. Every later stop used PIDs.
- The parameter screens were first started on the FPGA host. While they ran, the shared host's available memory fell to about 8 GB with swap nearly full, so they were stopped (by PID) and moved to the x86 host. The stopped run's log is `resources/host/params_fpga_host_stopped.log`.
- The image-geometry bench refused the stage-2 image in the first two regression runs, because `HAS_TENSOR_XPOSE 2` had no mapping to the board bench. Commit `d33456a` added the mapping. The refused runs are kept in §7's logs.
- Mode 1 of `NUCA_SHARED_DATA_READ` passed every regression but lost 0.552 ns in the tile screen (§4.1); it is kept in the RTL as a documented, non-recommended mode.
- §9's sizing runs used scratch variant profiles on GB10 (kept in `resources/sizing/`, not in `spec/r1_profiles.json`). The first Hub FP8 compile on the full shape list spent hours on host-side tensor setup with every launch skipped. It was stopped by PID and re-run on the shorter list (`hub_v5x8_stopped_long_shapes.log`).
- The first address-width synthesis (`g2addr35` at `40ceb89`) was stopped by PID and renamed `g2addr35_40ceb89e6555_stopped_superseded` on the x86 host, after the fold moved into `tnca_addr_fold` and the TMU notice port was left at 64 bits; `5430243` is the one reported.
- A local run of `make core-g2` was stopped by PID when the session's temporary-file quota filled; `g2_regress6` is the run reported.

## 9. The author's decisions on §6 and what was done

Source: the author, in the lead's chat, 2026-10-28. The decisions were:
1. Sharing the matrix MACs with the vector FMAs: **rejected**. The author's reason: NVIDIA keeps tensor-core and vector MACs separate.
2. Hardware contexts 4 → 2: **approved** (withdrawn on 2026-10-29; see the decision below).
3. Accumulator slots: choose the smallest number that loses no matrix throughput, derived from the matrix compute requirement; if that is 16, keep 16.
4. Fewer barriers and narrower epochs: **approved**, sized from what the compiler and runtime actually use.
5. Narrower internal physical address: **approved**, keeping enough bits for the delivered 8 GiB and the full HBM address space the image maps.
6. Smaller FMA: **rejected**.

Items 2, 4 and 5 are implemented as image parameters whose defaults build the delivered machine (`NUM_RCTX`/`RCTX_W`/`PHYS_VREGS`/`PHYS_W`, which already existed; and new `BAR_N`/`BAR_W`, `BAR_EPOCH_W`, `ACC_N`/`ACC_W` and `PHYS_ADDR_W`, plumbed from `tnca_multicore_hbm_xo` to the core and the home slice). Item 3 resolves to 16 slots, so nothing changes there. The reduced machine is the new profile `core-vrf-sync-v5-x8-g2` (`spec/r1_profiles.json`, generated target `compiler/targets/tnca-sim-vrf-v5-x8-g2.json`): two contexts with 128 physical vector registers, 16 accumulators, 8 barriers and 38-bit epochs, with every instruction, format and capability of v5-x8. The image parameters that build it, plus the home address width and the two §4 changes, are in `resources/screens/image_g2_all.tcl`.

**Decision of 2026-10-29: the contexts cut is withdrawn.** Author decision (source: the author, in the lead's chat, 2026-10-29, answering the lead's questions about the G4 §9 results): **restore 4 hardware contexts.** The 4→2 cut is withdrawn because of its measured slowdown. Every other G2 cut is kept.
- The measurement: one cut at a time on the run-1 image (rtl, `g4/g2_cuts/`). The contexts cut alone made the five points that were slower under G2 0.62–0.94× (geometric mean 0.807), the same as all G2 cuts together (0.808). Every other cut stayed within 1 %.
- The profile for run 2 is `core-vrf-sync-v5-x8-g2c4` (`spec/r1_profiles.json`, generated `compiler/targets/tnca-sim-vrf-v5-x8-g2c4.json`). It is v5-x8 with 8 barriers and 38-bit epochs, and four contexts with 256 physical vector registers. Its core bench is `make core-g2c4` (86 of 86 runs pass; `g4/g2c4/core_g2c4.log`).
- One program of that list runs differently: `t39_pair`'s two producer/consumer pairs need 5 barriers each, more than 8 in total. The bench runs it as one pair. A barrier index beyond a context's slice aliases into the next slice instead of faulting, which is a pre-existing property of the slice mapping that 8 barriers exposes earlier.
- The images: `g4/images/image_g2c4.tcl` (G2's cuts with four contexts), `image_s4c4.tcl`, and the run-2 candidate `g4/run2/image.tcl`.
- The contexts row's saving (−18,476 kernel LUTs, −6,741 per core) is given back; the other rows' savings stand.

### 9.1 How each size was chosen

The sizing runs compile the frozen 113-point suite (`spec/r40/suites/hbm.json`: every GEMM, linear, bmm, conv and attention shape, plus the GPT-2, LLaMA and ResNet blocks) with `default_change_audit.py run --profile P`. That is the B40-S compile knobs, compile only, with every launch skipped. P is a scratch variant profile: `resources/sizing/mkvariant.py` adds one, and gen_target derives its target from the RTL as for any profile. They also compile the Hub FP8 kernels that Qwen's FP8Linear loads (`tools/r41/fp8_hub_compile_only.py`, the shapes of `fp8_kernels_cases.py`). Code is compared with `default_change_audit.py compare` (the code text and launch structure of every kernel) and `tools/r41/asm_resource_scan.py --compare`. Identical code on the same image executes the same instructions, so it cannot lose throughput. Every report, comparison and dump is in `resources/sizing/`.

**Contexts (approved).** All 113 points compile for two contexts (`run-g2v0-ctx2`), and every Hub FP8 kernel compiles. Every kernel that runs on more than one context runs on both (`rctx_spmd` 2 for 451 of 497 kernels). Their code differs from the four-context machine's, as it must. The performance cost is the one R39 measured, quoted here and not re-measured: "R39-05 4 contexts (reverting to v3's 2)", S_geo 1.076×, +0.5 % suite cycles (`doc/r39_report.md` §3, RTL simulation).

**Accumulators: 16, unchanged.** The derivation:
- The matrix engine is one 16-MAC column array that issues one row per cycle, k-outer and m-inner. It returns to a row after 16 cycles, against a multiply-accumulate pipeline of `PIPE = FMA_LAT + OPND_REG` = 3 cycles (`tnca_matrix.v`). It runs one command at a time (`busy` until `S_IDLE`). So a single slot keeps the engine at full rate within a command, and the engine itself needs no more than one.
- The slots a program needs come from two other places:
  - the compiler's output blocking, one slot per 16×16 C tile held, which sets how often A and B fragments are reloaded per MMA;
  - context slicing (`RSLICE`): a program run on c contexts gets 1/c of the slots (`tnca_cc/rctx_spmd.slices`).
- On the two-context machine, the 113-point suite holds at most 6 accumulators per program: 5 or 6 in attention prefill, conv2d, the MLP and the GPT-2, LLaMA-prefill and ResNet blocks. Every such kernel runs on both contexts, so it needs 8 per context, which is 16 in all.
- With 8 slots, 23 of the 113 points change code. The compiler's recorded reason, for all 40 changed kernels, is `rctx_spmd_refused: the kernel uses 5 (or 6) accumulators, more than a slice of 4`: they would run on one context instead of two.
- The Hub FP8 matmul kernels hold all 16 accumulators at once. With 8 they do not compile (`ResourceError: 16 accumulators live at once, target has 8`, `resources/sizing/hub_g2v1.log`).
- 16 is the smallest count that loses nothing, so `ACC_N` stays 16. The slot index is a power of two (`ACC_W`), because slicing shifts the context into the index's top bits.

**Barriers: 8.**
- On the two-context machine, the suite's programs name at most 3 barriers each (every `addmm`, `bmm`, `gemv`, `linear` and `matmul` template). The Hub FP8 kernels name 1.
- With 8 barriers (a slice of 4 per context) the code of all 113 points is identical to the 32-barrier machine's (`cmp-g2v0-g2v2-ctx2-bar8.json`: 113 unchanged), and so is every Hub kernel (`hub-compare-g2v0-g2v2.txt`).
- With 4 barriers, 37 points change. The recorded reason is `rctx_spmd_refused: the kernel uses 3 barriers, more than a slice of 2`.
- The G2 profile, which combines 8 barriers with the 38-bit epochs, compiles all 113 points to the same code as the 32-barrier two-context machine (`cmp-g2v0-g2final.json`), and the Hub kernels identically (`hub-compare-g2v0-g2final.txt`).
- A caveat from older evidence: FP8 kernel dumps recorded earlier on the gather route (`evidence/r41/fp8/models_tnca`, without the X8 transposed transfer) name up to 32 barriers. On v5-x8 and G2, which have the transposed transfer, the same kernels take the X8 route and name one.

**Epoch width: 38 bits.**
- A barrier's epoch counts its generations since the last `mb.init` or launch (`flush` resets every barrier).
- An epoch can advance at most once per cycle, since it takes one `mb.arm`, and the runtime ends every launch at its deadline. The largest deadline the runtime uses is 200,000,000,000 cycles (the stage-2 board transport, `tools/performance_harness/stage2_board.py`, and the Qwen board manifests). That is below 2^38.
- An arm that would overflow the epoch is already a fault in `tnca_barrier.v` (it sets the barrier's error and never wraps). A program that outlived the bound would therefore stop with a located fault, not synchronise wrongly.

**Home physical address: 35 bits.**
- The image maps a core address window of 8 GiB above a 128 KiB local region: `MEM_BASE` 0x20000 in `tnca_multicore_axi.v`; `window_bytes` 8589934592 and `local_limit` 131072 in the board manifests. The TMU register page is at 0x7FFF0000. Every legal line address is below 0x2_0002_0000 < 2^34.
- The slice keeps 34 address bits plus a fold bit set when any higher bit was set (`rtl/nuca/tnca_addr_fold.v`, used by `tnca_tile.v`). An out-of-range address therefore reaches the outlet as an address at or above 2^34, which no aperture contains, and it still faults instead of aliasing a cached line.
- The outlet keeps the same 35 bits for its same-line hazard check.
- The LSU, the NoC flits and the AXI address stay 64-bit. Narrowing them would change the core's datapath and the NoC payload, and was not attempted.

### 9.2 Measured savings

Kernel synthesis, each build compared with `base` (`resources/tables/compare_g2.md`); tile screens of tile 0 in SLR0 at 10 ns. The base tile screen: 247,579 routed LUTs, 41,878 CLBs (76.20 % of SLR0), WNS +0.838 ns.

| Change (image parameters) | Kernel LUTs | Kernel FFs | Tile 0 LUTs (synthesis) | Tile-0 screen: LUTs / CLBs (share of SLR0) / WNS |
|---|---:|---:|---:|---|
| Contexts 4 → 2 (`NUM_RCTX 2 RCTX_W 1 PHYS_VREGS 128 PHYS_W 7`) | −18,476 | −10,733 | −6,060 | 241,188 / 41,175 (74.92 %) / +1.223 ns |
| Home address 64 → 35 bits (`PHYS_ADDR_W 35`) | −319 | −3,168 | −366 | 247,069 / 41,632 (75.75 %) / +0.983 ns |
| Epochs 48 → 38 bits (`BAR_EPOCH_W 38`) | −1,230 | −1,927 | −689 | not screened alone |
| Barriers 32 → 8 with 38-bit epochs (`BAR_N 8 BAR_W 3 BAR_EPOCH_W 38`) | −13,042 | −8,736 | −4,482 | 243,014 / 41,266 (75.08 %) / +0.881 ns |
| Accumulators | 0 (kept at 16) | | | |
| `NUCA_SHARED_DATA_READ 2` with `AXI_LINE_LUTRAM 1` (§4.3) | −27,077 | −12,877 | −2,399 | 245,101 / 41,471 (75.46 %) / +0.772 ns |
| **Everything together** (`image_g2_all.tcl`) | **−62,472** | **−35,807** | **−14,789** | **232,925 / 39,974 (72.73 %) / +0.603 ns** |

Notes on the rows:
- **Contexts.** The saving is in the core: −6,741 LUTs per core. The VRF read side loses 2,757 LUTs, the scalar file loses half its LUTRAM (2,072 → 1,036), and the barrier unit and pipeline lose their per-context state. The vector register file stays in 32 RAMB36 per core, because its width, not its depth, sets the BRAM count.
- **Home address.**
  - In synthesis the slice loses 419 LUTs, 794 FFs and 128 LUTRAM. The tag shrinks from 53 to 24 bits, in the 32 × 8 tag LUTRAM and in the eight-way compare.
  - The outlet adapters gain 431 LUTs and lose 232 FFs each in synthesis, with the hazard check at 35 bits. That LUT increase is measured, not explained.
  - The saving is small in area. In the screen, the tile's worst path ending in the slice is +1.217 ns (base +1.267), and the tile WNS moved from +0.838 to +0.983, within placement variance.
- **Barriers.** The barrier unit goes from 7,450 to 1,577 LUTs and from 3,891 to 929 FFs per core; with 38-bit epochs alone it is 6,613 LUTs.
- **Everything together** is 7.7 % of the kernel's LUTs and 14.8 % of its FFs in kernel synthesis.
  - Tile-0 screen: −14,654 routed LUTs and −1,904 CLBs, from 76.20 % to 72.73 % of SLR0. The outlets (§4.2) come on top of it in SLR0.
  - Its WNS is +0.603 ns, 0.235 ns below the base screen. The worst path is the F09 family, VRF bank read to MAC operand capture (`g_bank[0] mem_reg` → `g_mac[7] a_q`), which the run-1 image left at +0.079 ns in SLR0 and which the base screen has at +0.867.
  - The contexts change shrinks the VRF banks this path starts from. One placement per build cannot separate the change from placement variance, but this family must be watched in a link of the combined build.
- **Default-off identity.** Kernel syntheses with every new parameter at its default reproduce the base exactly: `off3` at `40ceb89` (the parameters' plumbing) and `off4` at `8057dea` (`synth/off4_8057dea1fb61/`). The only RTL change on the branch since `8057dea` is another workstream's removal of the unused stream-engine files (`60b7ab7`), which no image builds.

### 9.3 Regressions and coverage checks

| Run (commit) | Check | Result |
|---|---|---|
| `g2_regress6` (`41661b7`) | `make nuca-shared-read` (bench output width corrected, below) | 24/24 pass |
| | `make nuca-addr-fold`: the 35-bit slice against the 64-bit one in lockstep, addresses spread over 2^34, with and without out-of-range addresses (which must stay at or above 2^34 and change nothing else), with and without the data pipeline, three seeds × 200,000 cycles | 12/12 pass |
| | `make nuca`, `make nuca-dco`, `make top`, `make mesh`, `r40-line-axi`, `r40-cache-maintenance` (default build) | pass |
| | `make core-g2`: core-r39sw5's 44 programs, ISA and engine tests, on the G2 core (two contexts, 8 barriers, 38-bit epochs, X8), with in-order and out-of-order memory. It includes `t15_target_vrf_v5_x8_g2`, which reads the context, accumulator and barrier counts back from the hardware and checks them against the generated target | 88/88 pass |
| | `r1-profile-tests` | pass (6 checks; the G2 profile's own check is `t15` above) |
| | Board top at the image's parameters: base, `addr35`, `bar8_epoch38` | pass |
| | Board top, `ctx2` and `g2_all` | **refused**: the multicore runtime required a four-lane physical chunk contract. Fixed in `b36deb5` (the contract now follows the image profile's context count), re-run in `g2_regress7` |
| | `make compiler` with `tnca-opt` built | unit tests pass (157); `compiler-ir` **skipped**: FileCheck was not found at the default path. Re-run in `g2_regress7` with `TNCA_LLVM` set |
| `g2_regress7` (`18d045c`) | Board top at the image's parameters, `ctx2` and `g2_all`, with `--image-profile core-vrf-sync-v5-x8-g2` (two contexts, 8 barriers, 38-bit epochs, 35-bit home addresses and both §4 changes), and base again after the runtime change | 3/3 pass |
| | `r40-multicore-mmio-checks`, `make compiler` | failed on the environment: an evidence file missing from the GB10 copy, `torch` absent from the system Python, and `R1_STRICT` counting skips as failures. Re-run in `g2_regress8`/`g2_regress9` |
| `g2_regress9` (`18d045c`) | `r40-multicore-mmio-checks` with the torch Python | pass (the base tree `92b2c85` passes too) |
| | `make compiler` with the torch Python and `tnca-opt` built | the checks pass, `compiler-unit` **fails 16/157**. The unchanged base tree `92b2c85` fails the same 16 tests in the same environment (`compiler.failed.txt` = `base_compiler.failed.txt`; the errors are `unknown compile option(s): index_opt` from the torch environment's installed backend), so they are not caused by G2 |
| | `compiler/test/run_tests.py --no-rtl` with the system Python | 157 tests pass, 53 skipped (Triton absent) |
| | `compiler/test/IR/run_ir_tests.py` with `tnca-opt` and FileCheck (`TNCA_LLVM` = the toolchain's LLVM 20, `R1_STRICT=1`) | pass, 6 files, 6 RUN cases |
| GB10 compile runs | 113 points for two contexts, for the G2 profile and for each scratch variant | 113/113 compile in every run |
| | Hub FP8 kernels for v5-x8, two contexts, G2 | all compile; with 8 accumulators the matmul kernels do not (above) |

**A bench defect found and fixed.** Both lockstep benches (`tb_nuca_shared_read`, and `tb_nuca_addr_fold` derived from it) declared their output vector 32 bits shorter than the outputs they concatenate, so the top 32 bits were never compared: `maint_ready`/`maint_done`/`maint_busy`/`req_ready`, `rsp_valid`, the response source and tag, and the top bits of the response data. The §4.1 results above were obtained with the short vector. The corrected bench passes all 24 `nuca-shared-read` runs again in `g2_regress6`; the earlier logs are kept.
