# Where the run-1 image spends its FPGA resources, and what can be removed

Status: G2 of `BANDWIDTH_PLAN.md`, 2026-10-28. **`doc/engineering_discipline.md` is binding** (§2.5: subtract as well as add; removing an architectural capability needs the author's approval).

This document answers three questions about the run-1 100 MHz three-core U280 image (`evidence/r41/stage2/run1/README.md`):
1. where the LUTs, LUTRAM, FFs, BRAM, DSP and CARRY8 go, per module, per tile and per SLR, platform shell included;
2. why the large consumers are as large as they are;
3. what can be removed or resized without losing an instruction, a numeric format, a precision or an SFU function, and how much each change saves when measured.

Every number below is read from a report produced by a recorded run (§7 lists the commands, hosts, commits and hashes). The domain of each number is stated: **routed image** (the run-1 routed checkpoint), **kernel synthesis** (out-of-context synthesis of the whole kernel at 10 ns), **tile screen** or **outlet screen** (out-of-context place and route at 10 ns, in SLR0), or **RTL simulation**. No number here is a board measurement, and no screen is a substitute for a linked image.

## 1. Summary

- **The platform is not the problem; the three tiles are.** Of the routed design's 927,245 LUTs, the TNCA kernel holds 798,355 and the three tiles 740,674 (246,868 / 247,348 / 246,458). Each tile sits entirely in its own SLR. SLR0 additionally holds all three memory outlets, one DCO/TMU replica, the HBM subsystem and part of the static shell.
- **Floating point is the largest single cost.** The kernel instantiates 96 binary32 FMAs (16 in each core's vector FPU and 16 in each core's matrix engine). In kernel synthesis they take 228,912 LUTs, 28.2 % of the kernel; their normalisation stage alone takes 130,656 (16.1 %).
- **Next come the core's data-movement structures:** the two-unit LSU (25,376 LUTs and 13,660 FFs per core), the banked vector register file's read crossbar (13,801 LUTs per core outside the banks), the NUCA slice (30,491 LUTs per tile, 9,632 of them LUTRAM), the tensor transfer engine (14,369) and the barrier unit (7,450).
- **Two capability-neutral removals were built, regression-tested and screened** (both default-off, behind new image parameters):
  - **`AXI_LINE_LUTRAM=1`**: the outlet's eight 512-bit slot lines move from reset registers into single-write-port distributed RAM. Outlet screen: routed LUTs 6,942 → 1,389, FFs 6,131 → 2,027, CLBs 1,952 → 399 per outlet, WNS +4.083 → +4.057 ns. All three outlets sit in SLR0. *(Kernel synthesis: §4.2.)*
  - **`NUCA_SHARED_DATA_READ`**: fewer LUTRAM read ports on the slice's 128 Kbit line array. Mode 1 (one port) saves 4,684 LUTRAM per slice in kernel synthesis, but the tile-0 screen lost 0.552 ns of slack to a structural path, so it is not recommended. Mode 2 (lookup port kept, the rest shared) is the candidate. *(Measurements: §4.1.)*
- **Measured costs of performance and timing features** (removing them keeps every capability but is a performance or timing trade-off, so they are listed for the author, not recommended): §5.
- **Options that would remove or narrow an architectural capability** are listed in §6 for the author's decision and were not implemented.

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
| Other dynamic-region glue (`ulp` own logic, `ulp_cmp`, SLR crossings, resets) | 3,166 in total | | | | | | |
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

To split each SLR by owner, every placed leaf cell of the routed design was dumped with its SLR (`resources/run1_routed/leaf_cells.tsv.gz`, 1.37 M cells) and grouped by hierarchy (`tools/r41/util_hier.py slr`). LUT and LUTRAM columns count leaf cells (a site holding two LUT cells counts twice), so they exceed the site counts above; FF, CARRY8, RAMB and DSP counts are exact. SLR0:

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
- **Kernel synthesis** keeps the RTL module boundaries (`-flatten_hierarchy none`; `resources/synth/base_67a453b77be8/`). It attributes cost to the module that owns it.
- **The routed image** (v++'s own synthesis and implementation) moves logic across boundaries: for example, `u_valu` shrinks from 22,885 to 4,459 LUTs and `u_core`'s own logic grows from 7,962 to 23,268, and `u_cni` from 20 to 4,196. Totals per tile agree within 2 % (251,804 synthesis, 246,868 routed).

Tile 0, kernel synthesis (all three tiles are within 0.4 % of each other; full tables in `resources/tables/base_synth_hierarchy.md`):

| Module | LUT | LUTRAM | FF | RAMB36 | RAMB18 | DSP | Share of tile LUTs |
|---|---:|---:|---:|---:|---:|---:|---:|
| `u_vfpu` vector FPU (16 FMAs, 16 converters, SFU) | 63,181 | 0 | 8,502 | 0 | 15 | 46 | 25.1 % |
| `u_mtx` matrix engine (16 MACs, accumulator file) | 50,912 | 2,856 | 11,456 | 0 | 0 | 32 | 20.2 % |
| `u_slice` NUCA home slice | 30,491 | 9,632 | 4,964 | 0 | 0 | 0 | 12.1 % |
| `g_lsu_pair` two load/store units | 25,376 | 72 | 13,660 | 0 | 0 | 0 | 10.1 % |
| `u_valu` vector integer ALU | 22,885 | 0 | 1,963 | 0 | 0 | 64 | 9.1 % |
| `g_vrf_banked` vector register file | 21,305 | 0 | 6,291 | 32 | 0 | 0 | 8.5 % |
| `u_txf` tensor transfer engine | 14,369 | 596 | 8,439 | 0 | 0 | 13 | 5.7 % |
| `u_core` own logic (pipeline, scoreboard, counters) | 7,962 | 0 | 7,461 | 0 | 0 | 0 | 3.2 % |
| `u_bar` barrier unit | 7,450 | 0 | 3,891 | 0 | 0 | 0 | 3.0 % |
| `u_srf` scalar register file | 3,756 | 2,072 | 256 | 0 | 0 | 0 | 1.5 % |
| `u_salu`, `u_prf`, decoders, `u_mul`, memory mux, NIs | 4,021 in total | | | | | | 1.6 % |
| **Tile 0** | **251,804** | **15,228** | **68,613** | **32** | **15** | **171** | |

By module type, over the whole kernel (`resources/tables/base_synth_bytype.md`):

| Module type | Instances | LUT | Share of kernel LUTs | FF | DSP |
|---|---:|---:|---:|---:|---:|
| `tnca_fp32_fma` (binary32 FMA) | 96 | 228,912 | 28.2 % | 25,632 | 192 |
| … of which `tnca_fma_norm_xls` (normalise) | 96 | 130,656 | 16.1 % | 0 | 0 |
| … of which `tnca_fma_align_xls` (multiply, align) | 96 | 50,928 | 6.3 % | 0 | 192 |
| … of which `tnca_fma_round_xls` (round) | 96 | 47,232 | 5.8 % | 0 | 0 |
| `tnca_nuca_slice` | 3 | 91,764 | 11.3 % | 14,941 | 0 |
| `tnca_lsu` | 6 | 73,986 | 9.1 % | 40,962 | 0 |
| `tnca_vector_alu` | 3 | 68,653 | 8.5 % | 5,889 | 192 |
| `tnca_vrf_banked` | 3 | 63,915 | 7.9 % | 18,873 | 0 |
| `tnca_tensor` | 3 | 43,107 | 5.3 % | 25,317 | 39 |
| `tnca_line_axi` (outlets) | 3 | 23,028 | 2.8 % | 18,009 | 0 |
| `tnca_cvt_comb_xls` (format conversion) | 48 | 22,560 | 2.8 % | 0 | 0 |
| `tnca_barrier` | 3 | 22,350 | 2.8 % | 11,673 | 0 |
| `tnca_acc_rf` (accumulator files) | 3 | 19,117 | 2.4 % | 12,288 | 0 |
| `tnca_tmu` (three DCO replicas) | 3 | 17,079 | 2.1 % | 7,632 | 0 |

## 3. Why the large consumers are large

**Binary32 FMAs (96 instances, 28.2 % of the kernel).** Every vector FP operation goes through one FMA per lane (`tnca_vector_fpu.v` header: add, sub, mul and fma all map onto `tnca_fp32_fma`), and the matrix engine has a second set of 16 per core for its MACs (`tnca_fpu_multi`). Each FMA costs about 2,400 LUTs and 2 DSPs. The DSPs only form the 24×24 product; the alignment, the 80-bit addition, the leading-zero count, the normalising shift and the rounding are in LUTs. The generated normalise stage (`rtl/gen/tnca_fma_norm_xls.v`) evaluates both subtraction orders and both rounding candidates in parallel (`sub_bs_squeezed`, `sub_sb_squeezed`, `dif_squeezed`, `dif_m1_squeezed`, 81 bits each), which is duplicated logic chosen for speed. The device has 9,024 DSPs and the image uses 524, so the arithmetic is LUT-heavy on a part where DSPs are almost unused.

**NUCA slice (30,491 LUTs, 9,632 LUTRAM).** The line array is 32 sets × 8 ways × 512 bits = 128 Kbit (`SETS 32`, `WAYS 8`). Synthesis mapped it to `RAM256X1D ×512` plus `RAM64M8 ×592` (`resources/synth/base_67a453b77be8/synth.log`, distributed-RAM report): one copy per distinct read address. The engine reads the array in five states with four distinct address expressions (lookup, maintenance, write-back/wake service), although only one state is active per cycle. The other large state is the MSHR fill buffer (`ms_line`, 8 × 512 bits) and the request, MSHR and target tables, which keep full 64-bit addresses.

**Load/store units (25,376 LUTs, 13,660 FFs per core).** `LSU_INFLIGHT 2` builds `tnca_lsu_pair`: two complete `tnca_lsu` units, so that a second access can start while the first waits for memory. Each unit holds 16 lanes × (64-bit address, 64-bit write data, 64-bit read data) and a beat planner (6,827 FFs per unit). Only one unit issues at a time, but both carry a full issue path. The second unit never receives an atomic (an atomic needs both units idle and goes to unit 0), yet it carries the atomic path.

**Vector register file (21,305 LUTs per core).** The 256 × 512-bit file (4 contexts × 64 registers) is in BRAM (32 RAMB36). The LUTs are the read side: five read requesters across four banks of 512-bit words, with per-lane valid bits (`ZERO_INVALID 1`) and two writers (13,801 LUTs outside the banks).

**Matrix engine (50,912 LUTs).** Sixteen MACs at 2,628 LUTs each are 42,048. The accumulator file (16 slots × 16 rows × 16 lanes × binary32 = 128 Kbit) is efficient LUTRAM (`RAM64M8 ×20` per lane, 2,560 LUTRAM), but its per-lane valid bits (256 × 16) are registers (4,096 FFs) so that a zeroed slot reads as zeros without clearing the RAM. `MTX_ABUF 1` adds an 8 × 512-bit operand buffer (296 LUTRAM).

**Tensor transfer engine (14,369 LUTs, 8,439 FFs).** Sixteen descriptors (base, five extents, five strides, boxel, rank, dtype, pad) sit in LUTRAM. The registers are the 32-entry 64-bit line buffer (`dl_buf`), the eight store slots, the window table and, with `HAS_TENSOR_XPOSE 2`, the corner-turn state.

**Barrier unit (7,450 LUTs, 3,891 FFs).** 32 barriers, each with two 48-bit epochs and two 8-bit counters, all held in registers because every barrier is compared every cycle.

**Scalar register file (3,756 LUTs, 2,072 LUTRAM).** 256 × 64 bits (4 contexts × 64 registers) with seven read ports (`ra`…`rg` in `tnca_scalar_rf.v`): `RAM64M8 ×280`, one LUTRAM copy per read port.

**Outlets (7,676 LUTs, 6,003 FFs each, all in SLR0).** `tnca_line_axi` keeps eight 512-bit slot lines in reset registers (4,096 FFs). Each line can be written by two sources (a request's write data on admission and the AXI R beats), and is read by the W channel (one 256-bit beat of the issuing slot) and by the response (the whole line of `m_rid`). This gives a write multiplexer per bit and two wide read multiplexers.

**DCO/TMU (three replicas, 17,079 LUTs).** `DCO_LOCAL_REPLICAS 1` keeps one TMU state copy next to each slice (a timing fix for the cross-SLR TMU paths); each replica holds the 256-entry live table and the tensor ranges.

**Debug, trace and counters.** All `$display` tracing and the scheduling checkers in `tnca_core.v` are under `ifndef SYNTHESIS` and do not reach the image. The image keeps hardware performance counters: 20 × 64-bit counters per core (`perf_*` in `tnca_core.v`), ten 32-bit counters per slice and four 64-bit counters per outlet. They are read by the board diagnostics, so they are listed in §6, not removed.

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

**Mode 2, measured:** PENDING-C1B

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

The image has three outlets, all placed in SLR0. Kernel synthesis: PENDING-C2

### 4.3 Both together

PENDING-C12

## 5. Measured costs of performance and timing features (author's trade-off)

These parameters add hardware for performance or for timing closure. Turning one off keeps every instruction, format, precision and SFU function, but slows some programs or reopens a timing path, so the decision belongs to the author with a performance non-regression measurement (discipline §2.4). The costs are measured by kernel synthesis of the image with one parameter changed.

PENDING-P

## 6. Options that need the author's approval (not implemented)

Each of these would remove or narrow an architectural capability, a semantic range or an observability interface. None was built or measured.

| Option | What it saves (from the tables above) | What it removes |
|---|---|---|
| Share the matrix engine's 16 MACs with the vector FPU's 16 FMAs | Up to one set of 16 FMAs per core (16 × about 2,400 LUTs and 32 DSPs per core) | Concurrent matrix and vector FP execution; the arch overview's separate matrix unit |
| `NUM_RCTX` 4 → 2 (`PHYS_VREGS` 256 → 128) | Half of the VRF BRAM and of the scalar file's LUTRAM (2,072 per core), smaller per-context state | Two of the four hardware contexts |
| `ACC_N` 16 → 8 accumulator slots | Half of the accumulator file (2,560 LUTRAM and 4,096 FFs per core) | Accumulator slots visible to `a.*` and the matrix instructions |
| `BAR_N` 32 → 16 barriers, or narrower epochs (`EPOCH_W` 48) | Up to half of the barrier unit (7,450 LUTs, 3,891 FFs per core) | Barrier names and epoch range visible to software |
| Narrow internal physical addresses (64 → 34 bits) in the slice, LSU and outlet tables | Part of the 64-bit address registers and comparators in `rq_addr`, `ms_addr`, `ms_wbadr`, `lane_addr`, `address`/`bus_address` | The 64-bit address range seen by hardware; out-of-range addresses would alias instead of faulting |
| Narrow or remove hardware performance counters (20 × 64 bit per core, 10 × 32 per slice, 4 × 64 per outlet) | The counters' registers and incrementers | Board-level observability used for performance diagnosis |
| A smaller FMA: one subtraction order plus a swap, or a DSP-based normalising shift | Part of the 130,656 LUTs of the normalise stage | Nothing architectural if bit-exact, but it touches the single rounding implementation of the core, so it needs exhaustive numeric verification and the author's agreement before it is attempted |

## 7. Regression results

PENDING-REG

## 8. Method, provenance and preserved failures

PENDING-METHOD
