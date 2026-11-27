# R42 shrink, M0: what in a tile is memory, arithmetic, control, and what a hard block could hold

Branch `R42-shrink` (from `R42` `9dc0712b`), 2026-11-27. Domains: **synthesis** = Vivado 2023.2 kernel synthesis of the run-3 image parameters, hierarchy kept
(`tools/r42/area/images/image_base.tcl`), tile 0, 253,898 LUT (`evidence/r42/shrink/census/tile0_instances.txt`, from the area worker's `kern_base_utilization.rpt.gz`);
**cells** = a primitive census of the same tile's synthesized checkpoint (`census/tile0_cells.csv.gz`, `tools/r42/shrink/cell_census.tcl`); **RTL** = read from `rtl/`;
**sim** = a simulation run here; **estimate** = my derivation, basis stated. Nothing here is a routed, board or link number. The per-unit reading of the RTL was done by three
read-only agents whose raw notes are kept in `census/census_*.md`; every number that matters below was cross-checked against the instance table (which is the ground truth for LUT).

## 0. Hard-block facts used (measured or documented)

| Block | Shape | Read latency | Source |
|---|---|---|---|
| URAM288 | 72 x 4096 (288 Kbit), byte write enable (9 x 8 bit), 1 write + 1 read port in simple dual port, one clock | **1 + IREG_PRE + OREG cycles: 1, 2, 2 or 3** | **sim** of the unisim model (Vivado 2023.2 `URAM288.v`, iverilog), `sim/tb_uram_lat.v`, output in `sim/uram_lat.txt`: IREG_PRE=F,OREG=F: data after the address edge (1 cycle); T,F: 2; F,T: 2; T,T: 3. "3-cycle URAM" is the maximum-Fmax setting, not a requirement at 100 MHz. |
| RAMB36E2 / RAMB18E2 | 36 Kbit: 512 x 72 (SDP) ... 32768 x 1; 18 Kbit: 512 x 36 ... | 1 cycle without the output register, 2 with | documented (UG573), not re-measured here |
| DSP48E2 | 27 x 18 multiplier, 48-bit ALU, 3-input, pattern detect, pre-adder | registered output = extra cycle only when the DSP stage is added | documented; the area worker's inference rules (REPORT.md) |
| Device | 960 URAM288 (0 used), 2,160 RAMB36 (about 32 per tile used), 9,024 DSP48E2 (818 used) | | `utilization_slr.rpt.gz` via the headroom report |

A 512-bit line is 8 URAM (7.1 rounded up, 64 data bits per URAM x 8) or 8 RAMB36 (72 bits wide, 512 deep) however shallow the array: **width, not depth, sets the block count** for every wide array
of this tile (a 4,096-line, 256 KiB line array is still 8 URAM). The hard blocks therefore hold the arrays here nearly for free in capacity and the cost is the number of blocks (8 per 512-bit array).

## 1. Tile 0 in one table (synthesis)

Tile: 253,898 LUT = 236,253 logic + 17,020 LUTRAM + 625 SRL; 68,929 FF; 32 RAMB36 + 15 RAMB18; 0 URAM; 269 DSP.

| Unit | LUT | LUTRAM | FF | Memory-like LUT (arrays + their access/valid logic) | Arithmetic LUT | Datapath mux / crossbar | Control |
|---|---:|---:|---:|---|---|---|---|
| Matrix engine `u_mtx` (16 MAC lanes 51,248; acc file 6,293; own 4,011) | 61,552 | 2,856 | 15,643 | acc file 6,293 (2,560 LUTRAM + valid logic ~3.7k) + `abuf` 296 = ~6.6k | MAC lanes ~49.5k (multipliers 8 DSP/lane, aligners, adders, LZC, rounders) | own 3.6k | ~0.1k |
| Vector FPU `u_vfpu` | 49,907 | 0 | 8,506 | none addressable (all FF/delay registers; SFU tables already 15 RAMB18) | 16 FMA 24,720; converters 7,520 + `cvt_int_f` ~8.1k in own logic; compare/min/max ~3.3k; SFU 2.3k; reduction fold 1.6k | operand select 1.3k, output 0.9k | ~0 |
| NUCA home slice `g_home_pipe` | 39,488 | 9,902 | 6,175 | LUTRAM 9,902 + FF arrays + their muxes: **most of the 39.5k is array access** (see 2.1) | tag compares, AMO ALU (64-bit), victim select | merge/response muxes of 512-bit lines | arbiter, MSHR, fill FSM |
| Vector integer ALU `u_valu` | 23,173 | 0 | 1,970 | none | comparators 3.3k, add/sub ~0.9k, shifters ~3.8k, cxy/finish 3.2k, 1,568 clz/popc, mul 64 DSP | result mux ~5.4k, shuffle crossbar 2.5k | ~0.5k |
| Banked VRF `u_vrf` | 21,270 | 0 | 6,291 | **valid-bit emulation 5.4k + bypass merge 2.0k** (arrays are already 32 RAMB36) | 0 | **read-response crossbar ~11.7k, write steering ~1.9k** | arbitration ~0.1k |
| Transfer engine `u_txf` | 19,503 | 1,516 | 9,497 | LUTRAM 1,516 + FF index arrays 3.5k FF + write/alloc logic ~4.3k + descriptor capture ~1.4k (estimate) | address/bounds/stride ~5.6k, 15 DSP | line alignment ~5.9k | ~0.8k |
| Hybrid LSU `u_lsu` | 19,338 | 674 | 8,203 | `landing` 586, slot metadata 48, beat table 36, FF lane arrays 6,437 FF | 49 64-bit address adders ~3.7k, 512-bit byte shifters ~7k | lane extract/merge ~3k | ~1k |
| Core own logic `u_core` | 8,756 | 7 | 9,269 | scoreboard 592 FF (~2.0-2.3k LUT), pc/context arrays | 68 CARRY8 only | writeback/operand muxes ~3.6k | PC/fetch/trap ~1.9k |
| Scalar RF `u_srf` | 3,756 | 2,072 | 256 | all of it (256 x 64, seven read ports as seven LUTRAM copies) | 0 | 7 read muxes | 0 |
| Barrier `u_bar` | 1,577 | 0 | 929 | none (8 entries updated in parallel) | epoch compare ~83 % | | |
| Predicate RF `u_prf` | 998 | 0 | 1,008 | all of it (64 x 16 FF + reset loop + 3 async reads) | 0 | | |
| Scalar ALU 1,468; mul 445 (16 DSP); mem mux 646; decoders 963; host mux 982 | 4,504 | 0 | 648 | | shifters 576, adders | | decode |

Of 253,898 LUT: **array LUTRAM is 17,020 (6.7 %)**; including the LUT logic that exists only to access arrays held in FF or LUTRAM (valid-bit emulation, read/write steering of banked and
slotted arrays, the slice's array access) the memory-like part is about 65-75k (26-30 %, estimate: home slice ~30k, VRF ~19k, acc ~6.3k, srf 3.7k, LSU/txf ~12k, prf 1k). **About 70 % of the
tile is arithmetic and datapath muxing** (MAC lanes 49.5k, FMA 24.7k, converters ~15.6k, valu 23k, address/shift datapaths in LSU and txf ~20k).

## 2. Memory-like structures, one by one

Columns: bits resolved with the image parameters; today; combinational consumer of the read data in the same cycle (this decides the restructuring); hard block and read-latency consequence;
what in the RTL changes; expected LUT saving per tile (all **estimate** unless "attributed").

### 2.1 NUCA home slice (`rtl/nuca/tnca_home_pipe.v`; HOME_MSHR_N 64 -> MSHR_N 64, TARGET_N 128, RQ_N 16; SETS 32, WAYS 8; ADDR_W 35, tag 24 bit)

| Array (line) | Bits | Today | Same-cycle consumer | Hard block / latency | RTL change | LUT saved |
|---|---|---|---|---|---|---|
| `data_mem` (204) 256 lines x 512 | 131,072 | LUTRAM, two async read ports (S3 and the maintenance scan): cells: 512 RAM256X1D (8 LUT each) + r2 copy: **6.4k LUTRAM LUT raw** | S3: response data, AMO ALU, merge, victim writeback data, in the S3 cycle | 8 RAMB36 or 8 URAM288; latency 1 (read issued in S2 with the index S3 will use; **no added cycle**; held by read enable while S3 holds) | **built** (`DATA_RAM`, section 3): read in S2, this cycle's write bypassed into the read (found by the miter: a victim line read in the cycle S3 merges a store into it) | 4-6k |
| `tg_wdata` (221) 128 x 512 | 65,536 | LUTRAM 1,172 raw | fill unit: AMO/merge of the target in the cycle `fu_t` is valid | 8 BRAM/URAM, latency 1 if the next `fu_t` (known one cycle earlier: `tg_next[fu_t]`, `ms_head[...]`) is the read address | read address = the value loaded into `fu_t`; hold while the fill unit stalls | ~1.1k |
| `ms_line` (218) 64 x 512 | 32,768 | LUTRAM 586 raw | fill unit, `fu_first` | 8 BRAM/URAM, address = `fill_pick`/`fill_next_pick` (known a cycle early) | as above; write port is the memory response | ~0.55k |
| `rq_wdata`, `wbq_data`, `rsq_data`, `rq_cmp`, tag banks (8 x 32 x 24) | 8,192 / 2,048 / 2,048 / 1,024 / 6,144 | LUTRAM ~1.2k raw | queue heads | too small / not worth blocks | none | - |
| `ms_addr` 64 x 29, `rq_addr` 16 x 35, v/d/cl bits 3 x 256, `older` 16 x 16, `victim_ptr` 32 x 3 | | FF 1,856 + 560 + 768 + 240 | parallel compares (shadowing, MSHR match) | not RAM-shaped (parallel match) | none | 0 |
Home slice total 39,488 LUT; 9,902 LUTRAM; 6,175 FF. Remaining ~29.6k logic LUT: S2 decision (8-way tag compare 8 x 24 bit, victim select, MSHR match over 64 entries, request shadowing over 16 x 16), AMO ALU, 512-bit merge, response/wb queue muxes.

### 2.2 Accumulator file `tnca_acc_rf` (in `u_mtx`; ACC_N 16, ROWS 16, VLEN 16, ELEN 32)

| Array | Bits | Today | Same-cycle consumer | Hard block / latency | RTL change | LUT saved |
|---|---|---|---|---|---|---|
| lane data 16 lanes x 256 x 32 | 131,072 | LUTRAM 2,560 (attributed) | MAC lane operand `c`; `a.vread`; row ops | 8 RAMB36 (72 x 512, half used; two 32-bit lanes per block, `WEBWE` {4 x en[2k+1], 4 x en[2k]}) or 8 URAM; latency 1: the sweep issues its read address one cycle early (S_LOADB precedes every S_MAC); read row m at t, write at t+5, next read of m at t+17 (12 cycles of slack) | address one cycle earlier; row-op/`a.vread` paths need the row a stage earlier or one extra result cycle (largest edit, scoreboard not traced) | 2.6k |
| valid bits 256 x 16 | 4,096 | FF + 256:1 read mux ~1.0k LUT + set/clear/decode ~0.76k LUT | read zero where invalid | (a) physical zero + 256-cycle reset sweep (acc_rf -> ~50 LUT, frees the 4,096 FF); (b) valid in the parity bits (~550 LUT); (c) keep FF valid | (a)/(b) need a post-reset sweep through the idle write port | 4.0k with (c) kept; ~6k with (a) (estimate) |

### 2.3 Banked VRF (`tnca_vrf_banked` + 4 x `tnca_rf_bank`; 256 x 512, NUM_PORTS 5 read requesters, NUM_WRITERS 2, ZERO_INVALID 1, RDW_BYPASS 1, READ_LATENCY 1)

The 4 banks x 64 rows x 512 bit **are already RAMB36 (8 per bank, 32 total)**; they cost 0 LUT. The 21,270 LUT are the periphery:

| Part | LUT | What it is | Change | Saving |
|---|---:|---|---|---:|
| `vld` valid-bit emulation (4 banks: write enables 1,024 LUT4 + 64:1 read mux 16 lanes x 64 rows, per bank) | ~5.4k (attributed, cells scaled) | unwritten lane reads zero | valid bits into the BRAM parity bits (528 of 576 bits per row) + a 64-cycle reset sweep through the write port (the contract allows a scrub, `spec/r3v2_contract.json:67-72`; whether the launch gap exceeds 64 cycles after reset not determined) | ~5.4k LUT, 4.1k FF |
| bypass merge (`byp ? byp_data : vld|byp ? raw : 0`, 512 LUT4 per bank) + `s1_byp_data` 512 FF per bank | ~2.0k | read-during-write | keep (needed: the BRAM returns old data on collision) | 0 |
| read-response crossbar 4 banks -> 5 ports x 512 bit, selects decoded from the 3-bit port id per bit (5.6 cells per output bit) | ~11.7k (attributed) | | one-hot port selects registered in the bank (2 LUT6 per bit) | ~6.6k |
| write steering 2 writers -> 4 banks (2,048 LUT4 + row/mask muxes) | ~1.9k | | none | 0 |
URAM would not help: 32 URAM per tile for 1.6 % depth use and the RTL needs latency exactly 1.

### 2.4 Scalar RF, predicate RF, barrier, core arrays

| Structure | Bits | Today | Change | LUT saved |
|---|---|---|---|---:|
| `u_srf` 256 x 64, 7 read ports | 16,384 (x 7 copies) | 7 LUTRAM copies, 2,072 LUTRAM + 1,684 logic; ports A-C read in R and registered, D-G consumed in X (D on the redirect path) | one RAMB36 per read port (7), A-C address one stage earlier, selected by `stall_front` | ~3.3k (estimate), 256 FF |
| `u_prf` 64 x 16, 3 async reads, reset loop | 1,024 | FF 1,008 + 998 LUT | LUTRAM + valid bits | ~0.7k LUT, 0.9k FF |
| `u_bar` 8 entries x (epoch 38 x 3 + flags) | ~900 | FF, all entries update in parallel | none (not RAM-shaped) | 0; narrower epoch is a parameter |
| core `pc` 4 x 64 (5 write sources per entry), scoreboard `sb_*` 592 FF (3 writers per bit, ~16 reads per cycle), `instret`, trap record | | FF | scoreboard has no hard-block shape; `pc` single arbitrated next-PC ~0.5k; `instret`/trap record to LUTRAM ~0.3k | ~0.8k |

### 2.5 Hybrid LSU and transfer engine (agent note `census/census_lsu_txf.md`)

| Structure | Bits | Today | Hard block / latency | Change | LUT saved |
|---|---|---|---|---|---:|
| fast `landing` 64 x 512 | 32,768 | LUTRAM 586 | 8 BRAM/URAM; address one stage earlier; **+1 cycle per load completion** (agent) | read pick one stage earlier | 0.59k |
| `ls_data` 64 x 512 (txf line slots) | 32,768 | LUTRAM 586 (+20) | as above; +1 cycle per line | | 0.6k |
| `xb_d` 16 x 512 (transposed-load line buffer) | 8,192 | LUTRAM 296 | 8 BRAM; read address `xh_e` is a register: **no added cycle** | | 0.3k |
| descriptor table 16 x ~862 | 13.8k | LUTRAM 552 | 12 RAMB36 parallel, or 1 RAMB36 serial (+~15 cycles per command) | | 0.55k |
| `sl_dlo`/`sl_dhi` 8 x 128 | 1,024 | FF 960 + ~1.5k raw LUT (zero-init write at slot allocation plus data write) | LUTRAM if the zero write can go (RTL comment `tnca_tensor.v:1473` says the zero is never kept; not simulated) | | ~1.2k LUT, 0.96k FF |
| `dl_buf` 32 x 64 (888 FF read) | 2,048 | FF + 926 raw LUT write demux | remove: write each word into the descriptor RAM as it arrives | | ~0.9k |
| slow-LSU lane arrays (`lane_addr`, `ln_*`, 6,437 FF, all 16 lanes written/read per cycle) | | FF | not RAM-shaped; `ln_wlo` is a copy of `lane_addr[63:3]` (-976 FF) | | 0 LUT |
| `xbuf` 16 x 64 (column reads of all 16 rows per cycle) | 1,024 | FF crossbar | no RAM shape | | 0 |
| `abuf` 8 x 512 (matrix A buffer) | 4,096 | LUTRAM 296 | 8 BRAM (512-bit write) | | <= 0.3k |

### 2.6 Everything else that is memory-like in the vector units

`u_valu`, `u_vfpu` and the MAC lanes contain **no addressable array** (the FPU's `acc` 512 FF, `dly_v` 1,024 FF, `cvt_x` 516 FF are parallel-read registers; the SFU `src`/`res` are
indexed but 1,024 FF in a lane-serial unit; its four tables are already 15 RAMB18). Nothing there can move to URAM/BRAM.

### 2.7 Sum of the memory-like candidates (estimate, per tile)

NUCA data + aux arrays 5.5-7.7k; acc file 6.6k (with the valid logic changed) ; VRF valid bits 5.4k + crossbar selects 6.6k; scalar RF 3.3k, predicate RF 0.7k; LSU/txf arrays ~5.5k (several at +1 cycle);
core ~0.8k. **Total about 35-42k LUT (14-17 % of the tile)**, of which about 17k is array LUTRAM and the rest is the access and valid-bit logic around arrays. It does not reach the 44-49 % reduction
that two tiles per SLR need (section C of the area report), and it leaves the arithmetic (MAC lanes, FMA, converters, valu, address datapaths) untouched.

## 3. Arithmetic and what DSP48E2 could hold (all estimates; only the NUCA, acc and VRF items above are built in this branch)

| Block | Count x LUT (synthesis) | What it is | DSP evidence |
|---|---|---|---|
| MAC lane (`tnca_mac_lane`, 5 XLS stages) | 16 x 3,203 (b0 460, widen 2 x 124, s1 707 + 8 DSP, s2 217, f2 449, norm 535, round 490, own 97) | 4 x 21x18 (DP4), 2 x 11x11 (DP2), 24x24 on 2 DSP (FP32); aligners, 3 adders, LZC, rounder | the area worker's DSP adders: no gain (REPORT.md); the K-fused E4M3 lane (`R42-kfuse-rtl`) is 27,993 LUT per core at G=4 against 44,672, E4M3 only |
| FMA (`tnca_fp32_fma`) | 16 x 1,545 | 24x24 multiplier (2 DSP) + align + add + LZC + round | DSP adders: 1,396 LUT / 5 DSP vs 1,351 / 2 (no gain) |
| `cvt_int_f` in `u_vfpu` own logic | 16 x ~507 = ~8.1k; `cvt_comb_xls` 16 x 470 = 7.5k | 32-bit negate, LZC, shift, round / f32 -> int | `tnca_cvt_int` 418 LUT exists behind `FMA_IMPL` bit 8 (not set in the image): -1.4k per tile (area worker, bit-identical) |
| valu comparators (`mm_gt`) | 120 x 32 cells = ~3.3k | pairwise unsigned `>` for min/max | each has a flop behind it: 120 comparators could be DSP pattern-detect/48-bit ALU (untested) |
| valu shifters, cxy/finish, result mux | ~3.8k, ~3.2k, ~5.4k | | not DSP-shaped (byte/bit selection); the signed 33 x 33 multiply on four DSP per lane could absorb `cxy` and the finish subtracts (untested) |
| LSU address adders (slow unit) | 49 x 64-bit = ~3.7k | `base + (idx << size)` | DSP with a register behind: 2 DSP per 64-bit add; the operands sit between registers (untested) |
| LSU/txf byte shifters | ~7k + ~5.9k | 512-bit shift by 0..63 B | not DSP-shaped |
| txf bound/stride adders and compares | ~5.6k | 64-bit subtracts and compares of ~33-bit values | narrowing is cheaper than DSP (estimate -0.8k) |
| scalar ALU shifters, `u_mul` fabric adders | 576; ~400 | | `u_mul`: the 5 fabric adders absorbed by a DSP cascade (-400 LUT, -190 FF, estimate) |
Arithmetic that is a DSP candidate and is not already one: roughly 120 comparators 3.3k + 49 address adders 3.7k + the K-fused / converter items; none of it comes near the ~110k a halved tile would need.

## 4. Verdict of the census (facts, no recommendation)

1. LUTRAM is 6.7 % of the tile; the arrays that URAM/BRAM can take hold about 17k LUT of LUTRAM and about 20-25k LUT of valid-bit and steering logic. Capacity is not the issue (width is), so block counts are 8 per 512-bit array.
2. The synchronous read costs **no cycle** where the read address is known one cycle early (home data array, acc sweep, `xb_d`, scalar RF with an address move); it costs one cycle for the LSU `landing` and txf `ls_data`
   completion paths. URAM with latency 1 is possible at 100 MHz.
3. Arithmetic and datapath muxing is about 70 % of the tile and is untouched by hard-block memory.

## 5. Addendum: what the builds measured against this census (OOC, 10 ns synthesis; details in RESULTS.md)

- NUCA home slice unit (wrapper, run-3 parameters): shipped 38,455 LUT / 9,894 LUTRAM; with `DATA_RAM`/`AUX_RAM` 2 (24 URAM) 29,214 LUT / 1,696 LUTRAM: the census attribution above (data array 4.1-6.4k raw LUTRAM, `tg_wdata` 1.2k, `ms_line` 0.6k) was right in order of magnitude; the saving
  is -9.2k LUT (-24 %), 1.7k LUTRAM remain (request/write-back/response queues, tag banks).
- Banked VRF unit: shipped 21,289 LUT (the tile instance table says 21,270). Measured: response crossbar `XBAR` 2 -7,373 (census estimate 6.6k), valid bits in RAM -5,323 (census 5.4k), both -12,709.
- One tile (`tnca_tile` OOC): 249,921 shipped; 230,979 with `FMA_IMPL` 465 + home URAM + VRF `XBAR` 2: -18,942 (-7.6 %). Memory-like logic is therefore not where the area is: the remaining 231k is 70+ % arithmetic and datapath multiplexing.
- 4x slice (`SETS` 128): URAM version 42,903 LUT (line array in the same 8 URAM; the cost is the tag arrays and the 3 x 1,024 valid/dirty/claim flip-flops with their 8-way read and victim muxes, +13.7k logic); shipped LUTRAM version 76,444 (29,794 LUTRAM).
