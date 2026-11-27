# Census: vector integer ALU, vector FPU, matrix engine (tile 0, image_base parameters)

Resolved parameters: VLEN 16, ELEN 32 (512-bit vectors); VALU LATE = 2 (`TNCA_UARCH_VALU_LATE`, rtl/common/tnca_uarch_defaults.vh:62; the image does not override it); FP_LAT 2, HAS_SFU 1, FMA_IMPL 1 (so BE = (1/16)%16 = 0, converter select (1/256)%2 = 0, i.e. the `cvt_int_f` function is built, not `tnca_cvt_int`); MAC_DP 1, MAC_CUTS 29 = 0b011101 (cuts 0,2,3,4; MAC_LAT 4), OPND_REG 1, so PIPE = 1+4 = 5; MTX_ABUF 1; ACC_N 16 x ACC_ROWS 16 (core default, never overridden) = 256 rows.

Currency. "LUT" = instance-table LUT (tile0_rel.txt). Function-level attribution uses the census CSV cone names (Vivado names a LUT after the output net of the cone it was split from, so cones are real; the CSV counts LUT cells and is higher than the table: VALU own 25,043 cells vs 21,605, VFPU own 18,635 vs 15,337, MTX own 4,645 vs 3,691 logic, acc 4,115 vs 3,733). Each cone is scaled by its unit's ratio (0.863 / 0.823 / 0.795 / 0.907). Splits INSIDE a cone are estimates; I say so. yosys on isolated one-lane pieces was used only as a relative weight (it is off by x0.4 to x1.3 against the known `tnca_lzc` 63 and `tnca_popc` 35).

## 1. Vector integer ALU `u_valu` (tnca_vector_alu.v) -- 21,605 own + 1,568 (16 clz 63 + 16 popc 35) + 1,970 FF + 64 DSP

No memory arrays. FF = pipeline registers of the LATE=2 split (lines 424-464): q_pu 16x64 (544 FF + the rest in the DSP P registers), q_cxy 16x32, q_key 16x32, q_add 4x32, q_bit 4x32, q_gt 120, control ~26 = 1,970. `u_valu_ref` (core.v:3386) is under `ifndef SYNTHESIS`. With LATE=2 the t_add/t_bit trees and `red_result` (275-301, 336) and `ps_hi`/`lane_mul` (147-159) are dead: the CARRY8 inventory (724) matches the live logic only: vres 144 = 16 lanes x (32-bit add 4 + 33-bit sub 5); pres 32 (equality, 2/lane); q_cxy 64; q_gt 240; q_add 16; late_v 164; clz 64.

| Function (lines) | Structure: count x width | Evidence | LUT |
|---|---|---|---:|
| Pairwise min/max comparators `mm_gt` (312-321, 457) | 120 = 16*15/2 unsigned 32-bit `>` | cone q_gt 3,840 cells = 120 x 32 (16 chain positions, 2 CARRY8 each, S+DI cells) | 3,310 |
| Per-lane datapath, result mux (87-185): add, sub/borrow, 3 shifters, logic, min/max, sel, splat, iota, clz/popc select, 27-way `case(func)`; cmp case (188-200) | 16 lanes x 32 bit | cone `vres` 14,532 cells less the shuffle share below; add+sub exact from CARRY8 (144) ~0.9k; 3 barrel shifters (shl, lshr, ashr written separately) ~3.8k (estimate: yosys isolated 91/92/97 cells each x16); result mux + min/max + logic + decode ~5.4k (remainder) | 10,080 |
| Shuffle (207-242): `src_idx` (16 x ~18 cells), ONE crossbar 16 dst x 32 bit x 16:1, v.insert mux | 512 output bits | crossbar 16:1 = 4 LUT6 + 2 F7 + 1 F8 per bit = 2,048 cells; F8 count 581 in cone matches 512 (estimate for the split) | 2,460 |
| Multiplier operand mux `y = use_imm ? imm : b` | 16 x 32 | cone q_pu_reg 512 cells (LUT3) | 440 |
| Multiplier 32x32 unsigned (147) | 16 | 4 DSP48E2 per lane = 64 DSP, no further LUT visible | 0 |
| MULH.S correction operand `cxy` = (x31?y:0)+(y31?x:0) (153) | 16 x 32-bit add | cone q_cxy 992 cells, 4 CARRY8/lane | 860 |
| LATE=2 finish stage (466-510): `pu_hi - cxy` 16 x 33-bit sub + func select (68 cells/lane, yosys), reduction finish (f_add 3 adds, f_bit, f_win 16 x 15-AND, f_best 16:1 AND-OR x 32 ~270, f_red), q_late_v mux | 512 out bits | cone late_v 2,748 cells, 164 CARRY8 (split = estimate) | 2,370 |
| Reduction input stage: q_key (16x32 mask/xor), q_add (4 groups x 4 masked 32-bit operands, 4 CARRY8 each), q_bit (4 groups x 32 x and/or/xor) | | cones 518 / 512 / 775 cells | 450 / 440 / 670 |
| Predicate (342-370), v.extract `a[mod]` 16:1 x 32 (400), pres | | cones pres 276, sres 215, late_s 32 cells | 480 |
| clz 16 x 63, popc 16 x 35 (98-106) | child instances | tile table | (1,568) |

Check: sum of the rows = 21,560 vs 21,605. Arithmetic -> DSP48E2 (27x18, 48-bit ALU, pattern detect): the multiplier is already on DSP. The R42 area worker found (REPORT.md, dsp_infer.v) that an adder reaches a DSP only with a register behind it and a static add (1 DSP, 0 LUT for a registered 48-bit add); dynamic add/sub with carry-in stays in carry chains. Candidates with a register behind them: the 120 comparators (q_gt flop follows; as key_j - key_i with the sign at P[47]: one DSP each = 120 DSP, removes ~3.3k LUT; NOT tested: all tests registered the full word, and `>` as written maps to carry chains) and the 16 `cxy` adders (q_cxy flop follows; the operand gating `x31 ? y : 0` would need OPMODE-by-data or stay in LUT). A signed 33x33 product also fits the same 4 DSPs per lane (17+16 bit tiling), which would remove cxy (860) and the finish subtracts (~1.1k) -- inference of a 33-bit signed product to 4 DSPs is not tested. Add/sub/eq of the lane are combinational into the result mux: stay in carry chains. R42 REPORT.md "Failed or not done" lists the 120 comparators as "about 3.8k LUT, derived only"; the cone confirms 3,840 cells (3,310 at table scale).

## 2. Vector FPU `u_vfpu` (tnca_vector_fpu.v) -- own 15,337 LUT + 2,145 FF

| Function (lines) | Structure | Evidence | LUT |
|---|---|---|---:|
| int<->binary32 convert `cvt_l` = `cvt_int_f` (285-379, 397-400; `cvt[i]` at 478 is simulation-only) | 16 lanes, 32-bit negate, 32-bit LZC, left shift, 24-bit round; f32->int 64-bit shift/compare, saturate | OOC measured 507 LUT/lane (REPORT.md:62, 27 CARRY8; cone CARRY8 590 less 160 = 430 = 16 x 27 agrees); `tnca_cvt_int` 418-422 exists behind FMA_IMPL bit 8 (not set) | 8,110 |
| FP min/max, neg/abs, compare (430-475): `lt32` (two 32-bit +/- and a compare = 10 CARRY8), NaN/zero tests, 5-way select, 10-way cmp case | 16 lanes | 160 CARRY8 in cone; eq compare 2 CARRY8/lane (cone dly_p, 32); remainder of cone dly_v after the converter: 3,250 = min/max ~2,300 + nf_v major/func mux ~450 + `dly_v[1]` 3:1 injection mux (628-633) ~500 (estimates) | 3,250 |
| FMA operand select fa/fb/fc (173-203) | 16 x 3 x 32 = 1,536 bits, 1 LUT/bit | cone g_fma.u_fma 1,539 cells | 1,270 |
| Reduction fold (485-540): `red_step` lanes 0..7 (FADD pass, FMAX/FMIN: 2 NaN tests + `lt32` + 3:1 select), acc load `mask ? a : ident` (16 lanes, identity from func), capture mux | 8 x lt32 | cone acc 1,934 cells, 80 CARRY8 = 8 x 10 | 1,590 |
| Output: vres `out_from_fma ? fma_y : dly_v[1]`; mc_vres `sfu_done ? sfu : acc[0]` (672-680); dly_p/cmp case | 512 + 512 bits | cones 515 / 512 / 301 cells | 420 / 420 / 250 |
| Control (red_busy, lvl, rwait, ...) | | ~20 cells | 20 |

Sum = 15,330 vs 15,337. The 16 FMAs (1,545 each), 16 `cvt_comb_xls` (470 each, vf.cvt.fmt), SFU (2,330) are children.

Memory-like state (all FF; none is addressed by an index, so none is a RAM candidate):

| Name (line) | Dimensions | Implementation | Note |
|---|---|---|---|
| `acc` (131, 509-540) | 16 x 32 = 512 FF | FF, static wiring: all 16 entries are read at once (`fa = acc[2i%16]`, `fc = acc[(2i+1)%16]`), lanes 0..7 captured per level | not RAM-shaped (16 parallel reads/writes) |
| `dly_v` (603-635) | FP_LAT 2 x 512 = 1,024 FF, sync reset, stage 1 has a 3:1 injection | FF | pure 2-deep delay except reset and injection; an SRL32 form would cost 512 LUT to save 1,024 FF |
| `cvt_x` (222-232) | 16 x 32 (+4) = 516 FF | input register of the converters | timing register |
| `dly_p` 32, `dly_fn/maj` 26, `red_*` ~30 FF | | | |
| SFU children (tnca_sfu.v:110-111): `src` 16 x 32 (indexed read by `cnt`), `res` 16 x 32 (indexed write by `land_ix`) = 1,024 FF with reset loops; exp2/log2/rcp/rsqrt tables 15 RAMB18 | child, own 241 LUT / 1,041 FF | tables are already block RAM; `res` is written one lane/cycle but read as 512 bits, so it is not a RAM shape |

## 3. Matrix engine `u_mtx` (tnca_matrix.v) -- 61,552 LUT total

Own logic 4,011 = 296 LUTRAM + 24 SRL + 3,691 logic; 1,787 FF; 128 DSP (16 x 8).

| Own logic function | Structure | Evidence | LUT |
|---|---|---|---:|
| MAC operand select fa/fb/fc (309-321) and A-element select `a_word` (use_abuf ? abuf : vrf, 512-bit 2:1) + `a_dp_v`/`a_elem_v` (265-276: 16:1 x 32 twice, 64:1 byte x 2 x 8, 32:1 x 16-bit) | fa/fb/fc 16 x 3 x 32 = 1,536 bits; A select ~1,250 cells | cone u_mac inputs 2,863 cells (hand sum 2,784) | 2,280 |
| B row select `b_nd`/`b_dp` + `b_row` load (244-263, 530-533) | 16 lanes x 32 | cone b_row 1,086 cells, 120 MUXF | 860 |
| acc write-data mux `direct_w ? direct_data : mac_bus` (377) + acc address muxes (382-384) | 512 bits | cone u_acc 545 cells | 430 |
| Sequencer, k/m/la counters, tag pipe, vrf_addr (228-230), `a_flat = mm*k_limit + kk` (218) | | ~150 cells | 120 |

Memory-like state of own logic: `abuf` (179) 8 x 512 = 4,096 bit, LUTRAM 296 LUT attributed (census abuf_reg_RAMSLICE; 8-deep x 512 needs >= 256 in RAM32M16, the other 40 are not explained), one 512-bit write port (S_LOADA, 522-526), async read by `a_bi` (3 bits), consumer = `a_word` mux then the element select in the same cycle into the MAC input register. `b_row` 16 x 32 = 512 FF (244), `r_vdata` 512 FF (167), `direct_data` 512 FF (371; a one-cycle copy of `r_vdata`, which is stable then, line 444-451 comment) are plain registers. Tag pipe `tg_*` 5 x 25 bit = 24 SRL + FF. Hard-block fit of `abuf`: 512-bit write per cycle forces 8 RAMB36 (72-bit SDP) or 16 RAMB18 for 4 Kbit, saving at most the 296 LUTRAM LUT and leaving the 512-bit mux logic unchanged; not a gain.

### 3a. Accumulator file `u_acc` (tnca_acc_rf.v) -- 6,293 LUT = 2,560 LUTRAM + 3,733 logic, 4,096 FF

| Item | Value |
|---|---|
| Data array `lane_mem` (98-108) | 16 lanes x 256 x 32 = 131,072 bit; LUTRAM: 16 lanes x 4 depth quarters x 5 bit-slices (7,7,7,7,4) x 8 LUT (RAM64M8) = 2,560 LUT (attributed, census). One write port (per-lane enable `w_go && w_lane_en[g]`), one async read port, one address `ra` for all lanes |
| Valid array `valid` (71-82) | 256 x 16 = 4,096 FDRE (attributed), reset loop, `z_en` row clear, set `valid[wa] | w_lane_en` |
| Read `r_data` = `(z_en && ra==za) || !rv[g] ? 0 : lane_mem[ra]` (105-106) | cone r_data 2,125 cells (1,930 LUT, 802 MUXF): 512 x (4:1 depth mux over RAM outputs + gating) |
| Valid read mux 256:1 x 16 | 68 LUT6 + 34 F7 + 17 F8 per column = 1,088 cells; cone valid has 528 F7 + 256 F8, 1,926 cells (1,750 LUT): read mux ~1,000 + set/clear/decode ~760 (estimate from MUXF counts; read vs write side not separable by name) |

Timing as built: row m is read in cycle t (`acc_r_row = mm`, S_MAC and !vrf_wait; S_ROW uses `r_row`); the data goes the same cycle into the MAC input register (`fc = acc_rd`, k_q holds it); `tg_v[0]` is set at the end of t, `tg_v[PIPE-1]` high at t+5, so `acc_w_en` and the write land at the end of t+5 (PIPE = 5). The next read of row m is at t+17 (16 rows + one S_LOADB cycle), 12 cycles after the write; row m+5 is read in the cycle row m is written. `acc_collide` (404-417, `ifndef SYNTHESIS`) fires on equal {slot,row} between a used read (`st==S_MAC && !vrf_wait`, or S_ROW with rowop != 0) and `acc_w_en`. A row op or a.vread starts only after the previous command's `done` (drain = PIPE+1; a.vread is a FU_MATRIX instruction that stalls while `mtx_busy`, core.v:2493-2494). z and normal writes never coincide (S_ZERO runs with an empty write pipe).

Paths reading it: (1) MAC sweep (above). (2) row ops a.mul.row/a.fma.row: S_ROW cycle after accept, `fa/fc` from `acc_rd`, FMA result written PIPE later; a.vwrite reads nothing (`direct_w`). (3) a.vread: `ard_slot = x_ra_idx`, `ard_row = sb_data` (X-stage register or forwarded `w_sdata`, core.v:743, 1665-1666), data `acc_rd_data` used in X by the writeback AND-OR (core.v:2263, 2342).

Synchronous (BRAM/URAM) read, what changes:
- Sweep: address one cycle early. Row 0 is requested in S_LOADB (which always precedes S_MAC, so no extra cycle), rows 1..15 as `mm+1` while advancing. A stall (`vrf_wait`, only when `!use_abuf`, i.e. FP16 K32/K64, FP8 K64) is handled by dropping the read enable so DOUT holds (or re-reading the same row; nothing writes it). The address path gains `vrf_valid` (a flop when MTX_VRF_REG=1) into the address/enable mux. Read-during-write is unreachable (rows differ, `acc_collide`); the file's "read old" contract equals READ_FIRST anyway.
- 2-cycle read (DO_REG / URAM OREG): addresses two ahead; S_LOADB is one cycle, so +1 cycle per k step unless the last two rows' requests are issued during rows 14,15 (row 0 and 1 of the next step); stalls then hold two stages.
- Row ops: address `{cmd_slot, cmd_row}` is available only in the S_IDLE accept cycle (from `x_dsti`, `sa_data` = forward mux): either feed BRAM from that mux (long path) or add a state, +1 cycle per row op (a.mul.row/a.fma.row, ~10 cycles each today).
- a.vread: needs the row one stage earlier (R has `r_sb_rd`; the forwarded value is `nw_sdata`, a writeback-stage signal) or one extra result cycle through the late-writeback path (`ws_vl`-style scheduling). I did not trace the scoreboard and writeback scheduling; this is the largest RTL edit.
- Valid bits: the gate must use a registered copy of `ra` (valid can stay FF with the read mux on the pipelined address).

Hard-block fit: 8 x RAMB36E2 SDP 72-bit x 512 deep (256 used, 50 %) [or 16 RAMB18E2 36-bit], or 8 x URAM288 (72 x 4096, 6 % used). Width, not depth, sets the count (512 bits per cycle). Lane enables: a lane is 32 bit = 4 bytes, so each RAMB36 holds 2 lanes and `WEBWE[7:0]` = {4 x en[2k+1], 4 x en[2k]} (one bit per 9-bit byte lane; the 8 parity bits per RAMB36, 64 in total, are free); URAM BWE[8:0] likewise, 9th byte spare.

Valid bits, cost and replacement (estimates): 4,096 FF + ~1,750 LUT (read mux ~1,000, set/clear/decode ~760) + part of the 512 gating cells in r_data. Options: (a) physically zero: `z_en` rows already stream one row/cycle in S_ZERO with the write port idle, so the row-zero becomes a zero write (the data mux `direct_w ? direct_data : mac_bus` takes a zero term at no extra LUT); reset: BRAM/URAM INIT = 0 covers configuration, a later `rst_n` needs a 256-cycle sweep (8-bit counter, ~30 LUT, `busy` held). Whether the host asserts `rst_n` between kernels and relies on zeroed accumulators is not visible in the RTL I read; not checked in tb. With (a) the valid array, its read mux and the gating go: acc_rf 6,293 -> ~50. (b) keep set-only valid bits in the 8 spare parity bits per row (bit g written by lane g's byte-0 enable with data 1; no read-modify-write): read-mux and FFs go, the 512 AND gates stay: ~6,293 -> ~550. (c) BRAM data, FF valid kept: saves 2,560 LUTRAM + the 4:1 depth mux (~1,400) = ~4,000. These are structure estimates; none was built.

## 4. MAC lane `tnca_mac_lane` (x16, 3,203 LUT, 610 FF + 33 SRL, 8 DSP each)

Modes (header, tnca_mac_lane.v:1-18, xls/tnca_mac.x:1-48): mode 0 FP32: C <- RNE(a*b + C) on operands widened by `fmt` (bit-identical to the binary32 FMA); mode 1 DP2: two products from 16-bit slots (FP16/BF16/E5M2/E4M3 per operand), summed exactly, round-to-odd at 48 bit (RO_48), then one FMA rounding with C; mode 2 DP4: four E4M3xE4M3 products as 20x17-bit integer multiplies summed exactly, RO_48 is the identity. All modes hand a 48-bit-significand "product" to one shared align/normalise/round back end. Registers: IN_REG 105 + cut0 187+37 + cut2 64+37 + cut3 150 + cut4 122 bits; the 37-bit sideband (incl. C) becomes 33 SRL.

| Stage (module) | LUT | DSP | What it does | Multiplier / adder / shifter / LZC / round |
|---|---:|---:|---|---|
| `u_wa`,`u_wb` widen (tnca_widen_xls) | 124 x 2 | 0 | mode 0 operand widening of 5 formats to binary32 (subnormal normalise) | LZC + exponent adds |
| `u_s1` (mac_s1) | 707 | 8 | decode (4 x 16-bit-slot decoders, 8 x `e4m3_ms`); DP4 operand shifts (`ma<<sa2`, `mb<<sb2`, 4 each); products; DP2 reference choice (12-bit scale adds, compare, 2 abs diffs) | DSP: 4 x 21x18 signed (DP4), 2 x 11x11 (DP2), one 24x24 unsigned (FP32) on 2 DSPs; LUT adds: 39-bit, ~15 small |
| `u_s2` (mac_s2) | 217 | 0 | DP2 smaller product shifted by d <= 64 with sticky, one 67-bit add (carry-in folds the borrow) | 64-bit shifter ~100, 67-bit adder, 25-bit negate |
| `u_f2` (mac_f2) | 449 | 0 | mode's product: |sum| (negate 67/48 bit), RO_48 (18-bit LZC, 66-bit shift by r <= 18, lost-bit OR), exponent, 3-way select | shifter ~250 (estimate), negates, small LZC |
| `u_b0` (mac_b0 = fma32_align_p) | 460 | 0 | unpack C, 12-bit exponent difference (two subtracts), swap big/small, right-align small field by <= 80 with sticky | align shifter ~260 (estimate), 12-bit adders |
| `u_b1` (fmad_norm) | 535 | 0 | 4 x 89-bit add/sub (sum, dif, dif-1, rev dif) + 3:1 select, 89-bit LZC (tnca_lzc, 82 LUT per the xls header), exponent/shift amounts | 4 x 89 = 356 carry-chain LUT, LZC 82 |
| `u_b2` (fma_round) | 490 | 0 | normalising shift of the 90-bit magnitude (both directions, with guard/sticky mask), +1 on 24 bit, overflow test, assemble (combinational into the accumulator write) | shifter ~350-460 (REPORT.md: "stage-3 shifter 463 hierarchy kept, 351 flattened"), 24-bit incrementer |
| lane own | 97 (33 SRL) | | sideband, mode/fmt muxes | |

DSP evidence: the three registered adders on DSP in the lane changed LUT by -2 for +3 DSP (REPORT.md "Failed or not done"); the 4 norm adders are combinational (stay in carry chains by the dsp_infer rules); multiplier-based shifters on DSP were "not built" by the area worker.
Not determined: the split of s1/s2/f2/b0 into adder vs shifter vs mux is structure-based (generated netlists were not synthesised per piece here).
