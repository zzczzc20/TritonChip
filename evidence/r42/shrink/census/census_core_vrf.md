# Census: banked VRF + core-side operand path, core own logic, small core units (tile 0, R42 base image)

Conventions. "attributed" = read from the instance table (tile0_rel.txt) or from census-csv cell-name stems (cells, not LUTs: the csv double counts packed LUT6_2 pairs; I scale cells to LUT with the instance totals: vrf top 13,766/16,872 = 0.816, u_core own 8,756/10,681 = 0.82). "estimate" = my derivation, basis given. Parameters resolved from image_base.tcl and the core instantiation (tnca_core.v:1161-1168; rsp_ready tied 5'b11111 at :1173): NUM_RCTX 4, RCTX_W 2, PHYS_VREGS 256, PHYS_W 8, VLEN 16, ELEN 32 (row = 512 b), NUM_BANKS 4, READ_LATENCY 1, NUM_PORTS 5, NUM_WRITERS 2, ZERO_INVALID 1, PHYS_IN 5'b00111, RDW_BYPASS 1 (hard-wired, tnca_vrf_banked.v:311), BAR_N 8, EPOCH_W 38.

## 1. Banked VRF (tnca_vrf_banked.v 380 lines, tnca_rf_bank.v 228 lines): 21,270 LUT, 6,291 FF, 32 RAMB36

Requesters (NUM_PORTS 5): p0-p2 operand sources A/B/C (D-stage prefetch or R), p3 matrix, p4 transfer-engine window read (tnca_core.v:1098-1112). Writers (NUM_WRITERS 2): W-stage writeback, transfer engine (core:1136-1146). Mapping bank = phys % 4, row = phys / 4: BANK_DEPTH 256/4 = 64, ROW_W 6.

### 1.1 Memory-like structures

| structure (file:line) | entries x width = bits | today | cost | ports / semantics | same-cycle consumer |
|---|---|---|---|---|---|
| `mem` rf_bank.v:108 | per bank 64 x 512 = 32,768 b; 4 banks = 131,072 b | RAMB36E2, 8 per bank (64 data b each, 72-wide SDP) = 32 RAMB36, 11% of capacity (width-bound) | attributed: 32 RAMB36, 8 LUT2/bank on mem_reg (what they gate: not determined) | 1 write (byte enables from `wmask`, 16 lanes x 4 B), 1 clocked read, no output reg (READ_LATENCY 1) | none: s1_raw is the BRAM output latch |
| `vld` rf_bank.v:111 | per bank 64 x 16 = 1,024 b; total 4,096 FF | FF array, sync reset + CE | attributed: 4,096 FDRE, 4,096 LUT4 (CE = we & wmask[l] & row decode, 1,024/bank), 256 LUT6 (row decode) | per-lane set on masked write, cleared only by rst_n (not by launch) | s1_vld mux then zero-gate |
| `vld` read mux rf_bank.v:162 | 64:1 x 16 lanes per bank | LUT6 tree | attributed: 1,088 LUT6 + 512 MUXF7 + 256 MUXF8 (272 LUT6 per bank = 16 lanes x 17) | async 64:1 into s1_vld | merge LUT4 |
| `s1_byp_data` rf_bank.v:150 | 512 FF/bank = 2,048 FF | FF | attributed | copy of the write data, enabled by s1_free | merge LUT4 |
| bypass/zero merge rf_bank.v:176-184 | 512 bits/bank | LUT4 (byp, vld, byp_data, raw) | attributed: 2,048 LUT4 | out = byp ? byp_data : (vld|byp ? raw : 0) | rdata to crossbar |
| s1_valid/s1_dst/s1_byp_mask/s1_vld | 1 + 3 + 16 + 16 FF/bank | FF | 1,572 FF/bank total = 1,024+512+16+16+3+1 | dst is 8 b in RTL, 3 b kept | |

READ_LATENCY and RDW_BYPASS: LATENCY 1 means `s1_raw <= mem[raddr]` (rf_bank.v:161), absorbed by the BRAM's own output latch, so a request in cycle t answers in t+1 (core asserts this, tnca_core.v:861-864); LATENCY 2 would add an s2 register stage (not used). RDW_BYPASS 1 means a read of the row being written returns the new data: `rdw_hit` (6-bit compare) latches wmask, and `s1_byp_data` (512 FF) replaces the raw array output after the register; it exists because the BRAM returns old data on a same-cycle collision (UG573 collision rules not checked here). It is used by VRF_BYPASS 1 (core:1299-1325). rsp_ready is tied high (core:1173), so the hold (`s1_free`) is a constant in effect (does hierarchy keep it as logic: not determined; at most a few LUT).

### 1.2 LUT split of the 21,270

| part | LUT | basis |
|---|---|---|
| 4 banks (vrf rows `g_bank[i]`) | 7,504 (1,876 each) | attributed |
| per bank: vld CE 1,024 + vld row decode 64 + vld read mux 272 + merge 512 + BRAM glue 8 + rdw/valid ~4 | 1,884 cells | attributed from stems `vld[][]`, `s1_vld[]`, `rdata[]_INST` |
| top own logic: read response crossbar (rsp_data_c, vrf_banked.v:343-363: 4 banks x 512 b -> 5 ports x 512 b = 2,560 output bits) | ~11,740 (14,387 cells, 5.6 cells per output bit) | attributed (stem `rsp_data[]_INST_0`) x 0.816 |
| top own: write steering (bank_wdata_c, :134-165: 2 writers -> 4 banks, 4 x 512 LUT4 = 2,048 plus row/mask/we muxes) | ~1,940 (2,381 cells) | attributed (stem `g_bank[].u_bank_i`) |
| top own: read arbitration, round robin rr_ptr (3 FF), port_busy, bank_row mux, phys adders | ~85 (104 cells) | attributed |

So of 21,270: crossbar 55%, vld emulation 5.4k (25%), merge 2.0k (10%), write steering 1.9k (9%), arbitration 0.4%. The BRAM array itself costs 0 LUT. The crossbar is not memory. Inferred from the cell count: the bank response carries a 3-bit port id and valid, so each output bit is a 4-term AND-OR on undecoded dst bits (5.6 cells per bit); with 20 pre-decoded one-hot selects (registered with the response) it would be 2 LUT6 per bit.

### 1.3 What a BRAM-based change would do (it already is BRAM; changes are in the periphery)

| change (RTL to touch) | saves (estimate) | basis / constraint |
|---|---|---|
| valid bits into the row: 528-bit row (512 + 16 valid) still fits 8 RAMB36 x 72 = 576 b (2 parity bits each, written by the same byte enables with data 1); delete `vld`, `s1_vld`, row decode | ~5.4k LUT, ~4.1k FF, +0 BRAM | needs a zero for never-written rows: spec/r3v2_contract.json:67-72 sets data_array_reset false and says valid metadata "may be scrubbed within a bound". |
| or no valid at all: after rst_n a 64-cycle sweep writes zero rows (wdata is already 0 when both write selects are low; add a 6-bit counter, force we/mask/row, hold core_idle) | same ~5.4k LUT, 4.1k FF, sweep ~50 LUT | whether the host's reset-to-launch gap exceeds 64 cycles: not determined |
| remove same-cycle bypass (VRF_BYPASS 0): drops `s1_byp_data`, `s1_byp_mask`, rdw compare, merge LUT | ~2.0k LUT, 2.1k FF | costs one stall per producer/consumer pair (tnca_core.v:1283-1298 comment); not quantified |
| one-hot response select registered in the bank | ~6.6k LUT (11.7k -> 5.1k) | 2 LUT6 x 2,560 bits; no hard block involved |
| URAM288 | no gain | one bank = 1 read + 1 write per cycle, 512 b = 8 URAM per bank = 32, depth 64/4096 used (1.6%); RTL needs latency exactly 1; URAM minimum latency not verified |

## 1.4 Core-side operand path (tnca_core.v g_vrf_banked, 813-1176)

| name (census) | line | bits | FF | LUT (attributed) | use |
|---|---|---|---|---|---|
| of_d0/of_d1/of_d2 | 844, 898-900 | 3 x 512 | 1,536 | CE only | holds an arrived response while R waits (CE = of_arrive) |
| `of_v0..2` mux then x_va_f/x_vb_f/x_vc_f | 876-878, 930 | 3 x 512 | 1,536 | 1,536 LUT3 cells (~1,260 LUT) | (arrive ? rsp : held) into X; FPU copy, keep=true |
| x_va_q/x_vb_q/x_vc_q | 915 | 3 x 512 | not visible in census | | merged or absorbed (RTL claims VALU DSP input regs); where: not determined |
| g_mtx_vrf_reg.data_q | 952 | 512 | 512 | CE only | matrix response register (MTX_VRF_REG 1) |

Total 3,584 FF = 38.7% of the core's 9,269 FF. `tnca_operand_collector.v` is not instantiated in rtl/ (only tb/, tools/, flow/ reference it; no instance in the tile table); the core does its own collection (of_*, lines 844-905).

## 2. u_core own logic (tnca_core.v, 3,655 lines): 8,756 LUT, 9,269 FF, 7 SRL, 0 DSP, 0 BRAM

### 2.1 FF inventory (attributed from census)

| group | FF |
|---|---|
| 512-bit datapath registers: of_d x3, x_v*_f x3, data_q, w_vdata (core:697), red_vdata (:2177) | 4,608 |
| LSU command landing register `command` (:1786, 1,323 b declared, 1,281 kept; includes two 512-b fields q_index, q_vdata) | 1,281 |
| context state: pc 256, f_pc 256, instret 256, f_detail 96, f_code 16, arg_base 64, clock_ctr 64, pid/npid 192, perf_issued 64 | 1,264 |
| scoreboard sb_s 256, sb_v 256, sb_p 64, sb_acc 16, dsb/rsb/whit 12 | 604 |
| PC pipeline d_pc, f1_pc, fb_pc, r_pc, x_pc (64 each) | 320 |
| rest: d_insn 64, fb_insn 64, x_s*_q 192, w_sdata 64, r_/x_ bundles, fp/mu/vl tags, etc. | 1,192 |

### 2.2 Memory-like arrays (all FF; no BRAM or code store in the core: the instruction memory is behind the tile ports imem_addr/imem_data, core:431)

| array (line) | dims | cost | ports | hard block? |
|---|---|---|---|---|
| pc (306) | 4 x 64 = 256 b | 1,307 cells (~1,070 LUT) incl. 64-b pc+8, x_pc+br_off adders, 4:1 read (imem_addr 67 cells) | 1 async read (f_sel); writes from launch, fetch (pc+8), X redirect, R refetch (3026-3028, 3309, 3346, 3362): 5 sources/entry | none (3 concurrent writes). Steering costs 5.1 cells per FF; a single arbitrated next-pc then per-entry hold is ~2 (estimate ~-0.5k LUT) |
| instret (309) | 4 x 64 | 138 cells | read x_rctx (2209, 3302), clear-all on launch | LUTRAM possible (1 write/cycle) but clear-all needs a sweep: LUT-neutral, -256 FF (estimate) |
| f_pc, f_code, f_detail (310-312) | 4 x 64, 4 x 8 (4 b kept), 4 x 24 | 411 + 45 + 96 cells; 368 FF | read x_rctx (2232-2233); writes at 3237, 3282-3284, 3313-3337 (up to 2/cycle) | LUTRAM with a valid bit per context: ~-0.3k LUT, -0.3k FF (estimate; needs write arbitration) |
| sb_s/sb_v/sb_p/sb_acc (1232-1235) | 256 + 256 + 64 + 16 | sb_s 605, sb_v 711, sb_p 278, sb_acc 62, dsb_v 258, whit/rsb 97 cells, plus the `f1_valid` cone (800 cells with 316 MUXF7 + 156 MUXF8, about 10 more 256:1 lookups) | per bit: set (R, 3201), clear (W, 3190), clear (X reject, 3213), launch clear; reads: 8 x 256:1 on sb_s (1317-1375, 1386), 4 x 256:1 on sb_v, 4 x 64:1 on sb_p, 2 x 16:1 sb_acc, OR-reduce rctx_wr_busy (2457) | none: bit vectors with 3 writers and 16 reads per cycle. ~2.0k-2.3k LUT (24-26% of u_core) |
| fp_*/mu_*/wb_rsv (1994-2002, 1990) | FP_LAT 2, MUL_LAT 3 deep | 7 SRL16E + FF | shift | already SRL |

### 2.3 u_core LUT attribution (cells x 0.82; estimate for packing)

| part | LUT | stems |
|---|---|---|
| writeback AND-OR muxes (wb_vdata 6 sources x 512 = 1,549 cells; wb_sdata 11 sources, 813 cells incl. x_pc+8) | ~1,940 | core:2337-2343, 2325-2336 |
| operand mux of_v (1,536 cells) + command mux (579 cells) | ~1,730 | core:876, 1797 |
| pc/fetch/trap state (pc 1,307, f_pc 411, f_detail 96, f_code 45, instret 138, fb_insn 180, d_pc 64, imem_addr 67) | ~1,900 | |
| scoreboard and hazard lookups | ~1,650 attributed + up to 650 in the f1_valid cone | |
| rest (stall/squash, decode registers, mux to units) | ~890 | |

Arithmetic in own logic: 68 CARRY8 only: six 64-b adders/counters (pc+8, x_pc+br_off, x_pc+8, instret+1, clock_ctr, perf_issued; the last two cost 0 LUT) and about 14 CARRY8 of compares (br_taken, vreg_oob, 8-b phys adders at 2743-2745). No multiplier or shifter; nothing worth DSP (counters already cost no LUT).

## 3. Small units

| unit | LUT / FF / other | memory-like | arithmetic | control |
|---|---|---|---|---|
| u_srf tnca_scalar_rf (core:746) | 3,756 LUT (2,072 LUTRAM, 1,684 logic), 256 FF | see 3.1 | none | |
| u_prf tnca_pred_rf (core:1217) | 998 LUT, 1,008 FF | see 3.2 | none | |
| u_salu tnca_scalar_alu (core:1439) | 1,256 own + clz 134 + popc 78 = 1,468 LUT, 0 FF, 22+8 CARRY8 | none | add, sub, eq, signed and unsigned lt, 3 x 64-b variable shifters (:47-49), 4-way mov shift (:36), clz, popc | 23-group result mux (121 MUXF7) |
| u_mul tnca_mul64 (core:1432) | 445 LUT, 648 FF, 16 DSP, 62 CARRY8 | p0..p2 pipeline regs (not arrays) | four 32x32 unsigned products in 16 DSP (4 each); fabric: mid 65 b, lo_sum, hi_u 3-input 64 b, corr 64 b, hi_s sub 64 b = census cells 444 LUT2 + 63 LUT4 (instance table: 445 LUT) | |
| u_bar tnca_barrier (core:1675) | 1,577 LUT, 929 FF, 36 CARRY8 | see 3.3 | 5 x 38-b >=, 8 x 38-b !=, +1, decrements | wake/completion |
| u_dec (core:521) | 653 LUT | none | none | combinational decode; o_illegal alone is 182 of 650 cells |
| u_pfdec (core:558) | 310 LUT | none | none | same decoder module, only va/vb/vc outputs used |

### 3.1 u_srf (tnca_scalar_rf.v:66, :67)
`mem` DEPTH = 4 x 64 = 256 entries x 64 b = 16,384 b, 1 write, 7 async read ports, one LUTRAM copy per port: 36 RAM64M8 + 4 RAM64X1D = 296 LUT per copy, 7 x 296 = 2,072 (attributed). `valid` 256 FF (sync reset, set on write; the header's reset emulation) with 256 LUT5 CE cells. Per port logic (attributed, ~250 cells): 4:1 block mux 64 LUT6 (est.), merge `idx==0 ? 0 : hit ? w_data : valid ? mem : 0` 65 LUT5, valid 256:1 mux ~71 LUT6 + 34 MUXF7 + 17 MUXF8, address compare. Consumers: ports A-C read in R, registered into x_s*_q (core:726-745, x_sa_q at 735), then forwarded with x_s*_fw; ports D-G (x_sd_q..x_sg_q) are consumed in X: D's low 16 bits feed token_bar_valid -> x_bar_illegal -> x_redirect (core:1584), D-G also feed the transfer engine and matrix commands (2143-2144).
Hard block: 1 RAMB36 per copy in SDP 64 b x 512 (50% used), 7 RAMB36 total (or 7 URAM288, 6% depth used); latency 1 (address one stage earlier). A-C: address from the decoder in D, with `stall_front ? r_*_idx : dec_*_idx` so the held instruction re-reads, or gate RDEN with stall_front (about 1,000 loads); keep both write-first compares (W at t, x_s*_fw at t+1). D-G: address r_sb2_q..r_sb5_q (already D->R registers, :1360), data in X; X never stalls (bubble), so no hold; the output then feeds the redirect path directly. Valid bit moves to the row's parity bit (64 data + 1 of 8 spare) or a 256-cycle sweep. Saves (estimate): LUTRAM 2,072 + data mux 448 + valid read ~500 + valid decode 293 = ~3.3k LUT, 256 FF, for +7 RAMB36; ~440 LUT of merge stay.

### 3.2 u_prf (tnca_pred_rf.v:24)
`mem` 4 x 16 = 64 entries x 16 b = 1,024 b (1,008 FF kept = 63 x 16), sync reset loop (so FF), 1 write, 3 async reads (ra, rb, rm; all read in R, registered x_pa_q/x_pb_q/x_lm_q, core:1196-1215), P0 reads all ones. Cost: 3 x ~435 cells (289 LUT6 + 128 MUXF7 each: 64:1 x 16 b), write decode 85 cells. Hard block: not worth RAMB18 (1 Kib x 3 copies); same pattern as u_srf: 3 copies x (2 RAM64M8 + 2 RAM64X1D = 20 LUTRAM) = 60 LUTRAM, 64 valid FF, merge 3 x 16 LUT, valid mux ~70: ~290 LUT vs 998, -944 FF (estimate, basis: u_srf's own header records 8,192 FF + ~25k LUT before this change).

### 3.3 u_bar (tnca_barrier.v:43-51)
cur_epoch 8 x 38 = 304 FF, done_epoch 304, slp_epoch 4 x 38 = 152, n_arrive 8 x 8 = 64, n_ops 64, err/armed/slp_v/slp_bar/wake/fault 41 = 929. All 8 entries update in parallel (completion test, OP_DEC with 3 sources, :58-75, :150-161), so not RAM-shaped. Epoch-dependent logic (attributed cells): wake 640, wake_reg 184, cur_epoch 351, done_epoch 331, test_done 155, arm_token 120 = 1,781 of 2,143 (83%): 9 x (8:1 x 38 b) muxes, 5 x 38-b `>=`, 8 x 38-b `!=`. Per bit of BAR_EPOCH_W: 20 FF and ~34 LUT (estimate, 0.83 x 1,577 / 38). DSP: a DSP48E2 pattern-detect/subtract does one 48-b compare, so 13 DSP could replace ~250-300 LUT of comparators; the 8:1 muxes stay in fabric.

## 4. Arithmetic: DSP evidence

- u_mul: essentially all of its LUT is the five fabric adders after the DSP products (444 LUT2 cells + 62 CARRY8 remain in the census, so Vivado did not cascade them). A DSP-cascade tiling (PCOUT to PCIN, 17-b shift) would absorb them: ~-400 LUT, ~-190 FF (p1_* registers, estimate).
- u_salu: add/sub ~128 LUT; shifters ~3 x 192 LUT6 (64 b x 6 levels, 2 per LUT6, estimate) + mov shifter 64; DSP cannot do 64-b variable shifts. The multiplier already computes the shifts: SHL = lo(a x 2^n), LSHR = hi_u(a x 2^(64-n)), ASHR = hi_s(a x 2^(64-n)), n=0 special: ~-576 LUT but 3-cycle latency and a 2^n decoder (estimate).
- u_dec/u_pfdec: d_pf_v*_idx/en registers (core:3050-3052) equal dec_v*_idx/en of the same word (same enable, !stall_front), so u_dec's va/vb/vc outputs (~94 cells) are a duplicate decode (estimate ~-80 LUT).

## 5. Not determined
x_va_q location; what the 8 LUT2/bank on mem_reg gate; split of the 800-cell f1_valid cone; whether rsp_ready=1 folds the hold logic across the kept hierarchy; reset-to-launch gap for a sweep; URAM288 minimum latency and RAMB36 collision data (UG573 not in the repo); whether the 64-b PCs (1,088 FF in pc, f_pc, d/f1/fb/r/x_pc) need all upper bits.
