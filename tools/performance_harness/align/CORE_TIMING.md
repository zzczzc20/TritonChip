<!-- Status: DRAFT, written 2026-10-27 by reading the Verilog at 3e1a282 (an agent's pass,
     reviewed for structure only). It is the working reference for the cycle-level core
     rewrite; the alignment gate (README.md section 4) is the authority, and section 7
     lists what this document could not settle by reading. Correct it from traces, not
     from memory. -->

# TNCA core cycle-timing specification (draft, from the Verilog)

Source of truth: `rtl/core/tnca_core.v` and the units it instantiates, worktree
`TritonChip-model-align`, branch `model-align`, commit `3e1a282`. Line numbers
below are from that commit. Where `tools/performance_harness/systemc/README.md`
(section "Board timing structure ... (M2)") or
`configs/r40-board-m2.sources.json` say something different, section 8 says so.

## 0. Conventions and the v5 parameter set

Cycle `t` is the interval ending at clock edge `E_t`. A register assigned with
`<=` at `E_t` holds the new value during `t+1`. "In X at cycle x" means
`x_valid=1` during `x`; the align trace's `I` record is written in that cycle.
"Issues" means leaves R at `E_t` (`!stall_front && r_valid && !kill_r`).

Profile `core-vrf-sync-v5` (`spec/r1_profiles.json`, "profiles/core-vrf-sync-v5") and the
3-core U280 image (`evidence/r40/v2/convergence/attempt03/image.tcl`,
`TNCA_IMAGE_PARAMS`) elaborate the core with:

| parameter | value | where |
|---|---|---|
| NUM_RCTX / RCTX_W | 4 / 2 | profile elaboration, image |
| VREGS_PER_RCTX / PHYS_VREGS / PHYS_W | 64 / 256 / 8 | image; `rtl/board/tnca_multicore_hbm_xo.v:19` |
| USE_BANKED_VRF | 1 (4 banks, READ_LATENCY 1, 5 read ports, 2 writers) | `tnca_core.v:977-991` |
| VRF_BYPASS, FE_NOBUBBLE, VRF_PREFETCH | 1, 1, 1 | image |
| RCTX_SWITCH | 2 | image |
| LSU_INFLIGHT | 2 (`tnca_lsu_pair`) | image; `tnca_core.v:1448` |
| FPU_OVERLAP, MTX_ABUF | 1, 1 | image |
| RSLICE, RPAIR, HAS_SFU, HAS_ATOMICS, HAS_TENSOR_XPOSE | 1, 1, 1, 1, 1 | image |
| FP_LAT | 2 (default; no override in tile/top/board) | `tnca_core.v:142` |
| MUL_LAT | 3 (default) | `tnca_core.v:144`; `tnca_mul64.v` has three register stages |
| SFU_LAT | 6 (default) | `tnca_core.v:143` |
| WB_DEPTH | 8 (localparam) | `tnca_core.v:1557` |
| LSU_BEATS / CTAG_W / LSU_TAG_W | 8 / 4 / 5 | `tnca_core.v:148-150`; per LSU unit tag is 3 bits (`tnca_lsu_pair.v:96`) |
| VLEN / ELEN / LW | 16 / 32 / 4 | defaults |
| ACC_N, BAR_N, DESC_N, WIN_N | 16, 32, 16, 32 | defaults |
| matrix PIPE = FMA_LAT + OPND_REG | 2 + 1 = 3 | `tnca_matrix.v:278`, `tnca_core.v:1339` passes `FMA_LAT(FP_LAT)` |
| tensor NTAG / NSLOT | 16 / 8 | `tnca_core.v:1639`, `tnca_tensor.v:454` |

The instruction memory on the board is a one-cycle registered read
(`instruction <= code[ia>>3]`, `rtl/tnca_multicore_shell.v:130-134`), which is
exactly the core's assumption (`tnca_core.v:414-426`). `CODE_LINK_STAGES` only
delays host code writes, not fetches.

`tnca_operand_collector.v` is NOT instantiated by the core (no reference to it
anywhere else under `rtl/`); the core's own `of_*` logic (`tnca_core.v:746-811`)
plus `tnca_vrf_banked`'s arbiter do the job.

## 1. State that survives a clock edge

### 1.1 Per Role Context (index i)

| register | width | meaning | line |
|---|---|---|---|
| `pc[i]` | 64 | next fetch address | 286 |
| `running[i]` | 1 | started by the launch and not exited/faulted | 287 |
| `faulted[i]` | 1 | trapped | 288 |
| `blocked[i]` | 1 | asleep on an mbarrier (`mb.wait`) | 379 |
| `parked[i]` | 1 | gave way in R (RCTX_SWITCH), waiting for a wake | 385 |
| `launched[i]` | 1 | in the launch mask (for `NUM_ROLES`) | 389 |
| `instret[i]` | 64 | retired X slots (every `x_valid`) | 289 |
| `rr_ptr` | RCTX_W | fetch round-robin pointer (shared) | 370 |
| `vbase_q`, `slice_q`, `slice_k`, `pair_q` | | launch-time resource mapping | 294, 479-481 |

`fetchable = running & ~faulted & ~blocked & ~parked` (line 397).

### 1.2 Front-end bundles

| bundle | fields that matter for timing | line |
|---|---|---|
| F1 (`f1_valid, f1_rctx, f1_pc`) | a fetch is in flight; its word is `imem_data` next cycle | 427-429 |
| FB (`fb_valid, fb_rctx, fb_pc, fb_insn`) | word that arrived while D was stalled | 431-434 |
| D (`d_valid, d_rctx, d_pc, d_insn`, `d_pf_v{a,b,c}_{en,idx}`) | undecoded word plus pre-decoded vector source fields for the prefetch | 303-306, 536-537 |
| R, X (`TNCA_BUNDLE`) | `valid, rctx, pc, major, func, mod, fu, dstc, dsti, mask_dst, s{a,b,c}_{en,idx}, v{a,b,c}_{en,idx}, ra_idx, p{a,b,m}_{en,idx}, pm_neg, imm, is_branch, br_cond, br_on_zero, br_ind, br_link, br_off, is_exit, is_trap, mem_en, mem_store, mem_vector, mem_indexed, mem_size, amo, is_fence, illegal` | 309-365 |

For the timing model the X bundle additionally needs `x_fpctl` (1231) only for
correctness of FP data, not timing.

### 1.3 Operand fetch (banked VRF, lives beside R)

| register | meaning | line |
|---|---|---|
| `of_got[2:0]` | sources already held in `of_d0..2` | 749-750 |
| `pf_tag[2:0]` | ports of a D-prefetch accepted last cycle | 755 |
| `adv_q` | D advanced into R at the last edge | 756 |
| `x_va_q, x_vb_q, x_vc_q` | operands travelling with the X bundle | 785 |
| `u_vrf.rr_ptr` (3 bits, 5 ports) | read-port round robin | `tnca_vrf_banked.v:222` |
| per bank `s1_valid, s1_dst, s1_raw ...` | one read response in flight per bank | `tnca_rf_bank.v:147-170` |

### 1.4 Scoreboards and writeback

| register | meaning | line |
|---|---|---|
| `sb_s[NUM_RCTX*64]`, `sb_v[NUM_RCTX*64]`, `sb_p[NUM_RCTX*16]` | one busy bit per architectural register per context (S0/P0 never) | 1011-1013 |
| `sb_acc[16]` | accumulator slot has a command outstanding (core-wide) | 1014 |
| `wb_rsv[7:0]` | scheduled-writeback reservation shift register | 1559 |
| `fp_v/rctx/dstc/dsti/len[0..FP_LAT-1]` | FP tag pipeline | 1562-1566 |
| `mu_v/rctx/dsti/func[0..MUL_LAT-1]` | MUL tag pipeline | 1568-1571 |
| `w_en, w_rctx, w_class, w_idx, w_sdata, w_vdata, w_lane_en, w_pdata` | the WB stage (drives the RF write ports) | 648-655 |
| `red_pend, red_rctx, red_dstc, red_dsti, red_sdata, red_vdata, red_len` | one deferred multicycle-FP completion | 1675-1683 |
| `mtx_rctx`, `txf_rctx` | issuer of the running engine command (RPAIR arrive precision) | 589, 1913 |
| `mtx_busy_q`, `txf_busy_q` | last cycle's engine busy (for `eng_freed`) | 1996 |

### 1.5 Units

| unit | state | line |
|---|---|---|
| each `tnca_lsu` (x2) | `state` (IDLE/SETUP/RUN/DRAIN/DONE), latched request, `lane_addr[16]`, `pending`, `partial`, `active`, `bt_valid[8]`, `bt_lanes/low/high[8]`, `cpl_valid` + payload | `tnca_lsu.v:121-153` |
| `tnca_lsu_pair` | `old` (which unit holds the older access) | `tnca_lsu_pair.v:124` |
| `tnca_vector_fpu` | `red_busy, red_done, lvl, rwait, acc[16]`; `tnca_sfu.busy, done, cnt, src[16], res[16]`; `dly_*[FP_LAT]` delay line; `cvt_*` | `tnca_vector_fpu.v:129-133, 446, 543-546`; `tnca_sfu.v` |
| `tnca_mul64` | 3 pipeline stages, always advancing | `tnca_mul64.v` |
| `tnca_matrix` | `st`, latched command, `kk, mm, drain, la, use_abuf, abuf[8], b_row`, `tg_*[PIPE]`, `done/done_slot/done_bar*`, `src_free*` | `tnca_matrix.v:143-168, 279-283` |
| `tnca_tensor` | `st`, descriptor/window tables, latched command and normalised geometry, odometer `t_i0..3, t_P0..3, t_unit, t_flat, t_half`, load slots `sl_*[8]`, store tags `tg_v[16]`, `sr_*`, `st_prev`, `vw_*` (held write), `xbuf`, `done*`, `fault*` | `tnca_tensor.v:186-300, 457-498` |
| `tnca_barrier` | per barrier `cur_epoch, done_epoch, n_arrive, n_ops, err, armed`; per context `slp_v, slp_bar, slp_epoch`; `wake, wake_err` (one-cycle pulses) | `tnca_barrier.v:117-127` |
| `tnca_acc_rf` | accumulator rows (combinational read, `tnca_acc_rf.v:105`) | |

## 2. Next-state rules, stage by stage

Every rule below is "during cycle t these wires are computed from the pre-edge
registers; at `E_t` these registers change".

### 2.1 Combinational stall/kill summary (used by every stage)

| wire | expression | line |
|---|---|---|
| `hazard` | `r_valid && (hz_sa|hz_sb|hz_sc|hz_va|hz_vb|hz_vc|hz_pa|hz_pb|hz_pm|hz_d) | hz_acc | hz_sb2 | hz_sbx` | 1151 |
| `stall_other` | `hazard | stall_lsu | stall_fpu | stall_mtx | stall_txf | stall_fence | stall_arrive | stall_rsv | wb_steal` | 1983 |
| `stall_raw` | `stall_other | vrf_stall | vrf_wb_freeze` (`vrf_wb_freeze` is structurally 0: the WB stage is writer 0 and always accepted, 964-973) | 1987 |
| `rp_go` | `r_valid && r_of_idle && (hazard || stall_mtx || stall_txf) && d_valid && (d_rctx != r_rctx) && !x_redirect && !vrf_wb_freeze` | 2025-2028 |
| `stall_front` | `stall_raw && !rp_go` | 2029 |
| `x_redirect` | `x_valid && ((x_is_branch && br_taken) || x_is_exit || x_is_trap || x_illegal || x_access_fault || bar_sleep)` (`x_kill` is constant 0, 2089) | 1749-1751 |
| `kill_f` | `(x_redirect && f_hit && f_sel==x_rctx) | (rp_go && f_hit && f_sel==r_rctx)` | 2066 |
| `kill_f1`, `kill_fb` | same two terms on `f1_rctx` / `fb_rctx` | 2071-2074 |
| `kill_av` | `fb_valid ? kill_fb : kill_f1` | 2075 |
| `kill_d` | `x_redirect && d_valid && d_rctx==x_rctx` (NOT on rp_go: D's instruction is another context's) | 2068 |
| `kill_r` | `(x_redirect && r_valid && r_rctx==x_rctx) | rp_go` | 2084 |

`wb_steal` (1790) does not depend on `r_valid`: it freezes the whole front end
even when R is empty. Every other `stall_other` term requires `r_valid`.
`vrf_stall` requires `r_valid` through `of_enable` (746, 806).

### 2.2 F: context selection and fetch issue

Lines 399-411, 2081-2083, 2310-2316.

| condition (cycle t) | effect at E_t |
|---|---|
| `f_sel` = the first context in the order `rr_ptr, rr_ptr+1, ... (mod 4)` with `fetchable=1`; `f_hit` = any. (Loop 402-408 runs i from 3 down to 0 and the last assignment wins, so the lowest i whose candidate is fetchable is chosen.) | |
| `imem_addr = pc[f_sel]` (411) | the word is on `imem_data` during t+1 |
| `f_issue = f_hit && !(stall_front && (fb_valid || f1_valid))` (FE_NOBUBBLE=1, 2082) | `f1_valid <= f_issue && !kill_f`; if `f_issue`: `f1_rctx <= f_sel; f1_pc <= pc[f_sel]; rr_ptr <= f_sel+1`; `pc[f_sel] <= pc[f_sel]+8` unless `kill_f` |
| `rr_ptr` advances on every `f_issue`, killed or not (2314) | |

So: unstalled, one fetch issues every cycle, round robin over the fetchable
contexts; a context whose fetch is killed by its own redirect in the same
cycle loses that slot and `rr_ptr` still moves past it.

### 2.3 FB: the one-deep fetch buffer

Lines 2318-2327.

| condition (cycle t) | effect at E_t |
|---|---|
| `!stall_front` | `fb_valid <= 0` (D took whatever was available) |
| `stall_front && f1_valid && !fb_valid` | `fb_valid <= !kill_f1; fb_{rctx,pc} <= f1_*; fb_insn <= imem_data` |
| `stall_front && fb_valid && kill_fb` | `fb_valid <= 0` |

`avail = fb_valid | f1_valid`; `avail_insn = fb_valid ? fb_insn : imem_data`
(436-439). `f1_valid && fb_valid` never both hold (assert 2718).

### 2.4 F -> D

Lines 2332-2350.

| condition | effect at E_t |
|---|---|
| `!stall_front` | `d_valid <= avail && !kill_av; d_{rctx,pc,insn} <= avail_*; d_pf_v*_{en,idx} <= u_pfdec(avail_insn)` |
| `stall_front && kill_d` | `d_valid <= 0` |
| `stall_front && !kill_d` | D holds |

Decode (`u_dec`, 501-528) is combinational on `d_insn` during the D cycle;
`d_illegal` (571) folds in `vreg_oob`, `sfu_unsupported`,
`wrspecial_unsupported`, `atomic_unsupported`.

### 2.5 D -> R

Lines 2355-2401.

| condition | effect at E_t |
|---|---|
| `!stall_front` | `r_valid <= d_valid && !kill_d`; all `r_*` fields from the decoder; `r_dstc <= d_illegal ? DST_NONE : dec_dstc`; every `*_en` and `is_branch/br_link/is_exit/is_trap/mem_en` ANDed with `!d_illegal`; `r_dsti <= dec_dsti_sl`, `r_ra_idx <= dec_ra_sl` (RSLICE remap, 488-499) |
| `stall_front && kill_r` | `r_valid <= 0` |
| else | R holds |

### 2.6 D: the vector prefetch (VRF_PREFETCH=1)

Lines 927-947, 1076-1082.

| wire | expression |
|---|---|
| `pf_need` | `{d_pf_vc_en, d_pf_vb_en, d_pf_va_en}` |
| `PF_HAZ(idx)` | `(sb_v[d_rctx][idx] && !(WB_HIT_V(d_rctx,idx))) || (r_valid && r_dstc==DST_V && r_rctx==d_rctx && r_dsti==idx)` |
| `pf_go` | `d_valid && |pf_need && of_want==0 && !(r_valid && r_vsrc_haz) && !pf_haz_w` |
| request ports 0..2 | `pf_go ? pf_need @ d_rctx/d_pf_*_idx : of_want @ r_rctx/r_v*_idx` (930-937) |
| at E_t | `pf_tag <= pf_go ? (vrf_req_accept[2:0] & pf_need) : 0`; `adv_q <= !stall_front && d_valid && !kill_d` (944-945) |

D re-offers the same prefetch every cycle it stays in D; a response whose
instruction did not advance (`adv_q=0`) is dropped by `of_arrive` (757).

### 2.7 R: hazard check

Lines 1021-1152. All macros read pre-edge `sb_*`, `w_*`, `x_*`.

| macro | expression |
|---|---|
| `SB_S_BUSY(rc,idx)` | `idx!=0 && sb_s[rc*64+idx]` |
| `SB_V_BUSY(rc,idx)` | `sb_v[rc*64+idx]` |
| `SB_P_BUSY(rc,idx)` | `idx!=0 && sb_p[rc*16+idx]` |
| `WB_HIT_{S,V,P}(rc,idx)` | `w_en && w_class==class && w_rctx==rc && w_idx==idx` (the WB stage writes it THIS cycle) |
| `X_PROD_{S,V,P}(rc,idx)` | `x_has_dest && x_dstc==class && x_rctx==rc && x_dsti==idx` (a single-cycle producer is in X now) |
| `AVAIL_S`, `AVAIL_P` | `WB_HIT || X_PROD` |
| `AVAIL_V` (banked) | `VRF_BYPASS && WB_HIT_V` only (1067-1069): X_PROD does not count for vectors |

| hazard wire | condition (instruction in R) | line |
|---|---|---|
| `hz_sa/sb/sc` | `r_s*_en && SB_S_BUSY && !AVAIL_S` | 1084-1086 |
| `hz_va/vb/vc` | `r_v*_en && SB_V_BUSY && !AVAIL_V` | 1087-1089 |
| `hz_pa/pb/pm` | `r_p*_en && SB_P_BUSY && !AVAIL_P` | 1090-1092 |
| `hz_sb2` | TXFER major (t.ld/t.st/t.prefetch): `Sb+1`; MMA with `mod[4]`: `Sc+1`; `SB_S_BUSY && !AVAIL_S` | 1098-1103 |
| `hz_sbx` | TXFER major: `Sb+2, Sb+3, Sb+4` busy and not available (checked for every rank) | 1108-1115 |
| `hz_d` (WAW) | destination S/V/P busy and `!WB_HIT` (X_PROD does not count) | 1126-1131 |
| `hz_acc` | `r_dstc==DST_A && sb_acc[r_dsti] && !(mtx_done && mtx_done_slot==r_dsti)`; or `a.vread` whose `ra` slot is busy likewise | 1137-1147 |
| `r_vsrc_haz` | `hz_va|hz_vb|hz_vc` | 1149 |

Source enables per instruction class (decoder `rtl/core/tnca_decoder.v`):
`m.mma` sets `o_va_idx/o_vb_idx` but NOT `o_va_en/o_vb_en` (679-693), so the
fragment registers are neither read through the operand ports nor hazard
checked; it checks `Sc` (and `Sc+1` with `mod[4]`). `t.ld/t.st` check `Sb..Sb+4`
and `Sc`, no vector source (810-830). `mb.wait`/`mb.test` read `Sa`; `mb.arm`
writes S (626-647). Loads: `Sa` (+`Vb` for gather); stores: `Sa` and the data
register (`Sb` scalar / `Vb` vector / `Vb` index + `Vc` data for scatter)
(569-625).

### 2.8 R: the other stalls

| wire | condition | line |
|---|---|---|
| `stall_lsu` | `r_mem_en && !((r_amo ? lsu_req_ready : lsu_ready_load) && !x_mem_go)`; i.e. a memory op in X this cycle blocks the next one for one cycle, and the pair must offer `req_ready_load` (plain load/store) or `req_ready` (atomic) | 1863-1864, 1932 |
| `stall_fpu` (FPU_OVERLAP=1) | `(r_is_fpnorm && fold_next) | (r_is_fpred && (red_busy_next || fp_inflight || x_is_fpnorm)) | (r_is_sfu && red_busy_next)` with `fold_next = fpu_red_busy || x_is_fpred`, `red_busy_next = fpu_busy || red_pend || fpu_done || x_is_fpmc`, `fp_inflight = |fp_v` | 1871, 1953-1961 |
| `stall_mtx` | `r_fu==FU_MATRIX && (mtx_busy || mtx_cmd_valid)` (all MATRIX and ACC majors, a.vread included) | 1928-1930 |
| `stall_txf` | `r_fu==FU_TXFER && (txf_busy || txf_is_cfg || txf_is_xfr)` (TDESC and TXFER majors; a one-cycle `w.def` in X still costs the next TXFER-FU instruction one cycle) | 1924-1926 |
| `stall_fence` | `r_is_fence && (!lsu_req_ready || lsu_cpl_valid || rctx_wr_busy[r_rctx])` | 1946-1947 |
| `stall_arrive` | `mb.arrive && (rctx_wr_busy[r_rctx] || !lsu_req_ready || arr_mtx || arr_txf)`; RPAIR=1: `arr_mtx = (mtx_busy && mtx_rctx==r_rctx) || (mtx_cmd_valid && x_rctx==r_rctx)`, `arr_txf` likewise with `txf_busy/txf_rctx` and a cfg/xfr in X | 1904-1922 |
| `rctx_wr_busy[i]` | any of `sb_s[i][1..63]`, `sb_v[i][*]`, `sb_p[i][1..15]` set | 1890-1902 |
| `stall_rsv` | `r_books_wb && wb_rsv[r_wb_delay+1]`; `r_wb_delay = is_mul ? 3 : is_fpnorm ? 2 : 0`; `r_books_wb = r_valid && dstc in {S,V,P} && !(r_mem_en | r_is_fpmc)` | 1965-1974 |
| `wb_steal` | `(lsu_cpl_valid | red_pend) && (x_has_dest | fp_wb | mul_wb)` | 1786-1790 |
| `vrf_stall` | `of_enable && !of_ready` (vector operands not all held or arriving) | 806 |

`lsu_req_ready` = both LSU units idle; `lsu_ready_load` = both idle, or one idle
and the other in DRAIN/DONE with a non-atomic access (`tnca_lsu_pair.v:116-119`).

### 2.9 R: operand fetch from the banked VRF

Lines 746-811 and `tnca_vrf_banked.v:239-268`, `tnca_rf_bank.v:145-170`.

| wire / rule | expression |
|---|---|
| `of_enable` | `r_valid && !r_vsrc_haz` (no read while a vector source has an older writer in flight) |
| `of_need` | `{r_vc_en, r_vb_en, r_va_en} & {3{of_enable}}` |
| `of_arrive` | `vrf_rsp_valid[2:0] & ~(pf_tag & ~adv_q)` (responses this cycle, minus a dropped prefetch) |
| `of_have` | `of_got | of_arrive` |
| `of_want` | `of_need & ~of_have` -> offered to ports 0..2 this cycle (unless `pf_go`, which only happens when `of_want==0`) |
| `of_ready` | `(of_need & ~of_have) == 0` |
| `vrf_stall` | `of_enable && !of_ready` |
| `r_of_idle` | `of_got==0 && of_arrive==0 && of_want==0` |
| at E_t, `!stall_front` | `of_got <= 0`; `x_v{a,b,c}_q <= of_v*` (771-792) |
| at E_t, `stall_front` | `of_got <= of_have`; `of_d* <= arriving data` |

Bank side: port p (0 va, 1 vb, 2 vc, 3 matrix, 4 tensor store read) asks for
physical register `vbase[rctx] + idx`; bank = `phys % 4`, row = `phys / 4`
(`tnca_vrf_banked.v:13-14`). The port is the instruction FIELD, not the
operand's order: `a` reads on va, `b` on vb, `c` on vc (`TNCA_DEC_SRC_V*`,
`tnca_decoder.v:163-165`); a vector store's data (field d) reads on vb (601),
a scatter's on vc with its index on vb (613-614), `v.insert`'s Vd on vb (524),
`a.mul.row` / `a.fma.row`'s vector on vb (`TNCA_DEC_SRC_VB`, 9B). An
instruction whose only vector source is `b` therefore competes as port 1:
with `rr_ptr = 1` it beats the matrix's port 3 on a shared bank, as port 0
would not (seen on one launch of a graph-level point: an `a.fma.row` behind
an mma cost the engine one refused fragment read). Per cycle each bank accepts at most one read;
ports are scanned in order `u_vrf.rr_ptr, +1, ... (mod 5)`, a port is accepted
if its bank is not yet taken this cycle and the bank's response register is
free (`bank_accept`, always true here because `rsp_ready` is tied to 1, 987);
`rr_ptr` then moves past the FIRST port served (221-268). An accepted read
returns on `vrf_rsp_valid[p]` in the NEXT cycle (`s1_valid <= re`, bank 160),
and a read of the row being written at the same edge returns the new data
(`rdw_hit`, bank 145, 166). Writes: writer 0 = WB stage, writer 1 = transfer
engine; one write per bank per cycle, lowest writer index wins (135-158).

Consequences (no contention from ports 3/4 or another context):

| case | cycles the instruction spends in R because of operands |
|---|---|
| vector sources all prefetched by D (sources had no `PF_HAZ` while in D, R wanted nothing, D advanced) | 0 extra: responses arrive in the first R cycle |
| not prefetched, all sources in distinct banks | 1 extra (`vrf_stall` in the first R cycle) |
| not prefetched, k sources share a bank | k extra (one per cycle, serialised) |
| a source whose producer is in W this cycle (`WB_HIT_V`) | the read is issued this cycle, returns next cycle with the new value (bypass) |

### 2.10 R -> X, scoreboard claim, reservation

Lines 2406-2459, 2484-2492, 2179-2182.

| condition (cycle t) | effect at E_t |
|---|---|
| `!stall_front` | `x_valid <= r_valid && !kill_r`; whole bundle copied; `perf_issued++` if `|running && r_valid && !kill_r` |
| `stall_front` | `x_valid <= 0` (X is a bubble next cycle; X never repeats) |
| issue (`!stall_front && r_valid && !kill_r`) | claim: `sb_s[r_rctx][r_dsti]<=1` (idx!=0) / `sb_v<=1` / `sb_p<=1` (idx!=0) / `sb_acc[r_dsti]<=1` by `r_dstc`; claim is written after the release in the same block so claim wins (2461-2492) |
| issue && `r_books_wb` | `wb_rsv <= (wb_rsv>>1) | (1 << r_wb_delay)`; otherwise `wb_rsv <= wb_rsv>>1` |
| `rp_go` | `pc[r_rctx] <= r_pc`; `parked[r_rctx] <= 1` unless `rctx_wake[r_rctx]` this cycle; R's instruction is dropped (`kill_r`); F1/FB/F of that context killed (2634-2637, 2066-2074) |

Invariant checked at 2705: `sched_wb == wb_rsv[0]` every cycle. An instruction
with delay d leaving R at `E_t` lands on the port at cycle `t+1+d`.

### 2.11 X

X never stalls. Everything below is computed during the X cycle x and takes
effect at `E_x` unless stated.

| class | what happens | line |
|---|---|---|
| scalar ALU, SYS, CTRL (incl. `s.movk`), VINT, PRED, SHUF, integer RED, `a.vread`, `mb.arm`, `mb.test`, link | `x_has_dest = x_valid && dstc in {S,V,P} && !x_defers_wb`; result goes to `nw_*` this cycle, `w_*` at `E_x` | 1776-1779, 1834-1843 |
| `s.mul/s.muli/s.mulh.*` | `mu_v[0] <= 1` at `E_x` (`tnca_mul64` pipeline advances every cycle); `mul_wb = mu_v[2]` at `x+3` | 1254-1258, 2197-2206 |
| VFP, CVT (`TNCA_IS_FPNORM`) | `fp_v[0] <= 1` at `E_x`; `fp_wb = fp_v[1]` at `x+2` (FP_LAT=2) | 1259, 2184-2195 |
| FP RED, SFU (`TNCA_IS_FPMC`) | `start` pulse into `u_vfpu`; `red_rctx/dstc/dsti/len <= x_*` at `E_x`; completes later via `fpu_done` -> `red_pend` | 1239, 2524-2531 |
| load/store/atomic | `x_mem_go = x_valid && x_mem_en && !x_access_fault` -> `req_valid` to the LSU pair; accepted in this cycle (R guaranteed a free unit) | 1427, 1457 |
| MATRIX/ACC commands (not `a.vread`) | `mtx_cmd_valid` pulse; `mtx_rctx <= x_rctx` | 1318, 2497 |
| TDESC/TXFER | `txf_is_cfg` / `txf_is_xfr` pulse; `txf_rctx <= x_rctx`; `t.prefetch` is dropped | 1596-1607, 2498 |
| MBAR | `init/arm/arrive/inval_en` pulses; `mb.wait`: `bar_sleep = !bar_test_done` | 1369-1399 |
| branch | `br_taken` (1743); `x_redirect`; `pc[x_rctx] <= br_target` at `E_x` | 2618-2619 |
| `s.exit` | `running[x_rctx] <= 0` | 2616 |
| trap / illegal / misaligned atomic | `running<=0, faulted<=1`, fault outputs | 2583-2615 |
| `mb.wait` that sleeps | `blocked[x_rctx] <= 1; pc[x_rctx] <= x_pc` (re-executes from the same PC on wake) | 2577-2582 |
| every `x_valid` | `instret[x_rctx]++` | 2575 |

`x_redirect` kills F/F1/FB/D/R entries of the same context only (2.1).

### 2.12 Writeback port arbitration and the WB stage

Lines 1786-1844, 2211-2220, 2473-2480.

| priority during cycle t | taker of the port |
|---|---|
| `sched_wb = x_has_dest | fp_wb | mul_wb` (mutually exclusive by `wb_rsv`) | the scheduled result (fp_wb before mul_wb before x_has_dest in the mux, but only one is ever set) |
| else `lsu_cpl_valid` | LSU completion (`lsu_cpl_ack=1`; a store completion sets `nw_en=0` but still needs this free slot to be acknowledged) |
| else `red_pend` | reduction/SFU result (`red_pend <= 0` at `E_t`) |

At `E_t`: `w_* <= nw_*` (always; `vrf_wb_freeze` is 0). During `t+1`: `w_en=1`
drives the RF write ports (scalar/predicate files are write-first:
`tnca_scalar_rf.v:91-104`, `tnca_pred_rf.v:58-63`; the banked VRF writes bank
`phys%4` at `E_{t+1}` and bypasses a read accepted in `t+1`). At `E_{t+1}` the
scoreboard bit of `w_*` is released (2473-2480); during `t+1` `WB_HIT` exempts a
consumer in R.

**R41 F04 (VALU_LATE != 0; model `core_valu_late=1`).** `v.mul`, `v.muli`,
`v.mulh.s`, `v.mulh.u` (major VINT, func 0x03-0x05, 0x22) and the integer
reductions (major RED, func < FADD) write back one cycle after X through
`vl_*` (`tnca_core.v` 1532-1543, 2008-2014): they defer like the scalar
multiply with `r_wb_delay = 1`, so they book the reservation slot one ahead
(`stall_rsv` against an X-result instruction behind them), hold their
destination's scoreboard bit to the WB stage and are not forwarded from X.
On the port `ws_vl` follows `ws_fp` and `ws_mul` and precedes `ws_x`
(2310-2316; one at a time by the reservation). Seen as an X-result
instruction behind a vector multiply issuing a cycle later than R40's rules
gave.

### 2.13 Result latencies (consumer-visible)

Producer in X at cycle x. "Consumer X" is the earliest cycle a dependent
instruction can be in X, assuming it is already in R (or D for the prefetch
cases) and nothing else stalls.

| producer | result on the port (`nw_en`) | `w_en` cycle | consumer in R passes hazard at | consumer X (scalar/pred dest) | consumer X (vector dest) |
|---|---|---|---|---|---|
| scalar ALU, SYS, link, `mb.arm` token, `mb.test`, int RED to scalar, shuffle extract to scalar | x | x+1 | x (`X_PROD_S`) | x+1 | - |
| VINT, PRED, SHUF to V, int RED broadcast to V, `a.vread` | x | x+1 | x+1 (`WB_HIT_V`; X_PROD does not count on the banked file) | - | x+3 (read issued x+1, data x+2, X x+3) |
| predicate-writing single-cycle op (e.g. `v.cmp`) | x | x+1 | x (`X_PROD_P`) | x+1 | - |
| `s.mul*` (MUL_LAT 3) | x+3 | x+4 | x+4 | x+5 | - |
| VFP / CVT to V (FP_LAT 2) | x+2 | x+3 | x+3 | - | x+5 |
| `vf.cmp` to P (FP_LAT 2) | x+2 | x+3 | x+3 | x+4 | - |
| FP reduction (LW=4 levels of FP_LAT+1 cycles) | `red_done` at x+13, `red_pend` from x+14, port at x+14 (if free) | x+15 | x+15 | x+16 | x+17 |
| SFU (VLEN+SFU_LAT=22 issue+land, `done` at x+24) | `red_pend` x+25, port x+25 | x+26 | x+26 | - | x+28 |
| load, last memory response at cycle p | `cpl_valid` at p+3, port p+3 | p+4 | p+4 | p+5 | p+6 |

Derivations: FP reduction `rwait` counts 0..FP_LAT then captures
(`tnca_vector_fpu.v:466-477`): level k captures at x+3(k+1); the fourth capture
(x+12) sets `red_done` (x+13 visible), `red_pend <= 1` at `E_{x+13}`. SFU:
`cnt` = 0 at x+1, `busy<=0, done<=1` when `cnt == VLEN+LAT = 22` (x+23), so
`done` is high at x+24 (`tnca_sfu.v:129` start, `:135-137` the done edge;
`issuing`/`landing` at 76-77).

WAW: `hz_d` uses `WB_HIT` only, so two back-to-back single-cycle writers of
the same register cost the second one 1 cycle in R.

### 2.14 Writeback-slot stalls (`stall_rsv`)

With delays {0, 2, 3}: an instruction in R at cycle t with delay d stalls iff
`wb_rsv[d+1]` is set, i.e. an earlier issue would land at t+1+d. Examples:

| sequence | effect |
|---|---|
| VFP issued at `E_t`, then single-cycle writer in R at t+2 | stalls 1 cycle (`wb_rsv[1]` set at t+2) |
| `s.mul` issued at `E_t`, then single-cycle writer in R at t+3 | stalls 1 cycle |
| `s.mul` issued at `E_t`, VFP in R at t+1 | stalls 1 cycle (`wb_rsv[3]` set at t+1) |
| back-to-back VFPs, back-to-back muls | no stall |

### 2.15 Multicycle FP occupancy

| in R | blocked while | line |
|---|---|---|
| VFP/CVT | an FP reduction is folding (`fpu_red_busy`) or an FP reduction is in X | 1957 |
| FP reduction | `fpu_busy || red_pend || fpu_done || x_is_fpmc` or any `fp_v` set or a VFP/CVT in X | 1958 |
| SFU | `fpu_busy || red_pend || fpu_done || x_is_fpmc` | 1959 |

Hence the next SFU pass can leave R at x+26 and be in X at x+27 (`red_pend`
cleared at `E_{x+25}` when granted at x+25); the next FP reduction likewise
unless FP ops are in flight.

### 2.16 Context switching (RCTX_SWITCH=2)

| event (cycle t) | rule | line |
|---|---|---|
| `rp_go` | see 2.1; requires `r_of_idle`: an instruction that has issued or holds any vector read cannot give way (e.g. one stalled on a scalar hazard after its vector reads went out, or an `a.vwrite`/`a.fma.row` waiting for the matrix unit) | 2025 |
| wake `rctx_wake[i]` | `(w_en && w_rctx==i) || mtx_done || eng_freed`, with `eng_freed = (mtx_busy_q && !mtx_busy) || (txf_busy_q && !txf_busy)` (any writeback to that context, ANY matrix completion, ANY engine going idle) | 1997-2011 |
| at `E_t` | `parked[i] <= 0` for every `rctx_wake[i]`; `parked[r_rctx] <= 1` on `rp_go` unless woken the same cycle | 2632-2637 |
| refetch | context fetchable from t+1 after a wake at t; fetch issues at t+1 at the earliest (subject to `rr_ptr`), F1 t+2, D t+3, R t+4, X t+5 | 2.2 |

Cost comparison for a vector RAW chain V1 (X at x) -> V2: stalled in R: V2 in X
at x+3; parked: wake at x+2 (`w_en`), fetch x+3, X at x+7 (D-prefetch at x+5
succeeds since `sb_v` cleared at `E_{x+2}`).

### 2.17 Branches and faults

| event | timeline |
|---|---|
| taken branch in X at x | F/D/R of that context killed; its fetch at x killed (`kill_f`); `pc <= br_target`; fetch at x+1 (if selected), X of the target at x+5. The context loses the slots x+1..x+4 ("4 fetch slots"). Other contexts unaffected. |
| not taken | nothing |
| `s.exit`, trap, illegal, misaligned atomic | same squash; `running[i] <= 0` |
| `txf_fault` (async) | `running/faulted` of `txf_fault_rctx` at `E` of the fault pulse (2552-2562); nothing is squashed |
| `bar_wake_err[i]` | `running[i]<=0, faulted[i]<=1` (2507-2511) |

### 2.18 Exclusive stall attribution (for counter matching)

Lines 2037-2059, counted when `|running`: priority `hazard > stall_lsu >
stall_fpu > stall_mtx > stall_txf > stall_fence > stall_arrive > stall_rsv >
wb_steal > vrf_stall > vrf_wb_freeze`; `perf_stall_total` counts `stall_raw`
(including cycles that became `rp_go`); `perf_x_frontend` counts
`!stall_raw && !(r_valid && !kill_r)`; `perf_cycles` counts `|running`.

## 3. The memory port (`mem_req_*` / `mem_rsp_*`)

### 3.1 Mux

`tnca_mem_mux.v:69-91`: the transfer engine (client b) has strict priority:
`a_req_ready = !b_req_valid && m_req_ready`. Tag bit 4 = 1 for the engine, 0
for the LSU; the LSU pair's tag is `{unit, beat[2:0]}`
(`tnca_lsu_pair.v:154`). A response is routed by its tag only.

### 3.2 One `tnca_lsu` unit (`tnca_lsu.v`)

Request accepted in X at cycle c (`req_valid && state==S_IDLE`, 383-424).

| cycle | state | what happens | line |
|---|---|---|---|
| c | S_IDLE | lane addresses, write data, `pending/active` latched (`active = vector ? mask : mask[0]`) | 385-423 |
| c+1 | S_SETUP | per-lane word geometry registered | 234-246, 430 |
| c+2 .. | S_RUN | one beat per cycle: `mem_req_valid = plan_valid && have_slot` (`have_slot`: one of 8 beat tags free, 300-312); fires when `mem_req_ready`; lanes of the beat retired from `pending` at issue | 436-456 |
| i+1 (i = last beat issued) | S_RUN, `plan_valid=0` | `state <= S_DRAIN` | 454-455 |
| i+2 .. | S_DRAIN | `o_draining=1`; waits `bt_valid==0 && !mem_rsp_valid` | 158, 459-462 |
| p+1 (p = last response) | S_DRAIN | condition true -> `S_DONE` | 461 |
| p+2 | S_DONE | `cpl_valid <= 1` | 465-470 |
| p+3 .. | S_DONE | `cpl_valid=1` held until `cpl_ack`; on ack `cpl_valid<=0, state<=S_IDLE` | 471-474 |

Response handling runs in every state (352-378): `bt_valid[tag] <= 0` at the
edge of the response cycle, so `no_beats` is true from p+1.

Beats per instruction (coalesced by aligned 8-byte word, 248-295): one beat per
distinct word touched by active lanes; a lane that straddles a word boundary
costs a second beat (the high half).

| access | beats (aligned base, all lanes active) |
|---|---|
| scalar B/H/W/D | 1 (2 if it straddles a word) |
| vector W (32-bit lanes) contiguous | 8 |
| vector H contiguous | 4 |
| vector B contiguous | 2 |
| gather/scatter | number of distinct words among active lanes (up to 16), plus straddles |
| masked-off lanes | no beats |
| atomic | 1 (scalar; naturally aligned or `x_access_fault`) |

### 3.3 The pair (`tnca_lsu_pair.v`, LSU_INFLIGHT=2)

| rule | expression | line |
|---|---|---|
| which unit takes a request | unit 0 if idle, else unit 1 | 120-121 |
| `req_ready` (atomic, fence, arrive) | both idle | 116 |
| `req_ready_load` (plain load/store) | both idle, or one idle and the other in DRAIN/DONE with a non-atomic | 117-119 |
| port ownership | the unit with `mem_req_valid` (never both; assert 200-202) | 147-158 |
| completion order | `pick = u_cplv[old] ? old : ~old`: the older if it is complete, otherwise the younger (a younger load CAN complete first) | 136-145 |

Back-to-back memory instructions (M1 in X at c, last beat issued at i): M2 in R
passes `stall_lsu` at i+2 (unit 0 in DRAIN; `!x_mem_go`), M2 in X at i+3, its
first beat at i+5. The port is idle during i+1..i+4. A scalar load (i=c+2)
thus admits the next memory instruction into X at c+5.

### 3.4 Completion into the pipeline

`lsu_cpl_valid` (= `|u_cplv`) is a request for the writeback port (2.12):

| condition at cycle t (`lsu_cpl_valid=1`) | effect |
|---|---|
| `!sched_wb` | `cpl_grant`: `lsu_cpl_ack=1`, `nw_* <= cpl payload` (`nw_en=0` for a store), `w_en` during t+1, register written at `E_{t+1}`, scoreboard bit released at `E_{t+1}`; the unit returns to S_IDLE at `E_t` |
| `sched_wb` | `wb_steal=1`: front end frozen this cycle (R does not issue, X becomes a bubble at `E_t`), the completion retries next cycle; bounded by WB_DEPTH because no new booking is made while stalled |

Load-to-use from the last response p: scalar consumer in X at p+5, vector
consumer in X at p+6 (2.13), plus any `wb_steal` cycles.

### 3.5 Fence and arrive vs. memory

`fence`: in R until both units idle, no completion pending, no register write of
its context outstanding; in X it does nothing (1946-1947). `mb.arrive`: same
LSU condition plus engine conditions (2.8).

### 3.6 R41: the hybrid LSU (HYBRID_LSU=1)

R41's device (`spec/devices/u280-3core-r41.json`: HYBRID_LSU 1, LINE_SLOTS 64,
TENSOR_LINE 1; profile LSU_INFLIGHT 1) replaces 3.2-3.4 with `tnca_core.v`
`g_lsu_hybrid` (1762-1841) around `tnca_lsu_hybrid.v`. The core port is 512
bits (MEM_SHIFT 6); the model switch is `core_lsu_hybrid=1` with
`core_lsu_line_slots`, `core_tensor_line` and `lsu_slots=1`.

**Command register** (`g_lsu_hybrid` 1786-1802). X does not hand the access to
an engine: `x_mem_go` sets `pending` and latches the command at `E_x`. It
leaves at the edge of the first cycle with `consume = pending && (amo ?
hybrid_idle : hybrid_ready_load)`.

| core wire | hybrid | R40 (3.3) |
|---|---|---|
| `lsu_ready_load` (plain load/store in R) | `!pending` | idle unit, or the other draining |
| `lsu_req_ready` (atomic, fence, arrive, `core_idle`) | `!pending && hybrid_idle` (`hybrid_idle` = line path quiet, word unit idle, no completion held, no atomic pending) | both units idle |
| `stall_fence`, `stall_arrive` | add `x_mem_claim` (an access in X has not reached the register yet, 2484-2517) | -- |

So memory instructions issue at best every third cycle (M1 in X at c, pending
during c+1, M2 passes R at c+2 and is in X at c+3), however busy the paths
behind the register are. `stall_lsu` keeps `!x_mem_claim` (R41 F01: X's
opcode, not `x_mem_go`).

**Which path** (`tnca_lsu_line.v` 52-55, `req_eligible`): the line path takes a
contiguous (not indexed) vector access, not atomic, of B/H/W elements
(`req_size <= 2`), whose `16 << size` bytes from `address[5:0]` fit one 64-byte
line, below 8 GiB (`address[63:33] == 0`), outside the TMU page. Everything
else (scalars, 64-bit lanes, gathers/scatters, atomics, a vector access
crossing a line) takes the word path. One path at a time
(`tnca_lsu_hybrid.v` 82-87):

| path admits when | condition |
|---|---|
| line (`fast_mode && fast_ready`) | word unit idle, the completion register not holding a word completion, a free slot, the request register empty or leaving this cycle |
| word (`slow_mode && slow_ready`) | no line slot valid and the request register empty, the completion register not holding a line completion, the unit idle |

**Line path** (one access = one request; SLOTS = LINE_SLOTS = 64 tags 64..127
on the port, `{1, slot}`). Access consumed at k:

| cycle | what happens | line |
|---|---|---|
| k | `admit`: lowest free slot; `done <= (mask == 0)`; with lanes, `q_valid <= 1`, address = line, `be = packed_mask << offset` (loads too), write data packed and shifted (stores) | 111-128 |
| k+1 .. | `mem_req_valid = q_valid`; leaves when `mem_req_ready`, then `issued[tag]` | 109 |
| p (response) | `done[tag]`, `landing[tag] <= rdata` at `E_p` | 131 |
| p+1 | `cpl_valid` = a held pick, else round robin from `rr` over done slots; `retire` when the completion register takes it (then `rr <= pick+1`); not taken: the pick is held | 82-90, 132-135 |

A store waits for its response as a load does. With no active lane the slot is
done at admission and completes without a request.

**Word path** = one `tnca_lsu` (SLOW_UNITS 1 for LSU_INFLIGHT 1; 3.2) with the
R41 F10 request queue (REQ_REG=1, `tnca_lsu.v` 332-402), behind
`tnca_word_line_adapter.v`. Its 8-byte beats keep their address and travel at
`addr[5:0]` of the line (`be << addr[5:0]`, data likewise); the response is the
whole line. Changes to 3.2:

| cycle | what happens |
|---|---|
| k | access consumed: S_IDLE -> S_SETUP |
| k+1 | S_SETUP -> S_RUN |
| k+2 .. | S_RUN: a planned beat is pushed into a two-entry queue (`issue_ok = plan_valid && have_slot && !q_tv`); its beat slot is reserved and its lanes retired at the push |
| push+1 .. | the queue head is on the port (`mem_req_valid = q_hv`), popped on `mem_req_ready` |
| | S_RUN -> S_DRAIN on the edge the queue empties (`!plan_valid && !issue_fire && q_empty`), then as 3.2 |

**Completion register** (`tnca_lsu_hybrid.v` 97-121): either path's completion
lands in `out_*` when it is free or being taken this cycle (`buffer_ready =
!out_valid || cpl_ack`), the line path's first (the modes never overlap). The
core's `lsu_cpl_valid` is `out_valid`; writeback arbitration is 3.4's
(`ws_lsu = out_valid && !sched_wb`; a store takes the slot with `nw_en = 0`).
One cycle more than R40 from a path's completion to the WB stage.

| from the last response p | line path | word path |
|---|---|---|
| completion register loaded | `E_{p+1}` | `E_{p+3}` (DRAIN -> DONE at `E_{p+1}`, `cpl_valid` at `E_{p+2}`) |
| writeback port (no `sched_wb`) | p+2 | p+4 |
| next access of the same path consumed | as soon as a slot is free | p+4 (unit idle), first beat on the port p+7 |

**Port priority** (`tnca_mem_mux.v` A_FIRST = TENSOR_LINE && HYBRID_LSU): the
LSU first, the transfer engine only when the LSU has no request (the reverse of
3.1). Inside the LSU the line path's request comes first (both never have one).

Trace (`tnca_core_trace.v` at MEM_DW 512): `Q` carries 64 byte enables and 512
data bits, `P` the 512-bit line; the model's `CoreTrace::QW/PW`. Verified in
core replay, one core, on the smallest tier's elementwise and reduction kernels
that use no transfer engine: every launch exact (one of them 2,486 cycles, where
the R40 rules gave 4,384).

## 4. The engines

### 4.1 Matrix engine (`tnca_matrix.v`)

Command in X at cycle c (`mtx_cmd_valid`, `tnca_core.v:1318`); `busy = st !=
S_IDLE` from c+1; `done` is a one-cycle pulse in the cycle AFTER S_DONE
(busy already 0 in that cycle). `mtx_free_next = !mtx_busy && !mtx_cmd_valid`
lets the next MATRIX/ACC-FU instruction leave R in the `done` cycle.

| command | path | busy cycles (no VRF contention) | `done` pulse at |
|---|---|---|---|
| `m.zero` / `a.zero` | S_ZERO x16 (one row per cycle, 406-415) -> S_DONE | 17 | c+18 |
| `m.mma` with ABUF (FP16/BF16 K16; FP8 K16/K32: 396-400) | S_LOADA: a register reads, 2 cycles each (request, response; 419-423) -> per k: S_LOADB 2 cycles (427-436) + S_MAC 16 cycles (one m per cycle, `use_abuf`, 442-466) -> S_DRAIN PIPE+2 = 5 cycles (490-493) -> S_DONE 1 | `2a + 18K + PIPE + 3 = 2a + 18K + 6`; a = K*16*esz/64 = K/4 (16-bit) or K/8 (8-bit); K in {16,32,64} | c + 2a + 18K + 7 |
| `m.mma` without ABUF (FP16/BF16 K32/K64, FP8 K64) | per k: S_LOADB 2 + 16 MACs each needing its own read (2 cycles: request, response) = 34 | `34K + 6` | c + 34K + 7 |
| `a.vwrite` | S_ROW 1 (direct write, 472-478) -> S_DONE | 2 | c+3 |
| `a.mul.row`, `a.fma.row` | S_ROW 1 -> S_DRAIN 5 -> S_DONE 1 | 7 | c+8 |
| `a.vread` | no command: `acc_rd_data` read combinationally in X, single-cycle writeback | 0 | - |

Examples: FP16 K16: a=8 -> 310 busy cycles; FP8 K32: a=8 -> 598; FP16 K32
unbuffered: 1094.

`src_free` (second barrier event, `mod[4]`) pulses in the first S_DRAIN cycle
(`src_free <= r_sf_en` at the last MAC edge, 455-456), i.e. PIPE+3 cycles
before `done`. `mtx_done` clears `sb_acc[done_slot]` (2482); `hz_acc` is
exempt in the done cycle itself (1143-1144).

VRF port 3 requests (`vrf_req`, 216) go through the same round-robin arbiter
as the execute ports; each refused cycle adds one cycle to S_LOADA/S_LOADB
(or to every MAC when unbuffered). D's prefetch re-asking every stalled
cycle competes on the same banks.

**R41 (MAC_DP 1, MTX_VRF_REG 1; model `core_mtx_pipe=5`, `core_mtx_vrf_reg=1`).**

- Every MAC is a `tnca_mac_lane`, so `PIPE = OPND_REG + popcount(MAC_CUTS)` =
  1 + 4 = 5 for every command (m.mma, m.mma.dp, the row ops), not
  `FMA_LAT + OPND_REG` = 3: S_DRAIN lasts PIPE + 1 = 6 cycles.
- `m.mma.dp.async` (major MATRIX func 2, ISA 9C.4; B's format in `mod[6:5]`)
  steps k by the group size: 4 for E4M3 x E4M3, else 2 (`kstep`, 206-207).
  So a command is K/kstep LOADB + 16-cycle sweeps, ending when
  `kk == K - kstep`. B's register for k is `vb + kk/4` (8-bit B) or `kk/2`, by
  B's own format. A's addressing and the ABUF choice are m.mma's (A's format).
- MTX_VRF_REG (`tnca_core.v` 948-960) registers the banked file's port-3
  response: the matrix takes its data two cycles after the grant, and asks
  for nothing in either cycle (`rq_v[3] = req && !rsp_valid && !valid_q`). So
  every fragment read (LOADA, LOADB, an unbuffered MAC's A) costs three
  cycles, not two.

### 4.2 Transfer engine (`tnca_tensor.v`)

`busy = st != S_IDLE` (292). Command in X at c; `cfg` ops 1..3 (td.release,
w.def, w.release) take effect at `E_c` without making the engine busy
(779-790); `t.prefetch` is dropped in the core (1606-1607). `done` is a
one-cycle pulse in the cycle after S_DONE (`done <= 1` at the S_DONE edge,
1213-1219).

| phase | cycles | line |
|---|---|---|
| S_TSETUP (snapshot, normalise) | 1 (c+1) | 931-976 |
| S_TCHK (geometry; fault here has no side effect) | 1 (c+2) | 981-994 |
| S_TMUL (`t_acc += c[p]*s[p]` for outer positions, then one walk-setup cycle) | rank (c+3 .. c+2+rank) | 999-1022 |
| first walk cycle | c+3+rank | |
| LOAD walk S_TRUN, per 8-byte box unit | aligned (`r_m==0`): 1 request cycle when `mem_req_ready && have_slot` (8 slots, 597-601); misaligned row: 2 request cycles (low word, then `t_half` high word, 1059-1064); unit with no byte inside the tensor: 1 cycle writing the fill to the window, only when no completed slot wants the port (`!have_done`) and no held write (1032-1041) | 1029-1089 |
| load slot lifetime | issue at i, response at i+M (both words for a misaligned unit), `sl_done` at i+M+1, window write presented at i+M+2 (`vw_en`, one unit = 2 lanes per cycle, 824-832), slot free from i+M+2; writes to the banked VRF lose to the WB stage on a same-bank cycle and are re-presented (`vw_hold`, 740, 765) | 457-490, 795-832 |
| S_TDRAIN | until `sl_v==0 && !mem_rsp_valid` (the last write was presented) | 1091-1093 |
| S_DONE | 1 cycle, waits `!vw_hold`; `done` pulses the cycle after | 1213-1219 |
| STORE walk S_SRUN, per memory word | 1 request cycle when `t_rowv && !st_need_reg && st_be!=0 && have_tag` (16 tags, 603-604); a word with `st_be==0` or a row outside the tensor advances in 1 cycle without a request; a new 64-byte window register costs S_SREAD: request (port 4) one cycle, data the next, back in S_SRUN the third (3 cycles per register, `tnca_tensor.v:1107-1108, 1141-1148`, `tnca_core.v:931`) | 1100-1139 |
| store tag lifetime | until the write's response (`tg_v`, 804-806); S_SDRAIN waits `tg_v==0 && !mem_rsp_valid` | 1150-1152 |
| TRANSPOSED (`mod[5]`, rank 2, 16-bit) per column block (`t_row_units` blocks) | S_XRUN: `t_b3` cycles (one request per source row, tags), S_XDRAIN until all responses, S_XWRITE 4 cycles (one whole-lane row write each, `vw_hold` stalls); blocks do not overlap | 1159-1208 |
| `td.load` | 32 words strictly serial: S_DLOAD (request) / S_DWAIT (response) per word, then S_DPARSE 1, S_DONE 1; busy throughout, so the next TXFER-FU instruction waits (`stall_txf`, parks with RCTX_SWITCH=2) | 882-925 |
| inactive command (`cmd_active=0`) | S_IDLE -> S_DONE directly: busy 1 cycle, `done` at c+2 | 876 |
| fault | S_FAULT: `fault`, `done`, `done_err` pulses; the barrier's error path | 1225-1236 |

Throughput bounds with memory latency M (accept -> response): loads are slot
bound at 8 units per (M+2) cycles and port bound at 1 unit (aligned) per cycle;
stores are tag bound at 16 words per (M+1) cycles. The engine's requests take
the core port ahead of the LSU every cycle they are valid (3.1).

### 4.3 What the core does while an engine is busy

| situation | core behaviour |
|---|---|
| another MATRIX/ACC-FU instruction reaches R | `stall_mtx`; with another context in D and `r_of_idle` it parks (`m.mma`, `m.zero`, `a.zero`, `a.vread` have no vector source so they can park; `a.vwrite`/`a.fma.row` fetch `Vb` first and then cannot) |
| another TXFER-FU instruction reaches R | `stall_txf`; parks under the same rule (`t.ld/t.st/td.*/w.*` have no vector source) |
| engine goes idle at cycle t (`busy_q && !busy`) | `eng_freed`: every parked context wakes at `E_t` |
| `mtx_done` at cycle t | every parked context wakes; `sb_acc[slot]` released |
| `mb.arrive` of the issuing context (RPAIR) | waits for the engine command its own context issued (`arr_mtx`, `arr_txf`) |
| completion -> barrier | `mtx_done && mtx_done_bar_en` / `mtx_free` / `txf_done && txf_done_bar_en` are counted in the pulse cycle (5.2) |

### 4.4 R41: the line-wide, multi-command transfer engine (TENSOR_LINE)

`tnca_tensor.v` with LINE_ENGINE (core: TENSOR_LINE && HYBRID_LSU; model
`core_tensor_line=1`, `core_tensor_mul_pipe=1`) owns the core's 64-byte port
(3.6: the LSU before it). Tags: line slot `{1, slot}` (LSLOT_N = LINE_SLOTS,
64), TensorMap line `{0, 1, line}`, word walk `{0, 0, tag}`.

**Address multiply** (S_TMUL, MUL_PIPE 1 on every R41 build): one product a
cycle, registered, so S_TMUL lasts `rank + 1` cycles (one for rank 1), not
`rank`.

**TensorMap** (S_IDLE, 1239): at a line-aligned address td.load issues its
four lines back to back (S_LDLOAD, all in flight), waits for the fourth
response (S_LDWAIT: `dlr == 4` before the edge), folds the box element count
one dimension a step (S_LDFOLD, five cycles), then S_DPARSE and S_DONE as
before. Any other address takes R40's 32-word sequence, each word placed in
its line.

**Which transfers go by lines** (`le_elig`, 932-937, decided in S_TMUL's last
cycle): not transposed; row 0 starts on a line (`(t_acc + t_CB) mod 64 = 0`);
every outer stride a multiple of 64; the row a whole number of lines; the box
inside the tensor in every dimension (`0 <= c`, `c + block <= extent`). Then
the command takes a free record (CMD_N 8; none free: S_TMUL repeats its last
cycle), the barrier signal moves to the record, and the walk goes to S_LRUN
(load) or S_LSREAD (store). Everything else is the R40 word walk.

| state | per cycle |
|---|---|
| S_LRUN | with a free line slot, request line `t_line` of the row (`l_addr = t_ps + 64 t_line`, all 64 bytes enabled); on the port's acceptance the slot is taken (`{cid, register = window base + t_flat/64 + t_line}`) and the walk steps (next line; at the row's end the odometer; after the last row S_LEND) |
| S_LSREAD | the engine read port (port 4) asks for the line's register; `vr_valid` -> `sr_data`, S_LSRUN |
| S_LSRUN | as S_LRUN with the register in hand as the write data; meanwhile port 4 reads the next line's register ahead (`pf_req = !pf_v && l_more`). On a fire with the next register neither held (`pf_v`) nor arriving: S_LSREAD. So a store line goes every other cycle unless the port holds it longer |
| S_LEND | the record is closed (`cm_open <= 0`); S_IDLE: the walk takes the next command while the lines are in flight |

Line slots: a store's response frees its slot; a load's response marks it
ready, and its whole register is written through the window port in the next
cycle the word walk does not use it (`lw_fire = lrdy && !vw_hold &&
!legacy_vw`, the lowest ready slot). A record counts +1 per line issued, -1
per store response and per load register write; a closed record at zero
signals its barrier (`done`) unless the walk signals its own completion that
cycle. The core's `txf_busy` is the walk only (`stall_txf`, wakes); `mb.arrive`
also waits for its context's records (`inflight_rctx`, RPAIR); `core_idle`
waits for everything (`busy_any`).

**Transposed walk** (g_line_port 830-889): each word read that goes to memory
allocates the buffer entry of its source row (16) and fills it with the whole
line on its response; a later unit of the same line in that row is answered
from the entry (no port request; the answer is presented the next cycle and
yields to a memory response). Any write request, and S_TSETUP, empty the
buffer. 1-byte elements (HAS_XPOSE 2): eight destination rows a column block,
a row R bytes.

## 5. Barriers (`tnca_barrier.v`)

### 5.1 Instructions

| instruction | R | X (cycle x) | effect at `E_x` | result |
|---|---|---|---|---|
| `mb.init b` | - | `init_en` | counts/epochs of b zeroed; fault if a context sleeps on b (252-263) | none |
| `mb.arm b, arrivals, ops` | WAW on its S dest | `arm_en`; `arm_token = {cur_epoch[b]+1, b}` combinational (154-155) | `cur_epoch[b]++`, `n_arrive/n_ops` loaded, `armed[b]<=1` (268-281) | token in S at x+1 (single-cycle writeback) |
| `mb.arrive b` | `stall_arrive` (2.8) | `arrive_en` | `n_arrive[b]--` (290-298); if both counts reach 0 this cycle (`completes[b]`, 184-195) `done_epoch[b] <= cur_epoch[b]` (336-340) and every sleeper on b gets `wake<=1` (360-374) | none |
| `mb.test tok` | `Sa` hazard | `test_done = done_epoch[b] >= epoch` (151) | - | `{63'd0, test_done}` in S at x+1 |
| `mb.wait tok` | `Sa` hazard | if `test_done`: retires as a no-op; else `bar_sleep`: `x_redirect` (squash own F/D/R), `sleep_en` | `blocked[x_rctx]<=1; pc[x_rctx]<=x_pc`; `slp_v/bar/epoch` recorded (354-358) | re-executes on wake |
| `mb.inval b` | - | `inval_en` | `err[b]<=1`, `done_epoch[b]<=cur_epoch[b]`; sleepers on b woken with `wake_err` | none |

### 5.2 Operation completions

`OP_DEC(b)` (171-175) sums, in one cycle, `mtx_done_en` (= `mtx_done &&
mtx_done_bar_en`), `mtx_free_en`, and `txf_done_en && !txf_done_err` for
barrier b; `n_ops[b] -= OP_DEC(b)` at the edge (320-331); underflow is a
fault. A completion in the same cycle as an `mb.arm` of the same barrier is
not applied (322). `txf_done_err` sets `err` and completes the generation
(307-310).

### 5.3 Sleep and wake timeline

| cycle | event |
|---|---|
| x | `mb.wait` in X, `test_done=0` -> `bar_sleep` |
| `E_x` | `blocked[i]<=1`, `slp_v[i]<=1`, F/D/R of i squashed, `pc[i] <= x_pc` |
| w | the completing event (`arrive_en`, `OP_DEC`, `inval_en`) makes `completes[slp_bar[i]]` true, or `done_epoch >= slp_epoch` already holds (checked every cycle, 367-368) |
| `E_w` | `wake[i] <= 1`, `slp_v[i] <= 0` |
| w+1 | `bar_wake[i]=1` -> core: `blocked[i] <= 0` at `E_{w+1}` (2500-2512) |
| w+2 | context fetchable; fetch issues at w+2 at the earliest (`rr_ptr` permitting) |
| w+6 | the `mb.wait` re-executes in X (`x_valid` again, `instret++` again, `I` record again); now `test_done=1`, retires |
| w+7 | its successor in X |

A sleeper registered at `E_x` is first examined at x+1, so a completion in the
very cycle x (seen by `completes` but not by the sleep decision) wakes it at
`E_{x+1}` (via `done_epoch >= slp_epoch`), not lost.

For an engine completion the "w" above is the `done` pulse cycle: matrix
`c + 2a + 18K + 7` (4.1), transfer: the cycle after S_DONE (4.2).

## 6. Launch, `instret`, `core_idle`

### 6.1 Launch

`launch_valid` at cycle L (2255-2300), evaluated before every other sequential
rule (the launch branch is the `if`, the pipeline the `else`):

| at `E_L` |
|---|
| `pc[i] <= launch_pc` for masked i, `instret[i] <= 0` for all i |
| `running <= launch_role_mask; launched <= mask; faulted <= 0; blocked <= 0; parked <= 0; rr_ptr <= 0` |
| `slice_q/pair_q/slice_k`, `pid_*`, `arg_base`, `vbase_q` (odd contexts take the even partner's base under RPAIR) |
| `f1_valid, fb_valid, d_valid, r_valid, x_valid, w_en <= 0; wb_rsv <= 0; fp_v[*], mu_v[*] <= 0; sb_s/sb_v/sb_p/sb_acc <= 0` |
| `u_vrf.clear` (bank read round robin to 0), `u_mtx.flush`, `u_bar.flush`, `u_txf.flush` (same-cycle, combinational `flush=launch_valid`) |
| NOT cleared: the LSU units, `red_pend`, the FPU sequencers (core_idle guarantees they are idle before the dispatcher launches) |

| cycle | first instruction |
|---|---|
| L+1 | `fetchable` true, `f_sel=0`'s first fetchable context, `f_issue` |
| L+2 | word on `imem_data`, `f1_valid` -> `d_valid` at `E_{L+2}` |
| L+3 | D |
| L+4 | R |
| L+5 | X (`I` record), `instret` becomes 1 at `E_{L+5}` |

Outside the core: `tnca_top.v:285-320` puts a `DISPATCH_STAGES=2` elastic link
between `tnca_grid_dispatch` and `tile_launch`, and another on the completion
(`started && tile_idle`); `tile_idle` is `core_idle` directly
(`tnca_top.v:492`, `tnca_tile.v:227`).

### 6.2 `instret`

`instret[x_rctx]++` at `E_x` for every `x_valid` (2574-2575): sleeping
`mb.wait`s, faulting instructions and `s.exit` all count; a re-executed
`mb.wait` counts again. Reset to 0 by the launch.

### 6.3 `core_idle` (2729-2735)

`core_idle = (running == 0) && !f1_valid && !fb_valid && !d_valid && !r_valid
&& !x_valid && lsu_req_ready && !lsu_cpl_valid && !fpu_busy && !red_pend &&
!w_en && !mtx_busy && !txf_busy && (wb_rsv == 0)`.

Note `running` drops at the `s.exit`'s own `E_x`; a context asleep on a barrier
is still `running`; a parked context is still `running`; the FP/MUL tag
pipelines are covered by `wb_rsv`. The earliest `core_idle` after the last
context's `s.exit` in X at cycle x is x+1 if nothing else is outstanding (the
`s.exit` has no writeback; `w_en` for an earlier instruction in X at x-1 is
high during x and low from x+1).

## 7. Points I am not certain about (with where to look)

1. Matrix busy count vs. the profile's numbers. From `tnca_matrix.v:369-502` I
   count busy = `2a + 18K + 6` for ABUF (FP16 K16: 310) and `34K + 6`
   unbuffered (K16: 550), with the `done` pulse one cycle later. The profile
   description (`spec/r1_profiles.json`, core-vrf-sync-v5) says 551 -> 311 and
   the README's `mtx_zero_cycles` says 17. Either the oracle counts
   (done cycle - issue cycle) = busy+1 for MMA but busy for m.zero, or I have
   the S_LOADA/S_LOADB handshake off by one. Check
   `tools/performance_harness/systemc/rtl_oracle/matrix_oracle.py` (how it
   measures) and `tnca_matrix.v:419-436` with the banked file's one-cycle
   response (`tnca_core.v:931-932, 975`).
2. The FP reduction and SFU landing cycles (2.13) assume `fpu_done` is seen by
   the core in the cycle `red_done`/`sfu_done` registers are high and that
   `red_pend` then needs a free port; verify `tnca_vector_fpu.v:466-477`
   (`rwait` compares against `FP_LAT[1:0]`) and `tnca_sfu.v` (`cnt == VLEN +
   LAT`) against a waveform; I derived them, did not simulate.
3. The SFU's `done` cycle (x+24) comes from `tnca_sfu.v:129` (`busy <= 1` on
   `start`), `:135-137` (`done <= 1` when `cnt == VLEN + LAT`) and the fact
   that `cnt` is 0 in the first busy cycle; the header comment at line 33 says
   "done ... valid in the following cycle", which matches, but I did not run
   `tb_sfu_core` to confirm.
4. Vector load-to-use of +6 (3.4) relies on `AVAIL_V` admitting the read in
   the `w_en` cycle and the bank bypass returning the new data
   (`tnca_rf_bank.v:145, 166`); the model README states +5 without
   distinguishing scalar and vector destinations.
5. Bank conflicts between D's prefetch (re-offered every stalled cycle), the
   matrix port 3 and the tensor port 4 depend on `u_vrf.rr_ptr`
   (`tnca_vrf_banked.v:221-268`), which advances past the first port served
   in arbitration order. A model must replicate the pointer exactly; I have
   described the rule, not enumerated cases.
6. Transfer-engine fill units and the window-write port: when a completed slot
   (`have_done`) and a fill unit both want `vw_en`, the slot wins and the walk
   does not advance that cycle (`tnca_tensor.v:824-832, 1032-1041`); I did not
   trace the interaction with `vw_hold` under sustained WB-stage writes to the
   same bank.
7. The dispatcher path (`tnca_top.v:285-320`, `tnca_grid_dispatch.v:104-130`,
   `DISPATCH_STAGES=2`, `tnca_elastic_link`) is outside the core; the
   `launch_valid` cycle L is what the core sees, and I did not derive how many
   cycles after `grid_valid` it arrives nor the 6-cycle `dispatch_cycles` of
   the README.
8. Memory response ordering: the LSU and tensor engine accept responses in any
   order by tag; the `p` ("last response") in 3.2 is whichever arrives last.
   `lm_req_ready` is low whenever the tensor engine asserts `b_req_valid`
   (`tnca_mem_mux.v:82-83`); a model that does not give the engine strict
   priority will misplace LSU beats.
9. `x_access_fault` for a misaligned atomic (1421-1425) is raised in X; I did
   not check whether the LSU can also fault on anything else (it cannot in the
   RTL I read; the only fault producers are 2552-2615).
10. `hz_sb2`/`hz_sbx` for `t.prefetch`: it is a TXFER major, so it checks
    `Sb+1..Sb+4` and takes the `stall_txf` wait although it is then dropped
    (1606-1607); confirm the decoder's `TNCA_T_PREFETCH` arm (810-830) gives
    `o_fu = FU_TXFER` (it does, line 811 is above the case).
11. The behaviour when `rp_go` and `wb_steal` coincide: `stall_front=0`, the
    R instruction is dropped, D's instruction advances into R, X becomes a
    bubble (`kill_r`); the completion still waits for `sched_wb` to clear.
    I read this from 2025-2029 and 2406-2459 but did not simulate it.

## 8. Where the model-side description disagrees with the Verilog

| README / sources.json | Verilog |
|---|---|
| `scalar_mul_latency: "MUL_LAT 5"`, `vector_fp_latency: "FP_LAT 5"` | RTL parameters are `MUL_LAT=3`, `FP_LAT=2` (`tnca_core.v:142,144`); the consumer-visible x+5 of both (2.13) is what the 5 means |
| `predicate_latency = 2` ("M1 value") | a predicate from a single-cycle op is usable at x+1 (`X_PROD_P`, 1070, 2.13) |
| `barrier_latency = 2` ("M1 value") | `mb.arm` token and `mb.test` result are ordinary single-cycle scalar results, usable at x+1 |
| `load_latency: last response + 5` | +5 for a scalar destination, +6 for a vector destination (3.4) |
| `vrf_bank_conflict_cycles: "(tnca_operand_collector.v)"` | that module is not instantiated; the rule is `tnca_vrf_banked.v:239-268` plus `tnca_core.v:746-811` |
| `mtx_pipe_cycles: "+6 = pipe 4 + 1 + complete 1"` | S_DRAIN is PIPE+2 = 5 cycles (PIPE = 3) and S_DONE 1: same 6, different decomposition (`tnca_matrix.v:457, 490-493`) |
| `lsu_setup_cycles=1` in the cfg vs. `sources.json` "first beat at c+2" | first beat is at c+2 (S_IDLE, S_SETUP, then S_RUN) |
| `park_refetch_cycles=5: "refetches 5 cycles after its wake"` | the fetch issues 1 cycle after the wake; the refetched instruction is in R 4 and in X 5 cycles after the wake (2.16) |
| `dma_setup_cycles=2` ("2 + rank") | correct: S_TSETUP, S_TCHK, then `rank` S_TMUL cycles (4.2) |
| `fp_reduction_latency=16`, `sfu_latency=28`, `sfu_occupancy` 27, `acc_read_latency=3`, `fp_compare_latency=4`, `vector_int_latency=3`, `scalar_latency=1`, `branch_taken_cycles=4`, `mtx_zero_cycles` 17, `mtx_row_cycles` 2/7 | agree with 2.13, 2.17, 4.1 |
