# F04: writeback `w_rctx` / `w_en` → `w_sdata`, and the PC enable

Census families (attempt 3, routed, 100 MHz, residual after M1–M3):
- **`w_rctx` → `w_sdata` / `w_vdata`** (128 endpoints, WNS −0.908): the writeback's context and index enter the
  predicate file's write-first compare in X. From there the path runs `rm_data` → `lane_mask` (fanout 128) → the
  vector ALU's masked reduction → `x_sres` → `w_sdata`, 17 levels.
- **`x_pm_idx` → `w_vdata` / `w_sdata`** (93 endpoints, −35 ns): the same mask read, from its index.
- **`w_en` → `w_vdata`** (40 endpoints): through the scalar or predicate bypass into the vector ALU.
- **`w_en` → `pc_reg` CE** (223 endpoints, −50.7 ns): the scalar bypass, then the barrier test or branch compare,
  then the redirect, then the PC write.

## Design

"Context and data in the same row": the writeback's context/index match is made one stage earlier, against the
write being latched (`nw_*`), and it travels with the operand it selects.
- **Predicate operands:** the predicate file's ports (`pa`, `pb` and the mask port `pm`) are read in R, exactly as
  F02 does for the scalar file. All changes are in `rtl/core/tnca_core.v`.
  - `u_prf` is addressed by `r_rctx` and `r_p{a,b,m}_idx`.
  - `x_pa_q`/`x_pb_q` hold the read. `x_lm_q` holds the lane mask R computed from it:
    `r_pm_en ? (r_pm_neg ? ~rd : rd) : all ones`.
  - The forward bits are `x_pa_fw`, `x_pb_fw` and `x_lm_fw`. Each is `nw_en && nw_class == P &&
    nw_rctx == r_rctx && nw_idx == idx && idx != 0 && !vrf_wb_freeze`, and `x_lm_fw` is also gated by `r_pm_en`.
  - In X, `lane_mask = x_lm_fw ? (x_pm_neg ? ~w_pdata : w_pdata) : x_lm_q`: one LUT per lane from registers.
  - `pa/pb_data` use the same select.
  - These registers load every cycle, as F01/F02's do, so they add no load to `stall_front`.
- **The PC enable:** this family is cut by F02. After F02 the redirect starts from `x_sa_q`/`w_sdata`, not from
  the scalar file's read and bypass.
- **What remains of the W stage:** `w_rctx`/`w_en` no longer reach `w_sdata`/`w_vdata` through any X-stage read.
  The remaining compares against W are in R and end at registers: the `x_*_q` captures, the scoreboard's `WB_HIT`
  and the F05 busy-bit updates.

**Why it is exact.** The F02 argument applies with the predicate file's rules.
- The R read is write-first with W's write of the same cycle.
- The only write that can land between R and X is the one `nw_*` latches, and the forward bit names it.
- P0 reads all ones, is never written and never forwarded.
- The match is made against the actual next writer and context, not the scoreboard's prediction. So a write to the
  same predicate index by another context is never forwarded, and a completion returning in any order is
  forwarded exactly when it lands.

**Cycle cost: none.** No stage is added, and X sees the same mask and predicates in the same cycle.

**In-flight, replay, flush, fault and ownership.** As for F02: the scoreboard, squash, replay, launch, fault and
ownership logic are untouched, and X's mask is don't-care in a bubble.

## Verification

- **Bench assertion** (every valid X cycle, every bench): `pa_data`, `pb_data` and `lane_mask` equal the old reads
  of `u_prf.mem` at X's indices, with W's write-first bypass and the negate and enable applied. A mismatch prints
  `FAIL`.
- **Coverage:** `+r41cov` prints `fwd_p`.
- Directed test sections B (masked moves, negated mask, `p.and`/`p.ballot` and masked reductions on a mask
  produced by the instruction just ahead) and G (contexts writing the same indices).
- Regression list, results and cycle tables: `regression.md`. OOC: `ooc.md`.
