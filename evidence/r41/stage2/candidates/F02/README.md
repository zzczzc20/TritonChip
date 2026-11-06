# F02: scalar read → barrier / scoreboard / issue

Census families (attempt 3, routed, 100 MHz, residual after M1–M3), all starting at an X-stage scalar read:
- `x_sa_idx` → `sb_s` / `sb_v` D (782 endpoints, −206 ns): the read feeds the barrier test or branch compare, then
  `x_redirect`, `kill_r`, the scoreboard claim of the instruction leaving R, and finally the scoreboard bit.
- `x_sb_idx` → the matrix MAC input `a_q` (189 endpoints, −57 ns, WNS −0.902): `sb_data` is `a.vread`'s row. As the
  accumulator file's read address when the engine is idle, it reaches the MAC operand mux.
- The same read is the first ~2.8 ns of the F01 enable paths and of the F04 PC-enable paths.

## Design

The scalar file's operand ports A, B and C are read in R, and the value travels into X in a register. The one write
that can still land on a source between that read and X is forwarded, and the forward is decided in R together
with the value. "Read value and dependency check in separate stages": R does the read and the match, and X
starts from registers. All changes are in `rtl/core/tnca_core.v`:

- `u_srf` ports A–C are addressed by `r_rctx` / `r_s{a,b,c}_idx`. The read is asynchronous and write-first, as
  before, so it includes W's write of the same cycle.
- `x_s{a,b,c}_q` hold the read. `x_s{a,b,c}_fw` hold "the write latched into W at this edge targets this source":
  `nw_en && nw_class == S && nw_rctx == r_rctx && nw_idx == idx && idx != 0 && !vrf_wb_freeze`.
- In X, `s{a,b,c}_data = fw ? w_sdata : q`. That is one LUT from three registers.
- Like `x_va_q` (F01), these load every cycle. A bubble in X uses none of them, so they add no load to
  `stall_front`.

**Why it is exact.** Take the instruction that leaves R at the end of cycle t.
- The old design read the file in X at t+1, write-first with W's write at t+1.
- The new design reads in R at t, write-first with W's write at t. The file it sees is the file at the start of
  t+1.
- W's write at t+1 is exactly what `nw_*` latches into `w_*` at the end of t. So the R-stage forward bit names it,
  and `w_sdata` is its value.
- Exceptions:
  - A frozen writeback keeps `w_*`. It is a vector write, and it holds the front end, so R does not leave.
  - S0 reads zero and is never forwarded.
- The match is computed against the actual next writer, not the scoreboard's prediction. So it is exact for every
  writer: X results, the MUL pipe, the FP pipe, LSU completions in any return order, and reductions.
- This holds whatever the hazard logic does, and is not a consequence of it.
- Ports D–G stay in X: a transfer's coordinates and `m.mma`'s second token. They are not on the redirect, stall or
  matrix-operand paths.

**Cycle cost: none.** No stage is added, and X sees the same value in the same cycle. The bench assertion below
proves this cycle by cycle.

**In-flight, replay, flush, fault and ownership.**
- The scoreboard, the hazard logic and the squash rules are untouched. The claim and the redirect are computed as
  before, from the same values, only sooner after the clock.
- Replay (RCTX_SWITCH refetch) re-reads in R like any other entry.
- A launch clears X (`x_valid`). X's operand registers are don't-care until a valid entry arrives.
- Faults read the same address value. The access-fault check, the barrier fault and the transfer descriptor
  fault see identical operands.
- Ownership is unchanged.

## Verification

- **Bench assertion** (`tnca_core`, simulation only, every valid X cycle, every bench): `sa/sb/sc_data` equal the
  old read. That is `u_srf`'s array at X's own indices with the write-first bypass of W, recomputed independently
  from `u_srf.mem/valid`. A mismatch prints `FAIL`.
- **Coverage:** `+r41cov` prints `fwd_s`, the X cycles that used a forwarded scalar.
- Directed test, regression list, results and cycle tables: `regression.md`. OOC: `ooc.md`.
