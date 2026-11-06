# F01: the core's stall decision and the operand-register enables

Census family (attempt 3, routed, 100 MHz, after the M1–M3 cone exclusion): the core's `stall_front` is the clock
enable of the X-stage operand registers `x_va_q/x_vb_q/x_vc_q` (3 × 512 bits), of the operand holding registers
`of_d0..2` (3 × 512 bits) and of every D/R/X pipeline register. The placer put it on a BUFG (fanout 1,653, about
3 ns). Its cone started at X's scalar read: `x_sa_idx` → `u_srf` read → the barrier test's 48-bit compare →
`x_redirect` → `rp_go` → `stall_front` → CE, 20 levels. This class is 4,650 endpoints and −2,168 ns of the
residual census TNS.

## Design

Three changes in `rtl/core/tnca_core.v`. Together they take every X-stage data value out of the stall decision and
take the 3,072 wide operand registers off it.

1. **No stall enable on the wide operand registers.**
   - `x_va_q/x_vb_q/x_vc_q` load every cycle.
     - A stalled front end sends a bubble into X (`x_valid <= 0`). X never repeats or holds an instruction.
     - Every consumer of `va/vb/vc_data` acts only on a valid X: ALU/FPU results reach the writeback through
       `x_has_dest` and the `fp_v` pipe; the FPU reduction and the SFU latch their source on `start`; the matrix,
       transfer and memory units latch on their command valids. All of these include `x_valid`.
     - What these registers hold in a bubble cycle is therefore never used.
     - The VALU DSP input registers that Vivado folds them into lose the enable too.
   - `of_d0..2` are enabled by their response alone (`of_arrive`), not by `stall_front & of_arrive`.
     - A response is only ever R's own: `of_arrive` already drops a prefetch response when D did not move up.
     - When the instruction leaves R in the same cycle, the response goes straight into X through `of_v*`.
     - The next occupant of R starts with `of_got` clear and reads `of_d*` only after its own response lands.

2. **The give-way rule (RCTX_SWITCH) asks X's opcode, not X's result.**
   - `rp_go` used `!x_redirect`, the end of X's scalar read, branch compare and barrier test, and `rp_go` is a
     term of `stall_front`.
   - It now uses `!x_may_redirect`: a branch, exit, trap, illegal instruction, atomic (whose address can fault) or
     `mb.wait` in X. These are decode bits registered with the instruction.
   - `x_may_redirect` covers every cycle `x_redirect` covers. So a give-way and a redirect still never share a
     cycle, and the PC, park and squash updates keep their one-squash-a-cycle order.
   - The only difference: a give-way now also waits behind such an instruction when it does *not* redirect, for
     example a branch that falls through. That costs at most one cycle, and holding R is always legal.
   - An exact alternative was measured and rejected (Cycle impact, below). It let the give-way share a cycle with
     the redirect, with the redirect's PC taking priority.

3. **"X is taking the load/store unit" asks X's opcode.**
   - `lsu_free_next` used `!x_mem_go`, whose `x_access_fault` term is X's misaligned-atomic check on the scalar
     address. It now uses `!(x_valid && x_mem_en)`.
   - The two differ only for an access that faults, and that access stops its context. So the only effect is that
     an access in R may wait one extra cycle behind a faulting atomic.

What is left in `stall_front` is registered state: the scoreboard and its R-stage lookups, `w_*`, the unit busy
flags, the writeback reservation and the operand-collector state.

**In-flight, replay, flush, fault and ownership.**
- **In-flight work:** unchanged. The FP/MUL pipes, the reservation, the LSU and the engines never used the
  removed enables.
- **Replay** (the give-way's refetch from `r_pc` and park): fires in a subset of the cycles it fired in before.
  Its effects are unchanged.
- **Flush** (launch) and **squash** (`kill_*`): unchanged. They still use `x_redirect`, which no longer reaches
  the enable network.
- **Faults:** precise as before. An access fault, trap or illegal instruction still redirects in X, and nothing
  younger has claimed the scoreboard.
- **Ownership:** unchanged. Matrix and transfer issuer, LSU tags and barrier sleepers are as before.
- **Not done:** the stall is not delayed a cycle.

## Verification (GB10, iverilog 12)

`regression.md` has the full results. In summary:
- Every core bench (`core`, `core-slowmem`, `core-ooo`, `core-vrf`, `core-r39hw`, `core-r39sw`, `core-r39sw4`,
  `core-r39sw5`), plus `top`, `r38-xpose-rtl` and `r41-core-boundary`, passes. That is 538 cases.
- The new directed program `tb/core/progs/t41_boundary.s` runs on every core bench at 1, 2 and all contexts.
- Its `+r41cov` counters show the cases this candidate touches occurring:
  - an R stall holding collected operands;
  - a dropped prefetch response;
  - a give-way candidate meeting a redirect;
  - a give-way candidate held behind a non-redirecting branch.

## Cycle impact

- **No change on any bench without RCTX_SWITCH.**
- **With RCTX_SWITCH:**
  - on the image configuration (`r39sw4`/`r39sw5`), `t39_switch` is 1.5–1.9 % faster and every other program is
    identical;
  - on the two-context `r39sw`, `t39_switch` is +1 / +14 cycles (+0.5 % on the reordering memory).
- No pipeline stage is added.

## Unit OOC timing

`ooc.md`.
