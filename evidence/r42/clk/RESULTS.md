# R42 clk: path-family clock fixes (CLKFIX bits 0-7), draft in progress

Status line (updated as results arrive): see the "Results so far" section; nothing here is a board number. Domains: sim = RTL simulation (iverilog, GB10), OOC = out-of-context Vivado on the x86 host, routed = after route_design, TLM = SystemC model, derived = arithmetic on recorded numbers.

## 1. What was built (all default-off; `CLKFIX` bit field, threaded shell/axi/fpga/hbm_xo/tile/core and benches; `R42_CLKFIX=<mask>` Make variable)

| bit | module | change | cycles at the bit |
|---|---|---|---|
| 0 | `tnca_lsu_line` (hybrid LSU line path) | `PICK_REG`: the rotating completion pick over the done slots is registered (skips the slot being completed, starts behind it); +1 cycle on every fast-path completion | +1 per fast completion |
| 1 | `tnca_home_pipe` | `FREE_WR`: the MSHR address array is written into the first free entry every cycle (enable = MSHR states only) instead of by the S2 allocation decision | 0 (cycle-exact, miter) |
| 2 | `tnca_core` + `tnca_vrf_banked` | `W_PHYS_IN`: the writeback's physical vector register (base + index) is registered with `w_*` (like `w_bank_q`) and the VRF write decode starts from it | 0 (core-run2 identical) |
| 3 | not built | | |
| 4 | `tnca_home_pipe` | `S2SPLIT`: the S2 decision cone (tag compare, victim choice, MSHR CAM) is registered, S2 acts one cycle later; same-set requests behind an install/allocation are replayed (as S1 stale already is); a registered flag invalidates an MSHR match whose MSHR the fill unit freed | +1 per slice request |
| 5 | `tnca_home_pipe` | `ARB_SAME`: the arbiter's pairwise same-line compare is a registered matrix written at request acceptance | 0 (cycle-exact, miter) |
| 6 | `tnca_lsu` (word path) | `FREE_LOAD`: lane address/write-data/read-data arrays load in every idle cycle (enable = state only) | 0 |
| 7 | `tnca_multicore_mmio` | `IDLE_REG`: `all_idle` (every core quiescent, ANDed across the SLRs) is registered before it gates the doorbell and the command completion | +1 on admit/complete |

(Table is completed with measurements below.)
