# R42 clk: path-family clock fixes (CLKFIX bits 0-7), draft in progress

Status line (updated as results arrive): see the "Results so far" section; nothing here is a board number. Domains: sim = RTL simulation (iverilog, GB10), OOC = out-of-context Vivado on the x86 host, routed = after route_design, TLM = SystemC model, derived = arithmetic on recorded numbers.

## 1. What was built (all default-off; `CLKFIX` bit field, threaded shell/axi/fpga/hbm_xo/tile/core and benches; `R42_CLKFIX=<mask>` Make variable)

| bit | module | change | cycles at the bit |
|---|---|---|---|
| 0 | `tnca_lsu_line` (hybrid LSU line path) | `PICK_REG`: the rotating completion pick over the done slots is registered (skips the slot being completed, starts behind it); +1 cycle on every fast-path completion | +1 per fast completion |
| 1 | `tnca_home_pipe` | `FREE_WR`: the MSHR address array is written into the first free entry every cycle (enable = MSHR states only) instead of by the S2 allocation decision | 0 (cycle-exact, miter) |
| 2 | `tnca_core` + `tnca_vrf_banked` | `W_PHYS_IN`: the writeback's physical vector register (base + index) is registered with `w_*` (like `w_bank_q`) and the VRF write decode starts from it | 0 (core-run2 identical) |
| 3 | `tnca_core` | private copies of the writeback registers (select, context, index, data) for the scalar and the predicate register file write ports (`dont_touch`) | 0 (core-run2 identical) |
| 4 | `tnca_home_pipe` | `S2SPLIT`: the S2 decision cone (tag compare, victim choice, MSHR CAM) is registered, S2 acts one cycle later; same-set requests behind an install/allocation are replayed (as S1 stale already is); a registered flag invalidates an MSHR match whose MSHR the fill unit freed | +1 per slice request |
| 5 | `tnca_home_pipe` | `ARB_SAME`: the arbiter's pairwise same-line compare is a registered matrix written at request acceptance | 0 (cycle-exact, miter) |
| 6 | `tnca_lsu` (word path) | `FREE_LOAD`: lane address/write-data/read-data arrays load in every idle cycle (enable = state only) | 0 |
| 7 | `tnca_multicore_mmio` | `IDLE_REG`: `all_idle` (every core quiescent, ANDed across the SLRs) is registered before it gates the doorbell and the command completion | +1 on admit/complete |

(Table is completed with measurements below.)
| 8 | `tnca_core` + `tnca_scalar_rf` | `TOKR`: the second m.mma token (Sc+1) / first transfer coordinate (Sb+1) is read in R (new read port H) and registered like operands A-C, instead of an asynchronous RAM read in X in front of the barrier-token validity compare -> `x_bar_illegal` -> enables of the barrier, matrix and transfer engines | 0 (core-run2 identical) |

Parameter threading: `CLKFIX` through `tnca_multicore_hbm_xo`, `tnca_multicore_axi`, `tnca_multicore_fpga`, `tnca_multicore_shell`, `tnca_top`, `tnca_tile`, `tnca_core` (and `tnca_home_pipe`, `tnca_lsu_hybrid/pair/lsu/line`, `tnca_vrf_banked`, `tnca_scalar_rf`, `tnca_multicore_mmio` below it); benches `tb_tnca_core`, `tb_core_hybrid`, `tb_lsu_line`, `tb_lsu_hybrid`, `tb_home_pipe(_special)`, `tb_home_pipe_miter`, board-regs bench; `R42_CLKFIX` Make variable; `tools/r42/clk_images.py` writes the image parameter files of the arms (`tools/r42/clk/images/`); `tools/r40/image_geometry_tests.py` and `tools/r41/link_budget_rtl.py` know `CLKFIX`.

## 2. Verification (simulation; GB10)

| gate | command | result |
|---|---|---|
| core-run2 at CLKFIX 0 on the R42-clk head | `make core-run2` | 88 PASS, all 88 rows (cycles, issued, hazard, lsu, wb, squash) identical to origin/R42 (`sim/run2_*.cycles.gz`) |
| core-run2 at CLKFIX 1, 4, 8, 256 | `make core-run2 R42_CLKFIX=...` | 88 PASS each, rows identical to origin/R42 (the non-hybrid LSU has no fast path, so bit 1 is a no-op there) |
| core-g2c4, core-mmadp, unit | `make ...` | 86 PASS (rows identical to origin/R42), 5 PASS, 11 PASS |
| home-pipe miter, A = origin/R42's `tnca_home_pipe` (renamed copy), B = the new module at CLKFIX 0, 2, 32, 34 with data/aux RAM 0:0 and 2:2, 14 stimulus configurations (reads, byte stores, full-line stores, atomics, MMIO, maintenance, hot lines, tiny and 128-set geometries), 150k cycles each, every output compared every cycle (`mem_req_addr` only while valid) | `tools/r42/clk/run_home_miter_clk.sh` (REF=1) | 112 of 112 PASS, 0 mismatches (`sim/miter_ref*_summary.txt.gz`) |
| home-pipe functional regression (`tools/r41/g4_home_pipe_tests.py --quick`: byte-shadow traffic, maintenance, atomics, TMU page, service throughput) at CLKFIX 16 and 18 (S2 split alone, with FREE_WR), fill chain 0 and 1 | `CLKFIX=16 python3 tools/r41/g4_home_pipe_tests.py ...` | 81 of 81 cases each; CLKFIX 50: 81 of 81 |
| hybrid LSU line engine at PICK_REG 1 | `PICK_REG=1 python3 tools/r41/lsu_line_tests.py` | 48 of 48 configurations |
| hybrid core bench (`tb_core_hybrid`, line-wide port, memory latency 6 with random stalls and reordering), 19 programs | `tools/r42/clk/core_hybrid_clk.py` | CLKFIX 2, 3, 6, 7: cycle-identical to CLKFIX 0 on every program; CLKFIX 1 (PICK_REG): all programs pass, cycles differ by -6..+297 (the random stalls move with every latency change) |

Home-slice service throughput (`tb_home_pipe` SERVICE modes, 2048 requests): CLKFIX 0 and 2 identical; CLKFIX 16/50 add +0 to +2 cycles except mode 2 with 8 MSHRs (12,068 -> 12,279, +1.7 %): the S2 split adds latency, not a throughput limit.

## 3. Cycle cost (RTL stand-in and model)

Three-core board-top RTL stand-in at the run-3 image geometry (`tools/r42/clk_mc.sh`, profile `core-vrf-sync-v5-x8-g2c4-dp`), 7 ArchBench tiny points, numerics pass in every arm (`sim/mc7_*.tgz`; `tools/r42/clk/mc_table.py`):

| arm (CLKFIX) | geomean cycles vs base | per point (add, gemv, linear, matmul, rope, softmax, transpose) |
|---|---|---|
| pick (1) | 1.0032 | 1.0034, 0.9981, 1.0007, 0.9995, 1.0009, 1.0144, 1.0056 |
| s2 (16) | 1.0542 | 1.0237, 1.1757, 1.0435, 1.0518, 1.0645, 1.0183, 1.0102 |
| c23 (0,1,2,4) | 1.0557 | 1.0186, 1.1753, 1.0417, 1.0534, 1.0650, 1.0249, 1.0192 |
| all (0-8) | 1.0553 | (as c23 within 0.1 %) |
| idle (128, 3 points) | 1.0001 | 0.9998, 0.9996, 1.0010 |

The TLM (SystemC model, 77 tiny/small/medium points, slice hit latency +1 and slice-to-outlet +1): geomean +1.55 % cycles (tiny +1.98, small +1.24, medium +1.40; gemv +3.6, bmm +4.8). On the tiny stand-in points the RTL cost of the S2 split is 3 to 4 times the model's; the tiny tier is the most latency-bound.

(The OOC results follow in section 4 as they land.)
