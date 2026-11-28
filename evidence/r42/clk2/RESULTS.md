# R42 clk2: clock-pipelining fixes for the matrix engine, vector FPU, transfer engine and barrier (worker `clk2`, branch `R42-clk2`)

Status: DRAFT written while the first tile screen (T1) runs; the final text is the last message of the worker. Domains: **sim** = iverilog on GB10 (`make core-run2`), **OOC** = Vivado 2023.2 out-of-context synthesis, place and route on the x86 host (xcu280, one tile pinned to SLR0, no shell, no other tile), **TLM** = the SystemC model, **derived** = arithmetic on recorded figures. No link, no card, no board number.

## 1. What exists (all behind `CLKFIX2`, default 0 = the shipped RTL cycle for cycle)

`CLKFIX2` is threaded through shell, axi, fpga, hbm_xo, top, tile and core (and the benches, Make variable `R42_CLKFIX2`, profile entries `core-r42-clk2c` (control, FMA_IMPL 465) and `core-r42-clk2` (CLKFIX2 255, FMA_IMPL 465), image arms `tools/r42/clk2_images.py`).

| bit | unit | change | cycles |
|---|---|---|---|
| 0 | matrix | MAC operands (a, b, c, format controls) registered once more before the lane (`tnca_matrix` OPREG); the command payload registers (r_vdata, r_scale, r_mask) copy their inputs while the engine is idle, so their enable is the state register instead of the core's issue decision (CAPFREE) | +1 per command |
| 1 | matrix | MAC lane cuts inside S1 (S1A decode | register | S1B multipliers: new XLS stage functions `mac_s1a`, `mac_s1b`, `rtl/gen/tnca_mac_s1{a,b}_xls.v`) and after S2 (`MAC_CUTS` bits 6 and 1) | +2 per command |
| 2 | matrix | lane results registered before the accumulator file's write port, a.vwrite's direct write delayed with them (WREG); the valid-bit enables of the file decoded a cycle early (`tnca_acc_rf` WPRE) | +1 per command |
| 3 | vector FPU | FMA cut behind the multiplier (`tnca_fmax_fma` CUTS 4'b1011) | +1 FP latency |
| 4 | vector FPU | FMA cut behind the adders (CUTS \| 4'b0100) | +1 FP latency |
| 5 | vector FPU | register behind the FMA result (FOREG); with bits 3 and 4 also the XLS-pipelined float-to-float converter (`tnca_cvt_xls`) and the two-stage int<->binary32 converter (`tnca_cvt_int_p`) (CVTP); FP latency 2 -> 5 | +1 FP latency |
| 6 | transfer engine | the whole-box-in-tensor test registered (`g_inside_q`), the padding fault raised from S_TMUL's last cycle; the store prefetch's vector-file request no longer waits for the odometer's `l_more` (`TCHK_REG`) | 0 (+1 on a faulting transfer) |
| 7 | vector FPU | SFU operand select registered (`tnca_sfu` XREG) | +1 per SFU pass |

Not built, with the evidence why: **barrier** (the unit screen of `tnca_barrier` alone at 5.6 ns is WNS +1.88 ns, 6 levels, 3.7 ns; all 1,469 failing endpoints in the tile start at `u_srf/valid_reg` and arrive through the token compare `x_bar_token_bad` -> `x_is_bar`, which is core control, sent to `clk`); **generic reset-tree replication** (no failing endpoint in either 5.6 ns tile screen starts at a reset: counted 0 of 44,309 / 31,203).

## 2. Census of the baseline tile at 5.6 ns (OOC, routed)

`census_tile_p56.md`, `census_families_s3_p56.md` (from `~/r42_clk_out/tscr_s{0,3}_p56`): shrunk tile s3 31,203 failing endpoints, 11,733 of them with a path above 6.2 ns, 2,497 above 6.67 ns (default tile s0: 44,309 / 19,691 / 6,524).
My four units in s3 (failing at 5.6 / path > 6.2 / path > 6.67): matrix 4,342 / 1,476 / 326, vector FPU 1,976 / 1,505 / 551, transfer engine 3,318 / 731 / 119, barrier 1,469 / 688 / 65.

## 3. Unit screen against tile screen

Matrix engine alone (registered I/O, 5.6 ns, default RTL, s3 parameters): WNS -0.247 ns, 153 of 49,656 endpoints failing; the same logic in the tile screen: -1.367 ns, 4,342 failing. Longest stage paths unit vs tile: S1 5.72 / 6.95 ns, S2+F2 5.83 / 6.92, B1 5.31 / 5.85. A tile with 31k failing endpoints costs every stage about 1.1-1.2 ns: unit screens are not predictors of tile closure and the tile screen is the measure.

(Tile screen results T1 and later, gates and the stand-in cycle table follow in the final text.)
