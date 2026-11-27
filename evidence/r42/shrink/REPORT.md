# R42 shrink: final report (written by the lead from the worker's final message; the worker's narrative with all tables is `RESULTS.md`, M0 is `CENSUS.md`)

Branch `R42-shrink` (final commit `c6b177ec`, from `R42` `9dc0712b`), merged into R42 (merge commit `3dfe5014`). Worker `shrink`, 2026-11-27/28.
**Domains: OOC Vivado 2023.2 (x86 host), iverilog simulation, SystemC/TLM model (`S145` = predicted speed-up of the 145-point wall-time geometric mean against the R41 three-core run-3 image; the model gives 2.07 for today's hardware, the board measured 2.03), or derived arithmetic. Nothing is a routed image or a board number. No link, card or board load was requested.**

## Built (all behind parameters; defaults keep the shipped RTL bit- and cycle-identical)
- `tnca_home_pipe`: slice line/tag-aux arrays in URAM or BRAM (`DATA_RAM`, `AUX_RAM`), read issued one stage early, registered same-cycle write bypass after the RAM output: unit OOC 38,455 -> 29,214 LUT (-9.2k, 24 URAM).
- `tnca_vrf_banked` / `tnca_rf_bank`: response crossbar `XBAR` 2 (bank-number 4:1 mux): 21,289 -> 13,916 LUT (-7.4k), unit WNS at 6.67 ns +0.79 -> +1.11 ns.
- `VLD_RAM` (valid bits in the bank RAM): passes its unit miter, **deadlocks the core** (the core requires the writeback never to be refused), so it stays off and out of the tile.
- Make variables `R42_VRF_XBAR`, `R42_VRF_VLD_RAM`; parameters threaded through shell/top/tile/core.

## Tile LUT (OOC `tnca_tile`, run-3 parameters, 10 ns)
default 249,921; + area FMA/MAC back end 242,807; + home arrays in URAM 236,905; + VRF crossbar **230,979 (-18,942, -7.6 %)**. Tile totals are good to +-3k (other units drift between runs); the touched instances moved by exactly their unit deltas.

## Verification (sim)
Home-pipe miter 36 runs / 9.0M cycles / 0 mismatches; VRF miter 7 runs / 0 mismatches; `core-run2` default, `XBAR` 1, `XBAR` 2: 90 PASS each with identical cycles; `core-g2c4`, `core-mmadp`, `unit`, R41 home benches pass at defaults; mutation check 14/17 killed (3 survivors in the unintegrated `VLD_RAM`, 1 equivalent).

## What the evidence says
- Memory is a small part of the tile: array LUTRAM 6.7 %, memory-like logic 26-30 %, about 70 % is arithmetic and datapath mux. The shrink gives **no cycle gain by itself** (model 1.000).
- **More cores do not fit.** Two tiles per SLR need a tile of at most 143.2k (4 cores), 140.3k (5), 126.7k (6); built 231k, everything identified plus estimates about 190-200k. Blocking units: matrix lanes 49-51k, 16 FMA 24.7k, converters 15.6k, vector ALU 24k, home slice 29k, LSU 19k, transfer engine 19k, VRF 14k. Halving both lane arrays still leaves about 160k.
- **What the freed area buys (model, full 145-point bundles):** `sets128` 2.170; + `te32`, `le2`, `hl`, `vi1` 2.984 (vi1 about +14k LUT per core, inference); nine clock fixes at +1 cycle each 2.032 (-1.9 %) at 100 MHz, 2.448 at 125 MHz, 2.863 at 150 MHz if timing closes; area levers + 125 MHz fixes 3.46 (77 light points + inheritance). Cores 4/5/6 if the tile fitted: 2.41 / 3.01 / 3.57.
- **Clock:** isolated routed tile at 6.67 ns: default WNS -0.521 (3,488 failing endpoints, worst path 7.19 ns), step 3 WNS -0.066 (150 failing, 6.74 ns); image derate 1.4-1.6, so the shrunk netlist is about 100-105 MHz as it is; 120-125 MHz is plausible after the nine fixes (not built, not shown); no evidence for 150 MHz.

## Not done
Accumulator file, scalar RF, LSU and transfer-engine arrays in RAM (risk without a cycle-exact bench; 4-6k, 3.3k, 0.6k estimates); multi-format K-fused DSP lane (derived -12k to -19k per tile, never built or screened); DSP comparators/adders; all nine clock fixes; registered reset tree; `VLD_RAM` system reset stretch. `default_change_audit.py` not run.
Process: five of the worker's own jobs stopped by explicit PID on the shared x86 host (memory peak); one kill loop was fed by a `ps` listing of the worker's own wrappers (rule broken, only those two wrappers affected); no other user's or worker's process was signalled.
