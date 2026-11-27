# R42 shrink: what was built and how it was verified

Branch `R42-shrink`. Every new behaviour is behind a parameter whose default keeps the shipped RTL: default `make core-run2` = 90 PASS with all 88 reported cycle counts identical to the area worker's
default run (`verify/core_run2_default.log`, compared line by line against `evidence/r42/area/verify/core_run2_base.log`); the default tile synthesizes with the same unit sizes (CENSUS / OOC tables).

## Files

| File | Change |
|---|---|
| `rtl/nuca/tnca_home_pipe.v` | `DATA_RAM` (0 distributed RAM, shipped; 1 RAMB; 2 URAM288) for the 256 x 512 line array; `AUX_RAM` likewise for the MSHR line buffer `ms_line` (64 x 512) and the target write data `tg_wdata` (128 x 512). Synchronous reads issued one stage early, no added cycle; a registered write bypass after the RAM output for the one case the shipped design handled by reading late (below). |
| `rtl/core/tnca_rf_bank.v`, `rtl/core/tnca_vrf_banked.v` | `XBAR` (0 shipped; 1 AND-OR over registered one-hot selects, every output identical; 2 bank-number mux, data of a port with no response is not zero, every consumer in `tnca_core` uses data only under valid) and `VLD_RAM` (valid bits in the bank RAM with a post-reset sweep: unit-level only, breaks the core, see failures). |
| `rtl/core/tnca_core.v`, `rtl/tile/tnca_tile.v`, `rtl/tnca_top.v`, `rtl/tnca_multicore_shell.v`, `rtl/fpga/*`, `rtl/board/tnca_multicore_hbm_xo.v` | `SHRINK` bit field plumbed from the image parameters: [1:0] home `DATA_RAM`, [3:2] home `AUX_RAM`, [5:4] `VRF_XBAR`, [6] `VRF_VLD_RAM`. Default 0. |
| `tb/core/tb_tnca_core.v`, `Makefile` | parameters `VRF_XBAR`, `VRF_VLD_RAM`; `make core-run2 R42_VRF_XBAR=2` |
| `tb/nuca/tb_home_pipe*.v` | parameters `DATA_RAM`, `AUX_RAM` |
| `tools/r42/shrink/` | OOC wrapper generator and wrappers, job and queue scripts (own slot counting, memory gate, stop file, no pgrep), cell census, miters, mutation check, floorplan arithmetic, TLM configuration generator and evaluation |

## Verification (sim)

1. **Unit miter, `tb_home_pipe_miter`** (`tools/r42/shrink/sim/`): two `tnca_home_pipe` (shipped and DATA_RAM/AUX_RAM variant), identical random inputs (reads, byte-enabled and full-line stores, atomics, TMU
   register accesses, maintenance ranges, random back-pressure and out-of-order random-latency memory), every output of both compared in every cycle (`!==`). It found a real hazard the first RTL had (a victim line read
   by S2 in the cycle S3 merges a store into it returns the old line; fixed by the registered bypass). Results: see `verify/miter_*` (matrix of 18 structural/traffic configurations x DATA_RAM/AUX_RAM combinations).
2. **Unit miter, `tb_vrf_miter`**: shipped `tnca_vrf_banked` against `XBAR` 1 and 2 (and `VLD_RAM`), five requesters, two writers with lane masks, random `rsp_ready`.
3. **Mutation check** (`mutation_check.py`): single-point mutants of the new logic, each must be killed by the matching miter.
4. **Core bench**: `make core-run2` (90 program runs, in-order and reordered memory) with `R42_VRF_XBAR=2`: 90 PASS, all cycle counts identical to the default (`verify/core_run2_xbar2.log`).
5. **R41 home-pipe benches** (`tb_home_pipe`, `tb_home_pipe_special`) with DATA_RAM/AUX_RAM 1 and 2: PASS, finish times identical to the default (`verify/home_pipe_r41_tests.txt`).
