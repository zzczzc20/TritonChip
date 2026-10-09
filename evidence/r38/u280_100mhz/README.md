# U280 at 100 MHz, undivided (DEV_DIV = 1), 2026-10-09

The device on the U280 running at the kernel clock: 100 MHz, no divider.

## The image

- **RTL:** r38-dco `8d8e4e9` (timing-001..011, `FP_LAT = 2`), built with
  `DEV_DIV = 1` and a 100 MHz kernel clock.
- **Link:** `u280_nd100c`, started 2026-10-09 08:19 HKT, finished 12:05.
- **Routed timing** (`hw_bb_locked_timing_summary_routed.rpt`):
  - Design WNS +0.016 ns, 0 of 566,734 endpoints failing. This is the platform's own near-zero path.
  - With DEV_DIV = 1 the device is in the kernel clock domain: ap_clk WNS
    **+0.127 ns** against 10 ns, 0 of 149,372 failing.
  - The worst path is LSU `pending` → `u_mem` BRAM data input: 9.469 ns, 14 levels, 87 % routing.
- **Frequency:** v++ printed only AUTO-FREQ-SCALING-08 (timing closed with slack, requested frequencies kept). `xclbin_clocks.txt` reads back DATA_CLK = 100 MHz.
- **sha256** of `tnca_u280.xclbin`: `09cbd3d4992a07330b6f0567cb7f091682311adc3bf212479b4695d1a98323fb`. xclbin uuid `a446256a-e3be-6912-0701-117f82c9a70a`.

What made the difference from the first undivided attempt was routing
congestion. Before timing-011, SLR1's CLBs were 97.71 % used and the router
spent five hours on its first iteration without finishing. With the
accumulator file in distributed RAM, SLR1 was at 85.41 % (LUTs 62.36 %,
registers 15.92 %) and routing converged in about an hour
(`doc/r38_board_bringup.md` sec 6.12).

## The clock, measured

- The shell's frequency counter (`clock_counter_freqs` in sysfs) reads 100 MHz.
- `flow/fpga/measure_clock.py`, from two spin kernels: **100.067 MHz and 100.009 MHz** (`clock.txt`).

## The runs

Run by `boardrun.sh`; `program.sh --go` passed every gate (FPGA 31 °C, HBM 26 °C).

| File | Step | Result |
|---|---|---|
| `program_go.log` | load | `xbutil program` succeeded; xclbin `a446256a-…` |
| `b1_conformance.json` | B1, `tools/r36/conformance.py --require-measured` | PASS |
| `b2_profiler.log` | B2, `tools/r38/profiler_tests.py --quick` | PASS |
| `b3_tiny.log` | B3, ArchBench tiny on `fpga-u280` | 27/27 pass |
| `b4_compare.txt` | B4, `tools/r38/board_vs_sim.py` | 27/27 points with identical device cycles |
| `clock.txt` | `measure_clock.py` | 100.07 / 100.01 MHz |

The B4 reference is ArchBench tiny on `fpga-u50-sim` of the r38-dco
`1eb6b94` RTL, generated on the build host. timing-010 changes cycles on the
banked register file, so the 75 MHz run's reference would not do.
timing-011 changes none (the R2 cycle check passes 149/149), and the
comparison above shows it.
