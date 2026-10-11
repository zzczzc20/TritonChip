# U280 at 75 MHz (DEV_DIV = 2), 2026-10-09

The first board run of the device at /2: kernel clock 150 MHz, device clock 75 MHz.

## The image

- **RTL:** r38-dco at timing-003 (`f307eef`), built with `DEV_DIV = 2`.
- **Link:** started 2026-10-09 00:45 HKT, finished 05:41.
- **Routed timing** (`hw_bb_locked_timing_summary_routed.rpt`):
  - Design WNS +0.016 ns, 0 of 916,016 endpoints failing. This is the platform's own near-zero path, the same as in every U280 link.
  - clk_dev WNS **+0.471 ns** against 13.333 ns, 0 of 498,254 failing.
  - The worst clk_dev path is `x_pm_idx` → `w_vdata`: 12.681 ns, 28 levels, 80 % routing.
- **Frequency:** v++ printed only AUTO-FREQ-SCALING-08 (timing closed with slack, requested frequencies kept). `xclbin_clocks.txt` reads back DATA_CLK = 150 MHz.
- **sha256** of `tnca_u280.xclbin`: `16813fa9075f9192f9e3b19f97d9a68185fc13dca395885e8efcaa8c98cc3cee`.

## The clock, measured

- The shell's frequency counter (`clock_counter_freqs` in sysfs on the management function) reads 150 MHz for the kernel clock.
- `flow/fpga/measure_clock.py` times two spin kernels on the host and divides the difference in device cycles by the difference in wall time: **74.965 MHz and 75.017 MHz**.

## The runs

Run by `boardrun.sh`; `program.sh --go` passed every gate (FPGA 31 °C, HBM 26 °C).

| File | Step | Result |
|---|---|---|
| `program_go.log` | load | `xbutil program` succeeded; xclbin `dacad4d7-…` |
| `b1_conformance.json` | B1, `tools/r36/conformance.py --require-measured` | PASS: oracle, a second launch, a fault and recovery |
| `b2_profiler.log` | B2, `tools/r38/profiler_tests.py --quick` | PASS |
| `b3_tiny.log` | B3, ArchBench tiny on `fpga-u280` | 27/27 pass |
| `b4_compare.txt` | B4, `tools/r38/board_vs_sim.py` | 27/27 points with identical device cycles to the RTL simulation (`fpga-u50-sim`) |
