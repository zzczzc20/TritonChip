# Ownership/code-upload candidate package — not linked or accepted

Immutable source archive: `025cf62` (`rtl`, `fpga`, `flow`, `spec` plus the
image-profile generator). FPGA host root `~/tnca_build/u280_r40ownership`.
Vivado 2023.2, U280 part xcu280-fsvh2892-2L-e, multicore-hbm variant.

XO packaging completed. SHA256:
`b40a773e5984513f6bd2e5dfbbf2b45854ba74321e87f8941a503d617df9caa8`.
`package.log` preserves the complete packaging output. `image.json` records
the explicit profile-derived parameters, including CACHE_MAINTENANCE=1 and
two-stage code/NoC/dispatch/home/outlet links. The package is retained on the
FPGA host; it can be rebuilt from this source and configuration.

A separate OOC synthesis with general.maxThreads=2 is running to obtain a
new DCP for actual ownership-controller/pipeline hierarchy and resource checks.
There is no OOC, placement, route, timing/CDC, board or performance PASS here.
The two older immutable full-link candidates continue independently.
