# R40-05 large-grid transport checkpoint — not accepted

The R40 extension is additive to the unchanged R2 transport. Default
admission remains 4096 programs and opcode-3 packets remain identical.
Session(large_grid=True), or TNCA_LARGE_GRID=1, queries opcode 8 before
admitting larger work. A supported image advertises 32-bit axes and a
16777216-program streaming budget. Opcode 9 carries all six 32-bit values;
an old image refuses the query, and an unsupported wide launch cannot
silently truncate. Invalid capability replies poison the session before
argument validation/allocation. spec/r40/grid_transport_extension.json
records the public contract.

Unit tests pass 5/5. Both sim-file and the actual MMIO/physical-core RTL
stand-in preserve legacy values, values above 65535, and unsigned high bits.
The runtime then executes 6144-program 1-D and 3-D grids through the same
scalar diagnostic artifact with role-SPMD: every slot contains its distinct
linear ID, and exactly 6144 atomic increments prove no duplicated execution.
Default admission refuses before any device request. RTL tests pass 10/10.
These IR-builder diagnostics are not standard Torch performance evidence,
physical multicore delivery or R40-05 acceptance.

Legacy grid tests pass 2/2; runtime tests pass 7/7. The first FPGA regression
passes cases 01–07 and catches a stale test that used newly assigned opcode
9 as an unknown operation. The updated test uses reserved opcode 15 and
also verifies GRID_CAP; case 08 then passes. Both attempts are retained.
The currently loaded ac580b0 board image predates this extension; a new
image must pass timing/CDC before it can be used for large board graphs.
