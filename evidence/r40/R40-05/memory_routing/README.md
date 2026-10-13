# Global-dispatch memory routing and visibility

The topology now reaches the core and slice network interfaces. The frozen
2x2 home hash remains unchanged; other meshes use a bounded byte hash of all
physical line-address bytes, reduced to the configured slice count. Requester
IDs decode to x/y using the actual mesh width. The external memory source field
keeps its legacy two bits for up to four slices and expands for larger meshes.

Non-2x2 slices retain all physical line bits in their cache tags. They cannot
use the old omitted-home-bit reconstruction, which depended on the frozen
invertible 2x2 hash. The preserved failing two-grid run demonstrates the error:
address 0x10000 expected 0x1000 but held 0x1018 (program 24's value) after dirty
writeback. The corrected test checks all 455 words with a second read-only grid,
including dirty evictions and cross-launch cache visibility.

The memory arbiter now holds the selected requester until handshake under
backpressure. A dedicated bench adds a higher-priority requester while the
output is stalled, checks every source/tag response, and rejects invalid source
aliases. Running that bench on the old arbiter fails, as preserved here.

These are RTL functional diagnostics, not native Torch, routed SLR or performance
acceptance. Persistent transport, host-copy coherence, role scheduling and board
integration remain required for R40-05/R40-06. One shared external memory outlet
remains; this does not satisfy the multi-outlet R40-06 requirement.

Validation: all 20 configurations passed across the matrix run and its final
8-core retry. The initial runner had a 180-second process budget; its final
8-core two-grid case expired at that host limit, while the other 19 passed.
The retry retained the 100000 simulation-cycle watchdog and every assertion,
with a 600-second host process budget. Eight cores completed the write grid at
cycle 7014 and read-only verification at cycle 12761, all 455 PIDs per grid.
The runner now uses that longer wall-clock budget for real-memory cases.
The full old 2x2 system (4 cases), NUCA (9 cases) and parameter propagation
(6 configurations) also passed. The logs retain the timeout and final retry.
