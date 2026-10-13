# Bounded global dispatcher, real core integration

The opt-in `tnca_top.GLOBAL_GRID=1` profile dispatches one logical grid to
physical tiles. Each tile has one pending queue entry and one accepted
program; storage does not grow with grid size. Round-robin allocation advances
x/y/z on queue reservation. Ready/valid metadata remains stable until acceptance
or explicit fault cancellation. Arguments are isolated by core index and stride.

Fault notification stops new issue immediately, cancels pending commands, and
waits for accepted programs to reach core idle. Completion retains launch ID,
status, elapsed cycles and per-core issued/completed counts until acknowledged.
Reset discards dispatcher state and must accompany core reset. The caller supplies
per-core argument tables with shared tensor pointers and private scratch pointers;
this hardware does not manufacture argument tables or provide a memory fence.

`python3 tools/r40/grid_dispatch_tests.py` covers 1/2/3/4/8 cores: 1D/3D grids,
6145 programs, empty/invalid/overflow grids, masked cores, mixed latencies,
backpressure, all-core fault injection before completion, drain, and relaunch.
The second bench runs actual core instruction pipelines for all 455 three-dimensional
PIDs, checks their received coordinates/extents/private argument addresses, and
requires every core to execute and drain. Its program is EXIT; this is functional
RTL integration evidence, not standard Torch numeric or routed board evidence.
Legacy GLOBAL_GRID=0 retains the old broadcast launch and tile ID semantics.

Public runtime/multicore backend, compiler role-SPMD compatibility, memory traffic
with 8-core requester tags, pipelined SLR links, physical placement and board
performance remain required. This diagnostic does not mark R40-05 accepted.
