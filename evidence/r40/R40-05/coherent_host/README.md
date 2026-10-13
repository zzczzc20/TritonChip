# Coherent host word interface

`tnca_top` exposes an opt-in HOST_ACCESS interface on GLOBAL_GRID profiles.
Host words pass through tile 0's network interface to the unique home NUCA,
so reads see dirty cached data and masked writes preserve untouched bytes.
The mux reserves ownership even before a backpressured request is accepted,
holds it through the matching response, and keeps global launch admission
closed while ownership is held. Host requests are refused during a grid.
Legacy profiles keep the port disabled.

`make r40-coherent-host` checks the mux's core/host field routing, forbidden
access, reserved request under backpressure, matching response tags and reset.
Its integrated three-core test executes both 455-PID grids, then downloads
all 455 words, performs a full-word upload/readback and a two-byte masked
upload/readback. Every outstanding host transaction is challenged with a
new grid request. All 457 downloads and 2 uploads passed. No host access
uses testbench hierarchical memory writes or reads.

This is functional RTL evidence for the coherent word port. The public
persistent transport, instruction upload/fetch binding, bulk DMA coherence,
role scheduling, fault/deadline recovery and board integration remain required.
It is not a native Torch, board or complete R40-05/R40-06 acceptance claim.
