# R40-05 core parameter propagation checkpoint — not accepted

The mesh and tile wrappers now forward the banked VRF, bypass, fetch,
prefetch, context switch, LSU concurrency, FPU overlap, matrix A buffering,
resource slicing, role pairs, system atomics and tensor transpose parameters
to each elaborated core. Every legacy default is unchanged; the frozen
system-2x2 SFU stays off. An odd final tile has no paired slowdown peer,
rather than reading a counter outside the vector.

An Icarus elaboration inspection verifies actual core instance parameters
for legacy four-core and selected v5 one/two/three/four/eight-core geometries,
6/6. The legacy mesh and atomic programs pass single/all-role execution,
4/4, with cycles 18591/29075 and 1488/2829. Sources and logs are hashed.

This provides parameter plumbing only. It does not enable a new public
multicore profile, supply a persistent native system backend/global grid
dispatcher, change the currently fixed launch slicing, or prove physical
multicore Torch execution or board timing. Those remain required for R40-05.
