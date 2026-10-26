# R40-06 independent memory outlets: functional validation

Parent: 71397f9. Remote snapshots: ~/tritonchip-r40-ports-full and ~/tritonchip-r40-ports-final on the jump host. Tests use the locked Torch/Triton environment. This batch adds independently arbitrated home-to-controller ports before serialization, full source/tag returns, sticky protocol faults, public capability negotiation and hardware traffic counters.

- Mixed read/write protocol matrix: 15 combinations of 1/2/3/4/8 homes and 1/2/3/4/8 outlets; stalled payload stability, independent progress, simultaneous replies, fairness and malformed-source rejection pass.
- Real-core matrix: (cores,ports) = (1,1),(2,2),(3,2),(3,3),(4,4),(8,3),(8,8). Each runs two 455-program 3-D memory phases, exact-once completion, dirty eviction, 457 coherent host downloads and two masked host writes. All pass.
- Native resident Torch graph: (a+b)*0.5 on 4097 elements, 257 logical programs, three physical cores. Two-port reads 623/293 and writes 174/87; three-port reads 374/289/265 and writes 89/86/88. Each outlet has actual read/write traffic; peak physical concurrency is three; measured resident calls have zero tensor staging and one whole-grid launch. Numerical outputs pass.
- Legacy registry, conformance and 20 dispatcher configurations pass. Twelve option/capability/refusal tests pass.
- Final single-port malformed response test passes: invalid source latches fault, and the subsequent launch completes with fault status and zero issued work. This final change only affects the single-port generate branch; the tested native two/three-port implementation is unchanged. The CI matrix enables this injection in its existing one-core case; its two-phase and injection components were both executed successfully.

The controller assignment is home modulo port count, preserving unique home ownership and avoiding response collisions. Simulator controllers each expose the logical address space; their allocation is not physical memory capacity evidence. Counters measure accepted 64-byte line requests, not board HBM bytes. A lost response may require platform recovery: accepted work is never declared drained without completing.

R40-06 remains open: real HBM integration/performance, locality/DCO, lifetime/alias hints, resource/clock and bandwidth accounting, and the full acceptance matrix are still required. This evidence closes only the functional multi-outlet implementation subtask, not the work package or R40.
