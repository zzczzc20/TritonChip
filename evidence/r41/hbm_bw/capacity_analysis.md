# Streaming capacity analysis and candidate constraints

## Evidence and accounting

The target of 100 GB/s at 100 MHz requires 1,000 useful bytes per cycle. Effective copy traffic must count source bytes plus destination bytes, each once; read-reduce counts input bytes and its actual output, and write counts destination bytes. AXI traffic and useful traffic must be reported separately: cache fills, write allocation and evictions must not inflate effective bandwidth. Host staging time is a separate metric.

`rtl/core/tnca_mem_mux.v` merges the LSU and tensor transfer engine onto a single request port, with transfer priority. `rtl/core/tnca_core.v` defaults that path to 64 bits, with a single active tensor transfer command. `rtl/fpga/tnca_line_axi.v` accepts complete 64-byte lines but does not widen the core path. The board wrapper exposes three named AXI masters. Existing R40 attribution (`evidence/r40/v3/model_attribution/README.md`) reports unchanged copy time with three outlets versus one, and only 12% model outlet occupancy. These observations rule out treating outlet expansion alone as the solution.

| Option at 100 MHz | Ideal payload ceiling | Assessment |
|---|---:|---|
| Existing three 64-bit core paths | 2.4 GB/s request payload | Cannot reach target |
| Three widened 512-bit core paths | 19.2 GB/s request payload | Cannot reach target, even before shared-engine limits |
| Sixteen 512-bit lanes | 102.4 GB/s in one direction | Requires 97.7% utilization for read or write; insufficient margin |
| Twenty-four 512-bit lanes | 153.6 GB/s in one direction | Requires 65.1% utilization; candidate for investigation |
| Thirty-two 256-bit lanes | 102.4 GB/s in one direction | Requires 97.7% utilization; insufficient margin |
| Thirty-two 512-bit lanes | 204.8 GB/s in one direction | Requires 48.8% utilization; higher routing/area risk |

These are algebraic interface bounds, not simulation predictions. AXI read and write channels are independent; balanced copy can exploit both directions, but a design must also sustain the requested read and write workloads independently. A 512-bit kernel interface still needs a platform width conversion to the HBM controller's 256-bit, 450 MHz interface. Port connectivity, address allocation and actual HBM arbitration must be checked in the linked topology.

## Provisional architecture to evaluate

Investigate 24 independent 512-bit stream lanes, partitioned physically rather than gathered through a central crossbar. Each lane needs a disjoint PC assignment or a demonstrated collision-free mapping. Commands must originate in real core/compiler execution; contiguous transfer acceleration must preserve existing numeric semantics and fall back for unsupported shapes, aliases, atomics and masking. A synthetic AXI exerciser would establish controller capacity only and would not establish TritonChip bandwidth.

At 100 GB/s spread uniformly across 24 lanes, each lane must deliver 41.67 bytes per cycle. For a round-trip latency of L kernel cycles, the bandwidth-delay product is at least ceil(41.67 L / 64) outstanding 64-byte lines per lane. At illustrative latencies of 24, 50 and 100 cycles, that is 16, 33 and 66 lines. The historical eight-line window cannot simply be carried forward. Larger bursts amortize address issue overhead, but need independent ordering, 4 KiB boundary handling and a measured latency distribution.

SLR0 is already 96.7% packed per the handoff. Stream data and state must be distributed into SLR1/2 with registered command and completion crossings. No numerical area estimate is credible before synthesis of a lane and a replicated group. Wide return multiplexers, global reduction, centralized outstanding allocation and ownership maintenance are timing risks. Read-reduce additionally needs enough arithmetic throughput and an explicitly specified reduction order; changing floating-point associativity would violate the existing numeric contract unless already permitted by that instruction.

## Required next evidence

Use the actual SystemC model to compare baseline versus independent lane service, then the physical-map Ramulator path to expose PC collisions, read/write turnaround and controller queue limits. The existing model is uncalibrated and cannot qualify an FPGA bandwidth claim. RTL regression must exercise independent backpressure, reordered replies, exact byte counts, partial tails, overlaps, faults and cache ownership handoff. Free unit and tile screens must quantify area and 10 ns timing before a full link. Only board manifest records with image UUID, clock, correctness and counter windows can establish the target.
