# End-to-end operator acceptance audit

The governing requirements are HANDOFF section 7. The memory-system design target is 100 GB/s; acceptance requires several-times standard torch operator speedups on the board with unchanged numerics. No candidate board speedup has been measured. A stream-only image is not ready for a full-run allocation.

## Current path and missing proof

| Level | Existing evidence | Required next proof |
| --- | --- | --- |
| HBM controller | Bounded-admission Ramulator service sweep in model/; ideal independent-PC scaling only | Concurrent controller service under actual operator address, read/write mix and downstream backpressure |
| AXI outlets | Legacy three 256-bit outlets: 9.6 GB/s per direction at 100 MHz before overhead; compact 32-lane stream bank routed alone | Counters for accepted beats, outstanding occupancy, response latency and stalls with the complete candidate |
| NoC/NI | tnca_core_ni has fixed 64-bit data, 8-bit byte enables; tnca_noc_pkg has a 216-bit payload but only 64-bit request/response data | Coordinated wide request/response protocol, tags, masks, atomics and backpressure validation; accepted useful bytes/cycle, not packet width |
| NUCA/cache | Existing word service is retained by the stream candidate | Full-line/coalesced service throughput, hit/miss/writeback contention and downstream absorption in TLM and RTL |
| LSU | Three 64-bit core paths have a combined optimistic 2.4 GB/s per direction at 100 MHz; LSU width parameters alone do not widen the fixed NI | Coalescing efficiency, response landing capacity, sustained issue and stall attribution with real compiled traces |
| Vector RF / execution | Banked RF permits one read per bank per cycle; collisions and writer arbitration matter. SFU serializes VLEN lanes through one slice | RF conflicts/writeback and SFU occupancy per operator; show these can absorb the widened LSU traffic |
| Standard torch | Native 32 MiB copy dispatch software test passes against a register/byte-memory fixture | Manifest-bound board add/mul/silu/gelu/copy and all other memory-bound points, numerical checks and direct baseline/candidate speed ratios |

These are ceilings and outstanding proof obligations, not throughput guarantees. The NoC packet width must not be confused with useful data width. Increasing only LSU parameters or independent AXI lanes leaves the standard operator path constrained. The stream CU bypasses NoC/NUCA/LSU/RF and therefore supplies only a copy subtask and a memory diagnostic; it cannot establish the requested end-to-end guarantee.

## Baseline coverage

operator_baseline.json preserves all 102 tnca100 roofline points, including 64 historically classified mem_bytes and 38 fp_lane. Candidate time, speedup and numerical verdict remain null. The original board suite reports 113 points; the remaining eleven and explicit mul/gelu/copy coverage must be recovered from raw board records before an acceptance manifest is considered complete. The roofline classification is a starting list, not permission to omit an author-required operator or evidence that a point remains memory-bound after widening.

For each matched shape and dtype, report baseline device seconds / candidate device seconds and separately report standard torch wall time. Keep compilation and host transfers out of device-time comparisons, retain their wall-time cost, and use the existing manifest identity/clock guards. No new numerical tolerance or arithmetic substitution is authorized. Compute-bound normal GEMM speedup is informational.

## Current core synthesis

The free kernel synthesis completed successfully on acf3019 under ~/r41_hbmbw_kernel_v1, without a card open or a full image build. Raw reports are in kernel/: 819,109 LUTs, 242,714 FFs, 192 RAMB36, 45 RAMB18 and 520 DSPs. Synthesis timing reports WNS +4.032 ns and WHS +0.055 ns. This is neither routed timing nor a whole-image fit. It constrains the area available for widening the actual operator path; the historical loaded image checkpoint must not substitute for this source.

## Next implementation gate

Use compiled operator traces and existing board/model counters to split LSU issue, home/cache service, NoC backpressure, RF conflicts and SFU time. Extend the SystemC/TLM experiments to include these coupled limits before selecting a wide-line protocol. Verify every widened boundary and its receiving stage in RTL, then obtain tile/combined placement screens. Keep the copy integration as a tested partial path. Do not spend a full build solely to demonstrate stream-copy bandwidth.
