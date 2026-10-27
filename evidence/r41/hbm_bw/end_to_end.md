# End-to-end operator acceptance audit

The governing requirements are HANDOFF section 7. The memory-system design target is 100 GB/s; acceptance requires several-times standard torch operator speedups on the board with unchanged numerics. No candidate board speedup has been measured. A stream-only image is not ready for a full-run allocation.

## Current path and missing proof

| Level | Existing evidence | Required next proof |
| --- | --- | --- |
| HBM controller | Bounded-admission Ramulator service sweep in model/; ideal independent-PC scaling only | Concurrent controller service under actual operator address, read/write mix and downstream backpressure |
| AXI outlets | Legacy three 256-bit outlets: 9.6 GB/s per direction at 100 MHz before overhead; compact 32-lane stream bank routed alone | Counters for accepted beats, outstanding occupancy, response latency and stalls with the complete candidate |
| NoC/NI | Opt-in hybrid mode now carries 512-bit data and 64 byte enables in a 720-bit payload. Coupled one/three-core functional tests pass through the real mesh | Accepted useful bytes/cycle, tag occupancy and backpressure attribution under compiled operator traffic |
| NUCA/cache | Wide cache word supports positioned masks, scalar atomics and TMU accesses; unit routed SLR1 WNS +1.774 ns / WHS +0.007 ns; coupled full-vector tests pass | Full-line service throughput, hit/miss/writeback contention and downstream absorption in TLM and RTL |
| LSU | Opt-in hybrid has eight compact vector landing slots plus narrow fallback, with an R-to-X command landing register; one/three-core full-vector roundtrips pass | Coalescing efficiency, sustained issue and stall attribution with real compiled traces; measure the command register and completion bottlenecks |
| Vector RF / execution | Banked RF permits one read per bank per cycle; collisions and writer arbitration matter. Default SFU serializes VLEN lanes; the opt-in four-slice SFU passes exact-bit tests and a routed unit screen | RF conflicts/writeback and SFU occupancy per operator; show these can absorb the widened LSU traffic |
| Standard torch | Native 32 MiB copy dispatch software test passes against a register/byte-memory fixture | Manifest-bound board add/mul/silu/gelu/copy and all other memory-bound points, numerical checks and direct baseline/candidate speed ratios |

These are ceilings and outstanding proof obligations, not throughput guarantees. The NoC packet width must not be confused with useful data width. Increasing only LSU parameters or independent AXI lanes leaves the standard operator path constrained. The stream CU bypasses NoC/NUCA/LSU/RF and therefore supplies only a copy subtask and a memory diagnostic; it cannot establish the requested end-to-end guarantee.

## Baseline coverage

operator_baseline.json preserves all 102 tnca100 roofline points, including 64 historically classified mem_bytes and 38 fp_lane. Candidate time, speedup and numerical verdict remain null. The original board suite reports 113 points; the remaining eleven and explicit mul/gelu/copy coverage must be recovered from raw board records before an acceptance manifest is considered complete. The roofline classification is a starting list, not permission to omit an author-required operator or evidence that a point remains memory-bound after widening.

For each matched shape and dtype, report baseline device seconds / candidate device seconds and separately report standard torch wall time. Keep compilation and host transfers out of device-time comparisons, retain their wall-time cost, and use the existing manifest identity/clock guards. No new numerical tolerance or arithmetic substitution is authorized. Compute-bound normal GEMM speedup is informational.

## Current core synthesis

The free kernel synthesis completed successfully on acf3019 under ~/r41_hbmbw_kernel_v1, without a card open or a full image build. Raw reports are in kernel/: 819,109 LUTs, 242,714 FFs, 192 RAMB36, 45 RAMB18 and 520 DSPs. Synthesis timing reports WNS +4.032 ns and WHS +0.055 ns. This is neither routed timing nor a whole-image fit. It constrains the area available for widening the actual operator path; the historical loaded image checkpoint must not substitute for this source.

## Next implementation gate

Use compiled operator traces and existing board/model counters to split LSU issue, home/cache service, NoC backpressure, RF conflicts and SFU time. Extend the SystemC/TLM experiments to include these coupled limits before selecting a wide-line protocol. Verify every widened boundary and its receiving stage in RTL, then obtain tile/combined placement screens. Keep the copy integration as a tested partial path. Do not spend a full build solely to demonstrate stream-copy bandwidth.

## Integrated wide-path bounds and remaining architecture gate

The diagnostic candidate now widens all core-to-home boundaries together. Three one-request-per-cycle 512-bit core ports have an optimistic aggregate ceiling of 19.2 GB/s per direction at 100 MHz, before command admission, RF, cache or network stalls. The retained three 256-bit AXI outlets have only 9.6 GB/s per direction before protocol overhead. These are upper bounds, not achieved rates, and this configuration cannot demonstrate a 100 GB/s standard-core path. A wider core/cache word does not remove the outlet count/width bottleneck.

The 32-PC stream bank remains a separately routed memory-system component, not proof that the core or torch operator path absorbs its rate. Selecting a full-image candidate requires measured coupled service and an architecture that closes these bounds, plus several-times operator speedups. The current diagnostic kernel synthesis measures integration area; it does not authorize a full run. The landing-register fix must also be represented in future performance predictions: independent eight-unit TLM hypotheses do not describe this shared compact planner and completion port.

## Wide diagnostic kernel synthesis

The integrated hybrid/four-slice-SFU kernel has completed free out-of-context synthesis on the upgraded x86 host. It uses 907,599 LUTs, 264,242 FFs, 192 RAMB36, 180 RAMB18 and 646 DSPs. Relative to the earlier 819,109-LUT diagnostic core candidate, LUT use increases by 88,490 (10.8%). At a 10 ns constraint, synthesis WNS is +3.809 ns and WHS is +0.055 ns, with zero TNS/THS. This is not routed or linked timing. Raw reports and the remote input hashes are retained in kernel_wide/. The checkpoint is ~/r41_hbmbw_wide_kernel_v1/out_landing/synth.dcp. A free tile-0 SLR1 placement/routing screen has started from this exact checkpoint, under tile0_slr1 and tile0_screen.log.

Ordinary torch.compile add, mul, silu, gelu and clone/copy now pass float32 numerical checks on 33 elements through the actual three-core/three-outlet hybrid MMIO RTL stand-in. This first add includes 27 compiler tuning launches; the subsequent operations each launch once. It is a functional result with a small cache fixture, not an operator-speed comparison or a board measurement. Raw output is hybrid_lsu/coupled/torch_hybrid_pass.log. The current outlet ceiling and other end-to-end gates remain open.

## Wide tile routed screen

Tile 0 extracted from the wide-kernel checkpoint passes its free SLR1 routed screen at 10 ns: WNS +0.508 ns, WHS +0.007 ns, zero TNS/THS. It uses 273,894 LUTs, 75,791 FFs, 62 block RAM tiles and 213 DSPs. LUT occupancy is 63.4% of SLR1. Raw timing and utilization are in kernel_wide/tile0_slr1/. This isolates intra-tile placement and routing; it does not prove simultaneous three-tile packing, the static shell, inter-SLR paths, stream-CU coexistence, full-image 100 MHz timing or throughput. Those combined risks remain open before a full-run proposal.

## Actual home window and FP16 operator progress

The real tile uses eight MSHRs and 16 target slots. With that capacity and image-like registered-tag/victim controls, the checked cold-line fixture measures 0.5798 GB/s per home at assumed 100 MHz. Increasing the synthetic window to 16/32 has no effect on its 45,212-cycle completion; the eight-MSHR case has no MSHR-full events. This shows the receiving-stage serialization in this fixture, not a controller or board guarantee. Details and source hashes are in wide_home/windows/.

The standard torch 16,384-element float16 RTL comparison reaches 3.530x add, 3.623x mul, 2.509x silu, 1.898x gelu and 4.150x copy against a block16 legacy fixture, with unchanged numerical checks. The hybrid uses maximum block128 and the small-cache synthetic memory; therefore these are not hardware-only A/B ratios or board acceptance. Matching-block controls and a new integrated HIT_CHAIN kernel screen are running. The new option's unit-routed timing passes, but complete candidate placement, 100 GB/s and all required board speedups remain open.

## Single-read fill optimization

A conservative direct response for one ordinary read target cuts the image-capacity cold home fixture from 45,212 to 36,738 cycles for 4,096 checked 64-byte lines (0.5798 to 0.7136 GB/s at assumed 100 MHz). Its integrated full-vector and global cache-handoff regressions pass. The fast path is now enabled with HYBRID_LSU; multi-target, write and atomic misses retain the original service. The improvement is only 1.231x at one home and does not remove the roughly nine-cycle-per-line serial limit. Unit routing and standard torch coverage for this new variant are pending. Further parallel miss service, controller/outlet capacity and core/RF absorption still require evidence before proposing an image as meeting 100 GB/s.
