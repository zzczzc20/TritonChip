# Wide home/cache service

The actual tnca_nuca_slice now supports a complete 512-bit cache word without changing its 64-byte memory-line interface. This is the receiving stage for the wide NI and hybrid LSU. The 64-bit default remains the existing path, and no board wrapper enables the wide mode yet.

## Implementation

The word selector now derives its shift and width from WORD_BYTES, with a constant zero selector when one word occupies the entire line. The prior fixed three-bit, eight-byte selector could not safely serve 512-bit words. Both hit and miss-target wake paths now use that geometry.

A wide word still executes the original scalar atomic operation. Positioned byte enables identify the relevant eight-byte group; that group calls the unchanged atomic function with its own old/data bytes and the scalar compare operand. Existing masked writes preserve every other byte of the line. The response returns the complete pre-operation cache word, so the hybrid adapter can extract the original scalar at its remembered byte position. No arithmetic or comparison semantics were substituted.

Scalar TMU accesses preserve their original eight-byte address. The store word is extracted from its position in the wide request, and the 64-bit read result is placed at that same position in the wide response. Register offsets and the TMU's own logic remain unchanged. The compact vector path rejects the TMU page and reaches this logic through the narrow adapter.

## Functional evidence

`tools/r41/wide_home_tests.py` passes nine configurations:

- The existing default 64-bit slice passes 2,000 randomized requests.
- Four wide tests (two seeds, data pipeline off/on) each pass 2,000 randomized requests. A 512-bit shadow checks all read bytes after randomized partial writes, cache misses, merged misses, dirty evictions, variable DRAM service and reordered replies. The test emits hit/miss/merge/writeback and blocked-miss counters and drains every outstanding request.
- Two same-line pressure cases use registered ownership/range logic and a 200-cycle memory response to exhaust the target list. Each passes 100 mixed requests and explicitly requires a nonzero target-full counter; backpressure and eventual drain are both verified.
- Two special cases, with data pipeline off/on, each cover 288 atomic cases and 1,456 total transactions. Every scalar atomic opcode is exercised at every eight-byte position, including both 32-bit halves, first on a cold miss/wake and then a cached word containing distinct sentinel bytes. Full old-word returns, successful/failed CAS and preservation of untouched data are checked. All eight TMU word positions are read/written, and responses are held under backpressure.

The initial special fixture selected HOME_W=0, which the pre-existing address-recovery function does not support; the failed elaboration is preserved. Using FULL_LINE_TAG=1 with the supported HOME_W=2 supplies the intended small-cache fixture without modifying unrelated geometry.

These are functional receiving-stage checks, not a sustained throughput guarantee. The serialized cache engine's service rate, misses/writebacks and actual coupled NoC/core demand still need measurement with compiled operators and counters. A 512-bit response width alone does not establish 100 GB/s or several-times board operator speedup.

## Physical screen and host

The free 10 ns SLR1 screen uses the candidate's 32 sets, eight ways, full tags, registered owner/victim selection/rank, data pipeline and registered DCO ranges. It runs on the upgraded x86 host under ~/r41_hbmbw_wide_home_v1, with four threads after sourcing ~/x86_link_env.sh. The first attempt was terminal with a missing tnca_dco_owner_registered.vh in the transferred source bundle; its log is preserved. The header was supplied and the new out_v2/screen_v2.log run is live (launcher 4692, Vivado 4760, last verified during placement). This was a source-bundle omission, not a proven runtime/library gap. No full v++ link, budgeted image run or card access was started.

Next parameterize the real tile/mesh/host and board hierarchy, connect the hybrid core to this home through the wide NI, and validate ownership, tensor transfers, atomics and standard torch operators on the complete path. Final routed timing/area and the full image remain unqualified.

## Routed screen result

The free SLR1 screen completed on the x86 host after sourcing ~/x86_link_env.sh. At 10 ns, routed WNS is +1.774 ns and WHS is +0.007 ns. The routed slice uses 41,279 LUTs (30,807 logic and 10,472 LUTRAM) and 5,789 FFs, with no block RAM or DSP. Raw routed timing, utilization and DRC reports are retained in physical/. This is a unit implementation result, not a full-image timing or throughput qualification. No full-run allocation or card access was used.

## Measured service and opt-in hit chaining

The checked home service sweep now measures accepted requests through final responses instead of assuming that a 512-bit word implies one word per cycle. Run `PYTHONPATH=. python3 tools/r41/home_service_sweep.py --out FRESH_DIRECTORY`. Six cases return the same 262,144 useful bytes; every read and request/response count is checked. The fixture uses 32 sets/eight ways, four MSHRs, eight target slots, 32 source tags, a synthetic delayed/reordered memory and an always-ready response sink. This is standalone service, without AXI, NoC, RF or torch.

| Word / mode | Cycles | Useful GB/s at assumed 100 MHz |
| --- | ---: | ---: |
| 64-bit hot read | 98,475 | 0.266 |
| 512-bit hot read, original selection | 12,459 | 2.104 |
| 512-bit hot read, hit chaining | 8,385 | 3.126 |
| 64-bit streaming read | 266,233 | 0.0985 |
| 512-bit streaming read, original selection | 56,280 | 0.4658 |
| 512-bit streaming read, hit chaining | 56,280 | 0.4658 |

The original hit path spends a separate cycle selecting the next request after completing a hit. HIT_CHAIN=1 permits a retiring hit to initiate tag lookup for another eligible request. The arbiter excludes the retiring entry; older same-line dependencies remain shadowed, so conflicting requests keep their original serialization. Maintenance and pending MSHR completion work suppress chaining. The response queue reserves capacity for both the current response and the next one. The option defaults to zero and is not enabled in the tile/image candidate yet.

Chained randomized partial writes/reads pass 4,096 requests under response backpressure for both data-pipeline settings, including 1,297 owner-epoch retries in the pipelined case. Both sets of 288 atomic cases and all eight TMU positions pass. Raw checks and complete service cases, source hashes and configuration metadata are in service/. The 1.486x hot-service improvement does not improve the cold miss path, establish 100 GB/s or satisfy operator acceptance. The cold result identifies a remaining receiving-stage bottleneck that must be addressed and measured.

A free HIT_CHAIN=1 SLR1 screen is live on x86 under ~/r41_hbmbw_home_chain_v1, screen.log / out, launcher 16692 and Vivado 16760. The Tcl accepts an optional third argument selecting the chain option; two-argument invocations retain the original setting. New routed timing and coupled-core qualification remain pending. No full link or card operation was started.

## Image-capacity cold-service sweep and tile integration

The earlier service probe used a 4-MSHR/eight-target fixture. The actual tile defaults to **eight MSHRs and 16 targets**; the earlier 0.4658 GB/s cold value is not a measurement of that image capacity. A new 4,096-line cold-read sweep uses the full-line tag and registered owner, victim selection and rank settings, with synthetic 40-cycle delayed/reordered memory and an always-ready response sink. The eight-MSHR/16-target case returns all checked data in 45,212 cycles, or 0.5798 GB/s at an assumed 100 MHz. Eight/eight and hypothetical 16/32 windows produce the same 45,212 cycles, with no MSHR-full events in these cases. Four/eight takes 58,410 cycles and records 3,304 MSHR-full events. This identifies a service-engine bottleneck for this fixture; a bigger window alone does not improve it. The 16/32 case is hypothetical and would require a wider memory tag and integration. Full logs, geometry and verified source hashes are in windows/; reproduce with `PYTHONPATH=. python3 tools/r41/home_window_sweep.py --out FRESH_DIRECTORY`.

The standalone HIT_CHAIN=1 SLR1 screen completed at 10 ns with WNS +1.628 ns and WHS +0.011 ns. The observed terminal-log excerpt is retained in physical_chain/screen_excerpt.txt. Full reports could not yet be fetched because the x86 host timed out during SSH banner exchange; retry the existing files, not the completed screen. This is unit timing only.

HIT_CHAIN is now selected by HYBRID_LSU in the actual tile. The complete one-core/one-outlet, three-core/two-outlet and three-core/three-outlet full-vector matrix passes with unchanged numerics. Three-core configurations additionally assert that the chain path was exercised. Six maintenance unit configurations pass with the option enabled; a real three-core coherent handoff test also passes, checking 455 dirty results and a partial DMA update. Logs are retained in windows/. A new full-kernel **free synthesis screen**, not a full link, is live on acf3019 under ~/r41_hbmbw_kernel_chain_v1, screen.log / out, slot-wrapper PID 3170968. Its source includes the integrated chain. The earlier wide-kernel/tile routed reports predate this change and cannot qualify the new candidate.
