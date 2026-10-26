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
