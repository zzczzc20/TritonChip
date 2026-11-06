# Selected-MSHR service descriptor screening

This isolated fallback is not part of the running attempt03 image. No further
full implementation or module physical run has started for it.

The routed predecessor module's worst sampled path starts at
`cur_ms_reg[1]_rep` and ends at a data RAM input: 18 levels, 9.203 ns data delay
(1.446 ns logic, 7.757 ns routed net). Its leading RAM address net has fanout
3,031 and 2.608 ns net delay. The complete path includes a RAM read, atomic
ALU and RAM write data; it is not simply a write-enable path. The raw first
path is retained in `predecessor_critical_path.rpt`. This is predecessor-module
evidence, not a diagnosis of the still-running full attempt03 route.

When `CACHE_DATA_PIPELINE=1`, the candidate captures the selected MSHR's set,
way, bypass and tag on the existing E_SEL arbitration edge, alongside cur_ms.
Subsequent RAM addresses and fill controls use these registers. It adds no
service state or cycle. These descriptor fields cannot change while that MSHR
is serviced; returned line data, head/tail and coalescing state retain their
existing live paths. Simulation asserts the descriptor invariant at each
consuming state. The legacy parameter value retains its original paths.

The actual 32-set, eight-way, eight-MSHR, sixteen-target slice passed a cycle
miter against exact baseline RTL 54249f7 plus an independent numerical oracle.
Both sides explicitly enable full tags, registered owner, selected victim,
victim ranking and the data pipeline. Cases include dirty writeback, partial
writes, 32/64-bit atomics, backpressure, out-of-order DRAM at latency 400,
metadata epoch changes, and a global address base of 2^48. The bypass object's
registered bounds are rebased to that same aperture. Capturing the wrong set
is detected by mutation. Exact sources, fixture and logs are in
`functional_raw.tar.gz` and `functional_report.json`.

No new resource, delay, min/max timing, RAM mapping or routed-congestion
improvement is claimed. The atomic feedback path remains and may still limit
timing. Further screening is conditional on attempt03's terminal diagnosis:
if its actual blocking path matches this hypothesis, run the same constrained
module placement/route and inspect the new worst path, logic/net delay, actual
RAM primitives and packing. If the root cause differs, investigate that cause.
Do not spend the final full-implementation reserve on this functional result.
