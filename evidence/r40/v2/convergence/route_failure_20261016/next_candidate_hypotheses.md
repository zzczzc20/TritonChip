# Bounded candidate screening before either remaining full implementation

This is an engineering investigation within R40-05/06, not a contract change
or an additional work package. No complete implementation attempt is launched
by these notes. Gate status remains not-measured.

## What the failed checkpoint establishes

The retained critical-path report has 30 endpoints but only two distinct source
registers. Sixteen paths start in TMU base metadata and fourteen start in slice
2 tag storage. They terminate in writeback-address RAM write enables. Do not
interpret these duplicated endpoints as 30 independent bottlenecks.
The worst reported path has 91 logic levels and 50.082 ns data delay, including
43.934 ns attributed to routing. The other source has 98 logic levels. The
substantially unrouted design contains estimated delays; these are diagnostic
paths, not final setup/hold qualification.

The kernel hierarchy accounts for 962,157 LUTs. The four reported HBM soft
interconnect siblings total 9,999 LUTs. This contradicts the hypothesis that
those interconnects dominate the global LUT footprint. Their observed footprint
is not a floor for a modified design. Reducing the configured PC range without
further evidence is not the next experiment.

## First local hypothesis: clocked, banked tag reads

Actual tag storage has one write site in `E_FILL`, no reset of its contents,
and validity held separately. Current full-line tags are asynchronously indexed
for every way; the failed path starts in a physical tag flip-flop.

Evaluate one synchronous tag bank per way. Read the selected request's set on
the `E_SEL` admission edge, concurrently with capture of `cur_addr`, so the
registered tags are available in the existing `E_LOOK` stage. This could remove
large asynchronous muxes and allow block RAM inference without adding a normal
request cycle. That latency claim must be tested, not assumed.

Maintenance needs equally explicit prefetch: set zero at maintenance admission,
next set when advancing `maint_index` in `E_MSCAN`, and next set after a drained
writeback in `E_MSEND`. Hold the current set during `E_MREAD`/`E_MSEND` stalls.
Do not infer correctness from hit-only traces: fill, dirty eviction, maintenance,
reset, high-address tags, last-set wrap and replay must be exercised.

## Second local hypothesis: use the existing lookup stage for victim owners

Victim-owner membership currently feeds tile classification, dead checks,
victim priority and writeback admission in one combinational cone. Evaluate
registering each way's owner result during `E_LOOK` and consuming it during
`E_MISS`. Use the already source-checked unsigned-64 registered endpoint path.
Keep current-request owner/TLW behavior unchanged until separately evaluated.

Metadata commits can occur between those stages. Export an owner-table epoch
that changes on every accepted descriptor commit or table clear, using exactly
the same accepted-source and offset predicates as the real TMU update. If the
epoch changed, repeat lookup before allocating or dropping a victim. A one-bit
epoch is sufficient only if consumption always immediately follows lookup;
prove this state-transition invariant and do not generalize it to arbitrary
stalled snapshots. Dead-FIFO changes need no owner-epoch retry if dead checks
remain live in the consumption stage.

Preserve epoch-change failure/mutation cases. Test finite concurrent metadata
updates and eventual progress; do not conceal retry overhead. If classification
still requires long cross-region metadata paths, evaluate physical local
register replicas with identical commit edges. Delaying the global metadata
vector by one cycle is not an equivalent replica.

## Decision before a complete implementation

First run functional/mutation checks and actual paired TMU-plus-slice OOC
screening with complete immutable inputs and identical constraints. Require
actual bank inference and a materially shorter limiting cone; compare resource
changes and every competing path. If those observations support the hypothesis,
combine the remaining necessary hardware controls into one immutable candidate.
If they do not, retain the results and repair the specific mechanism. Do not
consume a full implementation attempt or rotate directives to bypass a failed
local screen. Positive OOC timing never qualifies a final image.
