# Selected-victim and TMU notice pipeline candidates

Required gates remain unchanged. These archives preserve isolated actual RTL, stimulus, raw failures, mutations and successful functional screens. They are candidates for physical convergence, not qualified production hardware.

The selected-victim candidate adds `E_MISS -> E_ALLOC`: way/eligibility/dead/bypass decisions end at registers, separating owner/victim selection from writeback allocation. Owner and selected-decision epochs are checked on adjacent edges; a mismatch retries before request/MSHR/TLW side effects. The single engine excludes pre-existing claimed ways and rechecks allocation resources at commit. Traffic covers replacement policies, metadata churn, dirty eviction, maintenance and full-tag operation. The additional pressure screen uses the actual 32-set/eight-way/eight-MSHR/sixteen-target geometry, DRAM latency 400, and requires dirty writebacks, resource exhaustion and hits beyond blocked misses. Both owner-guard and selected-guard mutations are detected. Hits retain their existing path; a miss gains one engine cycle.

The failed full-image DCP identifies a different downstream bottleneck: TLW address -> TMU dead update, 30 levels, 20.561 ns data delay. The TMU candidate registers the arbitrated address and then classified tile ID/nAcc before live-table counting. It processes one notice per cycle in steady state, with 0/1/2 cycles of additional notice/dead-counter latency by mode. A waiting register write stops new notices, waits for every accepted notice to retire, and only then asserts ready. Table publication and owner epoch remain atomic on the accepted write edge; CLEAR/DESC/default-granule changes cannot pass older notices. This ordering is tested against an independent original TMU receiving the same accepted transaction stream, compared after bounded drain. Depths 0/1/2, ports 3/4, overflow/high addresses, staging, reset, TLW/dead counts and write pressure are covered. Removing the write-drain interlock produces the expected failure.

Two unsuccessful stimulus attempts are retained explicitly. The replicated-state alternative used the original three-port bench at PORTS=1, illegally selecting port1; valid 3/4-port runs pass and detect a missing-broadcast mutation. The first pipeline bench provided no quiet post-registration traffic phase and failed its positive notice/death coverage check. The corrected bench adds 64 quiet cycles and bounded drain, without changing RTL. State replication alone was not selected because it leaves the newly observed classification/count chain uncut.

The prior paired module control hit its 2700-second observation cap during routing. That is not proof of an RTL failure or a successful routed comparison. The new combined module screen preserves placed timing/resources and a small checkpoint before attempting route, so a bounded route cannot discard the placement evidence. Its actual three-cache/shared-TMU geometry, source hashes, scope and live job are recorded remotely in `~/tnca_build/r40_tmu_pipeline_module_20261017/identity.json`. No cores, NoC or HBM shell are present; I/O timing is unspecified. Global full-image timing, CLB packing, CDC, image and workload performance remain unqualified.

The R3 whole-system [preflight archive](r3_system_preflight.json) binds the
local coherent TMU copies and pair-rank candidate to actual-geometry ownership,
four memory-control Torch configurations and backend checks. These are functional
simulation results; they do not qualify a board image or replace the frozen
77-point, HBM14, model or performance inventories.

The [residual-path archive](placed_residual_paths.json) contains raw placement
reports and a bounded auxiliary extraction by `gemini-3.8-flash-high` at high
effort. The primary agent verified both report hashes and every quoted line for
the six sampled paths. The local-TMU/pair-rank candidate's worst sampled placed
path is slice2 write ingress to replica0 metadata: 14 levels, 9.034 ns
(1.364 logic / 7.670 estimated net), slack +0.512 ns. The earlier shared-TMU
candidate exposes cache data read/AMO/whole-line rewrite: 24 levels, 9.067 ns
(1.527 / 7.540 estimated net), slack +0.542 ns. These estimates motivate local
write inbox registers and byte-enabled cache writes with E_LOOK hit-word capture.
Neither estimate establishes routed full-image timing.

Current blocker: no qualified three-core 100 MHz image. Direct evidence remains
`../attempt02/terminal_failure.json`; the residual reports identify the next RTL
hypotheses. Next experiment on the ARM jump host:
`tail -n 25 ~/r40_r4_functional_v2_20261017/runner.log`.
Success requires the numerical oracle, cycle miter, write-publication ordering,
maintenance and mutations, followed by matching module physical evidence.
Failure preserves the exact trace and repairs that cause before physical use.
The final full implementation budget remains reserved.

## R4 implementation and running validation

The optional convergence RTL is now connected through the board wrapper, FPGA
wrappers, shell, top and tile. Every new production parameter defaults to zero.
The candidate explicitly enables `NUCA_REGISTER_VICTIM_SELECT=1`,
`NUCA_REGISTER_VICTIM_RANK=1`, `NUCA_CACHE_DATA_PIPELINE=1`,
`DCO_TLW_PIPE_STAGES=2`, `DCO_LOCAL_REPLICAS=1` and
`DCO_REGISTER_WRITES=1`, alongside the existing registered ranges/owner settings.
The three leaf RTL bodies match the immutable simulation/module inputs after
normalizing those documented parameter defaults; both hashes are retained in
[r4_functional_snapshot.json](r4_functional_snapshot.json).

The local write inbox captures port, offset, data and source before acknowledging
a write. Accepted TLW notices drain first. Table publication, owner epoch and ACK
still occur on one edge. An isolated metadata write gains one cycle, with at most
one write every two cycles; capture can overlap notice drain. The notice pipeline
adds two cycles of counter/dead visibility, sustaining one accepted notice per
cycle. Copies consume the same global accepted stream and publish coherently;
only the consumed local state and pipeline boundaries receive preservation
attributes. The placement hook assigns each retained local TMU to its slice's
SLR. The final actual-location audit reports each copy independently of pblock
intentions, and rejects an incomplete set of replicas.

Cache hit-word capture uses the existing E_LOOK edge and adds no hit cycle.
Stores and atomics share one byte-enabled write port, avoiding readback and
rewrite of unaffected bytes. Fill writes all bytes; maintenance, dirty marking,
whole-old-word atomic replies and zero-byte stores retain their prior semantics.
The selected-victim stage adds one miss cycle; the pair-rank stage adds another
for full-set ranked replacement. Invalid-way and round-robin shortcuts skip the
rank stage. Adjacent epoch checks retry stale classification before side effects.
These architecture costs are explicit; performance remains not-measured.

The snapshot includes a cycle miter against the original data path, an independent
scalar oracle for masked stores and 32/64-bit ADD/AND/OR/XOR/exchange/CAS/load/store,
accepted-stream TMU checks at notice depths 0/1/2 with write registers off/on,
three/four-port replica comparisons, maintenance and pressure traffic. The
additional width test and all four mutations completed successfully: corrupting
byte enables, corrupting hit-word capture, removing notice drain and acknowledging
before inbox capture are detected. The full batch is still running in this
snapshot; completed subsets do not stand in for its terminal result. The initial
zero-width miter elaboration failure is preserved as a testbench failure, followed
by the explicit-width correction with unchanged DUT RTL.

The [R3 terminal module archive](r3_module_terminal.json) records 90,038 fully
routed routable nets and zero routing errors, but accurate min/max timing and a
routed checkpoint were not retained before the reporting cap. The
[R4 module identity](r4_module_identity.json) binds the next screen to specific
write-ingress/data-path hypotheses, the same geometry/10 ns clock/directive and
5400-second bound. It saves a routed DCP immediately after route. This is a
three-cache/three-TMU module experiment, with no cores, NoC, HBM or timed I/O;
it does not consume or replace the final full-image implementation.

Next commands, with independent outcomes retained:

```sh
# ARM jump host
cat ~/r40_r4_functional_v2_20261017/runner.exit
cat ~/r40_r4_functional_v2_20261017/mutations.exit
cat ~/r40_r4_preflight_20261017/preflight.exit
cat ~/r40_r4_legacy_20261017/legacy.exit
# FPGA host
tail -n 25 ~/tnca_build/r40_r4_module_20261017/stdout.log
```

Success permits review of source-matched exact setup/hold paths, tag primitives,
local TMU locations and packing before freezing attempt 3. Failure preserves the
first numerical, protocol, mapping or timing error and repairs that cause.
Required gates, final-image acceptance and the frozen contract are unchanged.


## Completed R4 physical and functional evidence

[r4_physical_terminal.json](r4_physical_terminal.json) and its raw archive
establish 75,184 fully routed routable nets with zero routing errors; accurate
internal setup WNS +0.630 ns and hold WHS +0.004 ns at a 10 ns clock; 96 RAM32M16
tag roots; and 2,571 mapped state rows per local TMU, each confined to its own
SLR. RAMD32/RAMS32 children are not additional tag roots. The owner chain ends
at first-pair registers (worst sampled owner-to-pair: 14 levels, 7.429 ns =
1.662 logic + 5.767 routed net), then selected-way registers (8 levels, 3.620 ns
= 1.227 + 2.393), then allocation outputs (6 levels, 7.044 ns = 0.859 + 6.185).
No direct combinational owner-to-selected path was reported after these cuts.

The remaining module worst path is slice2 `cur_addr[18]` to `tl_a[10]/CE`:
21 levels, 8.970 ns = 2.109 logic + 6.861 routed net. The local write inbox
separates a sampled 3-level/5.673 ns ingress path from a 10-level/6.486 ns
local commit path. Hit-word capture is 7 levels/7.884 ns; consumption is
13 levels/5.923 ns. These are actual routed module paths, not full-image timing.
Matched post-opt resource records show the byte-enabled/inbox changes trade
logic for LUTRAM: R3 115,247 total LUTs / 24,692 LUTRAMs / 19,725 FFs versus
R4 108,608 / 29,936 / 20,144. Module occupancy is much lower than the failed
complete design; neither resource savings nor positive module slack guarantee
full-image closure.

The original R4 report failed because a located RAM256X1S macro returned no
SLR when queried as a cell. Its actual site resolves correctly. The read-only
completion performs site-to-SLR checks, retains all rows, and completed with
no unresolved or misplaced state. The production actual-placement audit now
uses the same site relationship, including macros without IS_PLACED. No
physical implementation was rerun for the report correction.

[r4_completed_functional.json](r4_completed_functional.json) preserves the
complete R4 cache matrix, all maintenance configurations, metadata/replica
comparisons, atomic widths/mutations, whole-system ownership, all four native
memory-control Torch configurations and backend tests. Legacy default-mode
mesh/atomic programs passed in single-role and all-role modes. The original
maintenance aggregator rejected its existing `NUCA MAINTENANCE PASS` line;
all eight unchanged cases were subsequently completed with that exact success
marker. The DUT was not modified to satisfy the runner. CI run 39047006244 is
successful on commit `1f821c6`.

## Last local predicate correction before freezing

The exact TLW predicate can avoid `(1 << lsb) - 1`: a last-word address requires
all bit positions 3 through lsb-1 to be set. The replacement reduces missing
fixed bits and retains the original granule/owner/policy/read guards. Symbolic
SAT proves equality for every 64-bit address, every six-bit granule and every
guard input. An off-by-one mutant produces a counterexample. Complete-slice
cycle comparisons against immutable R4 pass with DBP enabled, partial writes
and upper/lower 32-bit and 64-bit atomics. No state or clock cycle is added.
Exact source, proof logs, mutant and cycle-miter stimulus are in the archived
`r40_tlw_predicate_20261017_complete.tar.gz`.

The matching module screen is `~/tnca_build/r40_tlw_predicate_module_20261017`.
It preserves geometry, clock and directive, corrects the report query, and saves
the routed checkpoint once before expensive reporting. Next experiment:
`tail -n 25 ~/tnca_build/r40_tlw_predicate_module_20261017/stdout.log`.
Success permits final configuration/source freeze and the last full-image run,
subject to the exact physical and functional evidence. Failure retains the new
worst path and repairs a demonstrated cause. No required gate, full-image
clock/CDC/loadability result or board performance is inferred from these screens.


## Complete the HIT owner boundary

The exact-mask candidate's post-opt worst path still starts at local TMU tensor
metadata and ends at the TLW buffer enable: 20 levels, 4.847 ns with estimated
routing. Source inspection identifies the reason: E_HIT bypassed the E_LOOK
request-owner registers. The final local candidate reuses that owner snapshot
on hits. If the adjacent owner epoch changed, it suppresses only the unformed
hint. It does not retry the data request, add state, or add a hit cycle. Policy
and default granule still use their current values. Existing formed notices
retain their original valid/ready behavior.

[hit_owner_functional.json](hit_owner_functional.json) retains all three bench
iterations with identical DUT RTL. Random churn initially missed stale last-word
hits in one seed; forcing every eligible hit then removed stable-hint coverage.
The completed stimulus invalidates the first four eligible hits and retains
independent periodic miss/rank churn, both coverage floors, original request
streams and numerical checks. All fixed/changing-metadata policy/seed cases and
the removed-epoch-guard mutation passed. The complete-slice data-cycle miter
checks response and backing-memory timing/values against the prior candidate;
the scalar oracle covers partial writes and 32/64-bit atomics.

The original STAT_DROP remains the busy-buffer loss counter. It does not measure
intentionally unformed stale-owner hints; zero STAT_DROP does not prove zero
stale-hint suppression. These tests and comparisons provide no board performance
measurement. The [matched module identity](hit_owner_module_identity.json) binds
this correction to the same geometry/clock/directive. Next command:
`tail -n 25 ~/tnca_build/r40_hit_owner_module_20261017/stdout.log`.
Success requires actual min/max timing, tag mapping and local-state placement,
followed by exact source CI and final immutable configuration freeze. Failure
preserves the remaining physical path; do not claim the earlier R4 module proves
this newer source's route.

The separate [CI regression record](site_query_ci_regression.json) preserves the
0827d03 failed job: the site-query correction had not updated the query stub.
The fixed tests model the observed RAM macro and retain wrong-location/missing/
unplaced refusal tests, including local TMUs. Both collector and physical-clock
local suites pass. New source CI remains pending until its own run completes.
The placement pre-hook now reads the launcher's immutable TNCA_IMAGE_CONFIG when
not already loaded, so final-image replica checks are bound to the actual config.
