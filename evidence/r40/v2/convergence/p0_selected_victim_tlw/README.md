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
