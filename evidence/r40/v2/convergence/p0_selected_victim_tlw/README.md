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
