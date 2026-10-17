# Final bounded implementation candidate

The failed image reconstructed cache-private lookup/replacement logic under
TMU and placed it across SLR boundaries. Explicit LOC-to-site queries confirmed
this detour even though named core/slice groups and pipeline state occupied
the intended regions. This candidate preserves cache instance hierarchy so
that tile pblocks own the complete private cone, while retaining required
code-queue protection and internal optimization.

The corrected screen connects all maintenance inputs and uses three real
slices sharing one real TMU under rebuilt synthesis and `opt_design`:

| Cache boundary | LUT | FF | Foreign combinational consumers by cache | Worst data ns | Levels | Estimated WNS ns |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| Unpreserved | 99,976 | 15,509 | 4,144 / 5,208 / 4,153 | 8.383 | 33 | +1.374 |
| Preserved | 100,039 | 15,524 | 0 / 0 / 0 | 8.344 | 33 | +1.413 |

The selected-private-register fanout audit covers tag outputs, victim owners,
ages and valid bits across all module namespaces, stopping at sequential
endpoints. Counts are per-cache dependencies, not unique whole-design cells.
Both versions retain 96 RAM32M16 tag macros and 1,272 tag-output flops across
three caches. Macro children are not additional independent storage.

The preserved worst path is victim-owner flop to writeback-address RAM enable:
2.465 ns logic and 5.879 ns estimated net delay. Tag reads register at
E_SEL/E_LOOK; owners register at E_LOOK/E_MISS with epoch retry; victim ranking
uses a balanced tree without another request cycle. Functional/mutation and
source-regression evidence is linked from the P0 package. The decoder repair
is included in this candidate, unlike the failed cf330dc image.

These are module-level, unplaced estimates. Cores, NoC and the HBM platform
are absent; these reports cannot qualify full packing, routed setup/hold, CDC,
a loadable image or frozen workloads. `preflight.json` binds scope, exact
sources, configuration, clock, immutable contract and raw archives. Missing
performance remains not-measured.

The final full attempt retains three cores/outlets, HBM[0:31], 100 MHz,
32 sets/eight ways, two-stage links and SSI_SpreadSLLs. It runs once in a fresh
build directory and never retries or loads a board. Success must be followed
by actual placement, route, clock/setup/hold, CDC and image qualification,
then the unscaled >=64 MiB standard Torch checkpoint and complete frozen
inventories. Failure retains logs/DCP and does not trigger another attempt.

Started at 2026-10-16 22:33:29 UTC from immutable source `0c39793`;
[launch.json](launch.json) records the exact archive, source, configuration,
clock, contract, process and observer identities. The implementation budget
is now two used, zero remaining. No acceptance gate is promoted at launch.

Current physical-gate blocker: this candidate has not completed full route.
Direct evidence: the runner has started; route/timing/CDC/image results are
not-measured. Next command on the FPGA host:
`cat ~/tnca_build/r40_attempt02_followup/state.json`.
Success: inspect complete actual-placement, route, 100 MHz setup/hold, CDC
and image reports before loading. Failure: retain logs and DCP and diagnose;
never retry automatically or lower the clock/cores/workload.

The vertical hardware/workload gates are blocked by that same unqualified
image. After physical qualification, run the existing original >=64 MiB
Torch checkpoint on the immutable software snapshot, then the complete
native 77, HBM 14 and frozen model inventories. Numerical, ownership,
three-core/outlet traffic and visibility evidence are required; a checkpoint
failure must preserve its exact inputs and output before any correction.

The running full-system kernel has completed cross-boundary/area optimization
and emitted its preliminary RAM inference report. `tag_inference.json` records
all reported synchronous tag-bank rows and binds the captured original log
prefix in `kernel_inference_prefix.log.gz`: 32 x 53 bits per way infer four
RAM32M16 macros per way. Repeated/specialized module rows are not a final
instantiated primitive census. This supports the intended storage mapping;
final mapped/placed counts, locality, routed timing, CDC and image remain
unqualified. No required gate is promoted.

At 2026-10-17 03:17 UTC, actual placement reached Phase 4.1 Post Commit
Optimization; route had not started. Kernel synthesis completed in 1:13:11
with 821,876 LUTs, 236,357 FFs, 214.5 BRAM tiles and 531 DSPs. The completed
logs, utilization report and generated synthesis script are preserved in
`synthesis_reports.tar.gz`. `post_synthesis_observation.json` binds their
hashes, the streamed SHA256 of the retained 156,828,189-byte kernel DCP,
constraint/configuration hashes and the captured placement-log prefix.
The DCP was hashed as bytes; no additional Vivado checkpoint reader was run.

The placement log reports aggregate SLR0-to-SLR1 demand of 15,497/23,040 SLLs.
This does not measure hotspot-column demand or establish final routability.
Actual placement/locality, complete route, routed setup/hold, CDC and image
qualification remain open. No required gate is promoted.

Pinned-source CI [38976369917](https://github.com/zzczzc20/TritonChip/actions/runs/38976369917)
completed successfully on exact hardware source `0c39793`; `source_ci.json`
retains the authoritative run and job results. CI is source regression
qualification, not board workload or physical implementation acceptance.
