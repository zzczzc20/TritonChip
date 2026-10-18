# R40 v2 convergence checkpoint

## Inventory collector correction after source-origin validation

The source-origin correction passed its actual child preflight and repeated V
successfully on software tree `cbc1340e1135ac301bfe6c39488bb3eb2b1571aafad46920e070ae701c58da52`.
The next inventory blocker was cold-call contamination: native FakeTensor
constant propagation launched `_copy_kernel` while compiling model graphs.
The first numerical results were correct, but the launch-count guard correctly
reported `not-measured`. [Cold records and second full checkpoint](attempt03/cold_inventory_3c403a4.json)
retain that evidence without dropping launches or weakening the guard.

The functional pipeline now requests one untimed warmup and one independent
steady graph call using the existing cold-setup separation in ArchBench, and
retains complete compiler/provenance bundles for every original point. This is
functional inventory sampling, explicitly not a performance measurement or a
change to the frozen 5/30/3 performance plan. Original 77 and 14 point identities
are unchanged. Its manifest binds the collector SHA. Success requires exact
steady wrapper launch counts, original numerical tolerances and zero implicit
staging, then the same full model stage. Failure preserves the first original
point; no development point or smaller shape can replace it.

## Vertical checkpoint passed; inventory source-origin repair

**Required gate delta: V passed on the preserved 90 MHz image. No complete
required performance gate is promoted.** The original 67,108,864-element FP16
add graph is numerically correct (maximum relative error 0.0002474, tolerance
0.02). Its actual 1 GiB BO spans PCs0..3; live graph tensors span PCs0/1. All
three physical cores completed work, peak active cores is three, and all three
outlets recorded positive read/write traffic before CPU validation. The event
records DRAINING followed by successful quiesce and SUCCEEDED; the host read
records successful ownership handoff and numerical validation.
[Full raw checkpoint and preflight](attempt03/vertical_checkpoint_90mhz.json)
binds these facts to the immutable source, image and qualified clock.

**Inventory blocker:** child processes inherited XRT's PYTHONPATH without the
snapshot's `compiler/python` directory, selecting an older installed compiler
whose scheduler model path does not exist. The original standalone checkpoint
used explicit source paths and is unaffected. Failed native rows and the
stopped runner are retained in
[native_source_origin_failure.json](attempt03/native_source_origin_failure.json).
`archbench.run._child_env` now prepends the snapshot compiler and assembler
paths. A regression starts a child with a deliberately stale compiler earlier
in its inherited installation path and verifies the actual imported compiler
and scheduler belong to the snapshot. The frozen contract is unchanged.

Next experiment: stage the committed corrected source as a new immutable
snapshot, verify actual child import origins, then rerun the unchanged complete
checkpoint/native77/HBM14/model pipeline on the same loaded image. Success
unlocks final-identity gate collectors; failure preserves the first original
point and repairs its root cause. No programming or implementation is needed.
The previously queued inventory was stopped at this known source-origin fault;
it is not reported as a hardware or numerical failure.

## Current delivery state — 2026-10-18

**Required gate delta: no new complete required gate has passed. The conditional
physical prerequisite P90 has passed and the preserved image is loaded.** Actual
DATA_CLK counter samples are 90 MHz (1 MHz readout resolution). Analysis at the
matching operating clock reports WNS +0.016 ns, TNS 0, WHS 0 and no setup, hold,
pulse-width or route failures. All three physical cores and TMU replicas are
placed in SLR0/1/2. The original 100 MHz report remains failed. The user has
explicitly authorized the approximately 10 MHz reduction; no other frozen
threshold, workload, sampling plan or gate has been relaxed.

[Conditional physical checkpoint](attempt03/conditional_physical_checkpoint.json),
[loaded clock qualification](attempt03/qualified_clock90_after_load.json), and
[exact endpoint review](attempt03/operating_preload_review_90mhz.json) bind the
image, software/configuration, actual clock and raw reports. The compressed
[raw qualification and retained failures](attempt03/qualified90_raw.tar.gz)
contains timing, route, placement, CDC, DRC, load log and board-clock samples.
All five raw evidence hashes in the qualified-clock proof were checked against
this archive. CDC review retains 56 Critical findings in unchanged vendor
platform structures, each reviewed by exact endpoints; it does not claim zero
raw Critical findings. No TNCA endpoint appears in the CDC crossing report.
Seven input and three output unconstrained board-management pins are retained
in the scoped review; internal endpoints have complete clock coverage.

**V — previous blocker (now passed; see current state above):** the original graph has not yet completed its
numerical, physical-traffic and drained-visibility assertions. The first launch
failed before compute submission because its archive omitted `tools/r38`.
The complete same-commit archive preserved that failure; its first preflight
then exposed an import-order cycle in Triton plugin discovery, also before
device open. The corrected preflight passed all entrypoint imports and native
extension readiness with zero sessions opened. It automatically started the
unchanged full delivery pipeline on the same loaded image. Neither repair
changes the software tree, graph size, tolerances or hardware.

Next experiment command (observe the already running experiment; do not launch
a concurrent board job):

```sh
cat ~/tnca_prepare_conditional12a0340_full_importfix/state.json
cat ~/tnca_runs_r40_conditional90_12a0340_full/checkpoint/report.json
cat ~/tnca_runs_r40_conditional90_12a0340_full/delivery.json
```

Success requires numerical correctness of the original 67,108,864-element
FP16 add graph, actual >=1 GiB multi-PC memory allocation, three physical cores,
all three physical outlets, and drained completion plus host visibility.
It unlocks the already queued original native77, HBM14 and frozen model stages;
it does not replace any complete gate. Failure preserves the first raw report
and stops the queue for that specific cause. No reduced workload, reprogramming
or further implementation is part of this experiment.

The final delivery PR must be **non-Draft and target `main`** after all remaining
gates pass, with the 90 MHz exception disclosed. Existing Draft PR #7 targets
`r38-dco` and is not the requested final delivery PR.

The table under “Every currently unpassed required gate” remains the per-gate
blocker/evidence/experiment ledger. Its V prerequisite is the running experiment
above; P90 is complete. Historical passed inventory evidence is not proof of
rerunning the inventory on this final identity. Missing measurements remain
`not-measured`. Earlier chronological observations below are retained history
and are superseded by this current state.


## Immediate experiment commands and branches

**Required gate delta: no additional gate passed. Attempt 3 completed route,
bitstream generation and xclbin packaging. Its original launcher exited 3
because Vitis moved DATA_CLK from the requested 100 MHz to 90 MHz. The complete
image is retained; this is a clock-policy rejection, not a missing bitstream.
The original routed 100 MHz report fails setup (WNS -1.002 ns, TNS -6679.026 ns).
Budget: maximum 4, used 3, remaining 1. No fourth implementation has started.**

[Terminal image evidence](attempt03/terminal_image_90mhz_20261018.json) binds the
79,202,018-byte xclbin, complete SHA, raw clock topology info and original link
outcome. Packaged DATA_CLK is **90 MHz**, not a board measurement. It is retained
under the user's [91 MHz conditional-acceptance authorization](attempt03/conditional_91mhz_authorization.md),
with all operating results explicitly labelled at their actual frequency.
The original 100 MHz gate is not passed. No conditional workload gate is yet
qualified; actual clock, timing, CDC, route and placement still require review.

[Independent read-only image backup](attempt03/preserved_image_90mhz_20261018.json)
now retains both the complete xclbin and generated partial bitstream outside the
implementation working directory, with matching source/copy SHA256 and read-only
permissions. No programming or frequency change was performed.
[The staged continuous plan](attempt03/conditional_delivery_plan_12a0340.json)
binds all original 77 native IDs and 14 HBM-core IDs to concrete commands on the
FPGA host. It remains planned, with no workload or clock measurement claimed.
The active serial qualification observer is recorded in
[conditional_90mhz_launch.json](attempt03/conditional_90mhz_launch.json).

[Qualification priority change](attempt03/qualification_priority_90mhz.json)
retains the currently running original 100 MHz diagnostic but defers the
not-yet-started cache-cut DCP load until after the conditional functional
baseline. The original watcher parent was stopped; its owned report continues.
A lightweight controller marks that historical report terminal, then the
existing operating-image collector proceeds. No implementation, running report
or qualification check was removed. This avoids an extra checkpoint load on
the critical path to V. Cache-cut evidence remains required before any later
root-cause/frequency optimization decision.

[Read-only load preflight](attempt03/load_preflight_90mhz.json) passes
platform/interface compatibility, device-idle, loader exclusion and thermal
checks on the actual complete image. The image remains unloaded.
[Runtime metadata preflight](attempt03/packaged_hbm_preflight_90mhz.json)
resolves all three kernel arguments to the same enabled 8 GiB group; an
illustrative 1 GiB range spans physical PCs 0..3. This is metadata evidence,
not an actual BO allocation, memory-window measurement or traffic pass.

[Critical-path release](attempt03/diagnostic_release_for90.json) supersedes
the earlier decision to let the extra historical report finish. Its owned
report group was stopped after preserving partial outputs, allowing mandatory
90 MHz qualification to proceed. The original implementation, routed checkpoint,
complete image, original routed timing and all 90 MHz qualification checks are
unchanged. Detailed 100 MHz cones/cache cuts are deferred until after the
conditional functional baseline, before any frequency-optimization decision.

[Live operating audit](attempt03/operating_audit_live_90mhz.json) confirms
Vivado PID 2192863 actually executing the serial 90 MHz timing/route/CDC/
placement report. Packaged clock topology and both preserved artifact hashes
are retained. This is a live report process, not a qualification result.
[Prepared execution environment](attempt03/conditional_delivery_ready.json)
records a compiled native Torch extension, matching frozen software tree,
zero opened sessions, and the exact post-qualification launch script. Its
single low-priority preparation worker performed no RTL/TLM run or board open.
The script inherits that prepared cache and remains unexecuted pending P90.

**Operating-audit guard correction.** The first report stopped before producing
operating timing/CDC evidence, at its generated-clock PERIOD identity guard;
[the failure is retained](attempt03/operating_audit_v1_guard_failure.json).
[Installed Vivado 2023.2 API probes](attempt03/clock_property_probe_vivado2023_2.json)
show PERIOD=0.000 before `update_timing`, then 11.111 after update for the
requested 11.111111... ns period. The correction updates timing before reading
the property and compares within its displayed 0.001 ns quantum. It does not
relax setup/hold slack or permit a different operating clock. Regression checks
cover deferred properties, three-decimal precision and a wrong-frequency refusal.
[The v2 report-only launch](attempt03/operating_audit_v2_launch.json) uses a fresh
output directory and the same immutable routed checkpoint. No implementation
or programming was performed; P90 still awaits actual report review.

The current authoritative v2 state is
`~/tnca_reports_r40_operating_v2/state.json`, with progress in `runner.log`.
Inspect this fresh report rather than the preserved v1 failure state under
the image-preservation directory. Success proceeds to full report review,
then the explicit load and actual clock capture. Any actual timing/CDC/placement
failure preserves the usable baseline and is diagnosed before further action.
The unchanged three-block, 77-point CPU/native device-time collector now accepts
the same reviewed operating-clock evidence as the functional runners; its
warmups, timed samples, independent blocks and frozen definitions are unchanged.

**P90 — conditional operating path within the authorized 91 MHz ceiling.**
Unique blocker: qualification of the preserved image at its actual operating
clock. Direct evidence: the complete image has packaged 90 MHz metadata but no
board-clock measurement or reviewed 90 MHz timing/CDC report. Next command:
inspect the serial operating-image audit state and reports. If timing/hold,
route, CDC and physical SLR placement pass, load the preserved image and verify
the actual clock; then execute V with `--operating-clock-evidence`. If they fail,
retain the baseline and repair that specific qualification issue without a new
implementation. The earlier planned 91 MHz report is superseded by analysis
matching the actual packaged 90 MHz. For conditional runs, the P prerequisite
in the unresolved-gate table means P90; P/100 MHz remains separately failed.
All downstream workload gates remain not-measured.

[Immutable conditional software](attempt03/conditional_software_12a0340.json)
is already staged on the FPGA host, with the contract, software-tree and archive
hashes verified. No device was opened. It includes continuous original >=64 MiB
checkpoint -> native77 -> HBM14 -> frozen-model execution and shared measured
clock provenance. The original graph sizes, numerical checks and traffic/
visibility assertions are unchanged. This is preparation, not workload evidence.

Hardware source is `54249f7f7ac0cf8f5d1b82d8e63a3046d27522d3`, with the explicit
immutable configuration and archive identity in [attempt03](attempt03/README.md).
The frozen contract is unchanged. No final-image timing, CDC, HBM or workload
result is claimed; missing measurements remain not-measured.

[Routed timing evidence](attempt03/routed_summary_20261018.json) binds the
completed route and routed timing-report SHA. P's current blocker is failed
setup in the actual routed candidate. The original build had no terminal
marker at observation, and CDC/image qualification remains outstanding.
Next experiment: run the existing terminal observer command below to extract
final min/max paths, clock/CDC evidence and actual cache mapping after the
original implementation exits. If the final audit passes all requirements,
review immutable image/source identity and proceed to V. If setup remains
negative, diagnose the actual limiting paths, validate a source-bound correction
and a matched physical screen before spending the final implementation attempt.
Do not load the timing-failing candidate as a qualified 100 MHz image.

[Route-start evidence](attempt03/route_started_20261018.json) confirms the
original implementation PID is live, physical optimization completed, and
the route precondition DRC finished with zero errors. Routing has started;
this is not a final route/DRC, WNS, CDC or image result. The current experiment
and success/failure branches below remain unchanged. No additional full
implementation has been launched.

[Post-hold/intermediate timing](attempt03/post_hold_intermediate_20261018.json)
retains the live original PID and raw intermediate summaries. Setup WNS was
-1.300 ns after hold repair and improved to -1.006 ns during skew/delay cleanup;
the latter summary still has TNS -7278.353 ns, WHS -0.027 ns and THS -0.080 ns.
These are intermediate, unqualified results, not a final 100 MHz pass or terminal
failure. P remains blocked by final route/timing/CDC/image evidence. The next
command is the existing terminal observer below. Success proceeds to identity
review and V; failure requires the actual final limiting paths before spending
the one remaining implementation attempt. No large DCP reader runs alongside
the active route process.

[Read-only device preflight](attempt03/preload_readonly_20261018.json) finds
the U280 healthy and idle, with no asynchronous error or firewall fault. It
records the pre-existing image UUID, actual clock table and memory inventory.
No memory bank is reported enabled in that existing image, and no buffer
objects are allocated. These are not attempt03 measurements. Final loading
remains conditional on its legal route, timing, CDC, DRC and source review;
actual multi-PC allocation and traffic must be measured after that load.


[Live placement evidence](attempt03/placement_observation_20261018.json)
records the running implementation and observer, absent terminal marker and
log identity. Two critical warnings remain for final review: a generated
DONT_PARTITION query matched no objects, and an HDPR-5 severity downgrade was
reported. [Generated constraint evidence](attempt03/generated_constraint_preflight_20261018.json)
traces HDPR-5 to the Vitis-generated pre-place script. The generated kernel
clock constraint uses a 12/12 generated-clock ratio, rather than a new
independent primary clock. Actual timing and clock roots still require final
report review. Neither critical warning is classified as harmless or terminal
without the final DRC and hierarchy evidence. No large DCP is opened beside the live placer.

**P — qualify the same final three-SLR build.** The unique blocker is the
absence of legal full-image route, 100 MHz min/max timing, CDC and a loadable
image. Direct evidence is [the launch manifest](attempt03/launch.json), whose
image SHA is null and qualifications are not-measured. Attempt 2's terminal
failure remains in [its raw record](attempt02/terminal_failure.json).
Next experiment on the FPGA host:

```sh
tail -n 30 ~/tnca_build/r40_attempt03_20261017/xo/link.log
cat ~/tnca_build/r40_attempt03_20261017/build/attempt.exit
cat ~/tnca_build/r40_attempt03_followup_cachecuts/state.json
```

[Actual placement completion](attempt03/placement_complete_20261018.json)
records `place_design completed successfully`, followed by `phys_opt_design`.
The retained post-placement SLR report shows CLB occupancy
95.24% / 88.39% / 85.41%. SLR0 remains tightly packed. Boundary SLL usage is
61.74% for SLR0/1 and 35.03% for SLR1/2; these aggregates do not establish
per-column capacity or legal routing. No timing or route gate is promoted.

The terminal marker is absent during execution. The existing read-only observer
collects reports after terminal completion. Success requires actual three-core
SLR placement, legal route, all-clock setup/hold, CDC and a complete image hash;
then proceed to V. Failure preserves the checkpoint, actual limiting paths and
route errors. The one remaining attempt requires a source-bound root-cause
correction and matched validation before launch; no retry is automatic.

[Prerequisite evidence](attempt03/prerequisites.json) retains the routed
predecessor module: WNS +0.361 ns, WHS +0.010 ns, zero TNS/THS and route errors,
96 RAM32M16 tag roots and actual local TMU state in SLR0/1/2. Its new worst
sampled path is MSHR index to data RAM, 18 levels / 9.203 ns (1.446 logic /
7.757 routed net). Sparse module timing excludes cores, NoC, HBM and timed I/O.
The final owner-mux completion passed exact data/hint/counter cycle comparisons;
the exact final mapped netlist has zero direct metadata-to-notice paths, with
nonempty metadata, notice and owner-snapshot endpoint sets. This final completion
has synth/opt evidence, not a second module route; full-image route is in progress.

**V — original unscaled Torch checkpoint after P.** Its unique blocker is the
qualified final image. The immutable software and concrete board command below
remain unchanged. Success must demonstrate numerical correctness, all three
physical cores/outlets, actual multi-PC HBM, ownership and drained visibility
for the original >=64 MiB workload; then run native77, HBM14 and frozen models.
Failure preserves the first incorrect result or missing traffic/visibility
requirement without shrinking the workload.

CI for exact source `54249f7` completed with 12 successful jobs and one
cancelled job: its serial MMIO/native sweep exceeded the 120-minute job cap.
[Raw cancellation evidence](attempt03/exact_source_ci_cancelled.json) is
retained; this is not a CI pass. The original checks and six simulations are
being scheduled as independent lanes, preserving all assertions and arguments.
Successful source qualification is mandatory before board load.
[Replacement CI](attempt03/split_ci_started.json) is run `39087952712` on
commit `16914f7`; its RTL tree exactly matches attempt03. Its software also
contains endpoint-auditing changes, so it is not represented as a rerun of
commit `54249f7`. [Terminal success](attempt03/split_ci_terminal_success_20261018.json)
records all lanes successful. This resolves the source-CI prerequisite for that
specific software/RTL snapshot, not physical qualification or CI of later model
tooling commits. Board loading remains blocked by P. The next experiment is the
existing terminal route/timing/CDC collector above; success permits identity
review and V, failure retains the terminal reports and keeps loading blocked.
A complete original native77 functional run now
uses source a1fed22, the final-candidate RTL geometry and explicit testbench
guards on the ARM host; [launch evidence](attempt03/final_native77_launch.json).
Its new per-sample raw endpoint records are independently reproducible. This
is functional RTL only, not the final-image inventory or a performance gate.
The [live partial snapshot](attempt03/native77_partial_20261018.json) retains
the first observed failure: the unchanged original medium llama_decode point
exceeded a 7,200-second host wait. The unique blocker for this diagnostic is
unresolved forward progress at that wait limit. Next command on the ARM host:
`tail -n 30 ~/r40_final_native77_a1fed22_20261017/native77_results/raw_logs/model.llama_decode__medium__fp16-b1-p256.log`.
If completion/progress evidence establishes host slowness, retain this timeout
and rerun the original point under a separately identified wait budget; if a
protocol stall or wrong result is found, retain its journal and reproduce it
before any candidate change. A busy simulator process alone proves neither
forward progress nor correctness.
[Original-point diagnosis](attempt03/llama_progress_started.json) now runs in
a separate immutable source directory on the ARM host. It preserves the
original final geometry and point hash, enables the existing timed mailbox
journal, and records read-only instruction/program progress every 65,536
clocks. The diagnostic harness compiled successfully and its live log emitted
the first progress record. No DUT signals are driven by this monitor. Device
instruction overhead is zero; host overhead is not measured, so the run is
excluded from performance comparisons. Next commands:

```sh
cat ~/r40_final_llama_progress_20261018/results/runner.exit
tail -n 12 ~/r40_final_llama_progress_20261018/sim/build/u50-sim/link-*/regs.log
```

Success requires the unchanged original numerical comparison. If it still
times out, preserve launch IDs, instruction/program progress and the journal
to isolate a simulator-rate limit or protocol/kernel stall. Neither a busy
process nor advancing time alone clears the original failure.

The earlier R3 run is retained separately. The earlier full R3 native77 simulation still has
original points exceeding the 1800-second host wait. Those rows remain timeouts.
The unchanged original argmax diagnostic now passes with exact integer reference
equality; its steady call took 3,200.963 host seconds.
[Raw terminal evidence](native_argmax_timeout_terminal.json) retains both the
original timeout and successful longer-wait result. Neither result substitutes
for the final-candidate or final-image inventory.

## Latest detailed diagnosis and candidate screening

The read-only [actual failed-DCP detail](attempt02/physical_detail.json)
confirms 96 RAM32M16 tag roots, 1,272 tag-output registers and these sampled
cuts: metadata-to-owner 7 levels/12.342 ns, tag-to-owner 9/10.849 ns,
owner-to-writeback 31/17.752 ns, request-owner-to-TLW 16/7.595 ns.
These are actual failed-DCP path delays, not legal-route qualification.

The sampled tile-boundary audit finds 2,331 canonical crossing nets and
actual LAG route nodes for 2,330. TMU contributes 1,600 of this boundary
sample, including 1,439 nets spanning all three SLRs. Actual physical
`LAG_LAG` routing-tile X attribution is retained separately from driver
SLICE X and the router's printed capacity-column indices. In the three-SLR
TMU sample, tile X87 is used by 778 unique nets and X96 by 468; a net may
use multiple tile Xs. These are connected-net counts, not channel demand,
and this tile-boundary sample does not cover every NoC/platform crossing.
The tag/cut/boundary reports completed; the final HBM endpoint query failed
to resolve a leaf driver and remains explicitly incomplete. Its exact failed
query and all prior reports are retained; no implementation was rerun.

The exact production response-guidance helper now passed a real small-DCP
API probe (517 pairs/1,034 attributes, exit 0), after correcting its collection
handling and a probe-only success-marker error. Both failures are preserved.
The matched routed Laguna experiment remains separate from that API probe.

[Follow-up candidates](p0_selected_victim_tlw/followup_report.json) retain
successful actual-geometry V1-select/shared-TMU integration regressions,
plus independent registered-pair-ranking and coherent local-TMU unit/mutation
checks. The V1 integrated result does not endorse the newer composition.
A new bounded module screen at
`~/tnca_build/r40_local_tmu_rank_module_20261017` combines local coherent
TMU state, raw/classified TLW registers, first-pair ranking registers and
selected-way registers, while retaining the same geometry, 10 ns clock and
directive. Its hypothesis addresses the actual broadcast and owner cones
above. This is still module screening, not the reserved full implementation.
No required acceptance gate has changed.

## Earlier P0/P1 evidence (historical identities retained)

**P0/P1 physical convergence baseline:**
`attempt01/physical_postmortem.tar.gz` retains all failed-checkpoint reports
and the actual-endpoint-region boundary audit. The worst sampled cone is
TMU `tb_reg[0][63]` to slice writeback-address storage: 53 logic levels,
28.999 ns data delay (3.447 ns logic, 25.552 ns net). This is substantially
unrouted diagnostic timing, never 100 MHz qualification. The 30 sampled
endpoints collapse to three logical source/storage families in
`critical_cones_ranked.json`; this sample is not exhaustive.
Current CLB occupancy is 95.83%/90.81%/87.37% for SLR0/1/2.
`cross_slr_modules_ranked.tsv` ranks 4,963 unique connected nets with
endpoints in multiple actual SLRs; it does not count routed SLL capacity.
The largest sampled driver group is tile 2's slice (2,536 nets between
SLR1/2); TMU metadata is also prominent. The failed snapshot's core and slice internals are each physically confined to their assigned SLR,
with no unplaced nonconstant cells; this does not qualify the newer candidate. `cross_slr_driver_columns.tsv` reports driver SLICE X positions,
**not crossing columns**. `sll_estimated_columns.tsv` separately preserves
the router's printed per-column demand estimates, including 203%; mapping
specific buses onto physical SLL columns is still not-measured.

The active P0 hypothesis is per-way synchronous tag RAM plus an E_LOOK
owner register boundary, with explicit TMU commit/clear epoch retry before
E_MISS effects. Independent functional and matched 10 ns TMU+slice screens
must report the cut boundary, new worst path/levels/logic/net delay and
actual tag primitives. No OOC timing or mapping result has yet qualified
this candidate. **Screen success:** review those results together with
specific SLL hotspot corrections and actual pipeline/core locations before
considering the final attempt. **Screen failure:** retain the counterexample
or critical cone and correct its cause; do not spend the remaining attempt.

The completed [P0 screen](p0_pipeline/README.md) now establishes both mapped
register boundaries, actual distributed tag RAM and a same-source balanced
victim comparison: 49 to 34 levels, 12.476 to 8.579 ns data delay, estimated
WNS -2.719 to +1.178 ns. This is an unplaced setup screen, not a P gate pass.
The pipeline-only candidate's retained downstream failure motivated that change.
The source-bound function and mutation results are retained with the reports.

The [physical boundary audit](attempt01/physical_boundaries_v2.json) rules out
core/slice SLR spill in the failed image and finds only six kernel DONT_TOUCH
code-link objects; the much larger protected population belongs to the platform.
No protection is removed. Its LAG-node query recovered no actual crossing
columns, so that attribution remains not-measured. The focused P1 read-only
probe separates actual pipeline state from combinational cells and traces the
sampled shared slice2 logic to its sequential producers and consumers.
**Success:** use those facts to screen a concrete SLL correction and audit the
new final candidate. **Failure:** preserve the missing information; do not
replace it with a placement-property check or spend the last attempt by default.

**Full-implementation budget:** the original two additional attempts, one
further attempt authorized on 2026-10-17, and one final additional attempt
authorized on 2026-10-18 (maximum 4, used 3, remaining 1). The exact
launch identity, state and remaining balance are recorded in
`implementation_budget.json`; report extraction and module-level screening
are not full implementations. The contract remains frozen.

**S — registered-range screening is complete.** The source-bound functional,
mutation and compositional proof package is in
[registered_ranges](../../R40-06/registered_ranges/README.md). A paired actual
TMU-plus-slice OOC screen reduced LUTs **83,353 → 78,543**, but estimated WNS
improved only **-6.333 → -6.037 ns**. Both variants still have the worst path
from `slice/cur_addr_reg[7]` to writeback-address RAM write enable, with
essentially unchanged logic delay. This supports an area reduction. The completed linked diagnosis and the
parameter-only geometry screen support the bounded first attempt above: success determines whether the local cone also
limits the complete design; failure preserves the local result without
attributing the linked congestion to it. No operating-frequency, graph-speedup
or gate claim follows from OOC estimates.

**V — vertical checkpoint after P yields a qualified loadable image.** From the verified immutable archive on the FPGA server:

```sh
cd ~/tnca_srcR40checkpoint_642c660
source /opt/xilinx/xrt/setup.sh
export TORCH_EXTENSIONS_DIR="$HOME/.cache/tnca/r40_checkpoint642_extensions"
PYTHONPATH=python:compiler/python:tools/asm CUDA_VISIBLE_DEVICES= \
  ~/tc_env/torchenv/bin/python tools/r40/board_checkpoint.py \
  --xclbin "$R40_FINAL_XCLBIN" --image-sha256 "$R40_FINAL_IMAGE_SHA256" \
  --software-tree-sha256 "$R40_FINAL_SOFTWARE_TREE" --out "$R40_CHECKPOINT_OUT"
```

These four variables must be bound in the retained run manifest before execution: actual final image path/full SHA, archive's exact `files:<sha256>` software tree, fresh output directory. They are intentionally not bound to the currently loaded single-core image. The runner refuses a mismatched image/software identity, fewer cores/outlets, an insufficient physical window, or a changed frozen point. It runs the unchanged `op.add/hbm/fp16-n67108864` point (two 128 MiB inputs plus output), checks numerical tolerance, three physical cores, all actual outlets before CPU-read maintenance, drained completion events/quiesce, and successful host ownership handoff before numerical readback. Raw DMA without ownership handoff is not certified. Mapping/buffer ranges show actual PCs intersected; full PC functional boundary/alias coverage remains separate. No sample from V is a full gate or a performance result. **Success:** unlock full inventories and frozen experiments. **Failure:** preserve report including partial counters/events and fix the specific first failing assertion; no reduced shape substitute.

## Every currently unpassed required gate

All rows are `not-measured`. Each row has one immediate missing prerequisite, its direct evidence, the next experiment, and both outcomes. P90 and V have passed. I is the full original inventory/model rerun after the documented child source-origin repair. Running a Make target alone does not supply absent board/performance evidence.

| Required gate | Unique current blocker | Direct evidence | Next experiment | If successful | If failed |
|---|---|---|---|---|---|
| R40-01.mixed-device | 最终快照/镜像上的 mixed-device 拒绝证据缺失 | `required_gates.json`: sim/build/r40/v2/R40-01.json does not name gate R40-01.mixed-device | I | After I, collect frozen `make r40-device` package evidence on final identity; judge this gate separately | Preserve I first failing evidence; fix that cause; gate remains not-measured |
| R40-01.device-time | 同一最终身份的三组随机交错 CPU/native 5/30 配对记录缺失 | `required_gates.json`: sim/build/r40/v2/R40-01.json does not name gate R40-01.device-time | I | After I, collect frozen `make r40-device` package evidence on final identity; judge this gate separately | Preserve I first failing evidence; fix that cause; gate remains not-measured |
| R40-02.zero-transfer | 最终镜像完整冻结 resident 图零搬运证据缺失 | `required_gates.json`: no evidence at sim/build/r40/v2/R40-02.json | I | After I, collect frozen `make r40-residency` package evidence on final identity; judge this gate separately | Preserve I first failing evidence; fix that cause; gate remains not-measured |
| R40-02.resident | 冻结 graphs3 的 staging/resident 匹配 A/B 缺失 | `required_gates.json`: no evidence at sim/build/r40/v2/R40-02.json | I | After I, collect frozen `make r40-residency` package evidence on final identity; judge this gate separately | Preserve I first failing evidence; fix that cause; gate remains not-measured |
| R40-02.batched | 冻结小核图 single/batched 匹配 A/B 缺失 | `required_gates.json`: no evidence at sim/build/r40/v2/R40-02.json | I | After I, collect frozen `make r40-residency` package evidence on final identity; judge this gate separately | Preserve I first failing evidence; fix that cause; gate remains not-measured |
| R40-02.stress | 最终镜像长期 replay/恢复/代码淘汰证据缺失 | `required_gates.json`: no evidence at sim/build/r40/v2/R40-02.json | I | After I, collect frozen `make r40-residency` package evidence on final identity; judge this gate separately | Preserve I first failing evidence; fix that cause; gate remains not-measured |
| R40-03.tiles | 最终镜像宽 block 与 tile 冻结功能库存缺失 | `required_gates.json`: no evidence at sim/build/r40/v2/R40-03.json | I | After I, collect frozen `make r40-tiles` package evidence on final identity; judge this gate separately | Preserve I first failing evidence; fix that cause; gate remains not-measured |
| R40-03.wide-blocks | 冻结 pointwise/reduction 宽 block 匹配 A/B 缺失 | `required_gates.json`: no evidence at sim/build/r40/v2/R40-03.json | I | After I, collect frozen `make r40-tiles` package evidence on final identity; judge this gate separately | Preserve I first failing evidence; fix that cause; gate remains not-measured |
| R40-03.gemm-tile | 冻结 GEMM reuse 的 tile 匹配 A/B 缺失 | `required_gates.json`: no evidence at sim/build/r40/v2/R40-03.json | I | After I, collect frozen `make r40-tiles` package evidence on final identity; judge this gate separately | Preserve I first failing evidence; fix that cause; gate remains not-measured |
| R40-04.hbm | 实际 ≥1 GiB 多 PC 窗口及完整原始 HBM 库存未完成验证 | `required_gates.json`: no evidence at sim/build/r40/v2/R40-04.json; `evidence/r40/R40-04/physical_map_v2/` proves legacy image only exposes 256 MiB / PC0 | I | After I, collect frozen `make r40-hbm` package evidence on final identity; judge this gate separately | Preserve I first failing evidence; fix that cause; gate remains not-measured |
| R40-04.bandwidth | 最终镜像方向带宽及 memory-service outstanding 1/8 匹配记录缺失 | `required_gates.json`: no evidence at sim/build/r40/v2/R40-04.json | I | After I, collect frozen `make r40-hbm` package evidence on final identity; judge this gate separately | Preserve I first failing evidence; fix that cause; gate remains not-measured |
| R40-04.robustness | 最终实际分配 PC 的边界/high64/alias/越界证据缺失 | `required_gates.json`: no evidence at sim/build/r40/v2/R40-04.json | I | After I, collect frozen `make r40-hbm` package evidence on final identity; judge this gate separately | Preserve I first failing evidence; fix that cause; gate remains not-measured |
| R40-05.multicore | P90 已通过；最终镜像三核运行功能证据尚未完成 | `attempt03/launch.json`: conditional_physical_checkpoint.json passes P90; V passed; I source-origin repair is required | I | After I, collect frozen `make r40-multicore` package evidence on final identity; judge this gate separately | Preserve I first failing evidence; fix that cause; gate remains not-measured |
| R40-05.scaling | 最终镜像冻结 compute 的 1/2/3 核 A/B 缺失 | `required_gates.json`: no evidence at sim/build/r40/v2/R40-05.json | I | After I, collect frozen `make r40-multicore` package evidence on final identity; judge this gate separately | Preserve I first failing evidence; fix that cause; gate remains not-measured |
| R40-05.dispatch | 最终镜像原始大图的三物理核完成/分发计数缺失 | `required_gates.json`: no evidence at sim/build/r40/v2/R40-05.json | I | After I, collect frozen `make r40-multicore` package evidence on final identity; judge this gate separately | Preserve I first failing evidence; fix that cause; gate remains not-measured |
| R40-06.nuca | 最终镜像 active outlet 的实际负载流量证据缺失 | `required_gates.json`: no evidence at sim/build/r40/v2/R40-06.json | I | After I, collect frozen `make r40-nuca` package evidence on final identity; judge this gate separately | Preserve I first failing evidence; fix that cause; gate remains not-measured |
| R40-06.streaming-scaling | 冻结 streaming 的 1/3 核匹配 A/B 缺失 | `required_gates.json`: no evidence at sim/build/r40/v2/R40-06.json | I | After I, collect frozen `make r40-nuca` package evidence on final identity; judge this gate separately | Preserve I first failing evidence; fix that cause; gate remains not-measured |
| R40-06.locality | 冻结 LOCALITY/DCO 开关尚未接入标准 Torch 编译/分配路径 | `pending_switch_consumers.json`; `frontend.py` has only explicit DCO importer input | `rg -n -e TNCA_LOCALITY -e TNCA_DCO -e 'dco=' python/tnca/aot.py python/tnca/inductor.py python/tnca/triton_backend/compiler.py` | If a consumer exists, trace emitted registrations and bytes on V-qualified hardware | If absent, connect standard-path DCO/locality within R40-06 after V; keep gate open |
| R40-06.semantics | 最终镜像 ownership/完成可见性纵向闭环缺失 | `required_gates.json`: no evidence at sim/build/r40/v2/R40-06.json | I | After I, collect frozen `make r40-nuca` package evidence on final identity; judge this gate separately | Preserve I first failing evidence; fix that cause; gate remains not-measured |
| R40-07.pipeline | 最终镜像标准 Torch 图中的实际流水重叠证据缺失 | `required_gates.json`: no evidence at sim/build/r40/v2/R40-07.json | I | After I, collect frozen `make r40-pipeline` package evidence on final identity; judge this gate separately | Preserve I first failing evidence; fix that cause; gate remains not-measured |
| R40-07.long-k | 最终合格镜像上的完整 long-K stage A/B 缺失 | `../../R40-07/frozen_stage_control/README.md`: frozen control reaches actual 1/2/3-stage Torch rings; final frozen performance evidence absent | I, then `make r40-pipeline` with final identity | Collect unchanged frozen shapes, hold roles constant, verify stage provenance and the frozen speedup threshold | Preserve first failing shape/program/numerical or protocol result; repair cause without shrinking workload; gate remains not-measured |
| R40-07.matrix | 冻结 compute-bound 的 matrix 匹配 A/B 缺失 | `required_gates.json`: no evidence at sim/build/r40/v2/R40-07.json | I | After I, collect frozen `make r40-pipeline` package evidence on final identity; judge this gate separately | Preserve I first failing evidence; fix that cause; gate remains not-measured |
| R40-07.policy | 最终镜像策略边界与资源拒绝证据缺失 | `required_gates.json`: no evidence at sim/build/r40/v2/R40-07.json | I | After I, collect frozen `make r40-pipeline` package evidence on final identity; judge this gate separately | Preserve I first failing evidence; fix that cause; gate remains not-measured |
| R40-08.attention | 修复后 attention 尚未在最终三核镜像完成冻结验证 | `required_gates.json`: no evidence at sim/build/r40/v2/R40-08.json; `evidence/r40/R40-08/descriptor_publication/` retains original failure, RAM snapshot and mutation | I | After I, collect frozen `make r40-attention` package evidence on final identity; judge this gate separately | Preserve I first failing evidence; fix that cause; gate remains not-measured |
| R40-08.prefill | 冻结 long-prefill 的时间及中间字节匹配 A/B 缺失 | `required_gates.json`: no evidence at sim/build/r40/v2/R40-08.json; `evidence/r40/R40-08/descriptor_publication/` retains original failure, RAM snapshot and mutation | I | After I, collect frozen `make r40-attention` package evidence on final identity; judge this gate separately | Preserve I first failing evidence; fix that cause; gate remains not-measured |
| R40-08.decode | 冻结模型原始 decode64 在最终镜像上的数值验证缺失 | `required_gates.json`: no evidence at sim/build/r40/v2/R40-08.json; `evidence/r40/R40-08/descriptor_publication/` retains original failure, RAM snapshot and mutation | I | After I, collect frozen `make r40-attention` package evidence on final identity; judge this gate separately | Preserve I first failing evidence; fix that cause; gate remains not-measured |
| R40-09.autotune | 标准路径 TNCA_TUNE 尚未连接冻结形状的真实候选选择 | `pending_switch_consumers.json`; compiled_candidates has actual compilation; autotune_native_tests uses a small graph | `rg -n -F -e TNCA_TUNE -e 'select(' python/tnca/inductor.py python/tnca/triton_backend/compiler.py` | Trace actual frozen-shape candidates into measured selection and cache, then frozen holdout | Connect frozen-shape selection using existing budget/workers after V; do not extend framework |
| R40-09.holdout | 冻结 14 点 holdout 的实际 tune/control 与 regret 记录缺失 | `required_gates.json`: no evidence at sim/build/r40/v2/R40-09.json; worker_deadline/formal_call is explicitly functional RTL only | I | After I, collect frozen `make r40-autotune` package evidence on final identity; judge this gate separately | Preserve I first failing evidence; fix that cause; gate remains not-measured |
| R40-09.suites | 最终不可变身份下完整原生77及HBM14记录缺失 | `required_gates.json`: no evidence at sim/build/r40/v2/R40-09.json; worker_deadline/formal_call is explicitly functional RTL only | I | After I, collect frozen `make r40-autotune` package evidence on final identity; judge this gate separately | Preserve I first failing evidence; fix that cause; gate remains not-measured |
| R40-09.model | 冻结四层原始模型在最终镜像的 1/3 核匹配记录缺失 | `required_gates.json`: no evidence at sim/build/r40/v2/R40-09.json; worker_deadline/formal_call is explicitly functional RTL only | I | After I, collect frozen `make r40-autotune` package evidence on final identity; judge this gate separately | Preserve I first failing evidence; fix that cause; gate remains not-measured |
| R40-10.acceptance | 上游 required gates 尚未全部通过 | `required_gates.json`: no evidence at sim/build/r40/v2/R40-10.json | `python3 tools/r40/acceptance.py --plan spec/r40/v2/plan.json --json /tmp/r40-v2-final.json` | Archive final report only if every required gate passes and final identities match | Keep acceptance open; work the first unsatisfied prerequisite |

## Execution order after V

1. Preserve descriptor failure reproduction and mutation; run the complete native77, complete HBM14 and frozen unscaled model on the same immutable source/image. Earlier native77 and attention14 functional results remain evidence, but attention14 is not HBM14 and neither substitutes for final-image inventories.
2. Use actual compiled candidates and the frozen holdout; keep the existing 16/kernel and 600 s/graph limits. The isolated-worker fix only moves supervisor compilation outside the formal sample; no more tuning-framework expansion is scheduled.
3. Collect each frozen A/B experiment with its prescribed warmups, samples, independent blocks, identity audit and confidence bounds. Bind software tree, harness, compiler versions, knobs/config, complete image SHA, measured/qualified clock and frozen contract SHA. Missing data stay not-measured.
4. Re-audit all final records against the image and source actually delivered. Do not recycle legacy-image timing into final signoff.

The packed-row optimization remains uncommitted experimental work and is excluded from immutable mainline archives. Its background RTL run is diagnostic only; it cannot promote a required gate.

Candidate archive and hardware qualification fields are recorded in `candidate_manifest.json`. The image hash and timing/CDC qualification remain absent until actually produced. The frozen active-core count knob is now wired to the runtime, with 1/2/3 counts mapping to masks 1/3/7. Its native RTL verification has completed for active counts 1/2/3; raw counters are in `evidence/r40/R40-05/active_core_knob/`. No board scaling gate is promoted. Real fixed-candidate compiler diagnostics for the deduplicated HBM/holdout inventories are retained in `evidence/r40/R40-09/compiled_candidates/`; all associated performance and suite gates remain not-measured.

R40-03.contract is now capability-pass from the complete disabled-option artifact audit at 85fa7d8; its full raw command, result, reference identity and source review are retained in `evidence/r40/v2/R40-03/contract/`. It is a software contract gate, so it does not depend on the pending routed image; R40-03.tiles and both performance gates remain not-measured.

## Necessary wiring after the vertical checkpoint

`pending_switch_consumers.json` is a source audit, not gate evidence. The remaining DCO/locality and standard-path tuning controls above are implementation prerequisites inside existing packages. Frozen TNCA_PIPELINE now selects actual 1/2/3-stage rings independently of TNCA_RCTX_ROLES; standard Torch functional validation and the default artifact guard passed. Final-image pipeline performance is still missing. Measured tuning integration currently covers the small graph. V remains the immediate cross-package checkpoint. After V, connect the mechanisms required by the frozen gates using existing implementations and retain failed candidates.

After V and full functional inventories, the concrete paired device-time collection command is:

```sh
PYTHONPATH=python:compiler/python:tools/asm CUDA_VISIBLE_DEVICES= \
  ~/tc_env/torchenv/bin/python tools/r40/device_time_v2_collect.py \
  --xclbin "$R40_FINAL_XCLBIN" --image-sha256 "$R40_FINAL_IMAGE_SHA256" \
  --software-tree-sha256 "$R40_FINAL_SOFTWARE_TREE" --out "$R40_DEVICE_TIME_OUT" --evidence
```

The collector fixes the complete frozen 77-point inventory, 5 warmups/30 samples, three blocks and frozen random order. Every independent subprocess report is preserved before judging. Full image SHA, compiler/Python versions and backend arguments participate in configuration matching. This command has not measured the final board. Success proceeds to the frozen non-regression audit; a collection failure preserves partial records and an audit failure preserves the actual verdict. Partial collection cannot pass.

The active V software snapshot is **642c660**, tree `files:ba3d9b0207b2520e1773e993b9ffce5db87410cc61237cfecf85908d29988613`, archive SHA `fab753e704ea5b56354f3d6ea01bb50877402a723abd2091100ff0557d56d802`. Both hosts verified this exact tree and frozen contract; paths and reconstruction command are in `checkpoint_snapshot_642c660.json`. FPGA Python 3.10.12, Torch 2.14.0+cpu and Triton 3.8.0 are available. Native device registration was prepared in a snapshot-specific extension cache without opening a device. Historical 8abf527/85fa7d8 archives and results remain attributed to those identities. This checkpoint snapshot includes the current runtime-memory/measurement fixes and pipeline control; it is not final R40 software qualification, and no legacy native77 or performance result is reattributed to it.

## Historical implementation and CI checkpoint (superseded by P above)

The same candidate advanced to Phase 4.1.1.3 Post Placement Timing Optimization; its original implementation process remains live. No implementation retry was started. Base post-route reports remain queued behind successful link completion. A second, sequential observer then queries actual placed primitive cells of each core using `get_slrs -of_objects`, records unplaced logic and excludes only unplaced GND/VCC netlist constants. It requires the three physical core hierarchies to occupy SLR0/1/2. This is independent of pblock intentions; it is not yet physical evidence. Its Tcl query API was checked against installed Vivado 2023.2 help. Collector refusal tests cover shared SLRs, missing cores, unplaced logic and preserved previous reports. The observers never mark a gate accepted or rerun implementation.

CI run 38679960975 failed only R36-BE-01: its broad Popen token check classified the generic measurement subprocess supervisor as simulator startup. The checker now permits only the single owned `run_measurement` worker call; simulator names anywhere in the module, additional launches, unowned groups and relocation into the runtime still fail mutation checks. The complete backend boundary suite passed against immutable execution source 8abf527 with the corrected checker loaded separately. No execution/RTL source or frozen contract changed. A fresh CI run is required before CI is reported green; this fixes no physical/performance gate by itself.

## Frozen-model validation readiness

`tools/r40/native_model.py` and `evidence/r40/R40-09/frozen_model_reference/` retain the unchanged full model, independent CPU reference comparison and a native validation runner. The current reference matches all original 64 tokens with 221,267,968 bytes of weights. Native execution remains conditional on qualified P and successful V; this is neither model execution nor performance gate evidence. The runner records resident calls, explicit validation transfers and ownership/completion histories, and never promotes a performance gate.

## Necessary HBM measurement correction

`evidence/r40/R40-04/measurement_boundary/` reproduces the v2 MMIO register-default bug on actual three-core RTL: legacy C0..D8 reads return zero despite positive traffic at all three outlets. The current backend correctly reports the legacy interface unavailable, and the HBM diagnostic uses opcode-17 outlet line counts. It captures traffic before CPU validation and keeps common-clock graph spans separate from summed kernel cycles. Request/control traffic outside launch endpoints is explicitly labelled; it is not certified HBM peak bandwidth. The active 642c660 checkpoint archive includes this correction; the historical immutable 8abf527 archive has not been modified. No required gate changes as a result of this diagnostic correction.

The [focused cache-cone audit](attempt01/cache_cone_ownership.json) resolves a
physical ownership issue: a sampled cache-private cone starts and ends in
slice2, but seventeen of its direct LUT neighbours are reconstructed under
TMU and physically placed in SLR1, confirmed independently from actual LOCs.
All192 queried pipeline state groups match their intended SLR assignments;
that does not establish locality of combinational logic moved outside those
named groups. A targeted cache-boundary comparison under rebuilt synthesis
is running before the remaining full attempt. The historical single-slice
OOC wrapper also omitted `maint_base`; the corrected input-complete screen
must replace its incomplete maintenance timing evidence. Required gate
statuses and actual performance remain unchanged/not-measured.

Source regression CI [38910228700](https://github.com/zzczzc20/TritonChip/actions/runs/38910228700)
completed successfully for `1df03591f933284a94370e7ecdf452e479ffebe7`, now
published to `r40`. This does not promote the hardware gates. The corrected
three-slice shared-TMU control has completed under rebuilt hierarchy: its
unplaced worst setup path is33 levels/8.383 ns, estimated WNS+1.374 ns,
but TMU combinational startpoints still include private tag-output/owner/age
registers from the caches. Positive unplaced timing therefore does not resolve
the observed physical ownership problem. The targeted boundary-preserved
variant is still running; no production hierarchy attribute has been applied.

## Candidate-worker source isolation

The existing R40-09 supervisor now has a
[real Torch worker bridge](../../R40-09/real_candidate_worker/README.md).
It fixes an observed stale compiler import, records actual module/model
origins and preserves compiler bundles plus complete raw endpoints. The
functional validation remains partial: one original tiny candidate completed,
the second reached the shared deadline. It does not implement standard-path
`TNCA_TUNE`, select a policy using holdout data, or replace the full frozen
inventories. Those required gates remain open.

## Conditional MSHR-service fallback

An isolated branch `r40-ms-service-snapshot` (commit `658c811a486f6c5c43691a69767f7887c2fa10c9`)
captures immutable service descriptors on the existing arbitration edge.
Cycle/oracle checks and mutation completed with explicit full tags, victim
ranking and actual cache/MSHR/target geometry. The motivating predecessor
path traverses RAM read, atomic ALU and RAM write data; a high-fanout read
address net contributes 2.608 ns. No new timing or resource improvement is
claimed, and the active attempt03 source is unchanged. Use this branch only
if the terminal full-image diagnosis supports that cause, followed by matched
physical screening. It has consumed no full-implementation reserve.
