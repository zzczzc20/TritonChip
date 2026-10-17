# R40 v2 delivery convergence ledger

This is a dated execution ledger, **not a contract amendment**. The frozen v2 contract and all thresholds remain unchanged. Latest physical observation: 2026-10-17 09:10 UTC; active vertical checkpoint software snapshot 642c660; previous failed hardware source 09ebe06; additional attempt 1 source cf330dc failed actual routing with global congestion level 7. Existing functional passes do not certify the final image. Required gate changes: R40-01.device and R40-03.contract now capability-pass; missing mixed-device/device-time entries are not-measured. All other required statuses are in required_gates.json. No performance gate has been promoted.

## Immediate experiment commands and branches

**P — diagnose the terminal failed route before choosing a replacement.** The
original `09ebe06` implementation stopped at 2026-10-15 14:53:10 UTC (22:53:10 HKT). Its
`build/link.exit` is **1** and PID 594866 is absent. Vivado reports
`Route 35-3: Design is not routable as its global congestion level is 7`,
followed by `Route 35-4445`. Raw log, terminal exit and checkpoint SHA are in
`route_failure_20261016/`; the retained error DCP is 595,473,030 bytes with SHA256
`8d82bd375ce489ca433aba15f83692b53b409bc045381a7a8f44ed44811a50d9`.
This was a failed implementation attempt, not a completed routed image.
Required gate statuses remain unchanged and final qualification is not-measured.

The existing `SSI_SpreadSLLs` directive did not resolve the problem. Earlier
local SLL demand reached 204%; placement CLB occupancy was
97.59%/96.54%/86.54%, despite LUT utilization of 84.08%/91.09%/75.83%.
The router identified at least 8,651 high-pin-utilization CLBs. These are
specific congestion observations, not proof that any one mechanism caused
all failures. The -38.938 ns intermediate WNS is not final routed timing.

**Current blocker:** no functionally validated, routable three-core final
candidate. Read-only checkpoint diagnosis completed successfully; all eight
reports and the exact script are in
`route_failure_20261016/diagnostic_reports_complete.tar.gz`. Thirty reported
endpoints reduce to two source registers: TMU base metadata and slice tag
storage, feeding writeback-address RAM enables. Worst reported data delay is
50.082 ns (91 logic levels); this includes estimates in a substantially
unrouted design and cannot qualify timing. The
[bounded candidate hypotheses](route_failure_20261016/next_candidate_hypotheses.md)
identify tag-bank/owner-pipeline screening and its correctness prerequisites.

**Candidate launch update:** the user requested starting the first of the two
remaining complete attempts before the session allowance expires. Candidate
parameters and the one-shot runner are in [attempt01](attempt01/README.md).
Registered ranges plus 32 sets/eight ways target the observed metadata and
packing pressure; the placement directive and HBM PC range are unchanged.
Snapshot 11 passed the runtime-control matrix, but snapshot 09's reservation-X
failure remains unexplained and is preserved in
[the runtime-control evidence](../../R40-04/runtime_memory_controls/README.md).
The hardware candidate predates the later decoder repair described below.
Mechanism passes cannot qualify its image; no required gate is promoted.

The new [timed mailbox capture](../../R40-04/runtime_memory_controls/timed_journal/README.md)
reproduced a reservation-X failure on the original four-configuration native
graph. Exact-harness replay matches the first assertion verbatim at the same
simulated time/core/context/PC. A separate first-unknown diagnostic replay is
launched against these fixed inputs; it consumes no full implementation budget.
The trace now locates illegal-opcode classification leaking into shared stall/fetch-valid control. A candidate repair passes the failing input prefix and core regressions; complete four-configuration native checks now pass with recording enabled and disabled. The full 77/14/model inventories and final-image evidence remain outstanding. The live cf330dc hardware build does not contain this later repair.

**Current P result:** additional attempt 1 has terminal exit 1. Its actual
route failed with global congestion level 7; no qualified image exists.
`attempt01/terminal_failure.json` binds the vendor failure and retained error
DCP (`ed1facba…90360e8`). The read-only observer completed all eight diagnostic reports without retry.
Registered ranges and half the cache sets did not solve routability.

**Final-candidate preflight:** the complete three-slice/shared-TMU paired
rebuilt-synthesis and post-optimization screen is retained in
[attempt02](attempt02/README.md). Preserving only the cache hierarchy removes
all sampled private-register combinational consumers outside each cache after
`opt_design` (4,144/5,208/4,153 -> 0/0/0), at +63 LUT/+15 FF. Actual failed-DCP
site queries establish that such private logic previously detoured through
SLR1 from SLR2. This is the structural correction for the final attempt;
module estimates do not promote any full-image gate.

**Current candidate is routing; one further full attempt is reserved.**
Immutable hardware source `0c39793` started at 2026-10-16 22:33:29 UTC.
At 2026-10-17 09:10 UTC, placement and physical optimization were complete,
and route was running at Phase 4.1 Global Iteration 0. Neither the terminal
marker nor a route error marker existed. The matched implementation process
was live. [attempt02/routing_observation.json](attempt02/routing_observation.json)
binds the captured original log prefix and placed utilization reports to the
hardware source, configuration, 100 MHz target and frozen contract.
No required gate is promoted.

Actual placed CLB occupancy is 93.53%/87.70%/86.07% for SLR0/1/2; unique
control sets are 4,178/5,066/2,612. Placed connectivity reports 9,306
SLR1–2 and 14,041 SLR0–1 crossings. These are distinct from the router's
estimated SLL demands of 16,541 and 16,082. The router's hottest SLR1–2
column is estimated at 3,062/1,440 (213%); the three hottest adjacent printed
columns span 199%–213%. Total demand below aggregate capacity does not resolve
this localized overload. The latest reported intermediate setup/hold
slacks are -0.952/-0.176 ns; they are not final qualification. Bus-to-actual
SLL-column attribution and final fmax remain not-measured.

**Current physical blocker:** no terminal complete route, 100 MHz setup/hold,
CDC or loadable-image qualification for this candidate. **Direct evidence:**
`attempt02/routing_observation.json` and its raw reports above.
**Next command:** `cat ~/tnca_build/r40_attempt02_followup/state.json` on the
FPGA host. The existing observer waits for terminal completion before opening
any DCP. **Success:** review actual placement, complete route, 100 MHz
setup/hold, CDC and image identity before V. **Failure:** retain terminal
logs and DCP; after the observer completes, use the staged diagnostic tools
for actual crossing contributors, register locations, tag mapping and
critical-path logic/net delay. Screen a correction to the observed failure
mechanism before considering the reserved attempt. No automatic retry.
The vertical and frozen-inventory gates remain blocked by the unqualified
image; the exact V command and per-gate success/failure branches below apply.

The user explicitly authorized one further full implementation on
2026-10-17 after the previous two launches. **Budget: maximum 3, used 2,
remaining 1; third attempt not launched.** This is a reserve, not an instruction
to consume it. Current routing must finish; the current image may still
qualify. If it does not, require a diagnosed physical cause, corresponding
structural correction, matched functional/resource/timing screens and a new
immutable candidate before spending this final reserve. The frozen contract,
three physical cores, 100 MHz and original workload sizes remain unchanged.

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

**Full-implementation budget:** the original two additional attempts plus one
further attempt authorized on 2026-10-17 (maximum 3, used 2, remaining 1). The exact
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

All rows are `not-measured`. Each row has one immediate missing prerequisite, its direct evidence, the next experiment, and both outcomes. P is the current runnable experiment for final-image-dependent gates; V is conditional on P qualification. Running a Make target alone does not supply absent board/performance evidence.

| Required gate | Unique current blocker | Direct evidence | Next experiment | If successful | If failed |
|---|---|---|---|---|---|
| R40-01.mixed-device | 最终快照/镜像上的 mixed-device 拒绝证据缺失 | `required_gates.json`: sim/build/r40/v2/R40-01.json does not name gate R40-01.mixed-device | P → V | After V, collect frozen `make r40-device` package evidence on final identity; judge this gate separately | Preserve P/V first failing evidence; fix that cause; gate remains not-measured |
| R40-01.device-time | 同一最终身份的三组随机交错 CPU/native 5/30 配对记录缺失 | `required_gates.json`: sim/build/r40/v2/R40-01.json does not name gate R40-01.device-time | P → V | After V, collect frozen `make r40-device` package evidence on final identity; judge this gate separately | Preserve P/V first failing evidence; fix that cause; gate remains not-measured |
| R40-02.zero-transfer | 最终镜像完整冻结 resident 图零搬运证据缺失 | `required_gates.json`: no evidence at sim/build/r40/v2/R40-02.json | P → V | After V, collect frozen `make r40-residency` package evidence on final identity; judge this gate separately | Preserve P/V first failing evidence; fix that cause; gate remains not-measured |
| R40-02.resident | 冻结 graphs3 的 staging/resident 匹配 A/B 缺失 | `required_gates.json`: no evidence at sim/build/r40/v2/R40-02.json | P → V | After V, collect frozen `make r40-residency` package evidence on final identity; judge this gate separately | Preserve P/V first failing evidence; fix that cause; gate remains not-measured |
| R40-02.batched | 冻结小核图 single/batched 匹配 A/B 缺失 | `required_gates.json`: no evidence at sim/build/r40/v2/R40-02.json | P → V | After V, collect frozen `make r40-residency` package evidence on final identity; judge this gate separately | Preserve P/V first failing evidence; fix that cause; gate remains not-measured |
| R40-02.stress | 最终镜像长期 replay/恢复/代码淘汰证据缺失 | `required_gates.json`: no evidence at sim/build/r40/v2/R40-02.json | P → V | After V, collect frozen `make r40-residency` package evidence on final identity; judge this gate separately | Preserve P/V first failing evidence; fix that cause; gate remains not-measured |
| R40-03.tiles | 最终镜像宽 block 与 tile 冻结功能库存缺失 | `required_gates.json`: no evidence at sim/build/r40/v2/R40-03.json | P → V | After V, collect frozen `make r40-tiles` package evidence on final identity; judge this gate separately | Preserve P/V first failing evidence; fix that cause; gate remains not-measured |
| R40-03.wide-blocks | 冻结 pointwise/reduction 宽 block 匹配 A/B 缺失 | `required_gates.json`: no evidence at sim/build/r40/v2/R40-03.json | P → V | After V, collect frozen `make r40-tiles` package evidence on final identity; judge this gate separately | Preserve P/V first failing evidence; fix that cause; gate remains not-measured |
| R40-03.gemm-tile | 冻结 GEMM reuse 的 tile 匹配 A/B 缺失 | `required_gates.json`: no evidence at sim/build/r40/v2/R40-03.json | P → V | After V, collect frozen `make r40-tiles` package evidence on final identity; judge this gate separately | Preserve P/V first failing evidence; fix that cause; gate remains not-measured |
| R40-04.hbm | 实际 ≥1 GiB 多 PC 窗口尚无合格三核镜像可验证 | `required_gates.json`: no evidence at sim/build/r40/v2/R40-04.json; `evidence/r40/R40-04/physical_map_v2/` proves legacy image only exposes 256 MiB / PC0 | P → V | After V, collect frozen `make r40-hbm` package evidence on final identity; judge this gate separately | Preserve P/V first failing evidence; fix that cause; gate remains not-measured |
| R40-04.bandwidth | 最终镜像方向带宽及 memory-service outstanding 1/8 匹配记录缺失 | `required_gates.json`: no evidence at sim/build/r40/v2/R40-04.json | P → V | After V, collect frozen `make r40-hbm` package evidence on final identity; judge this gate separately | Preserve P/V first failing evidence; fix that cause; gate remains not-measured |
| R40-04.robustness | 最终实际分配 PC 的边界/high64/alias/越界证据缺失 | `required_gates.json`: no evidence at sim/build/r40/v2/R40-04.json | P → V | After V, collect frozen `make r40-hbm` package evidence on final identity; judge this gate separately | Preserve P/V first failing evidence; fix that cause; gate remains not-measured |
| R40-05.multicore | 最后候选尚未完成 route 和物理资格审核，无合格镜像 | `attempt02/routing_observation.json`: placement and phys_opt complete; route Global Iteration 0, no terminal marker or qualified image (P) | P → V | After V, collect frozen `make r40-multicore` package evidence on final identity; judge this gate separately | Preserve P/V first failing evidence; fix that cause; gate remains not-measured |
| R40-05.scaling | 最终镜像冻结 compute 的 1/2/3 核 A/B 缺失 | `required_gates.json`: no evidence at sim/build/r40/v2/R40-05.json | P → V | After V, collect frozen `make r40-multicore` package evidence on final identity; judge this gate separately | Preserve P/V first failing evidence; fix that cause; gate remains not-measured |
| R40-05.dispatch | 最终镜像原始大图的三物理核完成/分发计数缺失 | `required_gates.json`: no evidence at sim/build/r40/v2/R40-05.json | P → V | After V, collect frozen `make r40-multicore` package evidence on final identity; judge this gate separately | Preserve P/V first failing evidence; fix that cause; gate remains not-measured |
| R40-06.nuca | 最终镜像 active outlet 的实际负载流量证据缺失 | `required_gates.json`: no evidence at sim/build/r40/v2/R40-06.json | P → V | After V, collect frozen `make r40-nuca` package evidence on final identity; judge this gate separately | Preserve P/V first failing evidence; fix that cause; gate remains not-measured |
| R40-06.streaming-scaling | 冻结 streaming 的 1/3 核匹配 A/B 缺失 | `required_gates.json`: no evidence at sim/build/r40/v2/R40-06.json | P → V | After V, collect frozen `make r40-nuca` package evidence on final identity; judge this gate separately | Preserve P/V first failing evidence; fix that cause; gate remains not-measured |
| R40-06.locality | 冻结 LOCALITY/DCO 开关尚未接入标准 Torch 编译/分配路径 | `pending_switch_consumers.json`; `frontend.py` has only explicit DCO importer input | `rg -n -e TNCA_LOCALITY -e TNCA_DCO -e 'dco=' python/tnca/aot.py python/tnca/inductor.py python/tnca/triton_backend/compiler.py` | If a consumer exists, trace emitted registrations and bytes on V-qualified hardware | If absent, connect standard-path DCO/locality within R40-06 after V; keep gate open |
| R40-06.semantics | 最终镜像 ownership/完成可见性纵向闭环缺失 | `required_gates.json`: no evidence at sim/build/r40/v2/R40-06.json | P → V | After V, collect frozen `make r40-nuca` package evidence on final identity; judge this gate separately | Preserve P/V first failing evidence; fix that cause; gate remains not-measured |
| R40-07.pipeline | 最终镜像标准 Torch 图中的实际流水重叠证据缺失 | `required_gates.json`: no evidence at sim/build/r40/v2/R40-07.json | P → V | After V, collect frozen `make r40-pipeline` package evidence on final identity; judge this gate separately | Preserve P/V first failing evidence; fix that cause; gate remains not-measured |
| R40-07.long-k | 最终合格镜像上的完整 long-K stage A/B 缺失 | `../../R40-07/frozen_stage_control/README.md`: frozen control reaches actual 1/2/3-stage Torch rings; final frozen performance evidence absent | P → V, then `make r40-pipeline` with final identity | Collect unchanged frozen shapes, hold roles constant, verify stage provenance and the frozen speedup threshold | Preserve first failing shape/program/numerical or protocol result; repair cause without shrinking workload; gate remains not-measured |
| R40-07.matrix | 冻结 compute-bound 的 matrix 匹配 A/B 缺失 | `required_gates.json`: no evidence at sim/build/r40/v2/R40-07.json | P → V | After V, collect frozen `make r40-pipeline` package evidence on final identity; judge this gate separately | Preserve P/V first failing evidence; fix that cause; gate remains not-measured |
| R40-07.policy | 最终镜像策略边界与资源拒绝证据缺失 | `required_gates.json`: no evidence at sim/build/r40/v2/R40-07.json | P → V | After V, collect frozen `make r40-pipeline` package evidence on final identity; judge this gate separately | Preserve P/V first failing evidence; fix that cause; gate remains not-measured |
| R40-08.attention | 修复后 attention 尚未在最终三核镜像完成冻结验证 | `required_gates.json`: no evidence at sim/build/r40/v2/R40-08.json; `evidence/r40/R40-08/descriptor_publication/` retains original failure, RAM snapshot and mutation | P → V | After V, collect frozen `make r40-attention` package evidence on final identity; judge this gate separately | Preserve P/V first failing evidence; fix that cause; gate remains not-measured |
| R40-08.prefill | 冻结 long-prefill 的时间及中间字节匹配 A/B 缺失 | `required_gates.json`: no evidence at sim/build/r40/v2/R40-08.json; `evidence/r40/R40-08/descriptor_publication/` retains original failure, RAM snapshot and mutation | P → V | After V, collect frozen `make r40-attention` package evidence on final identity; judge this gate separately | Preserve P/V first failing evidence; fix that cause; gate remains not-measured |
| R40-08.decode | 冻结模型原始 decode64 在最终镜像上的数值验证缺失 | `required_gates.json`: no evidence at sim/build/r40/v2/R40-08.json; `evidence/r40/R40-08/descriptor_publication/` retains original failure, RAM snapshot and mutation | P → V | After V, collect frozen `make r40-attention` package evidence on final identity; judge this gate separately | Preserve P/V first failing evidence; fix that cause; gate remains not-measured |
| R40-09.autotune | 标准路径 TNCA_TUNE 尚未连接冻结形状的真实候选选择 | `pending_switch_consumers.json`; compiled_candidates has actual compilation; autotune_native_tests uses a small graph | `rg -n -F -e TNCA_TUNE -e 'select(' python/tnca/inductor.py python/tnca/triton_backend/compiler.py` | Trace actual frozen-shape candidates into measured selection and cache, then frozen holdout | Connect frozen-shape selection using existing budget/workers after V; do not extend framework |
| R40-09.holdout | 冻结 14 点 holdout 的实际 tune/control 与 regret 记录缺失 | `required_gates.json`: no evidence at sim/build/r40/v2/R40-09.json; worker_deadline/formal_call is explicitly functional RTL only | P → V | After V, collect frozen `make r40-autotune` package evidence on final identity; judge this gate separately | Preserve P/V first failing evidence; fix that cause; gate remains not-measured |
| R40-09.suites | 最终不可变身份下完整原生77及HBM14记录缺失 | `required_gates.json`: no evidence at sim/build/r40/v2/R40-09.json; worker_deadline/formal_call is explicitly functional RTL only | P → V | After V, collect frozen `make r40-autotune` package evidence on final identity; judge this gate separately | Preserve P/V first failing evidence; fix that cause; gate remains not-measured |
| R40-09.model | 冻结四层原始模型在最终镜像的 1/3 核匹配记录缺失 | `required_gates.json`: no evidence at sim/build/r40/v2/R40-09.json; worker_deadline/formal_call is explicitly functional RTL only | P → V | After V, collect frozen `make r40-autotune` package evidence on final identity; judge this gate separately | Preserve P/V first failing evidence; fix that cause; gate remains not-measured |
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
