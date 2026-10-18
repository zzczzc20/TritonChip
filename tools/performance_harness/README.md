# TNCA Performance Analysis Harness

This harness provides a reproducible workflow for diagnosing TNCA performance
and screening architectural changes before FPGA synthesis and implementation.
It combines analytical cost bounds, device counters, Proton instrumentation,
controlled RTL experiments, and physical implementation reports. Engineers and
AI agents should use these tools together to identify bottlenecks, test specific
hypotheses, and determine whether a candidate warrants an implementation run.

Diagnostic evidence must state its provenance, assumptions, and measurement
uncertainty. Instrumentation inevitably perturbs execution; minimize that
perturbation and characterize its magnitude for the intended workload before
interpreting a trace. Screening results and diagnostic traces do not establish
acceptance compliance.

The [native SystemC architectural model](systemc/README.md) is being developed
for event-driven complete-chip exploration on the GB10. Its current ISA and
architecture coverage are incomplete; consult its coverage table before use.
Its memory replay path is conditional on captured traffic and cannot substitute
for native program execution or establish complete-chip prediction accuracy.
RTL remains an independent offline validation source, outside the TLM execution
path. No model build or simulation should run on the FPGA implementation host.

This workflow operates within the existing R40 work packages and frozen
acceptance contract. Start with the [required-gate blocker ledger](../../evidence/r40/v2/convergence/README.md).
Begin progress reports with changes to required-gate status. Diagnostic test
counts and commit counts do not establish acceptance progress. For each
unresolved gate, record the current blocker, supporting evidence, next
experiment command, and follow-up actions for both possible outcomes.

## 1. Define the hypothesis and establish experiment provenance

Write a falsifiable hypothesis before collecting data. For example:
“Three cores contend for one physical PC; distributing requests across two
allocated PCs should reduce outlet backpressure without changing useful work.”
Identify the expected counter changes, competing explanations, and the
smallest experiment that distinguishes them.

Bind every measurement to the complete software file-tree SHA, source commit
and dirty state, compiler/runtime configuration, target profile, actual
toolchain versions, loaded image SHA, device clock, and contract SHA.
Use `archbench.provenance.software_tree(refresh=True)` before and after an
experiment: a cached digest cannot detect source changes during execution.
Hash the diagnostic harness as well. Label simulator images and virtual
clocks explicitly; they are not FPGA images or measured operating frequencies.

Preserve the original graph, shapes, seed, and numerical reference.
Microbenchmarks can establish mechanisms; they cannot replace the original
77 native points, 14 frozen HBM points, frozen holdout, or full model.
Report missing measurements as `not-measured`, with unknown numeric fields
set to `null`. Never relabel a reduced workload as its original benchmark.

## 2. Establish an uninstrumented baseline

`Session(counters=True)` exposes counters supported by the selected device.
`tnca.costmodel.counter_diagnosis` checks the exclusive cycle partition for
one program/core, reports component fractions, and computes optimistic Amdahl
ceilings for eliminating individual stall components. These ceilings are not
predicted speedups.

Overlapping stall indicators must not be summed. Cumulative multicore cycles
divided by global makespan do not constitute a utilization metric. Some physical
multicore backends currently return program-level counters as unavailable;
use actual dispatcher/core/outlet telemetry and retain that limitation.
`tools/r40/board_checkpoint.py` demonstrates capturing outlet traffic before
the CPU ownership handoff.

For images advertising runtime memory controls, query
`session.dev.memory_configuration()` and retain the actual readback in the
experiment identity. `configure_memory(active_ports=1, outstanding=8)` changes
routing and admission on the same image only after the device confirms complete
drain. `TNCA_RTL_MEM_PORTS` and `TNCA_RTL_OUTSTANDING` select startup values;
explicit requests are refused on older images. The outstanding limit applies
per physical outlet to adapter slots, including writes awaiting B responses.
Requests buffered upstream are a separate quantity and can exceed that limit.
A one-outlet experiment still addresses every home and the same shared BO.
Query configuration before and after an experiment block, outside the measured
interval. These controls add no per-launch configuration-register probes.
Sample per-grid dispatch telemetry before any subsequent control grid. A policy
readback implemented as a core program is itself a grid and resets issued,
completed and peak-activity counters. Include its duration and traffic if it
runs inside the measured graph; retain cumulative outlet counters separately.
Keep image, allocation, ingress width, burst size and clock fixed during these
comparisons. Functional RTL checks are available through
`tools/r40/memory_control_tests.py`; they do not measure board bandwidth.

`tools/r38/profiler_tests.py` checks counter identities and the historical
noninterference baseline. Run it when changing counter RTL, rather than
repeating the full suite for presentation-only changes.

For timing-sensitive RTL failures, use the optional [timed mailbox journal](mmio_journal.md) to preserve the exact simulated input schedule before adding diagnostic probes. Journal capture and replay are functional diagnostics; their wall time is not a performance result.

New ArchBench `device-graph-makespan` entries retain
`endpoint_schema: "tnca-launch-endpoints/1"` and `endpoint_records`: one complete
list of launch start/end records for each timed graph call. Recompute the
samples with `tnca.device_timing.audit_graph_samples(samples, endpoint_records,
launch_counts=counts)`, using the recorded graph launch count for each sample.
The auditor checks unsigned 64-bit counters, wraps, complete samples and a
common clock/session/epoch within each call. The ablation reader rejects a
new-schema record whose endpoints are missing or disagree with its samples.
Historical records without these fields have no reconstructed raw endpoints.
The frozen plan's `graph-makespan` resolves to ArchBench's
`device-graph-makespan` only with this raw endpoint proof and cycle units.
The source clock label and the frozen metric retain their original meanings;
serial active cycles and host request latency remain separate measurements.

Endpoint retention reuses existing launch logs after all timed calls have
finished. It adds no device register reads or target instructions inside the
measured graph. JSON retention, validation and serialization still consume host
time and memory outside that region; their wall-time cost is not a device
performance result. Use the original raw endpoints for candidate ranking,
and keep simulator cycles separate from qualified board time.

## 3. Screen candidates using conservative bounds

```bash
python tools/r40/cost_screen.py INPUT.json --out NEW_OUTPUT.json
```

The input contains a reference measurement `identity`, a `demand` map,
`candidates` mapping names to capacity maps, and an optional `cycle_budget`.
Each demand entry has `work`, `unit`, and `source`; each corresponding
capacity entry has `per_cycle`, `unit`, and `source`.

Required identity fields are `software_sha256`, `config_sha256`,
`image_sha256`, `contract_sha256` (complete 64-digit lowercase SHA256 values),
`clock_hz`, and `backend`. A hypothetical candidate is identified by its
capacity specification and assumptions. Do not invent a measured image
identity for a candidate that has not been implemented.

Demands must be lower bounds that remain valid under the candidate's
assumptions; capacities must be optimistic upper bounds. Changing compilation,
caching, fusion, layout, or communication may invalidate traffic measured on
the reference. Re-derive the bound and document that dependency.

Use separate resource keys for actual PCs, directions, NoC cuts, and cores.
Shared bandwidth does not multiply with core count. Convert capacities across
clock domains into the same reference device cycles.

The model computes `max(work / capacity)`. Exceeding a cycle budget rejects
an impossible candidate under the stated assumptions. A candidate within the
budget remains eligible for further evaluation; the bound does not establish
implementation feasibility or predict achieved performance.

Optional `implementation` entries provide per-candidate `floors` and
`available` maps for proven physical resource minima. Each floor has
`minimum/unit/source`; each capacity has `capacity/unit/source`.
Check resources per actual region: an overloaded SLR cannot borrow capacity
from another SLR. Synthesis LUT counts are not automatically post-placement
lower bounds. Capacity checks cannot certify routability or 100 MHz timing.
Track placed CLB occupancy and unique control sets alongside LUT utilization:
packing pressure may exhaust a region before its LUT count reaches capacity.
An observed baseline footprint is not automatically a resource floor for a
modified candidate.

Inspect existing placement evidence before requesting another implementation
run:

```bash
python3 tools/r40/resources.py --diagnose-placement \
  --util PATH_TO_PLACED_UTILIZATION.rpt \
  --implementation-log PATH_TO_IMPLEMENTATION.log.gz \
  --build-manifest PATH_TO_BUILD_MANIFEST.json --out NEW_DIAGNOSIS.json
```

The build manifest identifies `hardware_source`, `implementation_root`,
`clock_target_hz`, and the frozen `contract_sha256`; retain the complete
software/configuration/build identity in it. The output binds the manifest,
raw reports, and analysis source by hash. It distinguishes global `Used`
columns from per-SLR counts, reports CLB density and unique control sets,
and checks per-column SLL demand independently of aggregate crossing capacity.
A per-SLR-only report leaves global totals unavailable. The default 90% CLB
density hint is configurable with `--pressure-threshold`; it is not a gate
or a routability predictor. Missing timing and measured clock remain unknown.
Do not convert an earlier build's observed footprint into a candidate minimum.
Retain an immutable manifest copy in the evidence bundle; a live convergence
ledger may change while an earlier diagnostic remains valid.

For the final registered-cache candidate, retain the actual mapped tag devices
and each register-boundary path from the saved routed checkpoint:

```sh
vivado -mode batch -source flow/fpga/diagnose_r40_checkpoint.tcl -tclargs \
  ROUTED_CHECKPOINT.dcp NEW_CUT_REPORTS final-cache-cuts
```

This read-only mode reports tag-to-owner, owner-to-pair, pair-to-selected and
selected-output paths, the global worst sampled paths, and tag storage roots
separately from their implementation children. Missing pair/selected registers
or required timing paths fail extraction and leave partial reports intact.
The raw direct owner-to-selected report may legitimately contain no path;
that absence is not a zero-delay measurement. Reports retain logic levels and
logic/net delay rather than treating LUT count as a timing result.
`watch_implementation.py --final-cache-cuts` adds these reports to the existing
terminal observer for both success and failure. Use a separate immutable
collector bundle if the build source is already frozen. Schedule checkpoint
reads after implementation has terminated, and keep report collection serial.
Successful extraction means review-ready evidence, not physical qualification.

For empirical predictions, provide `calibration` with `candidate`, `dev`,
and `validation`. Each record contains
`id/split/identity/instrumented=false/demand/cycles`.
The model fits startup and scale by resource-bottleneck regime, then calibrates
on disjoint validation records. A different identity, unseen regime, or
bottleneck bound outside the fitted range returns `out-of-domain`.
Never use frozen holdout data for fitting, threshold selection, or calibration.
At least nine validation residuals are needed for a potentially finite 90%
interval; its interpretation still depends on representative calibration data.
Production-workload accuracy of this new model is currently **not-measured**.

The legacy `tools/r38/costmodel/` model remains a single-core, serial-grid
operator model. Do not divide its estimates by core count.
`spec/r39_sched_model.json`, `tools/r39/latency_probe.py`, and
`tools/r39/load_latency.py` support instruction/LSU calibration. A one-cycle
bench memory cannot substitute for a board memory-path measurement.

For changes to DCO owner selection, use
`tools/r40/dco_owner_order_proof.py` to check the actual RTL through the
documented membership and order abstraction. The default SMT engine requires
Yosys and Z3; `--engine sat` retains the original bit-level proof path.
`tools/r40/dco_owner_screen.py --mapping-only --shared-calls 9 --out NEW_DIR`
compares generic LUT6 depth and cell counts for independent queries sharing a
tensor table, without repeating the bit-level equivalence run.
The screen first checks both Verilog-2005 and SystemVerilog-2012 with Icarus;
acceptance by Yosys alone does not establish compatibility with the system
simulation frontend.
Keep proof timeouts as unproven results. Generic mapping can justify further
evaluation, but does not establish FPGA timing, whole-slice area, or graph
performance. See the [candidate evidence and proof assumptions](
../../evidence/r40/R40-06/owner_selection/README.md).

## 4. Instrument execution to resolve specific hypotheses

The TNCA backend preserves upstream `proton.record start/end` scope semantics
through the pinned Triton 3.8.0 dialect and Python TNCA IR. At the TNCA IR level:

```python
from tnca_cc.proton import lower_records

b.proton_record("load_wait")
# Emit operations. Include an explicit wait if completion is being measured.
b.proton_record("load_wait", start=False)

instrumented, layout = lower_records(
    kernel, buffer_base=capture.addr, buffer_bytes=capture.size, grid=grid)
```

Invoke lowering explicitly after normal optimization and before CodeGen.
The caller owns a disjoint, freshly zeroed capture buffer, its lifetime,
launch, completion wait, and CPU ownership handoff. `enabled=False` removes
markers. Compilation rejects unlowered markers to prevent silent sample loss.
This is an explicit TNCA recording backend, not an automatic Torch
collector or a port of CUDA/CUPTI collection through `proton.start()`.

Scope names must be unique and properly nested. Each program/role owns a
fixed slot; launch dimensions and contexts must match the layout.
Loops may execute inside an outer scope. Branches that skip or repeat a
marker are unsupported; move the scope outside the loop.
Insufficient capture memory or scalar registers produces an explicit refusal,
with no silent event loss, wraparound, or register spilling.

Scopes read the existing 64-bit ISA clock. Timestamps remain in scalar
registers until epilogue stores, avoiding trace HBM stores inside the region.
Instrumentation still changes instruction count, register pressure,
scheduling, and contention. Issue timestamps do not establish asynchronous
engine completion. Each physical core has a reset-relative clock; separately
rebased viewer lanes do not establish cross-core ordering.

Run resource-intensive experiments on the jump host. The following commands
produce reproducible RTL instrumentation calibration bundles:

```bash
python tools/r40/proton_profile.py --out /tmp/proton-fine --iterations 128
python tools/r40/proton_profile.py --out /tmp/proton-coarse --iterations 8192
python tools/r40/proton_profile.py --out /tmp/proton-loop --iterations 128 --loop
```

These diagnose the recording mechanism, not production graph performance.
Open `trace.chrome_trace` with a Chrome Trace/Perfetto JSON viewer. Do not
automatically upload private traces to an external service.

## 5. Quantify instrumentation overhead and uncertainty

Warm up uninstrumented A and instrumented B independently, then alternate
AB/BA pairs under identical configuration and image identity. Validate
numerics after every execution. The current tool requires at least five
pairs; this minimum does not guarantee coverage of all noise sources.
Retain raw device cycles, order, code hashes, sample count, median overhead,
and observed range.

Recalibrate on representative workloads after changing concurrency, core
count, cache conditions, scope density, or register pressure. Existing RTL
microbenchmark observations are 77.551% whole-program overhead for 128
unrolled additions, 1.388% for 8192 additions, and 5.891% for 128 runtime-loop
iterations. See the [raw evidence](../../evidence/r40/R40-09/performance_tools/README.md).
Deterministic repeats do not establish zero systematic error.

The default 1% overhead reference is a configurable diagnostic advisory.
It does not define an acceptance gate or determine whether a trace is usable.
A trace with higher overhead can still resolve a specific hypothesis if the
analysis accounts for the possibility that instrumentation changes the
bottleneck. Choose scope granularity for the question being investigated, and
report the measured perturbation, observed range, and conditions under which
those observations apply.

Report whole-program perturbation separately from region timestamp error.
An observed whole-program overhead is not a bound on each region's error.
Unknown region error bounds remain `null`. Do not subtract one constant from
all regions and claim to reconstruct uninstrumented execution. Formal
performance results use instrumentation-disabled runs and the frozen protocol.

Audit a completed bundle and optionally combine it with resource screening:

```bash
python tools/performance_harness/audit.py \
  --report /tmp/proton-fine/report.json --out /tmp/proton-audit.json
# Optional: --cost-input INPUT.json, with exactly the report's identity.
```

The auditor regenerates events from the raw buffer, checks trace identity,
AB/BA ordering, code hashes, and raw cycles, and emits input and analysis-source
hashes. Compare those hashes against previously retained manifests or commits.
Self-consistent input files alone cannot prove that their source was truthful.
An audited diagnostic bundle remains diagnostic evidence.

## 6. Validate the mechanism and justify physical implementation

After screening, choose the smallest RTL experiment that resolves the main
uncertainty. Change the mechanism named by the hypothesis, preserve failure
reproductions and mutations, and compare numerics, device cycles, counters,
and resource effects.

Use `tools/r40/engine_overlap.py` for device-side overlap analysis and
`tools/r40/window_resources_tests.py` to audit implicit DMA/MMA window usage
from machine code. Explicit operand counts alone miss those resources.
`tools/r40/ablate.py` and `device_time_v2_collect.py` support formal comparisons
and frozen sampling. Never rank candidates by simulator wall time. Move from
necessary infrastructure repairs to real compiled candidates promptly.

Start synthesis or implementation only when physical evidence is needed.
For every rerun, record a concrete root-cause hypothesis, corresponding change,
expected improvement, and falsification condition. Changes to implementation
directives require an explicit technical rationale. Reused results must match
complete sources, parameters, constraints, and tool versions. Preserve immutable
build directories and avoid modifying an active implementation run.

Use `tools/r40/resources.py` for actual routed utilization/timing reports,
`flow/fpga/audit_r40_image.tcl` for CDC/timing audits, and
`flow/fpga/audit_r40_placed_cores.tcl` for actual core-to-SLR placement.
Pblock intent and synthesis checkpoints do not replace placement/route.
A loadable image, 100 MHz timing, CDC, and actual HBM/PC/ownership behavior
remain subject to the required-gate ledger.

Each investigation should produce a concise decision record containing:

- The complete experiment identity and the hypothesis under evaluation.
- The primary bottleneck, direct evidence, and competing explanations.
- The prediction, its assumptions, and its domain of applicability.
- Instrumentation overhead, observed variability, and unresolved error bounds.
- The next experiment command and the actions warranted by either outcome.

Record unsupported speedup, frequency, routing, area, and acceptance claims as
`not-measured`; retain `null` for unknown numerical values.
