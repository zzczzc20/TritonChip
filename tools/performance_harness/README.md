# TNCA Performance Analysis Harness

This is the guide to the tools that diagnose TNCA performance, screen architectural changes before FPGA
synthesis, run guarded board experiments, and audit that default behaviour did not change. It was
reorganised on 2026-10-31 after the R41 rounds so that every tool that remains useful has one entry here.

How to read it:

1. [Which tool for which question](#which-tool-for-which-question) is the entry point.
2. [Environments and server rules](#environments-and-server-rules) and
   [Measurement-integrity rules](#measurement-integrity-rules) apply before any run.
3. [Workflows](#workflows) are end-to-end recipes with the real commands and the evidence that used them.
4. [Tool catalog](#tool-catalog) lists every script with its purpose, who references it and a class.
5. [Known pitfalls](#known-pitfalls) and [Future work](#future-work) close the guide.
6. [Methodology](#methodology-hypotheses-baselines-bounds-instrumentation) keeps the original (R40) method text
   on hypotheses, baselines, conservative bounds, Proton instrumentation and decision records.

Every description of a tool comes from its source, its docstring or `--help` text, its tests and the evidence
that used it. Where that is not enough the entry says "unclear" and the tool is listed under
[Needs a decision](#needs-a-decision). Nothing was deleted or changed in behaviour when this guide was written;
`python3 tools/performance_harness/catalog_check.py` (`make harness-catalog`) keeps the catalog complete.

Binding rules: [`doc/engineering_discipline.md`](../../doc/engineering_discipline.md). Diagnostic evidence must
state its provenance, assumptions and measurement uncertainty. Screening results, model predictions and
diagnostic traces never establish acceptance compliance.

## Which tool for which question

| Question | Use | Notes and evidence |
|---|---|---|
| How do I run a board experiment safely? | A manifest for `experiment.py` with an `fpga` step, queued by `launches/acceptance_queue.py`; check first with `preflight_board_manifest.py` and `acceptance_queue.py --check-only` | The queue never programs or resets a card and requires an approved loaded image and 15 idle minutes. See [W1](#w1-run-a-board-experiment-safely) and `evidence/r41/SERVERS.md`. |
| How do I run a whole planned board campaign unattended? | `tools/r41/run2_board/prepare_plan.py`, then `watch_plan.py` or `auto.py`; `idle_monitor.py` records the idle interval | R41 run-2 machinery, guarded by image, clock and idle checks. See [W8](#w8-guarded-board-queue-and-idle-watcher). |
| How do I compare two images or two batches? | Same registered manifests on both, then `acceptance_compare.py` (paired records), `tools/r41/compare_tiled_batches.py` (same image, default vs tiled) or `tools/r41/membound_baseline_table.py` (tables); post-route: `tools/r41/run2_postlink/check_run2.py` on both analysis directories and `tools/r41/util_hier.py compare` | See [W2](#w2-compare-two-images-or-two-batches). |
| How do I prove default code is unchanged? | `tools/performance_harness/default_change_audit.py run` on both trees, then `compare` | Compile-only, no device. Required by the default-code rule of `doc/engineering_discipline.md`. See [W3](#w3-audit-that-default-code-is-unchanged). |
| How do I get a roofline or balance estimate? | `tools/performance_harness/roofline_report.py` (from ArchBench records); `tools/r41/balance_model.py` (analytic throughput matching); `tools/r41/stream_bandwidth_model.py`, `g4_issue_bound.py` | Rooflines use spec peaks, never measurements. Evidence: `evidence/r41/hbm_bw/BALANCE.md`, `evidence/r41/stage2/roofline/README.md`. See [W4](#w4-roofline-balance-and-issue-bounds). |
| How do I get a per-kernel census? | `tools/r41/kernel_census.py` on ArchBench `--bundle-dir` output; with `asm_resource_scan.py`, `barrier_census.py`, `resource_usage.py` | See [W5](#w5-per-kernel-census-loop-counts-and-resources). |
| How do I profile decode? | `tools/models/decode_profile.py`, then `tools/models/decode_profile_report.py` | Observation only. Evidence: `evidence/r41/chat/DECODE_PROFILE.md`. See [W6](#w6-profile-a-chat-decode-step). |
| How do I count loop instructions? | `tools/r41/loop_issue_count.py DIR_OR_FILE [--mnemonics]` | Instructions per innermost-loop iteration of compiled `.s` files. See [W5](#w5-per-kernel-census-loop-counts-and-resources). |
| How do I generate, run and tabulate the memory-bound suite? | `python3 -m archbench.suite --check`; `tools/r41/membound_manifests.py`; queue the manifests; `tools/r41/membound_baseline_table.py` | The suite is registered in `archbench/suites/r41_membound.json`. See [W7](#w7-the-memory-bound-suite-generate-run-tabulate). |
| How do I run the suite on the TLM model? | `tools/r41/g4_model_suite.py`, or `tools/performance_harness/systemc/run_graphs.py` for any suite | Model time is a prediction. See [W11](#w11-tlm-runs). |
| How do I check a routed image after a link? | `tools/r41/run2_postlink/post_run2.sh` (runs `analyze_run2.tcl` then `check_run2.py`) | Read-only. See [W9](#w9-post-link-checks). |
| How do I qualify the board clock of an image? | `tools/performance_harness/clock_qualify.py capture`, `assemble`, `verify` | `capture` only reads sysfs. See [W9](#w9-post-link-checks). |
| How do I run the x86 link? | `evidence/r41/stage2/x86_link/README.md`, `flow/fpga/run_r40_attempt01.sh`; rehearse with `tools/r41/x86_link_dryrun/` | See [W10](#w10-x86-link-flow). |
| How do I get resource tables from a routed image? | `tools/r41/util_hier.py` (`routed`, `compare`, `slr`, `bytype`) | Reads Vivado reports and leaf-cell dumps. Evidence: `evidence/r41/hbm_bw/RESOURCES.md`. See [W12](#w12-resource-analysis-and-unit-frequency-screens). |
| How do I screen a unit's frequency without a full link? | `tools/r41/fmax/run_unit.sh` (x86 host) | Out-of-context unit screens. See [W12](#w12-resource-analysis-and-unit-frequency-screens). |
| How do I probe the link budget with RTL? | `tools/r41/link_budget_rtl.py`, then `link_budget_report.py` | Run-1 image geometry on the RTL stand-in; never board evidence. See [W13](#w13-rtl-link-budget-probes). |
| Do barrier immediates fit the slice a launch gets? | `tools/r41/barrier_slice_check.py` | See the tool entry and `evidence/r41/hbm_bw/fma_dslx/barrier_slice/README.md`. |
| How do I chat with a model on the standard path? | `tools/models/chat.py`; quality: `tools/models/chat_quality.py` | See [W14](#w14-chat-and-model-runs). |
| How do I audit a Proton bundle or measure instrumentation overhead? | `tools/r40/proton_profile.py`, then `tools/performance_harness/audit.py` | Methodology below. |
| Which ISA or numeric reference do I regenerate? | `tools/isa/*` (schema, target, support matrix), `tools/model/*` (oracles, vector generators) | Make targets `isa`, `target`, `compiler`. |
| Which tests cover the harness? | `make r41-harness-tests`, `make harness-catalog`, `make archbench-selftest` | See [Unit tests](#unit-tests). |

## Environments and server rules

The authoritative rules are in [`evidence/r41/SERVERS.md`](../../evidence/r41/SERVERS.md) (three servers and the shared U280
card) and `evidence/r41/stage2/x86_link/README.md`. Credentials stay in helper scripts outside the repository; never
print, copy or commit them. Summary of what the tools assume:

- GB10 (build and simulation host): Torch, the SystemC/TLM model, RTL simulators. Shared: at most 6 cores and 16 GB
  per job. No model build or simulation runs on the FPGA implementation host. `systemc/bootstrap.sh` must never be run
  on the FPGA host.
- FPGA host (shared U280 card): board runs only through `experiment.py` manifests launched by a waiting script that
  enforces the 15-minute idle rule. A new image needs the author's explicit approval; the board interpreter is
  recorded in the manifest.
- x86 host (CPU-only Vivado): synthesis, place and route, full links, unit screens; no card.
- The local lead machine is small (about 7 GB RAM, no swap): keep builds and heavy outputs on the servers and commit
  only summaries.
- Environments: see [Known pitfalls](#known-pitfalls) (GB10 `torchenv` has no `transformers`; `~/flux-venv` has it).

## Measurement-integrity rules

These rules come from `doc/engineering_discipline.md` (binding) and are enforced in part by the tools themselves.

- Domain labels. Every number carries its domain: `rtl`, `spec`, `analytic`, `tlm` (model prediction, never FPGA
  evidence), `board`. `balance_model.py`, `link_budget_report.py` and `membound_baseline_table.py` state the domain
  of each figure; `run_graphs.py` labels records `model-prediction-uncalibrated`.
- Registered suites. The suite is registered and frozen before any design change is measured
  (`python3 -m archbench.suite --freeze` / `--check`; `archbench/suites/*.json`). Cases are never dropped,
  reordered or resized after seeing results. A point with no record, a failed run or a changed definition hash is
  listed, never omitted (`membound_baseline_table.py`).
- Standard path. Measured programs are public PyTorch programs on `device="tnca"` through
  `torch.compile` / Inductor / Triton / the TNCA compiler / the runtime, with default knobs. No hidden knob, no
  hand-written kernel, no benchmark-specific code.
- Default code unchanged. A change that is not a general improvement of the default must leave the 113 original
  points' numerics and default code unchanged (`default_change_audit.py`).
- No typed-in results. Tools compute from raw files (`util_hier.py`, `link_budget_report.py`, `balance_model.py`
  say so in their docstrings). Raw outputs are kept, or recorded with sha256 and host path.
- Identity. Board steps bind image sha256, UUID, qualified clock evidence and memory map (`experiment.py`); wrong
  image, clock or another user's context refuses the step.
- References and tolerances are fixed before the device result is examined.
- Exit zero of `experiment.py` means the commands finished; it never promotes evidence to an acceptance gate.
- Unknown or unmeasured values are `not-measured` / `null`.

## Workflows

Commands are the ones the tools document or the evidence used. Paths marked `SRC`, `OUT`, `PY` are yours. Run `--help`
for the exact options of any tool; this guide does not repeat every flag. Do not run a board step without the rules
above.

### W1 Run a board experiment safely

1. Write an experiment manifest (see `spec/r41/stage1/experiments/*.json`, `spec/r41/hbm_bw/membound/*.json`):
   hypothesis, competing explanations, expected counter changes, commands, budgets, and for an `fpga` step the board
   identity (image sha256, UUID, qualified clock evidence, clock, memory map).
2. Validate it without a device: `preflight_board_manifest.py --root SRC --manifest M.json --out OUT.json` runs the
   step up to, but not including, its first device probe (the probe raises a sentinel; process execution is
   forbidden).
3. Check the queue: `python3 tools/performance_harness/launches/acceptance_queue.py --root SRC --out OUT --python PY
   --manifest M.json --check-only` (manifest and shell checks; no sysfs or card access).
4. Launch it detached on the FPGA host with the same command minus `--check-only`
   (`--budget-seconds` defaults to 10800, idle waits excluded; `--manifest` may repeat). The queue waits for 15
   idle minutes and an approved loaded image, runs the manifests with before and after image guards and never
   programs or resets a card.
5. `experiment.py` runs/resumes the manifest; reuse is exact (key covers manifest, sources, software tree, command,
   environment, inputs, evidence kind, timing boundary, cache state, board identity). `board_batch.py` is the
   `fpga` step for registered batches: the manifest's `measurement` block must equal the frozen plan or the
   batch is refused; `--dry-run` and `--retry-failed` exist.
6. Evidence: `evidence/r41/acceptance/launch_records/README.md`, `evidence/r41/acceptance/STATUS.md`,
   `evidence/r41/hbm_bw/LINK_BUDGET.md`.

### W2 Compare two images or two batches

- Run the same registered manifests on both images (same suite, sampling rule and timing boundary).
- `acceptance_compare.py --before B.json --after A.json --manifest M.json --out OUT.json` compares actual paired
  board records and keeps numerical and performance failures visible.
- `tools/r41/compare_tiled_batches.py --default D --tiled T --batch NAME --out OUT.json` compares complete
  same-image default and tiled batches by frozen point (evidence: `evidence/r41/hbm_bw/g4/board_tiled2/README.md`).
- Tables per point, family and size: `tools/r41/membound_baseline_table.py` ([W7](#w7-the-memory-bound-suite-generate-run-tabulate)).
- Post-route: run `check_run2.py` on each analysis directory ([W9](#w9-post-link-checks)) and
  `util_hier.py compare BASE_RPT CAND_RPT` for kernel-synthesis deltas ([W12](#w12-resource-analysis-and-unit-frequency-screens)).
- Never rank candidates by simulator wall time or compare against a weakened control.

### W3 Audit that default code is unchanged

```sh
python3 tools/performance_harness/default_change_audit.py run --source-root TREE --software-tree files:<sha> \
    --label r40 --out DIR [--jobs 4]
python3 tools/performance_harness/default_change_audit.py compare --control DIR/report.json \
    --candidate DIR2/report.json [--ablation LABEL=DIR3/report.json ...] --out audit.json
```

`run` compiles every original point (the union of the 113 frozen points in `spec/r40/suites/hbm.json`) with the
delivered default compile knobs, each in a fresh process with private Triton/Inductor/temporary directories, over a
host-memory stand-in session: no board, simulator or device session; every launch is skipped
(`TNCA_LAUNCH=skip`). It records per launched kernel the sealed artifact text hash, launch structure and resources.
`compare` judges candidate against control. Evidence: `evidence/r41/acceptance/STATUS.md`,
`evidence/r41/acceptance/KIMI_HANDOFF.md`; the freeze gate of run 2 reports "audit 113/113 unchanged"
(`evidence/r41/HANDOFF_RUN2.md` section 5). `tools/r41/resource_usage.py` reads the audit's `report.json`.

### W4 Roofline, balance and issue bounds

- Roofline figure and efficiency table from ArchBench records:
  `python roofline_report.py --tnca100 FORMAL.jsonl --tnca90 DIR_OR_GZ... --gpu RAW.jsonl... --out OUTDIR`.
  Axes are arithmetic intensity (FLOP per required byte) and achieved FLOP/s; rooflines are spec peaks (TNCA
  strict-bound peaks, GB10 `archbench/specs/gpu_gb10.json`); `eta = bound / time`.
  Evidence: `evidence/r41/stage2/roofline/README.md`.
- Analytic throughput-matching model:
  `python3 tools/r41/balance_model.py --out evidence/r41/hbm_bw/balance/balance_model.json [--update-md evidence/r41/hbm_bw/BALANCE.md]`.
  Constants carry their RTL/spec source; results derive from recorded raw files in the repository; `--update-md`
  rewrites the `<!-- BEGIN name -->` blocks of the document. The constants describe the R41 architecture:
  re-check them before reuse ([reusable with care](#classes)).
- `tools/r41/g4_issue_bound.py` (issue and SFU ceilings of compiled memory-bound kernels, analytic),
  `stream_bandwidth_model.py` (cycle-level capacity of the per-PC AXI stream engines),
  `bandwidth_closure_model.py` (V3 bandwidth closure with the native SystemC streaming-add program).
- The R40 conservative-bound screen `tools/r40/cost_screen.py` is described under Methodology.

### W5 Per-kernel census, loop counts and resources

1. Obtain compiled assembly: run ArchBench points with `--bundle-dir DIR` (the provenance bundle keeps
   `kernels/<name>/<name>.s`), or dump with `TNCA_KERNEL_DUMP` (as `fp8_hub_compile_only.py` does with
   `TNCA_LAUNCH=skip`).
2. `python3 tools/r41/kernel_census.py BUNDLE_DIR [BUNDLE_DIR ...] --out FILE.json`: per kernel, static
   instructions by class, vector loads/stores and element width, tensor-transfer-engine instructions, SFU
   operations, loops and the compiler's per-program block size. Nothing executes; counts are the generated code's.
3. `python3 tools/r41/loop_issue_count.py DIR_OR_FILE [...] [--mnemonics]`: instructions per iteration of the
   innermost loop (between the last `for_N:` before the first `m.mma.*` and `endfor_(N+1):`), classified as in
   `evidence/r41/hbm_bw/BALANCE.md` section 5. Used in `evidence/r41/hbm_bw/mma_dp/README.md`.
4. `asm_resource_scan.py DIR [...]` / `--compare CONTROL CANDIDATE` (accumulators and barriers named by programs;
   compare reports identical, identical-up-to-barrier-register-names, or different), `barrier_census.py [--bar-n 8]
   [--contexts 4] [--json OUT] DIR [...]`, `resource_usage.py REPORT.json [...]` (from an audit report).

### W6 Profile a chat decode step

```sh
python tools/models/decode_profile.py --model vl --device tnca --layers 1 --prompt "Say hi." \
    --new-tokens 4 --bucket 64 --out OUT
python tools/models/decode_profile_report.py ...   # see --help: --full-layers, --clock-hz, --rates, --board-report, --json
```

`decode_profile.py` runs `chat.py` unchanged and only reads: per launch it records the kernel, grid, artifact,
device cycles, every tensor argument and its role, code-store loads and, on the TLM backend, the difference of
every model counter (read while the device is idle, so no device cycles are added). The report tool derives
per-class tables and scales to full depth analytically. Evidence: `evidence/r41/chat/DECODE_PROFILE.md`.

### W7 The memory-bound suite: generate, run, tabulate

1. Registration check: `python3 -m archbench.suite --check` (`--freeze` writes the registered file; it is never
   re-frozen after measurement).
2. Manifests: `python3 tools/r41/membound_manifests.py [--tiled2] [--check]` writes
   `spec/r41/hbm_bw/membound/<id>.json`; each runs `board_batch.py` over a slice of
   `archbench/suites/r41_membound.json`. Slicing exists only because a queue has at most three hours; the split
   uses an estimate of board time fitted from run-1 records.
3. Run through [W1](#w1-run-a-board-experiment-safely) on an approved image.
4. Tabulate: `python3 tools/r41/membound_baseline_table.py [--write]` (reads archived raw outputs
   `evidence/r41/hbm_bw/membound_baseline/membound_baseline_e57b64c.tgz` and the frozen suite; reports verdict,
   formal device time, full-link GB/s and eta per point). Result document: `evidence/r41/hbm_bw/MEMBOUND_BASELINE.md`.
   For a new image the archive path and image identity in this script need editing.
5. Model-side counterpart: `tools/r41/g4_model_suite.py` ([W11](#w11-tlm-runs)).

### W8 Guarded board queue and idle watcher

The run-2 machinery in `tools/r41/run2_board/`:

- `prepare_plan.py --work W --python PY --image XCLBIN --out PLAN [--xclbinutil ...]` publishes a complete plan after
  all preflight reviews (`source_review.py`, `identity.py`, `registrations.py`, `manifests.py`).
- `watch_plan.py --plan PLAN --image XCLBIN --out OUT [--poll 20]` waits for a complete plan and starts its guarded
  queue once.
- `auto.py --plan PLAN --image XCLBIN --program P --out OUT [--sysfs ROOT] [--poll 5]` runs the approved plan after
  a guarded idle interval. Missing inputs, malformed sysfs, a changed image or a failed stage stop the queue; it never
  resets a device, kills a process or retries a failed load. `stage.py --root --out --python --registration` runs one
  registered stage.
- `idle_monitor.py` records a monotonic idle interval without touching the card; it hard-codes `~/r41run3`.
- Rules: the card is shared, 15 idle minutes, no load without the author's approval (`evidence/r41/SERVERS.md`).
- The remaining files in that directory are campaign-specific waiters and reviews (candidate image, Q41/XL41 pilots,
  chat retries); read them as examples, not as tools ([catalog](#r41-guarded-board-queue-tools-r41-run2_board-chat-version-stamped-and-campaign-specific-scripts)).
- Tests: `tools/r41/run2_board/test_*.py` (for example `test_auto.py`).

### W9 Post-link checks

```sh
bash tools/r41/run2_postlink/post_run2.sh BUILD_ROOT SOURCE_ROOT        # waits for build/attempt.exit
python3 tools/r41/run2_postlink/check_run2.py ANALYSIS_DIR [--json OUT]  # reads report files only
```

`post_run2.sh` runs the read-only `analyze_run2.tcl` on the routed checkpoint (2 threads, never writes a checkpoint
or touches a board) and then `check_run2.py`, which checks the reports against the G4 screens and pass/fail
thresholds (hard FAIL gates include negative setup/hold slack of the design and kernel clock, failing endpoints,
routing errors and Critical CDC findings; see its docstring for the full list). Run it on run 1's reports
(`evidence/r41/stage2/run1/run1-reports.tgz`) to see reference values. Evidence:
`evidence/r41/hbm_bw/g4/run2/BRINGUP.md`, `evidence/r41/HANDOFF_RUN2.md` section 11.
Clock qualification: `clock_qualify.py capture --image X --uuid U --out board_clock.json` (reads sysfs only; refuses
unless the loaded UUID is the image's), `assemble --image X --audit DIR --capture board_clock.json --out
qualified_clock.json`, `verify --evidence qualified_clock.json --image X --out verify.json`.
Other post-route tools: `frequency_preflight.py`, `regional_preflight.py` (per-SLR resource preflight),
`check_irq_sticky.py`, `review_candidate_cdc.py`, the HBM-clock `.tcl` probes (one-off).

### W10 x86 link flow

Follow `evidence/r41/stage2/x86_link/README.md`:

```sh
source ~/x86_link_env.sh
setsid nohup nice -n 5 bash flow/fpga/run_r40_attempt01.sh FRESH_BUILD_ROOT IMAGE_TCL > /dev/null 2>&1 < /dev/null &
```

One link at a time (`VPP_JOBS=2`); a link counts against the Stage 2 budget of five full runs. Rehearse the flow
without a card with `tools/r41/x86_link_dryrun/` (`make_tiny.py` builds an isolated AXI interface probe,
`run_tiny.sh SRC OUT IMAGE_CONFIG` runs it in a disposable source copy, `resource_watch.py` runs a command in a fresh
session and caps summed process-tree RSS). `tiny_place_hook.tcl` must never ship. The dry run found the missing
`vitis_hls` and `unzip` (see pitfalls). Run [W9](#w9-post-link-checks) when the link ends.

### W11 TLM runs

- Any suite: `run_graphs.py --source-root SRC --binary BUILD/tnca-chip --out FRESH_DIR --suite SUITE.json [--tier tiny]
  [--point ID ...]` runs ArchBench points end to end through the native SystemC model on the standard path (fresh
  child process per point, compared with the PyTorch reference, host fallbacks counted); `--backend sim-file` runs
  the RTL simulator for the wall-time comparison only. Records are `model-prediction-uncalibrated`.
- Memory-bound suite: `$TORCH_PYTHON tools/r41/g4_model_suite.py --binary BUILD/tnca-chip --out DIR [--size small]
  [--variant run1 --variant g4 --variant ideal] [--jobs 3]` (all at 100 MHz, three active cores).
- Diagnosis: `diagnose_systemc.py --report RUN/report.json --out diagnosis.json [--candidate ...]`.
- Model details, coverage table, calibration, DRAM paths and recording policy: [`systemc/README.md`](systemc/README.md).
  Build and run on GB10 only. Evidence: `evidence/r41/hbm_bw/G4_DESIGN.md`.

### W12 Resource analysis and unit frequency screens

```sh
util_hier.py routed  ROUTED_HIER_RPT        # per-tile / per-module tables of the routed image
util_hier.py compare BASE_RPT CAND_RPT [...]  # kernel-synthesis deltas, base vs candidate
util_hier.py slr     LEAF_CELLS_TSV[.gz]    # per-SLR primitive counts per module group
util_hier.py bytype  HIER_RPT [MIN_LUT]     # totals per module type
```

Every number is read from a Vivado report (`report_utilization -hierarchical`) or a leaf-cell dump given on the
command line (`evidence/r41/hbm_bw/RESOURCES.md`). Unit screens on the x86 host:
`tools/r41/fmax/run_unit.sh NAME TOP ROUTE FILE.v [FILE.v ...]` wraps TOP in registers (`regwrap.py`), runs
`unit_ooc.tcl` and prints one line per unit with `unitsum.py` (LUT, logic/memory, FF, CARRY8, DSP, CLB, WNS);
`xlsc.sh` rebuilds the DSLX FMA/MAC to combinational Verilog with the pinned XLS. For the earlier R40 reports see
`tools/r40/resources.py` (Methodology).

### W13 RTL link-budget probes

`link_budget_rtl.py --point ID [--point ID ...] --out FILE.jsonl [--memory-bytes N]` runs each ArchBench point as the
board would (cold call, then one timed call) on the RTL stand-in of the board kernel at the run-1 image parameters,
printing cumulative stage counters per tile and outlet. The memory behind the outlets is the testbench line
model, not HBM: RTL simulation, never board. `link_budget_report.py [--write]` computes the stage ceilings of
the link from recorded runs only (`evidence/r41/hbm_bw/LINK_BUDGET.md`); `link_budget_home.py` gives home-slice
service ceilings; `g4_rtl_report.py` and `g4_g2_cuts_report.py` are the G4 per-experiment reports.

### W14 Chat and model runs

`python tools/models/chat.py --model vl --prompt "Hello" --device tnca --out OUT` (one-shot), `--interactive`, or
`--model q35 --sample ...`. The CPU W8A8 route is a teacher-forced reference, never a device fallback; `--layers`
is a reduced-depth integration proof only. Quality: `tools/models/chat_quality.py` (registered check
`spec/r41/chat/quality-v1.json`). Use `~/flux-venv` on GB10 for TLM chat runs (it has `transformers`).
Checkpoint fetch scripts: `fetch_qwen_fp8.py`, `fetch_qwen35_text.py`, `fetch_flux2_klein.py`.

## Tool catalog

### Classes

<a id="classes"></a>

| Class | Meaning |
|---|---|
| core | Core reusable: use it for any future round. |
| care | Reusable with care: needs an edit (hard-coded image, paths, constants) or a specific environment (torch, TLM binary, x86 Vivado, board). |
| one-off | One-off or round-specific: evidence generation for R41, kept for reproducibility. |
| unclear | Superseded or unclear: see [Needs a decision](#needs-a-decision). Nothing is deleted without the author. |

The classification is a reading of each tool's code, docstring and references, not a test result. The "Referenced by"
column lists up to two places that mention the file name (Makefile, `doc/`, `evidence/*.md`, `spec/`, `flow/`), else
one other tool that mentions it, else "none found". It shows use, not correctness.

<!-- BEGIN catalog -->
### Experiment runner, board safety and acceptance (`tools/performance_harness/`)

Manifest-driven experiments with explicit identities, guarded board access and the R41 acceptance ledger.

| Tool | Purpose (from its docstring) | Referenced by | Class |
|---|---|---|---|
| `tools/performance_harness/acceptance_compare.py` | Compare actual paired board records; preserve numerical/performance failures. | `spec/r41/stage2/acceptance/s2-p1-gemm.json` | care |
| `tools/performance_harness/acceptance_manifests.py` | Generate bounded lead-started acceptance registrations; never run a card. | `evidence/r41/acceptance/STATUS.md`, `evidence/r41/hbm_bw/AUDIT.md` | one-off |
| `tools/performance_harness/board_batch.py` | Run a registered board batch with the frozen Stage 1 sampling rule. | `evidence/r41/hbm_bw/LINK_BUDGET.md`, `evidence/r41/hbm_bw/g4/board_tiled2/README.md` | care |
| `tools/performance_harness/catalog_check.py` | Coverage check of this catalog: fails on a script that is neither listed here nor ignored, and on a listed path that no longer exists | `Makefile` | core |
| `tools/performance_harness/clock_qualify.py` | Qualified operating clock of the R41 Stage 2 run-1 image (100 MHz). | `evidence/r41/hbm_bw/g4/run2/BRINGUP.md`, `evidence/r41/stage2/board/README.md` | care |
| `tools/performance_harness/endurance_mixed.py` | Two-hour continuous mixed board run with session and resource-growth records (R41 S2-01 item 8). | `evidence/r41/stage2/board/README.md`, `spec/r41/run2/board/r2-s2-endurance.json` | one-off |
| `tools/performance_harness/experiment.py` | Run/resume a bounded experiment with explicit inputs, outputs and identities. | `Makefile`, `doc/r38_isca_roadmap.md` | core |
| `tools/performance_harness/frequency_preflight.py` | Rebuild the complete routed census and bound Stage 2 regional risk. | `evidence/r41/stage2/candidates/F08/README.md` | care |
| `tools/performance_harness/launches/acceptance_queue.py` | Lead-started queue: require an approved loaded image and 15 minutes idle. | `evidence/r41/acceptance/STATUS.md`, `evidence/r41/acceptance/launch_records/README.md` | core |
| `tools/performance_harness/lifecycle_closeout.py` | Validate archived lifecycle preflight reports and derive a scoped summary. | `evidence/r41/acceptance/STATUS.md` | one-off |
| `tools/performance_harness/preflight_board_manifest.py` | Validate a single board step up to, but without, its first device probe. | none found | core |
| `tools/performance_harness/regional_preflight.py` | Per-SLR resource preflight of the next implementation's changed RTL (R41 S1-04 B). | `evidence/r41/stage1/regional_preflight.md`, `evidence/r41/stage2/integration/README.md` | care |
| `tools/performance_harness/stage1_acceptance.py` | R41 Stage 1 acceptance: judge the frozen plan's required items from indexed evidence. | `Makefile` | one-off |
| `tools/performance_harness/stage2_acceptance.py` | Integrity-check the manually judged R41 Stage 2 ledger; never invent a pass. | `Makefile`, `evidence/r41/acceptance/STATUS.md` | one-off |
| `tools/performance_harness/stage2_board.py` | R41 Stage 2 board batch on the run-1 100 MHz image (30e79769..., UUID 49964b17...). | `evidence/r41/hbm_bw/RESOURCES.md`, `evidence/r41/stage2/board/README.md` | one-off |
| `tools/performance_harness/stage2_contract.py` | Validate prospective Stage 2 inputs without rewriting historical failures. | none found | one-off |
| `tools/performance_harness/stage_contract.py` | Verify the frozen R41 stage inputs without promoting any delivery gate. | `Makefile`, `spec/r41/stage1/experiments/harness-smoke.json` | care |
| `tools/performance_harness/tune_board_qualification.py` | Physical TNCA_TUNE qualification on the 90 MHz board (R41 S1-03 B). | `spec/r41/stage1/experiments/board-tune-qualification.json` | one-off |

### Audits, comparisons and residuals (`tools/performance_harness/`)

Audit of default code, Proton bundles and SystemC runs; roofline; comparison and residual scripts.

| Tool | Purpose (from its docstring) | Referenced by | Class |
|---|---|---|---|
| `tools/performance_harness/audit.py` | Audit a diagnostic Proton bundle and optionally screen architecture resources. | `doc/engineering_discipline.md`, `evidence/r40/v3/capability_fe1b.md` | core |
| `tools/performance_harness/audit_historical_ooc.py` | Verify source equality before reusing historical diagnostic OOC reports. | none found | one-off |
| `tools/performance_harness/captured_case_inputs.py` | Replay saved diagnostic CPU inputs exactly; performs no TNCA computation. | called by `tools/performance_harness/prepare_captured_model_cases.py` | one-off |
| `tools/performance_harness/compare_captured_model_cases.py` | Postdict measured diagnostics; refuse residuals when program/bindings differ. | none found | one-off |
| `tools/performance_harness/compare_fragment_bits.py` | Bitwise K-order check of matched fragment runs (S1-02 ordered K). | `evidence/r41/fp8/README.md` | one-off |
| `tools/performance_harness/compare_fragments.py` | Compare matched fragment development runs; model cycles stay predictions. | none found | one-off |
| `tools/performance_harness/default_change_audit.py` | R41 Stage 1: compile-only audit of default-on compiler changes against B40-OBS. | `doc/engineering_discipline.md`, `evidence/r41/acceptance/KIMI_HANDOFF.md` | core |
| `tools/performance_harness/diagnose_systemc.py` | Audit a native SystemC run and expose falsifiable performance hypotheses. | none found | core |
| `tools/performance_harness/matched_native_speed.py` | Compare complete native SystemC and RTL execution of one identical artifact. | `spec/r41/stage1/experiments/matched-native-speed.json` | care |
| `tools/performance_harness/memory_service_compare.py` | Retain service/depth observations without assuming more outlets win. | `spec/r41/stage1/experiments/board-memory-service.json` | one-off |
| `tools/performance_harness/mma_dp_summary.py` | Derive the m.mma.dp.async summaries from raw run reports (no number is typed here). | none found | one-off |
| `tools/performance_harness/monitor_audit.py` | Audit matched native off/aggregate/trace results and bounded trace metadata. | `spec/r41/stage1/experiments/harness-smoke.json` | care |
| `tools/performance_harness/pipeline_admission_compare.py` | Compare native ring cases without promoting model rankings to board wins. | `spec/r41/stage1/experiments/board-pipeline-admission.json`, `spec/r41/stage1/experiments/pipeline-admission.json` | one-off |
| `tools/performance_harness/pipeline_compare.py` | Compare actual three-sample pipeline records; do not infer acceptance. | `evidence/r41/hbm_bw/AUDIT.md`, `spec/r41/stage2/acceptance/s2-p2-pipeline.json` | one-off |
| `tools/performance_harness/prediction_residuals.py` | Fast- and slow-path residuals of a sealed prediction set against board records. | `evidence/r41/stage1/kickoff/predictions-v2/README.md`, `spec/r41/stage1/experiments/board-gemm-tile.json` | care |
| `tools/performance_harness/prepare_captured_model_cases.py` | Copy frozen diagnostic helpers with an explicit captured-input provider. | none found | one-off |
| `tools/performance_harness/prospective_residual.py` | Residual of a sealed prospective model prediction against a later board record. | none found | care |
| `tools/performance_harness/reference_compare.py` | Cross-validate CPU BF16/FP32 references before prospective TNCA numerics freeze. | none found | care |
| `tools/performance_harness/roofline_report.py` | Roofline figure and efficiency table from ArchBench records. | `evidence/r41/hbm_bw/HANDOFF.md`, `evidence/r41/stage2/roofline/README.md` | core |
| `tools/performance_harness/tests.py` | Replay preserved real RTL bundles; corrupted trace/cycles/code must refuse. | `Makefile`, `doc/architecture_overview.md` | core |

### R41 stage-item case drivers (`tools/performance_harness/`)

Standard-Torch case drivers written for one named R41 stage item; they are templates for future device-window, residency or locality cases.

| Tool | Purpose (from its docstring) | Referenced by | Class |
|---|---|---|---|
| `tools/performance_harness/addr64_case.py` | R41 S2-04 D: standard Torch on device addresses above 4 GiB and storages over 2 GiB. | `evidence/r41/fp8/models_tnca/README.md` | one-off |
| `tools/performance_harness/compare_old_mapping_board.py` | Audit all nine physical old/fixed/narrow GEMM comparisons from raw files. | none found | one-off |
| `tools/performance_harness/fragment_case.py` | Standard Torch GEMM residency case, with artifacts and native model counters. | `evidence/r41/fp8/README.md`, `evidence/r41/stage2/candidates/F08/README.md` | one-off |
| `tools/performance_harness/hbm8_case.py` | R41 hbm8: an 8 GiB HBM window on the three-core image, through standard Torch. | `evidence/r41/hbm8/README.md`, `spec/r41/hbm8/experiments/board-hbm8-big.json` | one-off |
| `tools/performance_harness/hbm8_manifest.py` | The registered R41 hbm8 board experiments, and their simulator twins. | `evidence/r41/hbm8/README.md`, `evidence/r41/stage2/board/README.md` | one-off |
| `tools/performance_harness/hbm8_sim.py` | Sparse 8 GiB-window RTL stand-in for the three-core U280 board binding. | `evidence/r41/hbm8/README.md` | one-off |
| `tools/performance_harness/hbm8_summary.py` | Summarize hbm8_case.py reports into the experiment's conclusion record. | `spec/r41/hbm8/experiments/board-hbm8-big.json`, `spec/r41/hbm8/experiments/board-hbm8.json` | one-off |
| `tools/performance_harness/hbm8_window_case.py` | Standard Torch on the native model with an 8 GiB device window (one case per process). | `evidence/r41/hbm8/tlm/README.md`, `spec/r41/hbm8/experiments/tlm8-window.json` | one-off |
| `tools/performance_harness/locality_case.py` | Two real standard Torch graphs through captured native locality off/on. | `evidence/r41/acceptance/STATUS.md`, `spec/r41/stage2/acceptance/s2-p2-locality.json` | one-off |
| `tools/performance_harness/locality_closeout.py` | Validate a real locality_case report and derive its comparison ledger. | none found | one-off |
| `tools/performance_harness/memory_service_case.py` | Resident standard Torch service ledger under actual outlet/depth controls. | `spec/r41/stage1/experiments/board-memory-service.json` | one-off |
| `tools/performance_harness/prepare_old_mapping_control.py` | Prepare an isolated, single-change negative control from frozen Stage 1. | called by `tools/performance_harness/prepare_captured_model_cases.py` | one-off |
| `tools/performance_harness/register_old_mapping_board.py` | Register the nine original GEMM points for the isolated old-mapping arm. | none found | one-off |
| `tools/performance_harness/validate_old_mapping_control.py` | Validate the isolated legacy-mapping control before any board measurement. | none found | one-off |

### Feature case drivers and numerics (`tools/performance_harness/`)

Functional and numeric test drivers for FP8, operators the standard path lacked, and primitives. They need a `tnca` backend (simulator, TLM or board) and a torch environment.

| Tool | Purpose (from its docstring) | Referenced by | Class |
|---|---|---|---|
| `tools/performance_harness/fp8_block_reference.py` | Plain-Torch references for block-scaled FP8 kernels, in the order TNCA executes them. | `evidence/r41/fp8/kernels/README.md`, `evidence/r41/hbm_bw/mma_dp/README.md` | care |
| `tools/performance_harness/fp8_blockwise_cases.py` | Blockwise-scaled `_scaled_mm` / `_scaled_mm_v2` on `tnca` through Inductor. | `evidence/r41/fp8/blockwise/README.md`, `evidence/r41/hbm_bw/g4/run2/BRINGUP.md` | care |
| `tools/performance_harness/fp8_cases.py` | FP8 as a generic Torch dtype on `tnca`: casts, tensor plumbing, _scaled_mm, linears. | `evidence/r41/fp8/README.md`, `evidence/r41/fp8/blockwise/README.md` | care |
| `tools/performance_harness/fp8_environment.py` | Build/check an isolated FPGA-host FP8 environment; never access a device. | none found | care |
| `tools/performance_harness/fp8_kernels_cases.py` | The Hub FP8 kernels (kernels-community/finegrained-fp8), unchanged, on tnca. | `doc/r41_chat.md`, `evidence/r41/fp8/blockwise/README.md` | care |
| `tools/performance_harness/fp8_numeric_exhaustive.py` | Exhaustive FP8 conversion semantics: the TNCA conversion vs torch CPU. | `evidence/r41/fp8/README.md` | care |
| `tools/performance_harness/primitive_cases.py` | Model-independent attention/conv/recurrent-state feasibility via standard Torch. | `spec/r41/stage1/experiments/depthwise-regression.json`, `spec/r41/stage1/experiments/harness-smoke.json` | care |
| `tools/performance_harness/std_ops_cases.py` | Generic operators the standard path lacked, on `tnca`: sin/cos, log1p and | `evidence/r41/fp8/models_tnca/README.md`, `evidence/r41/fp8/ops/README.md` | care |
| `tools/performance_harness/transcendental_numeric.py` | Accuracy of the compiler's log1p, sin, cos and erf over every binary32 input. | `evidence/r41/fp8/ops/README.md` | care |
| `tools/performance_harness/triangular_solve_cases.py` | `torch.linalg.solve_triangular` on `tnca`, through the standard path. | `evidence/r41/fp8/models_tnca/README.md`, `evidence/r41/fp8/ops/README.md` | care |

### Model estimation and compile-only probes (`tools/performance_harness/`)

Analytic or compile-only scripts for pinned language-model checkpoints; they never allocate checkpoint data or open a device.

| Tool | Purpose (from its docstring) | Referenced by | Class |
|---|---|---|---|
| `tools/performance_harness/language_meta_analysis.py` | Analyze the pinned official language model without allocating checkpoint data. | `evidence/r41/stage1/model_estimates.md` | care |
| `tools/performance_harness/model_capacity_estimate.py` | Analytical state/activation estimates from an audited language inventory. | `evidence/r41/stage1/model_feasibility.md` | care |
| `tools/performance_harness/model_compile_estimate.py` | Compile-only TNCA estimates for the official Qwen3.5 language branch. | `evidence/r41/fp8/ops/README.md`, `evidence/r41/stage1/model_estimates.md` | care |
| `tools/performance_harness/model_inventory.py` | Inventory checkpoint storage from pinned safetensors headers, without weights. | none found | care |
| `tools/performance_harness/vl_compile_probe.py` | Compile-only TNCA probe of the official Qwen3-VL text and vision branches. | `evidence/r41/fp8/ops/README.md` | care |

### SystemC / TLM model (`tools/performance_harness/systemc/`)

Native SystemC model drivers, validators, calibration, RTL oracles. See the model README for coverage.

| Tool | Purpose (from its docstring) | Referenced by | Class |
|---|---|---|---|
| `tools/performance_harness/systemc/analyze_replay.py` | Quantify validation error at an accepted-request RTL boundary, without fitting. | none found | care |
| `tools/performance_harness/systemc/benchmark_boundary.py` | Offline RTL/TLM memory-boundary cost and error; never whole-chip speedup. | none found | care |
| `tools/performance_harness/systemc/board_dataset.py` | Board timing dataset for calibrating and validating the SystemC model. | `evidence/r40/v3/systemc_m2/README.md` | care |
| `tools/performance_harness/systemc/bootstrap.sh` | Bootstrap of the SystemC build environment; run on the GB10, never on the FPGA implementation host | `doc/r1_signoff.md`, `doc/r1_status.md` | core |
| `tools/performance_harness/systemc/calibrate.py` | Judge SystemC model predictions against the board: fidelity on the frozen split. | `doc/r38_isca_roadmap.md`, `evidence/r40/v3/systemc_m2/README.md` | care |
| `tools/performance_harness/systemc/capture.py` | Capture the real randomized NUCA test and export lossless word transactions. | none found | care |
| `tools/performance_harness/systemc/directional.py` | Directional validation and per-family breakdown of model predictions. | none found | care |
| `tools/performance_harness/systemc/dram_address_map.py` | Audit an actual AXI BO/PC map and export a native-model mapping input. | `spec/r41/hbm8/experiments/tlm8-window.json`, `spec/r41/stage1/experiments/physical-dram-map.json` | care |
| `tools/performance_harness/systemc/fit.py` | Fit the SystemC model's free timing constants on the R40-04 calibration split. | `evidence/r40/v3/systemc_m2/README.md` | care |
| `tools/performance_harness/systemc/generate_isa.py` | Generate model opcode identifiers from the project's canonical ISA schema. | none found | core |
| `tools/performance_harness/systemc/generate_sfu.py` | Emit the SFU coefficient tables from the product's xls/tnca_sfu_tables.x. | none found | core |
| `tools/performance_harness/systemc/opcode_inventory.py` | Static opcode inventory of compiled TNCA kernels, with model coverage. | `evidence/r40/v3/systemc_m1/README.md` | core |
| `tools/performance_harness/systemc/prediction_domain.py` | Numerical execution and timing-prediction eligibility are separate. | none found | care |
| `tools/performance_harness/systemc/predictions.py` | Turn run_graphs.py results into calibrate.py prediction rows. | `evidence/r41/stage1/memory_paths.md` | care |
| `tools/performance_harness/systemc/ramulator/hbm2_u280_pc.py` | Ramulator2 configuration of one U280 HBM2 pseudo channel (the slow DRAM path); exported to hbm2_u280_pc.yaml with `python3 -m ramulator export` | none found | core |
| `tools/performance_harness/systemc/rtl_oracle/matrix_oracle.py` | Record tnca_matrix accumulator contents from the product RTL (offline). | `evidence/r40/v3/systemc_m1/README.md`, `evidence/r41/stage2/candidates/F08/README.md` | core |
| `tools/performance_harness/systemc/rtl_oracle/rtl_oracle.py` | Record vector-FPU results from the product RTL for the model's numeric check. | `evidence/r40/v3/systemc_m1/README.md`, `evidence/r41/stage2/candidates/F03/README.md` | core |
| `tools/performance_harness/systemc/rtl_oracle/xpose_oracle.py` | The transposed write's sequencing, RTL capture against the model (R41 X8). | `evidence/r41/stage2/candidates/X8/README.md` | core |
| `tools/performance_harness/systemc/run.py` | Run a sealed TLM experiment. Model time is a prediction, never FPGA evidence. | `Makefile`, `doc/compiler_overview.md` | core |
| `tools/performance_harness/systemc/run_graphs.py` | Run ArchBench points end to end through the native SystemC model. | `Makefile`, `evidence/r40/v3/systemc_m1/README.md` | core |
| `tools/performance_harness/systemc/tlm_backend.py` | TNCA runtime backends that launch into the native SystemC model. | `evidence/r40/v3/systemc_m1/README.md`, `evidence/r41/hbm8/addr64/AUDIT.md` | core |
| `tools/performance_harness/systemc/validate.py` | GB10 functional/resource validation, RTL differential and trace perturbation. | `evidence/r41/pmac/model/README.md` | core |
| `tools/performance_harness/systemc/validate_chip.py` | Exercise native machine-word execution using actual assembler/compiler outputs. | `spec/r41/stage1/experiments/harness-smoke.json`, `spec/r41/stage1/experiments/slow-process-lifetime.json` | core |
| `tools/performance_harness/systemc/validate_dma.py` | Native transfer execution against an independent rank/stride/byte oracle. | `evidence/r40/v3/systemc_m2/README.md`, `evidence/r41/stage2/candidates/X8/README.md` | core |
| `tools/performance_harness/systemc/validate_matrix.py` | Directed end-to-end checks of the M1 instructions in the native executor. | `evidence/r41/pmac/model/README.md` | core |
| `tools/performance_harness/systemc/validate_matrix_dp.py` | Exact-oracle validation of the SystemC model's m.mma.dp.async arithmetic. | `evidence/r41/hbm_bw/mma_dp/README.md` | core |
| `tools/performance_harness/systemc/validate_matrix_dp_chip.py` | Directed end-to-end checks of m.mma.dp.async (ISA sec 9C.4) in the native executor. | `evidence/r41/hbm_bw/mma_dp/README.md` | core |
| `tools/performance_harness/systemc/validate_monitor.py` | Matched off/aggregate/trace experiments; host overhead is workload-specific. | `spec/r41/stage1/experiments/harness-smoke.json` | core |
| `tools/performance_harness/systemc/validate_numeric.py` | Independent exact-oracle validation of native numeric helpers on the GB10. | `evidence/r40/v3/final_implementation_plan/README.md`, `evidence/r40/v3/systemc_m1/README.md` | core |
| `tools/performance_harness/systemc/validate_roles.py` | Validate role isolation, explicit RF aliasing, barrier generations and faults. | called by `tools/performance_harness/systemc/validate_matrix.py` | core |

### ArchBench (`archbench/`)

The PyTorch-level benchmark: points, suites, platform runners, scoring and provenance.

| Tool | Purpose (from its docstring) | Referenced by | Class |
|---|---|---|---|
| `archbench/__init__.py` | ArchBench: one PyTorch-level benchmark, measured on TNCA and on GPUs. | `doc/compiler_overview.md`, `doc/evidence/r35/acceptance-linux.json` | core |
| `archbench/bounds.py` | The strict lower bound of a point on a platform: L = max over resources of | `doc/r38_stage1_status.md`, `doc/r38_stage1_tasks_and_acceptance.md` | core |
| `archbench/generate.py` | Greedy LLaMA generation on TNCA, the way a model is used. | `doc/r40_v3/plan.json`, `evidence/r40/R40-09/frozen_model_reference/README.md` | core |
| `archbench/hbm.py` | The HBM tier (R40-00): working sets that need external memory. | `doc/r40_tasks_and_acceptance.md`, `spec/r40/suites/hbm-tier.json` | core |
| `archbench/membound.py` | The R41 memory-bound tier (`membound`): the registered bandwidth suite. | `evidence/r41/hbm_bw/LINK_BUDGET.md`, `spec/r41/hbm_bw/membound/membound-large-1.json` | core |
| `archbench/models.py` | End-to-end model points. | `doc/r39_tasks_and_acceptance.md`, `spec/r38/evidence_registry.json` | core |
| `archbench/ops.py` | Operator points: the torch ops an LLM (and a little beyond) spends its time in. | `doc/r39_tasks_and_acceptance.md`, `spec/r38/evidence_registry.json` | core |
| `archbench/platforms/__init__.py` | Platform runners: tnca (simulator or board) and gpu. Each turns a Point | `doc/compiler_overview.md`, `doc/evidence/r35/acceptance-linux.json` | core |
| `archbench/platforms/common.py` | What every platform runner shares: the reference, the comparison, the | `doc/r40_tasks_and_acceptance.md`, `spec/r40/api_lock.json` | core |
| `archbench/platforms/gpu.py` | GPU: torch.compile -> Inductor -> Triton (and the libraries Inductor calls). | called by `tools/r40/v3_gpu_boundaries.py` | core |
| `archbench/platforms/tnca.py` | TNCA: torch.compile -> Inductor -> Triton -> tnca plugin -> Session. | `doc/r38_board_bringup.md`, `doc/r40_tasks_and_acceptance.md` | core |
| `archbench/provenance.py` | Provenance of an ArchBench record (archbench/PROTOCOL.md, protocol 2). | `doc/r40_tasks_and_acceptance.md` | core |
| `archbench/registry.py` | The registry: every ArchBench point, defined once at the PyTorch level. | `spec/r38/evidence_registry.json`, `spec/r41/stage1/lock.json` | core |
| `archbench/report.py` | Score and compare ArchBench records. | `doc/phase3_5_tasks_and_acceptance.md`, `evidence/r41/chat/DECODE_PROFILE.md` | core |
| `archbench/run.py` | Run ArchBench points on a platform and append one JSON record per point. | `Makefile`, `doc/compiler_overview.md` | core |
| `archbench/selftest.py` | ArchBench's own checks: the definitions are what they claim, and the | `Makefile`, `doc/r1_signoff.md` | core |
| `archbench/suite.py` | Frozen suites: what each tier contains, by content hash. | `Makefile`, `evidence/r41/hbm_bw/G4_DESIGN.md` | core |

### R41 analytic models, census and resource tools (`tools/r41/`)

Tools that read compiled code, recorded runs or Vivado reports and derive numbers; none types a result in.

| Tool | Purpose (from its docstring) | Referenced by | Class |
|---|---|---|---|
| `tools/r41/asm_resource_scan.py` | Accumulators and barriers named by compiled TNCA programs (R41 G2). | `evidence/r41/hbm_bw/RESOURCES.md` | core |
| `tools/r41/balance_model.py` | R41 run-2 throughput-matching model (analytic; no simulation is run here). | `evidence/r41/hbm_bw/BALANCE.md` | care |
| `tools/r41/bandwidth_closure_model.py` | Check V3 bandwidth closure with the native SystemC streaming-add program. | `evidence/r41/hbm_bw/AUDIT.md`, `evidence/r41/hbm_bw/STATUS.md` | care |
| `tools/r41/barrier_census.py` | Barrier census of compiled TNCA assembly (R41 G2, run-2 freeze). | none found | core |
| `tools/r41/barrier_slice_check.py` | Check every compiled launch's barrier immediates against the slice the runtime gives it. | `evidence/r41/hbm_bw/fma_dslx/barrier_slice/README.md` | care |
| `tools/r41/compare_tiled_batches.py` | Compare complete same-image default and tiled board batches by frozen point. | none found | one-off |
| `tools/r41/compare_torch_rtl.py` | Compare completed, warmed standard-torch RTL operator runs (never board evidence). | `evidence/r41/hbm_bw/AUDIT.md`, `evidence/r41/hbm_bw/hybrid_lsu/README.md` | care |
| `tools/r41/fp8_hub_compile_only.py` | Compile-only run of the Hub FP8 kernels for one profile (R41 G2 sizing). | `evidence/r41/hbm_bw/RESOURCES.md`, `evidence/r41/hbm_bw/fma_dslx/barrier_slice/README.md` | care |
| `tools/r41/fp8_hub_rtl.py` | The Hub FP8 block-scaled matmul through the standard PyTorch path on the board-top RTL simulation. | none found | one-off |
| `tools/r41/g4_g2_cuts_report.py` | G2's capability cuts one at a time on the run-1 image (RTL link probes). | `evidence/r41/hbm_bw/g4/g2_cuts/README.md` | one-off |
| `tools/r41/g4_issue_bound.py` | Issue and SFU ceilings of the compiled memory-bound kernels (G4, analytic). | `evidence/r41/hbm_bw/G4_DESIGN.md` | care |
| `tools/r41/g4_model_suite.py` | Run the registered memory-bound suite through the SystemC/TLM model (G4). | `evidence/r41/hbm_bw/G4_DESIGN.md` | care |
| `tools/r41/g4_report.py` | Tables of G4_DESIGN.md, generated from the recorded G4 runs. | `evidence/r41/hbm_bw/G4_DESIGN.md` | one-off |
| `tools/r41/g4_rtl_report.py` | RTL link probes of the tiled element-wise lowering against the default (G4). | `evidence/r41/hbm_bw/G4_DESIGN.md`, `evidence/r41/hbm_bw/g4/rtl_s4/README.md` | one-off |
| `tools/r41/hybrid_torch_rtl.py` | Standard torch numerics through the coupled wide RTL MMIO stand-in. | `evidence/r41/hbm_bw/AUDIT.md`, `evidence/r41/hbm_bw/hybrid_lsu/README.md` | care |
| `tools/r41/hybrid_trace_summary.py` | Summarize completed MMIO grids from the optional hybrid RTL observer. | `evidence/r41/hbm_bw/hybrid_lsu/README.md` | care |
| `tools/r41/kernel_census.py` | What the standard path compiles for each point: block sizes and instruction mix. | `evidence/r41/hbm_bw/LINK_BUDGET.md` | core |
| `tools/r41/link_budget_home.py` | Home-slice service ceilings at the run-1 image geometry (RTL simulation). | `evidence/r41/hbm_bw/LINK_BUDGET.md` | care |
| `tools/r41/link_budget_report.py` | Stage ceilings of the full link, computed from the recorded runs only. | `evidence/r41/hbm_bw/LINK_BUDGET.md` | one-off |
| `tools/r41/link_budget_rtl.py` | Stage counters of standard-path ArchBench points on the run-1 image RTL. | `evidence/r41/hbm_bw/G4_DESIGN.md`, `evidence/r41/hbm_bw/LINK_BUDGET.md` | care |
| `tools/r41/loop_issue_count.py` | Instructions per iteration of the innermost loop of compiled TNCA programs. | `evidence/r41/hbm_bw/mma_dp/README.md`, `evidence/r41/hbm_bw/mma_dp/issue/row_scalar/README.md` | core |
| `tools/r41/matrix_latency_synth_check.py` | Check Vivado's synthesized MAC_LAT for all 32 matrix pipeline cut masks. | none found | care |
| `tools/r41/membound_baseline_table.py` | Per-point, per-family and per-size tables of the membound board baseline. | `evidence/r41/hbm_bw/LINK_BUDGET.md`, `evidence/r41/hbm_bw/MEMBOUND_BASELINE.md` | care |
| `tools/r41/membound_manifests.py` | Board manifests for the baseline of the registered R41 memory-bound suite. | `evidence/r41/hbm_bw/LINK_BUDGET.md`, `evidence/r41/hbm_bw/g4/board_tiled2/README.md` | care |
| `tools/r41/operator_path_model_sweep.py` | Run standard torch add/silu against coupled, unbuilt memory-path hypotheses. | `evidence/r41/hbm_bw/AUDIT.md`, `evidence/r41/hbm_bw/STATUS.md` | one-off |
| `tools/r41/resource_usage.py` | Per-kernel resource use from a compile audit (R41 G2). | none found | core |
| `tools/r41/scf_sdpa_tlm_check.py` | Functional dynamic-SDPA check on the SystemC/TLM model, not a board benchmark. | none found | care |
| `tools/r41/stream_bandwidth_model.py` | Cycle-level capacity model for the independent per-PC AXI stream engines. | `evidence/r41/hbm_bw/AUDIT.md`, `evidence/r41/hbm_bw/R41F_PLAN.md` | care |
| `tools/r41/util_hier.py` | Resource tables for R41 G2 (evidence/r41/hbm_bw/RESOURCES.md). | `evidence/r41/hbm_bw/RESOURCES.md` | core |
| `tools/r41/vector_schedule_model_tests.py` | Compare independent-vector batching with finite native-model LSU resources. | `evidence/r41/hbm_bw/vector_schedule/README.md` | one-off |
| `tools/r41/wide_path_model_tests.py` | Explore coupled LSU/NoC/home widths with real native-chip numeric checks. | `evidence/r41/hbm_bw/wide_model/README.md` | one-off |

### R41 RTL bench drivers (`tools/r41/*_tests.py` and sweeps)

Drivers that elaborate and run an RTL bench for one R41 hardware feature (RTL simulation, never board evidence).

| Tool | Purpose (from its docstring) | Referenced by | Class |
|---|---|---|---|
| `tools/r41/barrier_alias_tests.py` | Core RTL checks for immediate barrier indices at slice boundaries. | `evidence/r41/hbm_bw/barrier_alias/README.md` | care |
| `tools/r41/g4_home_calibrate.py` | Service rate of the pipelined home (rtl/nuca/tnca_home_pipe.v), to calibrate | `evidence/r41/hbm_bw/G4_DESIGN.md` | one-off |
| `tools/r41/g4_home_pipe_tests.py` | tnca_home_pipe (R41 G4): randomised and directed regressions (RTL simulation). | `evidence/r41/hbm_bw/G4_DESIGN.md` | care |
| `tools/r41/g4_home_tests.py` | G4 home-slice options: byte-shadow regression of the wide slice (RTL simulation). | `evidence/r41/hbm_bw/G4_DESIGN.md`, `evidence/r41/hbm_bw/g4/strided/README.md` | care |
| `tools/r41/hbm_layout_tests.py` | Host DMA/RTL address-layout agreement and capability refusal boundaries. | `evidence/r41/acceptance/gate_after_rectification/README.md`, `evidence/r41/hbm_bw/g4/run2/BRINGUP.md` | care |
| `tools/r41/home_service_sweep.py` | Measure accepted/completed home-word service with checked read responses. | `evidence/r41/hbm_bw/wide_home/README.md` | one-off |
| `tools/r41/home_window_sweep.py` | Check cold service at image-like home settings and larger finite windows. | `evidence/r41/hbm_bw/wide_home/README.md` | one-off |
| `tools/r41/hybrid_core_tests.py` | Execute existing ISA programs on the real hybrid core and a wide memory fixture. | `evidence/r41/hbm_bw/hybrid_lsu/README.md` | care |
| `tools/r41/lsu_bank_tests.py` | Compare real multi-unit LSU data and memory effects to the legacy pair. | `evidence/r41/hbm_bw/lsu_bank/README.md` | care |
| `tools/r41/lsu_hybrid_tests.py` | Full hybrid LSU completions and memory effects versus the legacy pair. | `evidence/r41/hbm_bw/G4_DESIGN.md`, `evidence/r41/hbm_bw/g4/strided/README.md` | care |
| `tools/r41/lsu_line_tests.py` | Finite shared vector-line planner: byte oracle, backpressure and throughput. | `evidence/r41/hbm_bw/G4_DESIGN.md`, `evidence/r41/hbm_bw/g4/tag_tests/README.md` | care |
| `tools/r41/mtx_vrf_reg_tests.py` | Lockstep RTL regression for the optional matrix VRF response register. | `Makefile` | care |
| `tools/r41/sfu_parallel_tests.py` | Verify parallel SFU bit equivalence, launch timing and reset behavior. | `evidence/r41/hbm_bw/sfu/README.md` | care |
| `tools/r41/tensor_line_tests.py` | The line-wide, multi-command transfer engine against the word engine | `evidence/r41/hbm_bw/G4_DESIGN.md` | care |
| `tools/r41/upload_path_tests.py` | Host upload path through the multicore MMIO backend on the RTL stand-in. | `doc/r41_chat.md` | care |
| `tools/r41/wide_home_tests.py` | Wide-home byte shadow, original narrow regression, atomics and MMIO. | `evidence/r41/hbm_bw/wide_home/README.md` | care |
| `tools/r41/wide_noc_tests.py` | Real mesh/NI packet integrity, backpressure and sustained issue checks. | `evidence/r41/hbm_bw/G4_DESIGN.md`, `evidence/r41/hbm_bw/g4/tag_tests/README.md` | care |

### R41 guarded board queue (`tools/r41/run2_board/`): reusable core

Plan, stage, identity and idle machinery for board runs that refuse to touch a busy card or a changed image.

| Tool | Purpose (from its docstring) | Referenced by | Class |
|---|---|---|---|
| `tools/r41/run2_board/auto.py` | Run the author's approved run-2 board plan after a guarded idle interval. | `spec/r41/stage2/board/r3-clean-repro-queue.json`, `spec/r41/stage2/board/r3-q41-bf16-pilot.json` | care |
| `tools/r41/run2_board/identity.py` | Capture and qualify the approved run-2 clock, then execute identity smoke. | `Makefile`, `doc/compiler_overview.md` | care |
| `tools/r41/run2_board/idle_monitor.py` | Continuously record a monotonic idle interval without touching the card. | called by `tools/r41/run2_board/candidate_queue.py` | care |
| `tools/r41/run2_board/manifests.py` | Derive run-2 board manifests from the registered run-1 workloads. | `evidence/r41/acceptance/STATUS.md`, `evidence/r41/hbm_bw/AUDIT.md` | care |
| `tools/r41/run2_board/prepare_plan.py` | Publish a complete guarded run-2 board plan after all preflight reviews. | none found | care |
| `tools/r41/run2_board/registrations.py` | Register run-2 stage groups and their explicit workload result checks. | none found | care |
| `tools/r41/run2_board/source_review.py` | Verify the staged frozen run-2 source tree against its original archive. | `spec/r41/run2/board/r2-s2-identity.json` | care |
| `tools/r41/run2_board/stage.py` | Run a registered sequence of board experiments and publish its result. | called by `tools/r41/chat_speed/queue_event_board.py` | care |
| `tools/r41/run2_board/watch_plan.py` | Wait for a complete run-2 plan, then start its guarded board queue once. | none found | care |

### R41 guarded board queue (`tools/r41/run2_board/`): chat-version-stamped and campaign-specific scripts

Waiters, reviews and queues written for one campaign (candidate image, Q41/XL41 pilots, chat retries). Several are copies of each other differing in hard-coded paths, ids and PIDs.

| Tool | Purpose (from its docstring) | Referenced by | Class |
|---|---|---|---|
| `tools/r41/run2_board/candidate_cdc_diff.py` | Compare candidate CDC endpoints to the reviewed baseline and run-2. | none found | one-off |
| `tools/r41/run2_board/candidate_clock.py` | Review the corrected candidate's original timing and xclbin metadata. | none found | one-off |
| `tools/r41/run2_board/candidate_identity.py` | Capture the corrected candidate's board clock and run identity smoke. | called by `tools/r41/run2_board/candidate_queue.py` | one-off |
| `tools/r41/run2_board/candidate_manifests.py` | Prepare the same registered workloads for a corrected candidate xclbin. | none found | one-off |
| `tools/r41/run2_board/candidate_map.py` | Extract the candidate xclbin HBM map without opening an FPGA device. | none found | one-off |
| `tools/r41/run2_board/candidate_queue.py` | Run the corrected image's complete board stages after explicit approval. | called by `tools/r41/run2_board/candidate_waiter.py` | one-off |
| `tools/r41/run2_board/candidate_resume.py` | Resume a stopped candidate queue without reprogramming or discarding evidence. | called by `tools/r41/run2_board/q41_pilot_waiter.py` | one-off |
| `tools/r41/run2_board/candidate_source.py` | Verify the corrected frozen image source and bind it to a candidate hash. | called by `tools/r41/run2_board/candidate_manifests.py` | one-off |
| `tools/r41/run2_board/candidate_waiter.py` | Start the corrected-image queue once its reviews and approval exist. | none found | one-off |
| `tools/r41/run2_board/chat_priority_queue.py` | (no docstring; from the code) waits for the candidate image to be loaded and the card idle at 100 MHz, then runs the registered stages chat-vl and chat-q35 through... | none found | unclear |
| `tools/r41/run2_board/chat_stall_watch.py` | Preserve and stop only a demonstrably idle priority chat so the queued diagnostic can run. | none found | one-off |
| `tools/r41/run2_board/chat_trace_stall_watch.py` | Advance from the traced VL diagnostic only after ten minutes of no measurable work. | none found | one-off |
| `tools/r41/run2_board/chat_trace_waiter.py` | Retry the preserved VL stall with timed stacks after the two priority chats. | none found | one-off |
| `tools/r41/run2_board/chat_trace_wrapper.py` | Run an unchanged chat script while periodically preserving Python stacks. | none found | one-off |
| `tools/r41/run2_board/clean_repro_waiter.py` | Rebuild and rerun bounded board points from a cache-free source archive. | called by `tools/r41/run2_board/perf_recheck_waiter.py` | one-off |
| `tools/r41/run2_board/clock_review.py` | Pre-load review of the author's specific run-2 HBM-clock exception. | none found | one-off |
| `tools/r41/run2_board/grid_rerun_waiter.py` | Rerun the restored grid admission case after the registered run-1 rechecks. | none found | one-off |
| `tools/r41/run2_board/hbm_review.py` | Check the original run-2 xclbin's previously extracted HBM mapping. | none found | one-off |
| `tools/r41/run2_board/nohost_prepare.py` | Register a candidate no-host-feedback chat retry in a separate staged tree. | none found | unclear |
| `tools/r41/run2_board/nohost_prepare_v3.py` | Register a candidate no-host-feedback chat retry in a separate staged tree. | none found | unclear |
| `tools/r41/run2_board/nohost_prepare_v4.py` | Register full quality, session and script stages for grouped-tail device chat. | none found | unclear |
| `tools/r41/run2_board/nohost_prepare_v5.py` | Register full quality, session and script stages for grouped-tail device chat. | none found | one-off |
| `tools/r41/run2_board/nohost_queue.py` | Run the two registered chat retries after the live chats and VL diagnostic release the card. | called by `tools/r41/run2_board/nohost_queue_v3.py` | unclear |
| `tools/r41/run2_board/nohost_queue_v3.py` | Run the two registered chat retries after the live chats and VL diagnostic release the card. | called by `tools/r41/run2_board/nohost_queue_v4.py` | unclear |
| `tools/r41/run2_board/nohost_queue_v4.py` | Run the two registered chat retries after the live chats and VL diagnostic release the card. | called by `tools/r41/run2_board/nohost_queue_v5.py` | unclear |
| `tools/r41/run2_board/nohost_queue_v5.py` | Run the two registered chat retries after the live chats and VL diagnostic release the card. | none found | one-off |
| `tools/r41/run2_board/perf_recheck_waiter.py` | Run registered regression checks after the candidate campaign, then restore run 1. | called by `tools/r41/run2_board/grid_rerun_waiter.py` | one-off |
| `tools/r41/run2_board/probe_chat_feedback_tlm.py` | (no docstring) mocks a one-layer model and a session around chat.py on the TLM backend; the question it answers is unclear | none found | unclear |
| `tools/r41/run2_board/q35_timeout_recovery.py` | Resume missing Qwen3.5 quality prompts only after the original stage times out. | none found | one-off |
| `tools/r41/run2_board/q41_full_waiter.py` | Continue Q41 core/tail after verified preregistered pilots, without loading. | called by `tools/r41/run2_board/xl41_matrix_waiter.py` | one-off |
| `tools/r41/run2_board/q41_pilot_waiter.py` | Wait for the completed candidate campaign, then run the registered Q41 pilot. | `spec/r41/stage2/board/r3-q41-full-queue.json` | one-off |
| `tools/r41/run2_board/supersede_on_compile.py` | Switch the known graph-reset retry only while its next wrapper is compiling. | none found | one-off |
| `tools/r41/run2_board/supersede_on_prompt.py` | Switch a known failing VL quality run after a completed, flushed prompt. | none found | one-off |
| `tools/r41/run2_board/xl41_extra_waiter.py` | Run bounded gather, attention and recurrence pilots after verified XL matrix tests. | called by `tools/r41/run2_board/clean_repro_waiter.py` | one-off |
| `tools/r41/run2_board/xl41_matrix_waiter.py` | Run registered XL matrix pilots after verified full Q41 sessions, without loading. | called by `tools/r41/run2_board/xl41_extra_waiter.py` | one-off |
| `tools/r41/run2_board/xl41_pilot_waiter.py` | Wait for a verified Q41 pilot, then run the registered XL41 GEMV pilot. | `spec/r41/stage2/board/r3-q41-full-queue.json` | one-off |
| `tools/r41/run2_board/xl41_stream_waiter.py` | Wait for verified XL41 GEMV, then run the registered >2 GiB stream pilot. | called by `tools/r41/run2_board/q41_full_waiter.py` | one-off |

### R41 post-link, x86 link probes and unit screens (`tools/r41/run2_postlink/`, `x86_link_dryrun/`, `fmax/`)

Read-only post-route analysis, x86 link dry run and out-of-context unit frequency screens.

| Tool | Purpose (from its docstring) | Referenced by | Class |
|---|---|---|---|
| `tools/r41/fmax/regwrap.py` | Register wrapper for a combinational module (R41 G2 unit screens). | called by `tools/r41/fmax/run_unit.sh` | care |
| `tools/r41/fmax/run_unit.sh` | Unit screen on the x86 host: run_unit.sh NAME TOP ROUTE FILE.v ...; wraps TOP in registers (regwrap.py), runs unit_ooc.tcl, prints unitsum.py | none found | care |
| `tools/r41/fmax/unit_ooc.tcl` | Out-of-context Vivado batch for one unit screen: -tclargs TOP OUT_DIR ROUTE(0/1) FILE.v ... | called by `tools/r41/fmax/run_unit.sh` | care |
| `tools/r41/fmax/unitsum.py` | One line per unit screen: LUT, logic/memory, FF, CARRY8, DSP, CLB, WNS. | called by `tools/r41/fmax/run_unit.sh` | care |
| `tools/r41/fmax/xlsc.sh` | R41 G2 FMA/MAC rebuild: DSLX to combinational Verilog with the pinned XLS | none found | care |
| `tools/r41/run2_postlink/analyze_run2.tcl` | One open_checkpoint of the routed checkpoint with 2 threads; writes the report set that check_run2.py reads; never writes a checkpoint or touches a board | `evidence/r41/hbm_bw/g4/run2/BRINGUP.md` | care |
| `tools/r41/run2_postlink/check_irq_sticky.py` | Check the routed platform IRQ source's LUT-level sticky condition. | `evidence/r41/hbm_bw/g4/run3/route_prepackage/README.md` | one-off |
| `tools/r41/run2_postlink/check_run2.py` | R41 run 2: check a routed image's reports against the G4 screens and the | `evidence/r41/hbm_bw/g4/run2/BRINGUP.md` | care |
| `tools/r41/run2_postlink/explore_hbm_clock.tcl` | Recreates the 450 MHz generated clock, then analyses 433.4 MHz on the routed checkpoint (HBM-clock exception review) | none found | one-off |
| `tools/r41/run2_postlink/inspect_hbm_clock.tcl` | Inspects the HBM clock constraints of the routed checkpoint; changes no constraint, writes no checkpoint, establishes no closure | `evidence/r41/hbm_bw/g4/run3/route_prepackage/README.md` | one-off |
| `tools/r41/run2_postlink/post_run2.sh` | After the link ends: read-only post-route analysis (analyze_run2.tcl) then check_run2.py; waits for build/attempt.exit; never touches the card | `evidence/r41/HANDOFF_RUN2.md`, `evidence/r41/hbm_bw/g4/run2/BRINGUP.md` | care |
| `tools/r41/run2_postlink/review_candidate_cdc.py` | Record the individually reviewed platform-scoped candidate CDC findings. | none found | one-off |
| `tools/r41/run2_postlink/trace_platform_irq.tcl` | Traces the platform IRQ path in the routed checkpoint as review evidence (not a CDC disposition) | called by `tools/r41/run2_postlink/analyze_run2.tcl` | one-off |
| `tools/r41/x86_link_dryrun/make_tiny.py` | Create an isolated, non-executable-on-board AXI interface link probe. | called by `tools/r41/x86_link_dryrun/run_tiny.sh` | care |
| `tools/r41/x86_link_dryrun/resource_watch.py` | Run a command in a fresh session, cap summed process-tree RSS, log samples. | called by `tools/r41/x86_link_dryrun/run_tiny.sh` | core |
| `tools/r41/x86_link_dryrun/run_tiny.sh` | Runs the tiny link probe in a disposable source copy on the x86 host; never uses a card | none found | care |
| `tools/r41/x86_link_dryrun/tiny_place_hook.tcl` | Placement hook for the tiny AXI link probe; its header says never ship it | called by `tools/r41/x86_link_dryrun/run_tiny.sh` | care |

### R41 chat-speed experiments (`tools/r41/chat_speed/`)

No-board TLM A/B scripts for chat launch overhead.

| Tool | Purpose (from its docstring) | Referenced by | Class |
|---|---|---|---|
| `tools/r41/chat_speed/bench_code_cache_dp.py` | Synthetic code-cache pressure replay plus exact DP compute, no board. | `evidence/r41/chat/speed/README.md` | one-off |
| `tools/r41/chat_speed/bench_hoisted_fp8.py` | Compare official fused FP8 with exact separately quantised FP8 on DP TLM. | none found | one-off |
| `tools/r41/chat_speed/bench_runtime_events.py` | Runtime-history A/B: standard Torch matmul on DP TLM, no board. | none found | one-off |
| `tools/r41/chat_speed/queue_event_board.py` | Run the event-index board comparison after the priority chat stage passes. | none found | one-off |
| `tools/r41/chat_speed/queue_activation_board.py` | Queue the gated activation-staging board session after prior tests pass. | none found | care |
| `tools/r41/chat_speed/queue_after_quality_failure.py` | Preserve the completed quality failure and queue gated board speed sessions. | none found | care |

### Model drivers and diagnostics (`tools/models/`)

Chat, decode profile, quality checks, checkpoint loaders and the diagnostics of the VL reason-prompt investigation.

| Tool | Purpose (from its docstring) | Referenced by | Class |
|---|---|---|---|
| `tools/models/chat.py` | Chat with the staged Qwen FP8 checkpoints through the standard Torch path. | `doc/r41_chat.md`, `evidence/r41/acceptance/STATUS.md` | core |
| `tools/models/chat_numerical_floor.py` | Measure a reduced-depth chat model's CPU W8A8 scale-rounding floor. | none found | care |
| `tools/models/chat_quality.py` | Registered chat-quality check (spec/r41/chat/quality-v1.json): CPU reference answers, then device scoring. | `doc/r41_chat.md`, `evidence/r41/chat/quality-ref/README.md` | core |
| `tools/models/compare_vl_reason_logits.py` | Compare saved CPU or TLM reason-prompt logits without modifying them. | none found | one-off |
| `tools/models/decode_profile.py` | Per-launch profile of the chat driver's decode steps (observation only). | `evidence/r41/chat/DECODE_PROFILE.md` | core |
| `tools/models/decode_profile_report.py` | Per-kernel and per-class tables for one steady decode step, from a decode_profile.py run. | `evidence/r41/chat/DECODE_PROFILE.md` | core |
| `tools/models/diagnose_chunked_prefill.py` | Compare full and chunked CPU prefill on the same checkpoint, inputs and cache length. | none found | one-off |
| `tools/models/diagnose_event_scans.py` | CPU-only measure of historical-event scans in an actual TNCA Session. | `evidence/r41/chat/run3_priority/README.md` | one-off |
| `tools/models/diagnose_q35_warmup_cpu.py` | Check full-Qwen3.5 cache reset after a dummy decode warmup on CPU. | none found | one-off |
| `tools/models/diagnose_vl_reason_attention.py` | Compare layer-2 CPU attention masks and rotary inputs at the reason prompt's short tail. | none found | one-off |
| `tools/models/diagnose_vl_reason_fp8.py` | Compare DP-group and exact FP8 dot arithmetic on a real reason-prompt activation. | none found | one-off |
| `tools/models/diagnose_vl_reason_fp8_all.py` | Compare grouped DP dots and exact W8A8 for seven real layer-2 projections. | none found | one-off |
| `tools/models/diagnose_vl_reason_fp8_tlm.py` | Run the reason prompt's captured layer-2 FP8 Q projection on DP-profile TLM. | none found | one-off |
| `tools/models/diagnose_vl_reason_head_tlm.py` | Compare selected real VL LM-head logits on CPU and DP-profile TLM. | none found | one-off |
| `tools/models/diagnose_vl_reason_layers.py` | Trace the real reason prompt on CPU to locate the chunk-tail divergence. | none found | one-off |
| `tools/models/diagnose_vl_reason_logits.py` | Compare reason-prompt prefill logits across CPU paths and a DP TLM run. | none found | one-off |
| `tools/models/diagnose_vl_reason_norm_amplification.py` | Measure FP8 V projection sensitivity to captured CPU versus TLM norm outputs. | none found | one-off |
| `tools/models/diagnose_vl_reason_norm_cpu_compile.py` | Compare eager CPU, CPU-Inductor and TLM RMSNorm for a real prompt input. | none found | one-off |
| `tools/models/diagnose_vl_reason_norm_tlm.py` | Compare layer-2 input normalization on the full reason-prompt TLM input. | none found | one-off |
| `tools/models/diagnose_vl_reason_sdpa.py` | Compare actual layer-2 SDPA operands for the reason prompt on CPU. | none found | one-off |
| `tools/models/diagnose_vl_reason_sdpa_replay.py` | Replay saved reason-prompt SDPA operands with controlled CPU query shapes. | none found | one-off |
| `tools/models/diagnose_vl_reason_sdpa_tlm.py` | Replay captured CPU reason-prompt attention operands on DP-profile TLM. | none found | one-off |
| `tools/models/diagnose_vl_reason_sdpa_tlm_math.py` | Compare captured prompt attention with DP TLM and CPU math SDPA. | none found | one-off |
| `tools/models/diagnose_vl_reason_sve_trace.py` | Locate the first CPU-flash versus SVE prototype divergence during real prefill. | none found | one-off |
| `tools/models/diagnose_vl_warmup_image.py` | Compare full-VL CPU image chat with and without an earlier decode warmup. | none found | one-off |
| `tools/models/diagnose_vl_warmup_multiturn.py` | Compare full-VL CPU multi-turn chat with and without dummy decode warmup. | none found | one-off |
| `tools/models/fetch_flux2_klein.py` | Fetch the pinned FLUX.2 [klein] 4B files into the standard Hugging Face cache layout. | `evidence/r41/fp8/flux/README.md` | care |
| `tools/models/fetch_qwen35_text.py` | Fetch the authentic pinned Qwen3.5-2B snapshot; no inference or device use. | none found | care |
| `tools/models/fetch_qwen_fp8.py` | Fetch the pinned Qwen FP8 checkpoints into the standard Hugging Face cache layout. | `evidence/r41/fp8/qwen/README.md` | care |
| `tools/models/flux2_klein_run.py` | FLUX.2 [klein] 4B end to end through the standard torch path, stage by stage. | `evidence/r41/fp8/flux/README.md`, `spec/r41/stage2/board/r3-clean-repro-queue.json` | care |
| `tools/models/fp8_blockwise_ref.py` | Independent reference of fine-grained (block-wise) W8A8 FP8 linear math, as computed by the Hub | `doc/r41_chat.md`, `evidence/r41/fp8/qwen/README.md` | care |
| `tools/models/fp8_checkpoint.py` | Generic loader for safetensors checkpoints that carry `_quantization_metadata` (static FP8 W8A8). | `evidence/r41/fp8/flux/README.md`, `spec/r41/stage2/board/r3-clean-repro-queue.json` | core |
| `tools/models/prepare_vl_reason_teacher.py` | Stage registered VL reason prompt tensors for layer-local TLM diagnosis. | none found | one-off |
| `tools/models/qwen35_reference.py` | Official all-layer Qwen3.5-2B CPU reference; no TNCA output is read. | `spec/r41/stage2/board/r3-clean-repro-queue.json`, `spec/r41/stage2/board/r3-q41-bf16-pilot.json` | care |
| `tools/models/qwen35_run.py` | lovedheart/Qwen3.5-4B-FP8 end to end through the standard torch path, stage by stage. | `evidence/r41/fp8/qwen/README.md` | care |
| `tools/models/qwen35_stage2_pilot.py` | Bounded, teacher-forced Q41 pilot of the pinned full BF16 language model. | none found | one-off |
| `tools/models/qwen3vl_run.py` | Qwen/Qwen3-VL-2B-Instruct-FP8 end to end through the standard torch path, stage by stage. | `evidence/r41/fp8/qwen/README.md` | care |
| `tools/models/qwen_fp8_lib.py` | Shared runner for transformers checkpoints in the fine-grained FP8 format (quantization_config.quant_method | `evidence/r41/chat/launch_overhead/README.md`, `evidence/r41/fp8/qwen/README.md` | core |
| `tools/models/qwen_tnca_layers.py` | Qwen FP8 checkpoints on tnca, layer by layer (R41 FP8 validation). | `doc/r41_chat.md`, `evidence/r41/fp8/models_tnca/README.md` | care |
| `tools/models/run_cpu_quality_with_warmup.py` | Run existing full CPU quality checks after one dummy decode warmup. | none found | one-off |
| `tools/models/run_precision_cast_experiment.py` | Run a diagnostic script with the non-default Inductor precision-cast setting. | none found | care |
| `tools/models/vision_split_check.py` | Check that running the vision tower once per image equals one joint call over all images. | `doc/r41_chat.md`, `evidence/r41/acceptance/STATUS.md` | care |
| `tools/models/xl41_attention_pilot.py` | Bounded standard-Torch pilot for frozen XL41 causal ragged GQA. | none found | one-off |
| `tools/models/xl41_gather_pilot.py` | Bounded XL41 gather pilot preserving the formula and probing boundary rows. | none found | one-off |
| `tools/models/xl41_gemv_pilot.py` | Bounded standard-Torch pilot for the frozen XL41 BF16 GEMV point. | none found | one-off |
| `tools/models/xl41_matrix_pilot.py` | Bounded standard-Torch pilots for frozen XL41 transpose and matrix points. | none found | one-off |
| `tools/models/xl41_recurrence_pilot.py` | Exploratory standard-Torch FP32 gated recurrence under the XL41 shape. | none found | one-off |
| `tools/models/xl41_stream_pilot.py` | Bounded standard-Torch pilot for the frozen >2 GiB XL41 stream point. | none found | one-off |

### Numeric oracles (`tools/model/`)

Exact reference semantics for the float formats, FMA, f2i, packed dot product, and the vector generators for RTL benches.

| Tool | Purpose (from its docstring) | Referenced by | Class |
|---|---|---|---|
| `tools/model/f2i.py` | f2i.py -- ISA sec 8.4 binary32 -> int32 / uint32 reference, from the ISA text. | `Makefile`, `evidence/r40/v3/final_image_candidate/README.md` | core |
| `tools/model/fdp.py` | fdp.py -- exact reference for m.mma.dp, the packed dot-product MAC lane. | `Makefile`, `doc/conformance.md` | core |
| `tools/model/floatfmt.py` | Exact reference for the five TNCA float formats. | `Makefile`, `doc/xls.md` | core |
| `tools/model/fp32.py` | fp32.py -- exact binary32 reference semantics for the TNCA vector FP unit. | `Makefile`, `doc/xls.md` | core |
| `tools/model/gen_cvt_vectors.py` | Generate vectors for tb/xls/tb_cvt_xls.v from the exact oracle. | `Makefile` | core |
| `tools/model/gen_decoder_vectors.py` | Generate encoding-legality vectors for tb_decoder. | `Makefile`, `evidence/r40/v3/systemc_m1/README.md` | core |
| `tools/model/gen_f2i_vectors.py` | Vectors for tb/core/tb_vector_fpu_f2i.v: binary32 -> int32/uint32 against the | `Makefile` | core |
| `tools/model/gen_fma_vectors.py` | Generate differential test vectors for tnca_fp32_fma. | `Makefile` | core |
| `tools/model/gen_fpu_multi_vectors.py` | Vectors for tb/xls/tb_fpu_multi.v -- multi-precision FMA. | `Makefile` | core |
| `tools/model/gen_sfu_tables.py` | Generate xls/tnca_sfu_tables.x -- the SFU's interpolation coefficients. | `Makefile`, `doc/xls.md` | core |
| `tools/model/gen_sfu_vectors.py` | Generate SFU vectors and the reference the hardware is measured against. | `Makefile` | core |

### ISA toolchain (`tools/isa/`)

Canonical ISA schema export, downstream generation, consistency and pin checks.

| Tool | Purpose (from its docstring) | Referenced by | Class |
|---|---|---|---|
| `tools/isa/check_consistency.py` | check_consistency.py -- the two existing ISA sources must agree. | `Makefile`, `doc/architecture_overview.md` | core |
| `tools/isa/export_schema.py` | export_schema.py -- emit the canonical machine-readable TNCA ISA schema. | `Makefile`, `doc/architecture_overview.md` | core |
| `tools/isa/gen.py` | gen.py -- generate every downstream ISA consumer from the canonical schema. | `Makefile`, `doc/architecture_overview.md` | core |
| `tools/isa/gen_support_matrix.py` | Generate doc/support_matrix.md -- doc/phase1_tasks_and_acceptance.md R1-01. | `Makefile`, `doc/architecture_overview.md` | core |
| `tools/isa/gen_target.py` | gen_target.py -- generate the TNCA target description the compiler consumes. | `Makefile`, `doc/architecture_overview.md` | core |
| `tools/isa/verify_pin.py` | verify_pin.py -- confirm the pinned toolchain still describes upstream. | `Makefile`, `spec/r1_toolchain.lock.json` | core |

### Unit tests

Tests are listed by glob. Whether a test needs torch or a built model is not stated in all of them; run it and read it.
`make r41-harness-tests` runs `test_experiment.py`, `test_board_tools.py`, `test_stage1_acceptance.py`,
`test_stage_contract.py` and `systemc/test_run_graphs.py`; `make harness-catalog` runs the catalog check and its test.

| Tests | Cover | Class |
|---|---|---|
| `tools/performance_harness/test_*.py` | Experiment runner and budget, board tools, stage contracts and acceptance, frequency preflight, fragment bits, hbm8, locality, old mapping, Inductor tuning, measurement scope, model inventory (26 files, includes `test_catalog_check.py`) | core |
| `tools/performance_harness/systemc/test_*.py` | DRAM address maps, fixed point, native hbm8, host handoff, run_graphs, slow-process lifetime | core |
| `tools/r41/run2_board/test_*.py` | Guarded queue (`test_auto.py`), manifests and the campaign waiters' refusal rules | care |
| `tools/r41/test_*.py` | `test_barrier_slice_check.py`: missing evidence must not become a passing verdict | core |
| `tools/r41/chat_speed/test_*.py` | `test_equal_rows_tlm.py`: equal-address and unequal-mask cases on the DP TLM | one-off |
| `tools/models/test_*.py` | Chat driver, device feedback, FP8 checkpoint loader, Q35 pilot | care |
<!-- END catalog -->

### Needs a decision

Nothing below was changed or deleted. The author decides.

Superseded by a later copy (the copies differ only in hard-coded paths, ids, waiting PIDs and, for v5, an added
dummy-decode-warmup line, as read with `diff`):
- `tools/r41/run2_board/nohost_prepare.py`, `nohost_prepare_v3.py`, `nohost_prepare_v4.py` (latest: `nohost_prepare_v5.py`).
- `tools/r41/run2_board/nohost_queue.py`, `nohost_queue_v3.py`, `nohost_queue_v4.py` (latest: `nohost_queue_v5.py`).

Unclear:
- `tools/r41/run2_board/chat_priority_queue.py`: no docstring; from the code it waits for one hard-coded candidate image
  at 100 MHz and runs the stages `chat-vl` and `chat-q35`. Superseded by later queues or not: unclear.
- `tools/r41/run2_board/probe_chat_feedback_tlm.py`: no docstring; mocks a one-layer model around `chat.py` on the TLM
  backend; the question it answered is unclear.

No reference found (purpose is clear from the docstring, use is not): `tools/r41/matrix_latency_synth_check.py`,
`scf_sdpa_tlm_check.py`, `fp8_hub_rtl.py`, `compare_tiled_batches.py` (the last is cited only by evidence JSON/README of
`g4/board_tiled2`), `tools/r41/fmax/xlsc.sh`.

## Known pitfalls

Found in the R41 rounds. The first group is recorded in `evidence/r41/HANDOFF_RUN2.md`; the others were reported by the
R41 sessions and are not backed by a retained log in this guide.

- Strict `make compiler` counts skipped tests as a failure: build the RTL benches first (see the freeze gate in
  `evidence/r41/HANDOFF_RUN2.md` section 5).
- The GB10 `~/tc_env/torchenv` has no `transformers`; use `~/flux-venv` (HANDOFF_RUN2 section 3). Never `pip install`
  into `torchenv`.
- A stale editable `tnca` install can shadow the tree under test: set `PYTHONPATH` to the tree and check
  `tnca.__file__` before trusting a result.
- `gh pr edit` fails on the Projects-classic GraphQL error; use `gh api -X PATCH repos/OWNER/REPO/pulls/N -f body=...`.
- The x86 link needs `vitis_hls` and `unzip` on the host; start the link detached with `setsid nohup`
  (a foreground tool call over 120 s is backgrounded and may die).
- `link.sh` can report `LINK_FAILED freq-moved` when `v++` auto-scales a platform clock (run 2: `hbm_aclk` 450 to 441
  MHz because of two failing setup endpoints). The author accepted it as a declared exception
  (`evidence/r41/HANDOFF_RUN2.md` section 11); the guard stays: a moved frequency is never silently accepted.
- Commands on the servers longer than 120 s are backgrounded by the helper; do not poll a heavy job in the foreground.
- A strict check can pass for the wrong reason: `check_run2.py` exits 1 when any hard gate fails even if the kernel clock
  meets timing (run 2: `post.exit = 1` only because of the platform WNS).
- Hard-coded run identities: `idle_monitor.py`, `chat_priority_queue.py`, `membound_baseline_table.py`, `clock_qualify.py`
  and the `nohost_*` copies name one image, home directory or archive. Edit or parameterise before use.
- Never `pkill -f` on shared servers; kill your own processes by PID.

## Future work

Recommendations only; nothing was changed.

1. Parameterise the `nohost_*`, `chat_trace_*`, `q41_*`/`xl41_*` waiters into one campaign runner that takes the
   registration, image identity and the predecessor to wait for as arguments. They share the same guard logic
   (image hash, 15-minute idle, flock) in many near-identical copies; a fix to one is not applied to the others.
2. Consolidate the comparison scripts (`pipeline_compare.py`, `pipeline_admission_compare.py`, `compare_fragments.py`,
   `compare_fragment_bits.py`, `memory_service_compare.py`, `acceptance_compare.py`, `compare_tiled_batches.py`) around
   one paired-record comparison library with per-experiment adapters.
3. Replace the many one-off RTL bench drivers (`tools/r41/*_tests.py`, sweeps) with a table-driven registry and Make
   targets (only `mtx_vrf_reg_tests.py` has a Make rule today), so a regression suite exists for the features they
   cover.
4. Generalise the stage-specific case drivers (`hbm8_*`, `addr64_case.py`, `fragment_case.py`, `locality_case.py`,
   `memory_service_case.py`) into one "standard Torch device-window case" driver.
5. Give the model-diagnostic scripts under `tools/models/diagnose_vl_reason_*.py` (about twenty, one per probe) a shared
   driver, or archive them with their evidence.
6. Make evidence-regenerating reports (`g4_report.py`, `link_budget_report.py`, `membound_baseline_table.py`) take the
   archive and image identity as arguments so a second run can reuse them.
7. Give `idle_monitor.py` and `prepare_plan.py` defaults that are not tied to `~/r41run3`.
8. Add `--help` smoke tests (import-only) for the tools that take only positional or `sys.argv` input
   (`loop_issue_count.py`, `resource_usage.py`, `util_hier.py`, `asm_resource_scan.py`).
9. Move the shell/Tcl helpers of the unit screens and post-link flow next to `flow/fpga/` once the flow is stable.

## Methodology: hypotheses, baselines, bounds, instrumentation

The text below is the original harness guide (R40), kept because it is still correct. Updates: R41 audits use the 113
original points (union of the frozen sets), and the R40 work-package ledger referenced here is historical for R41, whose
ledger is `evidence/r41/acceptance/stage2_index.md`.

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
for event-driven complete-chip exploration on the GB10. Since M1 it executes
the frozen suites' compiled kernels end to end through the standard runtime
path (`systemc/run_graphs.py`), with RTL-exact numerics; its timing is
uncalibrated and its architecture coverage incomplete. Consult its coverage
table before use.
Its memory replay path is conditional on captured traffic and cannot substitute
for native program execution or establish complete-chip prediction accuracy.
RTL remains an independent offline validation source, outside the TLM execution
path. Socket monitors and recording policies are described in the model README.
Use `diagnose_systemc.py --report RUN/report.json --out diagnosis.json` to audit
a sealed run and expose resource-pressure hypotheses; `--candidate` compares
a compatible sensitivity experiment. Uncalibrated predictions cannot justify
an implementation run by themselves. No model build or simulation should run on the FPGA implementation host.

This workflow operates within the existing R40 work packages and frozen
acceptance contract. Start with the [required-gate blocker ledger](../../evidence/r40/v2/convergence/README.md).
Begin progress reports with changes to required-gate status. Diagnostic test
counts and commit counts do not establish acceptance progress. For each
unresolved gate, record the current blocker, supporting evidence, next
experiment command, and follow-up actions for both possible outcomes.

### 1. Define the hypothesis and establish experiment provenance

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

### 2. Establish an uninstrumented baseline

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

### 3. Screen candidates using conservative bounds

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

### 4. Instrument execution to resolve specific hypotheses

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

### 5. Quantify instrumentation overhead and uncertainty

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

### 6. Validate the mechanism and justify physical implementation

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
