# Core alignment: the trace gate between the model and the RTL

This directory holds the **alignment gate** of the performance model: the
evidence that the model counts the cycles the RTL counts, taken cycle by cycle
on one core, and the tools that produce it. It is the validation path of the
model. It is not a prediction path: nothing here feeds a measured RTL trace
into a prediction (the model README's rule stands), and a replayed memory
interface is used only to decide whether the core agrees with the RTL.

## 1. What is compared

The **observable set** is deliberately coarse (decided 2026-10-26): what a
model of a different microstructure can still be held to.

| Stream | One record per | Fields compared |
|---|---|---|
| launch `L` … `E` | program launch on the core | length in cycles (`E` minus `L`) |
| issue `I` | instruction leaving the execute stage | cycle since launch, Role Context, pc |
| request `Q` | memory request accepted at the core's port | cycle since launch, we, address, byte enables (data with `--data`) |
| response `P` | memory response at the core's port | latency since its request (data with `--data`) |

Not compared: tags (an allocation detail of each implementation), the
instruction port, anything inside the core. Pipeline stage occupancy,
scoreboards and register ports are the model's business; a divergence shows
up as a shifted issue cycle, and the diff tool names the first one.

Cycles are counted at the traced core's clock, from the first edge with reset
released; every comparison is relative to the launch's `L` edge, so a launch
that starts later in one trace (host time, dispatch) does not shift the next.

## 2. Trace format (`TNCA_CORE_TRACE 1`)

Text, one record per line, in cycle order. Numbers are decimal unless marked
hex (fixed-width, lower case, no prefix).

```
TNCA_CORE_TRACE 1 core=<n>
L <cycle> <pc:016x> <role_mask>            launch accepted by the core
I <cycle> <rctx> <pc:016x>                  instruction retires its execute stage
Q <cycle> <we> <addr:016x> <be:02x> <tag> <amo> <wdata:016x>   request accepted
P <cycle> <tag> <rdata:016x>                response delivered
E <cycle>                                   core idle again: the launch is complete
END <launches> <cycles>
```

An `I` record is written on every cycle the execute stage holds a valid
instruction (`x_valid`); an `mb.wait` that sleeps and re-executes appears once
per execution, as the RTL's own `instret` counts it. A `Q` is written on a
cycle with `mem_req_valid && mem_req_ready`; the response with that tag is
the next `P` carrying it. The RTL writes both from the pre-edge values, which
are the values its own clocked logic acts on at that edge.

The RTL side is `tnca_core_trace.v`, a second simulation top compiled next to
`tb/device/tnca_multicore_device.v`; it reaches every core (`TNCA_TRACE_CORES`,
a compile-time define) by hierarchical reference and nothing in `rtl/`
changes. Core 0 writes `+core_trace=<file>`, core c>0 `<file>.core<c>`; the
recorders keep them as `<point>.trace` and `<point>.core<c>.trace`. The model
writes the same format from its own clock (see the model README for the
switch; `TNCA_MODEL_CORE_TRACE_CORE=all` for every core).

### 2b. The uncore trace (`TNCA_UNCORE_TRACE 1`, second slice)

With `+uncore_trace=<file>` the monitor also writes one trace of the path
between the cores' ports and the device's memory port (`<point>.uncore.trace`):

```
TNCA_UNCORE_TRACE 1 cores=<n>
L <cycle> <core> <pc:016x> <role_mask>      a core's launch (the records that follow are timed from it)
S <cycle> <home> <src> <tag> <we> <addr>    the home slice accepted a request (tnca_tile s_req_*)
R <cycle> <home> <src> <tag>                the slice's response handshake (s_rsp_*)
M <cycle> <home> <we> <addr> <mtag>         the tile's memory request accepted (slice -> outlet, a line)
D <cycle> <home> <mtag>                     the tile's memory response
X <cycle> <we> <addr> <src> <tag>           the device's memory port accepted a request (the DRAM side)
Y <cycle> <src> <tag>                       the device's memory response
END <cycles>
```

The gate pairs launches in order (any core's `L`) and compares each stream
record by record, cycles relative to the launch. In the validation path the
model replays the `X`/`Y` boundary (the DRAM model's latencies, which the
bench randomises); in the prediction path `Y` comes from the model's own
memory timing and is not compared.

## 3. Tools

| Tool | Does | Runs where |
|---|---|---|
| `tnca_core_trace.v` | the RTL monitor: every core's trace and the uncore trace | with the device bench (`sim-multicore`) or the board-top bench (`fpga-multicore-sim`, `-DTNCA_TRACE_BOARD`), under Icarus |
| `record_rtl.py` | builds the chosen bench with the monitor, runs ArchBench points on it with `--active-cores` cores, keeps every core's trace, the uncore trace and the run records per point | a host with Icarus and the Torch/Triton environment |
| `record_model.py` | runs the same points on the model (`run_graphs.py`), predicting or replaying the RTL recording at the core or the uncore level | a host with the model build and the Torch/Triton environment |
| `tracediff.py` | parses two traces (or two directories of them), pairs launches in order and reports the first divergence of every stream, per launch and per point | anywhere |
| `tracediff.py gate` | the verdict over a point set: `alignment.json` | anywhere; the harness adapter calls this |

### 3.1 Recording the RTL

```sh
python3 tools/performance_harness/align/record_rtl.py \
    --python /path/to/torch-env/bin/python --out RUN_DIR \
    --suite spec/r40/suites/existing.json --tier tiny --jobs 8
```

One ArchBench child per point, each with its own bench process, so one
`+core_trace=` file per point. The child is the standard path (`archbench.run
--platform tnca --backend sim-multicore`), with `TNCA_ACTIVE_CORES` from
`--active-cores` (default 1; 3 runs the grids over all three cores as the
board does), the monitor compiled into a private copy of the bench
(`RUN_DIR/bench/`), and `TNCA_DEVICE_PLUSARGS` carrying the trace paths
(`+core_trace=`, one `+core_trace<c>=` per further core, `+uncore_trace=`).
The run records (`RUN_DIR/records.jsonl`) keep every point's result and
provenance as any ArchBench run does; `RUN_DIR/traces/<point>.trace`,
`<point>.core<c>.trace` and `<point>.uncore.trace` are the traces;
`RUN_DIR/manifest.json` names the bench, its compiler and the source tree.

`--bench board` records the board-top stand-in instead: the bench the
`fpga-multicore-sim` backend elaborates (`tb/fpga/tnca_multicore_board_regs.v`,
the U280 image's link and pipeline stages, three AXI outlets through
`tnca_line_axi`, the bench's AXI line model behind them), built here with the
monitor as a second top and handed to the backend prebuilt (`harness=`). Its
uncore trace's `X`/`Y` records are every outlet's adapter request and
response (the adapter boundary), not the DRAM model's port.

### 3.1b Recording the model

```sh
python3 tools/performance_harness/align/record_model.py --binary BUILD/tnca-chip --out RUN_MODEL \
    --config tools/performance_harness/systemc/configs/r40-board-cycle.cfg \
    --suite spec/r40/suites/existing.json --tier tiny --jobs 6 [--replay RUN_RTL]
```

`record_model_container.sh` runs it inside the toolchain image with the host's
model build mounted; `TNCA_REPLAY_DIR=RUN_RTL` turns on the replay. Without
`--replay` the model predicts (its own memory system); with it every point's
model process replays the RTL recording at the chosen `--replay-level`:

- `core` (first slice): the core's port replays the recorded LSU beats
  (`TNCA_MODEL_CORE_REPLAY`, CYCLE_CORE.md); the uncore is not exercised.
- `uncore` (second slice, needs `uncore_model=1`): the device's memory port
  replays the recorded `X`/`Y` stream (`TNCA_MODEL_UNCORE_REPLAY`); the
  cores, the NoC, the slices and the arbiter run at the cycle level and their
  traces are compared whole.

`--active-cores` must match the recording. Configurations:
`r40-board-cycle.cfg` is the M2 board structure with `core_model=1`;
`r40-sim-cycle.cfg` is the simulation bench's structure (4 sets x 2 ways,
PORTS=1, no link stages) with `core_model=1` and `uncore_model=1`, the one to
replay the device-bench recordings against.

### 3.2 Comparing

```sh
python3 tools/performance_harness/align/tracediff.py RTL.trace MODEL.trace
python3 tools/performance_harness/align/tracediff.py gate --a RUN_RTL/traces --b RUN_MODEL/traces --out alignment.json
```

Options: `--data` also compares write data and read data; `--order-only`
compares the sequences without cycles (the first question when a model is
new: does it even execute the same instructions and requests?);
`--max-report N` limits the divergences listed per point.

## 4. The gate contract (for the agent harness)

`tracediff.py gate` is the interface the agent harness binds to (PR #9's
services take a candidate tree, run a measurement, and return aggregates).
Its inputs are two directories of traces named by point; its output is
`alignment.json`:

```
schema        tnca-core-alignment/1
verdict       aligned | diverged | no-verdict
reason        one line
points        { <point>: { launches, aligned_launches, streams: {issue, request, response, length}: equal|<first divergence>, cycles_a, cycles_b } }
aggregates    { points, aligned, diverged, missing, median_abs_cycle_error, p95_abs_cycle_error }
options       { data, order_only }
```

| Verdict | When | Exit code |
|---|---|---|
| `aligned` | every point present on both sides, every launch equal in every compared stream | 0 |
| `diverged` | at least one launch differs | 1 |
| `no-verdict` | a point's trace is missing or malformed on either side | 2 |

The harness adapter (to be added on the harness branch) produces the two
directories from a candidate tree: the RTL side with `record_rtl.py` on the
evaluator's bench, the model side from the model's own run, both on the
validation points, and hands the agent the `aggregates` only. The `points`
map names points and is for the evaluator's record, as `certification.json`
is. Point names never reach the agent through this file.

## 5. Scope (decided 2026-10-26 and 2026-10-27)

**First slice:** one core. The model's core runs execute/commit cycle by
cycle; the memory interface is replayed from the RTL's trace in the
validation path. The base is the PR #10 merge, `3e1a282` (three cores,
NUCA, HBM, the U280 at 90 MHz).

**Second slice:** the uncore, switchable (`uncore_model=1`, on in the
evaluation configurations because the agent's candidates are judged with
the uncore's effect): the grid dispatcher, the core and slice network
interfaces, the mesh, the NUCA slices, the memory arbiter and the DRAM
model at the cycle level (UNCORE_TIMING.md), validated by replaying the
device port. The board configuration (link stages, three outlets, the AXI
adapters) is the same model with the board's parameters, validated against
the board-top bench's recording at the adapter boundary
(UNCORE_TIMING_BOARD.md).
