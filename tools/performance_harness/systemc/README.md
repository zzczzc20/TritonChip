# Native SystemC Architectural Model

This directory develops the event-driven architectural reference for TNCA.
The intended platform covers program dispatch, core execution, register and
engine contention, DMA, MMA, synchronization, NoC, NUCA, HBM, and host ownership.
Its execution path must contain no RTL simulator, Verilator model, synthesis,
or measured workload stall trace. RTL is an independent offline validation
source. The model must generate dynamic memory traffic from program semantics
before it can predict the effects of compiler or compute-architecture changes.

**Qualification status: functionally executing the compiler's output;
timing uncalibrated.** Since M1 the native executor implements every opcode
the frozen suites' compiled kernels contain (`opcode_inventory.py`), with the
RTL as the numeric authority (bit-exact against offline RTL oracles, below).
Device atomics and anything else not implemented are refused explicitly. All
timing constants are **uncalibrated hypotheses**; no cycle count produced here
is an R40 performance result, and the model is not qualified as a
complete-chip timing reference. Do not bypass refusals or substitute a memory
replay for a complete-chip experiment.

**M2 (board calibration): uncalibrated by doc/r40_tasks_and_acceptance.md
sec 3.4.** With the RTL-structured board configuration
(`configs/r40-board-m2.cfg`) fitted on the R40-04 calibration split only, the
frozen holdout gives a median absolute graph-makespan error of 21.5 % and a
p95 of 34.7 % (limits 10 % / 20 %): load/store streaming kernels are 10-36 %
too fast, transfer/matrix-engine kernels and the llama graphs are within about
6 %. Evidence and the per-family breakdown: `evidence/r40/v3/systemc_m2/`.

## Execution host and dependency identity

Build and run on the GB10 jump host. The FPGA implementation server is reserved
for its existing implementation and necessary physical/device operations.

The dependency is Accellera SystemC **3.0.2**, including TLM-2.0, pinned to
commit `70b0fc8e4a74acc677b0fc73cea08f940c2115d5`.
Use the [official installation guidance](https://github.com/accellera-official/systemc/blob/3.0.2/INSTALL.md).
The build uses C++17 and the imported `SystemC::systemc` CMake target.

```sh
bash tools/performance_harness/systemc/bootstrap.sh "$HOME/tc_env/systemc-3.0.2"
cmake -S tools/performance_harness/systemc -B /tmp/tnca-systemc-build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="$HOME/tc_env/systemc-3.0.2/install"
cmake --build /tmp/tnca-systemc-build --parallel 4
```

No Verilator dependency is required. No clock-by-clock idle loop is used.
Dependencies and resource availability schedule SystemC events directly.

## Implemented execution paths

| Component | Current implementation | Qualification limitation |
| --- | --- | --- |
| Native executable | Words decoded once (`--classify` prints the verdict per word); scalar/control, integer and FP32 vectors (incl. `.ftz`), all conversions, predicates, shuffles, reductions, SFU (bit-exact port of `xls/tnca_sfu.x`), scalar/vector/gather/scatter LSU, `fence`, `trap` as an architectural fault, special registers per `tnca_core.v` | Device atomics refused; vector-FPU/conversion/SFU/reduction lane functions checked bit-exactly against RTL-recorded vectors |
| Matrix engine | Core-wide accumulators A0..A15, `m.zero`, `m.mma.async` (FP16/BF16/E5M2/E4M3, K16/32/64, source-free event), `a.vread/a.vwrite/a.mul.row/a.fma.row/a.zero`; one command at a time; result and source-free barrier completions | Arithmetic bit-exact against `tnca_matrix.v`; busy-cycle formula uncalibrated |
| Grid execution | `tnca_grid_dispatch.v` order: groups of 1/2/4 x along x, then y, z; per-core argument stride; active-core mask; fault stops dispatch; four persistent Role Context processes per core (no SystemC process per program) | Dispatcher cycle and grouping/pair behavior are not calibrated |
| Role execution | Private scalar/predicate state and scoreboards; timestamped, lane-masked physical VRF writeback; RPAIR vector-base aliasing; RSLICE remap of accumulator, barrier, descriptor, window indices and barrier tokens from the runtime's role mask | Arbitrary vector-base mappings beyond 64*r remain unmodelled |
| Barriers | Shared per-core generations, arrival/operation counters, sleeping waits; `mb.arrive` drains own register writes and own engine commands (RPAIR precision); DMA and MMA completions; faults invalidate and wake | Timing calibration remains incomplete |
| Runtime adapter | `tlm_backend.py` registers `systemc-tlm` (per-program) and `systemc-tlm-grid` (physical grid) into the runtime's backend table from outside the product tree; one long-lived `tnca-chip --serve` process per Session | Host transfers are functionally coherent and take no modeled time |
| DMA | Immutable descriptor snapshots, rank/stride/padding, bounded asynchronous transfer resources, RF leases and barrier completion | Partial descriptor modifiers; no measured DMA timing calibration |
| Core timing | Issue resource, per-class result latencies, SFU occupancy, matrix-engine occupancy, all from configuration keys | Every constant is an uncalibrated hypothesis; RF banks/ports and fetch are not modelled |
| LSU | Byte enables, unaligned splitting, gather/scatter, word coalescing, bounded outstanding beats on persistent workers; scalar accesses use lane 0 of the mask | A role waits for its own access (no non-blocking stores); atomics refused |
| NoC | Unique-home mapping from `tnca_core_ni.v`; XY routes and directed-link service contention | Packet-level estimate; finite router buffers/VC behavior are not implemented |
| NUCA | Set-associative writeback storage, invalid-first/round-robin replacement, finite MSHRs/targets, merges, same-line ordering and hit-under-miss | DCO policies, epochs and exact engine arbitration remain incomplete |
| Memory | Shared byte-addressed storage, explicit contiguous PC mapping, independent PC transfer resources, finite outlet outstanding slots and write acknowledgements | No DRAM bank/row/refresh model; bandwidth/latency need calibration |
| Visibility | Requester drain, parallel slice writeback/invalidation, completion after write acknowledgement, external host probes | Product ownership protocol adapter is not complete |
| Profiling | Passive socket monitors, bounded aggregate latency histograms, concurrency, resource queue/service counters; optional Chrome transactions | Host overhead is workload-specific; SCV recording is not implemented |

`r40-three-core.cfg` binds the final-candidate cache geometry and three outlets.
Its timing fields are explicitly **uncalibrated hypotheses**. The PC configuration
describes a hypothetical contiguous allocation, not an observed board allocation.
An actual allocation must supply its own verified map. The TMU MMIO window is
refused as ordinary data memory.

The TLM boundary uses generic payload addresses, read/write commands, data,
byte enables and response status. A typed extension carries source, request
identity and atomic operation metadata. Blocking transport runs in concurrent
SystemC processes with actual kernel-time waits. DMI and temporal decoupling
are disabled to prevent shared-resource contention from being bypassed.

## Executing ArchBench graphs (M1)

The standard path is unchanged: torch.compile -> Inductor -> Triton -> TNCA
artifact -> `Session`, which launches into the model through a backend the
harness registers at import (`tlm_backend.register()`); nothing under
`archbench/`, `python/`, `compiler/`, `tools/asm/`, `spec/`, `rtl/` or
`tb/device/` is modified. `run_graphs.py` runs each point in a fresh child
(one Session, one model process per graph) through
`archbench.platforms.tnca.run`, which compiles, compares with the PyTorch
reference at the point's own tolerance and counts host fallbacks. Records
carry `evidence_level: model-prediction-uncalibrated`.

```sh
python3 tools/performance_harness/systemc/run_graphs.py --source-root SRC   --binary BUILD/tnca-chip --out FRESH_DIR   --suite SRC/spec/r40/suites/existing.json --tier tiny   --knobs-from PREFLIGHT/A_fix2/runs --torch-extensions-dir EXT --jobs 4
```

The default backend `systemc-tlm-grid` reproduces the delivered board
configuration (native tensors, physical grid, `TNCA_ACTIVE_CORES=1`,
`TNCA_RTL_MEM_PORTS/OUTSTANDING` mapped onto the model's outlets); the
point's compile knobs are copied from the compile preflight's `result.json`.
`--backend systemc-tlm` uses per-program launches, the sequence the
`sim-file` RTL simulator receives, and `--backend sim-file` runs the same
child on the RTL simulator for wall-time comparisons only. The native path
measures the steady call after the cold call (ArchBench `cold_setup`).
Each model process leaves `identity.json`, `launches.json` (per-launch
modeled cycles, retired instructions, host seconds), `counters.json`
(including per-core instruction classes, dependency, engine and barrier
waits) and `sockets.json` under its run directory.

`tnca-chip --serve CONFIG OUT IN_FD OUT_FD` is the host protocol: `W addr n`
+payload, `R addr n`, `L` (one program: pc, argument base, role mask, pid,
npid, deadline), `G` (one grid: pc, argument base, role mask, gx gy gz,
active cores, argument stride, group width, deadline), `C` (counters), `F`
(writeback/invalidate maintenance), `X`. A launch replies `DONE status code
pc start end retired`; an architectural fault (TRAP_USER, BARRIER_FAULT,
DESCRIPTOR_FAULT, ...) is status 1, a deadline overrun status 2, and a model
refusal or deadlock terminates the process with `ERR`, which the backend
raises as a lost device. Host reads and writes are coherent with the NUCA
homes and cost no modeled time.

## Board timing structure, calibration and judgement (M2)

`configs/r40-board-m2.cfg` is the 3-core U280 board (attempt03 image, 90 MHz)
with the RTL's structure switched on; `configs/r40-board-m2.sources.json`
gives the RTL source (or FIT) of every key. `configs/r40-three-core.cfg`
keeps the M1 behaviour exactly.

- `memory_model=1`: the analytic timeline memory
  (`include/tnca_timeline_memory.hpp`): home-slice engine states, MSHRs,
  round-robin victims, one-flit NoC packets, outlet issue stages and
  outstanding lines; `tl_*` keys.
- `core_pipeline=1`: the core's single in-order pipeline: two LSU units
  (`LSU_INFLIGHT=2`), freezes for non-switching stalls, context parking and
  refetch (`RCTX_SWITCH=2`), taken-branch refetch, banked-VRF read conflicts
  with `VRF_PREFETCH`, the writeback-port steal, RTL latencies; and the
  transfer engine's RTL sequencing (setup 2+rank, 8 load slots, store
  register reads, transposed column blocks).
- Host time (server mode only): `host_launch_cycles` before every launch,
  `host_transfer_cycles` + `host_transfer_cycles_per_kib` per host transfer.
- `time_quantum`: conservative lookahead; a context skips a wait that ends
  before any other pending activity (only same-cycle event order changes).
- Serve command `T 0|1` turns timing off/on: untimed execution has every
  architectural effect and no modeled time (used for a graph's cold call).

Runner and judges:

```sh
# every distinct (point, knobs) of a software tree's board records, with exactly those knobs
run_graphs.py --source-root SRC_TREE --binary BUILD/tnca-chip --config configs/r40-board-m2.cfg \
    --dataset board_timing_dataset.json[.gz] --split calibration --tree files:fe1b \
    --memory-bytes 8589934592 --monitor off --timed-auto --untimed-cold --out FRESH_DIR
predictions.py --results FRESH_DIR/results.json --config CFG --binary BIN --out predictions.jsonl
fit.py host --records 'RECORDS/*.jsonl' --root SRC --out host_fit.json            # calibration rows only
fit.py latency --dataset D --root SRC --params hbm_latency_cycles,tl_outlet_to_slice_cycles \
    --grid 20:5=p1.jsonl --grid 35:5=p2.jsonl ... --out latency_fit.json         # calibration rows only
fit.py report --config CFG --host host_fit.json --latency latency_fit.json \
    --sources configs/r40-board-m2.sources.json --out fit_report.json
calibrate.py evaluate --dataset D --predictions predictions.jsonl --fit-report fit_report.json --out fidelity.json
directional.py --dataset D --predictions predictions.jsonl --fidelity fidelity.json --out directional.json
```

`--timed-auto` measures a one-launch graph on the cold call's graph launch
(no steady call) and a multi-launch graph on one steady call;
`--untimed-cold` runs that cold call untimed. `--env KEY=VALUE` sets a
runtime knob for every run (e.g. `TNCA_RTL_MEM_PORTS=3`).

## DRAM paths: fast and slow (M3)

The memory model asks one interface for every line that reaches the HBM side
(`include/tnca_dram.hpp`: `DramPath::complete(DramRequest{port, address,
write, arrival})` returns the completion cycle):

- **fast** (`dram_path=0`, the default): the M2 latency model,
  `hbm_latency_cycles` (+ the second beat) for a read and
  `tl_hbm_write_latency_cycles` for a write. Bit-identical to M2.
- **slow** (`dram_path=1`, `configs/r40-board-m3.cfg`): Ramulator2 HBM2, one
  FR-FCFS open-page controller with refresh per pseudo channel
  (`ramulator/hbm2_u280_pc.py` lists every assumption). Each timed command
  (launch or flush) is solved as a fixed point: a forked copy-on-write child
  runs the command reading per-request latencies from a table and logs its
  DRAM requests; the parent replays them through Ramulator2
  (`ramulator/tnca_dram_bridge.cpp`) and rebuilds the table
  (`dram_fixed_cycles` + the Ramulator2 latency); after at most
  `dram_max_passes` children, or once two children agree within
  `dram_tolerance_ppm`, the command runs for real. Untimed commands (`T 0`)
  run once. Per command: `<model dir>/dram_fixed_point.jsonl` (passes,
  residual, requests, row hits/misses/conflicts, refreshes, time); totals in
  the `dram_*` counters; path, Ramulator2 commit and config sha256 in
  `summary.json`.

Switch with the config key or the environment (the environment wins):

```sh
TNCA_DRAM_PATH=slow tnca-chip --serve configs/r40-board-m2.cfg ...      # slow path on any config
run_graphs.py ... --config configs/r40-board-m3.cfg                       # slow (dram_path=1)
run_graphs.py ... --config configs/r40-board-m3.cfg --dram-path fast      # the M2 fast path
```

The slow path needs a binary configured with the pinned Ramulator2 checkout:

```sh
git clone https://github.com/CMU-SAFARI/ramulator2 && git -C ramulator2 checkout $(cat ramulator/PIN)
cmake -S ramulator2 -B ramulator2/build_lib -DCMAKE_BUILD_TYPE=Release -DRAMULATOR_PYTHON_BINDINGS=OFF
cmake --build ramulator2/build_lib --parallel 6      # libramulator.so in the checkout root
cmake -S . -B build -DCMAKE_PREFIX_PATH=SYSTEMC -DRAMULATOR2_DIR=$PWD/ramulator2 && cmake --build build
build/tnca-dram-bridge-test ramulator/hbm2_u280_pc.yaml   # bridge unit tests
```

Without `RAMULATOR2_DIR` the binary has only the fast path and refuses
`dram_path=1`. Configure refuses a checkout that is not the pinned commit or
has local changes. `ramulator/hbm2_u280_pc.yaml` (committed) is exported
from `hbm2_u280_pc.py` by `python3 -m ramulator export`, which needs a build
with Python bindings; that build runs Ramulator2's code generator, which
rewrites (reformats) its generated DRAM sources, so do it in a separate
checkout and build the library above from the pristine pin.

## Numeric authority: offline RTL oracles

The RTL is the semantic authority, used offline only:
`rtl_oracle/rtl_oracle.py` records `tnca_vector_fpu` outputs (vf.*, all
conversions, FP reductions, SFU) and `rtl_oracle/matrix_oracle.py` records
`tnca_matrix` accumulators (MMA in every dtype/K, row operations) with
iverilog; `tnca-numeric-probe vfpu|matrix` then checks the executor's own
lane functions (`include/tnca_vector_ops.hpp`, `include/tnca_matrix_ops.hpp`)
word by word. Following the RTL exposes four places where
`tnca_vector_fpu.v`'s float-to-integer conversion differs from ISA sec 8.4
(8-bit shift wrap for |x| in [2^24, 2^31), RNE rounding of odd values in
[2^23, 2^24), unsigned saturation from 2^31, all-ones for negative fractions
to unsigned); `validate_numeric.py` records each divergence. No such
conversion occurs in the frozen suites.

`opcode_inventory.py` scans assembled kernel images, identifies opcodes with
the canonical ISA tables and asks `tnca-chip --classify` for the executor's
verdict and timing class per word, beside the pre-M1 acceptance logic.

## Performance diagnosis and recording policy

The primary objective is rapid bottleneck diagnosis and architecture screening
before an expensive implementation run. Functional execution generates demand;
coverage alone is not a performance qualification metric. The event-driven path
must demonstrate its speed and prediction error on matched workloads before
claiming an orders-of-magnitude advantage over RTL.

`run.py --monitor off|aggregate|trace` selects the recording policy and binds it
to experiment identity. The default is `aggregate`: no per-transaction trace
serialization. Passive `SocketMonitor` modules interpose standard TLM-2.0
sockets at each NoC-to-home and cache-to-HBM boundary. They forward the original
payload and delay without waits, additional events or payload extensions.
They use `tlm_utils` sockets, not a vendor Probe/Monitor Socket package.
**SCV Transaction Recording is not currently implemented.**

Each boundary reports transaction/byte counts, read/write counts, error counts,
peak outstanding calls, mean/max residence, a fixed 65-bin latency histogram,
and the union of intervals with active calls. Bin 0 contains zero cycles;
bin k contains [2^(k-1), 2^k-1]. Aggregate storage is bounded by socket count,
not transaction count. `trace` adds bounded Chrome events with socket-local
transaction sequence, address, length, source and home. `trace_events` limits
all emitted events; dropped events are reported. `off` retains mandatory
functional checks and existing architectural counters but disables socket
statistics and detailed event recording. It is not a no-instrumentation RTL
baseline.

Socket residence includes downstream waits. Home and HBM intervals overlap;
do not add their latency or busy fractions. Busy fraction describes an active
transaction interval, not physical port utilization. Queue/wait cycle sums
across concurrent requests can exceed elapsed cycles. Resource service counters
provide supporting demand evidence, not a critical-path attribution.

```sh
python3 tools/performance_harness/systemc/validate_monitor.py \
  --binary /tmp/tnca-systemc-build/tnca-tlm --out /fresh/path/monitor-study
python3 tools/performance_harness/diagnose_systemc.py \
  --report /fresh/path/monitor-study/run-0-aggregate/report.json \
  --out /fresh/path/diagnosis.json
```

The monitor study uses alternating off/aggregate/trace order with warmup and
reports the observed host-overhead range. It requires identical simulated
results, counters and byte-for-byte responses. Its memory read stream does not
establish native-chip speedup or a universal overhead bound. Host time includes
process startup and output, excludes compilation and identity hashing, and
must be interpreted against that scope. Prediction error and RTL speedup remain
null until measured independently.

`diagnose_systemc.py` verifies sealed outputs before interpreting counters.
It exposes socket residence and resource pressure, then supports `--candidate`
for a single-configuration sensitivity comparison with unchanged software and
workload. Change one capacity or mapping, preserve useful work, and check that
the expected queue and traffic changes accompany the predicted cycle change.
A lower modeled cycle count alone does not authorize synthesis. Calibrate on
independent RTL/board workloads, retain a frozen holdout and reject a ranking
whose improvement is smaller than its established prediction uncertainty.

## Run and preserve an experiment

```sh
python3 tools/performance_harness/systemc/run.py \
  --binary /tmp/tnca-systemc-build/tnca-tlm \
  --trace /absolute/path/requests.trace --out /fresh/path/memory-run
python3 tools/performance_harness/systemc/run.py --mode chip \
  --binary /tmp/tnca-systemc-build/tnca-chip \
  --trace /absolute/path/chip.commands --out /fresh/path/chip-run
```

Outputs include the executable and loaded SystemC-library hashes, configuration,
model-source hashes, input hashes, source identity when supplied, retained
stdout/stderr, functional status, resource counters and `trace.json`.
Chip scripts also bind every referenced code/data/reference file. Configuration
and input identities are rechecked after execution. Outputs require a new
directory; failures and partial files remain available for diagnosis.
`clock_hz` defines modeled reference time. It is never a measured device clock.

Native scripts support `code ABSOLUTE_HEX_FILE`,
`data ADDRESS ABSOLUTE_BINARY_FILE`, `launch ID PC ARG_BASE GX GY GZ`, `fence`,
and `expect ADDRESS ABSOLUTE_REFERENCE_FILE`. `launch_roles ID PC ARG_BASE GX GY GZ
ROLE_MASK PAIR_ALIAS` selects up to four active roles. The default maps role r
to physical vector base `64*r`; `PAIR_ALIAS=1` maps each odd role to its even
partner's vector base. This is explicit vector aliasing. `launch_mask ID PC ARG GX GY GZ MASK`
passes the runtime's full 32-bit role mask (contexts, RPAIR bit 28, RSLICE
bit 31 with log2 slices in bits 30:29). Direct data access requires initial or explicitly drained
host ownership.
`code` places the image at address 0 of model memory, where the runtime's
code allocator also starts; instructions are fetched from memory and decoded
once per address. Unknown commands and trailing launch fields are refused;
an illegal encoding is an ILLEGAL_INSTRUCTION fault and an unimplemented one
a model refusal. Opcode identifiers and predicate legality come from the
canonical ISA schema.

Memory replay rows contain eleven whitespace-separated fields:

```text
id release_cycle core op address value byte_enables compare check expected kind
```

Integers use decimal or a `0x` prefix. IDs strictly increase; release cycles
never decrease. Ordinary read/write operations are `0/1`; atomic operations are
`2 + TNCA_AOP`. `kind` is `T` for a transaction, `I` for initial memory, `F` for
global maintenance and drain, or `H` for an external-memory visibility probe.
Initialization precedes transactions. Atomic masks must select one aligned
32-bit half or a 64-bit word. All addresses and configuration keys are checked.
Ordinary NUCA reads return an entire aligned word; their trace byte mask is
ignored, matching the RTL boundary. The LSU extracts the requested subword.
TLM read payloads therefore enable all eight returned bytes; stores and atomics
retain their actual byte enables.

**Replay is conditional on its captured address and arrival sequence.** It can
test memory mechanisms and hypotheses preserving that sequence. It cannot
predict changed control flow, register dependencies, compiler transformations,
or full-chip makespan. Source backpressure can delay release times; that slip
is reported rather than silently treating all arrivals as exact.

## Validation and uncertainty

```sh
python3 tools/performance_harness/systemc/validate.py \
  --binary /tmp/tnca-systemc-build/tnca-tlm --out /fresh/path/memory-validation
python3 tools/performance_harness/systemc/validate_chip.py \
  --binary /tmp/tnca-systemc-build/tnca-chip --out /fresh/path/chip-validation
python3 tools/performance_harness/systemc/validate_numeric.py \
  --binary /tmp/tnca-systemc-build/tnca-numeric-probe --out /fresh/path/numeric-validation
python3 tools/performance_harness/systemc/validate_roles.py \
  --binary /tmp/tnca-systemc-build/tnca-chip --out /fresh/path/role-validation
python3 tools/performance_harness/systemc/validate_matrix.py \
  --binary /tmp/tnca-systemc-build/tnca-chip --out /fresh/path/matrix-validation
```

`validate_matrix.py` runs MMA in all dtype/K combinations with result and
source-free barriers, accumulator row operations under a lane mask, an RSLICE
launch whose unsliced negative control must fail, `trap` as a reported fault
and an illegal encoding, against exact Fraction references.

Memory validation checks partial writes, every implemented home-atomic ALU
operation at both widths, dirty thrashing, finite-resource stalls, one/all
outlet contention, multiple PC resources, maintenance visibility, negative
mutations, address refusals and trace perturbation. Independent randomized RTL
captures are checked against a separate byte-memory oracle before TLM replay.
The capture monitor is simulation-only and changes no product RTL.

Native validation consumes the actual assembler and compiler outputs. It checks
two three-dimensional grids sharing cache lines across physical cores, a
second grid reading the first grid's results, masked compiler-generated integer
and FP32 vector addition, explicit halfword-memory conversion, masked reductions,
aliasing shuffles, predicate selection, SFU specials, untouched tail sentinels
and unsupported-operation refusal. These are development diagnostics, not the original native77/HBM14 or
the frozen model/holdout acceptance runs.

Role validation checks private register/predicate banks, explicit pair aliasing,
future-generation sleeping, release visibility, a generation beyond 65535,
initialization under a waiter, invalidation, count underflow and deadlock refusal.
The retained campaign additionally compiles a separate source clone with only
the arrive writeback wait removed; the consumer then reads the old physical
value and numerical checking fails. Global shared readiness must never make
that mutation appear correct. See the [role evidence](../../../evidence/r40/R40-09/systemc_native/roles_v12/README.md).

`analyze_replay.py` reports signed and absolute per-request completion errors
against retained RTL responses. Fixed-arrival replay error is not a production
prediction interval. Future calibration must keep fitting, independent
validation and frozen holdout separate, and bind the complete model/configuration
and source identities. Never fit on frozen holdout.

Tracing adds host allocation and serialization cost. Validation alternates three
traced/untraced pairs, compares complete response files and simulated counters,
and reports the observed host overhead range. No physical instrumentation claim
follows from those results. A measured zero simulated-cycle difference does not
mean zero host overhead or perfect model accuracy.

## Complete-chip qualification still required

The remaining work includes timing calibration against board counters (every
latency key in the configuration), RF bank/read/write port resources,
instruction fetch, non-blocking stores, device atomics, DCO/ownership epochs
and host-transfer cost, bounded router queues, and independent end-to-end
timing validation on matched workloads.
Unsupported cases remain refused until their semantics and timing assumptions
have direct validation. Production timing error and complete-chip performance
are **not-measured**. The frozen R40 v2 contract is unchanged.
