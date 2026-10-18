# Native SystemC Architectural Model

This directory develops the event-driven architectural reference for TNCA.
The intended platform covers program dispatch, core execution, register and
engine contention, DMA, MMA, synchronization, NoC, NUCA, HBM, and host ownership.
Its execution path must contain no RTL simulator, Verilator model, synthesis,
or measured workload stall trace. RTL is an independent offline validation
source. The model must generate dynamic memory traffic from program semantics
before it can predict the effects of compiler or compute-architecture changes.

**Qualification status: incomplete.** The current native machine-word executor
supports only part of the ISA. The memory subsystem and the initial native
execution path require the validation described below. Neither executable is
qualified as a complete-chip reference or as an R40 performance result.
Unsupported instructions fail explicitly. Do not bypass those failures or
substitute a memory replay for a complete-chip experiment.

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
| Native executable | Canonical assembled 64-bit TNCA words; scalar/control, integer vectors, predicate operations, selected FP32/CVT/SFU, shuffles, reductions and scalar/vector LSU | Partial ISA; no claim of complete compiler coverage |
| Grid execution | Bounded concurrent physical-core workers, dynamic program assignment, per-core telemetry | Dispatcher cycle and grouping/pair behavior are not calibrated |
| Role execution | Private scalar/predicate state and scoreboards; timestamped, lane-masked physical VRF writeback; disjoint role banks or adjacent-pair aliasing | RSLICE, grouped programs, arbitrary vector-base mappings and complete role-mask coverage remain incomplete |
| Barriers | Shared per-core generations, arrival/operation counters, sleeping waits, release writeback drain and explicit faults | Native DMA/MMA completion hooks are not yet integrated; timing is uncalibrated |
| Core timing | Issue resource and scalar/vector result availability events | Timing constants are hypotheses; RF banks and all execution engines remain incomplete |
| LSU | Byte enables, unaligned splitting, gather/scatter, word coalescing, bounded outstanding beats | Additional ordering and atomics integration remain required |
| NoC | Unique-home mapping from `tnca_core_ni.v`; XY routes and directed-link service contention | Packet-level estimate; finite router buffers/VC behavior are not implemented |
| NUCA | Set-associative writeback storage, invalid-first/round-robin replacement, finite MSHRs/targets, merges, same-line ordering and hit-under-miss | DCO policies, epochs and exact engine arbitration remain incomplete |
| Memory | Shared byte-addressed storage, explicit contiguous PC mapping, independent PC transfer resources, finite outlet outstanding slots and write acknowledgements | No DRAM bank/row/refresh model; bandwidth/latency need calibration |
| Visibility | Requester drain, parallel slice writeback/invalidation, completion after write acknowledgement, external host probes | Product ownership protocol adapter is not complete |
| Profiling | Bounded Chrome trace, per-resource counters, deterministic simulated time with tracing enabled/disabled | Host tracing cost must be measured for each workload class |

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
partner's vector base. This is explicit vector aliasing, not RSLICE or grouped
program support. Direct data access requires initial or explicitly drained
host ownership.
Unknown commands, trailing launch fields and undefined opcodes are refused.
Opcode identifiers and predicate legality come from the canonical ISA schema.

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
```

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

The remaining coverage includes complete role-mask/RSLICE/grouping behavior,
RF bank/read/write resources, descriptors/windows, asynchronous DMA and MMA,
barrier generations and completion obligations, remaining numerical modifiers
and complete instruction integration coverage, DCO/ownership epochs, bounded router queues,
the standard Torch/runtime adapter, and independent end-to-end timing validation.
Unsupported cases remain refused until their semantics and timing assumptions
have direct validation. Production timing error and complete-chip performance
are **not-measured**. The frozen R40 v2 contract is unchanged.
