# Handing the hardware to a platform branch

This is what a downstream branch receives and what it is responsible
for. It is the deliverable of R36-09, and every claim in it is checked
by `tools/r36/handoff_tests.py`.

**The claim level is `engineering-ready`.** Nothing here has been
placed, routed, timed, or run on a device. Section 1 of the phase
contract puts device selection, vendor IP, placement and routing,
bitstreams, programming and board bring-up in a separate branch, and
this document does not pretend otherwise. `platform-verified` is a
reserved state the registry refuses to accept.

---

## 1. What you get

| thing | how to build it | what it is |
|---|---|---|
| the hardware export | `make r36-export` | `tnca_shell.v` and everything it instantiates, with a manifest, an elaboration script and a self-checking smoke test |
| the installable wheel | `make r2-package` | the runtime, and the compiler if you install Triton |
| the conformance suite | `tools/r36/conformance.py` | one suite, five groups, run against any backend |
| the rehearsal | `tools/r36/downstream/` | a worked example of all of the above |

Each carries its own hashes: `MANIFEST.json` in the export, and the
wheel's own record. `make r36-build --json` writes a build report with
both.

## 2. The interface you extend at

```python
from tnca import backend

class MyBoard(backend.Backend):
    name = 'my-board'
    state = backend.REFERENCE_AVAILABLE
    # start, stop, alive, upload, download, launch, quiesce, ping

backend.register('my-board', MyBoard,
                 state=backend.REFERENCE_AVAILABLE,
                 what='what this binding is')
```

Eight methods. That is the whole surface, and it is deliberately not
more: section 3.1 separates the compile target, the execution backend,
the hardware implementation and the platform binding, and a backend is
only the second. It does not compile, it does not own the buffer or
event state machines, and it never edits a kernel.

**`platform-verified` is reserved.** `register` raises if you pass it.
Nothing may claim a real platform ran it until a real platform has.

### The hardware side

`tnca_shell` takes a depth-one command port, returns one completion
record per command, and offers a single-word host access port that is
only ready while no kernel is running. `sim/tnca_device.v` in the export
is a worked example of driving all three; it is about two hundred lines
and contains no design logic.

## 3. What is yours and what is ours

### FPGA

| responsibility | whose | note |
|---|---|---|
| device and family selection | yours | not specified here, deliberately |
| clock generation and reset synchronisation | yours | the export has no reset synchroniser; see §5 |
| memory: the data and instruction port | yours | the contract is in `spec/r36_memories.json` under `DEVICE-MEMORY` |
| byte enables on that memory | yours | the ISA has sub-word stores. A memory without byte enables needs a read-modify-write in your adapter |
| the host transport | yours | a doorbell and a mailbox, or whatever the board gives you |
| pin and board constraints, bitstream, programming | yours | |
| timing closure | yours | no frequency is claimed here |
| the public top's behaviour | ours | `rtl/tnca_shell.v` |
| the ISA and the compiler | ours | |
| the conformance suite and its oracles | ours | you run it; you do not write a second one |

### ASIC

| responsibility | whose | note |
|---|---|---|
| process, library, corners | yours | |
| memory compilers and macro selection | yours | `spec/r36_memories.json` lists every array with its ports, latency, mask and read-during-write behaviour |
| the VRF bank as a macro | yours | `tnca_rf_bank` is the array to map: 1R/1W, 32 x 512 bits |
| DFT, scan, BIST | yours | nothing in the export is scan-aware |
| power intent, clock tree | yours | |
| floorplan, place and route, signoff | yours | |
| the RTL and its contracts | ours | |

### What is NOT a board task

These are **our** gaps, and they stay ours. Writing them into the tables
above would be disguising a missing semantic as somebody else's
integration work, which R36-HAND-02 forbids.

- **No memory technology mapping has been done.** Every array in
  `spec/r36_memories.json` reads `technology_mapping: not started`. An
  array yosys infers as memory is not a verified physical RAM.
- **The ARF is behavioural.** `CAP-ARF-SYNCHRONOUS` is recorded
  unimplemented in `spec/r36_capabilities.json`.
- **No cache coherence.** Host and device access is serial and the
  runtime enforces it.
- **The four-tile system runtime is incomplete.** R2-09, R2-10 and R2-14
  have open halves. The handoff baseline does not depend on any of it.
- **`tl.dot` does not reach the matrix unit.** Triton's own frontend
  refuses it for this target; the route to the matrix unit today is the
  compiler's GEMM planner.

## 4. Running the minimal correctness program

```sh
# build the export
make r36-export

# elaborate it, alone, with no library search of ours
cd sim/build/tnca-export && ./elaborate.sh

# and run a program on it
./run_smoke.sh          # prints SMOKE PASS
```

Then, from your own tree, with the wheel installed:

```sh
python run_rehearsal.py --package /path/to/tnca-export
```

`tools/r36/downstream/run_rehearsal.py` registers a backend, builds a
device from the package, runs the minimal program, checks it against an
oracle, and runs the conformance suite. Copy it and change
`platform_backend.py`.

## 5. Clocks, reset, and what the export does not contain

From `spec/r36_memories.json`, `cross_domain`:

- **One clock**, `clk`, for the whole public top. No domain crossing
  exists inside it.
- **One asynchronous-assert reset**, `rst_n`, active low.
- **Reset release is yours.** The export contains no reset synchroniser.
  Release `rst_n` synchronously to `clk`.
- **The memory port is synchronous to `clk`.** A memory in another
  domain needs a synchroniser the export does not provide.
- There are no other unstated board assumptions. If you find one, it is
  a gap in this list and belongs on the mainline.

## 6. Diagnosing a failure

| symptom | where to look |
|---|---|
| a launch returns status 1 | a fault. `done_fault_code` and `done_fault_pc`; the pc is an address in the artifact's disassembly (`TNCA_KERNEL_DUMP=1`) |
| a launch returns status 2 | the device deadline expired with work outstanding. The core is NOT reset by this; the shell stays busy-clean so you can tell "still running" from "went wrong" |
| a launch returns status 3 | the request was refused before any work started. Check alignment and range |
| `host_req_ready` never rises | a kernel is running, or `kernel_live` was never cleared. Send a QUIESCE |
| values wrong, cycles plausible | the memory. Run `make r36-mem-tests`: two independent implementations at seven configurations each |
| values right, cycles identical at every latency | your memory is ignoring the latency it was given. `MUT-MEM-LATENCY` exists because that is invisible to a correctness check |
| the export will not elaborate | `./elaborate.sh` names every file and passes no `-y`, so the error names the missing module |

### The counters

`tnca_shell` exposes the core's performance counters directly:
`perf_cycles`, `perf_issued`, `perf_stall_hazard`, `perf_stall_lsu`,
`perf_stall_wb`, `perf_squash`. The completion record carries
`done_cycles` (the cycles the kernel was observed running) and
`done_retired` (instructions retired). `done_cycles` counts the run, not
the shell's own state machine.

## 7. Which way a fix goes

| what you found | where it is fixed |
|---|---|
| the ISA, the compiler, the runtime state machine, the public top's behaviour | **the mainline.** Open an issue; do not patch your copy |
| a semantic the interface does not express | **the mainline.** The rehearsal found three of these and all three were fixed there, not worked around |
| your board's clocking, memory, transport, pinout | **your extension layer** |
| a parameter of the public top | **your instantiation.** `USE_BANKED_VRF`, `NUM_RCTX` and the rest are parameters for this reason |

A platform branch that patches `rtl/core/`, `python/tnca/runtime.py` or
the compiler has forked, not extended, and the conformance suite it runs
afterwards is testing something else. `tools/r36/handoff_tests.py`
checks the rehearsal against exactly this rule.

## 8. Versioning the interface

| interface | where its version lives | what a change means |
|---|---|---|
| the backend ABC | `tnca.backend.Backend` | adding a method is breaking for every binding. It has happened once: `ping` |
| the transport protocol | `spec/r2_transport_v1.json` | the file name carries the version |
| the relay protocol | `PROTOCOL_VERSION` in `socket_relay.py` | the `hello` exchange carries it |
| the export package | `format` in `MANIFEST.json` | `tnca-hardware-export/1` |
| the compile target | `config_hash` in the artifact | an artifact for another configuration is refused at load, before any device request |

A breaking change bumps the version in the file that owns it and the
conformance suite gains a case for the new behaviour. A binding that
does not implement it fails that case rather than being quietly
tolerated.

## 9. Who owns what

| area | owner | how a change is proposed |
|---|---|---|
| the ISA and `isa/` | the architecture owner | a spec change with a conformance case |
| `rtl/tnca_shell.v` and the public ports | the integration owner | a change here needs the mutation suite re-run: it is what `MUT-EARLY-DONE` watches |
| `tnca.backend` and the registry | the runtime owner | a new method is a version bump, §8 |
| the conformance suite | the verification owner | a case may be added; one may not be removed to make a run pass |
| a platform binding | the platform branch | no mainline review needed, and no mainline file touched |

This repository is one project and the "owners" above are roles rather
than people. The point of naming them is that a change to a public
interface has a named place to be argued about, not that there is a
committee.
