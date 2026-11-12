---
name: cycle-model-alignment
description: How to keep the TNCA cycle-aligned model (CA model, C++/SystemC) cycle-equal to the RTL and how to read a divergence. Use it when you change the RTL and want the model to keep predicting it, when you change the model's timing, or to find out why a model trace and an RTL trace differ. Its script records one test case of your own on both sides and compares them with the trace gate.
---

# Cycle-model alignment

The CA model (`tools/performance_harness/systemc/`, configuration
`configs/r41-board-cycle.cfg` for the R41 card's machine) is a cycle-level
description of the TNCA core and uncore, written to be held to the RTL cycle
by cycle. It runs a program in seconds where the RTL simulation takes minutes
to hours. It is judged by a **trace gate**: the RTL and the model each write a
trace of the same program, and the gate says whether every launch is equal in
every compared stream. "Equal" means exactly equal; there is no tolerance.
(The same binary also holds the Event Model, `configs/r41-sim.cfg`: a timeline
model with an error card, not held to the gate.)

What is known of it on this line: with the R41 machine's 32 home-cache sets,
replay was exact at both the core and the uncore level, on 30 one-core and 12
three-core cases. Nothing has checked it since against another machine. Nothing in the evaluation reads the model: it is an instrument, and
the gate is how far it can be trusted for a machine.

Read `tools/performance_harness/align/README.md` first: it defines the two
trace formats, the gate's verdicts and which files are the platform's (the
instruments, which you do not change) and which are the device's (the design,
which you do).

## What is compared

The observable set is deliberately coarse, so that a model of a different
microstructure can still be held to it:

| Stream | Record | Compared |
|---|---|---|
| launch | `L` … `E` | its length in cycles |
| issue | `I` | the cycle an instruction leaves the execute stage, its context, its pc |
| request | `Q` | the cycle a memory request is accepted at the core's port, its direction, address and byte enables |
| response | `P` | the latency of the response |
| uncore streams (second slice) | `S R M D X Y` | slice requests and responses, tile memory requests and responses, the device port |

Not compared: tags, the instruction port, anything inside the core. A wrong
pipeline rule shows up as a shifted issue cycle; the gate names the first.

## How the model is held to the RTL

The RTL's trace is a **recording**; the model is run in **replay** against
it. A replay boundary feeds the model the recorded behaviour at one interface
and masks everything beyond it:

- `core` level: the core's memory port replays the recorded beats (issue no
  earlier than the RTL's memory accepted; answer after the RTL's latency).
  The uncore is not exercised. Use it to validate the core alone.
- `uncore` level: the device's memory port replays the recorded accepts and
  responses; the cores, the NoC, the slices and the arbiter run on their own
  and every stream above the port is compared. This is the level that says
  whether the model predicts the machine above its memory.

The model never reads a trace. It takes a **replay input** (`TNCA_REPLAY 1`,
made by `replay_extract.py` from a recording): the one stream the level
replays, with cycles relative to each launch, and nothing else. What the
gate compares above the boundary, the model can only produce.

Moving the boundary outward exposes the next class of errors. A model that
is exact at the core level and wrong at the uncore level has a correct core
and a wrong uncore; look there.

## The discipline

The RTL is a clocked design and the model is written the same way: **every
decision of a cycle is taken from the pre-edge state, and every state update
happens at the edge.** `CYCLE_CORE.md` and the timing specifications
(`CORE_TIMING.md`, `UNCORE_TIMING.md`, `UNCORE_TIMING_BOARD.md`) are the rule
books; each rule cites the Verilog it comes from.

1. **A change is three things: a rule, the model, a switch.** Write the rule
   into the specification where the RTL's behaviour is described (cite the
   line of the Verilog); change the model to implement that rule; keep the
   change behind a configuration switch where the old behaviour has a
   meaning, so the model can be held to both. A model change with no rule
   in the specification is not finished.
2. **Pre-edge values, not post-edge.** Most divergences of one cycle come
   from using a value an edge has just written where the RTL reads the value
   before that edge (a free slot, a free tag, a round-robin pointer, an
   arbiter's grant). When a trace shifts by exactly one cycle at one event,
   look for this first.
3. **Resources are named by the instruction, not by its operands' order.**
   Register-file ports, banks and arbiters are bound by fields of the
   encoding. A model that numbers things in the order it meets them will
   agree on most programs and disagree on one.
4. **Everything that counts moves the shared state.** Host traffic between
   launches, warm-up accesses and the dispatcher's bookkeeping advance
   round-robin pointers and allocation tables just as a launch's own
   traffic does. A divergence at the very first event of a launch usually
   means the state the launch starts from is wrong.
5. **Scale by parameters only.** A larger configuration is the same model
   with other numbers (cores, mesh, ports, outstanding requests). Code that
   exists only for a large configuration has never been through the gate.

## How to read a divergence

`tracediff.py A.trace B.trace` prints the first divergence of every stream in
every launch: the launch, the stream, the index, the two records. Then:

- **Which side moved?** Compare the records' cycles. The model early by one
  at one event: rule 2. The model late and staying late: a stall or a
  dependency the RTL does not have.
- **What is the first event?** The first divergence is the cause; everything
  after it is consequence. Fix the first one, record again, look again.
- **A model trace that stops** (fewer launches, or `END` missing) is a
  refusal, not a divergence: the reason is in the model's `stdout.log`
  (`MODEL_REFUSED`, or a watchdog's "core made no progress" with a snapshot
  of the pipeline).
- **A replay mismatch** (the model's counters `*_replay_mismatches`,
  `*_replay_unrecorded`) means the model asked for a beat the RTL never
  made, or in another order: the program the model runs is not the program
  the RTL ran, or the model's memory requests are wrong before any timing is.
- **Debug dumps** the model can write per cycle, for one launch window:
  `TNCA_MODEL_TXF_DEBUG=core:launch:from:to:path` (the transfer engine and
  the port), `TNCA_MODEL_VRF_DEBUG=core:launch:from:to:path` (the register
  file's read arbiter), `TNCA_MODEL_PORT_DEBUG=path` (why a board port
  refuses). Launches are numbered from 0; cycles are relative to the launch.

## The script

```
python /opt/tnca-skills/cycle-model-alignment/scripts/align_case.py CASE.py [--level uncore] [--out DIR]
```

Run it from the repository checkout you work in. For one case of your own
(the `make()` file format of the test-case skill) it builds the model from
your tree (incrementally), elaborates the machine's board-top bench from its
device description in your tree with the trace monitor (the bench the
evaluator simulates), runs the case on the RTL and on the model in replay,
and prints the gate's verdict with the first divergence of every stream.
Both sides compile and launch the same program under the evaluator's
environment, on three active cores by default. Its `--help` lists the options
(the level, the device description, the model configuration, the active
cores, `--reuse-bench` when the RTL has not changed since the last run).

A change of the machine is a change on both sides: the RTL (or the device
description's parameters) and the model's configuration or code. A model that
was exact for one machine says nothing yet about the next until the gate has
seen it.
