# The cycle-level core (`core_model=1`)

The first slice's model of one core: `tools/performance_harness/systemc/src/cycle_core.inc`,
selected by the configuration key `core_model=1` (absent or 0: the event-driven
core of M1/M2, untouched). It follows CORE_TIMING.md and the TPCS discipline:

- **State** is what survives a clock edge (CORE_TIMING section 1): per-context
  pc/running/blocked/parked, the F1/FB/D/R/X bundles, scoreboards, the WB
  stage, the writeback reservation register, the LSU units, engine busy
  counters, barrier sleep records. Nothing else carries timing.
- **One step per cycle.** `step()` first computes every wire from the pre-edge
  state (stalls, kills, hazards, the port arbiter, the bank arbiter), then
  applies every register update at the edge. Reads never see a write of the
  same cycle except where the RTL forwards (the write-first scalar and
  predicate files, `WB_HIT`); writes are deferred entries with the cycle they
  take the writeback port.
- **X executes architecturally** with the decode, ALU, vector, SFU and matrix
  arithmetic of the existing model (the executor is reused); its results are
  deferred writes with the RTL's port cycle (single-cycle at x, FP at x+2, MUL
  at x+3, loads at the last response +3, reductions and SFU through `red_pend`).
- **Transactions**: the memory port issues at most one beat per cycle,
  bound to the timeline memory (prediction) or to the RTL's recorded trace
  (validation: `TNCA_MODEL_CORE_REPLAY=<rtl trace>`); the transfer engine is
  the cycle-level `cycle_tensor.inc` (tnca_tensor.v's states: the serial
  descriptor load, setup and geometry cycles, the load walk with eight slots
  and the window write port, the store walk with sixteen tags and the
  register reads through port 4 of the banked file, the transposed walk,
  drain and done) with strict priority on the port; the matrix engine is the
  cycle-level `cycle_matrix.inc` (tnca_matrix.v's states: the A-fragment
  buffer load, one B row per k, the MAC sweep, drain and done, with every
  fragment read a two-cycle request/response on port 3 of the banked file,
  repeated when the arbiter refuses it); the barrier model is the existing
  one, driven at the cycle level (sleep records, wake pulses).

Not modelled in this version, each a known divergence source to retire with
the gate when a kernel reaches it: response reordering on the port, atomics
(refused as before). On the suite's smallest tier (2026-10-27) every point
aligns exactly in replay mode (evidence/model-align/slice1).

**R41 (`core_lsu_hybrid=1`, CORE_TIMING 3.6).** The R41 core's memory side is
the hybrid LSU: a command register between X and the LSU, a line path that
sends a contiguous vector access as one 64-byte request (64 slots, completion
round robin), the word path as one unit with the two-entry request queue, a
completion register in front of the writeback port, and the LSU ahead of the
transfer engine on the port (`core_tensor_line=1`). The 64-byte port is traced
in the RTL monitor's 512-bit form. The transfer engine is R41's line-wide,
multi-command one (`core_tensor_line=1`, CORE_TIMING 4.4: line loads and stores
through 64 line slots with up to eight commands in flight after the walk is
free, the four-line TensorMap load, the transposed walk's line buffer, 1-byte
transposes, the registered address multiply); the vector multiply family and
the integer reductions write back a cycle after X (`core_valu_late=1`, 2.12).
With the switches off the R40 rules stand unchanged. Not yet in the model: R41's
matrix engine (MTX_VRF_REG, the packed MAC lanes), the G2 FMA rebuild
(FMA_IMPL 1), line requests in the cycle uncore (slice 2).

Validation path: `record_model.py --replay RTL_RUN` gives every point's
model process the RTL trace of that point; the replay port checks each LSU
beat's address and direction against the record, issues no earlier than the
RTL did (the memory system's back-pressure) and answers after the RTL's
latency. Mismatches are counted in the model's counters
(`core_N_replay_mismatches`) and never silently absorbed.

## The cycle-level uncore (`uncore_model=1`, second slice)

`tools/performance_harness/systemc/src/cycle_uncore.inc`, selected by the
configuration key `uncore_model=1` (which requires `core_model=1`), follows
UNCORE_TIMING.md: the grid dispatcher, the core and slice network
interfaces, the 3x1 mesh (per-port, per-virtual-network FIFOs, round-robin
output arbiters, the elastic links), the NUCA slices (request queue, MSHRs,
targets, response queue, the engine's states), the memory arbiter with its
hold rule and the bench's DRAM model (its LFSR-driven ready and reorder
pick, latency 11). It is one SystemC process that steps half a cycle after
the cores: the cores' requests of cycle t enter the source router's FIFO at
the edge of t, the responses the uncore delivers at cycle t reach the core
at t+1.

- **State**: every FIFO, the routers' arbiters, each slice's queues and
  engine registers, the arbiter's hold, the DRAM pool and LFSR.
- **One step per cycle**: wires from the pre-edge state (ready chains, the
  arbitration, the slice engines' choices, the DRAM's ready and the response
  it presents), then the edge.
- **Dispatch** (chip.cpp `dispatch`): the RTL's one-enqueue-per-cycle
  round robin, `rr` the core after the last choice, the block bound to the
  core at the enqueue, first launch at L+2, a core a candidate again three
  cycles after its last cycle. Two cores that exit in the same cycle relaunch
  in consecutive cycles in the RTL's order, which decides which block each
  core runs next; a model that let both relaunch together diverged on the
  three-core grids at the first such exit.
- **Validation path**: `--replay-level uncore` replays the device's memory
  port (`TNCA_MODEL_UNCORE_REPLAY=<uncore trace>`): the arbiter grants a
  request no earlier than the RTL's port accepted it (the recorded `X`), the
  DRAM answers at the recorded `Y`. Everything above the port is the model's
  own, so every core trace and every uncore stream (slice requests and
  responses, tile memory requests and responses, the port) is compared
  whole. With the recorded `X`/`Y` on the simulation bench, the three-core
  points recorded so far align exactly in every stream (2026-10-27).
- **Prediction path**: the model's own LFSR and pool; the DRAM's ready
  pattern depends on the absolute cycle, so a prediction's total differs
  from a recording by the phase of that pattern, not by structure.

**The board configuration** (`uncore_board=1`, UNCORE_TIMING_BOARD.md): the
same slices and mesh with two-stage links everywhere; per home a home link
(two 2-deep request stages, two response registers), its port (with every
outlet active home h is port h and nothing arbitrates; with one active
outlet, the runtime's default `TNCA_RTL_MEM_PORTS=1` that both recordings
and run_graphs.py use, the three homes share outlet 0 through the port
arbiter's round robin and hold), an outlet link of the same shape and a
`tnca_line_axi` adapter (`outstanding` slots, one with the runtime's default
`TNCA_RTL_OUTSTANDING=1`, eight on the image; one issue register: a read holds it
two cycles, a write three; same-line hazards refuse the request at the head
of the outlet link; a write's slot stays until its B and gives no response);
dispatch through two-stage command and completion links (first launch L+4,
relaunch i+6, done i+3). Behind the adapter the AXI slave is deterministic:
every ready high, reads answered after the port's latency and the bench's
per-slot skew, the lowest due read first, one beat per cycle; the bench's
seeded `$random` readies and read picks are not reproduced, so a board
prediction differs from the board-top bench by those stalls, not by
structure. The validation path replays the adapter boundary: each port's
accepts no earlier than its recorded `X`, each port's responses at its
recorded `Y` (then +2 +2 to the slice); writebacks have no record (no B in
the trace), so their slot release is the deterministic model's. The config
`r40-boardbench-cycle.cfg` is this with the bench's 4x2 slices, for the
gate; `r40-board-cycle.cfg` is the prediction configuration for the U280
(one HBM latency for every port, the fitted 23, an assumption until
measured; the image's 32x8 slice with E_ALLOC and the data-array pipeline
is still modelled as the bench's slice).

Rules the uncore level made visible (each fixed where it lives, each
invisible under core-level replay because the replayed port hid them):

- the transfer engine's load slot is the pre-edge `free_slot` of the
  request's tag, not one freed by the window write at the same edge;
- likewise its store / transposed-load tag is the pre-edge `free_tag`
  (`tnca_tensor.v` 506-511, 1110, 1164): a tag freed by a response at the
  same edge is reused by the next request. The model allocated the post-edge
  tag, so a fast slice hit landing in the middle of a 16-request burst left
  one tag allocated forever and the engine never drained (three points: a
  refusal "core made no progress");
- the dispatcher above;
- a vector source reads on the port of its instruction field (a: 0, b: 1,
  c: 2; store data and `v.insert`'s Vd on vb, a scatter's data on vc), not
  on the port of its position among the sources. The model numbered sources
  in order, so an `a.fma.row` (vector in `b`) asked on port 0 and lost its
  bank to the matrix engine's port 3 where the RTL's port 1 wins (round
  robin from `rr_ptr = 1`): one refused fragment read, the mma one cycle
  longer, the accumulator read behind it one cycle later (one launch of a
  graph-level point, invisible in 1.7 M cycles otherwise);
- the memory arbiter's round robin also moves on the host's traffic between
  launches: the host's words go through tile 0 and the homes
  (`tnca_top.v:488`), and every host miss is a fill (and maybe a writeback)
  on the port, so the pointer resumes after that home. The model's zero-time
  host warm-up (`host_touch`) now advances the pointer the same way; without
  it the first three-way tie of a launch could be granted in the wrong order
  (one launch of one point: 1,2,0 recorded, 2,0,1 modelled, a pure relabelling
  of the three slices from then on).
