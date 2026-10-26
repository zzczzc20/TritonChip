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
(refused as before). On the tiny tier of `spec/r40/suites/existing.json`
(2026-10-27) every operator point and the MLP model align exactly in replay
mode (evidence/model-align/slice1).

Validation path: `record_model.py --replay RTL_RUN` gives every point's
model process the RTL trace of that point; the replay port checks each LSU
beat's address and direction against the record, issues no earlier than the
RTL did (the memory system's back-pressure) and answers after the RTL's
latency. Mismatches are counted in the model's counters
(`core_N_replay_mismatches`) and never silently absorbed.
