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
  drain and done) with strict priority on the port; the matrix engine is a
  busy window with the RTL's formulas; the barrier model is the existing one,
  driven at the cycle level (sleep records, wake pulses).

Not modelled in this version, each a known divergence source to retire with
the gate: VRF port 3 (the matrix engine's fragment reads do not contend with
the execute ports), response reordering on the port, atomics (refused as
before).

Validation path: `record_model.py --replay RTL_RUN` gives every point's
model process the RTL trace of that point; the replay port checks each LSU
beat's address and direction against the record, issues no earlier than the
RTL did (the memory system's back-pressure) and answers after the RTL's
latency. Mismatches are counted in the model's counters
(`core_N_replay_mismatches`) and never silently absorbed.
