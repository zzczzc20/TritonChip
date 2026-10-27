# Slice 2 evidence: the uncore at the cycle level, trace gate (2026-10-27)

Base `3e1a282` (PR #10), branch `model-align`. RTL traces: `record_rtl.py` on the evaluation host's
toolchain container (Icarus 12, the device bench on `sim-multicore`, three cores elaborated, the
monitor's per-core and uncore traces, `TNCA_TORCH_DEVICE=native`, run_graphs.py's knob sets,
warmup 0 / timed 1). Model traces: `record_model.py --replay RUN --replay-level uncore` with
`configs/r40-sim-cycle.cfg` (the simulation bench's structure: 4 sets x 2 ways, PORTS=1, no link
stages, `core_model=1`, `uncore_model=1`): the device's memory port replays the recorded `X`/`Y`
stream; the cores, the NoC, the slices and the arbiter are the model's own.

Files: `three_cores_uncore_replay_vs_rtl.json` (op.add and matmul with `--active-cores 3` on both
sides: the three core traces and the uncore trace of each), `board_uncore_replay_vs_rtl.json` (op.add
with three active cores on the board-top bench, `record_rtl.py --bench board`, replayed with
`r40-boardbench-cycle.cfg`: the board's links, dispatch stages and AXI adapters, UNCORE_TIMING_BOARD.md),
`tiny_uncore_replay_vs_rtl.json` (the tiny tier of `spec/r40/suites/existing.json`, one active core).

## What the tables say

- **Three active cores align exactly**: every core's launches (37 / 11 / 10 for op.add, the
  dispatcher's distribution), every compared stream of every core, and every uncore stream (slice
  requests and responses, tile memory requests and responses, the port's requests and responses),
  with the same cycle counts.
- Two rules the uncore level made visible, both corrected in the model (CYCLE_CORE.md): the grid
  dispatcher enqueues one core per cycle in round-robin order from the core after its last choice,
  so two cores that exit in the same cycle relaunch in consecutive cycles and that order decides
  which block each runs next (the model had let both relaunch together: the first such exit put a
  different block on core 0 and the replay could not follow); and the transfer engine's load slot
  is the pre-edge free slot of the request's tag, not one the window write frees at the same edge
  (under core-level replay the replayed port hid this; with the modelled NoC the misdirected
  response cost two cycles per affected beat and a hang).
- **The board-top bench aligns exactly too** (three active cores, every core trace, every uncore
  stream with `X`/`Y` at the AXI adapters): the two-stage links on every path, the dispatcher's
  command and completion links (first launch L+4, relaunch i+6), the adapter's admission and its
  single issue register. The recording runs in the runtime's default memory mode, which both sides
  take from run_graphs.py's RUNTIME knobs (`TNCA_RTL_MEM_PORTS=1`, `TNCA_RTL_OUTSTANDING=1`): all
  three homes share outlet 0 through the port arbiter and the adapter admits one line at a time, so
  that is what the model's `active_outlets=1`, `outstanding=1` describe. Found on the way: a replayed
  read's slot is free in the cycle of its recorded response (released at the final beat's edge).
- **Prediction on the board-top bench** (`board_prediction_vs_rtl.json`, same recording, no replay):
  the model's AXI slave is deterministic (every ready high, the lowest due read first) where the
  bench's line model stalls each AXI channel on a seeded `$random` in 30-40% of cycles and picks
  among due reads at random. Over the three cores the prediction is 2.9% below the recording
  (48436 vs 47052 cycles summed; per core +1.3%, -9.2%, -4.8%) and the dispatcher hands one block
  to a different core (38/10/10 against 37/11/10): the timing difference is the bench's random
  stalls, not structure. The U280's HBM is not that line model either, so the prediction
  configuration (`r40-board-cycle.cfg`) keeps the deterministic slave with one latency per port.
- The uncore trace carries no cycle verdict of its own (its `END` is each side's own clock, the
  model's including the host path); the core traces carry it.

### Three active cores, uncore replay (device bench, r40-sim-cycle.cfg)

Verdict: **aligned** (all 8 point(s) equal in every compared stream). Points 8, aligned 8, diverged 0, missing 0; |cycle error| median 0.0%, p95 0.0%.

| point | launches equal | cycles RTL | cycles model | error | first issue divergence | first request divergence |
|---|---:|---:|---:|---:|---|---|
| op.add_tiny_fp16-n1000 | 37/37 | 16491 | 16491 | +0.00% | equal | equal |
| op.add_tiny_fp16-n1000.core1 | 11/11 | 5365 | 5365 | +0.00% | equal | equal |
| op.add_tiny_fp16-n1000.core2 | 10/10 | 5144 | 5144 | +0.00% | equal | equal |
| op.add_tiny_fp16-n1000.uncore | 58/58 | | | | slice: equal / equal | port: equal / equal |
| op.matmul_tiny_fp16-m16-k32-n16 | 28/28 | 17318 | 17318 | +0.00% | equal | equal |
| op.matmul_tiny_fp16-m16-k32-n16.core1 | 0/0 | 0 | 0 | +0.00% | equal | equal |
| op.matmul_tiny_fp16-m16-k32-n16.core2 | 0/0 | 0 | 0 | +0.00% | equal | equal |
| op.matmul_tiny_fp16-m16-k32-n16.uncore | 28/28 | | | | slice: equal / equal | port: equal / equal |

### Board-top bench, three active cores, uncore replay at the adapters (r40-boardbench-cycle.cfg)

Verdict: **aligned** (all 4 point(s) equal in every compared stream). Points 4, aligned 4, diverged 0, missing 0; |cycle error| median 0.0%, p95 0.0%.

| point | launches equal | cycles RTL | cycles model | error | first issue divergence | first request divergence |
|---|---:|---:|---:|---:|---|---|
| op.add_tiny_fp16-n1000 | 37/37 | 24335 | 24335 | +0.00% | equal | equal |
| op.add_tiny_fp16-n1000.core1 | 11/11 | 11970 | 11970 | +0.00% | equal | equal |
| op.add_tiny_fp16-n1000.core2 | 10/10 | 12131 | 12131 | +0.00% | equal | equal |
| op.add_tiny_fp16-n1000.uncore | 58/58 | | | | slice: equal / equal | port: equal / equal |

### Board-top bench, three active cores, prediction (no replay: the model's deterministic AXI slave)

Verdict: **diverged** (4 of 4 point(s) diverge). Points 4, aligned 0, diverged 4, missing 0; |cycle error| median 4.8%, p95 9.2%.

| point | launches equal | cycles RTL | cycles model | error | first issue divergence | first request divergence |
|---|---:|---:|---:|---:|---|---|
| op.add_tiny_fp16-n1000 | 10/37 | 24335 | 24640 | +1.25% | L2 #304 | L2 #3 |
| op.add_tiny_fp16-n1000.core1 | 0/11 | 11970 | 10868 | -9.21% | L0 #72 | L0 #16 |
| op.add_tiny_fp16-n1000.core2 | 0/10 | 12131 | 11544 | -4.84% | L0 #72 | L0 #16 |
| op.add_tiny_fp16-n1000.uncore | 16/58 | | | | slice: L2 #3 / L1 #3 | port: L1 #1 / L1 #0 |
