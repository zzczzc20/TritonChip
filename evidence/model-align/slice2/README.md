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
`tiny_uncore_replay_vs_rtl.json` (the tiny tier of `spec/r40/suites/existing.json`, one active core: the 25
points recorded so far; `model.llama_decode` and `model.llama_prefill` were still recording on the RTL when this
table was made and are added when they land).

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
  Assumption behind both benches: no cache-maintenance command occurs between launches (the bench's
  host words go through the homes, so nothing needs flushing); the cycle uncore has no maintenance
  path yet, and a recording with flushes would need one (those writebacks move the arbiter too).
- **Prediction on the board-top bench** (`board_prediction_vs_rtl.json`, same recording, no replay):
  the model's AXI slave is deterministic (every ready high, the lowest due read first) where the
  bench's line model stalls each AXI channel on a seeded `$random` in 30-40% of cycles and picks
  among due reads at random. Over the three cores the prediction is 2.9% below the recording
  (48436 vs 47052 cycles summed; per core +1.3%, -9.2%, -4.8%) and the dispatcher hands one block
  to a different core (38/10/10 against 37/11/10): the timing difference is the bench's random
  stalls, not structure. The U280's HBM is not that line model either, so the prediction
  configuration (`r40-board-cycle.cfg`) keeps the deterministic slave with one latency per port.
- **The tiny tier aligns exactly** (25 points, one active core, every core and uncore stream, 853522
  cycles of model.gpt2_block among them). Its first run diverged on four points and the two causes
  were both rules of the RTL the single-point runs had not exercised (CYCLE_CORE.md): the transfer
  engine's store / transposed-load tag is the pre-edge free tag (a tag freed by a response at the same
  edge is reused by the next request; the model's post-edge choice let a fast slice hit in the middle
  of a 16-request burst leave one tag allocated forever: op.gemv, model.mlp, model.gpt2_block stopped
  with "core made no progress"); and the memory arbiter's round robin moves on the host's traffic
  between launches too (host words go through tile 0 and the homes, every host miss is a port
  transaction), so the model's zero-time host warm-up now advances the pointer (op.transpose's launch 28
  had its first three-way tie granted 2,0,1 instead of 1,2,0 and every later port record relabelled).
- The uncore trace carries no cycle verdict of its own (its `END` is each side's own clock, the
  model's including the host path); the core traces carry it.

### Tiny tier, one active core, uncore replay (device bench, r40-sim-cycle.cfg)

Verdict: **aligned** (all 100 point(s) equal in every compared stream). Points 100, aligned 100, diverged 0, missing 0; |cycle error| median 0.0%, p95 0.0%.

| point | launches equal | cycles RTL | cycles model | error | first issue divergence | first request divergence |
|---|---:|---:|---:|---:|---|---|
| model.gpt2_block_tiny_fp16-d64-h4-b1-s16 | 270/270 | 853522 | 853522 | +0.00% | equal | equal |
| model.gpt2_block_tiny_fp16-d64-h4-b1-s16.core1 | 0/0 | 0 | 0 | +0.00% | equal | equal |
| model.gpt2_block_tiny_fp16-d64-h4-b1-s16.core2 | 0/0 | 0 | 0 | +0.00% | equal | equal |
| model.gpt2_block_tiny_fp16-d64-h4-b1-s16.uncore | 270/270 | | | | slice: equal / equal | port: equal / equal |
| model.mlp_tiny_fp16-t16-d64 | 46/46 | 337256 | 337256 | +0.00% | equal | equal |
| model.mlp_tiny_fp16-t16-d64.core1 | 0/0 | 0 | 0 | +0.00% | equal | equal |
| model.mlp_tiny_fp16-t16-d64.core2 | 0/0 | 0 | 0 | +0.00% | equal | equal |
| model.mlp_tiny_fp16-t16-d64.uncore | 46/46 | | | | slice: equal / equal | port: equal / equal |
| model.resnet_block_tiny_fp32-n1-c8-hw8 | 54/54 | 982677 | 982677 | +0.00% | equal | equal |
| model.resnet_block_tiny_fp32-n1-c8-hw8.core1 | 0/0 | 0 | 0 | +0.00% | equal | equal |
| model.resnet_block_tiny_fp32-n1-c8-hw8.core2 | 0/0 | 0 | 0 | +0.00% | equal | equal |
| model.resnet_block_tiny_fp32-n1-c8-hw8.uncore | 54/54 | | | | slice: equal / equal | port: equal / equal |
| op.add_tiny_fp16-n1000 | 58/58 | 24953 | 24953 | +0.00% | equal | equal |
| op.add_tiny_fp16-n1000.core1 | 0/0 | 0 | 0 | +0.00% | equal | equal |
| op.add_tiny_fp16-n1000.core2 | 0/0 | 0 | 0 | +0.00% | equal | equal |
| op.add_tiny_fp16-n1000.uncore | 58/58 | | | | slice: equal / equal | port: equal / equal |
| op.add_tiny_fp32-n256 | 34/34 | 15183 | 15183 | +0.00% | equal | equal |
| op.add_tiny_fp32-n256.core1 | 0/0 | 0 | 0 | +0.00% | equal | equal |
| op.add_tiny_fp32-n256.core2 | 0/0 | 0 | 0 | +0.00% | equal | equal |
| op.add_tiny_fp32-n256.uncore | 34/34 | | | | slice: equal / equal | port: equal / equal |
| op.argmax_tiny_fp16-r2-v256 | 28/28 | 17517 | 17517 | +0.00% | equal | equal |
| op.argmax_tiny_fp16-r2-v256.core1 | 0/0 | 0 | 0 | +0.00% | equal | equal |
| op.argmax_tiny_fp16-r2-v256.core2 | 0/0 | 0 | 0 | +0.00% | equal | equal |
| op.argmax_tiny_fp16-r2-v256.uncore | 28/28 | | | | slice: equal / equal | port: equal / equal |
| op.attention_decode_tiny_fp16-b1-h2-l16-d16 | 76/76 | 54465 | 54465 | +0.00% | equal | equal |
| op.attention_decode_tiny_fp16-b1-h2-l16-d16.core1 | 0/0 | 0 | 0 | +0.00% | equal | equal |
| op.attention_decode_tiny_fp16-b1-h2-l16-d16.core2 | 0/0 | 0 | 0 | +0.00% | equal | equal |
| op.attention_decode_tiny_fp16-b1-h2-l16-d16.uncore | 76/76 | | | | slice: equal / equal | port: equal / equal |
| op.attention_prefill_tiny_fp16-b1-h2-s16-d16 | 118/118 | 152382 | 152382 | +0.00% | equal | equal |
| op.attention_prefill_tiny_fp16-b1-h2-s16-d16.core1 | 0/0 | 0 | 0 | +0.00% | equal | equal |
| op.attention_prefill_tiny_fp16-b1-h2-s16-d16.core2 | 0/0 | 0 | 0 | +0.00% | equal | equal |
| op.attention_prefill_tiny_fp16-b1-h2-s16-d16.uncore | 118/118 | | | | slice: equal / equal | port: equal / equal |
| op.bmm_tiny_fp16-b2-m16-k16-n16 | 30/30 | 20118 | 20118 | +0.00% | equal | equal |
| op.bmm_tiny_fp16-b2-m16-k16-n16.core1 | 0/0 | 0 | 0 | +0.00% | equal | equal |
| op.bmm_tiny_fp16-b2-m16-k16-n16.core2 | 0/0 | 0 | 0 | +0.00% | equal | equal |
| op.bmm_tiny_fp16-b2-m16-k16-n16.uncore | 30/30 | | | | slice: equal / equal | port: equal / equal |
| op.conv2d_tiny_fp32-n1-cin4-h8-w8-cout8 | 38/38 | 442022 | 442022 | +0.00% | equal | equal |
| op.conv2d_tiny_fp32-n1-cin4-h8-w8-cout8.core1 | 0/0 | 0 | 0 | +0.00% | equal | equal |
| op.conv2d_tiny_fp32-n1-cin4-h8-w8-cout8.core2 | 0/0 | 0 | 0 | +0.00% | equal | equal |
| op.conv2d_tiny_fp32-n1-cin4-h8-w8-cout8.uncore | 38/38 | | | | slice: equal / equal | port: equal / equal |
| op.cumsum_tiny_fp32-r4-c64 | 28/28 | 16485 | 16485 | +0.00% | equal | equal |
| op.cumsum_tiny_fp32-r4-c64.core1 | 0/0 | 0 | 0 | +0.00% | equal | equal |
| op.cumsum_tiny_fp32-r4-c64.core2 | 0/0 | 0 | 0 | +0.00% | equal | equal |
| op.cumsum_tiny_fp32-r4-c64.uncore | 28/28 | | | | slice: equal / equal | port: equal / equal |
| op.embedding_tiny_fp16-v256-d64-t16 | 58/58 | 117585 | 117585 | +0.00% | equal | equal |
| op.embedding_tiny_fp16-v256-d64-t16.core1 | 0/0 | 0 | 0 | +0.00% | equal | equal |
| op.embedding_tiny_fp16-v256-d64-t16.core2 | 0/0 | 0 | 0 | +0.00% | equal | equal |
| op.embedding_tiny_fp16-v256-d64-t16.uncore | 58/58 | | | | slice: equal / equal | port: equal / equal |
| op.gelu_tanh_tiny_fp16-n256 | 34/34 | 16990 | 16990 | +0.00% | equal | equal |
| op.gelu_tanh_tiny_fp16-n256.core1 | 0/0 | 0 | 0 | +0.00% | equal | equal |
| op.gelu_tanh_tiny_fp16-n256.core2 | 0/0 | 0 | 0 | +0.00% | equal | equal |
| op.gelu_tanh_tiny_fp16-n256.uncore | 34/34 | | | | slice: equal / equal | port: equal / equal |
| op.gemv_tiny_fp16-k64-n64 | 30/30 | 57187 | 57187 | +0.00% | equal | equal |
| op.gemv_tiny_fp16-k64-n64.core1 | 0/0 | 0 | 0 | +0.00% | equal | equal |
| op.gemv_tiny_fp16-k64-n64.core2 | 0/0 | 0 | 0 | +0.00% | equal | equal |
| op.gemv_tiny_fp16-k64-n64.uncore | 30/30 | | | | slice: equal / equal | port: equal / equal |
| op.layernorm_tiny_fp16-r4-d64 | 28/28 | 18359 | 18359 | +0.00% | equal | equal |
| op.layernorm_tiny_fp16-r4-d64.core1 | 0/0 | 0 | 0 | +0.00% | equal | equal |
| op.layernorm_tiny_fp16-r4-d64.core2 | 0/0 | 0 | 0 | +0.00% | equal | equal |
| op.layernorm_tiny_fp16-r4-d64.uncore | 28/28 | | | | slice: equal / equal | port: equal / equal |
| op.linear_tiny_fp16-m16-k64-n64 | 30/30 | 47546 | 47546 | +0.00% | equal | equal |
| op.linear_tiny_fp16-m16-k64-n64.core1 | 0/0 | 0 | 0 | +0.00% | equal | equal |
| op.linear_tiny_fp16-m16-k64-n64.core2 | 0/0 | 0 | 0 | +0.00% | equal | equal |
| op.linear_tiny_fp16-m16-k64-n64.uncore | 30/30 | | | | slice: equal / equal | port: equal / equal |
| op.matmul_tiny_fp16-m16-k32-n16 | 28/28 | 17301 | 17301 | +0.00% | equal | equal |
| op.matmul_tiny_fp16-m16-k32-n16.core1 | 0/0 | 0 | 0 | +0.00% | equal | equal |
| op.matmul_tiny_fp16-m16-k32-n16.core2 | 0/0 | 0 | 0 | +0.00% | equal | equal |
| op.matmul_tiny_fp16-m16-k32-n16.uncore | 28/28 | | | | slice: equal / equal | port: equal / equal |
| op.matmul_tiny_fp32-m16-k32-n16 | 28/28 | 25970 | 25970 | +0.00% | equal | equal |
| op.matmul_tiny_fp32-m16-k32-n16.core1 | 0/0 | 0 | 0 | +0.00% | equal | equal |
| op.matmul_tiny_fp32-m16-k32-n16.core2 | 0/0 | 0 | 0 | +0.00% | equal | equal |
| op.matmul_tiny_fp32-m16-k32-n16.uncore | 28/28 | | | | slice: equal / equal | port: equal / equal |
| op.rmsnorm_tiny_fp16-r4-d64 | 28/28 | 16773 | 16773 | +0.00% | equal | equal |
| op.rmsnorm_tiny_fp16-r4-d64.core1 | 0/0 | 0 | 0 | +0.00% | equal | equal |
| op.rmsnorm_tiny_fp16-r4-d64.core2 | 0/0 | 0 | 0 | +0.00% | equal | equal |
| op.rmsnorm_tiny_fp16-r4-d64.uncore | 28/28 | | | | slice: equal / equal | port: equal / equal |
| op.rope_tiny_fp16-b1-h2-s16-d16 | 58/58 | 29251 | 29251 | +0.00% | equal | equal |
| op.rope_tiny_fp16-b1-h2-s16-d16.core1 | 0/0 | 0 | 0 | +0.00% | equal | equal |
| op.rope_tiny_fp16-b1-h2-s16-d16.core2 | 0/0 | 0 | 0 | +0.00% | equal | equal |
| op.rope_tiny_fp16-b1-h2-s16-d16.uncore | 58/58 | | | | slice: equal / equal | port: equal / equal |
| op.silu_tiny_fp16-n256 | 34/34 | 15798 | 15798 | +0.00% | equal | equal |
| op.silu_tiny_fp16-n256.core1 | 0/0 | 0 | 0 | +0.00% | equal | equal |
| op.silu_tiny_fp16-n256.core2 | 0/0 | 0 | 0 | +0.00% | equal | equal |
| op.silu_tiny_fp16-n256.uncore | 34/34 | | | | slice: equal / equal | port: equal / equal |
| op.softmax_tiny_fp16-r4-c64 | 30/30 | 21026 | 21026 | +0.00% | equal | equal |
| op.softmax_tiny_fp16-r4-c64.core1 | 0/0 | 0 | 0 | +0.00% | equal | equal |
| op.softmax_tiny_fp16-r4-c64.core2 | 0/0 | 0 | 0 | +0.00% | equal | equal |
| op.softmax_tiny_fp16-r4-c64.uncore | 30/30 | | | | slice: equal / equal | port: equal / equal |
| op.softmax_tiny_fp32-r3-c37 | 28/28 | 15019 | 15019 | +0.00% | equal | equal |
| op.softmax_tiny_fp32-r3-c37.core1 | 0/0 | 0 | 0 | +0.00% | equal | equal |
| op.softmax_tiny_fp32-r3-c37.core2 | 0/0 | 0 | 0 | +0.00% | equal | equal |
| op.softmax_tiny_fp32-r3-c37.uncore | 28/28 | | | | slice: equal / equal | port: equal / equal |
| op.swiglu_tiny_fp16-n256 | 34/34 | 16730 | 16730 | +0.00% | equal | equal |
| op.swiglu_tiny_fp16-n256.core1 | 0/0 | 0 | 0 | +0.00% | equal | equal |
| op.swiglu_tiny_fp16-n256.core2 | 0/0 | 0 | 0 | +0.00% | equal | equal |
| op.swiglu_tiny_fp16-n256.uncore | 34/34 | | | | slice: equal / equal | port: equal / equal |
| op.transpose_tiny_fp32-r32-c32 | 30/30 | 25134 | 25134 | +0.00% | equal | equal |
| op.transpose_tiny_fp32-r32-c32.core1 | 0/0 | 0 | 0 | +0.00% | equal | equal |
| op.transpose_tiny_fp32-r32-c32.core2 | 0/0 | 0 | 0 | +0.00% | equal | equal |
| op.transpose_tiny_fp32-r32-c32.uncore | 30/30 | | | | slice: equal / equal | port: equal / equal |

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
