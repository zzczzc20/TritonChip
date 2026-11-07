# Slice 1 evidence: one core, trace gate (2026-10-27)

Base `3e1a282` (PR #10), branch `model-align`. RTL traces: `tools/performance_harness/align/record_rtl.py`
on the evaluation host's toolchain container (Icarus 12, `sim-multicore`, three cores elaborated, one
active, `TNCA_TORCH_DEVICE=native`, run_graphs.py's knob sets, warmup 0 / timed 1), the tiny tier of
`spec/r40/suites/existing.json`. Model traces: `record_model.py --replay` with
`configs/r40-board-cycle.cfg` (the M2 board structure, `core_model=1`).

Files: `cycle_core_replay_vs_rtl_model_points.json` (the four model points on their full RTL recordings), `event_model_vs_rtl*.json` (the M2 event-driven core, `core_model=0`, its own memory; the
first table), `cycle_core_replay_vs_rtl*.json` (the cycle core replaying the RTL's beats; the
current table). `*_order.json` are the same comparisons with `--order-only --data`.

## What the tables say

- **The cycle core aligns exactly on all 23 operator points and on `model.mlp`**: every launch,
  every compared stream (issue events, memory-port beats, launch length), the same cycle count,
  and zero replay mismatches (the model's beats are the RTL's beats, in order).
- The four other `model.*` rows in the main table compare against RTL traces the recorder's
  two-hour host timeout cut short (`rc=1`). Their full recordings (`record_rtl.py --timeout
  24000`, 1.3 to 4.4 h each) are the third table: **gpt2_block and resnet_block exact; llama_decode
  (338 of 340 launches) and llama_prefill (561 of 564) exact in total cycles** (+0 and +2 cycles over
  1.7 M and 2.5 M), with the remaining launches shifted by one cycle somewhere inside and zero
  replay mismatches. The residual cases are recorded in `cycle_core_replay_vs_rtl_model_points.json`.
- `data` divergence at launch 1, beat 3 on every point (`--data` mode, not part of the coarse
  gate): the RTL's memory returns 0 for a word the model returns as the value a previous launch
  stored. Open question for the memory system (slice 2): a don't-care read, or a visibility rule
  the model does not have?
- The event-driven core, for comparison, diverges on every point from the 27th instruction (its
  context arbitration) and is 9.6% off in median cycles.
- What the gate corrected on the way (each a one-line rule in CORE_TIMING.md's terms): the
  transfer engine's S_TDRAIN reads the slot register before the edge that frees it; a td.load's
  first request is the cycle after its instruction; a completed load slot is seen the cycle
  after its response; `insert` reads its destination as a vector source; the matrix engine's
  fragment reads contend on port 3 of the banked file.

### The four model points on their full RTL recordings (record_rtl.py --timeout 24000)

Verdict: **diverged** (2 of 4 point(s) diverge). Points 4, aligned 2, diverged 2, missing 0; |cycle error| median 0.0%, p95 0.0%.

| point | launches equal | cycles RTL | cycles model | error | first issue divergence | first request divergence |
|---|---:|---:|---:|---:|---|---|
| model.gpt2_block_tiny_fp16-d64-h4-b1-s16 | 270/270 | 854124 | 854124 | +0.00% | equal | equal |
| model.llama_decode_tiny_fp16-b1-p16 | 338/340 | 1733826 | 1733826 | +0.00% | L168 #1042 | equal |
| model.llama_prefill_tiny_fp16-b1-s16 | 561/564 | 2487258 | 2487260 | +0.00% | L147 #720 | L147 #1238 |
| model.resnet_block_tiny_fp32-n1-c8-hw8 | 54/54 | 981275 | 981275 | +0.00% | equal | equal |

### Cycle core (core_model=1), replay of the RTL's beats, tiny (2026-10-27, commit after the matrix engine)

Verdict: **diverged** (4 of 27 point(s) diverge). Points 27, aligned 23, diverged 4, missing 0; |cycle error| median 0.0%, p95 166.4%.

| point | launches equal | cycles RTL | cycles model | error | first issue divergence | first request divergence | order-only: issue | request | data |
|---|---:|---:|---:|---:|---|---|---|---|---|
| model.gpt2_block_tiny_fp16-d64-h4-b1-s16 | 154/155 | 462540 | 834028 | +80.31% | L154 #513 | L154 #506 | L154 #513 | L154 #506 | L1 #3 |
| model.llama_decode_tiny_fp16-b1-p16 | 163/163 | 616411 | 1642341 | +166.44% | equal | equal | equal | equal | L1 #3 |
| model.llama_prefill_tiny_fp16-b1-s16 | 147/148 | 503323 | 2287766 | +354.53% | L97 #720 | L97 #1238 | equal | equal | L1 #3 |
| model.mlp_tiny_fp16-t16-d64 | 46/46 | 337256 | 337256 | +0.00% | equal | equal | equal | equal | L1 #3 |
| model.resnet_block_tiny_fp32-n1-c8-hw8 | 53/54 | 859327 | 959246 | +11.63% | L53 #1933 | L53 #1194 | L53 #1933 | L53 #1194 | L1 #3 |
| op.add_tiny_fp16-n1000 | 58/58 | 24954 | 24954 | +0.00% | equal | equal | equal | equal | L1 #3 |
| op.add_tiny_fp32-n256 | 34/34 | 15198 | 15198 | +0.00% | equal | equal | equal | equal | L1 #3 |
| op.argmax_tiny_fp16-r2-v256 | 28/28 | 17542 | 17542 | +0.00% | equal | equal | equal | equal | L1 #3 |
| op.attention_decode_tiny_fp16-b1-h2-l16-d16 | 76/76 | 54448 | 54448 | +0.00% | equal | equal | equal | equal | L1 #3 |
| op.attention_prefill_tiny_fp16-b1-h2-s16-d16 | 118/118 | 152454 | 152454 | +0.00% | equal | equal | equal | equal | L1 #3 |
| op.bmm_tiny_fp16-b2-m16-k16-n16 | 30/30 | 20112 | 20112 | +0.00% | equal | equal | equal | equal | L1 #3 |
| op.conv2d_tiny_fp32-n1-cin4-h8-w8-cout8 | 38/38 | 444430 | 444430 | +0.00% | equal | equal | equal | equal | L1 #3 |
| op.cumsum_tiny_fp32-r4-c64 | 28/28 | 16462 | 16462 | +0.00% | equal | equal | equal | equal | L1 #3 |
| op.embedding_tiny_fp16-v256-d64-t16 | 58/58 | 117596 | 117596 | +0.00% | equal | equal | equal | equal | L1 #3 |
| op.gelu_tanh_tiny_fp16-n256 | 34/34 | 16999 | 16999 | +0.00% | equal | equal | equal | equal | L1 #3 |
| op.gemv_tiny_fp16-k64-n64 | 30/30 | 57271 | 57271 | +0.00% | equal | equal | equal | equal | L1 #3 |
| op.layernorm_tiny_fp16-r4-d64 | 28/28 | 18276 | 18276 | +0.00% | equal | equal | equal | equal | L1 #3 |
| op.linear_tiny_fp16-m16-k64-n64 | 30/30 | 47628 | 47628 | +0.00% | equal | equal | equal | equal | L1 #3 |
| op.matmul_tiny_fp16-m16-k32-n16 | 28/28 | 17308 | 17308 | +0.00% | equal | equal | equal | equal | L1 #3 |
| op.matmul_tiny_fp32-m16-k32-n16 | 28/28 | 25981 | 25981 | +0.00% | equal | equal | equal | equal | L1 #3 |
| op.rmsnorm_tiny_fp16-r4-d64 | 28/28 | 16868 | 16868 | +0.00% | equal | equal | equal | equal | L1 #3 |
| op.rope_tiny_fp16-b1-h2-s16-d16 | 58/58 | 29320 | 29320 | +0.00% | equal | equal | equal | equal | L1 #3 |
| op.silu_tiny_fp16-n256 | 34/34 | 15783 | 15783 | +0.00% | equal | equal | equal | equal | L1 #3 |
| op.softmax_tiny_fp16-r4-c64 | 30/30 | 21046 | 21046 | +0.00% | equal | equal | equal | equal | L1 #3 |
| op.softmax_tiny_fp32-r3-c37 | 28/28 | 15111 | 15111 | +0.00% | equal | equal | equal | equal | L1 #3 |
| op.swiglu_tiny_fp16-n256 | 34/34 | 16714 | 16714 | +0.00% | equal | equal | equal | equal | L1 #3 |
| op.transpose_tiny_fp32-r32-c32 | 30/30 | 25158 | 25158 | +0.00% | equal | equal | equal | equal | L1 #3 |

### Event-driven M2 model (core_model=0), its own memory, tiny

Verdict: **diverged** (27 of 27 point(s) diverge). Points 27, aligned 0, diverged 27, missing 0; |cycle error| median 9.6%, p95 141.3%.

| point | launches equal | cycles RTL | cycles model | error | first issue divergence | first request divergence | order-only: issue | request | data |
|---|---:|---:|---:|---:|---|---|---|---|---|
| model.gpt2_block_tiny_fp16-d64-h4-b1-s16 | 0/155 | 462540 | 805142 | +74.07% | L0 #0 | L0 #0 | L0 #26 | L26 #7 | L1 #3 |
| model.llama_decode_tiny_fp16-b1-p16 | 0/163 | 616411 | 1487437 | +141.31% | L0 #0 | L0 #0 | L0 #26 | L27 #8 | L1 #3 |
| model.llama_prefill_tiny_fp16-b1-s16 | 0/148 | 503323 | 2232750 | +343.60% | L0 #0 | L0 #0 | L0 #26 | L26 #11 | L1 #3 |
| model.mlp_tiny_fp16-t16-d64 | 0/46 | 337256 | 266496 | -20.98% | L0 #0 | L0 #0 | L0 #26 | L26 #24 | L1 #3 |
| model.resnet_block_tiny_fp32-n1-c8-hw8 | 0/54 | 859327 | 749112 | -12.83% | L0 #0 | L0 #0 | L0 #26 | L26 #6 | L1 #3 |
| op.add_tiny_fp16-n1000 | 0/58 | 24954 | 22558 | -9.60% | L0 #0 | L0 #0 | L0 #26 | L26 #12 | L1 #3 |
| op.add_tiny_fp32-n256 | 0/34 | 15198 | 13852 | -8.86% | L0 #0 | L0 #0 | L0 #26 | L26 #16 | L1 #3 |
| op.argmax_tiny_fp16-r2-v256 | 0/28 | 17542 | 16164 | -7.86% | L0 #0 | L0 #0 | L0 #26 | L26 #12 | L1 #3 |
| op.attention_decode_tiny_fp16-b1-h2-l16-d16 | 0/76 | 54448 | 67343 | +23.68% | L0 #0 | L0 #0 | L0 #26 | L26 #45 | L1 #3 |
| op.attention_prefill_tiny_fp16-b1-h2-s16-d16 | 0/118 | 152454 | 160401 | +5.21% | L0 #0 | L0 #0 | L0 #26 | L26 #7 | L1 #3 |
| op.bmm_tiny_fp16-b2-m16-k16-n16 | 0/30 | 20112 | 20739 | +3.12% | L0 #0 | L0 #0 | L0 #26 | L26 #11 | L1 #3 |
| op.conv2d_tiny_fp32-n1-cin4-h8-w8-cout8 | 0/38 | 444430 | 346479 | -22.04% | L0 #0 | L0 #0 | L0 #26 | L26 #6 | L1 #3 |
| op.cumsum_tiny_fp32-r4-c64 | 0/28 | 16462 | 14869 | -9.68% | L0 #0 | L0 #0 | L0 #26 | L26 #16 | L1 #3 |
| op.embedding_tiny_fp16-v256-d64-t16 | 0/58 | 117596 | 112359 | -4.45% | L0 #0 | L0 #0 | L0 #26 | L26 #40 | L1 #3 |
| op.gelu_tanh_tiny_fp16-n256 | 0/34 | 16999 | 15964 | -6.09% | L0 #0 | L0 #0 | L0 #26 | L26 #24 | L1 #3 |
| op.gemv_tiny_fp16-k64-n64 | 0/30 | 57271 | 52931 | -7.58% | L0 #0 | L0 #0 | L0 #26 | L26 #8 | L1 #3 |
| op.layernorm_tiny_fp16-r4-d64 | 0/28 | 18276 | 16309 | -10.76% | L0 #0 | L0 #0 | L0 #26 | L26 #7 | L1 #3 |
| op.linear_tiny_fp16-m16-k64-n64 | 0/30 | 47628 | 40444 | -15.08% | L0 #0 | L0 #0 | L0 #26 | L26 #24 | L1 #3 |
| op.matmul_tiny_fp16-m16-k32-n16 | 0/28 | 17308 | 17574 | +1.54% | L0 #0 | L0 #0 | L0 #26 | L26 #11 | L1 #3 |
| op.matmul_tiny_fp32-m16-k32-n16 | 0/28 | 25981 | 25984 | +0.01% | L0 #0 | L0 #0 | L0 #26 | L26 #11 | L1 #3 |
| op.rmsnorm_tiny_fp16-r4-d64 | 0/28 | 16868 | 14870 | -11.84% | L0 #0 | L0 #0 | L0 #26 | L26 #7 | L1 #3 |
| op.rope_tiny_fp16-b1-h2-s16-d16 | 0/58 | 29320 | 24969 | -14.84% | L0 #0 | L0 #0 | L0 #26 | L26 #12 | L1 #3 |
| op.silu_tiny_fp16-n256 | 0/34 | 15783 | 14644 | -7.22% | L0 #0 | L0 #0 | L0 #26 | L26 #7 | L1 #3 |
| op.softmax_tiny_fp16-r4-c64 | 0/30 | 21046 | 22155 | +5.27% | L0 #0 | L0 #0 | L0 #26 | L26 #3 | L1 #3 |
| op.softmax_tiny_fp32-r3-c37 | 0/28 | 15111 | 13876 | -8.17% | L0 #0 | L0 #0 | L0 #26 | L26 #6 | L1 #3 |
| op.swiglu_tiny_fp16-n256 | 0/34 | 16714 | 15298 | -8.47% | L0 #0 | L0 #0 | L0 #26 | L26 #7 | L1 #3 |
| op.transpose_tiny_fp32-r32-c32 | 0/30 | 25158 | 21287 | -15.39% | L0 #0 | L0 #0 | L0 #26 | L26 #52 | L1 #3 |
