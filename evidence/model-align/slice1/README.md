# Slice 1 evidence: one core, trace gate (intermediate, 2026-10-27)

Base `3e1a282` (PR #10), branch `model-align`. RTL traces: `tools/performance_harness/align/record_rtl.py`
on the evaluation host's toolchain container (Icarus 12, `sim-multicore`, three cores elaborated, one
active, `TNCA_TORCH_DEVICE=native`, run_graphs.py's knob sets, warmup 0 / timed 1), the tiny tier of
`spec/r40/suites/existing.json`. Four `model.*` points hit the recorder's 2 h host timeout in this
run (`rc=1`): their RTL traces end early, so their rows compare a truncated RTL against the full
model and are not meaningful here; they are being re-recorded with a longer timeout.

Files: `event_model_vs_rtl*.json` (the M2 event-driven core, `core_model=0`, its own memory),
`cycle_core_replay_vs_rtl*.json` (the cycle core, `core_model=1`, replaying the RTL's LSU beats).
`*_order.json` are the same comparisons with `--order-only --data`.

## What the tables say

- The cycle core aligns **exactly** (every launch, every compared stream, identical cycle counts)
  on 12 of 27 points: both `add`, `argmax`, `embedding`, `gelu_tanh`, `layernorm`, `rmsnorm`,
  `rope`, `silu`, `softmax fp32`, `swiglu`, `transpose`; `cumsum` matches in total cycles with one
  instruction one cycle early in launch 26. These are the kernels that use the LSU alone.
- Every remaining divergence starts in launch 26 (the first launch of the timed call) at the
  first **transfer-engine** beat on the port (`td.load` reading its descriptor, then `t.ld/t.st`):
  the cycle core still runs the transfer engine as the M2 event-driven service, whose beats do
  not go through the replayed port and whose sequencing is not cycle-exact. That is the next
  piece of the first slice (tnca_tensor.v, CORE_TIMING.md section 4.2).
- `data` divergence at launch 1, beat 3 on every point: the RTL's memory returns 0 for a word the
  model returns as the value a previous launch stored. Not part of the coarse gate; recorded as an
  open question for the memory system (slice 2): is it a don't-care read, or a visibility rule the
  model does not have?
- The event-driven core, for comparison, diverges on every point from the 27th instruction (its
  context arbitration) and is 9.6% off in median cycles.

### Cycle core (core_model=1), replay of the RTL's LSU beats, tiny

Verdict: **diverged** (15 of 27 point(s) diverge). Points 27, aligned 12, diverged 15, missing 0; |cycle error| median 1.1%, p95 80.5%.

| point | launches equal | cycles RTL | cycles model | error | first issue divergence | first request divergence | order-only: issue | request | data |
|---|---:|---:|---:|---:|---|---|---|---|---|
| model.gpt2_block_tiny_fp16-d64-h4-b1-s16 | 46/155 | 462540 | 127233 | -72.49% | L30 #173 | L30 #24 | L30 #218 | L30 #24 | L1 #3 |
| model.llama_decode_tiny_fp16-b1-p16 | 27/163 | 616411 | 18022 | -97.08% | L27 #88 | L27 #21 | L27 #90 | L27 #21 | L1 #3 |
| model.llama_prefill_tiny_fp16-b1-s16 | 58/148 | 503323 | 268957 | -46.56% | L30 #165 | L30 #22 | L30 #204 | L30 #22 | L1 #3 |
| model.mlp_tiny_fp16-t16-d64 | 26/46 | 337256 | 340905 | +1.08% | L26 #167 | L26 #24 | L26 #206 | L26 #24 | L1 #3 |
| model.resnet_block_tiny_fp32-n1-c8-hw8 | 49/54 | 859327 | 963777 | +12.15% | L26 #80 | L26 #19 | L26 #82 | L26 #19 | L1 #3 |
| op.add_tiny_fp16-n1000 | 58/58 | 24954 | 24954 | +0.00% | equal | equal | equal | equal | L1 #3 |
| op.add_tiny_fp32-n256 | 34/34 | 15198 | 15198 | +0.00% | equal | equal | equal | equal | L1 #3 |
| op.argmax_tiny_fp16-r2-v256 | 28/28 | 17542 | 17542 | +0.00% | equal | equal | equal | equal | L1 #3 |
| op.attention_decode_tiny_fp16-b1-h2-l16-d16 | 42/76 | 54448 | 76411 | +40.34% | L34 #78 | L34 #23 | L34 #80 | L34 #23 | L1 #3 |
| op.attention_prefill_tiny_fp16-b1-h2-s16-d16 | 36/118 | 152454 | 32472 | -78.70% | L36 #52 | L36 #11 | L38 #616 | L36 #11 | L1 #3 |
| op.bmm_tiny_fp16-b2-m16-k16-n16 | 26/30 | 20112 | 21625 | +7.52% | L26 #52 | L26 #11 | equal | L26 #11 | L1 #3 |
| op.conv2d_tiny_fp32-n1-cin4-h8-w8-cout8 | 34/38 | 444430 | 449653 | +1.18% | L26 #78 | L26 #19 | L26 #80 | L26 #19 | L1 #3 |
| op.cumsum_tiny_fp32-r4-c64 | 27/28 | 16462 | 16462 | +0.00% | L26 #1141 | equal | equal | equal | L1 #3 |
| op.embedding_tiny_fp16-v256-d64-t16 | 58/58 | 117596 | 117596 | +0.00% | equal | equal | equal | equal | L1 #3 |
| op.gelu_tanh_tiny_fp16-n256 | 34/34 | 16999 | 16999 | +0.00% | equal | equal | equal | equal | L1 #3 |
| op.gemv_tiny_fp16-k64-n64 | 26/30 | 57271 | 11162 | -80.51% | L26 #88 | L26 #21 | L26 #90 | L26 #21 | L1 #3 |
| op.layernorm_tiny_fp16-r4-d64 | 28/28 | 18276 | 18276 | +0.00% | equal | equal | equal | equal | L1 #3 |
| op.linear_tiny_fp16-m16-k64-n64 | 26/30 | 47628 | 48996 | +2.87% | L26 #167 | L26 #24 | L26 #206 | L26 #24 | L1 #3 |
| op.matmul_tiny_fp16-m16-k32-n16 | 26/28 | 17308 | 18312 | +5.80% | L26 #52 | L26 #11 | equal | L26 #11 | L1 #3 |
| op.matmul_tiny_fp32-m16-k32-n16 | 26/28 | 25981 | 28347 | +9.11% | L26 #52 | L26 #11 | equal | L26 #11 | L1 #3 |
| op.rmsnorm_tiny_fp16-r4-d64 | 28/28 | 16868 | 16868 | +0.00% | equal | equal | equal | equal | L1 #3 |
| op.rope_tiny_fp16-b1-h2-s16-d16 | 58/58 | 29320 | 29320 | +0.00% | equal | equal | equal | equal | L1 #3 |
| op.silu_tiny_fp16-n256 | 34/34 | 15783 | 15783 | +0.00% | equal | equal | equal | equal | L1 #3 |
| op.softmax_tiny_fp16-r4-c64 | 26/30 | 21046 | 11132 | -47.11% | L26 #725 | L26 #150 | L26 #949 | L26 #150 | L1 #3 |
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
