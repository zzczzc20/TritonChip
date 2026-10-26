# Attribution of the R40 v3 performance misses (diagnosis, not gate records)

Each comparison takes the board configuration pair from the fe1b campaign
(`../campaign_fe1b/campaign.json`, exact knobs) and runs it through the
SystemC/TLM model. The model is binary `tnca-chip` 2402d46d… with
`configs/r40-board-m2.cfg` dc2c1763…, at M2, **uncalibrated** (holdout median
21.5 %), on the delivery tree fe1b. Its counters say where the time goes. The
board's own per-launch records (`../campaign_fe1b/judge/*.jsonl`) give the
device cycles.

A model result is used for attribution only when it reproduces the board's
direction, and the tables say whether it does. Raw summaries are in
`runs.tgz`, the counter differences in `counters_diff.txt`, and the GEMM-tile
pair in `../R40-03.gemm-tile/model_diagnosis/`.

| Point (experiment) | Board control → candidate | Board ratio | Model control → candidate | Model ratio | Reproduced |
|---|---|---:|---|---:|---|
| op.add/medium (wide-blocks 16→64) | 516,130 → 326,351 | 1.58x | 392,976 → 255,170 | 1.54x | yes |
| op.rope/medium (wide-blocks), device serial cycles | 797,647 → 592,163 | 1.35x | 645,391 → 509,366 | 1.27x | yes (device) |
| op.rope/medium (wide-blocks), graph makespan | 8,832,803 → 9,376,336 | 0.94x | — | — | host gaps; not modelled |
| op.layernorm/medium (wide-blocks) | 802,950 → 836,277 | 0.96x | 698,365 → 606,190 | 1.15x | **no** |
| op.copy/hbm (ports 1→3, B40-N) | 162,406,130 → 162,797,381 | 1.00x | 119,376,863 → 119,230,183 | 1.00x | yes |
| op.gemv/large (pipeline 1→2, roles 2) | 93,296,498 → 79,387,257 | 1.18x | 86,955,297 → 80,616,871 | 1.08x | yes (smaller) |
| op.matmul/large fp16 (pipeline 1→2) | 1,402,886,765 → 1,637,209,377 | 0.86x | 1,360,744,319 → 1,340,266,729 | 1.02x | **no** |
| op.linear/medium (gemm-tile 16,16,16→16,32,16) | 1,949,824 → 11,875,930 | 0.16x | 1,826,404 → 10,785,379 | 0.17x | yes |

## Findings

**Ports (R40-06.ports).**
- With one outlet, the outlet is busy for 14.7 M of the 119 M cycles of
  `op.copy/hbm`, about 12 %.
- Three outlets split that busy time three ways and change nothing.
- The home-slice engine queue (359 M context-cycles) and the cores' dependency
  waits are unchanged.

So the streaming limit is the NUCA home-slice engine and the per-core memory
path, not the outlets. This agrees with R40-06.streaming-cores, which gains
2.56x by adding cores.

**Wide blocks (R40-03.wide-blocks).**
- Streaming points gain 1.5-1.6x, and the model agrees.
- rope's kernels are 26 % faster on the device. Its graph makespan is 11x its
  device cycles: under the B40-S per-kernel transfer policy, host staging gaps
  sit between its three launches. The 0.94x is in those gaps, not in the
  device.
- layernorm is a single launch that is really 4 % slower on the device. The
  model does not reproduce this; its load/store path is the known M2 gap. The
  model's direction is the opposite (fewer instructions, more LSU wait), so
  this point is attributed only as far as "the LSU/memory path at 1 outstanding
  line".

**Pipeline (R40-07.long-k).**
- gemv gains 18 % on the board and 8 % in the model.
- In the model, both gemv and matmul trade barrier-sleep cycles (overlap
  gained: −35 % on matmul) against transfer-engine admission waits (+109 % on
  matmul). The transfer engine takes one command at a time, so the producer
  role's prefetch serialises with the other transfers.
- On the board, matmul loses 14 %. The model nets this to about zero, so it
  underestimates the loss. The direction of the trade-off is the useful
  finding.
- Pipelining is a win only where overlap exceeds admission contention, which
  is a per-point policy decision (R40-07.policy / TNCA_TUNE, deferred to R41).

**GEMM tile (R40-03.gemm-tile).** See `../R40-03.gemm-tile/model_diagnosis/`.
A non-16x16 accumulator is lowered with row-wise accumulator traffic, giving
12x the instructions and 48x the matrix-engine commands.
