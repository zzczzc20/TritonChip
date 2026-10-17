# Native Torch DMA/MMA service overlap

Source: `0f4639d` plus the trace-audit patch committed with this evidence.
The exact runner used is `runner_snapshot.py`; the final product runner adds
an earlier refusal for unsupported trace backends. Compiler/RTL are unchanged
from `0f4639d`. ARM jump host, locked Torch/Triton, Icarus 12, isolated sim-file
binding, core-vrf-sync-v5. Trace switches: `+engtrace +trace`.

The standard fullgraph native Torch mm/addmm runner still checks CPU reference
numerics and zero steady tensor staging. Runtime command history selects the
measured launches; engine trace is grouped by the same mailbox sequence.
Matched issue/completion intervals count transfer service and matrix op=1 only.
Descriptor configuration, accumulator zero/row ops, warmup and unrelated
launches cannot inflate the result. The parser refuses missing/incomplete,
faulted, duplicate or inconsistent command traces.

Stage 3, four measured launches:

| Torch graph / shape MxNxK | Sequence | Stage 2 overlap | Stage 3 overlap |
|---|---:|---:|---:|
| mm / 16x16x48 | 116 | 294 | 292 |
| mm / 17x31x65 | 128 | 342 | 540 |
| addmm / 16x16x48 | 140 | 294 | 292 |
| addmm / 17x31x65 | 152 | 342 | 540 |

All four numerical/ring/staging checks pass in each mode; total overlap is
1272/1664 cycles for stage 2/3. These counts are not speedups.
Both `stage2/` and `stage3/` retain `engine.log`, `overlap.json`, `run.log`
and the SHA-named `.tnca` artifacts.
`engine_overlap_tests.py` passes 4/4, including warmup exclusion and refusal
of malformed evidence. Raw trace can regenerate the selected interval report.

This is standard-path functional overlap evidence on deterministic RTL, not
seconds or a performance sign-off. Adverse HBM/fault/stage policy coverage,
frozen long-K and matrix A/B, board timing and LOO remain required for R40-07.
