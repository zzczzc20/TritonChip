# F02 regression and cycle impact (GB10, iverilog 12)

Tree: r41s2 + F01 + this candidate. The F01 commit has the full 11-target core regression. F02 ran the
targets that cover its change on every scalar-operand path:
- the behavioural and banked VRF benches (`core`, `core-vrf`);
- the image configuration and its 4-context sibling (`core-r39sw5`, `core-r39sw4`), each on both memories;
- the 2×2 system (`top`);
- `r41-core-boundary`: `t41_boundary` on 15 benches × 1/2/all contexts.

The full 11-target set ran again on the final tree (F01+F02+F04+F05): F05 `regression.md`.

| Target | Cases | PASS | Base cycles | Candidate cycles | Changed cases | Largest changes |
|---|---:|---:|---:|---:|---:|---|
| `core` | 40 | 40 | 94222 | 94222 | 0 | — |
| `core-vrf` | 40 | 40 | 103247 | 103247 | 0 | — |
| `core-r39sw4` | 80 | 80 | 236129 | 235989 | 2 | t39_switch (r39 4 rctx, ooo mem) 4568→4480 (-1.9 %); t39_switch (r39 4 rctx) 3554→3502 (-1.5 %) |
| `core-r39sw5` | 86 | 86 | 246160 | 246020 | 2 | t39_switch (r39 v5, ooo mem) 4568→4480 (-1.9 %); t39_switch (r39 v5) 3554→3502 (-1.5 %) |
| `top` | 4 | 4 | 51983 | 51983 | 0 | — |
| `r41-core-boundary` | 45 | 45 | — | — | 0 | — |

All cases PASS. In particular the new bench assertion passed on every valid X cycle of every case. It compares
`sa/sb/sc_data` with an independent recomputation of the old X-stage read.

## Cycle impact

None. Every case has the same cycle count as with F01 alone, compared case by case on all five targets above and
all 45 `t41_boundary` runs. The only differences from the baseline are F01's `t39_switch` ones.

## Coverage

`fwd_s` counts valid X cycles that used a scalar operand forwarded from W. Over the 45 `t41_boundary` runs the
total is 2,368: 82–86 per single-context run, where every instruction directly follows its producer, and fewer
when contexts interleave.
