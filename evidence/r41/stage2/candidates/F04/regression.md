# F04 regression and cycle impact (GB10, iverilog 12)

Tree: r41s2 + F01 + F02 + this candidate. The targets are F02's: `core`, `core-vrf`, `core-r39sw4`,
`core-r39sw5` (each on both memories), `top`, and `r41-core-boundary` (`t41_boundary` on 15 benches × 1/2/all
contexts). The full 11-target set ran again on the final tree: F05 `regression.md`.

| Target | Cases | PASS | Base cycles | Candidate cycles | Changed cases | Largest changes |
|---|---:|---:|---:|---:|---:|---|
| `core` | 40 | 40 | 94222 | 94222 | 0 | — |
| `core-vrf` | 40 | 40 | 103247 | 103247 | 0 | — |
| `core-r39sw4` | 80 | 80 | 236129 | 235989 | 2 | t39_switch (r39 4 rctx, ooo mem) 4568→4480 (-1.9 %); t39_switch (r39 4 rctx) 3554→3502 (-1.5 %) |
| `core-r39sw5` | 86 | 86 | 246160 | 246020 | 2 | t39_switch (r39 v5, ooo mem) 4568→4480 (-1.9 %); t39_switch (r39 v5) 3554→3502 (-1.5 %) |
| `top` | 4 | 4 | 51983 | 51983 | 0 | — |
| `r41-core-boundary` | 45 | 45 | — | — | 0 | — |

All cases PASS. The new bench assertion compares `pa_data`, `pb_data` and `lane_mask` with the old X-stage
predicate reads on every valid X cycle, and it never fired.

## Cycle impact

None. Every case has the same cycle count as with F01 alone, compared case by case. The only differences from the
baseline are F01's `t39_switch` ones.

## Coverage

`fwd_p` counts valid X cycles that used a forwarded predicate operand or lane mask. The total is 127 over the 45
`t41_boundary` runs, nearly all in the single-context runs (section B: a mask produced by the instruction just
ahead). Multi-context forwarding conflicts are also exercised: every context writes and reads the same predicate
indices concurrently, the forward requires `nw_rctx == r_rctx`, and the assertion would catch a cross-context
forward.
