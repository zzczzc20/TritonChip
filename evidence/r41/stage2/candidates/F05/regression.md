# F05 regression and cycle impact (GB10, iverilog 12)

Tree: r41s2 + F01 + F02 + F04 + this candidate. This is the final tree of the four candidates. It ran the full
11-target core set that F01 ran: every core bench, `top`, `r38-xpose-rtl`, and `r41-core-boundary`
(`t41_boundary` on 15 benches × 1/2/all contexts). The wider suites run on this tree are listed in
`final_tree_suites.md`.

| Target | Cases | PASS | Base cycles | Candidate cycles | Changed cases | Largest changes |
|---|---:|---:|---:|---:|---:|---|
| `core` | 40 | 40 | 94222 | 94222 | 0 | — |
| `core-slowmem` | 40 | 40 | 119246 | 119246 | 0 | — |
| `core-ooo` | 40 | 40 | 127253 | 127253 | 0 | — |
| `core-vrf` | 40 | 40 | 103247 | 103247 | 0 | — |
| `core-r39hw` | 80 | 80 | 229529 | 229529 | 0 | — |
| `core-r39sw` | 80 | 80 | 225666 | 225681 | 2 | t39_switch (r39 switch, ooo mem) 2780→2794 (+0.5 %); t39_switch (r39 switch) 2090→2091 (+0.0 %) |
| `core-r39sw4` | 80 | 80 | 236129 | 235989 | 2 | t39_switch (r39 4 rctx, ooo mem) 4568→4480 (-1.9 %); t39_switch (r39 4 rctx) 3554→3502 (-1.5 %) |
| `core-r39sw5` | 86 | 86 | 246160 | 246020 | 2 | t39_switch (r39 v5, ooo mem) 4568→4480 (-1.9 %); t39_switch (r39 v5) 3554→3502 (-1.5 %) |
| `top` | 4 | 4 | 51983 | 51983 | 0 | — |
| `r38-xpose-rtl` | 3 | 3 | 3985 | 3985 | 0 | — |
| `r41-core-boundary` | 45 | 45 | — | — | 0 | — |

All cases PASS. The new bench assertions never fired on any cycle of any case. They check that `dsb_v`/`rsb_v`
equal the scoreboard lookups and that `d_pf_v*_phys`/`r_v*_phys` equal the `vbase + idx` adds. The F02 and F04
assertions also stayed silent.

## Cycle impact

None from F05. Every case of every target has the same cycle count as with F01 alone, compared case by case. All
45 `t41_boundary` runs are also identical. Against the baseline, the four candidates together change only F01's
`t39_switch` counts.

## Coverage of the busy-bit corner cases (`+r41cov`)

The table counts the cases where the registered bit differs from a plain copy of the scoreboard's current value:
- `dsb_clm` / `dsb_rel`: the word entering D had a source claimed or released on that same edge;
- `rsb_clm` / `rsb_rel`: the same for the instruction entering R;
- `hold_rel`: a held source was released while it waited.

| `t41_boundary` run | `dsb_clm` | `dsb_rel` | `rsb_clm` | `rsb_rel` | `hold_rel` |
|---|---:|---:|---:|---:|---:|
| `tb_core`, 1 context | 10 | 11 | 13 | 7 | 1 |
| `tb_core_vrf`, 1 context | 10 | 7 | 13 | 2 | 19 |
| `tb_core_r39sw5`, 1 context | 6 | 6 | 19 | 7 | 19 |
| `tb_core_r39sw5`, 4 contexts | 4 | 37 | 0 | 4 | 7 |
| `tb_core_r39sw5_ooo`, 4 contexts | 4 | 35 | 0 | 10 | 10 |
| total over all 45 runs | 876 | 502 | 307 | 339 | 574 |
