# F01 regression and cycle impact (GB10, iverilog 12)

Tree: r41s2 at 544aed8 plus this candidate (`rtl/core/tnca_core.v`, `tb/core/tb_tnca_core.v`,
`tb/core/progs/t41_boundary.s`, `Makefile`). Baseline: the same targets on 544aed8, the same host and the same day.

## Every core bench

`make <target>` for each target below: the core programs (`CORE_PROGS`, now including `t41_boundary`). The
bench variants are:
- `core-slowmem`: a slow, back-pressuring memory;
- `core-ooo`: out-of-order returns;
- `core-vrf`: the banked VRF;
- `core-r39hw`/`r39sw`/`r39sw4`/`r39sw5`: the R39 hardware switches. `r39sw5` is the B40 image's core
  configuration: 4 contexts, RCTX_SWITCH 2, RPAIR, RSLICE, MTX_ABUF, LSU_INFLIGHT 2, FPU_OVERLAP and VRF_PREFETCH.
  Each runs on the bench memory and on the reordering one.

Also run: `top` (2×2 system with mesh and NUCA), `r38-xpose-rtl`, and `r41-core-boundary` (new, below). Base
cycles sum only the programs that exist in both runs, so the new `t41_boundary` is left out of the base column.

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

All targets return 0, and every case prints PASS. No bench assertion fired: the core's `FAIL tnca_core assert`
lines would fail the case.

## Cycle impact

- **No change on any bench without RCTX_SWITCH.** The give-way rule is the only behavioural change, and the
  load/store claim change acts only behind a faulting access.
- **With RCTX_SWITCH, only `t39_switch` moves:**
  - −52 / −88 cycles (−1.5 % / −1.9 %) on the image configuration (`r39sw4`, `r39sw5`);
  - +1 / +14 cycles (+0.0 % / +0.5 %) on the two-context `r39sw`, bench memory / reordering memory.
- The give-way rule was chosen by measurement. Three versions were run on `core-r39sw4` and `core-r39sw5`:

| `t39_switch`, `r39sw5` | Baseline | E: give way regardless, redirect wins | B: X same context excluded | **Bp: any `x_may_redirect` excluded (committed)** |
|---|---:|---:|---:|---:|
| bench memory | 3554 | 3586 | 3586 | **3502** |
| reordering memory | 4568 | 4485 | 4485 | **4480** |
| `t18_squash`, reordering | 1912 | 1936 | 1936 | **1912** |
| all 86 cases, total | 246160 | 246133 | 246133 | **246020** |

  Bp is also the simplest to argue: a give-way and a redirect still never share a cycle.

## Directed test: `t41_boundary` and `make r41-core-boundary`

`tb/core/progs/t41_boundary.s` runs on every context. Each value depends on the role, so a value from another
context, a lost or repeated instruction, or a stale operand changes an asserted result. Its sections:
- **A:** scalar operands A, B and C produced by the instruction just ahead, by the MUL pipe and by a load. Also S0
  and a branch on a fresh value.
- **B:** a predicate mask produced just ahead. Covers masked moves, a negated mask, `p.and`/`p.ballot` and masked
  reductions.
- **C:** dependent vector chains through the scoreboard with D's prefetch, a WAW, and an FP-pipe chain.
- **D:** a vector store held in R with its operand collected while D prefetches, then read back.
- **E:** taken branches squashing younger vector work.
- **H:** a load-use stall right behind a taken branch, so that a give-way meets a redirect.
- **G:** an `mb.arm`/`mb.arrive`/`mb.wait` ping-pong between the two roles of a pair over 6 rounds. The wait token
  is built by the instruction just before the wait. `mb.test` follows directly on `mb.arm`'s token.

`make r41-core-boundary` runs it on all 15 core benches with `+roles=1`, `+roles=2` and `+allroles` (45 runs, all
PASS). With `+r41cov` it prints how often each corner case occurred, for example on the image configuration
(`tb_core_r39sw5[_ooo]`):

| | `+roles=1` | `+roles=2` | `+allroles` | `+allroles`, reordering memory |
|---|---:|---:|---:|---:|
| `sw_redir`: give-way candidate while X redirected (held, as before) | 0 | 19 | 0 | 11 |
| `sw_inhibit`: give-way candidate held behind a non-redirecting branch/atomic/wait (new) | 0 | 1 | 0 | 3 |
| `hold_ops`: R stalled holding collected vector operands | 7 | 12 | 33 | 51 |
| `pf_drop`: a D prefetch response dropped while R held | 28 | 19 | 90 | 106 |

`mem_claim`, an access in R behind a faulting atomic, did not occur in this program. The fault itself is covered
by `t23_amofault` (`+expectfault=03`), which passes on every bench.

`t41_boundary` passes on the baseline too, with the same cycle counts. It is a semantic test, not a test of this
change only.
