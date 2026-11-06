# F05: scoreboard `sb_v` → VRF bank `s1_vld` / `s1_byp_mask`

Census families (attempt 3, routed, 100 MHz, residual after M1–M3): `sb_v` → bank `s1_vld` D, `s1_byp_mask` R and
the bank BRAM read address. 291 endpoints, −94 ns, WNS −0.904, 20–24 levels.

The path, as the census and the baseline OOC both show it:
1. `sb_v` → a 256:1 lookup by R's (or D's) context and source index → `hz_v*` / `PF_HAZ` → `of_want` / `pf_go`.
2. The request mux → `vbase[rctx] + idx` (an adder) → bank decode → the 5-port round-robin arbitration → bank
   `raddr` (fanout 265) → `vld[raddr]` → `s1_vld`.

## Design

"Local valid bits and a capture boundary": the request a bank sees is built from registers. Changes are in
`rtl/core/tnca_core.v` and `rtl/core/tnca_vrf_banked.v`.

- **Local busy bits.** `dsb_v[2:0]` and `rsb_v[2:0]` are registers equal, every cycle, to `SB_V_BUSY` of D's
  three prefetch sources and R's three sources. `hz_v*` and `PF_HAZ` read them instead of the two 256:1 lookups.
  - They are written with the scoreboard's own next state, `sb_v' = claim | (sb_v & ~release)`. The claim and
    release are those of the same edge, under the same conditions as the scoreboard block: release first, claim
    wins, and a launch clears everything.
  - **F→D** (front end moves): the word entering D is looked up now, through its predecoded fields `pf_*_n`. These
    are the same fields `d_pf_*` latch.
  - **D→R:** R's sources are D's. `d_pf_v*_idx` is the decode of `d_insn`, so R takes D's bits with this edge's
    claim and release applied.
  - **Holding:** a stalled front end claims nothing, since a claim needs `!stall_front`. So a held bit can only be
    released.
  - A squashed or empty stage holds don't-care bits, and is refilled only through an advance, which recomputes
    them.
- **Physical registers a stage early.**
  - `d_pf_v*_phys` is set at F→D: `vbase_q[avail_rctx] + pf_v*_idx_n`. `r_v*_phys` copies it at D→R.
  - `vbase_q` changes only at a launch, and a launch empties D and R.
  - `tnca_vrf_banked` gets `PHYS_IN`. Ports with the bit set present `req_phys` and skip `phys_of`. The default 0
    elaborates the module exactly as before. The core sets `PHYS_IN = 5'b00111`, the three operand ports. The
    matrix and transfer ports still send context and index.
- **What the request path is now:** busy-bit register → `!WB_HIT` → hazard / `pf_go` → physical-register mux →
  arbitration → `raddr` → `s1_vld`.

**Why it is exact.** By induction on the update rules above: the bits and physical registers equal what the
lookups and adds returned, whenever D or R holds an instruction. Bench assertions check this every cycle, in every
bench.

**Cycle cost: none.** No stage is added, the hazard and prefetch decisions are identical, and the bank sees the
same requests.

**In-flight, replay, flush, fault and ownership.**
- Bank conflicts, arbitration order, the response routing and the B1 writeback-hold rule are untouched.
- In-flight operand responses and dropped prefetches are untouched. `of_arrive`, `pf_tag` and `adv_q` are as
  before.
- **Replay:** a give-way (`kill_r`) is either an advance edge (D's bits move up) or leaves R empty.
- **Flush:** a launch zeroes the bits along with the scoreboard. The squash rules are unchanged.
- **Faults and ownership:** unchanged.

## Verification

- **Bench assertions** (every cycle, every bench): for each of D's and R's three sources, whenever the stage is
  valid and the index names a register of the context:
  - `dsb_v`/`rsb_v` equal `SB_V_BUSY` of the scoreboard;
  - `d_pf_v*_phys`/`r_v*_phys` equal `vbase + idx`.
- **Coverage** (`+r41cov`):
  - `dsb_clm`/`dsb_rel`: the word entering D had a source claimed or released on that edge.
  - `rsb_clm`/`rsb_rel`: the same, for the instruction entering R.
  - `hold_rel`: a held source was released.
- Directed test section C (dependent vector chains, a WAW and an FP-pipe chain behind D's prefetch), plus D and E.
- Regression list, results and cycle tables: `regression.md`. OOC: `ooc.md`.
