# Interface note: m.mma.dp.async and the operand-capture path (G2 <-> timing), final for run 2

Status 2026-10-29: Kimi's matrix VRF response register (`MTX_VRF_REG`, `b87a0cc`) and G2's shared MAC lane are both on `r41-hbm-bw` and agree. G2 owns the integration while Kimi is out of quota.

## What is where
- **Capture path, as Kimi left it.** `tnca_core` registers the banked file's matrix response (`g_mtx_vrf_reg`: `data_q`, `valid_q`) before `tnca_matrix`. `tnca_matrix` consumes `vrf_data` only in `S_LOADA`, `S_LOADB` and the A read of `S_MAC`, each qualified by `vrf_valid`, so a response one cycle later changes cycle counts and nothing else. G2 added nothing in front of this.
- **After `vrf_data` (G2, `MAC_DP=1` only; every line is constant-folded away at `MAC_DP=0`).**
  - B row: `b_row` is `VLEN*32` bits. `S_LOADB` fills each lane's word from `b_dp[i]` (m.mma.dp: G elements, one from each of G consecutive B rows: two 16-bit rows of one half-register, two 8-bit rows, or four E4M3 rows = one whole register) or `b_nd[i]` (m.mma: one element). Select: `r_dp`, a register set in `S_IDLE`.
  - A word: `a_dp_v` picks an aligned 32-bit word (16-bit elements, four E4M3 bytes) or two bytes (G = 2, 8-bit) from `a_word` (ABUF or the registered response). Same select bits as `a_elem_v`, one mux level more.
  - K step: `kstep` = 4 (E4M3 x E4M3), 2 (other m.mma.dp pairs), 1 (m.mma); one B load per step.
- **Ports and parameters.** `tnca_matrix`: `cmd_dp`, `cmd_bdt[1:0]`; `MAC_DP`, `MAC_CUTS`, `FMA_IMPL`. `tnca_core`: `x_is_mma_any`, `mmadp_unsupported` (ILLEGAL_INSTRUCTION without `MAC_DP`), CAPABILITY0 bit 12. Image parameters `FMA_IMPL`, `MAC_DP`, `MAC_CUTS` are plumbed from `tnca_multicore_hbm_xo` to the core.
- **Final cut set `MAC_CUTS = 29` (0b11101), result latency 5** (operand register + 4): cuts after S1 (decode, products), after S2+F2 (DP2 shift/add, |sum|, RO_48), after B0 (align with C) and after norm; round goes into the consumer's register. Unit screens (10 ns, SLR0): S1 6.17 ns, S2+F2 6.45, B0 4.9, norm 4.6, round 5.2. `tnca_matrix` derives `PIPE = OPND_REG + popcount(MAC_CUTS)`; the tag pipeline, the drain and `src_free` follow from it, and nothing outside the matrix assumes the old latency (completion is by `done`). A row returns every 16 cycles, so latency up to 15 does not stall the sweep.

## Checked together (run-2 parameter set of `g4/run2/image.tcl` at `35513d2d`)
`make core-run2`: g2c4's four contexts, `MTX_VRF_REG 1`, `MTX_ABUF 1`, `FMA_IMPL 1`, `MAC_DP 1`, `MAC_CUTS 29`, in-order and slow reordered memory: core-g2c4's program list (m.mma through the new lane, all four formats) plus `t41_mmadp` (m.mma.dp, 15 format/K combinations including E4M3 K32 through the A buffer and K64 without it, 16-bit K16 with it and K32 without). Results: `evidence/r41/hbm_bw/fma_dslx/logs/regression/`.

## For the timing owner
If a screen shows a lane stage critical, tell G2 which and the cut moves (`tnca_mac_lane.v` `CUTS`; any 5-bit value is functionally correct, tested at 0, 18, 29, 31).
