# Interface note: m.mma.dp.async and the operand-capture path (G2 -> timing, 2026-10-29)

Pushed in `tnca_matrix` (`MAC_DP`, default 0; nothing changes with it clear) and the core, so the timing owner can see what moved:
- **Capture path untouched.** `vrf_data`/`vrf_valid`/`vrf_req`, `MTX_VRF_REG` and the banked-file arbitration are as before. All new logic is after `vrf_data` lands in `S_LOADB`/`S_MAC` (the A word) and in the lane.
- **B operand, 32 bits per lane.** `b_row` is `VLEN*32` when `MAC_DP=1` (`VLEN*16` otherwise). `S_LOADB` loads each lane's word from `vrf_data` through `b_dp[i]` (G elements from G consecutive B rows) or `b_nd[i]` (the old single element); the select is `r_dp` (a register set in `S_IDLE`). The widening Kimi was asked to take is done here; if timing needs `vrf_data` registered again before `b_dp`/`b_nd`, wrap them, the arithmetic does not care.
- **A word.** `a_dp_v` selects one aligned 32-bit word (or two bytes) from `a_word` by `a_elem`; same inputs and same select bits as `a_elem_v`, one more level.
- **New ports of `tnca_matrix`:** `cmd_dp`, `cmd_bdt[1:0]`; params `MAC_DP`, `MAC_CUTS`. New core wires: `x_is_mma_any`, `mmadp_unsupported`.
- **Latency.** Result latency = `OPND_REG + popcount(MAC_CUTS)` (4 with cuts 0b1110, vs 3); `PIPE` follows it; nothing outside `tnca_matrix` assumes the old value (completion is by `done`).
- Please keep the lane's own register placement (`tnca_mac_lane.v CUTS`) in G2's hands; tell me which stage is critical in your screens and I move a cut.
