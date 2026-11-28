| family (bit) | base fail@5.6 / >6.2 | T1_bits0124 fail@5.6 / >6.2 | T2_bits0to6_8 fail@5.6 / >6.2 |
|---|---|---|---|
| 0 PICK_REG: LSU fast done/rr_reg -> out_vector/out_scalar/out_mask | 584 / 570 | 0 / 0 | 0 / 0 |
| 1 FREE_WR: home -> ms_addr | 1856 / 1184 | 82 / 0 | 0 / 0 |
| 2 VRF write phys: w_rctx/w_idx -> vld_reg | 3641 / 781 | 288 / 0 | 2 / 0 |
| 3 WREP: SRF endpoints | 176 / 80 | 0 / 0 | 0 / 0 |
| 4 S2 split: home s1_addr/ms_state/wbq -> state arrays, s3_*, tag WE, URAM index | 3259 / 1670 | 970 / 137 | 192 / 8 |
| 5 ARB_SAME: home rq_addr/s1_addr -> g_tag read_tag | 384 / 196 | 384 / 192 | 50 / 0 |
| 6 FREE_LOAD: LSU command_reg -> lane_* | 2346 / 250 | 2675 / 1119 | 0 / 0 |
| 8 TOKR: x_sd_q_reg -> any | 0 / 0 | 3434 / 1132 | 0 / 0 |
