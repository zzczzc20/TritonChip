
### matrix engine (shrunk tile s3, 5.6 ns, routed OOC; failing / endpoints below 1.5 ns, worst slack, levels)

| failing | of | worst slack | levels | start -> end |
|--:|--:|--:|--:|---|
| 953 | 986 | -1.337 | 23 | g_c0.q_reg[] -> g_c2.q_reg[] |
| 517 | 6461 | -0.799 | 7 | direct_w_reg -> u_acc/valid_reg[][] |
| 485 | 880 | -1.367 | 22 | g_in_reg.a_q_reg[] -> g_c0.q_reg[] |
| 395 | 512 | -0.767 | 12 | u_srf/valid_reg[] -> r_vdata_reg[] |
| 325 | 342 | -1.317 | 15 | w_sdata_reg[] -> g_in_reg.a_q_reg[] |
| 267 | 1221 | -1.305 | 22 | g_in_reg.b_q_reg[] -> g_c0.q_reg[] |
| 68 | 73 | -1.324 | 15 | r_k_reg[] -> g_in_reg.a_q_reg[] |
| 66 | 74 | -1.301 | 15 | mm_reg[]__0 -> g_in_reg.a_q_reg[] |
| 56 | 480 | -0.336 | 10 | w_sdata_reg[] -> g_c0.kq_reg[]_srl2 |
| 25 | 32 | -0.384 | 12 | u_srf/valid_reg[] -> r_scale_reg[] |
| 18 | 23 | -1.194 | 13 | x_sb_fw_reg -> g_in_reg.a_q_reg[] |
| 14 | 16 | -0.801 | 12 | g_c4.q_reg[] -> u_acc/g_rd[].lane_mem_reg_192_255_28_31/RAMB |
| 14 | 16 | -0.792 | 12 | g_c4.q_reg[] -> u_acc/g_rd[].lane_mem_reg_128_191_28_31/RAMC |
| 14 | 16 | -0.737 | 12 | g_c4.q_reg[] -> u_acc/g_rd[].lane_mem_reg_128_191_28_31/RAMB |

### vector FPU (shrunk tile s3, 5.6 ns, routed OOC; failing / endpoints below 1.5 ns, worst slack, levels)

| failing | of | worst slack | levels | start -> end |
|--:|--:|--:|--:|---|
| 904 | 1342 | -1.390 | 17 | g_vrf_banked.x_vb_f_reg[] -> g_fma[].u_fma/g_l2x.u_fma_x/g_c1.q_reg[] |
| 357 | 580 | -1.356 | 17 | x_fpctl_reg[] -> g_fma[].u_fma/g_l2x.u_fma_x/g_c1.q_reg[] |
| 245 | 298 | -1.365 | 21 | cvt_x_reg[][] -> dly_v_reg[][] |
| 76 | 139 | -0.931 | 15 | cvt_fn_reg[] -> dly_v_reg[][] |
| 76 | 365 | -0.858 | 20 | g_fma[].u_fma/g_l2x.u_fma_x/g_c1.q_reg[] -> g_fma[].u_fma/g_l2x.u_fma_x/g_c3.q_reg[] |
| 72 | 72 | -1.354 | 21 | cvt_mod_reg[] -> dly_v_reg[][] |
| 58 | 157 | -1.088 | 16 | red_busy_reg -> g_fma[].u_fma/g_l2x.u_fma_x/g_c1.q_reg[] |
| 39 | 226 | -0.949 | 14 | g_fma[].u_fma/g_l2x.u_fma_x/g_c3.q_reg[] -> acc_reg[][] |
| 32 | 32 | -0.858 | 16 | g_sfu.u_sfu/g_slice[].u_serial/cnt_reg[] -> g_sfu.u_sfu/g_slice[].u_serial/u_exp2/p2_i_reg[]_srl3___p2_eq_361_reg_r |
| 22 | 22 | -1.172 | 11 | g_sfu.u_sfu/g_slice[].u_serial/cnt_reg[] -> g_sfu.u_sfu/g_slice[].u_serial/u_exp2/smul53b_30b_x_23b_return__0/DSP_A_B_DATA_INST |
| 21 | 22 | -1.156 | 11 | g_sfu.u_sfu/g_slice[].u_serial/cnt_reg[] -> g_sfu.u_sfu/g_slice[].u_serial/u_exp2/smul53b_30b_x_23b_return/DSP_A_B_DATA_INST |
| 21 | 22 | -0.886 | 11 | g_sfu.u_sfu/g_slice[].u_serial/cnt_reg[] -> g_sfu.u_sfu/g_slice[].u_serial/u_exp2/p0_bit_slice_611_reg[] |
| 16 | 16 | -1.246 | 12 | g_sfu.u_sfu/g_slice[].u_serial/cnt_reg[] -> g_sfu.u_sfu/g_slice[].u_serial/u_exp2/p0_bit_slice_606_reg_rep_rep |
| 14 | 461 | -0.111 | 5 | x_func_reg[] -> acc_reg[][] |

### transfer engine (shrunk tile s3, 5.6 ns, routed OOC; failing / endpoints below 1.5 ns, worst slack, levels)

| failing | of | worst slack | levels | start -> end |
|--:|--:|--:|--:|---|
| 921 | 939 | -1.075 | 8 | vw_rctx_reg[] -> vw_data_reg[] |
| 140 | 321 | -0.362 | 13 | t_unit_reg[] -> sl_dlo_reg[][] |
| 127 | 327 | -0.531 | 13 | t_unit_reg[] -> sl_dhi_reg[][] |
| 109 | 125 | -1.182 | 16 | t_unit_reg[] -> st_prev_reg[] |
| 84 | 84 | -1.277 | 10 | sl_nlo_reg[] -> vw_data_reg[] |
| 80 | 80 | -1.230 | 13 | t_c0_reg[] -> cm_cnt_reg[][] |
| 68 | 114 | -0.424 | 5 | g_host_mux.u_host/host_busy_reg_rep -> sl_dlo_reg[][] |
| 64 | 64 | -1.100 | 17 | t_unit_reg[] -> t_P0_reg[] |
| 64 | 64 | -0.811 | 17 | t_unit_reg[] -> t_i0_reg[] |
| 64 | 64 | -0.796 | 17 | t_unit_reg[] -> t_i3_reg[] |
| 64 | 64 | -0.596 | 17 | t_unit_reg[] -> t_ps_reg[] |
| 64 | 64 | -0.591 | 17 | t_unit_reg[] -> t_P3_reg[] |
| 64 | 64 | -0.588 | 17 | t_unit_reg[] -> t_P2_reg[] |
| 64 | 64 | -0.499 | 17 | t_unit_reg[] -> t_P1_reg[] |

### barrier (shrunk tile s3, 5.6 ns, routed OOC; failing / endpoints below 1.5 ns, worst slack, levels)

| failing | of | worst slack | levels | start -> end |
|--:|--:|--:|--:|---|
| 608 | 608 | -1.326 | 20 | u_srf/valid_reg[] -> done_epoch_reg[][] |
| 442 | 607 | -0.880 | 15 | u_srf/valid_reg[] -> cur_epoch_reg[][] |
| 152 | 152 | -0.905 | 10 | w_sdata_reg[] -> slp_epoch_reg[][] |
| 123 | 128 | -0.814 | 17 | u_srf/valid_reg[] -> n_ops_reg[][] |
| 101 | 128 | -1.160 | 16 | u_srf/valid_reg[] -> n_arrive_reg[][] |
| 12 | 12 | -0.850 | 10 | w_sdata_reg[] -> slp_bar_reg[][] |
| 8 | 8 | -0.727 | 16 | u_srf/valid_reg[] -> armed_reg[] |
| 8 | 8 | -0.570 | 18 | u_srf/valid_reg[] -> err_reg[] |
| 6 | 6 | -1.333 | 18 | u_srf/valid_reg[] -> fault_bar_reg[] |
| 4 | 4 | -0.917 | 20 | u_srf/valid_reg[] -> slp_v_reg[] |
| 4 | 4 | -0.718 | 19 | u_srf/valid_reg[] -> wake_reg[] |
| 1 | 1 | -1.313 | 18 | u_srf/valid_reg[] -> fault_reg |
