| unit | endpoints < 1.5 | failing at 5.60 | path > 6.2 | path > 6.67 | worst slack | baseline failing / > 6.2 / > 6.67 |
|---|--:|--:|--:|--:|--:|---|
| u_core/g_lsu_hybrid.u_lsu | 12180 | 710 | 566 | 441 | -1.312 | 3305 / 843 / 478 |
| g_home_pipe.u_home_pipe | 11933 | 7480 | 3708 | 483 | -1.300 | 7190 / 3715 / 595 |
| u_core/(core own) | 5189 | 2453 | 687 | 34 | -1.226 | 4311 / 1279 / 203 |
| u_core/u_txf | 9017 | 4155 | 531 | 2 | -1.095 | 3318 / 731 / 120 |
| u_core/g_vrf_banked.u_vrf | 4935 | 344 | 96 | 0 | -1.015 | 4060 / 1105 / 86 |
| u_core/u_valu | 2790 | 198 | 34 | 0 | -1.013 | 573 / 307 / 103 |
| u_core/u_bar | 1671 | 895 | 101 | 0 | -0.914 | 1469 / 688 / 66 |
| u_core/u_vfpu | 5819 | 963 | 122 | 0 | -0.880 | 1976 / 1505 / 563 |
| u_core/u_mtx | 13713 | 738 | 44 | 0 | -0.880 | 4342 / 1476 / 336 |
| g_host_mux.u_host | 4 | 4 | 0 | 0 | -0.401 | 4 / 4 / 0 |
| u_core/u_prf | 439 | 0 | 0 | 0 | 0.178 | 0 / 0 / 0 |
| u_core/u_mul | 520 | 0 | 0 | 0 | 0.300 | 479 / 0 / 0 |
| u_core/u_srf | 364 | 0 | 0 | 0 | 0.594 | 176 / 80 / 8 |
| **tile** | 68574 | 17940 | 5889 | 960 | | |

### matrix engine: failing families

| failing | of | worst slack | levels | start -> end |
|--:|--:|--:|--:|---|
| 192 | 287 | -0.694 | 10 | x_sb_fw_reg -> g_mac[].g_oreg.o_c_reg[] |
| 155 | 225 | -0.826 | 9 | w_sdata_reg[] -> g_mac[].g_oreg.o_c_reg[] |
| 147 | 228 | -0.880 | 10 | x_sb_fw_reg -> g_mac[].g_oreg.o_a_reg[] |
| 131 | 175 | -0.659 | 9 | w_sdata_reg[] -> g_mac[].g_oreg.o_a_reg[] |
| 55 | 97 | -0.622 | 10 | g_seq0.mm_reg[] -> g_mac[].g_oreg.o_a_reg[] |
| 12 | 12 | -0.240 | 14 | x_sd_q_reg[] -> g_seq0.kk_reg[] |
| 8 | 347 | -0.222 | 20 | g_c3.q_reg[] -> g_c4.q_reg[] |
| 6 | 6 | -0.240 | 14 | x_sd_q_reg[] -> g_seq0.kk_reg[]_rep__0 |
| 6 | 6 | -0.236 | 14 | x_sd_q_reg[] -> g_seq0.kk_reg[]_rep |
| 5 | 9 | -0.280 | 7 | g_seq0.kk_reg[]_rep -> g_mac[].g_oreg.o_a_reg[] |
| 4 | 4 | -0.324 | 14 | x_sd_q_reg[] -> g_seq0.la_reg[] |
| 3 | 3 | -0.234 | 14 | x_sd_q_reg[] -> g_seq0.kk_reg[]_rep__1 |

### vector FPU: failing families

| failing | of | worst slack | levels | start -> end |
|--:|--:|--:|--:|---|
| 405 | 669 | -0.880 | 12 | x_fpctl_reg[] -> g_fma[].g_fx.g_ok.u_fma_x/g_na.g_c0.pq_reg[] |
| 152 | 1285 | -0.657 | 13 | g_fma[].g_fx.g_ok.u_fma_x/g_na.g_c0.cq_reg[] -> g_fma[].g_fx.g_ok.u_fma_x/g_c1.q_reg[] |
| 139 | 748 | -0.403 | 7 | x_func_reg[] -> acc_reg[][] |
| 33 | 512 | -0.081 | 5 | x_func_reg[] -> g_sfu.u_sfu/g_slice[].u_serial/src_reg[][] |
| 32 | 177 | -0.831 | 4 | x_major_reg[] -> dly_v_reg[][] |
| 32 | 32 | -0.525 | 17 | g_sfu.u_sfu/g_slice[].u_serial/cnt_reg[] -> g_sfu.u_sfu/g_slice[].u_serial/u_exp2/p2_i_reg[]_srl3___p2_eq_361_reg_r |
| 31 | 83 | -0.584 | 12 | red_busy_reg -> g_fma[].g_fx.g_ok.u_fma_x/g_na.g_c0.pq_reg[] |
| 23 | 69 | -0.420 | 9 | g_vrf_banked.x_vb_f_reg[] -> dly_v_reg[][] |
| 21 | 22 | -0.486 | 12 | g_sfu.u_sfu/g_slice[].u_serial/cnt_reg[] -> g_sfu.u_sfu/g_slice[].u_serial/u_exp2/smul53b_30b_x_23b_return/DSP_A_B_DATA_INST |
| 21 | 22 | -0.469 | 12 | g_sfu.u_sfu/g_slice[].u_serial/cnt_reg[] -> g_sfu.u_sfu/g_slice[].u_serial/u_exp2/smul53b_30b_x_23b_return__0/DSP_A_B_DATA_INST |
| 17 | 22 | -0.400 | 12 | g_sfu.u_sfu/g_slice[].u_serial/cnt_reg[] -> g_sfu.u_sfu/g_slice[].u_serial/u_exp2/p0_bit_slice_611_reg[] |
| 16 | 16 | -0.635 | 13 | g_sfu.u_sfu/g_slice[].u_serial/cnt_reg[] -> g_sfu.u_sfu/g_slice[].u_serial/u_exp2/p0_bit_slice_606_reg_rep_rep |

### transfer engine: failing families

| failing | of | worst slack | levels | start -> end |
|--:|--:|--:|--:|---|
| 800 | 991 | -1.095 | 14 | t_unit_reg[] -> vw_data_reg[] |
| 315 | 512 | -0.498 | 14 | t_unit_reg[] -> sl_dlo_reg[][] |
| 272 | 448 | -0.413 | 14 | t_unit_reg[] -> sl_dhi_reg[][] |
| 138 | 512 | -0.804 | 8 | ls_v_reg[] -> sr_next_reg[] |
| 105 | 512 | -0.548 | 8 | ls_v_reg[] -> sr_data_reg[] |
| 100 | 112 | -0.833 | 14 | t_unit_reg[] -> st_prev_reg[] |
| 78 | 80 | -0.859 | 18 | t_unit_reg[] -> cm_cnt_reg[][] |
| 64 | 64 | -0.988 | 16 | t_unit_reg[] -> t_P2_reg[] |
| 64 | 64 | -0.987 | 16 | t_unit_reg[] -> t_P0_reg[] |
| 64 | 64 | -0.789 | 16 | t_unit_reg[] -> t_P1_reg[] |
| 64 | 64 | -0.767 | 13 | t_unit_reg[] -> g_line_port.xf_e_reg[][] |
| 64 | 64 | -0.660 | 2 | st_reg[] -> t_e0_reg[] |

### barrier: failing families

| failing | of | worst slack | levels | start -> end |
|--:|--:|--:|--:|---|
| 562 | 608 | -0.836 | 22 | x_sd_q_reg[] -> done_epoch_reg[][] |
| 172 | 608 | -0.394 | 17 | x_sd_q_reg[] -> cur_epoch_reg[][] |
| 89 | 128 | -0.522 | 19 | x_sd_q_reg[] -> n_ops_reg[][] |
| 51 | 128 | -0.552 | 18 | x_sd_q_reg[] -> n_arrive_reg[][] |
| 6 | 6 | -0.611 | 20 | x_sd_q_reg[] -> fault_bar_reg[] |
| 4 | 4 | -0.914 | 22 | x_sd_q_reg[] -> slp_v_reg[] |
| 4 | 4 | -0.761 | 21 | x_sd_q_reg[] -> wake_reg[] |
| 3 | 8 | -0.272 | 18 | x_sd_q_reg[] -> armed_reg[] |
| 3 | 8 | -0.178 | 20 | x_sd_q_reg[] -> err_reg[] |
| 1 | 1 | -0.541 | 20 | x_sd_q_reg[] -> fault_reg |
