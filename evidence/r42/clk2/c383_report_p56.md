| unit | endpoints < 1.5 | failing at 5.60 | path > 6.2 | path > 6.67 | worst slack | baseline failing / > 6.2 / > 6.67 |
|---|--:|--:|--:|--:|--:|---|
| u_core/u_vfpu | 5413 | 1671 | 606 | 0 | -0.894 | 1976 / 1505 / 563 |
| g_home_pipe.u_home_pipe | 9695 | 1291 | 282 | 0 | -0.893 | 7190 / 3715 / 595 |
| u_core/g_vrf_banked.u_vrf | 8619 | 330 | 84 | 0 | -0.890 | 4060 / 1105 / 86 |
| u_core/u_mtx | 22529 | 2586 | 599 | 0 | -0.889 | 4342 / 1476 / 336 |
| u_core/(core own) | 6247 | 1313 | 75 | 0 | -0.883 | 4311 / 1279 / 203 |
| u_core/u_txf | 13519 | 3335 | 107 | 0 | -0.861 | 3318 / 731 / 120 |
| g_host_mux.u_host | 4 | 4 | 3 | 0 | -0.652 | 4 / 4 / 0 |
| u_core/g_lsu_hybrid.u_lsu | 7549 | 137 | 0 | 0 | -0.588 | 3305 / 843 / 478 |
| u_core/u_valu | 2665 | 110 | 0 | 0 | -0.376 | 573 / 307 / 103 |
| u_core/u_bar | 1671 | 193 | 0 | 0 | -0.375 | 1469 / 688 / 66 |
| u_core/u_mul | 520 | 0 | 0 | 0 | 0.061 | 479 / 0 / 0 |
| u_core/u_srf | 1721 | 0 | 0 | 0 | 0.453 | 176 / 80 / 8 |
| u_core/u_prf | 262 | 0 | 0 | 0 | 0.996 | 0 / 0 / 0 |
| **tile** | 80414 | 10970 | 1756 | 0 | | |

### matrix engine: failing families

| failing | of | worst slack | levels | start -> end |
|--:|--:|--:|--:|---|
| 921 | 984 | -0.859 | 24 | g_c0.q_reg[] -> g_c2.q_reg[] |
| 638 | 1487 | -0.876 | 22 | g_in_reg.a_q_reg[] -> g_c0.q_reg[] |
| 246 | 8033 | -0.480 | 7 | g_seq0.direct_w_reg -> u_acc/valid_reg[][] |
| 170 | 218 | -0.883 | 14 | x_sb_q_reg[] -> g_in_reg.a_q_reg[] |
| 131 | 183 | -0.889 | 15 | g_seq0.mm_reg[]__0 -> g_in_reg.a_q_reg[] |
| 83 | 105 | -0.881 | 15 | x_sb_fw_reg_rep -> g_in_reg.a_q_reg[] |
| 37 | 594 | -0.128 | 14 | g_in_reg.b_q_reg[] -> g_c0.q_reg[] |
| 17 | 350 | -0.422 | 20 | g_c3.q_reg[] -> g_c4.q_reg[] |
| 14 | 243 | -0.095 | 10 | g_c2.q_reg[] -> g_c3.q_reg[] |
| 12 | 242 | -0.215 | 6 | g_seq0.r_scale_reg[] -> g_in_reg.b_q_reg[] |
| 9 | 10 | -0.769 | 14 | g_seq0.mm_reg[] -> g_in_reg.a_q_reg[] |
| 7 | 343 | -0.256 | 9 | x_sb_q_reg[] -> g_c0.kq_reg[]_srl2 |

### vector FPU: failing families

| failing | of | worst slack | levels | start -> end |
|--:|--:|--:|--:|---|
| 688 | 1138 | -0.878 | 17 | g_vrf_banked.x_vb_f_reg[] -> g_fma[].u_fma/g_l2x.u_fma_x/g_c1.q_reg[] |
| 306 | 481 | -0.894 | 17 | red_busy_reg -> g_fma[].u_fma/g_l2x.u_fma_x/g_c1.q_reg[] |
| 234 | 371 | -0.884 | 17 | x_fpctl_reg[] -> g_fma[].u_fma/g_l2x.u_fma_x/g_c1.q_reg[] |
| 211 | 298 | -0.886 | 21 | cvt_x_reg[][] -> dly_v_reg[][] |
| 53 | 53 | -0.821 | 21 | cvt_mod_reg[] -> dly_v_reg[][] |
| 52 | 367 | -0.478 | 20 | g_fma[].u_fma/g_l2x.u_fma_x/g_c1.q_reg[] -> g_fma[].u_fma/g_l2x.u_fma_x/g_c3.q_reg[] |
| 43 | 512 | -0.396 | 5 | x_func_reg[] -> g_sfu.u_sfu/g_slice[].u_serial/src_reg[][] |
| 38 | 161 | -0.695 | 15 | cvt_fn_reg[] -> dly_v_reg[][] |
| 16 | 16 | -0.485 | 12 | g_sfu.u_sfu/g_slice[].u_serial/cnt_reg[] -> g_sfu.u_sfu/g_slice[].u_serial/u_exp2/p0_bit_slice_606_reg_rep_rep |
| 12 | 30 | -0.165 | 16 | g_sfu.u_sfu/g_slice[].u_serial/cnt_reg[] -> g_sfu.u_sfu/g_slice[].u_serial/u_exp2/p2_i_reg[]_srl3___p2_eq_361_reg_r |
| 7 | 10 | -0.294 | 5 | x_func_reg[] -> g_sfu.u_sfu/g_slice[].u_serial/op_l_reg[] |
| 5 | 16 | -0.842 | 15 | acc_reg[][] -> g_fma[].u_fma/g_l2x.u_fma_x/g_c1.q_reg[] |

### transfer engine: failing families

| failing | of | worst slack | levels | start -> end |
|--:|--:|--:|--:|---|
| 539 | 943 | -0.437 | 13 | t_unit_reg[] -> vw_data_reg[] |
| 498 | 512 | -0.664 | 13 | t_unit_reg[] -> sl_dlo_reg[][] |
| 440 | 447 | -0.830 | 13 | t_unit_reg[] -> sl_dhi_reg[][] |
| 112 | 132 | -0.729 | 13 | t_unit_reg[] -> st_prev_reg[] |
| 64 | 64 | -0.711 | 14 | t_unit_reg[] -> t_unit_reg[] |
| 64 | 64 | -0.617 | 13 | t_unit_reg[] -> t_P1_reg[] |
| 64 | 64 | -0.587 | 13 | t_unit_reg[] -> t_P2_reg[] |
| 64 | 64 | -0.565 | 13 | t_unit_reg[] -> t_P0_reg[] |
| 64 | 64 | -0.532 | 13 | t_unit_reg[] -> t_ps_reg[] |
| 64 | 64 | -0.524 | 13 | t_unit_reg[] -> t_P3_reg[] |
| 63 | 63 | -0.632 | 14 | t_unit_reg[] -> t_i0_reg[] |
| 63 | 63 | -0.581 | 14 | t_unit_reg[] -> t_i3_reg[] |

### barrier: failing families

| failing | of | worst slack | levels | start -> end |
|--:|--:|--:|--:|---|
| 154 | 608 | -0.375 | 14 | w_sdata_reg[] -> done_epoch_reg[][] |
| 16 | 86 | -0.216 | 11 | w_sdata_reg[] -> n_arrive_reg[][] |
| 11 | 128 | -0.211 | 12 | w_sdata_reg[] -> n_ops_reg[][] |
| 5 | 6 | -0.284 | 14 | w_sdata_reg[] -> fault_bar_reg[] |
| 3 | 4 | -0.287 | 17 | w_sdata_reg[] -> slp_v_reg[] |
| 3 | 4 | -0.241 | 16 | w_sdata_reg[] -> wake_reg[] |
| 1 | 1 | -0.151 | 14 | w_sdata_reg[] -> fault_reg |
