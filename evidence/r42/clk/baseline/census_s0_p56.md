| unit | endpoints < 1.5 ns | failing at 5.60 ns | path > 6.667 ns | path > 5 ns | worst slack | worst path (levels, ns, net %) | start -> end |
|---|--:|--:|--:|--:|--:|---|---|
| u_core/g_vrf_banked.u_vrf | 9270 | 1915 | 407 | 6944 | -1.642 | 22 lv, 7.22, 74 % | t_i2_reg[] -> s1_vld_reg[] |
| u_core/u_mtx | 25178 | 5700 | 631 | 13070 | -1.637 | 15 lv, 7.22, 81 % | w_sdata_reg[] -> g_in_reg.a_q_reg[] |
| u_core/(core own) | 9197 | 3889 | 528 | 6776 | -1.634 | 13 lv, 7.21, 82 % | g_l2.s1_q_reg[] -> w_vdata_reg[] |
| u_core/g_lsu_hybrid.u_lsu | 10300 | 4437 | 777 | 6785 | -1.634 | 17 lv, 7.22, 74 % | rr_reg[] -> out_vector_reg[] |
| g_home_pipe.u_home_pipe | 68285 | 19950 | 3223 | 60480 | -1.630 | 15 lv, 7.13, 80 % | wbq_addr_reg[][] -> ms_state_reg[][] |
| u_core/u_vfpu | 5497 | 2175 | 718 | 3251 | -1.616 | 21 lv, 7.20, 66 % | cvt_x_reg[][]_replica -> dly_v_reg[][] |
| u_core/u_txf | 10557 | 3660 | 208 | 5719 | -1.593 | 14 lv, 7.17, 80 % | t_unit_reg[] -> vw_data_reg[] |
| u_core/u_bar | 1671 | 1279 | 28 | 1671 | -1.460 | 18 lv, 7.04, 73 % | w_class_reg[] -> slp_v_reg[] |
| u_core/u_valu | 3211 | 1300 | 53 | 2631 | -1.199 | 6 lv, 6.78, 92 % | x_func_reg[] -> q_gt_reg[] |
| g_host_mux.u_host | 4 | 4 | 0 | 4 | -0.168 | 15 lv, 5.75, 73 % | t_unit_reg[] -> host_busy_reg_rep__1 |
| u_core/u_srf | 1368 | 0 | 0 | 192 | 0.194 | 0 lv, 5.30, 98 % | w_idx_reg[] -> RAMA |
| u_core/u_mul | 520 | 0 | 0 | 255 | 0.364 | 10 lv, 5.23, 53 % | x_func_reg[] -> DSP_OUTPUT_INST |
| u_core/u_prf | 301 | 0 | 0 | 1 | 0.471 | 2 lv, 5.02, 94 % | w_idx_reg[] -> mem_reg[][] |
| **tile** | 145359 | 44309 | 6573 | 107779 | | | |
