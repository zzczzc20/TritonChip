| unit | endpoints < 1.5 ns | failing at 6.67 ns | path > 6.667 ns | path > 5 ns | worst slack | worst path (levels, ns, net %) | start -> end |
|---|--:|--:|--:|--:|--:|---|---|
| u_core/g_lsu_hybrid.u_lsu | 8148 | 580 | 581 | 8148 | -0.521 | 20 lv, 7.17, 78 % | rr_reg[] -> out_vector_reg[] |
| g_home_pipe.u_home_pipe | 40110 | 852 | 860 | 40110 | -0.520 | 16 lv, 7.17, 81 % | rq_addr_reg[][] -> g_tag[].read_tag_reg[] |
| u_core/(core own) | 6359 | 491 | 500 | 6359 | -0.454 | 19 lv, 7.10, 76 % | x_sd_q_reg[] -> w_sdata_reg[] |
| u_core/u_vfpu | 2701 | 694 | 705 | 2701 | -0.452 | 21 lv, 7.10, 63 % | cvt_x_reg[][] -> dly_v_reg[][] |
| u_core/u_bar | 1671 | 162 | 162 | 1671 | -0.422 | 20 lv, 6.99, 72 % | x_sd_q_reg[] -> fault_bar_reg[] |
| u_core/u_mtx | 7713 | 566 | 574 | 7713 | -0.414 | 21 lv, 7.06, 51 % | g_in_reg.a_q_reg[] -> g_c0.q_reg[] |
| u_core/g_vrf_banked.u_vrf | 5669 | 94 | 95 | 5669 | -0.405 | 17 lv, 7.06, 82 % | st_reg[] -> s1_vld_reg[] |
| u_core/u_txf | 9613 | 49 | 49 | 9613 | -0.284 | 18 lv, 6.65, 49 % | dl_word_reg[] -> DSP_A_B_DATA_INST |
| u_core/u_valu | 2038 | 0 | 0 | 2038 | 0.051 | 5 lv, 6.60, 89 % | w_pdata_reg[] -> q_gt_reg[] |
| u_core/u_srf | 160 | 0 | 0 | 160 | 0.639 | 0 lv, 5.89, 99 % | w_idx_reg[] -> RAMA |
| g_host_mux.u_host | 4 | 0 | 0 | 4 | 0.669 | 17 lv, 5.98, 70 % | t_ps_reg[] -> host_busy_reg_rep__1 |
| u_core/u_prf | 92 | 0 | 0 | 92 | 0.825 | 2 lv, 5.74, 95 % | w_idx_reg[] -> mem_reg[][] |
| **tile** | 84278 | 3488 | 3526 | 84278 | | | |
