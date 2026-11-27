| unit | endpoints < 1.5 ns | failing at 6.67 ns | path > 6.667 ns | path > 5 ns | worst slack | worst path (levels, ns, net %) | start -> end |
|---|--:|--:|--:|--:|--:|---|---|
| u_core/u_mtx | 6182 | 32 | 34 | 6182 | -0.066 | 18 lv, 6.72, 68 % | g_c0.q_reg[] -> g_c2.q_reg[] |
| u_core/g_lsu_hybrid.u_lsu | 3763 | 85 | 87 | 3763 | -0.064 | 19 lv, 6.72, 72 % | rr_reg[] -> out_vector_reg[] |
| g_home_pipe.u_home_pipe | 8584 | 11 | 12 | 8584 | -0.061 | 13 lv, 6.38, 65 % | data_mem_uram_reg_uram_4 -> data_mem_uram_reg_uram_5 |
| u_core/g_vrf_banked.u_vrf | 4774 | 11 | 11 | 4774 | -0.058 | 20 lv, 6.71, 73 % | t_i1_reg[] -> s1_vld_reg[] |
| u_core/u_vfpu | 2204 | 6 | 8 | 2204 | -0.044 | 21 lv, 6.69, 61 % | cvt_x_reg[][] -> dly_v_reg[][] |
| u_core/(core own) | 5106 | 5 | 5 | 5106 | -0.038 | 12 lv, 6.69, 81 % | g_c3.q_reg[] -> w_vdata_reg[] |
| u_core/u_bar | 1634 | 0 | 0 | 1634 | 0.045 | 18 lv, 6.52, 77 % | x_sd_q_reg[] -> done_epoch_reg[][] |
| u_core/u_txf | 4813 | 0 | 0 | 4813 | 0.046 | 17 lv, 6.61, 52 % | dl_word_reg[] -> dl_boxel[]_i_1_psdsp |
| u_core/u_valu | 1632 | 0 | 0 | 1632 | 0.198 | 4 lv, 6.45, 92 % | x_func_reg[] -> q_gt_reg[] |
| g_host_mux.u_host | 4 | 0 | 0 | 4 | 0.265 | 17 lv, 6.39, 80 % | t_unit_reg[] -> host_busy_reg |
| u_core/u_mul | 208 | 0 | 0 | 208 | 1.250 | 8 lv, 5.41, 58 % | w_sdata_reg[] -> DSP_OUTPUT_INST |
| **tile** | 38904 | 150 | 157 | 38904 | | | |
