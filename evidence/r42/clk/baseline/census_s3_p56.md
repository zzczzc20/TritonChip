| unit | endpoints < 1.5 ns | failing at 5.60 ns | path > 6.667 ns | path > 5 ns | worst slack | worst path (levels, ns, net %) | start -> end |
|---|--:|--:|--:|--:|--:|---|---|
| u_core/g_lsu_hybrid.u_lsu | 8875 | 3305 | 478 | 5839 | -1.392 | 21 lv, 6.97, 69 % | done_reg[] -> out_vector_reg[] |
| u_core/u_vfpu | 5182 | 1976 | 563 | 2966 | -1.390 | 17 lv, 6.97, 54 % | g_vrf_banked.x_vb_f_reg[] -> g_c1.q_reg[] |
| u_core/(core own) | 7860 | 4311 | 203 | 5668 | -1.389 | 15 lv, 6.88, 81 % | valid_reg[] -> x_valu_imm_reg_replica_15 |
| g_home_pipe.u_home_pipe | 12445 | 7190 | 595 | 9606 | -1.387 | 19 lv, 6.88, 76 % | s1_addr_reg[]_rep -> perf_evict_reg[] |
| u_core/u_mtx | 17261 | 4342 | 336 | 10594 | -1.367 | 20 lv, 6.95, 53 % | g_in_reg.a_q_reg[] -> g_c0.q_reg[] |
| u_core/u_txf | 10512 | 3318 | 120 | 6621 | -1.353 | 9 lv, 6.93, 86 % | vw_rctx_reg[] -> vw_lane_en_reg[] |
| u_core/u_bar | 1671 | 1469 | 66 | 1653 | -1.333 | 18 lv, 6.83, 78 % | valid_reg[] -> fault_bar_reg[] |
| u_core/g_vrf_banked.u_vrf | 8487 | 4060 | 86 | 5143 | -1.329 | 19 lv, 6.91, 77 % | t_i1_reg[] -> s1_vld_reg[] |
| u_core/u_valu | 3048 | 573 | 103 | 1653 | -1.298 | 7 lv, 6.88, 74 % | g_vrf_banked.of_d1_reg[] -> q_pu_reg[][]__0 |
| u_core/u_srf | 1684 | 176 | 8 | 488 | -1.092 | 2 lv, 6.49, 95 % | w_idx_reg[] -> RAMA |
| g_host_mux.u_host | 4 | 4 | 0 | 4 | -0.991 | 16 lv, 6.57, 76 % | t_unit_reg[] -> host_busy_reg |
| u_core/u_mul | 520 | 479 | 0 | 520 | -0.405 | 9 lv, 6.00, 62 % | x_func_reg[] -> DSP_OUTPUT_INST |
| u_core/u_prf | 755 | 0 | 0 | 92 | 0.022 | 2 lv, 5.47, 94 % | w_idx_reg[] -> mem_reg[][] |
| **tile** | 78304 | 31203 | 2558 | 50847 | | | |
