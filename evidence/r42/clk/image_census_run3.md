| group | endpoints < 2.0 / 1.5 / 1.0 / 0.5 ns slack | of which start at the global reset synchroniser (reset_sync_reg) | worst slack | worst path (levels, ns, net ns) | top end registers |
|---|---|---|--:|---|---|
| A tile-internal: g_home_pipe.u_home_pipe | 26116 / 13834 / 4318 / 134 | 0 / 0 / 0 / 0 | 0.296 | 10 lv, 9.388, net 8.266 | ms_addr_reg[][](3712), RAMD(1481), RAMG(1421) |
| A tile-internal: u_core/(core own) | 6348 / 3117 / 757 / 32 | 0 / 0 / 0 / 0 | 0.248 | 10 lv, 9.740, net 8.636 | w_vdata_reg[](925), pc_reg[][](875), sb_s_reg[](755) |
| A tile-internal: u_core/g_vrf_banked.u_vrf | 5788 / 1920 / 328 / 46 | 0 / 0 / 0 / 0 | 0.284 | 16 lv, 9.552, net 7.831 | vld_reg[][](4136), s1_vld_reg[](192), mem_reg_0(188) |
| A tile-internal: u_core/g_lsu_hybrid.u_lsu | 5760 / 5178 / 4575 / 359 | 0 / 0 / 0 / 0 | 0.126 | 25 lv, 9.852, net 7.502 | lane_addr_reg[][](2048), out_vector_reg[](1512), lane_wdata_reg[][](1088) |
| A tile-internal: u_core/u_txf | 4655 / 2808 / 55 / 0 | 0 / 0 / 0 / 0 | 0.662 | 5 lv, 9.322, net 8.690 | vw_data_reg[](521), cm_cnt_reg[][](189), t_i2_reg[](180) |
| A tile-internal: u_core/u_mtx | 2793 / 950 / 161 / 9 | 0 / 0 / 0 / 0 | 0.216 | 15 lv, 9.707, net 8.300 | g_in_reg.a_q_reg[](920), valid_reg[][](662), DSP_A_B_DATA_INST(430) |
| A tile-internal: u_core/u_vfpu | 2097 / 572 / 106 / 1 | 0 / 0 / 0 / 0 | 0.493 | 11 lv, 9.145, net 7.997 | g_l2.s0_q_reg[](1228), dly_v_reg[][](576), DSP_A_B_DATA_INST(84) |
| A tile-internal: u_core/u_bar | 659 / 208 / 53 / 0 | 0 / 0 / 0 / 0 | 0.544 | 18 lv, 9.335, net 7.562 | done_epoch_reg[][](652), wake_reg[](4), slp_v_reg[](2) |
| A tile-internal: u_core/u_valu | 161 / 109 / 2 / 0 | 0 / 0 / 0 / 0 | 0.905 | 4 lv, 8.602, net 8.146 | DSP_OUTPUT_INST(96), DSP_A_B_DATA_INST(35), q_pu_reg[][]__0(15) |
| A tile-internal: u_core/u_mul | 1 / 0 / 0 / 0 | 0 / 0 / 0 / 0 | 1.988 | 7 lv, 8.133, net 6.362 | p0_hl_reg[](1) |
| B tile boundary: starts outside the tile (shell/mesh/link) | 13929 / 9590 / 5177 / 272 | 13784 / 9562 / 5174 / 272 (worst 0.168) | 0.168 | 1 lv, 9.462, net 9.244 | ms_addr_reg[][](1856), valid_reg[][](1629), rq_addr_reg[][](1189) |
| B mesh (u_mesh) | 5637 / 1931 / 688 / 0 | 3056 / 800 / 48 / 0 (worst 0.941) | 0.706 | 16 lv, 9.002, net 7.397 | RAME_D1(379), RAMC_D1(376), RAMB_D1(373) |
| B home links (SLR crossing) | 1972 / 1021 / 206 / 5 | 452 / 276 / 82 / 1 (worst 0.343) | 0.343 | 3 lv, 9.346, net 9.127 | RAME_D1(126), RAMG_D1(123), RAMF_D1(123) |
| B HBM outlet links (SLR crossing) | 526 / 156 / 7 / 0 | 0 / 0 / 0 / 0 | 0.801 | 6 lv, 9.030, net 8.424 | RAMB_D1(49), RAMA_D1(44), RAMD_D1(42) |
| B command links | 180 / 32 / 0 / 0 | 4 / 0 / 0 / 0 (worst 1.832) | 1.119 | 6 lv, 8.179, net 7.464 | RAMA(11), RAMA_D1(11), RAMB(11) |
| C HBM AXI outlet adapters | 8348 / 3815 / 422 / 0 | 2600 / 872 / 144 / 0 (worst 0.604) | 0.595 | 10 lv, 9.522, net 8.557 | bus_address_reg[][](4785), address_reg[][](2445), tag_reg[][](402) |
| C platform / HBM subsystem | 1640 / 1115 / 93 / 0 | 16 / 9 / 4 / 0 (worst 0.831) | 0.671 | 0 lv, 1.576, net 1.271 | gen_rd_b.gen_rd_b_synth_template.gen_rf_narrow_reg.doutb_reg_reg[](1548), dest_graysync_ff_reg[][](76), gen_thread_loop[].r_word_cnt_reg[][](8) |
| C TMU (shell, outside the tiles) | 1505 / 1302 / 1149 / 0 | 1248 / 1082 / 1081 / 0 (worst 0.818) | 0.779 | 3 lv, 8.280, net 7.807 | tb_reg[][](572), te_reg[][](531), mw_data_q_reg[](192) |
| C device top: control / MMIO | 1470 / 810 / 285 / 42 | 0 / 0 / 0 / 0 | 0.111 | 14 lv, 9.277, net 8.102 | rsp_r0_reg[](126), rsp_r1_reg[](126), launch_sequence_reg[](100) |
| C dispatch / code / deadline (shell) | 1284 / 657 / 212 / 0 | 0 / 0 / 0 / 0 | 0.650 | 8 lv, 8.478, net 7.572 | active_cycles_reg[](192), completed_reg[](96), issued_reg[](96) |
| C kernel registers | 24 / 11 / 3 / 0 | 0 / 0 / 0 / 0 | 0.506 | 5 lv, 9.067, net 8.622 | rd_data_s_reg[](24) |
| D other: system/shell/invalid_done_reg_replica | 1 / 1 / 1 / 0 | 0 / 0 / 0 / 0 | 0.989 | 12 lv, 8.287, net 7.403 | invalid_done_reg_replica(1) |
| D other: system/shell/invalid_done_reg | 1 / 1 / 0 / 0 | 0 / 0 / 0 / 0 | 1.044 | 12 lv, 8.232, net 7.348 | invalid_done_reg(1) |
| D other: g_multiple_memory.outlets/g_port[].arb/held_pick_reg[]_rep__1 | 1 / 0 / 0 / 0 | 0 / 0 / 0 / 0 | 1.682 | 7 lv, 8.319, net 7.548 | held_pick_reg[]_rep__1(1) |
| D other: shell/u_system/host_allow_reg | 1 / 0 / 0 / 0 | 0 / 0 / 0 / 0 | 1.691 | 7 lv, 7.419, net 6.735 | host_allow_reg(1) |
| D other: g_multiple_memory.outlets/g_port[].arb/held_pick_reg[]_rep__0 | 1 / 0 / 0 / 0 | 0 / 0 / 0 / 0 | 1.737 | 7 lv, 8.264, net 7.532 | held_pick_reg[]_rep__0(1) |
| D other: g_multiple_memory.outlets/g_port[].arb/held_reg | 1 / 0 / 0 / 0 | 0 / 0 / 0 / 0 | 1.747 | 7 lv, 8.254, net 7.544 | held_reg(1) |
| **all kernel-clock endpoints** | 90899 / 49138 / 18598 / 900 | 21160 / 12601 / 6533 / 273 | | | |
| group A total | 54378 / 28696 / 10355 / 581 | | | | |
| group B total | 22244 / 12730 / 6078 / 277 | | | | |
| group C total | 14271 / 7710 / 2164 / 42 | | | | |
| group D total | 6 / 2 / 1 / 0 | | | | |
