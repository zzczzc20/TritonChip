Report: `evidence/r41/hbm_bw/resources/run1_routed/util_hier_d20.rpt.gz`


#### Whole design: platform shell versus TNCA kernel (`tnca_u50_1`)

| Instance | LUT | logic | LUTRAM | SRL | FF | RAMB36 | RAMB18 | DSP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ulp/tnca_u50_1 | 798,355 | 742,721 | 55,472 | 162 | 226,528 | 192 | 45 | 520 |
| blp | 106,153 | 100,176 | 4,564 | 1,413 | 129,945 | 193 | 5 | 4 |
| ulp/hmss_0 | 13,902 | 8,860 | 1,996 | 3,046 | 27,063 | 4 | 0 | 0 |
| ulp/ulp_ucs | 4,493 | 4,473 | 0 | 20 | 7,375 | 0 | 0 | 0 |
| ii_level0_pipe | 1,482 | 1,482 | 0 | 0 | 6,141 | 0 | 0 | 0 |
| ulp (own logic) | 1,468 | 1,468 | 0 | 0 | 0 | 0 | 0 | 0 |
| ulp/ulp_cmp | 481 | 481 | 0 | 0 | 670 | 0 | 0 | 0 |
| ulp/SLR0 | 385 | 385 | 0 | 0 | 1,397 | 0 | 0 | 0 |
| ulp/SLR1 | 205 | 205 | 0 | 0 | 790 | 0 | 0 | 0 |
| ulp/SLR2 | 205 | 205 | 0 | 0 | 790 | 0 | 0 | 0 |
| ulp/axi_ic_ctrl_mgmt_slr1 | 121 | 121 | 0 | 0 | 123 | 0 | 0 | 0 |
| ulp/proc_sys_reset_kernel_slr0 | 17 | 16 | 0 | 1 | 37 | 0 | 0 | 0 |
| ulp/proc_sys_reset_ctrl_slr1 | 16 | 15 | 0 | 1 | 33 | 0 | 0 | 0 |
| ulp/proc_sys_reset_ctrl_slr2 | 16 | 15 | 0 | 1 | 33 | 0 | 0 | 0 |
| ulp/proc_sys_reset_ctrl_slr0 | 15 | 14 | 0 | 1 | 33 | 0 | 0 | 0 |
| ulp/axi_vip_data | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| ulp/ii_level0_wire | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| ulp/satellite_gpio_slice_1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| **total** | **927,245** | **860,568** | **62,032** | **4,645** | **400,958** | **389** | **50** | **524** |

#### TNCA kernel (`device/system`), by block

| Instance | LUT | logic | LUTRAM | SRL | FF | RAMB36 | RAMB18 | DSP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| shell/u_system/g_tile[1].u_tile | 247,348 | 232,074 | 15,220 | 54 | 65,382 | 32 | 15 | 171 |
| shell/u_system/g_tile[0].u_tile | 246,868 | 231,598 | 15,216 | 54 | 65,345 | 32 | 15 | 171 |
| shell/u_system/g_tile[2].u_tile | 246,458 | 231,184 | 15,220 | 54 | 65,303 | 32 | 15 | 171 |
| shell/u_system/g_local_tmu.u_tmu | 16,397 | 15,569 | 828 | 0 | 6,096 | 0 | 0 | 0 |
| g_outlet[0].adapter | 7,413 | 7,413 | 0 | 0 | 6,003 | 0 | 0 | 0 |
| g_outlet[1].adapter | 7,283 | 7,283 | 0 | 0 | 5,930 | 0 | 0 | 0 |
| g_outlet[2].adapter | 7,238 | 7,238 | 0 | 0 | 5,930 | 0 | 0 | 0 |
| shell/u_system/u_mesh | 5,130 | 2,502 | 2,628 | 0 | 264 | 0 | 0 | 0 |
| shell/u_system/g_multiple_memory.outlets | 2,136 | 2,136 | 0 | 0 | 8 | 0 | 0 | 0 |
| shell/u_system/g_dispatch.u_dispatch | 1,312 | 1,312 | 0 | 0 | 1,407 | 0 | 0 | 7 |
| g_outlet[0].link | 1,297 | 625 | 672 | 0 | 528 | 0 | 0 | 0 |
| shell/u_system/g_home_link[1].g_pipeline.link | 1,183 | 511 | 672 | 0 | 1,040 | 0 | 0 | 0 |
| g_outlet[1].link | 774 | 102 | 672 | 0 | 11 | 0 | 0 | 0 |
| g_outlet[2].link | 773 | 101 | 672 | 0 | 13 | 0 | 0 | 0 |
| shell/u_system/g_home_link[0].g_pipeline.link | 691 | 19 | 672 | 0 | 10 | 0 | 0 | 0 |
| shell/u_system/g_home_link[2].g_pipeline.link | 684 | 12 | 672 | 0 | 10 | 0 | 0 | 0 |
| shell/u_system/g_dispatch.g_command_link[2].commands | 648 | 264 | 384 | 0 | 8 | 0 | 0 | 0 |
| shell/u_system/g_dispatch.g_command_link[0].commands | 410 | 26 | 384 | 0 | 8 | 0 | 0 | 0 |
| shell/u_system/g_dispatch.g_command_link[1].commands | 408 | 24 | 384 | 0 | 8 | 0 | 0 | 0 |
| shell/g_code[2].commands | 171 | 11 | 160 | 0 | 8 | 0 | 0 | 0 |
| shell/g_code[0].commands | 170 | 10 | 160 | 0 | 8 | 0 | 0 | 0 |
| shell/g_code[1].commands | 170 | 10 | 160 | 0 | 8 | 0 | 0 | 0 |
| shell/u_system/g_cache_handoff.g_home_link[0].command | 156 | 8 | 148 | 0 | 8 | 0 | 0 | 0 |
| shell/u_system/g_cache_handoff.g_home_link[1].command | 156 | 8 | 148 | 0 | 8 | 0 | 0 | 0 |
| shell/u_system/g_cache_handoff.g_home_link[2].command | 156 | 8 | 148 | 0 | 8 | 0 | 0 | 0 |
| shell/(shell) | 117 | 117 | 0 | 0 | 139 | 96 | 0 | 0 |
| shell/g_code[0].replies | 89 | 9 | 80 | 0 | 8 | 0 | 0 | 0 |
| shell/g_code[1].replies | 89 | 9 | 80 | 0 | 8 | 0 | 0 | 0 |
| shell/g_code[2].replies | 89 | 9 | 80 | 0 | 8 | 0 | 0 | 0 |
| shell/u_system/(u_system) | 71 | 71 | 0 | 0 | 22 | 0 | 0 | 0 |
| shell/u_deadline | 58 | 58 | 0 | 0 | 129 | 0 | 0 | 0 |
| g_memory_mode.mode | 23 | 23 | 0 | 0 | 31 | 0 | 0 | 0 |
| shell/u_system/g_dispatch.g_command_link[0].g_pipeline.completions | 18 | 14 | 4 | 0 | 8 | 0 | 0 | 0 |
| shell/u_system/g_dispatch.g_command_link[2].g_pipeline.completions | 18 | 14 | 4 | 0 | 8 | 0 | 0 | 0 |
| shell/u_system/g_dispatch.g_command_link[1].g_pipeline.completions | 16 | 12 | 4 | 0 | 8 | 0 | 0 | 0 |
| shell/u_system/g_cache_handoff.ownership | 13 | 13 | 0 | 0 | 136 | 0 | 0 | 0 |
| shell/u_system/g_cache_handoff.g_home_link[0].response | 4 | 4 | 0 | 0 | 4 | 0 | 0 | 0 |
| shell/u_system/g_cache_handoff.g_home_link[1].response | 4 | 4 | 0 | 0 | 4 | 0 | 0 | 0 |
| shell/u_system/g_cache_handoff.g_home_link[2].response | 4 | 4 | 0 | 0 | 4 | 0 | 0 | 0 |
| **total** | **796,005** | **740,371** | **55,472** | **162** | **223,869** | **192** | **45** | **520** |

#### Per tile, per module: LUT

| Module | g_tile[0].u_tile | g_tile[1].u_tile | g_tile[2].u_tile | share of tile 0 |
|---|---:|---:|---:|---:|
| u_core/u_vfpu | 61,661 | 62,143 | 61,870 | 25.0 % |
| u_core/u_mtx | 49,650 | 49,794 | 48,654 | 20.1 % |
| u_slice | 30,714 | 31,344 | 31,340 | 12.4 % |
| u_core/g_lsu_pair.u_lsu | 23,444 | 23,080 | 24,081 | 9.5 % |
| u_core/(u_core) | 23,268 | 22,766 | 22,576 | 9.4 % |
| u_core/u_txf | 19,742 | 19,663 | 19,654 | 8.0 % |
| u_core/g_vrf_banked.u_vrf | 15,807 | 15,751 | 15,746 | 6.4 % |
| u_core/u_bar | 7,862 | 8,245 | 7,946 | 3.2 % |
| u_core/u_valu | 4,459 | 4,441 | 4,438 | 1.8 % |
| u_cni | 4,196 | 4,127 | 4,145 | 1.7 % |
| u_core/u_srf | 3,727 | 3,728 | 3,728 | 1.5 % |
| u_core/u_prf | 912 | 912 | 910 | 0.4 % |
| u_core/u_dec | 680 | 681 | 686 | 0.3 % |
| u_core/u_mul | 619 | 609 | 599 | 0.3 % |
| u_core/u_pfdec | 327 | 326 | 329 | 0.1 % |
| u_core/u_salu | 79 | 98 | 97 | 0.0 % |
| g_host_mux.u_host | 84 | 0 | 0 | 0.0 % |
| (g_tile[0].u_tile) | 4 | 0 | 0 | 0.0 % |
| (g_tile[1].u_tile) | 0 | 3 | 0 | 0.0 % |
| (g_tile[2].u_tile) | 0 | 0 | 3 | 0.0 % |
| **tile total** | **246,868** | **247,348** | **246,458** | |

#### Per tile, per module: LUTRAM

| Module | g_tile[0].u_tile | g_tile[1].u_tile | g_tile[2].u_tile | share of tile 0 |
|---|---:|---:|---:|---:|
| u_slice | 9,632 | 9,636 | 9,636 | 63.3 % |
| u_core/u_mtx | 2,856 | 2,856 | 2,856 | 18.8 % |
| u_core/u_srf | 2,072 | 2,072 | 2,072 | 13.6 % |
| u_core/u_txf | 596 | 596 | 596 | 3.9 % |
| u_core/g_lsu_pair.u_lsu | 60 | 60 | 60 | 0.4 % |
| **tile total** | **15,216** | **15,220** | **15,220** | |

#### Per tile, per module: FF

| Module | g_tile[0].u_tile | g_tile[1].u_tile | g_tile[2].u_tile | share of tile 0 |
|---|---:|---:|---:|---:|
| u_core/g_lsu_pair.u_lsu | 13,641 | 13,654 | 13,646 | 20.9 % |
| u_core/u_mtx | 11,463 | 11,462 | 11,462 | 17.5 % |
| u_core/u_vfpu | 8,407 | 8,407 | 8,405 | 12.9 % |
| u_core/u_txf | 8,324 | 8,323 | 8,324 | 12.7 % |
| u_core/g_vrf_banked.u_vrf | 6,570 | 6,501 | 6,530 | 10.1 % |
| u_core/(u_core) | 6,377 | 6,388 | 6,290 | 9.8 % |
| u_slice | 4,641 | 4,716 | 4,699 | 7.1 % |
| u_core/u_bar | 3,891 | 3,891 | 3,891 | 6.0 % |
| u_core/u_prf | 1,008 | 1,008 | 1,024 | 1.5 % |
| u_core/u_mul | 648 | 648 | 648 | 1.0 % |
| u_core/u_srf | 256 | 256 | 256 | 0.4 % |
| u_cni | 114 | 128 | 128 | 0.2 % |
| g_host_mux.u_host | 4 | 0 | 0 | 0.0 % |
| (g_tile[0].u_tile) | 1 | 0 | 0 | 0.0 % |
| **tile total** | **65,345** | **65,382** | **65,303** | |

#### g_tile[0].u_tile (SLR0): sub-blocks of at least 2,000 LUTs, three levels deep

| Instance | LUT | logic | LUTRAM | SRL | FF | RAMB36 | RAMB18 | DSP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| u_cni | 4,196 | 4,196 | 0 | 0 | 114 | 0 | 0 | 0 |
| u_core | 211,888 | 206,250 | 5,584 | 54 | 60,585 | 32 | 15 | 171 |
| u_core/g_lsu_pair.u_lsu | 23,444 | 23,384 | 60 | 0 | 13,641 | 0 | 0 | 0 |
| u_core/g_lsu_pair.u_lsu/g_unit[0].u | 13,090 | 13,060 | 30 | 0 | 6,819 | 0 | 0 | 0 |
| u_core/g_lsu_pair.u_lsu/g_unit[1].u | 10,355 | 10,325 | 30 | 0 | 6,819 | 0 | 0 | 0 |
| u_core/g_vrf_banked.u_vrf | 15,807 | 15,807 | 0 | 0 | 6,570 | 32 | 0 | 0 |
| u_core/g_vrf_banked.u_vrf/g_bank[0].u_bank | 2,991 | 2,991 | 0 | 0 | 1,634 | 8 | 0 | 0 |
| u_core/g_vrf_banked.u_vrf/g_bank[1].u_bank | 4,542 | 4,542 | 0 | 0 | 1,649 | 8 | 0 | 0 |
| u_core/g_vrf_banked.u_vrf/g_bank[2].u_bank | 3,428 | 3,428 | 0 | 0 | 1,635 | 8 | 0 | 0 |
| u_core/g_vrf_banked.u_vrf/g_bank[3].u_bank | 4,871 | 4,871 | 0 | 0 | 1,649 | 8 | 0 | 0 |
| u_core/u_bar | 7,862 | 7,862 | 0 | 0 | 3,891 | 0 | 0 | 0 |
| u_core/u_mtx | 49,650 | 46,794 | 2,856 | 0 | 11,463 | 0 | 0 | 32 |
| u_core/u_mtx/g_mac[0].u_mac | 2,567 | 2,567 | 0 | 0 | 364 | 0 | 0 | 2 |
| u_core/u_mtx/g_mac[10].u_mac | 2,427 | 2,427 | 0 | 0 | 364 | 0 | 0 | 2 |
| u_core/u_mtx/g_mac[11].u_mac | 2,433 | 2,433 | 0 | 0 | 364 | 0 | 0 | 2 |
| u_core/u_mtx/g_mac[12].u_mac | 2,430 | 2,430 | 0 | 0 | 366 | 0 | 0 | 2 |
| u_core/u_mtx/g_mac[13].u_mac | 2,424 | 2,424 | 0 | 0 | 364 | 0 | 0 | 2 |
| u_core/u_mtx/g_mac[14].u_mac | 2,429 | 2,429 | 0 | 0 | 364 | 0 | 0 | 2 |
| u_core/u_mtx/g_mac[15].u_mac | 2,847 | 2,847 | 0 | 0 | 364 | 0 | 0 | 2 |
| u_core/u_mtx/g_mac[1].u_mac | 2,428 | 2,428 | 0 | 0 | 364 | 0 | 0 | 2 |
| u_core/u_mtx/g_mac[2].u_mac | 2,425 | 2,425 | 0 | 0 | 364 | 0 | 0 | 2 |
| u_core/u_mtx/g_mac[3].u_mac | 2,426 | 2,426 | 0 | 0 | 364 | 0 | 0 | 2 |
| u_core/u_mtx/g_mac[4].u_mac | 2,427 | 2,427 | 0 | 0 | 364 | 0 | 0 | 2 |
| u_core/u_mtx/g_mac[5].u_mac | 2,425 | 2,425 | 0 | 0 | 364 | 0 | 0 | 2 |
| u_core/u_mtx/g_mac[6].u_mac | 2,426 | 2,426 | 0 | 0 | 364 | 0 | 0 | 2 |
| u_core/u_mtx/g_mac[7].u_mac | 2,430 | 2,430 | 0 | 0 | 364 | 0 | 0 | 2 |
| u_core/u_mtx/g_mac[8].u_mac | 2,426 | 2,426 | 0 | 0 | 364 | 0 | 0 | 2 |
| u_core/u_mtx/g_mac[9].u_mac | 2,424 | 2,424 | 0 | 0 | 364 | 0 | 0 | 2 |
| u_core/u_mtx/u_acc | 8,493 | 5,933 | 2,560 | 0 | 4,096 | 0 | 0 | 0 |
| u_core/u_srf | 3,727 | 1,655 | 2,072 | 0 | 256 | 0 | 0 | 0 |
| u_core/u_txf | 19,742 | 19,146 | 596 | 0 | 8,324 | 0 | 0 | 13 |
| u_core/u_valu | 4,459 | 4,459 | 0 | 0 | 0 | 0 | 0 | 64 |
| u_core/u_vfpu | 61,661 | 61,614 | 0 | 47 | 8,407 | 0 | 15 | 46 |
| u_core/u_vfpu/g_fma[0].u_fma | 2,469 | 2,469 | 0 | 0 | 271 | 0 | 0 | 2 |
| u_core/u_vfpu/g_fma[10].u_fma | 2,400 | 2,400 | 0 | 0 | 268 | 0 | 0 | 2 |
| u_core/u_vfpu/g_fma[11].u_fma | 2,400 | 2,400 | 0 | 0 | 268 | 0 | 0 | 2 |
| u_core/u_vfpu/g_fma[12].u_fma | 2,403 | 2,403 | 0 | 0 | 268 | 0 | 0 | 2 |
| u_core/u_vfpu/g_fma[13].u_fma | 2,411 | 2,411 | 0 | 0 | 268 | 0 | 0 | 2 |
| u_core/u_vfpu/g_fma[14].u_fma | 2,394 | 2,394 | 0 | 0 | 268 | 0 | 0 | 2 |
| u_core/u_vfpu/g_fma[15].u_fma | 2,412 | 2,412 | 0 | 0 | 268 | 0 | 0 | 2 |
| u_core/u_vfpu/g_fma[1].u_fma | 2,486 | 2,486 | 0 | 0 | 268 | 0 | 0 | 2 |
| u_core/u_vfpu/g_fma[2].u_fma | 2,489 | 2,489 | 0 | 0 | 268 | 0 | 0 | 2 |
| u_core/u_vfpu/g_fma[3].u_fma | 2,485 | 2,485 | 0 | 0 | 268 | 0 | 0 | 2 |
| u_core/u_vfpu/g_fma[4].u_fma | 2,486 | 2,486 | 0 | 0 | 268 | 0 | 0 | 2 |
| u_core/u_vfpu/g_fma[5].u_fma | 2,488 | 2,488 | 0 | 0 | 268 | 0 | 0 | 2 |
| u_core/u_vfpu/g_fma[6].u_fma | 2,486 | 2,486 | 0 | 0 | 268 | 0 | 0 | 2 |
| u_core/u_vfpu/g_fma[7].u_fma | 2,470 | 2,470 | 0 | 0 | 268 | 0 | 0 | 2 |
| u_core/u_vfpu/g_fma[8].u_fma | 2,406 | 2,406 | 0 | 0 | 268 | 0 | 0 | 2 |
| u_core/u_vfpu/g_fma[9].u_fma | 2,403 | 2,403 | 0 | 0 | 268 | 0 | 0 | 2 |
| u_core/u_vfpu/g_sfu.u_sfu | 2,526 | 2,479 | 0 | 47 | 1,951 | 0 | 15 | 14 |
| u_slice | 30,714 | 21,082 | 9,632 | 0 | 4,641 | 0 | 0 | 0 |
