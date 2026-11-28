
s0 default tile, routed OOC, 5.6 ns constraint (endpoints with slack < 1.5 ns; path = 5.6 - slack)
| unit | endpoints <1.5 | fail @5.6 | path>5.9 (slack<-0.3) | path>6.2 (<-0.6) | path>6.67 (<-1.07) | worst slack |
|---|--:|--:|--:|--:|--:|--:|
| u_core/g_vrf_banked.u_vrf | 9270 | 1915 | 763 | 512 | 407 | -1.642 |
| u_core/u_mtx | 25178 | 5700 | 3486 | 2071 | 624 | -1.637 |
| u_core/(core own) | 9197 | 3889 | 2646 | 1723 | 522 | -1.634 |
| u_core/g_lsu_hybrid.u_lsu | 10300 | 4437 | 2929 | 1387 | 776 | -1.634 |
| g_home_pipe.u_home_pipe | 68285 | 19950 | 14217 | 10614 | 3195 | -1.630 |
| u_core/u_vfpu | 5497 | 2175 | 1812 | 1549 | 714 | -1.616 |
| u_core/u_txf | 10557 | 3660 | 2354 | 1233 | 205 | -1.593 |
| u_core/u_bar | 1671 | 1279 | 837 | 316 | 28 | -1.460 |
| u_core/u_valu | 3211 | 1300 | 604 | 286 | 53 | -1.199 |
| g_host_mux.u_host | 4 | 4 | 0 | 0 | 0 | -0.168 |
| u_core/u_srf | 1368 | 0 | 0 | 0 | 0 | 0.194 |
| u_core/u_mul | 520 | 0 | 0 | 0 | 0 | 0.364 |
| u_core/u_prf | 301 | 0 | 0 | 0 | 0 | 0.471 |
| **tile** | 145359 | 44309 | 29648 | 19691 | 6524 | |

s3 shrunk tile, routed OOC, 5.6 ns constraint (endpoints with slack < 1.5 ns; path = 5.6 - slack)
| unit | endpoints <1.5 | fail @5.6 | path>5.9 (slack<-0.3) | path>6.2 (<-0.6) | path>6.67 (<-1.07) | worst slack |
|---|--:|--:|--:|--:|--:|--:|
| u_core/g_lsu_hybrid.u_lsu | 8875 | 3305 | 1692 | 843 | 469 | -1.392 |
| u_core/u_vfpu | 5182 | 1976 | 1718 | 1505 | 551 | -1.390 |
| u_core/(core own) | 7860 | 4311 | 2498 | 1279 | 201 | -1.389 |
| g_home_pipe.u_home_pipe | 12445 | 7190 | 5959 | 3715 | 572 | -1.387 |
| u_core/u_mtx | 17261 | 4342 | 2544 | 1476 | 326 | -1.367 |
| u_core/u_txf | 10512 | 3318 | 1756 | 731 | 119 | -1.353 |
| u_core/u_bar | 1671 | 1469 | 1165 | 688 | 65 | -1.333 |
| u_core/g_vrf_banked.u_vrf | 8487 | 4060 | 2616 | 1105 | 83 | -1.329 |
| u_core/u_valu | 3048 | 573 | 407 | 307 | 103 | -1.298 |
| u_core/u_srf | 1684 | 176 | 152 | 80 | 8 | -1.092 |
| g_host_mux.u_host | 4 | 4 | 4 | 4 | 0 | -0.991 |
| u_core/u_mul | 520 | 479 | 193 | 0 | 0 | -0.405 |
| u_core/u_prf | 755 | 0 | 0 | 0 | 0 | 0.022 |
| **tile** | 78304 | 31203 | 20704 | 11733 | 2497 | |
