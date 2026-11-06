Report: `evidence/r41/hbm_bw/resources/synth/run1src_c9f25787494a/utilization.rpt`; kernel total LUTs 809,018. Every instance of a module type, summed (an instance nested in one of its own type is not counted twice).

| Module type | instances | LUT | % of kernel LUT | LUTRAM | FF | DSP |
|---|---:|---:|---:|---:|---:|---:|
| tnca_multicore_shell | 1 | 782,070 | 96.7 % | 53,732 | 208,057 | 520 |
| tnca_top | 1 | 780,747 | 96.5 % | 53,012 | 207,547 | 520 |
| tnca_tile | 3 | 752,888 | 93.1 % | 45,716 | 194,877 | 513 |
| tnca_core | 3 | 660,185 | 81.6 % | 16,800 | 179,619 | 513 |
| tnca_fp32_fma | 96 | 229,008 | 28.3 % | 0 | 25,632 | 192 |
| tnca_vector_fpu | 3 | 188,502 | 23.3 % | 0 | 25,470 | 138 |
| tnca_matrix | 3 | 153,204 | 18.9 % | 8,568 | 34,320 | 96 |
| tnca_fma_norm_xls | 96 | 130,656 | 16.1 % | 0 | 0 | 0 |
| tnca_fpu_multi | 48 | 126,192 | 15.6 % | 0 | 17,424 | 96 |
| tnca_nuca_slice | 3 | 92,518 | 11.4 % | 28,916 | 15,044 | 0 |
| tnca_lsu_pair | 3 | 76,310 | 9.4 % | 216 | 40,922 | 0 |
| tnca_lsu | 6 | 74,178 | 9.2 % | 216 | 40,914 | 0 |
| tnca_vector_alu | 3 | 69,459 | 8.6 % | 0 | 0 | 192 |
| tnca_vrf_banked | 3 | 63,864 | 7.9 % | 0 | 18,873 | 0 |
| tnca_fma_align_xls | 96 | 51,024 | 6.3 % | 0 | 0 | 192 |
| tnca_fma_round_xls | 96 | 47,232 | 5.8 % | 0 | 0 | 0 |
| tnca_tensor | 3 | 43,476 | 5.4 % | 1,800 | 24,934 | 39 |
| tnca_line_axi | 3 | 22,869 | 2.8 % | 0 | 18,009 | 0 |
| tnca_cvt_comb_xls | 48 | 22,560 | 2.8 % | 0 | 0 | 0 |
| tnca_rf_bank | 12 | 22,512 | 2.8 % | 0 | 18,864 | 0 |
| tnca_barrier | 3 | 22,350 | 2.8 % | 0 | 11,673 | 0 |
| tnca_acc_rf | 3 | 19,581 | 2.4 % | 7,680 | 12,288 | 0 |
| tnca_replicated_tmu | 1 | 16,836 | 2.1 % | 1,020 | 7,632 | 0 |
| tnca_tmu | 3 | 16,836 | 2.1 % | 1,020 | 7,632 | 0 |
| tnca_widen_xls | 96 | 11,904 | 1.5 % | 0 | 0 | 0 |
| tnca_scalar_rf | 3 | 11,268 | 1.4 % | 6,216 | 768 | 0 |
| tnca_fifo | 94 | 9,597 | 1.2 % | 9,012 | 393 | 0 |
| tnca_sfu | 3 | 6,990 | 0.9 % | 0 | 6,219 | 42 |
| tnca_mesh | 1 | 5,083 | 0.6 % | 2,628 | 264 | 0 |
| tnca_scalar_alu | 3 | 4,404 | 0.5 % | 0 | 0 | 0 |
| tnca_line_link | 6 | 4,125 | 0.5 % | 4,032 | 6,255 | 0 |
| tnca_elastic_link | 26 | 4,072 | 0.5 % | 3,760 | 196 | 0 |
| tnca_router | 3 | 3,579 | 0.4 % | 1,220 | 200 | 0 |
| tnca_lzc | 51 | 3,426 | 0.4 % | 0 | 0 | 0 |
| tnca_pred_rf | 3 | 2,994 | 0.4 % | 0 | 3,024 | 0 |
| tnca_decoder | 6 | 2,886 | 0.4 % | 0 | 0 | 0 |
| tnca_sfu_log2_xls | 3 | 2,535 | 0.3 % | 0 | 846 | 9 |
| tnca_memory_mode | 1 | 2,007 | 0.2 % | 0 | 30 | 0 |
