Report: `evidence/r41/hbm_bw/resources/synth/base_67a453b77be8/utilization.rpt`; kernel total LUTs 812,108. Every instance of a module type, summed (an instance nested in one of its own type is not counted twice).

| Module type | instances | LUT | % of kernel LUT | LUTRAM | FF | DSP |
|---|---:|---:|---:|---:|---:|---:|
| tnca_multicore_shell | 1 | 785,001 | 96.7 % | 53,712 | 218,986 | 520 |
| tnca_top | 1 | 783,678 | 96.5 % | 52,992 | 218,476 | 520 |
| tnca_tile | 3 | 755,577 | 93.0 % | 45,696 | 205,806 | 513 |
| tnca_core | 3 | 663,628 | 81.7 % | 16,788 | 190,651 | 513 |
| tnca_fp32_fma | 96 | 228,912 | 28.2 % | 0 | 25,632 | 192 |
| tnca_vector_fpu | 3 | 189,543 | 23.3 % | 0 | 25,506 | 138 |
| tnca_matrix | 3 | 152,740 | 18.8 % | 8,568 | 34,368 | 96 |
| tnca_fma_norm_xls | 96 | 130,656 | 16.1 % | 0 | 0 | 0 |
| tnca_fpu_multi | 48 | 126,144 | 15.5 % | 0 | 17,424 | 96 |
| tnca_nuca_slice | 3 | 91,764 | 11.3 % | 28,908 | 14,941 | 0 |
| tnca_lsu_pair | 3 | 76,127 | 9.4 % | 216 | 40,979 | 0 |
| tnca_lsu | 6 | 73,986 | 9.1 % | 216 | 40,962 | 0 |
| tnca_vector_alu | 3 | 68,653 | 8.5 % | 0 | 5,889 | 192 |
| tnca_vrf_banked | 3 | 63,915 | 7.9 % | 0 | 18,873 | 0 |
| tnca_fma_align_xls | 96 | 50,928 | 6.3 % | 0 | 0 | 192 |
| tnca_fma_round_xls | 96 | 47,232 | 5.8 % | 0 | 0 | 0 |
| tnca_tensor | 3 | 43,107 | 5.3 % | 1,788 | 25,317 | 39 |
| tnca_line_axi | 3 | 23,028 | 2.8 % | 0 | 18,009 | 0 |
| tnca_cvt_comb_xls | 48 | 22,560 | 2.8 % | 0 | 0 | 0 |
| tnca_rf_bank | 12 | 22,512 | 2.8 % | 0 | 18,864 | 0 |
| tnca_barrier | 3 | 22,350 | 2.8 % | 0 | 11,673 | 0 |
| tnca_acc_rf | 3 | 19,117 | 2.4 % | 7,680 | 12,288 | 0 |
| tnca_replicated_tmu | 1 | 17,079 | 2.1 % | 1,020 | 7,632 | 0 |
| tnca_tmu | 3 | 17,079 | 2.1 % | 1,020 | 7,632 | 0 |
| tnca_widen_xls | 96 | 11,904 | 1.5 % | 0 | 0 | 0 |
| tnca_scalar_rf | 3 | 11,268 | 1.4 % | 6,216 | 768 | 0 |
| tnca_fifo | 94 | 9,597 | 1.2 % | 9,012 | 393 | 0 |
| tnca_sfu | 3 | 6,990 | 0.9 % | 0 | 6,219 | 42 |
| tnca_sfu_serial | 3 | 6,990 | 0.9 % | 0 | 6,219 | 42 |
| tnca_mesh | 1 | 5,082 | 0.6 % | 2,628 | 264 | 0 |
| tnca_scalar_alu | 3 | 4,404 | 0.5 % | 0 | 0 | 0 |
| tnca_line_link | 6 | 4,125 | 0.5 % | 4,032 | 6,255 | 0 |
| tnca_elastic_link | 26 | 4,072 | 0.5 % | 3,760 | 196 | 0 |
| tnca_router | 3 | 3,578 | 0.4 % | 1,220 | 200 | 0 |
| tnca_lzc | 51 | 3,426 | 0.4 % | 0 | 0 | 0 |
| tnca_pred_rf | 3 | 2,994 | 0.4 % | 0 | 3,024 | 0 |
| tnca_decoder | 6 | 2,886 | 0.4 % | 0 | 0 | 0 |
| tnca_sfu_log2_xls | 3 | 2,535 | 0.3 % | 0 | 846 | 9 |
| tnca_memory_mode | 1 | 2,007 | 0.2 % | 0 | 30 | 0 |
