# R41 G4 run-2 candidate image (not linked; the lead starts the link). Four Role Contexts (the author
# withdrew the contexts cut on 2026-10-29; profile core-vrf-sync-v5-x8-g2c4) and G2's other cuts, plus
# G4's one-lane path (G4_DESIGN.md section 8: hybrid line LSU, pipelined line home, full-line stores
# without fill, pipelined outlets, striped HBM) and the run-2 steps of section 9.6: the line-wide tensor
# transfer engine (TENSOR_LINE 1), chained home fills (HOME_LINE_PIPE 2), 512-bit outlets (AXI_DATA_W 512)
# and 64 lines in flight per core and per home (LINE_SLOTS, HOME_MSHR_N, AXI_OUT_N 64) = image_s7.tcl
# plus HBM_STRIPED; merged 2026-10-29 with G2's FMA_IMPL 1 and shared MAC lane (MAC_DP 1, MAC_CUTS 29 =
# 0b11101; fma_dslx/screens/image_g2c4_dp.tcl) and K8's registered matrix operand capture (MTX_VRF_REG 1;
# timing/image_g2c4_mtx_vrf_reg.tcl).
set TNCA_IMAGE_PARAMS [dict create AXI_DATA_W 512 AXI_ID_W 6 AXI_LINE_LUTRAM 1 BAR_EPOCH_W 38 BAR_N 8 BAR_W 3 AXI_OUT_N 64 CACHE_MAINTENANCE 1 CODE_LINK_STAGES 2 DCO_LOCAL_REPLICAS 1 DCO_REGISTERED_RANGES 1 DCO_REGISTER_WRITES 1 DCO_TLW_PIPE_STAGES 2 DISPATCH_STAGES 2 FE_NOBUBBLE 1 FPU_OVERLAP 1 HAS_ATOMICS 1 HAS_SFU 1 HAS_SYS_ATOMICS 0 HAS_TENSOR_XPOSE 2 LINK_STAGES 2 LSU_INFLIGHT 2 MEMORY_CONTROLS 1 MEM_HOME_STAGES 2 MEM_PORTS 3 MTX_ABUF 1 PHYS_ADDR_W 35 NOC_LINK_STAGES 2 NUCA_CACHE_DATA_PIPELINE 1 NUCA_REGISTER_VICTIM_RANK 1 NUCA_REGISTER_VICTIM_SELECT 1 NUCA_SHARED_DATA_READ 2 NUM_RCTX 4 RCTX_SWITCH 2 RCTX_W 2 PHYS_VREGS 256 PHYS_W 8 RPAIR 1 RSLICE 1 SETS 32 SET_W 5 USE_BANKED_VRF 1 VREGS_PER_RCTX 64 VRF_BYPASS 1 VRF_PREFETCH 1 WAYS 8 WAY_W 3 HYBRID_LSU 1 HOME_BACKGROUND_READS 1 HOME_LINE_WRITE_NOFILL 1 HOME_LINE_PIPE 2 HOME_MSHR_N 64 LINE_SLOTS 64 AXI_PIPELINED_ISSUE 1 HBM_STRIPED 1 TENSOR_LINE 1 FMA_IMPL 1 MAC_DP 1 MAC_CUTS 29 MTX_VRF_REG 1]
