#!/bin/bash
# creates cfg variants of cal.cfg in ~/r42/kfuse-speed_out/cfg
C=$HOME/r42/kfuse-speed_out/cfg
mk() { name=$1; shift; cp $C/cal.cfg $C/$name.cfg; for kv in "$@"; do echo $kv >> $C/$name.cfg; done; }
mk cal_g2 mtx_dp_g_override=2
mk cal_g4 mtx_dp_g_override=4 mtx_dp_b_regs=1
mk cal_g8 mtx_dp_g_override=8 mtx_dp_b_regs=1
mk cal_g16 mtx_dp_g_override=16 mtx_dp_b_regs=1
mk cal_s1 issue_free_mode=1
mk cal_s2 issue_free_mode=2
mk cal_g8_s1 mtx_dp_g_override=8 mtx_dp_b_regs=1 issue_free_mode=1
mk cal_g16_s1 mtx_dp_g_override=16 mtx_dp_b_regs=1 issue_free_mode=1
mk cal_g8_s2 mtx_dp_g_override=8 mtx_dp_b_regs=1 issue_free_mode=2
mk cal_g16_s2 mtx_dp_g_override=16 mtx_dp_b_regs=1 issue_free_mode=2
ls $C
mk cal_g4_s1 mtx_dp_g_override=4 mtx_dp_b_regs=1 issue_free_mode=1
mk cal_g4_s2 mtx_dp_g_override=4 mtx_dp_b_regs=1 issue_free_mode=2
mk cal_g8_pf mtx_dp_g_override=8 mtx_dp_b_regs=1 mtx_dp_prefetch=1
mk cal_g16_pf mtx_dp_g_override=16 mtx_dp_b_regs=1 mtx_dp_prefetch=1
mk cal_s1_pf issue_free_mode=1 mtx_dp_prefetch=1
mk cal_t4 dma_outstanding=64
mk cal_t4_s1 dma_outstanding=64 issue_free_mode=1
mk cal_t4_s2 dma_outstanding=64 issue_free_mode=2
mk cal_t4_g8 dma_outstanding=64 mtx_dp_g_override=8 mtx_dp_b_regs=1
mk cal_t4_g8_s1 dma_outstanding=64 mtx_dp_g_override=8 mtx_dp_b_regs=1 issue_free_mode=1
mk cal_t4_g8_s2 dma_outstanding=64 mtx_dp_g_override=8 mtx_dp_b_regs=1 issue_free_mode=2
mk cal_t4_g16_s2 dma_outstanding=64 mtx_dp_g_override=16 mtx_dp_b_regs=1 issue_free_mode=2
mk cal_t4u dma_outstanding=64 dma_unit_slots=32
mk cal_s3 issue_free_mode=3
mk cal_g8_s3 mtx_dp_g_override=8 mtx_dp_b_regs=1 issue_free_mode=3
mk cal_g16_s3 mtx_dp_g_override=16 mtx_dp_b_regs=1 issue_free_mode=3
mk cal_u32 dma_unit_slots=32
mk cal_u32_s1 dma_unit_slots=32 issue_free_mode=1
mk cal_u32_g8 dma_unit_slots=32 mtx_dp_g_override=8 mtx_dp_b_regs=1
mk cal_u32_g8_s1 dma_unit_slots=32 mtx_dp_g_override=8 mtx_dp_b_regs=1 issue_free_mode=1
mk cal_u32_g8_s2 dma_unit_slots=32 mtx_dp_g_override=8 mtx_dp_b_regs=1 issue_free_mode=2
mk cal_lat0 tl_outlet_to_slice_cycles=1 tl_slice_to_outlet_cycles=1 hbm_latency_cycles=1 tl_hbm_write_latency_cycles=1 tl_noc_inject_cycles=1 tl_noc_eject_cycles=1 tl_slice_hit_latency=1
mk cal_lat0_s1 tl_outlet_to_slice_cycles=1 tl_slice_to_outlet_cycles=1 hbm_latency_cycles=1 tl_hbm_write_latency_cycles=1 tl_noc_inject_cycles=1 tl_noc_eject_cycles=1 tl_slice_hit_latency=1 issue_free_mode=1
mk cal_lat0_g8 tl_outlet_to_slice_cycles=1 tl_slice_to_outlet_cycles=1 hbm_latency_cycles=1 tl_hbm_write_latency_cycles=1 tl_noc_inject_cycles=1 tl_noc_eject_cycles=1 tl_slice_hit_latency=1 mtx_dp_g_override=8 mtx_dp_b_regs=1
mk cal_lat0_g8_s1 tl_outlet_to_slice_cycles=1 tl_slice_to_outlet_cycles=1 hbm_latency_cycles=1 tl_hbm_write_latency_cycles=1 tl_noc_inject_cycles=1 tl_noc_eject_cycles=1 tl_slice_hit_latency=1 mtx_dp_g_override=8 mtx_dp_b_regs=1 issue_free_mode=1
