#!/bin/bash
C=$HOME/r42/kfuse-speed_out/cfg
mk() { name=$1; shift; cp $C/ideal.cfg $C/$name.cfg; for kv in "$@"; do echo $kv >> $C/$name.cfg; done; }
mk ideal_s1 issue_free_mode=1
mk ideal_s2 issue_free_mode=2
mk ideal_g8 mtx_dp_g_override=8 mtx_dp_b_regs=1
mk ideal_g8_s1 mtx_dp_g_override=8 mtx_dp_b_regs=1 issue_free_mode=1
mk ideal_g8_s2 mtx_dp_g_override=8 mtx_dp_b_regs=1 issue_free_mode=2
mk ideal_g16 mtx_dp_g_override=16 mtx_dp_b_regs=1
mk ideal_g16_s1 mtx_dp_g_override=16 mtx_dp_b_regs=1 issue_free_mode=1
mk ideal_g16_s2 mtx_dp_g_override=16 mtx_dp_b_regs=1 issue_free_mode=2
mk ideal_g8_pf mtx_dp_g_override=8 mtx_dp_b_regs=1 mtx_dp_prefetch=1
mk ideal_g16_pf mtx_dp_g_override=16 mtx_dp_b_regs=1 mtx_dp_prefetch=1
ls $C | wc -l
