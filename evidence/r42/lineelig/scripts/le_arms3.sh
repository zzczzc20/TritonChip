#!/bin/bash
# slot 3: items 3 and 2 on a small point (op.matmul/small, K128)
M=/home/zhongchun/r42/lineelig_m2/tools/r42/images; O=~/r42/lineelig_out/mc; R=~/r42/lineelig_out/le_mc3.sh
P="op.matmul/small/fp16-m64-k128-n128"
$R lineelig_m2 $O/line32_tag5_small $M/r42_line32_tag5_sim.tcl core-r42-line32 - $P
$R lineelig_m2 $O/tag5_small        $M/r42_tag5_sim.tcl core-vrf-sync-v5-x8-g2c4-dp - $P
$R lineelig_m2 $O/k32_line32_abuf2_small $M/r42_line32_abuf2_sim.tcl core-r42-line32 TNCA_GEMM_TILE=16,16,32 $P
$R lineelig_m2 $O/k32_old_small     $M/r41_run2_sim.tcl core-vrf-sync-v5-x8-g2c4-dp TNCA_GEMM_TILE=16,16,32 $P
echo ARMS3_DONE
