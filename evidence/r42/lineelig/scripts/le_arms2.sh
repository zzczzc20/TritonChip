#!/bin/bash
# slot 2: K32 tile arms on the unchanged engine, (2) alone, and the combined (1)+(3) at the default tile
M=/home/zhongchun/r42/lineelig_m2/tools/r42/images; O=~/r42/lineelig_out/mc; R=~/r42/lineelig_out/le_mc3.sh
K="TNCA_GEMM_TILE=16,16,32"
P="op.matmul/tiny/fp16-m16-k32-n16 op.linear/tiny/fp16-m16-k64-n64"
$R lineelig_m2 $O/k32_old_tiny       $M/r41_run2_sim.tcl core-vrf-sync-v5-x8-g2c4-dp "$K" $P
$R lineelig_m2 $O/k32_abuf2_tiny     $M/r42_abuf2_sim.tcl core-vrf-sync-v5-x8-g2c4-dp "$K" $P
$R lineelig_m2 $O/abuf2_tiny         $M/r42_abuf2_sim.tcl core-vrf-sync-v5-x8-g2c4-dp - $P
$R lineelig_m2 $O/line32_tag5_tiny   $M/r42_line32_tag5_sim.tcl core-r42-line32 - op.matmul/tiny/fp16-m16-k32-n16 op.linear/tiny/fp16-m16-k64-n64 op.gemv/tiny/fp16-k64-n64 op.bmm/tiny/fp16-b2-m16-k16-n16
echo ARMS2_DONE
