#!/bin/bash
# slot 1: K32 tile arms with the half-line engine (item 1) and the sixteen-register A buffer (item 2), tiny points
M=/home/zhongchun/r42/lineelig_m2/tools/r42/images; O=~/r42/lineelig_out/mc; R=~/r42/lineelig_out/le_mc3.sh
K="TNCA_GEMM_TILE=16,16,32"
P="op.matmul/tiny/fp16-m16-k32-n16 op.linear/tiny/fp16-m16-k64-n64"
$R lineelig_m2 $O/k32_line32_tiny    $M/r42_line32_sim.tcl            core-r42-line32 "$K" $P
$R lineelig_m2 $O/k32_line32_abuf2_tiny $M/r42_line32_abuf2_sim.tcl   core-r42-line32 "$K" $P
$R lineelig_m2 $O/k32_all3_tiny      $M/r42_line32_tag5_abuf2_sim.tcl core-r42-line32 "$K" $P
echo ARMS1_DONE
