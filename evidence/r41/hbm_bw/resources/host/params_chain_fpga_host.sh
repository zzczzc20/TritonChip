#!/bin/bash
# R41 G2: parameter kernel-synthesis screens on the FPGA host, one at a time,
# each through ~/r41s2_tools/vivado_slot.sh (kernel synthesis needs ~9 GB).
cd "$(dirname "$0")"
P=src/evidence/r41/hbm_bw/resources/screens
for j in p1_dco_shared_tmu p2_lsu_single p3_no_vrf_prefetch p4_no_mtx_abuf; do
  mkdir -p $j
  ~/r41s2_tools/vivado_slot.sh --min-gb 20 $PWD/$j/slot.log -- vivado -mode batch -nojournal \
     -log $j/synth.log -source src/flow/fpga/synth_r40_multicore.tcl -tclargs src $P/image_$j.tcl $j/out
  echo "$(date -u +%FT%TZ) $j exit $?" >> chain.log
done
