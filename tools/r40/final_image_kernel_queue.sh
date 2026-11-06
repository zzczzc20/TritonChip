#!/bin/bash
# Guarded kernel synthesis of the final-image candidate, then the cross-SLR cone audit before (A3) / after.
set -u
NEW=$HOME/tnca_src_r40_final_d2cc850
IMG=$HOME/tnca_src_r40_attempt03_54249f7/evidence/r40/v2/convergence/attempt03/image.tcl
A3DCP=$HOME/tnca_build/r40_attempt03_20261017/xo/tmp/link/vivado/vpl/prj/prj.runs/ulp_tnca_u50_1_0_synth_1/ulp_tnca_u50_1_0.dcp
Q=$HOME/tnca_ooc_final_d2cc850
source /tools/Xilinx/Vivado/2023.2/settings64.sh
run() {  # name minGB args...
  local name=$1 min=$2; shift 2
  for i in $(seq 1 300); do
    if OOC_MIN_AVAIL_GB=$min bash $NEW/tools/r40/final_image_ooc.sh "$@" > $Q/$name.out 2>&1; then
      echo "$(date -u +%FT%TZ) $name DONE" >> $Q/queue.log; return 0
    fi
    if grep -q REFUSED $Q/$name.out; then sleep 600; continue; fi
    echo "$(date -u +%FT%TZ) $name FAILED: $(tail -2 $Q/$name.out | tr '\n' ' ')" >> $Q/queue.log; return 1
  done
}
run kernel 19 kernel $NEW $IMG $Q/kernel || exit 1
NEWDCP=$(ls $Q/kernel/*.dcp | head -1)
run audit_a3 18 audit $A3DCP $IMG $NEW $Q/audit_a3
run audit_new 18 audit $NEWDCP $IMG $NEW $Q/audit_new
echo "$(date -u +%FT%TZ) KERNEL-QUEUE-DONE" >> $Q/queue.log
