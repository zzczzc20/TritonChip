#!/bin/bash
# Read-only census of attempt 3's routed checkpoint; starts only with >= 30 GB available.
set -u
NEW=$HOME/tnca_src_r40_final_d2cc850
DCP=$HOME/tnca_build/r40_attempt03_20261017/xo/tmp/link/vivado/vpl/prj/prj.runs/impl_1/level0_wrapper_routed.dcp
Q=$HOME/tnca_ooc_final_d2cc850
source /tools/Xilinx/Vivado/2023.2/settings64.sh
for i in $(seq 1 600); do
  avail=$(free -g | awk '/^Mem:/{print $7}')
  if [ "$avail" -ge 30 ]; then
    echo "$(date -u +%FT%TZ) census start (avail ${avail} GB)" >> $Q/queue.log
    nice -n 10 vivado -mode batch -nojournal -log $Q/census.log -source $NEW/tools/r40/census_failing_endpoints.tcl \
      -tclargs $DCP $Q/census 20000 > $Q/census.out 2>&1
    echo "$(date -u +%FT%TZ) census exit $?" >> $Q/queue.log
    exit 0
  fi
  sleep 300
done
echo "$(date -u +%FT%TZ) census gave up waiting for memory" >> $Q/queue.log
